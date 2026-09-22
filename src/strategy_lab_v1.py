#!/usr/bin/env python3
"""CFB Edge Engine — Strategy Lab V1.

Read-only research report for evaluating predefined model-side spread filters.

This script:
- reads immutable prediction snapshots and derived scores;
- does not call APIs;
- does not modify the neural model, snapshots, results, scores, or wager ledger;
- writes derived CSV/JSON research outputs only;
- does not create wagers or claim a validated betting rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfb_edge_paths import HISTORY_DB, STRATEGY_LAB_CSV, STRATEGY_LAB_FILE


OUTPUT_JSON = STRATEGY_LAB_FILE
OUTPUT_CSV = STRATEGY_LAB_CSV

EDGE_THRESHOLDS = (2.5, 4.0, 6.0)
VERIFIED_NEAR_CLOSE_MINUTES = 120.0
NEAR_CLOSE_PROXY_MINUTES = 360.0
DEFAULT_MINIMUM_SAMPLE = 30


def print_rule(char: str = "=", width: int = 108) -> None:
    print(char * width)


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    return int(number) if number is not None else None


def normalize_team(value: Any) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in str(value or ""))
        .split()
    )


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fmt(value: float | None, decimals: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{decimals}f}"


def choose_snapshots(rows: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    if policy == "all":
        return sorted(rows, key=lambda row: (str(row.get("captured_utc") or ""), str(row.get("snapshot_id") or "")))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["game_key"])].append(row)

    selected: list[dict[str, Any]] = []
    for game_rows in grouped.values():
        ordered = sorted(
            game_rows,
            key=lambda row: (str(row.get("captured_utc") or ""), str(row.get("snapshot_id") or "")),
        )
        selected.append(ordered[0] if policy == "first" else ordered[-1])

    return sorted(selected, key=lambda row: (str(row.get("kickoff_utc") or ""), str(row.get("game_key") or "")))


def load_scored_rows() -> list[dict[str, Any]]:
    if not HISTORY_DB.exists():
        raise FileNotFoundError(f"Prediction history database not found: {HISTORY_DB}")

    connection = sqlite3.connect(f"file:{HISTORY_DB}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        rows = connection.execute(
            """
            SELECT
                p.*,
                s.actual_home_margin,
                s.margin_error,
                s.absolute_margin_error,
                s.winner_correct,
                s.brier_score,
                s.model_side_ats_result,
                s.model_side_flat_units,
                r.away_points,
                r.home_points,
                r.winner AS actual_winner
            FROM prediction_snapshots p
            JOIN prediction_scores s ON s.snapshot_id = p.snapshot_id
            JOIN game_results r ON r.game_key = p.game_key
            ORDER BY p.captured_utc, p.snapshot_id
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def load_all_snapshots() -> dict[str, list[dict[str, Any]]]:
    connection = sqlite3.connect(f"file:{HISTORY_DB}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    snapshot_id, game_key, captured_utc, kickoff_utc,
                    model_favorite, hardrock_available,
                    hardrock_spread_for_model_favorite,
                    hardrock_price_for_model_favorite
                FROM prediction_snapshots
                ORDER BY captured_utc, snapshot_id
                """
            ).fetchall()
        ]
    finally:
        connection.close()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("game_key") or "")].append(row)
    return grouped


def is_full_input(row: dict[str, Any]) -> bool:
    return str(row.get("nn_input_status") or "").strip().upper().startswith("FULL")


def has_clean_signal(row: dict[str, Any]) -> bool:
    return str(row.get("signal_warning") or "NONE").strip().upper() in ("", "NONE")


def quoted_hardrock(row: dict[str, Any]) -> bool:
    return (
        safe_int(row.get("hardrock_available")) == 1
        and safe_float(row.get("hardrock_spread_for_model_favorite")) is not None
        and safe_float(row.get("hardrock_price_for_model_favorite")) is not None
    )


def snapshot_spread_for_team(snapshot: dict[str, Any], team: str) -> float | None:
    spread = safe_float(snapshot.get("hardrock_spread_for_model_favorite"))
    favorite = normalize_team(snapshot.get("model_favorite"))
    target = normalize_team(team)
    if spread is None or not favorite or not target:
        return None
    if favorite == target:
        return spread
    return -spread


def close_quality(minutes_before_kickoff: float | None) -> str:
    if minutes_before_kickoff is None or minutes_before_kickoff < 0:
        return "UNAVAILABLE"
    if minutes_before_kickoff <= VERIFIED_NEAR_CLOSE_MINUTES:
        return "VERIFIED_NEAR_CLOSE"
    if minutes_before_kickoff <= NEAR_CLOSE_PROXY_MINUTES:
        return "NEAR_CLOSE_PROXY"
    return "OLDER_PROXY"


def calculate_clv(row: dict[str, Any], game_snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    target_team = str(row.get("model_favorite") or "").strip()
    ticket_spread = safe_float(row.get("hardrock_spread_for_model_favorite"))
    captured = parse_datetime(row.get("captured_utc"))
    kickoff = parse_datetime(row.get("kickoff_utc"))

    if not target_team or ticket_spread is None or captured is None or kickoff is None:
        return {"status": "UNAVAILABLE", "clv_points": None, "close_spread": None, "close_quality": "UNAVAILABLE"}

    candidates: list[dict[str, Any]] = []
    for snapshot in game_snapshots:
        snapshot_time = parse_datetime(snapshot.get("captured_utc"))
        if snapshot_time is None or snapshot_time <= captured or snapshot_time >= kickoff:
            continue
        if safe_int(snapshot.get("hardrock_available")) != 1:
            continue
        close_spread = snapshot_spread_for_team(snapshot, target_team)
        if close_spread is not None:
            candidates.append({"snapshot": snapshot, "spread": close_spread, "captured": snapshot_time})

    if not candidates:
        return {"status": "NO_POST_CAPTURE_QUOTE", "clv_points": None, "close_spread": None, "close_quality": "UNAVAILABLE"}

    close = max(candidates, key=lambda candidate: (candidate["captured"], str(candidate["snapshot"].get("snapshot_id") or "")))
    minutes = (kickoff - close["captured"]).total_seconds() / 60.0
    return {
        "status": "AVAILABLE",
        "clv_points": round(ticket_spread - close["spread"], 3),
        "close_spread": round(close["spread"], 3),
        "close_captured_utc": close["snapshot"].get("captured_utc"),
        "close_minutes_before_kickoff": round(minutes, 2),
        "close_quality": close_quality(minutes),
    }


def strategy_metrics(rows: list[dict[str, Any]], minimum_sample: int) -> dict[str, Any]:
    graded = [row for row in rows if row.get("model_side_ats_result") in ("W", "L", "P")]
    wins = sum(row.get("model_side_ats_result") == "W" for row in graded)
    losses = sum(row.get("model_side_ats_result") == "L" for row in graded)
    pushes = sum(row.get("model_side_ats_result") == "P" for row in graded)
    decisions = wins + losses

    units = [safe_float(row.get("model_side_flat_units")) for row in graded]
    priced_units = [value for value in units if value is not None]
    priced_decisions = sum(
        row.get("model_side_ats_result") in ("W", "L")
        and safe_float(row.get("model_side_flat_units")) is not None
        for row in graded
    )

    edges = [abs(safe_float(row.get("model_edge_points")) or 0.0) for row in rows if safe_float(row.get("model_edge_points")) is not None]
    clv_rows = [row for row in rows if row.get("clv_close_quality") == "VERIFIED_NEAR_CLOSE"]
    clv_values = [safe_float(row.get("clv_points")) for row in clv_rows]
    clv_values = [value for value in clv_values if value is not None]

    return {
        "games": len(rows),
        "ats_games": len(graded),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "cover_rate": (wins / decisions) if decisions else None,
        "priced_decisions": priced_decisions,
        "flat_units": round(sum(priced_units), 4) if priced_units else None,
        "flat_roi": round(sum(priced_units) / priced_decisions, 6) if priced_units and priced_decisions else None,
        "average_abs_model_edge_points": round(sum(edges) / len(edges), 3) if edges else None,
        "verified_near_close_clv_games": len(clv_values),
        "average_verified_near_close_clv_points": round(sum(clv_values) / len(clv_values), 3) if clv_values else None,
        "positive_verified_clv": sum(value > 0 for value in clv_values),
        "negative_verified_clv": sum(value < 0 for value in clv_values),
        "zero_verified_clv": sum(value == 0 for value in clv_values),
        "sample_status": "MEETS_MINIMUM_SAMPLE" if decisions >= minimum_sample else "INSUFFICIENT_SAMPLE",
    }


def enrich_rows(rows: list[dict[str, Any]], all_snapshots: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        edge = safe_float(row.get("model_edge_points"))
        clv = calculate_clv(row, all_snapshots.get(str(row.get("game_key") or ""), []))
        row.update(
            {
                "absolute_model_edge_points": round(abs(edge), 3) if edge is not None else None,
                "full_input": is_full_input(row),
                "clean_signal": has_clean_signal(row),
                "hardrock_quoted": quoted_hardrock(row),
                "clv_points": clv.get("clv_points"),
                "clv_status": clv.get("status"),
                "clv_close_spread": clv.get("close_spread"),
                "clv_close_captured_utc": clv.get("close_captured_utc"),
                "clv_close_minutes_before_kickoff": clv.get("close_minutes_before_kickoff"),
                "clv_close_quality": clv.get("close_quality"),
            }
        )
        enriched.append(row)
    return enriched


def build_strategy_summary(rows: list[dict[str, Any]], minimum_sample: int) -> dict[str, Any]:
    strategies: dict[str, dict[str, Any]] = {
        "BASELINE_ALL_MODEL_SIDES": strategy_metrics(rows, minimum_sample),
        "FULL_INPUT_ONLY": strategy_metrics([row for row in rows if row["full_input"]], minimum_sample),
        "FULL_INPUT_CLEAN_SIGNAL_ONLY": strategy_metrics(
            [row for row in rows if row["full_input"] and row["clean_signal"]],
            minimum_sample,
        ),
    }

    for threshold in EDGE_THRESHOLDS:
        label = str(threshold).replace(".", "_")
        edge_rows = [
            row for row in rows
            if (safe_float(row.get("absolute_model_edge_points")) or -1.0) >= threshold
        ]
        strategies[f"EDGE_AT_LEAST_{label}"] = strategy_metrics(edge_rows, minimum_sample)
        strategies[f"FULL_INPUT_CLEAN_EDGE_AT_LEAST_{label}"] = strategy_metrics(
            [row for row in edge_rows if row["full_input"] and row["clean_signal"]],
            minimum_sample,
        )

    edge_buckets = {
        "UNDER_2_5": [row for row in rows if (safe_float(row.get("absolute_model_edge_points")) or -1.0) < 2.5],
        "2_5_TO_3_99": [row for row in rows if 2.5 <= (safe_float(row.get("absolute_model_edge_points")) or -1.0) < 4.0],
        "4_0_TO_5_99": [row for row in rows if 4.0 <= (safe_float(row.get("absolute_model_edge_points")) or -1.0) < 6.0],
        "6_0_PLUS": [row for row in rows if (safe_float(row.get("absolute_model_edge_points")) or -1.0) >= 6.0],
    }
    return {
        "predefined_strategy_results": strategies,
        "edge_bucket_results": {
            name: strategy_metrics(bucket_rows, minimum_sample)
            for name, bucket_rows in edge_buckets.items()
        },
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "game_key", "game_id", "season", "week", "kickoff_utc", "captured_utc",
        "away_team", "home_team", "model_favorite", "model_spread_display",
        "model_edge_points", "absolute_model_edge_points", "hardrock_display",
        "hardrock_spread_for_model_favorite", "hardrock_price_for_model_favorite",
        "model_side_ats_result", "model_side_flat_units", "strength_label",
        "nn_input_status", "nn_feature_coverage", "full_input", "clean_signal",
        "hardrock_quoted", "signal_warning", "clv_points", "clv_status",
        "clv_close_spread", "clv_close_captured_utc", "clv_close_minutes_before_kickoff",
        "clv_close_quality",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(name: str, metrics: dict[str, Any]) -> None:
    print(
        f"{name:<42} | n={metrics['ats_games']:<3} | "
        f"ATS={metrics['wins']}-{metrics['losses']}-{metrics['pushes']} | "
        f"cover={fmt(metrics['cover_rate'])} | "
        f"units={fmt(metrics['flat_units'], 2)} | "
        f"ROI={fmt(metrics['flat_roi'])} | {metrics['sample_status']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only prospective strategy research for frozen CFB Edge predictions."
    )
    parser.add_argument("--policy", choices=("latest", "first", "all"), default="latest")
    parser.add_argument("--season", type=int, default=None, help="Optional season filter.")
    parser.add_argument("--week", type=int, default=None, help="Optional week filter.")
    parser.add_argument("--minimum-sample", type=int, default=DEFAULT_MINIMUM_SAMPLE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_sample < 1:
        raise ValueError("--minimum-sample must be at least 1.")

    all_scored = load_scored_rows()
    selected = choose_snapshots(all_scored, args.policy)
    if args.season is not None:
        selected = [row for row in selected if safe_int(row.get("season")) == args.season]
    if args.week is not None:
        selected = [row for row in selected if safe_int(row.get("week")) == args.week]

    enriched = enrich_rows(selected, load_all_snapshots())
    strategy_report = build_strategy_summary(enriched, args.minimum_sample)
    summary = {
        "schema_version": 1,
        "generated_utc": utc_iso(),
        "snapshot_policy": args.policy,
        "filters": {"season": args.season, "week": args.week, "minimum_sample": args.minimum_sample},
        "contract": {
            "model_usage": "FROZEN_MODEL_RESEARCH_ONLY",
            "no_api_calls": True,
            "no_database_writes": True,
            "no_wager_creation": True,
            "important_note": (
                "This report evaluates predefined hypothetical model-side filters using immutable "
                "pre-kickoff snapshots. It is not a validated wagering rule or an automated betting system."
            ),
        },
        "data_quality": {
            "selected_scored_snapshots": len(enriched),
            "hardrock_quoted": sum(row["hardrock_quoted"] for row in enriched),
            "full_input": sum(row["full_input"] for row in enriched),
            "clean_signal": sum(row["clean_signal"] for row in enriched),
            "verified_near_close_clv": sum(row.get("clv_close_quality") == "VERIFIED_NEAR_CLOSE" for row in enriched),
        },
        **strategy_report,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(enriched)

    print_rule()
    print("CFB EDGE ENGINE — STRATEGY LAB V1")
    print_rule()
    print("Mode: READ-ONLY RESEARCH / FROZEN MODEL / NO API CALLS / NO WAGERS CREATED")
    print(f"History DB: {HISTORY_DB}")
    print(f"Policy: {args.policy.upper()} | Season: {args.season or 'ALL'} | Week: {args.week or 'ALL'}")
    print_rule()
    print("PREDEFINED STRATEGIES")
    print_rule("-")
    for name, metrics in summary["predefined_strategy_results"].items():
        print_metrics(name, metrics)
    print()
    print("MODEL-TO-MARKET EDGE BUCKETS")
    print_rule("-")
    for name, metrics in summary["edge_bucket_results"].items():
        print_metrics(name, metrics)
    print()
    print("DATA QUALITY")
    print_rule("-")
    for name, value in summary["data_quality"].items():
        print(f"{name}: {value}")
    print()
    print("FILES WRITTEN")
    print_rule("-")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print()
    print("Do not promote a strategy from this report without sufficient prospective sample size and verified near-close CLV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
