#!/usr/bin/env python3
"""CFB Edge Engine — Research Snapshot Policy Robustness V1.

Read-only companion to ``research_hindsight_audit_v1.py`` for the study
"Beyond Hindsight: Building an Auditable Neural Decision-Support System for
Real-Time Forecasting."

Why this report exists
----------------------
One game can have more than one valid forecast record before kickoff.  A
research report should not choose the most favorable record after results are
known.  This script evaluates two fixed, documented policies side by side:

* FIRST:  the earliest strictly pre-kickoff stored snapshot per game.
* LATEST: the final strictly pre-kickoff stored snapshot per game.

It verifies the same timing, hash, trigger, and run-count evidence used by the
hindsight audit, then reports descriptive outcome metrics for both policies.
It also compares them on the games scored under both policies.

Safety
------
* No API calls.
* Prediction-history SQLite database is opened read-only.
* No model retraining or model changes.
* No prediction, result, market, wager, or database writes.
* Only derived JSON and CSV research summaries are written unless --dry-run is
  supplied.

Run
---
python src/research_snapshot_policy_robustness_v1.py --season 2026

The output is descriptive research evidence.  It does not recommend a wager,
claim profitability, or select a policy based on which one performs better.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import research_hindsight_audit_v1 as audit
from cfb_edge_paths import PROCESSED_DIR


OUTPUT_JSON = PROCESSED_DIR / "research_snapshot_policy_robustness_v1.json"
OUTPUT_CSV = PROCESSED_DIR / "research_snapshot_policy_robustness_v1.csv"

POLICIES = ("first", "latest")


def rule(char: str = "=", width: int = 108) -> None:
    print(char * width)


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value: Any) -> float | None:
    return audit.safe_float(value)


def safe_int(value: Any) -> int | None:
    return audit.safe_int(value)


def parse_datetime(value: Any) -> datetime | None:
    return audit.parse_datetime(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def fmt(value: float | None, decimals: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{decimals}f}"


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def metric_delta(
    latest: float | None,
    first: float | None,
) -> float | None:
    if latest is None or first is None:
        return None
    return latest - first


def rows_with_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep snapshots with a derived score attached by the normal scorer."""
    return [row for row in rows if row.get("scored_utc")]


def fixed_policy_summary(
    policy: str,
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe one predeclared snapshot-selection policy."""
    scored = rows_with_scores(selected)
    timing_values = [
        value
        for row in selected
        if (value := safe_float(row.get("minutes_before_kickoff"))) is not None
    ]
    hash_verified = sum(bool(row.get("record_hash_verified")) for row in selected)
    metrics = audit.outcome_metrics(scored)

    return {
        "policy": policy,
        "policy_definition": (
            "Earliest strictly pre-kickoff stored snapshot per game."
            if policy == "first"
            else "Final strictly pre-kickoff stored snapshot per game."
        ),
        "selected_games": len(selected),
        "selected_scored_games": len(scored),
        "selected_awaiting_or_unmatched_games": len(selected) - len(scored),
        "selected_hashes_verified": hash_verified,
        "selected_all_hashes_verified": hash_verified == len(selected),
        "capture_timing_minutes_before_kickoff": {
            "n": len(timing_values),
            "mean": mean(timing_values),
            "median": median(timing_values),
            "minimum": min(timing_values) if timing_values else None,
            "maximum": max(timing_values) if timing_values else None,
        },
        "descriptive_outcomes": metrics,
    }


def rows_by_game(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        game_key = str(row.get("game_key") or "").strip()
        if game_key:
            result[game_key] = row
    return result


def score_equivalence_metrics(
    first_rows: list[dict[str, Any]],
    latest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare fixed policies only over games where each policy has a score."""
    first_by_game = rows_by_game(first_rows)
    latest_by_game = rows_by_game(latest_rows)
    common_game_keys = sorted(set(first_by_game) & set(latest_by_game))

    common_scored_keys = [
        game_key
        for game_key in common_game_keys
        if first_by_game[game_key].get("scored_utc")
        and latest_by_game[game_key].get("scored_utc")
    ]
    first_common_scored = [first_by_game[key] for key in common_scored_keys]
    latest_common_scored = [latest_by_game[key] for key in common_scored_keys]

    first_metrics = audit.outcome_metrics(first_common_scored)
    latest_metrics = audit.outcome_metrics(latest_common_scored)

    return {
        "common_games": len(common_game_keys),
        "common_scored_games": len(common_scored_keys),
        "first_only_scored_games": sum(
            bool(row.get("scored_utc"))
            for game_key, row in first_by_game.items()
            if game_key not in common_scored_keys
        ),
        "latest_only_scored_games": sum(
            bool(row.get("scored_utc"))
            for game_key, row in latest_by_game.items()
            if game_key not in common_scored_keys
        ),
        "first_policy": first_metrics,
        "latest_policy": latest_metrics,
        "latest_minus_first": {
            "winner_accuracy": metric_delta(
                latest_metrics["winner_accuracy"],
                first_metrics["winner_accuracy"],
            ),
            "brier_score": metric_delta(
                latest_metrics["brier_score"],
                first_metrics["brier_score"],
            ),
            "model_margin_mae": metric_delta(
                latest_metrics["model_margin_mae"],
                first_metrics["model_margin_mae"],
            ),
            "hardrock_margin_mae": metric_delta(
                latest_metrics["hardrock_margin_mae"],
                first_metrics["hardrock_margin_mae"],
            ),
            "model_minus_hardrock_mae": metric_delta(
                latest_metrics["model_minus_hardrock_mae"],
                first_metrics["model_minus_hardrock_mae"],
            ),
        },
    }


def selection_comparison(
    first_rows: list[dict[str, Any]],
    latest_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Document how the two fixed policies differ without judging either one."""
    first_by_game = rows_by_game(first_rows)
    latest_by_game = rows_by_game(latest_rows)
    all_game_keys = sorted(set(first_by_game) | set(latest_by_game))

    comparison_rows: list[dict[str, Any]] = []
    timing_gaps: list[float] = []
    model_margin_changes: list[float] = []
    same_snapshot_count = 0
    different_snapshot_count = 0
    missing_from_first = 0
    missing_from_latest = 0

    for game_key in all_game_keys:
        first = first_by_game.get(game_key)
        latest = latest_by_game.get(game_key)
        if first is None:
            missing_from_first += 1
            continue
        if latest is None:
            missing_from_latest += 1
            continue

        same_snapshot = first.get("snapshot_id") == latest.get("snapshot_id")
        if same_snapshot:
            same_snapshot_count += 1
        else:
            different_snapshot_count += 1

        first_captured = parse_datetime(first.get("captured_utc"))
        latest_captured = parse_datetime(latest.get("captured_utc"))
        capture_gap_minutes = None
        if first_captured is not None and latest_captured is not None:
            capture_gap_minutes = round(
                (latest_captured - first_captured).total_seconds() / 60.0,
                2,
            )
            if math.isfinite(capture_gap_minutes):
                timing_gaps.append(capture_gap_minutes)

        first_margin = safe_float(first.get("model_home_margin"))
        latest_margin = safe_float(latest.get("model_home_margin"))
        model_margin_change = metric_delta(latest_margin, first_margin)
        if model_margin_change is not None:
            model_margin_changes.append(model_margin_change)

        comparison_rows.append(
            {
                "game_key": game_key,
                "season": first.get("season") or latest.get("season"),
                "week": first.get("week") or latest.get("week"),
                "away_team": first.get("away_team") or latest.get("away_team"),
                "home_team": first.get("home_team") or latest.get("home_team"),
                "kickoff_utc": first.get("kickoff_utc") or latest.get("kickoff_utc"),
                "same_snapshot_selected": same_snapshot,
                "first_snapshot_id": first.get("snapshot_id"),
                "first_captured_utc": first.get("captured_utc"),
                "first_minutes_before_kickoff": first.get("minutes_before_kickoff"),
                "first_model_favorite": first.get("model_favorite"),
                "first_model_home_margin": first_margin,
                "first_scored_utc": first.get("scored_utc"),
                "first_winner_correct": first.get("winner_correct"),
                "first_absolute_margin_error": first.get("absolute_margin_error"),
                "first_hardrock_absolute_margin_error": first.get(
                    "hardrock_absolute_margin_error"
                ),
                "latest_snapshot_id": latest.get("snapshot_id"),
                "latest_captured_utc": latest.get("captured_utc"),
                "latest_minutes_before_kickoff": latest.get("minutes_before_kickoff"),
                "latest_model_favorite": latest.get("model_favorite"),
                "latest_model_home_margin": latest_margin,
                "latest_scored_utc": latest.get("scored_utc"),
                "latest_winner_correct": latest.get("winner_correct"),
                "latest_absolute_margin_error": latest.get("absolute_margin_error"),
                "latest_hardrock_absolute_margin_error": latest.get(
                    "hardrock_absolute_margin_error"
                ),
                "capture_gap_minutes_latest_after_first": capture_gap_minutes,
                "model_home_margin_change_latest_minus_first": model_margin_change,
            }
        )

    return (
        {
            "games_compared": len(comparison_rows),
            "same_snapshot_selected": same_snapshot_count,
            "different_snapshot_selected": different_snapshot_count,
            "missing_from_first_policy": missing_from_first,
            "missing_from_latest_policy": missing_from_latest,
            "latest_capture_after_first_minutes": {
                "n": len(timing_gaps),
                "mean": mean(timing_gaps),
                "median": median(timing_gaps),
                "minimum": min(timing_gaps) if timing_gaps else None,
                "maximum": max(timing_gaps) if timing_gaps else None,
            },
            "model_home_margin_change_latest_minus_first": {
                "n": len(model_margin_changes),
                "mean": mean(model_margin_changes),
                "mean_absolute_change": mean(
                    [abs(value) for value in model_margin_changes]
                ),
                "median": median(model_margin_changes),
            },
        },
        comparison_rows,
    )


def audit_checks(
    *,
    timing: dict[str, int],
    hash_counts: dict[str, int],
    triggers: dict[str, str],
    runs: list[dict[str, Any]],
    first_rows: list[dict[str, Any]],
    latest_rows: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, bool]:
    alignment = audit.audit_run_alignment(runs)
    required_triggers_present = (
        audit.REQUIRED_IMMUTABILITY_TRIGGERS <= set(triggers)
    )
    return {
        "all_stored_snapshots_strictly_pre_kickoff": (
            timing["at_or_after_kickoff"] == 0
            and timing["missing_or_invalid_time"] == 0
        ),
        "all_stored_record_hashes_verified": (
            hash_counts["verified"] == hash_counts["total"]
        ),
        "normal_immutability_triggers_present": required_triggers_present,
        "run_record_counts_aligned": (
            alignment["row_count_mismatch_runs"] == 0
        ),
        "first_selected_records_have_verified_hashes": all(
            bool(row.get("record_hash_verified")) for row in first_rows
        ),
        "latest_selected_records_have_verified_hashes": all(
            bool(row.get("record_hash_verified")) for row in latest_rows
        ),
        "both_policies_cover_the_same_games": (
            comparison["missing_from_first_policy"] == 0
            and comparison["missing_from_latest_policy"] == 0
        ),
    }


def build_payload(season: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = audit.load_audit_data(season)
    snapshots = data["snapshots"]
    timing, strictly_pre_kickoff = audit.audit_timing(snapshots)
    hash_counts = audit.audit_record_hashes(snapshots)

    first_rows = audit.choose_snapshots(strictly_pre_kickoff, "first")
    latest_rows = audit.choose_snapshots(strictly_pre_kickoff, "latest")

    first_summary = fixed_policy_summary("first", first_rows)
    latest_summary = fixed_policy_summary("latest", latest_rows)
    selection, comparison_rows = selection_comparison(first_rows, latest_rows)
    common_score_comparison = score_equivalence_metrics(first_rows, latest_rows)
    checks = audit_checks(
        timing=timing,
        hash_counts=hash_counts,
        triggers=data["triggers"],
        runs=data["runs"],
        first_rows=first_rows,
        latest_rows=latest_rows,
        comparison=selection,
    )
    audit_pass = all(checks.values())

    return (
        {
            "schema_version": 1,
            "generated_utc": utc_iso(),
            "study_title": (
                "Beyond Hindsight: Building an Auditable Neural Decision-Support "
                "System for Real-Time Forecasting"
            ),
            "mode": "READ_ONLY_POLICY_ROBUSTNESS_NO_API_CALLS_NO_MODEL_CHANGES",
            "season": season,
            "scope_note": (
                "College football is the live evaluation setting. This report compares "
                "two fixed pre-kickoff snapshot policies and does not recommend wagers, "
                "claim profitability, or select a policy based on performance."
            ),
            "predeclared_policies": {
                "first": first_summary["policy_definition"],
                "latest": latest_summary["policy_definition"],
            },
            "audit_pass": audit_pass,
            "audit_checks": checks,
            "snapshot_evidence": {
                "stored_snapshots": len(snapshots),
                "unique_games_with_snapshots": len(
                    {
                        str(row.get("game_key") or "")
                        for row in snapshots
                        if str(row.get("game_key") or "")
                    }
                ),
                "time_integrity": timing,
                "record_integrity": hash_counts,
                "run_alignment": audit.audit_run_alignment(data["runs"]),
                "immutability_guard_names_found": sorted(data["triggers"]),
                "required_immutability_guard_names": sorted(
                    audit.REQUIRED_IMMUTABILITY_TRIGGERS
                ),
            },
            "policy_summaries": {
                "first": first_summary,
                "latest": latest_summary,
            },
            "selection_comparison": selection,
            "common_scored_game_comparison": common_score_comparison,
            "research_interpretation": {
                "finding": (
                    "The report compares first and latest strictly pre-kickoff snapshots "
                    "without selecting a result-dependent record. Differences in outcomes, "
                    "if any, describe timing sensitivity rather than establish a better "
                    "forecasting or wagering policy."
                ),
                "limits": [
                    "This is a prospective evidence audit, not a randomized trial.",
                    "The report does not choose the better-performing policy after results are known.",
                    "A high winner-selection rate does not by itself establish reliable decision advantage.",
                    "Margin accuracy and comparison to the captured public baseline remain necessary context.",
                    "No profitability, automated-action, or causal claim is made by this report.",
                ],
            },
        },
        comparison_rows,
    )


def write_outputs(payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    fields = [
        "game_key",
        "season",
        "week",
        "away_team",
        "home_team",
        "kickoff_utc",
        "same_snapshot_selected",
        "first_snapshot_id",
        "first_captured_utc",
        "first_minutes_before_kickoff",
        "first_model_favorite",
        "first_model_home_margin",
        "first_scored_utc",
        "first_winner_correct",
        "first_absolute_margin_error",
        "first_hardrock_absolute_margin_error",
        "latest_snapshot_id",
        "latest_captured_utc",
        "latest_minutes_before_kickoff",
        "latest_model_favorite",
        "latest_model_home_margin",
        "latest_scored_utc",
        "latest_winner_correct",
        "latest_absolute_margin_error",
        "latest_hardrock_absolute_margin_error",
        "capture_gap_minutes_latest_after_first",
        "model_home_margin_change_latest_minus_first",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_policy_metrics(label: str, summary: dict[str, Any]) -> None:
    timing = summary["capture_timing_minutes_before_kickoff"]
    metrics = summary["descriptive_outcomes"]
    print(label)
    rule("-")
    print(f"Games selected:                   {summary['selected_games']}")
    print(f"Completed games scored:           {summary['selected_scored_games']}")
    print(f"Awaiting/unmatched games:         {summary['selected_awaiting_or_unmatched_games']}")
    print(f"Selected hashes verified:         {summary['selected_hashes_verified']}/{summary['selected_games']}")
    print(f"Median capture lead time:         {fmt(timing['median'], 1)} min before kickoff")
    print(f"Winner accuracy:                  {pct(metrics['winner_accuracy'])}")
    print(f"Brier score:                      {fmt(metrics['brier_score'], 3)}")
    print(f"Model margin MAE:                 {fmt(metrics['model_margin_mae'], 2)} pts")
    print(f"Hard Rock margin MAE:             {fmt(metrics['hardrock_margin_mae'], 2)} pts")
    print(f"Model minus Hard Rock MAE:        {fmt(metrics['model_minus_hardrock_mae'], 2)} pts")
    print()


def print_summary(payload: dict[str, Any]) -> None:
    evidence = payload["snapshot_evidence"]
    timing = evidence["time_integrity"]
    hashes = evidence["record_integrity"]
    selection = payload["selection_comparison"]
    common = payload["common_scored_game_comparison"]
    deltas = common["latest_minus_first"]

    rule()
    print("CFB EDGE ENGINE — RESEARCH SNAPSHOT POLICY ROBUSTNESS V1")
    rule()
    print(f"Mode:   {payload['mode']}")
    print(f"Season: {payload['season']}")
    print("Policies: FIRST and LATEST strictly pre-kickoff snapshots per game")
    print()

    print("SHARED EVIDENCE INTEGRITY")
    rule("-")
    print(f"Stored prediction snapshots:      {timing['total']}")
    print(f"Captured before kickoff:          {timing['pre_kickoff']}")
    print(f"At/after kickoff:                 {timing['at_or_after_kickoff']}")
    print(f"Missing/invalid timing:           {timing['missing_or_invalid_time']}")
    print(f"Stored record hashes verified:    {hashes['verified']}/{hashes['total']}")
    print(f"Hash mismatches:                  {hashes['hash_mismatch']}")
    print(f"Immutability guards present:      {payload['audit_checks']['normal_immutability_triggers_present']}")
    print(f"Run row-count alignment:          {payload['audit_checks']['run_record_counts_aligned']}")
    print()

    print_policy_metrics("FIRST PRE-KICKOFF POLICY", payload["policy_summaries"]["first"])
    print_policy_metrics("LATEST PRE-KICKOFF POLICY", payload["policy_summaries"]["latest"])

    print("SELECTION COMPARISON")
    rule("-")
    print(f"Games compared:                   {selection['games_compared']}")
    print(f"Same snapshot selected:           {selection['same_snapshot_selected']}")
    print(f"Different snapshot selected:      {selection['different_snapshot_selected']}")
    print(
        "Median latest-after-first gap:    "
        f"{fmt(selection['latest_capture_after_first_minutes']['median'], 1)} min"
    )
    print(
        "Mean absolute model-margin change: "
        f"{fmt(selection['model_home_margin_change_latest_minus_first']['mean_absolute_change'], 2)} pts"
    )
    print()

    print("COMMON SCORED-GAME SENSITIVITY")
    rule("-")
    print(f"Common scored games:              {common['common_scored_games']}")
    print(f"Latest minus first winner accuracy:{pct(deltas['winner_accuracy'])}")
    print(f"Latest minus first Brier score:   {fmt(deltas['brier_score'], 3)}")
    print(f"Latest minus first model MAE:     {fmt(deltas['model_margin_mae'], 2)} pts")
    print(f"Latest minus first HR MAE:        {fmt(deltas['hardrock_margin_mae'], 2)} pts")
    print()

    print("RESEARCH CONCLUSION")
    rule("-")
    print("PASS" if payload["audit_pass"] else "REVIEW REQUIRED")
    print(payload["research_interpretation"]["finding"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed first and latest pre-kickoff CFB prediction snapshots "
            "without API calls or model changes."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season to evaluate (default: 2026).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the report without writing derived JSON/CSV outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload, comparison_rows = build_payload(args.season)
    print_summary(payload)

    if args.dry_run:
        print()
        print("DRY RUN COMPLETE — no derived research files were written.")
    else:
        write_outputs(payload, comparison_rows)
        print()
        print("FILES WRITTEN")
        rule("-")
        print(OUTPUT_JSON)
        print(OUTPUT_CSV)

    return 0 if payload["audit_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
