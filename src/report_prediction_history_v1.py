#!/usr/bin/env python3
"""Generate human-readable prospective evaluation reports from prediction history."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from prediction_history_v1 import HISTORY_DB, PROCESSED_DIR, connect_history, safe_float, utc_iso


def print_rule(char: str = "=", width: int = 104) -> None:
    print(char * width)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rmse(values: list[float]) -> float | None:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def fmt(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def choose_snapshots(rows: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    if policy == "all":
        return rows

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


def metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margin_errors = [
        float(row["margin_error"])
        for row in rows
        if safe_float(row.get("margin_error")) is not None
    ]
    abs_errors = [abs(value) for value in margin_errors]

    winner_values = [
        int(row["winner_correct"])
        for row in rows
        if row.get("winner_correct") is not None
    ]
    brier_values = [
        float(row["brier_score"])
        for row in rows
        if safe_float(row.get("brier_score")) is not None
    ]
    hardrock_abs = [
        float(row["hardrock_absolute_margin_error"])
        for row in rows
        if safe_float(row.get("hardrock_absolute_margin_error")) is not None
    ]

    ats_rows = [row for row in rows if row.get("model_side_ats_result") in ("W", "L", "P")]
    ats_wins = sum(row.get("model_side_ats_result") == "W" for row in ats_rows)
    ats_losses = sum(row.get("model_side_ats_result") == "L" for row in ats_rows)
    ats_pushes = sum(row.get("model_side_ats_result") == "P" for row in ats_rows)
    unit_values = [
        float(row["model_side_flat_units"])
        for row in ats_rows
        if safe_float(row.get("model_side_flat_units")) is not None
    ]
    unit_decisions = sum(row.get("model_side_ats_result") in ("W", "L") for row in ats_rows)
    units = sum(unit_values) if unit_values else None
    roi = units / unit_decisions if units is not None and unit_decisions else None

    model_mae = mean(abs_errors)
    hardrock_mae = mean(hardrock_abs)

    return {
        "games": len(rows),
        "margin_games": len(margin_errors),
        "model_margin_mae": model_mae,
        "model_margin_rmse": rmse(margin_errors),
        "winner_games": len(winner_values),
        "winner_accuracy": mean([float(value) for value in winner_values]),
        "brier_games": len(brier_values),
        "brier_score": mean(brier_values),
        "hardrock_margin_games": len(hardrock_abs),
        "hardrock_margin_mae": hardrock_mae,
        "model_minus_hardrock_mae": (
            model_mae - hardrock_mae
            if model_mae is not None and hardrock_mae is not None
            else None
        ),
        "model_side_ats_games": len(ats_rows),
        "model_side_ats_wins": ats_wins,
        "model_side_ats_losses": ats_losses,
        "model_side_ats_pushes": ats_pushes,
        "model_side_flat_units": units,
        "model_side_flat_roi": roi,
    }


def load_scored_rows() -> list[dict[str, Any]]:
    connection = connect_history()
    try:
        sql = """
            SELECT
                p.*,
                s.actual_home_margin,
                s.predicted_home_margin,
                s.margin_error,
                s.absolute_margin_error,
                s.squared_margin_error,
                s.winner_correct,
                s.brier_score,
                s.hardrock_implied_home_margin,
                s.hardrock_absolute_margin_error,
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
        return [dict(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "game_key", "game_id", "season", "week", "kickoff_utc", "captured_utc",
        "away_team", "home_team", "away_points", "home_points", "actual_winner",
        "model_favorite", "model_spread_display", "model_home_margin",
        "model_favorite_win_probability", "model_home_win_probability",
        "hardrock_display", "hardrock_spread_for_model_favorite",
        "hardrock_price_for_model_favorite", "reference_market_display",
        "model_edge_points", "strength_label", "strength_score", "signal_warning",
        "nn_input_status", "nn_feature_coverage", "nn_margin_std",
        "nn_win_probability_std", "actual_home_margin", "margin_error",
        "absolute_margin_error", "winner_correct", "brier_score",
        "hardrock_implied_home_margin", "hardrock_absolute_margin_error",
        "model_side_ats_result", "model_side_flat_units", "wager_decision",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    overall = metric_block(rows)
    by_strength: dict[str, Any] = {}
    for strength in ("HIGH", "MEDIUM", "LOW", "N/A"):
        subset = [row for row in rows if str(row.get("strength_label") or "N/A").upper() == strength]
        by_strength[strength] = metric_block(subset)

    warned = [row for row in rows if str(row.get("signal_warning") or "NONE").upper() not in ("", "NONE")]
    clean = [row for row in rows if str(row.get("signal_warning") or "NONE").upper() in ("", "NONE")]

    return {
        "generated_utc": utc_iso(),
        "snapshot_policy": policy,
        "important_note": (
            "Model-side ATS is a diagnostic of backing the model favorite at the captured Hard Rock line. "
            "It is not the result of a validated wagering rule unless wager_decision later says so."
        ),
        "overall": overall,
        "by_strength": by_strength,
        "warning_split": {
            "no_warning": metric_block(clean),
            "warning_present": metric_block(warned),
        },
    }


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(label)
    print_rule("-")
    print(f"Games scored:              {metrics['games']}")
    print(f"Winner accuracy:           {fmt(metrics['winner_accuracy'], 3)}")
    print(f"Brier score:               {fmt(metrics['brier_score'], 3)}")
    print(f"Model spread MAE:          {fmt(metrics['model_margin_mae'], 2)} pts")
    print(f"Model spread RMSE:         {fmt(metrics['model_margin_rmse'], 2)} pts")
    print(f"Hard Rock implied MAE:     {fmt(metrics['hardrock_margin_mae'], 2)} pts")
    print(f"Model minus Hard Rock MAE: {fmt(metrics['model_minus_hardrock_mae'], 2)} pts")
    print(
        "Model-side ATS diagnostic:  "
        f"{metrics['model_side_ats_wins']}-{metrics['model_side_ats_losses']}-{metrics['model_side_ats_pushes']}"
    )
    print(f"Model-side flat units:     {fmt(metrics['model_side_flat_units'], 2)}")
    print(f"Model-side flat ROI:       {fmt(metrics['model_side_flat_roi'], 3)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report prospective CFB model performance.")
    parser.add_argument(
        "--policy",
        choices=("latest", "first", "all"),
        default="latest",
        help=(
            "How to handle multiple pre-kickoff snapshots per game. "
            "latest=last prediction before kickoff, first=earliest recorded prediction, all=every snapshot."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows = load_scored_rows()
    selected = choose_snapshots(all_rows, args.policy)
    summary = build_summary(selected, args.policy)

    csv_path = PROCESSED_DIR / f"prediction_history_evaluation_{args.policy}.csv"
    json_path = PROCESSED_DIR / f"prediction_history_summary_{args.policy}.json"
    write_csv(selected, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_rule()
    print("PREDICTION HISTORY REPORT — V1")
    print_rule()
    print(f"History DB:       {HISTORY_DB}")
    print(f"Snapshot policy:  {args.policy.upper()}")
    print(f"Scored snapshots: {len(selected)}")
    print_rule()
    print()
    print_metrics("OVERALL", summary["overall"])

    print()
    print("BY STRENGTH")
    print_rule("-")
    for strength in ("HIGH", "MEDIUM", "LOW", "N/A"):
        block = summary["by_strength"][strength]
        print(
            f"{strength:<6} | n={block['games']:<3} | "
            f"win acc={fmt(block['winner_accuracy'], 3):>5} | "
            f"MAE={fmt(block['model_margin_mae'], 2):>5} | "
            f"Brier={fmt(block['brier_score'], 3):>5} | "
            f"ATS={block['model_side_ats_wins']}-{block['model_side_ats_losses']}-{block['model_side_ats_pushes']}"
        )

    print()
    print("FILES WRITTEN")
    print_rule("-")
    print(csv_path)
    print(json_path)
    print()
    print("Lower MAE/Brier is better. Model-minus-Hard-Rock MAE below zero means the model beat the captured Hard Rock line on margin error for that sample.")
    print("The ATS line is diagnostic only; we still do not have a validated automatic betting rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
