#!/usr/bin/env python3
"""
CFB Edge Engine — Model vs Market Performance V1

Purpose
-------
Evaluate clean, strictly scored prospective predictions against the captured
Hard Rock market.

This script is descriptive only. It does NOT:
- invent wagers
- treat diagnostic ATS results as real bets
- call any API
- retrain or modify the neural network
- modify prediction history
- modify wager history

Default snapshot policy is LATEST: for each completed game, evaluate the most
recent immutable prediction captured before kickoff.

Outputs
-------
data/processed/model_market_performance_2026_v1.json
data/processed/model_market_performance_2026_v1.csv
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_DB = (
    PROJECT_ROOT
    / "data"
    / "prospective"
    / "prediction_history.sqlite"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_market_performance_2026_v1.json"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_market_performance_2026_v1.csv"
)

SEASON = 2026


# ============================================================================
# HELPERS
# ============================================================================

def utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


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
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_team(value: Any) -> str:
    return " ".join(
        str(value or "")
        .strip()
        .casefold()
        .split()
    )


def same_team(left: Any, right: Any) -> bool:
    a = normalize_team(left)
    b = normalize_team(right)
    return bool(a and b and a == b)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def rmse(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(
        sum(value * value for value in values)
        / len(values)
    )


def rounded(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def percent(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def record_string(
    wins: int,
    losses: int,
    pushes: int,
) -> str:
    return f"{wins}-{losses}-{pushes}"


# ============================================================================
# DATA LOADING
# ============================================================================

def connect_history_read_only() -> sqlite3.Connection:
    if not HISTORY_DB.exists():
        raise FileNotFoundError(
            f"Prediction-history database not found: {HISTORY_DB}"
        )

    connection = sqlite3.connect(
        f"file:{HISTORY_DB}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute(
        "PRAGMA query_only = ON"
    )
    connection.execute(
        "PRAGMA busy_timeout = 5000"
    )
    return connection


def load_scored_rows(
    season: int,
) -> list[dict[str, Any]]:
    connection = connect_history_read_only()

    try:
        rows = connection.execute(
            """
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
            JOIN prediction_scores s
              ON s.snapshot_id = p.snapshot_id
            JOIN game_results r
              ON r.game_key = p.game_key
            WHERE p.season = ?
            ORDER BY
                p.kickoff_utc,
                p.captured_utc,
                p.snapshot_id
            """,
            (
                season,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        connection.close()


# ============================================================================
# SNAPSHOT POLICY
# ============================================================================

def apply_policy(
    rows: list[dict[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    policy = policy.lower()

    if policy == "all":
        return list(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = str(
            row.get("game_key") or ""
        )

        if key:
            grouped[key].append(row)

    selected = []

    for game_rows in grouped.values():
        ordered = sorted(
            game_rows,
            key=lambda row: (
                str(
                    row.get("captured_utc")
                    or ""
                ),
                str(
                    row.get("snapshot_id")
                    or ""
                ),
            ),
        )

        if policy == "first":
            selected.append(
                ordered[0]
            )

        else:
            selected.append(
                ordered[-1]
            )

    return sorted(
        selected,
        key=lambda row: (
            str(
                row.get("kickoff_utc")
                or ""
            ),
            str(
                row.get("game_key")
                or ""
            ),
        ),
    )


# ============================================================================
# HARD ROCK FAVORITE
# ============================================================================

def hardrock_favorite(
    row: dict[str, Any],
) -> str | None:
    spread = safe_float(
        row.get(
            "hardrock_spread_for_model_favorite"
        )
    )

    if (
        spread is None
        or
        math.isclose(
            spread,
            0.0,
            abs_tol=1e-9,
        )
    ):
        return None

    model_favorite = str(
        row.get(
            "model_favorite"
        )
        or ""
    ).strip()

    home = str(
        row.get(
            "home_team"
        )
        or ""
    ).strip()

    away = str(
        row.get(
            "away_team"
        )
        or ""
    ).strip()

    if same_team(
        model_favorite,
        home,
    ):
        opponent = away

    elif same_team(
        model_favorite,
        away,
    ):
        opponent = home

    else:
        return None

    if spread < 0:
        return model_favorite

    return opponent


def hardrock_favorite_correct(
    row: dict[str, Any],
) -> int | None:
    favorite = hardrock_favorite(
        row
    )

    actual = str(
        row.get(
            "actual_winner"
        )
        or ""
    ).strip()

    if not favorite or not actual:
        return None

    return int(
        same_team(
            favorite,
            actual,
        )
    )


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(rows)

    model_winner_results = [
        safe_int(
            row.get(
                "winner_correct"
            )
        )
        for row in rows
    ]

    model_winner_results = [
        value
        for value in model_winner_results
        if value in {
            0,
            1,
        }
    ]

    model_winner_correct = sum(
        model_winner_results
    )

    hardrock_results = [
        hardrock_favorite_correct(
            row
        )
        for row in rows
    ]

    hardrock_results = [
        value
        for value in hardrock_results
        if value in {
            0,
            1,
        }
    ]

    hardrock_correct = sum(
        hardrock_results
    )

    model_abs_errors = [
        value
        for row in rows
        if (
            value := safe_float(
                row.get(
                    "absolute_margin_error"
                )
            )
        )
        is not None
    ]

    model_signed_errors = [
        value
        for row in rows
        if (
            value := safe_float(
                row.get(
                    "margin_error"
                )
            )
        )
        is not None
    ]

    hardrock_abs_errors = [
        value
        for row in rows
        if (
            value := safe_float(
                row.get(
                    "hardrock_absolute_margin_error"
                )
            )
        )
        is not None
    ]

    brier_values = [
        value
        for row in rows
        if (
            value := safe_float(
                row.get(
                    "brier_score"
                )
            )
        )
        is not None
    ]

    ats_results = [
        str(
            row.get(
                "model_side_ats_result"
            )
            or ""
        ).upper()
        for row in rows
    ]

    ats_w = sum(
        result in {
            "W",
            "WIN",
            "WON",
        }
        for result in ats_results
    )

    ats_l = sum(
        result in {
            "L",
            "LOSS",
            "LOST",
        }
        for result in ats_results
    )

    ats_p = sum(
        result in {
            "P",
            "PUSH",
        }
        for result in ats_results
    )

    hr_comparison_rows = [
        row
        for row in rows
        if (
            safe_float(
                row.get(
                    "absolute_margin_error"
                )
            )
            is not None
            and
            safe_float(
                row.get(
                    "hardrock_absolute_margin_error"
                )
            )
            is not None
        )
    ]

    model_better = 0
    hardrock_better = 0
    margin_ties = 0

    for row in hr_comparison_rows:
        model_error = float(
            row[
                "absolute_margin_error"
            ]
        )

        hardrock_error = float(
            row[
                "hardrock_absolute_margin_error"
            ]
        )

        if math.isclose(
            model_error,
            hardrock_error,
            abs_tol=1e-9,
        ):
            margin_ties += 1

        elif model_error < hardrock_error:
            model_better += 1

        else:
            hardrock_better += 1

    return {
        "games": n,
        "model_winner_games": len(
            model_winner_results
        ),
        "model_winner_correct": model_winner_correct,
        "model_winner_accuracy": rounded(
            percent(
                model_winner_correct,
                len(
                    model_winner_results
                ),
            )
        ),
        "hardrock_favorite_games": len(
            hardrock_results
        ),
        "hardrock_favorite_correct": hardrock_correct,
        "hardrock_favorite_accuracy": rounded(
            percent(
                hardrock_correct,
                len(
                    hardrock_results
                ),
            )
        ),
        "model_margin_mae": rounded(
            mean(
                model_abs_errors
            ),
            3,
        ),
        "model_margin_rmse": rounded(
            rmse(
                model_signed_errors
            ),
            3,
        ),
        "hardrock_margin_mae": rounded(
            mean(
                hardrock_abs_errors
            ),
            3,
        ),
        "brier_score": rounded(
            mean(
                brier_values
            ),
            4,
        ),
        "model_side_spread_record": record_string(
            ats_w,
            ats_l,
            ats_p,
        ),
        "model_side_spread_wins": ats_w,
        "model_side_spread_losses": ats_l,
        "model_side_spread_pushes": ats_p,
        "margin_error_head_to_head": {
            "comparable_games": len(
                hr_comparison_rows
            ),
            "model_better": model_better,
            "hardrock_better": hardrock_better,
            "ties": margin_ties,
        },
    }


def group_by_strength(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups = {
        "HIGH": [],
        "MEDIUM": [],
        "LOW": [],
        "N/A": [],
    }

    for row in rows:
        strength = str(
            row.get(
                "strength_label"
            )
            or "N/A"
        ).upper()

        if strength not in groups:
            strength = "N/A"

        groups[
            strength
        ].append(
            row
        )

    return {
        key: aggregate(
            group
        )
        for key, group in groups.items()
    }


def edge_bucket(
    row: dict[str, Any],
) -> str:
    edge = safe_float(
        row.get(
            "model_edge_points"
        )
    )

    if edge is None:
        return "N/A"

    if edge < 0:
        return "NO_POSITIVE_EDGE"

    if edge < 1:
        return "0_TO_0.9"

    if edge < 3:
        return "1_TO_2.9"

    if edge < 6:
        return "3_TO_5.9"

    return "6_PLUS"


def group_by_edge(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        groups[
            edge_bucket(
                row
            )
        ].append(
            row
        )

    order = [
        "6_PLUS",
        "3_TO_5.9",
        "1_TO_2.9",
        "0_TO_0.9",
        "NO_POSITIVE_EDGE",
        "N/A",
    ]

    return {
        key: aggregate(
            groups.get(
                key,
                [],
            )
        )
        for key in order
    }


def group_by_week(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        week = safe_int(
            row.get(
                "week"
            )
        )

        if week is not None:
            groups[
                week
            ].append(
                row
            )

    return {
        str(
            week
        ): aggregate(
            groups[
                week
            ]
        )
        for week in sorted(
            groups
        )
    }


# ============================================================================
# ROW EXPORT
# ============================================================================

def export_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    hr_favorite = hardrock_favorite(
        row
    )

    hr_correct = hardrock_favorite_correct(
        row
    )

    return {
        "game_key": row.get(
            "game_key"
        ),
        "game_id": row.get(
            "game_id"
        ),
        "week": row.get(
            "week"
        ),
        "kickoff_utc": row.get(
            "kickoff_utc"
        ),
        "captured_utc": row.get(
            "captured_utc"
        ),
        "away_team": row.get(
            "away_team"
        ),
        "home_team": row.get(
            "home_team"
        ),
        "actual_winner": row.get(
            "actual_winner"
        ),
        "model_favorite": row.get(
            "model_favorite"
        ),
        "model_spread_display": row.get(
            "model_spread_display"
        ),
        "model_winner_correct": row.get(
            "winner_correct"
        ),
        "model_absolute_margin_error": row.get(
            "absolute_margin_error"
        ),
        "brier_score": row.get(
            "brier_score"
        ),
        "hardrock_display": row.get(
            "hardrock_display"
        ),
        "hardrock_favorite": hr_favorite,
        "hardrock_favorite_correct": hr_correct,
        "hardrock_absolute_margin_error": row.get(
            "hardrock_absolute_margin_error"
        ),
        "model_side_spread_result": row.get(
            "model_side_ats_result"
        ),
        "model_edge_points": row.get(
            "model_edge_points"
        ),
        "strength_label": row.get(
            "strength_label"
        ),
        "signal_warning": row.get(
            "signal_warning"
        ),
    }


# ============================================================================
# REPORT
# ============================================================================

def build_report(
    rows: list[dict[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_utc": utc_iso(),
        "season": SEASON,
        "snapshot_policy": policy.upper(),
        "sample_warning": (
            "Prospective sample sizes may be very small. "
            "Descriptive performance is not evidence of a profitable strategy."
        ),
        "overall": aggregate(
            rows
        ),
        "by_strength": group_by_strength(
            rows
        ),
        "by_model_edge": group_by_edge(
            rows
        ),
        "by_week": group_by_week(
            rows
        ),
        "rows": [
            export_row(
                row
            )
            for row in rows
        ],
        "notes": [
            (
                "Hard Rock favorite accuracy is based only on games with a "
                "usable non-pick'em captured Hard Rock spread."
            ),
            (
                "Model-side spread record is diagnostic only and must not be "
                "reported as an actual betting record unless a real wager was logged."
            ),
            (
                "Lower MAE and Brier are better."
            ),
            (
                "Margin-error head-to-head counts whether the model or the "
                "captured Hard Rock line was closer to the final scoring margin."
            ),
        ],
    }


def write_outputs(
    report: dict[str, Any],
) -> None:
    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            default=str,
        )
        handle.write(
            "\n"
        )

    rows = report.get(
        "rows"
    )

    if not isinstance(
        rows,
        list,
    ):
        rows = []

    fieldnames = [
        "game_key",
        "game_id",
        "week",
        "kickoff_utc",
        "captured_utc",
        "away_team",
        "home_team",
        "actual_winner",
        "model_favorite",
        "model_spread_display",
        "model_winner_correct",
        "model_absolute_margin_error",
        "brier_score",
        "hardrock_display",
        "hardrock_favorite",
        "hardrock_favorite_correct",
        "hardrock_absolute_margin_error",
        "model_side_spread_result",
        "model_edge_points",
        "strength_label",
        "signal_warning",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================================
# CONSOLE
# ============================================================================

def fmt_pct(
    value: Any,
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "N/A"

    return f"{number * 100:.1f}%"


def fmt_num(
    value: Any,
    digits: int = 2,
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "N/A"

    return f"{number:.{digits}f}"


def print_summary(
    report: dict[str, Any],
) -> None:
    overall = report[
        "overall"
    ]

    print(
        "=" * 108
    )

    print(
        "CFB EDGE ENGINE — MODEL VS MARKET PERFORMANCE V1"
    )

    print(
        "=" * 108
    )

    print(
        f"Season:            {report['season']}"
    )

    print(
        f"Snapshot policy:   {report['snapshot_policy']}"
    )

    print(
        f"Scored games:      {overall['games']}"
    )

    print(
        "Mode:              READ ONLY / DESCRIPTIVE"
    )

    print(
        "=" * 108
    )

    print()

    print(
        "OVERALL"
    )

    print(
        "-" * 108
    )

    print(
        f"Model winner accuracy:       "
        f"{fmt_pct(overall['model_winner_accuracy'])} "
        f"({overall['model_winner_correct']}/"
        f"{overall['model_winner_games']})"
    )

    print(
        f"Hard Rock favorite accuracy: "
        f"{fmt_pct(overall['hardrock_favorite_accuracy'])} "
        f"({overall['hardrock_favorite_correct']}/"
        f"{overall['hardrock_favorite_games']})"
    )

    print(
        f"Model margin MAE:            "
        f"{fmt_num(overall['model_margin_mae'])} pts"
    )

    print(
        f"Hard Rock margin MAE:        "
        f"{fmt_num(overall['hardrock_margin_mae'])} pts"
    )

    print(
        f"Model margin RMSE:           "
        f"{fmt_num(overall['model_margin_rmse'])} pts"
    )

    print(
        f"Model Brier score:           "
        f"{fmt_num(overall['brier_score'], 3)}"
    )

    print(
        f"Model-side spread diagnostic:"
        f" {overall['model_side_spread_record']}"
    )

    head = overall[
        "margin_error_head_to_head"
    ]

    print(
        f"Closer to final margin:      "
        f"Model {head['model_better']} | "
        f"Hard Rock {head['hardrock_better']} | "
        f"Ties {head['ties']}"
    )

    print()

    print(
        "BY STRENGTH"
    )

    print(
        "-" * 108
    )

    for strength in (
        "HIGH",
        "MEDIUM",
        "LOW",
        "N/A",
    ):
        row = report[
            "by_strength"
        ][
            strength
        ]

        print(
            f"{strength:<7} "
            f"n={row['games']:<3} | "
            f"Model win={fmt_pct(row['model_winner_accuracy']):>6} | "
            f"HR fav={fmt_pct(row['hardrock_favorite_accuracy']):>6} | "
            f"Model MAE={fmt_num(row['model_margin_mae']):>6} | "
            f"HR MAE={fmt_num(row['hardrock_margin_mae']):>6} | "
            f"Spread diag={row['model_side_spread_record']}"
        )

    print()

    print(
        "FILES WRITTEN"
    )

    print(
        "-" * 108
    )

    print(
        OUTPUT_JSON
    )

    print(
        OUTPUT_CSV
    )

    print()

    print(
        "IMPORTANT: spread results above are diagnostic model-vs-line results, "
        "not an actual wager record."
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build read-only model-vs-market prospective performance metrics."
        )
    )

    parser.add_argument(
        "--policy",
        choices=(
            "latest",
            "first",
            "all",
        ),
        default="latest",
        help=(
            "Which immutable prediction snapshot to evaluate for each game."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rows = load_scored_rows(
        SEASON
    )

    selected = apply_policy(
        rows,
        args.policy,
    )

    report = build_report(
        selected,
        policy=args.policy,
    )

    write_outputs(
        report
    )

    print_summary(
        report
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
