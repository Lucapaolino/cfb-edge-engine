#!/usr/bin/env python3
"""
CFB Edge Engine — Weekly Recap V1

Purpose
-------
Create a read-only weekly report from strictly scored prospective predictions
and the separate Actual Wager Ledger.

This script combines:
- model vs market performance
- actual wager performance
- Strength-bucket results
- biggest model hit / miss
- largest model-vs-market disagreement
- largest captured model edge
- optional CLV statistics from real wagers when available

Important
---------
Prediction-history spread results remain DIAGNOSTIC ONLY.
Only tickets stored in wager_ledger.sqlite count as actual wagers.

No API calls are made. No production data is modified.

Outputs
-------
data/processed/weekly_recap_2026_v1.json
data/processed/weekly_recap_2026_v1.csv
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

from model_market_performance_v1 import (
    SEASON,
    aggregate,
    apply_policy,
    hardrock_favorite,
    load_scored_rows,
    safe_float,
    safe_int,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WAGER_DB = (
    PROJECT_ROOT
    / "data"
    / "prospective"
    / "wager_ledger.sqlite"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weekly_recap_2026_v1.json"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weekly_recap_2026_v1.csv"
)


# ============================================================================
# HELPERS
# ============================================================================

def utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


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


def rounded(
    value: float | None,
    digits: int = 3,
) -> float | None:
    if value is None:
        return None
    return round(
        value,
        digits,
    )


def mean(
    values: list[float],
) -> float | None:
    if not values:
        return None
    return sum(
        values
    ) / len(
        values
    )


def game_name(
    row: dict[str, Any],
) -> str:
    away = str(
        row.get(
            "away_team"
        )
        or ""
    ).strip()

    home = str(
        row.get(
            "home_team"
        )
        or ""
    ).strip()

    if away or home:
        return f"{away} @ {home}".strip()

    return str(
        row.get(
            "game_key"
        )
        or "Unknown game"
    )


def actual_final(
    row: dict[str, Any],
) -> str:
    away = str(
        row.get(
            "away_team"
        )
        or ""
    ).strip()

    home = str(
        row.get(
            "home_team"
        )
        or ""
    ).strip()

    away_points = safe_float(
        row.get(
            "away_points"
        )
    )

    home_points = safe_float(
        row.get(
            "home_points"
        )
    )

    if (
        away_points is None
        or
        home_points is None
    ):
        return "N/A"

    return (
        f"{away} {away_points:g} - "
        f"{home_points:g} {home}"
    )


def metric_record(
    wins: int,
    losses: int,
    pushes: int,
) -> str:
    return (
        f"{wins}-{losses}-{pushes}"
    )


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
    suffix: str = "",
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "N/A"

    return (
        f"{number:.{digits}f}"
        f"{suffix}"
    )


def fmt_money(
    value: Any,
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "N/A"

    sign = "+" if number > 0 else ""

    return (
        f"{sign}${number:,.2f}"
    )


# ============================================================================
# ACTUAL WAGER ACCESS
# ============================================================================

def table_exists(
    connection: sqlite3.Connection,
    table: str,
) -> bool:
    return (
        connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table'
              AND name=?
            LIMIT 1
            """,
            (
                table,
            ),
        ).fetchone()
        is not None
    )


def load_actual_wagers() -> list[dict[str, Any]]:
    if not WAGER_DB.exists():
        return []

    connection = sqlite3.connect(
        f"file:{WAGER_DB}?mode=ro",
        uri=True,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA query_only = ON"
    )

    try:
        if not table_exists(
            connection,
            "wagers",
        ):
            return []

        rows = connection.execute(
            """
            SELECT *
            FROM wagers
            WHERE season = ?
            ORDER BY placed_utc, wager_id
            """,
            (
                SEASON,
            ),
        ).fetchall()

        return [
            dict(
                row
            )
            for row in rows
        ]

    finally:
        connection.close()


def wagers_for_week(
    wagers: list[dict[str, Any]],
    week: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in wagers
        if safe_int(
            row.get(
                "week"
            )
        )
        ==
        week
    ]


def wager_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    settled = []

    pending = 0

    for row in rows:
        status = str(
            row.get(
                "status"
            )
            or ""
        ).upper()

        if status == "PENDING":
            pending += 1

        if status in {
            "WON",
            "LOST",
            "PUSH",
        }:
            settled.append(
                row
            )

    wins = sum(
        str(
            row.get(
                "status"
            )
            or ""
        ).upper()
        ==
        "WON"
        for row in settled
    )

    losses = sum(
        str(
            row.get(
                "status"
            )
            or ""
        ).upper()
        ==
        "LOST"
        for row in settled
    )

    pushes = sum(
        str(
            row.get(
                "status"
            )
            or ""
        ).upper()
        ==
        "PUSH"
        for row in settled
    )

    risked = sum(
        safe_float(
            row.get(
                "stake"
            )
        )
        or
        0.0
        for row in settled
    )

    profit = sum(
        safe_float(
            row.get(
                "profit_loss"
            )
        )
        or
        0.0
        for row in settled
    )

    roi = (
        profit / risked
        if risked > 0
        else None
    )

    clv_values = [
        value
        for row in settled
        if (
            value := safe_float(
                row.get(
                    "clv_points"
                )
            )
        )
        is not None
    ]

    high = [
        row
        for row in settled
        if str(
            row.get(
                "strength_label"
            )
            or ""
        ).upper()
        ==
        "HIGH"
    ]

    medium = [
        row
        for row in settled
        if str(
            row.get(
                "strength_label"
            )
            or ""
        ).upper()
        ==
        "MEDIUM"
    ]

    low = [
        row
        for row in settled
        if str(
            row.get(
                "strength_label"
            )
            or ""
        ).upper()
        ==
        "LOW"
    ]

    def group_record(
        group: list[dict[str, Any]],
    ) -> str:
        group_w = sum(
            str(
                row.get(
                    "status"
                )
                or ""
            ).upper()
            ==
            "WON"
            for row in group
        )

        group_l = sum(
            str(
                row.get(
                    "status"
                )
                or ""
            ).upper()
            ==
            "LOST"
            for row in group
        )

        group_p = sum(
            str(
                row.get(
                    "status"
                )
                or ""
            ).upper()
            ==
            "PUSH"
            for row in group
        )

        return metric_record(
            group_w,
            group_l,
            group_p,
        )

    return {
        "tickets": len(
            rows
        ),
        "settled": len(
            settled
        ),
        "pending": pending,
        "record": metric_record(
            wins,
            losses,
            pushes,
        ),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "amount_risked": rounded(
            risked,
            2,
        ),
        "profit_loss": rounded(
            profit,
            2,
        ),
        "roi": rounded(
            roi,
            4,
        ),
        "average_clv_points": rounded(
            mean(
                clv_values
            ),
            3,
        ),
        "clv_sample_size": len(
            clv_values
        ),
        "by_strength": {
            "HIGH": group_record(
                high
            ),
            "MEDIUM": group_record(
                medium
            ),
            "LOW": group_record(
                low
            ),
        },
    }


# ============================================================================
# WEEKLY MODEL DIAGNOSTICS
# ============================================================================

def rows_for_week(
    rows: list[dict[str, Any]],
    week: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if safe_int(
            row.get(
                "week"
            )
        )
        ==
        week
    ]


def strength_breakdown(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups = {
        "HIGH": [],
        "MEDIUM": [],
        "LOW": [],
        "N/A": [],
    }

    for row in rows:
        label = str(
            row.get(
                "strength_label"
            )
            or "N/A"
        ).upper()

        if label not in groups:
            label = "N/A"

        groups[
            label
        ].append(
            row
        )

    return {
        label: aggregate(
            group
        )
        for label, group in groups.items()
    }


def candidate_summary(
    row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if row is None:
        return None

    return {
        "game": game_name(
            row
        ),
        "game_id": row.get(
            "game_id"
        ),
        "kickoff_utc": row.get(
            "kickoff_utc"
        ),
        "model_spread": row.get(
            "model_spread_display"
        ),
        "hardrock": row.get(
            "hardrock_display"
        ),
        "reference_market": row.get(
            "reference_market_display"
        ),
        "final": actual_final(
            row
        ),
        "actual_winner": row.get(
            "actual_winner"
        ),
        "absolute_margin_error": safe_float(
            row.get(
                "absolute_margin_error"
            )
        ),
        "hardrock_absolute_margin_error": safe_float(
            row.get(
                "hardrock_absolute_margin_error"
            )
        ),
        "model_edge_points": safe_float(
            row.get(
                "model_edge_points"
            )
        ),
        "model_vs_consensus_abs_points": safe_float(
            row.get(
                "model_vs_consensus_abs_points"
            )
        ),
        "strength": str(
            row.get(
                "strength_label"
            )
            or "N/A"
        ).upper(),
        "warning": row.get(
            "signal_warning"
        ),
    }


def weekly_extremes(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    model_error_rows = [
        row
        for row in rows
        if safe_float(
            row.get(
                "absolute_margin_error"
            )
        )
        is not None
    ]

    biggest_hit = (
        min(
            model_error_rows,
            key=lambda row: float(
                row[
                    "absolute_margin_error"
                ]
            ),
        )
        if model_error_rows
        else None
    )

    biggest_miss = (
        max(
            model_error_rows,
            key=lambda row: float(
                row[
                    "absolute_margin_error"
                ]
            ),
        )
        if model_error_rows
        else None
    )

    disagreement_rows = [
        row
        for row in rows
        if safe_float(
            row.get(
                "model_vs_consensus_abs_points"
            )
        )
        is not None
    ]

    largest_disagreement = (
        max(
            disagreement_rows,
            key=lambda row: float(
                row[
                    "model_vs_consensus_abs_points"
                ]
            ),
        )
        if disagreement_rows
        else None
    )

    edge_rows = [
        row
        for row in rows
        if safe_float(
            row.get(
                "model_edge_points"
            )
        )
        is not None
    ]

    largest_edge = (
        max(
            edge_rows,
            key=lambda row: float(
                row[
                    "model_edge_points"
                ]
            ),
        )
        if edge_rows
        else None
    )

    model_beats_market_rows = []

    hardrock_beats_model_rows = []

    for row in rows:
        model_error = safe_float(
            row.get(
                "absolute_margin_error"
            )
        )

        hardrock_error = safe_float(
            row.get(
                "hardrock_absolute_margin_error"
            )
        )

        if (
            model_error is None
            or
            hardrock_error is None
        ):
            continue

        difference = (
            hardrock_error
            -
            model_error
        )

        if difference > 0:
            model_beats_market_rows.append(
                (
                    difference,
                    row,
                )
            )

        elif difference < 0:
            hardrock_beats_model_rows.append(
                (
                    abs(
                        difference
                    ),
                    row,
                )
            )

    biggest_model_advantage = (
        max(
            model_beats_market_rows,
            key=lambda item: item[0],
        )
        if model_beats_market_rows
        else None
    )

    biggest_market_advantage = (
        max(
            hardrock_beats_model_rows,
            key=lambda item: item[0],
        )
        if hardrock_beats_model_rows
        else None
    )

    return {
        "best_model_margin_prediction": candidate_summary(
            biggest_hit
        ),
        "worst_model_margin_prediction": candidate_summary(
            biggest_miss
        ),
        "largest_model_market_disagreement": candidate_summary(
            largest_disagreement
        ),
        "largest_captured_model_edge": candidate_summary(
            largest_edge
        ),
        "biggest_model_margin_advantage_over_hardrock": (
            {
                "advantage_points": rounded(
                    biggest_model_advantage[0],
                    3,
                ),
                "game": candidate_summary(
                    biggest_model_advantage[1]
                ),
            }
            if biggest_model_advantage
            else None
        ),
        "biggest_hardrock_margin_advantage_over_model": (
            {
                "advantage_points": rounded(
                    biggest_market_advantage[0],
                    3,
                ),
                "game": candidate_summary(
                    biggest_market_advantage[1]
                ),
            }
            if biggest_market_advantage
            else None
        ),
    }


# ============================================================================
# WEEK BUILDING
# ============================================================================

def build_week(
    prediction_rows: list[dict[str, Any]],
    wagers: list[dict[str, Any]],
    week: int,
) -> dict[str, Any]:
    week_rows = rows_for_week(
        prediction_rows,
        week,
    )

    week_wagers = wagers_for_week(
        wagers,
        week,
    )

    overall = aggregate(
        week_rows
    )

    strengths = strength_breakdown(
        week_rows
    )

    extremes = weekly_extremes(
        week_rows
    )

    actual_wagers = wager_summary(
        week_wagers
    )

    return {
        "week": week,
        "games_scored": len(
            week_rows
        ),
        "model_vs_market": overall,
        "by_strength": strengths,
        "actual_wagers": actual_wagers,
        "extremes": extremes,
        "sample_warning": (
            "Small weekly samples are descriptive only. "
            "Do not infer a validated wagering edge from one week."
        ),
    }


def available_weeks(
    rows: list[dict[str, Any]],
) -> list[int]:
    return sorted(
        {
            week
            for row in rows
            if (
                week := safe_int(
                    row.get(
                        "week"
                    )
                )
            )
            is not None
        }
    )


def build_report(
    prediction_rows: list[dict[str, Any]],
    wagers: list[dict[str, Any]],
    *,
    selected_week: int | None,
    policy: str,
) -> dict[str, Any]:
    weeks = available_weeks(
        prediction_rows
    )

    all_week_reports = {
        str(
            week
        ): build_week(
            prediction_rows,
            wagers,
            week,
        )
        for week in weeks
    }

    if selected_week is None:
        selected_week = (
            max(
                weeks
            )
            if weeks
            else None
        )

    selected = (
        all_week_reports.get(
            str(
                selected_week
            )
        )
        if selected_week is not None
        else None
    )

    return {
        "schema_version": 1,
        "generated_utc": utc_iso(),
        "season": SEASON,
        "snapshot_policy": policy.upper(),
        "available_weeks": weeks,
        "selected_week": selected_week,
        "selected_week_report": selected,
        "weeks": all_week_reports,
        "notes": [
            (
                "Prediction spread results are diagnostic only. "
                "Only Wager Ledger tickets count as actual wagers."
            ),
            (
                "The default LATEST snapshot policy evaluates the most recent "
                "immutable prediction captured before kickoff."
            ),
            (
                "Weekly metrics are descriptive and can be extremely noisy "
                "at small sample sizes."
            ),
        ],
    }


# ============================================================================
# OUTPUTS
# ============================================================================

def flatten_week(
    week_report: dict[str, Any],
) -> dict[str, Any]:
    model = week_report.get(
        "model_vs_market"
    ) or {}

    wagers = week_report.get(
        "actual_wagers"
    ) or {}

    head = (
        model.get(
            "margin_error_head_to_head"
        )
        or
        {}
    )

    return {
        "week": week_report.get(
            "week"
        ),
        "games_scored": week_report.get(
            "games_scored"
        ),
        "model_winner_accuracy": model.get(
            "model_winner_accuracy"
        ),
        "hardrock_favorite_accuracy": model.get(
            "hardrock_favorite_accuracy"
        ),
        "model_margin_mae": model.get(
            "model_margin_mae"
        ),
        "hardrock_margin_mae": model.get(
            "hardrock_margin_mae"
        ),
        "model_brier_score": model.get(
            "brier_score"
        ),
        "model_side_spread_diagnostic": model.get(
            "model_side_spread_record"
        ),
        "model_margin_closer_games": head.get(
            "model_better"
        ),
        "hardrock_margin_closer_games": head.get(
            "hardrock_better"
        ),
        "margin_ties": head.get(
            "ties"
        ),
        "actual_wager_tickets": wagers.get(
            "tickets"
        ),
        "actual_wager_record": wagers.get(
            "record"
        ),
        "actual_amount_risked": wagers.get(
            "amount_risked"
        ),
        "actual_profit_loss": wagers.get(
            "profit_loss"
        ),
        "actual_roi": wagers.get(
            "roi"
        ),
        "actual_average_clv_points": wagers.get(
            "average_clv_points"
        ),
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

    rows = [
        flatten_week(
            week_report
        )
        for _, week_report in sorted(
            report.get(
                "weeks",
                {}
            ).items(),
            key=lambda item: int(
                item[0]
            ),
        )
    ]

    fieldnames = [
        "week",
        "games_scored",
        "model_winner_accuracy",
        "hardrock_favorite_accuracy",
        "model_margin_mae",
        "hardrock_margin_mae",
        "model_brier_score",
        "model_side_spread_diagnostic",
        "model_margin_closer_games",
        "hardrock_margin_closer_games",
        "margin_ties",
        "actual_wager_tickets",
        "actual_wager_record",
        "actual_amount_risked",
        "actual_profit_loss",
        "actual_roi",
        "actual_average_clv_points",
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

def print_candidate(
    label: str,
    candidate: dict[str, Any] | None,
    *,
    value_key: str | None = None,
    value_suffix: str = "",
) -> None:
    if not candidate:
        print(
            f"{label:<31} N/A"
        )
        return

    value_text = ""

    if value_key:
        value = safe_float(
            candidate.get(
                value_key
            )
        )

        if value is not None:
            value_text = (
                f" | {value:.2f}{value_suffix}"
            )

    print(
        f"{label:<31} "
        f"{candidate.get('game')}"
        f"{value_text}"
    )


def print_summary(
    report: dict[str, Any],
) -> None:
    week_report = report.get(
        "selected_week_report"
    )

    print(
        "=" * 112
    )

    print(
        "CFB EDGE ENGINE — WEEKLY RECAP V1"
    )

    print(
        "=" * 112
    )

    print(
        f"Season:            {report['season']}"
    )

    print(
        f"Snapshot policy:   {report['snapshot_policy']}"
    )

    print(
        f"Selected week:     {report.get('selected_week')}"
    )

    print(
        "Mode:              READ ONLY / DESCRIPTIVE"
    )

    print(
        "=" * 112
    )

    if not week_report:
        print()
        print(
            "No safely scored games are available for the selected week."
        )
        return

    model = week_report[
        "model_vs_market"
    ]

    wagers = week_report[
        "actual_wagers"
    ]

    print()

    print(
        "WEEK PERFORMANCE"
    )

    print(
        "-" * 112
    )

    print(
        f"Games scored:                 "
        f"{week_report['games_scored']}"
    )

    print(
        f"Model winner accuracy:        "
        f"{fmt_pct(model['model_winner_accuracy'])}"
    )

    print(
        f"Hard Rock favorite accuracy:  "
        f"{fmt_pct(model['hardrock_favorite_accuracy'])}"
    )

    print(
        f"Model margin MAE:             "
        f"{fmt_num(model['model_margin_mae'], 2, ' pts')}"
    )

    print(
        f"Hard Rock margin MAE:         "
        f"{fmt_num(model['hardrock_margin_mae'], 2, ' pts')}"
    )

    print(
        f"Model Brier score:            "
        f"{fmt_num(model['brier_score'], 3)}"
    )

    print(
        f"Model-side spread diagnostic: "
        f"{model['model_side_spread_record']}"
    )

    head = model[
        "margin_error_head_to_head"
    ]

    print(
        f"Closer to final margin:       "
        f"Model {head['model_better']} | "
        f"Hard Rock {head['hardrock_better']} | "
        f"Ties {head['ties']}"
    )

    print()

    print(
        "ACTUAL WAGERS"
    )

    print(
        "-" * 112
    )

    print(
        f"Tickets logged:               "
        f"{wagers['tickets']}"
    )

    print(
        f"Settled record:               "
        f"{wagers['record']}"
    )

    print(
        f"Pending:                      "
        f"{wagers['pending']}"
    )

    print(
        f"Amount risked:                "
        f"${wagers['amount_risked']:,.2f}"
    )

    print(
        f"Profit / loss:                "
        f"{fmt_money(wagers['profit_loss'])}"
    )

    print(
        f"ROI:                          "
        f"{fmt_pct(wagers['roi'])}"
    )

    print(
        f"Average CLV:                  "
        f"{fmt_num(wagers['average_clv_points'], 2, ' pts')} "
        f"(n={wagers['clv_sample_size']})"
    )

    print()

    print(
        "WEEK HIGHLIGHTS"
    )

    print(
        "-" * 112
    )

    extremes = week_report[
        "extremes"
    ]

    print_candidate(
        "Best model margin prediction:",
        extremes.get(
            "best_model_margin_prediction"
        ),
        value_key="absolute_margin_error",
        value_suffix=" pts error",
    )

    print_candidate(
        "Worst model margin prediction:",
        extremes.get(
            "worst_model_margin_prediction"
        ),
        value_key="absolute_margin_error",
        value_suffix=" pts error",
    )

    print_candidate(
        "Largest model/market gap:",
        extremes.get(
            "largest_model_market_disagreement"
        ),
        value_key="model_vs_consensus_abs_points",
        value_suffix=" pts",
    )

    print_candidate(
        "Largest captured model edge:",
        extremes.get(
            "largest_captured_model_edge"
        ),
        value_key="model_edge_points",
        value_suffix=" pts",
    )

    model_advantage = extremes.get(
        "biggest_model_margin_advantage_over_hardrock"
    )

    if model_advantage:
        game = (
            model_advantage.get(
                "game"
            )
            or
            {}
        )

        print(
            f"{'Biggest model vs HR margin win:':<31} "
            f"{game.get('game')} | "
            f"{model_advantage.get('advantage_points'):.2f} pts closer"
        )

    else:
        print(
            f"{'Biggest model vs HR margin win:':<31} N/A"
        )

    market_advantage = extremes.get(
        "biggest_hardrock_margin_advantage_over_model"
    )

    if market_advantage:
        game = (
            market_advantage.get(
                "game"
            )
            or
            {}
        )

        print(
            f"{'Biggest HR vs model margin win:':<31} "
            f"{game.get('game')} | "
            f"{market_advantage.get('advantage_points'):.2f} pts closer"
        )

    else:
        print(
            f"{'Biggest HR vs model margin win:':<31} N/A"
        )

    print()

    print(
        "BY STRENGTH"
    )

    print(
        "-" * 112
    )

    for label in (
        "HIGH",
        "MEDIUM",
        "LOW",
        "N/A",
    ):
        row = week_report[
            "by_strength"
        ][
            label
        ]

        print(
            f"{label:<7} "
            f"n={row['games']:<3} | "
            f"Model win={fmt_pct(row['model_winner_accuracy']):>6} | "
            f"Model MAE={fmt_num(row['model_margin_mae']):>6} | "
            f"HR MAE={fmt_num(row['hardrock_margin_mae']):>6} | "
            f"Spread diag={row['model_side_spread_record']}"
        )

    print()

    print(
        "FILES WRITTEN"
    )

    print(
        "-" * 112
    )

    print(
        OUTPUT_JSON
    )

    print(
        OUTPUT_CSV
    )

    print()

    print(
        "IMPORTANT: Prediction spread results are diagnostic. "
        "Only Wager Ledger tickets are actual bets."
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a weekly CFB Edge prospective recap."
        )
    )

    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help=(
            "Week to print. Default is the latest week with safely scored games."
        ),
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
            "Prediction snapshot policy. Default: latest."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scored = load_scored_rows(
        SEASON
    )

    selected = apply_policy(
        scored,
        args.policy,
    )

    wagers = load_actual_wagers()

    report = build_report(
        selected,
        wagers,
        selected_week=args.week,
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
