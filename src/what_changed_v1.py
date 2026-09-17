#!/usr/bin/env python3
"""
CFB Edge Engine — What Changed V1

Purpose
-------
Compare the two most recent immutable pre-kickoff prediction snapshots for each
college-football game. This is the backend for a future GUI "WHAT CHANGED?"
panel.

The script is intentionally read-only with respect to prediction history. It
never calls CFBD, The Odds API, OddsPapi, or any other network service.

Why this exists
---------------
The prediction-history database already preserves every successful GAME DAY run
as a separate snapshot. That means a Saturday game can have a Monday snapshot,
a Wednesday snapshot, a Friday snapshot, and so on. This script turns those raw
snapshots into a clean comparison so the GUI can answer questions such as:

    MODEL SPREAD        Georgia -15.8 -> Georgia -16.4
    HARD ROCK           Georgia -13.5 -> Georgia -14.5
    MODEL EDGE          +2.3 -> +1.9 pts
    WIN PROBABILITY     76.8% -> 78.1%
    STRENGTH            MEDIUM -> MEDIUM

Integrity notes
---------------
- prediction_history.sqlite is opened in SQLite read-only/query-only mode.
- Comparison is performed only within the exact same game_key.
- There is no team-pair-only matching.
- Canonical home-oriented values are used internally where possible so a change
  is not accidentally reversed just because a favorite label changes.
- If the model favorite changes between snapshots, model-edge change is not
  treated as directly comparable; the side flip is explicitly flagged.
- Missing Hard Rock or reference-market values stay missing. No substitute book
  is invented.

Outputs
-------
By default:
    data/processed/what_changed_2026_v1.json
    data/processed/what_changed_2026_v1.csv

The JSON output is intended to be the primary GUI/backend contract. The CSV is
for easy inspection.
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
HISTORY_DB = PROJECT_ROOT / "data" / "prospective" / "prediction_history.sqlite"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_JSON = PROCESSED_DIR / "what_changed_2026_v1.json"
OUTPUT_CSV = PROCESSED_DIR / "what_changed_2026_v1.csv"

SCHEMA_VERSION = 1


# ============================================================================
# GENERIC HELPERS
# ============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    value = value or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
    return " ".join(str(value or "").strip().casefold().split())


def same_team(left: Any, right: Any) -> bool:
    left_norm = normalize_team(left)
    right_norm = normalize_team(right)
    return bool(left_norm and right_norm and left_norm == right_norm)


def rounded(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def numeric_delta(current: Any, previous: Any, digits: int = 3) -> float | None:
    current_number = safe_float(current)
    previous_number = safe_float(previous)
    if current_number is None or previous_number is None:
        return None
    return rounded(current_number - previous_number, digits)


def changed_numeric(current: Any, previous: Any, tolerance: float = 1e-9) -> bool:
    current_number = safe_float(current)
    previous_number = safe_float(previous)
    if current_number is None and previous_number is None:
        return False
    if current_number is None or previous_number is None:
        return True
    return not math.isclose(current_number, previous_number, abs_tol=tolerance)


def changed_text(current: Any, previous: Any) -> bool:
    return str(current or "").strip() != str(previous or "").strip()


def probability_percent(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    if number <= 1.0:
        number *= 100.0
    return rounded(number, 1)


def capture_gap_hours(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    left = parse_datetime(previous.get("captured_utc"))
    right = parse_datetime(current.get("captured_utc"))
    if left is None or right is None:
        return None
    return rounded((right - left).total_seconds() / 3600.0, 2)


# ============================================================================
# CANONICAL MARKET ORIENTATION
# ============================================================================


def spread_for_model_favorite_to_home_margin(
    snapshot: dict[str, Any],
    field: str,
) -> float | None:
    """Convert a spread quoted for the model favorite to implied home margin.

    Example:
      Model favorite = away Alabama
      Hard Rock line for model favorite = -8.5
      Implied home margin = -8.5

      Model favorite = home Georgia
      Hard Rock line for model favorite = -14.5
      Implied home margin = +14.5

    A sportsbook spread is the negative of expected margin for that side, so
    the magnitude is re-oriented here into the same sign convention used by
    model_home_margin: positive means home team favored, negative means away.
    """

    spread = safe_float(snapshot.get(field))
    if spread is None:
        return None

    favorite = snapshot.get("model_favorite")
    home = snapshot.get("home_team")
    away = snapshot.get("away_team")

    if same_team(favorite, home):
        return -spread
    if same_team(favorite, away):
        return spread
    return None


def hardrock_home_margin(snapshot: dict[str, Any]) -> float | None:
    return spread_for_model_favorite_to_home_margin(
        snapshot,
        "hardrock_spread_for_model_favorite",
    )


def reference_home_margin(snapshot: dict[str, Any]) -> float | None:
    return spread_for_model_favorite_to_home_margin(
        snapshot,
        "reference_spread_for_model_favorite",
    )


def favorite_probability_for_team(snapshot: dict[str, Any], team: Any) -> float | None:
    """Return model probability for a named team using canonical home probability."""

    home_p = safe_float(snapshot.get("model_home_win_probability"))
    if home_p is None:
        return None

    home = snapshot.get("home_team")
    away = snapshot.get("away_team")
    if same_team(team, home):
        return home_p
    if same_team(team, away):
        return 1.0 - home_p
    return None


# ============================================================================
# HISTORY ACCESS
# ============================================================================


def connect_history_read_only(path: Path = HISTORY_DB) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction-history database not found: {path}\n"
            "Run GAME DAY successfully at least once before building changes."
        )

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def history_columns(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(prediction_snapshots)").fetchall()
    }


def validate_history_schema(connection: sqlite3.Connection) -> None:
    required = {
        "snapshot_id",
        "run_id",
        "captured_utc",
        "game_key",
        "game_id",
        "season",
        "week",
        "kickoff_utc",
        "away_team",
        "home_team",
        "model_favorite",
        "model_spread_display",
        "model_home_margin",
        "model_favorite_win_probability",
        "model_home_win_probability",
        "hardrock_available",
        "hardrock_spread_for_model_favorite",
        "hardrock_price_for_model_favorite",
        "hardrock_display",
        "reference_spread_for_model_favorite",
        "reference_market_display",
        "reference_market_books",
        "model_edge_points",
        "model_vs_consensus_abs_points",
        "hardrock_vs_consensus_points",
        "strength_label",
        "strength_score",
        "signal_warning",
        "nn_input_status",
        "nn_feature_coverage",
        "nn_margin_std",
        "nn_win_probability_std",
        "nn_heads_agree",
    }
    columns = history_columns(connection)
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(
            "prediction_snapshots schema is missing required fields: "
            + ", ".join(missing)
        )


def load_snapshots(
    connection: sqlite3.Connection,
    *,
    season: int | None = 2026,
) -> list[dict[str, Any]]:
    validate_history_schema(connection)

    if season is None:
        query = "SELECT * FROM prediction_snapshots ORDER BY game_key, captured_utc, snapshot_id"
        parameters: tuple[Any, ...] = ()
    else:
        query = (
            "SELECT * FROM prediction_snapshots "
            "WHERE season = ? "
            "ORDER BY game_key, captured_utc, snapshot_id"
        )
        parameters = (season,)

    return [dict(row) for row in connection.execute(query, parameters).fetchall()]


# ============================================================================
# COMPARISON ENGINE
# ============================================================================


def compare_two_snapshots(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    if str(previous.get("game_key") or "") != str(current.get("game_key") or ""):
        raise ValueError("Snapshots must belong to the exact same game_key.")

    model_side_flipped = not same_team(
        previous.get("model_favorite"),
        current.get("model_favorite"),
    )

    previous_hr_home = hardrock_home_margin(previous)
    current_hr_home = hardrock_home_margin(current)
    previous_ref_home = reference_home_margin(previous)
    current_ref_home = reference_home_margin(current)

    current_model_favorite = current.get("model_favorite")
    previous_probability_for_current_favorite = favorite_probability_for_team(
        previous,
        current_model_favorite,
    )
    current_probability_for_current_favorite = favorite_probability_for_team(
        current,
        current_model_favorite,
    )

    # model_edge_points is stored from the model-favorite side. If the model
    # favorite changes, subtracting the two edge values would compare opposite
    # sides and create a misleading delta. Preserve the raw values but fail
    # closed on the numerical change.
    edge_delta = None
    if not model_side_flipped:
        edge_delta = numeric_delta(
            current.get("model_edge_points"),
            previous.get("model_edge_points"),
        )

    flags: list[str] = []
    if model_side_flipped:
        flags.append("MODEL_SIDE_FLIPPED")
    if safe_int(previous.get("hardrock_available")) != safe_int(current.get("hardrock_available")):
        flags.append("HARD_ROCK_AVAILABILITY_CHANGED")
    if changed_text(current.get("strength_label"), previous.get("strength_label")):
        flags.append("STRENGTH_CHANGED")
    if changed_text(current.get("signal_warning"), previous.get("signal_warning")):
        flags.append("WARNING_CHANGED")
    if changed_text(current.get("nn_input_status"), previous.get("nn_input_status")):
        flags.append("INPUT_STATUS_CHANGED")
    if safe_int(current.get("nn_feature_coverage")) != safe_int(previous.get("nn_feature_coverage")):
        flags.append("FEATURE_COVERAGE_CHANGED")
    if safe_int(current.get("nn_heads_agree")) != safe_int(previous.get("nn_heads_agree")):
        flags.append("MODEL_HEAD_AGREEMENT_CHANGED")

    model_home_delta = numeric_delta(
        current.get("model_home_margin"),
        previous.get("model_home_margin"),
    )
    hardrock_home_delta = numeric_delta(current_hr_home, previous_hr_home)
    reference_home_delta = numeric_delta(current_ref_home, previous_ref_home)
    probability_delta = numeric_delta(
        current_probability_for_current_favorite,
        previous_probability_for_current_favorite,
        digits=5,
    )
    strength_score_delta = numeric_delta(
        current.get("strength_score"),
        previous.get("strength_score"),
    )

    meaningful_change = any(
        (
            changed_numeric(current.get("model_home_margin"), previous.get("model_home_margin"), 0.049),
            changed_numeric(current_hr_home, previous_hr_home, 0.049),
            changed_numeric(current_ref_home, previous_ref_home, 0.049),
            changed_numeric(
                current_probability_for_current_favorite,
                previous_probability_for_current_favorite,
                0.001,
            ),
            changed_numeric(current.get("model_edge_points"), previous.get("model_edge_points"), 0.049),
            changed_text(current.get("strength_label"), previous.get("strength_label")),
            changed_text(current.get("signal_warning"), previous.get("signal_warning")),
            model_side_flipped,
            bool(flags),
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "game_key": current.get("game_key"),
        "game_id": current.get("game_id"),
        "season": safe_int(current.get("season")),
        "week": safe_int(current.get("week")),
        "kickoff_utc": current.get("kickoff_utc"),
        "away_team": current.get("away_team"),
        "home_team": current.get("home_team"),
        "previous_snapshot_id": previous.get("snapshot_id"),
        "current_snapshot_id": current.get("snapshot_id"),
        "previous_run_id": previous.get("run_id"),
        "current_run_id": current.get("run_id"),
        "previous_captured_utc": previous.get("captured_utc"),
        "current_captured_utc": current.get("captured_utc"),
        "hours_between_snapshots": capture_gap_hours(previous, current),
        "meaningful_change": bool(meaningful_change),
        "flags": flags,
        "model": {
            "previous_favorite": previous.get("model_favorite"),
            "current_favorite": current.get("model_favorite"),
            "side_flipped": model_side_flipped,
            "previous_spread_display": previous.get("model_spread_display"),
            "current_spread_display": current.get("model_spread_display"),
            "previous_home_margin": rounded(safe_float(previous.get("model_home_margin"))),
            "current_home_margin": rounded(safe_float(current.get("model_home_margin"))),
            "home_margin_delta": model_home_delta,
            "previous_probability_for_current_favorite": probability_percent(
                previous_probability_for_current_favorite
            ),
            "current_probability_for_current_favorite": probability_percent(
                current_probability_for_current_favorite
            ),
            "probability_delta_percentage_points": (
                None if probability_delta is None else rounded(probability_delta * 100.0, 2)
            ),
            "previous_margin_std": rounded(safe_float(previous.get("nn_margin_std"))),
            "current_margin_std": rounded(safe_float(current.get("nn_margin_std"))),
            "margin_std_delta": numeric_delta(
                current.get("nn_margin_std"),
                previous.get("nn_margin_std"),
            ),
            "previous_probability_std": rounded(
                safe_float(previous.get("nn_win_probability_std")),
                5,
            ),
            "current_probability_std": rounded(
                safe_float(current.get("nn_win_probability_std")),
                5,
            ),
            "probability_std_delta": numeric_delta(
                current.get("nn_win_probability_std"),
                previous.get("nn_win_probability_std"),
                digits=5,
            ),
            "previous_heads_agree": safe_int(previous.get("nn_heads_agree")),
            "current_heads_agree": safe_int(current.get("nn_heads_agree")),
        },
        "hardrock": {
            "previous_available": safe_int(previous.get("hardrock_available")),
            "current_available": safe_int(current.get("hardrock_available")),
            "previous_display": previous.get("hardrock_display"),
            "current_display": current.get("hardrock_display"),
            "previous_home_margin": rounded(previous_hr_home),
            "current_home_margin": rounded(current_hr_home),
            "home_margin_delta": hardrock_home_delta,
            "previous_price_for_model_favorite": rounded(
                safe_float(previous.get("hardrock_price_for_model_favorite"))
            ),
            "current_price_for_model_favorite": rounded(
                safe_float(current.get("hardrock_price_for_model_favorite"))
            ),
        },
        "reference_market": {
            "previous_display": previous.get("reference_market_display"),
            "current_display": current.get("reference_market_display"),
            "previous_home_margin": rounded(previous_ref_home),
            "current_home_margin": rounded(current_ref_home),
            "home_margin_delta": reference_home_delta,
            "previous_books": safe_int(previous.get("reference_market_books")),
            "current_books": safe_int(current.get("reference_market_books")),
        },
        "edge": {
            "previous_points": rounded(safe_float(previous.get("model_edge_points"))),
            "current_points": rounded(safe_float(current.get("model_edge_points"))),
            "delta_points": edge_delta,
            "directly_comparable": not model_side_flipped,
            "previous_model_vs_consensus_abs_points": rounded(
                safe_float(previous.get("model_vs_consensus_abs_points"))
            ),
            "current_model_vs_consensus_abs_points": rounded(
                safe_float(current.get("model_vs_consensus_abs_points"))
            ),
            "previous_hardrock_vs_consensus_points": rounded(
                safe_float(previous.get("hardrock_vs_consensus_points"))
            ),
            "current_hardrock_vs_consensus_points": rounded(
                safe_float(current.get("hardrock_vs_consensus_points"))
            ),
        },
        "strength": {
            "previous_label": str(previous.get("strength_label") or "N/A").upper(),
            "current_label": str(current.get("strength_label") or "N/A").upper(),
            "previous_score": rounded(safe_float(previous.get("strength_score"))),
            "current_score": rounded(safe_float(current.get("strength_score"))),
            "score_delta": strength_score_delta,
        },
        "quality": {
            "previous_warning": previous.get("signal_warning") or "NONE",
            "current_warning": current.get("signal_warning") or "NONE",
            "previous_input_status": previous.get("nn_input_status"),
            "current_input_status": current.get("nn_input_status"),
            "previous_feature_coverage": safe_int(previous.get("nn_feature_coverage")),
            "current_feature_coverage": safe_int(current.get("nn_feature_coverage")),
        },
    }


def build_change_rows(
    snapshots: list[dict[str, Any]],
    *,
    include_single_snapshot_games: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        game_key = str(snapshot.get("game_key") or "").strip()
        if not game_key:
            continue
        grouped[game_key].append(snapshot)

    comparisons: list[dict[str, Any]] = []
    single_snapshot_games = 0

    for game_key, game_rows in grouped.items():
        ordered = sorted(
            game_rows,
            key=lambda row: (
                str(row.get("captured_utc") or ""),
                str(row.get("snapshot_id") or ""),
            ),
        )

        if len(ordered) < 2:
            single_snapshot_games += 1
            if not include_single_snapshot_games:
                continue
            current = ordered[-1]
            comparisons.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "game_key": game_key,
                    "game_id": current.get("game_id"),
                    "season": safe_int(current.get("season")),
                    "week": safe_int(current.get("week")),
                    "kickoff_utc": current.get("kickoff_utc"),
                    "away_team": current.get("away_team"),
                    "home_team": current.get("home_team"),
                    "previous_snapshot_id": None,
                    "current_snapshot_id": current.get("snapshot_id"),
                    "previous_run_id": None,
                    "current_run_id": current.get("run_id"),
                    "previous_captured_utc": None,
                    "current_captured_utc": current.get("captured_utc"),
                    "hours_between_snapshots": None,
                    "meaningful_change": False,
                    "flags": ["NO_PREVIOUS_SNAPSHOT"],
                    "model": {
                        "previous_favorite": None,
                        "current_favorite": current.get("model_favorite"),
                        "side_flipped": False,
                        "previous_spread_display": None,
                        "current_spread_display": current.get("model_spread_display"),
                    },
                    "hardrock": {
                        "previous_display": None,
                        "current_display": current.get("hardrock_display"),
                    },
                    "reference_market": {
                        "previous_display": None,
                        "current_display": current.get("reference_market_display"),
                    },
                    "edge": {
                        "previous_points": None,
                        "current_points": rounded(safe_float(current.get("model_edge_points"))),
                        "delta_points": None,
                        "directly_comparable": False,
                    },
                    "strength": {
                        "previous_label": None,
                        "current_label": str(current.get("strength_label") or "N/A").upper(),
                    },
                    "quality": {
                        "previous_warning": None,
                        "current_warning": current.get("signal_warning") or "NONE",
                    },
                }
            )
            continue

        comparisons.append(compare_two_snapshots(ordered[-2], ordered[-1]))

    comparisons.sort(
        key=lambda row: (
            str(row.get("kickoff_utc") or ""),
            str(row.get("away_team") or ""),
            str(row.get("home_team") or ""),
        )
    )
    return comparisons, single_snapshot_games


# ============================================================================
# OUTPUTS
# ============================================================================


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    model = row.get("model") or {}
    hardrock = row.get("hardrock") or {}
    reference = row.get("reference_market") or {}
    edge = row.get("edge") or {}
    strength = row.get("strength") or {}
    quality = row.get("quality") or {}

    return {
        "game_key": row.get("game_key"),
        "game_id": row.get("game_id"),
        "season": row.get("season"),
        "week": row.get("week"),
        "kickoff_utc": row.get("kickoff_utc"),
        "away_team": row.get("away_team"),
        "home_team": row.get("home_team"),
        "previous_captured_utc": row.get("previous_captured_utc"),
        "current_captured_utc": row.get("current_captured_utc"),
        "hours_between_snapshots": row.get("hours_between_snapshots"),
        "meaningful_change": int(bool(row.get("meaningful_change"))),
        "flags": "|".join(row.get("flags") or []),
        "previous_model_favorite": model.get("previous_favorite"),
        "current_model_favorite": model.get("current_favorite"),
        "model_side_flipped": int(bool(model.get("side_flipped"))),
        "previous_model_spread": model.get("previous_spread_display"),
        "current_model_spread": model.get("current_spread_display"),
        "model_home_margin_delta": model.get("home_margin_delta"),
        "previous_win_probability_current_favorite_pct": model.get(
            "previous_probability_for_current_favorite"
        ),
        "current_win_probability_current_favorite_pct": model.get(
            "current_probability_for_current_favorite"
        ),
        "win_probability_delta_percentage_points": model.get(
            "probability_delta_percentage_points"
        ),
        "previous_hardrock": hardrock.get("previous_display"),
        "current_hardrock": hardrock.get("current_display"),
        "hardrock_home_margin_delta": hardrock.get("home_margin_delta"),
        "previous_reference": reference.get("previous_display"),
        "current_reference": reference.get("current_display"),
        "reference_home_margin_delta": reference.get("home_margin_delta"),
        "previous_edge_points": edge.get("previous_points"),
        "current_edge_points": edge.get("current_points"),
        "edge_delta_points": edge.get("delta_points"),
        "previous_strength": strength.get("previous_label"),
        "current_strength": strength.get("current_label"),
        "strength_score_delta": strength.get("score_delta"),
        "previous_warning": quality.get("previous_warning"),
        "current_warning": quality.get("current_warning"),
        "previous_feature_coverage": quality.get("previous_feature_coverage"),
        "current_feature_coverage": quality.get("current_feature_coverage"),
        "previous_snapshot_id": row.get("previous_snapshot_id"),
        "current_snapshot_id": row.get("current_snapshot_id"),
        "previous_run_id": row.get("previous_run_id"),
        "current_run_id": row.get("current_run_id"),
    }


def write_outputs(
    comparisons: list[dict[str, Any]],
    *,
    total_snapshots: int,
    total_games: int,
    single_snapshot_games: int,
    season: int | None,
    output_json: Path = OUTPUT_JSON,
    output_csv: Path = OUTPUT_CSV,
) -> dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_iso(),
        "source_database": str(HISTORY_DB.relative_to(PROJECT_ROOT)),
        "season_filter": season,
        "total_snapshots": total_snapshots,
        "total_games": total_games,
        "games_with_comparisons": len(comparisons),
        "single_snapshot_games": single_snapshot_games,
        "meaningful_change_games": sum(
            1 for row in comparisons if bool(row.get("meaningful_change"))
        ),
        "comparisons": comparisons,
    }

    output_json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    flattened = [flatten_for_csv(row) for row in comparisons]
    if flattened:
        fieldnames = list(flattened[0].keys())
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened)
    else:
        output_csv.write_text("", encoding="utf-8")

    return payload


# ============================================================================
# CLI
# ============================================================================


def print_rule(character: str = "=") -> None:
    print(character * 100)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the latest two immutable prediction snapshots for each game."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season to compare. Default: 2026. Use 0 for all seasons.",
    )
    parser.add_argument(
        "--game-key",
        default=None,
        help="Print only one exact game_key after building comparisons.",
    )
    parser.add_argument(
        "--include-single",
        action="store_true",
        help="Include games with only one snapshot as NO_PREVIOUS_SNAPSHOT rows.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Read and compare without writing JSON/CSV outputs.",
    )
    args = parser.parse_args()

    season = None if args.season == 0 else args.season

    print_rule()
    print("CFB EDGE ENGINE — WHAT CHANGED V1")
    print_rule()
    print(f"History DB: {HISTORY_DB}")
    print(f"Season:     {season if season is not None else 'ALL'}")
    print("Network:    NONE — local prediction history only")
    print("Mode:       READ-ONLY prediction comparison")
    print_rule()

    connection = connect_history_read_only()
    try:
        snapshots = load_snapshots(connection, season=season)
    finally:
        connection.close()

    game_keys = {
        str(row.get("game_key") or "")
        for row in snapshots
        if str(row.get("game_key") or "")
    }

    comparisons, single_snapshot_games = build_change_rows(
        snapshots,
        include_single_snapshot_games=args.include_single,
    )

    if not args.no_write:
        payload = write_outputs(
            comparisons,
            total_snapshots=len(snapshots),
            total_games=len(game_keys),
            single_snapshot_games=single_snapshot_games,
            season=season,
        )
        print()
        print("OUTPUT")
        print_rule("-")
        print(f"JSON: {OUTPUT_JSON.relative_to(PROJECT_ROOT)}")
        print(f"CSV:  {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
        print(f"Generated: {payload['generated_utc']}")

    print()
    print("COMPARISON STATUS")
    print_rule("-")
    print(f"Snapshots read:             {len(snapshots)}")
    print(f"Games found:                {len(game_keys)}")
    print(f"Games with >=2 snapshots:   {len(game_keys) - single_snapshot_games}")
    print(f"Single-snapshot games:      {single_snapshot_games}")
    print(f"Comparison rows produced:   {len(comparisons)}")
    print(
        "Meaningful changes:         "
        + str(sum(1 for row in comparisons if bool(row.get("meaningful_change"))))
    )

    selected = comparisons
    if args.game_key:
        selected = [
            row for row in comparisons if str(row.get("game_key") or "") == args.game_key
        ]
        if not selected:
            print()
            print(f"No comparison found for exact game_key: {args.game_key}")
            return 1

    if args.game_key:
        print()
        print("GAME COMPARISON")
        print_rule("-")
        print(json.dumps(selected[0], indent=2, default=str))
    elif comparisons:
        print()
        print("LATEST SAMPLE")
        print_rule("-")
        for row in comparisons[:5]:
            model = row.get("model") or {}
            hardrock = row.get("hardrock") or {}
            strength = row.get("strength") or {}
            print(
                f"{row.get('away_team')} @ {row.get('home_team')} | "
                f"MODEL {model.get('previous_spread_display') or 'N/A'} -> "
                f"{model.get('current_spread_display') or 'N/A'} | "
                f"HR {hardrock.get('previous_display') or 'N/A'} -> "
                f"{hardrock.get('current_display') or 'N/A'} | "
                f"STRENGTH {strength.get('previous_label') or 'N/A'} -> "
                f"{strength.get('current_label') or 'N/A'}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
