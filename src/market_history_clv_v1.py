#!/usr/bin/env python3
"""
CFB Edge Engine — Market History + CLV V1

Purpose
-------
Build a truthful pre-kickoff line-movement history from immutable Prediction
History snapshots and calculate Closing Line Value (CLV) for ACTUAL wagers.

Important terminology
---------------------
This project cannot claim a "true closing line" unless it actually captured a
market snapshot close enough to kickoff. Therefore every closing candidate is
quality-labeled by how long before kickoff it was captured.

Primary actual-wager CLV:
    1. Hard Rock closing snapshot, when a valid post-wager Hard Rock snapshot
       exists. This is preferred because actual tickets are placed at Hard Rock.
    2. Multi-book reference consensus closing snapshot as a fallback.

Spread CLV convention:
    CLV points = ticket spread - closing spread for the SAME wagered team

Examples:
    Ticket -7.5, close -9.5  -> +2.0 CLV points
    Ticket +3.5, close +2.0  -> +1.5 CLV points

Positive = the ticket captured a better spread than the later close.
Negative = the ticket captured a worse spread than the later close.

This script:
- makes NO API calls
- never retrains the model
- never changes Prediction History
- never changes market snapshots
- updates ONLY wager_ledger.wagers.clv_status / clv_points when
  --update-ledger is explicitly supplied
- writes line-movement / CLV research outputs

Outputs
-------
data/processed/market_history_2026_v1.json
data/processed/market_history_2026_v1.csv
data/processed/wager_clv_2026_v1.json
data/processed/wager_clv_2026_v1.csv
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


# ============================================================================
# PATHS / CONSTANTS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_DB = (
    PROJECT_ROOT
    / "data"
    / "prospective"
    / "prediction_history.sqlite"
)

WAGER_DB = (
    PROJECT_ROOT
    / "data"
    / "prospective"
    / "wager_ledger.sqlite"
)

MARKET_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_history_2026_v1.json"
)

MARKET_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_history_2026_v1.csv"
)

CLV_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wager_clv_2026_v1.json"
)

CLV_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wager_clv_2026_v1.csv"
)

SEASON = 2026

# These thresholds classify capture quality. They do NOT claim that a line is
# an official sportsbook close.
VERIFIED_CLOSE_MINUTES = 120.0
NEAR_CLOSE_MINUTES = 360.0
SAME_DAY_CLOSE_MINUTES = 1440.0

MIN_REFERENCE_BOOKS = 2


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return (
        utc_now()
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


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
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
    digits: int = 4,
) -> float | None:
    if value is None:
        return None

    return round(value, digits)


def american_break_even_probability(
    odds: int | float | None,
) -> float | None:
    value = safe_float(odds)

    if value is None:
        return None

    if -100 < value < 100:
        return None

    if value > 0:
        return 100.0 / (value + 100.0)

    return abs(value) / (abs(value) + 100.0)


def minutes_before_kickoff(
    captured_utc: Any,
    kickoff_utc: Any,
) -> float | None:
    captured = parse_datetime(captured_utc)
    kickoff = parse_datetime(kickoff_utc)

    if captured is None or kickoff is None:
        return None

    return (
        kickoff - captured
    ).total_seconds() / 60.0


def close_quality(
    minutes_to_kickoff: float | None,
    source: str,
) -> str:
    prefix = source.upper()

    if minutes_to_kickoff is None:
        return f"{prefix}_TIME_UNKNOWN"

    if minutes_to_kickoff < 0:
        return f"{prefix}_INVALID_AFTER_KICKOFF"

    if minutes_to_kickoff <= VERIFIED_CLOSE_MINUTES:
        return f"{prefix}_VERIFIED_NEAR_CLOSE"

    if minutes_to_kickoff <= NEAR_CLOSE_MINUTES:
        return f"{prefix}_NEAR_CLOSE_PROXY"

    if minutes_to_kickoff <= SAME_DAY_CLOSE_MINUTES:
        return f"{prefix}_SAME_DAY_CLOSE_PROXY"

    return f"{prefix}_STALE_CLOSE_PROXY"


# ============================================================================
# SQLITE HELPERS
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
            (table,),
        ).fetchone()
        is not None
    )


def table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    }


# ============================================================================
# PREDICTION HISTORY LOADING
# ============================================================================

def load_prediction_snapshots() -> list[dict[str, Any]]:
    if not HISTORY_DB.exists():
        raise FileNotFoundError(
            f"Prediction History database not found: {HISTORY_DB}"
        )

    connection = sqlite3.connect(
        f"file:{HISTORY_DB}?mode=ro",
        uri=True,
    )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")

    try:
        if not table_exists(
            connection,
            "prediction_snapshots",
        ):
            raise RuntimeError(
                "prediction_snapshots table is missing."
            )

        columns = table_columns(
            connection,
            "prediction_snapshots",
        )

        required = {
            "snapshot_id",
            "run_id",
            "game_key",
            "game_id",
            "season",
            "week",
            "kickoff_utc",
            "captured_utc",
            "away_team",
            "home_team",
            "model_favorite",
            "model_spread",
            "model_spread_display",
            "hardrock_available",
            "hardrock_spread_for_model_favorite",
            "hardrock_price_for_model_favorite",
            "hardrock_display",
            "reference_spread_for_model_favorite",
            "reference_market_display",
            "reference_market_books",
            "strength_label",
        }

        missing = sorted(required - columns)

        if missing:
            raise RuntimeError(
                "prediction_snapshots schema is missing required fields: "
                + ", ".join(missing)
            )

        rows = connection.execute(
            """
            SELECT *
            FROM prediction_snapshots
            WHERE season = ?
            ORDER BY game_key, captured_utc, snapshot_id
            """,
            (SEASON,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        connection.close()


# ============================================================================
# SPREAD ORIENTATION
# ============================================================================

def opponent_of_model_favorite(
    snapshot: dict[str, Any],
) -> str | None:
    favorite = str(
        snapshot.get("model_favorite") or ""
    ).strip()

    away = str(
        snapshot.get("away_team") or ""
    ).strip()

    home = str(
        snapshot.get("home_team") or ""
    ).strip()

    if same_team(favorite, away):
        return home

    if same_team(favorite, home):
        return away

    return None


def spread_for_team_from_model_side_field(
    snapshot: dict[str, Any],
    field: str,
    team: Any,
) -> float | None:
    spread = safe_float(
        snapshot.get(field)
    )

    if spread is None:
        return None

    favorite = str(
        snapshot.get("model_favorite") or ""
    ).strip()

    opponent = opponent_of_model_favorite(
        snapshot
    )

    if same_team(team, favorite):
        return spread

    if opponent and same_team(team, opponent):
        return -spread

    return None


def model_spread_for_team(
    snapshot: dict[str, Any],
    team: Any,
) -> float | None:
    magnitude = safe_float(
        snapshot.get("model_spread")
    )

    if magnitude is None:
        return None

    magnitude = abs(magnitude)

    favorite = str(
        snapshot.get("model_favorite") or ""
    ).strip()

    opponent = opponent_of_model_favorite(
        snapshot
    )

    if same_team(team, favorite):
        return -magnitude

    if opponent and same_team(team, opponent):
        return magnitude

    return None


def hardrock_price_for_team(
    snapshot: dict[str, Any],
    team: Any,
) -> int | None:
    # The production snapshot stores price only for that snapshot's model
    # favorite. We intentionally refuse to invent the other side's juice.
    favorite = str(
        snapshot.get("model_favorite") or ""
    ).strip()

    if not same_team(team, favorite):
        return None

    price = safe_int(
        snapshot.get(
            "hardrock_price_for_model_favorite"
        )
    )

    if price is None or -100 < price < 100:
        return None

    return price


# ============================================================================
# SNAPSHOT VALIDITY
# ============================================================================

def valid_pre_kickoff_snapshot(
    snapshot: dict[str, Any],
) -> bool:
    captured = parse_datetime(
        snapshot.get("captured_utc")
    )

    kickoff = parse_datetime(
        snapshot.get("kickoff_utc")
    )

    if captured is None or kickoff is None:
        return False

    return captured < kickoff


def valid_reference_snapshot(
    snapshot: dict[str, Any],
) -> bool:
    if not valid_pre_kickoff_snapshot(snapshot):
        return False

    spread = safe_float(
        snapshot.get(
            "reference_spread_for_model_favorite"
        )
    )

    books = safe_int(
        snapshot.get(
            "reference_market_books"
        )
    )

    return (
        spread is not None
        and
        books is not None
        and
        books >= MIN_REFERENCE_BOOKS
    )


def valid_hardrock_snapshot(
    snapshot: dict[str, Any],
) -> bool:
    if not valid_pre_kickoff_snapshot(snapshot):
        return False

    available = safe_int(
        snapshot.get(
            "hardrock_available"
        )
    )

    spread = safe_float(
        snapshot.get(
            "hardrock_spread_for_model_favorite"
        )
    )

    return (
        available == 1
        and
        spread is not None
    )


# ============================================================================
# GAME TIMELINES
# ============================================================================

def tracked_team_for_game(
    snapshots: list[dict[str, Any]],
) -> str:
    # Stable chart orientation uses the model favorite from the most recent
    # pre-kickoff snapshot. Earlier snapshots are converted to this SAME team.
    pregame = [
        row
        for row in snapshots
        if valid_pre_kickoff_snapshot(row)
    ]

    source = (
        pregame[-1]
        if pregame
        else snapshots[-1]
    )

    favorite = str(
        source.get("model_favorite") or ""
    ).strip()

    if favorite:
        return favorite

    home = str(
        source.get("home_team") or ""
    ).strip()

    if home:
        return home

    return str(
        source.get("away_team") or ""
    ).strip()


def timeline_point(
    snapshot: dict[str, Any],
    tracked_team: str,
) -> dict[str, Any]:
    reference_spread = (
        spread_for_team_from_model_side_field(
            snapshot,
            "reference_spread_for_model_favorite",
            tracked_team,
        )
    )

    hardrock_spread = (
        spread_for_team_from_model_side_field(
            snapshot,
            "hardrock_spread_for_model_favorite",
            tracked_team,
        )
    )

    model_spread = model_spread_for_team(
        snapshot,
        tracked_team,
    )

    hardrock_price = hardrock_price_for_team(
        snapshot,
        tracked_team,
    )

    return {
        "snapshot_id": snapshot.get(
            "snapshot_id"
        ),
        "run_id": snapshot.get(
            "run_id"
        ),
        "captured_utc": snapshot.get(
            "captured_utc"
        ),
        "minutes_before_kickoff": rounded(
            minutes_before_kickoff(
                snapshot.get("captured_utc"),
                snapshot.get("kickoff_utc"),
            ),
            2,
        ),
        "snapshot_model_favorite": snapshot.get(
            "model_favorite"
        ),
        "tracked_team": tracked_team,
        "model_spread_for_tracked_team": rounded(
            model_spread,
            3,
        ),
        "hardrock_spread_for_tracked_team": rounded(
            hardrock_spread,
            3,
        ),
        "hardrock_price_for_tracked_team": hardrock_price,
        "reference_spread_for_tracked_team": rounded(
            reference_spread,
            3,
        ),
        "reference_market_books": safe_int(
            snapshot.get(
                "reference_market_books"
            )
        ),
        "model_spread_display": snapshot.get(
            "model_spread_display"
        ),
        "hardrock_display": snapshot.get(
            "hardrock_display"
        ),
        "reference_market_display": snapshot.get(
            "reference_market_display"
        ),
        "strength_label": snapshot.get(
            "strength_label"
        ),
        "signal_warning": snapshot.get(
            "signal_warning"
        ),
    }


def latest_valid_snapshot(
    snapshots: list[dict[str, Any]],
    validator,
    *,
    not_before: datetime | None = None,
) -> dict[str, Any] | None:
    valid = []

    for snapshot in snapshots:
        if not validator(snapshot):
            continue

        captured = parse_datetime(
            snapshot.get("captured_utc")
        )

        if captured is None:
            continue

        if (
            not_before is not None
            and
            captured < not_before
        ):
            continue

        valid.append(
            snapshot
        )

    if not valid:
        return None

    return max(
        valid,
        key=lambda row: (
            parse_datetime(
                row.get("captured_utc")
            )
            or
            datetime.min.replace(
                tzinfo=timezone.utc
            ),
            str(
                row.get("snapshot_id") or ""
            ),
        ),
    )


def close_summary(
    snapshot: dict[str, Any] | None,
    source: str,
    tracked_team: str,
) -> dict[str, Any] | None:
    if snapshot is None:
        return None

    field = (
        "hardrock_spread_for_model_favorite"
        if source == "HARDROCK"
        else
        "reference_spread_for_model_favorite"
    )

    spread = (
        spread_for_team_from_model_side_field(
            snapshot,
            field,
            tracked_team,
        )
    )

    minutes = minutes_before_kickoff(
        snapshot.get("captured_utc"),
        snapshot.get("kickoff_utc"),
    )

    price = (
        hardrock_price_for_team(
            snapshot,
            tracked_team,
        )
        if source == "HARDROCK"
        else
        None
    )

    return {
        "source": source,
        "snapshot_id": snapshot.get(
            "snapshot_id"
        ),
        "captured_utc": snapshot.get(
            "captured_utc"
        ),
        "minutes_before_kickoff": rounded(
            minutes,
            2,
        ),
        "quality": close_quality(
            minutes,
            source,
        ),
        "tracked_team": tracked_team,
        "spread_for_tracked_team": rounded(
            spread,
            3,
        ),
        "american_odds_for_tracked_team": price,
        "reference_market_books": safe_int(
            snapshot.get(
                "reference_market_books"
            )
        ),
        "display": (
            snapshot.get("hardrock_display")
            if source == "HARDROCK"
            else snapshot.get(
                "reference_market_display"
            )
        ),
    }


def build_market_history(
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )

    for snapshot in snapshots:
        game_key = str(
            snapshot.get("game_key") or ""
        ).strip()

        if game_key:
            grouped[game_key].append(
                snapshot
            )

    games = []

    verified_hr_closes = 0
    verified_ref_closes = 0

    for game_key, game_snapshots in grouped.items():
        ordered = sorted(
            game_snapshots,
            key=lambda row: (
                parse_datetime(
                    row.get("captured_utc")
                )
                or
                datetime.min.replace(
                    tzinfo=timezone.utc
                ),
                str(
                    row.get("snapshot_id") or ""
                ),
            ),
        )

        latest = ordered[-1]

        tracked_team = tracked_team_for_game(
            ordered
        )

        timeline = [
            timeline_point(
                snapshot,
                tracked_team,
            )
            for snapshot in ordered
            if valid_pre_kickoff_snapshot(
                snapshot
            )
        ]

        hardrock_close_snapshot = latest_valid_snapshot(
            ordered,
            valid_hardrock_snapshot,
        )

        reference_close_snapshot = latest_valid_snapshot(
            ordered,
            valid_reference_snapshot,
        )

        hardrock_close = close_summary(
            hardrock_close_snapshot,
            "HARDROCK",
            tracked_team,
        )

        reference_close = close_summary(
            reference_close_snapshot,
            "REFERENCE",
            tracked_team,
        )

        if (
            hardrock_close
            and
            hardrock_close["quality"]
            ==
            "HARDROCK_VERIFIED_NEAR_CLOSE"
        ):
            verified_hr_closes += 1

        if (
            reference_close
            and
            reference_close["quality"]
            ==
            "REFERENCE_VERIFIED_NEAR_CLOSE"
        ):
            verified_ref_closes += 1

        games.append(
            {
                "game_key": game_key,
                "game_id": latest.get(
                    "game_id"
                ),
                "season": latest.get(
                    "season"
                ),
                "week": latest.get(
                    "week"
                ),
                "kickoff_utc": latest.get(
                    "kickoff_utc"
                ),
                "away_team": latest.get(
                    "away_team"
                ),
                "home_team": latest.get(
                    "home_team"
                ),
                "tracked_team": tracked_team,
                "snapshot_count": len(
                    timeline
                ),
                "hardrock_close": hardrock_close,
                "reference_close": reference_close,
                "timeline": timeline,
            }
        )

    games = sorted(
        games,
        key=lambda game: (
            safe_int(
                game.get("week")
            )
            or
            -1,
            str(
                game.get("kickoff_utc") or ""
            ),
            str(
                game.get("game_key") or ""
            ),
        ),
    )

    return {
        "schema_version": 1,
        "generated_utc": utc_iso(),
        "season": SEASON,
        "games": games,
        "summary": {
            "games_with_history": len(
                games
            ),
            "hardrock_verified_near_close_games": (
                verified_hr_closes
            ),
            "reference_verified_near_close_games": (
                verified_ref_closes
            ),
        },
        "close_quality_rules": {
            "verified_near_close_minutes": VERIFIED_CLOSE_MINUTES,
            "near_close_proxy_minutes": NEAR_CLOSE_MINUTES,
            "same_day_proxy_minutes": SAME_DAY_CLOSE_MINUTES,
            "minimum_reference_books": MIN_REFERENCE_BOOKS,
        },
        "warning": (
            "A close is only labeled VERIFIED_NEAR_CLOSE when a valid market "
            "snapshot was actually captured within 120 minutes of kickoff. "
            "Older snapshots are explicitly labeled proxies."
        ),
    }


# ============================================================================
# ACTUAL WAGERS
# ============================================================================

def load_wagers() -> tuple[list[dict[str, Any]], set[str]]:
    if not WAGER_DB.exists():
        return [], set()

    connection = sqlite3.connect(
        f"file:{WAGER_DB}?mode=ro",
        uri=True,
    )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")

    try:
        if not table_exists(
            connection,
            "wagers",
        ):
            return [], set()

        columns = table_columns(
            connection,
            "wagers",
        )

        rows = connection.execute(
            """
            SELECT *
            FROM wagers
            WHERE season = ?
            ORDER BY placed_utc, wager_id
            """,
            (SEASON,),
        ).fetchall()

        return (
            [
                dict(row)
                for row in rows
            ],
            columns,
        )

    finally:
        connection.close()


def game_snapshot_index(
    snapshots: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )

    for snapshot in snapshots:
        game_key = str(
            snapshot.get("game_key") or ""
        ).strip()

        if game_key:
            result[game_key].append(
                snapshot
            )

    for game_key in result:
        result[game_key] = sorted(
            result[game_key],
            key=lambda row: (
                parse_datetime(
                    row.get("captured_utc")
                )
                or
                datetime.min.replace(
                    tzinfo=timezone.utc
                ),
                str(
                    row.get("snapshot_id") or ""
                ),
            ),
        )

    return dict(result)


def clv_against_snapshot(
    wager: dict[str, Any],
    snapshot: dict[str, Any] | None,
    source: str,
) -> dict[str, Any] | None:
    if snapshot is None:
        return None

    wager_team = str(
        wager.get("wager_team") or ""
    ).strip()

    ticket_line = safe_float(
        wager.get("wager_line")
    )

    if not wager_team or ticket_line is None:
        return None

    field = (
        "hardrock_spread_for_model_favorite"
        if source == "HARDROCK"
        else
        "reference_spread_for_model_favorite"
    )

    close_line = (
        spread_for_team_from_model_side_field(
            snapshot,
            field,
            wager_team,
        )
    )

    if close_line is None:
        return None

    minutes = minutes_before_kickoff(
        snapshot.get("captured_utc"),
        snapshot.get("kickoff_utc"),
    )

    clv_points = (
        ticket_line
        -
        close_line
    )

    result = {
        "source": source,
        "snapshot_id": snapshot.get(
            "snapshot_id"
        ),
        "captured_utc": snapshot.get(
            "captured_utc"
        ),
        "minutes_before_kickoff": rounded(
            minutes,
            2,
        ),
        "quality": close_quality(
            minutes,
            source,
        ),
        "wager_team": wager_team,
        "ticket_spread": rounded(
            ticket_line,
            3,
        ),
        "closing_spread_for_wager_team": rounded(
            close_line,
            3,
        ),
        "clv_points": rounded(
            clv_points,
            3,
        ),
        "reference_market_books": safe_int(
            snapshot.get(
                "reference_market_books"
            )
        ),
        "closing_display": (
            snapshot.get("hardrock_display")
            if source == "HARDROCK"
            else snapshot.get(
                "reference_market_display"
            )
        ),
    }

    if source == "HARDROCK":
        closing_price = hardrock_price_for_team(
            snapshot,
            wager_team,
        )

        ticket_price = safe_int(
            wager.get(
                "american_odds"
            )
        )

        ticket_break_even = (
            american_break_even_probability(
                ticket_price
            )
        )

        closing_break_even = (
            american_break_even_probability(
                closing_price
            )
        )

        result[
            "closing_american_odds_for_wager_team"
        ] = closing_price

        result[
            "ticket_american_odds"
        ] = ticket_price

        result[
            "ticket_break_even_probability"
        ] = rounded(
            ticket_break_even,
            6,
        )

        result[
            "closing_break_even_probability"
        ] = rounded(
            closing_break_even,
            6,
        )

        if (
            ticket_break_even is not None
            and
            closing_break_even is not None
            and
            math.isclose(
                ticket_line,
                close_line,
                abs_tol=1e-9,
            )
        ):
            # At identical spread, a higher later break-even probability means
            # the ticket was cheaper/better than the closing price.
            result[
                "same_line_price_clv_break_even_pp"
            ] = rounded(
                (
                    closing_break_even
                    -
                    ticket_break_even
                )
                *
                100.0,
                4,
            )

        else:
            result[
                "same_line_price_clv_break_even_pp"
            ] = None

    return result


def build_wager_clv_report(
    wagers: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    index = game_snapshot_index(
        snapshots
    )

    rows = []

    positive = 0
    negative = 0
    zero = 0
    no_post_wager_close = 0
    verified_near_close = 0

    for wager in wagers:
        game_key = str(
            wager.get("game_key") or ""
        ).strip()

        placed = parse_datetime(
            wager.get("placed_utc")
        )

        game_snapshots = index.get(
            game_key,
            [],
        )

        if placed is None:
            hardrock_close_snapshot = None
            reference_close_snapshot = None

        else:
            # A CLV close must be at or AFTER the wager timestamp and before
            # kickoff. A market snapshot from before the bet is not CLV.
            hardrock_close_snapshot = latest_valid_snapshot(
                game_snapshots,
                valid_hardrock_snapshot,
                not_before=placed,
            )

            reference_close_snapshot = latest_valid_snapshot(
                game_snapshots,
                valid_reference_snapshot,
                not_before=placed,
            )

        hardrock = clv_against_snapshot(
            wager,
            hardrock_close_snapshot,
            "HARDROCK",
        )

        reference = clv_against_snapshot(
            wager,
            reference_close_snapshot,
            "REFERENCE",
        )

        # Primary CLV uses same-book Hard Rock whenever available.
        primary = (
            hardrock
            if hardrock is not None
            else reference
        )

        if primary is None:
            primary_status = (
                "NO_POST_WAGER_CLOSE_CAPTURE"
            )

            primary_clv = None

            no_post_wager_close += 1

        else:
            primary_status = str(
                primary.get("quality")
                or
                "UNKNOWN"
            )

            primary_clv = safe_float(
                primary.get("clv_points")
            )

            if primary_status.endswith(
                "VERIFIED_NEAR_CLOSE"
            ):
                verified_near_close += 1

            if primary_clv is not None:
                if primary_clv > 0:
                    positive += 1

                elif primary_clv < 0:
                    negative += 1

                else:
                    zero += 1

        rows.append(
            {
                "wager_id": wager.get(
                    "wager_id"
                ),
                "game_key": game_key,
                "game_id": wager.get(
                    "game_id"
                ),
                "week": wager.get(
                    "week"
                ),
                "kickoff_utc": wager.get(
                    "kickoff_utc"
                ),
                "placed_utc": wager.get(
                    "placed_utc"
                ),
                "away_team": wager.get(
                    "away_team"
                ),
                "home_team": wager.get(
                    "home_team"
                ),
                "wager_team": wager.get(
                    "wager_team"
                ),
                "wager_line": safe_float(
                    wager.get(
                        "wager_line"
                    )
                ),
                "american_odds": safe_int(
                    wager.get(
                        "american_odds"
                    )
                ),
                "stake": safe_float(
                    wager.get(
                        "stake"
                    )
                ),
                "wager_status": wager.get(
                    "status"
                ),
                "primary_clv_source": (
                    primary.get("source")
                    if primary
                    else None
                ),
                "primary_clv_status": (
                    primary_status
                ),
                "primary_clv_points": rounded(
                    primary_clv,
                    3,
                ),
                "hardrock_clv": hardrock,
                "reference_clv": reference,
            }
        )

    clv_values = [
        safe_float(
            row.get(
                "primary_clv_points"
            )
        )
        for row in rows
    ]

    clv_values = [
        value
        for value in clv_values
        if value is not None
    ]

    average_clv = (
        sum(clv_values)
        /
        len(clv_values)
        if clv_values
        else None
    )

    return {
        "schema_version": 1,
        "generated_utc": utc_iso(),
        "season": SEASON,
        "wagers": rows,
        "summary": {
            "wagers_logged": len(
                wagers
            ),
            "wagers_with_post_wager_close_capture": (
                len(clv_values)
            ),
            "wagers_without_post_wager_close_capture": (
                no_post_wager_close
            ),
            "verified_near_close_wagers": (
                verified_near_close
            ),
            "positive_clv": positive,
            "negative_clv": negative,
            "zero_clv": zero,
            "average_primary_clv_points": rounded(
                average_clv,
                3,
            ),
        },
        "primary_source_rule": (
            "Use a post-wager Hard Rock pre-kickoff close when available; "
            "otherwise use a post-wager multi-book reference consensus close."
        ),
        "warning": (
            "Only *_VERIFIED_NEAR_CLOSE captures occurred within 120 minutes "
            "of kickoff. Proxy statuses must not be presented as official closes."
        ),
    }


# ============================================================================
# OPTIONAL LEDGER UPDATE
# ============================================================================

def update_ledger_clv(
    clv_report: dict[str, Any],
) -> dict[str, int]:
    if not WAGER_DB.exists():
        return {
            "updated": 0,
            "skipped": 0,
        }

    connection = sqlite3.connect(
        str(WAGER_DB)
    )

    connection.row_factory = sqlite3.Row

    try:
        if not table_exists(
            connection,
            "wagers",
        ):
            raise RuntimeError(
                "wagers table is missing."
            )

        columns = table_columns(
            connection,
            "wagers",
        )

        required = {
            "wager_id",
            "clv_status",
            "clv_points",
        }

        missing = sorted(
            required
            -
            columns
        )

        if missing:
            raise RuntimeError(
                "Wager Ledger is missing CLV columns: "
                + ", ".join(
                    missing
                )
            )

        updated = 0
        skipped = 0

        for row in clv_report.get(
            "wagers",
            [],
        ):
            wager_id = str(
                row.get(
                    "wager_id"
                )
                or
                ""
            ).strip()

            if not wager_id:
                skipped += 1
                continue

            status = str(
                row.get(
                    "primary_clv_status"
                )
                or
                "NO_POST_WAGER_CLOSE_CAPTURE"
            )

            points = safe_float(
                row.get(
                    "primary_clv_points"
                )
            )

            cursor = connection.execute(
                """
                UPDATE wagers
                SET
                    clv_status = ?,
                    clv_points = ?
                WHERE wager_id = ?
                """,
                (
                    status,
                    points,
                    wager_id,
                ),
            )

            if cursor.rowcount == 1:
                updated += 1

            else:
                skipped += 1

        connection.commit()

        return {
            "updated": updated,
            "skipped": skipped,
        }

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ============================================================================
# OUTPUTS
# ============================================================================

def write_market_outputs(
    market_report: dict[str, Any],
) -> None:
    MARKET_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MARKET_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            market_report,
            handle,
            indent=2,
            default=str,
        )

        handle.write(
            "\n"
        )

    rows = []

    for game in market_report.get(
        "games",
        [],
    ):
        for point in game.get(
            "timeline",
            [],
        ):
            rows.append(
                {
                    "game_key": game.get(
                        "game_key"
                    ),
                    "game_id": game.get(
                        "game_id"
                    ),
                    "week": game.get(
                        "week"
                    ),
                    "kickoff_utc": game.get(
                        "kickoff_utc"
                    ),
                    "away_team": game.get(
                        "away_team"
                    ),
                    "home_team": game.get(
                        "home_team"
                    ),
                    "tracked_team": game.get(
                        "tracked_team"
                    ),
                    **point,
                }
            )

    fieldnames = [
        "game_key",
        "game_id",
        "week",
        "kickoff_utc",
        "away_team",
        "home_team",
        "tracked_team",
        "snapshot_id",
        "run_id",
        "captured_utc",
        "minutes_before_kickoff",
        "snapshot_model_favorite",
        "model_spread_for_tracked_team",
        "hardrock_spread_for_tracked_team",
        "hardrock_price_for_tracked_team",
        "reference_spread_for_tracked_team",
        "reference_market_books",
        "model_spread_display",
        "hardrock_display",
        "reference_market_display",
        "strength_label",
        "signal_warning",
    ]

    with MARKET_CSV.open(
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


def write_clv_outputs(
    clv_report: dict[str, Any],
) -> None:
    CLV_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CLV_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            clv_report,
            handle,
            indent=2,
            default=str,
        )

        handle.write(
            "\n"
        )

    rows = []

    for row in clv_report.get(
        "wagers",
        [],
    ):
        hardrock = row.get(
            "hardrock_clv"
        ) or {}

        reference = row.get(
            "reference_clv"
        ) or {}

        rows.append(
            {
                "wager_id": row.get(
                    "wager_id"
                ),
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
                "placed_utc": row.get(
                    "placed_utc"
                ),
                "away_team": row.get(
                    "away_team"
                ),
                "home_team": row.get(
                    "home_team"
                ),
                "wager_team": row.get(
                    "wager_team"
                ),
                "wager_line": row.get(
                    "wager_line"
                ),
                "american_odds": row.get(
                    "american_odds"
                ),
                "stake": row.get(
                    "stake"
                ),
                "wager_status": row.get(
                    "wager_status"
                ),
                "primary_clv_source": row.get(
                    "primary_clv_source"
                ),
                "primary_clv_status": row.get(
                    "primary_clv_status"
                ),
                "primary_clv_points": row.get(
                    "primary_clv_points"
                ),
                "hardrock_close_spread": hardrock.get(
                    "closing_spread_for_wager_team"
                ),
                "hardrock_clv_points": hardrock.get(
                    "clv_points"
                ),
                "hardrock_close_odds": hardrock.get(
                    "closing_american_odds_for_wager_team"
                ),
                "hardrock_same_line_price_clv_pp": hardrock.get(
                    "same_line_price_clv_break_even_pp"
                ),
                "hardrock_close_quality": hardrock.get(
                    "quality"
                ),
                "reference_close_spread": reference.get(
                    "closing_spread_for_wager_team"
                ),
                "reference_clv_points": reference.get(
                    "clv_points"
                ),
                "reference_close_books": reference.get(
                    "reference_market_books"
                ),
                "reference_close_quality": reference.get(
                    "quality"
                ),
            }
        )

    fieldnames = [
        "wager_id",
        "game_key",
        "game_id",
        "week",
        "kickoff_utc",
        "placed_utc",
        "away_team",
        "home_team",
        "wager_team",
        "wager_line",
        "american_odds",
        "stake",
        "wager_status",
        "primary_clv_source",
        "primary_clv_status",
        "primary_clv_points",
        "hardrock_close_spread",
        "hardrock_clv_points",
        "hardrock_close_odds",
        "hardrock_same_line_price_clv_pp",
        "hardrock_close_quality",
        "reference_close_spread",
        "reference_clv_points",
        "reference_close_books",
        "reference_close_quality",
    ]

    with CLV_CSV.open(
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

def print_summary(
    market_report: dict[str, Any],
    clv_report: dict[str, Any],
    ledger_update: dict[str, int] | None,
) -> None:
    market = market_report[
        "summary"
    ]

    clv = clv_report[
        "summary"
    ]

    print(
        "=" * 112
    )

    print(
        "CFB EDGE ENGINE — MARKET HISTORY + CLV V1"
    )

    print(
        "=" * 112
    )

    print(
        "Mode: NO API CALLS / IMMUTABLE PREDICTION HISTORY"
    )

    print(
        f"Season: {SEASON}"
    )

    print(
        "=" * 112
    )

    print()

    print(
        "MARKET HISTORY"
    )

    print(
        "-" * 112
    )

    print(
        f"Games with stored snapshot history:       "
        f"{market['games_with_history']}"
    )

    print(
        f"Hard Rock closes within 120 min:          "
        f"{market['hardrock_verified_near_close_games']}"
    )

    print(
        f"Reference closes within 120 min:          "
        f"{market['reference_verified_near_close_games']}"
    )

    print()

    print(
        "ACTUAL WAGER CLV"
    )

    print(
        "-" * 112
    )

    print(
        f"Actual wagers logged:                     "
        f"{clv['wagers_logged']}"
    )

    print(
        f"With a post-wager closing capture:        "
        f"{clv['wagers_with_post_wager_close_capture']}"
    )

    print(
        f"Without post-wager closing capture:       "
        f"{clv['wagers_without_post_wager_close_capture']}"
    )

    print(
        f"Verified near-close actual wagers:        "
        f"{clv['verified_near_close_wagers']}"
    )

    print(
        f"Positive / Negative / Zero CLV:           "
        f"{clv['positive_clv']} / "
        f"{clv['negative_clv']} / "
        f"{clv['zero_clv']}"
    )

    average = safe_float(
        clv.get(
            "average_primary_clv_points"
        )
    )

    print(
        "Average primary CLV:                     "
        + (
            "N/A"
            if average is None
            else
            f"{average:+.3f} pts"
        )
    )

    if ledger_update is not None:
        print()

        print(
            "WAGER LEDGER CLV UPDATE"
        )

        print(
            "-" * 112
        )

        print(
            f"Rows updated:                             "
            f"{ledger_update['updated']}"
        )

        print(
            f"Rows skipped:                             "
            f"{ledger_update['skipped']}"
        )

    print()

    print(
        "FILES WRITTEN"
    )

    print(
        "-" * 112
    )

    print(
        MARKET_JSON
    )

    print(
        MARKET_CSV
    )

    print(
        CLV_JSON
    )

    print(
        CLV_CSV
    )

    print()

    print(
        "CLV RULE: ticket spread minus later closing spread for the SAME team."
    )

    print(
        "Only *_VERIFIED_NEAR_CLOSE means the final stored quote was captured within 120 minutes of kickoff."
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build line-movement history and actual-wager CLV from local snapshots."
        )
    )

    parser.add_argument(
        "--update-ledger",
        action="store_true",
        help=(
            "Write derived clv_status/clv_points into the Actual Wager Ledger. "
            "Original ticket fields are never changed."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    snapshots = load_prediction_snapshots()

    market_report = build_market_history(
        snapshots
    )

    wagers, wager_columns = load_wagers()

    clv_report = build_wager_clv_report(
        wagers,
        snapshots,
    )

    write_market_outputs(
        market_report
    )

    write_clv_outputs(
        clv_report
    )

    ledger_update = None

    if args.update_ledger:
        ledger_update = update_ledger_clv(
            clv_report
        )

    print_summary(
        market_report,
        clv_report,
        ledger_update,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
