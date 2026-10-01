"""Read-only schedule-to-market linkage checks for NFL research V1.

This module verifies that a captured Odds API event can be linked to a game in
the latest captured nflverse schedule snapshot.  It makes no provider calls,
does not alter either source ledger, and never creates a model, prediction,
signal, or wager record.

The matching rule is deliberately strict: home team, away team, and a small
UTC-date window must agree.  A possible match is reported as ambiguous rather
than silently forcing a game identifier.
"""

from __future__ import annotations

import re
import sqlite3

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping


REQUIRED_TABLES = (
    "nfl_market_snapshots",
    "nfl_market_book_quotes",
    "nfl_schedule_snapshots",
    "nfl_schedule_rows",
)


def _team_key(value: Any) -> str:
    """Normalize a team label without guessing a team identity."""
    text = re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper())
    return " ".join(text.split())


# nflverse schedule rows use the short code at right.  The Odds API uses the
# full display name, so this is an explicit cross-source identity contract.
TEAM_CODE_BY_ALIAS = {
    "ARI": "ARI",
    "ARIZONA CARDINALS": "ARI",
    "ATL": "ATL",
    "ATLANTA FALCONS": "ATL",
    "BAL": "BAL",
    "BALTIMORE RAVENS": "BAL",
    "BUF": "BUF",
    "BUFFALO BILLS": "BUF",
    "CAR": "CAR",
    "CAROLINA PANTHERS": "CAR",
    "CHI": "CHI",
    "CHICAGO BEARS": "CHI",
    "CIN": "CIN",
    "CINCINNATI BENGALS": "CIN",
    "CLE": "CLE",
    "CLEVELAND BROWNS": "CLE",
    "DAL": "DAL",
    "DALLAS COWBOYS": "DAL",
    "DEN": "DEN",
    "DENVER BRONCOS": "DEN",
    "DET": "DET",
    "DETROIT LIONS": "DET",
    "GB": "GB",
    "GREEN BAY PACKERS": "GB",
    "HOU": "HOU",
    "HOUSTON TEXANS": "HOU",
    "IND": "IND",
    "INDIANAPOLIS COLTS": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "JACKSONVILLE JAGUARS": "JAX",
    "KC": "KC",
    "KANSAS CITY CHIEFS": "KC",
    "LV": "LV",
    "LAS VEGAS RAIDERS": "LV",
    "LAC": "LAC",
    "LA CHARGERS": "LAC",
    "LOS ANGELES CHARGERS": "LAC",
    # nflverse represents the Rams with LA; LAC remains the Chargers.
    "LA": "LAR",
    "LAR": "LAR",
    "LA RAMS": "LAR",
    "LOS ANGELES RAMS": "LAR",
    "MIA": "MIA",
    "MIAMI DOLPHINS": "MIA",
    "MIN": "MIN",
    "MINNESOTA VIKINGS": "MIN",
    "NE": "NE",
    "NEW ENGLAND PATRIOTS": "NE",
    "NO": "NO",
    "NEW ORLEANS SAINTS": "NO",
    "NYG": "NYG",
    "NEW YORK GIANTS": "NYG",
    "NYJ": "NYJ",
    "NEW YORK JETS": "NYJ",
    "PHI": "PHI",
    "PHILADELPHIA EAGLES": "PHI",
    "PIT": "PIT",
    "PITTSBURGH STEELERS": "PIT",
    "SEA": "SEA",
    "SEATTLE SEAHAWKS": "SEA",
    "SF": "SF",
    "SAN FRANCISCO 49ERS": "SF",
    "TB": "TB",
    "TAMPA BAY BUCCANEERS": "TB",
    "TEN": "TEN",
    "TENNESSEE TITANS": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
    "WASHINGTON COMMANDERS": "WAS",
}


def canonical_team_code(value: Any) -> str | None:
    """Return a known NFL team code or ``None`` for an unknown label."""
    return TEAM_CODE_BY_ALIAS.get(_team_key(value))


def _iso_date(value: Any) -> date | None:
    """Read an ISO date or timestamp without converting or changing it."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _event_date_window(commence_time_utc: Any) -> set[date]:
    """Allow a one-day UTC boundary around the scheduled local game date."""
    event_date = _iso_date(commence_time_utc)
    if event_date is None:
        return set()
    return {
        event_date - timedelta(days=1),
        event_date,
        event_date + timedelta(days=1),
    }


def _event_label(event: Mapping[str, Any]) -> str:
    away = str(event.get("away_team") or "Unknown away team")
    home = str(event.get("home_team") or "Unknown home team")
    return f"{away} @ {home}"


def match_market_events_to_schedule(
    market_events: list[Mapping[str, Any]],
    schedule_games: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Link market events to schedules, never forcing an uncertain match."""
    results: list[dict[str, Any]] = []

    for event in market_events:
        home_code = canonical_team_code(event.get("home_team"))
        away_code = canonical_team_code(event.get("away_team"))
        date_window = _event_date_window(event.get("commence_time_utc"))
        base_result = {
            "odds_api_event_id": str(event.get("odds_api_event_id") or ""),
            "event_label": _event_label(event),
            "commence_time_utc": event.get("commence_time_utc"),
            "home_team_code": home_code,
            "away_team_code": away_code,
        }

        if home_code is None or away_code is None:
            results.append(
                {
                    **base_result,
                    "status": "unmatched",
                    "reason": "unknown_team_identity",
                    "game_id": None,
                    "candidate_game_ids": [],
                    "same_matchup_schedule_games": [],
                }
            )
            continue

        if not date_window:
            results.append(
                {
                    **base_result,
                    "status": "unmatched",
                    "reason": "missing_or_invalid_commence_time",
                    "game_id": None,
                    "candidate_game_ids": [],
                    "same_matchup_schedule_games": [],
                }
            )
            continue

        same_matchup_schedule_games = sorted(
            [
                {
                    "game_id": str(game.get("game_id") or ""),
                    "gameday": game.get("gameday"),
                    "week": game.get("week"),
                    "game_type": game.get("game_type"),
                }
                for game in schedule_games
                if canonical_team_code(game.get("home_team")) == home_code
                and canonical_team_code(game.get("away_team")) == away_code
            ],
            key=lambda game: (
                str(game.get("gameday") or ""),
                str(game.get("game_id") or ""),
            ),
        )
        candidates = [
            game
            for game in schedule_games
            if canonical_team_code(game.get("home_team")) == home_code
            and canonical_team_code(game.get("away_team")) == away_code
            and _iso_date(game.get("gameday")) in date_window
        ]
        candidate_ids = sorted(
            str(game.get("game_id") or "")
            for game in candidates
            if str(game.get("game_id") or "")
        )

        if len(candidates) == 1:
            results.append(
                {
                    **base_result,
                    "status": "linked",
                    "reason": "home_away_and_date_window_match",
                    "game_id": candidate_ids[0],
                    "candidate_game_ids": candidate_ids,
                    "same_matchup_schedule_games": same_matchup_schedule_games,
                }
            )
        elif not candidates:
            results.append(
                {
                    **base_result,
                    "status": "unmatched",
                    "reason": "no_home_away_date_match",
                    "game_id": None,
                    "candidate_game_ids": [],
                    "same_matchup_schedule_games": same_matchup_schedule_games,
                }
            )
        else:
            results.append(
                {
                    **base_result,
                    "status": "ambiguous",
                    "reason": "multiple_home_away_date_matches",
                    "game_id": None,
                    "candidate_game_ids": candidate_ids,
                    "same_matchup_schedule_games": same_matchup_schedule_games,
                }
            )

    return results


def _empty_summary(db_path: Path, status: str, message: str) -> dict[str, Any]:
    return {
        "database_exists": db_path.exists(),
        "database_path": str(db_path),
        "linkage_status": status,
        "message": message,
        "latest_market_snapshot_id": None,
        "latest_market_capture_utc": None,
        "latest_schedule_snapshot_id": None,
        "latest_schedule_capture_utc": None,
        "market_event_count": 0,
        "market_bookmaker_rows": 0,
        "market_bookmaker_count": 0,
        "schedule_game_count": 0,
        "linked_event_count": 0,
        "unmatched_event_count": 0,
        "ambiguous_event_count": 0,
        "unmatched_event_examples": [],
        "ambiguous_event_examples": [],
    }


def _latest_snapshot(
    connection: sqlite3.Connection,
    table_name: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        f"""
        SELECT snapshot_id, captured_at_utc
        FROM {table_name}
        ORDER BY captured_at_utc DESC, snapshot_id DESC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row is not None else None


def _required_tables_present(connection: sqlite3.Connection) -> bool:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()
    present = {str(row[0]) for row in rows}
    return all(table_name in present for table_name in REQUIRED_TABLES)


def build_linkage_summary(db_path: Path) -> dict[str, Any]:
    """Return a read-only cross-source linkage report for the latest snapshots."""
    if not db_path.exists():
        return _empty_summary(
            db_path,
            "NOT INITIALIZED",
            "No NFL research database is present yet.",
        )

    connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if not _required_tables_present(connection):
            return _empty_summary(
                db_path,
                "SCHEMA NOT READY",
                "The NFL research database does not contain the V1 source tables.",
            )

        latest_market = _latest_snapshot(connection, "nfl_market_snapshots")
        latest_schedule = _latest_snapshot(connection, "nfl_schedule_snapshots")

        if latest_market is None and latest_schedule is None:
            return _empty_summary(
                db_path,
                "WAITING FOR SOURCES",
                "Capture an NFL schedule and reference-market snapshot first.",
            )
        if latest_market is None:
            return _empty_summary(
                db_path,
                "WAITING FOR MARKET",
                "A schedule exists, but no NFL reference-market snapshot exists yet.",
            )
        if latest_schedule is None:
            return _empty_summary(
                db_path,
                "WAITING FOR SCHEDULE",
                "A market snapshot exists, but no NFL schedule snapshot exists yet.",
            )

        market_metrics = connection.execute(
            """
            SELECT
                COUNT(*) AS bookmaker_rows,
                COUNT(DISTINCT odds_api_event_id) AS event_count,
                COUNT(DISTINCT bookmaker_key) AS bookmaker_count
            FROM nfl_market_book_quotes
            WHERE snapshot_id = ?
            """,
            (latest_market["snapshot_id"],),
        ).fetchone()
        market_events = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    odds_api_event_id,
                    MAX(commence_time_utc) AS commence_time_utc,
                    MAX(home_team) AS home_team,
                    MAX(away_team) AS away_team
                FROM nfl_market_book_quotes
                WHERE snapshot_id = ?
                GROUP BY odds_api_event_id
                ORDER BY odds_api_event_id
                """,
                (latest_market["snapshot_id"],),
            ).fetchall()
        ]
        schedule_games = [
            dict(row)
            for row in connection.execute(
                """
                SELECT game_id, gameday, home_team, away_team, week, game_type
                FROM nfl_schedule_rows
                WHERE snapshot_id = ?
                ORDER BY game_id
                """,
                (latest_schedule["snapshot_id"],),
            ).fetchall()
        ]
        results = match_market_events_to_schedule(market_events, schedule_games)
        linked = [item for item in results if item["status"] == "linked"]
        unmatched = [item for item in results if item["status"] == "unmatched"]
        ambiguous = [item for item in results if item["status"] == "ambiguous"]

        if not market_events:
            status = "WAITING FOR MARKET EVENTS"
            message = "The latest market snapshot has no normalized bookmaker events."
        elif not ambiguous and len(linked) == len(market_events):
            status = "PASS"
            message = "Every latest-market event has one unambiguous schedule link."
        else:
            status = "REVIEW"
            message = "Some latest-market events could not be linked unambiguously."

        return {
            "database_exists": True,
            "database_path": str(db_path),
            "linkage_status": status,
            "message": message,
            "latest_market_snapshot_id": latest_market["snapshot_id"],
            "latest_market_capture_utc": latest_market["captured_at_utc"],
            "latest_schedule_snapshot_id": latest_schedule["snapshot_id"],
            "latest_schedule_capture_utc": latest_schedule["captured_at_utc"],
            "market_event_count": len(market_events),
            "market_bookmaker_rows": int(market_metrics["bookmaker_rows"] or 0),
            "market_bookmaker_count": int(market_metrics["bookmaker_count"] or 0),
            "schedule_game_count": len(schedule_games),
            "linked_event_count": len(linked),
            "unmatched_event_count": len(unmatched),
            "ambiguous_event_count": len(ambiguous),
            "unmatched_event_examples": unmatched[:8],
            "ambiguous_event_examples": ambiguous[:8],
        }
    finally:
        connection.close()
