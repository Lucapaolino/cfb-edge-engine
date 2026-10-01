"""Immutable storage helpers for NFL research collection V1.

The NFL collection track is deliberately independent of the frozen CFB
research system.  It stores source snapshots and normalized rows only; it
does not create predictions, signals, wager records, or model outputs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from pathlib import Path
from typing import Any, Iterable, Mapping


IMMUTABLE_TABLES = (
    "nfl_market_snapshots",
    "nfl_market_book_quotes",
    "nfl_schedule_snapshots",
    "nfl_schedule_rows",
)


def canonical_json(payload: Any) -> str:
    """Serialize data deterministically for fingerprints and durable storage."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(payload: Any) -> str:
    """Return the SHA-256 fingerprint of a canonical JSON payload."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def connect_writable(db_path: Path) -> sqlite3.Connection:
    """Open and initialize the separate NFL research database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    initialize_schema(connection)
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create append-only NFL collection tables and immutability triggers."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS nfl_market_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            captured_at_utc TEXT NOT NULL,
            source TEXT NOT NULL,
            sport_key TEXT NOT NULL,
            request_json TEXT NOT NULL,
            response_sha256 TEXT NOT NULL,
            raw_file TEXT NOT NULL,
            response_json TEXT NOT NULL,
            requests_remaining TEXT,
            requests_used TEXT,
            requests_last TEXT,
            created_at_utc TEXT NOT NULL,
            CHECK (sport_key = 'americanfootball_nfl')
        );

        CREATE TABLE IF NOT EXISTS nfl_market_book_quotes (
            snapshot_id TEXT NOT NULL,
            odds_api_event_id TEXT NOT NULL,
            commence_time_utc TEXT,
            home_team TEXT,
            away_team TEXT,
            bookmaker_key TEXT NOT NULL,
            bookmaker_title TEXT,
            bookmaker_last_update_utc TEXT,
            h2h_last_update_utc TEXT,
            spread_last_update_utc TEXT,
            total_last_update_utc TEXT,
            home_moneyline REAL,
            away_moneyline REAL,
            home_spread REAL,
            home_spread_price REAL,
            away_spread REAL,
            away_spread_price REAL,
            total REAL,
            over_price REAL,
            under_price REAL,
            raw_bookmaker_json TEXT NOT NULL,
            PRIMARY KEY (snapshot_id, odds_api_event_id, bookmaker_key),
            FOREIGN KEY (snapshot_id)
                REFERENCES nfl_market_snapshots(snapshot_id)
        );

        CREATE TABLE IF NOT EXISTS nfl_schedule_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            captured_at_utc TEXT NOT NULL,
            source TEXT NOT NULL,
            source_version TEXT,
            requested_season INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            response_sha256 TEXT NOT NULL,
            raw_file TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nfl_schedule_rows (
            snapshot_id TEXT NOT NULL,
            game_id TEXT NOT NULL,
            season INTEGER,
            week INTEGER,
            game_type TEXT,
            gameday TEXT,
            weekday TEXT,
            gametime TEXT,
            away_team TEXT,
            away_score REAL,
            home_team TEXT,
            home_score REAL,
            result REAL,
            total REAL,
            overtime INTEGER,
            location TEXT,
            roof TEXT,
            surface TEXT,
            temperature REAL,
            wind REAL,
            raw_game_json TEXT NOT NULL,
            PRIMARY KEY (snapshot_id, game_id),
            FOREIGN KEY (snapshot_id)
                REFERENCES nfl_schedule_snapshots(snapshot_id)
        );
        """
    )

    for table_name in IMMUTABLE_TABLES:
        connection.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS nfl_reject_{table_name}_update
            BEFORE UPDATE ON {table_name}
            BEGIN
                SELECT RAISE(ABORT, 'NFL research records are immutable');
            END;
            """
        )
        connection.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS nfl_reject_{table_name}_delete
            BEFORE DELETE ON {table_name}
            BEGIN
                SELECT RAISE(ABORT, 'NFL research records are immutable');
            END;
            """
        )

    connection.commit()


def insert_market_snapshot(
    connection: sqlite3.Connection,
    snapshot: Mapping[str, Any],
    quote_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Insert one raw market capture plus its normalized bookmaker rows."""
    snapshot_columns = (
        "snapshot_id",
        "captured_at_utc",
        "source",
        "sport_key",
        "request_json",
        "response_sha256",
        "raw_file",
        "response_json",
        "requests_remaining",
        "requests_used",
        "requests_last",
        "created_at_utc",
    )
    quote_columns = (
        "snapshot_id",
        "odds_api_event_id",
        "commence_time_utc",
        "home_team",
        "away_team",
        "bookmaker_key",
        "bookmaker_title",
        "bookmaker_last_update_utc",
        "h2h_last_update_utc",
        "spread_last_update_utc",
        "total_last_update_utc",
        "home_moneyline",
        "away_moneyline",
        "home_spread",
        "home_spread_price",
        "away_spread",
        "away_spread_price",
        "total",
        "over_price",
        "under_price",
        "raw_bookmaker_json",
    )

    quote_values = [
        tuple(row.get(column) for column in quote_columns)
        for row in quote_rows
    ]
    snapshot_column_sql = ", ".join(snapshot_columns)
    quote_column_sql = ", ".join(quote_columns)
    snapshot_placeholders = ", ".join("?" for _ in snapshot_columns)
    quote_placeholders = ", ".join("?" for _ in quote_columns)

    with connection:
        connection.execute(
            f"""
            INSERT INTO nfl_market_snapshots ({snapshot_column_sql})
            VALUES ({snapshot_placeholders})
            """,
            tuple(snapshot.get(column) for column in snapshot_columns),
        )
        if quote_values:
            connection.executemany(
                f"""
                INSERT INTO nfl_market_book_quotes ({quote_column_sql})
                VALUES ({quote_placeholders})
                """,
                quote_values,
            )


def insert_schedule_snapshot(
    connection: sqlite3.Connection,
    snapshot: Mapping[str, Any],
    schedule_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Insert one source schedule/result capture and its normalized game rows."""
    snapshot_columns = (
        "snapshot_id",
        "captured_at_utc",
        "source",
        "source_version",
        "requested_season",
        "row_count",
        "response_sha256",
        "raw_file",
        "response_json",
        "created_at_utc",
    )
    row_columns = (
        "snapshot_id",
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "weekday",
        "gametime",
        "away_team",
        "away_score",
        "home_team",
        "home_score",
        "result",
        "total",
        "overtime",
        "location",
        "roof",
        "surface",
        "temperature",
        "wind",
        "raw_game_json",
    )
    row_values = [
        tuple(row.get(column) for column in row_columns)
        for row in schedule_rows
    ]
    snapshot_column_sql = ", ".join(snapshot_columns)
    row_column_sql = ", ".join(row_columns)
    snapshot_placeholders = ", ".join("?" for _ in snapshot_columns)
    row_placeholders = ", ".join("?" for _ in row_columns)

    with connection:
        connection.execute(
            f"""
            INSERT INTO nfl_schedule_snapshots ({snapshot_column_sql})
            VALUES ({snapshot_placeholders})
            """,
            tuple(snapshot.get(column) for column in snapshot_columns),
        )
        if row_values:
            connection.executemany(
                f"""
                INSERT INTO nfl_schedule_rows ({row_column_sql})
                VALUES ({row_placeholders})
                """,
                row_values,
            )


def database_summary(db_path: Path) -> dict[str, Any]:
    """Return read-only collection status without initializing or changing data."""
    if not db_path.exists():
        return {
            "database_exists": False,
            "database_path": str(db_path),
            "market_snapshots": 0,
            "market_quotes": 0,
            "schedule_snapshots": 0,
            "schedule_rows": 0,
            "immutability_triggers": 0,
            "latest_market_capture_utc": None,
            "latest_schedule_capture_utc": None,
        }

    connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        def count_rows(table_name: str) -> int:
            return int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])

        trigger_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM sqlite_master
                WHERE type = 'trigger'
                  AND name LIKE 'nfl_reject_%'
                """
            ).fetchone()[0]
        )

        return {
            "database_exists": True,
            "database_path": str(db_path),
            "market_snapshots": count_rows("nfl_market_snapshots"),
            "market_quotes": count_rows("nfl_market_book_quotes"),
            "schedule_snapshots": count_rows("nfl_schedule_snapshots"),
            "schedule_rows": count_rows("nfl_schedule_rows"),
            "immutability_triggers": trigger_count,
            "latest_market_capture_utc": connection.execute(
                "SELECT MAX(captured_at_utc) FROM nfl_market_snapshots"
            ).fetchone()[0],
            "latest_schedule_capture_utc": connection.execute(
                "SELECT MAX(captured_at_utc) FROM nfl_schedule_snapshots"
            ).fetchone()[0],
        }
    finally:
        connection.close()
