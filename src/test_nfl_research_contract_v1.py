#!/usr/bin/env python3
"""Offline contract test for NFL research collection V1.

No provider calls are made.  The test uses a temporary SQLite database and
verifies that the new NFL ledger is append-only and separate from CFB paths.
"""

from __future__ import annotations

import sqlite3
import tempfile

from pathlib import Path

from nfl_research_market_capture_v1 import normalize_quotes
from nfl_research_store_v1 import (
    IMMUTABLE_TABLES,
    canonical_json,
    connect_writable,
    database_summary,
    insert_market_snapshot,
    insert_schedule_snapshot,
    sha256_json,
)


def expect_immutable(operation, label: str) -> None:
    try:
        operation()
    except sqlite3.DatabaseError:
        print(f"PASS     {label}")
        return
    raise AssertionError(f"Expected immutable-record rejection: {label}")


def main() -> int:
    normalized_quote_rows = normalize_quotes(
        "market-1",
        [
            {
                "id": "event-1",
                "commence_time": "2026-10-04T17:00:00Z",
                "home_team": "Home",
                "away_team": "Away",
                "bookmakers": [
                    {
                        "key": "example_book",
                        "title": "Example Book",
                        "last_update": "2026-10-01T00:00:00Z",
                        "markets": [
                            {
                                "key": "h2h",
                                "last_update": "2026-10-01T00:00:00Z",
                                "outcomes": [
                                    {"name": "Home", "price": -120},
                                    {"name": "Away", "price": 100},
                                ],
                            },
                            {
                                "key": "spreads",
                                "last_update": "2026-10-01T00:00:00Z",
                                "outcomes": [
                                    {"name": "Home", "point": -2.5, "price": -110},
                                    {"name": "Away", "point": 2.5, "price": -110},
                                ],
                            },
                            {
                                "key": "totals",
                                "last_update": "2026-10-01T00:00:00Z",
                                "outcomes": [
                                    {"name": "Over", "point": 44.5, "price": -110},
                                    {"name": "Under", "point": 44.5, "price": -110},
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    )
    assert len(normalized_quote_rows) == 1
    assert normalized_quote_rows[0]["home_spread"] == -2.5
    assert normalized_quote_rows[0]["total"] == 44.5

    with tempfile.TemporaryDirectory(prefix="nfl_research_contract_") as temp_dir:
        db_path = Path(temp_dir) / "nfl_research.sqlite"
        connection = connect_writable(db_path)
        try:
            market_payload = [{"id": "event-1", "home_team": "Home", "away_team": "Away"}]
            insert_market_snapshot(
                connection,
                {
                    "snapshot_id": "market-1",
                    "captured_at_utc": "2026-10-01T00:00:00+00:00",
                    "source": "test_source",
                    "sport_key": "americanfootball_nfl",
                    "request_json": canonical_json({"markets": "h2h,spreads,totals"}),
                    "response_sha256": sha256_json(market_payload),
                    "raw_file": "data/nfl/raw/market_snapshots/market-1.json",
                    "response_json": canonical_json(market_payload),
                    "requests_remaining": "999",
                    "requests_used": "1",
                    "requests_last": "1",
                    "created_at_utc": "2026-10-01T00:00:01+00:00",
                },
                [
                    {
                        "snapshot_id": "market-1",
                        "odds_api_event_id": "event-1",
                        "commence_time_utc": "2026-10-04T17:00:00Z",
                        "home_team": "Home",
                        "away_team": "Away",
                        "bookmaker_key": "example_book",
                        "bookmaker_title": "Example Book",
                        "bookmaker_last_update_utc": "2026-10-01T00:00:00Z",
                        "h2h_last_update_utc": "2026-10-01T00:00:00Z",
                        "spread_last_update_utc": "2026-10-01T00:00:00Z",
                        "total_last_update_utc": "2026-10-01T00:00:00Z",
                        "home_moneyline": -120,
                        "away_moneyline": 100,
                        "home_spread": -2.5,
                        "home_spread_price": -110,
                        "away_spread": 2.5,
                        "away_spread_price": -110,
                        "total": 44.5,
                        "over_price": -110,
                        "under_price": -110,
                        "raw_bookmaker_json": canonical_json({"key": "example_book"}),
                    }
                ],
            )

            schedule_payload = [{"game_id": "2026_01_AWY_HME", "season": 2026, "week": 1}]
            insert_schedule_snapshot(
                connection,
                {
                    "snapshot_id": "schedule-1",
                    "captured_at_utc": "2026-10-01T00:00:00+00:00",
                    "source": "test_source",
                    "source_version": "test",
                    "requested_season": 2026,
                    "row_count": 1,
                    "response_sha256": sha256_json(schedule_payload),
                    "raw_file": "data/nfl/raw/schedule_snapshots/schedule-1.json",
                    "response_json": canonical_json(schedule_payload),
                    "created_at_utc": "2026-10-01T00:00:01+00:00",
                },
                [
                    {
                        "snapshot_id": "schedule-1",
                        "game_id": "2026_01_AWY_HME",
                        "season": 2026,
                        "week": 1,
                        "game_type": "REG",
                        "gameday": "2026-09-10",
                        "weekday": "Thursday",
                        "gametime": "20:20",
                        "away_team": "AWY",
                        "away_score": None,
                        "home_team": "HME",
                        "home_score": None,
                        "result": None,
                        "total": None,
                        "overtime": 0,
                        "location": "Home",
                        "roof": "dome",
                        "surface": "fieldturf",
                        "temperature": None,
                        "wind": None,
                        "raw_game_json": canonical_json(schedule_payload[0]),
                    }
                ],
            )

            expect_immutable(
                lambda: connection.execute(
                    "UPDATE nfl_market_snapshots SET source = 'changed' WHERE snapshot_id = 'market-1'"
                ),
                "market snapshot update rejected",
            )
            expect_immutable(
                lambda: connection.execute(
                    "DELETE FROM nfl_schedule_rows WHERE snapshot_id = 'schedule-1'"
                ),
                "schedule row delete rejected",
            )
        finally:
            connection.close()

        summary = database_summary(db_path)
        assert summary["market_snapshots"] == 1
        assert summary["market_quotes"] == 1
        assert summary["schedule_snapshots"] == 1
        assert summary["schedule_rows"] == 1
        assert summary["immutability_triggers"] == len(IMMUTABLE_TABLES) * 2

        print("PASS     isolated temporary NFL ledger")
        print("PASS     NFL h2h/spread/total quote normalization")
        print("PASS     immutable triggers installed")
        print("PASS     no CFB file or provider call required")
        print("NFL RESEARCH CONTRACT TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
