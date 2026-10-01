#!/usr/bin/env python3
"""Offline contract test for the NFL schedule-to-market linkage audit."""

from __future__ import annotations

import tempfile

from pathlib import Path

from nfl_research_linkage_v1 import (
    build_linkage_summary,
    canonical_team_code,
    match_market_events_to_schedule,
)
from nfl_research_store_v1 import (
    canonical_json,
    connect_writable,
    insert_market_snapshot,
    insert_schedule_snapshot,
    sha256_json,
)


def market_snapshot(snapshot_id: str) -> dict[str, object]:
    payload = [{"id": "event-1"}]
    return {
        "snapshot_id": snapshot_id,
        "captured_at_utc": "2026-09-01T00:00:00+00:00",
        "source": "test_source",
        "sport_key": "americanfootball_nfl",
        "request_json": canonical_json({"markets": "h2h,spreads,totals"}),
        "response_sha256": sha256_json(payload),
        "raw_file": f"data/nfl/raw/market_snapshots/{snapshot_id}.json",
        "response_json": canonical_json(payload),
        "requests_remaining": "999",
        "requests_used": "1",
        "requests_last": "1",
        "created_at_utc": "2026-09-01T00:00:01+00:00",
    }


def market_quote(snapshot_id: str, bookmaker_key: str) -> dict[str, object]:
    return {
        "snapshot_id": snapshot_id,
        "odds_api_event_id": "event-1",
        # Thursday night is Friday in UTC. The one-day window is intentional.
        "commence_time_utc": "2026-09-11T00:20:00Z",
        "home_team": "Kansas City Chiefs",
        "away_team": "Los Angeles Chargers",
        "bookmaker_key": bookmaker_key,
        "bookmaker_title": bookmaker_key.title(),
        "bookmaker_last_update_utc": "2026-09-01T00:00:00Z",
        "h2h_last_update_utc": "2026-09-01T00:00:00Z",
        "spread_last_update_utc": "2026-09-01T00:00:00Z",
        "total_last_update_utc": "2026-09-01T00:00:00Z",
        "home_moneyline": -130,
        "away_moneyline": 110,
        "home_spread": -2.5,
        "home_spread_price": -110,
        "away_spread": 2.5,
        "away_spread_price": -110,
        "total": 45.5,
        "over_price": -110,
        "under_price": -110,
        "raw_bookmaker_json": canonical_json({"key": bookmaker_key}),
    }


def schedule_snapshot(snapshot_id: str) -> dict[str, object]:
    payload = [{"game_id": "2026_01_LAC_KC"}]
    return {
        "snapshot_id": snapshot_id,
        "captured_at_utc": "2026-09-01T00:00:00+00:00",
        "source": "test_source",
        "source_version": "test",
        "requested_season": 2026,
        "row_count": 1,
        "response_sha256": sha256_json(payload),
        "raw_file": f"data/nfl/raw/schedule_snapshots/{snapshot_id}.json",
        "response_json": canonical_json(payload),
        "created_at_utc": "2026-09-01T00:00:01+00:00",
    }


def schedule_game(snapshot_id: str) -> dict[str, object]:
    raw = {"game_id": "2026_01_LAC_KC"}
    return {
        "snapshot_id": snapshot_id,
        "game_id": "2026_01_LAC_KC",
        "season": 2026,
        "week": 1,
        "game_type": "REG",
        "gameday": "2026-09-10",
        "weekday": "Thursday",
        "gametime": "20:20",
        "away_team": "LAC",
        "away_score": None,
        "home_team": "KC",
        "home_score": None,
        "result": None,
        "total": None,
        "overtime": 0,
        "location": "Home",
        "roof": "dome",
        "surface": "fieldturf",
        "temperature": None,
        "wind": None,
        "raw_game_json": canonical_json(raw),
    }


def main() -> int:
    assert canonical_team_code("Kansas City Chiefs") == "KC"
    assert canonical_team_code("LAC") == "LAC"
    assert canonical_team_code("LA") == "LAR"
    assert canonical_team_code("Not a team") is None

    direct = match_market_events_to_schedule(
        [
            {
                "odds_api_event_id": "event-1",
                "commence_time_utc": "2026-09-11T00:20:00Z",
                "home_team": "Kansas City Chiefs",
                "away_team": "Los Angeles Chargers",
            }
        ],
        [schedule_game("schedule-test")],
    )
    assert direct[0]["status"] == "linked"
    assert direct[0]["game_id"] == "2026_01_LAC_KC"
    assert direct[0]["same_matchup_schedule_games"][0]["gameday"] == "2026-09-10"

    date_conflict = match_market_events_to_schedule(
        [
            {
                "odds_api_event_id": "event-1",
                "commence_time_utc": "2026-10-11T00:20:00Z",
                "home_team": "Kansas City Chiefs",
                "away_team": "Los Angeles Chargers",
            }
        ],
        [schedule_game("schedule-test")],
    )
    assert date_conflict[0]["status"] == "unmatched"
    assert date_conflict[0]["same_matchup_schedule_games"][0]["game_id"] == "2026_01_LAC_KC"

    ambiguous = match_market_events_to_schedule(
        [
            {
                "odds_api_event_id": "event-1",
                "commence_time_utc": "2026-09-11T00:20:00Z",
                "home_team": "Kansas City Chiefs",
                "away_team": "Los Angeles Chargers",
            }
        ],
        [
            schedule_game("schedule-test"),
            {**schedule_game("schedule-test"), "game_id": "duplicate"},
        ],
    )
    assert ambiguous[0]["status"] == "ambiguous"

    with tempfile.TemporaryDirectory(prefix="nfl_research_linkage_") as temp_dir:
        db_path = Path(temp_dir) / "nfl_research.sqlite"
        connection = connect_writable(db_path)
        try:
            insert_schedule_snapshot(
                connection,
                schedule_snapshot("schedule-1"),
                [schedule_game("schedule-1")],
            )
            insert_market_snapshot(
                connection,
                market_snapshot("market-1"),
                [
                    market_quote("market-1", "book_one"),
                    market_quote("market-1", "book_two"),
                ],
            )
        finally:
            connection.close()

        summary = build_linkage_summary(db_path)
        assert summary["linkage_status"] == "PASS"
        assert summary["market_event_count"] == 1
        assert summary["market_bookmaker_rows"] == 2
        assert summary["market_bookmaker_count"] == 2
        assert summary["schedule_game_count"] == 1
        assert summary["linked_event_count"] == 1
        assert summary["unmatched_event_count"] == 0
        assert summary["ambiguous_event_count"] == 0

    print("PASS     NFL team identity crosswalk")
    print("PASS     UTC-boundary market event links to nflverse schedule")
    print("PASS     unmatched events retain same-matchup schedule diagnostics")
    print("PASS     ambiguous links are never forced")
    print("PASS     isolated read-only linkage summary")
    print("PASS     no CFB file or provider call required")
    print("NFL RESEARCH LINKAGE TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
