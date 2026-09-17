#!/usr/bin/env python3
"""Attach completed CFB results to immutable prediction snapshots and score them."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any

from prediction_history_v1 import (
    CFB_DB,
    HISTORY_DB,
    american_profit_for_one_unit,
    connect_history,
    normalize_team,
    safe_float,
    safe_int,
    utc_iso,
)


@dataclass(frozen=True)
class GameResult:
    game_id: str | None
    season: int | None
    week: int | None
    away_team: str
    home_team: str
    away_points: float
    home_points: float

    @property
    def actual_home_margin(self) -> float:
        return self.home_points - self.away_points

    @property
    def winner(self) -> str | None:
        if self.home_points > self.away_points:
            return self.home_team
        if self.away_points > self.home_points:
            return self.away_team
        return None


def print_rule(char: str = "=", width: int = 104) -> None:
    print(char * width)


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


def choose_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {column.casefold(): column for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def discover_games_source(connection: sqlite3.Connection) -> tuple[str, dict[str, str | None]]:
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    if "games" in tables:
        tables.remove("games")
        tables.insert(0, "games")

    for table in tables:
        columns = table_columns(connection, table)
        mapping = {
            "game_id": choose_column(columns, ("game_id", "id")),
            "season": choose_column(columns, ("season", "year")),
            "week": choose_column(columns, ("week",)),
            "home_team": choose_column(columns, ("home_team", "home_team_name", "home")),
            "away_team": choose_column(columns, ("away_team", "away_team_name", "away")),
            "home_points": choose_column(columns, ("home_points", "home_score", "home_pts")),
            "away_points": choose_column(columns, ("away_points", "away_score", "away_pts")),
        }
        required = ("home_team", "away_team", "home_points", "away_points")
        if all(mapping[key] for key in required):
            return table, mapping

    raise RuntimeError(
        "Could not find a completed-games table with home/away teams and points in data/cfb.db."
    )


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_completed_games() -> tuple[str, list[GameResult]]:
    if not CFB_DB.exists():
        raise FileNotFoundError(f"CFB database not found: {CFB_DB}")

    connection = sqlite3.connect(CFB_DB)
    connection.row_factory = sqlite3.Row
    try:
        table, mapping = discover_games_source(connection)
        selected: list[tuple[str, str]] = []
        for logical, actual in mapping.items():
            if actual:
                selected.append((logical, actual))

        select_sql = ", ".join(
            f"{quote_identifier(actual)} AS {quote_identifier(logical)}"
            for logical, actual in selected
        )
        sql = f"SELECT {select_sql} FROM {quote_identifier(table)}"

        results: list[GameResult] = []
        for row in connection.execute(sql):
            home_points = safe_float(row["home_points"])
            away_points = safe_float(row["away_points"])
            home_team = str(row["home_team"] or "").strip()
            away_team = str(row["away_team"] or "").strip()
            if home_points is None or away_points is None or not home_team or not away_team:
                continue
            results.append(
                GameResult(
                    game_id=str(row["game_id"]).strip() if "game_id" in row.keys() and row["game_id"] is not None else None,
                    season=safe_int(row["season"]) if "season" in row.keys() else None,
                    week=safe_int(row["week"]) if "week" in row.keys() else None,
                    away_team=away_team,
                    home_team=home_team,
                    away_points=away_points,
                    home_points=home_points,
                )
            )
        return table, results
    finally:
        connection.close()


def build_result_indexes(results: list[GameResult]) -> tuple[dict[str, GameResult], dict[tuple[Any, ...], GameResult], dict[tuple[str, str], list[GameResult]]]:
    by_id: dict[str, GameResult] = {}
    by_full: dict[tuple[Any, ...], GameResult] = {}
    by_teams: dict[tuple[str, str], list[GameResult]] = {}

    for result in results:
        if result.game_id:
            by_id[result.game_id] = result
        full_key = (
            result.season,
            result.week,
            normalize_team(result.away_team),
            normalize_team(result.home_team),
        )
        by_full[full_key] = result
        team_key = (normalize_team(result.away_team), normalize_team(result.home_team))
        by_teams.setdefault(team_key, []).append(result)

    return by_id, by_full, by_teams


def match_result(snapshot: sqlite3.Row, indexes: tuple[dict[str, GameResult], dict[tuple[Any, ...], GameResult], dict[tuple[str, str], list[GameResult]]]) -> GameResult | None:
    by_id, by_full, by_teams = indexes
    game_id = str(snapshot["game_id"]).strip() if snapshot["game_id"] is not None else None
    if game_id and game_id in by_id:
        return by_id[game_id]

    away = normalize_team(snapshot["away_team"])
    home = normalize_team(snapshot["home_team"])
    full_key = (snapshot["season"], snapshot["week"], away, home)
    if full_key in by_full:
        return by_full[full_key]

    team_matches = by_teams.get((away, home), [])
    if len(team_matches) == 1:
        return team_matches[0]
    return None


def implied_home_margin_from_execution(snapshot: sqlite3.Row) -> float | None:
    spread = safe_float(snapshot["hardrock_spread_for_model_favorite"])
    favorite = normalize_team(snapshot["model_favorite"])
    home = normalize_team(snapshot["home_team"])
    away = normalize_team(snapshot["away_team"])
    if spread is None:
        return None
    if favorite == home:
        return -spread
    if favorite == away:
        return spread
    return None


def score_ats(snapshot: sqlite3.Row, actual_home_margin: float) -> tuple[str | None, float | None]:
    spread = safe_float(snapshot["hardrock_spread_for_model_favorite"])
    price = safe_float(snapshot["hardrock_price_for_model_favorite"])
    favorite = normalize_team(snapshot["model_favorite"])
    home = normalize_team(snapshot["home_team"])
    away = normalize_team(snapshot["away_team"])
    if spread is None or not favorite:
        return None, None

    if favorite == home:
        team_margin = actual_home_margin
    elif favorite == away:
        team_margin = -actual_home_margin
    else:
        return None, None

    cover_margin = team_margin + spread
    if math.isclose(cover_margin, 0.0, abs_tol=1e-9):
        return "P", 0.0
    if cover_margin > 0:
        profit = american_profit_for_one_unit(price)
        return "W", profit
    return "L", -1.0 if price is not None else None


def score_history() -> dict[str, int]:
    source_table, results = load_completed_games()
    indexes = build_result_indexes(results)

    history = connect_history()
    try:
        snapshots = history.execute(
            "SELECT * FROM prediction_snapshots ORDER BY captured_utc, snapshot_id"
        ).fetchall()

        matched = 0
        newly_scored = 0
        unmatched = 0
        now = utc_iso()

        for snapshot in snapshots:
            result = match_result(snapshot, indexes)
            if result is None:
                unmatched += 1
                continue
            matched += 1

            actual_home_margin = result.actual_home_margin
            model_home_margin = safe_float(snapshot["model_home_margin"])
            model_home_p = safe_float(snapshot["model_home_win_probability"])

            margin_error = None
            absolute_error = None
            squared_error = None
            if model_home_margin is not None:
                margin_error = actual_home_margin - model_home_margin
                absolute_error = abs(margin_error)
                squared_error = margin_error * margin_error

            actual_home_win: int | None
            if result.home_points > result.away_points:
                actual_home_win = 1
            elif result.away_points > result.home_points:
                actual_home_win = 0
            else:
                actual_home_win = None

            favorite = normalize_team(snapshot["model_favorite"])
            predicted_home = favorite == normalize_team(snapshot["home_team"])
            predicted_away = favorite == normalize_team(snapshot["away_team"])
            winner_correct = None
            if actual_home_win is not None and (predicted_home or predicted_away):
                winner_correct = int((actual_home_win == 1 and predicted_home) or (actual_home_win == 0 and predicted_away))

            brier = None
            if actual_home_win is not None and model_home_p is not None:
                brier = (model_home_p - actual_home_win) ** 2

            hardrock_home_margin = implied_home_margin_from_execution(snapshot)
            hardrock_abs_error = None
            if hardrock_home_margin is not None:
                hardrock_abs_error = abs(actual_home_margin - hardrock_home_margin)

            ats_result, ats_units = score_ats(snapshot, actual_home_margin)

            game_key = str(snapshot["game_key"])
            history.execute(
                """
                INSERT INTO game_results (
                    game_key, game_id, season, week, away_team, home_team,
                    away_points, home_points, actual_home_margin, winner,
                    result_source, result_captured_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_key) DO UPDATE SET
                    game_id=excluded.game_id,
                    season=excluded.season,
                    week=excluded.week,
                    away_team=excluded.away_team,
                    home_team=excluded.home_team,
                    away_points=excluded.away_points,
                    home_points=excluded.home_points,
                    actual_home_margin=excluded.actual_home_margin,
                    winner=excluded.winner,
                    result_source=excluded.result_source,
                    result_captured_utc=excluded.result_captured_utc
                """,
                (
                    game_key,
                    result.game_id,
                    result.season,
                    result.week,
                    result.away_team,
                    result.home_team,
                    result.away_points,
                    result.home_points,
                    actual_home_margin,
                    result.winner,
                    f"data/cfb.db:{source_table}",
                    now,
                ),
            )

            existed = history.execute(
                "SELECT 1 FROM prediction_scores WHERE snapshot_id = ?",
                (snapshot["snapshot_id"],),
            ).fetchone()

            history.execute(
                """
                INSERT INTO prediction_scores (
                    snapshot_id, game_key, scored_utc, actual_home_margin,
                    predicted_home_margin, margin_error, absolute_margin_error,
                    squared_margin_error, winner_correct, brier_score,
                    hardrock_implied_home_margin, hardrock_absolute_margin_error,
                    model_side_ats_result, model_side_flat_units
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    game_key=excluded.game_key,
                    scored_utc=excluded.scored_utc,
                    actual_home_margin=excluded.actual_home_margin,
                    predicted_home_margin=excluded.predicted_home_margin,
                    margin_error=excluded.margin_error,
                    absolute_margin_error=excluded.absolute_margin_error,
                    squared_margin_error=excluded.squared_margin_error,
                    winner_correct=excluded.winner_correct,
                    brier_score=excluded.brier_score,
                    hardrock_implied_home_margin=excluded.hardrock_implied_home_margin,
                    hardrock_absolute_margin_error=excluded.hardrock_absolute_margin_error,
                    model_side_ats_result=excluded.model_side_ats_result,
                    model_side_flat_units=excluded.model_side_flat_units
                """,
                (
                    snapshot["snapshot_id"],
                    game_key,
                    now,
                    actual_home_margin,
                    model_home_margin,
                    margin_error,
                    absolute_error,
                    squared_error,
                    winner_correct,
                    brier,
                    hardrock_home_margin,
                    hardrock_abs_error,
                    ats_result,
                    ats_units,
                ),
            )
            if not existed:
                newly_scored += 1

        history.commit()
        return {
            "source_rows": len(results),
            "snapshots": len(snapshots),
            "matched": matched,
            "newly_scored": newly_scored,
            "unmatched": unmatched,
        }
    finally:
        history.close()


def main() -> int:
    print_rule()
    print("PREDICTION HISTORY SCORER — V1")
    print_rule()
    print(f"History DB: {HISTORY_DB}")
    print(f"Results DB: {CFB_DB}")
    print("Purpose:    score old pre-kickoff predictions after final scores arrive")
    print_rule()

    stats = score_history()
    print()
    print("SCORING STATUS")
    print_rule("-")
    print(f"Completed game rows available: {stats['source_rows']}")
    print(f"Prediction snapshots stored:   {stats['snapshots']}")
    print(f"Snapshots matched to results:  {stats['matched']}")
    print(f"Newly scored this run:         {stats['newly_scored']}")
    print(f"Still awaiting/unmatched:      {stats['unmatched']}")
    print()
    print("Prediction records themselves were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
