#!/usr/bin/env python3
"""Attach completed CFB results to immutable prediction snapshots and score them.

V2 safety change:
- Exact CFBD game ID is preferred, but the result must still agree on season and teams.
- Fallback matching requires season + week + normalized away team + normalized home team.
- There is NO team-pair-only fallback.
- Derived scoring tables are rebuilt from scratch on every run so previously contaminated
  result matches are automatically removed without touching immutable predictions.
"""

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


ResultIndexes = tuple[
    dict[str, list[GameResult]],
    dict[tuple[int, int, str, str], list[GameResult]],
]


def print_rule(char: str = "=", width: int = 104) -> None:
    print(char * width)


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [
        str(row[1])
        for row in connection.execute(
            f'PRAGMA table_info("{table}")'
        )
    ]


def choose_column(
    columns: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    lookup = {
        column.casefold(): column
        for column in columns
    }
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def discover_games_source(
    connection: sqlite3.Connection,
) -> tuple[str, dict[str, str | None]]:
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]

    # The canonical games table should always win when it exists.
    if "games" in tables:
        tables.remove("games")
        tables.insert(0, "games")

    for table in tables:
        columns = table_columns(
            connection,
            table,
        )
        mapping = {
            "game_id": choose_column(
                columns,
                ("game_id", "id"),
            ),
            "season": choose_column(
                columns,
                ("season", "year"),
            ),
            "week": choose_column(
                columns,
                ("week",),
            ),
            "home_team": choose_column(
                columns,
                ("home_team", "home_team_name", "home"),
            ),
            "away_team": choose_column(
                columns,
                ("away_team", "away_team_name", "away"),
            ),
            "home_points": choose_column(
                columns,
                ("home_points", "home_score", "home_pts"),
            ),
            "away_points": choose_column(
                columns,
                ("away_points", "away_score", "away_pts"),
            ),
            "completed": choose_column(
                columns,
                ("completed", "is_completed"),
            ),
        }

        required = (
            "home_team",
            "away_team",
            "home_points",
            "away_points",
        )
        if all(mapping[key] for key in required):
            return table, mapping

    raise RuntimeError(
        "Could not find a completed-games table with home/away teams "
        "and points in data/cfb.db."
    )


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def completed_value_is_true(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return int(value) == 1

    text = str(value).strip().casefold()
    return text in {
        "1",
        "true",
        "yes",
        "y",
        "complete",
        "completed",
        "final",
    }


def load_completed_games() -> tuple[str, list[GameResult]]:
    if not CFB_DB.exists():
        raise FileNotFoundError(
            f"CFB database not found: {CFB_DB}"
        )

    connection = sqlite3.connect(
        CFB_DB
    )
    connection.row_factory = sqlite3.Row

    try:
        table, mapping = discover_games_source(
            connection
        )

        selected: list[tuple[str, str]] = []
        for logical, actual in mapping.items():
            if actual:
                selected.append(
                    (
                        logical,
                        actual,
                    )
                )

        select_sql = ", ".join(
            f"{quote_identifier(actual)} AS {quote_identifier(logical)}"
            for logical, actual in selected
        )
        sql = (
            f"SELECT {select_sql} "
            f"FROM {quote_identifier(table)}"
        )

        results: list[GameResult] = []

        for row in connection.execute(
            sql
        ):
            # If the source table has an explicit completed flag, require it.
            # If it does not, non-null final scores are the completion signal,
            # matching the behavior of the original scorer.
            if (
                "completed" in row.keys()
                and
                not completed_value_is_true(
                    row["completed"]
                )
            ):
                continue

            home_points = safe_float(
                row["home_points"]
            )
            away_points = safe_float(
                row["away_points"]
            )
            home_team = str(
                row["home_team"] or ""
            ).strip()
            away_team = str(
                row["away_team"] or ""
            ).strip()

            if (
                home_points is None
                or
                away_points is None
                or
                not home_team
                or
                not away_team
            ):
                continue

            results.append(
                GameResult(
                    game_id=(
                        str(
                            row["game_id"]
                        ).strip()
                        if (
                            "game_id" in row.keys()
                            and
                            row["game_id"] is not None
                        )
                        else None
                    ),
                    season=(
                        safe_int(
                            row["season"]
                        )
                        if "season" in row.keys()
                        else None
                    ),
                    week=(
                        safe_int(
                            row["week"]
                        )
                        if "week" in row.keys()
                        else None
                    ),
                    away_team=away_team,
                    home_team=home_team,
                    away_points=away_points,
                    home_points=home_points,
                )
            )

        return table, results

    finally:
        connection.close()


def build_result_indexes(
    results: list[GameResult],
) -> ResultIndexes:
    by_id: dict[
        str,
        list[GameResult],
    ] = {}
    by_full: dict[
        tuple[int, int, str, str],
        list[GameResult],
    ] = {}

    for result in results:
        if result.game_id:
            by_id.setdefault(
                result.game_id,
                [],
            ).append(
                result
            )

        # A fallback result is only indexable when season and week both exist.
        # This is intentionally strict: a repeated matchup from another year
        # must never be allowed to satisfy a 2026 prediction.
        if (
            result.season is not None
            and
            result.week is not None
        ):
            full_key = (
                result.season,
                result.week,
                normalize_team(
                    result.away_team
                ),
                normalize_team(
                    result.home_team
                ),
            )
            by_full.setdefault(
                full_key,
                [],
            ).append(
                result
            )

    return by_id, by_full


def result_matches_snapshot_identity(
    snapshot: sqlite3.Row,
    result: GameResult,
) -> bool:
    snapshot_season = safe_int(
        snapshot["season"]
    )
    snapshot_week = safe_int(
        snapshot["week"]
    )

    # Season is mandatory for scoring. This is the core guard that prevents
    # old meetings between the same teams from being attached to a new pick.
    if (
        snapshot_season is None
        or
        result.season is None
        or
        snapshot_season != result.season
    ):
        return False

    snapshot_away = normalize_team(
        snapshot["away_team"]
    )
    snapshot_home = normalize_team(
        snapshot["home_team"]
    )

    if (
        not snapshot_away
        or
        not snapshot_home
        or
        snapshot_away != normalize_team(
            result.away_team
        )
        or
        snapshot_home != normalize_team(
            result.home_team
        )
    ):
        return False

    # If both sides know the week, they must agree. Exact game ID remains
    # usable when one source lacks week, but a known disagreement is rejected.
    if (
        snapshot_week is not None
        and
        result.week is not None
        and
        snapshot_week != result.week
    ):
        return False

    return True


def match_result(
    snapshot: sqlite3.Row,
    indexes: ResultIndexes,
) -> GameResult | None:
    by_id, by_full = indexes

    game_id = (
        str(
            snapshot["game_id"]
        ).strip()
        if snapshot["game_id"] is not None
        else None
    )

    # Preferred path: exact CFBD game ID, plus identity sanity checks.
    if game_id:
        id_matches = [
            result
            for result in by_id.get(
                game_id,
                [],
            )
            if result_matches_snapshot_identity(
                snapshot,
                result,
            )
        ]
        if len(id_matches) == 1:
            return id_matches[0]
        if len(id_matches) > 1:
            # Ambiguous source data: fail closed instead of guessing.
            return None

    # Safe fallback: season + week + away team + home team.
    snapshot_season = safe_int(
        snapshot["season"]
    )
    snapshot_week = safe_int(
        snapshot["week"]
    )
    away = normalize_team(
        snapshot["away_team"]
    )
    home = normalize_team(
        snapshot["home_team"]
    )

    if (
        snapshot_season is None
        or
        snapshot_week is None
        or
        not away
        or
        not home
    ):
        return None

    full_key = (
        snapshot_season,
        snapshot_week,
        away,
        home,
    )
    full_matches = [
        result
        for result in by_full.get(
            full_key,
            [],
        )
        if result_matches_snapshot_identity(
            snapshot,
            result,
        )
    ]

    if len(full_matches) == 1:
        return full_matches[0]

    # There is deliberately NO away-team/home-team-only fallback here.
    # If the exact game cannot be established, leave the snapshot unscored.
    return None


def implied_home_margin_from_execution(
    snapshot: sqlite3.Row,
) -> float | None:
    spread = safe_float(
        snapshot[
            "hardrock_spread_for_model_favorite"
        ]
    )
    favorite = normalize_team(
        snapshot["model_favorite"]
    )
    home = normalize_team(
        snapshot["home_team"]
    )
    away = normalize_team(
        snapshot["away_team"]
    )

    if spread is None:
        return None

    if favorite == home:
        return -spread

    if favorite == away:
        return spread

    return None


def score_ats(
    snapshot: sqlite3.Row,
    actual_home_margin: float,
) -> tuple[str | None, float | None]:
    spread = safe_float(
        snapshot[
            "hardrock_spread_for_model_favorite"
        ]
    )
    price = safe_float(
        snapshot[
            "hardrock_price_for_model_favorite"
        ]
    )
    favorite = normalize_team(
        snapshot["model_favorite"]
    )
    home = normalize_team(
        snapshot["home_team"]
    )
    away = normalize_team(
        snapshot["away_team"]
    )

    if spread is None or not favorite:
        return None, None

    if favorite == home:
        team_margin = actual_home_margin
    elif favorite == away:
        team_margin = -actual_home_margin
    else:
        return None, None

    cover_margin = team_margin + spread

    if math.isclose(
        cover_margin,
        0.0,
        abs_tol=1e-9,
    ):
        return "P", 0.0

    if cover_margin > 0:
        profit = american_profit_for_one_unit(
            price
        )
        return "W", profit

    return (
        "L",
        -1.0 if price is not None else None,
    )


def existing_scored_snapshot_ids(
    history: sqlite3.Connection,
) -> set[str]:
    return {
        str(row[0])
        for row in history.execute(
            "SELECT snapshot_id FROM prediction_scores"
        )
    }


def clear_derived_scoring_tables(
    history: sqlite3.Connection,
) -> tuple[int, int]:
    """Clear only derived outcomes/scores, never immutable predictions."""
    score_count = int(
        history.execute(
            "SELECT COUNT(*) FROM prediction_scores"
        ).fetchone()[0]
    )
    result_count = int(
        history.execute(
            "SELECT COUNT(*) FROM game_results"
        ).fetchone()[0]
    )

    # Foreign-key order matters: scores reference game_results.
    history.execute(
        "DELETE FROM prediction_scores"
    )
    history.execute(
        "DELETE FROM game_results"
    )

    return score_count, result_count


def score_history() -> dict[str, int]:
    source_table, results = load_completed_games()
    indexes = build_result_indexes(
        results
    )

    history = connect_history()

    try:
        snapshots = history.execute(
            "SELECT * FROM prediction_snapshots "
            "ORDER BY captured_utc, snapshot_id"
        ).fetchall()

        previous_ids = existing_scored_snapshot_ids(
            history
        )
        previous_score_rows, previous_result_rows = (
            clear_derived_scoring_tables(
                history
            )
        )

        matched = 0
        unmatched = 0
        now = utc_iso()
        current_scored_ids: set[str] = set()

        for snapshot in snapshots:
            result = match_result(
                snapshot,
                indexes,
            )

            if result is None:
                unmatched += 1
                continue

            matched += 1

            actual_home_margin = (
                result.actual_home_margin
            )
            model_home_margin = safe_float(
                snapshot["model_home_margin"]
            )
            model_home_p = safe_float(
                snapshot[
                    "model_home_win_probability"
                ]
            )

            margin_error = None
            absolute_error = None
            squared_error = None

            if model_home_margin is not None:
                margin_error = (
                    actual_home_margin
                    -
                    model_home_margin
                )
                absolute_error = abs(
                    margin_error
                )
                squared_error = (
                    margin_error
                    *
                    margin_error
                )

            actual_home_win: int | None
            if result.home_points > result.away_points:
                actual_home_win = 1
            elif result.away_points > result.home_points:
                actual_home_win = 0
            else:
                actual_home_win = None

            favorite = normalize_team(
                snapshot["model_favorite"]
            )
            predicted_home = (
                favorite
                ==
                normalize_team(
                    snapshot["home_team"]
                )
            )
            predicted_away = (
                favorite
                ==
                normalize_team(
                    snapshot["away_team"]
                )
            )

            winner_correct = None
            if (
                actual_home_win is not None
                and
                (
                    predicted_home
                    or
                    predicted_away
                )
            ):
                winner_correct = int(
                    (
                        actual_home_win == 1
                        and
                        predicted_home
                    )
                    or
                    (
                        actual_home_win == 0
                        and
                        predicted_away
                    )
                )

            brier = None
            if (
                actual_home_win is not None
                and
                model_home_p is not None
            ):
                brier = (
                    model_home_p
                    -
                    actual_home_win
                ) ** 2

            hardrock_home_margin = (
                implied_home_margin_from_execution(
                    snapshot
                )
            )
            hardrock_abs_error = None
            if hardrock_home_margin is not None:
                hardrock_abs_error = abs(
                    actual_home_margin
                    -
                    hardrock_home_margin
                )

            ats_result, ats_units = score_ats(
                snapshot,
                actual_home_margin,
            )

            game_key = str(
                snapshot["game_key"]
            )

            history.execute(
                """
                INSERT INTO game_results (
                    game_key,
                    game_id,
                    season,
                    week,
                    away_team,
                    home_team,
                    away_points,
                    home_points,
                    actual_home_margin,
                    winner,
                    result_source,
                    result_captured_utc
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

            history.execute(
                """
                INSERT INTO prediction_scores (
                    snapshot_id,
                    game_key,
                    scored_utc,
                    actual_home_margin,
                    predicted_home_margin,
                    margin_error,
                    absolute_margin_error,
                    squared_margin_error,
                    winner_correct,
                    brier_score,
                    hardrock_implied_home_margin,
                    hardrock_absolute_margin_error,
                    model_side_ats_result,
                    model_side_flat_units
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

            current_scored_ids.add(
                str(
                    snapshot["snapshot_id"]
                )
            )

        history.commit()

        removed_invalid = len(
            previous_ids
            -
            current_scored_ids
        )
        newly_scored = len(
            current_scored_ids
            -
            previous_ids
        )
        retained_scored = len(
            current_scored_ids
            &
            previous_ids
        )

        return {
            "source_rows": len(results),
            "snapshots": len(snapshots),
            "matched": matched,
            "newly_scored": newly_scored,
            "retained_scored": retained_scored,
            "removed_invalid": removed_invalid,
            "unmatched": unmatched,
            "previous_score_rows": previous_score_rows,
            "previous_result_rows": previous_result_rows,
            "current_score_rows": len(current_scored_ids),
        }

    except Exception:
        history.rollback()
        raise

    finally:
        history.close()


def main() -> int:
    print_rule()
    print(
        "PREDICTION HISTORY SCORER — V2 STRICT GAME MATCHING"
    )
    print_rule()
    print(
        f"History DB: {HISTORY_DB}"
    )
    print(
        f"Results DB: {CFB_DB}"
    )
    print(
        "Purpose:    score old pre-kickoff predictions only when the exact game can be established"
    )
    print(
        "Safety:     NO team-pair-only fallback; season/team identity is mandatory"
    )
    print(
        "Cleanup:    derived scores/results are rebuilt; immutable predictions are untouched"
    )
    print_rule()

    stats = score_history()

    print()
    print(
        "SCORING STATUS"
    )
    print_rule(
        "-"
    )
    print(
        f"Completed game rows available: {stats['source_rows']}"
    )
    print(
        f"Prediction snapshots stored:   {stats['snapshots']}"
    )
    print(
        f"Snapshots safely matched:      {stats['matched']}"
    )
    print(
        f"Still awaiting/unmatched:      {stats['unmatched']}"
    )
    print()
    print(
        "DERIVED SCORE REBUILD"
    )
    print_rule(
        "-"
    )
    print(
        f"Previous score rows:           {stats['previous_score_rows']}"
    )
    print(
        f"Previous result rows:          {stats['previous_result_rows']}"
    )
    print(
        f"Retained valid snapshots:      {stats['retained_scored']}"
    )
    print(
        f"Newly scoreable snapshots:     {stats['newly_scored']}"
    )
    print(
        f"Invalid old scores removed:    {stats['removed_invalid']}"
    )
    print(
        f"Current score rows:            {stats['current_score_rows']}"
    )
    print()
    print(
        "Prediction snapshots and prediction runs were NOT modified."
    )
    print(
        "Only the derived game_results and prediction_scores tables were rebuilt."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
