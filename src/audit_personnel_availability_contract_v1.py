#!/usr/bin/env python3
"""
CFB Edge Engine — Personnel Availability Contract Audit V1

Read-only audit before building the personnel/injury layer.

It inspects:
- current live-board identity fields
- SQLite player-stat tables
- game/week tables
- roster/position tables
- athlete IDs / player names / team fields

Safety:
- NO API calls
- NO imports of project scripts
- NO database writes
- NO file mutations
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

CFB_DB = DATA_DIR / "cfb.db"
BOARD_JSON = PROCESSED_DIR / "live_model_market_board_2026_v1.json"

WIDTH = 112


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def safe_count(
    connection: sqlite3.Connection,
    table: str,
) -> int:
    try:
        return int(
            connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
        )
    except sqlite3.Error:
        return -1


def table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        f'PRAGMA table_info("{table}")'
    ).fetchall()

    return [
        {
            "name": str(row[1]),
            "type": str(row[2]),
            "notnull": int(row[3]),
            "pk": int(row[5]),
        }
        for row in rows
    ]


def table_names(
    connection: sqlite3.Connection,
) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """
        ).fetchall()
    ]


def normalized_column_set(
    columns: list[dict[str, Any]],
) -> set[str]:
    return {
        item["name"].casefold()
        for item in columns
    }


def has_any(
    columns: set[str],
    names: set[str],
) -> bool:
    return any(
        name.casefold() in columns
        for name in names
    )


def player_stat_candidate(
    columns: set[str],
) -> bool:
    requirements = [
        {"game_id", "gameid"},
        {"team", "team_name"},
        {"athlete_id", "player_id", "athleteid"},
        {"athlete", "player", "player_name", "athlete_name"},
        {"category"},
        {"stat_type", "stat", "stat_name", "type"},
        {"stat_value", "value"},
    ]

    return all(
        has_any(columns, requirement)
        for requirement in requirements
    )


def game_candidate(
    columns: set[str],
) -> bool:
    return (
        has_any(columns, {"id", "game_id", "gameid"})
        and
        has_any(columns, {"season", "year"})
        and
        has_any(columns, {"week"})
    )


def roster_candidate(
    columns: set[str],
) -> bool:
    return (
        has_any(columns, {"team", "team_name", "school"})
        and
        has_any(columns, {"athlete_id", "player_id", "id", "athleteid"})
        and
        has_any(columns, {"athlete", "player", "player_name", "name", "athlete_name"})
        and
        has_any(columns, {"position", "pos"})
    )


def sample_rows(
    connection: sqlite3.Connection,
    table: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    try:
        rows = connection.execute(
            f'SELECT * FROM "{table}" LIMIT ?',
            (limit,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    except sqlite3.Error as error:
        return [
            {
                "__read_error__": str(error)
            }
        ]


def print_table(
    connection: sqlite3.Connection,
    table: str,
    title: str,
) -> None:
    columns = table_columns(
        connection,
        table,
    )

    print()
    rule("-")
    print(
        f"{title}: {table}"
    )
    rule("-")

    print(
        f"Rows: {safe_count(connection, table)}"
    )

    print(
        "Columns:"
    )

    for column in columns:
        flags = []

        if column["pk"]:
            flags.append("PK")

        if column["notnull"]:
            flags.append("NOT NULL")

        flag_text = (
            " [" + ", ".join(flags) + "]"
            if flags
            else
            ""
        )

        print(
            f"  {column['name']:<32} "
            f"{column['type']}{flag_text}"
        )

    print(
        "Sample rows:"
    )

    for row in sample_rows(
        connection,
        table,
    ):
        print(
            "  "
            +
            json.dumps(
                row,
                sort_keys=True,
                default=str,
            )
        )


def normalize_board_records(
    payload: Any,
) -> list[dict[str, Any]]:
    if isinstance(
        payload,
        list,
    ):
        return [
            row
            for row in payload
            if isinstance(row, dict)
        ]

    if isinstance(
        payload,
        dict,
    ):
        for key in (
            "rows",
            "records",
            "games",
            "signals",
            "board",
            "data",
        ):
            value = payload.get(key)

            if isinstance(
                value,
                list,
            ):
                rows = [
                    row
                    for row in value
                    if isinstance(row, dict)
                ]

                if rows:
                    return rows

    return []


def inspect_board() -> None:
    rule()
    print(
        "LIVE BOARD CONTRACT"
    )
    rule()

    print(
        f"Path: {BOARD_JSON}"
    )

    if not BOARD_JSON.exists():
        print(
            "STATUS: MISSING"
        )
        return

    payload = json.loads(
        BOARD_JSON.read_text(
            encoding="utf-8"
        )
    )

    rows = normalize_board_records(
        payload
    )

    print(
        f"Rows: {len(rows)}"
    )

    if not rows:
        return

    first = rows[0]

    print(
        "First-row keys:"
    )

    for key in sorted(
        first.keys()
    ):
        print(
            f"  {key}"
        )

    print()
    print(
        "First five game identities:"
    )

    for row in rows[:5]:
        print(
            "  "
            +
            json.dumps(
                {
                    "game_id": row.get("game_id"),
                    "season": row.get("season"),
                    "week": row.get("week"),
                    "kickoff_utc": (
                        row.get("kickoff_utc")
                        or
                        row.get("start_date")
                    ),
                    "away_team": row.get("away_team"),
                    "home_team": row.get("home_team"),
                },
                sort_keys=True,
                default=str,
            )
        )


def inspect_database() -> None:
    print()
    rule()
    print(
        "CFB DATABASE CONTRACT"
    )
    rule()

    print(
        f"Path: {CFB_DB}"
    )

    if not CFB_DB.exists():
        print(
            "STATUS: MISSING"
        )
        return

    connection = sqlite3.connect(
        f"file:{CFB_DB}?mode=ro",
        uri=True,
    )

    connection.row_factory = sqlite3.Row
    connection.execute(
        "PRAGMA query_only = ON"
    )

    try:
        tables = table_names(
            connection
        )

        print(
            f"Tables: {len(tables)}"
        )

        player_tables = []
        game_tables = []
        roster_tables = []

        for table in tables:
            columns = table_columns(
                connection,
                table,
            )

            normalized = normalized_column_set(
                columns
            )

            if player_stat_candidate(
                normalized
            ):
                player_tables.append(
                    table
                )

            if game_candidate(
                normalized
            ):
                game_tables.append(
                    table
                )

            if roster_candidate(
                normalized
            ):
                roster_tables.append(
                    table
                )

        print()
        print(
            "Detected player-stat table candidates:"
        )

        if player_tables:
            for table in player_tables:
                print(
                    f"  {table}"
                )
        else:
            print(
                "  NONE"
            )

        print()
        print(
            "Detected game/week table candidates:"
        )

        if game_tables:
            for table in game_tables:
                print(
                    f"  {table}"
                )
        else:
            print(
                "  NONE"
            )

        print()
        print(
            "Detected roster/position table candidates:"
        )

        if roster_tables:
            for table in roster_tables:
                print(
                    f"  {table}"
                )
        else:
            print(
                "  NONE"
            )

        for table in player_tables:
            print_table(
                connection,
                table,
                "PLAYER STAT TABLE"
            )

        for table in roster_tables:
            print_table(
                connection,
                table,
                "ROSTER / POSITION TABLE"
            )

        # Show only the strongest-looking game candidates so the output
        # remains manageable.
        prioritized_games = sorted(
            game_tables,
            key=lambda name: (
                "game" not in name.casefold(),
                name.casefold(),
            ),
        )

        for table in prioritized_games[:3]:
            print_table(
                connection,
                table,
                "GAME / WEEK TABLE"
            )

    finally:
        connection.close()


def print_decision() -> None:
    print()
    rule()
    print(
        "NEXT PERSONNEL BUILD"
    )
    rule()

    print(
        "Paste this output back into the project chat."
    )

    print()
    print(
        "The next whole-file implementation will use these exact local contracts to:"
    )

    print(
        "  1. create an immutable personnel_availability.sqlite research store,"
    )

    print(
        "  2. enforce source publication time < kickoff,"
    )

    print(
        "  3. resolve player/team/game identity without team-pair-only fallbacks,"
    )

    print(
        "  4. calculate season-to-date player production importance,"
    )

    print(
        "  5. treat QB availability separately,"
    )

    print(
        "  6. emit personnel-loss research context WITHOUT changing the frozen model."
    )


def main() -> int:
    rule()
    print(
        "CFB EDGE ENGINE — PERSONNEL AVAILABILITY CONTRACT AUDIT V1"
    )
    rule()

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        "Mode: READ ONLY / NO API CALLS / NO DATABASE WRITES"
    )

    inspect_board()
    inspect_database()
    print_decision()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
