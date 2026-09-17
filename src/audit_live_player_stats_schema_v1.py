import sqlite3
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATABASE_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "cfb.db"
)


# ============================================================
# START
# ============================================================

print(
    "LIVE PLAYER STATS SCHEMA AUDIT — V1"
)

print(
    "=" * 110
)


connection = sqlite3.connect(
    DATABASE_FILE
)


# ============================================================
# TABLE SCHEMAS
# ============================================================

tables_to_check = [
    "players",
    "player_game_stat_values"
]


for table in tables_to_check:

    print()

    print(
        table.upper()
    )

    print(
        "-" * 110
    )


    exists = connection.execute(
        """
        SELECT name

        FROM sqlite_master

        WHERE
            type = 'table'
            AND name = ?
        """,
        (
            table,
        )
    ).fetchone()


    if exists is None:

        print(
            "TABLE DOES NOT EXIST"
        )

        continue


    columns = connection.execute(
        f"""
        PRAGMA table_info({table})
        """
    ).fetchall()


    print(
        "COLUMNS"
    )


    for column in columns:

        (
            cid,
            name,
            data_type,
            not_null,
            default_value,
            primary_key
        ) = column


        print(
            f"  {name:32} "
            f"{data_type:12} "
            f"PK={primary_key}"
        )


    count = connection.execute(
        f"""
        SELECT COUNT(*)

        FROM {table}
        """
    ).fetchone()[0]


    print()

    print(
        f"Total rows: {count:,}"
    )


    print()

    print(
        "SAMPLE ROWS"
    )


    sample = connection.execute(
        f"""
        SELECT *

        FROM {table}

        LIMIT 5
        """
    ).fetchall()


    for row in sample:

        print(
            row
        )


# ============================================================
# PASSING ROW EXAMPLES
# ============================================================

print()

print(
    "=" * 110
)

print(
    "HISTORICAL PASSING DATA"
)

print(
    "=" * 110
)


passing_examples = connection.execute(
    """
    SELECT *

    FROM player_game_stat_values

    WHERE LOWER(category) = 'passing'

    LIMIT 15
    """
).fetchall()


for row in passing_examples:

    print(
        row
    )


# ============================================================
# DISTINCT PASSING STAT TYPES
# ============================================================

columns = connection.execute(
    """
    PRAGMA table_info(player_game_stat_values)
    """
).fetchall()


column_names = {
    row[1]
    for row in columns
}


if "stat_type" in column_names:

    print()

    print(
        "PASSING STAT TYPES"
    )

    print(
        "-" * 110
    )


    passing_types = connection.execute(
        """
        SELECT
            stat_type,
            COUNT(*)

        FROM player_game_stat_values

        WHERE LOWER(category) = 'passing'

        GROUP BY stat_type

        ORDER BY stat_type
        """
    ).fetchall()


    for (
        stat_type,
        count
    ) in passing_types:

        print(
            f"{stat_type:30} "
            f"{count:,}"
        )


# ============================================================
# 2026 PLAYER DATA COVERAGE
#
# Determine whether any player rows already exist for the
# newly imported 2026 game IDs.
# ============================================================

print()

print(
    "=" * 110
)

print(
    "2026 PLAYER DATA COVERAGE"
)

print(
    "=" * 110
)


player_game_columns = {
    row[1]
    for row in connection.execute(
        """
        PRAGMA table_info(player_game_stat_values)
        """
    ).fetchall()
}


if "game_id" in player_game_columns:

    player_rows_2026 = connection.execute(
        """
        SELECT COUNT(*)

        FROM player_game_stat_values AS p

        INNER JOIN games AS g
            ON g.game_id = p.game_id

        WHERE g.season = 2026
        """
    ).fetchone()[0]


    passing_rows_2026 = connection.execute(
        """
        SELECT COUNT(*)

        FROM player_game_stat_values AS p

        INNER JOIN games AS g
            ON g.game_id = p.game_id

        WHERE
            g.season = 2026
            AND LOWER(p.category) = 'passing'
        """
    ).fetchone()[0]


    games_with_player_stats = connection.execute(
        """
        SELECT COUNT(
            DISTINCT g.game_id
        )

        FROM games AS g

        INNER JOIN player_game_stat_values AS p
            ON p.game_id = g.game_id

        WHERE
            g.season = 2026
            AND g.home_points IS NOT NULL
            AND g.away_points IS NOT NULL
        """
    ).fetchone()[0]


    games_with_passing = connection.execute(
        """
        SELECT COUNT(
            DISTINCT g.game_id
        )

        FROM games AS g

        INNER JOIN player_game_stat_values AS p
            ON p.game_id = g.game_id

        WHERE
            g.season = 2026
            AND g.home_points IS NOT NULL
            AND g.away_points IS NOT NULL
            AND LOWER(p.category) = 'passing'
        """
    ).fetchone()[0]


    print(
        f"2026 player-stat rows:       "
        f"{player_rows_2026:,}"
    )


    print(
        f"2026 passing-stat rows:      "
        f"{passing_rows_2026:,}"
    )


    print(
        f"Games with player stats:     "
        f"{games_with_player_stats:,}/99"
    )


    print(
        f"Games with passing stats:    "
        f"{games_with_passing:,}/99"
    )


# ============================================================
# RECENT HISTORICAL SAMPLE
#
# Helps us verify how 2025 records map players to games/teams.
# ============================================================

print()

print(
    "=" * 110
)

print(
    "2025 PASSING SAMPLE"
)

print(
    "=" * 110
)


recent_sample = connection.execute(
    """
    SELECT
        p.*

    FROM player_game_stat_values AS p

    INNER JOIN games AS g
        ON g.game_id = p.game_id

    WHERE
        g.season = 2025
        AND LOWER(p.category) = 'passing'

    LIMIT 20
    """
).fetchall()


for row in recent_sample:

    print(
        row
    )


connection.close()


# ============================================================
# COMPLETE
# ============================================================

print()

print(
    "=" * 110
)

print(
    "AUDIT COMPLETE"
)

print(
    "=" * 110
)


print()

print(
    "Next step:"
)


print(
    "  Build the live 2026 player/QB updater using "
    "this exact schema."
)