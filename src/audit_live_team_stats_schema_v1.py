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
    "LIVE TEAM STATS SCHEMA AUDIT — V1"
)

print(
    "=" * 110
)


connection = sqlite3.connect(
    DATABASE_FILE
)


tables_to_check = [

    "team_game_stats",

    "team_game_stat_values"
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
            f"  {name:30} "
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
# COMPLETED 2026 WEEKS
# ============================================================

print()

print(
    "=" * 110
)

print(
    "COMPLETED 2026 WEEKS"
)

print(
    "=" * 110
)


weeks = connection.execute(
    """
    SELECT
        week,
        COUNT(*) AS games

    FROM games

    WHERE
        season = 2026
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL

    GROUP BY week

    ORDER BY week
    """
).fetchall()


for week, games in weeks:

    print(
        f"Week {week}: "
        f"{games:,} completed games"
    )


connection.close()


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