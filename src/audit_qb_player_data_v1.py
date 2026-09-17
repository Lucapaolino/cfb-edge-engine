import sqlite3
import re

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATABASE_FILE = "data/cfb.db"

PLAYER_STATS_TABLE = "player_game_stat_values"

PLAYERS_TABLE = "players"


# ============================================================
# HELPERS
# ============================================================

def normalize_name(value):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )


def find_column(
    columns,
    candidates
):

    normalized_columns = {
        normalize_name(column): column
        for column in columns
    }


    for candidate in candidates:

        normalized_candidate = normalize_name(
            candidate
        )


        if normalized_candidate in normalized_columns:

            return normalized_columns[
                normalized_candidate
            ]


    return None


def quote_identifier(
    value
):

    return (
        '"'
        +
        str(value).replace(
            '"',
            '""'
        )
        +
        '"'
    )


# ============================================================
# START
# ============================================================

print(
    "QB / PLAYER PASSING DATA AUDIT — V1"
)

print(
    "=" * 100
)


connection = sqlite3.connect(
    DATABASE_FILE
)


# ============================================================
# PLAYER GAME STAT SCHEMA
# ============================================================

schema = pd.read_sql_query(

    f"""
    PRAGMA table_info(
        {PLAYER_STATS_TABLE}
    )
    """,

    connection
)


if len(schema) == 0:

    raise RuntimeError(
        f"{PLAYER_STATS_TABLE} was not found."
    )


columns = schema[
    "name"
].tolist()


print()

print(
    "PLAYER_GAME_STAT_VALUES COLUMNS"
)

print(
    "-" * 100
)


for column in columns:

    print(
        column
    )


# ============================================================
# DETECT IMPORTANT FIELDS
# ============================================================

game_id_column = find_column(
    columns,
    [
        "game_id",
        "gameId"
    ]
)


athlete_id_column = find_column(
    columns,
    [
        "athlete_id",
        "athleteId",
        "player_id",
        "playerId"
    ]
)


team_id_column = find_column(
    columns,
    [
        "team_id",
        "teamId"
    ]
)


category_column = find_column(
    columns,
    [
        "category"
    ]
)


stat_type_column = find_column(
    columns,
    [
        "stat_type",
        "statType",
        "stat_name",
        "statName",
        "type"
    ]
)


stat_value_column = find_column(
    columns,
    [
        "stat_value",
        "statValue",
        "value"
    ]
)


print()

print(
    "DETECTED PLAYER STAT FIELDS"
)

print(
    "-" * 100
)


print(
    f"Game ID:      {game_id_column}"
)

print(
    f"Athlete ID:   {athlete_id_column}"
)

print(
    f"Team ID:      {team_id_column}"
)

print(
    f"Category:     {category_column}"
)

print(
    f"Stat type:    {stat_type_column}"
)

print(
    f"Stat value:   {stat_value_column}"
)


# ============================================================
# SAMPLE RAW ROWS
# ============================================================

print()

print(
    "SAMPLE RAW PLAYER STAT ROWS"
)

print(
    "-" * 100
)


sample = pd.read_sql_query(

    f"""
    SELECT *
    FROM {PLAYER_STATS_TABLE}
    LIMIT 30
    """,

    connection
)


print(
    sample.to_string(
        index=False
    )
)


# ============================================================
# CATEGORY AUDIT
# ============================================================

if category_column is not None:

    category_sql = quote_identifier(
        category_column
    )


    categories = pd.read_sql_query(

        f"""
        SELECT
            {category_sql} AS category,
            COUNT(*) AS rows
        FROM {PLAYER_STATS_TABLE}
        GROUP BY {category_sql}
        ORDER BY rows DESC
        """,

        connection
    )


    print()

    print(
        "PLAYER STAT CATEGORIES"
    )

    print(
        "-" * 100
    )


    print(
        categories.to_string(
            index=False
        )
    )


# ============================================================
# PASSING STAT TYPES
# ============================================================

if (
    category_column is not None
    and
    stat_type_column is not None
):

    category_sql = quote_identifier(
        category_column
    )


    stat_type_sql = quote_identifier(
        stat_type_column
    )


    passing_types = pd.read_sql_query(

        f"""
        SELECT
            {stat_type_sql} AS stat_type,
            COUNT(*) AS rows
        FROM {PLAYER_STATS_TABLE}
        WHERE LOWER(
            CAST(
                {category_sql}
                AS TEXT
            )
        ) LIKE '%pass%'
        GROUP BY {stat_type_sql}
        ORDER BY rows DESC
        """,

        connection
    )


    print()

    print(
        "PASSING STAT TYPES"
    )

    print(
        "-" * 100
    )


    print(
        passing_types.to_string(
            index=False
        )
    )


# ============================================================
# PASSING COVERAGE BY SEASON
# ============================================================

if (
    game_id_column is not None
    and
    category_column is not None
):

    game_id_sql = quote_identifier(
        game_id_column
    )


    category_sql = quote_identifier(
        category_column
    )


    coverage = pd.read_sql_query(

        f"""
        SELECT
            g.season AS season,

            COUNT(
                DISTINCT p.{game_id_sql}
            ) AS games_with_passing_stats,

            COUNT(*) AS passing_stat_rows

        FROM {PLAYER_STATS_TABLE} p

        JOIN games g
            ON g.game_id = p.{game_id_sql}

        WHERE LOWER(
            CAST(
                p.{category_sql}
                AS TEXT
            )
        ) LIKE '%pass%'

        GROUP BY g.season

        ORDER BY g.season
        """,

        connection
    )


    print()

    print(
        "PASSING DATA COVERAGE BY SEASON"
    )

    print(
        "-" * 100
    )


    print(
        coverage.to_string(
            index=False
        )
    )


# ============================================================
# SAMPLE PASSING ROWS WITH SEASON
# ============================================================

if (
    game_id_column is not None
    and
    category_column is not None
):

    game_id_sql = quote_identifier(
        game_id_column
    )


    category_sql = quote_identifier(
        category_column
    )


    query = f"""
        SELECT
            g.season,
            g.week,
            p.*
        FROM {PLAYER_STATS_TABLE} p

        JOIN games g
            ON g.game_id = p.{game_id_sql}

        WHERE LOWER(
            CAST(
                p.{category_sql}
                AS TEXT
            )
        ) LIKE '%pass%'

        ORDER BY
            g.season DESC,
            g.week DESC

        LIMIT 50
    """


    passing_sample = pd.read_sql_query(

        query,

        connection
    )


    print()

    print(
        "SAMPLE PASSING ROWS"
    )

    print(
        "-" * 100
    )


    print(
        passing_sample.to_string(
            index=False
        )
    )


# ============================================================
# PLAYER TABLE SCHEMA
# ============================================================

player_schema = pd.read_sql_query(

    f"""
    PRAGMA table_info(
        {PLAYERS_TABLE}
    )
    """,

    connection
)


print()

print(
    "PLAYERS TABLE COLUMNS"
)

print(
    "-" * 100
)


if len(
    player_schema
) == 0:

    print(
        "players table was not found."
    )


else:

    for column in player_schema[
        "name"
    ].tolist():

        print(
            column
        )


    print()

    print(
        "SAMPLE PLAYERS"
    )

    print(
        "-" * 100
    )


    player_sample = pd.read_sql_query(

        f"""
        SELECT *
        FROM {PLAYERS_TABLE}
        LIMIT 20
        """,

        connection
    )


    print(
        player_sample.to_string(
            index=False
        )
    )


connection.close()


# ============================================================
# FINISH
# ============================================================

print()

print(
    "=" * 100
)

print(
    "AUDIT COMPLETE"
)

print(
    "=" * 100
)


print(
    "No database data was modified."
)