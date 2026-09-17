import json
import os
import sqlite3
import time

import pandas as pd
import requests

from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

DATABASE_FILE = "data/cfb.db"

RAW_DIRECTORY = (
    "data/raw/historical_passing"
)

API_BASE_URL = (
    "https://api.collegefootballdata.com"
)

ENDPOINT = (
    "/games/players"
)

YEARS = [
    2021,
    2022,
    2023,
    2024
]

SEASON_TYPE = "regular"

CLASSIFICATION = "fbs"

CATEGORY = "passing"

REQUEST_TIMEOUT = 60

PAUSE_BETWEEN_REQUESTS = 0.20


# ============================================================
# START
# ============================================================

print(
    "HISTORICAL PASSING BACKFILL — V1"
)

print(
    "=" * 110
)


load_dotenv()


api_key = os.getenv(
    "CFBD_API_KEY"
)


if not api_key:

    raise RuntimeError(
        "CFBD_API_KEY was not found in .env"
    )


headers = {
    "Authorization":
        f"Bearer {api_key}"
}


os.makedirs(
    RAW_DIRECTORY,
    exist_ok=True
)


# ============================================================
# READ THE WEEKS WE ACTUALLY NEED
#
# Rather than guessing Week 1 through Week 16, use the
# schedule already stored in our database.
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


weeks = pd.read_sql_query(

    """
    SELECT DISTINCT
        season,
        week

    FROM games

    WHERE season IN (
        2021,
        2022,
        2023,
        2024
    )

    AND week IS NOT NULL

    ORDER BY
        season,
        week
    """,

    connection
)


connection.close()


if len(
    weeks
) == 0:

    raise RuntimeError(
        "No 2021-2024 weeks were found "
        "in the games table."
    )


print(
    f"Season/week combinations needed: "
    f"{len(weeks):,}"
)


print()

print(
    weeks.to_string(
        index=False
    )
)


# ============================================================
# DOWNLOAD / CACHE
#
# CFBD currently requires a week, team, or conference when
# querying player game stats by year.
#
# We query one week at a time.
#
# If a raw file already exists, we reuse it and make ZERO
# API calls for that week.
# ============================================================

print()

print(
    "=" * 110
)

print(
    "DOWNLOADING PASSING BOX SCORES"
)

print(
    "=" * 110
)


raw_files = []


for _, row in weeks.iterrows():

    year = int(
        row[
            "season"
        ]
    )


    week = int(
        row[
            "week"
        ]
    )


    raw_file = os.path.join(

        RAW_DIRECTORY,

        (
            f"passing_"
            f"{year}_"
            f"week_{week}.json"
        )
    )


    raw_files.append(
        {
            "season":
                year,

            "week":
                week,

            "path":
                raw_file
        }
    )


    # --------------------------------------------------------
    # CACHE HIT
    # --------------------------------------------------------

    if os.path.exists(
        raw_file
    ):

        with open(
            raw_file,
            "r",
            encoding="utf-8"
        ) as file:

            cached = json.load(
                file
            )


        print(

            f"{year} Week {week:>2} | "
            f"CACHE | "
            f"{len(cached):>3} games"
        )


        continue


    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    params = {

        "year":
            year,

        "week":
            week,

        "seasonType":
            SEASON_TYPE,

        "classification":
            CLASSIFICATION,

        "category":
            CATEGORY
    }


    response = requests.get(

        API_BASE_URL
        +
        ENDPOINT,

        headers=headers,

        params=params,

        timeout=REQUEST_TIMEOUT
    )


    if response.status_code != 200:

        print()

        print(
            f"FAILED: {year} Week {week}"
        )

        print(
            f"HTTP status: "
            f"{response.status_code}"
        )

        print(
            response.text[
                :1000
            ]
        )


        raise RuntimeError(
            "Historical passing download failed. "
            "Database has NOT been modified."
        )


    payload = response.json()


    if not isinstance(
        payload,
        list
    ):

        raise RuntimeError(

            f"Unexpected API response for "
            f"{year} Week {week}. "
            f"Database has NOT been modified."
        )


    with open(
        raw_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            payload,

            file,

            indent=2
        )


    print(

        f"{year} Week {week:>2} | "
        f"API   | "
        f"{len(payload):>3} games"
    )


    time.sleep(
        PAUSE_BETWEEN_REQUESTS
    )


# ============================================================
# IMPORTANT CHECK
#
# We reach this point only if EVERY requested week either:
#
# 1. already existed in cache
#
# or
#
# 2. downloaded successfully.
#
# Only now will we prepare database rows.
# ============================================================

print()

print(
    "=" * 110
)

print(
    "FLATTENING PASSING DATA"
)

print(
    "=" * 110
)


database_rows = []


game_ids_seen = set()


for file_info in raw_files:

    path = file_info[
        "path"
    ]


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        payload = json.load(
            file
        )


    for game in payload:

        game_id = game.get(
            "id"
        )


        if game_id is None:

            continue


        try:

            game_id = int(
                game_id
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        game_ids_seen.add(
            game_id
        )


        teams = game.get(
            "teams",
            []
        )


        if not isinstance(
            teams,
            list
        ):

            continue


        for team_data in teams:

            team = team_data.get(
                "team"
            )


            categories = team_data.get(
                "categories",
                []
            )


            if not isinstance(
                categories,
                list
            ):

                continue


            for category_data in categories:

                category_name = (
                    category_data.get(
                        "name"
                    )
                )


                if (
                    str(
                        category_name
                    ).lower()
                    !=
                    CATEGORY
                ):

                    continue


                stat_types = category_data.get(
                    "types",
                    []
                )


                if not isinstance(
                    stat_types,
                    list
                ):

                    continue


                for stat_type_data in stat_types:

                    stat_type = (
                        stat_type_data.get(
                            "name"
                        )
                    )


                    athletes = (
                        stat_type_data.get(
                            "athletes",
                            []
                        )
                    )


                    if not isinstance(
                        athletes,
                        list
                    ):

                        continue


                    for athlete in athletes:

                        athlete_id = athlete.get(
                            "id"
                        )


                        athlete_name = athlete.get(
                            "name"
                        )


                        stat_value = athlete.get(
                            "stat"
                        )


                        if athlete_id is not None:

                            athlete_id = str(
                                athlete_id
                            )


                        if stat_value is not None:

                            stat_value = str(
                                stat_value
                            )


                        database_rows.append(
                            (
                                game_id,
                                team,
                                athlete_id,
                                athlete_name,
                                CATEGORY,
                                stat_type,
                                stat_value
                            )
                        )


print(
    f"Unique API games: "
    f"{len(game_ids_seen):,}"
)


print(
    f"Passing stat rows prepared: "
    f"{len(database_rows):,}"
)


if len(
    database_rows
) == 0:

    raise RuntimeError(
        "Zero passing rows were prepared. "
        "Database has NOT been modified."
    )


# ============================================================
# CHECK THE TARGET TABLE BEFORE MODIFYING IT
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


table_schema = pd.read_sql_query(

    """
    PRAGMA table_info(
        player_game_stat_values
    )
    """,

    connection
)


actual_columns = set(
    table_schema[
        "name"
    ].tolist()
)


required_columns = {
    "game_id",
    "team",
    "athlete_id",
    "athlete_name",
    "category",
    "stat_type",
    "stat_value"
}


missing_columns = (
    required_columns
    -
    actual_columns
)


if missing_columns:

    connection.close()


    raise RuntimeError(

        "player_game_stat_values is missing: "

        +

        ", ".join(
            sorted(
                missing_columns
            )
        )
    )


# ============================================================
# DATABASE TRANSACTION
#
# Delete only historical PASSING rows for 2021-2024.
#
# The existing 2025 player data is left untouched.
#
# If inserting fails, rollback restores the previous state.
# ============================================================

print()

print(
    "=" * 110
)

print(
    "IMPORTING INTO SQLITE"
)

print(
    "=" * 110
)


try:

    connection.execute(
        "BEGIN"
    )


    before_count = connection.execute(

        """
        SELECT COUNT(*)

        FROM player_game_stat_values

        WHERE category = 'passing'

        AND game_id IN (

            SELECT game_id

            FROM games

            WHERE season IN (
                2021,
                2022,
                2023,
                2024
            )
        )
        """

    ).fetchone()[0]


    print(
        f"Historical passing rows "
        f"already in DB: "
        f"{before_count:,}"
    )


    connection.execute(

        """
        DELETE FROM player_game_stat_values

        WHERE category = 'passing'

        AND game_id IN (

            SELECT game_id

            FROM games

            WHERE season IN (
                2021,
                2022,
                2023,
                2024
            )
        )
        """
    )


    connection.executemany(

        """
        INSERT INTO player_game_stat_values (
            game_id,
            team,
            athlete_id,
            athlete_name,
            category,
            stat_type,
            stat_value
        )

        VALUES (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        )
        """,

        database_rows
    )


    connection.commit()


except Exception:

    connection.rollback()

    connection.close()

    raise


# ============================================================
# FINAL COVERAGE AUDIT
# ============================================================

coverage = pd.read_sql_query(

    """
    SELECT
        g.season,

        COUNT(
            DISTINCT p.game_id
        ) AS games_with_passing_stats,

        COUNT(*) AS passing_stat_rows

    FROM player_game_stat_values p

    JOIN games g
        ON g.game_id = p.game_id

    WHERE p.category = 'passing'

    AND g.season BETWEEN 2021 AND 2025

    GROUP BY g.season

    ORDER BY g.season
    """,

    connection
)


# ============================================================
# FBS-FBS MODEL POPULATION COVERAGE
#
# We also compare against the exact game population used by
# our features dataset.
# ============================================================

feature_games = pd.read_csv(

    "data/processed/features_v3.csv",

    usecols=[
        "game_id",
        "season"
    ]
)


passing_game_ids = pd.read_sql_query(

    """
    SELECT DISTINCT
        game_id

    FROM player_game_stat_values

    WHERE category = 'passing'
    """,

    connection
)


connection.close()


passing_game_id_set = set(

    pd.to_numeric(

        passing_game_ids[
            "game_id"
        ],

        errors="coerce"

    ).dropna().astype(
        int
    )
)


feature_games[
    "has_passing_data"
] = (

    pd.to_numeric(

        feature_games[
            "game_id"
        ],

        errors="coerce"

    ).isin(
        passing_game_id_set
    )
)


model_coverage = (

    feature_games[
        feature_games[
            "season"
        ].between(
            2021,
            2025
        )
    ]

    .groupby(
        "season"
    )

    .agg(

        model_games=(
            "game_id",
            "size"
        ),

        games_with_passing_data=(
            "has_passing_data",
            "sum"
        )
    )

    .reset_index()
)


model_coverage[
    "coverage"
] = (

    model_coverage[
        "games_with_passing_data"
    ]

    /

    model_coverage[
        "model_games"
    ]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()

print(
    "=" * 110
)

print(
    "DATABASE PASSING COVERAGE"
)

print(
    "=" * 110
)


print(
    coverage.to_string(
        index=False
    )
)


print()

print(
    "=" * 110
)

print(
    "MODEL-POPULATION PASSING COVERAGE"
)

print(
    "=" * 110
)


print(

    model_coverage.to_string(

        index=False,

        formatters={

            "coverage":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# QB / PASSER SUMMARY
#
# C/ATT is particularly useful because attempts will allow us
# to determine each team's primary passer for each game.
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


passer_summary = pd.read_sql_query(

    """
    SELECT
        g.season,

        COUNT(
            DISTINCT p.athlete_id
        ) AS unique_passers,

        COUNT(
            DISTINCT CASE
                WHEN p.stat_type = 'C/ATT'
                THEN p.game_id || ':' || p.team || ':' || p.athlete_id
            END
        ) AS passer_game_records

    FROM player_game_stat_values p

    JOIN games g
        ON g.game_id = p.game_id

    WHERE p.category = 'passing'

    AND g.season BETWEEN 2021 AND 2025

    GROUP BY g.season

    ORDER BY g.season
    """,

    connection
)


connection.close()


print()

print(
    "=" * 110
)

print(
    "PASSER RECORD SUMMARY"
)

print(
    "=" * 110
)


print(
    passer_summary.to_string(
        index=False
    )
)


print()

print(
    "=" * 110
)

print(
    "BACKFILL COMPLETE"
)

print(
    "=" * 110
)


print(
    "2025 player statistics were not deleted "
    "or modified."
)


print(
    "Historical raw responses were cached in:"
)


print(
    RAW_DIRECTORY
)