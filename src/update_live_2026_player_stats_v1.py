import json
import os
import sqlite3

from datetime import datetime, timezone
from pathlib import Path

import requests

from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

SEASON = 2026

API_URL = (
    "https://api.collegefootballdata.com/"
    "games/players"
)


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


RAW_DIRECTORY = (
    PROJECT_ROOT
    /
    "data"
    /
    "raw"
    /
    "live_2026"
    /
    "player_stats"
)


# ============================================================
# API KEY
# ============================================================

load_dotenv(
    PROJECT_ROOT
    /
    ".env"
)


API_KEY = os.getenv(
    "CFBD_API_KEY"
)


if not API_KEY:

    raise RuntimeError(
        "CFBD_API_KEY was not found in .env"
    )


# ============================================================
# HELPERS
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )


# ============================================================
# START
# ============================================================

captured_datetime = utc_now()


captured_at = (
    captured_datetime
    .isoformat()
)


batch_id = (
    captured_datetime
    .strftime(
        "%Y%m%dT%H%M%SZ"
    )
)


print(
    "UPDATE LIVE 2026 PLAYER STATS — V1"
)

print(
    "=" * 110
)


print(
    f"Season:       {SEASON}"
)


print(
    f"Captured UTC: {captured_at}"
)


# ============================================================
# DATABASE
# ============================================================

if not DATABASE_FILE.exists():

    raise RuntimeError(
        f"Database does not exist: "
        f"{DATABASE_FILE}"
    )


connection = sqlite3.connect(
    DATABASE_FILE
)


# ============================================================
# VERIFY TABLES
# ============================================================

required_tables = {

    "games",

    "player_game_stat_values"
}


existing_tables = {

    row[0]

    for row in connection.execute(
        """
        SELECT name

        FROM sqlite_master

        WHERE type = 'table'
        """
    ).fetchall()
}


missing_tables = (

    required_tables

    -
    existing_tables
)


if missing_tables:

    connection.close()

    raise RuntimeError(

        "Missing required database tables:\n"

        +

        "\n".join(
            sorted(
                missing_tables
            )
        )
    )


# ============================================================
# VERIFY PLAYER TABLE SCHEMA
# ============================================================

player_stat_columns = {

    row[1]

    for row in connection.execute(
        """
        PRAGMA table_info(
            player_game_stat_values
        )
        """
    ).fetchall()
}


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
    player_stat_columns
)


if missing_columns:

    connection.close()

    raise RuntimeError(

        "player_game_stat_values is "
        "missing required columns:\n"

        +

        "\n".join(
            sorted(
                missing_columns
            )
        )
    )


# ============================================================
# COMPLETED WEEKS
# ============================================================

completed_weeks = connection.execute(
    """
    SELECT DISTINCT week

    FROM games

    WHERE
        season = ?
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
        AND week IS NOT NULL

    ORDER BY week
    """,

    (
        SEASON,
    )
).fetchall()


completed_weeks = [

    int(
        row[0]
    )

    for row in completed_weeks
]


if len(
    completed_weeks
) == 0:

    connection.close()


    print()

    print(
        "No completed 2026 games."
    )


    raise SystemExit(
        0
    )


print()

print(
    "Completed weeks detected:"
)


print(
    "  "
    +
    ", ".join(

        str(
            week
        )

        for week
        in completed_weeks
    )
)


# ============================================================
# COMPLETED GAME IDs
#
# Only games we already know are completed locally are
# eligible for player-stat import.
# ============================================================

completed_game_rows = connection.execute(
    """
    SELECT
        game_id,
        week

    FROM games

    WHERE
        season = ?
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
    """,

    (
        SEASON,
    )
).fetchall()


completed_game_ids = {

    int(
        game_id
    )

    for (
        game_id,
        week
    ) in completed_game_rows
}


print(
    f"Completed games eligible: "
    f"{len(completed_game_ids):,}"
)


# ============================================================
# RAW STORAGE
# ============================================================

RAW_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# NORMALIZED STORAGE
#
# Composite primary key in our database:
#
# game_id
# + team
# + athlete_id
# + category
# + stat_type
#
# Dictionary storage also protects us from duplicate rows
# inside the API response.
# ============================================================

player_rows_by_key = {}


api_game_ids = set()


request_count = 0


category_counts = {}


# ============================================================
# DOWNLOAD COMPLETED WEEKS
# ============================================================

for week in completed_weeks:

    print()

    print(
        "=" * 110
    )

    print(
        f"DOWNLOADING WEEK {week}"
    )

    print(
        "=" * 110
    )


    response = requests.get(

        API_URL,

        params={

            "year":
                SEASON,

            "week":
                week,

            "classification":
                "fbs",

            "seasonType":
                "regular"
        },

        headers={

            "Authorization":
                f"Bearer {API_KEY}"
        },

        timeout=60
    )


    request_count += 1


    print(
        f"HTTP status: "
        f"{response.status_code}"
    )


    if response.status_code != 200:

        connection.close()

        raise RuntimeError(

            f"CFBD Week {week} request failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:2000]}"
        )


    payload = response.json()


    if not isinstance(
        payload,
        list
    ):

        connection.close()

        raise RuntimeError(

            f"Unexpected player-stat API "
            f"response for Week {week}."
        )


    print(
        f"Game player box scores returned: "
        f"{len(payload):,}"
    )


    # ========================================================
    # SAVE RAW RESPONSE
    # ========================================================

    raw_file = (

        RAW_DIRECTORY

        /

        (
            f"player_stats_{SEASON}_"
            f"week_{week}_"
            f"{batch_id}.json"
        )
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
        f"Raw response: {raw_file}"
    )


    # ========================================================
    # NORMALIZE RESPONSE
    # ========================================================

    week_rows = 0

    week_games = set()

    week_categories = {}


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


        # ----------------------------------------------------
        # Never write stats for an upcoming/unverified game.
        # ----------------------------------------------------

        if (
            game_id
            not in completed_game_ids
        ):

            continue


        api_game_ids.add(
            game_id
        )


        week_games.add(
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


        # ====================================================
        # TEAM
        # ====================================================

        for team_entry in teams:

            team_name = team_entry.get(
                "team"
            )


            if not team_name:

                continue


            categories = team_entry.get(
                "categories",
                []
            )


            if not isinstance(
                categories,
                list
            ):

                continue


            # =================================================
            # CATEGORY
            #
            # Examples:
            #
            # passing
            # rushing
            # receiving
            # defensive
            # kicking
            # =================================================

            for category_entry in categories:

                category = category_entry.get(
                    "name"
                )


                if not category:

                    continue


                category = str(
                    category
                )


                types = category_entry.get(
                    "types",
                    []
                )


                if not isinstance(
                    types,
                    list
                ):

                    continue


                # =============================================
                # STAT TYPE
                #
                # Passing examples:
                #
                # C/ATT
                # YDS
                # AVG
                # TD
                # INT
                # QBR
                # =============================================

                for type_entry in types:

                    stat_type = type_entry.get(
                        "name"
                    )


                    if not stat_type:

                        continue


                    stat_type = str(
                        stat_type
                    )


                    athletes = type_entry.get(
                        "athletes",
                        []
                    )


                    if not isinstance(
                        athletes,
                        list
                    ):

                        continue


                    # =========================================
                    # ATHLETE
                    # =========================================

                    for athlete_entry in athletes:

                        athlete_id = athlete_entry.get(
                            "id"
                        )


                        athlete_name = athlete_entry.get(
                            "name"
                        )


                        stat_value = athlete_entry.get(
                            "stat"
                        )


                        if athlete_id is None:

                            continue


                        athlete_id = str(
                            athlete_id
                        )


                        if athlete_name is None:

                            athlete_name = ""


                        else:

                            athlete_name = str(
                                athlete_name
                            )


                        if stat_value is None:

                            stored_value = None


                        else:

                            stored_value = str(
                                stat_value
                            )


                        key = (

                            game_id,

                            team_name,

                            athlete_id,

                            category,

                            stat_type
                        )


                        player_rows_by_key[
                            key
                        ] = (

                            game_id,

                            team_name,

                            athlete_id,

                            athlete_name,

                            category,

                            stat_type,

                            stored_value
                        )


                        week_rows += 1


                        week_categories[
                            category
                        ] = (

                            week_categories.get(
                                category,
                                0
                            )

                            +
                            1
                        )


                        category_counts[
                            category
                        ] = (

                            category_counts.get(
                                category,
                                0
                            )

                            +
                            1
                        )


    print(
        f"Completed games normalized: "
        f"{len(week_games):,}"
    )


    print(
        f"Player-stat rows normalized: "
        f"{week_rows:,}"
    )


    print()

    print(
        "Category rows:"
    )


    for category in sorted(
        week_categories
    ):

        print(
            f"  "
            f"{category:25} "
            f"{week_categories[category]:,}"
        )


# ============================================================
# FINAL ROWS
# ============================================================

player_rows = list(
    player_rows_by_key.values()
)


print()

print(
    "=" * 110
)

print(
    "NORMALIZED 2026 PLAYER DATA"
)

print(
    "=" * 110
)


print(
    f"API requests made:       "
    f"{request_count:,}"
)


print(
    f"Unique games returned:   "
    f"{len(api_game_ids):,}"
)


print(
    f"Unique player-stat rows: "
    f"{len(player_rows):,}"
)


# ============================================================
# UPSERT
# ============================================================

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

    ON CONFLICT(
        game_id,
        team,
        athlete_id,
        category,
        stat_type
    )

    DO UPDATE SET

        athlete_name =
            CASE

                WHEN
                    excluded.athlete_name
                    IS NOT NULL

                    AND
                    excluded.athlete_name != ''

                THEN
                    excluded.athlete_name

                ELSE
                    player_game_stat_values.athlete_name

            END,

        stat_value =
            COALESCE(
                excluded.stat_value,
                player_game_stat_values.stat_value
            )
    """,

    player_rows
)


connection.commit()


# ============================================================
# 2026 COVERAGE
# ============================================================

total_rows_2026 = connection.execute(
    """
    SELECT COUNT(*)

    FROM player_game_stat_values AS p

    INNER JOIN games AS g
        ON g.game_id = p.game_id

    WHERE g.season = ?
    """,

    (
        SEASON,
    )
).fetchone()[0]


passing_rows_2026 = connection.execute(
    """
    SELECT COUNT(*)

    FROM player_game_stat_values AS p

    INNER JOIN games AS g
        ON g.game_id = p.game_id

    WHERE
        g.season = ?
        AND LOWER(p.category) = 'passing'
    """,

    (
        SEASON,
    )
).fetchone()[0]


games_with_any_player_stats = connection.execute(
    """
    SELECT COUNT(
        DISTINCT g.game_id
    )

    FROM games AS g

    INNER JOIN player_game_stat_values AS p
        ON p.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
    """,

    (
        SEASON,
    )
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
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
        AND LOWER(p.category) = 'passing'
    """,

    (
        SEASON,
    )
).fetchone()[0]


# ============================================================
# UNIQUE ATHLETES
# ============================================================

unique_athletes_2026 = connection.execute(
    """
    SELECT COUNT(
        DISTINCT p.athlete_id
    )

    FROM player_game_stat_values AS p

    INNER JOIN games AS g
        ON g.game_id = p.game_id

    WHERE g.season = ?
    """,

    (
        SEASON,
    )
).fetchone()[0]


unique_passers_2026 = connection.execute(
    """
    SELECT COUNT(
        DISTINCT p.athlete_id
    )

    FROM player_game_stat_values AS p

    INNER JOIN games AS g
        ON g.game_id = p.game_id

    WHERE
        g.season = ?
        AND LOWER(p.category) = 'passing'
    """,

    (
        SEASON,
    )
).fetchone()[0]


# ============================================================
# PASSING STAT TYPES
# ============================================================

passing_types = connection.execute(
    """
    SELECT
        p.stat_type,
        COUNT(*)

    FROM player_game_stat_values AS p

    INNER JOIN games AS g
        ON g.game_id = p.game_id

    WHERE
        g.season = ?
        AND LOWER(p.category) = 'passing'

    GROUP BY p.stat_type

    ORDER BY p.stat_type
    """,

    (
        SEASON,
    )
).fetchall()


# ============================================================
# WEEK COVERAGE
# ============================================================

week_coverage = connection.execute(
    """
    SELECT
        g.week,

        COUNT(
            DISTINCT g.game_id
        ) AS completed_games,

        COUNT(
            DISTINCT CASE
                WHEN p.game_id IS NOT NULL
                THEN g.game_id
            END
        ) AS games_with_player_stats

    FROM games AS g

    LEFT JOIN player_game_stat_values AS p
        ON p.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL

    GROUP BY g.week

    ORDER BY g.week
    """,

    (
        SEASON,
    )
).fetchall()


# ============================================================
# DATABASE STATUS
# ============================================================

print()

print(
    "=" * 110
)

print(
    "2026 PLAYER-STATS DATABASE STATUS"
)

print(
    "=" * 110
)


print(
    f"Completed games:          "
    f"{len(completed_game_ids):,}"
)


print(
    f"Games with player stats: "
    f"{games_with_any_player_stats:,}"
)


print(
    f"Games with passing:      "
    f"{games_with_passing:,}"
)


print(
    f"2026 player-stat rows:   "
    f"{total_rows_2026:,}"
)


print(
    f"2026 passing rows:       "
    f"{passing_rows_2026:,}"
)


print(
    f"Unique athletes:         "
    f"{unique_athletes_2026:,}"
)


print(
    f"Unique passers:          "
    f"{unique_passers_2026:,}"
)


# ============================================================
# PASSING TYPES
# ============================================================

print()

print(
    "2026 PASSING STAT TYPES"
)

print(
    "-" * 110
)


for (
    stat_type,
    count
) in passing_types:

    print(
        f"{stat_type:30} "
        f"{count:,}"
    )


# ============================================================
# CATEGORY COUNTS
# ============================================================

database_categories = connection.execute(
    """
    SELECT
        p.category,
        COUNT(*)

    FROM player_game_stat_values AS p

    INNER JOIN games AS g
        ON g.game_id = p.game_id

    WHERE g.season = ?

    GROUP BY p.category

    ORDER BY p.category
    """,

    (
        SEASON,
    )
).fetchall()


print()

print(
    "2026 PLAYER CATEGORIES"
)

print(
    "-" * 110
)


for (
    category,
    count
) in database_categories:

    print(
        f"{category:30} "
        f"{count:,}"
    )


# ============================================================
# WEEK COVERAGE
# ============================================================

print()

print(
    "WEEK COVERAGE"
)

print(
    "-" * 110
)


for (
    week,
    completed,
    covered
) in week_coverage:

    print(
        f"Week {week}: "
        f"{covered:,}/"
        f"{completed:,} "
        f"completed games"
    )


# ============================================================
# MISSING PLAYER-STAT GAMES
# ============================================================

missing_games = connection.execute(
    """
    SELECT
        g.game_id,
        g.week,
        g.away_team,
        g.home_team

    FROM games AS g

    LEFT JOIN (

        SELECT DISTINCT
            game_id

        FROM player_game_stat_values

    ) AS coverage

        ON coverage.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
        AND coverage.game_id IS NULL

    ORDER BY
        g.week,
        g.start_date
    """,

    (
        SEASON,
    )
).fetchall()


if len(
    missing_games
) > 0:

    print()

    print(
        "COMPLETED GAMES MISSING PLAYER STATS"
    )

    print(
        "-" * 110
    )


    for (
        game_id,
        week,
        away_team,
        home_team
    ) in missing_games:

        print(
            f"{game_id} | "
            f"W{week} | "
            f"{away_team} @ {home_team}"
        )


# ============================================================
# PASSING SAMPLE
# ============================================================

passing_sample = connection.execute(
    """
    SELECT
        g.week,
        p.team,
        p.athlete_name,
        p.stat_type,
        p.stat_value

    FROM player_game_stat_values AS p

    INNER JOIN games AS g
        ON g.game_id = p.game_id

    WHERE
        g.season = ?
        AND LOWER(p.category) = 'passing'

    ORDER BY
        g.start_date,
        p.team,
        p.athlete_name,
        p.stat_type

    LIMIT 30
    """,

    (
        SEASON,
    )
).fetchall()


print()

print(
    "2026 PASSING SAMPLE"
)

print(
    "-" * 110
)


for (
    week,
    team,
    athlete_name,
    stat_type,
    stat_value
) in passing_sample:

    print(
        f"W{week} | "
        f"{team:25} | "
        f"{athlete_name:25} | "
        f"{stat_type:8} | "
        f"{stat_value}"
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
    "LIVE 2026 PLAYER STATS UPDATE COMPLETE"
)

print(
    "=" * 110
)


print()

print(
    "The players table was intentionally left untouched."
)


print(
    "Athlete identity for this layer comes directly from "
    "player_game_stat_values athlete_id + athlete_name."
)


print()

print(
    "Next prerequisite:"
)


print(
    "  Build 2026 live pregame football features."
)