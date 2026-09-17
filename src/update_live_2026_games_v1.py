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
    "https://api.collegefootballdata.com/games"
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


def get_value(
    row,
    *possible_names
):

    for name in possible_names:

        if name in row:

            return row[
                name
            ]


    return None


def bool_to_int(
    value
):

    if value is None:

        return None


    return int(
        bool(
            value
        )
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
    "UPDATE LIVE 2026 GAMES — V1"
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

print()


# ============================================================
# API REQUEST
#
# classification=fbs is deliberate.
#
# This includes games involving FBS programs, including
# FBS-v-FCS games, which we can classify/filter later.
# ============================================================

response = requests.get(

    API_URL,

    params={

        "year":
            SEASON,

        "classification":
            "fbs",

        "seasonType":
            "both"
    },

    headers={

        "Authorization":
            f"Bearer {API_KEY}"
    },

    timeout=60
)


print(
    f"HTTP status: {response.status_code}"
)


if response.status_code != 200:

    raise RuntimeError(

        f"CFBD request failed.\n"
        f"HTTP {response.status_code}\n"
        f"{response.text[:2000]}"
    )


payload = response.json()


if not isinstance(
    payload,
    list
):

    raise RuntimeError(
        "Unexpected CFBD response format."
    )


print(
    f"Games returned: {len(payload):,}"
)


# ============================================================
# PRESERVE RAW RESPONSE
#
# We never throw away the original API response.
# ============================================================

RAW_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


raw_file = (

    RAW_DIRECTORY

    /

    f"games_{SEASON}_{batch_id}.json"
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


# ============================================================
# DATABASE CHECK
# ============================================================

if not DATABASE_FILE.exists():

    raise RuntimeError(
        f"Database does not exist: {DATABASE_FILE}"
    )


connection = sqlite3.connect(
    DATABASE_FILE
)


game_columns = connection.execute(
    """
    PRAGMA table_info(games)
    """
).fetchall()


existing_columns = {

    row[1]

    for row in game_columns
}


required_columns = {

    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "neutral_site",
    "conference_game",
    "venue",
    "venue_id",

    "home_team",
    "home_team_id",
    "home_conference",
    "home_classification",
    "home_points",
    "home_pregame_elo",
    "home_postgame_elo",

    "away_team",
    "away_team_id",
    "away_conference",
    "away_classification",
    "away_points",
    "away_pregame_elo",
    "away_postgame_elo"
}


missing_columns = (

    required_columns

    -
    existing_columns
)


if missing_columns:

    connection.close()

    raise RuntimeError(

        "games table is missing required columns:\n"

        +

        "\n".join(
            sorted(
                missing_columns
            )
        )
    )


# ============================================================
# NORMALIZE API ROWS
#
# We support both the normal CFBD camelCase names and a few
# alternate names so this updater is less fragile.
# ============================================================

database_rows = []


for game in payload:

    game_id = get_value(
        game,
        "id",
        "gameId",
        "game_id"
    )


    if game_id is None:

        print(
            "WARNING: Skipping game with no ID."
        )

        continue


    row = {

        "game_id":
            game_id,

        "season":
            get_value(
                game,
                "season"
            ),

        "week":
            get_value(
                game,
                "week"
            ),

        "season_type":
            get_value(
                game,
                "seasonType",
                "season_type"
            ),

        "start_date":
            get_value(
                game,
                "startDate",
                "startTime",
                "start_date"
            ),

        "neutral_site":
            bool_to_int(
                get_value(
                    game,
                    "neutralSite",
                    "neutral_site"
                )
            ),

        "conference_game":
            bool_to_int(
                get_value(
                    game,
                    "conferenceGame",
                    "conference_game"
                )
            ),

        "venue":
            get_value(
                game,
                "venue"
            ),

        "venue_id":
            get_value(
                game,
                "venueId",
                "venue_id"
            ),

        "home_team":
            get_value(
                game,
                "homeTeam",
                "home_team"
            ),

        "home_team_id":
            get_value(
                game,
                "homeId",
                "homeTeamId",
                "home_team_id"
            ),

        "home_conference":
            get_value(
                game,
                "homeConference",
                "home_conference"
            ),

        "home_classification":
            get_value(
                game,
                "homeClassification",
                "home_classification"
            ),

        "home_points":
            get_value(
                game,
                "homePoints",
                "home_points"
            ),

        "home_pregame_elo":
            get_value(
                game,
                "homePregameElo",
                "home_pregame_elo"
            ),

        "home_postgame_elo":
            get_value(
                game,
                "homePostgameElo",
                "home_postgame_elo"
            ),

        "away_team":
            get_value(
                game,
                "awayTeam",
                "away_team"
            ),

        "away_team_id":
            get_value(
                game,
                "awayId",
                "awayTeamId",
                "away_team_id"
            ),

        "away_conference":
            get_value(
                game,
                "awayConference",
                "away_conference"
            ),

        "away_classification":
            get_value(
                game,
                "awayClassification",
                "away_classification"
            ),

        "away_points":
            get_value(
                game,
                "awayPoints",
                "away_points"
            ),

        "away_pregame_elo":
            get_value(
                game,
                "awayPregameElo",
                "away_pregame_elo"
            ),

        "away_postgame_elo":
            get_value(
                game,
                "awayPostgameElo",
                "away_postgame_elo"
            )
    }


    database_rows.append(
        row
    )


print(
    f"Normalized games: {len(database_rows):,}"
)


# ============================================================
# UPSERT
#
# Existing 2018-2025 data is untouched.
#
# For an existing 2026 game:
#
# - kickoff changes can update
# - scores can update after the game
# - Elo values can update
# - NULL API values do NOT erase an already-known value
# ============================================================

columns = [

    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "neutral_site",
    "conference_game",
    "venue",
    "venue_id",

    "home_team",
    "home_team_id",
    "home_conference",
    "home_classification",
    "home_points",
    "home_pregame_elo",
    "home_postgame_elo",

    "away_team",
    "away_team_id",
    "away_conference",
    "away_classification",
    "away_points",
    "away_pregame_elo",
    "away_postgame_elo"
]


update_columns = [

    column

    for column in columns

    if column != "game_id"
]


update_clause = ",\n".join(

    [

        (
            f"{column} = "
            f"COALESCE("
            f"excluded.{column}, "
            f"games.{column}"
            f")"
        )

        for column in update_columns
    ]
)


placeholders = ",".join(

    [
        "?"
        for _ in columns
    ]
)


sql = f"""
INSERT INTO games (

    {",".join(columns)}

)

VALUES (

    {placeholders}

)

ON CONFLICT(game_id)

DO UPDATE SET

    {update_clause}
"""


insert_values = []


for row in database_rows:

    insert_values.append(

        tuple(

            row[
                column
            ]

            for column in columns
        )
    )


connection.executemany(
    sql,
    insert_values
)


connection.commit()


# ============================================================
# AUDIT RESULT
# ============================================================

total_2026 = connection.execute(
    """
    SELECT COUNT(*)

    FROM games

    WHERE season = 2026
    """
).fetchone()[0]


completed_2026 = connection.execute(
    """
    SELECT COUNT(*)

    FROM games

    WHERE
        season = 2026
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
    """
).fetchone()[0]


upcoming_2026 = connection.execute(
    """
    SELECT COUNT(*)

    FROM games

    WHERE
        season = 2026
        AND (
            home_points IS NULL
            OR away_points IS NULL
        )
    """
).fetchone()[0]


fbs_vs_fbs = connection.execute(
    """
    SELECT COUNT(*)

    FROM games

    WHERE
        season = 2026
        AND LOWER(home_classification) = 'fbs'
        AND LOWER(away_classification) = 'fbs'
    """
).fetchone()[0]


fbs_other = (

    total_2026

    -
    fbs_vs_fbs
)


print()

print(
    "=" * 110
)

print(
    "2026 DATABASE STATUS"
)

print(
    "=" * 110
)


print(
    f"2026 games stored: "
    f"{total_2026:,}"
)


print(
    f"Completed games:    "
    f"{completed_2026:,}"
)


print(
    f"Upcoming games:     "
    f"{upcoming_2026:,}"
)


print(
    f"FBS vs FBS:         "
    f"{fbs_vs_fbs:,}"
)


print(
    f"FBS vs other:       "
    f"{fbs_other:,}"
)


# ============================================================
# CURRENT / UPCOMING SAMPLE
# ============================================================

sample = connection.execute(
    """
    SELECT
        week,
        start_date,
        away_team,
        home_team,
        away_points,
        home_points

    FROM games

    WHERE season = 2026

    ORDER BY start_date

    LIMIT 15
    """
).fetchall()


print()

print(
    "EARLIEST 2026 GAMES"
)

print(
    "-" * 110
)


for row in sample:

    (
        week,
        start_date,
        away_team,
        home_team,
        away_points,
        home_points
    ) = row


    score = ""


    if (
        away_points is not None
        and
        home_points is not None
    ):

        score = (
            f" | "
            f"{away_points}-"
            f"{home_points}"
        )


    print(
        f"W{week} | "
        f"{start_date} | "
        f"{away_team} @ {home_team}"
        f"{score}"
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
    "LIVE 2026 GAME UPDATE COMPLETE"
)

print(
    "=" * 110
)


print()

print(
    "Next prerequisite:"
)

print(
    "  Update 2026 completed-game team statistics."
)