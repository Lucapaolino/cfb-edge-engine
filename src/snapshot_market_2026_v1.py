import argparse
import json
import os
import sqlite3

from datetime import (
    datetime,
    timezone
)

import pandas as pd
import requests

from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

API_BASE_URL = (
    "https://api.collegefootballdata.com"
)

DATABASE_FILE = (
    "data/cfb.db"
)

RAW_DIRECTORY = (
    "data/raw/market_snapshots"
)

DEFAULT_YEAR = 2026

SEASON_TYPE = "regular"

REQUEST_TIMEOUT = 60


# ============================================================
# COMMAND LINE
#
# Normally:
#
# python src/snapshot_market_2026_v1.py
#
# The script will detect the current week.
#
# Or force a week:
#
# python src/snapshot_market_2026_v1.py --week 2
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--year",
    type=int,
    default=DEFAULT_YEAR
)


parser.add_argument(
    "--week",
    type=int,
    default=None
)


args = parser.parse_args()


YEAR = args.year

REQUESTED_WEEK = args.week


# ============================================================
# LOAD API KEY
# ============================================================

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


# ============================================================
# TIME HELPERS
# ============================================================

def parse_api_datetime(
    value
):

    if value is None:

        return None


    value = str(
        value
    ).strip()


    if value == "":

        return None


    try:

        return datetime.fromisoformat(

            value.replace(
                "Z",
                "+00:00"
            )
        )


    except ValueError:

        return None


# ============================================================
# DETECT CURRENT WEEK
#
# CFBD's calendar endpoint gives us the date range associated
# with each football week.
#
# If --week was supplied, this step is skipped.
# ============================================================

def detect_current_week(
    year
):

    response = requests.get(

        API_BASE_URL
        +
        "/calendar",

        headers=headers,

        params={
            "year":
                year
        },

        timeout=REQUEST_TIMEOUT
    )


    if response.status_code != 200:

        raise RuntimeError(

            "Calendar request failed: "
            f"HTTP {response.status_code}\n"
            f"{response.text[:1000]}"
        )


    calendar = response.json()


    now = datetime.now(
        timezone.utc
    )


    regular_weeks = []


    for week_data in calendar:

        season_type = str(

            week_data.get(
                "seasonType",
                ""
            )

        ).lower()


        if season_type != SEASON_TYPE:

            continue


        week_number = week_data.get(
            "week"
        )


        start = parse_api_datetime(

            week_data.get(
                "startDate"
            )
        )


        end = parse_api_datetime(

            week_data.get(
                "endDate"
            )
        )


        if (
            week_number is None
            or
            start is None
            or
            end is None
        ):

            continue


        regular_weeks.append(
            {
                "week":
                    int(
                        week_number
                    ),

                "start":
                    start,

                "end":
                    end
            }
        )


    if len(
        regular_weeks
    ) == 0:

        raise RuntimeError(
            "No regular-season calendar weeks returned."
        )


    # --------------------------------------------------------
    # Best case:
    # current UTC time falls directly inside one week.
    # --------------------------------------------------------

    for week_data in regular_weeks:

        if (
            week_data[
                "start"
            ]
            <=
            now
            <=
            week_data[
                "end"
            ]
        ):

            return week_data[
                "week"
            ]


    # --------------------------------------------------------
    # If we're between calendar windows, choose the next
    # upcoming regular-season week.
    # --------------------------------------------------------

    upcoming = [

        week_data

        for week_data
        in regular_weeks

        if week_data[
            "start"
        ]
        >
        now
    ]


    if len(
        upcoming
    ) > 0:

        upcoming.sort(
            key=lambda value:
                value[
                    "start"
                ]
        )


        return upcoming[
            0
        ][
            "week"
        ]


    # --------------------------------------------------------
    # Otherwise use the latest week in the season.
    # --------------------------------------------------------

    regular_weeks.sort(
        key=lambda value:
            value[
                "week"
            ]
    )


    return regular_weeks[
        -1
    ][
        "week"
    ]


# ============================================================
# CHOOSE WEEK
# ============================================================

if REQUESTED_WEEK is not None:

    WEEK = int(
        REQUESTED_WEEK
    )


    week_source = (
        "command-line override"
    )


else:

    WEEK = detect_current_week(
        YEAR
    )


    week_source = (
        "CFBD calendar auto-detection"
    )


# ============================================================
# CAPTURE TIME
#
# ONE timestamp is used for every line in this API response.
# ============================================================

captured_datetime = datetime.now(
    timezone.utc
)


captured_at_utc = (

    captured_datetime

    .replace(
        microsecond=0
    )

    .isoformat()
)


file_timestamp = (

    captured_datetime

    .strftime(
        "%Y%m%dT%H%M%SZ"
    )
)


# ============================================================
# START
# ============================================================

print(
    "LIVE MARKET SNAPSHOT COLLECTOR — V1"
)

print(
    "=" * 110
)


print(
    f"Season:          {YEAR}"
)


print(
    f"Week:            {WEEK}"
)


print(
    f"Week source:     {week_source}"
)


print(
    f"Captured at UTC: {captured_at_utc}"
)


print()


# ============================================================
# DOWNLOAD CURRENT LINES
#
# This is the same CFBD betting-lines endpoint we've already
# used for the historical database.
# ============================================================

response = requests.get(

    API_BASE_URL
    +
    "/lines",

    headers=headers,

    params={
        "year":
            YEAR,

        "week":
            WEEK,

        "seasonType":
            SEASON_TYPE
    },

    timeout=REQUEST_TIMEOUT
)


if response.status_code != 200:

    raise RuntimeError(

        "Betting lines request failed: "
        f"HTTP {response.status_code}\n"
        f"{response.text[:1000]}"
    )


payload = response.json()


if not isinstance(
    payload,
    list
):

    raise RuntimeError(
        "Unexpected /lines response."
    )


print(
    f"Games returned by API: "
    f"{len(payload):,}"
)


# ============================================================
# SAVE COMPLETE RAW RESPONSE
#
# Never overwrite an earlier snapshot.
#
# This is extremely important. If our database parser ever
# turns out to be wrong, we can rebuild it from the original
# response exactly as it existed at capture time.
# ============================================================

os.makedirs(
    RAW_DIRECTORY,
    exist_ok=True
)


raw_file = os.path.join(

    RAW_DIRECTORY,

    (
        f"lines_{YEAR}_"
        f"week_{WEEK}_"
        f"{file_timestamp}.json"
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
    f"Raw snapshot: "
    f"{raw_file}"
)


# ============================================================
# FLATTEN PROVIDER LINES
# ============================================================

rows = []


for game in payload:

    game_id = game.get(
        "id"
    )


    season = game.get(
        "season",
        YEAR
    )


    week = game.get(
        "week",
        WEEK
    )


    start_date = game.get(
        "startDate"
    )


    home_team = game.get(
        "homeTeam"
    )


    away_team = game.get(
        "awayTeam"
    )


    game_start = parse_api_datetime(
        start_date
    )


    if game_start is None:

        is_pregame = None


    else:

        is_pregame = int(

            captured_datetime

            <
            game_start
        )


    provider_lines = game.get(
        "lines",
        []
    )


    if not isinstance(
        provider_lines,
        list
    ):

        continue


    for line in provider_lines:

        provider = line.get(
            "provider"
        )


        if provider is None:

            continue


        rows.append(
            {
                "captured_at_utc":
                    captured_at_utc,

                "game_id":
                    game_id,

                "season":
                    season,

                "week":
                    week,

                "season_type":
                    game.get(
                        "seasonType"
                    ),

                "start_date_utc":
                    start_date,

                "is_pregame":
                    is_pregame,

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "provider":
                    provider,

                "spread":
                    line.get(
                        "spread"
                    ),

                "spread_open":
                    line.get(
                        "spreadOpen"
                    ),

                "formatted_spread":
                    line.get(
                        "formattedSpread"
                    ),

                "over_under":
                    line.get(
                        "overUnder"
                    ),

                "over_under_open":
                    line.get(
                        "overUnderOpen"
                    ),

                "home_moneyline":
                    line.get(
                        "homeMoneyline"
                    ),

                "away_moneyline":
                    line.get(
                        "awayMoneyline"
                    ),

                "raw_provider_json":
                    json.dumps(
                        line,
                        separators=(
                            ",",
                            ":"
                        )
                    )
            }
        )


snapshot = pd.DataFrame(
    rows
)


if len(
    snapshot
) == 0:

    print()

    print(
        "No sportsbook lines were returned."
    )


    print(
        "Raw response was still saved."
    )


    raise SystemExit(
        0
    )


# ============================================================
# CLEAN BASIC NUMERIC TYPES
#
# We DO NOT delete suspicious prices here.
#
# This is a collection layer, not a cleaning layer.
# Raw market observations should be preserved exactly.
# ============================================================

numeric_columns = [

    "game_id",
    "season",
    "week",
    "spread",
    "spread_open",
    "over_under",
    "over_under_open",
    "home_moneyline",
    "away_moneyline"
]


for column in numeric_columns:

    if column in snapshot.columns:

        snapshot[
            column
        ] = pd.to_numeric(

            snapshot[
                column
            ],

            errors="coerce"
        )


# ============================================================
# SQLITE SNAPSHOT TABLE
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


connection.execute(

    """
    CREATE TABLE IF NOT EXISTS market_snapshots (

        captured_at_utc TEXT NOT NULL,

        game_id INTEGER NOT NULL,

        season INTEGER,

        week INTEGER,

        season_type TEXT,

        start_date_utc TEXT,

        is_pregame INTEGER,

        home_team TEXT,

        away_team TEXT,

        provider TEXT NOT NULL,

        spread REAL,

        spread_open REAL,

        formatted_spread TEXT,

        over_under REAL,

        over_under_open REAL,

        home_moneyline REAL,

        away_moneyline REAL,

        raw_provider_json TEXT,

        UNIQUE (
            captured_at_utc,
            game_id,
            provider
        )
    )
    """
)


connection.execute(

    """
    CREATE INDEX IF NOT EXISTS
    idx_market_snapshots_game_provider_time

    ON market_snapshots (
        game_id,
        provider,
        captured_at_utc
    )
    """
)


connection.execute(

    """
    CREATE INDEX IF NOT EXISTS
    idx_market_snapshots_pregame

    ON market_snapshots (
        season,
        week,
        is_pregame
    )
    """
)


# ============================================================
# INSERT
# ============================================================

insert_rows = []


for _, row in snapshot.iterrows():

    insert_rows.append(
        (
            row[
                "captured_at_utc"
            ],

            None
            if pd.isna(
                row[
                    "game_id"
                ]
            )
            else
            int(
                row[
                    "game_id"
                ]
            ),

            None
            if pd.isna(
                row[
                    "season"
                ]
            )
            else
            int(
                row[
                    "season"
                ]
            ),

            None
            if pd.isna(
                row[
                    "week"
                ]
            )
            else
            int(
                row[
                    "week"
                ]
            ),

            row[
                "season_type"
            ],

            row[
                "start_date_utc"
            ],

            row[
                "is_pregame"
            ],

            row[
                "home_team"
            ],

            row[
                "away_team"
            ],

            row[
                "provider"
            ],

            None
            if pd.isna(
                row[
                    "spread"
                ]
            )
            else
            float(
                row[
                    "spread"
                ]
            ),

            None
            if pd.isna(
                row[
                    "spread_open"
                ]
            )
            else
            float(
                row[
                    "spread_open"
                ]
            ),

            row[
                "formatted_spread"
            ],

            None
            if pd.isna(
                row[
                    "over_under"
                ]
            )
            else
            float(
                row[
                    "over_under"
                ]
            ),

            None
            if pd.isna(
                row[
                    "over_under_open"
                ]
            )
            else
            float(
                row[
                    "over_under_open"
                ]
            ),

            None
            if pd.isna(
                row[
                    "home_moneyline"
                ]
            )
            else
            float(
                row[
                    "home_moneyline"
                ]
            ),

            None
            if pd.isna(
                row[
                    "away_moneyline"
                ]
            )
            else
            float(
                row[
                    "away_moneyline"
                ]
            ),

            row[
                "raw_provider_json"
            ]
        )
    )


connection.executemany(

    """
    INSERT OR IGNORE INTO market_snapshots (

        captured_at_utc,
        game_id,
        season,
        week,
        season_type,
        start_date_utc,
        is_pregame,
        home_team,
        away_team,
        provider,
        spread,
        spread_open,
        formatted_spread,
        over_under,
        over_under_open,
        home_moneyline,
        away_moneyline,
        raw_provider_json

    )

    VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?, ?, ?
    )
    """,

    insert_rows
)


connection.commit()


# ============================================================
# TOTAL DATABASE COUNTS
# ============================================================

total_snapshots = connection.execute(

    """
    SELECT COUNT(*)
    FROM market_snapshots
    """

).fetchone()[0]


unique_capture_times = connection.execute(

    """
    SELECT COUNT(
        DISTINCT captured_at_utc
    )
    FROM market_snapshots
    """

).fetchone()[0]


connection.close()


# ============================================================
# SNAPSHOT SUMMARY
# ============================================================

print()

print(
    "=" * 110
)

print(
    "SNAPSHOT SUMMARY"
)

print(
    "=" * 110
)


print(
    f"Provider rows captured: "
    f"{len(snapshot):,}"
)


print(
    f"Unique games:           "
    f"{snapshot['game_id'].nunique():,}"
)


print(
    f"Providers:              "
    f"{snapshot['provider'].nunique():,}"
)


pregame_count = (

    snapshot[
        "is_pregame"
    ]
    ==
    1

).sum()


print(
    f"Pregame provider rows:  "
    f"{pregame_count:,}"
)


print()

print(
    "ROWS BY PROVIDER"
)

print(
    "-" * 70
)


provider_summary = (

    snapshot

    .groupby(
        "provider"
    )

    .agg(

        games=(
            "game_id",
            "nunique"
        ),

        spreads=(
            "spread",
            lambda values:
                values.notna().sum()
        ),

        moneylines=(
            "home_moneyline",
            lambda values:
                values.notna().sum()
        )
    )

    .sort_values(
        "games",
        ascending=False
    )
)


print(
    provider_summary.to_string()
)


# ============================================================
# SAMPLE
# ============================================================

print()

print(
    "SAMPLE CURRENT LINES"
)

print(
    "-" * 110
)


sample_columns = [

    "start_date_utc",
    "away_team",
    "home_team",
    "provider",
    "spread",
    "home_moneyline",
    "away_moneyline",
    "is_pregame"
]


print(

    snapshot[
        sample_columns
    ]

    .sort_values(
        [
            "start_date_utc",
            "away_team",
            "provider"
        ]
    )

    .head(
        30
    )

    .to_string(
        index=False
    )
)


print()

print(
    "=" * 110
)

print(
    "DATABASE STATUS"
)

print(
    "=" * 110
)


print(
    f"Total stored provider snapshots: "
    f"{total_snapshots:,}"
)


print(
    f"Distinct capture times:          "
    f"{unique_capture_times:,}"
)


print()

print(
    "Snapshot successfully saved."
)


print(
    "No game outcomes were used."
)