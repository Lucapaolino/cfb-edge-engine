import argparse
import json
import os
import sqlite3
import time

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
    "data/raw/market_snapshots_v2"
)

DEFAULT_YEAR = 2026

DEFAULT_WEEKS_AHEAD = 2

SEASON_TYPE = "regular"

REQUEST_TIMEOUT = 60

PAUSE_BETWEEN_REQUESTS = 0.20


# ============================================================
# PROVIDER NORMALIZATION
#
# We preserve provider_raw exactly as CFBD returns it.
#
# provider is the normalized name used later for analysis.
# ============================================================

PROVIDER_ALIASES = {

    "Draft Kings":
        "DraftKings",

    "DraftKings":
        "DraftKings",

    "Bovada":
        "Bovada",

    "ESPN Bet":
        "ESPN Bet"
}


def normalize_provider(
    provider
):

    if provider is None:
        return None

    provider = str(
        provider
    ).strip()

    return PROVIDER_ALIASES.get(
        provider,
        provider
    )


# ============================================================
# COMMAND LINE
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


parser.add_argument(
    "--weeks-ahead",
    type=int,
    default=DEFAULT_WEEKS_AHEAD
)


args = parser.parse_args()


YEAR = args.year

REQUESTED_WEEK = args.week

WEEKS_AHEAD = args.weeks_ahead


if WEEKS_AHEAD < 0:

    raise RuntimeError(
        "--weeks-ahead cannot be negative."
    )


# ============================================================
# API KEY
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
# DATETIME PARSER
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

        parsed = datetime.fromisoformat(

            value.replace(
                "Z",
                "+00:00"
            )
        )


        if parsed.tzinfo is None:

            parsed = parsed.replace(
                tzinfo=timezone.utc
            )


        return parsed.astimezone(
            timezone.utc
        )


    except ValueError:

        return None


# ============================================================
# LOAD CALENDAR
# ============================================================

def get_regular_season_weeks(
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


    payload = response.json()


    weeks = []


    for row in payload:

        season_type = str(

            row.get(
                "seasonType",
                ""
            )

        ).lower()


        if season_type != SEASON_TYPE:

            continue


        week = row.get(
            "week"
        )


        start = parse_api_datetime(

            row.get(
                "startDate"
            )
        )


        end = parse_api_datetime(

            row.get(
                "endDate"
            )
        )


        if (
            week is None
            or
            start is None
            or
            end is None
        ):

            continue


        weeks.append(
            {
                "week":
                    int(
                        week
                    ),

                "start":
                    start,

                "end":
                    end
            }
        )


    weeks.sort(
        key=lambda row:
            row[
                "week"
            ]
    )


    return weeks


# ============================================================
# DETECT CURRENT WEEK
# ============================================================

def detect_current_week(
    calendar_weeks,
    now
):

    # --------------------------------------------------------
    # Current calendar window
    # --------------------------------------------------------

    for row in calendar_weeks:

        if (
            row[
                "start"
            ]
            <=
            now
            <=
            row[
                "end"
            ]
        ):

            return row[
                "week"
            ]


    # --------------------------------------------------------
    # Between weeks:
    # use next upcoming week
    # --------------------------------------------------------

    upcoming = [

        row

        for row
        in calendar_weeks

        if row[
            "start"
        ]
        >
        now
    ]


    if len(
        upcoming
    ) > 0:

        return upcoming[
            0
        ][
            "week"
        ]


    # --------------------------------------------------------
    # Season over:
    # use final regular-season week
    # --------------------------------------------------------

    return calendar_weeks[
        -1
    ][
        "week"
    ]


# ============================================================
# CAPTURE TIME
#
# One timestamp identifies the entire multi-week snapshot.
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


batch_id = (

    captured_datetime

    .strftime(
        "%Y%m%dT%H%M%SZ"
    )
)


# ============================================================
# DETERMINE WEEKS
# ============================================================

calendar_weeks = get_regular_season_weeks(
    YEAR
)


available_week_numbers = [

    row[
        "week"
    ]

    for row
    in calendar_weeks
]


if REQUESTED_WEEK is not None:

    starting_week = int(
        REQUESTED_WEEK
    )


    week_source = (
        "command-line override"
    )


else:

    starting_week = detect_current_week(

        calendar_weeks,

        captured_datetime
    )


    week_source = (
        "CFBD calendar auto-detection"
    )


requested_weeks = []


for week in range(

    starting_week,

    starting_week
    +
    WEEKS_AHEAD
    +
    1
):

    if week in available_week_numbers:

        requested_weeks.append(
            week
        )


if len(
    requested_weeks
) == 0:

    raise RuntimeError(
        "No valid regular-season weeks selected."
    )


# ============================================================
# START
# ============================================================

print(
    "LIVE MARKET SNAPSHOT COLLECTOR — V2"
)

print(
    "=" * 110
)


print(
    f"Season:          {YEAR}"
)


print(
    f"Starting week:   {starting_week}"
)


print(
    f"Weeks captured:  {requested_weeks}"
)


print(
    f"Week source:     {week_source}"
)


print(
    f"Captured at UTC: {captured_at_utc}"
)


print(
    f"Batch ID:        {batch_id}"
)


# ============================================================
# DOWNLOAD EACH WEEK
# ============================================================

os.makedirs(
    RAW_DIRECTORY,
    exist_ok=True
)


all_games = []


print()

print(
    "=" * 110
)

print(
    "DOWNLOADING MARKET DATA"
)

print(
    "=" * 110
)


for week in requested_weeks:

    response = requests.get(

        API_BASE_URL
        +
        "/lines",

        headers=headers,

        params={
            "year":
                YEAR,

            "week":
                week,

            "seasonType":
                SEASON_TYPE
        },

        timeout=REQUEST_TIMEOUT
    )


    if response.status_code != 200:

        raise RuntimeError(

            f"Week {week} lines request failed: "
            f"HTTP {response.status_code}\n"
            f"{response.text[:1000]}"
        )


    payload = response.json()


    if not isinstance(
        payload,
        list
    ):

        raise RuntimeError(

            f"Unexpected /lines response "
            f"for Week {week}."
        )


    raw_file = os.path.join(

        RAW_DIRECTORY,

        (
            f"lines_{YEAR}_"
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

        f"Week {week:>2} | "
        f"{len(payload):>3} games | "
        f"{raw_file}"
    )


    for game in payload:

        game[
            "_snapshot_requested_week"
        ] = week


        all_games.append(
            game
        )


    time.sleep(
        PAUSE_BETWEEN_REQUESTS
    )


# ============================================================
# FLATTEN PROVIDER LINES
# ============================================================

rows = []


for game in all_games:

    game_id = game.get(
        "id"
    )


    week = game.get(
        "week"
    )


    if week is None:

        week = game.get(
            "_snapshot_requested_week"
        )


    start_date = game.get(
        "startDate"
    )


    game_start = parse_api_datetime(
        start_date
    )


    if game_start is None:

        is_pregame = None

        hours_until_kickoff = None


    else:

        is_pregame = int(

            captured_datetime
            <
            game_start
        )


        hours_until_kickoff = (

            game_start

            -
            captured_datetime

        ).total_seconds() / 3600.0


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

        provider_raw = line.get(
            "provider"
        )


        provider = normalize_provider(
            provider_raw
        )


        if provider is None:

            continue


        rows.append(
            {
                "batch_id":
                    batch_id,

                "captured_at_utc":
                    captured_at_utc,

                "game_id":
                    game_id,

                "season":
                    game.get(
                        "season",
                        YEAR
                    ),

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

                "hours_until_kickoff":
                    hours_until_kickoff,

                "home_team":
                    game.get(
                        "homeTeam"
                    ),

                "away_team":
                    game.get(
                        "awayTeam"
                    ),

                "provider":
                    provider,

                "provider_raw":
                    provider_raw,

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
        "No provider rows returned."
    )


    raise SystemExit(
        0
    )


# ============================================================
# NUMERIC CLEANING
#
# Collection only.
# Nothing suspicious is deleted.
# ============================================================

numeric_columns = [

    "game_id",
    "season",
    "week",
    "hours_until_kickoff",
    "spread",
    "spread_open",
    "over_under",
    "over_under_open",
    "home_moneyline",
    "away_moneyline"
]


for column in numeric_columns:

    snapshot[
        column
    ] = pd.to_numeric(

        snapshot[
            column
        ],

        errors="coerce"
    )


# ============================================================
# SQLITE TABLE
#
# We use a new V2 table so the first V1 snapshot remains
# untouched.
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


connection.execute(

    """
    CREATE TABLE IF NOT EXISTS market_snapshots_v2 (

        batch_id TEXT NOT NULL,

        captured_at_utc TEXT NOT NULL,

        game_id INTEGER NOT NULL,

        season INTEGER,

        week INTEGER,

        season_type TEXT,

        start_date_utc TEXT,

        is_pregame INTEGER,

        hours_until_kickoff REAL,

        home_team TEXT,

        away_team TEXT,

        provider TEXT NOT NULL,

        provider_raw TEXT NOT NULL,

        spread REAL,

        spread_open REAL,

        formatted_spread TEXT,

        over_under REAL,

        over_under_open REAL,

        home_moneyline REAL,

        away_moneyline REAL,

        raw_provider_json TEXT,

        UNIQUE (
            batch_id,
            game_id,
            provider_raw
        )
    )
    """
)


connection.execute(

    """
    CREATE INDEX IF NOT EXISTS
    idx_market_snapshots_v2_game_time

    ON market_snapshots_v2 (
        game_id,
        captured_at_utc
    )
    """
)


connection.execute(

    """
    CREATE INDEX IF NOT EXISTS
    idx_market_snapshots_v2_provider

    ON market_snapshots_v2 (
        provider,
        captured_at_utc
    )
    """
)


connection.execute(

    """
    CREATE INDEX IF NOT EXISTS
    idx_market_snapshots_v2_pregame

    ON market_snapshots_v2 (
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
                "batch_id"
            ],

            row[
                "captured_at_utc"
            ],

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

            None
            if pd.isna(
                row[
                    "hours_until_kickoff"
                ]
            )
            else
            float(
                row[
                    "hours_until_kickoff"
                ]
            ),

            row[
                "home_team"
            ],

            row[
                "away_team"
            ],

            row[
                "provider"
            ],

            row[
                "provider_raw"
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
    INSERT OR IGNORE INTO market_snapshots_v2 (

        batch_id,
        captured_at_utc,
        game_id,
        season,
        week,
        season_type,
        start_date_utc,
        is_pregame,
        hours_until_kickoff,
        home_team,
        away_team,
        provider,
        provider_raw,
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
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    )
    """,

    insert_rows
)


connection.commit()


# ============================================================
# DATABASE COUNTS
# ============================================================

total_rows = connection.execute(

    """
    SELECT COUNT(*)
    FROM market_snapshots_v2
    """

).fetchone()[0]


total_batches = connection.execute(

    """
    SELECT COUNT(
        DISTINCT batch_id
    )
    FROM market_snapshots_v2
    """

).fetchone()[0]


connection.close()


# ============================================================
# SUMMARY
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
    f"Provider rows:          "
    f"{len(snapshot):,}"
)


print(
    f"Unique games:           "
    f"{snapshot['game_id'].nunique():,}"
)


print(
    f"Raw provider names:     "
    f"{snapshot['provider_raw'].nunique():,}"
)


print(
    f"Normalized providers:   "
    f"{snapshot['provider'].nunique():,}"
)


print(
    f"Pregame provider rows:  "
    f"{(snapshot['is_pregame'] == 1).sum():,}"
)


print()


# ============================================================
# PREGAME COVERAGE BY WEEK
# ============================================================

print(
    "PREGAME COVERAGE BY WEEK"
)

print(
    "-" * 90
)


week_summary = (

    snapshot

    .groupby(
        "week"
    )

    .agg(

        games=(
            "game_id",
            "nunique"
        ),

        provider_rows=(
            "game_id",
            "size"
        ),

        pregame_rows=(
            "is_pregame",
            lambda values:
                (
                    values
                    ==
                    1
                ).sum()
        )
    )

    .reset_index()
)


print(
    week_summary.to_string(
        index=False
    )
)


# ============================================================
# RAW PROVIDER NAMES
# ============================================================

print()

print(
    "RAW PROVIDER NAMES"
)

print(
    "-" * 90
)


raw_provider_summary = (

    snapshot

    .groupby(
        [
            "provider_raw",
            "provider"
        ]
    )

    .agg(

        rows=(
            "game_id",
            "size"
        ),

        games=(
            "game_id",
            "nunique"
        ),

        moneylines=(
            "home_moneyline",
            lambda values:
                values.notna().sum()
        )
    )

    .reset_index()
)


print(
    raw_provider_summary.to_string(
        index=False
    )
)


# ============================================================
# NORMALIZED PROVIDER SUMMARY
# ============================================================

print()

print(
    "NORMALIZED PROVIDER SUMMARY"
)

print(
    "-" * 90
)


normalized_summary = (

    snapshot[
        snapshot[
            "is_pregame"
        ]
        ==
        1
    ]

    .groupby(
        "provider"
    )

    .agg(

        rows=(
            "game_id",
            "size"
        ),

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

    .reset_index()
)


print(
    normalized_summary.to_string(
        index=False
    )
)


# ============================================================
# UPCOMING GAME SAMPLE
# ============================================================

print()

print(
    "UPCOMING GAME SAMPLE"
)

print(
    "-" * 110
)


upcoming = (

    snapshot[
        snapshot[
            "is_pregame"
        ]
        ==
        1
    ]

    .sort_values(
        [
            "start_date_utc",
            "away_team",
            "provider_raw"
        ]
    )
)


print(

    upcoming[
        [
            "week",
            "start_date_utc",
            "away_team",
            "home_team",
            "provider_raw",
            "provider",
            "spread",
            "home_moneyline",
            "away_moneyline",
            "hours_until_kickoff"
        ]
    ]

    .head(
        40
    )

    .to_string(
        index=False
    )
)


# ============================================================
# DATABASE STATUS
# ============================================================

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
    f"Total V2 snapshot rows: "
    f"{total_rows:,}"
)


print(
    f"Distinct V2 batches:    "
    f"{total_batches:,}"
)


print()

print(
    "Snapshot batch successfully saved."
)


print(
    "No outcomes were used."
)