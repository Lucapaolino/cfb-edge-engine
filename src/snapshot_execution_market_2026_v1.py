import json
import os
import sqlite3

from datetime import datetime, timezone

import pandas as pd
import requests

from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

API_URL = (
    "https://api.the-odds-api.com/"
    "v4/sports/americanfootball_ncaaf/odds"
)

DATABASE_FILE = "data/cfb.db"

RAW_DIRECTORY = (
    "data/raw/execution_market_snapshots"
)


# ============================================================
# MARKET BOOKS
#
# Hard Rock Florida:
#     our eventual EXECUTION sportsbook
#
# Everyone else:
#     reference books used to estimate market consensus
#
# The Odds API allows up to 10 specified bookmakers to count
# as one bookmaker-region group.
# ============================================================

BOOKMAKERS = [

    "hardrockbet_fl",

    "draftkings",

    "fanduel",

    "betmgm",

    "betrivers",

    "bovada",

    "betonlineag",

    "betus",

    "pinnacle",

    "espnbet"
]


MARKETS = [
    "h2h",
    "spreads",
    "totals"
]


REQUEST_TIMEOUT = 60


# ============================================================
# API KEY
# ============================================================

load_dotenv()


api_key = os.getenv(
    "THE_ODDS_API_KEY"
)


if not api_key:

    raise RuntimeError(
        "THE_ODDS_API_KEY was not found in .env"
    )


# ============================================================
# CAPTURE TIME
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
# HELPERS
# ============================================================

def parse_datetime(value):

    if value is None:

        return None


    try:

        parsed = datetime.fromisoformat(

            str(value).replace(
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


def find_outcome(
    outcomes,
    name
):

    for outcome in outcomes:

        if outcome.get(
            "name"
        ) == name:

            return outcome


    return None


# ============================================================
# START
# ============================================================

print(
    "EXECUTION + CONSENSUS MARKET SNAPSHOT — V2"
)

print(
    "=" * 110
)


print(
    f"Captured at UTC: {captured_at_utc}"
)


print(
    f"Books requested: {len(BOOKMAKERS)}"
)


print(
    f"Markets:         {', '.join(MARKETS)}"
)


print()

print(
    "EXECUTION BOOK:"
)


print(
    "  hardrockbet_fl"
)


print()

print(
    "REFERENCE BOOKS:"
)


for bookmaker in BOOKMAKERS:

    if bookmaker != "hardrockbet_fl":

        print(
            f"  {bookmaker}"
        )


# ============================================================
# REQUEST
# ============================================================

response = requests.get(

    API_URL,

    params={

        "apiKey":
            api_key,

        "bookmakers":
            ",".join(
                BOOKMAKERS
            ),

        "markets":
            ",".join(
                MARKETS
            ),

        "oddsFormat":
            "american",

        "dateFormat":
            "iso"
    },

    timeout=REQUEST_TIMEOUT
)


if response.status_code != 200:

    raise RuntimeError(

        f"Odds API request failed: "
        f"HTTP {response.status_code}\n"
        f"{response.text[:1500]}"
    )


payload = response.json()


if not isinstance(
    payload,
    list
):

    raise RuntimeError(
        "Unexpected Odds API response."
    )


# ============================================================
# QUOTA
# ============================================================

credits_remaining = response.headers.get(
    "x-requests-remaining"
)


credits_used = response.headers.get(
    "x-requests-used"
)


credits_last = response.headers.get(
    "x-requests-last"
)


print()

print(
    f"Events returned:   {len(payload):,}"
)


print(
    f"Credits this call: {credits_last}"
)


print(
    f"Credits used:      {credits_used}"
)


print(
    f"Credits remaining: {credits_remaining}"
)


# ============================================================
# RAW RESPONSE
#
# NEVER overwrite old snapshots.
# ============================================================

os.makedirs(
    RAW_DIRECTORY,
    exist_ok=True
)


raw_file = os.path.join(

    RAW_DIRECTORY,

    (
        f"ncaaf_execution_market_"
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


print()

print(
    f"Raw snapshot: {raw_file}"
)


# ============================================================
# FLATTEN
# ============================================================

rows = []


for event in payload:

    event_id = event.get(
        "id"
    )


    commence_time = event.get(
        "commence_time"
    )


    home_team = event.get(
        "home_team"
    )


    away_team = event.get(
        "away_team"
    )


    kickoff = parse_datetime(
        commence_time
    )


    if kickoff is None:

        is_pregame = None

        hours_until_kickoff = None


    else:

        is_pregame = int(

            captured_datetime
            <
            kickoff
        )


        hours_until_kickoff = (

            kickoff
            -
            captured_datetime

        ).total_seconds() / 3600.0


    for bookmaker in event.get(
        "bookmakers",
        []
    ):

        bookmaker_key = bookmaker.get(
            "key"
        )


        bookmaker_title = bookmaker.get(
            "title"
        )


        bookmaker_last_update = bookmaker.get(
            "last_update"
        )


        # ----------------------------------------------------
        # Defaults
        # ----------------------------------------------------

        home_moneyline = None

        away_moneyline = None


        home_spread = None

        away_spread = None

        home_spread_price = None

        away_spread_price = None


        total = None

        over_price = None

        under_price = None


        h2h_last_update = None

        spread_last_update = None

        total_last_update = None


        # ----------------------------------------------------
        # MARKETS
        # ----------------------------------------------------

        for market in bookmaker.get(
            "markets",
            []
        ):

            market_key = market.get(
                "key"
            )


            market_last_update = market.get(
                "last_update"
            )


            outcomes = market.get(
                "outcomes",
                []
            )


            # =================================================
            # MONEYLINE
            # =================================================

            if market_key == "h2h":

                h2h_last_update = (
                    market_last_update
                )


                home = find_outcome(

                    outcomes,

                    home_team
                )


                away = find_outcome(

                    outcomes,

                    away_team
                )


                if home is not None:

                    home_moneyline = (
                        home.get(
                            "price"
                        )
                    )


                if away is not None:

                    away_moneyline = (
                        away.get(
                            "price"
                        )
                    )


            # =================================================
            # SPREAD
            # =================================================

            elif market_key == "spreads":

                spread_last_update = (
                    market_last_update
                )


                home = find_outcome(

                    outcomes,

                    home_team
                )


                away = find_outcome(

                    outcomes,

                    away_team
                )


                if home is not None:

                    home_spread = (
                        home.get(
                            "point"
                        )
                    )


                    home_spread_price = (
                        home.get(
                            "price"
                        )
                    )


                if away is not None:

                    away_spread = (
                        away.get(
                            "point"
                        )
                    )


                    away_spread_price = (
                        away.get(
                            "price"
                        )
                    )


            # =================================================
            # TOTAL
            # =================================================

            elif market_key == "totals":

                total_last_update = (
                    market_last_update
                )


                over = find_outcome(

                    outcomes,

                    "Over"
                )


                under = find_outcome(

                    outcomes,

                    "Under"
                )


                if over is not None:

                    total = over.get(
                        "point"
                    )


                    over_price = over.get(
                        "price"
                    )


                if under is not None:

                    if total is None:

                        total = under.get(
                            "point"
                        )


                    under_price = under.get(
                        "price"
                    )


        jurisdiction = None


        if bookmaker_key == "hardrockbet_fl":

            jurisdiction = "FL"


        is_execution_book = int(

            bookmaker_key
            ==
            "hardrockbet_fl"
        )


        rows.append(
            {
                "batch_id":
                    batch_id,

                "captured_at_utc":
                    captured_at_utc,

                "odds_api_event_id":
                    event_id,

                "sport":
                    "americanfootball_ncaaf",

                "commence_time_utc":
                    commence_time,

                "is_pregame":
                    is_pregame,

                "hours_until_kickoff":
                    hours_until_kickoff,

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "bookmaker_key":
                    bookmaker_key,

                "bookmaker_title":
                    bookmaker_title,

                "jurisdiction":
                    jurisdiction,

                "is_execution_book":
                    is_execution_book,

                "bookmaker_last_update_utc":
                    bookmaker_last_update,

                "h2h_last_update_utc":
                    h2h_last_update,

                "spread_last_update_utc":
                    spread_last_update,

                "total_last_update_utc":
                    total_last_update,

                "home_moneyline":
                    home_moneyline,

                "away_moneyline":
                    away_moneyline,

                "home_spread":
                    home_spread,

                "home_spread_price":
                    home_spread_price,

                "away_spread":
                    away_spread,

                "away_spread_price":
                    away_spread_price,

                "total":
                    total,

                "over_price":
                    over_price,

                "under_price":
                    under_price,

                "raw_bookmaker_json":
                    json.dumps(
                        bookmaker,
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
        "No bookmaker rows returned."
    )


    raise SystemExit(
        0
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [

    "hours_until_kickoff",

    "home_moneyline",
    "away_moneyline",

    "home_spread",
    "home_spread_price",

    "away_spread",
    "away_spread_price",

    "total",
    "over_price",
    "under_price"
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
# DATABASE
#
# SAME TABLE AS V1.
#
# Your September 6 snapshot stays intact.
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


connection.execute(

    """
    CREATE TABLE IF NOT EXISTS
    execution_market_snapshots (

        batch_id TEXT NOT NULL,

        captured_at_utc TEXT NOT NULL,

        odds_api_event_id TEXT NOT NULL,

        sport TEXT,

        commence_time_utc TEXT,

        is_pregame INTEGER,

        hours_until_kickoff REAL,

        home_team TEXT,

        away_team TEXT,

        bookmaker_key TEXT NOT NULL,

        bookmaker_title TEXT,

        jurisdiction TEXT,

        is_execution_book INTEGER,

        bookmaker_last_update_utc TEXT,

        h2h_last_update_utc TEXT,

        spread_last_update_utc TEXT,

        total_last_update_utc TEXT,

        home_moneyline REAL,

        away_moneyline REAL,

        home_spread REAL,

        home_spread_price REAL,

        away_spread REAL,

        away_spread_price REAL,

        total REAL,

        over_price REAL,

        under_price REAL,

        raw_bookmaker_json TEXT,

        UNIQUE (
            batch_id,
            odds_api_event_id,
            bookmaker_key
        )
    )
    """
)


columns = [

    "batch_id",
    "captured_at_utc",
    "odds_api_event_id",
    "sport",
    "commence_time_utc",
    "is_pregame",
    "hours_until_kickoff",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "jurisdiction",
    "is_execution_book",
    "bookmaker_last_update_utc",
    "h2h_last_update_utc",
    "spread_last_update_utc",
    "total_last_update_utc",
    "home_moneyline",
    "away_moneyline",
    "home_spread",
    "home_spread_price",
    "away_spread",
    "away_spread_price",
    "total",
    "over_price",
    "under_price",
    "raw_bookmaker_json"
]


insert_rows = []


for _, row in snapshot.iterrows():

    values = []


    for column in columns:

        value = row[
            column
        ]


        if pd.isna(
            value
        ):

            value = None


        values.append(
            value
        )


    insert_rows.append(
        tuple(
            values
        )
    )


placeholders = ",".join(

    [
        "?"
        for _ in columns
    ]
)


connection.executemany(

    f"""
    INSERT OR IGNORE INTO
    execution_market_snapshots (

        {",".join(columns)}

    )

    VALUES (
        {placeholders}
    )
    """,

    insert_rows
)


connection.commit()


total_rows = connection.execute(

    """
    SELECT COUNT(*)
    FROM execution_market_snapshots
    """

).fetchone()[0]


total_batches = connection.execute(

    """
    SELECT COUNT(
        DISTINCT batch_id
    )
    FROM execution_market_snapshots
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
    "BOOKMAKER COVERAGE"
)

print(
    "=" * 110
)


book_summary = (

    snapshot

    .groupby(
        [
            "bookmaker_key",
            "bookmaker_title"
        ]
    )

    .agg(

        games=(
            "odds_api_event_id",
            "nunique"
        ),

        moneylines=(
            "home_moneyline",
            lambda x:
                x.notna().sum()
        ),

        spreads=(
            "home_spread",
            lambda x:
                x.notna().sum()
        ),

        totals=(
            "total",
            lambda x:
                x.notna().sum()
        )
    )

    .reset_index()

    .sort_values(
        "games",
        ascending=False
    )
)


print(
    book_summary.to_string(
        index=False
    )
)


returned_books = set(
    snapshot[
        "bookmaker_key"
    ]
)


missing_books = [

    book

    for book in BOOKMAKERS

    if book not in returned_books
]


print()

print(
    "REQUESTED BUT NOT RETURNED"
)

print(
    "-" * 70
)


if len(
    missing_books
) == 0:

    print(
        "None"
    )


else:

    for book in missing_books:

        print(
            book
        )


# ============================================================
# HARD ROCK
# ============================================================

hard_rock = snapshot[

    snapshot[
        "bookmaker_key"
    ]
    ==
    "hardrockbet_fl"

]


print()

print(
    "=" * 110
)

print(
    "HARD ROCK BET FLORIDA"
)

print(
    "=" * 110
)


print(

    f"Games available: "
    f"{hard_rock['odds_api_event_id'].nunique():,}"
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
    f"Rows captured this batch: "
    f"{len(snapshot):,}"
)


print(
    f"Total stored rows:        "
    f"{total_rows:,}"
)


print(
    f"Distinct batches:         "
    f"{total_batches:,}"
)


print()

print(
    "No outcomes were used."
)