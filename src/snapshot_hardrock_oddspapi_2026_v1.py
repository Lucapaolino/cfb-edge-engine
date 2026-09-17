import json
import os
import sqlite3

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from dotenv import load_dotenv


# ============================================================
# PURPOSE
#
# Production Hard Rock Bet NCAAF market collector.
#
# SOURCE:
#   OddsPapi
#
# BOOK:
#   Hard Rock Bet
#
# NORMAL API COST:
#   1 OddsPapi request
#
#
# IMPORTANT:
#
# Hard Rock often exposes an entire ladder of alternate
# spreads. OddsPapi's "mainLine" flag does NOT reliably
# identify the sportsbook's actual central/headline spread.
#
# Our resolver:
#
#   1. Finds all active full-game spread rungs.
#   2. Removes the vig from both sides.
#   3. Finds the rung nearest a true 50/50 market.
#   4. Requires:
#
#          spread_balance_score <= 0.05
#
# A balance score of 0.05 means the de-vigged market is
# approximately 45% / 55% or better.
#
#
# If the best available rung FAILS this test:
#
#   - the raw line is still stored
#   - it is labeled NO_BALANCED_HEADLINE
#   - it must NOT be used as our Hard Rock headline spread
#   - it must NOT generate MODEL EDGE
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://api.oddspapi.io/v4"

SPORT_ID = 14

NCAAF_TOURNAMENT_ID = 27653

BOOKMAKER = "hardrockbet"

SOURCE_NAME = "oddspapi"

HEADLINE_RESOLVER_VERSION = "balanced_devig_v2"

MAX_BALANCE_SCORE = 0.05


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)


DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cfb.db"
)


ODDSPAPI_RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "oddspapi"
)


RAW_DIRECTORY = (
    ODDSPAPI_RAW_DIRECTORY
    / "hardrock_execution"
)


PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


LATEST_CSV = (
    PROCESSED_DIRECTORY
    / "hardrock_execution_latest.csv"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    ENV_FILE
)


API_KEY = os.getenv(
    "ODDSPAPI_API_KEY"
)


if not API_KEY:

    raise RuntimeError(
        "\nODDSPAPI_API_KEY was not found in .env\n"
    )


# ============================================================
# GENERAL HELPERS
# ============================================================

def header(text):

    print()

    print(
        "=" * 110
    )

    print(
        text
    )

    print(
        "=" * 110
    )


def utc_now():

    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )


def safe_float(value):

    try:

        if value is None:

            return None


        if pd.isna(
            value
        ):

            return None


        return float(
            value
        )


    except Exception:

        return None


def american_display(
    price
):

    if price is None:

        return "N/A"


    if pd.isna(
        price
    ):

        return "N/A"


    price = int(
        round(
            float(
                price
            )
        )
    )


    if price > 0:

        return (
            f"+{price}"
        )


    return str(
        price
    )


def spread_display(
    team,
    spread,
    price
):

    if spread is None:

        return "Unavailable"


    if pd.isna(
        spread
    ):

        return "Unavailable"


    spread = float(
        spread
    )


    if abs(
        spread
    ) < 0.05:

        spread_text = "PK"


    elif spread > 0:

        spread_text = (
            f"+{spread:.1f}"
        )


    else:

        spread_text = (
            f"{spread:.1f}"
        )


    if (
        price is None
        or
        pd.isna(
            price
        )
    ):

        return (
            f"{team} "
            f"{spread_text}"
        )


    return (

        f"{team} "
        f"{spread_text} "
        f"({american_display(price)})"
    )


# ============================================================
# AMERICAN ODDS → IMPLIED PROBABILITY
# ============================================================

def american_implied_probability(
    price
):

    price = safe_float(
        price
    )


    if price is None:

        return None


    if price == 0:

        return None


    if price < 0:

        return (

            -price

            /

            (
                -price
                +
                100.0
            )
        )


    return (

        100.0

        /

        (
            price
            +
            100.0
        )
    )


# ============================================================
# API
# ============================================================

def api_get(
    endpoint,
    **params
):

    params[
        "apiKey"
    ] = API_KEY


    url = (

        f"{BASE_URL}/"
        f"{endpoint}"
    )


    response = requests.get(

        url,

        params=params,

        timeout=90
    )


    print(

        f"GET /{endpoint} "
        f"HTTP {response.status_code}"
    )


    if response.status_code != 200:

        print()

        print(
            "RESPONSE BODY:"
        )

        print(
            response.text[
                :3000
            ]
        )


        raise RuntimeError(

            f"OddsPapi request failed "
            f"with HTTP "
            f"{response.status_code}."
        )


    return response.json()


# ============================================================
# PAYLOAD HELPERS
# ============================================================

def extract_list(
    payload
):

    if isinstance(
        payload,
        list
    ):

        return payload


    if isinstance(
        payload,
        dict
    ):

        for key in [

            "data",
            "fixtures",
            "events",
            "results"

        ]:

            value = payload.get(
                key
            )


            if isinstance(
                value,
                list
            ):

                return value


        if "fixtureId" in payload:

            return [
                payload
            ]


    return []


# ============================================================
# CACHED MARKET CATALOG
#
# The catalog contains OddsPapi's market definitions.
#
# We already downloaded it during testing, so the normal
# Saturday collector does not waste another API request.
# ============================================================

def find_latest_market_catalog():

    files = sorted(

        ODDSPAPI_RAW_DIRECTORY.glob(
            "markets_*.json"
        )
    )


    if not files:

        raise RuntimeError(

            "\nNo cached OddsPapi market catalog found.\n\n"
            "Expected something like:\n"
            "data/raw/oddspapi/markets_YYYYMMDDTHHMMSSZ.json\n"
        )


    return files[
        -1
    ]


def load_market_catalog():

    path = find_latest_market_catalog()


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        payload = json.load(
            file
        )


    markets = extract_list(
        payload
    )


    catalog = {}


    for market in markets:

        market_id = market.get(
            "marketId"
        )


        if market_id is None:

            continue


        outcome_names = {}


        for outcome in market.get(
            "outcomes",
            []
        ):

            outcome_id = outcome.get(
                "outcomeId"
            )


            if outcome_id is None:

                continue


            outcome_names[
                str(
                    outcome_id
                )
            ] = str(

                outcome.get(
                    "outcomeName"
                )
            )


        catalog[
            str(
                market_id
            )
        ] = {

            "market_name":
                market.get(
                    "marketName"
                ),

            "market_type":
                market.get(
                    "marketType"
                ),

            "period":
                market.get(
                    "period"
                ),

            "handicap":
                safe_float(
                    market.get(
                        "handicap"
                    )
                ),

            "player_prop":
                market.get(
                    "playerProp"
                ),

            "outcomes":
                outcome_names
        }


    return (
        path,
        catalog
    )


# ============================================================
# ACTIVE GAME PRICE
#
# players["0"] is the game-level price in the payload we
# audited.
# ============================================================

def active_game_price(
    market_data,
    market_meta,
    wanted_outcome_name
):

    outcomes = market_data.get(
        "outcomes",
        {}
    )


    outcome_names = market_meta.get(
        "outcomes",
        {}
    )


    for (
        outcome_id,
        outcome_data
    ) in outcomes.items():

        outcome_name = outcome_names.get(
            str(
                outcome_id
            )
        )


        if str(
            outcome_name
        ) != str(
            wanted_outcome_name
        ):

            continue


        price_entry = (

            outcome_data
            .get(
                "players",
                {}
            )
            .get(
                "0"
            )
        )


        if not price_entry:

            continue


        if not price_entry.get(
            "active",
            False
        ):

            continue


        return {

            "outcome_id":
                str(
                    outcome_id
                ),

            "american":
                safe_float(
                    price_entry.get(
                        "priceAmerican"
                    )
                ),

            "decimal":
                safe_float(
                    price_entry.get(
                        "price"
                    )
                ),

            "main_line":
                bool(
                    price_entry.get(
                        "mainLine",
                        False
                    )
                ),

            "changed_at":
                price_entry.get(
                    "changedAt"
                )
        }


    return None


# ============================================================
# MONEYLINE
#
# We only accept:
#
#   Winner (incl. overtime)
#   period=result
# ============================================================

def resolve_moneyline(
    hardrock,
    catalog
):

    candidates = []


    for (
        market_id,
        market_data
    ) in hardrock.get(
        "markets",
        {}
    ).items():

        meta = catalog.get(
            str(
                market_id
            )
        )


        if not meta:

            continue


        if meta.get(
            "player_prop"
        ) is True:

            continue


        if (

            str(
                meta.get(
                    "market_type"
                )
            ).lower()

            !=

            "moneyline"
        ):

            continue


        if (

            str(
                meta.get(
                    "period"
                )
            ).lower()

            !=

            "result"
        ):

            continue


        if (

            str(
                meta.get(
                    "market_name"
                )
            ).strip().lower()

            !=

            "winner (incl. overtime)"
        ):

            continue


        participant1 = active_game_price(

            market_data,

            meta,

            "1"
        )


        participant2 = active_game_price(

            market_data,

            meta,

            "2"
        )


        if not participant1 or not participant2:

            continue


        candidates.append(

            {
                "market_id":
                    str(
                        market_id
                    ),

                "p1":
                    participant1,

                "p2":
                    participant2
            }
        )


    if not candidates:

        return None


    # --------------------------------------------------------
    # Normally there is exactly one full-game moneyline.
    #
    # If duplicates appear, prefer the pair whose total
    # implied probability looks most like a normal two-way
    # sportsbook market.
    # --------------------------------------------------------

    def candidate_score(
        candidate
    ):

        q1 = american_implied_probability(

            candidate[
                "p1"
            ][
                "american"
            ]
        )


        q2 = american_implied_probability(

            candidate[
                "p2"
            ][
                "american"
            ]
        )


        if q1 is None or q2 is None:

            return float(
                "inf"
            )


        return abs(

            (
                q1
                +
                q2
            )

            -

            1.047619
        )


    return min(

        candidates,

        key=candidate_score
    )


# ============================================================
# BALANCED TWO-WAY RUNG RESOLVER
#
# For each spread/total rung:
#
# raw implied probability
#            ↓
# remove vig
#            ↓
# compare fair probability to 50%
#
#
# Example:
#
# -110 / -110
#
# raw:
#   52.38%
#   52.38%
#
# de-vig:
#   50%
#   50%
#
# balance_score = 0.000
#
#
# A balance score of 0.05 corresponds roughly to:
#
#   45% / 55%
#
# Anything farther away is treated as an alternate/truncated
# ladder candidate rather than a trustworthy headline line.
# ============================================================

def resolve_balanced_rung(
    candidates
):

    scored = []


    for candidate in candidates:

        side1_price = candidate[
            "side1"
        ][
            "american"
        ]


        side2_price = candidate[
            "side2"
        ][
            "american"
        ]


        q1 = american_implied_probability(
            side1_price
        )


        q2 = american_implied_probability(
            side2_price
        )


        if q1 is None or q2 is None:

            continue


        total_implied_probability = (

            q1
            +
            q2
        )


        if total_implied_probability <= 0:

            continue


        fair_side1_probability = (

            q1

            /

            total_implied_probability
        )


        fair_side2_probability = (

            q2

            /

            total_implied_probability
        )


        balance_score = abs(

            fair_side1_probability

            -

            0.50
        )


        # Typical -110/-110 market:
        #
        # 0.5238095 + 0.5238095 = 1.047619
        #
        # This serves as a secondary tie-breaker.

        vig_shape_score = abs(

            total_implied_probability

            -

            1.047619
        )


        result = dict(
            candidate
        )


        result[
            "fair_side1_probability"
        ] = fair_side1_probability


        result[
            "fair_side2_probability"
        ] = fair_side2_probability


        result[
            "balance_score"
        ] = balance_score


        result[
            "total_implied_probability"
        ] = total_implied_probability


        result[
            "vig_shape_score"
        ] = vig_shape_score


        result[
            "quality_pass"
        ] = int(

            balance_score

            <=

            MAX_BALANCE_SCORE
        )


        scored.append(
            result
        )


    if not scored:

        return None


    scored = sorted(

        scored,

        key=lambda candidate: (

            candidate[
                "balance_score"
            ],

            candidate[
                "vig_shape_score"
            ],

            abs(
                float(
                    candidate[
                        "line"
                    ]
                )
            )
        )
    )


    return scored[
        0
    ]


# ============================================================
# FULL-GAME SPREAD
#
# Audit established:
#
# participant1 → outcome "1"
# participant2 → outcome "2"
#
# OddsPapi handicap belongs to participant1.
#
# participant2 receives the opposite spread sign.
# ============================================================

def resolve_spread(
    hardrock,
    catalog
):

    candidates = []


    for (
        market_id,
        market_data
    ) in hardrock.get(
        "markets",
        {}
    ).items():

        meta = catalog.get(
            str(
                market_id
            )
        )


        if not meta:

            continue


        if meta.get(
            "player_prop"
        ) is True:

            continue


        if (

            str(
                meta.get(
                    "market_type"
                )
            ).lower()

            !=

            "spreads"
        ):

            continue


        if (

            str(
                meta.get(
                    "period"
                )
            ).lower()

            !=

            "result"
        ):

            continue


        if (

            str(
                meta.get(
                    "market_name"
                )
            ).strip().lower()

            !=

            "handicap (incl. overtime)"
        ):

            continue


        handicap = safe_float(

            meta.get(
                "handicap"
            )
        )


        if handicap is None:

            continue


        participant1 = active_game_price(

            market_data,

            meta,

            "1"
        )


        participant2 = active_game_price(

            market_data,

            meta,

            "2"
        )


        if not participant1 or not participant2:

            continue


        candidates.append(

            {
                "market_id":
                    str(
                        market_id
                    ),

                "line":
                    handicap,

                "participant1_spread":
                    handicap,

                "participant2_spread":
                    -handicap,

                "side1":
                    participant1,

                "side2":
                    participant2
            }
        )


    return resolve_balanced_rung(
        candidates
    )


# ============================================================
# FULL-GAME TOTAL
#
# Same balanced-market logic is calculated and stored.
#
# Our current betting system is still spread-focused, so the
# total-quality result is diagnostic for now.
# ============================================================

def resolve_total(
    hardrock,
    catalog
):

    candidates = []


    for (
        market_id,
        market_data
    ) in hardrock.get(
        "markets",
        {}
    ).items():

        meta = catalog.get(
            str(
                market_id
            )
        )


        if not meta:

            continue


        if meta.get(
            "player_prop"
        ) is True:

            continue


        if (

            str(
                meta.get(
                    "market_type"
                )
            ).lower()

            !=

            "totals"
        ):

            continue


        if (

            str(
                meta.get(
                    "period"
                )
            ).lower()

            !=

            "result"
        ):

            continue


        if (

            str(
                meta.get(
                    "market_name"
                )
            ).strip().lower()

            !=

            "total (incl. overtime)"
        ):

            continue


        total = safe_float(

            meta.get(
                "handicap"
            )
        )


        if total is None:

            continue


        over = active_game_price(

            market_data,

            meta,

            "Over"
        )


        under = active_game_price(

            market_data,

            meta,

            "Under"
        )


        if not over or not under:

            continue


        candidates.append(

            {
                "market_id":
                    str(
                        market_id
                    ),

                "line":
                    total,

                "side1":
                    over,

                "side2":
                    under
            }
        )


    return resolve_balanced_rung(
        candidates
    )


# ============================================================
# SQLITE TABLE
# ============================================================

def create_table(
    connection
):

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS hardrock_execution_snapshots (

            batch_id TEXT NOT NULL,

            captured_at_utc TEXT NOT NULL,

            source TEXT NOT NULL,

            fixture_id TEXT NOT NULL,

            start_time_utc TEXT,

            status_name TEXT,

            bookmaker_key TEXT NOT NULL,

            bookmaker_active INTEGER,

            suspended INTEGER,

            participant1_id TEXT,

            participant1_name TEXT,

            participant1_short_name TEXT,

            participant1_abbr TEXT,

            participant2_id TEXT,

            participant2_name TEXT,

            participant2_short_name TEXT,

            participant2_abbr TEXT,

            participant1_moneyline REAL,

            participant2_moneyline REAL,

            participant1_spread REAL,

            participant1_spread_price REAL,

            participant2_spread REAL,

            participant2_spread_price REAL,

            spread_market_id TEXT,

            spread_balance_score REAL,

            spread_selected_mainline_flag INTEGER,

            spread_quality_pass INTEGER,

            spread_quality_status TEXT,

            total REAL,

            over_price REAL,

            under_price REAL,

            total_market_id TEXT,

            total_balance_score REAL,

            total_selected_mainline_flag INTEGER,

            total_quality_pass INTEGER,

            fixture_path TEXT,

            headline_resolver_version TEXT,

            raw_fixture_json TEXT,

            UNIQUE (
                batch_id,
                fixture_id
            )
        )
        """
    )


    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_hardrock_execution_fixture

        ON hardrock_execution_snapshots (
            fixture_id,
            captured_at_utc
        )
        """
    )


    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_hardrock_execution_time

        ON hardrock_execution_snapshots (
            captured_at_utc
        )
        """
    )


# ============================================================
# DATABASE MIGRATION
#
# Our V1 table already exists.
#
# SQLite's CREATE TABLE IF NOT EXISTS does not add new
# columns to an existing table, so explicitly add the V2
# quality fields when needed.
# ============================================================

def ensure_column(
    connection,
    name,
    sql_type
):

    existing_columns = {

        row[
            1
        ]

        for row in connection.execute(
            """
            PRAGMA table_info(
                hardrock_execution_snapshots
            )
            """
        )
    }


    if name not in existing_columns:

        print(

            f"Adding database column: "
            f"{name}"
        )


        connection.execute(

            f"""
            ALTER TABLE hardrock_execution_snapshots
            ADD COLUMN {name} {sql_type}
            """
        )


# ============================================================
# MAIN
# ============================================================

def main():

    captured_at = utc_now()


    batch_id = captured_at.strftime(
        "%Y%m%dT%H%M%SZ"
    )


    print(
        "HARD ROCK EXECUTION SNAPSHOT — ODDSPAPI V2"
    )


    print(
        "=" * 110
    )


    print(

        f"Captured UTC:         "
        f"{captured_at.isoformat()}"
    )


    print(

        f"Bookmaker:            "
        f"{BOOKMAKER}"
    )


    print(

        f"NCAAF tournament:     "
        f"{NCAAF_TOURNAMENT_ID}"
    )


    print(

        f"Resolver:             "
        f"{HEADLINE_RESOLVER_VERSION}"
    )


    print(

        f"Max balance score:    "
        f"{MAX_BALANCE_SCORE:.3f}"
    )


    print(
        "Expected API requests: 1"
    )


    # ========================================================
    # LOAD MARKET CATALOG
    # ========================================================

    header(
        "LOAD CACHED MARKET CATALOG"
    )


    (
        catalog_path,
        catalog
    ) = load_market_catalog()


    print(
        "Catalog:"
    )


    print(
        f"  {catalog_path}"
    )


    print(

        f"Market definitions: "
        f"{len(catalog):,}"
    )


    # ========================================================
    # HARD ROCK DOWNLOAD
    # ========================================================

    header(
        "DOWNLOAD HARD ROCK NCAAF"
    )


    payload = api_get(

        "odds-by-tournaments",

        tournamentIds=str(
            NCAAF_TOURNAMENT_ID
        ),

        bookmakers=BOOKMAKER,

        oddsFormat="american",

        language="en",

        verbosity=3
    )


    RAW_DIRECTORY.mkdir(

        parents=True,

        exist_ok=True
    )


    raw_path = (

        RAW_DIRECTORY

        /

        (
            "hardrock_ncaaf_"
            f"{batch_id}.json"
        )
    )


    with open(
        raw_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            payload,

            file,

            indent=2
        )


    fixtures = extract_list(
        payload
    )


    print()

    print(

        f"Fixtures returned: "
        f"{len(fixtures):,}"
    )


    print(
        "Raw snapshot:"
    )


    print(
        f"  {raw_path}"
    )


    # ========================================================
    # RESOLVE MARKETS
    # ========================================================

    header(
        "RESOLVE HARD ROCK MARKETS"
    )


    rows = []


    hardrock_fixture_count = 0

    active_fixture_count = 0

    moneyline_count = 0

    spread_candidate_count = 0

    usable_spread_count = 0

    rejected_spread_count = 0

    total_candidate_count = 0

    usable_total_count = 0


    for fixture in fixtures:

        hardrock = (

            fixture
            .get(
                "bookmakerOdds",
                {}
            )
            .get(
                BOOKMAKER
            )
        )


        if not hardrock:

            continue


        hardrock_fixture_count += 1


        bookmaker_active = bool(

            hardrock.get(
                "bookmakerIsActive",
                False
            )
        )


        suspended = bool(

            hardrock.get(
                "suspended",
                False
            )
        )


        if (

            bookmaker_active

            and

            not suspended
        ):

            active_fixture_count += 1


        moneyline = resolve_moneyline(

            hardrock,

            catalog
        )


        spread = resolve_spread(

            hardrock,

            catalog
        )


        total = resolve_total(

            hardrock,

            catalog
        )


        if moneyline:

            moneyline_count += 1


        if spread:

            spread_candidate_count += 1


            if int(
                spread[
                    "quality_pass"
                ]
            ) == 1:

                usable_spread_count += 1


            else:

                rejected_spread_count += 1


        if total:

            total_candidate_count += 1


            if int(
                total[
                    "quality_pass"
                ]
            ) == 1:

                usable_total_count += 1


        # ----------------------------------------------------
        # SPREAD QUALITY
        # ----------------------------------------------------

        if spread is None:

            spread_quality_pass = 0

            spread_quality_status = (
                "NO_SPREAD_MARKET"
            )


        elif int(
            spread[
                "quality_pass"
            ]
        ) == 1:

            spread_quality_pass = 1

            spread_quality_status = (
                "HEADLINE_USABLE"
            )


        else:

            spread_quality_pass = 0

            spread_quality_status = (
                "NO_BALANCED_HEADLINE"
            )


        # ----------------------------------------------------
        # TOTAL QUALITY
        # ----------------------------------------------------

        if total is None:

            total_quality_pass = 0


        else:

            total_quality_pass = int(

                total[
                    "quality_pass"
                ]
            )


        # ----------------------------------------------------
        # DATABASE ROW
        # ----------------------------------------------------

        row = {

            "batch_id":
                batch_id,

            "captured_at_utc":
                captured_at.isoformat(),

            "source":
                SOURCE_NAME,

            "fixture_id":
                fixture.get(
                    "fixtureId"
                ),

            "start_time_utc":
                fixture.get(
                    "startTime"
                ),

            "status_name":
                fixture.get(
                    "statusName"
                ),

            "bookmaker_key":
                BOOKMAKER,

            "bookmaker_active":
                int(
                    bookmaker_active
                ),

            "suspended":
                int(
                    suspended
                ),

            "participant1_id":
                str(
                    fixture.get(
                        "participant1Id"
                    )
                    or
                    ""
                ),

            "participant1_name":
                fixture.get(
                    "participant1Name"
                ),

            "participant1_short_name":
                fixture.get(
                    "participant1ShortName"
                ),

            "participant1_abbr":
                fixture.get(
                    "participant1Abbr"
                ),

            "participant2_id":
                str(
                    fixture.get(
                        "participant2Id"
                    )
                    or
                    ""
                ),

            "participant2_name":
                fixture.get(
                    "participant2Name"
                ),

            "participant2_short_name":
                fixture.get(
                    "participant2ShortName"
                ),

            "participant2_abbr":
                fixture.get(
                    "participant2Abbr"
                ),

            "participant1_moneyline":
                (
                    moneyline[
                        "p1"
                    ][
                        "american"
                    ]

                    if moneyline

                    else None
                ),

            "participant2_moneyline":
                (
                    moneyline[
                        "p2"
                    ][
                        "american"
                    ]

                    if moneyline

                    else None
                ),

            "participant1_spread":
                (
                    spread[
                        "participant1_spread"
                    ]

                    if spread

                    else None
                ),

            "participant1_spread_price":
                (
                    spread[
                        "side1"
                    ][
                        "american"
                    ]

                    if spread

                    else None
                ),

            "participant2_spread":
                (
                    spread[
                        "participant2_spread"
                    ]

                    if spread

                    else None
                ),

            "participant2_spread_price":
                (
                    spread[
                        "side2"
                    ][
                        "american"
                    ]

                    if spread

                    else None
                ),

            "spread_market_id":
                (
                    spread[
                        "market_id"
                    ]

                    if spread

                    else None
                ),

            "spread_balance_score":
                (
                    spread[
                        "balance_score"
                    ]

                    if spread

                    else None
                ),

            "spread_selected_mainline_flag":
                (
                    int(

                        bool(
                            spread[
                                "side1"
                            ][
                                "main_line"
                            ]
                        )

                        or

                        bool(
                            spread[
                                "side2"
                            ][
                                "main_line"
                            ]
                        )
                    )

                    if spread

                    else None
                ),

            "spread_quality_pass":
                spread_quality_pass,

            "spread_quality_status":
                spread_quality_status,

            "total":
                (
                    total[
                        "line"
                    ]

                    if total

                    else None
                ),

            "over_price":
                (
                    total[
                        "side1"
                    ][
                        "american"
                    ]

                    if total

                    else None
                ),

            "under_price":
                (
                    total[
                        "side2"
                    ][
                        "american"
                    ]

                    if total

                    else None
                ),

            "total_market_id":
                (
                    total[
                        "market_id"
                    ]

                    if total

                    else None
                ),

            "total_balance_score":
                (
                    total[
                        "balance_score"
                    ]

                    if total

                    else None
                ),

            "total_selected_mainline_flag":
                (
                    int(

                        bool(
                            total[
                                "side1"
                            ][
                                "main_line"
                            ]
                        )

                        or

                        bool(
                            total[
                                "side2"
                            ][
                                "main_line"
                            ]
                        )
                    )

                    if total

                    else None
                ),

            "total_quality_pass":
                total_quality_pass,

            "fixture_path":
                hardrock.get(
                    "fixturePath"
                ),

            "headline_resolver_version":
                HEADLINE_RESOLVER_VERSION,

            "raw_fixture_json":
                json.dumps(

                    fixture,

                    separators=(
                        ",",
                        ":"
                    )
                )
        }


        rows.append(
            row
        )


    frame = pd.DataFrame(
        rows
    )


    print(

        f"Hard Rock fixtures:         "
        f"{hardrock_fixture_count:,}"
    )


    print(

        f"Active + unsuspended:       "
        f"{active_fixture_count:,}"
    )


    print(

        f"Moneylines resolved:        "
        f"{moneyline_count:,}"
    )


    print(

        f"Spread candidates found:    "
        f"{spread_candidate_count:,}"
    )


    print(

        f"Usable headline spreads:    "
        f"{usable_spread_count:,}"
    )


    print(

        f"Rejected spread ladders:    "
        f"{rejected_spread_count:,}"
    )


    print(

        f"Total candidates found:     "
        f"{total_candidate_count:,}"
    )


    print(

        f"Usable headline totals:     "
        f"{usable_total_count:,}"
    )


    # ========================================================
    # SQLITE
    # ========================================================

    header(
        "WRITE SQLITE"
    )


    connection = sqlite3.connect(
        DATABASE_FILE
    )


    try:

        create_table(
            connection
        )


        ensure_column(

            connection,

            "spread_quality_pass",

            "INTEGER"
        )


        ensure_column(

            connection,

            "spread_quality_status",

            "TEXT"
        )


        ensure_column(

            connection,

            "total_quality_pass",

            "INTEGER"
        )


        columns = list(
            frame.columns
        )


        placeholders = ",".join(

            "?"

            for _ in columns
        )


        sql = (

            "INSERT OR IGNORE INTO "
            "hardrock_execution_snapshots "

            "("

            +

            ",".join(
                columns
            )

            +

            ") "

            "VALUES ("

            +

            placeholders

            +

            ")"
        )


        inserted = 0


        for _, frame_row in frame.iterrows():

            values = []


            for column in columns:

                value = frame_row[
                    column
                ]


                if pd.isna(
                    value
                ):

                    value = None


                values.append(
                    value
                )


            cursor = connection.execute(

                sql,

                values
            )


            inserted += cursor.rowcount


        connection.commit()


    finally:

        connection.close()


    print(

        f"Rows inserted: "
        f"{inserted:,}"
    )


    # ========================================================
    # LATEST CSV
    # ========================================================

    PROCESSED_DIRECTORY.mkdir(

        parents=True,

        exist_ok=True
    )


    frame.to_csv(

        LATEST_CSV,

        index=False
    )


    print(
        "Latest CSV:"
    )


    print(
        f"  {LATEST_CSV}"
    )


    # ========================================================
    # QUALITY EXAMPLES
    # ========================================================

    header(
        "QUALITY EXAMPLES"
    )


    usable = frame[

        frame[
            "spread_quality_pass"
        ]
        ==
        1

    ].copy()


    rejected = frame[

        frame[
            "spread_quality_status"
        ]
        ==
        "NO_BALANCED_HEADLINE"

    ].copy()


    # --------------------------------------------------------
    # USABLE
    # --------------------------------------------------------

    print(
        "BEST USABLE HEADLINES"
    )


    print(
        "-" * 110
    )


    if len(
        usable
    ) == 0:

        print(
            "None"
        )


    else:

        usable_show = (

            usable

            .sort_values(
                "spread_balance_score",
                ascending=True
            )

            .head(
                10
            )
        )


        for _, usable_row in usable_show.iterrows():

            participant1_display = spread_display(

                usable_row[
                    "participant1_short_name"
                ],

                usable_row[
                    "participant1_spread"
                ],

                usable_row[
                    "participant1_spread_price"
                ]
            )


            participant2_display = spread_display(

                usable_row[
                    "participant2_short_name"
                ],

                usable_row[
                    "participant2_spread"
                ],

                usable_row[
                    "participant2_spread_price"
                ]
            )


            print()

            print(

                f"{usable_row['participant2_short_name']} "
                f"@ "
                f"{usable_row['participant1_short_name']}"
            )


            print(

                f"  "
                f"{participant1_display}"
            )


            print(

                f"  "
                f"{participant2_display}"
            )


            print(

                f"  balance="
                f"{float(usable_row['spread_balance_score']):.4f}"
            )


    # --------------------------------------------------------
    # REJECTED
    # --------------------------------------------------------

    print()

    print(
        "REJECTED / ALTERNATE-LADDER CANDIDATES"
    )


    print(
        "-" * 110
    )


    if len(
        rejected
    ) == 0:

        print(
            "None"
        )


    else:

        rejected_show = (

            rejected

            .sort_values(
                "spread_balance_score",
                ascending=False
            )

            .head(
                15
            )
        )


        for _, rejected_row in rejected_show.iterrows():

            participant1_display = spread_display(

                rejected_row[
                    "participant1_short_name"
                ],

                rejected_row[
                    "participant1_spread"
                ],

                rejected_row[
                    "participant1_spread_price"
                ]
            )


            participant2_display = spread_display(

                rejected_row[
                    "participant2_short_name"
                ],

                rejected_row[
                    "participant2_spread"
                ],

                rejected_row[
                    "participant2_spread_price"
                ]
            )


            print()

            print(

                f"{rejected_row['participant2_short_name']} "
                f"@ "
                f"{rejected_row['participant1_short_name']}"
            )


            print(

                f"  RAW CANDIDATE: "
                f"{participant1_display}"
            )


            print(

                f"  RAW OPPOSITE:  "
                f"{participant2_display}"
            )


            print(

                f"  balance="
                f"{float(rejected_row['spread_balance_score']):.4f}"
            )


            print(
                "  status=NO_BALANCED_HEADLINE"
            )


    # ========================================================
    # TARGET VERIFICATION
    # ========================================================

    header(
        "TARGET GAME VERIFICATION"
    )


    target_games = [

        (
            "Kentucky",
            "Alabama"
        ),

        (
            "Texas",
            "Ohio State"
        ),

        (
            "Georgia Tech",
            "Tennessee"
        )
    ]


    for (
        participant1_term,
        participant2_term
    ) in target_games:

        matches = frame[

            frame[
                "participant1_name"
            ].str.contains(
                participant1_term,
                case=False,
                na=False
            )

            &

            frame[
                "participant2_name"
            ].str.contains(
                participant2_term,
                case=False,
                na=False
            )

        ]


        print()


        if len(
            matches
        ) == 0:

            print(

                f"{participant2_term} "
                f"@ "
                f"{participant1_term}: "
                f"NOT FOUND"
            )


            continue


        target_row = matches.iloc[
            0
        ]


        print(

            f"{target_row['participant2_short_name']} "
            f"@ "
            f"{target_row['participant1_short_name']}"
        )


        print(

            f"  Quality: "
            f"{target_row['spread_quality_status']}"
        )


        if pd.notna(
            target_row[
                "spread_balance_score"
            ]
        ):

            print(

                f"  Balance score: "
                f"{float(target_row['spread_balance_score']):.4f}"
            )


        # ----------------------------------------------------
        # Only call it a Hard Rock headline when it passed.
        # ----------------------------------------------------

        if int(
            target_row[
                "spread_quality_pass"
            ]
        ) == 1:

            print(
                "  HARD ROCK HEADLINE:"
            )


            print(

                "    "

                +

                spread_display(

                    target_row[
                        "participant1_short_name"
                    ],

                    target_row[
                        "participant1_spread"
                    ],

                    target_row[
                        "participant1_spread_price"
                    ]
                )
            )


            print(

                "    "

                +

                spread_display(

                    target_row[
                        "participant2_short_name"
                    ],

                    target_row[
                        "participant2_spread"
                    ],

                    target_row[
                        "participant2_spread_price"
                    ]
                )
            )


        else:

            print(
                "  HARD ROCK HEADLINE: Unavailable"
            )


            if pd.notna(
                target_row[
                    "participant1_spread"
                ]
            ):

                print(
                    "  Raw alternate candidate:"
                )


                print(

                    "    "

                    +

                    spread_display(

                        target_row[
                            "participant1_short_name"
                        ],

                        target_row[
                            "participant1_spread"
                        ],

                        target_row[
                            "participant1_spread_price"
                        ]
                    )
                )


                print(

                    "    "

                    +

                    spread_display(

                        target_row[
                            "participant2_short_name"
                        ],

                        target_row[
                            "participant2_spread"
                        ],

                        target_row[
                            "participant2_spread_price"
                        ]
                    )
                )


        print(
            "  MONEYLINE:"
        )


        print(

            f"    "
            f"{target_row['participant1_short_name']} "
            f"{american_display(target_row['participant1_moneyline'])}"
        )


        print(

            f"    "
            f"{target_row['participant2_short_name']} "
            f"{american_display(target_row['participant2_moneyline'])}"
        )


        if pd.notna(
            target_row[
                "total"
            ]
        ):

            print(
                "  TOTAL CANDIDATE:"
            )


            print(

                f"    "
                f"{float(target_row['total']):.1f} | "
                f"Over "
                f"{american_display(target_row['over_price'])} | "
                f"Under "
                f"{american_display(target_row['under_price'])}"
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    header(
        "HARD ROCK SNAPSHOT COMPLETE"
    )


    print(

        f"Batch: "
        f"{batch_id}"
    )


    print()


    print(
        "HEADLINE SPREAD RULE:"
    )


    print(

        f"  spread_balance_score "
        f"<= "
        f"{MAX_BALANCE_SCORE:.2f}"
    )


    print(
        "      → HEADLINE_USABLE"
    )


    print()

    print(

        f"  spread_balance_score "
        f"> "
        f"{MAX_BALANCE_SCORE:.2f}"
    )


    print(
        "      → NO_BALANCED_HEADLINE"
    )


    print()


    print(
        "Raw alternate-line candidates remain stored."
    )


    print(
        "They will not be used for MODEL EDGE or "
        "spread Strength."
    )


    print()

    print(
        "SQLite table:"
    )


    print(
        "  hardrock_execution_snapshots"
    )


    print()

    print(
        "Latest CSV:"
    )


    print(
        f"  {LATEST_CSV}"
    )


    print()

    print(
        "No scores or game outcomes were used."
    )


if __name__ == "__main__":

    main()