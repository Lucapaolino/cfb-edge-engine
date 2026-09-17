import json
import os
import time

from datetime import datetime, timezone
from pathlib import Path

import requests

from dotenv import load_dotenv


# ============================================================
# PURPOSE
#
# Answer ONE question:
#
# Does OddsPapi currently provide usable Hard Rock Bet odds
# for NCAAF?
#
# We are specifically looking for:
#
# - moneyline
# - spreads + juice
# - totals + juice
#
# This script:
#
# - does NOT modify the database
# - does NOT modify our betting board
# - does NOT train anything
# - does NOT use The Odds API credits
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://api.oddspapi.io/v4"

SPORT_ID = 14

NCAAF_TOURNAMENT_ID = 27653

BOOKMAKER = "hardrockbet"


TARGET_GAMES = [

    (
        "Alabama",
        "Kentucky"
    ),

    (
        "Ohio State",
        "Texas"
    ),

    (
        "Tennessee",
        "Georgia Tech"
    )
]


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
    /
    ".env"
)


RAW_DIRECTORY = (
    PROJECT_ROOT
    /
    "data"
    /
    "raw"
    /
    "oddspapi"
)


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv(
    ENV_FILE
)


API_KEY = os.getenv(
    "ODDSPAPI_API_KEY"
)


if not API_KEY:

    raise RuntimeError(

        "\nODDSPAPI_API_KEY was not found.\n\n"
        f"Expected it in:\n{ENV_FILE}\n"
    )


# ============================================================
# HELPERS
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


def normalize_text(value):

    if value is None:

        return ""

    return (

        str(value)
        .lower()
        .replace("'", "")
        .replace("’", "")
        .strip()
    )


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


    for attempt in range(
        1,
        4
    ):

        response = requests.get(

            url,

            params=params,

            timeout=90
        )


        print(

            f"GET /{endpoint} "
            f"HTTP {response.status_code}"
        )


        if response.status_code == 200:

            return response.json()


        if response.status_code == 429:

            try:

                payload = response.json()

                retry_ms = (

                    payload
                    .get(
                        "error",
                        {}
                    )
                    .get(
                        "retryMs",
                        1500
                    )
                )

            except Exception:

                retry_ms = 1500


            wait_seconds = (

                float(
                    retry_ms
                )

                /

                1000.0

                +

                0.5
            )


            print(

                f"Rate limited. "
                f"Waiting {wait_seconds:.1f}s..."
            )


            time.sleep(
                wait_seconds
            )

            continue


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


    raise RuntimeError(

        "OddsPapi request remained "
        "rate limited after retries."
    )


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
# MARKET CATALOG
# ============================================================

def build_market_catalog(
    market_payload
):

    markets = extract_list(
        market_payload
    )


    by_id = {}


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
            ] = outcome.get(
                "outcomeName"
            )


        by_id[
            str(
                market_id
            )
        ] = {

            "name":
                market.get(
                    "marketName"
                ),

            "short_name":
                market.get(
                    "marketNameShort"
                ),

            "type":
                market.get(
                    "marketType"
                ),

            "period":
                market.get(
                    "period"
                ),

            "handicap":
                market.get(
                    "handicap"
                ),

            "player_prop":
                market.get(
                    "playerProp"
                ),

            "outcomes":
                outcome_names
        }


    return by_id


# ============================================================
# FIND TARGET GAME
# ============================================================

def fixture_contains_teams(
    fixture,
    team_a,
    team_b
):

    names = [

        normalize_text(
            fixture.get(
                "participant1Name"
            )
        ),

        normalize_text(
            fixture.get(
                "participant2Name"
            )
        )
    ]


    team_a = normalize_text(
        team_a
    )


    team_b = normalize_text(
        team_b
    )


    a_found = any(

        team_a in name

        for name in names
    )


    b_found = any(

        team_b in name

        for name in names
    )


    return (

        a_found

        and

        b_found
    )


# ============================================================
# GET STANDARD GAME-LINE PRICES
# ============================================================

def get_game_line_rows(
    fixture,
    market_catalog
):

    bookmaker_odds = fixture.get(
        "bookmakerOdds",
        {}
    )


    hardrock = bookmaker_odds.get(
        BOOKMAKER
    )


    if not hardrock:

        return []


    rows = []


    for (
        market_id,
        market
    ) in hardrock.get(
        "markets",
        {}
    ).items():

        meta = market_catalog.get(

            str(
                market_id
            ),

            {}
        )


        market_name = str(

            meta.get(
                "name"
            )
            or
            ""
        )


        market_type = str(

            meta.get(
                "type"
            )
            or
            ""
        )


        period = str(

            meta.get(
                "period"
            )
            or
            ""
        )


        player_prop = meta.get(
            "player_prop"
        )


        # ----------------------------------------------------
        # Ignore player props.
        # ----------------------------------------------------

        if player_prop is True:

            continue


        market_name_lower = (
            market_name.lower()
        )


        market_type_lower = (
            market_type.lower()
        )


        # ----------------------------------------------------
        # Identify the three market families we care about.
        # ----------------------------------------------------

        family = None


        if (

            market_type_lower
            ==
            "moneyline"

            or

            (
                "winner"
                in
                market_name_lower

                and

                "overtime"
                in
                market_name_lower
            )
        ):

            family = "MONEYLINE"


        elif (

            "handicap"
            in
            market_name_lower

            and

            "overtime"
            in
            market_name_lower
        ):

            family = "SPREAD"


        elif (

            (
                "total"
                in
                market_name_lower

                or

                "over under"
                in
                market_name_lower
            )

            and

            "overtime"
            in
            market_name_lower
        ):

            family = "TOTAL"


        if family is None:

            continue


        outcomes = market.get(
            "outcomes",
            {}
        )


        for (
            outcome_id,
            outcome
        ) in outcomes.items():

            players = outcome.get(
                "players",
                {}
            )


            # ------------------------------------------------
            # "0" means this is a game-line price rather than
            # a player prop.
            # ------------------------------------------------

            price_entry = players.get(
                "0"
            )


            if not price_entry:

                continue


            if not price_entry.get(
                "active",
                False
            ):

                continue


            outcome_name = (

                meta.get(
                    "outcomes",
                    {}
                )

                .get(
                    str(
                        outcome_id
                    )
                )
            )


            rows.append(

                {
                    "family":
                        family,

                    "market_id":
                        str(
                            market_id
                        ),

                    "market_name":
                        market_name,

                    "period":
                        period,

                    "handicap":
                        meta.get(
                            "handicap"
                        ),

                    "outcome_id":
                        str(
                            outcome_id
                        ),

                    "outcome_name":
                        outcome_name,

                    "price_american":
                        price_entry.get(
                            "priceAmerican"
                        ),

                    "price_decimal":
                        price_entry.get(
                            "price"
                        ),

                    "main_line":
                        price_entry.get(
                            "mainLine"
                        ),

                    "changed_at":
                        price_entry.get(
                            "changedAt"
                        )
                }
            )


    return rows


# ============================================================
# DISPLAY GAME
# ============================================================

def display_fixture(
    fixture,
    market_catalog
):

    participant1 = fixture.get(
        "participant1Name"
    )


    participant2 = fixture.get(
        "participant2Name"
    )


    print()

    print(
        f"{participant1} vs "
        f"{participant2}"
    )


    print(
        f"Fixture ID: "
        f"{fixture.get('fixtureId')}"
    )


    print(
        f"Start UTC:  "
        f"{fixture.get('startTime')}"
    )


    print(
        f"Status:     "
        f"{fixture.get('statusName')}"
    )


    bookmaker_odds = fixture.get(
        "bookmakerOdds",
        {}
    )


    hardrock = bookmaker_odds.get(
        BOOKMAKER
    )


    if not hardrock:

        print()

        print(
            "HARD ROCK: NOT PRESENT"
        )

        return False


    print()

    print(
        "HARD ROCK: PRESENT"
    )


    print(
        f"Active:     "
        f"{hardrock.get('bookmakerIsActive')}"
    )


    print(
        f"Suspended:  "
        f"{hardrock.get('suspended')}"
    )


    fixture_path = hardrock.get(
        "fixturePath"
    )


    if fixture_path:

        print(
            f"Fixture URL: "
            f"{fixture_path}"
        )


    rows = get_game_line_rows(

        fixture,

        market_catalog
    )


    print()

    print(
        f"Active standard game-line "
        f"prices found: "
        f"{len(rows)}"
    )


    # ========================================================
    # MONEYLINE
    # ========================================================

    moneylines = [

        row

        for row in rows

        if row[
            "family"
        ]
        ==
        "MONEYLINE"
    ]


    print()

    print(
        "MONEYLINE"
    )

    print(
        "-" * 80
    )


    if not moneylines:

        print(
            "  None found"
        )


    for row in moneylines:

        print(

            f"  market={row['market_id']} | "
            f"outcome={row['outcome_name']} | "
            f"American={row['price_american']} | "
            f"decimal={row['price_decimal']} | "
            f"mainLine={row['main_line']}"
        )


    # ========================================================
    # SPREADS
    # ========================================================

    spreads = [

        row

        for row in rows

        if row[
            "family"
        ]
        ==
        "SPREAD"
    ]


    spreads = sorted(

        spreads,

        key=lambda row: (

            float(
                row[
                    "handicap"
                ]
            )

            if row[
                "handicap"
            ]
            is not None

            else 9999,

            row[
                "outcome_id"
            ]
        )
    )


    print()

    print(
        "SPREAD / HANDICAP"
    )

    print(
        "-" * 80
    )


    if not spreads:

        print(
            "  None found"
        )


    for row in spreads:

        print(

            f"  line={row['handicap']} | "
            f"market={row['market_id']} | "
            f"outcome={row['outcome_name']} | "
            f"American={row['price_american']} | "
            f"decimal={row['price_decimal']} | "
            f"mainLine={row['main_line']}"
        )


    # ========================================================
    # TOTALS
    # ========================================================

    totals = [

        row

        for row in rows

        if row[
            "family"
        ]
        ==
        "TOTAL"
    ]


    totals = sorted(

        totals,

        key=lambda row: (

            float(
                row[
                    "handicap"
                ]
            )

            if row[
                "handicap"
            ]
            is not None

            else 9999,

            row[
                "outcome_id"
            ]
        )
    )


    print()

    print(
        "TOTAL"
    )

    print(
        "-" * 80
    )


    if not totals:

        print(
            "  None found"
        )


    for row in totals:

        print(

            f"  total={row['handicap']} | "
            f"market={row['market_id']} | "
            f"outcome={row['outcome_name']} | "
            f"American={row['price_american']} | "
            f"decimal={row['price_decimal']} | "
            f"mainLine={row['main_line']}"
        )


    return True


# ============================================================
# MAIN
# ============================================================

def main():

    captured_at = datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )


    batch_id = captured_at.strftime(
        "%Y%m%dT%H%M%SZ"
    )


    RAW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "TEST ODDSPAPI HARD ROCK NCAAF — V1"
    )


    print(
        "=" * 110
    )


    print(
        f"Captured UTC: "
        f"{captured_at.isoformat()}"
    )


    print(
        f"Sport ID:     "
        f"{SPORT_ID}"
    )


    print(
        f"Tournament:   "
        f"{NCAAF_TOURNAMENT_ID}"
    )


    print(
        f"Bookmaker:    "
        f"{BOOKMAKER}"
    )


    print()

    print(
        "Expected API requests: 2"
    )


    # ========================================================
    # REQUEST 1 — MARKET CATALOG
    # ========================================================

    header(
        "REQUEST 1/2 — MARKET CATALOG"
    )


    market_payload = api_get(

        "markets",

        sportId=SPORT_ID
    )


    market_catalog_path = (

        RAW_DIRECTORY

        /

        f"markets_{batch_id}.json"
    )


    with open(
        market_catalog_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            market_payload,

            file,

            indent=2
        )


    market_catalog = build_market_catalog(
        market_payload
    )


    print()

    print(
        f"Market definitions loaded: "
        f"{len(market_catalog):,}"
    )


    print(
        f"Raw file:"
    )


    print(
        f"  {market_catalog_path}"
    )


    # Small pause to respect endpoint cooldowns.
    time.sleep(
        1.0
    )


    # ========================================================
    # REQUEST 2 — HARD ROCK NCAAF BOARD
    # ========================================================

    header(
        "REQUEST 2/2 — HARD ROCK NCAAF BOARD"
    )


    odds_payload = api_get(

        "odds-by-tournaments",

        tournamentIds=str(
            NCAAF_TOURNAMENT_ID
        ),

        bookmakers=BOOKMAKER,

        oddsFormat="american",

        language="en",

        verbosity=3
    )


    odds_path = (

        RAW_DIRECTORY

        /

        f"hardrock_ncaaf_{batch_id}.json"
    )


    with open(
        odds_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            odds_payload,

            file,

            indent=2
        )


    fixtures = extract_list(
        odds_payload
    )


    print()

    print(
        f"Fixtures returned: "
        f"{len(fixtures):,}"
    )


    print(
        f"Raw file:"
    )


    print(
        f"  {odds_path}"
    )


    # ========================================================
    # HARD ROCK COVERAGE
    # ========================================================

    hardrock_fixtures = []


    for fixture in fixtures:

        bookmaker_odds = fixture.get(
            "bookmakerOdds",
            {}
        )


        if BOOKMAKER in bookmaker_odds:

            hardrock_fixtures.append(
                fixture
            )


    header(
        "HARD ROCK NCAAF COVERAGE"
    )


    print(
        f"Fixtures returned:          "
        f"{len(fixtures):,}"
    )


    print(
        f"Fixtures with Hard Rock:    "
        f"{len(hardrock_fixtures):,}"
    )


    if len(
        hardrock_fixtures
    ) == 0:

        print()

        print(
            "RESULT: HARD ROCK NCAAF NOT PRESENT"
        )


        print()

        print(
            "The API key worked, but this request "
            "returned no Hard Rock NCAAF prices."
        )


        print()

        print(
            "Do NOT integrate OddsPapi into the "
            "production pipeline yet."
        )


        return


    print()

    print(
        "RESULT: HARD ROCK NCAAF DATA EXISTS"
    )


    # ========================================================
    # TARGET GAMES
    # ========================================================

    header(
        "TARGET GAME TEST"
    )


    targets_found = 0


    for (
        team_a,
        team_b
    ) in TARGET_GAMES:

        matches = [

            fixture

            for fixture
            in hardrock_fixtures

            if fixture_contains_teams(

                fixture,

                team_a,

                team_b
            )
        ]


        print()

        print(
            "#" * 110
        )


        print(
            f"TARGET: "
            f"{team_a} / {team_b}"
        )


        print(
            "#" * 110
        )


        if not matches:

            print()

            print(
                "Target game not found "
                "in current Hard Rock payload."
            )


            continue


        targets_found += 1


        for fixture in matches:

            display_fixture(

                fixture,

                market_catalog
            )


    # ========================================================
    # FALLBACK SAMPLE
    #
    # If today's three target games aren't all available,
    # inspect a few other Hard Rock NCAAF games anyway.
    # ========================================================

    if targets_found < len(
        TARGET_GAMES
    ):

        header(
            "OTHER HARD ROCK NCAAF SAMPLE"
        )


        sample_count = 0


        for fixture in hardrock_fixtures:

            if sample_count >= 5:

                break


            display_fixture(

                fixture,

                market_catalog
            )


            sample_count += 1


    # ========================================================
    # FINAL VERDICT
    # ========================================================

    header(
        "DIAGNOSTIC COMPLETE"
    )


    print(
        f"Hard Rock NCAAF fixtures: "
        f"{len(hardrock_fixtures):,}"
    )


    print(
        f"Target games found:       "
        f"{targets_found}/"
        f"{len(TARGET_GAMES)}"
    )


    print()

    print(
        "If the output above contains real "
        "moneyline, spread, and total prices, "
        "OddsPapi passes our first Hard Rock test."
    )


    print()

    print(
        "Do NOT make production changes yet."
    )


    print(
        "Next we will determine exactly how to "
        "resolve Hard Rock's headline spread from "
        "its alternate-line ladder."
    )


if __name__ == "__main__":

    main()