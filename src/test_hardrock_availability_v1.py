import os

import requests

from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

API_BASE_URL = (
    "https://api.the-odds-api.com/v4/sports"
)

BOOKMAKER = "hardrockbet_fl"

SPORTS = [
    (
        "NCAAF",
        "americanfootball_ncaaf"
    ),
    (
        "NFL",
        "americanfootball_nfl"
    )
]

MARKET = "h2h"

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
# START
# ============================================================

print(
    "HARD ROCK BET FL AVAILABILITY TEST — V1"
)

print(
    "=" * 100
)


print(
    f"Bookmaker key: {BOOKMAKER}"
)


print(
    f"Market tested: {MARKET}"
)


print()

print(
    "Each sport request should cost approximately 1 credit."
)


# ============================================================
# TEST EACH SPORT
# ============================================================

for sport_name, sport_key in SPORTS:

    print()

    print(
        "=" * 100
    )


    print(
        sport_name
    )


    print(
        "=" * 100
    )


    url = (
        f"{API_BASE_URL}/"
        f"{sport_key}/odds"
    )


    response = requests.get(

        url,

        params={
            "apiKey":
                api_key,

            "bookmakers":
                BOOKMAKER,

            "markets":
                MARKET,

            "oddsFormat":
                "american",

            "dateFormat":
                "iso"
        },

        timeout=REQUEST_TIMEOUT
    )


    print(
        f"HTTP status:       "
        f"{response.status_code}"
    )


    print(
        f"Credits this call: "
        f"{response.headers.get('x-requests-last')}"
    )


    print(
        f"Credits used:      "
        f"{response.headers.get('x-requests-used')}"
    )


    print(
        f"Credits remaining: "
        f"{response.headers.get('x-requests-remaining')}"
    )


    if response.status_code != 200:

        print()

        print(
            "ERROR RESPONSE"
        )


        print(
            response.text[
                :1500
            ]
        )


        continue


    payload = response.json()


    print(
        f"Events returned:   "
        f"{len(payload):,}"
    )


    # ========================================================
    # COUNT ACTUAL HARD ROCK BOOKMAKER OBJECTS
    # ========================================================

    hard_rock_events = []


    for event in payload:

        bookmakers = event.get(
            "bookmakers",
            []
        )


        for bookmaker in bookmakers:

            if (
                bookmaker.get(
                    "key"
                )
                ==
                BOOKMAKER
            ):

                hard_rock_events.append(
                    {
                        "commence_time":
                            event.get(
                                "commence_time"
                            ),

                        "away_team":
                            event.get(
                                "away_team"
                            ),

                        "home_team":
                            event.get(
                                "home_team"
                            ),

                        "last_update":
                            bookmaker.get(
                                "last_update"
                            )
                    }
                )


    print(
        f"Hard Rock events:  "
        f"{len(hard_rock_events):,}"
    )


    # ========================================================
    # SAMPLE
    # ========================================================

    if len(
        hard_rock_events
    ) > 0:

        print()

        print(
            "SAMPLE HARD ROCK EVENTS"
        )


        print(
            "-" * 100
        )


        for event in hard_rock_events[
            :10
        ]:

            print(

                f"{event['commence_time']} | "
                f"{event['away_team']} @ "
                f"{event['home_team']} | "
                f"updated "
                f"{event['last_update']}"
            )


    else:

        print()

        print(
            "No Hard Rock FL prices returned."
        )


print()

print(
    "=" * 100
)

print(
    "TEST COMPLETE"
)

print(
    "=" * 100
)