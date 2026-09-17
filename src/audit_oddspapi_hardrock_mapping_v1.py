import json
from pathlib import Path


# ============================================================
# PURPOSE
#
# Inspect the already-downloaded OddsPapi responses and
# determine exactly how:
#
#   participant 1 / participant 2
#   outcome 1 / outcome 2
#   handicap sign
#
# map to Hard Rock's displayed sportsbook spread.
#
# NO API CALLS.
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "oddspapi"
)


TARGET_MARKET_IDS = {
    "141",
    "14446",
    "142140",

    "14344",   # Alabama/Kentucky main spread
    "14264",   # Ohio State/Texas main spread
    "14356",   # Tennessee/Georgia Tech main spread
}


TARGET_TEAMS = [
    ("Alabama", "Kentucky"),
    ("Ohio State", "Texas"),
    ("Tennessee", "Georgia Tech"),
]


def header(text):

    print()

    print("=" * 110)

    print(text)

    print("=" * 110)


def normalize(value):

    return (
        str(value or "")
        .lower()
        .replace("'", "")
        .strip()
    )


def find_latest(prefix):

    files = sorted(
        RAW_DIRECTORY.glob(
            f"{prefix}_*.json"
        )
    )


    if not files:

        raise RuntimeError(
            f"No files found for {prefix}"
        )


    return files[-1]


def extract_list(payload):

    if isinstance(payload, list):

        return payload


    if isinstance(payload, dict):

        for key in [
            "data",
            "fixtures",
            "events",
            "results"
        ]:

            value = payload.get(key)

            if isinstance(value, list):

                return value


    return []


def contains_teams(
    fixture,
    team_a,
    team_b
):

    p1 = normalize(
        fixture.get(
            "participant1Name"
        )
    )


    p2 = normalize(
        fixture.get(
            "participant2Name"
        )
    )


    a = normalize(team_a)

    b = normalize(team_b)


    return (
        (a in p1 or a in p2)
        and
        (b in p1 or b in p2)
    )


def main():

    markets_path = find_latest(
        "markets"
    )


    odds_path = find_latest(
        "hardrock_ncaaf"
    )


    print(
        "AUDIT ODDSPAPI HARD ROCK MAPPING — V1"
    )

    print("=" * 110)

    print(
        f"Market catalog:"
    )

    print(
        f"  {markets_path}"
    )

    print()

    print(
        f"Odds payload:"
    )

    print(
        f"  {odds_path}"
    )


    with open(
        markets_path,
        "r",
        encoding="utf-8"
    ) as file:

        markets_payload = json.load(
            file
        )


    with open(
        odds_path,
        "r",
        encoding="utf-8"
    ) as file:

        odds_payload = json.load(
            file
        )


    markets = extract_list(
        markets_payload
    )


    fixtures = extract_list(
        odds_payload
    )


    # ========================================================
    # MARKET DEFINITIONS
    # ========================================================

    header(
        "RELEVANT MARKET DEFINITIONS"
    )


    market_lookup = {}


    for market in markets:

        market_id = str(
            market.get(
                "marketId"
            )
        )


        market_lookup[
            market_id
        ] = market


        if market_id not in TARGET_MARKET_IDS:

            continue


        print()

        print(
            f"MARKET ID: {market_id}"
        )


        print(
            f"Name:      "
            f"{market.get('marketName')}"
        )


        print(
            f"Short:     "
            f"{market.get('marketNameShort')}"
        )


        print(
            f"Type:      "
            f"{market.get('marketType')}"
        )


        print(
            f"Period:    "
            f"{market.get('period')}"
        )


        print(
            f"Handicap:  "
            f"{market.get('handicap')}"
        )


        print(
            f"PlayerProp:"
            f" {market.get('playerProp')}"
        )


        print(
            "Outcomes:"
        )


        for outcome in market.get(
            "outcomes",
            []
        ):

            print(
                "  "
                f"id={outcome.get('outcomeId')} | "
                f"name={outcome.get('outcomeName')}"
            )


    # ========================================================
    # FIXTURE DETAILS
    # ========================================================

    for (
        team_a,
        team_b
    ) in TARGET_TEAMS:

        matches = [

            fixture

            for fixture in fixtures

            if contains_teams(
                fixture,
                team_a,
                team_b
            )
        ]


        header(
            f"{team_a} / {team_b}"
        )


        if not matches:

            print(
                "Fixture not found."
            )

            continue


        for fixture in matches:

            print(
                f"Fixture ID: "
                f"{fixture.get('fixtureId')}"
            )


            print(
                f"Start:      "
                f"{fixture.get('startTime')}"
            )


            print()

            print(
                "PARTICIPANTS"
            )

            print(
                "-" * 70
            )


            for key in sorted(
                fixture.keys()
            ):

                if (
                    "participant"
                    in key.lower()
                ):

                    print(
                        f"{key}: "
                        f"{fixture.get(key)}"
                    )


            hardrock = (

                fixture
                .get(
                    "bookmakerOdds",
                    {}
                )
                .get(
                    "hardrockbet"
                )
            )


            if not hardrock:

                print(
                    "No Hard Rock object."
                )

                continue


            print()

            print(
                "HARD ROCK BOOKMAKER METADATA"
            )

            print(
                "-" * 70
            )


            for key, value in hardrock.items():

                if key == "markets":

                    continue


                print(
                    f"{key}: {value}"
                )


            print()

            print(
                "MAIN-LINE MARKETS"
            )

            print(
                "-" * 70
            )


            for (
                market_id,
                market_data
            ) in hardrock.get(
                "markets",
                {}
            ).items():

                meta = market_lookup.get(
                    str(
                        market_id
                    ),
                    {}
                )


                main_entries = []


                for (
                    outcome_id,
                    outcome_data
                ) in market_data.get(
                    "outcomes",
                    {}
                ).items():

                    player_zero = (
                        outcome_data
                        .get(
                            "players",
                            {}
                        )
                        .get(
                            "0"
                        )
                    )


                    if not player_zero:

                        continue


                    if not player_zero.get(
                        "active",
                        False
                    ):

                        continue


                    if not player_zero.get(
                        "mainLine",
                        False
                    ):

                        continue


                    main_entries.append(
                        (
                            outcome_id,
                            player_zero
                        )
                    )


                if not main_entries:

                    continue


                print()

                print(
                    f"Market {market_id}"
                )


                print(
                    f"  name:     "
                    f"{meta.get('marketName')}"
                )


                print(
                    f"  short:    "
                    f"{meta.get('marketNameShort')}"
                )


                print(
                    f"  type:     "
                    f"{meta.get('marketType')}"
                )


                print(
                    f"  period:   "
                    f"{meta.get('period')}"
                )


                print(
                    f"  handicap: "
                    f"{meta.get('handicap')}"
                )


                outcome_names = {

                    str(
                        item.get(
                            "outcomeId"
                        )
                    ):
                        item.get(
                            "outcomeName"
                        )

                    for item
                    in meta.get(
                        "outcomes",
                        []
                    )
                }


                for (
                    outcome_id,
                    price
                ) in main_entries:

                    print(

                        f"  outcome "
                        f"{outcome_id}"

                        f" | name="
                        f"{outcome_names.get(str(outcome_id))}"

                        f" | American="
                        f"{price.get('priceAmerican')}"

                        f" | decimal="
                        f"{price.get('price')}"

                        f" | changedAt="
                        f"{price.get('changedAt')}"
                    )


    header(
        "AUDIT COMPLETE"
    )


    print(
        "No API requests were made."
    )


    print()

    print(
        "Next:"
    )


    print(
        "  Use the participant/outcome mapping above "
        "to build the production Hard Rock collector."
    )


if __name__ == "__main__":

    main()