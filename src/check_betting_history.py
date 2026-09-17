import os
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

years = range(2018, 2026)

# We are sampling Week 1 from every season.
week = 1

print("HISTORICAL BETTING DATA AUDIT")
print("=" * 80)

for year in years:

    # --------------------------------------------------
    # GET WEEK 1 GAMES
    # --------------------------------------------------

    game_response = requests.get(
        "https://api.collegefootballdata.com/games",
        headers=headers,
        params={
            "year": year,
            "week": week,
            "seasonType": "regular",
            "classification": "fbs"
        },
        timeout=60
    )

    game_response.raise_for_status()

    games = game_response.json()

    # Restrict our audit to FBS vs FBS games.
    fbs_games = []

    for game in games:

        if (
            game.get("homeClassification") == "fbs"
            and
            game.get("awayClassification") == "fbs"
        ):
            fbs_games.append(game)

    fbs_game_ids = {
        game["id"]
        for game in fbs_games
    }


    # --------------------------------------------------
    # GET WEEK 1 BETTING LINES
    # --------------------------------------------------

    line_response = requests.get(
        "https://api.collegefootballdata.com/lines",
        headers=headers,
        params={
            "year": year,
            "week": week,
            "seasonType": "regular"
        },
        timeout=60
    )

    line_response.raise_for_status()

    betting_games = line_response.json()

    lines_by_game = {
        game["id"]: game
        for game in betting_games
    }


    # --------------------------------------------------
    # MEASURE GAME COVERAGE
    # --------------------------------------------------

    games_with_lines = 0

    provider_counts = Counter()

    provider_records = 0

    spread_count = 0
    spread_open_count = 0

    total_count = 0
    total_open_count = 0

    home_moneyline_count = 0
    away_moneyline_count = 0


    for game_id in fbs_game_ids:

        betting_game = lines_by_game.get(game_id)

        if not betting_game:
            continue

        lines = betting_game.get("lines", [])

        if not lines:
            continue

        games_with_lines += 1

        for line in lines:

            provider_records += 1

            provider = line.get("provider")

            if provider:
                provider_counts[provider] += 1

            if line.get("spread") is not None:
                spread_count += 1

            if line.get("spreadOpen") is not None:
                spread_open_count += 1

            if line.get("overUnder") is not None:
                total_count += 1

            if line.get("overUnderOpen") is not None:
                total_open_count += 1

            if line.get("homeMoneyline") is not None:
                home_moneyline_count += 1

            if line.get("awayMoneyline") is not None:
                away_moneyline_count += 1


    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    if fbs_games:

        game_coverage = (
            games_with_lines /
            len(fbs_games)
        ) * 100

    else:
        game_coverage = 0


    print(f"\n{year}")
    print("-" * 80)

    print(
        f"FBS vs FBS Week 1 games: "
        f"{len(fbs_games)}"
    )

    print(
        f"Games with betting lines: "
        f"{games_with_lines} / {len(fbs_games)} "
        f"({game_coverage:.2f}%)"
    )

    print(
        f"Sportsbook line records: "
        f"{provider_records}"
    )

    if provider_records:

        print(
            f"Spread coverage: "
            f"{spread_count / provider_records * 100:.2f}%"
        )

        print(
            f"Opening spread coverage: "
            f"{spread_open_count / provider_records * 100:.2f}%"
        )

        print(
            f"Total coverage: "
            f"{total_count / provider_records * 100:.2f}%"
        )

        print(
            f"Opening total coverage: "
            f"{total_open_count / provider_records * 100:.2f}%"
        )

        print(
            f"Home moneyline coverage: "
            f"{home_moneyline_count / provider_records * 100:.2f}%"
        )

        print(
            f"Away moneyline coverage: "
            f"{away_moneyline_count / provider_records * 100:.2f}%"
        )

    print(
        "Providers:",
        ", ".join(
            sorted(provider_counts.keys())
        )
        if provider_counts
        else "None"
    )