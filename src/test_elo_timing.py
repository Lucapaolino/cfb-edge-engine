import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

team = "Alabama"
year = 2025


# --------------------------------------------------
# GET ALABAMA'S 2025 REGULAR-SEASON GAMES
# --------------------------------------------------

response = requests.get(
    "https://api.collegefootballdata.com/games",
    headers=headers,
    params={
        "year": year,
        "team": team,
        "seasonType": "regular"
    },
    timeout=60
)

response.raise_for_status()

games = response.json()


# Sort games by week
games = sorted(
    games,
    key=lambda game: game.get("week", 0)
)


print("ELO TIMING TEST")
print("=" * 80)


# --------------------------------------------------
# CHECK FIRST 5 ALABAMA GAMES
# --------------------------------------------------

games_checked = 0

for game in games:

    if games_checked >= 5:
        break

    week = game.get("week")

    # Determine whether Alabama was home or away
    if game.get("homeTeam") == team:

        opponent = game.get("awayTeam")

        pregame_elo = game.get("homePregameElo")
        postgame_elo = game.get("homePostgameElo")

    elif game.get("awayTeam") == team:

        opponent = game.get("homeTeam")

        pregame_elo = game.get("awayPregameElo")
        postgame_elo = game.get("awayPostgameElo")

    else:
        continue


    # --------------------------------------------------
    # GET /ratings/elo FOR THAT SAME WEEK
    # --------------------------------------------------

    rating_response = requests.get(
        "https://api.collegefootballdata.com/ratings/elo",
        headers=headers,
        params={
            "year": year,
            "week": week,
            "seasonType": "regular",
            "team": team
        },
        timeout=60
    )

    rating_response.raise_for_status()

    ratings = rating_response.json()

    rating_elo = None

    if ratings:
        rating_elo = ratings[0].get("elo")


    # --------------------------------------------------
    # DISPLAY COMPARISON
    # --------------------------------------------------

    print(f"\nWeek {week}: Alabama vs {opponent}")
    print("-" * 80)

    print(f"Pregame Elo:        {pregame_elo}")
    print(f"Postgame Elo:       {postgame_elo}")
    print(f"/ratings/elo value: {rating_elo}")

    if rating_elo == pregame_elo:
        print("MATCH: ratings endpoint appears to be PRE-GAME for this week")

    elif rating_elo == postgame_elo:
        print("MATCH: ratings endpoint appears to be POST-GAME for this week")

    else:
        print("NO EXACT MATCH - needs investigation")

    games_checked += 1