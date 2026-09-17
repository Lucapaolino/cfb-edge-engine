import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

url = "https://api.collegefootballdata.com/games"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Seasons we want to inspect
years = [2015, 2018, 2020, 2022, 2024, 2025]

for year in years:

    params = {
        "year": year,
        "seasonType": "regular",
        "classification": "fbs"
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    games = response.json()

    if games:
        first_game = games[0]

        print(
            year,
            "| Games:", len(games),
            "| Pregame Elo:", first_game.get("homePregameElo"),
            "| Points:", first_game.get("homePoints")
        )
    else:
        print(year, "| No games returned")