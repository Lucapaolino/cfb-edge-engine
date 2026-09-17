import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

url = "https://api.collegefootballdata.com/lines"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Start small: 2025 Week 1
params = {
    "year": 2025,
    "week": 1,
    "seasonType": "regular"
}

response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=60
)

print("Status:", response.status_code)

response.raise_for_status()

games = response.json()

print("Games with betting data returned:", len(games))

if games:

    print("\nAvailable game fields:")
    print(games[0].keys())

    print("\nFIRST GAME")
    print("-" * 70)
    print(games[0])

    print("\nFIRST 5 GAMES - SUMMARY")
    print("-" * 70)

    for game in games[:5]:

        print(
            f"\n{game.get('awayTeam')} @ "
            f"{game.get('homeTeam')}"
        )

        print(
            f"Game ID: {game.get('id')}"
        )

        lines = game.get("lines", [])

        print(
            f"Number of line providers: {len(lines)}"
        )

        for line in lines[:3]:
            print(line)