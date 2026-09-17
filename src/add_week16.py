import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

url = "https://api.collegefootballdata.com/games/teams"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

params = {
    "year": 2025,
    "week": 16,
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

week_16_games = response.json()

print(f"Week 16 games returned: {len(week_16_games)}")

# Load what we already downloaded
with open("data/raw/team_stats_2025.json", "r") as file:
    existing_games = json.load(file)

# Combine old + new
combined_games = existing_games + week_16_games

# Deduplicate by game ID just to be safe
games_by_id = {
    game["id"]: game
    for game in combined_games
}

final_games = list(games_by_id.values())

with open("data/raw/team_stats_2025.json", "w") as file:
    json.dump(final_games, file, indent=2)

print(f"Total games now stored: {len(final_games)}")