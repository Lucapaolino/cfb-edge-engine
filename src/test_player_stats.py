import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

url = "https://api.collegefootballdata.com/games/players"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Small test request:
# Ohio's Week 1 player stats only
params = {
    "year": 2025,
    "week": 1,
    "team": "Ohio",
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

print("Games returned:", len(games))

if games:
    print("\nFirst game's available fields:")
    print(games[0].keys())

    print("\nFirst game:")
    print(games[0])