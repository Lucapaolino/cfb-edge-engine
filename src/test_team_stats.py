import os
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
    "week": 1,
    "seasonType": "regular",
    "classification": "fbs"
}

response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30
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