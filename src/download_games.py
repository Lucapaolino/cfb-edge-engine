import os
import json
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

params = {
    "year": 2025,
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

output_file = "data/raw/games_2025.json"

with open(output_file, "w") as file:
    json.dump(games, file, indent=2)

print(f"Downloaded {len(games)} games.")
print(f"Saved to {output_file}")