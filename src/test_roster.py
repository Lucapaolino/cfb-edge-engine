import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

url = "https://api.collegefootballdata.com/roster"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

params = {
    "year": 2025,
    "team": "Alabama"
}

response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=60
)

print("Status:", response.status_code)

response.raise_for_status()

players = response.json()

print("Players returned:", len(players))

if players:

    print("\nAvailable player fields:")
    print(players[0].keys())

    print("\nFIRST 5 PLAYERS")
    print("-" * 70)

    for player in players[:5]:

        print(
            f"ID: {player.get('id')} | "
            f"{player.get('firstName')} {player.get('lastName')} | "
            f"Position: {player.get('position')} | "
            f"Year: {player.get('year')} | "
            f"Jersey: {player.get('jersey')} | "
            f"Recruit IDs: {player.get('recruitIds')}"
        )