import os
import json
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

url = "https://api.collegefootballdata.com/games/teams"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

all_games = []

# Regular-season CFB can extend through Week 15
for week in range(1, 16):

    params = {
        "year": 2025,
        "week": week,
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

    print(f"Week {week}: {len(games)} games")

    all_games.extend(games)

print(f"\nTotal games downloaded: {len(all_games)}")

output_file = "data/raw/team_stats_2025.json"

with open(output_file, "w") as file:
    json.dump(all_games, file, indent=2)

print(f"Saved to {output_file}")

# Measure stat coverage
stat_counts = Counter()
team_records = 0

for game in all_games:

    for team in game.get("teams", []):

        team_records += 1

        categories = {
            stat["category"]
            for stat in team.get("stats", [])
        }

        stat_counts.update(categories)

print(f"\nTeam-game records: {team_records}")

print("\nSTAT COVERAGE")
print("-" * 50)

for category, count in stat_counts.most_common():

    coverage = (count / team_records) * 100

    print(
        f"{category:<25}"
        f"{count:>5} / {team_records}"
        f"   {coverage:>6.2f}%"
    )