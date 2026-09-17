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

searches = [
    {
        "year": 2018,
        "team": "Alabama"
    },
    {
        "year": 2019,
        "team": "Oklahoma"
    }
]

player_name = "Jalen Hurts"

print("PLAYER ID TRANSFER TEST")
print("-" * 60)

for search in searches:

    params = {
        "year": search["year"],
        "team": search["team"],
        "seasonType": "regular"
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    games = response.json()

    found_ids = set()

    for game in games:

        for team_data in game.get("teams", []):

            if team_data.get("team") != search["team"]:
                continue

            for category in team_data.get("categories", []):

                for stat_type in category.get("types", []):

                    for athlete in stat_type.get("athletes", []):

                        if athlete.get("name") == player_name:

                            found_ids.add(
                                athlete.get("id")
                            )

    print(
        f"{search['year']} | "
        f"{search['team']:<10} | "
        f"{player_name} | "
        f"IDs: {found_ids}"
    )