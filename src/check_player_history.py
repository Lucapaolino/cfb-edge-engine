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

years = [
    2018,
    2020,
    2022,
    2023,
    2024,
    2025
]

team = "Alabama"

print("PLAYER DATA HISTORY CHECK")
print("-" * 70)

for year in years:

    params = {
        "year": year,
        "team": team,
        "seasonType": "regular"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        games = response.json()

        category_names = set()
        player_ids = set()

        for game in games:

            for team_data in game.get("teams", []):

                for category in team_data.get("categories", []):

                    category_names.add(category.get("name"))

                    for stat_type in category.get("types", []):

                        for athlete in stat_type.get("athletes", []):

                            player_id = athlete.get("id")

                            if player_id:
                                player_ids.add(player_id)

        print(
            f"{year} | "
            f"Games: {len(games):>2} | "
            f"Categories: {len(category_names):>2} | "
            f"Players: {len(player_ids):>3}"
        )

        print(
            "       Categories:",
            ", ".join(sorted(category_names))
        )

    except requests.exceptions.RequestException as error:

        print(f"{year} | ERROR: {error}")