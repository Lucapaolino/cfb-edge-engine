import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

url = "https://api.collegefootballdata.com/ratings/elo"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

years = range(2018, 2026)

print("HISTORICAL ELO AUDIT")
print("=" * 70)

for year in years:

    # Check Week 1 and a later regular-season week
    weeks = [1, 8]

    for week in weeks:

        params = {
            "year": year,
            "week": week,
            "seasonType": "regular"
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        ratings = response.json()

        ratings_with_elo = [
            rating
            for rating in ratings
            if rating.get("elo") is not None
        ]

        print(
            f"{year} Week {week:<2} | "
            f"Teams returned: {len(ratings):>3} | "
            f"Teams with Elo: {len(ratings_with_elo):>3}"
        )

        # Show Alabama as an easy spot check
        for rating in ratings:

            if rating.get("team") == "Alabama":

                print(
                    f"    Alabama Elo: "
                    f"{rating.get('elo')}"
                )

                break