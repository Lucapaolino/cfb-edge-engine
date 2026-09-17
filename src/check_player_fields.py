import os
import requests
from collections import defaultdict
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

# This will store:
#
# year
#   -> category
#       -> set of stat fields
#
# Example:
# 2025
#   -> passing
#       -> C/ATT, YDS, AVG, TD, INT, QBR

fields_by_year = defaultdict(lambda: defaultdict(set))


print("DOWNLOADING PLAYER FIELD INFORMATION")
print("-" * 60)

for year in years:

    params = {
        "year": year,
        "team": team,
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

    print(f"{year}: {len(games)} games")

    for game in games:

        for team_data in game.get("teams", []):

            # Only inspect Alabama.
            # Ignore Alabama's opponents.
            if team_data.get("team") != team:
                continue

            for category in team_data.get("categories", []):

                category_name = category.get("name")

                for stat_type in category.get("types", []):

                    stat_name = stat_type.get("name")

                    if category_name and stat_name:
                        fields_by_year[year][category_name].add(
                            stat_name
                        )


# --------------------------------------------------
# DISPLAY FIELDS BY CATEGORY AND YEAR
# --------------------------------------------------

all_categories = set()

for year in years:
    all_categories.update(
        fields_by_year[year].keys()
    )


print("\n\nPLAYER STAT FIELD COVERAGE")
print("=" * 70)

for category in sorted(all_categories):

    print(f"\n{category.upper()}")
    print("-" * 70)

    # Find every stat field that appeared
    # in this category in any season.
    all_fields = set()

    for year in years:
        all_fields.update(
            fields_by_year[year].get(category, set())
        )

    for stat_field in sorted(all_fields):

        coverage = []

        for year in years:

            if stat_field in fields_by_year[year].get(
                category,
                set()
            ):
                coverage.append(f"{year}: YES")
            else:
                coverage.append(f"{year}: NO")

        print(
            f"{stat_field:<15} | "
            + " | ".join(coverage)
        )