import os
import json
import requests
from collections import defaultdict
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

# CFBD play-by-play endpoint
url = "https://api.collegefootballdata.com/plays"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Request 2025 Week 1 FBS regular-season plays
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

# Stop if CFBD returns an error
response.raise_for_status()

# Convert JSON response into Python data
plays = response.json()

print("Plays returned:", len(plays))

# Inspect the structure of the data
if plays:
    print("\nAvailable fields:")
    print(plays[0].keys())

    print("\nFirst play:")
    print(plays[0])


# --------------------------------------------------
# OVERALL PPA COVERAGE
# --------------------------------------------------

ppa_count = sum(
    1 for play in plays
    if play.get("ppa") is not None
)

if plays:
    ppa_coverage = (ppa_count / len(plays)) * 100
else:
    ppa_coverage = 0

print(
    f"\nOverall PPA coverage: "
    f"{ppa_count} / {len(plays)} "
    f"({ppa_coverage:.2f}%)"
)


# --------------------------------------------------
# SAVE RAW WEEK 1 DATA LOCALLY
# --------------------------------------------------

output_file = "data/raw/plays_2025_week1.json"

with open(output_file, "w") as file:
    json.dump(plays, file, indent=2)

print(f"\nSaved raw play data to {output_file}")


# --------------------------------------------------
# PPA COVERAGE BY PLAY TYPE
# --------------------------------------------------

play_type_counts = defaultdict(int)
play_type_ppa_counts = defaultdict(int)

for play in plays:

    play_type = play.get("playType")

    if play_type is None:
        play_type = "Unknown"

    play_type_counts[play_type] += 1

    if play.get("ppa") is not None:
        play_type_ppa_counts[play_type] += 1


print("\nPPA COVERAGE BY PLAY TYPE")
print("-" * 70)

for play_type, total in sorted(
    play_type_counts.items(),
    key=lambda item: item[1],
    reverse=True
):

    ppa_total = play_type_ppa_counts[play_type]

    coverage = (ppa_total / total) * 100

    print(
        f"{play_type:<40}"
        f"{ppa_total:>5} / {total:<5}"
        f"{coverage:>7.2f}%"
    )