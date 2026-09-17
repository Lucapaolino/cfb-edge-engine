import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

team = "Alabama"
year = 2025


# --------------------------------------------------
# GET 2025 ALABAMA ROSTER
# --------------------------------------------------

roster_response = requests.get(
    "https://api.collegefootballdata.com/roster",
    headers=headers,
    params={
        "year": year,
        "team": team
    },
    timeout=60
)

roster_response.raise_for_status()

roster = roster_response.json()

print(f"Roster players: {len(roster)}")


# --------------------------------------------------
# GET ALABAMA RECRUITING HISTORY
# --------------------------------------------------

recruit_response = requests.get(
    "https://api.collegefootballdata.com/recruiting/players",
    headers=headers,
    params={
        "team": team
    },
    timeout=60
)

recruit_response.raise_for_status()

recruits = recruit_response.json()

print(f"Recruiting records: {len(recruits)}")


# Create a lookup table:
#
# recruit ID -> recruiting record

recruits_by_id = {
    recruit["id"]: recruit
    for recruit in recruits
}


# --------------------------------------------------
# MATCH ROSTER PLAYERS TO RECRUITING RECORDS
# --------------------------------------------------

players_with_recruit_ids = 0
players_matched = 0

matches = []

for player in roster:

    recruit_ids = player.get("recruitIds") or []

    if recruit_ids:
        players_with_recruit_ids += 1

    for recruit_id in recruit_ids:

        recruit = recruits_by_id.get(recruit_id)

        if recruit:

            players_matched += 1

            matches.append({
                "player_id": player.get("id"),
                "player": (
                    f"{player.get('firstName')} "
                    f"{player.get('lastName')}"
                ),
                "position": player.get("position"),
                "recruit_id": recruit_id,
                "recruit_year": recruit.get("year"),
                "stars": recruit.get("stars"),
                "rating": recruit.get("rating"),
                "ranking": recruit.get("ranking"),
                "athlete_id": recruit.get("athleteId")
            })

            break


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nRECRUITING LINKAGE")
print("-" * 60)

print(
    f"Players with recruit IDs: "
    f"{players_with_recruit_ids} / {len(roster)}"
)

print(
    f"Players successfully matched: "
    f"{players_matched} / {len(roster)}"
)

if roster:

    coverage = (
        players_matched /
        len(roster)
    ) * 100

    print(
        f"Overall roster recruiting coverage: "
        f"{coverage:.2f}%"
    )


print("\nFIRST 10 MATCHES")
print("-" * 60)

for match in matches[:10]:

    print(
        f"{match['player']} | "
        f"{match['position']} | "
        f"Recruit year: {match['recruit_year']} | "
        f"Stars: {match['stars']} | "
        f"Rating: {match['rating']} | "
        f"National rank: {match['ranking']} | "
        f"CFBD athlete ID: {match['athlete_id']}"
    )