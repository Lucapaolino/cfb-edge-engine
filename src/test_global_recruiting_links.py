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
roster_year = 2025


# --------------------------------------------------
# GET 2025 ALABAMA ROSTER
# --------------------------------------------------

response = requests.get(
    "https://api.collegefootballdata.com/roster",
    headers=headers,
    params={
        "year": roster_year,
        "team": team
    },
    timeout=60
)

response.raise_for_status()

roster = response.json()

print(f"Roster players: {len(roster)}")


# --------------------------------------------------
# COLLECT ALL RECRUIT IDS FROM THE ROSTER
# --------------------------------------------------

needed_recruit_ids = set()

for player in roster:

    recruit_ids = player.get("recruitIds") or []

    for recruit_id in recruit_ids:
        needed_recruit_ids.add(recruit_id)

print(
    f"Unique recruit IDs attached to roster: "
    f"{len(needed_recruit_ids)}"
)


# --------------------------------------------------
# DOWNLOAD GLOBAL RECRUITING DATA
# --------------------------------------------------

recruiting_lookup = {}

years = range(2019, 2026)

classifications = [
    "HighSchool",
    "JUCO",
    "PrepSchool"
]

for year in years:

    for classification in classifications:

        params = {
            "year": year,
            "classification": classification
        }

        response = requests.get(
            "https://api.collegefootballdata.com/recruiting/players",
            headers=headers,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        recruits = response.json()

        print(
            f"{year} {classification:<10}: "
            f"{len(recruits)} recruiting records"
        )

        for recruit in recruits:

            recruit_id = recruit.get("id")

            if recruit_id in needed_recruit_ids:
                recruiting_lookup[recruit_id] = recruit


# --------------------------------------------------
# MATCH ROSTER PLAYERS
# --------------------------------------------------

players_with_recruit_ids = 0
players_matched = 0
matches = []
unmatched = []

for player in roster:

    recruit_ids = player.get("recruitIds") or []

    player_name = (
        f"{player.get('firstName')} "
        f"{player.get('lastName')}"
    )

    if recruit_ids:
        players_with_recruit_ids += 1

    matched_recruit = None

    for recruit_id in recruit_ids:

        if recruit_id in recruiting_lookup:
            matched_recruit = recruiting_lookup[recruit_id]
            break

    if matched_recruit:

        players_matched += 1

        matches.append({
            "player": player_name,
            "position": player.get("position"),
            "college_team": player.get("team"),
            "recruit_year": matched_recruit.get("year"),
            "recruit_type": matched_recruit.get("recruitType"),
            "original_school": matched_recruit.get("school"),
            "committed_to": matched_recruit.get("committedTo"),
            "stars": matched_recruit.get("stars"),
            "rating": matched_recruit.get("rating"),
            "ranking": matched_recruit.get("ranking")
        })

    elif recruit_ids:

        unmatched.append({
            "player": player_name,
            "position": player.get("position"),
            "recruit_ids": recruit_ids
        })


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nGLOBAL RECRUITING LINKAGE")
print("-" * 70)

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

if players_with_recruit_ids:

    id_coverage = (
        players_matched /
        players_with_recruit_ids
    ) * 100

    print(
        f"Coverage among players with recruit IDs: "
        f"{id_coverage:.2f}%"
    )


print("\nFIRST 15 MATCHES")
print("-" * 70)

for match in matches[:15]:

    print(
        f"{match['player']} | "
        f"{match['position']} | "
        f"{match['recruit_type']} | "
        f"{match['recruit_year']} | "
        f"{match['stars']} stars | "
        f"Rating: {match['rating']} | "
        f"Rank: {match['ranking']} | "
        f"Committed to: {match['committed_to']}"
    )


print("\nUNMATCHED PLAYERS WHO HAVE RECRUIT IDS")
print("-" * 70)

for player in unmatched:

    print(
        f"{player['player']} | "
        f"{player['position']} | "
        f"IDs: {player['recruit_ids']}"
    )