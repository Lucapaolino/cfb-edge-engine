import json
from collections import defaultdict

# Load our locally saved raw files
with open("data/raw/games_2025.json", "r") as file:
    games = json.load(file)

with open("data/raw/team_stats_2025.json", "r") as file:
    team_stats = json.load(file)

# --------------------------------------------------
# CHECK 1:
# Which game exists in /games but not /games/teams?
# --------------------------------------------------

game_ids = {game["id"] for game in games}
team_stat_ids = {game["id"] for game in team_stats}

missing_game_ids = game_ids - team_stat_ids

print("GAME ID CHECK")
print("-" * 50)

print("Games file:", len(game_ids))
print("Team stats file:", len(team_stat_ids))
print("Missing game IDs:", missing_game_ids)

for game in games:

    if game["id"] in missing_game_ids:

        print("\nMissing game:")
        print(
            game.get("awayTeam"),
            "@",
            game.get("homeTeam"),
            "| Week:",
            game.get("week"),
            "| Date:",
            game.get("startDate")
        )


# --------------------------------------------------
# CHECK 2:
# Show examples of teams where categories are absent
# --------------------------------------------------

categories_to_check = [
    "kickReturns",
    "puntReturns",
    "fumblesRecovered",
    "passesIntercepted"
]

missing_examples = defaultdict(list)

for game in team_stats:

    for team in game.get("teams", []):

        available_categories = {
            stat["category"]
            for stat in team.get("stats", [])
        }

        for category in categories_to_check:

            if category not in available_categories:

                if len(missing_examples[category]) < 5:

                    missing_examples[category].append(
                        (
                            game["id"],
                            team["team"]
                        )
                    )

print("\n\nMISSING CATEGORY EXAMPLES")
print("-" * 50)

for category in categories_to_check:

    print(f"\n{category}:")

    for game_id, team in missing_examples[category]:
        print(f"  Game {game_id} | {team}")