import json
import sqlite3


# --------------------------------------------------
# FILE LOCATIONS
# --------------------------------------------------

json_file = "data/raw/team_stats_2025.json"
database_file = "data/cfb.db"


# --------------------------------------------------
# LOAD RAW TEAM STATS
# --------------------------------------------------

with open(json_file, "r") as file:
    games = json.load(file)

print(f"Games loaded from team stats JSON: {len(games)}")


# --------------------------------------------------
# CONNECT TO DATABASE
# --------------------------------------------------

connection = sqlite3.connect(database_file)

# Enforce relationships between tables
connection.execute("PRAGMA foreign_keys = ON")

cursor = connection.cursor()


# --------------------------------------------------
# CREATE FLEXIBLE STAT-VALUE TABLE
#
# Example:
#
# game_id | team   | category       | stat_value
# ------------------------------------------------
# 12345   | Ohio   | rushingYards   | 201
# 12345   | Ohio   | thirdDownEff   | 8-14
#
# We preserve the raw text exactly as CFBD returned it.
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS team_game_stat_values (

    game_id INTEGER NOT NULL,
    team TEXT NOT NULL,
    category TEXT NOT NULL,
    stat_value TEXT,

    PRIMARY KEY (
        game_id,
        team,
        category
    ),

    FOREIGN KEY (game_id)
        REFERENCES games(game_id)
)
""")


# --------------------------------------------------
# IMPORT DATA
# --------------------------------------------------

team_records_processed = 0
stat_values_processed = 0

for game in games:

    game_id = game.get("id")

    for team in game.get("teams", []):

        team_name = team.get("team")


        # --------------------------------------------------
        # BASIC TEAM/GAME RECORD
        # --------------------------------------------------

        cursor.execute(
            """
            INSERT OR REPLACE INTO team_game_stats (

                game_id,
                team_id,
                team,
                conference,
                home_away,
                points

            )

            VALUES (?, ?, ?, ?, ?, ?)
            """,

            (
                game_id,
                team.get("teamId"),
                team_name,
                team.get("conference"),
                team.get("homeAway"),
                team.get("points")
            )
        )

        team_records_processed += 1


        # --------------------------------------------------
        # INDIVIDUAL STAT CATEGORIES
        # --------------------------------------------------

        for stat in team.get("stats", []):

            category = stat.get("category")
            stat_value = stat.get("stat")

            if category is None:
                continue

            cursor.execute(
                """
                INSERT OR REPLACE INTO team_game_stat_values (

                    game_id,
                    team,
                    category,
                    stat_value

                )

                VALUES (?, ?, ?, ?)
                """,

                (
                    game_id,
                    team_name,
                    category,
                    stat_value
                )
            )

            stat_values_processed += 1


# --------------------------------------------------
# SAVE EVERYTHING
# --------------------------------------------------

connection.commit()


# --------------------------------------------------
# VERIFY COUNTS
# --------------------------------------------------

cursor.execute("""
SELECT COUNT(*)
FROM team_game_stats
""")

team_record_count = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(*)
FROM team_game_stat_values
""")

stat_value_count = cursor.fetchone()[0]


print(f"\nTeam-game records processed: {team_records_processed}")
print(f"Team-game records stored:    {team_record_count}")

print(f"\nStat values processed:       {stat_values_processed}")
print(f"Stat values stored:          {stat_value_count}")


# --------------------------------------------------
# SHOW ONE EXAMPLE
# --------------------------------------------------

cursor.execute("""
SELECT
    game_id,
    team,
    category,
    stat_value

FROM team_game_stat_values

WHERE team = 'Alabama'

ORDER BY game_id, category

LIMIT 15
""")

sample_stats = cursor.fetchall()


print("\nALABAMA SAMPLE STATS")
print("-" * 70)

for row in sample_stats:

    game_id = row[0]
    team = row[1]
    category = row[2]
    stat_value = row[3]

    print(
        f"Game {game_id} | "
        f"{team:<12} | "
        f"{category:<25} | "
        f"{stat_value}"
    )


# --------------------------------------------------
# CLOSE DATABASE
# --------------------------------------------------

connection.close()