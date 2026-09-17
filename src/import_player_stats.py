import os
import re
import json
import time
import sqlite3
import requests
from dotenv import load_dotenv


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

year = 2025

database_file = "data/cfb.db"
raw_directory = f"data/raw/player_stats_{year}"

os.makedirs(raw_directory, exist_ok=True)

headers = {
    "Authorization": f"Bearer {API_KEY}"
}


# --------------------------------------------------
# CONNECT TO DATABASE
# --------------------------------------------------

connection = sqlite3.connect(database_file)

connection.execute(
    "PRAGMA foreign_keys = ON"
)

cursor = connection.cursor()


# --------------------------------------------------
# CREATE PLAYER GAME STAT TABLE
#
# One row represents one stat for one player
# in one game.
#
# Example:
#
# game_id | team    | player | category | stat_type | value
# ----------------------------------------------------------
# 12345   | Alabama | QB     | passing  | YDS       | 280
# 12345   | Alabama | QB     | passing  | TD        | 3
# --------------------------------------------------
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS player_game_stat_values (

    game_id INTEGER NOT NULL,

    team TEXT NOT NULL,

    athlete_id TEXT NOT NULL,
    athlete_name TEXT,

    category TEXT NOT NULL,
    stat_type TEXT NOT NULL,

    stat_value TEXT,

    PRIMARY KEY (
        game_id,
        team,
        athlete_id,
        category,
        stat_type
    ),

    FOREIGN KEY (game_id)
        REFERENCES games(game_id)
)
""")

connection.commit()


# --------------------------------------------------
# GET ALL FBS TEAMS FROM OUR 2025 DATABASE
#
# We are using our own database instead of making
# another API request just to get the team list.
# --------------------------------------------------

cursor.execute("""
SELECT team
FROM (

    SELECT home_team AS team
    FROM games
    WHERE
        season = ?
        AND home_classification = 'fbs'

    UNION

    SELECT away_team AS team
    FROM games
    WHERE
        season = ?
        AND away_classification = 'fbs'
)

ORDER BY team
""", (year, year))

fbs_teams = [
    row[0]
    for row in cursor.fetchall()
]

print(
    f"FBS teams found in database: "
    f"{len(fbs_teams)}"
)


# --------------------------------------------------
# GET VALID GAME IDS
# --------------------------------------------------

cursor.execute("""
SELECT game_id
FROM games
WHERE season = ?
""", (year,))

valid_game_ids = {
    row[0]
    for row in cursor.fetchall()
}


# --------------------------------------------------
# SAFE FILE NAME
#
# Converts something like:
#
# Hawai'i
#
# into:
#
# Hawai_i
# --------------------------------------------------

def safe_filename(team_name):

    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        team_name
    ).strip("_")


# --------------------------------------------------
# DOWNLOAD FUNCTION WITH RETRIES
# --------------------------------------------------

def download_team_player_stats(team_name):

    for attempt in range(1, 4):

        try:

            response = requests.get(
                "https://api.collegefootballdata.com/games/players",
                headers=headers,
                params={
                    "year": year,
                    "team": team_name,
                    "seasonType": "regular"
                },
                timeout=120
            )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as error:

            print(
                f"  Attempt {attempt} failed: "
                f"{error}"
            )

            if attempt < 3:
                time.sleep(2)

    return None


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

api_calls = 0
cached_teams = 0
failed_teams = []

player_stat_rows_processed = 0


# --------------------------------------------------
# PROCESS EACH FBS TEAM
# --------------------------------------------------

for team_number, team in enumerate(
    fbs_teams,
    start=1
):

    print(
        f"\n[{team_number}/{len(fbs_teams)}] "
        f"{team}"
    )

    filename = safe_filename(team)

    raw_file = (
        f"{raw_directory}/"
        f"{filename}.json"
    )


    # --------------------------------------------------
    # USE CACHED FILE IF AVAILABLE
    # --------------------------------------------------

    if os.path.exists(raw_file):

        print("  Using cached file")

        with open(raw_file, "r") as file:
            games = json.load(file)

        cached_teams += 1


    # --------------------------------------------------
    # OTHERWISE DOWNLOAD FROM CFBD
    # --------------------------------------------------

    else:

        print("  Downloading from CFBD...")

        games = download_team_player_stats(
            team
        )

        if games is None:

            print(
                "  FAILED after 3 attempts"
            )

            failed_teams.append(team)

            continue

        api_calls += 1

        with open(raw_file, "w") as file:

            json.dump(
                games,
                file,
                indent=2
            )

        # Small pause between successful requests
        time.sleep(0.25)


    print(
        f"  Games returned: "
        f"{len(games)}"
    )


    # --------------------------------------------------
    # EXTRACT ONLY THIS TEAM
    #
    # CFBD gives us both teams in each game's
    # player box score. We only store the team
    # we specifically requested so that we do
    # not create duplicate data.
    # --------------------------------------------------

    team_rows = 0

    for game in games:

        game_id = game.get("id")

        if game_id not in valid_game_ids:
            continue

        for team_data in game.get(
            "teams",
            []
        ):

            if team_data.get("team") != team:
                continue

            for category in team_data.get(
                "categories",
                []
            ):

                category_name = category.get(
                    "name"
                )

                if not category_name:
                    continue

                for stat_type in category.get(
                    "types",
                    []
                ):

                    stat_type_name = stat_type.get(
                        "name"
                    )

                    if not stat_type_name:
                        continue

                    for athlete in stat_type.get(
                        "athletes",
                        []
                    ):

                        athlete_id = athlete.get(
                            "id"
                        )

                        if athlete_id is None:
                            continue

                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO
                            player_game_stat_values (

                                game_id,

                                team,

                                athlete_id,
                                athlete_name,

                                category,
                                stat_type,

                                stat_value

                            )

                            VALUES (
                                ?, ?, ?, ?, ?, ?, ?
                            )
                            """,

                            (
                                game_id,

                                team,

                                str(athlete_id),
                                athlete.get("name"),

                                category_name,
                                stat_type_name,

                                athlete.get("stat")
                            )
                        )

                        team_rows += 1
                        player_stat_rows_processed += 1


    # Save after every team so progress is preserved.
    connection.commit()

    print(
        f"  Player stat rows imported: "
        f"{team_rows}"
    )


# --------------------------------------------------
# CREATE INDEXES
# --------------------------------------------------

cursor.execute("""
CREATE INDEX IF NOT EXISTS
idx_player_game_stats_game
ON player_game_stat_values(game_id)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS
idx_player_game_stats_athlete
ON player_game_stat_values(athlete_id)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS
idx_player_game_stats_team
ON player_game_stat_values(team)
""")

connection.commit()


# --------------------------------------------------
# VERIFY DATABASE
# --------------------------------------------------

cursor.execute("""
SELECT COUNT(*)
FROM player_game_stat_values
""")

rows_stored = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(DISTINCT athlete_id)
FROM player_game_stat_values
""")

unique_athletes = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(DISTINCT game_id)
FROM player_game_stat_values
""")

games_with_player_stats = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(DISTINCT team)
FROM player_game_stat_values
""")

teams_with_player_stats = cursor.fetchone()[0]


print("\n")
print("PLAYER STAT IMPORT RESULTS")
print("=" * 70)

print(
    f"API calls made:            "
    f"{api_calls}"
)

print(
    f"Teams loaded from cache:   "
    f"{cached_teams}"
)

print(
    f"Failed teams:              "
    f"{len(failed_teams)}"
)

print(
    f"Player stat rows processed:"
    f" {player_stat_rows_processed}"
)

print(
    f"Player stat rows stored:   "
    f"{rows_stored}"
)

print(
    f"Unique athletes:           "
    f"{unique_athletes}"
)

print(
    f"Games with player stats:   "
    f"{games_with_player_stats}"
)

print(
    f"Teams with player stats:   "
    f"{teams_with_player_stats}"
)

if failed_teams:

    print("\nFAILED TEAMS")
    print("-" * 70)

    for team in failed_teams:
        print(team)


# --------------------------------------------------
# ALABAMA PASSING SAMPLE
# --------------------------------------------------

cursor.execute("""
SELECT

    g.week,

    p.athlete_name,
    p.stat_type,
    p.stat_value

FROM player_game_stat_values AS p

JOIN games AS g
    ON p.game_id = g.game_id

WHERE
    g.season = ?
    AND p.team = 'Alabama'
    AND p.category = 'passing'

ORDER BY
    g.week,
    p.athlete_name,
    CASE p.stat_type
        WHEN 'C/ATT' THEN 1
        WHEN 'YDS'   THEN 2
        WHEN 'AVG'   THEN 3
        WHEN 'TD'    THEN 4
        WHEN 'INT'   THEN 5
        WHEN 'QBR'   THEN 6
        ELSE 99
    END

LIMIT 30
""", (year,))

rows = cursor.fetchall()


print("\nALABAMA PASSING SAMPLE")
print("-" * 70)

for row in rows:

    print(
        f"Week {row[0]:<2} | "
        f"{row[1]:<25} | "
        f"{row[2]:<6} | "
        f"{row[3]}"
    )


# --------------------------------------------------
# CLOSE DATABASE
# --------------------------------------------------

connection.close()