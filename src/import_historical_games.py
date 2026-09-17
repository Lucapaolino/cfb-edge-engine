import os
import json
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

database_file = "data/cfb.db"

start_year = 2018
end_year = 2024

years = range(
    start_year,
    end_year + 1
)

headers = {
    "Authorization": f"Bearer {API_KEY}"
}


# --------------------------------------------------
# CONNECT TO DATABASE
# --------------------------------------------------

connection = sqlite3.connect(
    database_file
)

connection.execute(
    "PRAGMA foreign_keys = ON"
)

cursor = connection.cursor()


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

api_calls = 0
cached_seasons = 0
total_games_processed = 0


print("HISTORICAL GAME IMPORT")
print("=" * 80)


# --------------------------------------------------
# PROCESS EACH SEASON
# --------------------------------------------------

for year in years:

    raw_file = (
        f"data/raw/games_{year}.json"
    )

    print(
        f"\nProcessing {year}"
    )

    print("-" * 80)


    # --------------------------------------------------
    # USE CACHED FILE IF IT ALREADY EXISTS
    # --------------------------------------------------

    if os.path.exists(raw_file):

        print(
            "Using cached raw file"
        )

        with open(
            raw_file,
            "r"
        ) as file:

            games = json.load(file)

        cached_seasons += 1


    # --------------------------------------------------
    # OTHERWISE DOWNLOAD FROM CFBD
    # --------------------------------------------------

    else:

        print(
            "Downloading from CFBD..."
        )

        response = requests.get(
            "https://api.collegefootballdata.com/games",
            headers=headers,
            params={
                "year": year,
                "seasonType": "regular",
                "classification": "fbs"
            },
            timeout=120
        )

        response.raise_for_status()

        games = response.json()

        api_calls += 1


        # --------------------------------------------------
        # SAVE RAW COPY BEFORE DATABASE WORK
        # --------------------------------------------------

        with open(
            raw_file,
            "w"
        ) as file:

            json.dump(
                games,
                file,
                indent=2
            )

        print(
            f"Raw file saved: "
            f"{raw_file}"
        )


    print(
        f"Games returned: "
        f"{len(games)}"
    )


    # --------------------------------------------------
    # INSERT / UPDATE GAMES
    #
    # ON CONFLICT updates the existing row without
    # deleting it first.
    #
    # This is safer than INSERT OR REPLACE once
    # other tables reference game_id.
    # --------------------------------------------------

    season_processed = 0

    for game in games:

        cursor.execute(
            """
            INSERT INTO games (

                game_id,

                season,
                week,
                season_type,

                start_date,

                neutral_site,
                conference_game,

                venue_id,
                venue,

                home_team_id,
                home_team,
                home_conference,
                home_classification,

                away_team_id,
                away_team,
                away_conference,
                away_classification,

                home_points,
                away_points,

                home_pregame_elo,
                away_pregame_elo,

                home_postgame_elo,
                away_postgame_elo

            )

            VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?
            )

            ON CONFLICT(game_id)
            DO UPDATE SET

                season = excluded.season,
                week = excluded.week,
                season_type = excluded.season_type,

                start_date = excluded.start_date,

                neutral_site = excluded.neutral_site,
                conference_game = excluded.conference_game,

                venue_id = excluded.venue_id,
                venue = excluded.venue,

                home_team_id = excluded.home_team_id,
                home_team = excluded.home_team,
                home_conference = excluded.home_conference,
                home_classification = excluded.home_classification,

                away_team_id = excluded.away_team_id,
                away_team = excluded.away_team,
                away_conference = excluded.away_conference,
                away_classification = excluded.away_classification,

                home_points = excluded.home_points,
                away_points = excluded.away_points,

                home_pregame_elo = excluded.home_pregame_elo,
                away_pregame_elo = excluded.away_pregame_elo,

                home_postgame_elo = excluded.home_postgame_elo,
                away_postgame_elo = excluded.away_postgame_elo
            """,

            (
                game.get("id"),

                game.get("season"),
                game.get("week"),
                game.get("seasonType"),

                game.get("startDate"),

                int(
                    game.get(
                        "neutralSite",
                        False
                    )
                ),

                int(
                    game.get(
                        "conferenceGame",
                        False
                    )
                ),

                game.get("venueId"),
                game.get("venue"),

                game.get(
                    "homeId",
                    game.get(
                        "homeTeamId"
                    )
                ),

                game.get("homeTeam"),
                game.get("homeConference"),
                game.get("homeClassification"),

                game.get(
                    "awayId",
                    game.get(
                        "awayTeamId"
                    )
                ),

                game.get("awayTeam"),
                game.get("awayConference"),
                game.get("awayClassification"),

                game.get("homePoints"),
                game.get("awayPoints"),

                game.get("homePregameElo"),
                game.get("awayPregameElo"),

                game.get("homePostgameElo"),
                game.get("awayPostgameElo")
            )
        )

        season_processed += 1
        total_games_processed += 1


    # --------------------------------------------------
    # SAVE EACH SEASON IMMEDIATELY
    # --------------------------------------------------

    connection.commit()


    # --------------------------------------------------
    # VERIFY SEASON COUNT
    # --------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM games
        WHERE season = ?
        """,
        (year,)
    )

    database_count = cursor.fetchone()[0]


    print(
        f"Games processed: "
        f"{season_processed}"
    )

    print(
        f"Games stored in database: "
        f"{database_count}"
    )


# --------------------------------------------------
# SHOW SEASON-BY-SEASON DATABASE COUNTS
# --------------------------------------------------

print("\n")
print("DATABASE SEASON COUNTS")
print("=" * 80)


cursor.execute("""
SELECT
    season,
    COUNT(*)

FROM games

WHERE season BETWEEN ? AND ?

GROUP BY season

ORDER BY season
""", (
    start_year,
    2025
))


season_counts = cursor.fetchall()


for season, count in season_counts:

    marker = ""

    if season == 2025:
        marker = "  <-- FINAL HOLDOUT"

    print(
        f"{season}: "
        f"{count:,} games"
        f"{marker}"
    )


# --------------------------------------------------
# TOTAL DATABASE SIZE
# --------------------------------------------------

cursor.execute("""
SELECT COUNT(*)
FROM games
""")

database_total = cursor.fetchone()[0]


print("\nSUMMARY")
print("=" * 80)

print(
    f"API calls made:       "
    f"{api_calls}"
)

print(
    f"Cached seasons used:  "
    f"{cached_seasons}"
)

print(
    f"Historical games "
    f"processed:         "
    f"{total_games_processed:,}"
)

print(
    f"Total games now in "
    f"database:          "
    f"{database_total:,}"
)


# --------------------------------------------------
# CLOSE DATABASE
# --------------------------------------------------

connection.close()