import json
import sqlite3


# --------------------------------------------------
# FILE LOCATIONS
# --------------------------------------------------

json_file = "data/raw/games_2025.json"
database_file = "data/cfb.db"


# --------------------------------------------------
# LOAD RAW GAME DATA
# --------------------------------------------------

with open(json_file, "r") as file:
    games = json.load(file)

print(f"Games loaded from JSON: {len(games)}")


# --------------------------------------------------
# CONNECT TO DATABASE
# --------------------------------------------------

connection = sqlite3.connect(database_file)

cursor = connection.cursor()


# --------------------------------------------------
# INSERT GAMES
# --------------------------------------------------

games_inserted = 0

for game in games:

    cursor.execute(
        """
        INSERT OR REPLACE INTO games (

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
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?
        )
        """,

        (
            game.get("id"),

            game.get("season"),
            game.get("week"),
            game.get("seasonType"),

            game.get("startDate"),

            int(game.get("neutralSite", False)),
            int(game.get("conferenceGame", False)),

            game.get("venueId"),
            game.get("venue"),

            # CFBD may use homeId instead of homeTeamId
            game.get("homeId", game.get("homeTeamId")),
            game.get("homeTeam"),
            game.get("homeConference"),
            game.get("homeClassification"),

            # Same idea for away team
            game.get("awayId", game.get("awayTeamId")),
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

    games_inserted += 1


# --------------------------------------------------
# SAVE CHANGES
# --------------------------------------------------

connection.commit()


# --------------------------------------------------
# VERIFY WHAT IS ACTUALLY IN THE DATABASE
# --------------------------------------------------

cursor.execute(
    """
    SELECT COUNT(*)
    FROM games
    """
)

database_game_count = cursor.fetchone()[0]


cursor.execute(
    """
    SELECT
        season,
        week,
        away_team,
        home_team,
        away_points,
        home_points

    FROM games

    ORDER BY
        season,
        week,
        start_date

    LIMIT 5
    """
)

sample_games = cursor.fetchall()


print(f"Games processed: {games_inserted}")
print(f"Games stored in database: {database_game_count}")

print("\nFIRST 5 DATABASE RECORDS")
print("-" * 70)

for game in sample_games:

    season = game[0]
    week = game[1]
    away_team = game[2]
    home_team = game[3]
    away_points = game[4]
    home_points = game[5]

    print(
        f"{season} Week {week} | "
        f"{away_team} {away_points} @ "
        f"{home_team} {home_points}"
    )


# --------------------------------------------------
# CLOSE DATABASE
# --------------------------------------------------

connection.close()