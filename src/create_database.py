import os
import sqlite3

# --------------------------------------------------
# DATABASE LOCATION
# --------------------------------------------------

database_file = "data/cfb.db"

# Make sure the data directory exists
os.makedirs("data", exist_ok=True)

# Connect to SQLite.
# If cfb.db does not exist yet, SQLite creates it.
connection = sqlite3.connect(database_file)

cursor = connection.cursor()


# --------------------------------------------------
# GAMES TABLE
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS games (

    game_id INTEGER PRIMARY KEY,

    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    season_type TEXT,

    start_date TEXT,

    neutral_site INTEGER,
    conference_game INTEGER,

    venue_id INTEGER,
    venue TEXT,

    home_team_id INTEGER,
    home_team TEXT,
    home_conference TEXT,
    home_classification TEXT,

    away_team_id INTEGER,
    away_team TEXT,
    away_conference TEXT,
    away_classification TEXT,

    home_points INTEGER,
    away_points INTEGER,

    home_pregame_elo REAL,
    away_pregame_elo REAL,

    home_postgame_elo REAL,
    away_postgame_elo REAL
)
""")


# --------------------------------------------------
# TEAM GAME STATS TABLE
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS team_game_stats (

    game_id INTEGER NOT NULL,
    team_id INTEGER,
    team TEXT NOT NULL,
    conference TEXT,
    home_away TEXT,
    points INTEGER,

    PRIMARY KEY (game_id, team),

    FOREIGN KEY (game_id)
        REFERENCES games(game_id)
)
""")


# --------------------------------------------------
# PLAYS TABLE
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS plays (

    play_id TEXT PRIMARY KEY,

    game_id INTEGER NOT NULL,
    drive_id TEXT,

    drive_number INTEGER,
    play_number INTEGER,

    offense TEXT,
    defense TEXT,

    period INTEGER,

    clock_minutes INTEGER,
    clock_seconds INTEGER,

    yards_to_goal INTEGER,
    down INTEGER,
    distance INTEGER,
    yards_gained INTEGER,

    play_type TEXT,
    play_text TEXT,

    ppa REAL,

    scoring INTEGER,
    wallclock TEXT,

    FOREIGN KEY (game_id)
        REFERENCES games(game_id)
)
""")


# --------------------------------------------------
# PLAYERS TABLE
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS players (

    athlete_id TEXT PRIMARY KEY,

    first_name TEXT,
    last_name TEXT
)
""")


# --------------------------------------------------
# ROSTERS TABLE
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS rosters (

    season INTEGER NOT NULL,
    athlete_id TEXT NOT NULL,

    team TEXT NOT NULL,
    position TEXT,

    jersey INTEGER,
    height INTEGER,
    weight INTEGER,

    PRIMARY KEY (season, athlete_id, team),

    FOREIGN KEY (athlete_id)
        REFERENCES players(athlete_id)
)
""")


# --------------------------------------------------
# TEAM TALENT TABLE
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS team_talent (

    season INTEGER NOT NULL,
    team TEXT NOT NULL,

    talent REAL,

    PRIMARY KEY (season, team)
)
""")


# --------------------------------------------------
# RETURNING PRODUCTION TABLE
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS returning_production (

    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    conference TEXT,

    total_ppa REAL,
    total_passing_ppa REAL,
    total_receiving_ppa REAL,
    total_rushing_ppa REAL,

    percent_ppa REAL,
    percent_passing_ppa REAL,
    percent_receiving_ppa REAL,
    percent_rushing_ppa REAL,

    usage REAL,
    passing_usage REAL,
    receiving_usage REAL,
    rushing_usage REAL,

    PRIMARY KEY (season, team)
)
""")


# --------------------------------------------------
# BETTING LINES TABLE
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS betting_lines (

    game_id INTEGER NOT NULL,
    provider TEXT NOT NULL,

    spread REAL,
    spread_open REAL,

    over_under REAL,
    over_under_open REAL,

    home_moneyline INTEGER,
    away_moneyline INTEGER,

    PRIMARY KEY (game_id, provider),

    FOREIGN KEY (game_id)
        REFERENCES games(game_id)
)
""")


# --------------------------------------------------
# SAVE CHANGES
# --------------------------------------------------

connection.commit()

connection.close()

print("Database created successfully.")
print(f"Saved to: {database_file}")