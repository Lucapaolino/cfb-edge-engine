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

year = 2025

database_file = "data/cfb.db"
raw_file = f"data/raw/betting_lines_{year}.json"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}


# --------------------------------------------------
# DOWNLOAD BETTING LINES WEEK BY WEEK
# --------------------------------------------------

all_betting_games = []

for week in range(1, 17):

    response = requests.get(
        "https://api.collegefootballdata.com/lines",
        headers=headers,
        params={
            "year": year,
            "week": week,
            "seasonType": "regular"
        },
        timeout=60
    )

    response.raise_for_status()

    games = response.json()

    print(
        f"Week {week}: "
        f"{len(games)} games with betting data"
    )

    all_betting_games.extend(games)


# --------------------------------------------------
# DEDUPLICATE GAMES BY GAME ID
# --------------------------------------------------

betting_games_by_id = {
    game["id"]: game
    for game in all_betting_games
}

betting_games = list(
    betting_games_by_id.values()
)

print(
    f"\nUnique games with betting data downloaded: "
    f"{len(betting_games)}"
)


# --------------------------------------------------
# SAVE RAW RESPONSE LOCALLY
# --------------------------------------------------

with open(raw_file, "w") as file:

    json.dump(
        betting_games,
        file,
        indent=2
    )

print(
    f"Raw betting data saved to: "
    f"{raw_file}"
)


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
# GET VALID GAME IDS FROM OUR DATABASE
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

print(
    f"\n2025 games already in database: "
    f"{len(valid_game_ids)}"
)


# --------------------------------------------------
# IMPORT BETTING LINES
# --------------------------------------------------

games_matched = 0
games_skipped = 0
line_records_processed = 0

for game in betting_games:

    game_id = game.get("id")

    # Only import betting lines for games
    # that exist in our games table.
    if game_id not in valid_game_ids:

        games_skipped += 1
        continue

    lines = game.get("lines", [])

    if not lines:
        continue

    games_matched += 1

    for line in lines:

        provider = line.get("provider")

        if not provider:
            continue

        cursor.execute(
            """
            INSERT OR REPLACE INTO betting_lines (

                game_id,
                provider,

                spread,
                spread_open,

                over_under,
                over_under_open,

                home_moneyline,
                away_moneyline

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (
                game_id,
                provider,

                line.get("spread"),
                line.get("spreadOpen"),

                line.get("overUnder"),
                line.get("overUnderOpen"),

                line.get("homeMoneyline"),
                line.get("awayMoneyline")
            )
        )

        line_records_processed += 1


# --------------------------------------------------
# SAVE DATABASE
# --------------------------------------------------

connection.commit()


# --------------------------------------------------
# VERIFY DATABASE
# --------------------------------------------------

cursor.execute("""
SELECT COUNT(*)
FROM betting_lines
""")

line_records_stored = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(DISTINCT game_id)
FROM betting_lines
""")

games_with_lines = cursor.fetchone()[0]


print("\nDATABASE RESULTS")
print("-" * 70)

print(
    f"Betting games matched to our database: "
    f"{games_matched}"
)

print(
    f"Betting games skipped: "
    f"{games_skipped}"
)

print(
    f"Line records processed: "
    f"{line_records_processed}"
)

print(
    f"Line records stored: "
    f"{line_records_stored}"
)

print(
    f"Database games with at least one line: "
    f"{games_with_lines}"
)


# --------------------------------------------------
# SHOW ONE EXAMPLE
# --------------------------------------------------

cursor.execute("""
SELECT

    g.week,
    g.away_team,
    g.home_team,

    b.provider,
    b.spread,
    b.spread_open,

    b.over_under,
    b.over_under_open,

    b.home_moneyline,
    b.away_moneyline

FROM betting_lines AS b

JOIN games AS g
    ON b.game_id = g.game_id

WHERE
    g.season = ?
    AND g.home_team = 'Alabama'

ORDER BY
    g.week,
    b.provider

LIMIT 12
""", (year,))

rows = cursor.fetchall()


print("\nALABAMA BETTING SAMPLE")
print("-" * 70)

for row in rows:

    print(
        f"Week {row[0]} | "
        f"{row[1]} @ {row[2]} | "
        f"{row[3]} | "
        f"Spread: {row[4]} "
        f"(open {row[5]}) | "
        f"Total: {row[6]} "
        f"(open {row[7]}) | "
        f"ML: {row[8]} / {row[9]}"
    )


# --------------------------------------------------
# CLOSE DATABASE
# --------------------------------------------------

connection.close()