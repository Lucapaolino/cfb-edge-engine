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
# GET VALID GAME IDS
#
# We only want plays belonging to games that are
# already part of our 2025 database.
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
    f"Valid 2025 games in database: "
    f"{len(valid_game_ids)}"
)


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

total_plays_read = 0
total_plays_imported = 0
total_plays_skipped = 0

api_calls = 0


# --------------------------------------------------
# PROCESS EACH REGULAR-SEASON WEEK
# --------------------------------------------------

for week in range(1, 17):

    raw_file = (
        f"data/raw/"
        f"plays_{year}_week{week}.json"
    )


    # --------------------------------------------------
    # USE CACHED FILE IF IT ALREADY EXISTS
    # --------------------------------------------------

    if os.path.exists(raw_file):

        print(
            f"\nWeek {week}: "
            f"using cached file"
        )

        with open(raw_file, "r") as file:
            plays = json.load(file)


    # --------------------------------------------------
    # OTHERWISE DOWNLOAD FROM CFBD
    # --------------------------------------------------

    else:

        print(
            f"\nWeek {week}: "
            f"downloading from CFBD..."
        )

        response = requests.get(
            "https://api.collegefootballdata.com/plays",
            headers=headers,
            params={
                "year": year,
                "week": week,
                "seasonType": "regular",
                "classification": "fbs"
            },
            timeout=120
        )

        response.raise_for_status()

        plays = response.json()

        api_calls += 1

        # Cache raw response immediately
        with open(raw_file, "w") as file:

            json.dump(
                plays,
                file,
                indent=2
            )


    # --------------------------------------------------
    # WEEK SUMMARY
    # --------------------------------------------------

    print(
        f"Week {week}: "
        f"{len(plays)} plays read"
    )

    total_plays_read += len(plays)


    # --------------------------------------------------
    # IMPORT EACH PLAY
    # --------------------------------------------------

    week_imported = 0
    week_skipped = 0

    for play in plays:

        game_id = play.get("gameId")

        # Ignore plays from games that are not
        # in our games table.
        if game_id not in valid_game_ids:

            week_skipped += 1
            total_plays_skipped += 1
            continue


        clock = play.get("clock") or {}

        play_id = play.get("id")

        if play_id is None:
            continue


        cursor.execute(
            """
            INSERT OR REPLACE INTO plays (

                play_id,
                game_id,
                drive_id,

                drive_number,
                play_number,

                offense,
                defense,

                period,

                clock_minutes,
                clock_seconds,

                yards_to_goal,
                down,
                distance,
                yards_gained,

                play_type,
                play_text,

                ppa,

                scoring,
                wallclock

            )

            VALUES (
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?,
                ?, ?
            )
            """,

            (
                str(play_id),
                game_id,
                play.get("driveId"),

                play.get("driveNumber"),
                play.get("playNumber"),

                play.get("offense"),
                play.get("defense"),

                play.get("period"),

                clock.get("minutes"),
                clock.get("seconds"),

                play.get("yardsToGoal"),
                play.get("down"),
                play.get("distance"),
                play.get("yardsGained"),

                play.get("playType"),
                play.get("playText"),

                play.get("ppa"),

                int(
                    play.get(
                        "scoring",
                        False
                    )
                ),

                play.get("wallclock")
            )
        )

        week_imported += 1
        total_plays_imported += 1


    # Save after each week so progress is preserved.
    connection.commit()

    print(
        f"Week {week}: "
        f"{week_imported} plays imported"
    )

    if week_skipped:

        print(
            f"Week {week}: "
            f"{week_skipped} plays skipped "
            f"(game not in database)"
        )


# --------------------------------------------------
# CREATE INDEXES
#
# Indexes make future searches much faster.
# --------------------------------------------------

cursor.execute("""
CREATE INDEX IF NOT EXISTS
idx_plays_game_id
ON plays(game_id)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS
idx_plays_offense
ON plays(offense)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS
idx_plays_defense
ON plays(defense)
""")

connection.commit()


# --------------------------------------------------
# VERIFY DATABASE
# --------------------------------------------------

cursor.execute("""
SELECT COUNT(*)
FROM plays
""")

plays_stored = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(DISTINCT game_id)
FROM plays
""")

games_with_plays = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(*)
FROM plays
WHERE ppa IS NOT NULL
""")

plays_with_ppa = cursor.fetchone()[0]


print("\n")
print("PLAY-BY-PLAY IMPORT RESULTS")
print("=" * 70)

print(
    f"API calls made:          "
    f"{api_calls}"
)

print(
    f"Total plays read:        "
    f"{total_plays_read}"
)

print(
    f"Total plays imported:    "
    f"{total_plays_imported}"
)

print(
    f"Total plays skipped:     "
    f"{total_plays_skipped}"
)

print(
    f"Unique plays stored:     "
    f"{plays_stored}"
)

print(
    f"Games with play data:    "
    f"{games_with_plays}"
)

print(
    f"Plays with PPA:          "
    f"{plays_with_ppa}"
)


# --------------------------------------------------
# ALABAMA EXAMPLE
# --------------------------------------------------

cursor.execute("""
SELECT

    g.week,
    p.offense,
    p.defense,
    p.down,
    p.distance,
    p.yards_gained,
    p.play_type,
    p.ppa

FROM plays AS p

JOIN games AS g
    ON p.game_id = g.game_id

WHERE
    g.season = ?
    AND p.offense = 'Alabama'
    AND p.ppa IS NOT NULL

ORDER BY
    g.week,
    p.game_id,
    p.drive_number,
    p.play_number

LIMIT 15
""", (year,))

rows = cursor.fetchall()


print("\nALABAMA PLAY SAMPLE")
print("-" * 70)

for row in rows:

    print(
        f"Week {row[0]} | "
        f"{row[1]} vs {row[2]} | "
        f"{row[3]} & {row[4]} | "
        f"{row[5]} yards | "
        f"{row[6]} | "
        f"PPA: {row[7]}"
    )


# --------------------------------------------------
# CLOSE DATABASE
# --------------------------------------------------

connection.close()