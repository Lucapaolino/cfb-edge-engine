import os
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
    raise RuntimeError(
        "CFBD_API_KEY was not found"
    )

database_file = "data/cfb.db"

start_year = 2018
end_year = 2024

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
# REQUEST SESSION
# --------------------------------------------------

session = requests.Session()


# --------------------------------------------------
# DOWNLOAD FUNCTION
#
# Tries each request up to 3 times.
# --------------------------------------------------

def download_week(
    year,
    week
):

    for attempt in range(
        1,
        4
    ):

        try:

            response = session.get(
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

            return response.json()

        except requests.exceptions.RequestException as error:

            print(
                f"    Attempt {attempt} failed: "
                f"{error}"
            )

            if attempt < 3:

                time.sleep(2)

    return None


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

api_calls = 0
cached_weeks = 0

failed_weeks = []

total_plays_read = 0
total_plays_imported = 0
total_plays_skipped = 0
total_ppa_plays = 0


print("HISTORICAL PLAY-BY-PLAY IMPORT")
print("=" * 90)


# --------------------------------------------------
# PROCESS EACH SEASON
# --------------------------------------------------

for year in range(
    start_year,
    end_year + 1
):

    print("\n")
    print("=" * 90)

    print(
        f"SEASON {year}"
    )

    print("=" * 90)


    # --------------------------------------------------
    # GET VALID GAME IDS
    #
    # We only import plays for games already in our
    # games table.
    # --------------------------------------------------

    cursor.execute(
        """
        SELECT game_id

        FROM games

        WHERE season = ?
        """,
        (year,)
    )


    valid_game_ids = {
        row[0]
        for row in cursor.fetchall()
    }


    print(
        f"Games in database: "
        f"{len(valid_game_ids):,}"
    )


    # --------------------------------------------------
    # SEASON COUNTERS
    # --------------------------------------------------

    season_plays_read = 0
    season_plays_imported = 0
    season_plays_skipped = 0
    season_ppa_plays = 0


    # --------------------------------------------------
    # PROCESS EACH WEEK
    # --------------------------------------------------

    for week in range(
        1,
        17
    ):

        raw_file = (
            f"data/raw/"
            f"plays_{year}_week{week}.json"
        )


        print(
            f"\n{year} Week {week}"
        )


        # --------------------------------------------------
        # USE CACHE
        # --------------------------------------------------

        if os.path.exists(
            raw_file
        ):

            print(
                "  Using cached file"
            )

            with open(
                raw_file,
                "r"
            ) as file:

                plays = json.load(
                    file
                )

            cached_weeks += 1


        # --------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------

        else:

            print(
                "  Downloading from CFBD..."
            )

            plays = download_week(
                year,
                week
            )


            if plays is None:

                print(
                    "  FAILED after 3 attempts"
                )

                failed_weeks.append(
                    (
                        year,
                        week
                    )
                )

                continue


            api_calls += 1


            # Save raw response immediately.
            with open(
                raw_file,
                "w"
            ) as file:

                json.dump(
                    plays,
                    file,
                    indent=2
                )


            time.sleep(
                0.20
            )


        # --------------------------------------------------
        # WEEK COUNTS
        # --------------------------------------------------

        week_plays_read = len(
            plays
        )

        season_plays_read += (
            week_plays_read
        )

        total_plays_read += (
            week_plays_read
        )


        print(
            f"  Plays returned: "
            f"{week_plays_read:,}"
        )


        # --------------------------------------------------
        # PREPARE DATABASE ROWS
        #
        # We collect an entire week first and then use
        # executemany(), which is much faster than doing
        # one SQLite operation at a time.
        # --------------------------------------------------

        database_rows = []

        week_skipped = 0
        week_ppa = 0


        for play in plays:

            game_id = play.get(
                "gameId"
            )


            if game_id not in valid_game_ids:

                week_skipped += 1

                continue


            play_id = play.get(
                "id"
            )


            if play_id is None:

                week_skipped += 1

                continue


            # --------------------------------------------------
            # CLOCK
            # --------------------------------------------------

            clock = play.get(
                "clock"
            )


            if isinstance(
                clock,
                dict
            ):

                clock_minutes = (
                    clock.get(
                        "minutes"
                    )
                )

                clock_seconds = (
                    clock.get(
                        "seconds"
                    )
                )

            else:

                clock_minutes = None
                clock_seconds = None


            # --------------------------------------------------
            # PPA COUNT
            # --------------------------------------------------

            ppa = play.get(
                "ppa"
            )


            if ppa is not None:

                week_ppa += 1


            # --------------------------------------------------
            # SCORING FLAG
            # --------------------------------------------------

            scoring = play.get(
                "scoring"
            )


            if scoring:

                scoring_value = 1

            else:

                scoring_value = 0


            # --------------------------------------------------
            # DATABASE ROW
            # --------------------------------------------------

            database_rows.append(
                (
                    str(
                        play_id
                    ),

                    game_id,

                    play.get(
                        "driveId"
                    ),

                    play.get(
                        "driveNumber"
                    ),

                    play.get(
                        "playNumber"
                    ),

                    play.get(
                        "offense"
                    ),

                    play.get(
                        "defense"
                    ),

                    play.get(
                        "period"
                    ),

                    clock_minutes,

                    clock_seconds,

                    play.get(
                        "yardsToGoal"
                    ),

                    play.get(
                        "down"
                    ),

                    play.get(
                        "distance"
                    ),

                    play.get(
                        "yardsGained"
                    ),

                    play.get(
                        "playType"
                    ),

                    play.get(
                        "playText"
                    ),

                    ppa,

                    scoring_value,

                    play.get(
                        "wallclock"
                    )
                )
            )


        # --------------------------------------------------
        # BULK INSERT
        # --------------------------------------------------

        cursor.executemany(
            """
            INSERT INTO plays (

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
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?,
                ?, ?,
                ?,
                ?, ?,
                ?,
                ?, ?,
                ?,
                ?, ?
            )

            ON CONFLICT(play_id)

            DO UPDATE SET

                game_id =
                    excluded.game_id,

                drive_id =
                    excluded.drive_id,

                drive_number =
                    excluded.drive_number,

                play_number =
                    excluded.play_number,

                offense =
                    excluded.offense,

                defense =
                    excluded.defense,

                period =
                    excluded.period,

                clock_minutes =
                    excluded.clock_minutes,

                clock_seconds =
                    excluded.clock_seconds,

                yards_to_goal =
                    excluded.yards_to_goal,

                down =
                    excluded.down,

                distance =
                    excluded.distance,

                yards_gained =
                    excluded.yards_gained,

                play_type =
                    excluded.play_type,

                play_text =
                    excluded.play_text,

                ppa =
                    excluded.ppa,

                scoring =
                    excluded.scoring,

                wallclock =
                    excluded.wallclock
            """,
            database_rows
        )


        connection.commit()


        week_imported = len(
            database_rows
        )


        season_plays_imported += (
            week_imported
        )

        season_plays_skipped += (
            week_skipped
        )

        season_ppa_plays += (
            week_ppa
        )


        total_plays_imported += (
            week_imported
        )

        total_plays_skipped += (
            week_skipped
        )

        total_ppa_plays += (
            week_ppa
        )


        print(
            f"  Plays imported: "
            f"{week_imported:,}"
        )

        print(
            f"  Plays with PPA: "
            f"{week_ppa:,}"
        )


        if week_skipped:

            print(
                f"  Plays skipped: "
                f"{week_skipped:,}"
            )


    # ==================================================
    # SEASON COVERAGE AUDIT
    # ==================================================

    cursor.execute(
        """
        SELECT COUNT(*)

        FROM games

        WHERE
            season = ?

            AND home_points IS NOT NULL
            AND away_points IS NOT NULL

            AND home_classification = 'fbs'
            AND away_classification = 'fbs'
        """,
        (year,)
    )


    expected_fbs_games = (
        cursor.fetchone()[0]
    )


    cursor.execute(
        """
        SELECT COUNT(
            DISTINCT p.game_id
        )

        FROM plays AS p

        JOIN games AS g
            ON p.game_id = g.game_id

        WHERE
            g.season = ?

            AND g.home_points IS NOT NULL
            AND g.away_points IS NOT NULL

            AND g.home_classification = 'fbs'
            AND g.away_classification = 'fbs'
        """,
        (year,)
    )


    fbs_games_with_plays = (
        cursor.fetchone()[0]
    )


    cursor.execute(
        """
        SELECT COUNT(*)

        FROM plays AS p

        JOIN games AS g
            ON p.game_id = g.game_id

        WHERE
            g.season = ?

            AND g.home_points IS NOT NULL
            AND g.away_points IS NOT NULL

            AND g.home_classification = 'fbs'
            AND g.away_classification = 'fbs'

            AND p.ppa IS NOT NULL
        """,
        (year,)
    )


    fbs_ppa_plays = (
        cursor.fetchone()[0]
    )


    print("\nSEASON AUDIT")
    print("-" * 90)

    print(
        f"FBS-v-FBS games expected:     "
        f"{expected_fbs_games:,}"
    )

    print(
        f"FBS-v-FBS games with plays:   "
        f"{fbs_games_with_plays:,}"
    )

    print(
        f"Missing play-by-play games:    "
        f"{expected_fbs_games - fbs_games_with_plays:,}"
    )

    print(
        f"FBS-v-FBS plays with PPA:     "
        f"{fbs_ppa_plays:,}"
    )

    print(
        f"All season plays read:        "
        f"{season_plays_read:,}"
    )

    print(
        f"All season plays imported:    "
        f"{season_plays_imported:,}"
    )


# --------------------------------------------------
# ADD USEFUL INDEX
# --------------------------------------------------

cursor.execute(
    """
    CREATE INDEX IF NOT EXISTS
    idx_plays_game_offense

    ON plays(
        game_id,
        offense
    )
    """
)


cursor.execute(
    """
    CREATE INDEX IF NOT EXISTS
    idx_plays_game_defense

    ON plays(
        game_id,
        defense
    )
    """
)


connection.commit()


# ==================================================
# FINAL DATABASE SUMMARY
# ==================================================

cursor.execute(
    """
    SELECT COUNT(*)
    FROM plays
    """
)

database_plays = (
    cursor.fetchone()[0]
)


cursor.execute(
    """
    SELECT COUNT(*)

    FROM plays

    WHERE ppa IS NOT NULL
    """
)

database_ppa_plays = (
    cursor.fetchone()[0]
)


cursor.execute(
    """
    SELECT COUNT(
        DISTINCT game_id
    )

    FROM plays
    """
)

database_games_with_plays = (
    cursor.fetchone()[0]
)


print("\n\n")
print("HISTORICAL PLAY IMPORT RESULTS")
print("=" * 90)

print(
    f"API calls made:             "
    f"{api_calls}"
)

print(
    f"Cached weeks used:          "
    f"{cached_weeks}"
)

print(
    f"Failed weeks:               "
    f"{len(failed_weeks)}"
)

print(
    f"Historical plays read:      "
    f"{total_plays_read:,}"
)

print(
    f"Historical plays imported:  "
    f"{total_plays_imported:,}"
)

print(
    f"Historical plays skipped:   "
    f"{total_plays_skipped:,}"
)

print(
    f"Historical plays with PPA:  "
    f"{total_ppa_plays:,}"
)

print(
    f"TOTAL plays in database:    "
    f"{database_plays:,}"
)

print(
    f"TOTAL plays with PPA:       "
    f"{database_ppa_plays:,}"
)

print(
    f"Games represented by plays: "
    f"{database_games_with_plays:,}"
)


if failed_weeks:

    print("\nFAILED WEEKS")
    print("-" * 90)

    for (
        year,
        week
    ) in failed_weeks:

        print(
            f"{year} Week {week}"
        )


print("\n")
print(
    "2025 data was not downloaded or "
    "modified by this historical import."
)


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

connection.close()

session.close()