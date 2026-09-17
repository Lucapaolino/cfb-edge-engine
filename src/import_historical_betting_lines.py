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
    raise RuntimeError("CFBD_API_KEY was not found")

database_file = "data/cfb.db"

start_year = 2018
end_year = 2024

headers = {
    "Authorization": f"Bearer {API_KEY}"
}


# --------------------------------------------------
# CONNECT TO DATABASE
# --------------------------------------------------

connection = sqlite3.connect(database_file)

connection.execute("PRAGMA foreign_keys = ON")

cursor = connection.cursor()


# --------------------------------------------------
# DOWNLOAD FUNCTION WITH RETRIES
# --------------------------------------------------

def download_week(year, week):

    for attempt in range(1, 4):

        try:

            response = requests.get(
                "https://api.collegefootballdata.com/lines",
                headers=headers,
                params={
                    "year": year,
                    "week": week,
                    "seasonType": "regular"
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

total_games_returned = 0
total_games_matched = 0
total_games_skipped = 0
total_lines_processed = 0


print("HISTORICAL BETTING LINE IMPORT")
print("=" * 80)


# --------------------------------------------------
# PROCESS EACH SEASON
# --------------------------------------------------

for year in range(
    start_year,
    end_year + 1
):

    print("\n" + "=" * 80)
    print(f"SEASON {year}")
    print("=" * 80)


    # --------------------------------------------------
    # VALID GAME IDS FOR THIS SEASON
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
    # PROCESS WEEKS
    # --------------------------------------------------

    for week in range(1, 17):

        raw_file = (
            f"data/raw/"
            f"betting_lines_{year}_week{week}.json"
        )


        print(
            f"\n{year} Week {week}"
        )


        # --------------------------------------------------
        # CACHE
        # --------------------------------------------------

        if os.path.exists(raw_file):

            print(
                "  Using cached file"
            )

            with open(
                raw_file,
                "r"
            ) as file:

                betting_games = json.load(file)

            cached_weeks += 1


        # --------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------

        else:

            print(
                "  Downloading from CFBD..."
            )

            betting_games = download_week(
                year,
                week
            )

            if betting_games is None:

                print(
                    "  FAILED after 3 attempts"
                )

                failed_weeks.append(
                    (year, week)
                )

                continue


            api_calls += 1


            with open(
                raw_file,
                "w"
            ) as file:

                json.dump(
                    betting_games,
                    file,
                    indent=2
                )

            time.sleep(0.20)


        print(
            f"  Games returned: "
            f"{len(betting_games)}"
        )


        total_games_returned += len(
            betting_games
        )


        # --------------------------------------------------
        # IMPORT
        # --------------------------------------------------

        week_matched = 0
        week_skipped = 0
        week_lines = 0


        for game in betting_games:

            game_id = game.get("id")


            # Only keep games that exist in our
            # historical games table.
            if game_id not in valid_game_ids:

                week_skipped += 1
                total_games_skipped += 1

                continue


            lines = game.get(
                "lines",
                []
            )


            if not lines:
                continue


            week_matched += 1
            total_games_matched += 1


            for line in lines:

                provider = line.get(
                    "provider"
                )

                if not provider:
                    continue


                cursor.execute(
                    """
                    INSERT INTO betting_lines (

                        game_id,
                        provider,

                        spread,
                        spread_open,

                        over_under,
                        over_under_open,

                        home_moneyline,
                        away_moneyline

                    )

                    VALUES (
                        ?, ?, ?, ?,
                        ?, ?, ?, ?
                    )

                    ON CONFLICT(
                        game_id,
                        provider
                    )

                    DO UPDATE SET

                        spread =
                            excluded.spread,

                        spread_open =
                            excluded.spread_open,

                        over_under =
                            excluded.over_under,

                        over_under_open =
                            excluded.over_under_open,

                        home_moneyline =
                            excluded.home_moneyline,

                        away_moneyline =
                            excluded.away_moneyline
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


                week_lines += 1
                total_lines_processed += 1


        connection.commit()


        print(
            f"  Database games matched: "
            f"{week_matched}"
        )

        print(
            f"  Line records processed: "
            f"{week_lines}"
        )

        if week_skipped:

            print(
                f"  Non-database games skipped: "
                f"{week_skipped}"
            )


    # --------------------------------------------------
    # SEASON GAME-LEVEL COVERAGE
    #
    # Only completed games count toward our
    # historical modeling coverage.
    # --------------------------------------------------

    cursor.execute("""
    SELECT COUNT(*)

    FROM games

    WHERE
        season = ?
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
    """, (year,))

    played_games = cursor.fetchone()[0]


    cursor.execute("""
    SELECT COUNT(DISTINCT b.game_id)

    FROM betting_lines AS b

    JOIN games AS g
        ON b.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
    """, (year,))

    games_with_lines = cursor.fetchone()[0]


    cursor.execute("""
    SELECT COUNT(DISTINCT b.game_id)

    FROM betting_lines AS b

    JOIN games AS g
        ON b.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
        AND b.spread IS NOT NULL
    """, (year,))

    games_with_spread = cursor.fetchone()[0]


    cursor.execute("""
    SELECT COUNT(DISTINCT b.game_id)

    FROM betting_lines AS b

    JOIN games AS g
        ON b.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
        AND b.spread_open IS NOT NULL
    """, (year,))

    games_with_open_spread = (
        cursor.fetchone()[0]
    )


    cursor.execute("""
    SELECT COUNT(DISTINCT b.game_id)

    FROM betting_lines AS b

    JOIN games AS g
        ON b.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
        AND b.over_under IS NOT NULL
    """, (year,))

    games_with_total = cursor.fetchone()[0]


    cursor.execute("""
    SELECT COUNT(DISTINCT b.game_id)

    FROM betting_lines AS b

    JOIN games AS g
        ON b.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
        AND b.over_under_open IS NOT NULL
    """, (year,))

    games_with_open_total = (
        cursor.fetchone()[0]
    )


    cursor.execute("""
    SELECT COUNT(DISTINCT b.game_id)

    FROM betting_lines AS b

    JOIN games AS g
        ON b.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
        AND (
            b.home_moneyline IS NOT NULL
            OR b.away_moneyline IS NOT NULL
        )
    """, (year,))

    games_with_moneyline = (
        cursor.fetchone()[0]
    )


    print("\nSEASON BETTING COVERAGE")
    print("-" * 80)

    print(
        f"Completed games:            "
        f"{played_games}"
    )

    print(
        f"Games with any line:        "
        f"{games_with_lines}"
    )

    print(
        f"Games with spread:          "
        f"{games_with_spread}"
    )

    print(
        f"Games with opening spread:  "
        f"{games_with_open_spread}"
    )

    print(
        f"Games with total:           "
        f"{games_with_total}"
    )

    print(
        f"Games with opening total:   "
        f"{games_with_open_total}"
    )

    print(
        f"Games with moneyline:       "
        f"{games_with_moneyline}"
    )


# --------------------------------------------------
# FULL COVERAGE TABLE
# --------------------------------------------------

print("\n\n")
print("HISTORICAL BETTING COVERAGE")
print("=" * 80)


for year in range(
    start_year,
    2026
):

    cursor.execute("""
    SELECT COUNT(*)

    FROM games

    WHERE
        season = ?
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
    """, (year,))

    played = cursor.fetchone()[0]


    cursor.execute("""
    SELECT COUNT(DISTINCT b.game_id)

    FROM betting_lines AS b

    JOIN games AS g
        ON b.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
    """, (year,))

    with_lines = cursor.fetchone()[0]


    if played:

        percent = (
            with_lines
            / played
            * 100
        )

    else:

        percent = 0


    marker = ""

    if year == 2025:

        marker = "  <-- HOLDOUT"


    print(
        f"{year}: "
        f"{with_lines:,} / "
        f"{played:,} "
        f"({percent:.2f}%)"
        f"{marker}"
    )


# --------------------------------------------------
# TOTAL DATABASE COUNTS
# --------------------------------------------------

cursor.execute("""
SELECT COUNT(*)
FROM betting_lines
""")

total_line_rows = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(DISTINCT game_id)
FROM betting_lines
""")

total_games_with_lines = (
    cursor.fetchone()[0]
)


print("\nSUMMARY")
print("=" * 80)

print(
    f"API calls made:               "
    f"{api_calls}"
)

print(
    f"Cached weeks used:            "
    f"{cached_weeks}"
)

print(
    f"Failed weeks:                 "
    f"{len(failed_weeks)}"
)

print(
    f"Betting games returned:       "
    f"{total_games_returned:,}"
)

print(
    f"Database games matched:       "
    f"{total_games_matched:,}"
)

print(
    f"Non-database games skipped:   "
    f"{total_games_skipped:,}"
)

print(
    f"Line records processed:       "
    f"{total_lines_processed:,}"
)

print(
    f"Total line rows in database:  "
    f"{total_line_rows:,}"
)

print(
    f"Database games with lines:    "
    f"{total_games_with_lines:,}"
)


if failed_weeks:

    print("\nFAILED WEEKS")
    print("-" * 80)

    for year, week in failed_weeks:

        print(
            f"{year} Week {week}"
        )


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

connection.close()