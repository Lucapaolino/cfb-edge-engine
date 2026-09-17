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
                "https://api.collegefootballdata.com/games/teams",
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

total_team_rows_processed = 0
total_stat_rows_processed = 0

failed_weeks = []


print("HISTORICAL TEAM STAT IMPORT")
print("=" * 80)


# --------------------------------------------------
# PROCESS EACH SEASON
# --------------------------------------------------

for year in range(
    start_year,
    end_year + 1
):

    print(f"\n{'=' * 80}")
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

    print(
        f"Games in database: "
        f"{len(valid_game_ids)}"
    )


    # --------------------------------------------------
    # PROCESS WEEKS 1 THROUGH 16
    # --------------------------------------------------

    for week in range(1, 17):

        raw_file = (
            f"data/raw/"
            f"team_stats_{year}_week{week}.json"
        )

        print(
            f"\n{year} Week {week}"
        )


        # --------------------------------------------------
        # USE CACHE IF AVAILABLE
        # --------------------------------------------------

        if os.path.exists(raw_file):

            print(
                "  Using cached file"
            )

            with open(
                raw_file,
                "r"
            ) as file:

                games = json.load(file)

            cached_weeks += 1


        # --------------------------------------------------
        # OTHERWISE DOWNLOAD
        # --------------------------------------------------

        else:

            print(
                "  Downloading from CFBD..."
            )

            games = download_week(
                year,
                week
            )

            if games is None:

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
                    games,
                    file,
                    indent=2
                )

            time.sleep(0.20)


        print(
            f"  Games returned: "
            f"{len(games)}"
        )


        # --------------------------------------------------
        # IMPORT
        # --------------------------------------------------

        week_team_rows = 0
        week_stat_rows = 0
        week_skipped_games = 0

        for game in games:

            game_id = game.get("id")

            if game_id not in valid_game_ids:

                week_skipped_games += 1
                continue


            for team in game.get(
                "teams",
                []
            ):

                team_name = team.get(
                    "team"
                )

                if not team_name:
                    continue


                # --------------------------------------------------
                # TEAM-GAME BASE RECORD
                # --------------------------------------------------

                cursor.execute(
                    """
                    INSERT INTO team_game_stats (

                        game_id,
                        team_id,
                        team,
                        conference,
                        home_away,
                        points

                    )

                    VALUES (?, ?, ?, ?, ?, ?)

                    ON CONFLICT(game_id, team)
                    DO UPDATE SET

                        team_id =
                            excluded.team_id,

                        conference =
                            excluded.conference,

                        home_away =
                            excluded.home_away,

                        points =
                            excluded.points
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

                week_team_rows += 1
                total_team_rows_processed += 1


                # --------------------------------------------------
                # INDIVIDUAL STAT VALUES
                # --------------------------------------------------

                for stat in team.get(
                    "stats",
                    []
                ):

                    category = stat.get(
                        "category"
                    )

                    stat_value = stat.get(
                        "stat"
                    )

                    if category is None:
                        continue

                    if stat_value is not None:
                        stat_value = str(
                            stat_value
                        )


                    cursor.execute(
                        """
                        INSERT INTO
                        team_game_stat_values (

                            game_id,
                            team,
                            category,
                            stat_value

                        )

                        VALUES (?, ?, ?, ?)

                        ON CONFLICT(
                            game_id,
                            team,
                            category
                        )

                        DO UPDATE SET

                            stat_value =
                                excluded.stat_value
                        """,

                        (
                            game_id,
                            team_name,
                            category,
                            stat_value
                        )
                    )

                    week_stat_rows += 1
                    total_stat_rows_processed += 1


        # --------------------------------------------------
        # SAVE AFTER EACH WEEK
        # --------------------------------------------------

        connection.commit()


        print(
            f"  Team rows processed: "
            f"{week_team_rows}"
        )

        print(
            f"  Stat rows processed: "
            f"{week_stat_rows}"
        )

        if week_skipped_games:

            print(
                f"  Games skipped because they "
                f"were not in our database: "
                f"{week_skipped_games}"
            )


    # --------------------------------------------------
    # SEASON COVERAGE AUDIT
    # --------------------------------------------------

    cursor.execute("""
    SELECT COUNT(DISTINCT game_id)

    FROM team_game_stats

    WHERE game_id IN (

        SELECT game_id
        FROM games
        WHERE season = ?
    )
    """, (year,))

    games_with_team_stats = (
        cursor.fetchone()[0]
    )


    cursor.execute("""
    SELECT COUNT(*)

    FROM team_game_stats

    WHERE game_id IN (

        SELECT game_id
        FROM games
        WHERE season = ?
    )
    """, (year,))

    team_rows_stored = (
        cursor.fetchone()[0]
    )


    cursor.execute("""
    SELECT COUNT(*)

    FROM (

        SELECT
            t.game_id,
            COUNT(*) AS team_count

        FROM team_game_stats AS t

        JOIN games AS g
            ON t.game_id = g.game_id

        WHERE g.season = ?

        GROUP BY t.game_id

        HAVING COUNT(*) != 2
    )
    """, (year,))

    games_bad_team_count = (
        cursor.fetchone()[0]
    )


    missing_games = (
        len(valid_game_ids)
        - games_with_team_stats
    )


    print("\nSEASON AUDIT")
    print("-" * 80)

    print(
        f"Games expected:                 "
        f"{len(valid_game_ids)}"
    )

    print(
        f"Games with team stats:          "
        f"{games_with_team_stats}"
    )

    print(
        f"Games missing team stats:       "
        f"{missing_games}"
    )

    print(
        f"Team-game rows stored:          "
        f"{team_rows_stored}"
    )

    print(
        f"Games without exactly 2 teams:  "
        f"{games_bad_team_count}"
    )


# --------------------------------------------------
# FULL HISTORICAL COVERAGE SUMMARY
# --------------------------------------------------

print("\n\n")
print("HISTORICAL TEAM STAT COVERAGE")
print("=" * 80)


cursor.execute("""
SELECT

    g.season,

    COUNT(
        DISTINCT g.game_id
    ) AS total_games,

    COUNT(
        DISTINCT t.game_id
    ) AS games_with_stats

FROM games AS g

LEFT JOIN team_game_stats AS t
    ON g.game_id = t.game_id

WHERE g.season BETWEEN ? AND ?

GROUP BY g.season

ORDER BY g.season
""", (
    start_year,
    2025
))


coverage_rows = cursor.fetchall()


for (
    season,
    total_games,
    games_with_stats
) in coverage_rows:

    if total_games:

        percent = (
            games_with_stats
            / total_games
            * 100
        )

    else:

        percent = 0

    marker = ""

    if season == 2025:
        marker = "  <-- HOLDOUT"

    print(
        f"{season}: "
        f"{games_with_stats:,} / "
        f"{total_games:,} "
        f"({percent:.2f}%)"
        f"{marker}"
    )


# --------------------------------------------------
# DATABASE TOTAL COUNTS
# --------------------------------------------------

cursor.execute("""
SELECT COUNT(*)
FROM team_game_stats
""")

total_team_rows_stored = (
    cursor.fetchone()[0]
)


cursor.execute("""
SELECT COUNT(*)
FROM team_game_stat_values
""")

total_stat_rows_stored = (
    cursor.fetchone()[0]
)


print("\nSUMMARY")
print("=" * 80)

print(
    f"API calls made:              "
    f"{api_calls}"
)

print(
    f"Cached weeks used:           "
    f"{cached_weeks}"
)

print(
    f"Failed weeks:                "
    f"{len(failed_weeks)}"
)

print(
    f"Team rows processed:         "
    f"{total_team_rows_processed:,}"
)

print(
    f"Stat rows processed:         "
    f"{total_stat_rows_processed:,}"
)

print(
    f"Total team rows in database: "
    f"{total_team_rows_stored:,}"
)

print(
    f"Total stat rows in database: "
    f"{total_stat_rows_stored:,}"
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