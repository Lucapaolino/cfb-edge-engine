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
# DOWNLOAD HELPER
# --------------------------------------------------

def download_json(url, params):

    for attempt in range(1, 4):

        try:

            response = requests.get(
                url,
                headers=headers,
                params=params,
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

cached_talent_files = 0
cached_returning_files = 0

failed_talent_years = []
failed_returning_years = []

talent_rows_processed = 0
returning_rows_processed = 0


print("HISTORICAL TEAM CONTEXT IMPORT")
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


    # ==================================================
    # TEAM TALENT
    # ==================================================

    talent_file = (
        f"data/raw/"
        f"team_talent_{year}.json"
    )


    print("\nTEAM TALENT")
    print("-" * 80)


    if os.path.exists(talent_file):

        print("Using cached file")

        with open(
            talent_file,
            "r"
        ) as file:

            talent_records = json.load(file)

        cached_talent_files += 1


    else:

        print("Downloading from CFBD...")

        talent_records = download_json(
            "https://api.collegefootballdata.com/talent",
            {
                "year": year
            }
        )


        if talent_records is None:

            print(
                "FAILED to download talent data"
            )

            failed_talent_years.append(year)

            talent_records = []

        else:

            api_calls += 1

            with open(
                talent_file,
                "w"
            ) as file:

                json.dump(
                    talent_records,
                    file,
                    indent=2
                )

            print(
                f"Saved raw file: "
                f"{talent_file}"
            )

            time.sleep(0.20)


    print(
        f"Talent records returned: "
        f"{len(talent_records)}"
    )


    season_talent_processed = 0


    for record in talent_records:

        record_year = record.get(
            "year",
            year
        )

        team = record.get("team")

        if not team:
            continue


        cursor.execute(
            """
            INSERT INTO team_talent (

                season,
                team,
                talent

            )

            VALUES (?, ?, ?)

            ON CONFLICT(season, team)
            DO UPDATE SET

                talent =
                    excluded.talent
            """,

            (
                record_year,
                team,
                record.get("talent")
            )
        )


        season_talent_processed += 1
        talent_rows_processed += 1


    connection.commit()


    print(
        f"Talent rows processed: "
        f"{season_talent_processed}"
    )


    # ==================================================
    # RETURNING PRODUCTION
    # ==================================================

    returning_file = (
        f"data/raw/"
        f"returning_production_{year}.json"
    )


    print("\nRETURNING PRODUCTION")
    print("-" * 80)


    if os.path.exists(returning_file):

        print("Using cached file")

        with open(
            returning_file,
            "r"
        ) as file:

            returning_records = json.load(file)

        cached_returning_files += 1


    else:

        print("Downloading from CFBD...")

        returning_records = download_json(
            "https://api.collegefootballdata.com/player/returning",
            {
                "year": year
            }
        )


        if returning_records is None:

            print(
                "FAILED to download returning data"
            )

            failed_returning_years.append(year)

            returning_records = []

        else:

            api_calls += 1

            with open(
                returning_file,
                "w"
            ) as file:

                json.dump(
                    returning_records,
                    file,
                    indent=2
                )

            print(
                f"Saved raw file: "
                f"{returning_file}"
            )

            time.sleep(0.20)


    print(
        f"Returning records returned: "
        f"{len(returning_records)}"
    )


    season_returning_processed = 0


    for record in returning_records:

        record_year = record.get(
            "season",
            year
        )

        team = record.get("team")

        if not team:
            continue


        cursor.execute(
            """
            INSERT INTO returning_production (

                season,
                team,
                conference,

                total_ppa,
                total_passing_ppa,
                total_receiving_ppa,
                total_rushing_ppa,

                percent_ppa,
                percent_passing_ppa,
                percent_receiving_ppa,
                percent_rushing_ppa,

                usage,
                passing_usage,
                receiving_usage,
                rushing_usage

            )

            VALUES (
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )

            ON CONFLICT(season, team)
            DO UPDATE SET

                conference =
                    excluded.conference,

                total_ppa =
                    excluded.total_ppa,

                total_passing_ppa =
                    excluded.total_passing_ppa,

                total_receiving_ppa =
                    excluded.total_receiving_ppa,

                total_rushing_ppa =
                    excluded.total_rushing_ppa,

                percent_ppa =
                    excluded.percent_ppa,

                percent_passing_ppa =
                    excluded.percent_passing_ppa,

                percent_receiving_ppa =
                    excluded.percent_receiving_ppa,

                percent_rushing_ppa =
                    excluded.percent_rushing_ppa,

                usage =
                    excluded.usage,

                passing_usage =
                    excluded.passing_usage,

                receiving_usage =
                    excluded.receiving_usage,

                rushing_usage =
                    excluded.rushing_usage
            """,

            (
                record_year,
                team,
                record.get("conference"),

                record.get("totalPPA"),
                record.get("totalPassingPPA"),
                record.get("totalReceivingPPA"),
                record.get("totalRushingPPA"),

                record.get("percentPPA"),
                record.get("percentPassingPPA"),
                record.get("percentReceivingPPA"),
                record.get("percentRushingPPA"),

                record.get("usage"),
                record.get("passingUsage"),
                record.get("receivingUsage"),
                record.get("rushingUsage")
            )
        )


        season_returning_processed += 1
        returning_rows_processed += 1


    connection.commit()


    print(
        f"Returning rows processed: "
        f"{season_returning_processed}"
    )


    # ==================================================
    # SEASON COVERAGE AUDIT
    #
    # We only care about FBS teams that actually
    # participated in at least one completed game.
    # ==================================================

    cursor.execute(
        """
        WITH fbs_teams AS (

            SELECT home_team AS team

            FROM games

            WHERE
                season = ?
                AND home_classification = 'fbs'
                AND home_points IS NOT NULL
                AND away_points IS NOT NULL

            UNION

            SELECT away_team AS team

            FROM games

            WHERE
                season = ?
                AND away_classification = 'fbs'
                AND home_points IS NOT NULL
                AND away_points IS NOT NULL
        )

        SELECT team
        FROM fbs_teams
        ORDER BY team
        """,
        (
            year,
            year
        )
    )


    fbs_teams = [
        row[0]
        for row in cursor.fetchall()
    ]


    # --------------------------------------------------
    # MISSING TALENT
    # --------------------------------------------------

    missing_talent = []


    for team in fbs_teams:

        cursor.execute(
            """
            SELECT 1

            FROM team_talent

            WHERE
                season = ?
                AND team = ?
            """,
            (
                year,
                team
            )
        )

        if cursor.fetchone() is None:

            missing_talent.append(team)


    # --------------------------------------------------
    # MISSING RETURNING
    # --------------------------------------------------

    missing_returning = []


    for team in fbs_teams:

        cursor.execute(
            """
            SELECT 1

            FROM returning_production

            WHERE
                season = ?
                AND team = ?
            """,
            (
                year,
                team
            )
        )

        if cursor.fetchone() is None:

            missing_returning.append(team)


    print("\nSEASON CONTEXT AUDIT")
    print("-" * 80)

    print(
        f"FBS teams in completed games: "
        f"{len(fbs_teams)}"
    )

    print(
        f"Missing talent records:       "
        f"{len(missing_talent)}"
    )

    if missing_talent:

        for team in missing_talent:

            print(
                f"  TALENT MISSING: "
                f"{team}"
            )


    print(
        f"Missing returning records:    "
        f"{len(missing_returning)}"
    )

    if missing_returning:

        for team in missing_returning:

            print(
                f"  RETURNING MISSING: "
                f"{team}"
            )


# --------------------------------------------------
# FINAL COVERAGE SUMMARY
# --------------------------------------------------

print("\n\n")
print("HISTORICAL CONTEXT COVERAGE")
print("=" * 80)


for year in range(
    start_year,
    2026
):

    cursor.execute(
        """
        WITH fbs_teams AS (

            SELECT home_team AS team

            FROM games

            WHERE
                season = ?
                AND home_classification = 'fbs'
                AND home_points IS NOT NULL
                AND away_points IS NOT NULL

            UNION

            SELECT away_team AS team

            FROM games

            WHERE
                season = ?
                AND away_classification = 'fbs'
                AND home_points IS NOT NULL
                AND away_points IS NOT NULL
        )

        SELECT COUNT(*)
        FROM fbs_teams
        """,
        (
            year,
            year
        )
    )

    fbs_count = cursor.fetchone()[0]


    cursor.execute(
        """
        SELECT COUNT(*)

        FROM team_talent

        WHERE season = ?
        """,
        (year,)
    )

    talent_count = cursor.fetchone()[0]


    cursor.execute(
        """
        SELECT COUNT(*)

        FROM returning_production

        WHERE season = ?
        """,
        (year,)
    )

    returning_count = cursor.fetchone()[0]


    marker = ""

    if year == 2025:
        marker = "  <-- HOLDOUT"


    print(
        f"{year}: "
        f"{fbs_count} FBS teams | "
        f"{talent_count} talent | "
        f"{returning_count} returning"
        f"{marker}"
    )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nSUMMARY")
print("=" * 80)

print(
    f"API calls made:               "
    f"{api_calls}"
)

print(
    f"Cached talent files used:     "
    f"{cached_talent_files}"
)

print(
    f"Cached returning files used:  "
    f"{cached_returning_files}"
)

print(
    f"Talent rows processed:        "
    f"{talent_rows_processed:,}"
)

print(
    f"Returning rows processed:     "
    f"{returning_rows_processed:,}"
)

print(
    f"Failed talent years:          "
    f"{len(failed_talent_years)}"
)

print(
    f"Failed returning years:       "
    f"{len(failed_returning_years)}"
)


if failed_talent_years:

    print(
        "\nFAILED TALENT YEARS"
    )

    for year in failed_talent_years:

        print(year)


if failed_returning_years:

    print(
        "\nFAILED RETURNING YEARS"
    )

    for year in failed_returning_years:

        print(year)


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

connection.close()