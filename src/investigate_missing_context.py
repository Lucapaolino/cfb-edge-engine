import json
import sqlite3
from difflib import get_close_matches


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

year = 2025

database_file = "data/cfb.db"

talent_file = (
    f"data/raw/team_talent_{year}.json"
)

returning_file = (
    f"data/raw/returning_production_{year}.json"
)


# --------------------------------------------------
# LOAD RAW FILES
# --------------------------------------------------

with open(talent_file, "r") as file:
    talent_records = json.load(file)

with open(returning_file, "r") as file:
    returning_records = json.load(file)


talent_teams = {
    record.get("team")
    for record in talent_records
    if record.get("team")
}

returning_teams = {
    record.get("team")
    for record in returning_records
    if record.get("team")
}


# --------------------------------------------------
# CONNECT TO DATABASE
# --------------------------------------------------

connection = sqlite3.connect(
    database_file
)

cursor = connection.cursor()


# --------------------------------------------------
# BUILD FBS TEAM LIST FROM GAMES
# --------------------------------------------------

cursor.execute("""
SELECT DISTINCT

    team,
    conference

FROM (

    SELECT
        home_team AS team,
        home_conference AS conference

    FROM games

    WHERE
        season = ?
        AND home_classification = 'fbs'

    UNION

    SELECT
        away_team AS team,
        away_conference AS conference

    FROM games

    WHERE
        season = ?
        AND away_classification = 'fbs'
)

ORDER BY team
""", (year, year))


fbs_teams = cursor.fetchall()


# --------------------------------------------------
# FIND MISSING TEAMS
# --------------------------------------------------

missing_talent = []

missing_returning = []


for team, conference in fbs_teams:

    if team not in talent_teams:

        missing_talent.append(
            (team, conference)
        )

    if team not in returning_teams:

        missing_returning.append(
            (team, conference)
        )


# --------------------------------------------------
# PRINT TALENT RESULTS
# --------------------------------------------------

print("MISSING TEAM TALENT INVESTIGATION")
print("=" * 80)

print(
    f"Raw talent records: "
    f"{len(talent_records)}"
)

print(
    f"Unique talent teams: "
    f"{len(talent_teams)}"
)

print(
    f"Missing FBS teams: "
    f"{len(missing_talent)}"
)


for team, conference in missing_talent:

    print("\n" + "-" * 80)

    print(
        f"Team:       {team}"
    )

    print(
        f"Conference: {conference}"
    )

    close_matches = get_close_matches(
        team,
        sorted(talent_teams),
        n=5,
        cutoff=0.5
    )

    print(
        f"Closest names in raw talent data:"
    )

    if close_matches:

        for match in close_matches:

            print(
                f"  - {match}"
            )

    else:

        print(
            "  None"
        )


# --------------------------------------------------
# PRINT RETURNING PRODUCTION RESULTS
# --------------------------------------------------

print("\n\n")
print("MISSING RETURNING PRODUCTION INVESTIGATION")
print("=" * 80)

print(
    f"Raw returning records: "
    f"{len(returning_records)}"
)

print(
    f"Unique returning teams: "
    f"{len(returning_teams)}"
)

print(
    f"Missing FBS teams: "
    f"{len(missing_returning)}"
)


for team, conference in missing_returning:

    print("\n" + "-" * 80)

    print(
        f"Team:       {team}"
    )

    print(
        f"Conference: {conference}"
    )

    close_matches = get_close_matches(
        team,
        sorted(returning_teams),
        n=5,
        cutoff=0.5
    )

    print(
        f"Closest names in raw returning data:"
    )

    if close_matches:

        for match in close_matches:

            print(
                f"  - {match}"
            )

    else:

        print(
            "  None"
        )


# --------------------------------------------------
# SHOW GAME CLASSIFICATION / CONFERENCE HISTORY
# FOR EACH MISSING TEAM
# --------------------------------------------------

all_missing_teams = sorted(
    {
        team
        for team, _ in (
            missing_talent
            + missing_returning
        )
    }
)


print("\n\n")
print("DATABASE TEAM DETAILS")
print("=" * 80)


for team in all_missing_teams:

    cursor.execute("""
    SELECT DISTINCT

        CASE
            WHEN home_team = ?
            THEN home_classification
            ELSE away_classification
        END AS classification,

        CASE
            WHEN home_team = ?
            THEN home_conference
            ELSE away_conference
        END AS conference

    FROM games

    WHERE
        season = ?
        AND (
            home_team = ?
            OR away_team = ?
        )
    """, (
        team,
        team,
        year,
        team,
        team
    ))

    records = cursor.fetchall()

    print(
        f"\n{team}"
    )

    for classification, conference in records:

        print(
            f"  Classification: "
            f"{classification}"
        )

        print(
            f"  Conference:     "
            f"{conference}"
        )


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

connection.close()