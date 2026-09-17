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

talent_file = f"data/raw/team_talent_{year}.json"
returning_file = f"data/raw/returning_production_{year}.json"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}


# --------------------------------------------------
# DOWNLOAD TEAM TALENT
# --------------------------------------------------

talent_response = requests.get(
    "https://api.collegefootballdata.com/talent",
    headers=headers,
    params={
        "year": year
    },
    timeout=60
)

talent_response.raise_for_status()

talent_records = talent_response.json()

with open(talent_file, "w") as file:
    json.dump(
        talent_records,
        file,
        indent=2
    )

print(
    f"Team talent downloaded: "
    f"{len(talent_records)} teams"
)


# --------------------------------------------------
# DOWNLOAD RETURNING PRODUCTION
# --------------------------------------------------

returning_response = requests.get(
    "https://api.collegefootballdata.com/player/returning",
    headers=headers,
    params={
        "year": year
    },
    timeout=60
)

returning_response.raise_for_status()

returning_records = returning_response.json()

with open(returning_file, "w") as file:
    json.dump(
        returning_records,
        file,
        indent=2
    )

print(
    f"Returning production downloaded: "
    f"{len(returning_records)} teams"
)


# --------------------------------------------------
# CONNECT TO DATABASE
# --------------------------------------------------

connection = sqlite3.connect(
    database_file
)

cursor = connection.cursor()


# --------------------------------------------------
# IMPORT TEAM TALENT
# --------------------------------------------------

talent_processed = 0

for record in talent_records:

    cursor.execute(
        """
        INSERT OR REPLACE INTO team_talent (

            season,
            team,
            talent

        )

        VALUES (?, ?, ?)
        """,

        (
            record.get("year"),
            record.get("team"),
            record.get("talent")
        )
    )

    talent_processed += 1


# --------------------------------------------------
# IMPORT RETURNING PRODUCTION
# --------------------------------------------------

returning_processed = 0

for record in returning_records:

    cursor.execute(
        """
        INSERT OR REPLACE INTO returning_production (

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
        """,

        (
            record.get("season"),
            record.get("team"),
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

    returning_processed += 1


# --------------------------------------------------
# SAVE DATABASE
# --------------------------------------------------

connection.commit()


# --------------------------------------------------
# VERIFY COUNTS
# --------------------------------------------------

cursor.execute("""
SELECT COUNT(*)
FROM team_talent
WHERE season = ?
""", (year,))

talent_database_count = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(*)
FROM returning_production
WHERE season = ?
""", (year,))

returning_database_count = cursor.fetchone()[0]


print("\nDATABASE RESULTS")
print("-" * 70)

print(
    f"Talent records stored: "
    f"{talent_database_count}"
)

print(
    f"Returning production records stored: "
    f"{returning_database_count}"
)


# --------------------------------------------------
# TEST A MULTI-TABLE JOIN FOR ALABAMA
# --------------------------------------------------

cursor.execute("""
SELECT

    tt.team,
    tt.talent,

    rp.percent_ppa,
    rp.percent_passing_ppa,
    rp.percent_receiving_ppa,
    rp.percent_rushing_ppa,

    rp.usage,
    rp.passing_usage,
    rp.receiving_usage,
    rp.rushing_usage

FROM team_talent AS tt

JOIN returning_production AS rp

    ON tt.season = rp.season
    AND tt.team = rp.team

WHERE
    tt.season = ?
    AND tt.team = 'Alabama'
""", (year,))


alabama = cursor.fetchone()


print("\nALABAMA 2025 CONTEXT")
print("-" * 70)

if alabama:

    print(f"Team:                  {alabama[0]}")
    print(f"Talent:                {alabama[1]}")
    print(f"Returning PPA:         {alabama[2]}")
    print(f"Returning passing PPA: {alabama[3]}")
    print(f"Returning receiving:   {alabama[4]}")
    print(f"Returning rushing:     {alabama[5]}")
    print(f"Returning usage:       {alabama[6]}")
    print(f"Passing usage:         {alabama[7]}")
    print(f"Receiving usage:       {alabama[8]}")
    print(f"Rushing usage:         {alabama[9]}")

else:

    print("Alabama record was not found.")


# --------------------------------------------------
# CLOSE DATABASE
# --------------------------------------------------

connection.close()