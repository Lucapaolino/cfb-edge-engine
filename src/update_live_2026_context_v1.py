import json
import os
import sqlite3

from datetime import datetime, timezone
from pathlib import Path

import requests

from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

SEASON = 2026

BASE_URL = (
    "https://api.collegefootballdata.com"
)

TALENT_URL = (
    f"{BASE_URL}/talent"
)

RETURNING_URL = (
    f"{BASE_URL}/player/returning"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DATABASE_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "cfb.db"
)


RAW_DIRECTORY = (
    PROJECT_ROOT
    /
    "data"
    /
    "raw"
    /
    "live_2026"
    /
    "team_context"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    PROJECT_ROOT
    /
    ".env"
)


API_KEY = os.getenv(
    "CFBD_API_KEY"
)


if not API_KEY:

    raise RuntimeError(
        "CFBD_API_KEY was not found in .env"
    )


HEADERS = {

    "Authorization":
        f"Bearer {API_KEY}"
}


# ============================================================
# HELPERS
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )


def get_value(
    row,
    *names
):

    for name in names:

        if (
            name in row
            and
            row[name] is not None
        ):

            return row[name]


    return None


def numeric_value(
    value
):

    if value is None:

        return None


    try:

        return float(
            value
        )


    except (
        TypeError,
        ValueError
    ):

        return None


def request_json(
    url,
    label
):

    print()

    print(
        "=" * 110
    )

    print(
        f"DOWNLOADING {label}"
    )

    print(
        "=" * 110
    )


    response = requests.get(

        url,

        params={
            "year":
                SEASON
        },

        headers=HEADERS,

        timeout=60
    )


    print(
        f"HTTP status: "
        f"{response.status_code}"
    )


    if response.status_code != 200:

        raise RuntimeError(

            f"{label} request failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:2000]}"
        )


    payload = response.json()


    if not isinstance(
        payload,
        list
    ):

        raise RuntimeError(
            f"{label} returned an unexpected "
            f"response type."
        )


    print(
        f"Rows returned: "
        f"{len(payload):,}"
    )


    return payload


# ============================================================
# START
# ============================================================

captured_datetime = utc_now()


captured_at = (
    captured_datetime
    .isoformat()
)


batch_id = (
    captured_datetime
    .strftime(
        "%Y%m%dT%H%M%SZ"
    )
)


print(
    "UPDATE LIVE 2026 TEAM CONTEXT — V1"
)

print(
    "=" * 110
)


print(
    f"Season:       {SEASON}"
)


print(
    f"Captured UTC: {captured_at}"
)


# ============================================================
# FETCH BOTH DATASETS FIRST
#
# We do not touch the database until both responses have
# successfully returned.
# ============================================================

talent_payload = request_json(
    TALENT_URL,
    "TEAM TALENT"
)


returning_payload = request_json(
    RETURNING_URL,
    "RETURNING PRODUCTION"
)


# ============================================================
# FAIL CLOSED
#
# These are mandatory V1 model inputs.
#
# If the API returns nothing, we do NOT invent values.
# ============================================================

if len(
    talent_payload
) == 0:

    raise RuntimeError(
        "CFBD returned zero 2026 talent rows. "
        "Live model context cannot be completed."
    )


if len(
    returning_payload
) == 0:

    raise RuntimeError(
        "CFBD returned zero 2026 returning-production rows. "
        "Live model context cannot be completed."
    )


# ============================================================
# SAVE RAW RESPONSES
# ============================================================

RAW_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


talent_raw_file = (

    RAW_DIRECTORY

    /

    (
        f"talent_{SEASON}_"
        f"{batch_id}.json"
    )
)


returning_raw_file = (

    RAW_DIRECTORY

    /

    (
        f"returning_production_{SEASON}_"
        f"{batch_id}.json"
    )
)


with open(
    talent_raw_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        talent_payload,
        file,
        indent=2
    )


with open(
    returning_raw_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        returning_payload,
        file,
        indent=2
    )


print()

print(
    f"Talent raw response:"
)

print(
    f"  {talent_raw_file}"
)


print()

print(
    f"Returning raw response:"
)

print(
    f"  {returning_raw_file}"
)


# ============================================================
# NORMALIZE TALENT
#
# Database schema:
#
# season
# team
# talent
# ============================================================

talent_rows_by_team = {}


for row in talent_payload:

    season = get_value(
        row,
        "year",
        "season"
    )


    team = get_value(
        row,
        "team",
        "school"
    )


    talent = numeric_value(
        get_value(
            row,
            "talent"
        )
    )


    if team is None:

        continue


    if season is None:

        season = SEASON


    try:

        season = int(
            season
        )

    except (
        TypeError,
        ValueError
    ):

        continue


    if season != SEASON:

        continue


    team = str(
        team
    ).strip()


    if not team:

        continue


    talent_rows_by_team[
        team
    ] = (

        season,

        team,

        talent
    )


talent_rows = list(
    talent_rows_by_team.values()
)


# ============================================================
# NORMALIZE RETURNING PRODUCTION
#
# Database schema exactly matches our historical table.
# ============================================================

returning_rows_by_team = {}


for row in returning_payload:

    season = get_value(
        row,
        "year",
        "season"
    )


    team = get_value(
        row,
        "team",
        "school"
    )


    if team is None:

        continue


    if season is None:

        season = SEASON


    try:

        season = int(
            season
        )

    except (
        TypeError,
        ValueError
    ):

        continue


    if season != SEASON:

        continue


    team = str(
        team
    ).strip()


    if not team:

        continue


    returning_rows_by_team[
        team
    ] = (

        season,

        team,

        get_value(
            row,
            "conference"
        ),

        numeric_value(
            get_value(
                row,
                "totalPPA",
                "total_ppa"
            )
        ),

        numeric_value(
            get_value(
                row,
                "totalPassingPPA",
                "total_passing_ppa"
            )
        ),

        numeric_value(
            get_value(
                row,
                "totalReceivingPPA",
                "total_receiving_ppa"
            )
        ),

        numeric_value(
            get_value(
                row,
                "totalRushingPPA",
                "total_rushing_ppa"
            )
        ),

        numeric_value(
            get_value(
                row,
                "percentPPA",
                "percent_ppa"
            )
        ),

        numeric_value(
            get_value(
                row,
                "percentPassingPPA",
                "percent_passing_ppa"
            )
        ),

        numeric_value(
            get_value(
                row,
                "percentReceivingPPA",
                "percent_receiving_ppa"
            )
        ),

        numeric_value(
            get_value(
                row,
                "percentRushingPPA",
                "percent_rushing_ppa"
            )
        ),

        numeric_value(
            get_value(
                row,
                "usage"
            )
        ),

        numeric_value(
            get_value(
                row,
                "passingUsage",
                "passing_usage"
            )
        ),

        numeric_value(
            get_value(
                row,
                "receivingUsage",
                "receiving_usage"
            )
        ),

        numeric_value(
            get_value(
                row,
                "rushingUsage",
                "rushing_usage"
            )
        )
    )


returning_rows = list(
    returning_rows_by_team.values()
)


# ============================================================
# NORMALIZATION AUDIT
# ============================================================

print()

print(
    "=" * 110
)

print(
    "NORMALIZED TEAM CONTEXT"
)

print(
    "=" * 110
)


print(
    f"Talent rows:              "
    f"{len(talent_rows):,}"
)


print(
    f"Returning-production rows:"
    f" {len(returning_rows):,}"
)


if len(
    talent_rows
) == 0:

    raise RuntimeError(
        "No valid 2026 talent rows survived normalization."
    )


if len(
    returning_rows
) == 0:

    raise RuntimeError(
        "No valid 2026 returning-production rows "
        "survived normalization."
    )


# ============================================================
# DATABASE
# ============================================================

if not DATABASE_FILE.exists():

    raise RuntimeError(
        f"Database not found: "
        f"{DATABASE_FILE}"
    )


connection = sqlite3.connect(
    DATABASE_FILE
)


try:

    connection.execute(
        "BEGIN"
    )


    # ========================================================
    # VERIFY TABLES
    # ========================================================

    existing_tables = {

        row[0]

        for row in connection.execute(
            """
            SELECT name

            FROM sqlite_master

            WHERE type = 'table'
            """
        ).fetchall()
    }


    for required_table in (

        "team_talent",

        "returning_production",

        "games"
    ):

        if (
            required_table
            not in existing_tables
        ):

            raise RuntimeError(
                f"Required table missing: "
                f"{required_table}"
            )


    # ========================================================
    # UPSERT TEAM TALENT
    # ========================================================

    connection.executemany(
        """
        INSERT INTO team_talent (

            season,
            team,
            talent
        )

        VALUES (
            ?,
            ?,
            ?
        )

        ON CONFLICT(
            season,
            team
        )

        DO UPDATE SET

            talent =
                COALESCE(
                    excluded.talent,
                    team_talent.talent
                )
        """,

        talent_rows
    )


    # ========================================================
    # UPSERT RETURNING PRODUCTION
    # ========================================================

    connection.executemany(
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
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        )

        ON CONFLICT(
            season,
            team
        )

        DO UPDATE SET

            conference =
                COALESCE(
                    excluded.conference,
                    returning_production.conference
                ),

            total_ppa =
                COALESCE(
                    excluded.total_ppa,
                    returning_production.total_ppa
                ),

            total_passing_ppa =
                COALESCE(
                    excluded.total_passing_ppa,
                    returning_production.total_passing_ppa
                ),

            total_receiving_ppa =
                COALESCE(
                    excluded.total_receiving_ppa,
                    returning_production.total_receiving_ppa
                ),

            total_rushing_ppa =
                COALESCE(
                    excluded.total_rushing_ppa,
                    returning_production.total_rushing_ppa
                ),

            percent_ppa =
                COALESCE(
                    excluded.percent_ppa,
                    returning_production.percent_ppa
                ),

            percent_passing_ppa =
                COALESCE(
                    excluded.percent_passing_ppa,
                    returning_production.percent_passing_ppa
                ),

            percent_receiving_ppa =
                COALESCE(
                    excluded.percent_receiving_ppa,
                    returning_production.percent_receiving_ppa
                ),

            percent_rushing_ppa =
                COALESCE(
                    excluded.percent_rushing_ppa,
                    returning_production.percent_rushing_ppa
                ),

            usage =
                COALESCE(
                    excluded.usage,
                    returning_production.usage
                ),

            passing_usage =
                COALESCE(
                    excluded.passing_usage,
                    returning_production.passing_usage
                ),

            receiving_usage =
                COALESCE(
                    excluded.receiving_usage,
                    returning_production.receiving_usage
                ),

            rushing_usage =
                COALESCE(
                    excluded.rushing_usage,
                    returning_production.rushing_usage
                )
        """,

        returning_rows
    )


    connection.commit()


except Exception:

    connection.rollback()

    connection.close()

    raise


# ============================================================
# DATABASE COUNTS
# ============================================================

talent_2026_count = connection.execute(
    """
    SELECT COUNT(*)

    FROM team_talent

    WHERE season = ?
    """,

    (
        SEASON,
    )
).fetchone()[0]


returning_2026_count = connection.execute(
    """
    SELECT COUNT(*)

    FROM returning_production

    WHERE season = ?
    """,

    (
        SEASON,
    )
).fetchone()[0]


# ============================================================
# CURRENT FBS TEAM POPULATION
# ============================================================

fbs_teams = {

    row[0]

    for row in connection.execute(
        """
        SELECT home_team

        FROM games

        WHERE
            season = ?
            AND LOWER(
                home_classification
            ) = 'fbs'

        UNION

        SELECT away_team

        FROM games

        WHERE
            season = ?
            AND LOWER(
                away_classification
            ) = 'fbs'
        """,

        (
            SEASON,
            SEASON
        )
    ).fetchall()

    if row[0]
}


talent_teams = {

    row[0]

    for row in connection.execute(
        """
        SELECT team

        FROM team_talent

        WHERE season = ?
        """,

        (
            SEASON,
        )
    ).fetchall()
}


returning_teams = {

    row[0]

    for row in connection.execute(
        """
        SELECT team

        FROM returning_production

        WHERE season = ?
        """,

        (
            SEASON,
        )
    ).fetchall()
}


missing_talent = sorted(
    fbs_teams
    -
    talent_teams
)


missing_returning = sorted(
    fbs_teams
    -
    returning_teams
)


# ============================================================
# NULL AUDIT FOR MODEL-REQUIRED RETURNING FIELDS
# ============================================================

returning_required_nulls = connection.execute(
    """
    SELECT COUNT(*)

    FROM returning_production

    WHERE
        season = ?
        AND (
            percent_ppa IS NULL
            OR percent_passing_ppa IS NULL
            OR percent_receiving_ppa IS NULL
            OR percent_rushing_ppa IS NULL
        )
    """,

    (
        SEASON,
    )
).fetchone()[0]


talent_nulls = connection.execute(
    """
    SELECT COUNT(*)

    FROM team_talent

    WHERE
        season = ?
        AND talent IS NULL
    """,

    (
        SEASON,
    )
).fetchone()[0]


# ============================================================
# RESULT
# ============================================================

print()

print(
    "=" * 110
)

print(
    "2026 TEAM CONTEXT DATABASE STATUS"
)

print(
    "=" * 110
)


print(
    f"FBS teams in schedule:          "
    f"{len(fbs_teams):,}"
)


print(
    f"2026 talent rows:               "
    f"{talent_2026_count:,}"
)


print(
    f"2026 returning-production rows: "
    f"{returning_2026_count:,}"
)


print(
    f"Talent rows with NULL value:    "
    f"{talent_nulls:,}"
)


print(
    f"Returning rows missing required "
    f"PPA percentages: "
    f"{returning_required_nulls:,}"
)


# ============================================================
# MISSING TEAMS
# ============================================================

print()

print(
    "TALENT COVERAGE"
)

print(
    "-" * 110
)


if not missing_talent:

    print(
        "All scheduled FBS teams have "
        "2026 talent data."
    )


else:

    print(
        f"Missing talent teams: "
        f"{len(missing_talent):,}"
    )


    for team in missing_talent:

        print(
            f"  {team}"
        )


print()

print(
    "RETURNING-PRODUCTION COVERAGE"
)

print(
    "-" * 110
)


if not missing_returning:

    print(
        "All scheduled FBS teams have "
        "2026 returning-production data."
    )


else:

    print(
        f"Missing returning-production teams: "
        f"{len(missing_returning):,}"
    )


    for team in missing_returning:

        print(
            f"  {team}"
        )


# ============================================================
# SAMPLE
# ============================================================

sample = connection.execute(
    """
    SELECT
        t.team,
        t.talent,
        r.percent_ppa,
        r.percent_passing_ppa,
        r.percent_receiving_ppa,
        r.percent_rushing_ppa

    FROM team_talent AS t

    LEFT JOIN returning_production AS r
        ON
            r.season = t.season
            AND r.team = t.team

    WHERE t.season = ?

    ORDER BY t.talent DESC

    LIMIT 15
    """,

    (
        SEASON,
    )
).fetchall()


print()

print(
    "2026 CONTEXT SAMPLE"
)

print(
    "-" * 110
)


for row in sample:

    (
        team,
        talent,
        percent_ppa,
        passing,
        receiving,
        rushing
    ) = row


    print(
        f"{team:30} | "
        f"talent={talent} | "
        f"PPA={percent_ppa} | "
        f"pass={passing} | "
        f"recv={receiving} | "
        f"rush={rushing}"
    )


connection.close()


# ============================================================
# COMPLETE
# ============================================================

print()

print(
    "=" * 110
)

print(
    "LIVE 2026 TEAM CONTEXT UPDATE COMPLETE"
)

print(
    "=" * 110
)


print()

print(
    "Model context now available:"
)


print(
    "  neutral_site"
)


print(
    "  pregame_elo_diff"
)


print(
    "  talent_diff"
)


print(
    "  returning_ppa_diff"
)


print(
    "  returning_passing_ppa_diff"
)


print(
    "  returning_receiving_ppa_diff"
)


print(
    "  returning_rushing_ppa_diff"
)


print()

print(
    "Next step:"
)


print(
    "  Build live 2026 pregame features "
    "for upcoming FBS-v-FBS games."
)