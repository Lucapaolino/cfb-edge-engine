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

API_URL = (
    "https://api.collegefootballdata.com/"
    "games/teams"
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
    "team_stats"
)


# ============================================================
# API KEY
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


# ============================================================
# HELPERS
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )


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
    "UPDATE LIVE 2026 TEAM STATS — V1"
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
# DATABASE
# ============================================================

if not DATABASE_FILE.exists():

    raise RuntimeError(
        f"Database does not exist: "
        f"{DATABASE_FILE}"
    )


connection = sqlite3.connect(
    DATABASE_FILE
)


# ============================================================
# VERIFY TABLES
# ============================================================

required_tables = {

    "games",

    "team_game_stats",

    "team_game_stat_values"
}


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


missing_tables = (

    required_tables

    -
    existing_tables
)


if missing_tables:

    connection.close()

    raise RuntimeError(

        "Missing required database tables:\n"

        +

        "\n".join(
            sorted(
                missing_tables
            )
        )
    )


# ============================================================
# FIND COMPLETED 2026 WEEKS
#
# We do NOT blindly request the entire season.
#
# Only weeks with at least one completed game need team stats.
# ============================================================

completed_weeks = connection.execute(
    """
    SELECT DISTINCT week

    FROM games

    WHERE
        season = ?
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
        AND week IS NOT NULL

    ORDER BY week
    """,

    (
        SEASON,
    )
).fetchall()


completed_weeks = [

    int(
        row[0]
    )

    for row in completed_weeks
]


if len(
    completed_weeks
) == 0:

    connection.close()

    print()

    print(
        "No completed 2026 games yet."
    )

    raise SystemExit(
        0
    )


print()

print(
    "Completed weeks detected:"
)


print(
    "  "
    +
    ", ".join(
        str(week)
        for week
        in completed_weeks
    )
)


# ============================================================
# COMPLETED GAME IDs
#
# Only these games may be imported.
#
# This protects us if the API happens to return a box score
# structure for a game that has not actually been completed.
# ============================================================

completed_game_rows = connection.execute(
    """
    SELECT
        game_id,
        week

    FROM games

    WHERE
        season = ?
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
    """,

    (
        SEASON,
    )
).fetchall()


completed_game_ids = {

    int(
        game_id
    )

    for (
        game_id,
        week
    ) in completed_game_rows
}


print(
    f"Completed games eligible: "
    f"{len(completed_game_ids):,}"
)


# ============================================================
# RAW DIRECTORY
# ============================================================

RAW_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STORAGE
# ============================================================

team_rows_by_key = {}

stat_rows_by_key = {}


api_game_ids = set()


request_count = 0


# ============================================================
# DOWNLOAD EACH COMPLETED WEEK
#
# CFBD requires week/team/conference when filtering team
# box scores by year, so week-by-week is the cleanest method.
# ============================================================

for week in completed_weeks:

    print()

    print(
        "=" * 110
    )

    print(
        f"DOWNLOADING WEEK {week}"
    )

    print(
        "=" * 110
    )


    response = requests.get(

        API_URL,

        params={

            "year":
                SEASON,

            "week":
                week,

            "classification":
                "fbs",

            "seasonType":
                "regular"
        },

        headers={

            "Authorization":
                f"Bearer {API_KEY}"
        },

        timeout=60
    )


    request_count += 1


    print(
        f"HTTP status: "
        f"{response.status_code}"
    )


    if response.status_code != 200:

        connection.close()

        raise RuntimeError(

            f"CFBD Week {week} request failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:2000]}"
        )


    payload = response.json()


    if not isinstance(
        payload,
        list
    ):

        connection.close()

        raise RuntimeError(

            f"Unexpected API response "
            f"for Week {week}."
        )


    print(
        f"Game box scores returned: "
        f"{len(payload):,}"
    )


    # ========================================================
    # SAVE RAW WEEK RESPONSE
    # ========================================================

    raw_file = (

        RAW_DIRECTORY

        /

        (
            f"team_stats_{SEASON}_"
            f"week_{week}_"
            f"{batch_id}.json"
        )
    )


    with open(
        raw_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            payload,
            file,
            indent=2
        )


    print(
        f"Raw response: {raw_file}"
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    week_team_rows = 0

    week_stat_rows = 0

    skipped_uncompleted = 0


    for game in payload:

        game_id = game.get(
            "id"
        )


        if game_id is None:

            continue


        try:

            game_id = int(
                game_id
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        # ----------------------------------------------------
        # Only completed games from our local schedule.
        # ----------------------------------------------------

        if (
            game_id
            not in completed_game_ids
        ):

            skipped_uncompleted += 1

            continue


        api_game_ids.add(
            game_id
        )


        teams = game.get(
            "teams",
            []
        )


        if not isinstance(
            teams,
            list
        ):

            continue


        for team_entry in teams:

            team_name = team_entry.get(
                "team"
            )


            if not team_name:

                continue


            team_id = team_entry.get(
                "teamId"
            )


            conference = team_entry.get(
                "conference"
            )


            home_away = team_entry.get(
                "homeAway"
            )


            points = team_entry.get(
                "points"
            )


            # =================================================
            # TEAM BASE ROW
            # =================================================

            team_key = (

                game_id,

                team_name
            )


            team_rows_by_key[
                team_key
            ] = (

                game_id,

                team_id,

                team_name,

                conference,

                home_away,

                points
            )


            week_team_rows += 1


            # =================================================
            # STAT VALUE ROWS
            # =================================================

            stats = team_entry.get(
                "stats",
                []
            )


            if not isinstance(
                stats,
                list
            ):

                continue


            for stat_entry in stats:

                category = stat_entry.get(
                    "category"
                )


                stat_value = stat_entry.get(
                    "stat"
                )


                if not category:

                    continue


                stat_key = (

                    game_id,

                    team_name,

                    category
                )


                # Store as TEXT to match historical schema.
                if stat_value is None:

                    stored_value = None


                else:

                    stored_value = str(
                        stat_value
                    )


                stat_rows_by_key[
                    stat_key
                ] = (

                    game_id,

                    team_name,

                    category,

                    stored_value
                )


                week_stat_rows += 1


    print(
        f"Normalized team rows: "
        f"{week_team_rows:,}"
    )


    print(
        f"Normalized stat rows: "
        f"{week_stat_rows:,}"
    )


    if skipped_uncompleted > 0:

        print(
            f"API games skipped because "
            f"local game not completed: "
            f"{skipped_uncompleted:,}"
        )


# ============================================================
# FINAL NORMALIZED DATA
# ============================================================

team_rows = list(
    team_rows_by_key.values()
)


stat_rows = list(
    stat_rows_by_key.values()
)


print()

print(
    "=" * 110
)

print(
    "NORMALIZED 2026 DATA"
)

print(
    "=" * 110
)


print(
    f"API requests made:       "
    f"{request_count:,}"
)


print(
    f"Unique games returned:   "
    f"{len(api_game_ids):,}"
)


print(
    f"Unique team rows:        "
    f"{len(team_rows):,}"
)


print(
    f"Unique stat-value rows:  "
    f"{len(stat_rows):,}"
)


# ============================================================
# UPSERT TEAM_GAME_STATS
#
# Composite PK:
#
# game_id + team
#
# Existing historical data is untouched.
# Rerunning this script safely refreshes current-season data.
# ============================================================

connection.executemany(
    """
    INSERT INTO team_game_stats (

        game_id,

        team_id,

        team,

        conference,

        home_away,

        points
    )

    VALUES (
        ?,
        ?,
        ?,
        ?,
        ?,
        ?
    )

    ON CONFLICT(
        game_id,
        team
    )

    DO UPDATE SET

        team_id =
            COALESCE(
                excluded.team_id,
                team_game_stats.team_id
            ),

        conference =
            COALESCE(
                excluded.conference,
                team_game_stats.conference
            ),

        home_away =
            COALESCE(
                excluded.home_away,
                team_game_stats.home_away
            ),

        points =
            COALESCE(
                excluded.points,
                team_game_stats.points
            )
    """,

    team_rows
)


# ============================================================
# UPSERT TEAM_GAME_STAT_VALUES
#
# Composite PK:
#
# game_id + team + category
# ============================================================

connection.executemany(
    """
    INSERT INTO team_game_stat_values (

        game_id,

        team,

        category,

        stat_value
    )

    VALUES (
        ?,
        ?,
        ?,
        ?
    )

    ON CONFLICT(
        game_id,
        team,
        category
    )

    DO UPDATE SET

        stat_value =
            COALESCE(
                excluded.stat_value,
                team_game_stat_values.stat_value
            )
    """,

    stat_rows
)


connection.commit()


# ============================================================
# 2026 TEAM ROW COUNT
#
# There is no season column in these tables, so we identify
# 2026 records by joining back to games.
# ============================================================

team_rows_2026 = connection.execute(
    """
    SELECT COUNT(*)

    FROM team_game_stats AS t

    INNER JOIN games AS g
        ON g.game_id = t.game_id

    WHERE g.season = ?
    """,

    (
        SEASON,
    )
).fetchone()[0]


stat_rows_2026 = connection.execute(
    """
    SELECT COUNT(*)

    FROM team_game_stat_values AS t

    INNER JOIN games AS g
        ON g.game_id = t.game_id

    WHERE g.season = ?
    """,

    (
        SEASON,
    )
).fetchone()[0]


# ============================================================
# GAME COVERAGE
#
# A normal completed game should have two team rows.
# ============================================================

covered_games = connection.execute(
    """
    SELECT COUNT(*)

    FROM (

        SELECT
            g.game_id

        FROM games AS g

        INNER JOIN team_game_stats AS t
            ON t.game_id = g.game_id

        WHERE
            g.season = ?
            AND g.home_points IS NOT NULL
            AND g.away_points IS NOT NULL

        GROUP BY g.game_id

        HAVING COUNT(
            DISTINCT t.team
        ) >= 2
    )
    """,

    (
        SEASON,
    )
).fetchone()[0]


completed_games = len(
    completed_game_ids
)


missing_games = (

    completed_games

    -
    covered_games
)


# ============================================================
# STAT COVERAGE
# ============================================================

games_with_stats = connection.execute(
    """
    SELECT COUNT(
        DISTINCT g.game_id
    )

    FROM games AS g

    INNER JOIN team_game_stat_values AS t
        ON t.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
    """,

    (
        SEASON,
    )
).fetchone()[0]


# ============================================================
# WEEK COVERAGE
# ============================================================

week_coverage = connection.execute(
    """
    SELECT
        g.week,

        COUNT(
            DISTINCT g.game_id
        ) AS completed_games,

        COUNT(
            DISTINCT CASE
                WHEN t.game_id IS NOT NULL
                THEN g.game_id
            END
        ) AS games_with_team_rows

    FROM games AS g

    LEFT JOIN team_game_stats AS t
        ON t.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL

    GROUP BY g.week

    ORDER BY g.week
    """,

    (
        SEASON,
    )
).fetchall()


# ============================================================
# DATABASE RESULT
# ============================================================

print()

print(
    "=" * 110
)

print(
    "2026 TEAM-STATS DATABASE STATUS"
)

print(
    "=" * 110
)


print(
    f"Completed games:         "
    f"{completed_games:,}"
)


print(
    f"Games with 2 team rows: "
    f"{covered_games:,}"
)


print(
    f"Games with stat values: "
    f"{games_with_stats:,}"
)


print(
    f"Missing completed games: "
    f"{missing_games:,}"
)


print(
    f"2026 team rows:          "
    f"{team_rows_2026:,}"
)


print(
    f"2026 stat-value rows:    "
    f"{stat_rows_2026:,}"
)


# ============================================================
# WEEK SUMMARY
# ============================================================

print()

print(
    "WEEK COVERAGE"
)

print(
    "-" * 110
)


for (
    week,
    week_completed,
    week_covered
) in week_coverage:

    print(

        f"Week {week}: "

        f"{week_covered:,}/"
        f"{week_completed:,} "

        f"games with team rows"
    )


# ============================================================
# FIND MISSING GAMES
# ============================================================

missing_game_rows = connection.execute(
    """
    SELECT
        g.game_id,
        g.week,
        g.away_team,
        g.home_team

    FROM games AS g

    LEFT JOIN (

        SELECT
            game_id,

            COUNT(
                DISTINCT team
            ) AS team_count

        FROM team_game_stats

        GROUP BY game_id

    ) AS coverage

        ON coverage.game_id = g.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
        AND COALESCE(
            coverage.team_count,
            0
        ) < 2

    ORDER BY
        g.week,
        g.start_date
    """,

    (
        SEASON,
    )
).fetchall()


if len(
    missing_game_rows
) > 0:

    print()

    print(
        "MISSING COMPLETED GAMES"
    )

    print(
        "-" * 110
    )


    for (
        game_id,
        week,
        away_team,
        home_team
    ) in missing_game_rows:

        print(
            f"{game_id} | "
            f"W{week} | "
            f"{away_team} @ "
            f"{home_team}"
        )


# ============================================================
# SAMPLE STATS
# ============================================================

sample_stats = connection.execute(
    """
    SELECT
        g.week,
        t.team,
        t.home_away,
        t.points,
        COUNT(
            s.category
        ) AS stat_categories

    FROM team_game_stats AS t

    INNER JOIN games AS g
        ON g.game_id = t.game_id

    LEFT JOIN team_game_stat_values AS s
        ON
            s.game_id = t.game_id
            AND s.team = t.team

    WHERE g.season = ?

    GROUP BY
        g.week,
        t.game_id,
        t.team,
        t.home_away,
        t.points

    ORDER BY
        g.start_date,
        t.home_away

    LIMIT 12
    """,

    (
        SEASON,
    )
).fetchall()


print()

print(
    "2026 SAMPLE"
)

print(
    "-" * 110
)


for (
    week,
    team,
    home_away,
    points,
    categories
) in sample_stats:

    print(

        f"W{week} | "

        f"{team:30} | "

        f"{home_away:4} | "

        f"{points:3} pts | "

        f"{categories} stat categories"
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
    "LIVE 2026 TEAM STATS UPDATE COMPLETE"
)

print(
    "=" * 110
)


print()

print(
    "Next prerequisite:"
)


print(
    "  Update 2026 passing / QB data."
)