import sqlite3
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

database_file = "data/cfb.db"

game_id = 401643737


# --------------------------------------------------
# CONNECT
# --------------------------------------------------

connection = sqlite3.connect(
    database_file
)


print("PLAY-BY-PLAY ANOMALY INVESTIGATION")
print("=" * 100)


# --------------------------------------------------
# GAME INFORMATION
# --------------------------------------------------

game_query = """
SELECT
    game_id,
    season,
    week,
    start_date,

    away_team,
    away_points,

    home_team,
    home_points

FROM games

WHERE game_id = ?
"""


game = pd.read_sql_query(
    game_query,
    connection,
    params=(game_id,)
)


print("\nGAME")
print("-" * 100)

print(
    game.to_string(
        index=False
    )
)


# --------------------------------------------------
# TOTAL PLAY COUNTS
# --------------------------------------------------

summary_query = """
SELECT

    COUNT(*) AS total_plays,

    SUM(
        CASE
            WHEN ppa IS NOT NULL
            THEN 1
            ELSE 0
        END
    ) AS plays_with_ppa,

    MIN(period) AS first_period,

    MAX(period) AS last_period,

    MIN(play_number) AS first_play_number,

    MAX(play_number) AS last_play_number

FROM plays

WHERE game_id = ?
"""


summary = pd.read_sql_query(
    summary_query,
    connection,
    params=(game_id,)
)


print("\nPLAY SUMMARY")
print("-" * 100)

print(
    summary.to_string(
        index=False
    )
)


# --------------------------------------------------
# PLAYS BY OFFENSE
# --------------------------------------------------

offense_query = """
SELECT

    offense,

    COUNT(*) AS plays,

    SUM(
        CASE
            WHEN ppa IS NOT NULL
            THEN 1
            ELSE 0
        END
    ) AS ppa_plays,

    AVG(ppa) AS average_ppa

FROM plays

WHERE game_id = ?

GROUP BY offense

ORDER BY plays DESC
"""


offense = pd.read_sql_query(
    offense_query,
    connection,
    params=(game_id,)
)


print("\nPLAYS BY OFFENSE")
print("-" * 100)

print(
    offense.to_string(
        index=False
    )
)


# --------------------------------------------------
# PLAYS BY PERIOD
# --------------------------------------------------

period_query = """
SELECT

    period,

    COUNT(*) AS plays,

    SUM(
        CASE
            WHEN ppa IS NOT NULL
            THEN 1
            ELSE 0
        END
    ) AS ppa_plays

FROM plays

WHERE game_id = ?

GROUP BY period

ORDER BY period
"""


periods = pd.read_sql_query(
    period_query,
    connection,
    params=(game_id,)
)


print("\nPLAYS BY PERIOD")
print("-" * 100)

print(
    periods.to_string(
        index=False
    )
)


# --------------------------------------------------
# PLAY TYPES
# --------------------------------------------------

type_query = """
SELECT

    play_type,

    COUNT(*) AS count

FROM plays

WHERE game_id = ?

GROUP BY play_type

ORDER BY count DESC
"""


play_types = pd.read_sql_query(
    type_query,
    connection,
    params=(game_id,)
)


print("\nPLAY TYPES")
print("-" * 100)

print(
    play_types.to_string(
        index=False
    )
)


# --------------------------------------------------
# FIRST 15 PLAYS
# --------------------------------------------------

first_query = """
SELECT

    period,
    clock_minutes,
    clock_seconds,

    offense,
    defense,

    down,
    distance,

    yards_gained,

    play_type,
    ppa,

    play_text

FROM plays

WHERE game_id = ?

ORDER BY
    period,
    play_number

LIMIT 15
"""


first_plays = pd.read_sql_query(
    first_query,
    connection,
    params=(game_id,)
)


print("\nFIRST 15 RECORDED PLAYS")
print("-" * 100)

print(
    first_plays.to_string(
        index=False
    )
)


# --------------------------------------------------
# LAST 15 PLAYS
# --------------------------------------------------

last_query = """
SELECT

    period,
    clock_minutes,
    clock_seconds,

    offense,
    defense,

    down,
    distance,

    yards_gained,

    play_type,
    ppa,

    play_text

FROM plays

WHERE game_id = ?

ORDER BY
    period DESC,
    play_number DESC

LIMIT 15
"""


last_plays = pd.read_sql_query(
    last_query,
    connection,
    params=(game_id,)
)


print("\nLAST 15 RECORDED PLAYS")
print("-" * 100)

print(
    last_plays.to_string(
        index=False
    )
)


# --------------------------------------------------
# TEAM GAME STAT VALUES
#
# This lets us compare the partial play log against
# the box score information we already downloaded.
# --------------------------------------------------

stats_query = """
SELECT

    team,
    category,
    stat_value

FROM team_game_stat_values

WHERE game_id = ?

ORDER BY
    team,
    category
"""


stats = pd.read_sql_query(
    stats_query,
    connection,
    params=(game_id,)
)


print("\nTEAM BOX-SCORE STATISTICS")
print("-" * 100)

print(
    stats.to_string(
        index=False
    )
)


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

connection.close()