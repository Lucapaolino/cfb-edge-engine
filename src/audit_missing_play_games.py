import sqlite3
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

database_file = "data/cfb.db"


# --------------------------------------------------
# CONNECT
# --------------------------------------------------

connection = sqlite3.connect(
    database_file
)


print("PLAY-BY-PLAY COVERAGE AUDIT")
print("=" * 100)


# --------------------------------------------------
# COMPLETED GAMES WITH ZERO PLAYS
# --------------------------------------------------

missing_query = """
SELECT

    g.game_id,
    g.season,
    g.week,
    g.start_date,

    g.away_team,
    g.away_classification,
    g.away_points,

    g.home_team,
    g.home_classification,
    g.home_points,

    COUNT(p.play_id) AS play_count

FROM games AS g

LEFT JOIN plays AS p
    ON g.game_id = p.game_id

WHERE
    g.home_points IS NOT NULL
    AND g.away_points IS NOT NULL

GROUP BY
    g.game_id

HAVING
    COUNT(p.play_id) = 0

ORDER BY
    g.season,
    g.week,
    g.start_date
"""


missing = pd.read_sql_query(
    missing_query,
    connection
)


print("\nCOMPLETED GAMES WITH NO PLAY-BY-PLAY")
print("-" * 100)

print(
    f"Count: "
    f"{len(missing):,}"
)


if len(missing) > 0:

    print(
        missing.to_string(
            index=False
        )
    )

else:

    print(
        "None"
    )


# --------------------------------------------------
# FBS-V-FBS MISSING ONLY
# --------------------------------------------------

missing_fbs = missing[
    (
        missing[
            "home_classification"
        ] == "fbs"
    )
    &
    (
        missing[
            "away_classification"
        ] == "fbs"
    )
].copy()


print("\n")
print("FBS-V-FBS GAMES WITH NO PLAY-BY-PLAY")
print("-" * 100)

print(
    f"Count: "
    f"{len(missing_fbs):,}"
)


if len(missing_fbs) > 0:

    print(
        missing_fbs.to_string(
            index=False
        )
    )

else:

    print(
        "None"
    )


# --------------------------------------------------
# SEASON COVERAGE
# --------------------------------------------------

coverage_query = """
SELECT

    g.season,

    COUNT(DISTINCT g.game_id)
        AS completed_fbs_games,

    COUNT(
        DISTINCT CASE
            WHEN p.game_id IS NOT NULL
            THEN g.game_id
        END
    )
        AS games_with_plays

FROM games AS g

LEFT JOIN plays AS p
    ON g.game_id = p.game_id

WHERE
    g.home_points IS NOT NULL
    AND g.away_points IS NOT NULL

    AND g.home_classification = 'fbs'
    AND g.away_classification = 'fbs'

GROUP BY
    g.season

ORDER BY
    g.season
"""


coverage = pd.read_sql_query(
    coverage_query,
    connection
)


coverage[
    "missing_games"
] = (
    coverage[
        "completed_fbs_games"
    ]
    -
    coverage[
        "games_with_plays"
    ]
)


coverage[
    "coverage_pct"
] = (
    coverage[
        "games_with_plays"
    ]
    /
    coverage[
        "completed_fbs_games"
    ]
    * 100
)


print("\n")
print("FBS-V-FBS PLAY COVERAGE BY SEASON")
print("-" * 100)

print(
    coverage.to_string(
        index=False,
        formatters={
            "coverage_pct":
                lambda value:
                f"{value:.2f}%"
        }
    )
)


# --------------------------------------------------
# PPA COVERAGE BY SEASON
# --------------------------------------------------

ppa_query = """
SELECT

    g.season,

    COUNT(p.play_id)
        AS total_plays,

    SUM(
        CASE
            WHEN p.ppa IS NOT NULL
            THEN 1
            ELSE 0
        END
    )
        AS plays_with_ppa

FROM games AS g

JOIN plays AS p
    ON g.game_id = p.game_id

WHERE
    g.home_points IS NOT NULL
    AND g.away_points IS NOT NULL

    AND g.home_classification = 'fbs'
    AND g.away_classification = 'fbs'

GROUP BY
    g.season

ORDER BY
    g.season
"""


ppa = pd.read_sql_query(
    ppa_query,
    connection
)


ppa[
    "ppa_pct"
] = (
    ppa[
        "plays_with_ppa"
    ]
    /
    ppa[
        "total_plays"
    ]
    * 100
)


print("\n")
print("FBS-V-FBS PPA COVERAGE")
print("-" * 100)

print(
    ppa.to_string(
        index=False,
        formatters={
            "ppa_pct":
                lambda value:
                f"{value:.2f}%"
        }
    )
)


# --------------------------------------------------
# SUSPICIOUSLY LOW PLAY COUNTS
#
# A completed FBS game normally contains far more
# than 50 plays. This lets us detect games where the
# API may contain only a partial play log.
# --------------------------------------------------

low_play_query = """
SELECT

    g.game_id,
    g.season,
    g.week,

    g.away_team,
    g.home_team,

    COUNT(p.play_id)
        AS play_count,

    SUM(
        CASE
            WHEN p.ppa IS NOT NULL
            THEN 1
            ELSE 0
        END
    )
        AS ppa_count

FROM games AS g

JOIN plays AS p
    ON g.game_id = p.game_id

WHERE
    g.home_points IS NOT NULL
    AND g.away_points IS NOT NULL

    AND g.home_classification = 'fbs'
    AND g.away_classification = 'fbs'

GROUP BY
    g.game_id

HAVING
    COUNT(p.play_id) < 50

ORDER BY
    g.season,
    g.week
"""


low_play = pd.read_sql_query(
    low_play_query,
    connection
)


print("\n")
print("FBS-V-FBS GAMES WITH FEWER THAN 50 PLAYS")
print("-" * 100)

print(
    f"Count: "
    f"{len(low_play):,}"
)


if len(low_play) > 0:

    print(
        low_play.to_string(
            index=False
        )
    )

else:

    print(
        "None"
    )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n")
print("SUMMARY")
print("=" * 100)

print(
    f"Completed games missing plays: "
    f"{len(missing):,}"
)

print(
    f"FBS-v-FBS missing plays:       "
    f"{len(missing_fbs):,}"
)

print(
    f"Suspicious low-play FBS games: "
    f"{len(low_play):,}"
)


connection.close()