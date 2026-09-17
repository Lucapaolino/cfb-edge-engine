import sqlite3
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

database_file = "data/cfb.db"

# Confirmed partial play-by-play game.
invalid_game_id = 401643737


# --------------------------------------------------
# CONNECT
# --------------------------------------------------

connection = sqlite3.connect(
    database_file
)


print("PPA PLAY-TYPE AUDIT")
print("=" * 100)

print(
    "Using 2018-2024 FBS-v-FBS games only."
)

print(
    "2025 reserved test data is not used "
    "for this audit."
)


# --------------------------------------------------
# PLAY TYPES
#
# We want to see the exact labels CFBD actually
# uses before deciding what counts as:
#
# - passing
# - rushing
# - other
#
# This prevents us from guessing.
# --------------------------------------------------

query = """
SELECT

    p.play_type,

    COUNT(*) AS total_plays,

    SUM(
        CASE
            WHEN p.ppa IS NOT NULL
            THEN 1
            ELSE 0
        END
    ) AS plays_with_ppa,

    AVG(
        CASE
            WHEN p.ppa IS NOT NULL
            THEN p.ppa
        END
    ) AS average_ppa

FROM plays AS p

JOIN games AS g
    ON p.game_id = g.game_id

WHERE
    g.season BETWEEN 2018 AND 2024

    AND g.home_points IS NOT NULL
    AND g.away_points IS NOT NULL

    AND g.home_classification = 'fbs'
    AND g.away_classification = 'fbs'

    AND g.game_id != ?

GROUP BY
    p.play_type

ORDER BY
    plays_with_ppa DESC,
    total_plays DESC
"""


play_types = pd.read_sql_query(
    query,
    connection,
    params=(invalid_game_id,)
)


play_types[
    "ppa_coverage_pct"
] = (
    play_types[
        "plays_with_ppa"
    ]
    /
    play_types[
        "total_plays"
    ]
    * 100
)


print("\n")
print("ALL PLAY TYPES")
print("-" * 100)


print(
    play_types.to_string(
        index=False,
        formatters={
            "average_ppa":
                lambda value:
                (
                    f"{value:.4f}"
                    if pd.notna(value)
                    else "-"
                ),

            "ppa_coverage_pct":
                lambda value:
                f"{value:.2f}%"
        }
    )
)


# --------------------------------------------------
# TYPES WITH PPA
# --------------------------------------------------

with_ppa = play_types[
    play_types[
        "plays_with_ppa"
    ] > 0
].copy()


print("\n")
print("PLAY TYPES THAT HAVE PPA")
print("-" * 100)


print(
    with_ppa[
        [
            "play_type",
            "total_plays",
            "plays_with_ppa",
            "ppa_coverage_pct"
        ]
    ].to_string(
        index=False,
        formatters={
            "ppa_coverage_pct":
                lambda value:
                f"{value:.2f}%"
        }
    )
)


# --------------------------------------------------
# PLAY TYPES WITHOUT PPA
# --------------------------------------------------

without_ppa = play_types[
    play_types[
        "plays_with_ppa"
    ] == 0
].copy()


print("\n")
print("PLAY TYPES WITH ZERO PPA")
print("-" * 100)


if len(without_ppa) > 0:

    print(
        without_ppa[
            [
                "play_type",
                "total_plays"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "None"
    )


# --------------------------------------------------
# TOTALS
# --------------------------------------------------

total_plays = int(
    play_types[
        "total_plays"
    ].sum()
)


total_ppa = int(
    play_types[
        "plays_with_ppa"
    ].sum()
)


print("\n")
print("SUMMARY")
print("=" * 100)

print(
    f"Distinct play types: "
    f"{len(play_types):,}"
)

print(
    f"Total plays audited: "
    f"{total_plays:,}"
)

print(
    f"Plays with PPA:      "
    f"{total_ppa:,}"
)

print(
    f"Overall PPA rate:    "
    f"{total_ppa / total_plays * 100:.2f}%"
)


connection.close()