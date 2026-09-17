import os
import sqlite3
import numpy as np
import pandas as pd


# ==================================================
# SETTINGS
# ==================================================

database_file = "data/cfb.db"

v1_file = (
    "data/processed/"
    "features_v1.csv"
)

v2_file = (
    "data/processed/"
    "features_v2.csv"
)


# ==================================================
# KNOWN UNUSABLE PPA GAMES
#
# These games remain in the model dataset.
#
# We simply refuse to use their play-by-play when
# creating PPA features.
# ==================================================

unusable_ppa_games = {

    # ----------------------------------------------
    # No play-by-play available
    # ----------------------------------------------

    401019521,  # 2018 Tulane @ South Florida

    401015050,  # 2018 Charlotte @ Florida Atlantic

    401121942,  # 2019 Georgia State @ Texas State


    # ----------------------------------------------
    # Complete play-by-play, but CFBD has no PPA
    # ----------------------------------------------

    401641034,  # 2024 New Mexico State @ Sam Houston

    401644689,  # 2024 Kent State @ Miami (OH)

    401644780,  # 2024 Eastern Michigan @ Western Michigan


    # ----------------------------------------------
    # Partial play-by-play
    # Only first quarter recorded
    # ----------------------------------------------

    401643737,  # 2024 New Mexico @ Utah State


    # ----------------------------------------------
    # Corrupted team labels
    #
    # Every play incorrectly says:
    #
    # Rice offense
    # Army defense
    #
    # despite both teams having offensive plays.
    # ----------------------------------------------

    401645328   # 2024 Rice @ Army
}


# ==================================================
# PLAY-TYPE DEFINITIONS
# ==================================================

# --------------------------------------------------
# RUSHING PLAYS
# --------------------------------------------------

rush_types = {

    "Rush",

    "Rushing Touchdown"
}


# --------------------------------------------------
# DROPBACK PLAYS
#
# A dropback is a passing-play attempt.
#
# This includes:
#
# completed passes
# incomplete passes
# passing touchdowns
# interceptions
# sacks
#
# A sack is treated as a failed passing play.
# --------------------------------------------------

dropback_types = {

    "Pass",

    "Pass Reception",

    "Pass Incompletion",

    "Passing Touchdown",

    "Sack",

    "Interception",

    "Pass Interception Return",

    "Interception Return Touchdown"
}


# --------------------------------------------------
# OTHER SCRIMMAGE PLAYS WITH USEFUL PPA
#
# These contribute to overall PPA but aren't cleanly
# classified as either rushing or passing.
# --------------------------------------------------

other_scrimmage_types = {

    "Fumble Recovery (Own)",

    "Fumble Recovery (Opponent)",

    "Fumble Return Touchdown",

    "Safety"
}


core_scrimmage_types = (

    rush_types

    |

    dropback_types

    |

    other_scrimmage_types
)


# ==================================================
# LOAD FEATURE DATASET V1
# ==================================================

print(
    "BUILDING FEATURE DATASET V2 — PPA"
)

print(
    "=" * 100
)


features = pd.read_csv(
    v1_file
)


print(
    f"V1 rows loaded:    "
    f"{len(features):,}"
)

print(
    f"V1 columns loaded: "
    f"{len(features.columns):,}"
)


# ==================================================
# DATABASE
# ==================================================

connection = sqlite3.connect(
    database_file
)


# ==================================================
# LOAD FBS-V-FBS GAMES
# ==================================================

games_query = """
SELECT

    game_id,

    season,

    week,

    start_date,

    home_team,

    away_team

FROM games

WHERE

    home_points IS NOT NULL

    AND away_points IS NOT NULL

    AND home_classification = 'fbs'

    AND away_classification = 'fbs'

    AND season BETWEEN 2018 AND 2025

ORDER BY

    season,

    start_date,

    game_id
"""


games = pd.read_sql_query(
    games_query,
    connection
)


print(
    f"FBS-v-FBS games:   "
    f"{len(games):,}"
)


# ==================================================
# LOAD PLAYS WITH PPA
# ==================================================

plays_query = """
SELECT

    p.game_id,

    p.offense,

    p.defense,

    p.play_type,

    p.ppa

FROM plays AS p

JOIN games AS g

    ON p.game_id = g.game_id

WHERE

    g.home_points IS NOT NULL

    AND g.away_points IS NOT NULL

    AND g.home_classification = 'fbs'

    AND g.away_classification = 'fbs'

    AND g.season BETWEEN 2018 AND 2025

    AND p.ppa IS NOT NULL
"""


plays = pd.read_sql_query(
    plays_query,
    connection
)


connection.close()


print(
    f"PPA plays loaded:  "
    f"{len(plays):,}"
)


# ==================================================
# REMOVE KNOWN BAD GAMES
# ==================================================

before_bad_filter = len(
    plays
)


plays = plays[
    ~plays[
        "game_id"
    ].isin(
        unusable_ppa_games
    )
].copy()


removed_bad_plays = (

    before_bad_filter

    -

    len(
        plays
    )
)


print(
    f"Known-bad PPA plays removed: "
    f"{removed_bad_plays:,}"
)


# ==================================================
# KEEP CORE SCRIMMAGE PLAYS
# ==================================================

plays = plays[
    plays[
        "play_type"
    ].isin(
        core_scrimmage_types
    )
].copy()


print(
    f"Core PPA plays:    "
    f"{len(plays):,}"
)


# ==================================================
# CLASSIFY PLAYS
# ==================================================

plays[
    "is_rush"
] = (
    plays[
        "play_type"
    ]
    .isin(
        rush_types
    )
)


plays[
    "is_dropback"
] = (
    plays[
        "play_type"
    ]
    .isin(
        dropback_types
    )
)


# --------------------------------------------------
# POSITIVE PPA
#
# Positive PPA means the play increased the
# offense's expected scoring value.
#
# The percentage of plays above zero is similar to
# an efficiency / success measure.
# --------------------------------------------------

plays[
    "positive_ppa"
] = (
    plays[
        "ppa"
    ] > 0
).astype(
    float
)


# --------------------------------------------------
# CONDITIONAL PPA VALUES
#
# NaN means the play isn't that type.
# --------------------------------------------------

plays[
    "rush_ppa"
] = (
    plays[
        "ppa"
    ].where(
        plays[
            "is_rush"
        ]
    )
)


plays[
    "dropback_ppa"
] = (
    plays[
        "ppa"
    ].where(
        plays[
            "is_dropback"
        ]
    )
)


# ==================================================
# OFFENSIVE GAME-LEVEL PPA
# ==================================================

offense = (

    plays

    .groupby(
        [
            "game_id",
            "offense"
        ],
        as_index=False
    )

    .agg(

        off_ppa_per_play=(
            "ppa",
            "mean"
        ),

        off_ppa_plays=(
            "ppa",
            "count"
        ),

        off_positive_ppa_rate=(
            "positive_ppa",
            "mean"
        ),

        off_rush_ppa_per_play=(
            "rush_ppa",
            "mean"
        ),

        off_rush_ppa_plays=(
            "rush_ppa",
            "count"
        ),

        off_dropback_ppa_per_play=(
            "dropback_ppa",
            "mean"
        ),

        off_dropback_ppa_plays=(
            "dropback_ppa",
            "count"
        )
    )
)


offense = offense.rename(
    columns={
        "offense":
            "team"
    }
)


# ==================================================
# DEFENSIVE GAME-LEVEL PPA
#
# Defensive PPA here means:
#
# how much offensive PPA the opponent generated
# while this team was on defense.
#
# LOWER is therefore better for defense.
# ==================================================

defense = (

    plays

    .groupby(
        [
            "game_id",
            "defense"
        ],
        as_index=False
    )

    .agg(

        def_ppa_per_play=(
            "ppa",
            "mean"
        ),

        def_ppa_plays=(
            "ppa",
            "count"
        ),

        def_positive_ppa_rate=(
            "positive_ppa",
            "mean"
        ),

        def_rush_ppa_per_play=(
            "rush_ppa",
            "mean"
        ),

        def_rush_ppa_plays=(
            "rush_ppa",
            "count"
        ),

        def_dropback_ppa_per_play=(
            "dropback_ppa",
            "mean"
        ),

        def_dropback_ppa_plays=(
            "dropback_ppa",
            "count"
        )
    )
)


defense = defense.rename(
    columns={
        "defense":
            "team"
    }
)


# ==================================================
# CREATE ONE ROW PER TEAM PER GAME
# ==================================================

home_rows = games[
    [
        "game_id",
        "season",
        "week",
        "start_date",
        "home_team",
        "away_team"
    ]
].copy()


home_rows = home_rows.rename(
    columns={
        "home_team":
            "team",

        "away_team":
            "opponent"
    }
)


home_rows[
    "home_away"
] = "home"


away_rows = games[
    [
        "game_id",
        "season",
        "week",
        "start_date",
        "away_team",
        "home_team"
    ]
].copy()


away_rows = away_rows.rename(
    columns={
        "away_team":
            "team",

        "home_team":
            "opponent"
    }
)


away_rows[
    "home_away"
] = "away"


team_games = pd.concat(
    [
        home_rows,
        away_rows
    ],
    ignore_index=True
)


# ==================================================
# MERGE GAME PPA
# ==================================================

team_games = team_games.merge(

    offense,

    on=[
        "game_id",
        "team"
    ],

    how="left"
)


team_games = team_games.merge(

    defense,

    on=[
        "game_id",
        "team"
    ],

    how="left"
)


# ==================================================
# DETERMINE WHETHER A TEAM HAS COMPLETE PPA
#
# We require BOTH:
#
# offensive PPA
# defensive PPA
#
# This is also a safety net. If another malformed
# game ever appears, partial data cannot enter later
# rolling averages.
# ==================================================

team_games[
    "ppa_game_available"
] = (

    (
        team_games[
            "off_ppa_plays"
        ]
        .fillna(0)
        > 0
    )

    &

    (
        team_games[
            "def_ppa_plays"
        ]
        .fillna(0)
        > 0
    )
)


team_games[
    "ppa_game_available"
] = (
    team_games[
        "ppa_game_available"
    ]
    .astype(int)
)


# ==================================================
# MASK INCOMPLETE GAME METRICS
#
# This is the important correction.
#
# If a team does not have BOTH offensive and
# defensive PPA for the game, NONE of that game's
# PPA metrics are allowed to contribute to future
# averages.
# ==================================================

raw_metric_columns = [

    "off_ppa_per_play",

    "def_ppa_per_play",

    "off_positive_ppa_rate",

    "def_positive_ppa_rate",

    "off_rush_ppa_per_play",

    "def_rush_ppa_per_play",

    "off_dropback_ppa_per_play",

    "def_dropback_ppa_per_play"
]


invalid_team_rows = (
    team_games[
        "ppa_game_available"
    ] == 0
)


team_games.loc[
    invalid_team_rows,
    raw_metric_columns
] = np.nan


# ==================================================
# NET PPA
#
# Higher is better.
#
# Example:
#
# offense creates +0.25 PPA/play
# defense allows  +0.05 PPA/play
#
# net = +0.20
# ==================================================

team_games[
    "net_ppa_per_play"
] = (

    team_games[
        "off_ppa_per_play"
    ]

    -

    team_games[
        "def_ppa_per_play"
    ]
)


team_games[
    "net_rush_ppa_per_play"
] = (

    team_games[
        "off_rush_ppa_per_play"
    ]

    -

    team_games[
        "def_rush_ppa_per_play"
    ]
)


team_games[
    "net_dropback_ppa_per_play"
] = (

    team_games[
        "off_dropback_ppa_per_play"
    ]

    -

    team_games[
        "def_dropback_ppa_per_play"
    ]
)


team_games[
    "net_positive_ppa_rate"
] = (

    team_games[
        "off_positive_ppa_rate"
    ]

    -

    team_games[
        "def_positive_ppa_rate"
    ]
)


# ==================================================
# CHRONOLOGICAL ORDER
# ==================================================

team_games[
    "start_date"
] = pd.to_datetime(
    team_games[
        "start_date"
    ],
    utc=True
)


team_games = (
    team_games

    .sort_values(
        [
            "season",
            "team",
            "start_date",
            "game_id"
        ]
    )

    .reset_index(
        drop=True
    )
)


# ==================================================
# PPA METRICS TO ROLL FORWARD
# ==================================================

ppa_metrics = [

    "off_ppa_per_play",

    "def_ppa_per_play",

    "net_ppa_per_play",

    "off_rush_ppa_per_play",

    "def_rush_ppa_per_play",

    "net_rush_ppa_per_play",

    "off_dropback_ppa_per_play",

    "def_dropback_ppa_per_play",

    "net_dropback_ppa_per_play",

    "off_positive_ppa_rate",

    "def_positive_ppa_rate",

    "net_positive_ppa_rate"
]


# ==================================================
# LEAKAGE-SAFE ROLLING FEATURES
#
# shift(1) means the current game NEVER contributes
# to its own prediction.
# ==================================================

group_keys = [
    "season",
    "team"
]


rolling_columns = []


new_rolling_data = {}


for metric in ppa_metrics:

    average_name = (
        f"pregame_avg_"
        f"{metric}"
    )


    last3_name = (
        f"pregame_last3_"
        f"{metric}"
    )


    shifted = (
        team_games

        .groupby(
            group_keys
        )[
            metric
        ]

        .shift(1)
    )


    new_rolling_data[
        average_name
    ] = (

        shifted

        .groupby(
            [
                team_games[
                    "season"
                ],

                team_games[
                    "team"
                ]
            ]
        )

        .transform(
            lambda series:
                series
                .expanding(
                    min_periods=1
                )
                .mean()
        )
    )


    new_rolling_data[
        last3_name
    ] = (

        shifted

        .groupby(
            [
                team_games[
                    "season"
                ],

                team_games[
                    "team"
                ]
            ]
        )

        .transform(
            lambda series:
                series
                .rolling(
                    window=3,
                    min_periods=1
                )
                .mean()
        )
    )


    rolling_columns.extend(
        [
            average_name,
            last3_name
        ]
    )


rolling_frame = pd.DataFrame(
    new_rolling_data,
    index=team_games.index
)


team_games = pd.concat(
    [
        team_games,
        rolling_frame
    ],
    axis=1
)


# ==================================================
# NUMBER OF PRIOR GAMES WITH GOOD PPA
# ==================================================

team_games[
    "pregame_ppa_games_available"
] = (

    team_games

    .groupby(
        group_keys
    )[
        "ppa_game_available"
    ]

    .transform(
        lambda series:
            series
            .shift(1)
            .fillna(0)
            .cumsum()
    )
)


# ==================================================
# CURRENT-GAME COVERAGE AUDIT
# ==================================================

game_coverage = (

    team_games

    .groupby(
        [
            "game_id",
            "season"
        ],
        as_index=False
    )

    .agg(

        teams_with_ppa=(
            "ppa_game_available",
            "sum"
        )
    )
)


game_coverage[
    "game_has_usable_ppa"
] = (
    game_coverage[
        "teams_with_ppa"
    ] == 2
)


coverage_by_season = (

    game_coverage

    .groupby(
        "season",
        as_index=False
    )

    .agg(

        games=(
            "game_id",
            "count"
        ),

        games_with_ppa=(
            "game_has_usable_ppa",
            "sum"
        )
    )
)


coverage_by_season[
    "missing_ppa_games"
] = (

    coverage_by_season[
        "games"
    ]

    -

    coverage_by_season[
        "games_with_ppa"
    ]
)


coverage_by_season[
    "coverage_pct"
] = (

    coverage_by_season[
        "games_with_ppa"
    ]

    /

    coverage_by_season[
        "games"
    ]

    * 100
)


# ==================================================
# HOME PPA FEATURE TABLE
# ==================================================

home_ppa = team_games[
    [
        "game_id",

        "team",

        "pregame_ppa_games_available"
    ]

    +

    rolling_columns
].copy()


home_ppa = home_ppa.rename(
    columns={
        "team":
            "home_team",

        "pregame_ppa_games_available":
            "home_pregame_ppa_games_available"
    }
)


home_rename = {

    column:
        f"home_{column}"

    for column in rolling_columns
}


home_ppa = home_ppa.rename(
    columns=home_rename
)


# ==================================================
# AWAY PPA FEATURE TABLE
# ==================================================

away_ppa = team_games[
    [
        "game_id",

        "team",

        "pregame_ppa_games_available"
    ]

    +

    rolling_columns
].copy()


away_ppa = away_ppa.rename(
    columns={
        "team":
            "away_team",

        "pregame_ppa_games_available":
            "away_pregame_ppa_games_available"
    }
)


away_rename = {

    column:
        f"away_{column}"

    for column in rolling_columns
}


away_ppa = away_ppa.rename(
    columns=away_rename
)


# ==================================================
# MERGE INTO V1
# ==================================================

features_v2 = features.merge(

    home_ppa,

    on=[
        "game_id",
        "home_team"
    ],

    how="left"
)


features_v2 = features_v2.merge(

    away_ppa,

    on=[
        "game_id",
        "away_team"
    ],

    how="left"
)


# ==================================================
# CREATE HOME-AWAY DIFFERENCE FEATURES
#
# Build them all together instead of inserting one
# at a time. This avoids the pandas fragmentation
# warning from the first version.
# ==================================================

difference_data = {}


for column in rolling_columns:

    difference_data[
        f"diff_{column}"
    ] = (

        features_v2[
            f"home_{column}"
        ]

        -

        features_v2[
            f"away_{column}"
        ]
    )


difference_data[
    "diff_pregame_ppa_games_available"
] = (

    features_v2[
        "home_pregame_ppa_games_available"
    ]

    -

    features_v2[
        "away_pregame_ppa_games_available"
    ]
)


difference_data[
    "min_pregame_ppa_games_available"
] = (

    features_v2[
        [
            "home_pregame_ppa_games_available",

            "away_pregame_ppa_games_available"
        ]
    ]

    .min(
        axis=1
    )
)


difference_frame = pd.DataFrame(
    difference_data,
    index=features_v2.index
)


features_v2 = pd.concat(
    [
        features_v2,
        difference_frame
    ],
    axis=1
)


# ==================================================
# INTEGRITY CHECKS
# ==================================================

duplicate_games = (
    features_v2[
        "game_id"
    ]
    .duplicated()
    .sum()
)


if len(
    features_v2
) != len(
    features
):

    raise RuntimeError(
        "V2 row count does not match V1."
    )


if duplicate_games != 0:

    raise RuntimeError(
        "Duplicate game IDs found in V2."
    )


# ==================================================
# SAVE
# ==================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


features_v2.to_csv(
    v2_file,
    index=False
)


# ==================================================
# OUTPUT
# ==================================================

print("\n")

print(
    "PPA GAME COVERAGE"
)

print(
    "=" * 100
)


print(
    coverage_by_season.to_string(
        index=False,
        formatters={
            "coverage_pct":
                lambda value:
                f"{value:.2f}%"
        }
    )
)


print("\n")

print(
    "FEATURE DATASET V2 RESULTS"
)

print(
    "=" * 100
)


print(
    f"Rows created:       "
    f"{len(features_v2):,}"
)


print(
    f"Columns created:    "
    f"{len(features_v2.columns):,}"
)


print(
    f"New V2 columns:     "
    f"{len(features_v2.columns) - len(features.columns):,}"
)


print(
    f"Duplicate game IDs: "
    f"{duplicate_games}"
)


print(
    f"Saved to:           "
    f"{v2_file}"
)


# ==================================================
# PREGAME PPA HISTORY
# ==================================================

print("\n")

print(
    "PREGAME PPA HISTORY AVAILABILITY"
)

print(
    "=" * 100
)


history_summary = (

    features_v2[
        features_v2[
            "season"
        ] <= 2024
    ]

    .groupby(
        "season"
    )

    .agg(

        games=(
            "game_id",
            "count"
        ),

        avg_home_prior_ppa_games=(
            "home_pregame_ppa_games_available",
            "mean"
        ),

        avg_away_prior_ppa_games=(
            "away_pregame_ppa_games_available",
            "mean"
        )
    )
)


print(
    history_summary.to_string()
)


# ==================================================
# 2024 SAMPLE
#
# We intentionally do not display 2025 outcomes.
# ==================================================

sample_columns = [

    "week",

    "away_team",

    "home_team",

    "away_pregame_elo",

    "home_pregame_elo",

    "away_pregame_avg_off_ppa_per_play",

    "home_pregame_avg_off_ppa_per_play",

    "away_pregame_avg_def_ppa_per_play",

    "home_pregame_avg_def_ppa_per_play",

    "diff_pregame_avg_net_ppa_per_play"
]


sample = (

    features_v2[
        features_v2[
            "season"
        ] == 2024
    ][
        sample_columns
    ]

    .head(12)
)


print("\n")

print(
    "2024 PPA FEATURE SAMPLE"
)

print(
    "=" * 100
)


print(
    sample.to_string(
        index=False
    )
)


print("\n")

print(
    "2025 was included only so leakage-safe "
    "pregame feature rows exist. No 2025 model "
    "performance was evaluated."
)