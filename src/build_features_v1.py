import os
import sqlite3
import numpy as np
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

database_file = "data/cfb.db"

output_directory = "data/processed"

output_file = (
    f"{output_directory}/"
    f"features_v1.csv"
)

os.makedirs(
    output_directory,
    exist_ok=True
)


# --------------------------------------------------
# CONNECT
# --------------------------------------------------

connection = sqlite3.connect(
    database_file
)


print("BUILDING FEATURE DATASET V1")
print("=" * 80)


# ==================================================
# LOAD TEAM-GAME BASE DATA
# ==================================================

team_games = pd.read_sql_query(
    """
    SELECT

        g.game_id,
        g.season,
        g.week,
        g.start_date,

        g.neutral_site,
        g.conference_game,

        g.home_team,
        g.away_team,

        g.home_classification,
        g.away_classification,

        g.home_points,
        g.away_points,

        g.home_pregame_elo,
        g.away_pregame_elo,

        t.team,
        t.home_away,
        t.points AS team_points,

        CASE

            WHEN t.team = g.home_team
            THEN g.away_team

            ELSE g.home_team

        END AS opponent,

        CASE

            WHEN t.team = g.home_team
            THEN g.home_classification

            ELSE g.away_classification

        END AS team_classification,

        CASE

            WHEN t.team = g.home_team
            THEN g.away_classification

            ELSE g.home_classification

        END AS opponent_classification,

        CASE

            WHEN t.team = g.home_team
            THEN g.home_points

            ELSE g.away_points

        END AS points_for,

        CASE

            WHEN t.team = g.home_team
            THEN g.away_points

            ELSE g.home_points

        END AS points_against

    FROM team_game_stats AS t

    JOIN games AS g
        ON t.game_id = g.game_id

    WHERE
        g.season BETWEEN 2018 AND 2025
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
    """,
    connection
)


team_games["start_date"] = pd.to_datetime(
    team_games["start_date"],
    utc=True,
    errors="coerce"
)


print(
    f"Team-game base rows loaded: "
    f"{len(team_games):,}"
)


# ==================================================
# LOAD SELECTED TEAM STAT VALUES
# ==================================================

stat_categories = [

    "totalYards",
    "netPassingYards",
    "rushingYards",

    "turnovers",

    "firstDowns",

    "sacks",
    "TFL",

    "yardsPerPass",
    "yardsPerRushAttempt",

    "thirdDownEff"
]


placeholders = ",".join(
    ["?"] * len(stat_categories)
)


stats = pd.read_sql_query(
    f"""
    SELECT

        s.game_id,
        s.team,
        s.category,
        s.stat_value

    FROM team_game_stat_values AS s

    JOIN games AS g
        ON s.game_id = g.game_id

    WHERE
        g.season BETWEEN 2018 AND 2025
        AND s.category IN ({placeholders})
    """,
    connection,
    params=stat_categories
)


print(
    f"Selected stat values loaded: "
    f"{len(stats):,}"
)


# ==================================================
# PIVOT STATS
#
# Turns:
#
# game | team | totalYards | 450
# game | team | turnovers  | 2
#
# into one row:
#
# game | team | totalYards | turnovers
# ==================================================

stats_wide = stats.pivot_table(
    index=[
        "game_id",
        "team"
    ],
    columns="category",
    values="stat_value",
    aggfunc="first"
).reset_index()


stats_wide.columns.name = None


# ==================================================
# STAT PARSING HELPERS
# ==================================================

def numeric_value(value):

    try:

        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return np.nan


def efficiency_percentage(value):

    if value is None:
        return np.nan

    text = str(value).strip()

    # CFBD efficiency fields generally look like:
    #
    # 6-17
    #
    # meaning 6 conversions in 17 attempts.

    for separator in [
        "-",
        "/"
    ]:

        if separator in text:

            parts = text.split(
                separator
            )

            if len(parts) != 2:
                continue

            try:

                made = float(
                    parts[0]
                )

                attempts = float(
                    parts[1]
                )

            except ValueError:

                return np.nan


            if attempts == 0:

                return np.nan


            return (
                made
                / attempts
            )


    return np.nan


# ==================================================
# CONVERT STATS TO NUMBERS
# ==================================================

numeric_categories = [

    "totalYards",
    "netPassingYards",
    "rushingYards",

    "turnovers",

    "firstDowns",

    "sacks",
    "TFL",

    "yardsPerPass",
    "yardsPerRushAttempt"
]


for category in numeric_categories:

    if category in stats_wide.columns:

        stats_wide[category] = (
            stats_wide[category]
            .apply(numeric_value)
        )


if "thirdDownEff" in stats_wide.columns:

    stats_wide[
        "thirdDownPct"
    ] = (
        stats_wide[
            "thirdDownEff"
        ]
        .apply(
            efficiency_percentage
        )
    )


# We no longer need the raw string.
stats_wide = stats_wide.drop(
    columns=[
        "thirdDownEff"
    ],
    errors="ignore"
)


# ==================================================
# MERGE STATS INTO TEAM GAME DATA
# ==================================================

team_games = team_games.merge(
    stats_wide,
    on=[
        "game_id",
        "team"
    ],
    how="left"
)


# ==================================================
# ADD OPPONENT'S GAME STATS
#
# This lets us calculate defensive history.
#
# Example:
#
# Alabama allowed 310 yards
#
# = opponent totalYards in that game.
# ==================================================

opponent_stats = stats_wide.copy()


stat_columns = [
    column
    for column in opponent_stats.columns
    if column not in [
        "game_id",
        "team"
    ]
]


opponent_stats = opponent_stats.rename(
    columns={
        "team": "opponent",

        **{
            column:
            f"opponent_{column}"

            for column
            in stat_columns
        }
    }
)


team_games = team_games.merge(
    opponent_stats,
    on=[
        "game_id",
        "opponent"
    ],
    how="left"
)


# ==================================================
# DERIVED SINGLE-GAME TEAM METRICS
#
# THESE ARE STILL GAME RESULTS.
#
# They are not going directly into the model for
# this game. They are used only to calculate
# PREVIOUS-game averages after we shift them.
# ==================================================

team_games[
    "margin"
] = (
    team_games["points_for"]
    - team_games["points_against"]
)


team_games[
    "win"
] = (
    team_games["margin"] > 0
).astype(float)


team_games[
    "turnover_margin"
] = (
    team_games.get(
        "opponent_turnovers",
        np.nan
    )
    - team_games.get(
        "turnovers",
        np.nan
    )
)


# ==================================================
# KEEP FBS-v-FBS HISTORY FOR V1
# ==================================================

history = team_games[
    (
        team_games[
            "team_classification"
        ] == "fbs"
    )
    &
    (
        team_games[
            "opponent_classification"
        ] == "fbs"
    )
].copy()


history = history.sort_values(
    by=[
        "season",
        "team",
        "start_date",
        "game_id"
    ]
).reset_index(
    drop=True
)


print(
    f"FBS-v-FBS team-game history rows: "
    f"{len(history):,}"
)


# ==================================================
# METRICS TO TURN INTO PREGAME HISTORY
# ==================================================

history_metrics = [

    "win",

    "points_for",
    "points_against",
    "margin",

    "totalYards",
    "opponent_totalYards",

    "netPassingYards",
    "opponent_netPassingYards",

    "rushingYards",
    "opponent_rushingYards",

    "turnovers",
    "opponent_turnovers",
    "turnover_margin",

    "firstDowns",
    "opponent_firstDowns",

    "sacks",
    "opponent_sacks",

    "TFL",
    "opponent_TFL",

    "yardsPerPass",
    "opponent_yardsPerPass",

    "yardsPerRushAttempt",
    "opponent_yardsPerRushAttempt",

    "thirdDownPct",
    "opponent_thirdDownPct"
]


history_metrics = [
    metric
    for metric in history_metrics
    if metric in history.columns
]


# ==================================================
# CREATE PREGAME EXPANDING + LAST-3 FEATURES
#
# CRITICAL:
#
# .shift(1)
#
# means the CURRENT GAME is excluded.
#
# If Alabama is playing Week 6, the Week 6 row only
# receives statistics from Weeks 1-5.
# ==================================================

feature_frames = []


for (
    season,
    team
), group in history.groupby(
    [
        "season",
        "team"
    ],
    sort=False
):

    group = group.sort_values(
        by=[
            "start_date",
            "game_id"
        ]
    ).copy()


    # Number of previous FBS-v-FBS games
    group[
        "pregame_games_played"
    ] = np.arange(
        len(group)
    )


    for metric in history_metrics:

        prior_values = (
            group[metric]
            .shift(1)
        )


        # Season-to-date average
        group[
            f"pregame_avg_{metric}"
        ] = (
            prior_values
            .expanding(
                min_periods=1
            )
            .mean()
        )


        # Recent form — last 3 games
        group[
            f"pregame_last3_{metric}"
        ] = (
            prior_values
            .rolling(
                window=3,
                min_periods=1
            )
            .mean()
        )


    feature_frames.append(
        group
    )


history_features = pd.concat(
    feature_frames,
    ignore_index=True
)


# ==================================================
# KEEP ONLY PREGAME FEATURES
# ==================================================

pregame_columns = [
    column
    for column in history_features.columns
    if (
        column.startswith(
            "pregame_"
        )
    )
]


team_feature_table = (
    history_features[
        [
            "game_id",
            "team"
        ]
        + pregame_columns
    ]
    .copy()
)


# ==================================================
# MODEL POPULATION
#
# One row per completed FBS-v-FBS game.
# ==================================================

games = pd.read_sql_query(
    """
    SELECT

        game_id,

        season,
        week,
        start_date,

        neutral_site,
        conference_game,

        home_team,
        away_team,

        home_points,
        away_points,

        home_pregame_elo,
        away_pregame_elo

    FROM games

    WHERE
        season BETWEEN 2018 AND 2025

        AND home_points IS NOT NULL
        AND away_points IS NOT NULL

        AND home_classification = 'fbs'
        AND away_classification = 'fbs'

    ORDER BY
        season,
        week,
        start_date
    """,
    connection
)


games["start_date"] = pd.to_datetime(
    games["start_date"],
    utc=True,
    errors="coerce"
)


print(
    f"Model population games: "
    f"{len(games):,}"
)


# ==================================================
# HOME TEAM PREGAME HISTORY
# ==================================================

home_features = (
    team_feature_table.copy()
)


home_features = home_features.rename(
    columns={
        "team": "home_team",

        **{
            column:
            f"home_{column}"

            for column
            in pregame_columns
        }
    }
)


games = games.merge(
    home_features,
    on=[
        "game_id",
        "home_team"
    ],
    how="left"
)


# ==================================================
# AWAY TEAM PREGAME HISTORY
# ==================================================

away_features = (
    team_feature_table.copy()
)


away_features = away_features.rename(
    columns={
        "team": "away_team",

        **{
            column:
            f"away_{column}"

            for column
            in pregame_columns
        }
    }
)


games = games.merge(
    away_features,
    on=[
        "game_id",
        "away_team"
    ],
    how="left"
)


# ==================================================
# LOAD TEAM TALENT
# ==================================================

talent = pd.read_sql_query(
    """
    SELECT
        season,
        team,
        talent

    FROM team_talent

    WHERE season BETWEEN 2018 AND 2025
    """,
    connection
)


home_talent = talent.rename(
    columns={
        "team":
            "home_team",

        "talent":
            "home_talent"
    }
)


away_talent = talent.rename(
    columns={
        "team":
            "away_team",

        "talent":
            "away_talent"
    }
)


games = games.merge(
    home_talent,
    on=[
        "season",
        "home_team"
    ],
    how="left"
)


games = games.merge(
    away_talent,
    on=[
        "season",
        "away_team"
    ],
    how="left"
)


# Availability flags
games[
    "home_talent_available"
] = (
    games[
        "home_talent"
    ].notna()
).astype(int)


games[
    "away_talent_available"
] = (
    games[
        "away_talent"
    ].notna()
).astype(int)


# ==================================================
# RETURNING PRODUCTION
# ==================================================

returning = pd.read_sql_query(
    """
    SELECT

        season,
        team,

        percent_ppa,
        percent_passing_ppa,
        percent_receiving_ppa,
        percent_rushing_ppa,

        usage,
        passing_usage,
        receiving_usage,
        rushing_usage

    FROM returning_production

    WHERE season BETWEEN 2018 AND 2025
    """,
    connection
)


returning_columns = [
    column
    for column in returning.columns
    if column not in [
        "season",
        "team"
    ]
]


home_returning = returning.rename(
    columns={
        "team":
            "home_team",

        **{
            column:
            f"home_{column}"

            for column
            in returning_columns
        }
    }
)


away_returning = returning.rename(
    columns={
        "team":
            "away_team",

        **{
            column:
            f"away_{column}"

            for column
            in returning_columns
        }
    }
)


games = games.merge(
    home_returning,
    on=[
        "season",
        "home_team"
    ],
    how="left"
)


games = games.merge(
    away_returning,
    on=[
        "season",
        "away_team"
    ],
    how="left"
)


games[
    "home_returning_available"
] = (
    games[
        "home_percent_ppa"
    ].notna()
).astype(int)


games[
    "away_returning_available"
] = (
    games[
        "away_percent_ppa"
    ].notna()
).astype(int)


# ==================================================
# MARKET BENCHMARK
#
# IMPORTANT:
#
# These columns are being stored for benchmarking.
#
# They are NOT automatically allowed into our
# market-blind model.
# ==================================================

betting = pd.read_sql_query(
    """
    SELECT

        game_id,

        spread,
        spread_open,

        over_under,
        over_under_open

    FROM betting_lines
    """,
    connection
)


market = (
    betting
    .groupby(
        "game_id",
        as_index=False
    )
    .agg(
        benchmark_spread=(
            "spread",
            "median"
        ),

        benchmark_open_spread=(
            "spread_open",
            "median"
        ),

        benchmark_total=(
            "over_under",
            "median"
        ),

        benchmark_open_total=(
            "over_under_open",
            "median"
        )
    )
)


games = games.merge(
    market,
    on="game_id",
    how="left"
)


# Negative spread means home favored.
#
# Example:
#
# Home spread = -7
#
# Market expected home margin = +7

games[
    "benchmark_market_home_margin"
] = (
    -games[
        "benchmark_spread"
    ]
)


# ==================================================
# TARGET LABELS
#
# THESE ARE WHAT WE ARE TRYING TO PREDICT.
#
# They are not model inputs.
# ==================================================

games[
    "target_home_margin"
] = (
    games[
        "home_points"
    ]
    - games[
        "away_points"
    ]
)


games[
    "target_total_points"
] = (
    games[
        "home_points"
    ]
    + games[
        "away_points"
    ]
)


games[
    "target_home_win"
] = (
    games[
        "target_home_margin"
    ] > 0
).astype(int)


# ==================================================
# PREGAME ELO FEATURES
# ==================================================

games[
    "home_pregame_elo_available"
] = (
    games[
        "home_pregame_elo"
    ].notna()
).astype(int)


games[
    "away_pregame_elo_available"
] = (
    games[
        "away_pregame_elo"
    ].notna()
).astype(int)


games[
    "pregame_elo_diff"
] = (
    games[
        "home_pregame_elo"
    ]
    - games[
        "away_pregame_elo"
    ]
)


# ==================================================
# HOME-v-AWAY DIFFERENCE FEATURES
#
# Helpful for linear models and useful later for
# debugging the neural network.
# ==================================================

difference_pairs = [

    (
        "home_talent",
        "away_talent",
        "talent_diff"
    ),

    (
        "home_percent_ppa",
        "away_percent_ppa",
        "returning_ppa_diff"
    ),

    (
        "home_percent_passing_ppa",
        "away_percent_passing_ppa",
        "returning_passing_ppa_diff"
    ),

    (
        "home_percent_receiving_ppa",
        "away_percent_receiving_ppa",
        "returning_receiving_ppa_diff"
    ),

    (
        "home_percent_rushing_ppa",
        "away_percent_rushing_ppa",
        "returning_rushing_ppa_diff"
    )
]


for (
    home_column,
    away_column,
    new_column
) in difference_pairs:

    if (
        home_column in games.columns
        and
        away_column in games.columns
    ):

        games[
            new_column
        ] = (
            games[
                home_column
            ]
            - games[
                away_column
            ]
        )


# --------------------------------------------------
# DIFFERENCES FOR ROLLING TEAM FEATURES
# --------------------------------------------------

for feature in pregame_columns:

    home_column = (
        f"home_{feature}"
    )

    away_column = (
        f"away_{feature}"
    )


    if (
        home_column in games.columns
        and
        away_column in games.columns
    ):

        games[
            f"diff_{feature}"
        ] = (
            games[
                home_column
            ]
            - games[
                away_column
            ]
        )


# ==================================================
# DATASET SPLITS
#
# 2025 is never used for selecting/tuning models.
# ==================================================

def dataset_split(season):

    if season <= 2023:

        return "train"

    if season == 2024:

        return "validation"

    if season == 2025:

        return "holdout"

    return "unknown"


games[
    "dataset_split"
] = (
    games[
        "season"
    ]
    .apply(
        dataset_split
    )
)


# ==================================================
# DUPLICATE / INTEGRITY CHECKS
# ==================================================

duplicate_games = (
    games[
        "game_id"
    ]
    .duplicated()
    .sum()
)


if duplicate_games != 0:

    raise RuntimeError(
        f"Duplicate game rows detected: "
        f"{duplicate_games}"
    )


expected_games = 5705


if len(games) != expected_games:

    print(
        f"\nWARNING: Expected approximately "
        f"{expected_games:,} FBS-v-FBS games "
        f"but found {len(games):,}."
    )


# ==================================================
# SORT
# ==================================================

games = games.sort_values(
    by=[
        "season",
        "week",
        "start_date",
        "game_id"
    ]
).reset_index(
    drop=True
)


# ==================================================
# SAVE
# ==================================================

games.to_csv(
    output_file,
    index=False
)


# ==================================================
# REPORT
# ==================================================

print("\n")
print("FEATURE DATASET V1 RESULTS")
print("=" * 80)

print(
    f"Rows created:             "
    f"{len(games):,}"
)

print(
    f"Columns created:          "
    f"{len(games.columns):,}"
)

print(
    f"Duplicate game IDs:       "
    f"{duplicate_games}"
)

print(
    f"Saved to:                 "
    f"{output_file}"
)


print("\nDATASET SPLITS")
print("-" * 80)


split_counts = (
    games[
        "dataset_split"
    ]
    .value_counts()
)


for split in [
    "train",
    "validation",
    "holdout"
]:

    print(
        f"{split:<15} "
        f"{split_counts.get(split, 0):,}"
    )


# ==================================================
# MARKET COVERAGE
# ==================================================

print("\nMARKET BENCHMARK COVERAGE")
print("-" * 80)


market_coverage = (
    games[
        "benchmark_spread"
    ]
    .notna()
    .mean()
    * 100
)


print(
    f"Median spread available:  "
    f"{market_coverage:.2f}%"
)


# ==================================================
# ELO COVERAGE
# ==================================================

print("\nPREGAME ELO COVERAGE")
print("-" * 80)


home_elo_coverage = (
    games[
        "home_pregame_elo"
    ]
    .notna()
    .mean()
    * 100
)


away_elo_coverage = (
    games[
        "away_pregame_elo"
    ]
    .notna()
    .mean()
    * 100
)


print(
    f"Home pregame Elo:         "
    f"{home_elo_coverage:.2f}%"
)

print(
    f"Away pregame Elo:         "
    f"{away_elo_coverage:.2f}%"
)


# ==================================================
# SAMPLE: ALABAMA 2025
# ==================================================

sample = games[
    (
        games["season"] == 2025
    )
    &
    (
        (
            games["home_team"]
            == "Alabama"
        )
        |
        (
            games["away_team"]
            == "Alabama"
        )
    )
].copy()


print("\nALABAMA 2025 FEATURE SAMPLE")
print("-" * 80)


sample_columns = [

    "week",

    "away_team",
    "home_team",

    "away_pregame_elo",
    "home_pregame_elo",

    "away_pregame_games_played",
    "home_pregame_games_played",

    "away_pregame_avg_margin",
    "home_pregame_avg_margin",

    "away_talent",
    "home_talent",

    "benchmark_spread",

    "target_home_margin"
]


sample_columns = [
    column
    for column in sample_columns
    if column in sample.columns
]


print(
    sample[
        sample_columns
    ]
    .head(8)
    .to_string(
        index=False
    )
)


# ==================================================
# CLOSE
# ==================================================

connection.close()