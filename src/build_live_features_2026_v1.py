import sqlite3

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

LIVE_SEASON = 2026

PARITY_SEASON = 2025

PARITY_TOLERANCE = 1e-6


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


HISTORICAL_FEATURE_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "features_v2.csv"
)


OUTPUT_CSV = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "live_features_2026_v1.csv"
)


# ============================================================
# HELPERS
# ============================================================

def header(text):

    print()

    print(
        "=" * 110
    )

    print(
        text
    )

    print(
        "=" * 110
    )


def numeric_value(value):

    try:

        return float(
            value
        )


    except (
        TypeError,
        ValueError
    ):

        return np.nan


def efficiency_percentage(value):

    if value is None:

        return np.nan


    text = str(
        value
    ).strip()


    for separator in [

        "-",
        "/"

    ]:

        if separator not in text:

            continue


        parts = text.split(
            separator
        )


        if len(
            parts
        ) != 2:

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
            /
            attempts
        )


    return np.nan


# ============================================================
# EXACT MODEL CONTRACT
#
# Do NOT manually reinvent the feature list.
#
# The original neural-network script selected every
# diff_pregame_* column occurring in the first 192 columns
# of features_v2.csv, then added the seven context features.
# ============================================================

def get_model_features():

    feature_header = pd.read_csv(
        HISTORICAL_FEATURE_FILE,
        nrows=0
    )


    v1_columns = list(
        feature_header.columns[
            :192
        ]
    )


    features = [

        column

        for column in v1_columns

        if column.startswith(
            "diff_pregame_"
        )
    ]


    features += [

        "neutral_site",

        "pregame_elo_diff",

        "talent_diff",

        "returning_ppa_diff",

        "returning_passing_ppa_diff",

        "returning_receiving_ppa_diff",

        "returning_rushing_ppa_diff"
    ]


    features = sorted(
        set(
            features
        )
    )


    if len(
        features
    ) != 54:

        raise RuntimeError(

            f"Expected exactly 54 model features. "
            f"Found {len(features)}."
        )


    return features


# ============================================================
# MODEL ROLLING METRICS
#
# These are the 23 single-team statistics responsible for
# the 46 average/last-three difference features.
#
# + pregame_games_played difference = 47 rolling inputs.
# ============================================================

ROLLING_METRICS = [

    "firstDowns",

    "margin",

    "netPassingYards",

    "opponent_firstDowns",

    "opponent_netPassingYards",

    "opponent_rushingYards",

    "opponent_sacks",

    "opponent_thirdDownPct",

    "opponent_totalYards",

    "opponent_turnovers",

    "opponent_yardsPerPass",

    "opponent_yardsPerRushAttempt",

    "points_against",

    "points_for",

    "rushingYards",

    "sacks",

    "thirdDownPct",

    "totalYards",

    "turnover_margin",

    "turnovers",

    "win",

    "yardsPerPass",

    "yardsPerRushAttempt"
]


RAW_STAT_CATEGORIES = [

    "totalYards",

    "netPassingYards",

    "rushingYards",

    "turnovers",

    "firstDowns",

    "sacks",

    "yardsPerPass",

    "yardsPerRushAttempt",

    "thirdDownEff"
]


# ============================================================
# LOAD TEAM-GAME HISTORY
# ============================================================

def load_team_history(
    connection,
    season
):

    team_games = pd.read_sql_query(
        """
        SELECT

            t.game_id,

            g.season,

            g.week,

            g.start_date,

            t.team,

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

        INNER JOIN games AS g
            ON g.game_id = t.game_id

        WHERE
            g.season = ?
            AND g.home_points IS NOT NULL
            AND g.away_points IS NOT NULL
        """,

        connection,

        params=[
            season
        ]
    )


    team_games[
        "start_date"
    ] = pd.to_datetime(

        team_games[
            "start_date"
        ],

        utc=True,

        errors="coerce"
    )


    # ========================================================
    # LOAD RAW STATS
    # ========================================================

    placeholders = ",".join(
        [
            "?"
            for _ in RAW_STAT_CATEGORIES
        ]
    )


    stats = pd.read_sql_query(
        f"""
        SELECT

            s.game_id,

            s.team,

            s.category,

            s.stat_value

        FROM team_game_stat_values AS s

        INNER JOIN games AS g
            ON g.game_id = s.game_id

        WHERE
            g.season = ?
            AND s.category IN (
                {placeholders}
            )
        """,

        connection,

        params=[
            season,
            *RAW_STAT_CATEGORIES
        ]
    )


    # ========================================================
    # PIVOT STATS
    # ========================================================

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


    # ========================================================
    # NUMERIC PARSING
    # ========================================================

    numeric_categories = [

        "totalYards",

        "netPassingYards",

        "rushingYards",

        "turnovers",

        "firstDowns",

        "sacks",

        "yardsPerPass",

        "yardsPerRushAttempt"
    ]


    for category in numeric_categories:

        if category in stats_wide.columns:

            stats_wide[
                category
            ] = (

                stats_wide[
                    category
                ]

                .apply(
                    numeric_value
                )
            )


    # ========================================================
    # THIRD-DOWN %
    #
    # Historical builder turned strings such as 6-17 into
    # 6 / 17.
    # ========================================================

    if (
        "thirdDownEff"
        in stats_wide.columns
    ):

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


    stats_wide = stats_wide.drop(

        columns=[
            "thirdDownEff"
        ],

        errors="ignore"
    )


    # ========================================================
    # TEAM STATS
    # ========================================================

    team_games = team_games.merge(

        stats_wide,

        on=[
            "game_id",
            "team"
        ],

        how="left"
    )


    # ========================================================
    # OPPONENT STATS
    # ========================================================

    opponent_stats = stats_wide.copy()


    stat_columns = [

        column

        for column
        in opponent_stats.columns

        if column not in [

            "game_id",
            "team"
        ]
    ]


    opponent_stats = opponent_stats.rename(

        columns={

            "team":
                "opponent",

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


    # ========================================================
    # DERIVED GAME METRICS
    # ========================================================

    team_games[
        "margin"
    ] = (

        team_games[
            "points_for"
        ]

        -

        team_games[
            "points_against"
        ]
    )


    team_games[
        "win"
    ] = (

        team_games[
            "margin"
        ]
        >
        0

    ).astype(
        float
    )


    team_games[
        "turnover_margin"
    ] = (

        team_games.get(
            "opponent_turnovers",
            np.nan
        )

        -

        team_games.get(
            "turnovers",
            np.nan
        )
    )


    # ========================================================
    # EXACT V1 POPULATION
    #
    # Only FBS-v-FBS games contribute rolling history.
    # ========================================================

    history = team_games[

        (
            team_games[
                "team_classification"
            ]
            .astype(str)
            .str.lower()
            ==
            "fbs"
        )

        &

        (
            team_games[
                "opponent_classification"
            ]
            .astype(str)
            .str.lower()
            ==
            "fbs"
        )

    ].copy()


    history = history.sort_values(

        by=[

            "team",

            "start_date",

            "game_id"
        ]

    ).reset_index(
        drop=True
    )


    return history


# ============================================================
# LOAD STATIC TEAM CONTEXT
# ============================================================

def load_context(
    connection,
    season
):

    talent = pd.read_sql_query(
        """
        SELECT
            team,
            talent

        FROM team_talent

        WHERE season = ?
        """,

        connection,

        params=[
            season
        ]
    )


    returning = pd.read_sql_query(
        """
        SELECT

            team,

            percent_ppa,

            percent_passing_ppa,

            percent_receiving_ppa,

            percent_rushing_ppa

        FROM returning_production

        WHERE season = ?
        """,

        connection,

        params=[
            season
        ]
    )


    talent_map = {

        row[
            "team"
        ]:
            row[
                "talent"
            ]

        for _, row
        in talent.iterrows()
    }


    returning_map = {

        row[
            "team"
        ]: {

            "percent_ppa":
                row[
                    "percent_ppa"
                ],

            "percent_passing_ppa":
                row[
                    "percent_passing_ppa"
                ],

            "percent_receiving_ppa":
                row[
                    "percent_receiving_ppa"
                ],

            "percent_rushing_ppa":
                row[
                    "percent_rushing_ppa"
                ]
        }

        for _, row
        in returning.iterrows()
    }


    return (
        talent_map,
        returning_map
    )


# ============================================================
# TEAM HISTORY SUMMARY
#
# This reproduces:
#
#   .shift(1)
#   expanding().mean()
#   rolling(window=3).mean()
#
# from the historical feature builder.
# ============================================================

def summarize_prior_history(
    team_history,
    target_start_date,
    target_game_id
):

    if (
        team_history is None
        or
        len(team_history) == 0
    ):

        result = {

            "pregame_games_played":
                0.0
        }


        for metric in ROLLING_METRICS:

            result[
                f"pregame_avg_{metric}"
            ] = np.nan


            result[
                f"pregame_last3_{metric}"
            ] = np.nan


        return result


    prior = team_history[

        (
            team_history[
                "start_date"
            ]
            <
            target_start_date
        )

        |

        (
            (
                team_history[
                    "start_date"
                ]
                ==
                target_start_date
            )

            &

            (
                team_history[
                    "game_id"
                ]
                <
                target_game_id
            )
        )

    ].copy()


    prior = prior.sort_values(

        by=[
            "start_date",
            "game_id"
        ]
    )


    result = {

        "pregame_games_played":
            float(
                len(
                    prior
                )
            )
    }


    for metric in ROLLING_METRICS:

        if metric not in prior.columns:

            result[
                f"pregame_avg_{metric}"
            ] = np.nan


            result[
                f"pregame_last3_{metric}"
            ] = np.nan


            continue


        values = pd.to_numeric(

            prior[
                metric
            ],

            errors="coerce"
        )


        if len(
            values
        ) == 0:

            avg_value = np.nan


        else:

            avg_value = values.mean()


        last_three = values.tail(
            3
        )


        if len(
            last_three
        ) == 0:

            last_three_value = np.nan


        else:

            last_three_value = (
                last_three.mean()
            )


        result[
            f"pregame_avg_{metric}"
        ] = avg_value


        result[
            f"pregame_last3_{metric}"
        ] = last_three_value


    return result


# ============================================================
# BUILD ONE GAME
# ============================================================

def build_game_features(
    game,
    history_by_team,
    talent_map,
    returning_map,
    model_features
):

    game_id = int(
        game[
            "game_id"
        ]
    )


    target_start_date = game[
        "start_date"
    ]


    home_team = game[
        "home_team"
    ]


    away_team = game[
        "away_team"
    ]


    home_history = history_by_team.get(
        home_team
    )


    away_history = history_by_team.get(
        away_team
    )


    home_summary = summarize_prior_history(

        home_history,

        target_start_date,

        game_id
    )


    away_summary = summarize_prior_history(

        away_history,

        target_start_date,

        game_id
    )


    row = {

        "game_id":
            game_id,

        "season":
            int(
                game[
                    "season"
                ]
            ),

        "week":
            int(
                game[
                    "week"
                ]
            ),

        "start_date":
            target_start_date,

        "away_team":
            away_team,

        "home_team":
            home_team
    }


    # ========================================================
    # ROLLING DIFFERENCES
    # ========================================================

    row[
        "diff_pregame_games_played"
    ] = (

        home_summary[
            "pregame_games_played"
        ]

        -

        away_summary[
            "pregame_games_played"
        ]
    )


    for metric in ROLLING_METRICS:

        for window_name in [

            "avg",
            "last3"

        ]:

            feature_name = (

                f"pregame_"
                f"{window_name}_"
                f"{metric}"
            )


            difference_name = (

                f"diff_"
                f"{feature_name}"
            )


            row[
                difference_name
            ] = (

                home_summary[
                    feature_name
                ]

                -

                away_summary[
                    feature_name
                ]
            )


    # ========================================================
    # NEUTRAL SITE
    # ========================================================

    neutral_site = game[
        "neutral_site"
    ]


    if pd.isna(
        neutral_site
    ):

        row[
            "neutral_site"
        ] = np.nan


    else:

        row[
            "neutral_site"
        ] = float(
            neutral_site
        )


    # ========================================================
    # ELO DIFFERENCE
    # ========================================================

    home_elo = pd.to_numeric(

        game[
            "home_pregame_elo"
        ],

        errors="coerce"
    )


    away_elo = pd.to_numeric(

        game[
            "away_pregame_elo"
        ],

        errors="coerce"
    )


    row[
        "pregame_elo_diff"
    ] = (

        home_elo

        -

        away_elo
    )


    # ========================================================
    # TALENT
    # ========================================================

    home_talent = talent_map.get(
        home_team,
        np.nan
    )


    away_talent = talent_map.get(
        away_team,
        np.nan
    )


    row[
        "talent_diff"
    ] = (

        home_talent

        -

        away_talent
    )


    # ========================================================
    # RETURNING PRODUCTION
    # ========================================================

    home_returning = returning_map.get(
        home_team,
        {}
    )


    away_returning = returning_map.get(
        away_team,
        {}
    )


    returning_pairs = [

        (
            "percent_ppa",
            "returning_ppa_diff"
        ),

        (
            "percent_passing_ppa",
            "returning_passing_ppa_diff"
        ),

        (
            "percent_receiving_ppa",
            "returning_receiving_ppa_diff"
        ),

        (
            "percent_rushing_ppa",
            "returning_rushing_ppa_diff"
        )
    ]


    for (
        source_name,
        feature_name
    ) in returning_pairs:

        home_value = home_returning.get(
            source_name,
            np.nan
        )


        away_value = away_returning.get(
            source_name,
            np.nan
        )


        row[
            feature_name
        ] = (

            home_value

            -

            away_value
        )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    row[
        "home_history_games"
    ] = home_summary[
        "pregame_games_played"
    ]


    row[
        "away_history_games"
    ] = away_summary[
        "pregame_games_played"
    ]


    row[
        "home_talent_available"
    ] = int(
        pd.notna(
            home_talent
        )
    )


    row[
        "away_talent_available"
    ] = int(
        pd.notna(
            away_talent
        )
    )


    row[
        "home_returning_available"
    ] = int(
        home_team
        in
        returning_map
    )


    row[
        "away_returning_available"
    ] = int(
        away_team
        in
        returning_map
    )


    # ========================================================
    # EXACT FEATURE COVERAGE
    # ========================================================

    feature_values = pd.Series(
        {
            feature:
                row.get(
                    feature,
                    np.nan
                )

            for feature
            in model_features
        }
    )


    row[
        "model_feature_count"
    ] = len(
        model_features
    )


    row[
        "model_non_null_features"
    ] = int(
        feature_values.notna().sum()
    )


    row[
        "model_missing_features"
    ] = int(
        feature_values.isna().sum()
    )


    # Static context is not something we want to impute
    # silently at the live-data stage.
    required_context = [

        "neutral_site",

        "pregame_elo_diff",

        "talent_diff",

        "returning_ppa_diff",

        "returning_passing_ppa_diff",

        "returning_receiving_ppa_diff",

        "returning_rushing_ppa_diff"
    ]


    row[
        "context_ready"
    ] = int(

        all(

            pd.notna(
                row.get(
                    feature
                )
            )

            for feature
            in required_context
        )
    )


    return row


# ============================================================
# BUILD FEATURES FOR A SET OF GAMES
# ============================================================

def build_feature_frame(
    games,
    history,
    talent_map,
    returning_map,
    model_features
):

    history_by_team = {

        team:
            group.copy()

        for team, group
        in history.groupby(
            "team",
            sort=False
        )
    }


    rows = []


    for _, game in games.iterrows():

        rows.append(

            build_game_features(

                game,

                history_by_team,

                talent_map,

                returning_map,

                model_features
            )
        )


    frame = pd.DataFrame(
        rows
    )


    return frame


# ============================================================
# HISTORICAL PARITY TEST
#
# Before generating live features, reconstruct 2025.
#
# If this does not match the exact features already produced
# by the original historical builder, STOP.
# ============================================================

def run_parity_test(
    connection,
    model_features
):

    header(
        "HISTORICAL PARITY TEST — 2025"
    )


    history = load_team_history(
        connection,
        PARITY_SEASON
    )


    talent_map, returning_map = load_context(
        connection,
        PARITY_SEASON
    )


    games = pd.read_sql_query(
        """
        SELECT

            game_id,

            season,

            week,

            start_date,

            neutral_site,

            home_team,

            away_team,

            home_pregame_elo,

            away_pregame_elo

        FROM games

        WHERE
            season = ?
            AND home_points IS NOT NULL
            AND away_points IS NOT NULL
            AND LOWER(home_classification) = 'fbs'
            AND LOWER(away_classification) = 'fbs'

        ORDER BY
            start_date,
            game_id
        """,

        connection,

        params=[
            PARITY_SEASON
        ]
    )


    games[
        "start_date"
    ] = pd.to_datetime(

        games[
            "start_date"
        ],

        utc=True,

        errors="coerce"
    )


    reconstructed = build_feature_frame(

        games,

        history,

        talent_map,

        returning_map,

        model_features
    )


    historical = pd.read_csv(

        HISTORICAL_FEATURE_FILE,

        usecols=[
            "game_id",
            *model_features
        ]
    )


    historical[
        "game_id"
    ] = pd.to_numeric(

        historical[
            "game_id"
        ],

        errors="coerce"
    )


    historical = historical.dropna(
        subset=[
            "game_id"
        ]
    )


    historical[
        "game_id"
    ] = historical[
        "game_id"
    ].astype(
        int
    )


    comparison = reconstructed[
        [
            "game_id",
            *model_features
        ]
    ].merge(

        historical,

        on="game_id",

        how="inner",

        suffixes=(
            "_live",
            "_historical"
        )
    )


    print(
        f"2025 FBS-v-FBS games reconstructed: "
        f"{len(reconstructed):,}"
    )


    print(
        f"Historical rows matched by game ID: "
        f"{len(comparison):,}"
    )


    if len(
        comparison
    ) != len(
        reconstructed
    ):

        raise RuntimeError(

            "Parity test could not match every "
            "reconstructed 2025 game to features_v2.csv."
        )


    total_mismatches = 0


    mismatch_features = []


    max_absolute_difference = 0.0


    for feature in model_features:

        live_values = pd.to_numeric(

            comparison[
                f"{feature}_live"
            ],

            errors="coerce"
        ).to_numpy(
            dtype=float
        )


        historical_values = pd.to_numeric(

            comparison[
                f"{feature}_historical"
            ],

            errors="coerce"
        ).to_numpy(
            dtype=float
        )


        equal = np.isclose(

            live_values,

            historical_values,

            rtol=PARITY_TOLERANCE,

            atol=PARITY_TOLERANCE,

            equal_nan=True
        )


        mismatch_count = int(
            (
                ~equal
            ).sum()
        )


        if mismatch_count > 0:

            mismatch_features.append(
                (
                    feature,
                    mismatch_count
                )
            )


            total_mismatches += (
                mismatch_count
            )


            valid = (

                np.isfinite(
                    live_values
                )

                &

                np.isfinite(
                    historical_values
                )
            )


            if valid.any():

                feature_max_difference = float(

                    np.max(

                        np.abs(

                            live_values[
                                valid
                            ]

                            -

                            historical_values[
                                valid
                            ]
                        )
                    )
                )


                max_absolute_difference = max(

                    max_absolute_difference,

                    feature_max_difference
                )


    if total_mismatches == 0:

        print()

        print(
            "PARITY RESULT: PASS"
        )


        print(
            "All 54 reconstructed features match "
            "the historical feature file within tolerance."
        )


    else:

        print()

        print(
            "PARITY RESULT: FAIL"
        )


        print(
            f"Total mismatched values: "
            f"{total_mismatches:,}"
        )


        print(
            f"Maximum numeric difference: "
            f"{max_absolute_difference}"
        )


        print()

        print(
            "Features with mismatches:"
        )


        for (
            feature,
            mismatch_count
        ) in mismatch_features:

            print(
                f"  {feature:55} "
                f"{mismatch_count:,}"
            )


        raise RuntimeError(

            "Historical parity failed. "
            "Live 2026 features were NOT generated."
        )


# ============================================================
# LIVE 2026
# ============================================================

def build_live_2026(
    connection,
    model_features
):

    header(
        "BUILD LIVE 2026 FEATURES"
    )


    now_utc = pd.Timestamp.now(
        tz="UTC"
    )


    # ========================================================
    # FIND FUTURE FBS-v-FBS GAMES
    # ========================================================

    games = pd.read_sql_query(
        """
        SELECT

            game_id,

            season,

            week,

            start_date,

            neutral_site,

            home_team,

            away_team,

            home_pregame_elo,

            away_pregame_elo

        FROM games

        WHERE
            season = ?
            AND (
                home_points IS NULL
                OR away_points IS NULL
            )
            AND LOWER(home_classification) = 'fbs'
            AND LOWER(away_classification) = 'fbs'

        ORDER BY
            start_date,
            game_id
        """,

        connection,

        params=[
            LIVE_SEASON
        ]
    )


    games[
        "start_date"
    ] = pd.to_datetime(

        games[
            "start_date"
        ],

        utc=True,

        errors="coerce"
    )


    games = games[

        games[
            "start_date"
        ].notna()

        &

        (
            games[
                "start_date"
            ]
            >
            now_utc
        )

    ].copy()


    if len(
        games
    ) == 0:

        raise RuntimeError(
            "No future 2026 FBS-v-FBS games found."
        )


    # ========================================================
    # ACTIVE WEEK
    #
    # Game-day V1 only scores the nearest upcoming week.
    #
    # This prevents us from pretending that today's Week 1
    # information is a final pregame assessment of a Week 10
    # matchup.
    # ========================================================

    active_week = int(

        pd.to_numeric(

            games[
                "week"
            ],

            errors="coerce"
        )

        .dropna()

        .min()
    )


    active_games = games[

        pd.to_numeric(

            games[
                "week"
            ],

            errors="coerce"
        )
        ==
        active_week

    ].copy()


    print(
        f"As of UTC: "
        f"{now_utc.isoformat()}"
    )


    print(
        f"Future FBS-v-FBS games: "
        f"{len(games):,}"
    )


    print(
        f"Active upcoming week: "
        f"{active_week}"
    )


    print(
        f"Games in active week: "
        f"{len(active_games):,}"
    )


    # ========================================================
    # HISTORY
    # ========================================================

    history = load_team_history(
        connection,
        LIVE_SEASON
    )


    print(
        f"Completed 2026 FBS-v-FBS "
        f"team-game history rows: "
        f"{len(history):,}"
    )


    print(
        f"Teams with FBS-v-FBS history: "
        f"{history['team'].nunique():,}"
    )


    # ========================================================
    # CONTEXT
    # ========================================================

    talent_map, returning_map = load_context(
        connection,
        LIVE_SEASON
    )


    # ========================================================
    # BUILD
    # ========================================================

    live = build_feature_frame(

        active_games,

        history,

        talent_map,

        returning_map,

        model_features
    )


    # ========================================================
    # FINAL CONTRACT VALIDATION
    # ========================================================

    missing_columns = [

        feature

        for feature
        in model_features

        if feature not in live.columns
    ]


    if missing_columns:

        raise RuntimeError(

            "Live feature frame is missing "
            "model columns:\n"

            +

            "\n".join(
                missing_columns
            )
        )


    # ========================================================
    # COLUMN ORDER
    # ========================================================

    metadata_columns = [

        "game_id",

        "season",

        "week",

        "start_date",

        "away_team",

        "home_team",

        "home_history_games",

        "away_history_games",

        "home_talent_available",

        "away_talent_available",

        "home_returning_available",

        "away_returning_available",

        "model_feature_count",

        "model_non_null_features",

        "model_missing_features",

        "context_ready"
    ]


    live = live[

        metadata_columns

        +

        model_features
    ]


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    live.to_csv(
        OUTPUT_CSV,
        index=False
    )


    # ========================================================
    # AUDIT
    # ========================================================

    header(
        "LIVE FEATURE STATUS"
    )


    print(
        f"Rows written:             "
        f"{len(live):,}"
    )


    print(
        f"Exact model features:     "
        f"{len(model_features)}"
    )


    print(
        f"Context-ready games:      "
        f"{int(live['context_ready'].sum()):,}/"
        f"{len(live):,}"
    )


    both_have_history = (

        (
            live[
                "home_history_games"
            ]
            >
            0
        )

        &

        (
            live[
                "away_history_games"
            ]
            >
            0
        )
    )


    print(
        f"Both teams have 2026 "
        f"FBS history:              "
        f"{int(both_have_history.sum()):,}/"
        f"{len(live):,}"
    )


    print()

    print(
        "MODEL FEATURE COMPLETENESS"
    )

    print(
        "-" * 110
    )


    completeness = (

        live[
            "model_non_null_features"
        ]

        .value_counts()

        .sort_index(
            ascending=False
        )
    )


    for feature_count, game_count in completeness.items():

        print(

            f"{int(feature_count):2}/54 "
            f"non-null features: "
            f"{int(game_count):,} games"
        )


    # ========================================================
    # MISSING CONTEXT GAMES
    # ========================================================

    missing_context = live[

        live[
            "context_ready"
        ]
        ==
        0

    ]


    if len(
        missing_context
    ) > 0:

        print()

        print(
            "GAMES WITH MISSING STATIC CONTEXT"
        )

        print(
            "-" * 110
        )


        for _, row in missing_context.iterrows():

            missing = []


            for feature in [

                "neutral_site",

                "pregame_elo_diff",

                "talent_diff",

                "returning_ppa_diff",

                "returning_passing_ppa_diff",

                "returning_receiving_ppa_diff",

                "returning_rushing_ppa_diff"
            ]:

                if pd.isna(
                    row[
                        feature
                    ]
                ):

                    missing.append(
                        feature
                    )


            print(

                f"{row['away_team']} @ "
                f"{row['home_team']} | "

                f"missing: "
                f"{', '.join(missing)}"
            )


    # ========================================================
    # SAMPLE
    # ========================================================

    print()

    print(
        "ACTIVE-WEEK SAMPLE"
    )

    print(
        "-" * 110
    )


    sample_columns = [

        "away_team",

        "home_team",

        "home_history_games",

        "away_history_games",

        "pregame_elo_diff",

        "talent_diff",

        "model_non_null_features",

        "context_ready"
    ]


    print(

        live[
            sample_columns
        ]

        .head(
            15
        )

        .to_string(
            index=False
        )
    )


    print()

    print(
        f"Output:"
    )


    print(
        f"  {OUTPUT_CSV}"
    )


    return live


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "BUILD LIVE FEATURES 2026 — V1"
    )

    print(
        "=" * 110
    )


    print(
        "Purpose:"
    )


    print(
        "  Reproduce the exact 54-feature neural-network "
        "input contract for the active 2026 slate."
    )


    print()

    print(
        "Safety:"
    )


    print(
        "  No APIs are called."
    )


    print(
        "  Historical parity must pass before live "
        "features are written."
    )


    print(
        "  Current-game results are never used as "
        "pregame history."
    )


    model_features = get_model_features()


    print()

    print(
        f"Exact model feature count: "
        f"{len(model_features)}"
    )


    if not DATABASE_FILE.exists():

        raise RuntimeError(
            f"Database not found: "
            f"{DATABASE_FILE}"
        )


    connection = sqlite3.connect(
        DATABASE_FILE
    )


    try:

        run_parity_test(

            connection,

            model_features
        )


        live = build_live_2026(

            connection,

            model_features
        )


    finally:

        connection.close()


    header(
        "LIVE FEATURE BUILD COMPLETE"
    )


    print(
        f"Active-slate rows: "
        f"{len(live):,}"
    )


    print(
        "Historical parity: PASS"
    )


    print(
        "2026 live feature contract: PASS"
    )


    print()

    print(
        "Next step:"
    )


    print(
        "  Freeze the production neural-network "
        "model and preprocessing artifacts."
    )


if __name__ == "__main__":

    main()