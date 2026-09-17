import os

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

MARKET_FILE = (
    "data/processed/"
    "nn_vs_bovada_market_v1.csv"
)

PREDICTIONS_FILE = (
    "data/processed/"
    "nn_seed_stability_predictions_v1.csv"
)

THRESHOLD_OUTPUT_FILE = (
    "data/processed/"
    "moneyline_edge_threshold_backtest_v1.csv"
)

SEASON_OUTPUT_FILE = (
    "data/processed/"
    "moneyline_edge_threshold_by_season_v1.csv"
)

BUCKET_OUTPUT_FILE = (
    "data/processed/"
    "moneyline_edge_buckets_v1.csv"
)


DEVELOPMENT_YEARS = [
    2021,
    2022,
    2023,
    2024
]


# ------------------------------------------------------------
# These are predetermined diagnostic thresholds.
#
# IMPORTANT:
# We are NOT going to simply pick whichever threshold has
# the highest historical ROI and call it our strategy.
# ------------------------------------------------------------

EDGE_THRESHOLDS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15
]


# ------------------------------------------------------------
# Disjoint buckets let us see whether larger predicted
# edges actually correspond to better realized performance.
# ------------------------------------------------------------

EDGE_BUCKETS = [
    (-np.inf, 0.00, "<0%"),
    (0.00, 0.02, "0-2%"),
    (0.02, 0.04, "2-4%"),
    (0.04, 0.06, "4-6%"),
    (0.06, 0.08, "6-8%"),
    (0.08, 0.10, "8-10%"),
    (0.10, 0.15, "10-15%"),
    (0.15, np.inf, "15%+")
]


# ------------------------------------------------------------
# Used only to express drawdown as an intuitive percentage.
#
# All actual betting calculations remain flat 1-unit wagers.
# ------------------------------------------------------------

STARTING_BANKROLL_UNITS = 100.0


# ============================================================
# START
# ============================================================

print(
    "MONEYLINE EDGE BACKTEST — V1"
)

print(
    "=" * 100
)


print(
    "Every qualifying wager risks exactly 1 unit."
)


print(
    "2025 will NOT be loaded or evaluated."
)


# ============================================================
# LOAD MARKET COMPARISON DATA
# ============================================================

games = pd.read_csv(
    MARKET_FILE
)


games = games[
    games[
        "season"
    ].isin(
        DEVELOPMENT_YEARS
    )
].copy()


if len(
    games
) == 0:

    raise RuntimeError(
        "No development games were loaded."
    )


if (
    games[
        "season"
    ].max()
    >= 2025
):

    raise RuntimeError(
        "2025 leakage detected."
    )


print()

print(
    f"Market/model games loaded: "
    f"{len(games):,}"
)


# ============================================================
# LOAD GAME METADATA
#
# The earlier market-comparison file intentionally contained
# only one row per game, but it did not preserve week/team
# metadata.
#
# We recover that from the NN prediction file so bets can be
# placed in chronological season/week order.
# ============================================================

prediction_metadata = pd.read_csv(
    PREDICTIONS_FILE,
    usecols=[
        "game_id",
        "season",
        "week",
        "away_team",
        "home_team",
        "architecture"
    ]
)


prediction_metadata = prediction_metadata[
    (
        prediction_metadata[
            "architecture"
        ]
        ==
        "NN Small"
    )
    &
    (
        prediction_metadata[
            "season"
        ].isin(
            DEVELOPMENT_YEARS
        )
    )
].copy()


prediction_metadata = (

    prediction_metadata

    .drop_duplicates(
        subset=[
            "game_id",
            "season"
        ]
    )

    .drop(
        columns=[
            "architecture"
        ]
    )
)


games = games.merge(

    prediction_metadata,

    on=[
        "game_id",
        "season"
    ],

    how="left"
)


if (
    games[
        "week"
    ].isna().any()
):

    missing_metadata = int(
        games[
            "week"
        ].isna().sum()
    )


    raise RuntimeError(

        f"Missing week metadata for "
        f"{missing_metadata} games."
    )


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [

    "actual_home_win",

    "model_best_bet_side",

    "model_best_ev_per_dollar",

    "model_price_edge",

    "model_fair_market_edge",

    "model_best_side_probability",

    "best_side_raw_break_even_probability",

    "best_side_market_fair_probability",

    "best_side_flat_profit",

    "home_moneyline",

    "away_moneyline",

    "home_decimal_odds",

    "away_decimal_odds"
]


missing_columns = [

    column

    for column in required_columns

    if column not in games.columns
]


if missing_columns:

    raise RuntimeError(

        "Required columns missing: "

        +

        ", ".join(
            missing_columns
        )
    )


# ============================================================
# SELECTED BET PRICE
# ============================================================

games[
    "selected_moneyline"
] = np.where(

    games[
        "model_best_bet_side"
    ]
    ==
    "HOME",

    games[
        "home_moneyline"
    ],

    games[
        "away_moneyline"
    ]
)


games[
    "selected_decimal_odds"
] = np.where(

    games[
        "model_best_bet_side"
    ]
    ==
    "HOME",

    games[
        "home_decimal_odds"
    ],

    games[
        "away_decimal_odds"
    ]
)


# ============================================================
# ACTUAL BET RESULT
# ============================================================

games[
    "bet_won"
] = np.where(

    games[
        "model_best_bet_side"
    ]
    ==
    "HOME",

    games[
        "actual_home_win"
    ]
    ==
    1,

    games[
        "actual_home_win"
    ]
    ==
    0
)


games[
    "bet_won"
] = games[
    "bet_won"
].astype(
    int
)


# ============================================================
# FAVORITE / UNDERDOG LABEL
# ============================================================

games[
    "bet_type"
] = np.where(

    games[
        "selected_moneyline"
    ]
    < 0,

    "FAVORITE",

    "UNDERDOG"
)


# ============================================================
# CHRONOLOGICAL ORDER
#
# Game ID is used only as a deterministic order inside the
# same week because exact line timestamps are unavailable.
# ============================================================

games = (

    games

    .sort_values(
        [
            "season",
            "week",
            "game_id"
        ]
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# RISK HELPERS
# ============================================================

def maximum_drawdown(
    profits
):

    profits = np.asarray(
        profits,
        dtype=float
    )


    if len(
        profits
    ) == 0:

        return np.nan


    cumulative = np.cumsum(
        profits
    )


    bankroll_curve = np.concatenate(
        [
            np.array(
                [0.0]
            ),
            cumulative
        ]
    )


    running_peak = np.maximum.accumulate(
        bankroll_curve
    )


    drawdowns = (

        running_peak

        -
        bankroll_curve
    )


    return float(
        np.max(
            drawdowns
        )
    )


def longest_losing_streak(
    wins
):

    longest = 0

    current = 0


    for won in wins:

        if int(
            won
        ) == 0:

            current += 1

            longest = max(
                longest,
                current
            )

        else:

            current = 0


    return int(
        longest
    )


# ============================================================
# PERFORMANCE SUMMARY FUNCTION
# ============================================================

def summarize_bets(
    bet_data
):

    bet_data = bet_data.copy()


    bets = len(
        bet_data
    )


    if bets == 0:

        return {
            "bets":
                0,

            "wins":
                0,

            "losses":
                0,

            "win_rate":
                np.nan,

            "profit_units":
                0.0,

            "roi":
                np.nan,

            "expected_profit_units":
                0.0,

            "expected_roi":
                np.nan,

            "average_price_edge":
                np.nan,

            "average_fair_market_edge":
                np.nan,

            "average_model_probability":
                np.nan,

            "average_break_even_probability":
                np.nan,

            "average_decimal_odds":
                np.nan,

            "favorite_pct":
                np.nan,

            "underdog_pct":
                np.nan,

            "max_drawdown_units":
                np.nan,

            "max_drawdown_pct_of_100_unit_bankroll":
                np.nan,

            "longest_losing_streak":
                np.nan
        }


    wins = int(
        bet_data[
            "bet_won"
        ].sum()
    )


    losses = (
        bets
        -
        wins
    )


    profit_units = float(

        bet_data[
            "best_side_flat_profit"
        ].sum()
    )


    # --------------------------------------------------------
    # Since each wager risks exactly 1 unit:
    #
    # ROI = profit / total units risked
    #     = profit / number of bets
    # --------------------------------------------------------

    roi = (
        profit_units
        /
        bets
    )


    expected_profit_units = float(

        bet_data[
            "model_best_ev_per_dollar"
        ].sum()
    )


    expected_roi = (

        expected_profit_units
        /
        bets
    )


    drawdown_units = maximum_drawdown(

        bet_data[
            "best_side_flat_profit"
        ].to_numpy()
    )


    return {

        "bets":
            bets,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            wins
            /
            bets,

        "profit_units":
            profit_units,

        "roi":
            roi,

        "expected_profit_units":
            expected_profit_units,

        "expected_roi":
            expected_roi,

        "average_price_edge":
            bet_data[
                "model_price_edge"
            ].mean(),

        "average_fair_market_edge":
            bet_data[
                "model_fair_market_edge"
            ].mean(),

        "average_model_probability":
            bet_data[
                "model_best_side_probability"
            ].mean(),

        "average_break_even_probability":
            bet_data[
                "best_side_raw_break_even_probability"
            ].mean(),

        "average_decimal_odds":
            bet_data[
                "selected_decimal_odds"
            ].mean(),

        "favorite_pct":
            (
                bet_data[
                    "bet_type"
                ]
                ==
                "FAVORITE"
            ).mean(),

        "underdog_pct":
            (
                bet_data[
                    "bet_type"
                ]
                ==
                "UNDERDOG"
            ).mean(),

        "max_drawdown_units":
            drawdown_units,

        "max_drawdown_pct_of_100_unit_bankroll":
            (
                drawdown_units
                /
                STARTING_BANKROLL_UNITS
            ),

        "longest_losing_streak":
            longest_losing_streak(

                bet_data[
                    "bet_won"
                ].to_numpy()
            )
    }


# ============================================================
# CUMULATIVE EDGE THRESHOLD TESTS
#
# Example:
#
# 6% threshold means:
#
# bet every game where model probability exceeds the
# actual break-even price by at least 6 percentage points.
# ============================================================

threshold_rows = []

season_rows = []


for threshold in EDGE_THRESHOLDS:

    qualifying = games[
        games[
            "model_price_edge"
        ]
        >=
        threshold
    ].copy()


    summary = summarize_bets(
        qualifying
    )


    summary[
        "minimum_price_edge"
    ] = threshold


    threshold_rows.append(
        summary
    )


    # --------------------------------------------------------
    # Same threshold broken out by season.
    # --------------------------------------------------------

    for season in DEVELOPMENT_YEARS:

        season_bets = qualifying[
            qualifying[
                "season"
            ]
            ==
            season
        ].copy()


        season_summary = summarize_bets(
            season_bets
        )


        season_summary[
            "minimum_price_edge"
        ] = threshold


        season_summary[
            "season"
        ] = season


        season_rows.append(
            season_summary
        )


threshold_results = pd.DataFrame(
    threshold_rows
)


season_results = pd.DataFrame(
    season_rows
)


# ============================================================
# DISJOINT EDGE BUCKETS
#
# This tells us whether bigger claimed edges actually perform
# better instead of merely looking better in the model.
# ============================================================

bucket_rows = []


for (
    lower_bound,
    upper_bound,
    label
) in EDGE_BUCKETS:

    if np.isneginf(
        lower_bound
    ):

        mask = (
            games[
                "model_price_edge"
            ]
            <
            upper_bound
        )


    elif np.isposinf(
        upper_bound
    ):

        mask = (
            games[
                "model_price_edge"
            ]
            >=
            lower_bound
        )


    else:

        mask = (

            (
                games[
                    "model_price_edge"
                ]
                >=
                lower_bound
            )

            &

            (
                games[
                    "model_price_edge"
                ]
                <
                upper_bound
            )
        )


    bucket_data = games[
        mask
    ].copy()


    summary = summarize_bets(
        bucket_data
    )


    summary[
        "edge_bucket"
    ] = label


    summary[
        "lower_edge"
    ] = lower_bound


    summary[
        "upper_edge"
    ] = upper_bound


    bucket_rows.append(
        summary
    )


bucket_results = pd.DataFrame(
    bucket_rows
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


threshold_results.to_csv(
    THRESHOLD_OUTPUT_FILE,
    index=False
)


season_results.to_csv(
    SEASON_OUTPUT_FILE,
    index=False
)


bucket_results.to_csv(
    BUCKET_OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT CUMULATIVE THRESHOLD RESULTS
# ============================================================

print()

print(
    "=" * 120
)

print(
    "FLAT-BET RESULTS BY MINIMUM MODEL PRICE EDGE"
)

print(
    "=" * 120
)


threshold_display = threshold_results[
    [
        "minimum_price_edge",
        "bets",
        "wins",
        "win_rate",
        "profit_units",
        "roi",
        "expected_roi",
        "average_price_edge",
        "average_decimal_odds",
        "favorite_pct",
        "max_drawdown_units",
        "longest_losing_streak"
    ]
].copy()


print(

    threshold_display.to_string(

        index=False,

        formatters={

            "minimum_price_edge":
                lambda value:
                    f"{value:.0%}",

            "win_rate":
                lambda value:
                    (
                        f"{value:.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "profit_units":
                lambda value:
                    f"{value:+.2f}",

            "roi":
                lambda value:
                    (
                        f"{value:+.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "expected_roi":
                lambda value:
                    (
                        f"{value:+.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "average_price_edge":
                lambda value:
                    (
                        f"{value:.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "average_decimal_odds":
                lambda value:
                    (
                        f"{value:.3f}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "favorite_pct":
                lambda value:
                    (
                        f"{value:.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "max_drawdown_units":
                lambda value:
                    (
                        f"{value:.2f}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    )
        }
    )
)


# ============================================================
# PRINT EDGE BUCKET RESULTS
# ============================================================

print()

print(
    "=" * 120
)

print(
    "DISJOINT MODEL PRICE-EDGE BUCKETS"
)

print(
    "=" * 120
)


bucket_display = bucket_results[
    [
        "edge_bucket",
        "bets",
        "win_rate",
        "profit_units",
        "roi",
        "expected_roi",
        "average_price_edge",
        "average_decimal_odds",
        "favorite_pct",
        "max_drawdown_units"
    ]
].copy()


print(

    bucket_display.to_string(

        index=False,

        formatters={

            "win_rate":
                lambda value:
                    (
                        f"{value:.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "profit_units":
                lambda value:
                    f"{value:+.2f}",

            "roi":
                lambda value:
                    (
                        f"{value:+.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "expected_roi":
                lambda value:
                    (
                        f"{value:+.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "average_price_edge":
                lambda value:
                    (
                        f"{value:.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "average_decimal_odds":
                lambda value:
                    (
                        f"{value:.3f}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "favorite_pct":
                lambda value:
                    (
                        f"{value:.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "max_drawdown_units":
                lambda value:
                    (
                        f"{value:.2f}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    )
        }
    )
)


# ============================================================
# PRINT BY-SEASON RESULTS FOR KEY THRESHOLDS
# ============================================================

print()

print(
    "=" * 120
)

print(
    "BY-SEASON RESULTS — KEY THRESHOLDS"
)

print(
    "=" * 120
)


key_thresholds = [
    0.00,
    0.04,
    0.06,
    0.08,
    0.10
]


season_display = season_results[
    season_results[
        "minimum_price_edge"
    ].isin(
        key_thresholds
    )
][
    [
        "minimum_price_edge",
        "season",
        "bets",
        "win_rate",
        "profit_units",
        "roi",
        "expected_roi",
        "max_drawdown_units"
    ]
].copy()


print(

    season_display.to_string(

        index=False,

        formatters={

            "minimum_price_edge":
                lambda value:
                    f"{value:.0%}",

            "win_rate":
                lambda value:
                    (
                        f"{value:.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "profit_units":
                lambda value:
                    f"{value:+.2f}",

            "roi":
                lambda value:
                    (
                        f"{value:+.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "expected_roi":
                lambda value:
                    (
                        f"{value:+.2%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "max_drawdown_units":
                lambda value:
                    (
                        f"{value:.2f}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    )
        }
    )
)


# ============================================================
# FAVORITE VS UNDERDOG DIAGNOSTIC
#
# We are NOT using this to select a strategy yet.
# ============================================================

print()

print(
    "=" * 120
)

print(
    "ALL MODEL-LABELED +EV BETS — FAVORITE VS UNDERDOG"
)

print(
    "=" * 120
)


positive_ev = games[
    games[
        "model_price_edge"
    ]
    >=
    0.0
].copy()


side_rows = []


for bet_type in [
    "FAVORITE",
    "UNDERDOG"
]:

    side_data = positive_ev[
        positive_ev[
            "bet_type"
        ]
        ==
        bet_type
    ]


    side_summary = summarize_bets(
        side_data
    )


    side_summary[
        "bet_type"
    ] = bet_type


    side_rows.append(
        side_summary
    )


side_results = pd.DataFrame(
    side_rows
)


print(

    side_results[
        [
            "bet_type",
            "bets",
            "win_rate",
            "profit_units",
            "roi",
            "expected_roi",
            "average_price_edge",
            "average_decimal_odds",
            "max_drawdown_units",
            "longest_losing_streak"
        ]
    ].to_string(

        index=False,

        formatters={

            "win_rate":
                lambda value:
                    f"{value:.2%}",

            "profit_units":
                lambda value:
                    f"{value:+.2f}",

            "roi":
                lambda value:
                    f"{value:+.2%}",

            "expected_roi":
                lambda value:
                    f"{value:+.2%}",

            "average_price_edge":
                lambda value:
                    f"{value:.2%}",

            "average_decimal_odds":
                lambda value:
                    f"{value:.3f}",

            "max_drawdown_units":
                lambda value:
                    f"{value:.2f}"
        }
    )
)


# ============================================================
# FINISH
# ============================================================

print()

print(
    "=" * 100
)

print(
    "RESULT FILES"
)

print(
    "=" * 100
)


print(
    THRESHOLD_OUTPUT_FILE
)


print(
    SEASON_OUTPUT_FILE
)


print(
    BUCKET_OUTPUT_FILE
)


print()

print(
    "IMPORTANT:"
)


print(
    "These are DEVELOPMENT backtests."
)


print(
    "Do not choose the most profitable historical threshold "
    "and call it proven."
)


print(
    "2025 was not loaded, scored, or used."
)