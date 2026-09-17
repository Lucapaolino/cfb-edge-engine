import os

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

PREDICTIONS_FILE = (
    "data/processed/"
    "market_aware_win_predictions_v1.csv"
)

METADATA_FILE = (
    "data/processed/"
    "nn_seed_stability_predictions_v1.csv"
)

THRESHOLD_OUTPUT_FILE = (
    "data/processed/"
    "market_aware_bet_thresholds_v1.csv"
)

SEASON_OUTPUT_FILE = (
    "data/processed/"
    "market_aware_bets_by_season_v1.csv"
)

BUCKET_OUTPUT_FILE = (
    "data/processed/"
    "market_aware_edge_buckets_v1.csv"
)


MODEL_NAME = (
    "Tuned Blend"
)


TEST_YEARS = [
    2022,
    2023,
    2024
]


# ------------------------------------------------------------
# Price-edge thresholds.
#
# Example:
#
# model says 57%
# sportsbook break-even = 54%
#
# price edge = 3 percentage points
#
# It qualifies for the 0%, 1%, 2%, and 3% thresholds,
# but NOT 4%.
# ------------------------------------------------------------

EDGE_THRESHOLDS = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08
]


# ------------------------------------------------------------
# Disjoint buckets.
#
# These help us see whether larger claimed edges actually
# perform better.
# ------------------------------------------------------------

EDGE_BUCKETS = [
    (-np.inf, 0.00, "<0%"),
    (0.00, 0.01, "0-1%"),
    (0.01, 0.02, "1-2%"),
    (0.02, 0.03, "2-3%"),
    (0.03, 0.04, "3-4%"),
    (0.04, 0.05, "4-5%"),
    (0.05, 0.06, "5-6%"),
    (0.06, 0.08, "6-8%"),
    (0.08, np.inf, "8%+")
]


STARTING_BANKROLL_UNITS = 100.0


# ============================================================
# START
# ============================================================

print(
    "MARKET-AWARE MONEYLINE BET BACKTEST — V1"
)

print(
    "=" * 110
)


print(
    f"Model: {MODEL_NAME}"
)


print(
    f"Test years: {TEST_YEARS}"
)


print(
    "Each qualifying wager risks exactly 1 unit."
)


print(
    "2025 will NOT be loaded or evaluated."
)


# ============================================================
# MONEYLINE FUNCTIONS
# ============================================================

def american_to_break_even_probability(
    odds
):

    if pd.isna(
        odds
    ):

        return np.nan


    odds = float(
        odds
    )


    if odds < 0:

        return (

            abs(
                odds
            )

            /

            (
                abs(
                    odds
                )

                +
                100.0
            )
        )


    if odds > 0:

        return (

            100.0

            /

            (
                odds
                +
                100.0
            )
        )


    return np.nan


# ============================================================
# LOAD MARKET-AWARE PREDICTIONS
# ============================================================

predictions = pd.read_csv(
    PREDICTIONS_FILE
)


predictions = predictions[
    (
        predictions[
            "model"
        ]
        ==
        MODEL_NAME
    )
    &
    (
        predictions[
            "season"
        ].isin(
            TEST_YEARS
        )
    )
].copy()


if len(
    predictions
) == 0:

    raise RuntimeError(
        "No Tuned Blend predictions found."
    )


if (
    predictions[
        "season"
    ].max()
    >= 2025
):

    raise RuntimeError(
        "2025 leakage detected."
    )


print()

print(
    f"Tuned Blend games loaded: "
    f"{len(predictions):,}"
)


# ------------------------------------------------------------
# Every game should appear exactly once for this model.
# ------------------------------------------------------------

duplicate_games = predictions.duplicated(
    subset=[
        "game_id",
        "season"
    ],
    keep=False
)


if duplicate_games.any():

    raise RuntimeError(
        "Duplicate Tuned Blend game predictions found."
    )


# ============================================================
# LOAD GAME METADATA
#
# Needed for chronological drawdown calculations.
# ============================================================

metadata = pd.read_csv(

    METADATA_FILE,

    usecols=[
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "architecture"
    ]
)


metadata = metadata[
    (
        metadata[
            "architecture"
        ]
        ==
        "NN Small"
    )
    &
    (
        metadata[
            "season"
        ].isin(
            TEST_YEARS
        )
    )
].copy()


metadata = (

    metadata

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


games = predictions.merge(

    metadata,

    on=[
        "game_id",
        "season"
    ],

    how="left"
)


if games[
    "week"
].isna().any():

    raise RuntimeError(
        "Missing week metadata."
    )


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "predicted_home_win_probability",

    "home_moneyline",

    "away_moneyline",

    "home_decimal_odds",

    "away_decimal_odds",

    "actual_home_win"
]


for column in numeric_columns:

    games[
        column
    ] = pd.to_numeric(

        games[
            column
        ],

        errors="coerce"
    )


games = games.dropna(
    subset=numeric_columns
).copy()


# ============================================================
# MODEL PROBABILITIES
# ============================================================

games[
    "model_home_probability"
] = games[
    "predicted_home_win_probability"
]


games[
    "model_away_probability"
] = (

    1.0

    -
    games[
        "model_home_probability"
    ]
)


# ============================================================
# SPORTSBOOK BREAK-EVEN PROBABILITIES
#
# IMPORTANT:
#
# These include the sportsbook vig.
#
# This is what the bet actually has to overcome to make money.
# ============================================================

games[
    "home_break_even_probability"
] = (

    games[
        "home_moneyline"
    ]

    .apply(
        american_to_break_even_probability
    )
)


games[
    "away_break_even_probability"
] = (

    games[
        "away_moneyline"
    ]

    .apply(
        american_to_break_even_probability
    )
)


# ============================================================
# PRICE EDGE
# ============================================================

games[
    "home_price_edge"
] = (

    games[
        "model_home_probability"
    ]

    -

    games[
        "home_break_even_probability"
    ]
)


games[
    "away_price_edge"
] = (

    games[
        "model_away_probability"
    ]

    -

    games[
        "away_break_even_probability"
    ]
)


# ============================================================
# EXPECTED VALUE PER $1 RISKED
#
# EV = probability * decimal odds - 1
# ============================================================

games[
    "home_expected_roi"
] = (

    games[
        "model_home_probability"
    ]

    *

    games[
        "home_decimal_odds"
    ]

    -

    1.0
)


games[
    "away_expected_roi"
] = (

    games[
        "model_away_probability"
    ]

    *

    games[
        "away_decimal_odds"
    ]

    -

    1.0
)


# ============================================================
# CHOOSE BEST SIDE
#
# We are NOT automatically betting it.
#
# We simply identify whichever side has the larger model edge.
# ============================================================

games[
    "selected_side"
] = np.where(

    games[
        "home_price_edge"
    ]

    >=

    games[
        "away_price_edge"
    ],

    "HOME",

    "AWAY"
)


games[
    "selected_price_edge"
] = np.where(

    games[
        "selected_side"
    ]
    ==
    "HOME",

    games[
        "home_price_edge"
    ],

    games[
        "away_price_edge"
    ]
)


games[
    "selected_model_probability"
] = np.where(

    games[
        "selected_side"
    ]
    ==
    "HOME",

    games[
        "model_home_probability"
    ],

    games[
        "model_away_probability"
    ]
)


games[
    "selected_break_even_probability"
] = np.where(

    games[
        "selected_side"
    ]
    ==
    "HOME",

    games[
        "home_break_even_probability"
    ],

    games[
        "away_break_even_probability"
    ]
)


games[
    "selected_expected_roi"
] = np.where(

    games[
        "selected_side"
    ]
    ==
    "HOME",

    games[
        "home_expected_roi"
    ],

    games[
        "away_expected_roi"
    ]
)


games[
    "selected_moneyline"
] = np.where(

    games[
        "selected_side"
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
        "selected_side"
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
# FAVORITE / UNDERDOG
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
# ACTUAL RESULT
# ============================================================

games[
    "bet_won"
] = np.where(

    games[
        "selected_side"
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
).astype(
    int
)


# ============================================================
# FLAT 1-UNIT PROFIT
#
# Win:
# decimal odds - 1
#
# Loss:
# -1 unit
# ============================================================

games[
    "flat_profit"
] = np.where(

    games[
        "bet_won"
    ]
    ==
    1,

    games[
        "selected_decimal_odds"
    ]

    -
    1.0,

    -1.0
)


# ============================================================
# CHRONOLOGICAL ORDER
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
# RISK FUNCTIONS
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


    bankroll = np.concatenate(
        [
            np.array(
                [0.0]
            ),

            cumulative
        ]
    )


    running_peak = np.maximum.accumulate(
        bankroll
    )


    drawdown = (

        running_peak

        -
        bankroll
    )


    return float(
        np.max(
            drawdown
        )
    )


def longest_losing_streak(
    wins
):

    longest = 0

    current = 0


    for win in wins:

        if int(
            win
        ) == 0:

            current += 1

            longest = max(
                longest,
                current
            )

        else:

            current = 0


    return longest


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize(
    bets
):

    bets = bets.copy()


    count = len(
        bets
    )


    if count == 0:

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

            "expected_roi":
                np.nan,

            "average_price_edge":
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

            "drawdown_pct_of_100_unit_bankroll":
                np.nan,

            "longest_losing_streak":
                np.nan
        }


    wins = int(
        bets[
            "bet_won"
        ].sum()
    )


    profit = float(

        bets[
            "flat_profit"
        ].sum()
    )


    drawdown = maximum_drawdown(

        bets[
            "flat_profit"
        ]
    )


    return {

        "bets":
            count,

        "wins":
            wins,

        "losses":
            count
            -
            wins,

        "win_rate":
            wins
            /
            count,

        "profit_units":
            profit,

        "roi":
            profit
            /
            count,

        "expected_roi":
            bets[
                "selected_expected_roi"
            ].mean(),

        "average_price_edge":
            bets[
                "selected_price_edge"
            ].mean(),

        "average_model_probability":
            bets[
                "selected_model_probability"
            ].mean(),

        "average_break_even_probability":
            bets[
                "selected_break_even_probability"
            ].mean(),

        "average_decimal_odds":
            bets[
                "selected_decimal_odds"
            ].mean(),

        "favorite_pct":
            (
                bets[
                    "bet_type"
                ]
                ==
                "FAVORITE"
            ).mean(),

        "underdog_pct":
            (
                bets[
                    "bet_type"
                ]
                ==
                "UNDERDOG"
            ).mean(),

        "max_drawdown_units":
            drawdown,

        "drawdown_pct_of_100_unit_bankroll":
            drawdown
            /
            STARTING_BANKROLL_UNITS,

        "longest_losing_streak":
            longest_losing_streak(

                bets[
                    "bet_won"
                ]
            )
    }


# ============================================================
# THRESHOLD BACKTEST
# ============================================================

threshold_rows = []

season_rows = []


for threshold in EDGE_THRESHOLDS:

    qualifying = games[
        games[
            "selected_price_edge"
        ]
        >=
        threshold
    ].copy()


    summary = summarize(
        qualifying
    )


    summary[
        "minimum_price_edge"
    ] = threshold


    threshold_rows.append(
        summary
    )


    # --------------------------------------------------------
    # Same threshold by season
    # --------------------------------------------------------

    for season in TEST_YEARS:

        season_bets = qualifying[
            qualifying[
                "season"
            ]
            ==
            season
        ].copy()


        season_summary = summarize(
            season_bets
        )


        season_summary[
            "season"
        ] = season


        season_summary[
            "minimum_price_edge"
        ] = threshold


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
# ============================================================

bucket_rows = []


for (
    lower,
    upper,
    label
) in EDGE_BUCKETS:

    if np.isneginf(
        lower
    ):

        mask = (
            games[
                "selected_price_edge"
            ]
            <
            upper
        )


    elif np.isposinf(
        upper
    ):

        mask = (
            games[
                "selected_price_edge"
            ]
            >=
            lower
        )


    else:

        mask = (

            (
                games[
                    "selected_price_edge"
                ]
                >=
                lower
            )

            &

            (
                games[
                    "selected_price_edge"
                ]
                <
                upper
            )
        )


    bucket_data = games[
        mask
    ].copy()


    summary = summarize(
        bucket_data
    )


    summary[
        "edge_bucket"
    ] = label


    bucket_rows.append(
        summary
    )


bucket_results = pd.DataFrame(
    bucket_rows
)


# ============================================================
# SAVE
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
# PRINT THRESHOLD RESULTS
# ============================================================

print()

print(
    "=" * 120
)

print(
    "TUNED BLEND — FLAT-BET RESULTS BY MINIMUM PRICE EDGE"
)

print(
    "=" * 120
)


display = threshold_results[
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
]


print(

    display.to_string(

        index=False,

        formatters={

            "minimum_price_edge":
                lambda x:
                    f"{x:.0%}",

            "win_rate":
                lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "-",

            "profit_units":
                lambda x:
                    f"{x:+.2f}",

            "roi":
                lambda x:
                    f"{x:+.2%}"
                    if pd.notna(x)
                    else "-",

            "expected_roi":
                lambda x:
                    f"{x:+.2%}"
                    if pd.notna(x)
                    else "-",

            "average_price_edge":
                lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "-",

            "average_decimal_odds":
                lambda x:
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "-",

            "favorite_pct":
                lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "-",

            "max_drawdown_units":
                lambda x:
                    f"{x:.2f}"
                    if pd.notna(x)
                    else "-"
        }
    )
)


# ============================================================
# PRINT BUCKET RESULTS
# ============================================================

print()

print(
    "=" * 120
)

print(
    "TUNED BLEND — DISJOINT PRICE-EDGE BUCKETS"
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
]


print(

    bucket_display.to_string(

        index=False,

        formatters={

            "win_rate":
                lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "-",

            "profit_units":
                lambda x:
                    f"{x:+.2f}",

            "roi":
                lambda x:
                    f"{x:+.2%}"
                    if pd.notna(x)
                    else "-",

            "expected_roi":
                lambda x:
                    f"{x:+.2%}"
                    if pd.notna(x)
                    else "-",

            "average_price_edge":
                lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "-",

            "average_decimal_odds":
                lambda x:
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "-",

            "favorite_pct":
                lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "-",

            "max_drawdown_units":
                lambda x:
                    f"{x:.2f}"
                    if pd.notna(x)
                    else "-"
        }
    )
)


# ============================================================
# PRINT BY-SEASON RESULTS
# ============================================================

print()

print(
    "=" * 120
)

print(
    "TUNED BLEND — BY-SEASON RESULTS"
)

print(
    "=" * 120
)


season_display = season_results[
    season_results[
        "minimum_price_edge"
    ].isin(
        [
            0.00,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05
        ]
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
]


print(

    season_display.to_string(

        index=False,

        formatters={

            "minimum_price_edge":
                lambda x:
                    f"{x:.0%}",

            "win_rate":
                lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "-",

            "profit_units":
                lambda x:
                    f"{x:+.2f}",

            "roi":
                lambda x:
                    f"{x:+.2%}"
                    if pd.notna(x)
                    else "-",

            "expected_roi":
                lambda x:
                    f"{x:+.2%}"
                    if pd.notna(x)
                    else "-",

            "max_drawdown_units":
                lambda x:
                    f"{x:.2f}"
                    if pd.notna(x)
                    else "-"
        }
    )
)


# ============================================================
# FAVORITE VS UNDERDOG
# ============================================================

print()

print(
    "=" * 120
)

print(
    "ALL POSITIVE-EDGE TUNED-BLEND BETS — FAVORITE VS UNDERDOG"
)

print(
    "=" * 120
)


positive_edge = games[
    games[
        "selected_price_edge"
    ]
    >=
    0.0
]


side_rows = []


for bet_type in [
    "FAVORITE",
    "UNDERDOG"
]:

    subset = positive_edge[
        positive_edge[
            "bet_type"
        ]
        ==
        bet_type
    ]


    result = summarize(
        subset
    )


    result[
        "bet_type"
    ] = bet_type


    side_rows.append(
        result
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
                lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "-",

            "profit_units":
                lambda x:
                    f"{x:+.2f}",

            "roi":
                lambda x:
                    f"{x:+.2%}"
                    if pd.notna(x)
                    else "-",

            "expected_roi":
                lambda x:
                    f"{x:+.2%}"
                    if pd.notna(x)
                    else "-",

            "average_price_edge":
                lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "-",

            "average_decimal_odds":
                lambda x:
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "-",

            "max_drawdown_units":
                lambda x:
                    f"{x:.2f}"
                    if pd.notna(x)
                    else "-"
        }
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print(
    "=" * 110
)

print(
    "RESULT FILES"
)

print(
    "=" * 110
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
    "These are development backtests."
)


print(
    "Do NOT choose the historically best threshold "
    "and call it proven."
)


print(
    "2025 was not loaded, evaluated, or used."
)