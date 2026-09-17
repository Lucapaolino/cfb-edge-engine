import os
import sqlite3

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)


# ============================================================
# SETTINGS
# ============================================================

DATABASE_FILE = (
    "data/cfb.db"
)

PREDICTIONS_FILE = (
    "data/processed/"
    "nn_seed_stability_predictions_v1.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "nn_vs_bovada_market_v1.csv"
)

SEASON_RESULTS_FILE = (
    "data/processed/"
    "nn_vs_bovada_market_by_season_v1.csv"
)


DEVELOPMENT_YEARS = [
    2021,
    2022,
    2023,
    2024
]


PROVIDER = (
    "Bovada"
)


ARCHITECTURE = (
    "NN Small"
)


# ============================================================
# START
# ============================================================

print(
    "NN VS SPORTSBOOK MONEYLINE MARKET — V1"
)

print(
    "=" * 100
)


print(
    f"Market provider: "
    f"{PROVIDER}"
)


print(
    f"Model:           "
    f"{ARCHITECTURE} five-seed ensemble"
)


print(
    f"Development:     "
    f"{DEVELOPMENT_YEARS}"
)


print(
    "\n2025 will NOT be loaded into this comparison."
)


# ============================================================
# MONEYLINE FUNCTIONS
# ============================================================

def american_to_implied_probability(
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


def american_to_decimal_odds(
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

            1.0

            +

            (
                100.0

                /

                abs(
                    odds
                )
            )
        )


    if odds > 0:

        return (

            1.0

            +

            (
                odds

                /

                100.0
            )
        )


    return np.nan


# ============================================================
# LOAD NN PREDICTIONS
# ============================================================

predictions = pd.read_csv(
    PREDICTIONS_FILE
)


print()

print(
    f"Prediction rows loaded: "
    f"{len(predictions):,}"
)


# ------------------------------------------------------------
# Only the Small architecture advances.
# ------------------------------------------------------------

predictions = predictions[
    predictions[
        "architecture"
    ]
    ==
    ARCHITECTURE
].copy()


# ------------------------------------------------------------
# HARD 2025 SAFETY WALL
# ------------------------------------------------------------

predictions = predictions[
    predictions[
        "season"
    ].isin(
        DEVELOPMENT_YEARS
    )
].copy()


if (
    predictions[
        "season"
    ].max()
    >= 2025
):

    raise RuntimeError(
        "2025 leakage detected."
    )


# ============================================================
# BUILD FIVE-SEED ENSEMBLE
#
# Average the five independently trained networks for
# each individual game.
# ============================================================

ensemble = (

    predictions

    .groupby(
        [
            "game_id",
            "season"
        ],
        as_index=False
    )

    .agg(

        actual_home_win=(
            "actual_home_win",
            "first"
        ),

        actual_margin=(
            "actual_margin",
            "first"
        ),

        model_home_win_probability=(
            "predicted_home_win_probability",
            "mean"
        ),

        model_predicted_margin=(
            "predicted_margin",
            "mean"
        ),

        seed_count=(
            "seed",
            "nunique"
        )
    )
)


# ------------------------------------------------------------
# Safety check:
#
# Every ensemble game should have five model predictions.
# ------------------------------------------------------------

bad_seed_counts = ensemble[
    ensemble[
        "seed_count"
    ]
    != 5
]


if len(
    bad_seed_counts
) > 0:

    raise RuntimeError(

        "Some games do not contain exactly "
        "five NN seed predictions."
    )


ensemble[
    "game_id"
] = pd.to_numeric(

    ensemble[
        "game_id"
    ],

    errors="coerce"
).astype(
    "Int64"
)


print(
    f"Ensemble games: "
    f"{len(ensemble):,}"
)


# ============================================================
# LOAD BOVADA MONEYLINES
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


moneylines = pd.read_sql_query(

    """
    SELECT
        game_id,
        provider,
        spread,
        home_moneyline,
        away_moneyline
    FROM betting_lines
    WHERE provider = ?
    """,

    connection,

    params=[
        PROVIDER
    ]
)


connection.close()


print(
    f"Bovada line rows loaded: "
    f"{len(moneylines):,}"
)


# ============================================================
# CLEAN MONEYLINES
# ============================================================

moneylines[
    "game_id"
] = pd.to_numeric(

    moneylines[
        "game_id"
    ],

    errors="coerce"
).astype(
    "Int64"
)


for column in [

    "spread",

    "home_moneyline",

    "away_moneyline"
]:

    moneylines[
        column
    ] = pd.to_numeric(

        moneylines[
            column
        ],

        errors="coerce"
    )


# ------------------------------------------------------------
# Zero is not a valid American moneyline.
# ------------------------------------------------------------

moneylines.loc[
    moneylines[
        "home_moneyline"
    ]
    == 0,
    "home_moneyline"
] = np.nan


moneylines.loc[
    moneylines[
        "away_moneyline"
    ]
    == 0,
    "away_moneyline"
] = np.nan


# ------------------------------------------------------------
# Require both sides so we can remove vig.
# ------------------------------------------------------------

moneylines = moneylines[
    moneylines[
        "home_moneyline"
    ].notna()

    &

    moneylines[
        "away_moneyline"
    ].notna()
].copy()


# ------------------------------------------------------------
# According to our audit, Bovada currently has one row
# per game. Stop if that changes rather than silently
# choosing a duplicate.
# ------------------------------------------------------------

duplicate_games = (

    moneylines[
        "game_id"
    ]

    .duplicated(
        keep=False
    )
)


if duplicate_games.any():

    duplicate_count = (

        moneylines.loc[
            duplicate_games,
            "game_id"
        ]
        .nunique()
    )


    raise RuntimeError(

        f"Bovada contains duplicate rows for "
        f"{duplicate_count} games. "
        f"Do not silently choose one."
    )


# ============================================================
# JOIN MODEL AND MARKET
# ============================================================

games = ensemble.merge(

    moneylines,

    on="game_id",

    how="inner"
)


games = games[
    games[
        "season"
    ].isin(
        DEVELOPMENT_YEARS
    )
].copy()


if (
    games[
        "season"
    ].max()
    >= 2025
):

    raise RuntimeError(
        "2025 leakage detected after join."
    )


print()

print(
    f"Games with both NN predictions "
    f"and Bovada paired moneylines: "
    f"{len(games):,}"
)


# ============================================================
# RAW IMPLIED PROBABILITIES
#
# These are the probabilities implied by the ACTUAL PRICE.
#
# These include sportsbook vig.
# ============================================================

games[
    "market_home_raw_probability"
] = (

    games[
        "home_moneyline"
    ]

    .apply(
        american_to_implied_probability
    )
)


games[
    "market_away_raw_probability"
] = (

    games[
        "away_moneyline"
    ]

    .apply(
        american_to_implied_probability
    )
)


# ============================================================
# SPORTSBOOK OVERROUND / VIG
#
# Example:
#
# Home raw = 60%
# Away raw = 43%
#
# Total = 103%
#
# Overround = 3%
# ============================================================

games[
    "market_probability_sum"
] = (

    games[
        "market_home_raw_probability"
    ]

    +

    games[
        "market_away_raw_probability"
    ]
)


games[
    "market_overround"
] = (

    games[
        "market_probability_sum"
    ]

    -

    1.0
)


# ============================================================
# REMOVE VIG
#
# Simple proportional normalization:
#
# home fair =
# home raw / (home raw + away raw)
#
# away fair =
# away raw / (home raw + away raw)
# ============================================================

games[
    "market_home_fair_probability"
] = (

    games[
        "market_home_raw_probability"
    ]

    /

    games[
        "market_probability_sum"
    ]
)


games[
    "market_away_fair_probability"
] = (

    games[
        "market_away_raw_probability"
    ]

    /

    games[
        "market_probability_sum"
    ]
)


# ============================================================
# DECIMAL ODDS
#
# Needed to calculate expected profit.
# ============================================================

games[
    "home_decimal_odds"
] = (

    games[
        "home_moneyline"
    ]

    .apply(
        american_to_decimal_odds
    )
)


games[
    "away_decimal_odds"
] = (

    games[
        "away_moneyline"
    ]

    .apply(
        american_to_decimal_odds
    )
)


# ============================================================
# MARKET FAVORITE
# ============================================================

games[
    "market_home_pick"
] = (

    games[
        "market_home_fair_probability"
    ]

    >=
    0.50
).astype(
    int
)


# ============================================================
# MODEL WIN PICK
# ============================================================

games[
    "model_home_pick"
] = (

    games[
        "model_home_win_probability"
    ]

    >=
    0.50
).astype(
    int
)


# ============================================================
# MODEL VS FAIR MARKET DISAGREEMENT
#
# This tells us how much our football model disagrees
# with the sportsbook after removing vig.
# ============================================================

games[
    "model_minus_market_home_probability"
] = (

    games[
        "model_home_win_probability"
    ]

    -

    games[
        "market_home_fair_probability"
    ]
)


games[
    "absolute_model_market_disagreement"
] = (

    games[
        "model_minus_market_home_probability"
    ]

    .abs()
)


# ============================================================
# EXPECTED VALUE USING OUR MODEL
#
# EV per $1 risked:
#
# probability * decimal odds - 1
#
# Positive value means our probability says the offered
# price is theoretically profitable.
# ============================================================

games[
    "model_home_ev_per_dollar"
] = (

    games[
        "model_home_win_probability"
    ]

    *

    games[
        "home_decimal_odds"
    ]

    -

    1.0
)


games[
    "model_away_win_probability"
] = (

    1.0

    -

    games[
        "model_home_win_probability"
    ]
)


games[
    "model_away_ev_per_dollar"
] = (

    games[
        "model_away_win_probability"
    ]

    *

    games[
        "away_decimal_odds"
    ]

    -

    1.0
)


# ============================================================
# BEST SIDE ACCORDING TO MODEL
#
# This does NOT yet mean we should bet it.
#
# It simply records whichever side our probability says
# has the better expected value at the stored price.
# ============================================================

games[
    "model_best_bet_side"
] = np.where(

    games[
        "model_home_ev_per_dollar"
    ]

    >=

    games[
        "model_away_ev_per_dollar"
    ],

    "HOME",

    "AWAY"
)


games[
    "model_best_ev_per_dollar"
] = np.where(

    games[
        "model_best_bet_side"
    ]
    ==
    "HOME",

    games[
        "model_home_ev_per_dollar"
    ],

    games[
        "model_away_ev_per_dollar"
    ]
)


games[
    "model_best_side_probability"
] = np.where(

    games[
        "model_best_bet_side"
    ]
    ==
    "HOME",

    games[
        "model_home_win_probability"
    ],

    games[
        "model_away_win_probability"
    ]
)


games[
    "best_side_raw_break_even_probability"
] = np.where(

    games[
        "model_best_bet_side"
    ]
    ==
    "HOME",

    games[
        "market_home_raw_probability"
    ],

    games[
        "market_away_raw_probability"
    ]
)


games[
    "best_side_market_fair_probability"
] = np.where(

    games[
        "model_best_bet_side"
    ]
    ==
    "HOME",

    games[
        "market_home_fair_probability"
    ],

    games[
        "market_away_fair_probability"
    ]
)


# ------------------------------------------------------------
# PRICE EDGE:
#
# Our probability minus ACTUAL break-even probability.
#
# This is the more directly relevant betting edge.
# ------------------------------------------------------------

games[
    "model_price_edge"
] = (

    games[
        "model_best_side_probability"
    ]

    -

    games[
        "best_side_raw_break_even_probability"
    ]
)


# ------------------------------------------------------------
# FAIR-MARKET EDGE:
#
# Our probability minus the de-vigged market estimate.
#
# Useful for measuring disagreement with the market.
# ------------------------------------------------------------

games[
    "model_fair_market_edge"
] = (

    games[
        "model_best_side_probability"
    ]

    -

    games[
        "best_side_market_fair_probability"
    ]
)


# ============================================================
# REALIZED FLAT-BET PROFIT
#
# This is saved for the NEXT strategy experiment.
#
# We are NOT choosing a betting threshold yet.
#
# Every hypothetical wager risks exactly 1 unit.
# ============================================================

home_win = (

    games[
        "actual_home_win"
    ]

    ==
    1
)


away_win = (

    games[
        "actual_home_win"
    ]

    ==
    0
)


games[
    "best_side_flat_profit"
] = np.where(

    games[
        "model_best_bet_side"
    ]
    ==
    "HOME",

    np.where(

        home_win,

        games[
            "home_decimal_odds"
        ]

        -
        1.0,

        -1.0
    ),

    np.where(

        away_win,

        games[
            "away_decimal_odds"
        ]

        -
        1.0,

        -1.0
    )
)


# ============================================================
# INTERNAL MARKET CONSISTENCY AUDIT
#
# Spread:
# negative = home favorite
# positive = away favorite
#
# Compare spread favorite with moneyline favorite.
# ============================================================

spread_available = (

    games[
        "spread"
    ].notna()

    &

    (
        games[
            "spread"
        ]
        != 0
    )
)


games[
    "spread_home_favorite"
] = np.nan


games.loc[
    spread_available,
    "spread_home_favorite"
] = (

    games.loc[
        spread_available,
        "spread"
    ]

    < 0
).astype(
    int
)


games[
    "spread_moneyline_favorite_match"
] = np.nan


games.loc[
    spread_available,
    "spread_moneyline_favorite_match"
] = (

    games.loc[
        spread_available,
        "spread_home_favorite"
    ]

    ==

    games.loc[
        spread_available,
        "market_home_pick"
    ]

).astype(
    int
)


favorite_match_rate = (

    games.loc[
        spread_available,
        "spread_moneyline_favorite_match"
    ]

    .mean()
)


print()

print(
    "BOVADA INTERNAL CONSISTENCY"
)

print(
    "-" * 100
)


print(

    f"Spread / moneyline favorite agreement: "
    f"{favorite_match_rate:.2%}"
)


print(

    f"Games checked: "
    f"{int(spread_available.sum()):,}"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def probability_metrics(

    actual,
    probability
):

    probability = np.clip(

        np.asarray(
            probability,
            dtype=float
        ),

        1e-6,

        1.0 - 1e-6
    )


    actual = np.asarray(
        actual,
        dtype=int
    )


    predicted = (

        probability

        >=
        0.50

    ).astype(
        int
    )


    accuracy = accuracy_score(

        actual,

        predicted
    )


    brier = brier_score_loss(

        actual,

        probability
    )


    loss = log_loss(

        actual,

        probability,

        labels=[
            0,
            1
        ]
    )


    return (
        accuracy,
        brier,
        loss
    )


# ============================================================
# COMPARE MODEL VS MARKET BY SEASON
# ============================================================

season_rows = []


for season in DEVELOPMENT_YEARS:

    season_data = games[
        games[
            "season"
        ]
        ==
        season
    ].copy()


    if len(
        season_data
    ) == 0:

        continue


    actual = (

        season_data[
            "actual_home_win"
        ]
        .astype(
            int
        )
    )


    (
        model_accuracy,
        model_brier,
        model_logloss
    ) = probability_metrics(

        actual,

        season_data[
            "model_home_win_probability"
        ]
    )


    (
        market_accuracy,
        market_brier,
        market_logloss
    ) = probability_metrics(

        actual,

        season_data[
            "market_home_fair_probability"
        ]
    )


    season_rows.append(
        {
            "season":
                season,

            "games":
                len(
                    season_data
                ),

            "model_accuracy":
                model_accuracy,

            "market_accuracy":
                market_accuracy,

            "model_brier":
                model_brier,

            "market_brier":
                market_brier,

            "model_logloss":
                model_logloss,

            "market_logloss":
                market_logloss,

            "accuracy_difference":
                (
                    model_accuracy
                    -
                    market_accuracy
                ),

            "brier_difference":
                (
                    model_brier
                    -
                    market_brier
                ),

            "logloss_difference":
                (
                    model_logloss
                    -
                    market_logloss
                ),

            "average_market_overround":
                season_data[
                    "market_overround"
                ].mean(),

            "average_abs_disagreement":
                season_data[
                    "absolute_model_market_disagreement"
                ].mean(),

            "model_positive_ev_games":
                int(

                    (
                        season_data[
                            "model_best_ev_per_dollar"
                        ]
                        >
                        0
                    )

                    .sum()
                ),

            "model_positive_ev_pct":
                (

                    (
                        season_data[
                            "model_best_ev_per_dollar"
                        ]
                        >
                        0
                    )

                    .mean()
                )
        }
    )


season_results = pd.DataFrame(
    season_rows
)


# ============================================================
# PRINT BY-SEASON RESULTS
# ============================================================

print()

print(
    "=" * 100
)

print(
    "NN VS DE-VIGGED BOVADA — BY SEASON"
)

print(
    "=" * 100
)


print(

    season_results.to_string(

        index=False,

        formatters={

            "model_accuracy":
                lambda value:
                    f"{value:.3f}",

            "market_accuracy":
                lambda value:
                    f"{value:.3f}",

            "model_brier":
                lambda value:
                    f"{value:.3f}",

            "market_brier":
                lambda value:
                    f"{value:.3f}",

            "model_logloss":
                lambda value:
                    f"{value:.3f}",

            "market_logloss":
                lambda value:
                    f"{value:.3f}",

            "accuracy_difference":
                lambda value:
                    f"{value:+.4f}",

            "brier_difference":
                lambda value:
                    f"{value:+.4f}",

            "logloss_difference":
                lambda value:
                    f"{value:+.4f}",

            "average_market_overround":
                lambda value:
                    f"{value:.2%}",

            "average_abs_disagreement":
                lambda value:
                    f"{value:.2%}",

            "model_positive_ev_pct":
                lambda value:
                    f"{value:.2%}"
        }
    )
)


# ============================================================
# EQUAL-WEIGHT SEASON AVERAGES
#
# Matches the methodology we've been using for rolling
# validation.
# ============================================================

average_results = {

    "games":
        int(
            season_results[
                "games"
            ].sum()
        ),

    "model_accuracy":
        season_results[
            "model_accuracy"
        ].mean(),

    "market_accuracy":
        season_results[
            "market_accuracy"
        ].mean(),

    "model_brier":
        season_results[
            "model_brier"
        ].mean(),

    "market_brier":
        season_results[
            "market_brier"
        ].mean(),

    "model_logloss":
        season_results[
            "model_logloss"
        ].mean(),

    "market_logloss":
        season_results[
            "market_logloss"
        ].mean(),

    "average_market_overround":
        season_results[
            "average_market_overround"
        ].mean(),

    "average_abs_disagreement":
        season_results[
            "average_abs_disagreement"
        ].mean(),

    "positive_ev_pct":
        season_results[
            "model_positive_ev_pct"
        ].mean()
}


print()

print(
    "=" * 100
)

print(
    "AVERAGE PROBABILITY BENCHMARK"
)

print(
    "=" * 100
)


print(

    f"Games compared:             "
    f"{average_results['games']:,}"
)


print()

print(

    f"NN win accuracy:            "
    f"{average_results['model_accuracy']:.3f}"
)


print(

    f"Bovada win accuracy:        "
    f"{average_results['market_accuracy']:.3f}"
)


print(

    f"Accuracy difference:        "
    f"{average_results['model_accuracy'] - average_results['market_accuracy']:+.4f}"
)


print()

print(

    f"NN Brier:                   "
    f"{average_results['model_brier']:.3f}"
)


print(

    f"Bovada Brier:               "
    f"{average_results['market_brier']:.3f}"
)


print(

    f"Brier difference:           "
    f"{average_results['model_brier'] - average_results['market_brier']:+.4f}"
)


print()

print(

    f"NN LogLoss:                 "
    f"{average_results['model_logloss']:.3f}"
)


print(

    f"Bovada LogLoss:             "
    f"{average_results['market_logloss']:.3f}"
)


print(

    f"LogLoss difference:         "
    f"{average_results['model_logloss'] - average_results['market_logloss']:+.4f}"
)


print()

print(

    f"Average Bovada overround:   "
    f"{average_results['average_market_overround']:.2%}"
)


print(

    f"Average NN/market disagreement: "
    f"{average_results['average_abs_disagreement']:.2%}"
)


print(

    f"Games NN labels +EV:        "
    f"{average_results['positive_ev_pct']:.2%}"
)


print()

print(
    "For Brier and LogLoss:"
)

print(
    "negative model-market difference = NN better"
)

print(
    "positive model-market difference = market better"
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(

    "data/processed",

    exist_ok=True
)


games.to_csv(

    OUTPUT_FILE,

    index=False
)


season_results.to_csv(

    SEASON_RESULTS_FILE,

    index=False
)


print()

print(
    "=" * 100
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 100
)


print(
    OUTPUT_FILE
)


print(
    SEASON_RESULTS_FILE
)


print()

print(
    "2025 was not loaded, scored, or used."
)