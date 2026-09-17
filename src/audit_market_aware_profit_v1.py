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

OUTPUT_FILE = (
    "data/processed/"
    "market_aware_profit_audit_v1.csv"
)

MODEL_NAME = "Tuned Blend"

TEST_YEARS = [
    2022,
    2023,
    2024
]


# ============================================================
# FUNCTIONS
# ============================================================

def american_to_break_even_probability(odds):

    if pd.isna(odds):
        return np.nan

    odds = float(odds)

    if odds < 0:

        return (
            abs(odds)
            /
            (abs(odds) + 100.0)
        )

    if odds > 0:

        return (
            100.0
            /
            (odds + 100.0)
        )

    return np.nan


def odds_bucket(odds):

    if pd.isna(odds):
        return "UNKNOWN"

    odds = float(odds)

    if odds <= -200:
        return "-200 or shorter"

    if -200 < odds < 0:
        return "-199 to -101"

    if 100 <= odds < 200:
        return "+100 to +199"

    if 200 <= odds < 300:
        return "+200 to +299"

    if 300 <= odds < 500:
        return "+300 to +499"

    if 500 <= odds < 800:
        return "+500 to +799"

    if odds >= 800:
        return "+800 or longer"

    return "OTHER"


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print(
    "MARKET-AWARE PROFIT CONCENTRATION AUDIT — V1"
)

print("=" * 110)


predictions = pd.read_csv(
    PREDICTIONS_FILE
)


predictions = predictions[
    (
        predictions["model"] == MODEL_NAME
    )
    &
    (
        predictions["season"].isin(TEST_YEARS)
    )
].copy()


if len(predictions) == 0:

    raise RuntimeError(
        "No Tuned Blend predictions found."
    )


if predictions["season"].max() >= 2025:

    raise RuntimeError(
        "2025 leakage detected."
    )


# ============================================================
# LOAD TEAM / WEEK METADATA
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
        metadata["architecture"] == "NN Small"
    )
    &
    (
        metadata["season"].isin(TEST_YEARS)
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


# ============================================================
# CLEAN NUMBERS
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

    games[column] = pd.to_numeric(
        games[column],
        errors="coerce"
    )


games = games.dropna(
    subset=numeric_columns
).copy()


# ============================================================
# HOME / AWAY MODEL PROBABILITY
# ============================================================

games["model_home_probability"] = (
    games[
        "predicted_home_win_probability"
    ]
)


games["model_away_probability"] = (
    1.0
    -
    games[
        "model_home_probability"
    ]
)


# ============================================================
# BREAK-EVEN PROBABILITY
# ============================================================

games[
    "home_break_even_probability"
] = games[
    "home_moneyline"
].apply(
    american_to_break_even_probability
)


games[
    "away_break_even_probability"
] = games[
    "away_moneyline"
].apply(
    american_to_break_even_probability
)


# ============================================================
# PRICE EDGE
# ============================================================

games["home_price_edge"] = (
    games["model_home_probability"]
    -
    games["home_break_even_probability"]
)


games["away_price_edge"] = (
    games["model_away_probability"]
    -
    games["away_break_even_probability"]
)


# ============================================================
# CHOOSE SIDE WITH LARGER EDGE
# ============================================================

games["selected_side"] = np.where(

    games["home_price_edge"]
    >=
    games["away_price_edge"],

    "HOME",

    "AWAY"
)


games["selected_price_edge"] = np.where(

    games["selected_side"] == "HOME",

    games["home_price_edge"],

    games["away_price_edge"]
)


games["selected_moneyline"] = np.where(

    games["selected_side"] == "HOME",

    games["home_moneyline"],

    games["away_moneyline"]
)


games["selected_decimal_odds"] = np.where(

    games["selected_side"] == "HOME",

    games["home_decimal_odds"],

    games["away_decimal_odds"]
)


games["selected_team"] = np.where(

    games["selected_side"] == "HOME",

    games["home_team"],

    games["away_team"]
)


games["opponent"] = np.where(

    games["selected_side"] == "HOME",

    games["away_team"],

    games["home_team"]
)


# ============================================================
# ACTUAL BET RESULT
# ============================================================

games["bet_won"] = np.where(

    games["selected_side"] == "HOME",

    games["actual_home_win"] == 1,

    games["actual_home_win"] == 0
).astype(int)


games["flat_profit"] = np.where(

    games["bet_won"] == 1,

    games["selected_decimal_odds"] - 1.0,

    -1.0
)


# ============================================================
# ONLY POSITIVE-EDGE BETS
#
# This reproduces the broadest rule from the prior test:
#
# model price edge >= 0%
# ============================================================

bets = games[
    games["selected_price_edge"] >= 0.0
].copy()


bets["odds_bucket"] = bets[
    "selected_moneyline"
].apply(
    odds_bucket
)


bets = bets.sort_values(
    [
        "season",
        "week",
        "game_id"
    ]
).reset_index(
    drop=True
)


print()

print(
    f"Positive-edge bets: "
    f"{len(bets):,}"
)


total_profit = float(
    bets["flat_profit"].sum()
)


total_roi = (
    total_profit
    /
    len(bets)
)


print(
    f"Total profit:       "
    f"{total_profit:+.2f} units"
)


print(
    f"Total ROI:          "
    f"{total_roi:+.2%}"
)


# ============================================================
# ODDS-BUCKET RESULTS
# ============================================================

bucket_order = [
    "-200 or shorter",
    "-199 to -101",
    "+100 to +199",
    "+200 to +299",
    "+300 to +499",
    "+500 to +799",
    "+800 or longer",
    "OTHER"
]


bucket_rows = []


for bucket in bucket_order:

    subset = bets[
        bets["odds_bucket"] == bucket
    ]


    if len(subset) == 0:
        continue


    profit = float(
        subset["flat_profit"].sum()
    )


    wins = int(
        subset["bet_won"].sum()
    )


    bucket_rows.append(
        {
            "odds_bucket":
                bucket,

            "bets":
                len(subset),

            "wins":
                wins,

            "win_rate":
                wins / len(subset),

            "profit_units":
                profit,

            "roi":
                profit / len(subset),

            "average_moneyline":
                subset[
                    "selected_moneyline"
                ].mean(),

            "average_edge":
                subset[
                    "selected_price_edge"
                ].mean(),

            "share_of_total_profit":
                (
                    profit / total_profit
                    if total_profit != 0
                    else np.nan
                )
        }
    )


bucket_results = pd.DataFrame(
    bucket_rows
)


print()

print("=" * 110)

print(
    "PROFIT BY ODDS RANGE"
)

print("=" * 110)


print(

    bucket_results.to_string(

        index=False,

        formatters={

            "win_rate":
                lambda x:
                    f"{x:.2%}",

            "profit_units":
                lambda x:
                    f"{x:+.2f}",

            "roi":
                lambda x:
                    f"{x:+.2%}",

            "average_moneyline":
                lambda x:
                    f"{x:+.0f}",

            "average_edge":
                lambda x:
                    f"{x:.2%}",

            "share_of_total_profit":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# MOST PROFITABLE INDIVIDUAL BETS
# ============================================================

top_winners = (

    bets[
        bets["flat_profit"] > 0
    ]

    .sort_values(
        "flat_profit",
        ascending=False
    )

    .head(15)
)


print()

print("=" * 110)

print(
    "TOP 15 MOST PROFITABLE INDIVIDUAL BETS"
)

print("=" * 110)


print(

    top_winners[
        [
            "season",
            "week",
            "selected_team",
            "opponent",
            "selected_moneyline",
            "selected_price_edge",
            "flat_profit"
        ]
    ].to_string(

        index=False,

        formatters={

            "selected_moneyline":
                lambda x:
                    f"{x:+.0f}",

            "selected_price_edge":
                lambda x:
                    f"{x:.2%}",

            "flat_profit":
                lambda x:
                    f"{x:+.2f}"
        }
    )
)


# ============================================================
# PROFIT CONCENTRATION
#
# Remove the best winning bets and see whether the rest
# of the strategy still makes money.
# ============================================================

winning_profits = (

    bets[
        bets["flat_profit"] > 0
    ]

    .sort_values(
        "flat_profit",
        ascending=False
    )
)


concentration_rows = []


for remove_count in [
    0,
    1,
    3,
    5,
    10,
    15
]:

    if remove_count == 0:

        remaining = bets.copy()

    else:

        ids_to_remove = (
            winning_profits
            .head(remove_count)
            .index
        )

        remaining = bets.drop(
            index=ids_to_remove
        )


    remaining_profit = float(
        remaining[
            "flat_profit"
        ].sum()
    )


    concentration_rows.append(
        {
            "top_wins_removed":
                remove_count,

            "remaining_bets":
                len(remaining),

            "remaining_profit":
                remaining_profit,

            "remaining_roi":
                (
                    remaining_profit
                    /
                    len(remaining)
                )
        }
    )


concentration = pd.DataFrame(
    concentration_rows
)


print()

print("=" * 110)

print(
    "WHAT HAPPENS IF WE REMOVE THE BIGGEST WINNERS?"
)

print("=" * 110)


print(

    concentration.to_string(

        index=False,

        formatters={

            "remaining_profit":
                lambda x:
                    f"{x:+.2f}",

            "remaining_roi":
                lambda x:
                    f"{x:+.2%}"
        }
    )
)


# ============================================================
# PROFIT BY SEASON AND ODDS RANGE
# ============================================================

season_bucket = (

    bets

    .groupby(
        [
            "season",
            "odds_bucket"
        ],
        as_index=False
    )

    .agg(

        bets=(
            "game_id",
            "size"
        ),

        wins=(
            "bet_won",
            "sum"
        ),

        profit_units=(
            "flat_profit",
            "sum"
        )
    )
)


season_bucket["win_rate"] = (
    season_bucket["wins"]
    /
    season_bucket["bets"]
)


season_bucket["roi"] = (
    season_bucket["profit_units"]
    /
    season_bucket["bets"]
)


print()

print("=" * 110)

print(
    "PROFIT BY SEASON AND ODDS RANGE"
)

print("=" * 110)


print(

    season_bucket.to_string(

        index=False,

        formatters={

            "win_rate":
                lambda x:
                    f"{x:.2%}",

            "profit_units":
                lambda x:
                    f"{x:+.2f}",

            "roi":
                lambda x:
                    f"{x:+.2%}"
        }
    )
)


# ============================================================
# SAVE GAME-LEVEL AUDIT
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


bets.to_csv(
    OUTPUT_FILE,
    index=False
)


print()

print("=" * 110)

print(
    "OUTPUT FILE"
)

print("=" * 110)


print(
    OUTPUT_FILE
)


print()

print(
    "2025 was not loaded or evaluated."
)