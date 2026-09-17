import os

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "market_aware_profit_audit_v1.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "market_aware_bootstrap_results_v1.csv"
)

BLOCK_OUTPUT_FILE = (
    "data/processed/"
    "market_aware_week_block_bootstrap_results_v1.csv"
)


TEST_YEARS = [
    2022,
    2023,
    2024
]


BOOTSTRAP_SAMPLES = 20_000

RANDOM_SEED = 20260905


# ============================================================
# START
# ============================================================

print(
    "MARKET-AWARE STRATEGY BOOTSTRAP — V1"
)

print(
    "=" * 110
)


print(
    f"Bootstrap samples: "
    f"{BOOTSTRAP_SAMPLES:,}"
)


print(
    "Strategy being tested:"
)


print(
    "Tuned Blend with model price edge >= 0%"
)


print(
    "Each wager risks exactly 1 unit."
)


print(
    "2025 will NOT be loaded or evaluated."
)


# ============================================================
# LOAD BETS
# ============================================================

bets = pd.read_csv(
    INPUT_FILE
)


bets = bets[
    bets[
        "season"
    ].isin(
        TEST_YEARS
    )
].copy()


if len(
    bets
) == 0:

    raise RuntimeError(
        "No bets loaded."
    )


if (
    bets[
        "season"
    ].max()
    >= 2025
):

    raise RuntimeError(
        "2025 leakage detected."
    )


required_columns = [

    "season",

    "week",

    "game_id",

    "flat_profit",

    "bet_won",

    "selected_moneyline",

    "selected_price_edge"
]


missing = [

    column

    for column in required_columns

    if column not in bets.columns
]


if missing:

    raise RuntimeError(

        "Missing required columns: "

        +

        ", ".join(
            missing
        )
    )


for column in [

    "season",

    "week",

    "flat_profit",

    "bet_won",

    "selected_moneyline",

    "selected_price_edge"
]:

    bets[
        column
    ] = pd.to_numeric(

        bets[
            column
        ],

        errors="coerce"
    )


bets = bets.dropna(
    subset=[
        "season",
        "week",
        "flat_profit",
        "bet_won"
    ]
).copy()


bets = (

    bets

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


print()

print(
    f"Bets loaded: "
    f"{len(bets):,}"
)


# ============================================================
# OBSERVED PERFORMANCE
# ============================================================

observed_profit = float(

    bets[
        "flat_profit"
    ].sum()
)


observed_roi = (

    observed_profit

    /
    len(
        bets
    )
)


observed_wins = int(

    bets[
        "bet_won"
    ].sum()
)


observed_win_rate = (

    observed_wins

    /
    len(
        bets
    )
)


print()

print(
    "OBSERVED STRATEGY"
)

print(
    "-" * 110
)


print(
    f"Bets:       {len(bets):,}"
)


print(
    f"Wins:       {observed_wins:,}"
)


print(
    f"Win rate:   {observed_win_rate:.2%}"
)


print(
    f"Profit:     {observed_profit:+.2f} units"
)


print(
    f"ROI:        {observed_roi:+.2%}"
)


# ============================================================
# DRAWDOWN
# ============================================================

def maximum_drawdown(
    profit_array
):

    profits = np.asarray(
        profit_array,
        dtype=float
    )


    if len(
        profits
    ) == 0:

        return np.nan


    cumulative = np.cumsum(
        profits
    )


    curve = np.concatenate(
        [
            np.array(
                [0.0]
            ),
            cumulative
        ]
    )


    running_peak = np.maximum.accumulate(
        curve
    )


    drawdowns = (

        running_peak

        -
        curve
    )


    return float(
        np.max(
            drawdowns
        )
    )


observed_drawdown = maximum_drawdown(

    bets[
        "flat_profit"
    ].to_numpy()
)


print(
    f"Max drawdown: {observed_drawdown:.2f} units"
)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


profits = bets[
    "flat_profit"
].to_numpy(
    dtype=float
)


wins = bets[
    "bet_won"
].to_numpy(
    dtype=int
)


n_bets = len(
    bets
)


# ============================================================
# BET-LEVEL BOOTSTRAP
#
# Sample 226 bets WITH replacement.
#
# Example:
#
# Original historical bets:
# A B C D E
#
# A bootstrap sample might be:
# C C A E B
#
# Repeat 20,000 times.
# ============================================================

print()

print(
    "Running bet-level bootstrap..."
)


bet_bootstrap_rows = []


for iteration in range(
    BOOTSTRAP_SAMPLES
):

    sample_indices = rng.integers(

        low=0,

        high=n_bets,

        size=n_bets
    )


    sample_profit = profits[
        sample_indices
    ]


    sample_wins = wins[
        sample_indices
    ]


    total_profit = float(
        sample_profit.sum()
    )


    roi = (
        total_profit
        /
        n_bets
    )


    win_rate = float(
        sample_wins.mean()
    )


    drawdown = maximum_drawdown(
        sample_profit
    )


    bet_bootstrap_rows.append(
        {
            "iteration":
                iteration,

            "profit_units":
                total_profit,

            "roi":
                roi,

            "win_rate":
                win_rate,

            "max_drawdown_units":
                drawdown
        }
    )


bet_bootstrap = pd.DataFrame(
    bet_bootstrap_rows
)


# ============================================================
# WEEK-BLOCK BOOTSTRAP
#
# Bets occurring in the same season/week stay together.
#
# This recognizes that one football Saturday can contain
# common shocks and multiple related wagers.
# ============================================================

print(
    "Running week-block bootstrap..."
)


bets[
    "week_block"
] = (

    bets[
        "season"
    ].astype(
        int
    ).astype(
        str
    )

    +

    "_W"

    +

    bets[
        "week"
    ].astype(
        int
    ).astype(
        str
    )
)


week_blocks = {

    block:
        group.copy()

    for block, group
    in bets.groupby(
        "week_block"
    )
}


block_names = list(
    week_blocks.keys()
)


number_of_blocks = len(
    block_names
)


print(
    f"Historical week blocks: "
    f"{number_of_blocks}"
)


block_bootstrap_rows = []


for iteration in range(
    BOOTSTRAP_SAMPLES
):

    sampled_blocks = rng.choice(

        block_names,

        size=number_of_blocks,

        replace=True
    )


    sampled_profit_arrays = []

    sampled_win_arrays = []


    for block in sampled_blocks:

        block_data = week_blocks[
            block
        ]


        sampled_profit_arrays.append(

            block_data[
                "flat_profit"
            ].to_numpy(
                dtype=float
            )
        )


        sampled_win_arrays.append(

            block_data[
                "bet_won"
            ].to_numpy(
                dtype=int
            )
        )


    sample_profit = np.concatenate(
        sampled_profit_arrays
    )


    sample_wins = np.concatenate(
        sampled_win_arrays
    )


    total_profit = float(
        sample_profit.sum()
    )


    sample_bet_count = len(
        sample_profit
    )


    roi = (

        total_profit

        /
        sample_bet_count
    )


    win_rate = float(
        sample_wins.mean()
    )


    drawdown = maximum_drawdown(
        sample_profit
    )


    block_bootstrap_rows.append(
        {
            "iteration":
                iteration,

            "bets":
                sample_bet_count,

            "profit_units":
                total_profit,

            "roi":
                roi,

            "win_rate":
                win_rate,

            "max_drawdown_units":
                drawdown
        }
    )


block_bootstrap = pd.DataFrame(
    block_bootstrap_rows
)


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize_bootstrap(
    frame,
    name
):

    roi = frame[
        "roi"
    ]


    profit = frame[
        "profit_units"
    ]


    drawdown = frame[
        "max_drawdown_units"
    ]


    return {

        "method":
            name,

        "samples":
            len(
                frame
            ),

        "mean_roi":
            roi.mean(),

        "median_roi":
            roi.median(),

        "roi_2_5_percentile":
            roi.quantile(
                0.025
            ),

        "roi_5_percentile":
            roi.quantile(
                0.05
            ),

        "roi_95_percentile":
            roi.quantile(
                0.95
            ),

        "roi_97_5_percentile":
            roi.quantile(
                0.975
            ),

        "probability_positive_roi":
            (
                roi
                >
                0
            ).mean(),

        "probability_negative_roi":
            (
                roi
                <
                0
            ).mean(),

        "probability_roi_above_5_percent":
            (
                roi
                >=
                0.05
            ).mean(),

        "probability_roi_below_minus_5_percent":
            (
                roi
                <=
                -0.05
            ).mean(),

        "mean_profit_units":
            profit.mean(),

        "profit_2_5_percentile":
            profit.quantile(
                0.025
            ),

        "profit_97_5_percentile":
            profit.quantile(
                0.975
            ),

        "median_max_drawdown":
            drawdown.median(),

        "drawdown_95_percentile":
            drawdown.quantile(
                0.95
            ),

        "probability_drawdown_over_25_units":
            (
                drawdown
                >=
                25
            ).mean(),

        "probability_drawdown_over_40_units":
            (
                drawdown
                >=
                40
            ).mean()
    }


summary_rows = [

    summarize_bootstrap(

        bet_bootstrap,

        "Bet-level bootstrap"
    ),

    summarize_bootstrap(

        block_bootstrap,

        "Week-block bootstrap"
    )
]


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()

print(
    "=" * 120
)

print(
    "BOOTSTRAP UNCERTAINTY SUMMARY"
)

print(
    "=" * 120
)


display_columns = [

    "method",

    "mean_roi",

    "median_roi",

    "roi_2_5_percentile",

    "roi_97_5_percentile",

    "probability_positive_roi",

    "probability_negative_roi",

    "probability_roi_above_5_percent",

    "probability_roi_below_minus_5_percent",

    "median_max_drawdown",

    "drawdown_95_percentile"
]


print(

    summary[
        display_columns
    ].to_string(

        index=False,

        formatters={

            "mean_roi":
                lambda x:
                    f"{x:+.2%}",

            "median_roi":
                lambda x:
                    f"{x:+.2%}",

            "roi_2_5_percentile":
                lambda x:
                    f"{x:+.2%}",

            "roi_97_5_percentile":
                lambda x:
                    f"{x:+.2%}",

            "probability_positive_roi":
                lambda x:
                    f"{x:.2%}",

            "probability_negative_roi":
                lambda x:
                    f"{x:.2%}",

            "probability_roi_above_5_percent":
                lambda x:
                    f"{x:.2%}",

            "probability_roi_below_minus_5_percent":
                lambda x:
                    f"{x:.2%}",

            "median_max_drawdown":
                lambda x:
                    f"{x:.2f}",

            "drawdown_95_percentile":
                lambda x:
                    f"{x:.2f}"
        }
    )
)


# ============================================================
# MORE INTUITIVE INTERPRETATION
# ============================================================

print()

print(
    "=" * 120
)

print(
    "RISK INTERPRETATION"
)

print(
    "=" * 120
)


for _, row in summary.iterrows():

    print()

    print(
        row[
            "method"
        ]
    )


    print(
        "-" * 80
    )


    print(

        f"Estimated chance of positive ROI: "
        f"{row['probability_positive_roi']:.2%}"
    )


    print(

        f"Estimated chance of losing money: "
        f"{row['probability_negative_roi']:.2%}"
    )


    print(

        f"Middle 95% ROI range: "
        f"{row['roi_2_5_percentile']:+.2%} "
        f"to "
        f"{row['roi_97_5_percentile']:+.2%}"
    )


    print(

        f"Chance ROI is at least +5%: "
        f"{row['probability_roi_above_5_percent']:.2%}"
    )


    print(

        f"Chance ROI is -5% or worse: "
        f"{row['probability_roi_below_minus_5_percent']:.2%}"
    )


    print(

        f"Median maximum drawdown: "
        f"{row['median_max_drawdown']:.2f} units"
    )


    print(

        f"95th percentile drawdown: "
        f"{row['drawdown_95_percentile']:.2f} units"
    )


# ============================================================
# POST-HOC +800 SENSITIVITY
#
# IMPORTANT:
#
# We discovered the +800 issue AFTER seeing the results.
#
# Therefore this is NOT a new validated betting strategy.
# It is shown only to understand sensitivity.
# ============================================================

extreme_longshot_count = (

    (
        bets[
            "selected_moneyline"
        ]
        >=
        800
    )

    .sum()
)


non_extreme = bets[
    bets[
        "selected_moneyline"
    ]
    <
    800
].copy()


non_extreme_profit = (

    non_extreme[
        "flat_profit"
    ].sum()
)


non_extreme_roi = (

    non_extreme_profit

    /
    len(
        non_extreme
    )
)


print()

print(
    "=" * 120
)

print(
    "POST-HOC +800 LONGSHOT SENSITIVITY"
)

print(
    "=" * 120
)


print(
    "WARNING: This filter was motivated by results we already saw."
)


print(
    "It must NOT be treated as a validated strategy."
)


print()

print(

    f"+800 or longer bets removed: "
    f"{extreme_longshot_count}"
)


print(

    f"Remaining bets: "
    f"{len(non_extreme)}"
)


print(

    f"Remaining profit: "
    f"{non_extreme_profit:+.2f} units"
)


print(

    f"Remaining ROI: "
    f"{non_extreme_roi:+.2%}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


summary.to_csv(
    OUTPUT_FILE,
    index=False
)


block_bootstrap.to_csv(
    BLOCK_OUTPUT_FILE,
    index=False
)


print()

print(
    "=" * 110
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 110
)


print(
    OUTPUT_FILE
)


print(
    BLOCK_OUTPUT_FILE
)


print()

print(
    "2025 was not loaded, evaluated, "
    "or used."
)