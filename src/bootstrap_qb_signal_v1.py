import os

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "qb_market_residual_predictions_v1.csv"
)

SUMMARY_FILE = (
    "data/processed/"
    "qb_signal_bootstrap_summary_v1.csv"
)

DISTRIBUTION_FILE = (
    "data/processed/"
    "qb_signal_week_block_bootstrap_v1.csv"
)


BOOTSTRAP_SAMPLES = 20_000

RANDOM_SEED = 20260905


TEST_YEARS = [
    2023,
    2024,
    2025
]


# ============================================================
# START
# ============================================================

print(
    "QB SIGNAL PAIRED BOOTSTRAP — V1"
)

print(
    "=" * 110
)


print(
    f"Bootstrap samples: "
    f"{BOOTSTRAP_SAMPLES:,}"
)


print()

print(
    "Negative score difference means:"
)

print(
    "QB model is BETTER than Bovada."
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

data = pd.read_csv(
    INPUT_FILE
)


data = data[
    data[
        "season"
    ].isin(
        TEST_YEARS
    )
].copy()


required_columns = [
    "game_id",
    "season",
    "week",
    "actual_home_win",
    "market_home_probability",
    "qb_home_probability",
    "both_qb_continuity_available"
]


missing = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing:

    raise RuntimeError(

        "Missing required columns: "

        +

        ", ".join(
            missing
        )
    )


numeric_columns = [
    "season",
    "week",
    "actual_home_win",
    "market_home_probability",
    "qb_home_probability",
    "both_qb_continuity_available"
]


for column in numeric_columns:

    data[
        column
    ] = pd.to_numeric(

        data[
            column
        ],

        errors="coerce"
    )


data = data.dropna(
    subset=[
        "season",
        "week",
        "actual_home_win",
        "market_home_probability",
        "qb_home_probability"
    ]
).copy()


data = data.sort_values(
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
    f"Games loaded: "
    f"{len(data):,}"
)


print()

print(
    "GAMES BY SEASON"
)

print(
    "-" * 70
)


print(

    data.groupby(
        "season"
    ).size().to_string()
)


# ============================================================
# CLIP PROBABILITIES
# ============================================================

EPSILON = 1e-12


data[
    "market_probability"
] = np.clip(

    data[
        "market_home_probability"
    ].astype(
        float
    ),

    EPSILON,

    1.0
    -
    EPSILON
)


data[
    "qb_probability"
] = np.clip(

    data[
        "qb_home_probability"
    ].astype(
        float
    ),

    EPSILON,

    1.0
    -
    EPSILON
)


y = data[
    "actual_home_win"
].astype(
    int
)


# ============================================================
# PER-GAME BRIER CONTRIBUTION
# ============================================================

data[
    "market_brier"
] = (

    data[
        "market_probability"
    ]

    -
    y

) ** 2


data[
    "qb_brier"
] = (

    data[
        "qb_probability"
    ]

    -
    y

) ** 2


data[
    "brier_difference"
] = (

    data[
        "qb_brier"
    ]

    -

    data[
        "market_brier"
    ]
)


# ============================================================
# PER-GAME LOGLOSS CONTRIBUTION
# ============================================================

data[
    "market_logloss"
] = -(

    y
    *
    np.log(
        data[
            "market_probability"
        ]
    )

    +

    (
        1
        -
        y
    )
    *
    np.log(
        1.0
        -
        data[
            "market_probability"
        ]
    )
)


data[
    "qb_logloss"
] = -(

    y
    *
    np.log(
        data[
            "qb_probability"
        ]
    )

    +

    (
        1
        -
        y
    )
    *
    np.log(
        1.0
        -
        data[
            "qb_probability"
        ]
    )
)


data[
    "logloss_difference"
] = (

    data[
        "qb_logloss"
    ]

    -

    data[
        "market_logloss"
    ]
)


# ============================================================
# OBSERVED RESULT FUNCTION
# ============================================================

def observed_result(
    frame
):

    return {

        "games":
            len(
                frame
            ),

        "market_brier":
            frame[
                "market_brier"
            ].mean(),

        "qb_brier":
            frame[
                "qb_brier"
            ].mean(),

        "brier_difference":
            frame[
                "brier_difference"
            ].mean(),

        "market_logloss":
            frame[
                "market_logloss"
            ].mean(),

        "qb_logloss":
            frame[
                "qb_logloss"
            ].mean(),

        "logloss_difference":
            frame[
                "logloss_difference"
            ].mean()
    }


# ============================================================
# OVERALL OBSERVED RESULTS
# ============================================================

overall_observed = observed_result(
    data
)


continuity_data = data[
    data[
        "both_qb_continuity_available"
    ]
    ==
    1
].copy()


continuity_observed = observed_result(
    continuity_data
)


print()

print(
    "=" * 110
)

print(
    "OBSERVED PAIRED RESULTS"
)

print(
    "=" * 110
)


print()

print(
    "ALL GAMES"
)

print(
    "-" * 70
)


print(

    f"Games:              "
    f"{overall_observed['games']:,}"
)


print(

    f"Bovada Brier:       "
    f"{overall_observed['market_brier']:.6f}"
)


print(

    f"QB Brier:           "
    f"{overall_observed['qb_brier']:.6f}"
)


print(

    f"Brier difference:   "
    f"{overall_observed['brier_difference']:+.6f}"
)


print()

print(

    f"Bovada LogLoss:     "
    f"{overall_observed['market_logloss']:.6f}"
)


print(

    f"QB LogLoss:         "
    f"{overall_observed['qb_logloss']:.6f}"
)


print(

    f"LogLoss difference: "
    f"{overall_observed['logloss_difference']:+.6f}"
)


print()

print(
    "BOTH TEAMS HAVE QB CONTINUITY"
)

print(
    "-" * 70
)


print(

    f"Games:              "
    f"{continuity_observed['games']:,}"
)


print(

    f"Brier difference:   "
    f"{continuity_observed['brier_difference']:+.6f}"
)


print(

    f"LogLoss difference: "
    f"{continuity_observed['logloss_difference']:+.6f}"
)


# ============================================================
# SEASON-BY-SEASON RESULTS
# ============================================================

print()

print(
    "=" * 110
)

print(
    "OBSERVED RESULTS BY SEASON"
)

print(
    "=" * 110
)


season_rows = []


for season in TEST_YEARS:

    subset = data[
        data[
            "season"
        ]
        ==
        season
    ]


    result = observed_result(
        subset
    )


    season_rows.append(
        {
            "season":
                season,

            "games":
                result[
                    "games"
                ],

            "brier_difference":
                result[
                    "brier_difference"
                ],

            "logloss_difference":
                result[
                    "logloss_difference"
                ]
        }
    )


season_results = pd.DataFrame(
    season_rows
)


print(

    season_results.to_string(

        index=False,

        formatters={

            "brier_difference":
                lambda value:
                    f"{value:+.6f}",

            "logloss_difference":
                lambda value:
                    f"{value:+.6f}"
        }
    )
)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# GAME-LEVEL PAIRED BOOTSTRAP
#
# Both models remain paired on each sampled game.
# ============================================================

def game_level_bootstrap(
    frame
):

    brier_differences = frame[
        "brier_difference"
    ].to_numpy(
        dtype=float
    )


    logloss_differences = frame[
        "logloss_difference"
    ].to_numpy(
        dtype=float
    )


    n_games = len(
        frame
    )


    output = np.zeros(
        (
            BOOTSTRAP_SAMPLES,
            2
        ),
        dtype=float
    )


    for iteration in range(
        BOOTSTRAP_SAMPLES
    ):

        indices = rng.integers(

            0,

            n_games,

            size=n_games
        )


        output[
            iteration,
            0
        ] = brier_differences[
            indices
        ].mean()


        output[
            iteration,
            1
        ] = logloss_differences[
            indices
        ].mean()


    return pd.DataFrame(
        output,
        columns=[
            "brier_difference",
            "logloss_difference"
        ]
    )


# ============================================================
# SEASON-WEEK BLOCK BOOTSTRAP
#
# We sample complete football weeks instead of individual
# games.
#
# We also preserve season composition:
#
# 2023 weeks are resampled from 2023,
# 2024 weeks from 2024,
# 2025 weeks from 2025.
#
# This is more conservative and avoids one random bootstrap
# sample becoming mostly one season.
# ============================================================

def season_week_block_bootstrap(
    frame
):

    season_blocks = {}


    for season in TEST_YEARS:

        season_frame = frame[
            frame[
                "season"
            ]
            ==
            season
        ]


        blocks = {

            int(
                week
            ):
                group.copy()

            for week, group
            in season_frame.groupby(
                "week"
            )
        }


        season_blocks[
            season
        ] = blocks


    rows = []


    for iteration in range(
        BOOTSTRAP_SAMPLES
    ):

        sampled_frames = []


        for season in TEST_YEARS:

            blocks = season_blocks[
                season
            ]


            block_names = list(
                blocks.keys()
            )


            sampled_names = rng.choice(

                block_names,

                size=len(
                    block_names
                ),

                replace=True
            )


            for block_name in sampled_names:

                sampled_frames.append(

                    blocks[
                        int(
                            block_name
                        )
                    ]
                )


        sample = pd.concat(

            sampled_frames,

            ignore_index=True
        )


        rows.append(
            {
                "iteration":
                    iteration,

                "games":
                    len(
                        sample
                    ),

                "brier_difference":
                    sample[
                        "brier_difference"
                    ].mean(),

                "logloss_difference":
                    sample[
                        "logloss_difference"
                    ].mean()
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# RUN BOOTSTRAPS
# ============================================================

print()

print(
    "Running game-level paired bootstrap..."
)


overall_game_bootstrap = game_level_bootstrap(
    data
)


continuity_game_bootstrap = game_level_bootstrap(
    continuity_data
)


print(
    "Running season/week block bootstrap..."
)


overall_block_bootstrap = (
    season_week_block_bootstrap(
        data
    )
)


continuity_block_bootstrap = (
    season_week_block_bootstrap(
        continuity_data
    )
)


# ============================================================
# SUMMARY
# ============================================================

def summarize_distribution(
    frame,
    sample_name,
    bootstrap_name
):

    brier = frame[
        "brier_difference"
    ]


    logloss = frame[
        "logloss_difference"
    ]


    return {

        "sample":
            sample_name,

        "bootstrap":
            bootstrap_name,

        "samples":
            len(
                frame
            ),

        # ----------------------------------------------------
        # Brier
        # ----------------------------------------------------

        "mean_brier_difference":
            brier.mean(),

        "brier_2_5_percentile":
            brier.quantile(
                0.025
            ),

        "brier_97_5_percentile":
            brier.quantile(
                0.975
            ),

        "probability_qb_improves_brier":
            (
                brier
                <
                0
            ).mean(),

        # ----------------------------------------------------
        # LogLoss
        # ----------------------------------------------------

        "mean_logloss_difference":
            logloss.mean(),

        "logloss_2_5_percentile":
            logloss.quantile(
                0.025
            ),

        "logloss_97_5_percentile":
            logloss.quantile(
                0.975
            ),

        "probability_qb_improves_logloss":
            (
                logloss
                <
                0
            ).mean()
    }


summary = pd.DataFrame(
    [
        summarize_distribution(

            overall_game_bootstrap,

            "All games",

            "Game-level paired"
        ),

        summarize_distribution(

            overall_block_bootstrap,

            "All games",

            "Season-week block"
        ),

        summarize_distribution(

            continuity_game_bootstrap,

            "Both-team continuity",

            "Game-level paired"
        ),

        summarize_distribution(

            continuity_block_bootstrap,

            "Both-team continuity",

            "Season-week block"
        )
    ]
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()

print(
    "=" * 130
)

print(
    "BOOTSTRAP SIGNAL SUMMARY"
)

print(
    "=" * 130
)


print(

    summary.to_string(

        index=False,

        formatters={

            "mean_brier_difference":
                lambda x:
                    f"{x:+.6f}",

            "brier_2_5_percentile":
                lambda x:
                    f"{x:+.6f}",

            "brier_97_5_percentile":
                lambda x:
                    f"{x:+.6f}",

            "probability_qb_improves_brier":
                lambda x:
                    f"{x:.2%}",

            "mean_logloss_difference":
                lambda x:
                    f"{x:+.6f}",

            "logloss_2_5_percentile":
                lambda x:
                    f"{x:+.6f}",

            "logloss_97_5_percentile":
                lambda x:
                    f"{x:+.6f}",

            "probability_qb_improves_logloss":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# HUMAN-READABLE INTERPRETATION
# ============================================================

print()

print(
    "=" * 110
)

print(
    "SIGNAL INTERPRETATION"
)

print(
    "=" * 110
)


for _, row in summary.iterrows():

    print()

    print(

        f"{row['sample']} | "
        f"{row['bootstrap']}"
    )


    print(
        "-" * 90
    )


    print(

        f"Chance QB model improves Brier:   "
        f"{row['probability_qb_improves_brier']:.2%}"
    )


    print(

        f"95% Brier difference interval:    "
        f"{row['brier_2_5_percentile']:+.6f} "
        f"to "
        f"{row['brier_97_5_percentile']:+.6f}"
    )


    print()

    print(

        f"Chance QB model improves LogLoss: "
        f"{row['probability_qb_improves_logloss']:.2%}"
    )


    print(

        f"95% LogLoss difference interval:  "
        f"{row['logloss_2_5_percentile']:+.6f} "
        f"to "
        f"{row['logloss_97_5_percentile']:+.6f}"
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


overall_block_save = (
    overall_block_bootstrap.copy()
)


overall_block_save[
    "sample"
] = "All games"


continuity_block_save = (
    continuity_block_bootstrap.copy()
)


continuity_block_save[
    "sample"
] = "Both-team continuity"


block_output = pd.concat(

    [
        overall_block_save,
        continuity_block_save
    ],

    ignore_index=True
)


block_output.to_csv(
    DISTRIBUTION_FILE,
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
    SUMMARY_FILE
)


print(
    DISTRIBUTION_FILE
)