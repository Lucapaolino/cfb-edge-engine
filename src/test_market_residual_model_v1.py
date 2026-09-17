import os
import sqlite3

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)

from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = (
    "data/processed/"
    "features_v2.csv"
)

DATABASE_FILE = (
    "data/cfb.db"
)

RESULT_FILE = (
    "data/processed/"
    "market_residual_results_v1.csv"
)

PREDICTION_FILE = (
    "data/processed/"
    "market_residual_predictions_v1.csv"
)

COEFFICIENT_FILE = (
    "data/processed/"
    "market_residual_coefficients_v1.csv"
)


PROVIDER = "Bovada"


# ------------------------------------------------------------
# 2025 is no longer reserved.
#
# It has already been revealed and can now be treated as
# development/history.
# ------------------------------------------------------------

MARKET_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025
]


OUTER_TEST_YEARS = [
    2023,
    2024,
    2025
]


# ------------------------------------------------------------
# Small controlled regularization grid.
#
# These are selected using ONLY the historical inner
# validation season for each outer test.
# ------------------------------------------------------------

L2_VALUES = [
    0.01,
    0.10,
    1.00
]


LEARNING_RATE = 0.01

MAX_EPOCHS = 1000

PATIENCE = 60

MIN_DELTA = 0.00001


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# START
# ============================================================

print(
    "MARKET RESIDUAL WIN MODEL — V1"
)

print(
    "=" * 110
)


print(
    f"Provider: {PROVIDER}"
)


print(
    f"Outer test years: "
    f"{OUTER_TEST_YEARS}"
)


print()

print(
    "MODEL IDEA:"
)

print(
    "Final log-odds = Bovada log-odds + learned football correction"
)


# ============================================================
# HELPERS
# ============================================================

def american_to_probability(
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


def clip_probability(
    probability
):

    return np.clip(
        probability,
        1e-6,
        1.0 - 1e-6
    )


def probability_to_logit(
    probability
):

    probability = clip_probability(
        probability
    )


    return np.log(

        probability

        /

        (
            1.0
            -
            probability
        )
    )


def calculate_metrics(
    actual,
    probability
):

    actual = np.asarray(
        actual,
        dtype=int
    )


    probability = clip_probability(
        np.asarray(
            probability,
            dtype=float
        )
    )


    predicted = (

        probability
        >=
        0.50

    ).astype(
        int
    )


    return {

        "accuracy":
            accuracy_score(
                actual,
                predicted
            ),

        "brier":
            brier_score_loss(
                actual,
                probability
            ),

        "logloss":
            log_loss(
                actual,
                probability,
                labels=[
                    0,
                    1
                ]
            )
    }


# ============================================================
# LOAD FEATURE DATA
# ============================================================

features = pd.read_csv(
    FEATURE_FILE
)


print()

print(
    f"Feature rows loaded: "
    f"{len(features):,}"
)


# ============================================================
# EXACT 54-FOOTBALL-FEATURE SET
# ============================================================

v1_columns = list(
    features.columns[
        :192
    ]
)


FOOTBALL_FEATURES = [

    column

    for column in v1_columns

    if column.startswith(
        "diff_pregame_"
    )
]


FOOTBALL_FEATURES += [

    "neutral_site",

    "pregame_elo_diff",

    "talent_diff",

    "returning_ppa_diff",

    "returning_passing_ppa_diff",

    "returning_receiving_ppa_diff",

    "returning_rushing_ppa_diff"
]


FOOTBALL_FEATURES = sorted(
    set(
        FOOTBALL_FEATURES
    )
)


if len(
    FOOTBALL_FEATURES
) != 54:

    raise RuntimeError(

        f"Expected 54 football features. "
        f"Found {len(FOOTBALL_FEATURES)}."
    )


print(
    f"Football features: "
    f"{len(FOOTBALL_FEATURES)}"
)


# ============================================================
# LOAD BOVADA MONEYLINES
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


market = pd.read_sql_query(

    """
    SELECT
        game_id,
        provider,
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


market[
    "game_id"
] = pd.to_numeric(

    market[
        "game_id"
    ],

    errors="coerce"
).astype(
    "Int64"
)


for column in [

    "home_moneyline",
    "away_moneyline"
]:

    market[
        column
    ] = pd.to_numeric(

        market[
            column
        ],

        errors="coerce"
    )


market.loc[
    market[
        "home_moneyline"
    ] == 0,
    "home_moneyline"
] = np.nan


market.loc[
    market[
        "away_moneyline"
    ] == 0,
    "away_moneyline"
] = np.nan


market = market[
    market[
        "home_moneyline"
    ].notna()

    &

    market[
        "away_moneyline"
    ].notna()
].copy()


duplicates = market[
    "game_id"
].duplicated(
    keep=False
)


if duplicates.any():

    raise RuntimeError(
        "Duplicate Bovada game rows detected."
    )


# ============================================================
# RAW + DE-VIGGED MARKET PROBABILITIES
# ============================================================

market[
    "home_raw_probability"
] = market[
    "home_moneyline"
].apply(
    american_to_probability
)


market[
    "away_raw_probability"
] = market[
    "away_moneyline"
].apply(
    american_to_probability
)


market[
    "probability_sum"
] = (

    market[
        "home_raw_probability"
    ]

    +

    market[
        "away_raw_probability"
    ]
)


market[
    "market_home_probability"
] = (

    market[
        "home_raw_probability"
    ]

    /

    market[
        "probability_sum"
    ]
)


# ============================================================
# JOIN FOOTBALL + MARKET
# ============================================================

data = features.merge(

    market[
        [
            "game_id",
            "home_moneyline",
            "away_moneyline",
            "home_raw_probability",
            "away_raw_probability",
            "market_home_probability"
        ]
    ],

    on="game_id",

    how="inner"
)


data = data[
    data[
        "season"
    ].isin(
        MARKET_YEARS
    )
].copy()


data[
    "actual_home_win"
] = (

    data[
        "target_home_margin"
    ]

    >
    0

).astype(
    int
)


data[
    "market_logit"
] = probability_to_logit(

    data[
        "market_home_probability"
    ].to_numpy(
        dtype=float
    )
)


print()

print(
    f"Market + football games: "
    f"{len(data):,}"
)


print()

print(
    "COVERAGE BY SEASON"
)

print(
    "-" * 70
)


coverage = (

    data

    .groupby(
        "season"
    )

    .size()
)


print(
    coverage.to_string()
)


# ============================================================
# PREPROCESS FOOTBALL FEATURES
# ============================================================

def prepare_features(
    train_frame,
    evaluation_frame
):

    train_numeric = (

        train_frame[
            FOOTBALL_FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    evaluation_numeric = (

        evaluation_frame[
            FOOTBALL_FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    medians = (

        train_numeric

        .median()

        .fillna(
            0.0
        )
    )


    train_imputed = train_numeric.fillna(
        medians
    )


    evaluation_imputed = evaluation_numeric.fillna(
        medians
    )


    scaler = StandardScaler()


    x_train = (

        scaler

        .fit_transform(
            train_imputed
        )

        .astype(
            np.float32
        )
    )


    x_evaluation = (

        scaler

        .transform(
            evaluation_imputed
        )

        .astype(
            np.float32
        )
    )


    return (
        x_train,
        x_evaluation
    )


# ============================================================
# TRUE MARKET-RESIDUAL MODEL
#
# Bovada's logit is FIXED.
#
# The model only learns a correction based on the 54
# football features.
#
# Starting weights are all zero, so before training:
#
# corrected probability == Bovada probability.
# ============================================================

class MarketResidualModel(
    nn.Module
):

    def __init__(
        self,
        feature_count
    ):

        super().__init__()


        self.correction = nn.Linear(
            feature_count,
            1
        )


        nn.init.zeros_(
            self.correction.weight
        )


        nn.init.zeros_(
            self.correction.bias
        )


    def forward(
        self,
        football_features,
        market_logit
    ):

        correction = (

            self.correction(
                football_features
            )

            .squeeze(
                1
            )
        )


        final_logit = (

            market_logit

            +

            correction
        )


        return (
            final_logit,
            correction
        )


# ============================================================
# TRAIN WITH INNER VALIDATION
# ============================================================

def select_hyperparameters(
    inner_train,
    inner_validation
):

    (
        x_train,
        x_validation
    ) = prepare_features(

        inner_train,

        inner_validation
    )


    train_market_logit = torch.tensor(

        inner_train[
            "market_logit"
        ].to_numpy(
            dtype=np.float32
        ),

        dtype=torch.float32
    ).to(
        DEVICE
    )


    validation_market_logit = torch.tensor(

        inner_validation[
            "market_logit"
        ].to_numpy(
            dtype=np.float32
        ),

        dtype=torch.float32
    ).to(
        DEVICE
    )


    x_train_tensor = torch.tensor(

        x_train,

        dtype=torch.float32
    ).to(
        DEVICE
    )


    x_validation_tensor = torch.tensor(

        x_validation,

        dtype=torch.float32
    ).to(
        DEVICE
    )


    y_train = torch.tensor(

        inner_train[
            "actual_home_win"
        ].to_numpy(
            dtype=np.float32
        ),

        dtype=torch.float32
    ).to(
        DEVICE
    )


    y_validation = torch.tensor(

        inner_validation[
            "actual_home_win"
        ].to_numpy(
            dtype=np.float32
        ),

        dtype=torch.float32
    ).to(
        DEVICE
    )


    bce = nn.BCEWithLogitsLoss()


    best_overall = None


    for l2_value in L2_VALUES:

        model = MarketResidualModel(
            len(
                FOOTBALL_FEATURES
            )
        ).to(
            DEVICE
        )


        optimizer = torch.optim.Adam(

            model.parameters(),

            lr=LEARNING_RATE
        )


        best_validation_loss = float(
            "inf"
        )

        best_epoch = 1

        stale_epochs = 0


        for epoch in range(
            1,
            MAX_EPOCHS + 1
        ):

            model.train()


            optimizer.zero_grad()


            (
                train_logit,
                _
            ) = model(

                x_train_tensor,

                train_market_logit
            )


            classification_loss = bce(

                train_logit,

                y_train
            )


            weight_penalty = (

                model
                .correction
                .weight
                .pow(
                    2
                )
                .mean()
            )


            train_loss = (

                classification_loss

                +

                l2_value
                *
                weight_penalty
            )


            train_loss.backward()

            optimizer.step()


            # ================================================
            # INNER VALIDATION
            # ================================================

            model.eval()


            with torch.no_grad():

                (
                    validation_logit,
                    _
                ) = model(

                    x_validation_tensor,

                    validation_market_logit
                )


                validation_loss = float(

                    bce(

                        validation_logit,

                        y_validation
                    ).item()
                )


            if (

                validation_loss

                <
                (
                    best_validation_loss
                    -
                    MIN_DELTA
                )
            ):

                best_validation_loss = (
                    validation_loss
                )

                best_epoch = epoch

                stale_epochs = 0


            else:

                stale_epochs += 1


            if stale_epochs >= PATIENCE:

                break


        candidate = {

            "l2":
                l2_value,

            "epoch":
                best_epoch,

            "validation_logloss":
                best_validation_loss
        }


        if (

            best_overall is None

            or

            candidate[
                "validation_logloss"
            ]

            <
            best_overall[
                "validation_logloss"
            ]
        ):

            best_overall = candidate


    return best_overall


# ============================================================
# FINAL TRAINING
# ============================================================

def train_final_model(
    train_frame,
    test_frame,
    l2_value,
    epochs
):

    (
        x_train,
        x_test
    ) = prepare_features(

        train_frame,

        test_frame
    )


    x_train_tensor = torch.tensor(

        x_train,

        dtype=torch.float32
    ).to(
        DEVICE
    )


    x_test_tensor = torch.tensor(

        x_test,

        dtype=torch.float32
    ).to(
        DEVICE
    )


    train_market_logit = torch.tensor(

        train_frame[
            "market_logit"
        ].to_numpy(
            dtype=np.float32
        ),

        dtype=torch.float32
    ).to(
        DEVICE
    )


    test_market_logit = torch.tensor(

        test_frame[
            "market_logit"
        ].to_numpy(
            dtype=np.float32
        ),

        dtype=torch.float32
    ).to(
        DEVICE
    )


    y_train = torch.tensor(

        train_frame[
            "actual_home_win"
        ].to_numpy(
            dtype=np.float32
        ),

        dtype=torch.float32
    ).to(
        DEVICE
    )


    bce = nn.BCEWithLogitsLoss()


    model = MarketResidualModel(
        len(
            FOOTBALL_FEATURES
        )
    ).to(
        DEVICE
    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )


    for _ in range(
        epochs
    ):

        model.train()


        optimizer.zero_grad()


        (
            train_logit,
            _
        ) = model(

            x_train_tensor,

            train_market_logit
        )


        classification_loss = bce(

            train_logit,

            y_train
        )


        weight_penalty = (

            model
            .correction
            .weight
            .pow(
                2
            )
            .mean()
        )


        loss = (

            classification_loss

            +

            l2_value
            *
            weight_penalty
        )


        loss.backward()

        optimizer.step()


    # ========================================================
    # TEST PREDICTION
    # ========================================================

    model.eval()


    with torch.no_grad():

        (
            test_logit,
            correction
        ) = model(

            x_test_tensor,

            test_market_logit
        )


        corrected_probability = (

            torch.sigmoid(
                test_logit
            )

            .cpu()

            .numpy()
        )


        correction = (

            correction

            .cpu()

            .numpy()
        )


        coefficients = (

            model
            .correction
            .weight

            .detach()

            .cpu()

            .numpy()

            .reshape(
                -1
            )
        )


        intercept = float(

            model
            .correction
            .bias

            .detach()

            .cpu()

            .item()
        )


    return (
        corrected_probability,
        correction,
        coefficients,
        intercept
    )


# ============================================================
# STORAGE
# ============================================================

result_rows = []

prediction_rows = []

coefficient_rows = []


# ============================================================
# ROLLING OUTER VALIDATION
# ============================================================

for outer_year in OUTER_TEST_YEARS:

    inner_validation_year = (
        outer_year
        -
        1
    )


    inner_train = data[
        data[
            "season"
        ]
        <
        inner_validation_year
    ].copy()


    inner_validation = data[
        data[
            "season"
        ]
        ==
        inner_validation_year
    ].copy()


    final_train = data[
        data[
            "season"
        ]
        <
        outer_year
    ].copy()


    outer_test = data[
        data[
            "season"
        ]
        ==
        outer_year
    ].copy()


    print()

    print(
        "=" * 110
    )

    print(
        f"OUTER TEST YEAR {outer_year}"
    )

    print(
        "=" * 110
    )


    print(

        f"Inner train:      "
        f"{int(inner_train['season'].min())}-"
        f"{int(inner_train['season'].max())} "
        f"({len(inner_train):,} games)"
    )


    print(

        f"Inner validation: "
        f"{inner_validation_year} "
        f"({len(inner_validation):,} games)"
    )


    print(

        f"Final train:      "
        f"{int(final_train['season'].min())}-"
        f"{int(final_train['season'].max())} "
        f"({len(final_train):,} games)"
    )


    print(

        f"Outer test:       "
        f"{outer_year} "
        f"({len(outer_test):,} games)"
    )


    # ========================================================
    # SELECT REGULARIZATION + TRAINING LENGTH
    # ========================================================

    selected = select_hyperparameters(

        inner_train,

        inner_validation
    )


    print()

    print(

        f"Selected L2:      "
        f"{selected['l2']}"
    )


    print(

        f"Selected epoch:   "
        f"{selected['epoch']}"
    )


    print(

        f"Inner LogLoss:    "
        f"{selected['validation_logloss']:.4f}"
    )


    # ========================================================
    # FINAL MODEL
    # ========================================================

    (
        corrected_probability,
        corrections,
        coefficients,
        intercept
    ) = train_final_model(

        final_train,

        outer_test,

        selected[
            "l2"
        ],

        selected[
            "epoch"
        ]
    )


    actual = (

        outer_test[
            "actual_home_win"
        ]

        .astype(
            int
        )

        .to_numpy()
    )


    market_probability = (

        outer_test[
            "market_home_probability"
        ]

        .to_numpy(
            dtype=float
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

    market_metrics = calculate_metrics(

        actual,

        market_probability
    )


    residual_metrics = calculate_metrics(

        actual,

        corrected_probability
    )


    print()

    print(
        "RESULTS"
    )

    print(
        "-" * 110
    )


    print(

        f"Bovada          | "
        f"ACC {market_metrics['accuracy']:.3f} | "
        f"Brier {market_metrics['brier']:.3f} | "
        f"LogLoss {market_metrics['logloss']:.3f}"
    )


    print(

        f"Residual Model  | "
        f"ACC {residual_metrics['accuracy']:.3f} | "
        f"Brier {residual_metrics['brier']:.3f} | "
        f"LogLoss {residual_metrics['logloss']:.3f}"
    )


    print(

        f"Change          | "
        f"ACC {residual_metrics['accuracy'] - market_metrics['accuracy']:+.4f} | "
        f"Brier {residual_metrics['brier'] - market_metrics['brier']:+.4f} | "
        f"LogLoss {residual_metrics['logloss'] - market_metrics['logloss']:+.4f}"
    )


    print(

        f"Average absolute probability move: "
        f"{np.mean(np.abs(corrected_probability - market_probability)):.2%}"
    )


    print(

        f"Average absolute logit correction: "
        f"{np.mean(np.abs(corrections)):.4f}"
    )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    result_rows.append(
        {
            "outer_year":
                outer_year,

            "test_games":
                len(
                    outer_test
                ),

            "selected_l2":
                selected[
                    "l2"
                ],

            "selected_epoch":
                selected[
                    "epoch"
                ],

            "market_accuracy":
                market_metrics[
                    "accuracy"
                ],

            "residual_accuracy":
                residual_metrics[
                    "accuracy"
                ],

            "accuracy_change":
                (
                    residual_metrics[
                        "accuracy"
                    ]

                    -

                    market_metrics[
                        "accuracy"
                    ]
                ),

            "market_brier":
                market_metrics[
                    "brier"
                ],

            "residual_brier":
                residual_metrics[
                    "brier"
                ],

            "brier_change":
                (
                    residual_metrics[
                        "brier"
                    ]

                    -

                    market_metrics[
                        "brier"
                    ]
                ),

            "market_logloss":
                market_metrics[
                    "logloss"
                ],

            "residual_logloss":
                residual_metrics[
                    "logloss"
                ],

            "logloss_change":
                (
                    residual_metrics[
                        "logloss"
                    ]

                    -

                    market_metrics[
                        "logloss"
                    ]
                ),

            "average_probability_move":
                np.mean(
                    np.abs(
                        corrected_probability
                        -
                        market_probability
                    )
                )
        }
    )


    # ========================================================
    # GAME-LEVEL PREDICTIONS
    # ========================================================

    for position in range(
        len(
            outer_test
        )
    ):

        game = outer_test.iloc[
            position
        ]


        prediction_rows.append(
            {
                "game_id":
                    game[
                        "game_id"
                    ],

                "season":
                    outer_year,

                "week":
                    game[
                        "week"
                    ],

                "home_team":
                    game[
                        "home_team"
                    ],

                "away_team":
                    game[
                        "away_team"
                    ],

                "actual_home_win":
                    actual[
                        position
                    ],

                "home_moneyline":
                    game[
                        "home_moneyline"
                    ],

                "away_moneyline":
                    game[
                        "away_moneyline"
                    ],

                "home_raw_probability":
                    game[
                        "home_raw_probability"
                    ],

                "away_raw_probability":
                    game[
                        "away_raw_probability"
                    ],

                "market_home_probability":
                    market_probability[
                        position
                    ],

                "residual_home_probability":
                    corrected_probability[
                        position
                    ],

                "probability_adjustment":
                    (
                        corrected_probability[
                            position
                        ]

                        -

                        market_probability[
                            position
                        ]
                    ),

                "logit_correction":
                    corrections[
                        position
                    ]
            }
        )


    # ========================================================
    # COEFFICIENTS
    # ========================================================

    for feature_name, coefficient in zip(
        FOOTBALL_FEATURES,
        coefficients
    ):

        coefficient_rows.append(
            {
                "outer_year":
                    outer_year,

                "feature":
                    feature_name,

                "coefficient":
                    coefficient
            }
        )


    coefficient_rows.append(
        {
            "outer_year":
                outer_year,

            "feature":
                "INTERCEPT",

            "coefficient":
                intercept
        }
    )


# ============================================================
# FINAL DATAFRAMES
# ============================================================

results = pd.DataFrame(
    result_rows
)


predictions = pd.DataFrame(
    prediction_rows
)


coefficients = pd.DataFrame(
    coefficient_rows
)


# ============================================================
# AVERAGE RESULTS
# ============================================================

average = {

    "market_accuracy":
        results[
            "market_accuracy"
        ].mean(),

    "residual_accuracy":
        results[
            "residual_accuracy"
        ].mean(),

    "market_brier":
        results[
            "market_brier"
        ].mean(),

    "residual_brier":
        results[
            "residual_brier"
        ].mean(),

    "market_logloss":
        results[
            "market_logloss"
        ].mean(),

    "residual_logloss":
        results[
            "residual_logloss"
        ].mean(),

    "average_probability_move":
        results[
            "average_probability_move"
        ].mean()
}


print()

print(
    "=" * 110
)

print(
    "AVERAGE OUT-OF-SAMPLE RESULTS"
)

print(
    "=" * 110
)


print(

    f"Bovada accuracy:          "
    f"{average['market_accuracy']:.3f}"
)


print(

    f"Residual accuracy:        "
    f"{average['residual_accuracy']:.3f}"
)


print(

    f"Accuracy change:          "
    f"{average['residual_accuracy'] - average['market_accuracy']:+.4f}"
)


print()

print(

    f"Bovada Brier:             "
    f"{average['market_brier']:.4f}"
)


print(

    f"Residual Brier:           "
    f"{average['residual_brier']:.4f}"
)


print(

    f"Brier change:             "
    f"{average['residual_brier'] - average['market_brier']:+.4f}"
)


print()

print(

    f"Bovada LogLoss:           "
    f"{average['market_logloss']:.4f}"
)


print(

    f"Residual LogLoss:         "
    f"{average['residual_logloss']:.4f}"
)


print(

    f"LogLoss change:           "
    f"{average['residual_logloss'] - average['market_logloss']:+.4f}"
)


print()

print(

    f"Average probability move: "
    f"{average['average_probability_move']:.2%}"
)


# ============================================================
# COEFFICIENT STABILITY
#
# Features with the same sign across all three folds are more
# interesting than features whose sign constantly flips.
# ============================================================

coefficient_only = coefficients[
    coefficients[
        "feature"
    ]
    !=
    "INTERCEPT"
].copy()


coefficient_summary = (

    coefficient_only

    .groupby(
        "feature",
        as_index=False
    )

    .agg(

        mean_coefficient=(
            "coefficient",
            "mean"
        ),

        std_coefficient=(
            "coefficient",
            "std"
        ),

        min_coefficient=(
            "coefficient",
            "min"
        ),

        max_coefficient=(
            "coefficient",
            "max"
        )
    )
)


coefficient_summary[
    "same_sign_all_folds"
] = (

    (
        coefficient_summary[
            "min_coefficient"
        ]
        >
        0
    )

    |

    (
        coefficient_summary[
            "max_coefficient"
        ]
        <
        0
    )
)


coefficient_summary[
    "absolute_mean"
] = coefficient_summary[
    "mean_coefficient"
].abs()


top_stable = (

    coefficient_summary[
        coefficient_summary[
            "same_sign_all_folds"
        ]
    ]

    .sort_values(
        "absolute_mean",
        ascending=False
    )

    .head(
        15
    )
)


print()

print(
    "=" * 110
)

print(
    "TOP STABLE MARKET-CORRECTION FEATURES"
)

print(
    "=" * 110
)


if len(
    top_stable
) == 0:

    print(
        "No feature kept the same coefficient sign "
        "across all outer folds."
    )

else:

    print(

        top_stable[
            [
                "feature",
                "mean_coefficient",
                "std_coefficient",
                "min_coefficient",
                "max_coefficient"
            ]
        ].to_string(

            index=False,

            formatters={

                "mean_coefficient":
                    lambda x:
                        f"{x:+.4f}",

                "std_coefficient":
                    lambda x:
                        f"{x:.4f}",

                "min_coefficient":
                    lambda x:
                        f"{x:+.4f}",

                "max_coefficient":
                    lambda x:
                        f"{x:+.4f}"
            }
        )
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


results.to_csv(
    RESULT_FILE,
    index=False
)


predictions.to_csv(
    PREDICTION_FILE,
    index=False
)


coefficients.to_csv(
    COEFFICIENT_FILE,
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
    RESULT_FILE
)


print(
    PREDICTION_FILE
)


print(
    COEFFICIENT_FILE
)