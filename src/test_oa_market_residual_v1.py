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
    "features_v3.csv"
)

DATABASE_FILE = (
    "data/cfb.db"
)

RESULT_FILE = (
    "data/processed/"
    "oa_market_residual_results_v1.csv"
)

PREDICTION_FILE = (
    "data/processed/"
    "oa_market_residual_predictions_v1.csv"
)


PROVIDER = "Bovada"


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
# Only the clean opponent-adjusted features.
#
# We deliberately exclude:
#
# oa_expected_margin
# oa_home_field_estimate
#
# because the estimated home-field coefficient looked
# suspiciously large in the audit.
# ------------------------------------------------------------

OA_FEATURES = [
    "oa_strength_diff",
    "oa_home_prior_games",
    "oa_away_prior_games",
    "oa_available"
]


# ------------------------------------------------------------
# Conservative regularization.
#
# Higher L2 means:
# "Stay closer to Bovada unless the historical evidence
# strongly supports a correction."
# ------------------------------------------------------------

L2_VALUES = [
    0.10,
    1.00,
    10.00
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
    "OPPONENT-ADJUSTED MARKET RESIDUAL MODEL — V1"
)

print(
    "=" * 110
)


print(
    f"Provider: {PROVIDER}"
)


print(
    f"Opponent-adjusted features: "
    f"{OA_FEATURES}"
)


print(
    f"Outer test years: "
    f"{OUTER_TEST_YEARS}"
)


print()

print(
    "MODEL:"
)

print(
    "Final log-odds = Bovada log-odds "
    "+ opponent-adjusted correction"
)


# ============================================================
# ODDS HELPERS
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
        np.asarray(
            probability,
            dtype=float
        )
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


    prediction = (

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
                prediction
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
# LOAD V3 FEATURES
# ============================================================

features = pd.read_csv(
    FEATURE_FILE
)


print()

print(
    f"Feature rows loaded: "
    f"{len(features):,}"
)


print(
    f"Feature columns:     "
    f"{len(features.columns):,}"
)


# ============================================================
# REQUIRED FEATURE CHECK
# ============================================================

required_columns = [

    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "target_home_margin"
]


required_columns += OA_FEATURES


missing = [

    column

    for column in required_columns

    if column not in features.columns
]


if missing:

    raise RuntimeError(

        "Missing required columns: "

        +

        ", ".join(
            missing
        )
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
    ]
    ==
    0,
    "home_moneyline"
] = np.nan


market.loc[
    market[
        "away_moneyline"
    ]
    ==
    0,
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


# ------------------------------------------------------------
# We expect one Bovada row per game.
# ------------------------------------------------------------

duplicate_games = (

    market[
        "game_id"
    ]

    .duplicated(
        keep=False
    )
)


if duplicate_games.any():

    raise RuntimeError(
        "Duplicate Bovada game rows detected."
    )


# ============================================================
# MARKET PROBABILITIES
# ============================================================

market[
    "home_raw_probability"
] = (

    market[
        "home_moneyline"
    ]

    .apply(
        american_to_probability
    )
)


market[
    "away_raw_probability"
] = (

    market[
        "away_moneyline"
    ]

    .apply(
        american_to_probability
    )
)


market[
    "raw_probability_sum"
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
        "raw_probability_sum"
    ]
)


# ============================================================
# JOIN MARKET + FOOTBALL
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
    f"Market + V3 games: "
    f"{len(data):,}"
)


print()

print(
    "COVERAGE"
)

print(
    "-" * 70
)


coverage = (

    data

    .groupby(
        "season"
    )

    .agg(

        games=(
            "game_id",
            "size"
        ),

        oa_available=(
            "oa_available",
            "sum"
        )
    )

    .reset_index()
)


coverage[
    "oa_coverage"
] = (

    coverage[
        "oa_available"
    ]

    /

    coverage[
        "games"
    ]
)


print(

    coverage.to_string(

        index=False,

        formatters={

            "oa_coverage":
                lambda value:
                    f"{value:.2%}"
        }
    )
)


# ============================================================
# PREPROCESS FEATURES
# ============================================================

def prepare_features(
    training_frame,
    evaluation_frame
):

    training_numeric = (

        training_frame[
            OA_FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    evaluation_numeric = (

        evaluation_frame[
            OA_FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    # --------------------------------------------------------
    # Missing OA strength occurs mainly early in the season.
    #
    # Median values are calculated using TRAINING data only.
    # oa_available tells the model whether the rating
    # actually existed.
    # --------------------------------------------------------

    medians = (

        training_numeric

        .median()

        .fillna(
            0.0
        )
    )


    training_imputed = (

        training_numeric

        .fillna(
            medians
        )
    )


    evaluation_imputed = (

        evaluation_numeric

        .fillna(
            medians
        )
    )


    scaler = StandardScaler()


    x_train = (

        scaler

        .fit_transform(
            training_imputed
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
        x_evaluation,
        scaler
    )


# ============================================================
# RESIDUAL MODEL
#
# Start at exactly Bovada.
#
# The model can only learn an adjustment.
# ============================================================

class OAResidualModel(
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


        # ----------------------------------------------------
        # Zero initialization means:
        #
        # before training,
        # corrected probability = Bovada probability.
        # ----------------------------------------------------

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
# INNER MODEL SELECTION
# ============================================================

def select_model_settings(
    inner_train,
    inner_validation
):

    (
        x_train,
        x_validation,
        _
    ) = prepare_features(

        inner_train,

        inner_validation
    )


    x_train = torch.tensor(

        x_train,

        dtype=torch.float32
    ).to(
        DEVICE
    )


    x_validation = torch.tensor(

        x_validation,

        dtype=torch.float32
    ).to(
        DEVICE
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

        model = OAResidualModel(
            len(
                OA_FEATURES
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

                x_train,

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


            total_loss = (

                classification_loss

                +

                l2_value
                *
                weight_penalty
            )


            total_loss.backward()

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

                    x_validation,

                    validation_market_logit
                )


                validation_loss = float(

                    bce(

                        validation_logit,

                        y_validation
                    )

                    .item()
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


                best_epoch = (
                    epoch
                )


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
    training_frame,
    test_frame,
    l2_value,
    epochs
):

    (
        x_train,
        x_test,
        scaler
    ) = prepare_features(

        training_frame,

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

        training_frame[
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

        training_frame[
            "actual_home_win"
        ].to_numpy(
            dtype=np.float32
        ),

        dtype=torch.float32
    ).to(
        DEVICE
    )


    bce = nn.BCEWithLogitsLoss()


    model = OAResidualModel(
        len(
            OA_FEATURES
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
    # OUTER TEST
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
# ROLLING OUTER TEST
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
    # SELECT SETTINGS
    # ========================================================

    selected = select_model_settings(

        inner_train,

        inner_validation
    )


    print()

    print(

        f"Selected L2:    "
        f"{selected['l2']}"
    )


    print(

        f"Selected epoch: "
        f"{selected['epoch']}"
    )


    print(

        f"Inner LogLoss:  "
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


    market_metrics = calculate_metrics(

        actual,

        market_probability
    )


    oa_metrics = calculate_metrics(

        actual,

        corrected_probability
    )


    probability_move = float(

        np.mean(

            np.abs(

                corrected_probability

                -

                market_probability
            )
        )
    )


    print()

    print(
        "RESULTS"
    )


    print(
        "-" * 110
    )


    print(

        f"Bovada        | "
        f"ACC {market_metrics['accuracy']:.3f} | "
        f"Brier {market_metrics['brier']:.4f} | "
        f"LogLoss {market_metrics['logloss']:.4f}"
    )


    print(

        f"Bovada + OA   | "
        f"ACC {oa_metrics['accuracy']:.3f} | "
        f"Brier {oa_metrics['brier']:.4f} | "
        f"LogLoss {oa_metrics['logloss']:.4f}"
    )


    print(

        f"Change        | "
        f"ACC {oa_metrics['accuracy'] - market_metrics['accuracy']:+.4f} | "
        f"Brier {oa_metrics['brier'] - market_metrics['brier']:+.4f} | "
        f"LogLoss {oa_metrics['logloss'] - market_metrics['logloss']:+.4f}"
    )


    print(

        f"Average absolute probability move: "
        f"{probability_move:.2%}"
    )


    print()

    print(
        "OA COEFFICIENTS"
    )


    print(
        "-" * 70
    )


    for feature_name, coefficient in zip(
        OA_FEATURES,
        coefficients
    ):

        print(

            f"{feature_name:<25} "
            f"{coefficient:+.4f}"
        )


    print(

        f"{'INTERCEPT':<25} "
        f"{intercept:+.4f}"
    )


    # ========================================================
    # SUMMARY
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

            "oa_accuracy":
                oa_metrics[
                    "accuracy"
                ],

            "accuracy_change":
                (
                    oa_metrics[
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

            "oa_brier":
                oa_metrics[
                    "brier"
                ],

            "brier_change":
                (
                    oa_metrics[
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

            "oa_logloss":
                oa_metrics[
                    "logloss"
                ],

            "logloss_change":
                (
                    oa_metrics[
                        "logloss"
                    ]

                    -

                    market_metrics[
                        "logloss"
                    ]
                ),

            "average_probability_move":
                probability_move
        }
    )


    # ========================================================
    # GAME PREDICTIONS
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

                "market_home_probability":
                    market_probability[
                        position
                    ],

                "oa_home_probability":
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

                "oa_available":
                    game[
                        "oa_available"
                    ],

                "oa_strength_diff":
                    game[
                        "oa_strength_diff"
                    ]
            }
        )


    # ========================================================
    # COEFFICIENTS
    # ========================================================

    for feature_name, coefficient in zip(
        OA_FEATURES,
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
# FINAL RESULTS
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


average_market_accuracy = (

    results[
        "market_accuracy"
    ].mean()
)


average_oa_accuracy = (

    results[
        "oa_accuracy"
    ].mean()
)


average_market_brier = (

    results[
        "market_brier"
    ].mean()
)


average_oa_brier = (

    results[
        "oa_brier"
    ].mean()
)


average_market_logloss = (

    results[
        "market_logloss"
    ].mean()
)


average_oa_logloss = (

    results[
        "oa_logloss"
    ].mean()
)


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

    f"Bovada accuracy:     "
    f"{average_market_accuracy:.3f}"
)


print(

    f"Bovada + OA accuracy:"
    f" {average_oa_accuracy:.3f}"
)


print(

    f"Accuracy change:     "
    f"{average_oa_accuracy - average_market_accuracy:+.4f}"
)


print()

print(

    f"Bovada Brier:        "
    f"{average_market_brier:.4f}"
)


print(

    f"Bovada + OA Brier:   "
    f"{average_oa_brier:.4f}"
)


print(

    f"Brier change:        "
    f"{average_oa_brier - average_market_brier:+.4f}"
)


print()

print(

    f"Bovada LogLoss:      "
    f"{average_market_logloss:.4f}"
)


print(

    f"Bovada + OA LogLoss: "
    f"{average_oa_logloss:.4f}"
)


print(

    f"LogLoss change:      "
    f"{average_oa_logloss - average_market_logloss:+.4f}"
)


print()

print(

    f"Average probability move: "
    f"{results['average_probability_move'].mean():.2%}"
)


# ============================================================
# COEFFICIENT STABILITY
# ============================================================

coefficient_pivot = (

    coefficients[
        coefficients[
            "feature"
        ]
        !=
        "INTERCEPT"
    ]

    .pivot(

        index="feature",

        columns="outer_year",

        values="coefficient"
    )
)


print()

print(
    "=" * 110
)


print(
    "COEFFICIENT STABILITY"
)


print(
    "=" * 110
)


print(
    coefficient_pivot.to_string(
        float_format=lambda value:
            f"{value:+.4f}"
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