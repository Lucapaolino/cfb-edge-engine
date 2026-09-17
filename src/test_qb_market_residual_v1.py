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
    "features_v4.csv"
)

DATABASE_FILE = (
    "data/cfb.db"
)

RESULT_FILE = (
    "data/processed/"
    "qb_market_residual_results_v1.csv"
)

PREDICTION_FILE = (
    "data/processed/"
    "qb_market_residual_predictions_v1.csv"
)

COEFFICIENT_FILE = (
    "data/processed/"
    "qb_market_residual_coefficients_v1.csv"
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


# ============================================================
# CLEAN QB FEATURES ONLY
#
# We deliberately DO NOT use:
#
# qb_*_last_primary_attempt_share
# qb_*_top_passer_attempt_share
#
# because the current builder does not count secondary-passer
# attempts toward those season-level shares.
# ============================================================

QB_FEATURES = [

    # --------------------------------------------------------
    # Difference in how many previous games each team's
    # current primary passer has remained primary.
    # --------------------------------------------------------
    "qb_diff_primary_streak",

    # --------------------------------------------------------
    # Difference in the fraction of previous games handled
    # by the most recent primary passer.
    # --------------------------------------------------------
    "qb_diff_last_primary_start_share",

    # --------------------------------------------------------
    # How dominant the primary passer was in the immediately
    # previous game.
    #
    # This feature IS clean because it uses all attempts in
    # that specific team/game.
    # --------------------------------------------------------
    "qb_diff_last_game_primary_attempt_share",

    # --------------------------------------------------------
    # Difference in number of different primary passers.
    # More different QBs usually means less continuity.
    # --------------------------------------------------------
    "qb_diff_distinct_primary_passers",

    # --------------------------------------------------------
    # Difference in historical QB-change frequency.
    # --------------------------------------------------------
    "qb_diff_qb_change_rate",

    # --------------------------------------------------------
    # Whether each team used the same primary passer in its
    # previous two games.
    # --------------------------------------------------------
    "qb_diff_same_primary_last2",

    # --------------------------------------------------------
    # Missingness indicators.
    #
    # These tell the model whether continuity information was
    # actually available for each team.
    # --------------------------------------------------------
    "qb_home_continuity_available",
    "qb_away_continuity_available"
]


# ============================================================
# REGULARIZATION
#
# Large values force the model to remain close to Bovada.
# ============================================================

L2_VALUES = [
    1.0,
    10.0,
    100.0
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
    "QB CONTINUITY MARKET RESIDUAL MODEL — V1"
)

print(
    "=" * 110
)


print(
    f"Provider: {PROVIDER}"
)


print(
    f"QB features: {len(QB_FEATURES)}"
)


for feature in QB_FEATURES:

    print(
        f"  {feature}"
    )


print()

print(
    "MODEL:"
)


print(
    "Final log-odds = Bovada log-odds "
    "+ learned QB-continuity correction"
)


print()

print(
    "IMPORTANT:"
)


print(
    "Epoch 0 is allowed."
)


print(
    "If historical validation says QB continuity adds no value,"
)


print(
    "the selected model can remain exactly equal to Bovada."
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
# LOAD FEATURES
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
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [

    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "target_home_margin"
]


required_columns += QB_FEATURES


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
# LOAD BOVADA
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


# ============================================================
# DUPLICATE AUDIT
# ============================================================

if market[
    "game_id"
].duplicated(
    keep=False
).any():

    raise RuntimeError(
        "Duplicate Bovada game rows detected."
    )


# ============================================================
# MARKET PROBABILITY
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
# JOIN
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
    f"Market + QB games: "
    f"{len(data):,}"
)


# ============================================================
# COVERAGE
# ============================================================

data[
    "both_qb_continuity_available"
] = (

    (
        data[
            "qb_home_continuity_available"
        ]
        ==
        1
    )

    &

    (
        data[
            "qb_away_continuity_available"
        ]
        ==
        1
    )
).astype(
    int
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

        both_qb_continuity=(
            "both_qb_continuity_available",
            "sum"
        )
    )

    .reset_index()
)


coverage[
    "coverage"
] = (

    coverage[
        "both_qb_continuity"
    ]

    /

    coverage[
        "games"
    ]
)


print()

print(
    "MARKET-GAME QB COVERAGE"
)


print(
    "-" * 80
)


print(

    coverage.to_string(

        index=False,

        formatters={

            "coverage":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# FEATURE PREPROCESSING
# ============================================================

def prepare_features(
    training_frame,
    evaluation_frame
):

    train_numeric = (

        training_frame[
            QB_FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    evaluation_numeric = (

        evaluation_frame[
            QB_FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    # --------------------------------------------------------
    # Training-only median imputation.
    # --------------------------------------------------------

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
# MARKET RESIDUAL MODEL
#
# Starting state:
#
# correction = 0
#
# therefore:
#
# final prediction = Bovada
# ============================================================

class QBResidualModel(
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
# SELECT SETTINGS
#
# IMPORTANT:
#
# We evaluate EPOCH ZERO before making a single update.
#
# That gives the model permission to say:
#
# "Bovada is better. Do nothing."
# ============================================================

def select_model_settings(
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

        model = QBResidualModel(
            len(
                QB_FEATURES
            )
        ).to(
            DEVICE
        )


        optimizer = torch.optim.Adam(

            model.parameters(),

            lr=LEARNING_RATE
        )


        # ====================================================
        # EPOCH ZERO
        #
        # Model correction is exactly zero.
        # ====================================================

        model.eval()


        with torch.no_grad():

            (
                initial_validation_logit,
                _
            ) = model(

                x_validation_tensor,

                validation_market_logit
            )


            initial_validation_loss = float(

                bce(

                    initial_validation_logit,

                    y_validation
                )

                .item()
            )


        best_validation_loss = (
            initial_validation_loss
        )


        best_epoch = 0


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


            training_loss = (

                classification_loss

                +

                l2_value
                *
                weight_penalty
            )


            training_loss.backward()

            optimizer.step()


            # ================================================
            # VALIDATION
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
                best_validation_loss,

            "bovada_validation_logloss":
                initial_validation_loss
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
        x_test
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


    model = QBResidualModel(
        len(
            QB_FEATURES
        )
    ).to(
        DEVICE
    )


    bce = nn.BCEWithLogitsLoss()


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # epochs may legitimately be ZERO.
    # --------------------------------------------------------

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
    # PREDICT
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

    validation_year = (
        outer_year
        -
        1
    )


    inner_train = data[
        data[
            "season"
        ]
        <
        validation_year
    ].copy()


    inner_validation = data[
        data[
            "season"
        ]
        ==
        validation_year
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
        f"{validation_year} "
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

        f"Selected L2:          "
        f"{selected['l2']}"
    )


    print(

        f"Selected epoch:       "
        f"{selected['epoch']}"
    )


    print(

        f"Bovada inner LogLoss: "
        f"{selected['bovada_validation_logloss']:.4f}"
    )


    print(

        f"Best inner LogLoss:   "
        f"{selected['validation_logloss']:.4f}"
    )


    # ========================================================
    # TRAIN FINAL MODEL
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
    # OVERALL METRICS
    # ========================================================

    market_metrics = calculate_metrics(

        actual,

        market_probability
    )


    qb_metrics = calculate_metrics(

        actual,

        corrected_probability
    )


    average_move = float(

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
        "OVERALL RESULTS"
    )


    print(
        "-" * 110
    )


    print(

        f"Bovada      | "
        f"ACC {market_metrics['accuracy']:.3f} | "
        f"Brier {market_metrics['brier']:.4f} | "
        f"LogLoss {market_metrics['logloss']:.4f}"
    )


    print(

        f"Bovada + QB | "
        f"ACC {qb_metrics['accuracy']:.3f} | "
        f"Brier {qb_metrics['brier']:.4f} | "
        f"LogLoss {qb_metrics['logloss']:.4f}"
    )


    print(

        f"Change      | "
        f"ACC {qb_metrics['accuracy'] - market_metrics['accuracy']:+.4f} | "
        f"Brier {qb_metrics['brier'] - market_metrics['brier']:+.4f} | "
        f"LogLoss {qb_metrics['logloss'] - market_metrics['logloss']:+.4f}"
    )


    print(

        f"Average absolute probability move: "
        f"{average_move:.2%}"
    )


    # ========================================================
    # BOTH-TEAMS-CONTINUITY SUBSET
    #
    # Diagnostic only.
    #
    # We are NOT declaring this a betting rule.
    # ========================================================

    both_mask = (

        outer_test[
            "both_qb_continuity_available"
        ].to_numpy(
            dtype=int
        )

        ==
        1
    )


    if both_mask.sum() > 0:

        subset_market_metrics = calculate_metrics(

            actual[
                both_mask
            ],

            market_probability[
                both_mask
            ]
        )


        subset_qb_metrics = calculate_metrics(

            actual[
                both_mask
            ],

            corrected_probability[
                both_mask
            ]
        )


        print()

        print(
            "BOTH-TEAMS QB CONTINUITY AVAILABLE"
        )


        print(
            "-" * 110
        )


        print(

            f"Games: "
            f"{both_mask.sum():,}"
        )


        print(

            f"Bovada      | "
            f"ACC {subset_market_metrics['accuracy']:.3f} | "
            f"Brier {subset_market_metrics['brier']:.4f} | "
            f"LogLoss {subset_market_metrics['logloss']:.4f}"
        )


        print(

            f"Bovada + QB | "
            f"ACC {subset_qb_metrics['accuracy']:.3f} | "
            f"Brier {subset_qb_metrics['brier']:.4f} | "
            f"LogLoss {subset_qb_metrics['logloss']:.4f}"
        )


        print(

            f"Change      | "
            f"ACC {subset_qb_metrics['accuracy'] - subset_market_metrics['accuracy']:+.4f} | "
            f"Brier {subset_qb_metrics['brier'] - subset_market_metrics['brier']:+.4f} | "
            f"LogLoss {subset_qb_metrics['logloss'] - subset_market_metrics['logloss']:+.4f}"
        )


    # ========================================================
    # COEFFICIENTS
    # ========================================================

    print()

    print(
        "QB COEFFICIENTS"
    )


    print(
        "-" * 80
    )


    for feature_name, coefficient in zip(
        QB_FEATURES,
        coefficients
    ):

        print(

            f"{feature_name:<45}"
            f"{coefficient:+.5f}"
        )


    print(

        f"{'INTERCEPT':<45}"
        f"{intercept:+.5f}"
    )


    # ========================================================
    # SAVE RESULT ROW
    # ========================================================

    result_rows.append(
        {
            "outer_year":
                outer_year,

            "games":
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

            "qb_accuracy":
                qb_metrics[
                    "accuracy"
                ],

            "accuracy_change":
                (
                    qb_metrics[
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

            "qb_brier":
                qb_metrics[
                    "brier"
                ],

            "brier_change":
                (
                    qb_metrics[
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

            "qb_logloss":
                qb_metrics[
                    "logloss"
                ],

            "logloss_change":
                (
                    qb_metrics[
                        "logloss"
                    ]

                    -

                    market_metrics[
                        "logloss"
                    ]
                ),

            "average_probability_move":
                average_move
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

                "market_home_probability":
                    market_probability[
                        position
                    ],

                "qb_home_probability":
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
                    ],

                "both_qb_continuity_available":
                    game[
                        "both_qb_continuity_available"
                    ]
            }
        )


    # ========================================================
    # COEFFICIENT STORAGE
    # ========================================================

    for feature_name, coefficient in zip(
        QB_FEATURES,
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
# AVERAGE OUTER RESULTS
# ============================================================

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


market_accuracy = results[
    "market_accuracy"
].mean()


qb_accuracy = results[
    "qb_accuracy"
].mean()


market_brier = results[
    "market_brier"
].mean()


qb_brier = results[
    "qb_brier"
].mean()


market_logloss = results[
    "market_logloss"
].mean()


qb_logloss = results[
    "qb_logloss"
].mean()


print(

    f"Bovada accuracy:      "
    f"{market_accuracy:.4f}"
)


print(

    f"Bovada + QB accuracy: "
    f"{qb_accuracy:.4f}"
)


print(

    f"Accuracy change:      "
    f"{qb_accuracy - market_accuracy:+.4f}"
)


print()

print(

    f"Bovada Brier:         "
    f"{market_brier:.4f}"
)


print(

    f"Bovada + QB Brier:    "
    f"{qb_brier:.4f}"
)


print(

    f"Brier change:         "
    f"{qb_brier - market_brier:+.4f}"
)


print()

print(

    f"Bovada LogLoss:       "
    f"{market_logloss:.4f}"
)


print(

    f"Bovada + QB LogLoss:  "
    f"{qb_logloss:.4f}"
)


print(

    f"LogLoss change:       "
    f"{qb_logloss - market_logloss:+.4f}"
)


print()

print(

    f"Average probability move: "
    f"{results['average_probability_move'].mean():.2%}"
)


# ============================================================
# COEFFICIENT STABILITY
# ============================================================

pivot = (

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

    pivot.to_string(

        float_format=lambda x:
            f"{x:+.5f}"
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