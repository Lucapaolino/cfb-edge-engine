import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "nn_vs_bovada_market_v1.csv"
)


RESULT_FILE = (
    "data/processed/"
    "market_aware_win_results_v1.csv"
)


PREDICTION_FILE = (
    "data/processed/"
    "market_aware_win_predictions_v1.csv"
)


# ------------------------------------------------------------
# 2021 is our first season with historical moneyline coverage.
#
# Therefore:
#
# 2021 can TRAIN the first market-aware model,
# but it cannot itself be an out-of-sample market-aware test.
# ------------------------------------------------------------

TEST_YEARS = [
    2022,
    2023,
    2024
]


# ------------------------------------------------------------
# Weight assigned to the NN in the simple blend.
#
# 0.00 = 100% Bovada
# 1.00 = 100% NN
#
# We deliberately use a coarse grid to avoid excessive
# optimization.
# ------------------------------------------------------------

BLEND_WEIGHTS = np.arange(
    0.00,
    1.001,
    0.05
)


# ============================================================
# HELPERS
# ============================================================

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
        >= 0.50
    ).astype(
        int
    )


    accuracy = accuracy_score(
        actual,
        prediction
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
# LOAD DATA
# ============================================================

print(
    "MARKET-AWARE WIN PROBABILITY MODEL — V1"
)

print(
    "=" * 100
)


data = pd.read_csv(
    INPUT_FILE
)


# ------------------------------------------------------------
# Hard development-only restriction.
# ------------------------------------------------------------

data = data[
    data[
        "season"
    ].isin(
        [
            2021,
            2022,
            2023,
            2024
        ]
    )
].copy()


if (
    data[
        "season"
    ].max()
    >= 2025
):

    raise RuntimeError(
        "2025 leakage detected."
    )


required_columns = [

    "game_id",

    "season",

    "actual_home_win",

    "model_home_win_probability",

    "market_home_fair_probability",

    "home_moneyline",

    "away_moneyline",

    "home_decimal_odds",

    "away_decimal_odds"
]


missing = [

    column

    for column in required_columns

    if column not in data.columns
]


if missing:

    raise RuntimeError(

        "Missing columns: "

        +

        ", ".join(
            missing
        )
    )


print(
    f"Games loaded: "
    f"{len(data):,}"
)


print(
    "2025 is excluded."
)


# ============================================================
# STORAGE
# ============================================================

results = []

prediction_rows = []


# ============================================================
# ROLLING MARKET-AWARE TEST
# ============================================================

for test_year in TEST_YEARS:

    print()

    print(
        "=" * 100
    )

    print(
        f"TEST YEAR {test_year}"
    )

    print(
        "=" * 100
    )


    # --------------------------------------------------------
    # Meta-model can only learn from prior seasons.
    # --------------------------------------------------------

    train = data[
        data[
            "season"
        ]
        <
        test_year
    ].copy()


    test = data[
        data[
            "season"
        ]
        ==
        test_year
    ].copy()


    if len(
        train
    ) == 0:

        raise RuntimeError(
            "No training data."
        )


    if len(
        test
    ) == 0:

        raise RuntimeError(
            "No test data."
        )


    print(

        f"Train seasons: "
        f"{int(train['season'].min())}-"
        f"{int(train['season'].max())}"
    )


    print(
        f"Train games:   "
        f"{len(train):,}"
    )


    print(
        f"Test games:    "
        f"{len(test):,}"
    )


    y_train = (

        train[
            "actual_home_win"
        ]

        .astype(
            int
        )

        .to_numpy()
    )


    y_test = (

        test[
            "actual_home_win"
        ]

        .astype(
            int
        )

        .to_numpy()
    )


    train_market = clip_probability(

        train[
            "market_home_fair_probability"
        ].to_numpy()
    )


    train_nn = clip_probability(

        train[
            "model_home_win_probability"
        ].to_numpy()
    )


    test_market = clip_probability(

        test[
            "market_home_fair_probability"
        ].to_numpy()
    )


    test_nn = clip_probability(

        test[
            "model_home_win_probability"
        ].to_numpy()
    )


    # ========================================================
    # BASELINE 1 — BOVADA
    # ========================================================

    (
        market_accuracy,
        market_brier,
        market_logloss
    ) = calculate_metrics(

        y_test,

        test_market
    )


    # ========================================================
    # BASELINE 2 — NN
    # ========================================================

    (
        nn_accuracy,
        nn_brier,
        nn_logloss
    ) = calculate_metrics(

        y_test,

        test_nn
    )


    # ========================================================
    # 50 / 50 BLEND
    # ========================================================

    fifty_fifty_probability = (

        0.50
        *
        test_market

        +

        0.50
        *
        test_nn
    )


    (
        fifty_accuracy,
        fifty_brier,
        fifty_logloss
    ) = calculate_metrics(

        y_test,

        fifty_fifty_probability
    )


    # ========================================================
    # TUNED SIMPLE BLEND
    #
    # Choose the NN weight using ONLY PREVIOUS seasons.
    #
    # The optimization metric is LogLoss because our
    # ultimate use depends on accurate probabilities.
    # ========================================================

    best_weight = None

    best_train_logloss = float(
        "inf"
    )


    for nn_weight in BLEND_WEIGHTS:

        blended_train_probability = (

            (
                1.0
                -
                nn_weight
            )

            *
            train_market

            +

            nn_weight
            *
            train_nn
        )


        (
            _,
            _,
            blend_train_logloss
        ) = calculate_metrics(

            y_train,

            blended_train_probability
        )


        if (
            blend_train_logloss
            <
            best_train_logloss
        ):

            best_train_logloss = (
                blend_train_logloss
            )

            best_weight = float(
                nn_weight
            )


    tuned_probability = (

        (
            1.0
            -
            best_weight
        )

        *
        test_market

        +

        best_weight
        *
        test_nn
    )


    (
        tuned_accuracy,
        tuned_brier,
        tuned_logloss
    ) = calculate_metrics(

        y_test,

        tuned_probability
    )


    # ========================================================
    # LOGISTIC STACK
    #
    # Instead of manually averaging probabilities, let a
    # regularized logistic model learn how much information
    # to take from:
    #
    #   Bovada probability
    #   NN probability
    #
    # We feed the LOG-ODDS representation because logistic
    # relationships are naturally expressed in log-odds.
    # ========================================================

    x_train_stack = np.column_stack(
        [
            probability_to_logit(
                train_market
            ),

            probability_to_logit(
                train_nn
            )
        ]
    )


    x_test_stack = np.column_stack(
        [
            probability_to_logit(
                test_market
            ),

            probability_to_logit(
                test_nn
            )
        ]
    )


    stack = LogisticRegression(

        C=1.0,

        max_iter=5000,

        solver="lbfgs"
    )


    stack.fit(

        x_train_stack,

        y_train
    )


    stacked_probability = (

        stack

        .predict_proba(
            x_test_stack
        )[
            :,
            1
        ]
    )


    (
        stack_accuracy,
        stack_brier,
        stack_logloss
    ) = calculate_metrics(

        y_test,

        stacked_probability
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()

    print(
        "RESULTS"
    )

    print(
        "-" * 100
    )


    print(

        f"Bovada          | "
        f"ACC {market_accuracy:.3f} | "
        f"Brier {market_brier:.3f} | "
        f"LogLoss {market_logloss:.3f}"
    )


    print(

        f"NN              | "
        f"ACC {nn_accuracy:.3f} | "
        f"Brier {nn_brier:.3f} | "
        f"LogLoss {nn_logloss:.3f}"
    )


    print(

        f"50/50 Blend     | "
        f"ACC {fifty_accuracy:.3f} | "
        f"Brier {fifty_brier:.3f} | "
        f"LogLoss {fifty_logloss:.3f}"
    )


    print(

        f"Tuned Blend     | "
        f"ACC {tuned_accuracy:.3f} | "
        f"Brier {tuned_brier:.3f} | "
        f"LogLoss {tuned_logloss:.3f}"
    )


    print(

        f"Selected NN weight: "
        f"{best_weight:.0%}"
    )


    print(

        f"Logistic Stack  | "
        f"ACC {stack_accuracy:.3f} | "
        f"Brier {stack_brier:.3f} | "
        f"LogLoss {stack_logloss:.3f}"
    )


    print(

        f"Stack coefficients — "
        f"Market: {stack.coef_[0][0]:.3f} | "
        f"NN: {stack.coef_[0][1]:.3f} | "
        f"Intercept: {stack.intercept_[0]:.3f}"
    )


    # ========================================================
    # SAVE SUMMARY ROWS
    # ========================================================

    year_models = [

        (
            "Bovada",
            test_market,
            market_accuracy,
            market_brier,
            market_logloss,
            np.nan
        ),

        (
            "NN",
            test_nn,
            nn_accuracy,
            nn_brier,
            nn_logloss,
            np.nan
        ),

        (
            "50/50 Blend",
            fifty_fifty_probability,
            fifty_accuracy,
            fifty_brier,
            fifty_logloss,
            0.50
        ),

        (
            "Tuned Blend",
            tuned_probability,
            tuned_accuracy,
            tuned_brier,
            tuned_logloss,
            best_weight
        ),

        (
            "Logistic Stack",
            stacked_probability,
            stack_accuracy,
            stack_brier,
            stack_logloss,
            np.nan
        )
    ]


    for (
        model_name,
        probabilities,
        accuracy,
        brier,
        loss,
        nn_weight
    ) in year_models:

        results.append(
            {
                "test_year":
                    test_year,

                "model":
                    model_name,

                "train_games":
                    len(
                        train
                    ),

                "test_games":
                    len(
                        test
                    ),

                "nn_weight":
                    nn_weight,

                "accuracy":
                    accuracy,

                "brier":
                    brier,

                "logloss":
                    loss
            }
        )


        for position in range(
            len(
                test
            )
        ):

            game = test.iloc[
                position
            ]


            prediction_rows.append(
                {
                    "game_id":
                        game[
                            "game_id"
                        ],

                    "season":
                        test_year,

                    "model":
                        model_name,

                    "actual_home_win":
                        int(
                            y_test[
                                position
                            ]
                        ),

                    "predicted_home_win_probability":
                        probabilities[
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

                    "home_decimal_odds":
                        game[
                            "home_decimal_odds"
                        ],

                    "away_decimal_odds":
                        game[
                            "away_decimal_odds"
                        ],

                    "market_home_fair_probability":
                        game[
                            "market_home_fair_probability"
                        ],

                    "original_nn_probability":
                        game[
                            "model_home_win_probability"
                        ]
                }
            )


# ============================================================
# RESULTS DATAFRAMES
# ============================================================

results = pd.DataFrame(
    results
)


predictions = pd.DataFrame(
    prediction_rows
)


# ============================================================
# AVERAGE RESULTS
#
# Each outer test season receives equal weight.
# ============================================================

average_results = (

    results

    .groupby(
        "model",
        as_index=False
    )

    .agg(

        accuracy=(
            "accuracy",
            "mean"
        ),

        brier=(
            "brier",
            "mean"
        ),

        logloss=(
            "logloss",
            "mean"
        ),

        average_nn_weight=(
            "nn_weight",
            "mean"
        )
    )
)


average_results = (

    average_results

    .sort_values(
        "logloss"
    )
)


print()

print(
    "=" * 100
)

print(
    "AVERAGE OUT-OF-SAMPLE MARKET-AWARE RESULTS"
)

print(
    "=" * 100
)


print(

    average_results.to_string(

        index=False,

        formatters={

            "accuracy":
                lambda value:
                    f"{value:.3f}",

            "brier":
                lambda value:
                    f"{value:.3f}",

            "logloss":
                lambda value:
                    f"{value:.3f}",

            "average_nn_weight":
                lambda value:
                    (
                        f"{value:.1%}"

                        if pd.notna(
                            value
                        )

                        else
                        "-"
                    )
        }
    )
)


# ============================================================
# DIRECT COMPARISON TO BOVADA
# ============================================================

market_row = average_results[
    average_results[
        "model"
    ]
    ==
    "Bovada"
].iloc[
    0
]


print()

print(
    "=" * 100
)

print(
    "CHANGE VS BOVADA"
)

print(
    "=" * 100
)


for _, row in average_results.iterrows():

    if (
        row[
            "model"
        ]
        ==
        "Bovada"
    ):

        continue


    print()

    print(
        row[
            "model"
        ]
    )


    print(

        f"Accuracy change: "
        f"{row['accuracy'] - market_row['accuracy']:+.4f}"
    )


    print(

        f"Brier change:    "
        f"{row['brier'] - market_row['brier']:+.4f}"
    )


    print(

        f"LogLoss change:  "
        f"{row['logloss'] - market_row['logloss']:+.4f}"
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
    "=" * 100
)

print(
    "RESULT FILES"
)

print(
    "=" * 100
)


print(
    RESULT_FILE
)


print(
    PREDICTION_FILE
)


print()

print(
    "2025 was not loaded, trained on, "
    "tuned on, or scored."
)