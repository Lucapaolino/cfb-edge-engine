import os

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    Ridge
)

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

feature_file = (
    "data/processed/"
    "features_v2.csv"
)


results_file = (
    "data/processed/"
    "ppa_rolling_validation_results_v2.csv"
)


predictions_file = (
    "data/processed/"
    "ppa_rolling_validation_predictions_v2.csv"
)


validation_years = [
    2021,
    2022,
    2023,
    2024
]


# ============================================================
# LOAD DATA
# ============================================================

print(
    "PPA FEATURE VALIDATION — V2"
)

print(
    "=" * 100
)


data = pd.read_csv(
    feature_file
)


print(
    f"Rows loaded:    "
    f"{len(data):,}"
)

print(
    f"Columns loaded: "
    f"{len(data.columns):,}"
)


# ============================================================
# RESERVED TEST SAFETY CHECK
# ============================================================

if (
    data[
        "season"
    ].max()
    < 2025
):

    raise RuntimeError(
        "Expected 2025 feature rows "
        "to exist."
    )


print(
    "\n2025 rows exist in the feature file, "
    "but will NOT be trained on, validated on, "
    "or scored."
)


# ============================================================
# TARGETS
# ============================================================

margin_target = (
    "target_home_margin"
)


if margin_target not in data.columns:

    raise RuntimeError(
        f"Missing required target column: "
        f"{margin_target}"
    )


# ------------------------------------------------------------
# HOME WIN TARGET
#
# Instead of depending on another stored target column,
# derive it directly from the actual point margin.
#
# Positive margin = home team won.
# ------------------------------------------------------------

data[
    "model_target_home_win"
] = (
    data[
        margin_target
    ] > 0
).astype(
    int
)


win_target = (
    "model_target_home_win"
)


# ============================================================
# ORIGINAL V1 FEATURE BOUNDARY
#
# V1 had exactly 192 columns.
#
# Everything after that belongs to the new V2 PPA work.
# ============================================================

v1_columns = list(
    data.columns[
        :192
    ]
)


# ============================================================
# ORIGINAL V1 MATCHUP-DIFFERENCE FEATURES
#
# The prior V1 experiment used:
#
# 47 rolling matchup-difference features
#
# + neutral_site
# + pregame_elo_diff
# + talent_diff
#
# + 4 returning-production PPA differences
#
# = 54 total features
# ============================================================

v1_difference_features = [

    column

    for column in v1_columns

    if column.startswith(
        "diff_pregame_"
    )
]


# ------------------------------------------------------------
# ORIGINAL NON-ROLLING MATCHUP FEATURES
# ------------------------------------------------------------

v1_extra_features = [

    "neutral_site",

    "pregame_elo_diff",

    "talent_diff",

    "returning_ppa_diff",

    "returning_passing_ppa_diff",

    "returning_receiving_ppa_diff",

    "returning_rushing_ppa_diff"
]


for feature in v1_extra_features:

    if feature not in data.columns:

        raise RuntimeError(
            f"Expected V1 feature is missing: "
            f"{feature}"
        )


v1_difference_features.extend(
    v1_extra_features
)


v1_difference_features = sorted(
    set(
        v1_difference_features
    )
)


# ============================================================
# V1 FEATURE AUDIT
# ============================================================

print("\n")
print(
    "V1 DIFFERENCE FEATURES"
)

print(
    "-" * 100
)


print(
    f"Feature count: "
    f"{len(v1_difference_features)}"
)


for feature in v1_difference_features:

    print(
        feature
    )


# ------------------------------------------------------------
# Critical safety check.
#
# We do NOT proceed if this is not the exact 54-feature
# representation used previously.
# ------------------------------------------------------------

if len(
    v1_difference_features
) != 54:

    print("\n")

    print(
        "ERROR:"
    )

    print(
        "The exact V1 matchup feature set "
        "did not contain 54 features."
    )

    print(
        "Nothing was trained."
    )

    raise SystemExit(
        1
    )


# ============================================================
# NEW V2 PPA DIFFERENCE FEATURES
#
# These are completely separate from the old
# returning_ppa_diff variables.
#
# These come from actual historical play-by-play.
# ============================================================

ppa_difference_features = [

    column

    for column in data.columns

    if (
        column.startswith(
            "diff_pregame_"
        )

        and

        "ppa" in column.lower()

        and

        column not in v1_difference_features
    )
]


# ------------------------------------------------------------
# Add amount of PPA history available.
#
# This tells the model whether the rolling PPA numbers
# are based on one prior game or several.
# ------------------------------------------------------------

if (
    "min_pregame_ppa_games_available"
    in data.columns
):

    ppa_difference_features.append(
        "min_pregame_ppa_games_available"
    )


ppa_difference_features = sorted(
    set(
        ppa_difference_features
    )
)


# ============================================================
# PPA FEATURE AUDIT
# ============================================================

print("\n")
print(
    "NEW V2 PPA FEATURES"
)

print(
    "-" * 100
)


print(
    f"Feature count: "
    f"{len(ppa_difference_features)}"
)


for feature in ppa_difference_features:

    print(
        feature
    )


if len(
    ppa_difference_features
) == 0:

    raise RuntimeError(
        "No new V2 PPA features were found."
    )


# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {

    "V1 Difference":
        v1_difference_features,

    "PPA Only":
        ppa_difference_features,

    "V1 + PPA":
        sorted(
            set(
                v1_difference_features
                +
                ppa_difference_features
            )
        )
}


print("\n")
print(
    "FEATURE SET SIZES"
)

print(
    "-" * 100
)


for (
    name,
    features
) in feature_sets.items():

    print(
        f"{name:<20} "
        f"{len(features):>3} features"
    )


# ============================================================
# NUMERIC DATA HELPER
#
# Converts booleans/numbers safely into numeric model input.
#
# Missing values remain NaN.
# ============================================================

def numeric_features(
    frame,
    columns
):

    return (
        frame[
            columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_ridge():

    return Pipeline(
        steps=[
            (
                "imputer",

                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),

            (
                "scaler",

                StandardScaler()
            ),

            (
                "model",

                Ridge(
                    alpha=10.0
                )
            )
        ]
    )


def build_logistic():

    return Pipeline(
        steps=[
            (
                "imputer",

                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),

            (
                "scaler",

                StandardScaler()
            ),

            (
                "model",

                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    solver="lbfgs"
                )
            )
        ]
    )


def build_gradient_margin():

    return HistGradientBoostingRegressor(

        learning_rate=0.05,

        max_iter=250,

        max_leaf_nodes=15,

        min_samples_leaf=20,

        l2_regularization=1.0,

        # --------------------------------------------
        # We disable random internal early stopping.
        #
        # This keeps the training procedure cleaner
        # for chronological forecasting.
        # --------------------------------------------

        early_stopping=False,

        random_state=42
    )


def build_gradient_win():

    return HistGradientBoostingClassifier(

        learning_rate=0.05,

        max_iter=250,

        max_leaf_nodes=15,

        min_samples_leaf=20,

        l2_regularization=1.0,

        early_stopping=False,

        random_state=42
    )


# ============================================================
# METRICS
# ============================================================

def calculate_margin_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )


    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )


    return (
        mae,
        rmse
    )


def calculate_win_metrics(
    actual,
    probability
):

    predicted = (
        probability
        >= 0.50
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
# STORAGE
# ============================================================

results = []

prediction_rows = []


# ============================================================
# ROLLING HISTORICAL VALIDATION
# ============================================================

for validation_year in validation_years:

    print("\n")
    print(
        "=" * 100
    )

    print(
        f"VALIDATION YEAR "
        f"{validation_year}"
    )

    print(
        "=" * 100
    )


    # --------------------------------------------------------
    # EXPANDING TRAINING WINDOW
    #
    # Example:
    #
    # 2023 validation
    #
    # training = 2018-2022
    # validation = 2023
    # --------------------------------------------------------

    train = data[
        data[
            "season"
        ]
        <
        validation_year
    ].copy()


    validation = data[
        data[
            "season"
        ]
        ==
        validation_year
    ].copy()


    # --------------------------------------------------------
    # Ensure 2025 cannot accidentally enter this experiment.
    # --------------------------------------------------------

    train = train[
        train[
            "season"
        ] <= 2024
    ].copy()


    validation = validation[
        validation[
            "season"
        ] <= 2024
    ].copy()


    # --------------------------------------------------------
    # CHRONOLOGICAL LEAKAGE CHECK
    # --------------------------------------------------------

    if (
        train[
            "season"
        ].max()
        >=
        validation_year
    ):

        raise RuntimeError(
            "Chronological leakage detected."
        )


    print(
        f"Training seasons: "
        f"{int(train['season'].min())}-"
        f"{int(train['season'].max())}"
    )


    print(
        f"Training games:   "
        f"{len(train):,}"
    )


    print(
        f"Validation games: "
        f"{len(validation):,}"
    )


    # ========================================================
    # TARGETS
    # ========================================================

    y_train_margin = (
        train[
            margin_target
        ]
    )


    y_validation_margin = (
        validation[
            margin_target
        ]
    )


    y_train_win = (
        train[
            win_target
        ]
        .astype(
            int
        )
    )


    y_validation_win = (
        validation[
            win_target
        ]
        .astype(
            int
        )
    )


    # ========================================================
    # TEST EACH FEATURE GROUP
    # ========================================================

    for (
        feature_name,
        features
    ) in feature_sets.items():

        print("\n")
        print(
            feature_name
        )

        print(
            "-" * 100
        )


        X_train = numeric_features(
            train,
            features
        )


        X_validation = numeric_features(
            validation,
            features
        )


        # ====================================================
        # RIDGE — MARGIN
        #
        # Ridge is our regularized linear model.
        #
        # It predicts expected home point margin.
        # ====================================================

        ridge = build_ridge()


        ridge.fit(
            X_train,
            y_train_margin
        )


        ridge_margin_prediction = (
            ridge.predict(
                X_validation
            )
        )


        (
            ridge_mae,
            ridge_rmse
        ) = calculate_margin_metrics(

            y_validation_margin,

            ridge_margin_prediction
        )


        # ====================================================
        # LOGISTIC REGRESSION — WIN PROBABILITY
        #
        # Predicts probability that home team wins.
        # ====================================================

        logistic = build_logistic()


        logistic.fit(
            X_train,
            y_train_win
        )


        logistic_win_probability = (

            logistic

            .predict_proba(
                X_validation
            )[
                :,
                1
            ]
        )


        (
            logistic_accuracy,
            logistic_brier,
            logistic_logloss
        ) = calculate_win_metrics(

            y_validation_win,

            logistic_win_probability
        )


        # ====================================================
        # GRADIENT BOOSTING — MARGIN
        #
        # Gradient boosting combines many small decision
        # trees and can capture nonlinear relationships.
        # ====================================================

        gradient_margin = (
            build_gradient_margin()
        )


        gradient_margin.fit(
            X_train,
            y_train_margin
        )


        gradient_margin_prediction = (

            gradient_margin

            .predict(
                X_validation
            )
        )


        (
            gradient_mae,
            gradient_rmse
        ) = calculate_margin_metrics(

            y_validation_margin,

            gradient_margin_prediction
        )


        # ====================================================
        # GRADIENT BOOSTING — WIN PROBABILITY
        # ====================================================

        gradient_win = (
            build_gradient_win()
        )


        gradient_win.fit(
            X_train,
            y_train_win
        )


        gradient_win_probability = (

            gradient_win

            .predict_proba(
                X_validation
            )[
                :,
                1
            ]
        )


        (
            gradient_accuracy,
            gradient_brier,
            gradient_logloss
        ) = calculate_win_metrics(

            y_validation_win,

            gradient_win_probability
        )


        # ====================================================
        # PRINT YEAR RESULTS
        # ====================================================

        print(
            f"Ridge margin:       "
            f"MAE {ridge_mae:.3f} | "
            f"RMSE {ridge_rmse:.3f}"
        )


        print(
            f"Logistic win:       "
            f"ACC {logistic_accuracy:.3f} | "
            f"Brier {logistic_brier:.3f} | "
            f"LogLoss {logistic_logloss:.3f}"
        )


        print(
            f"Gradient margin:    "
            f"MAE {gradient_mae:.3f} | "
            f"RMSE {gradient_rmse:.3f}"
        )


        print(
            f"Gradient win:       "
            f"ACC {gradient_accuracy:.3f} | "
            f"Brier {gradient_brier:.3f} | "
            f"LogLoss {gradient_logloss:.3f}"
        )


        # ====================================================
        # SAVE SUMMARY RESULTS
        # ====================================================

        results.append(
            {
                "validation_year":
                    validation_year,

                "feature_set":
                    feature_name,

                "model":
                    "Linear",

                "margin_mae":
                    ridge_mae,

                "margin_rmse":
                    ridge_rmse,

                "win_accuracy":
                    logistic_accuracy,

                "brier":
                    logistic_brier,

                "logloss":
                    logistic_logloss,

                "train_games":
                    len(
                        train
                    ),

                "validation_games":
                    len(
                        validation
                    )
            }
        )


        results.append(
            {
                "validation_year":
                    validation_year,

                "feature_set":
                    feature_name,

                "model":
                    "Gradient Boosting",

                "margin_mae":
                    gradient_mae,

                "margin_rmse":
                    gradient_rmse,

                "win_accuracy":
                    gradient_accuracy,

                "brier":
                    gradient_brier,

                "logloss":
                    gradient_logloss,

                "train_games":
                    len(
                        train
                    ),

                "validation_games":
                    len(
                        validation
                    )
            }
        )


        # ====================================================
        # SAVE GAME-LEVEL PREDICTIONS
        #
        # These are important because later we can compare
        # errors game-by-game rather than only averages.
        # ====================================================

        for position in range(
            len(
                validation
            )
        ):

            game = validation.iloc[
                position
            ]


            prediction_rows.append(
                {
                    "game_id":
                        game[
                            "game_id"
                        ],

                    "season":
                        validation_year,

                    "feature_set":
                        feature_name,

                    "actual_margin":
                        y_validation_margin.iloc[
                            position
                        ],

                    "actual_home_win":
                        y_validation_win.iloc[
                            position
                        ],

                    "ridge_margin":
                        ridge_margin_prediction[
                            position
                        ],

                    "logistic_home_win_probability":
                        logistic_win_probability[
                            position
                        ],

                    "gradient_margin":
                        gradient_margin_prediction[
                            position
                        ],

                    "gradient_home_win_probability":
                        gradient_win_probability[
                            position
                        ]
                }
            )


    # ========================================================
    # SPORTSBOOK BENCHMARK
    # ========================================================

    if (
        "benchmark_spread"
        in validation.columns
    ):

        market_mask = (
            validation[
                "benchmark_spread"
            ]
            .notna()
        )


        market_actual_margin = (
            validation.loc[
                market_mask,
                margin_target
            ]
        )


        # ----------------------------------------------------
        # Stored spread convention:
        #
        # -7 = home favored by seven.
        #
        # Therefore predicted home margin = +7.
        # ----------------------------------------------------

        market_margin_prediction = (

            -validation.loc[
                market_mask,
                "benchmark_spread"
            ]
        )


        (
            market_mae,
            market_rmse
        ) = calculate_margin_metrics(

            market_actual_margin,

            market_margin_prediction
        )


        # ----------------------------------------------------
        # FAVORITE WIN ACCURACY
        #
        # This is NOT ATS betting performance.
        #
        # It simply asks whether the market favorite
        # won the game outright.
        # ----------------------------------------------------

        favorite_mask = (

            market_mask

            &

            (
                validation[
                    "benchmark_spread"
                ]
                != 0
            )
        )


        market_home_win_prediction = (

            validation.loc[
                favorite_mask,
                "benchmark_spread"
            ]

            < 0
        ).astype(
            int
        )


        market_accuracy = accuracy_score(

            validation.loc[
                favorite_mask,
                win_target
            ].astype(
                int
            ),

            market_home_win_prediction
        )


        print("\n")

        print(
            "SPORTSBOOK BENCHMARK"
        )

        print(
            "-" * 100
        )


        print(
            f"Margin:             "
            f"MAE {market_mae:.3f} | "
            f"RMSE {market_rmse:.3f}"
        )


        print(
            f"Favorite accuracy:  "
            f"{market_accuracy:.3f}"
        )


        results.append(
            {
                "validation_year":
                    validation_year,

                "feature_set":
                    "Sportsbook",

                "model":
                    "Market",

                "margin_mae":
                    market_mae,

                "margin_rmse":
                    market_rmse,

                "win_accuracy":
                    market_accuracy,

                "brier":
                    np.nan,

                "logloss":
                    np.nan,

                "train_games":
                    np.nan,

                "validation_games":
                    int(
                        market_mask.sum()
                    )
            }
        )


# ============================================================
# CREATE OUTPUT TABLES
# ============================================================

results = pd.DataFrame(
    results
)


predictions = pd.DataFrame(
    prediction_rows
)


os.makedirs(
    "data/processed",
    exist_ok=True
)


results.to_csv(
    results_file,
    index=False
)


predictions.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# AVERAGE ROLLING VALIDATION RESULTS
#
# Each validation season receives equal weight,
# matching our earlier rolling validation summaries.
# ============================================================

print("\n\n")

print(
    "=" * 100
)

print(
    "AVERAGE ROLLING VALIDATION RESULTS"
)

print(
    "=" * 100
)


average_results = (

    results

    .groupby(
        [
            "feature_set",
            "model"
        ],
        dropna=False
    )

    .agg(

        margin_mae=(
            "margin_mae",
            "mean"
        ),

        margin_rmse=(
            "margin_rmse",
            "mean"
        ),

        win_accuracy=(
            "win_accuracy",
            "mean"
        ),

        brier=(
            "brier",
            "mean"
        ),

        logloss=(
            "logloss",
            "mean"
        )
    )

    .reset_index()
)


average_results = (
    average_results

    .sort_values(
        [
            "margin_mae",
            "brier"
        ],

        na_position="last"
    )
)


print(
    average_results.to_string(
        index=False,
        formatters={
            "margin_mae":
                lambda value:
                    f"{value:.3f}",

            "margin_rmse":
                lambda value:
                    f"{value:.3f}",

            "win_accuracy":
                lambda value:
                    f"{value:.3f}",

            "brier":
                lambda value:
                    (
                        f"{value:.3f}"

                        if pd.notna(
                            value
                        )

                        else "-"
                    ),

            "logloss":
                lambda value:
                    (
                        f"{value:.3f}"

                        if pd.notna(
                            value
                        )

                        else "-"
                    )
        }
    )
)


# ============================================================
# DIRECT EFFECT OF ADDING PPA
#
# Negative MAE/Brier/LogLoss changes are improvements.
# ============================================================

print("\n")

print(
    "=" * 100
)

print(
    "DIRECT EFFECT OF ADDING PPA"
)

print(
    "=" * 100
)


for model_name in [
    "Linear",
    "Gradient Boosting"
]:

    v1 = average_results[
        (
            average_results[
                "feature_set"
            ]
            ==
            "V1 Difference"
        )

        &

        (
            average_results[
                "model"
            ]
            ==
            model_name
        )
    ]


    v2 = average_results[
        (
            average_results[
                "feature_set"
            ]
            ==
            "V1 + PPA"
        )

        &

        (
            average_results[
                "model"
            ]
            ==
            model_name
        )
    ]


    if (
        len(
            v1
        ) == 1

        and

        len(
            v2
        ) == 1
    ):

        mae_change = (

            v2[
                "margin_mae"
            ].iloc[
                0
            ]

            -

            v1[
                "margin_mae"
            ].iloc[
                0
            ]
        )


        rmse_change = (

            v2[
                "margin_rmse"
            ].iloc[
                0
            ]

            -

            v1[
                "margin_rmse"
            ].iloc[
                0
            ]
        )


        accuracy_change = (

            v2[
                "win_accuracy"
            ].iloc[
                0
            ]

            -

            v1[
                "win_accuracy"
            ].iloc[
                0
            ]
        )


        brier_change = (

            v2[
                "brier"
            ].iloc[
                0
            ]

            -

            v1[
                "brier"
            ].iloc[
                0
            ]
        )


        logloss_change = (

            v2[
                "logloss"
            ].iloc[
                0
            ]

            -

            v1[
                "logloss"
            ].iloc[
                0
            ]
        )


        print("\n")

        print(
            model_name
        )


        print(
            f"MAE change:       "
            f"{mae_change:+.3f}"
        )


        print(
            f"RMSE change:      "
            f"{rmse_change:+.3f}"
        )


        print(
            f"Accuracy change:  "
            f"{accuracy_change:+.4f}"
        )


        print(
            f"Brier change:     "
            f"{brier_change:+.4f}"
        )


        print(
            f"LogLoss change:   "
            f"{logloss_change:+.4f}"
        )


        if mae_change < 0:

            print(
                "Margin prediction improved."
            )

        elif mae_change > 0:

            print(
                "Margin prediction worsened."
            )

        else:

            print(
                "Margin prediction unchanged."
            )


        if brier_change < 0:

            print(
                "Win probabilities improved."
            )

        elif brier_change > 0:

            print(
                "Win probabilities worsened."
            )

        else:

            print(
                "Win probabilities unchanged."
            )


# ============================================================
# FINISH
# ============================================================

print("\n")

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
    results_file
)


print(
    predictions_file
)


print("\n")

print(
    "2025 was not trained on, validated on, "
    "or scored."
)