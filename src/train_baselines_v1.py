import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import (
    DummyRegressor,
    DummyClassifier
)

from sklearn.linear_model import (
    Ridge,
    LogisticRegression
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    brier_score_loss,
    log_loss
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

feature_file = (
    "data/processed/"
    "features_v1.csv"
)

prediction_file = (
    "data/processed/"
    "baseline_validation_predictions_v1.csv"
)


print("BASELINE MODEL V1")
print("=" * 80)


# --------------------------------------------------
# LOAD FEATURE DATA
# --------------------------------------------------

data = pd.read_csv(
    feature_file
)


print(
    f"Feature rows loaded: "
    f"{len(data):,}"
)

print(
    f"Feature columns loaded: "
    f"{len(data.columns):,}"
)


# --------------------------------------------------
# SPLITS
#
# IMPORTANT:
#
# 2018-2023 = training
# 2024      = validation
# 2025      = reserved test
#
# We intentionally DO NOT evaluate 2025 here.
# --------------------------------------------------

train = data[
    data["dataset_split"] == "train"
].copy()


validation = data[
    data["dataset_split"] == "validation"
].copy()


print("\nDATA SPLITS")
print("-" * 80)

print(
    f"Training rows:    "
    f"{len(train):,}"
)

print(
    f"Validation rows:  "
    f"{len(validation):,}"
)

print(
    "2025 reserved test rows are "
    "not evaluated by this script."
)


# --------------------------------------------------
# TARGETS
# --------------------------------------------------

y_margin_train = (
    train[
        "target_home_margin"
    ]
    .astype(float)
)


y_margin_validation = (
    validation[
        "target_home_margin"
    ]
    .astype(float)
)


y_win_train = (
    train[
        "target_home_win"
    ]
    .astype(int)
)


y_win_validation = (
    validation[
        "target_home_win"
    ]
    .astype(int)
)


# --------------------------------------------------
# SELECT SAFE MODEL FEATURES
#
# This is one of the most important sections.
#
# We explicitly block:
#
# - final scores
# - targets
# - market benchmark information
# - IDs
# - team names
#
# This prevents accidental leakage.
# --------------------------------------------------

blocked_columns = {

    "game_id",
    "season",
    "start_date",

    "home_team",
    "away_team",

    "home_points",
    "away_points",

    "dataset_split"
}


feature_columns = []


for column in data.columns:

    # Explicitly blocked
    if column in blocked_columns:
        continue

    # Anything beginning with target_ is
    # information from after the game.
    if column.startswith("target_"):
        continue

    # Sportsbook information is intentionally
    # excluded from the MARKET-BLIND model.
    if column.startswith("benchmark_"):
        continue

    # Only numeric inputs for V1.
    if not pd.api.types.is_numeric_dtype(
        data[column]
    ):
        continue

    feature_columns.append(
        column
    )


# --------------------------------------------------
# REMOVE FEATURES THAT HAVE NO TRAINING DATA
# --------------------------------------------------

usable_feature_columns = []


for column in feature_columns:

    if train[column].notna().any():

        usable_feature_columns.append(
            column
        )


removed_empty_features = (
    len(feature_columns)
    - len(usable_feature_columns)
)


feature_columns = (
    usable_feature_columns
)


print("\nFEATURE SELECTION")
print("-" * 80)

print(
    f"Safe numeric features: "
    f"{len(feature_columns)}"
)

print(
    f"All-empty training features removed: "
    f"{removed_empty_features}"
)


# --------------------------------------------------
# MODEL MATRICES
# --------------------------------------------------

X_train = (
    train[
        feature_columns
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
)


X_validation = (
    validation[
        feature_columns
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
)


# --------------------------------------------------
# MISSINGNESS REPORT
# --------------------------------------------------

missing_rates = (
    X_train
    .isna()
    .mean()
    .sort_values(
        ascending=False
    )
)


print("\nHIGHEST TRAINING MISSINGNESS")
print("-" * 80)


for column, rate in (
    missing_rates
    .head(10)
    .items()
):

    print(
        f"{column:<55} "
        f"{rate * 100:6.2f}%"
    )


# --------------------------------------------------
# ELO-ONLY FEATURE SET
#
# This gives us a simple strength-rating baseline.
# --------------------------------------------------

elo_features = [
    "pregame_elo_diff",
    "neutral_site"
]


elo_features = [
    column
    for column in elo_features
    if column in data.columns
]


X_elo_train = (
    train[
        elo_features
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
)


X_elo_validation = (
    validation[
        elo_features
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
)


# --------------------------------------------------
# PIPELINE BUILDERS
#
# Imputer:
# fills missing values using TRAINING medians.
#
# Missing indicators:
# tell the model when a value was unavailable.
#
# StandardScaler:
# puts differently-scaled features onto comparable
# numeric scales.
# --------------------------------------------------

def regression_pipeline(model):

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
                model
            )
        ]
    )


def classification_pipeline(model):

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
                model
            )
        ]
    )


# ==================================================
# MARGIN MODELS
# ==================================================

print("\nTRAINING MARGIN MODELS")
print("-" * 80)


# --------------------------------------------------
# DUMMY MEAN BASELINE
# --------------------------------------------------

mean_model = DummyRegressor(
    strategy="mean"
)


mean_model.fit(
    X_train,
    y_margin_train
)


mean_margin_prediction = (
    mean_model.predict(
        X_validation
    )
)


print(
    "Mean baseline trained."
)


# --------------------------------------------------
# ELO-ONLY RIDGE
# --------------------------------------------------

elo_margin_model = (
    regression_pipeline(
        Ridge(
            alpha=10.0
        )
    )
)


elo_margin_model.fit(
    X_elo_train,
    y_margin_train
)


elo_margin_prediction = (
    elo_margin_model.predict(
        X_elo_validation
    )
)


print(
    "Elo-only Ridge trained."
)


# --------------------------------------------------
# FULL FOOTBALL RIDGE
# --------------------------------------------------

football_margin_model = (
    regression_pipeline(
        Ridge(
            alpha=10.0
        )
    )
)


football_margin_model.fit(
    X_train,
    y_margin_train
)


football_margin_prediction = (
    football_margin_model.predict(
        X_validation
    )
)


print(
    "Full football Ridge trained."
)


# ==================================================
# WIN-PROBABILITY MODELS
# ==================================================

print("\nTRAINING WIN MODELS")
print("-" * 80)


# --------------------------------------------------
# PRIOR-PROBABILITY BASELINE
# --------------------------------------------------

prior_model = DummyClassifier(
    strategy="prior"
)


prior_model.fit(
    X_train,
    y_win_train
)


prior_probability = (
    prior_model.predict_proba(
        X_validation
    )[:, 1]
)


prior_prediction = (
    prior_probability >= 0.5
).astype(int)


print(
    "Prior-probability baseline trained."
)


# --------------------------------------------------
# ELO-ONLY LOGISTIC REGRESSION
# --------------------------------------------------

elo_win_model = (
    classification_pipeline(
        LogisticRegression(
            C=1.0,
            max_iter=5000
        )
    )
)


elo_win_model.fit(
    X_elo_train,
    y_win_train
)


elo_win_probability = (
    elo_win_model.predict_proba(
        X_elo_validation
    )[:, 1]
)


elo_win_prediction = (
    elo_win_probability >= 0.5
).astype(int)


print(
    "Elo-only logistic regression trained."
)


# --------------------------------------------------
# FULL FOOTBALL LOGISTIC REGRESSION
# --------------------------------------------------

football_win_model = (
    classification_pipeline(
        LogisticRegression(
            C=1.0,
            max_iter=5000
        )
    )
)


football_win_model.fit(
    X_train,
    y_win_train
)


football_win_probability = (
    football_win_model.predict_proba(
        X_validation
    )[:, 1]
)


football_win_prediction = (
    football_win_probability >= 0.5
).astype(int)


print(
    "Full football logistic regression trained."
)


# ==================================================
# METRIC HELPERS
# ==================================================

def margin_metrics(
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

    return mae, rmse


def win_metrics(
    actual,
    probability
):

    prediction = (
        probability >= 0.5
    ).astype(int)

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
        probability
    )

    return (
        accuracy,
        brier,
        loss
    )


# ==================================================
# MARGIN EVALUATION
# ==================================================

mean_mae, mean_rmse = (
    margin_metrics(
        y_margin_validation,
        mean_margin_prediction
    )
)


elo_mae, elo_rmse = (
    margin_metrics(
        y_margin_validation,
        elo_margin_prediction
    )
)


football_mae, football_rmse = (
    margin_metrics(
        y_margin_validation,
        football_margin_prediction
    )
)


# --------------------------------------------------
# MARKET BENCHMARK
# --------------------------------------------------

market_mask = (
    validation[
        "benchmark_market_home_margin"
    ]
    .notna()
)


market_actual_margin = (
    y_margin_validation[
        market_mask
    ]
)


market_predicted_margin = (
    validation.loc[
        market_mask,
        "benchmark_market_home_margin"
    ]
)


market_mae, market_rmse = (
    margin_metrics(
        market_actual_margin,
        market_predicted_margin
    )
)


print("\n")
print("2024 VALIDATION — MARGIN")
print("=" * 80)

print(
    f"{'Model':<32} "
    f"{'MAE':>10} "
    f"{'RMSE':>10}"
)

print("-" * 80)

print(
    f"{'Mean baseline':<32} "
    f"{mean_mae:>10.3f} "
    f"{mean_rmse:>10.3f}"
)

print(
    f"{'Elo-only Ridge':<32} "
    f"{elo_mae:>10.3f} "
    f"{elo_rmse:>10.3f}"
)

print(
    f"{'Full football Ridge':<32} "
    f"{football_mae:>10.3f} "
    f"{football_rmse:>10.3f}"
)

print(
    f"{'Sportsbook spread':<32} "
    f"{market_mae:>10.3f} "
    f"{market_rmse:>10.3f}"
)


# ==================================================
# WIN-PROBABILITY EVALUATION
# ==================================================

prior_accuracy, prior_brier, prior_logloss = (
    win_metrics(
        y_win_validation,
        prior_probability
    )
)


elo_accuracy, elo_brier, elo_logloss = (
    win_metrics(
        y_win_validation,
        elo_win_probability
    )
)


(
    football_accuracy,
    football_brier,
    football_logloss
) = (
    win_metrics(
        y_win_validation,
        football_win_probability
    )
)


print("\n")
print("2024 VALIDATION — WIN PROBABILITY")
print("=" * 80)

print(
    f"{'Model':<32} "
    f"{'Accuracy':>10} "
    f"{'Brier':>10} "
    f"{'LogLoss':>10}"
)

print("-" * 80)

print(
    f"{'Prior baseline':<32} "
    f"{prior_accuracy:>10.3f} "
    f"{prior_brier:>10.3f} "
    f"{prior_logloss:>10.3f}"
)

print(
    f"{'Elo-only Logistic':<32} "
    f"{elo_accuracy:>10.3f} "
    f"{elo_brier:>10.3f} "
    f"{elo_logloss:>10.3f}"
)

print(
    f"{'Full football Logistic':<32} "
    f"{football_accuracy:>10.3f} "
    f"{football_brier:>10.3f} "
    f"{football_logloss:>10.3f}"
)


# ==================================================
# MARKET FAVORITE WIN ACCURACY
#
# We can compare favorite-vs-underdog accuracy,
# but NOT Brier score yet because a point spread
# is not itself a probability.
# ==================================================

market_win_mask = (
    validation[
        "benchmark_market_home_margin"
    ]
    .notna()
    &
    (
        validation[
            "benchmark_market_home_margin"
        ] != 0
    )
)


market_home_margin = (
    validation.loc[
        market_win_mask,
        "benchmark_market_home_margin"
    ]
)


market_win_prediction = (
    market_home_margin > 0
).astype(int)


market_actual_win = (
    validation.loc[
        market_win_mask,
        "target_home_win"
    ]
    .astype(int)
)


market_win_accuracy = (
    accuracy_score(
        market_actual_win,
        market_win_prediction
    )
)


print("\nMARKET FAVORITE")
print("-" * 80)

print(
    f"Games excluding pick'em lines: "
    f"{len(market_actual_win):,}"
)

print(
    f"Sportsbook favorite accuracy:   "
    f"{market_win_accuracy:.3f}"
)


# ==================================================
# SAVE VALIDATION PREDICTIONS
# ==================================================

predictions = validation[
    [
        "game_id",
        "season",
        "week",
        "start_date",

        "away_team",
        "home_team",

        "target_home_margin",
        "target_home_win",

        "benchmark_spread",
        "benchmark_market_home_margin"
    ]
].copy()


predictions[
    "mean_predicted_margin"
] = (
    mean_margin_prediction
)


predictions[
    "elo_predicted_margin"
] = (
    elo_margin_prediction
)


predictions[
    "football_predicted_margin"
] = (
    football_margin_prediction
)


predictions[
    "prior_home_win_probability"
] = (
    prior_probability
)


predictions[
    "elo_home_win_probability"
] = (
    elo_win_probability
)


predictions[
    "football_home_win_probability"
] = (
    football_win_probability
)


predictions.to_csv(
    prediction_file,
    index=False
)


print("\n")
print("VALIDATION PREDICTIONS")
print("=" * 80)

print(
    f"Saved to: "
    f"{prediction_file}"
)


# ==================================================
# SHOW SAMPLE PREDICTIONS
# ==================================================

print("\nSAMPLE 2024 PREDICTIONS")
print("-" * 80)


sample_columns = [

    "week",
    "away_team",
    "home_team",

    "target_home_margin",

    "benchmark_market_home_margin",

    "elo_predicted_margin",
    "football_predicted_margin",

    "elo_home_win_probability",
    "football_home_win_probability"
]


print(
    predictions[
        sample_columns
    ]
    .head(12)
    .to_string(
        index=False
    )
)


print("\n")
print(
    "2025 reserved test set was not "
    "evaluated."
)