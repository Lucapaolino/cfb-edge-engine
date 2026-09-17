import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    Ridge,
    LogisticRegression
)

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
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

output_file = (
    "data/processed/"
    "nonlinear_validation_results_v1.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv(
    feature_file
)


print("NONLINEAR MODEL TEST V1")
print("=" * 100)

print(
    f"Rows loaded: "
    f"{len(data):,}"
)


# --------------------------------------------------
# FEATURE HELPERS
# --------------------------------------------------

def existing(columns):

    return [
        column
        for column in columns
        if column in data.columns
    ]


# --------------------------------------------------
# ELO FEATURES
# --------------------------------------------------

elo_features = existing(
    [
        "pregame_elo_diff",
        "neutral_site"
    ]
)


# --------------------------------------------------
# DIFFERENCE FEATURES
# --------------------------------------------------

difference_features = [

    column

    for column in data.columns

    if (
        column.startswith(
            "diff_pregame_"
        )
        and
        pd.api.types.is_numeric_dtype(
            data[column]
        )
    )
]


difference_features += existing(
    [
        "talent_diff",

        "returning_ppa_diff",
        "returning_passing_ppa_diff",
        "returning_receiving_ppa_diff",
        "returning_rushing_ppa_diff"
    ]
)


# Remove duplicates while preserving order
difference_features = list(
    dict.fromkeys(
        difference_features
    )
)


model_features = list(
    dict.fromkeys(
        elo_features
        + difference_features
    )
)


print(
    f"Model features: "
    f"{len(model_features)}"
)


# --------------------------------------------------
# LINEAR MODEL BUILDERS
# --------------------------------------------------

def ridge_model():

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


def logistic_model():

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
                    max_iter=5000
                )
            )
        ]
    )


# --------------------------------------------------
# NONLINEAR MODEL BUILDERS
#
# HistGradientBoosting:
#
# - models nonlinear relationships
# - models interactions between features
# - handles missing values natively
# - works well on tabular data
#
# We intentionally use conservative settings.
# --------------------------------------------------

def boosted_margin_model():

    return HistGradientBoostingRegressor(

        learning_rate=0.05,

        max_iter=250,

        max_leaf_nodes=15,

        min_samples_leaf=30,

        l2_regularization=5.0,

        early_stopping=True,

        random_state=42
    )


def boosted_win_model():

    return HistGradientBoostingClassifier(

        learning_rate=0.05,

        max_iter=250,

        max_leaf_nodes=15,

        min_samples_leaf=30,

        l2_regularization=5.0,

        early_stopping=True,

        random_state=42
    )


# --------------------------------------------------
# VALIDATION YEARS
#
# 2025 IS STILL EXCLUDED.
# --------------------------------------------------

validation_years = [
    2021,
    2022,
    2023,
    2024
]


results = []


# --------------------------------------------------
# ROLLING VALIDATION
# --------------------------------------------------

for validation_year in validation_years:

    print("\n")
    print("=" * 100)

    print(
        f"VALIDATION SEASON: "
        f"{validation_year}"
    )

    print("=" * 100)


    # Expanding history:
    #
    # Only seasons BEFORE the validation season.
    train = data[
        data["season"]
        < validation_year
    ].copy()


    validation = data[
        data["season"]
        == validation_year
    ].copy()


    print(
        f"Training seasons: "
        f"{int(train['season'].min())}"
        f"-"
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


    # --------------------------------------------------
    # REMOVE FEATURES WITH NO TRAINING DATA
    # --------------------------------------------------

    usable_features = [

        column

        for column in model_features

        if (
            column in train.columns
            and
            train[column]
            .notna()
            .any()
        )
    ]


    X_train = (
        train[
            usable_features
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
            usable_features
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


    # ==================================================
    # LINEAR MARGIN MODEL
    # ==================================================

    linear_margin = (
        ridge_model()
    )


    linear_margin.fit(
        X_train,
        y_margin_train
    )


    linear_margin_prediction = (
        linear_margin.predict(
            X_validation
        )
    )


    # ==================================================
    # BOOSTED MARGIN MODEL
    # ==================================================

    boosted_margin = (
        boosted_margin_model()
    )


    boosted_margin.fit(
        X_train,
        y_margin_train
    )


    boosted_margin_prediction = (
        boosted_margin.predict(
            X_validation
        )
    )


    # ==================================================
    # LINEAR WIN MODEL
    # ==================================================

    linear_win = (
        logistic_model()
    )


    linear_win.fit(
        X_train,
        y_win_train
    )


    linear_win_probability = (
        linear_win.predict_proba(
            X_validation
        )[:, 1]
    )


    # ==================================================
    # BOOSTED WIN MODEL
    # ==================================================

    boosted_win = (
        boosted_win_model()
    )


    boosted_win.fit(
        X_train,
        y_win_train
    )


    boosted_win_probability = (
        boosted_win.predict_proba(
            X_validation
        )[:, 1]
    )


    # ==================================================
    # STORE BOTH MODEL TYPES
    # ==================================================

    model_outputs = {

        "Linear":

            (
                linear_margin_prediction,
                linear_win_probability
            ),

        "Gradient Boosting":

            (
                boosted_margin_prediction,
                boosted_win_probability
            )
    }


    for (
        model_name,
        (
            margin_prediction,
            win_probability
        )
    ) in model_outputs.items():


        margin_mae = (
            mean_absolute_error(
                y_margin_validation,
                margin_prediction
            )
        )


        margin_rmse = np.sqrt(
            mean_squared_error(
                y_margin_validation,
                margin_prediction
            )
        )


        win_prediction = (
            win_probability >= 0.5
        ).astype(int)


        accuracy = (
            accuracy_score(
                y_win_validation,
                win_prediction
            )
        )


        brier = (
            brier_score_loss(
                y_win_validation,
                win_probability
            )
        )


        loss = (
            log_loss(
                y_win_validation,
                win_probability
            )
        )


        results.append(
            {
                "validation_year":
                    validation_year,

                "model":
                    model_name,

                "feature_count":
                    len(
                        usable_features
                    ),

                "margin_mae":
                    margin_mae,

                "margin_rmse":
                    margin_rmse,

                "win_accuracy":
                    accuracy,

                "brier":
                    brier,

                "log_loss":
                    loss
            }
        )


        print(
            f"{model_name:<25} | "
            f"MAE {margin_mae:>6.3f} | "
            f"ACC {accuracy:>6.3f} | "
            f"Brier {brier:>6.3f} | "
            f"LogLoss {loss:>6.3f}"
        )


    # ==================================================
    # SPORTSBOOK BENCHMARK
    # ==================================================

    market_margin = (
        validation[
            "benchmark_market_home_margin"
        ]
    )


    market_mae = (
        mean_absolute_error(
            y_margin_validation,
            market_margin
        )
    )


    market_rmse = np.sqrt(
        mean_squared_error(
            y_margin_validation,
            market_margin
        )
    )


    market_mask = (
        market_margin.notna()
        &
        (
            market_margin != 0
        )
    )


    market_prediction = (
        market_margin[
            market_mask
        ] > 0
    ).astype(int)


    market_actual = (
        y_win_validation[
            market_mask
        ]
    )


    market_accuracy = (
        accuracy_score(
            market_actual,
            market_prediction
        )
    )


    results.append(
        {
            "validation_year":
                validation_year,

            "model":
                "Sportsbook",

            "feature_count":
                np.nan,

            "margin_mae":
                market_mae,

            "margin_rmse":
                market_rmse,

            "win_accuracy":
                market_accuracy,

            "brier":
                np.nan,

            "log_loss":
                np.nan
        }
    )


    print(
        f"{'Sportsbook':<25} | "
        f"MAE {market_mae:>6.3f} | "
        f"ACC {market_accuracy:>6.3f}"
    )


# --------------------------------------------------
# RESULTS DATAFRAME
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)


# --------------------------------------------------
# AVERAGE ACROSS YEARS
# --------------------------------------------------

average_results = (
    results_df
    .groupby(
        "model",
        as_index=False
    )
    .agg(
        folds=(
            "validation_year",
            "count"
        ),

        margin_mae=(
            "margin_mae",
            "mean"
        ),

        margin_rmse=(
            "margin_rmse",
            "mean"
        ),

        accuracy=(
            "win_accuracy",
            "mean"
        ),

        brier=(
            "brier",
            "mean"
        ),

        log_loss=(
            "log_loss",
            "mean"
        )
    )
)


average_results = (
    average_results
    .sort_values(
        by="margin_mae"
    )
)


# --------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------

print("\n\n")
print("NONLINEAR MODEL — AVERAGE RESULTS")
print("=" * 110)

print(
    f"{'Model':<30} "
    f"{'MAE':>9} "
    f"{'RMSE':>9} "
    f"{'Accuracy':>10} "
    f"{'Brier':>9} "
    f"{'LogLoss':>9}"
)

print("-" * 110)


for _, row in average_results.iterrows():

    if pd.notna(
        row["brier"]
    ):

        brier_text = (
            f"{row['brier']:.3f}"
        )

    else:

        brier_text = "-"


    if pd.notna(
        row["log_loss"]
    ):

        loss_text = (
            f"{row['log_loss']:.3f}"
        )

    else:

        loss_text = "-"


    print(
        f"{row['model']:<30} "
        f"{row['margin_mae']:>9.3f} "
        f"{row['margin_rmse']:>9.3f} "
        f"{row['accuracy']:>10.3f} "
        f"{brier_text:>9} "
        f"{loss_text:>9}"
    )


# --------------------------------------------------
# DIRECT LINEAR VS BOOSTING DIFFERENCE
# --------------------------------------------------

linear_row = (
    average_results[
        average_results[
            "model"
        ] == "Linear"
    ]
    .iloc[0]
)


boosted_row = (
    average_results[
        average_results[
            "model"
        ] == "Gradient Boosting"
    ]
    .iloc[0]
)


print("\n")
print("BOOSTING VS LINEAR")
print("=" * 100)


mae_change = (
    boosted_row[
        "margin_mae"
    ]
    - linear_row[
        "margin_mae"
    ]
)


accuracy_change = (
    boosted_row[
        "accuracy"
    ]
    - linear_row[
        "accuracy"
    ]
)


brier_change = (
    boosted_row[
        "brier"
    ]
    - linear_row[
        "brier"
    ]
)


loss_change = (
    boosted_row[
        "log_loss"
    ]
    - linear_row[
        "log_loss"
    ]
)


print(
    f"Margin MAE change: "
    f"{mae_change:+.3f}"
)

print(
    f"Accuracy change:   "
    f"{accuracy_change:+.3f}"
)

print(
    f"Brier change:      "
    f"{brier_change:+.3f}"
)

print(
    f"LogLoss change:    "
    f"{loss_change:+.3f}"
)


print(
    "\nNegative MAE/Brier/LogLoss change "
    "is better."
)

print(
    "Positive Accuracy change is better."
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

results_df.to_csv(
    output_file,
    index=False
)


print("\n")
print("RESULT FILE")
print("=" * 100)

print(
    f"Saved to: "
    f"{output_file}"
)


print("\n")
print(
    "2025 reserved test set was not "
    "evaluated."
)