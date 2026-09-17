import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

output_file = (
    "data/processed/"
    "training_window_results_v1.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv(
    feature_file
)


print("TRAINING WINDOW TEST V1")
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
# ELO BASE
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
# PIPELINES
# --------------------------------------------------

def regression_model():

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


def classification_model():

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
# TRAINING WINDOWS
#
# None = every season before validation year.
# --------------------------------------------------

training_windows = {

    "Expanding":
        None,

    "Previous 4 seasons":
        4,

    "Previous 3 seasons":
        3,

    "Previous 2 seasons":
        2,

    "Previous season only":
        1
}


# --------------------------------------------------
# VALIDATION YEARS
#
# 2025 remains completely excluded.
# --------------------------------------------------

validation_years = [
    2021,
    2022,
    2023,
    2024
]


results = []


# --------------------------------------------------
# RUN TESTS
# --------------------------------------------------

for validation_year in validation_years:

    print("\n")
    print("=" * 100)

    print(
        f"VALIDATION YEAR: "
        f"{validation_year}"
    )

    print("=" * 100)


    validation = data[
        data["season"]
        == validation_year
    ].copy()


    y_margin_validation = (
        validation[
            "target_home_margin"
        ]
        .astype(float)
    )


    y_win_validation = (
        validation[
            "target_home_win"
        ]
        .astype(int)
    )


    for (
        window_name,
        window_size
    ) in training_windows.items():


        # --------------------------------------------------
        # CHOOSE TRAINING SEASONS
        # --------------------------------------------------

        if window_size is None:

            train = data[
                data["season"]
                < validation_year
            ].copy()

        else:

            first_training_year = max(
                2018,
                validation_year
                - window_size
            )


            train = data[
                (
                    data["season"]
                    >= first_training_year
                )
                &
                (
                    data["season"]
                    < validation_year
                )
            ].copy()


        if len(train) == 0:
            continue


        # --------------------------------------------------
        # REMOVE FEATURES WITH NO TRAINING VALUES
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


        y_margin_train = (
            train[
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


        # --------------------------------------------------
        # MARGIN MODEL
        # --------------------------------------------------

        margin_model = (
            regression_model()
        )


        margin_model.fit(
            X_train,
            y_margin_train
        )


        margin_prediction = (
            margin_model.predict(
                X_validation
            )
        )


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


        # --------------------------------------------------
        # WIN MODEL
        # --------------------------------------------------

        win_model = (
            classification_model()
        )


        win_model.fit(
            X_train,
            y_win_train
        )


        win_probability = (
            win_model.predict_proba(
                X_validation
            )[:, 1]
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


        training_min = int(
            train["season"].min()
        )


        training_max = int(
            train["season"].max()
        )


        print(
            f"{window_name:<24} | "
            f"{training_min}-{training_max} | "
            f"{len(train):>5,} games | "
            f"MAE {margin_mae:>6.3f} | "
            f"ACC {accuracy:>6.3f} | "
            f"Brier {brier:>6.3f} | "
            f"LogLoss {loss:>6.3f}"
        )


        results.append(
            {
                "validation_year":
                    validation_year,

                "training_window":
                    window_name,

                "training_start":
                    training_min,

                "training_end":
                    training_max,

                "training_games":
                    len(train),

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


# --------------------------------------------------
# RESULTS DATAFRAME
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)


# --------------------------------------------------
# AVERAGES BY WINDOW
# --------------------------------------------------

average_results = (
    results_df
    .groupby(
        "training_window",
        as_index=False
    )
    .agg(
        folds=(
            "validation_year",
            "count"
        ),

        average_training_games=(
            "training_games",
            "mean"
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
# PRINT AVERAGES
# --------------------------------------------------

print("\n\n")
print("TRAINING WINDOW — AVERAGE RESULTS")
print("=" * 110)

print(
    f"{'Training window':<26} "
    f"{'Avg Games':>10} "
    f"{'MAE':>9} "
    f"{'RMSE':>9} "
    f"{'Accuracy':>10} "
    f"{'Brier':>9} "
    f"{'LogLoss':>9}"
)

print("-" * 110)


for _, row in average_results.iterrows():

    print(
        f"{row['training_window']:<26} "
        f"{row['average_training_games']:>10.0f} "
        f"{row['margin_mae']:>9.3f} "
        f"{row['margin_rmse']:>9.3f} "
        f"{row['accuracy']:>10.3f} "
        f"{row['brier']:>9.3f} "
        f"{row['log_loss']:>9.3f}"
    )


# --------------------------------------------------
# BEST WINDOW BY METRIC
# --------------------------------------------------

best_mae = (
    average_results
    .sort_values(
        "margin_mae"
    )
    .iloc[0]
)


best_brier = (
    average_results
    .sort_values(
        "brier"
    )
    .iloc[0]
)


best_logloss = (
    average_results
    .sort_values(
        "log_loss"
    )
    .iloc[0]
)


print("\n")
print("BEST AVERAGE WINDOWS")
print("=" * 100)

print(
    f"Margin MAE: "
    f"{best_mae['training_window']} "
    f"({best_mae['margin_mae']:.3f})"
)

print(
    f"Brier:     "
    f"{best_brier['training_window']} "
    f"({best_brier['brier']:.3f})"
)

print(
    f"Log loss:  "
    f"{best_logloss['training_window']} "
    f"({best_logloss['log_loss']:.3f})"
)


# --------------------------------------------------
# SAVE RESULTS
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