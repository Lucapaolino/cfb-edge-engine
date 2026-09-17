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
    "rolling_validation_results_v1.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv(
    feature_file
)


print("ROLLING VALIDATION V1")
print("=" * 100)

print(
    f"Rows loaded: "
    f"{len(data):,}"
)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def existing(columns):

    return [
        column
        for column in columns
        if column in data.columns
    ]


def numeric_columns_starting(prefix):

    return [
        column
        for column in data.columns

        if (
            column.startswith(prefix)
            and
            pd.api.types.is_numeric_dtype(
                data[column]
            )
        )
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
# RETURNING PRODUCTION
# --------------------------------------------------

returning_features = []


for column in data.columns:

    if (
        "percent_ppa" in column
        or "percent_passing_ppa" in column
        or "percent_receiving_ppa" in column
        or "percent_rushing_ppa" in column
        or "returning_ppa_diff" in column
        or "returning_passing_ppa_diff" in column
        or "returning_receiving_ppa_diff" in column
        or "returning_rushing_ppa_diff" in column
        or column.endswith(
            "returning_available"
        )
    ):

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):

            returning_features.append(
                column
            )


returning_features = sorted(
    set(returning_features)
)


# --------------------------------------------------
# DIFFERENCE FEATURES
# --------------------------------------------------

difference_features = (
    numeric_columns_starting(
        "diff_pregame_"
    )
)


difference_features += existing(
    [
        "talent_diff",

        "returning_ppa_diff",
        "returning_passing_ppa_diff",
        "returning_receiving_ppa_diff",
        "returning_rushing_ppa_diff"
    ]
)


difference_features = sorted(
    set(difference_features)
)


# --------------------------------------------------
# FULL FEATURE SET
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


full_features = []


for column in data.columns:

    if column in blocked_columns:
        continue

    if column.startswith(
        "target_"
    ):
        continue

    if column.startswith(
        "benchmark_"
    ):
        continue

    if not pd.api.types.is_numeric_dtype(
        data[column]
    ):
        continue

    full_features.append(
        column
    )


# --------------------------------------------------
# FEATURE GROUPS
# --------------------------------------------------

feature_groups = {

    "Elo only":
        elo_features,

    "Elo + returning":
        list(
            dict.fromkeys(
                elo_features
                + returning_features
            )
        ),

    "Elo + difference features":
        list(
            dict.fromkeys(
                elo_features
                + difference_features
            )
        ),

    "Full football":
        full_features
}


# --------------------------------------------------
# MODELS
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
# ROLLING FOLDS
#
# Each validation season can only use seasons
# before it.
#
# 2025 is intentionally absent.
# --------------------------------------------------

validation_years = [
    2021,
    2022,
    2023,
    2024
]


results = []


for validation_year in validation_years:

    print("\n")
    print("=" * 100)

    print(
        f"VALIDATION SEASON: "
        f"{validation_year}"
    )

    print("=" * 100)


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
    # EACH FEATURE GROUP
    # --------------------------------------------------

    for (
        group_name,
        features
    ) in feature_groups.items():


        # Remove features that contain absolutely
        # no training observations for this fold.
        usable_features = [

            column

            for column in features

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
        # MARGIN
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
        # WIN PROBABILITY
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
            win_probability
            >= 0.5
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
                    group_name,

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
            f"{group_name:<30} | "
            f"MAE {margin_mae:>6.3f} | "
            f"ACC {accuracy:>6.3f} | "
            f"Brier {brier:>6.3f} | "
            f"LogLoss {loss:>6.3f}"
        )


    # --------------------------------------------------
    # SPORTSBOOK MARGIN BENCHMARK
    # --------------------------------------------------

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


    # --------------------------------------------------
    # SPORTSBOOK FAVORITE ACCURACY
    # --------------------------------------------------

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
        f"{'Sportsbook':<30} | "
        f"MAE {market_mae:>6.3f} | "
        f"ACC {market_accuracy:>6.3f}"
    )


# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)


# --------------------------------------------------
# AVERAGE ACROSS VALIDATION YEARS
# --------------------------------------------------

average_results = (
    results_df
    .groupby(
        "model",
        as_index=False
    )
    .agg(
        seasons_tested=(
            "validation_year",
            "count"
        ),

        average_margin_mae=(
            "margin_mae",
            "mean"
        ),

        average_margin_rmse=(
            "margin_rmse",
            "mean"
        ),

        average_win_accuracy=(
            "win_accuracy",
            "mean"
        ),

        average_brier=(
            "brier",
            "mean"
        ),

        average_log_loss=(
            "log_loss",
            "mean"
        )
    )
)


# --------------------------------------------------
# PRINT AVERAGE RESULTS
# --------------------------------------------------

print("\n\n")
print("ROLLING VALIDATION — AVERAGE RESULTS")
print("=" * 110)


average_results = (
    average_results
    .sort_values(
        by="average_margin_mae"
    )
)


print(
    f"{'Model':<32} "
    f"{'MAE':>9} "
    f"{'RMSE':>9} "
    f"{'Accuracy':>10} "
    f"{'Brier':>9} "
    f"{'LogLoss':>9}"
)

print("-" * 110)


for _, row in average_results.iterrows():

    brier_text = (
        f"{row['average_brier']:.3f}"
        if pd.notna(
            row[
                "average_brier"
            ]
        )
        else "-"
    )


    loss_text = (
        f"{row['average_log_loss']:.3f}"
        if pd.notna(
            row[
                "average_log_loss"
            ]
        )
        else "-"
    )


    print(
        f"{row['model']:<32} "
        f"{row['average_margin_mae']:>9.3f} "
        f"{row['average_margin_rmse']:>9.3f} "
        f"{row['average_win_accuracy']:>10.3f} "
        f"{brier_text:>9} "
        f"{loss_text:>9}"
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