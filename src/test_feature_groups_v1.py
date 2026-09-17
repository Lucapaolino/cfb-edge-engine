import os
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
    "feature_group_results_v1.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv(
    feature_file
)


train = data[
    data["dataset_split"] == "train"
].copy()


validation = data[
    data["dataset_split"] == "validation"
].copy()


print("FEATURE GROUP ABLATION V1")
print("=" * 90)

print(
    f"Training games:   "
    f"{len(train):,}"
)

print(
    f"Validation games: "
    f"{len(validation):,}"
)

print(
    "2025 holdout is not evaluated."
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
# HELPERS
# --------------------------------------------------

def existing(columns):

    return [
        column
        for column in columns
        if column in data.columns
    ]


def columns_containing(
    required_text,
    excluded_text=None
):

    results = []

    for column in data.columns:

        if required_text not in column:
            continue

        if excluded_text is not None:

            if excluded_text in column:
                continue

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):

            continue

        results.append(
            column
        )

    return results


def difference_columns_starting(prefix):

    return [
        column
        for column in data.columns
        if column.startswith(prefix)
        and pd.api.types.is_numeric_dtype(
            data[column]
        )
    ]


# --------------------------------------------------
# EXACT ELO BASELINE
#
# Same two inputs used by our previous model.
# --------------------------------------------------

elo_base = existing(
    [
        "pregame_elo_diff",
        "neutral_site"
    ]
)


# --------------------------------------------------
# SCHEDULE CONTEXT
# --------------------------------------------------

schedule_context = existing(
    [
        "week",
        "conference_game"
    ]
)


# --------------------------------------------------
# TALENT FEATURES
# --------------------------------------------------

talent_features = existing(
    [
        "home_talent",
        "away_talent",
        "talent_diff",

        "home_talent_available",
        "away_talent_available"
    ]
)


# --------------------------------------------------
# RETURNING PRODUCTION FEATURES
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
    set(
        returning_features
    )
)


# --------------------------------------------------
# SCORING / RESULTS FORM
# --------------------------------------------------

scoring_terms = [
    "win",
    "points_for",
    "points_against",
    "margin"
]


scoring_features = []


for column in data.columns:

    if not column.startswith(
        "diff_pregame_"
    ):
        continue

    if any(
        term in column
        for term in scoring_terms
    ):

        scoring_features.append(
            column
        )


# --------------------------------------------------
# YARDAGE / EFFICIENCY FORM
# --------------------------------------------------

efficiency_terms = [
    "totalYards",
    "netPassingYards",
    "rushingYards",
    "yardsPerPass",
    "yardsPerRushAttempt"
]


efficiency_features = []


for column in data.columns:

    if not column.startswith(
        "diff_pregame_"
    ):
        continue

    if any(
        term in column
        for term in efficiency_terms
    ):

        efficiency_features.append(
            column
        )


# --------------------------------------------------
# TURNOVERS / DOWNS / PASS RUSH
# --------------------------------------------------

situational_terms = [
    "turnovers",
    "turnover_margin",
    "firstDowns",
    "thirdDownPct",
    "sacks",
    "TFL"
]


situational_features = []


for column in data.columns:

    if not column.startswith(
        "diff_pregame_"
    ):
        continue

    if any(
        term in column
        for term in situational_terms
    ):

        situational_features.append(
            column
        )


# --------------------------------------------------
# ALL SEASON-AVERAGE DIFFERENCES
# --------------------------------------------------

season_average_features = (
    difference_columns_starting(
        "diff_pregame_avg_"
    )
)


# --------------------------------------------------
# ALL LAST-3 DIFFERENCES
# --------------------------------------------------

last3_features = (
    difference_columns_starting(
        "diff_pregame_last3_"
    )
)


# --------------------------------------------------
# ALL DIFFERENCE FEATURES
#
# This is important because our full model contains
# home, away, AND difference versions of many stats.
#
# This test asks whether a simpler matchup-difference
# representation works better.
# --------------------------------------------------

all_difference_features = (
    difference_columns_starting(
        "diff_pregame_"
    )
)


all_difference_features += existing(
    [
        "talent_diff",

        "returning_ppa_diff",
        "returning_passing_ppa_diff",
        "returning_receiving_ppa_diff",
        "returning_rushing_ppa_diff"
    ]
)


all_difference_features = sorted(
    set(
        all_difference_features
    )
)


# --------------------------------------------------
# FULL SAFE FEATURE SET
#
# Recreate the same general feature selection used
# by train_baselines_v1.py.
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

    if not train[column].notna().any():
        continue

    full_features.append(
        column
    )


# --------------------------------------------------
# FEATURE GROUPS
#
# Each group begins with the same Elo baseline.
# --------------------------------------------------

feature_groups = {

    "Elo only":
        elo_base,

    "Elo + schedule context":
        elo_base
        + schedule_context,

    "Elo + talent":
        elo_base
        + talent_features,

    "Elo + returning":
        elo_base
        + returning_features,

    "Elo + preseason context":
        elo_base
        + talent_features
        + returning_features,

    "Elo + scoring form":
        elo_base
        + scoring_features,

    "Elo + efficiency form":
        elo_base
        + efficiency_features,

    "Elo + situational form":
        elo_base
        + situational_features,

    "Elo + season averages":
        elo_base
        + season_average_features,

    "Elo + last 3":
        elo_base
        + last3_features,

    "Elo + avg + last 3":
        elo_base
        + season_average_features
        + last3_features,

    "Elo + all difference features":
        elo_base
        + all_difference_features,

    "Full football model":
        full_features
}


# --------------------------------------------------
# REMOVE DUPLICATE FEATURE NAMES
# --------------------------------------------------

for group_name in feature_groups:

    feature_groups[
        group_name
    ] = list(
        dict.fromkeys(
            feature_groups[
                group_name
            ]
        )
    )


# --------------------------------------------------
# PIPELINES
# --------------------------------------------------

def make_regression_model():

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


def make_classification_model():

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
# RUN EACH FEATURE GROUP
# --------------------------------------------------

results = []


for (
    group_name,
    features
) in feature_groups.items():

    print(
        f"\nTesting: "
        f"{group_name}"
    )

    print(
        f"Features: "
        f"{len(features)}"
    )


    X_train = (
        train[
            features
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
            features
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
    # MARGIN MODEL
    # --------------------------------------------------

    margin_model = (
        make_regression_model()
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
        make_classification_model()
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


    accuracy = accuracy_score(
        y_win_validation,
        win_prediction
    )


    brier = brier_score_loss(
        y_win_validation,
        win_probability
    )


    loss = log_loss(
        y_win_validation,
        win_probability
    )


    results.append(
        {
            "feature_group":
                group_name,

            "feature_count":
                len(features),

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
# MARKET BENCHMARK
# --------------------------------------------------

market_actual = (
    validation[
        "target_home_margin"
    ]
)


market_prediction = (
    validation[
        "benchmark_market_home_margin"
    ]
)


market_mae = (
    mean_absolute_error(
        market_actual,
        market_prediction
    )
)


market_rmse = np.sqrt(
    mean_squared_error(
        market_actual,
        market_prediction
    )
)


# --------------------------------------------------
# MARKET FAVORITE ACCURACY
# --------------------------------------------------

market_mask = (
    validation[
        "benchmark_market_home_margin"
    ].notna()
    &
    (
        validation[
            "benchmark_market_home_margin"
        ] != 0
    )
)


market_favorite_prediction = (
    validation.loc[
        market_mask,
        "benchmark_market_home_margin"
    ] > 0
).astype(int)


market_actual_win = (
    validation.loc[
        market_mask,
        "target_home_win"
    ]
    .astype(int)
)


market_accuracy = (
    accuracy_score(
        market_actual_win,
        market_favorite_prediction
    )
)


# --------------------------------------------------
# PRINT MARGIN RESULTS
# --------------------------------------------------

margin_results = (
    results_df
    .sort_values(
        by="margin_mae"
    )
)


print("\n\n")
print("2024 FEATURE GROUPS — MARGIN")
print("=" * 90)

print(
    f"{'Feature group':<38} "
    f"{'Features':>8} "
    f"{'MAE':>10} "
    f"{'RMSE':>10}"
)

print("-" * 90)


for _, row in margin_results.iterrows():

    print(
        f"{row['feature_group']:<38} "
        f"{int(row['feature_count']):>8} "
        f"{row['margin_mae']:>10.3f} "
        f"{row['margin_rmse']:>10.3f}"
    )


print("-" * 90)

print(
    f"{'Sportsbook spread':<38} "
    f"{'-':>8} "
    f"{market_mae:>10.3f} "
    f"{market_rmse:>10.3f}"
)


# --------------------------------------------------
# PRINT WIN RESULTS
# --------------------------------------------------

win_results = (
    results_df
    .sort_values(
        by=[
            "brier",
            "log_loss"
        ]
    )
)


print("\n\n")
print("2024 FEATURE GROUPS — WIN PROBABILITY")
print("=" * 100)

print(
    f"{'Feature group':<38} "
    f"{'Features':>8} "
    f"{'Accuracy':>10} "
    f"{'Brier':>10} "
    f"{'LogLoss':>10}"
)

print("-" * 100)


for _, row in win_results.iterrows():

    print(
        f"{row['feature_group']:<38} "
        f"{int(row['feature_count']):>8} "
        f"{row['win_accuracy']:>10.3f} "
        f"{row['brier']:>10.3f} "
        f"{row['log_loss']:>10.3f}"
    )


print("\nMARKET FAVORITE")
print("-" * 100)

print(
    f"Sportsbook favorite accuracy: "
    f"{market_accuracy:.3f}"
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
print("=" * 90)

print(
    f"Saved to: "
    f"{output_file}"
)


print("\n")
print(
    "2025 holdout was not evaluated."
)