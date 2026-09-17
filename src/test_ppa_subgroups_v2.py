import os

import numpy as np
import pandas as pd

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
    "ppa_subgroup_results_v2.csv"
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
    "PPA SUBGROUP ABLATION — V2"
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
# TARGETS
# ============================================================

margin_target = (
    "target_home_margin"
)


if margin_target not in data.columns:

    raise RuntimeError(
        "target_home_margin is missing."
    )


# ------------------------------------------------------------
# Derive win target from actual home margin.
# ------------------------------------------------------------

win_target_series = (
    data[
        margin_target
    ] > 0
).astype(
    int
)


# ============================================================
# ORIGINAL V1 FEATURES
# ============================================================

v1_columns = list(
    data.columns[
        :192
    ]
)


v1_features = [

    column

    for column in v1_columns

    if column.startswith(
        "diff_pregame_"
    )
]


v1_features += [

    "neutral_site",

    "pregame_elo_diff",

    "talent_diff",

    "returning_ppa_diff",

    "returning_passing_ppa_diff",

    "returning_receiving_ppa_diff",

    "returning_rushing_ppa_diff"
]


v1_features = sorted(
    set(
        v1_features
    )
)


if len(
    v1_features
) != 54:

    raise RuntimeError(
        f"Expected 54 V1 features, "
        f"found {len(v1_features)}."
    )


print(
    f"\nExact V1 features: "
    f"{len(v1_features)}"
)


# ============================================================
# NEW PPA FEATURES
# ============================================================

new_ppa_features = [

    column

    for column in data.columns

    if (
        column.startswith(
            "diff_pregame_"
        )

        and

        "ppa" in column.lower()

        and

        column not in v1_features
    )
]


history_features = []


for feature in [

    "diff_pregame_ppa_games_available",

    "min_pregame_ppa_games_available"
]:

    if feature in data.columns:

        history_features.append(
            feature
        )


new_ppa_features = sorted(
    set(
        new_ppa_features
        +
        history_features
    )
)


print(
    f"New PPA features:  "
    f"{len(new_ppa_features)}"
)


# ============================================================
# BUILD LOGICAL PPA GROUPS
# ============================================================

overall_ppa = [

    column

    for column in new_ppa_features

    if (
        "_ppa_per_play"
        in column

        and
        "rush_ppa_per_play"
        not in column

        and
        "dropback_ppa_per_play"
        not in column

        and
        "positive_ppa_rate"
        not in column
    )
]


dropback_ppa = [

    column

    for column in new_ppa_features

    if (
        "dropback_ppa_per_play"
        in column
    )
]


rush_ppa = [

    column

    for column in new_ppa_features

    if (
        "rush_ppa_per_play"
        in column
    )
]


positive_ppa = [

    column

    for column in new_ppa_features

    if (
        "positive_ppa_rate"
        in column
    )
]


net_ppa = [

    column

    for column in new_ppa_features

    if (
        "_net_"
        in column
    )
]


season_average_ppa = [

    column

    for column in new_ppa_features

    if column.startswith(
        "diff_pregame_avg_"
    )
]


last3_ppa = [

    column

    for column in new_ppa_features

    if column.startswith(
        "diff_pregame_last3_"
    )
]


# ============================================================
# FEATURE SET HELPER
# ============================================================

def with_v1(
    extra_features
):

    return sorted(
        set(
            v1_features
            +
            extra_features
            +
            history_features
        )
    )


# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {

    "V1":
        v1_features,

    "V1 + PPA History":
        with_v1(
            []
        ),

    "V1 + Overall PPA":
        with_v1(
            overall_ppa
        ),

    "V1 + Dropback PPA":
        with_v1(
            dropback_ppa
        ),

    "V1 + Rush PPA":
        with_v1(
            rush_ppa
        ),

    "V1 + Positive PPA":
        with_v1(
            positive_ppa
        ),

    "V1 + Net PPA":
        with_v1(
            net_ppa
        ),

    "V1 + Season Avg PPA":
        with_v1(
            season_average_ppa
        ),

    "V1 + Last3 PPA":
        with_v1(
            last3_ppa
        ),

    "V1 + All PPA":
        sorted(
            set(
                v1_features
                +
                new_ppa_features
            )
        )
}


# ============================================================
# DISPLAY GROUP SIZES
# ============================================================

print("\n")

print(
    "FEATURE GROUP SIZES"
)

print(
    "-" * 100
)


for (
    name,
    features
) in feature_sets.items():

    print(
        f"{name:<24} "
        f"{len(features):>3}"
    )


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_ridge():

    return Pipeline(
        [
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
        [
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


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# ROLLING VALIDATION
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


    train_mask = (
        data[
            "season"
        ]
        <
        validation_year
    )


    validation_mask = (
        data[
            "season"
        ]
        ==
        validation_year
    )


    # --------------------------------------------------------
    # 2025 cannot enter the experiment.
    # --------------------------------------------------------

    train_mask &= (
        data[
            "season"
        ]
        <= 2024
    )


    validation_mask &= (
        data[
            "season"
        ]
        <= 2024
    )


    train = data[
        train_mask
    ].copy()


    validation = data[
        validation_mask
    ].copy()


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
        win_target_series.loc[
            train.index
        ]
    )


    y_validation_win = (
        win_target_series.loc[
            validation.index
        ]
    )


    print(
        f"Train: "
        f"{int(train['season'].min())}-"
        f"{int(train['season'].max())} "
        f"({len(train):,} games)"
    )


    print(
        f"Validate: "
        f"{validation_year} "
        f"({len(validation):,} games)"
    )


    # ========================================================
    # FEATURE GROUP LOOP
    # ========================================================

    for (
        group_name,
        features
    ) in feature_sets.items():

        X_train = (
            train[
                features
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
        )


        X_validation = (
            validation[
                features
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
        )


        # ----------------------------------------------------
        # MARGIN
        # ----------------------------------------------------

        ridge = build_ridge()


        ridge.fit(
            X_train,
            y_train_margin
        )


        margin_prediction = (
            ridge.predict(
                X_validation
            )
        )


        mae = mean_absolute_error(
            y_validation_margin,
            margin_prediction
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_validation_margin,
                margin_prediction
            )
        )


        # ----------------------------------------------------
        # WIN PROBABILITY
        # ----------------------------------------------------

        logistic = build_logistic()


        logistic.fit(
            X_train,
            y_train_win
        )


        probability = (
            logistic.predict_proba(
                X_validation
            )[
                :,
                1
            ]
        )


        classification = (
            probability
            >= 0.50
        ).astype(
            int
        )


        accuracy = accuracy_score(
            y_validation_win,
            classification
        )


        brier = brier_score_loss(
            y_validation_win,
            probability
        )


        loss = log_loss(
            y_validation_win,
            probability,
            labels=[
                0,
                1
            ]
        )


        results.append(
            {
                "validation_year":
                    validation_year,

                "feature_set":
                    group_name,

                "feature_count":
                    len(
                        features
                    ),

                "margin_mae":
                    mae,

                "margin_rmse":
                    rmse,

                "win_accuracy":
                    accuracy,

                "brier":
                    brier,

                "logloss":
                    loss
            }
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    results
)


os.makedirs(
    "data/processed",
    exist_ok=True
)


results.to_csv(
    results_file,
    index=False
)


# ============================================================
# AVERAGE RESULTS
# ============================================================

average = (

    results

    .groupby(
        [
            "feature_set",
            "feature_count"
        ],
        as_index=False
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
)


average = average.sort_values(
    "margin_mae"
)


print("\n\n")

print(
    "=" * 100
)

print(
    "AVERAGE PPA SUBGROUP RESULTS"
)

print(
    "=" * 100
)


print(
    average.to_string(
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
                    f"{value:.3f}",

            "logloss":
                lambda value:
                    f"{value:.3f}"
        }
    )
)


# ============================================================
# COMPARE EVERY GROUP TO V1
# ============================================================

baseline = average[
    average[
        "feature_set"
    ]
    ==
    "V1"
].iloc[
    0
]


comparison_rows = []


for _, row in average.iterrows():

    if (
        row[
            "feature_set"
        ]
        ==
        "V1"
    ):

        continue


    group_results = results[
        results[
            "feature_set"
        ]
        ==
        row[
            "feature_set"
        ]
    ]


    baseline_results = results[
        results[
            "feature_set"
        ]
        ==
        "V1"
    ]


    merged = baseline_results.merge(

        group_results,

        on="validation_year",

        suffixes=(
            "_v1",
            "_group"
        )
    )


    mae_year_wins = int(
        (
            merged[
                "margin_mae_group"
            ]
            <
            merged[
                "margin_mae_v1"
            ]
        )
        .sum()
    )


    brier_year_wins = int(
        (
            merged[
                "brier_group"
            ]
            <
            merged[
                "brier_v1"
            ]
        )
        .sum()
    )


    comparison_rows.append(
        {
            "feature_set":
                row[
                    "feature_set"
                ],

            "mae_change":
                (
                    row[
                        "margin_mae"
                    ]
                    -
                    baseline[
                        "margin_mae"
                    ]
                ),

            "rmse_change":
                (
                    row[
                        "margin_rmse"
                    ]
                    -
                    baseline[
                        "margin_rmse"
                    ]
                ),

            "accuracy_change":
                (
                    row[
                        "win_accuracy"
                    ]
                    -
                    baseline[
                        "win_accuracy"
                    ]
                ),

            "brier_change":
                (
                    row[
                        "brier"
                    ]
                    -
                    baseline[
                        "brier"
                    ]
                ),

            "logloss_change":
                (
                    row[
                        "logloss"
                    ]
                    -
                    baseline[
                        "logloss"
                    ]
                ),

            "mae_years_better":
                mae_year_wins,

            "brier_years_better":
                brier_year_wins
        }
    )


comparison = pd.DataFrame(
    comparison_rows
)


comparison = comparison.sort_values(
    "mae_change"
)


print("\n")

print(
    "=" * 100
)

print(
    "CHANGE VS V1"
)

print(
    "=" * 100
)


print(
    comparison.to_string(
        index=False,
        formatters={

            "mae_change":
                lambda value:
                    f"{value:+.3f}",

            "rmse_change":
                lambda value:
                    f"{value:+.3f}",

            "accuracy_change":
                lambda value:
                    f"{value:+.4f}",

            "brier_change":
                lambda value:
                    f"{value:+.4f}",

            "logloss_change":
                lambda value:
                    f"{value:+.4f}"
        }
    )
)


print("\n")

print(
    f"Results saved to: "
    f"{results_file}"
)


print(
    "2025 was not trained on, validated on, "
    "or scored."
)