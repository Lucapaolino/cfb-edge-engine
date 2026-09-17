import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

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

from sklearn.impute import SimpleImputer

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
    "neural_network_v1_results.csv"
)


predictions_file = (
    "data/processed/"
    "neural_network_v1_predictions.csv"
)


validation_years = [
    2021,
    2022,
    2023,
    2024
]


# ------------------------------------------------------------
# Neural-network training settings
# ------------------------------------------------------------

BATCH_SIZE = 64

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

DROPOUT = 0.20

MAX_EPOCHS = 400

PATIENCE = 30

MIN_DELTA = 0.0001


BASE_SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device(
        "mps"
    )

else:

    device = torch.device(
        "cpu"
    )


print(
    "NEURAL NETWORK V1"
)

print(
    "=" * 100
)


print(
    f"PyTorch version: "
    f"{torch.__version__}"
)


print(
    f"Training device: "
    f"{device}"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    feature_file
)


print(
    f"Rows loaded:     "
    f"{len(data):,}"
)


print(
    f"Columns loaded:  "
    f"{len(data.columns):,}"
)


# ============================================================
# RESERVED TEST SAFETY
# ============================================================

if (
    data[
        "season"
    ].max()
    < 2025
):

    raise RuntimeError(
        "Expected 2025 rows to exist."
    )


print(
    "\n2025 rows exist but will NOT be "
    "trained on, validated on, or scored."
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


data[
    "model_target_home_win"
] = (
    data[
        margin_target
    ]
    > 0
).astype(
    int
)


win_target = (
    "model_target_home_win"
)


# ============================================================
# EXACT 54-FEATURE V1 MODEL
# ============================================================

v1_columns = list(
    data.columns[
        :192
    ]
)


features = [

    column

    for column in v1_columns

    if column.startswith(
        "diff_pregame_"
    )
]


features += [

    "neutral_site",

    "pregame_elo_diff",

    "talent_diff",

    "returning_ppa_diff",

    "returning_passing_ppa_diff",

    "returning_receiving_ppa_diff",

    "returning_rushing_ppa_diff"
]


features = sorted(
    set(
        features
    )
)


if len(
    features
) != 54:

    raise RuntimeError(
        f"Expected exactly 54 V1 features. "
        f"Found {len(features)}."
    )


print(
    f"\nModel input features: "
    f"{len(features)}"
)


# ============================================================
# NETWORK ARCHITECTURES
# ============================================================

architectures = {

    "NN Small":
        [
            64,
            32
        ],

    "NN Medium":
        [
            128,
            64,
            32
        ],

    "NN Large":
        [
            256,
            128,
            64
        ]
}


# ============================================================
# NETWORK CLASS
#
# One shared representation feeds TWO output heads:
#
# 1. expected home margin
# 2. home win probability
#
# This is called multi-task learning.
# ============================================================

class FootballNetwork(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_sizes,
        dropout
    ):

        super().__init__()


        layers = []


        previous_size = (
            input_size
        )


        for hidden_size in hidden_sizes:

            layers.append(
                nn.Linear(
                    previous_size,
                    hidden_size
                )
            )


            layers.append(
                nn.ReLU()
            )


            layers.append(
                nn.Dropout(
                    dropout
                )
            )


            previous_size = (
                hidden_size
            )


        self.shared = (
            nn.Sequential(
                *layers
            )
        )


        # ----------------------------------------------------
        # Regression head:
        #
        # expected point margin
        # ----------------------------------------------------

        self.margin_head = (
            nn.Linear(
                previous_size,
                1
            )
        )


        # ----------------------------------------------------
        # Classification head:
        #
        # raw win-probability logit
        # ----------------------------------------------------

        self.win_head = (
            nn.Linear(
                previous_size,
                1
            )
        )


        self.initialize_weights()


    def initialize_weights(
        self
    ):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear
            ):

                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu"
                )


                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )


    def forward(
        self,
        x
    ):

        shared = self.shared(
            x
        )


        margin = (
            self.margin_head(
                shared
            )
            .squeeze(
                1
            )
        )


        win_logit = (
            self.win_head(
                shared
            )
            .squeeze(
                1
            )
        )


        return (
            margin,
            win_logit
        )


# ============================================================
# PARAMETER COUNTER
# ============================================================

def count_parameters(
    model
):

    return sum(
        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )


# ============================================================
# DISPLAY ARCHITECTURES
# ============================================================

print("\n")

print(
    "NETWORK ARCHITECTURES"
)

print(
    "-" * 100
)


for (
    name,
    hidden_sizes
) in architectures.items():

    temporary_model = (
        FootballNetwork(

            input_size=len(
                features
            ),

            hidden_sizes=hidden_sizes,

            dropout=DROPOUT
        )
    )


    parameters = count_parameters(
        temporary_model
    )


    architecture_string = (
        "54"
        +
        "".join(
            f" → {size}"

            for size
            in hidden_sizes
        )
        +
        " → 2 heads"
    )


    print(
        f"{name:<12} "
        f"{architecture_string:<35} "
        f"{parameters:,} parameters"
    )


# ============================================================
# FEATURE PREPROCESSING
#
# Median values and scaling are fit ONLY using
# training data.
#
# The validation/test year never determines these.
# ============================================================

def prepare_features(
    training_frame,
    evaluation_frame
):

    training_numeric = (

        training_frame[
            features
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    evaluation_numeric = (

        evaluation_frame[
            features
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    # --------------------------------------------------------
    # Training-only medians
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


    # --------------------------------------------------------
    # Training-only scaling
    #
    # Neural networks usually train better when inputs
    # are on approximately comparable scales.
    # --------------------------------------------------------

    scaler = (
        StandardScaler()
    )


    training_scaled = (
        scaler
        .fit_transform(
            training_imputed
        )
        .astype(
            np.float32
        )
    )


    evaluation_scaled = (
        scaler
        .transform(
            evaluation_imputed
        )
        .astype(
            np.float32
        )
    )


    return (
        training_scaled,
        evaluation_scaled
    )


# ============================================================
# DATA LOADER
# ============================================================

def make_loader(
    x,
    margin,
    win,
    shuffle
):

    dataset = TensorDataset(

        torch.tensor(
            x,
            dtype=torch.float32
        ),

        torch.tensor(
            margin,
            dtype=torch.float32
        ),

        torch.tensor(
            win,
            dtype=torch.float32
        )
    )


    return DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

        num_workers=0
    )


# ============================================================
# LOSS FUNCTIONS
#
# Margin uses Smooth L1 loss.
#
# Win probability uses binary cross entropy.
#
# Margin is standardized before training so the two
# losses operate on roughly similar numerical scales.
# ============================================================

margin_loss_function = (
    nn.SmoothL1Loss()
)


win_loss_function = (
    nn.BCEWithLogitsLoss()
)


# ============================================================
# INNER EARLY-STOPPING TRAINING
#
# This DOES NOT evaluate the outer test season.
#
# Example for 2024:
#
# train 2018-2022
# validate 2023
#
# It only chooses how many epochs to use.
# ============================================================

def select_best_epoch(

    inner_train,

    inner_validation,

    hidden_sizes,

    seed
):

    set_seed(
        seed
    )


    (
        x_train,
        x_validation
    ) = prepare_features(

        inner_train,

        inner_validation
    )


    # --------------------------------------------------------
    # Standardize margin target using INNER TRAIN only.
    # --------------------------------------------------------

    margin_mean = float(
        inner_train[
            margin_target
        ].mean()
    )


    margin_std = float(
        inner_train[
            margin_target
        ].std(
            ddof=0
        )
    )


    if (
        margin_std
        < 0.000001
    ):

        margin_std = 1.0


    y_train_margin = (

        (
            inner_train[
                margin_target
            ]
            .to_numpy(
                dtype=np.float32
            )

            -
            margin_mean
        )

        /
        margin_std
    )


    y_validation_margin = (

        (
            inner_validation[
                margin_target
            ]
            .to_numpy(
                dtype=np.float32
            )

            -
            margin_mean
        )

        /
        margin_std
    )


    y_train_win = (

        inner_train[
            win_target
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    y_validation_win = (

        inner_validation[
            win_target
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    train_loader = make_loader(

        x_train,

        y_train_margin,

        y_train_win,

        shuffle=True
    )


    x_validation_tensor = (

        torch.tensor(
            x_validation,
            dtype=torch.float32
        )
        .to(
            device
        )
    )


    y_validation_margin_tensor = (

        torch.tensor(
            y_validation_margin,
            dtype=torch.float32
        )
        .to(
            device
        )
    )


    y_validation_win_tensor = (

        torch.tensor(
            y_validation_win,
            dtype=torch.float32
        )
        .to(
            device
        )
    )


    model = FootballNetwork(

        input_size=len(
            features
        ),

        hidden_sizes=hidden_sizes,

        dropout=DROPOUT
    ).to(
        device
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    best_validation_loss = (
        float(
            "inf"
        )
    )


    best_epoch = 1

    epochs_without_improvement = 0


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()


        for (
            batch_x,
            batch_margin,
            batch_win
        ) in train_loader:

            batch_x = batch_x.to(
                device
            )


            batch_margin = (
                batch_margin.to(
                    device
                )
            )


            batch_win = batch_win.to(
                device
            )


            optimizer.zero_grad()


            (
                predicted_margin,
                predicted_win_logit
            ) = model(
                batch_x
            )


            margin_loss = (
                margin_loss_function(

                    predicted_margin,

                    batch_margin
                )
            )


            win_loss = (
                win_loss_function(

                    predicted_win_logit,

                    batch_win
                )
            )


            total_loss = (

                margin_loss

                +

                win_loss
            )


            total_loss.backward()


            optimizer.step()


        # ====================================================
        # INNER VALIDATION
        # ====================================================

        model.eval()


        with torch.no_grad():

            (
                validation_margin_prediction,
                validation_win_logit
            ) = model(
                x_validation_tensor
            )


            validation_margin_loss = (
                margin_loss_function(

                    validation_margin_prediction,

                    y_validation_margin_tensor
                )
            )


            validation_win_loss = (
                win_loss_function(

                    validation_win_logit,

                    y_validation_win_tensor
                )
            )


            validation_loss = float(

                (
                    validation_margin_loss
                    +
                    validation_win_loss
                )
                .item()
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

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


            epochs_without_improvement = 0


        else:

            epochs_without_improvement += 1


        if (
            epochs_without_improvement
            >=
            PATIENCE
        ):

            break


    return (
        best_epoch,
        best_validation_loss
    )


# ============================================================
# FINAL TRAINING
#
# After choosing the epoch count from historical data,
# we rebuild the model from scratch using ALL data prior
# to the outer validation season.
#
# The outer validation season remains unseen.
# ============================================================

def train_final_model(

    outer_train,

    outer_validation,

    hidden_sizes,

    epochs,

    seed
):

    set_seed(
        seed
    )


    (
        x_train,
        x_validation
    ) = prepare_features(

        outer_train,

        outer_validation
    )


    # --------------------------------------------------------
    # Standardize margin using full OUTER TRAIN only.
    # --------------------------------------------------------

    margin_mean = float(
        outer_train[
            margin_target
        ].mean()
    )


    margin_std = float(
        outer_train[
            margin_target
        ].std(
            ddof=0
        )
    )


    if (
        margin_std
        < 0.000001
    ):

        margin_std = 1.0


    y_train_margin = (

        (
            outer_train[
                margin_target
            ]
            .to_numpy(
                dtype=np.float32
            )

            -
            margin_mean
        )

        /
        margin_std
    )


    y_train_win = (

        outer_train[
            win_target
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    train_loader = make_loader(

        x_train,

        y_train_margin,

        y_train_win,

        shuffle=True
    )


    model = FootballNetwork(

        input_size=len(
            features
        ),

        hidden_sizes=hidden_sizes,

        dropout=DROPOUT
    ).to(
        device
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # FIXED NUMBER OF EPOCHS
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()


        for (
            batch_x,
            batch_margin,
            batch_win
        ) in train_loader:

            batch_x = batch_x.to(
                device
            )


            batch_margin = (
                batch_margin.to(
                    device
                )
            )


            batch_win = batch_win.to(
                device
            )


            optimizer.zero_grad()


            (
                predicted_margin,
                predicted_win_logit
            ) = model(
                batch_x
            )


            margin_loss = (
                margin_loss_function(

                    predicted_margin,

                    batch_margin
                )
            )


            win_loss = (
                win_loss_function(

                    predicted_win_logit,

                    batch_win
                )
            )


            total_loss = (

                margin_loss

                +

                win_loss
            )


            total_loss.backward()


            optimizer.step()


    # ========================================================
    # OUTER PREDICTIONS
    # ========================================================

    model.eval()


    x_validation_tensor = (

        torch.tensor(
            x_validation,
            dtype=torch.float32
        )
        .to(
            device
        )
    )


    with torch.no_grad():

        (
            standardized_margin_prediction,
            win_logit
        ) = model(
            x_validation_tensor
        )


        standardized_margin_prediction = (

            standardized_margin_prediction

            .cpu()

            .numpy()
        )


        win_probability = (

            torch.sigmoid(
                win_logit
            )

            .cpu()

            .numpy()
        )


    # --------------------------------------------------------
    # Convert standardized margin back into POINTS.
    # --------------------------------------------------------

    margin_prediction = (

        standardized_margin_prediction

        *
        margin_std

        +
        margin_mean
    )


    return (
        margin_prediction,
        win_probability
    )


# ============================================================
# BASELINE MODELS
#
# Re-run our exact V1 Ridge/logistic benchmark in the same
# outer folds so comparison is apples-to-apples.
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
# METRICS
# ============================================================

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


    return (
        mae,
        rmse
    )


def win_metrics(
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
# OUTER ROLLING VALIDATION
# ============================================================

for validation_year in validation_years:

    print("\n")

    print(
        "=" * 100
    )

    print(
        f"OUTER VALIDATION YEAR "
        f"{validation_year}"
    )

    print(
        "=" * 100
    )


    # --------------------------------------------------------
    # All previous seasons
    # --------------------------------------------------------

    outer_train = data[
        data[
            "season"
        ]
        <
        validation_year
    ].copy()


    # --------------------------------------------------------
    # Unseen outer test year
    # --------------------------------------------------------

    outer_validation = data[
        data[
            "season"
        ]
        ==
        validation_year
    ].copy()


    # --------------------------------------------------------
    # Hard safety wall around 2025
    # --------------------------------------------------------

    outer_train = outer_train[
        outer_train[
            "season"
        ]
        <= 2024
    ].copy()


    outer_validation = outer_validation[
        outer_validation[
            "season"
        ]
        <= 2024
    ].copy()


    inner_validation_year = (
        validation_year
        -
        1
    )


    inner_train = outer_train[
        outer_train[
            "season"
        ]
        <
        inner_validation_year
    ].copy()


    inner_validation = outer_train[
        outer_train[
            "season"
        ]
        ==
        inner_validation_year
    ].copy()


    if len(
        inner_train
    ) == 0:

        raise RuntimeError(
            "Inner training set is empty."
        )


    if len(
        inner_validation
    ) == 0:

        raise RuntimeError(
            "Inner validation set is empty."
        )


    print(
        f"Inner train:       "
        f"{int(inner_train['season'].min())}-"
        f"{int(inner_train['season'].max())} "
        f"({len(inner_train):,} games)"
    )


    print(
        f"Inner validation:  "
        f"{inner_validation_year} "
        f"({len(inner_validation):,} games)"
    )


    print(
        f"Final train:       "
        f"{int(outer_train['season'].min())}-"
        f"{int(outer_train['season'].max())} "
        f"({len(outer_train):,} games)"
    )


    print(
        f"Outer validation:  "
        f"{validation_year} "
        f"({len(outer_validation):,} games)"
    )


    y_outer_margin = (

        outer_validation[
            margin_target
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    y_outer_win = (

        outer_validation[
            win_target
        ]

        .to_numpy(
            dtype=np.int64
        )
    )


    # ========================================================
    # V1 LINEAR BENCHMARK
    # ========================================================

    X_outer_train = (

        outer_train[
            features
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    X_outer_validation = (

        outer_validation[
            features
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    ridge = build_ridge()


    ridge.fit(

        X_outer_train,

        outer_train[
            margin_target
        ]
    )


    ridge_prediction = (
        ridge.predict(
            X_outer_validation
        )
    )


    logistic = build_logistic()


    logistic.fit(

        X_outer_train,

        outer_train[
            win_target
        ]
    )


    logistic_probability = (

        logistic

        .predict_proba(
            X_outer_validation
        )[
            :,
            1
        ]
    )


    (
        ridge_mae,
        ridge_rmse
    ) = margin_metrics(

        y_outer_margin,

        ridge_prediction
    )


    (
        logistic_accuracy,
        logistic_brier,
        logistic_logloss
    ) = win_metrics(

        y_outer_win,

        logistic_probability
    )


    print("\n")

    print(
        "V1 LINEAR BENCHMARK"
    )

    print(
        "-" * 100
    )


    print(
        f"Margin MAE:       "
        f"{ridge_mae:.3f}"
    )


    print(
        f"Margin RMSE:      "
        f"{ridge_rmse:.3f}"
    )


    print(
        f"Win accuracy:     "
        f"{logistic_accuracy:.3f}"
    )


    print(
        f"Brier:            "
        f"{logistic_brier:.3f}"
    )


    print(
        f"LogLoss:          "
        f"{logistic_logloss:.3f}"
    )


    results.append(
        {
            "validation_year":
                validation_year,

            "model":
                "V1 Linear",

            "parameters":
                np.nan,

            "best_epoch":
                np.nan,

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

            "head_disagreement_rate":
                np.nan
        }
    )


    # ========================================================
    # SPORTSBOOK BENCHMARK
    # ========================================================

    if (
        "benchmark_spread"
        in outer_validation.columns
    ):

        market_mask = (
            outer_validation[
                "benchmark_spread"
            ]
            .notna()
        )


        market_margin_prediction = (

            -outer_validation.loc[
                market_mask,
                "benchmark_spread"
            ]
        )


        (
            market_mae,
            market_rmse
        ) = margin_metrics(

            outer_validation.loc[
                market_mask,
                margin_target
            ],

            market_margin_prediction
        )


        favorite_mask = (

            market_mask

            &

            (
                outer_validation[
                    "benchmark_spread"
                ]
                != 0
            )
        )


        market_win_prediction = (

            outer_validation.loc[
                favorite_mask,
                "benchmark_spread"
            ]

            < 0
        ).astype(
            int
        )


        market_accuracy = (
            accuracy_score(

                outer_validation.loc[
                    favorite_mask,
                    win_target
                ],

                market_win_prediction
            )
        )


        results.append(
            {
                "validation_year":
                    validation_year,

                "model":
                    "Sportsbook",

                "parameters":
                    np.nan,

                "best_epoch":
                    np.nan,

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

                "head_disagreement_rate":
                    np.nan
            }
        )


    # ========================================================
    # NEURAL NETWORKS
    # ========================================================

    for architecture_index, (
        architecture_name,
        hidden_sizes
    ) in enumerate(
        architectures.items()
    ):

        print("\n")

        print(
            architecture_name
        )

        print(
            "-" * 100
        )


        seed = (

            BASE_SEED

            +
            validation_year

            +
            architecture_index
            * 100
        )


        # ----------------------------------------------------
        # STEP 1
        #
        # Use previous historical season to decide
        # when training should stop.
        # ----------------------------------------------------

        (
            best_epoch,
            inner_validation_loss
        ) = select_best_epoch(

            inner_train=inner_train,

            inner_validation=inner_validation,

            hidden_sizes=hidden_sizes,

            seed=seed
        )


        print(
            f"Selected epoch:    "
            f"{best_epoch}"
        )


        print(
            f"Inner val loss:    "
            f"{inner_validation_loss:.4f}"
        )


        # ----------------------------------------------------
        # STEP 2
        #
        # Start a fresh model.
        #
        # Train using ALL historical seasons before
        # the outer validation year.
        # ----------------------------------------------------

        (
            margin_prediction,
            win_probability
        ) = train_final_model(

            outer_train=outer_train,

            outer_validation=outer_validation,

            hidden_sizes=hidden_sizes,

            epochs=best_epoch,

            seed=seed
        )


        # ----------------------------------------------------
        # OUTER METRICS
        # ----------------------------------------------------

        (
            mae,
            rmse
        ) = margin_metrics(

            y_outer_margin,

            margin_prediction
        )


        (
            accuracy,
            brier,
            loss
        ) = win_metrics(

            y_outer_win,

            win_probability
        )


        # ----------------------------------------------------
        # HEAD DISAGREEMENT
        #
        # Example disagreement:
        #
        # predicted margin says home team wins
        # but probability head says < 50%.
        #
        # Some disagreement is possible because they're
        # separate tasks, but a very large number would be
        # worth investigating.
        # ----------------------------------------------------

        margin_home_win = (
            margin_prediction
            > 0
        )


        probability_home_win = (
            win_probability
            >= 0.50
        )


        disagreement_rate = float(

            np.mean(
                margin_home_win
                !=
                probability_home_win
            )
        )


        model_for_count = FootballNetwork(

            input_size=len(
                features
            ),

            hidden_sizes=hidden_sizes,

            dropout=DROPOUT
        )


        parameters = count_parameters(
            model_for_count
        )


        print(
            f"Parameters:        "
            f"{parameters:,}"
        )


        print(
            f"Margin MAE:        "
            f"{mae:.3f}"
        )


        print(
            f"Margin RMSE:       "
            f"{rmse:.3f}"
        )


        print(
            f"Win accuracy:      "
            f"{accuracy:.3f}"
        )


        print(
            f"Brier:             "
            f"{brier:.3f}"
        )


        print(
            f"LogLoss:           "
            f"{loss:.3f}"
        )


        print(
            f"Head disagreement: "
            f"{disagreement_rate:.3%}"
        )


        results.append(
            {
                "validation_year":
                    validation_year,

                "model":
                    architecture_name,

                "parameters":
                    parameters,

                "best_epoch":
                    best_epoch,

                "margin_mae":
                    mae,

                "margin_rmse":
                    rmse,

                "win_accuracy":
                    accuracy,

                "brier":
                    brier,

                "logloss":
                    loss,

                "head_disagreement_rate":
                    disagreement_rate
            }
        )


        # ====================================================
        # SAVE GAME-LEVEL NN PREDICTIONS
        # ====================================================

        for position in range(
            len(
                outer_validation
            )
        ):

            game = outer_validation.iloc[
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

                    "week":
                        game[
                            "week"
                        ],

                    "away_team":
                        game[
                            "away_team"
                        ],

                    "home_team":
                        game[
                            "home_team"
                        ],

                    "model":
                        architecture_name,

                    "actual_margin":
                        y_outer_margin[
                            position
                        ],

                    "actual_home_win":
                        y_outer_win[
                            position
                        ],

                    "predicted_margin":
                        margin_prediction[
                            position
                        ],

                    "predicted_home_win_probability":
                        win_probability[
                            position
                        ],

                    "benchmark_spread":
                        game.get(
                            "benchmark_spread",
                            np.nan
                        )
                }
            )


# ============================================================
# SAVE RESULTS
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
# AVERAGE ROLLING RESULTS
#
# Each season gets equal weight, matching our previous
# rolling-validation summaries.
# ============================================================

average_results = (

    results

    .groupby(
        "model",
        as_index=False
    )

    .agg(

        parameters=(
            "parameters",
            "max"
        ),

        average_best_epoch=(
            "best_epoch",
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
        ),

        head_disagreement_rate=(
            "head_disagreement_rate",
            "mean"
        )
    )
)


average_results = (
    average_results

    .sort_values(
        "margin_mae"
    )
)


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


print(
    average_results.to_string(

        index=False,

        formatters={

            "parameters":
                lambda value:
                    (
                        f"{value:,.0f}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

            "average_best_epoch":
                lambda value:
                    (
                        f"{value:.1f}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    ),

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
                    ),

            "head_disagreement_rate":
                lambda value:
                    (
                        f"{value:.3%}"
                        if pd.notna(
                            value
                        )
                        else "-"
                    )
        }
    )
)


# ============================================================
# DIRECT NN VS V1
# ============================================================

print("\n")

print(
    "=" * 100
)

print(
    "DIRECT NEURAL NETWORK CHANGE VS V1 LINEAR"
)

print(
    "=" * 100
)


baseline_row = average_results[
    average_results[
        "model"
    ]
    ==
    "V1 Linear"
]


if len(
    baseline_row
) == 1:

    baseline_row = (
        baseline_row.iloc[
            0
        ]
    )


    for architecture_name in architectures:

        architecture_row = (
            average_results[
                average_results[
                    "model"
                ]
                ==
                architecture_name
            ]
        )


        if len(
            architecture_row
        ) != 1:

            continue


        architecture_row = (
            architecture_row.iloc[
                0
            ]
        )


        print("\n")

        print(
            architecture_name
        )


        print(
            f"MAE change:       "
            f"{architecture_row['margin_mae'] - baseline_row['margin_mae']:+.3f}"
        )


        print(
            f"RMSE change:      "
            f"{architecture_row['margin_rmse'] - baseline_row['margin_rmse']:+.3f}"
        )


        print(
            f"Accuracy change:  "
            f"{architecture_row['win_accuracy'] - baseline_row['win_accuracy']:+.4f}"
        )


        print(
            f"Brier change:     "
            f"{architecture_row['brier'] - baseline_row['brier']:+.4f}"
        )


        print(
            f"LogLoss change:   "
            f"{architecture_row['logloss'] - baseline_row['logloss']:+.4f}"
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