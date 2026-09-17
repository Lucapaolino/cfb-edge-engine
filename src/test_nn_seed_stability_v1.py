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

FEATURE_FILE = (
    "data/processed/"
    "features_v2.csv"
)

RUN_RESULTS_FILE = (
    "data/processed/"
    "nn_seed_stability_runs_v1.csv"
)

SEED_SUMMARY_FILE = (
    "data/processed/"
    "nn_seed_stability_by_seed_v1.csv"
)

PREDICTIONS_FILE = (
    "data/processed/"
    "nn_seed_stability_predictions_v1.csv"
)

ENSEMBLE_SEASON_FILE = (
    "data/processed/"
    "nn_seed_ensemble_by_season_v1.csv"
)

ENSEMBLE_SUMMARY_FILE = (
    "data/processed/"
    "nn_seed_ensemble_summary_v1.csv"
)


VALIDATION_YEARS = [
    2021,
    2022,
    2023,
    2024
]


# ------------------------------------------------------------
# Five different random starting points.
#
# The architecture and data stay exactly the same.
#
# Only the network's initial random weights and training
# randomness change.
# ------------------------------------------------------------

SEEDS = [
    11,
    23,
    37,
    53,
    71
]


# ------------------------------------------------------------
# Large does NOT advance.
#
# It had more than 55,000 parameters and performed worse
# than both smaller networks.
# ------------------------------------------------------------

ARCHITECTURES = {

    "NN Small": [
        64,
        32
    ],

    "NN Medium": [
        128,
        64,
        32
    ]
}


# ------------------------------------------------------------
# Training settings.
#
# These stay identical across all seeds so this remains
# a controlled experiment.
# ------------------------------------------------------------

BATCH_SIZE = 64

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

DROPOUT = 0.20

MAX_EPOCHS = 400

PATIENCE = 30

MIN_DELTA = 0.0001


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


print(
    "NEURAL NETWORK SEED STABILITY — V1"
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
    f"{DEVICE}"
)

print(
    f"Seeds: "
    f"{SEEDS}"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

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
    FEATURE_FILE
)


# Make a clean contiguous copy.
data = data.copy()


print()

print(
    f"Rows loaded:    "
    f"{len(data):,}"
)

print(
    f"Columns loaded: "
    f"{len(data.columns):,}"
)


# ============================================================
# RESERVED 2025 SAFETY CHECK
# ============================================================

if 2025 not in set(
    data[
        "season"
    ]
):

    raise RuntimeError(
        "Expected 2025 rows to exist in the "
        "feature dataset."
    )


print(
    "\n2025 exists in the dataset but will NOT be "
    "trained on, validated on, or scored."
)


# ============================================================
# TARGETS
# ============================================================

MARGIN_TARGET = (
    "target_home_margin"
)


if MARGIN_TARGET not in data.columns:

    raise RuntimeError(
        "target_home_margin is missing."
    )


# ------------------------------------------------------------
# Positive home margin = home team won.
#
# Keep this as a separate Series instead of adding another
# column to the dataframe.
# ------------------------------------------------------------

home_win_target = (

    data[
        MARGIN_TARGET
    ]

    > 0

).astype(
    int
)


# ============================================================
# EXACT 54-FEATURE V1 SET
# ============================================================

# V1 originally contained 192 columns.
# V2 appended the new PPA features after those.
v1_columns = list(
    data.columns[
        :192
    ]
)


FEATURES = [

    column

    for column in v1_columns

    if column.startswith(
        "diff_pregame_"
    )
]


FEATURES += [

    "neutral_site",

    "pregame_elo_diff",

    "talent_diff",

    "returning_ppa_diff",

    "returning_passing_ppa_diff",

    "returning_receiving_ppa_diff",

    "returning_rushing_ppa_diff"
]


FEATURES = sorted(
    set(
        FEATURES
    )
)


if len(
    FEATURES
) != 54:

    raise RuntimeError(

        f"Expected exactly 54 V1 features. "
        f"Found {len(FEATURES)}."
    )


print(
    f"\nModel features: "
    f"{len(FEATURES)}"
)


# ============================================================
# NEURAL NETWORK
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


        # ----------------------------------------------------
        # Shared hidden representation
        # ----------------------------------------------------

        self.shared = nn.Sequential(
            *layers
        )


        # ----------------------------------------------------
        # Head 1:
        #
        # Predict expected home point margin.
        # ----------------------------------------------------

        self.margin_head = nn.Linear(
            previous_size,
            1
        )


        # ----------------------------------------------------
        # Head 2:
        #
        # Predict home-win logit.
        #
        # We later convert the logit into a probability.
        # ----------------------------------------------------

        self.win_head = nn.Linear(
            previous_size,
            1
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
# DISPLAY NETWORKS
# ============================================================

print()

print(
    "NETWORKS BEING TESTED"
)

print(
    "-" * 100
)


for (
    architecture_name,
    hidden_sizes
) in ARCHITECTURES.items():

    temporary_model = FootballNetwork(

        input_size=len(
            FEATURES
        ),

        hidden_sizes=hidden_sizes,

        dropout=DROPOUT
    )


    parameter_count = count_parameters(
        temporary_model
    )


    architecture_text = (

        "54"

        +

        "".join(

            f" → {size}"

            for size in hidden_sizes
        )

        +

        " → 2 heads"
    )


    print(

        f"{architecture_name:<12} "
        f"{architecture_text:<35} "
        f"{parameter_count:,} parameters"
    )


# ============================================================
# FEATURE PREPROCESSING
# ============================================================

def prepare_features(
    training_frame,
    evaluation_frame
):

    training_numeric = (

        training_frame[
            FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    evaluation_numeric = (

        evaluation_frame[
            FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    # --------------------------------------------------------
    # Missing values are filled using medians calculated ONLY
    # from the training set.
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
    # Standardize inputs.
    #
    # Training data determines the scaling.
    # --------------------------------------------------------

    scaler = StandardScaler()


    x_train = (

        scaler

        .fit_transform(
            training_imputed
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
# DATA LOADER
# ============================================================

def make_loader(
    x,
    margin,
    win,
    seed
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


    # --------------------------------------------------------
    # Controls the random order that games are shown to the
    # network during mini-batch training.
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    return DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        generator=generator,

        num_workers=0
    )


# ============================================================
# LOSS FUNCTIONS
# ============================================================

# Margin prediction
margin_loss_function = (
    nn.SmoothL1Loss()
)


# Win probability
win_loss_function = (
    nn.BCEWithLogitsLoss()
)


# ============================================================
# INNER EARLY STOPPING
#
# Example when outer test year is 2024:
#
# Inner training:
# 2018-2022
#
# Inner validation:
# 2023
#
# 2024 is NEVER touched here.
# ============================================================

def select_best_epoch(

    inner_train,
    inner_train_win,

    inner_validation,
    inner_validation_win,

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
    # Standardize the point-margin target.
    #
    # This prevents margin loss from numerically overwhelming
    # the win-probability loss.
    # --------------------------------------------------------

    margin_mean = float(

        inner_train[
            MARGIN_TARGET
        ].mean()
    )


    margin_std = float(

        inner_train[
            MARGIN_TARGET
        ].std(
            ddof=0
        )
    )


    if margin_std < 1e-6:

        margin_std = 1.0


    y_train_margin = (

        (
            inner_train[
                MARGIN_TARGET
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
                MARGIN_TARGET
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

        inner_train_win

        .to_numpy(
            dtype=np.float32
        )
    )


    y_validation_win = (

        inner_validation_win

        .to_numpy(
            dtype=np.float32
        )
    )


    train_loader = make_loader(

        x_train,

        y_train_margin,

        y_train_win,

        seed
    )


    x_validation_tensor = (

        torch.tensor(
            x_validation,
            dtype=torch.float32
        )

        .to(
            DEVICE
        )
    )


    margin_validation_tensor = (

        torch.tensor(
            y_validation_margin,
            dtype=torch.float32
        )

        .to(
            DEVICE
        )
    )


    win_validation_tensor = (

        torch.tensor(
            y_validation_win,
            dtype=torch.float32
        )

        .to(
            DEVICE
        )
    )


    model = FootballNetwork(

        input_size=len(
            FEATURES
        ),

        hidden_sizes=hidden_sizes,

        dropout=DROPOUT
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    best_loss = float(
        "inf"
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
                DEVICE
            )

            batch_margin = (
                batch_margin.to(
                    DEVICE
                )
            )

            batch_win = batch_win.to(
                DEVICE
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
        # CHECK INNER VALIDATION SEASON
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

                    margin_validation_tensor
                )
            )


            validation_win_loss = (
                win_loss_function(

                    validation_win_logit,

                    win_validation_tensor
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
                best_loss
                -
                MIN_DELTA
            )
        ):

            best_loss = (
                validation_loss
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = 0


        else:

            epochs_without_improvement += 1


        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            break


    return (
        best_epoch,
        best_loss
    )


# ============================================================
# FINAL OUTER MODEL
#
# Once the epoch count is selected using older seasons,
# start over and train on ALL seasons before the test year.
# ============================================================

def train_final_model(

    outer_train,
    outer_train_win,

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


    margin_mean = float(

        outer_train[
            MARGIN_TARGET
        ].mean()
    )


    margin_std = float(

        outer_train[
            MARGIN_TARGET
        ].std(
            ddof=0
        )
    )


    if margin_std < 1e-6:

        margin_std = 1.0


    y_train_margin = (

        (
            outer_train[
                MARGIN_TARGET
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

        outer_train_win

        .to_numpy(
            dtype=np.float32
        )
    )


    train_loader = make_loader(

        x_train,

        y_train_margin,

        y_train_win,

        seed
    )


    model = FootballNetwork(

        input_size=len(
            FEATURES
        ),

        hidden_sizes=hidden_sizes,

        dropout=DROPOUT
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # TRAIN FOR SELECTED NUMBER OF EPOCHS
    # ========================================================

    for _ in range(
        epochs
    ):

        model.train()


        for (
            batch_x,
            batch_margin,
            batch_win
        ) in train_loader:

            batch_x = batch_x.to(
                DEVICE
            )

            batch_margin = (
                batch_margin.to(
                    DEVICE
                )
            )

            batch_win = batch_win.to(
                DEVICE
            )


            optimizer.zero_grad()


            (
                predicted_margin,
                predicted_win_logit
            ) = model(
                batch_x
            )


            total_loss = (

                margin_loss_function(

                    predicted_margin,

                    batch_margin
                )

                +

                win_loss_function(

                    predicted_win_logit,

                    batch_win
                )
            )


            total_loss.backward()

            optimizer.step()


    # ========================================================
    # PREDICT OUTER TEST YEAR
    # ========================================================

    model.eval()


    x_validation_tensor = (

        torch.tensor(
            x_validation,
            dtype=torch.float32
        )

        .to(
            DEVICE
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
    # Convert standardized margin back into actual points.
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
# METRICS
# ============================================================

def calculate_metrics(

    actual_margin,
    actual_win,

    predicted_margin,
    win_probability
):

    mae = mean_absolute_error(

        actual_margin,

        predicted_margin
    )


    rmse = np.sqrt(

        mean_squared_error(

            actual_margin,

            predicted_margin
        )
    )


    predicted_winner = (

        win_probability
        >= 0.50

    ).astype(
        int
    )


    accuracy = accuracy_score(

        actual_win,

        predicted_winner
    )


    brier = brier_score_loss(

        actual_win,

        win_probability
    )


    loss = log_loss(

        actual_win,

        win_probability,

        labels=[
            0,
            1
        ]
    )


    return (
        mae,
        rmse,
        accuracy,
        brier,
        loss
    )


# ============================================================
# V1 LINEAR BASELINE BUILDERS
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
# STORAGE
# ============================================================

run_results = []

prediction_rows = []

baseline_results = []


# ============================================================
# ROLLING OUTER VALIDATION
# ============================================================

for validation_year in VALIDATION_YEARS:

    print()

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
    # Everything before the validation year is trainable.
    # --------------------------------------------------------

    outer_train = (

        data[
            data[
                "season"
            ]
            <
            validation_year
        ]

        .copy()
    )


    # --------------------------------------------------------
    # This is the completely unseen outer test season.
    # --------------------------------------------------------

    outer_validation = (

        data[
            data[
                "season"
            ]
            ==
            validation_year
        ]

        .copy()
    )


    # --------------------------------------------------------
    # Hard 2025 safety wall.
    # --------------------------------------------------------

    if (
        outer_train[
            "season"
        ].max()
        >= 2025
    ):

        raise RuntimeError(
            "2025 leakage detected."
        )


    if (
        outer_validation[
            "season"
        ].max()
        >= 2025
    ):

        raise RuntimeError(
            "2025 leakage detected."
        )


    outer_train_win = (

        home_win_target.loc[
            outer_train.index
        ]
    )


    outer_validation_win = (

        home_win_target.loc[
            outer_validation.index
        ]
    )


    # --------------------------------------------------------
    # Previous season becomes the inner validation season.
    # --------------------------------------------------------

    inner_validation_year = (
        validation_year
        -
        1
    )


    inner_train = (

        outer_train[
            outer_train[
                "season"
            ]
            <
            inner_validation_year
        ]

        .copy()
    )


    inner_validation = (

        outer_train[
            outer_train[
                "season"
            ]
            ==
            inner_validation_year
        ]

        .copy()
    )


    inner_train_win = (

        home_win_target.loc[
            inner_train.index
        ]
    )


    inner_validation_win = (

        home_win_target.loc[
            inner_validation.index
        ]
    )


    actual_margin = (

        outer_validation[
            MARGIN_TARGET
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    actual_win = (

        outer_validation_win

        .to_numpy(
            dtype=np.int64
        )
    )


    print(

        f"Inner train:       "
        f"{int(inner_train['season'].min())}-"
        f"{int(inner_train['season'].max())}"
    )


    print(

        f"Inner validation:  "
        f"{inner_validation_year}"
    )


    print(

        f"Final train:       "
        f"{int(outer_train['season'].min())}-"
        f"{int(outer_train['season'].max())}"
    )


    print(

        f"Outer validation:  "
        f"{validation_year} "
        f"({len(outer_validation):,} games)"
    )


    # ========================================================
    # V1 LINEAR BASELINE
    # ========================================================

    x_outer_train = (

        outer_train[
            FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    x_outer_validation = (

        outer_validation[
            FEATURES
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    ridge = build_ridge()


    ridge.fit(

        x_outer_train,

        outer_train[
            MARGIN_TARGET
        ]
    )


    ridge_margin_prediction = (

        ridge.predict(
            x_outer_validation
        )
    )


    logistic = build_logistic()


    logistic.fit(

        x_outer_train,

        outer_train_win
    )


    logistic_probability = (

        logistic

        .predict_proba(
            x_outer_validation
        )[
            :,
            1
        ]
    )


    (
        baseline_mae,
        baseline_rmse,
        baseline_accuracy,
        baseline_brier,
        baseline_logloss
    ) = calculate_metrics(

        actual_margin,
        actual_win,

        ridge_margin_prediction,
        logistic_probability
    )


    baseline_results.append(
        {
            "validation_year":
                validation_year,

            "margin_mae":
                baseline_mae,

            "margin_rmse":
                baseline_rmse,

            "win_accuracy":
                baseline_accuracy,

            "brier":
                baseline_brier,

            "logloss":
                baseline_logloss
        }
    )


    print()

    print(
        "V1 Linear Benchmark"
    )

    print(
        "-" * 100
    )

    print(

        f"MAE {baseline_mae:.3f} | "
        f"ACC {baseline_accuracy:.3f} | "
        f"Brier {baseline_brier:.3f}"
    )


    # ========================================================
    # NEURAL NETWORKS
    # ========================================================

    for (
        architecture_name,
        hidden_sizes
    ) in ARCHITECTURES.items():

        print()

        print(
            architecture_name
        )

        print(
            "-" * 100
        )


        for seed in SEEDS:

            # ------------------------------------------------
            # STEP 1:
            #
            # Select number of epochs using older data.
            # ------------------------------------------------

            (
                best_epoch,
                inner_validation_loss
            ) = select_best_epoch(

                inner_train,
                inner_train_win,

                inner_validation,
                inner_validation_win,

                hidden_sizes,
                seed
            )


            # ------------------------------------------------
            # STEP 2:
            #
            # Start over and train final model using all data
            # before the outer test year.
            # ------------------------------------------------

            (
                margin_prediction,
                win_probability
            ) = train_final_model(

                outer_train,
                outer_train_win,

                outer_validation,

                hidden_sizes,
                best_epoch,
                seed
            )


            (
                mae,
                rmse,
                accuracy,
                brier,
                loss
            ) = calculate_metrics(

                actual_margin,
                actual_win,

                margin_prediction,
                win_probability
            )


            temporary_model = FootballNetwork(

                input_size=len(
                    FEATURES
                ),

                hidden_sizes=hidden_sizes,

                dropout=DROPOUT
            )


            parameters = count_parameters(
                temporary_model
            )


            print(

                f"Seed {seed:>2} | "
                f"Epoch {best_epoch:>3} | "
                f"MAE {mae:.3f} | "
                f"ACC {accuracy:.3f} | "
                f"Brier {brier:.3f}"
            )


            run_results.append(
                {
                    "validation_year":
                        validation_year,

                    "architecture":
                        architecture_name,

                    "seed":
                        seed,

                    "parameters":
                        parameters,

                    "best_epoch":
                        best_epoch,

                    "inner_validation_loss":
                        inner_validation_loss,

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


            # ================================================
            # SAVE GAME-LEVEL PREDICTIONS
            # ================================================

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

                        "architecture":
                            architecture_name,

                        "seed":
                            seed,

                        "actual_margin":
                            actual_margin[
                                position
                            ],

                        "actual_home_win":
                            actual_win[
                                position
                            ],

                        "predicted_margin":
                            margin_prediction[
                                position
                            ],

                        "predicted_home_win_probability":
                            win_probability[
                                position
                            ]
                    }
                )


# ============================================================
# CREATE DATAFRAMES
# ============================================================

run_results = pd.DataFrame(
    run_results
)


predictions = pd.DataFrame(
    prediction_rows
)


baseline_results = pd.DataFrame(
    baseline_results
)


os.makedirs(
    "data/processed",
    exist_ok=True
)


run_results.to_csv(
    RUN_RESULTS_FILE,
    index=False
)


predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# V1 BASELINE AVERAGE
#
# Equal weight for each validation season.
# ============================================================

baseline_average = {

    "margin_mae":
        baseline_results[
            "margin_mae"
        ].mean(),

    "margin_rmse":
        baseline_results[
            "margin_rmse"
        ].mean(),

    "win_accuracy":
        baseline_results[
            "win_accuracy"
        ].mean(),

    "brier":
        baseline_results[
            "brier"
        ].mean(),

    "logloss":
        baseline_results[
            "logloss"
        ].mean()
}


# ============================================================
# EACH SEED'S FOUR-SEASON AVERAGE
#
# This answers:
#
# "If we rerun the entire experiment from another random
# starting point, how much does performance change?"
# ============================================================

seed_results = (

    run_results

    .groupby(
        [
            "architecture",
            "seed"
        ],
        as_index=False
    )

    .agg(

        average_epoch=(
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
        )
    )
)


seed_results.to_csv(
    SEED_SUMMARY_FILE,
    index=False
)


print()

print(
    "=" * 100
)

print(
    "FOUR-SEASON RESULT FOR EACH RANDOM SEED"
)

print(
    "=" * 100
)


print(

    seed_results.to_string(

        index=False,

        formatters={

            "average_epoch":
                lambda value:
                    f"{value:.1f}",

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
# SEED STABILITY SUMMARY
#
# Mean and standard deviation across the five complete
# four-season runs.
# ============================================================

stability_summary = (

    seed_results

    .groupby(
        "architecture",
        as_index=False
    )

    .agg(

        mean_mae=(
            "margin_mae",
            "mean"
        ),

        std_mae=(
            "margin_mae",
            "std"
        ),

        mean_accuracy=(
            "win_accuracy",
            "mean"
        ),

        std_accuracy=(
            "win_accuracy",
            "std"
        ),

        mean_brier=(
            "brier",
            "mean"
        ),

        std_brier=(
            "brier",
            "std"
        ),

        mean_logloss=(
            "logloss",
            "mean"
        ),

        std_logloss=(
            "logloss",
            "std"
        )
    )
)


print()

print(
    "=" * 100
)

print(
    "INDIVIDUAL-SEED STABILITY"
)

print(
    "=" * 100
)


print(

    stability_summary.to_string(

        index=False,

        formatters={

            "mean_mae":
                lambda value:
                    f"{value:.3f}",

            "std_mae":
                lambda value:
                    f"{value:.3f}",

            "mean_accuracy":
                lambda value:
                    f"{value:.3f}",

            "std_accuracy":
                lambda value:
                    f"{value:.4f}",

            "mean_brier":
                lambda value:
                    f"{value:.3f}",

            "std_brier":
                lambda value:
                    f"{value:.4f}",

            "mean_logloss":
                lambda value:
                    f"{value:.3f}",

            "std_logloss":
                lambda value:
                    f"{value:.4f}"
        }
    )
)


# ============================================================
# FIVE-SEED ENSEMBLE
#
# For every game:
#
# average the five margin predictions
# average the five win probabilities
# ============================================================

ensemble_predictions = (

    predictions

    .groupby(
        [
            "game_id",
            "season",
            "architecture"
        ],
        as_index=False
    )

    .agg(

        actual_margin=(
            "actual_margin",
            "first"
        ),

        actual_home_win=(
            "actual_home_win",
            "first"
        ),

        predicted_margin=(
            "predicted_margin",
            "mean"
        ),

        predicted_home_win_probability=(
            "predicted_home_win_probability",
            "mean"
        )
    )
)


# ============================================================
# ENSEMBLE RESULTS BY SEASON
# ============================================================

ensemble_season_rows = []


for (
    architecture_name,
    architecture_data
) in ensemble_predictions.groupby(
    "architecture"
):

    for (
        season,
        season_data
    ) in architecture_data.groupby(
        "season"
    ):

        (
            mae,
            rmse,
            accuracy,
            brier,
            loss
        ) = calculate_metrics(

            season_data[
                "actual_margin"
            ],

            season_data[
                "actual_home_win"
            ],

            season_data[
                "predicted_margin"
            ],

            season_data[
                "predicted_home_win_probability"
            ]
        )


        ensemble_season_rows.append(
            {
                "architecture":
                    architecture_name,

                "validation_year":
                    season,

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


ensemble_by_season = pd.DataFrame(
    ensemble_season_rows
)


ensemble_by_season.to_csv(
    ENSEMBLE_SEASON_FILE,
    index=False
)


# ============================================================
# AVERAGE ENSEMBLE RESULT
#
# Each season receives equal weight.
# ============================================================

ensemble_summary = (

    ensemble_by_season

    .groupby(
        "architecture",
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


ensemble_summary.to_csv(
    ENSEMBLE_SUMMARY_FILE,
    index=False
)


print()

print(
    "=" * 100
)

print(
    "FIVE-SEED ENSEMBLE — BY SEASON"
)

print(
    "=" * 100
)


print(

    ensemble_by_season.to_string(

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


print()

print(
    "=" * 100
)

print(
    "FIVE-SEED ENSEMBLE — AVERAGE"
)

print(
    "=" * 100
)


print(

    ensemble_summary.to_string(

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
# DIRECT COMPARISON TO V1
# ============================================================

print()

print(
    "=" * 100
)

print(
    "V1 LINEAR BENCHMARK"
)

print(
    "=" * 100
)


print(

    f"Margin MAE:   "
    f"{baseline_average['margin_mae']:.3f}"
)

print(

    f"Margin RMSE:  "
    f"{baseline_average['margin_rmse']:.3f}"
)

print(

    f"Win Accuracy: "
    f"{baseline_average['win_accuracy']:.3f}"
)

print(

    f"Brier:        "
    f"{baseline_average['brier']:.3f}"
)

print(

    f"LogLoss:      "
    f"{baseline_average['logloss']:.3f}"
)


print()

print(
    "=" * 100
)

print(
    "ENSEMBLE CHANGE VS V1 LINEAR"
)

print(
    "=" * 100
)


for _, row in ensemble_summary.iterrows():

    print()

    print(
        row[
            "architecture"
        ]
    )


    print(

        f"MAE change:       "
        f"{row['margin_mae'] - baseline_average['margin_mae']:+.3f}"
    )

    print(

        f"RMSE change:      "
        f"{row['margin_rmse'] - baseline_average['margin_rmse']:+.3f}"
    )

    print(

        f"Accuracy change:  "
        f"{row['win_accuracy'] - baseline_average['win_accuracy']:+.4f}"
    )

    print(

        f"Brier change:     "
        f"{row['brier'] - baseline_average['brier']:+.4f}"
    )

    print(

        f"LogLoss change:   "
        f"{row['logloss'] - baseline_average['logloss']:+.4f}"
    )


# ============================================================
# FINISH
# ============================================================

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
    RUN_RESULTS_FILE
)

print(
    SEED_SUMMARY_FILE
)

print(
    PREDICTIONS_FILE
)

print(
    ENSEMBLE_SEASON_FILE
)

print(
    ENSEMBLE_SUMMARY_FILE
)


print()

print(
    "2025 was not trained on, validated on, "
    "or scored."
)