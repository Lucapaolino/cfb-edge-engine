import argparse
import copy
import hashlib
import json
import pickle
import random

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# ============================================================
# PURPOSE
#
# Train and freeze the research neural-network ensemble that
# will be used for prospective 2026 inference.
#
# IMPORTANT:
#
# 2025 is now development history.
#
# Procedure:
#
#   2018-2024
#       ↓
#   choose epoch count against 2025
#       ↓
#   rebuild from scratch
#       ↓
#   train 2018-2025 for chosen epoch count
#       ↓
#   save frozen artifact
#
# 2026 is NOT used for training or model selection.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

TRAIN_END_SEASON = 2024

EPOCH_SELECTION_SEASON = 2025

FINAL_TRAIN_END_SEASON = 2025


HIDDEN_SIZES = [
    64,
    32
]


ENSEMBLE_SEEDS = [
    42,
    43,
    44,
    45,
    46
]


BATCH_SIZE = 64

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

DROPOUT = 0.20

MAX_EPOCHS = 400

PATIENCE = 30

MIN_DELTA = 0.0001


MARGIN_TARGET = (
    "target_home_margin"
)

WIN_TARGET = (
    "target_home_win"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


FEATURE_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "features_v2.csv"
)


MODEL_DIRECTORY = (
    PROJECT_ROOT
    /
    "models"
    /
    "cfb_nn_2026_v1"
)


FEATURE_NAMES_FILE = (
    MODEL_DIRECTORY
    /
    "feature_names.json"
)


MEDIANS_FILE = (
    MODEL_DIRECTORY
    /
    "imputer_medians.json"
)


SCALER_FILE = (
    MODEL_DIRECTORY
    /
    "scaler.pkl"
)


TARGET_NORMALIZATION_FILE = (
    MODEL_DIRECTORY
    /
    "target_normalization.json"
)


METADATA_FILE = (
    MODEL_DIRECTORY
    /
    "metadata.json"
)


# ============================================================
# DEVICE
#
# Deliberately use CPU for the frozen artifact.
#
# This network is tiny. CPU training is fast enough and gives
# us a cleaner reproducibility story than changing between
# CPU / MPS depending on the machine.
# ============================================================

DEVICE = torch.device(
    "cpu"
)


# ============================================================
# HELPERS
# ============================================================

def header(text):

    print()

    print(
        "=" * 110
    )

    print(
        text
    )

    print(
        "=" * 110
    )


def utc_now():

    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )


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


def sha256_file(path):

    digest = hashlib.sha256()


    with open(
        path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )


            if not chunk:

                break


            digest.update(
                chunk
            )


    return digest.hexdigest()


# ============================================================
# NETWORK
#
# Mirrors the original NN Small architecture.
#
# 54 inputs
#     ↓
# 64
#     ↓
# 32
#     ↓
# margin head
# win-logit head
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


        self.shared = nn.Sequential(
            *layers
        )


        self.margin_head = nn.Linear(
            previous_size,
            1
        )


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
# FEATURE CONTRACT
#
# Reproduce exactly what train_neural_network_v1.py did:
#
# - first 192 feature-file columns
# - every diff_pregame_* among those
# - seven context features
# - sorted unique list
# ============================================================

def get_feature_contract(
    data
):

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

            f"Expected exactly 54 features. "
            f"Found {len(features)}."
        )


    return features


# ============================================================
# PREPROCESSING
#
# Mirrors the original script:
#
# 1. numeric coercion
# 2. training-only median
# 3. median fallback to zero only if the entire training
#    column has no usable values
# 4. StandardScaler fit only on training data
# ============================================================

def numeric_feature_frame(
    frame,
    features
):

    return (

        frame[
            features
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


def fit_preprocessor(
    frame,
    features
):

    numeric = numeric_feature_frame(
        frame,
        features
    )


    medians = (

        numeric
        .median()
        .fillna(
            0.0
        )
    )


    imputed = numeric.fillna(
        medians
    )


    scaler = StandardScaler()


    scaled = (

        scaler

        .fit_transform(
            imputed
        )

        .astype(
            np.float32
        )
    )


    return (
        medians,
        scaler,
        scaled
    )


def transform_features(
    frame,
    features,
    medians,
    scaler
):

    numeric = numeric_feature_frame(
        frame,
        features
    )


    imputed = numeric.fillna(
        medians
    )


    scaled = (

        scaler

        .transform(
            imputed
        )

        .astype(
            np.float32
        )
    )


    return scaled


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def margin_normalization(
    frame
):

    margin_mean = float(

        frame[
            MARGIN_TARGET
        ].mean()
    )


    margin_std = float(

        frame[
            MARGIN_TARGET
        ]

        .std(
            ddof=0
        )
    )


    if margin_std < 0.000001:

        margin_std = 1.0


    return (
        margin_mean,
        margin_std
    )


# ============================================================
# DATA LOADER
# ============================================================

def make_loader(
    x,
    y_margin,
    y_win,
    seed
):

    dataset = TensorDataset(

        torch.tensor(
            x,
            dtype=torch.float32
        ),

        torch.tensor(
            y_margin,
            dtype=torch.float32
        ),

        torch.tensor(
            y_win,
            dtype=torch.float32
        )
    )


    generator = torch.Generator()


    generator.manual_seed(
        seed
    )


    return DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        generator=generator
    )


# ============================================================
# LOSS FUNCTIONS
# ============================================================

MARGIN_LOSS_FUNCTION = (
    nn.SmoothL1Loss()
)


WIN_LOSS_FUNCTION = (
    nn.BCEWithLogitsLoss()
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer
):

    model.train()


    for (
        batch_x,
        batch_margin,
        batch_win
    ) in loader:

        batch_x = batch_x.to(
            DEVICE
        )


        batch_margin = batch_margin.to(
            DEVICE
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
            MARGIN_LOSS_FUNCTION(

                predicted_margin,

                batch_margin
            )
        )


        win_loss = (
            WIN_LOSS_FUNCTION(

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


# ============================================================
# EVALUATE STANDARDIZED LOSS
# ============================================================

def validation_loss(
    model,
    x,
    y_margin,
    y_win
):

    model.eval()


    x_tensor = torch.tensor(
        x,
        dtype=torch.float32,
        device=DEVICE
    )


    margin_tensor = torch.tensor(
        y_margin,
        dtype=torch.float32,
        device=DEVICE
    )


    win_tensor = torch.tensor(
        y_win,
        dtype=torch.float32,
        device=DEVICE
    )


    with torch.no_grad():

        (
            predicted_margin,
            predicted_win_logit
        ) = model(
            x_tensor
        )


        margin_loss = (
            MARGIN_LOSS_FUNCTION(

                predicted_margin,

                margin_tensor
            )
        )


        win_loss = (
            WIN_LOSS_FUNCTION(

                predicted_win_logit,

                win_tensor
            )
        )


        total = (

            margin_loss

            +

            win_loss
        )


    return float(
        total.item()
    )


# ============================================================
# RAW VALIDATION METRICS
# ============================================================

def prediction_metrics(
    model,
    x,
    actual_margin,
    actual_win,
    margin_mean,
    margin_std
):

    model.eval()


    x_tensor = torch.tensor(
        x,
        dtype=torch.float32,
        device=DEVICE
    )


    with torch.no_grad():

        (
            standardized_margin,
            win_logit
        ) = model(
            x_tensor
        )


        predicted_margin = (

            standardized_margin
            .cpu()
            .numpy()

            *

            margin_std

            +

            margin_mean
        )


        predicted_probability = (

            torch.sigmoid(
                win_logit
            )

            .cpu()
            .numpy()
        )


    actual_margin = np.asarray(
        actual_margin,
        dtype=float
    )


    actual_win = np.asarray(
        actual_win,
        dtype=float
    )


    mae = float(

        np.mean(

            np.abs(

                predicted_margin

                -

                actual_margin
            )
        )
    )


    brier = float(

        np.mean(

            (
                predicted_probability

                -

                actual_win
            )

            ** 2
        )
    )


    return {

        "margin_mae":
            mae,

        "win_brier":
            brier
    }


# ============================================================
# SELECT EPOCH COUNT
#
# 2018-2024 → training
# 2025      → development validation
#
# 2025 is NOT a prospective holdout anymore.
# ============================================================

def select_epoch_count(
    training_frame,
    validation_frame,
    features,
    seed
):

    set_seed(
        seed
    )


    (
        medians,
        scaler,
        x_train
    ) = fit_preprocessor(

        training_frame,

        features
    )


    x_validation = transform_features(

        validation_frame,

        features,

        medians,

        scaler
    )


    (
        margin_mean,
        margin_std
    ) = margin_normalization(
        training_frame
    )


    y_train_margin = (

        (
            training_frame[
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
            validation_frame[
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

        training_frame[
            WIN_TARGET
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    y_validation_win = (

        validation_frame[
            WIN_TARGET
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    loader = make_loader(

        x_train,

        y_train_margin,

        y_train_win,

        seed
    )


    model = FootballNetwork(

        input_size=len(
            features
        ),

        hidden_sizes=HIDDEN_SIZES,

        dropout=DROPOUT

    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    best_validation_loss = float(
        "inf"
    )


    best_epoch = 1


    epochs_without_improvement = 0


    best_state = None


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        train_epoch(

            model,

            loader,

            optimizer
        )


        current_validation_loss = (
            validation_loss(

                model,

                x_validation,

                y_validation_margin,

                y_validation_win
            )
        )


        if (

            current_validation_loss

            <

            (
                best_validation_loss

                -

                MIN_DELTA
            )
        ):

            best_validation_loss = (
                current_validation_loss
            )


            best_epoch = epoch


            epochs_without_improvement = 0


            best_state = copy.deepcopy(
                model.state_dict()
            )


        else:

            epochs_without_improvement += 1


        if (

            epochs_without_improvement

            >=
            PATIENCE
        ):

            break


    if best_state is None:

        raise RuntimeError(
            "Early stopping failed to capture "
            "a valid model state."
        )


    model.load_state_dict(
        best_state
    )


    metrics = prediction_metrics(

        model,

        x_validation,

        validation_frame[
            MARGIN_TARGET
        ].to_numpy(
            dtype=float
        ),

        validation_frame[
            WIN_TARGET
        ].to_numpy(
            dtype=float
        ),

        margin_mean,

        margin_std
    )


    return {

        "seed":
            seed,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_validation_loss":
            float(
                best_validation_loss
            ),

        "validation_margin_mae":
            float(
                metrics[
                    "margin_mae"
                ]
            ),

        "validation_win_brier":
            float(
                metrics[
                    "win_brier"
                ]
            )
    }


# ============================================================
# FINAL TRAINING
#
# Train from scratch on 2018-2025 using the epoch count chosen
# above.
# ============================================================

def train_final_model(
    final_frame,
    features,
    x_final,
    margin_mean,
    margin_std,
    seed,
    epochs
):

    set_seed(
        seed
    )


    y_margin = (

        (
            final_frame[
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


    y_win = (

        final_frame[
            WIN_TARGET
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    loader = make_loader(

        x_final,

        y_margin,

        y_win,

        seed
    )


    model = FootballNetwork(

        input_size=len(
            features
        ),

        hidden_sizes=HIDDEN_SIZES,

        dropout=DROPOUT

    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    for epoch in range(
        1,
        epochs + 1
    ):

        train_epoch(

            model,

            loader,

            optimizer
        )


    model.eval()


    return model


# ============================================================
# ARTIFACT SMOKE TEST
# ============================================================

def smoke_test(
    model_paths,
    final_frame,
    features,
    medians,
    scaler,
    margin_mean,
    margin_std
):

    header(
        "ARTIFACT SMOKE TEST"
    )


    sample = final_frame.head(
        5
    )


    x_sample = transform_features(

        sample,

        features,

        medians,

        scaler
    )


    x_tensor = torch.tensor(

        x_sample,

        dtype=torch.float32
    )


    for path in model_paths:

        checkpoint = torch.load(

            path,

            map_location="cpu"
        )


        model = FootballNetwork(

            input_size=checkpoint[
                "input_size"
            ],

            hidden_sizes=checkpoint[
                "hidden_sizes"
            ],

            dropout=checkpoint[
                "dropout"
            ]
        )


        model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )


        model.eval()


        with torch.no_grad():

            (
                standardized_margin,
                win_logit
            ) = model(
                x_tensor
            )


            margin = (

                standardized_margin

                *

                margin_std

                +

                margin_mean
            )


            win_probability = torch.sigmoid(
                win_logit
            )


        if not torch.isfinite(
            margin
        ).all():

            raise RuntimeError(

                f"Non-finite margin output "
                f"from {path.name}"
            )


        if not torch.isfinite(
            win_probability
        ).all():

            raise RuntimeError(

                f"Non-finite win probability "
                f"from {path.name}"
            )


        print(

            f"{path.name:24} "
            f"PASS"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--force",

        action="store_true",

        help=(
            "Allow replacement of an existing frozen "
            "cfb_nn_2026_v1 artifact."
        )
    )


    args = parser.parse_args()


    print(
        "TRAIN AND FREEZE CFB NN 2026 — V1"
    )

    print(
        "=" * 110
    )


    print(
        f"Device:             {DEVICE}"
    )


    print(
        f"Architecture:       "
        f"54 → 64 → 32 → 2 heads"
    )


    print(
        f"Ensemble models:    "
        f"{len(ENSEMBLE_SEEDS)}"
    )


    print(
        f"Seeds:              "
        f"{ENSEMBLE_SEEDS}"
    )


    print(
        f"Epoch selection:    "
        f"2018-{TRAIN_END_SEASON} → "
        f"{EPOCH_SELECTION_SEASON}"
    )


    print(
        f"Final training:     "
        f"2018-{FINAL_TRAIN_END_SEASON}"
    )


    print(
        "2026 usage:         prospective inference only"
    )


    if not FEATURE_FILE.exists():

        raise RuntimeError(

            f"Feature file not found: "
            f"{FEATURE_FILE}"
        )


    if (

        METADATA_FILE.exists()

        and

        not args.force
    ):

        raise RuntimeError(

            "\nA frozen model already exists at:\n"
            f"{MODEL_DIRECTORY}\n\n"
            "The script will not silently overwrite it.\n"
            "If you intentionally want to replace it, run:\n\n"
            "python src/train_and_freeze_nn_2026_v1.py --force"
        )


    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    header(
        "LOAD HISTORICAL FEATURE DATA"
    )


    data = pd.read_csv(
        FEATURE_FILE
    )


    print(
        f"Rows loaded:    "
        f"{len(data):,}"
    )


    print(
        f"Columns loaded: "
        f"{len(data.columns):,}"
    )


    required_columns = [

        "season",

        MARGIN_TARGET,

        WIN_TARGET
    ]


    for column in required_columns:

        if column not in data.columns:

            raise RuntimeError(

                f"Required column missing: "
                f"{column}"
            )


    data[
        "season"
    ] = pd.to_numeric(

        data[
            "season"
        ],

        errors="coerce"
    )


    # ========================================================
    # EXACT FEATURES
    # ========================================================

    features = get_feature_contract(
        data
    )


    print(
        f"Exact model features: "
        f"{len(features)}"
    )


    # ========================================================
    # TARGET CHECK
    # ========================================================

    target_data = data[

        data[
            "season"
        ].between(
            2018,
            FINAL_TRAIN_END_SEASON
        )

    ].copy()


    missing_targets = (

        target_data[
            [
                MARGIN_TARGET,
                WIN_TARGET
            ]
        ]

        .isna()

        .any(
            axis=1
        )

        .sum()
    )


    if missing_targets > 0:

        raise RuntimeError(

            f"Historical training population contains "
            f"{missing_targets:,} rows with missing targets."
        )


    # ========================================================
    # SPLITS
    # ========================================================

    epoch_training = data[

        data[
            "season"
        ]
        <=
        TRAIN_END_SEASON

    ].copy()


    epoch_validation = data[

        data[
            "season"
        ]
        ==
        EPOCH_SELECTION_SEASON

    ].copy()


    final_training = data[

        data[
            "season"
        ]
        <=
        FINAL_TRAIN_END_SEASON

    ].copy()


    print()

    print(
        f"Epoch-selection training rows: "
        f"{len(epoch_training):,}"
    )


    print(
        f"2025 development rows:         "
        f"{len(epoch_validation):,}"
    )


    print(
        f"Final 2018-2025 rows:           "
        f"{len(final_training):,}"
    )


    if len(
        epoch_validation
    ) == 0:

        raise RuntimeError(
            "No 2025 development rows found."
        )


    # ========================================================
    # EPOCH SELECTION
    # ========================================================

    header(
        "SELECT EPOCH COUNT FOR EACH ENSEMBLE MEMBER"
    )


    epoch_results = []


    for seed in ENSEMBLE_SEEDS:

        print()

        print(
            f"Seed {seed}: training..."
        )


        result = select_epoch_count(

            epoch_training,

            epoch_validation,

            features,

            seed
        )


        epoch_results.append(
            result
        )


        print(
            f"  best epoch:       "
            f"{result['best_epoch']}"
        )


        print(
            f"  validation loss:  "
            f"{result['best_validation_loss']:.6f}"
        )


        print(
            f"  2025 margin MAE:  "
            f"{result['validation_margin_mae']:.3f}"
        )


        print(
            f"  2025 win Brier:   "
            f"{result['validation_win_brier']:.4f}"
        )


    # ========================================================
    # FINAL PREPROCESSING
    #
    # This is what every 2026 live row will use.
    # ========================================================

    header(
        "FIT FINAL 2018-2025 PREPROCESSING"
    )


    (
        final_medians,
        final_scaler,
        x_final
    ) = fit_preprocessor(

        final_training,

        features
    )


    (
        final_margin_mean,
        final_margin_std
    ) = margin_normalization(
        final_training
    )


    print(
        f"Margin mean: "
        f"{final_margin_mean:.6f}"
    )


    print(
        f"Margin std:  "
        f"{final_margin_std:.6f}"
    )


    # ========================================================
    # SAVE FEATURE CONTRACT
    # ========================================================

    with open(
        FEATURE_NAMES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            features,
            file,
            indent=2
        )


    # ========================================================
    # SAVE MEDIANS
    # ========================================================

    with open(
        MEDIANS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            {
                feature:
                    float(
                        final_medians[
                            feature
                        ]
                    )

                for feature
                in features
            },

            file,

            indent=2,

            sort_keys=True
        )


    # ========================================================
    # SAVE SCALER
    # ========================================================

    with open(
        SCALER_FILE,
        "wb"
    ) as file:

        pickle.dump(
            final_scaler,
            file
        )


    # ========================================================
    # SAVE TARGET NORMALIZATION
    # ========================================================

    with open(
        TARGET_NORMALIZATION_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            {
                "margin_mean":
                    final_margin_mean,

                "margin_std":
                    final_margin_std
            },

            file,

            indent=2
        )


    # ========================================================
    # FINAL ENSEMBLE TRAINING
    # ========================================================

    header(
        "TRAIN FINAL 2018-2025 ENSEMBLE"
    )


    model_paths = []


    result_by_seed = {

        result[
            "seed"
        ]:
            result

        for result
        in epoch_results
    }


    for seed in ENSEMBLE_SEEDS:

        epochs = result_by_seed[
            seed
        ][
            "best_epoch"
        ]


        print()

        print(
            f"Seed {seed}: "
            f"training final model for "
            f"{epochs} epochs..."
        )


        model = train_final_model(

            final_training,

            features,

            x_final,

            final_margin_mean,

            final_margin_std,

            seed,

            epochs
        )


        model_path = (

            MODEL_DIRECTORY

            /

            f"model_seed_{seed}.pt"
        )


        checkpoint = {

            "artifact_version":
                "cfb_nn_2026_v1",

            "seed":
                seed,

            "epochs":
                epochs,

            "input_size":
                len(
                    features
                ),

            "hidden_sizes":
                HIDDEN_SIZES,

            "dropout":
                DROPOUT,

            "state_dict":
                {
                    key:
                        value.detach().cpu()

                    for key, value
                    in model.state_dict().items()
                }
        }


        torch.save(
            checkpoint,
            model_path
        )


        model_paths.append(
            model_path
        )


        print(
            f"  saved: "
            f"{model_path.name}"
        )


    # ========================================================
    # METADATA
    # ========================================================

    created_at = utc_now()


    metadata = {

        "artifact_version":
            "cfb_nn_2026_v1",

        "created_utc":
            created_at.isoformat(),

        "status":
            "RESEARCH_ONLY",

        "prospective_validation_season":
            2026,

        "training_seasons":
            [
                2018,
                2019,
                2020,
                2021,
                2022,
                2023,
                2024,
                2025
            ],

        "epoch_selection": {

            "training_through":
                TRAIN_END_SEASON,

            "validation_season":
                EPOCH_SELECTION_SEASON,

            "method":
                "early stopping"
        },

        "feature_file":
            str(
                FEATURE_FILE
            ),

        "feature_file_sha256":
            sha256_file(
                FEATURE_FILE
            ),

        "feature_count":
            len(
                features
            ),

        "architecture": {

            "input_size":
                len(
                    features
                ),

            "hidden_sizes":
                HIDDEN_SIZES,

            "outputs":
                [
                    "standardized_home_margin",
                    "home_win_logit"
                ],

            "dropout":
                DROPOUT
        },

        "training": {

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "max_epoch_selection":
                MAX_EPOCHS,

            "patience":
                PATIENCE,

            "min_delta":
                MIN_DELTA,

            "margin_loss":
                "SmoothL1Loss",

            "win_loss":
                "BCEWithLogitsLoss",

            "optimizer":
                "AdamW"
        },

        "ensemble_seeds":
            ENSEMBLE_SEEDS,

        "epoch_results":
            epoch_results,

        "final_training_rows":
            len(
                final_training
            ),

        "margin_normalization": {

            "mean":
                final_margin_mean,

            "std":
                final_margin_std
        },

        "preprocessing": {

            "missing_values":
                "training median",

            "all_missing_fallback":
                0.0,

            "scaling":
                "StandardScaler fit on 2018-2025"
        },

        "software": {

            "python_torch":
                torch.__version__,

            "scikit_learn":
                sklearn.__version__,

            "numpy":
                np.__version__,

            "pandas":
                pd.__version__
        },

        "notes": [

            (
                "This artifact is a research model, "
                "not a validated profitable betting system."
            ),

            (
                "2025 was used as development history "
                "for epoch-count selection."
            ),

            (
                "2026 must remain prospective for "
                "evaluation."
            ),

            (
                "Raw sigmoid output should not be "
                "described as a proven fair market "
                "probability."
            )
        ]
    }


    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


    # ========================================================
    # SMOKE TEST
    # ========================================================

    smoke_test(

        model_paths,

        final_training,

        features,

        final_medians,

        final_scaler,

        final_margin_mean,

        final_margin_std
    )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    header(
        "FROZEN MODEL ARTIFACT COMPLETE"
    )


    print(
        f"Artifact directory:"
    )


    print(
        f"  {MODEL_DIRECTORY}"
    )


    print()

    print(
        "Files:"
    )


    for path in sorted(
        MODEL_DIRECTORY.iterdir()
    ):

        print(
            f"  {path.name}"
        )


    print()

    print(
        "Ensemble:"
    )


    for result in epoch_results:

        print(

            f"  seed "
            f"{result['seed']} | "

            f"epochs "
            f"{result['best_epoch']} | "

            f"2025 MAE "
            f"{result['validation_margin_mae']:.3f} | "

            f"Brier "
            f"{result['validation_win_brier']:.4f}"
        )


    print()

    print(
        "STATUS: RESEARCH ONLY"
    )


    print(
        "2026 has not been used for training."
    )


    print()

    print(
        "Next step:"
    )


    print(
        "  Run frozen ensemble inference on "
        "live_features_2026_v1.csv."
    )


if __name__ == "__main__":

    main()