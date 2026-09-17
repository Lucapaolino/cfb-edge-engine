import json
import pickle

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# PURPOSE
#
# Load the frozen 2026 research neural-network ensemble and
# generate predictions for the current live feature slate.
#
# USER-FACING PRESENTATION:
#
#     Alabama -10.7
#
# rather than:
#
#     home_margin = -10.7
#
# The neural network itself is NOT changed.
#
# NO TRAINING OCCURS HERE.
# NO APIs ARE CALLED HERE.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

ARTIFACT_VERSION = "cfb_nn_2026_v1"

PREDICTION_VERSION = (
    "live_nn_predictions_2026_v2"
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


MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / ARTIFACT_VERSION
)


LIVE_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_features_2026_v1.csv"
)


OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_nn_predictions_2026_v1.csv"
)


OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_nn_predictions_2026_v1.json"
)


FEATURE_NAMES_FILE = (
    MODEL_DIRECTORY
    / "feature_names.json"
)


MEDIANS_FILE = (
    MODEL_DIRECTORY
    / "imputer_medians.json"
)


SCALER_FILE = (
    MODEL_DIRECTORY
    / "scaler.pkl"
)


TARGET_NORMALIZATION_FILE = (
    MODEL_DIRECTORY
    / "target_normalization.json"
)


METADATA_FILE = (
    MODEL_DIRECTORY
    / "metadata.json"
)


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


# ============================================================
# NETWORK
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
# REQUIRED FILES
# ============================================================

required_files = [

    FEATURE_NAMES_FILE,

    MEDIANS_FILE,

    SCALER_FILE,

    TARGET_NORMALIZATION_FILE,

    METADATA_FILE,

    LIVE_FEATURE_FILE
]


for path in required_files:

    if not path.exists():

        raise RuntimeError(

            f"Required file missing:\n"
            f"{path}"
        )


# ============================================================
# START
# ============================================================

prediction_time = utc_now()


print(
    "PREDICT LIVE 2026 NN — V2 PRESENTATION"
)

print(
    "=" * 110
)


print(
    f"Prediction UTC: "
    f"{prediction_time.isoformat()}"
)


print(
    f"Artifact:       "
    f"{ARTIFACT_VERSION}"
)


print(
    "Presentation:   SPORTSBOOK STYLE"
)


print(
    "Training:       NONE"
)


print(
    "API calls:      NONE"
)


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(
        file
    )


if (
    metadata.get(
        "artifact_version"
    )
    !=
    ARTIFACT_VERSION
):

    raise RuntimeError(
        "Artifact version mismatch."
    )


# ============================================================
# LOAD FEATURE CONTRACT
# ============================================================

with open(
    FEATURE_NAMES_FILE,
    "r",
    encoding="utf-8"
) as file:

    features = json.load(
        file
    )


if len(
    features
) != 54:

    raise RuntimeError(

        f"Expected 54 model features. "
        f"Found {len(features)}."
    )


# ============================================================
# LOAD PREPROCESSING
# ============================================================

with open(
    MEDIANS_FILE,
    "r",
    encoding="utf-8"
) as file:

    median_dictionary = json.load(
        file
    )


medians = pd.Series(

    {
        feature:
            float(
                median_dictionary[
                    feature
                ]
            )

        for feature in features
    }
)


with open(
    SCALER_FILE,
    "rb"
) as file:

    scaler = pickle.load(
        file
    )


with open(
    TARGET_NORMALIZATION_FILE,
    "r",
    encoding="utf-8"
) as file:

    target_normalization = json.load(
        file
    )


margin_mean = float(

    target_normalization[
        "margin_mean"
    ]
)


margin_std = float(

    target_normalization[
        "margin_std"
    ]
)


# ============================================================
# LOAD LIVE FEATURES
# ============================================================

header(
    "LOAD LIVE FEATURE SLATE"
)


live = pd.read_csv(
    LIVE_FEATURE_FILE
)


print(
    f"Games loaded: "
    f"{len(live):,}"
)


if len(
    live
) == 0:

    raise RuntimeError(
        "No live games found."
    )


missing_columns = [

    feature

    for feature in features

    if feature not in live.columns
]


if missing_columns:

    raise RuntimeError(

        "Missing required model features:\n"

        +

        "\n".join(
            missing_columns
        )
    )


# ============================================================
# PREPROCESS
# ============================================================

raw_numeric = (

    live[
        features
    ]

    .apply(
        pd.to_numeric,
        errors="coerce"
    )
)


imputed = raw_numeric.fillna(
    medians
)


if imputed.isna().any().any():

    raise RuntimeError(

        "NaN values remain after "
        "frozen median imputation."
    )


x = (

    scaler

    .transform(
        imputed
    )

    .astype(
        np.float32
    )
)


if not np.isfinite(
    x
).all():

    raise RuntimeError(
        "Non-finite scaled model input."
    )


x_tensor = torch.tensor(

    x,

    dtype=torch.float32,

    device=DEVICE
)


# ============================================================
# LOAD ENSEMBLE
# ============================================================

expected_seeds = metadata.get(
    "ensemble_seeds",
    []
)


if not expected_seeds:

    raise RuntimeError(
        "No ensemble seeds in metadata."
    )


margin_predictions = []

win_predictions = []


header(
    "RUN FROZEN ENSEMBLE"
)


for seed in expected_seeds:

    model_path = (

        MODEL_DIRECTORY

        /

        f"model_seed_{seed}.pt"
    )


    if not model_path.exists():

        raise RuntimeError(

            f"Missing frozen model: "
            f"{model_path}"
        )


    checkpoint = torch.load(

        model_path,

        map_location=DEVICE
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

    ).to(
        DEVICE
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


        projected_margin = (

            standardized_margin

            *

            margin_std

            +

            margin_mean
        )


        home_win_probability = (

            torch.sigmoid(
                win_logit
            )
        )


    margin_predictions.append(

        projected_margin
        .cpu()
        .numpy()
    )


    win_predictions.append(

        home_win_probability
        .cpu()
        .numpy()
    )


    print(
        f"Seed {seed}: PASS"
    )


# ============================================================
# ENSEMBLE MATRICES
# ============================================================

margin_matrix = np.column_stack(
    margin_predictions
)


win_matrix = np.column_stack(
    win_predictions
)


# ============================================================
# RAW INTERNAL MODEL OUTPUTS
#
# Keep these for math/debugging.
#
# They should NOT be the primary GUI presentation.
# ============================================================

live[
    "nn_projected_home_margin"
] = np.mean(

    margin_matrix,

    axis=1
)


live[
    "nn_home_win_probability_raw"
] = np.mean(

    win_matrix,

    axis=1
)


live[
    "nn_margin_std"
] = np.std(

    margin_matrix,

    axis=1,

    ddof=0
)


live[
    "nn_win_probability_std"
] = np.std(

    win_matrix,

    axis=1,

    ddof=0
)


live[
    "nn_margin_range"
] = (

    np.max(
        margin_matrix,
        axis=1
    )

    -

    np.min(
        margin_matrix,
        axis=1
    )
)


live[
    "nn_win_probability_range"
] = (

    np.max(
        win_matrix,
        axis=1
    )

    -

    np.min(
        win_matrix,
        axis=1
    )
)


# ============================================================
# SPORTSBOOK-STYLE MODEL PRESENTATION
#
# Examples:
#
# Internal:
#   Alabama @ Kentucky
#   home margin = -10.66
#
# Display:
#   Alabama -10.7
#
#
# Internal:
#   Arizona @ BYU
#   home margin = +4.94
#
# Display:
#   BYU -4.9
# ============================================================

model_favorites = []

model_underdogs = []

model_spreads = []

favorite_probabilities = []

underdog_probabilities = []

spread_displays = []


for _, row in live.iterrows():

    home_team = str(
        row[
            "home_team"
        ]
    )


    away_team = str(
        row[
            "away_team"
        ]
    )


    margin = float(

        row[
            "nn_projected_home_margin"
        ]
    )


    home_probability = float(

        row[
            "nn_home_win_probability_raw"
        ]
    )


    away_probability = (
        1.0
        -
        home_probability
    )


    # --------------------------------------------------------
    # HOME TEAM FAVORED
    # --------------------------------------------------------

    if margin > 0:

        favorite = home_team

        underdog = away_team

        spread = (
            -abs(
                margin
            )
        )

        favorite_probability = (
            home_probability
        )

        underdog_probability = (
            away_probability
        )


    # --------------------------------------------------------
    # AWAY TEAM FAVORED
    # --------------------------------------------------------

    elif margin < 0:

        favorite = away_team

        underdog = home_team

        spread = (
            -abs(
                margin
            )
        )

        favorite_probability = (
            away_probability
        )

        underdog_probability = (
            home_probability
        )


    # --------------------------------------------------------
    # EXACT PICK'EM
    # --------------------------------------------------------

    else:

        favorite = "PICK'EM"

        underdog = "PICK'EM"

        spread = 0.0

        favorite_probability = 0.50

        underdog_probability = 0.50


    model_favorites.append(
        favorite
    )


    model_underdogs.append(
        underdog
    )


    model_spreads.append(
        spread
    )


    favorite_probabilities.append(
        favorite_probability
    )


    underdog_probabilities.append(
        underdog_probability
    )


    if favorite == "PICK'EM":

        spread_displays.append(
            "PICK'EM"
        )


    else:

        spread_displays.append(

            f"{favorite} "
            f"{spread:.1f}"
        )


live[
    "model_favorite"
] = model_favorites


live[
    "model_underdog"
] = model_underdogs


live[
    "model_spread"
] = model_spreads


live[
    "model_spread_display"
] = spread_displays


live[
    "model_favorite_win_probability"
] = favorite_probabilities


live[
    "model_underdog_win_probability"
] = underdog_probabilities


live[
    "model_favorite_win_probability_pct"
] = (

    live[
        "model_favorite_win_probability"
    ]

    *

    100.0
)


# ============================================================
# MODEL VOTING
# ============================================================

live[
    "nn_home_votes"
] = (

    win_matrix

    >=
    0.50

).sum(
    axis=1
)


live[
    "nn_away_votes"
] = (

    len(
        expected_seeds
    )

    -

    live[
        "nn_home_votes"
    ]
)


# ============================================================
# MARGIN HEAD / WIN HEAD AGREEMENT
# ============================================================

live[
    "nn_margin_head_pick"
] = np.where(

    live[
        "nn_projected_home_margin"
    ]
    >
    0,

    live[
        "home_team"
    ],

    live[
        "away_team"
    ]
)


live[
    "nn_win_head_pick"
] = np.where(

    live[
        "nn_home_win_probability_raw"
    ]
    >=
    0.50,

    live[
        "home_team"
    ],

    live[
        "away_team"
    ]
)


live[
    "nn_heads_agree"
] = (

    live[
        "nn_margin_head_pick"
    ]

    ==

    live[
        "nn_win_head_pick"
    ]

).astype(
    int
)


# ============================================================
# INPUT QUALITY
# ============================================================

def input_status(
    row
):

    non_null = int(

        row.get(
            "model_non_null_features",
            0
        )
    )


    context_ready = int(

        row.get(
            "context_ready",
            0
        )
    )


    if context_ready != 1:

        return (
            "MISSING_STATIC_CONTEXT"
        )


    if non_null == 54:

        return (
            "FULL_54"
        )


    if non_null >= 40:

        return (
            "PARTIAL_HISTORY"
        )


    return (
        "EARLY_SEASON_HEAVY_IMPUTATION"
    )


live[
    "nn_input_status"
] = live.apply(

    input_status,

    axis=1
)


# ============================================================
# STRENGTH
#
# Reserved now.
#
# LOW / MEDIUM / HIGH will be calculated later from:
#
# - model vs market difference
# - actual sportsbook price
# - ensemble agreement
# - feature/data quality
# - personnel uncertainty
# - market staleness
# - prospective validation evidence
#
# Do NOT fake a strength rating from NN output alone.
# ============================================================

live[
    "strength_score"
] = np.nan


live[
    "strength_label"
] = "PENDING"


# ============================================================
# VERSIONING
# ============================================================

live[
    "nn_artifact_version"
] = ARTIFACT_VERSION


live[
    "nn_prediction_version"
] = PREDICTION_VERSION


live[
    "nn_prediction_utc"
] = prediction_time.isoformat()


# ============================================================
# SORT
# ============================================================

if "start_date" in live.columns:

    live[
        "start_date"
    ] = pd.to_datetime(

        live[
            "start_date"
        ],

        utc=True,

        errors="coerce"
    )


    live = live.sort_values(

        by=[
            "start_date",
            "game_id"
        ]

    ).reset_index(
        drop=True
    )


# ============================================================
# SAVE CSV
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


live.to_csv(

    OUTPUT_CSV,

    index=False
)


# ============================================================
# SAVE JSON
# ============================================================

live.to_json(

    OUTPUT_JSON,

    orient="records",

    indent=2,

    date_format="iso"
)


# ============================================================
# STATUS
# ============================================================

header(
    "SPORTSBOOK-STYLE MODEL OUTPUT"
)


print(
    f"Games predicted: "
    f"{len(live):,}"
)


print()

print(
    "IMPORTANT:"
)


print(
    "  Negative spread ALWAYS belongs to "
    "the projected favorite."
)


print()

print(
    "  Example:"
)


print(
    "    Alabama -10.7"
)


print()

print(
    "  means the model projects Alabama "
    "as approximately a 10.7-point favorite."
)


print()

print(
    "  You no longer need to interpret "
    "home-margin signs."
)


# ============================================================
# GAME DISPLAY
# ============================================================

header(
    "LIVE MODEL BOARD"
)


for _, row in live.iterrows():

    favorite = row[
        "model_favorite"
    ]


    favorite_probability = float(

        row[
            "model_favorite_win_probability"
        ]
    )


    print(
        f"{row['away_team']} @ "
        f"{row['home_team']}"
    )


    print()


    print(
        "MODEL SPREAD"
    )


    print(
        f"  {row['model_spread_display']}"
    )


    print()


    print(
        "WIN PROBABILITY"
    )


    if favorite == "PICK'EM":

        print(
            "  50.0%"
        )


    else:

        print(

            f"  {favorite} "
            f"{favorite_probability * 100:.1f}%"
        )


    print()


    print(
        "STRENGTH"
    )


    print(
        "  PENDING"
    )


    print()


    print(
        "MODEL DIAGNOSTICS"
    )


    print(

        f"  Ensemble margin SD: "
        f"{float(row['nn_margin_std']):.2f}"
    )


    print(

        f"  Win probability SD: "
        f"{float(row['nn_win_probability_std']):.3f}"
    )


    print(

        f"  Feature coverage: "
        f"{int(row.get('model_non_null_features', 0))}/54"
    )


    print(

        f"  Input quality: "
        f"{row['nn_input_status']}"
    )


    print(
        "-" * 110
    )


# ============================================================
# COMPLETE
# ============================================================

header(
    "PREDICTION RUN COMPLETE"
)


print(
    "Files written:"
)


print(
    f"  {OUTPUT_CSV}"
)


print(
    f"  {OUTPUT_JSON}"
)


print()

print(
    "USER-FACING CONTRACT:"
)


print(
    "  model_spread_display"
)


print(
    "  model_favorite"
)


print(
    "  model_spread"
)


print(
    "  model_favorite_win_probability"
)


print(
    "  strength_label"
)


print()

print(
    "Next:"
)


print(
    "  Join sportsbook prices to produce:"
)


print()

print(
    "    MODEL SPREAD"
)


print(
    "    HARD ROCK"
)


print(
    "    MODEL EDGE"
)


print(
    "    WIN PROBABILITY"
)


print(
    "    STRENGTH"
)


if __name__ == "__main__":

    pass