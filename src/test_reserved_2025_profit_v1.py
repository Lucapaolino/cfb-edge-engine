import os
import random
import sqlite3

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)

from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = (
    "data/processed/"
    "features_v2.csv"
)

DEVELOPMENT_MARKET_FILE = (
    "data/processed/"
    "nn_vs_bovada_market_v1.csv"
)

DATABASE_FILE = (
    "data/cfb.db"
)

NN_OUTPUT_FILE = (
    "data/processed/"
    "reserved_2025_nn_predictions_v1.csv"
)

BET_OUTPUT_FILE = (
    "data/processed/"
    "reserved_2025_market_aware_bets_v1.csv"
)

SUMMARY_OUTPUT_FILE = (
    "data/processed/"
    "reserved_2025_profit_summary_v1.csv"
)


TEST_YEAR = 2025

PROVIDER = "Bovada"


# ============================================================
# FROZEN BETTING RULE
#
# IMPORTANT:
#
# These values are defined BEFORE 2025 outcomes are revealed.
#
# We will NOT change them based on 2025 results.
# ============================================================

MINIMUM_PRICE_EDGE = 0.00

MAXIMUM_MONEYLINE = 800.0


# ============================================================
# NEURAL NETWORK SETTINGS
#
# Same Small NN architecture that survived seed testing.
# ============================================================

SEEDS = [
    11,
    23,
    37,
    53,
    71
]


HIDDEN_SIZES = [
    64,
    32
]


BATCH_SIZE = 64

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

DROPOUT = 0.20

MAX_EPOCHS = 400

PATIENCE = 30

MIN_DELTA = 0.0001


# ============================================================
# BLEND SETTINGS
#
# Same coarse grid used during development.
# ============================================================

BLEND_WEIGHTS = np.arange(
    0.00,
    1.001,
    0.05
)


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
    "RESERVED 2025 PROFIT TEST — V1"
)

print(
    "=" * 110
)


print(
    f"PyTorch: {torch.__version__}"
)

print(
    f"Device:  {DEVICE}"
)


print()

print(
    "FROZEN BETTING RULE"
)

print(
    "-" * 110
)


print(
    f"Minimum price edge: "
    f"{MINIMUM_PRICE_EDGE:.0%}"
)


print(
    f"Maximum allowed moneyline: "
    f"less than +{MAXIMUM_MONEYLINE:.0f}"
)


print(
    "Bet size: 1 flat unit"
)


print()

print(
    "2025 outcomes will NOT be accessed until:"
)

print(
    "1. NN predictions are generated"
)

print(
    "2. market blend weight is selected using 2021-2024"
)

print(
    "3. 2025 bet candidates are frozen"
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
# PROBABILITY / ODDS HELPERS
# ============================================================

def american_to_implied_probability(
    odds
):

    if pd.isna(
        odds
    ):

        return np.nan


    odds = float(
        odds
    )


    if odds < 0:

        return (

            abs(
                odds
            )

            /

            (
                abs(
                    odds
                )

                +
                100.0
            )
        )


    if odds > 0:

        return (

            100.0

            /

            (
                odds
                +
                100.0
            )
        )


    return np.nan


def american_to_decimal_odds(
    odds
):

    if pd.isna(
        odds
    ):

        return np.nan


    odds = float(
        odds
    )


    if odds < 0:

        return (

            1.0

            +

            (
                100.0
                /
                abs(
                    odds
                )
            )
        )


    if odds > 0:

        return (

            1.0

            +

            odds
            /
            100.0
        )


    return np.nan


def clip_probability(
    probability
):

    return np.clip(
        probability,
        1e-6,
        1.0 - 1e-6
    )


# ============================================================
# LOAD FEATURES
# ============================================================

data = pd.read_csv(
    FEATURE_FILE
)


data = data.copy()


print()

print(
    f"Feature rows loaded: "
    f"{len(data):,}"
)


if TEST_YEAR not in set(
    data[
        "season"
    ]
):

    raise RuntimeError(
        "2025 is missing from the feature dataset."
    )


# ============================================================
# EXACT 54 V1 FEATURES
# ============================================================

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

        f"Expected exactly 54 features. "
        f"Found {len(FEATURES)}."
    )


print(
    f"Model features: "
    f"{len(FEATURES)}"
)


# ============================================================
# SPLIT TRAINING AND RESERVED 2025 FEATURES
#
# IMPORTANT:
#
# We deliberately do NOT access 2025 target_home_margin here.
# ============================================================

training_data = (

    data[
        data[
            "season"
        ]
        <
        TEST_YEAR
    ]

    .copy()
)


reserved_features = (

    data[
        data[
            "season"
        ]
        ==
        TEST_YEAR
    ]

    .copy()
)


print(
    f"Training games through 2024: "
    f"{len(training_data):,}"
)


print(
    f"Reserved 2025 games:         "
    f"{len(reserved_features):,}"
)


# ============================================================
# TRAINING TARGETS ONLY
# ============================================================

MARGIN_TARGET = (
    "target_home_margin"
)


if MARGIN_TARGET not in training_data.columns:

    raise RuntimeError(
        "Training margin target missing."
    )


training_home_win = (

    training_data[
        MARGIN_TARGET
    ]

    >
    0

).astype(
    int
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

        previous_size = input_size


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

            previous_size = hidden_size


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
# FEATURE PREPROCESSING
# ============================================================

def prepare_features(
    train_frame,
    evaluation_frame
):

    train_numeric = (

        train_frame[
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


    medians = (

        train_numeric

        .median()

        .fillna(
            0.0
        )
    )


    train_imputed = (

        train_numeric

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


    scaler = StandardScaler()


    x_train = (

        scaler

        .fit_transform(
            train_imputed
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
# LOSSES
# ============================================================

margin_loss_function = (
    nn.SmoothL1Loss()
)


win_loss_function = (
    nn.BCEWithLogitsLoss()
)


# ============================================================
# SELECT BEST EPOCH
#
# For 2025:
#
# Inner train      = 2018-2023
# Inner validation = 2024
#
# 2025 remains untouched.
# ============================================================

def select_best_epoch(
    inner_train,
    inner_validation,
    seed
):

    set_seed(
        seed
    )


    inner_train_win = (

        inner_train[
            MARGIN_TARGET
        ]

        >
        0

    ).astype(
        int
    )


    inner_validation_win = (

        inner_validation[
            MARGIN_TARGET
        ]

        >
        0

    ).astype(
        int
    )


    (
        x_train,
        x_validation
    ) = prepare_features(

        inner_train,

        inner_validation
    )


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
            ].to_numpy(
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
            ].to_numpy(
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


    loader = make_loader(

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


    best_loss = float(
        "inf"
    )

    best_epoch = 1

    stale_epochs = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1
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


            loss = (

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


            loss.backward()

            optimizer.step()


        model.eval()


        with torch.no_grad():

            (
                validation_margin,
                validation_win_logit
            ) = model(
                x_validation_tensor
            )


            validation_loss = (

                margin_loss_function(

                    validation_margin,

                    margin_validation_tensor
                )

                +

                win_loss_function(

                    validation_win_logit,

                    win_validation_tensor
                )
            )


            validation_loss = float(
                validation_loss.item()
            )


        if (

            validation_loss

            <
            (
                best_loss
                -
                MIN_DELTA
            )
        ):

            best_loss = validation_loss

            best_epoch = epoch

            stale_epochs = 0


        else:

            stale_epochs += 1


        if stale_epochs >= PATIENCE:

            break


    return (
        best_epoch,
        best_loss
    )


# ============================================================
# FINAL TRAINING / PREDICTION
#
# Train using 2018-2024.
#
# Predict 2025 WITHOUT accessing its outcome.
# ============================================================

def train_and_predict_2025(
    train_frame,
    reserved_frame,
    epochs,
    seed
):

    set_seed(
        seed
    )


    train_win = (

        train_frame[
            MARGIN_TARGET
        ]

        >
        0

    ).astype(
        int
    )


    (
        x_train,
        x_reserved
    ) = prepare_features(

        train_frame,

        reserved_frame
    )


    margin_mean = float(

        train_frame[
            MARGIN_TARGET
        ].mean()
    )


    margin_std = float(

        train_frame[
            MARGIN_TARGET
        ].std(
            ddof=0
        )
    )


    if margin_std < 1e-6:

        margin_std = 1.0


    y_train_margin = (

        (
            train_frame[
                MARGIN_TARGET
            ].to_numpy(
                dtype=np.float32
            )

            -
            margin_mean
        )

        /
        margin_std
    )


    y_train_win = (

        train_win

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
            FEATURES
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


    for _ in range(
        epochs
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


            loss = (

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


            loss.backward()

            optimizer.step()


    model.eval()


    x_reserved_tensor = (

        torch.tensor(
            x_reserved,
            dtype=torch.float32
        )

        .to(
            DEVICE
        )
    )


    with torch.no_grad():

        (
            standardized_margin,
            win_logit
        ) = model(
            x_reserved_tensor
        )


        standardized_margin = (

            standardized_margin

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


    predicted_margin = (

        standardized_margin

        *
        margin_std

        +
        margin_mean
    )


    return (
        predicted_margin,
        win_probability
    )


# ============================================================
# INNER TRAIN / VALIDATION FOR 2025
# ============================================================

inner_train = (

    training_data[
        training_data[
            "season"
        ]
        <
        2024
    ]

    .copy()
)


inner_validation = (

    training_data[
        training_data[
            "season"
        ]
        ==
        2024
    ]

    .copy()
)


print()

print(
    "=" * 110
)

print(
    "GENERATING RESERVED 2025 NN PREDICTIONS"
)

print(
    "=" * 110
)


print(
    "Inner train:      2018-2023"
)


print(
    "Inner validation: 2024"
)


print(
    "Final train:      2018-2024"
)


print(
    "Prediction:       2025"
)


# ============================================================
# FIVE-SEED SMALL NN ENSEMBLE
# ============================================================

seed_prediction_rows = []


for seed in SEEDS:

    (
        best_epoch,
        best_loss
    ) = select_best_epoch(

        inner_train,

        inner_validation,

        seed
    )


    (
        margin_prediction,
        win_probability
    ) = train_and_predict_2025(

        training_data,

        reserved_features,

        best_epoch,

        seed
    )


    print(

        f"Seed {seed:>2} | "
        f"Epoch {best_epoch:>3} | "
        f"Inner loss {best_loss:.4f}"
    )


    for position in range(
        len(
            reserved_features
        )
    ):

        game = reserved_features.iloc[
            position
        ]


        seed_prediction_rows.append(
            {
                "game_id":
                    game[
                        "game_id"
                    ],

                "season":
                    TEST_YEAR,

                "week":
                    game[
                        "week"
                    ],

                "home_team":
                    game[
                        "home_team"
                    ],

                "away_team":
                    game[
                        "away_team"
                    ],

                "seed":
                    seed,

                "predicted_margin":
                    margin_prediction[
                        position
                    ],

                "nn_home_win_probability":
                    win_probability[
                        position
                    ]
            }
        )


seed_predictions = pd.DataFrame(
    seed_prediction_rows
)


# ============================================================
# ENSEMBLE
# ============================================================

ensemble_2025 = (

    seed_predictions

    .groupby(
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team"
        ],
        as_index=False
    )

    .agg(

        nn_predicted_margin=(
            "predicted_margin",
            "mean"
        ),

        nn_home_win_probability=(
            "nn_home_win_probability",
            "mean"
        ),

        seed_count=(
            "seed",
            "nunique"
        )
    )
)


if (

    ensemble_2025[
        "seed_count"
    ]
    !=
    5

).any():

    raise RuntimeError(
        "Some 2025 games do not have five seed predictions."
    )


ensemble_2025.to_csv(
    NN_OUTPUT_FILE,
    index=False
)


print()

print(
    f"2025 NN ensemble predictions: "
    f"{len(ensemble_2025):,}"
)


# ============================================================
# SELECT 2025 BLEND WEIGHT USING ONLY 2021-2024
# ============================================================

development = pd.read_csv(
    DEVELOPMENT_MARKET_FILE
)


development = development[
    development[
        "season"
    ].isin(
        [
            2021,
            2022,
            2023,
            2024
        ]
    )
].copy()


if (

    development[
        "season"
    ].max()
    >=
    TEST_YEAR

):

    raise RuntimeError(
        "2025 found in development market data."
    )


development_actual = (

    development[
        "actual_home_win"
    ]

    .astype(
        int
    )

    .to_numpy()
)


development_market = clip_probability(

    development[
        "market_home_fair_probability"
    ].to_numpy()
)


development_nn = clip_probability(

    development[
        "model_home_win_probability"
    ].to_numpy()
)


best_weight = None

best_logloss = float(
    "inf"
)


for nn_weight in BLEND_WEIGHTS:

    probability = (

        (
            1.0
            -
            nn_weight
        )

        *
        development_market

        +

        nn_weight
        *
        development_nn
    )


    loss = log_loss(

        development_actual,

        probability,

        labels=[
            0,
            1
        ]
    )


    if loss < best_logloss:

        best_logloss = loss

        best_weight = float(
            nn_weight
        )


print()

print(
    "=" * 110
)

print(
    "FROZEN 2025 MARKET BLEND"
)

print(
    "=" * 110
)


print(

    f"Development games used: "
    f"{len(development):,}"
)


print(

    f"Selected NN weight: "
    f"{best_weight:.0%}"
)


print(

    f"Selected market weight: "
    f"{1.0 - best_weight:.0%}"
)


print(

    f"Development LogLoss: "
    f"{best_logloss:.4f}"
)


# ============================================================
# LOAD 2025 BOVADA MONEYLINES
#
# Still no outcomes used.
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


market_2025 = pd.read_sql_query(

    """
    SELECT
        game_id,
        provider,
        spread,
        home_moneyline,
        away_moneyline
    FROM betting_lines
    WHERE provider = ?
    """,

    connection,

    params=[
        PROVIDER
    ]
)


connection.close()


market_2025[
    "game_id"
] = pd.to_numeric(

    market_2025[
        "game_id"
    ],

    errors="coerce"
).astype(
    "Int64"
)


for column in [

    "spread",
    "home_moneyline",
    "away_moneyline"
]:

    market_2025[
        column
    ] = pd.to_numeric(

        market_2025[
            column
        ],

        errors="coerce"
    )


market_2025.loc[
    market_2025[
        "home_moneyline"
    ]
    ==
    0,
    "home_moneyline"
] = np.nan


market_2025.loc[
    market_2025[
        "away_moneyline"
    ]
    ==
    0,
    "away_moneyline"
] = np.nan


market_2025 = market_2025[
    market_2025[
        "home_moneyline"
    ].notna()

    &

    market_2025[
        "away_moneyline"
    ].notna()
].copy()


duplicate_market_games = (

    market_2025[
        "game_id"
    ]

    .duplicated(
        keep=False
    )
)


if duplicate_market_games.any():

    raise RuntimeError(
        "Duplicate Bovada rows detected."
    )


# ============================================================
# JOIN 2025 NN + MARKET
# ============================================================

games = ensemble_2025.merge(

    market_2025,

    on="game_id",

    how="inner"
)


print()

print(
    f"2025 games with paired Bovada prices: "
    f"{len(games):,}"
)


# ============================================================
# MARKET PROBABILITIES
# ============================================================

games[
    "market_home_raw_probability"
] = (

    games[
        "home_moneyline"
    ]

    .apply(
        american_to_implied_probability
    )
)


games[
    "market_away_raw_probability"
] = (

    games[
        "away_moneyline"
    ]

    .apply(
        american_to_implied_probability
    )
)


games[
    "market_probability_sum"
] = (

    games[
        "market_home_raw_probability"
    ]

    +

    games[
        "market_away_raw_probability"
    ]
)


games[
    "market_home_fair_probability"
] = (

    games[
        "market_home_raw_probability"
    ]

    /

    games[
        "market_probability_sum"
    ]
)


games[
    "market_away_fair_probability"
] = (

    games[
        "market_away_raw_probability"
    ]

    /

    games[
        "market_probability_sum"
    ]
)


games[
    "home_decimal_odds"
] = (

    games[
        "home_moneyline"
    ]

    .apply(
        american_to_decimal_odds
    )
)


games[
    "away_decimal_odds"
] = (

    games[
        "away_moneyline"
    ]

    .apply(
        american_to_decimal_odds
    )
)


# ============================================================
# MARKET-AWARE 2025 PROBABILITY
#
# Weight was already selected using 2021-2024 only.
# ============================================================

games[
    "tuned_home_probability"
] = (

    (
        1.0
        -
        best_weight
    )

    *
    games[
        "market_home_fair_probability"
    ]

    +

    best_weight
    *
    games[
        "nn_home_win_probability"
    ]
)


games[
    "tuned_away_probability"
] = (

    1.0

    -

    games[
        "tuned_home_probability"
    ]
)


# ============================================================
# PRICE EDGE
# ============================================================

games[
    "home_price_edge"
] = (

    games[
        "tuned_home_probability"
    ]

    -

    games[
        "market_home_raw_probability"
    ]
)


games[
    "away_price_edge"
] = (

    games[
        "tuned_away_probability"
    ]

    -

    games[
        "market_away_raw_probability"
    ]
)


# ============================================================
# SELECT SIDE
# ============================================================

games[
    "selected_side"
] = np.where(

    games[
        "home_price_edge"
    ]

    >=

    games[
        "away_price_edge"
    ],

    "HOME",

    "AWAY"
)


games[
    "selected_team"
] = np.where(

    games[
        "selected_side"
    ]
    ==
    "HOME",

    games[
        "home_team"
    ],

    games[
        "away_team"
    ]
)


games[
    "selected_price_edge"
] = np.where(

    games[
        "selected_side"
    ]
    ==
    "HOME",

    games[
        "home_price_edge"
    ],

    games[
        "away_price_edge"
    ]
)


games[
    "selected_probability"
] = np.where(

    games[
        "selected_side"
    ]
    ==
    "HOME",

    games[
        "tuned_home_probability"
    ],

    games[
        "tuned_away_probability"
    ]
)


games[
    "selected_moneyline"
] = np.where(

    games[
        "selected_side"
    ]
    ==
    "HOME",

    games[
        "home_moneyline"
    ],

    games[
        "away_moneyline"
    ]
)


games[
    "selected_decimal_odds"
] = np.where(

    games[
        "selected_side"
    ]
    ==
    "HOME",

    games[
        "home_decimal_odds"
    ],

    games[
        "away_decimal_odds"
    ]
)


# ============================================================
# FREEZE THE ACTUAL BET LIST
#
# Still NO 2025 outcomes have been accessed.
# ============================================================

games[
    "frozen_bet"
] = (

    (
        games[
            "selected_price_edge"
        ]
        >=
        MINIMUM_PRICE_EDGE
    )

    &

    (
        games[
            "selected_moneyline"
        ]
        <
        MAXIMUM_MONEYLINE
    )
)


frozen_bets_before_outcomes = games[
    games[
        "frozen_bet"
    ]
].copy()


print()

print(
    "=" * 110
)

print(
    "2025 BET LIST FROZEN"
)

print(
    "=" * 110
)


print(

    f"Total Bovada games: "
    f"{len(games):,}"
)


print(

    f"BET decisions:      "
    f"{len(frozen_bets_before_outcomes):,}"
)


print(

    f"NO BET decisions:   "
    f"{len(games) - len(frozen_bets_before_outcomes):,}"
)


print()

print(
    "The model, blend weight, threshold, odds cap,"
)

print(
    "and exact 2025 bet list are now fixed."
)


# ============================================================
# ============================================================
#
# REVEAL RESERVED 2025 OUTCOMES
#
# Everything above this line was decided without accessing
# the 2025 result target.
#
# ============================================================
# ============================================================

print()

print(
    "=" * 110
)

print(
    "REVEALING RESERVED 2025 OUTCOMES"
)

print(
    "=" * 110
)


reserved_outcomes = (

    data[
        data[
            "season"
        ]
        ==
        TEST_YEAR
    ][
        [
            "game_id",
            "target_home_margin"
        ]
    ]

    .copy()
)


reserved_outcomes[
    "actual_home_win"
] = (

    reserved_outcomes[
        "target_home_margin"
    ]

    >
    0

).astype(
    int
)


games = games.merge(

    reserved_outcomes,

    on="game_id",

    how="left"
)


if games[
    "actual_home_win"
].isna().any():

    raise RuntimeError(
        "Some 2025 market games are missing outcomes."
    )


# ============================================================
# PROBABILITY METRICS
# ============================================================

def calculate_metrics(
    actual,
    probability
):

    actual = np.asarray(
        actual,
        dtype=int
    )


    probability = clip_probability(
        np.asarray(
            probability,
            dtype=float
        )
    )


    prediction = (

        probability
        >=
        0.50

    ).astype(
        int
    )


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


actual = (

    games[
        "actual_home_win"
    ]

    .astype(
        int
    )
)


(
    market_accuracy,
    market_brier,
    market_logloss
) = calculate_metrics(

    actual,

    games[
        "market_home_fair_probability"
    ]
)


(
    nn_accuracy,
    nn_brier,
    nn_logloss
) = calculate_metrics(

    actual,

    games[
        "nn_home_win_probability"
    ]
)


(
    tuned_accuracy,
    tuned_brier,
    tuned_logloss
) = calculate_metrics(

    actual,

    games[
        "tuned_home_probability"
    ]
)


# ============================================================
# SCORE FROZEN BETS
# ============================================================

bets = games[
    games[
        "frozen_bet"
    ]
].copy()


bets[
    "bet_won"
] = np.where(

    bets[
        "selected_side"
    ]
    ==
    "HOME",

    bets[
        "actual_home_win"
    ]
    ==
    1,

    bets[
        "actual_home_win"
    ]
    ==
    0

).astype(
    int
)


bets[
    "flat_profit"
] = np.where(

    bets[
        "bet_won"
    ]
    ==
    1,

    bets[
        "selected_decimal_odds"
    ]

    -
    1.0,

    -1.0
)


bets[
    "bet_type"
] = np.where(

    bets[
        "selected_moneyline"
    ]
    <
    0,

    "FAVORITE",

    "UNDERDOG"
)


bets = (

    bets

    .sort_values(
        [
            "week",
            "game_id"
        ]
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# RISK METRICS
# ============================================================

def maximum_drawdown(
    profit_array
):

    profits = np.asarray(
        profit_array,
        dtype=float
    )


    if len(
        profits
    ) == 0:

        return np.nan


    cumulative = np.cumsum(
        profits
    )


    curve = np.concatenate(
        [
            np.array(
                [0.0]
            ),
            cumulative
        ]
    )


    running_peak = np.maximum.accumulate(
        curve
    )


    drawdown = (

        running_peak

        -
        curve
    )


    return float(
        drawdown.max()
    )


def longest_losing_streak(
    wins
):

    longest = 0

    current = 0


    for won in wins:

        if int(
            won
        ) == 0:

            current += 1

            longest = max(
                longest,
                current
            )

        else:

            current = 0


    return longest


# ============================================================
# FINAL BETTING RESULT
# ============================================================

bet_count = len(
    bets
)


wins = int(
    bets[
        "bet_won"
    ].sum()
)


losses = (
    bet_count
    -
    wins
)


profit_units = float(

    bets[
        "flat_profit"
    ].sum()
)


roi = (

    profit_units
    /
    bet_count

    if bet_count > 0

    else
    np.nan
)


win_rate = (

    wins
    /
    bet_count

    if bet_count > 0

    else
    np.nan
)


drawdown = (

    maximum_drawdown(
        bets[
            "flat_profit"
        ]
    )

    if bet_count > 0

    else
    np.nan
)


losing_streak = (

    longest_losing_streak(
        bets[
            "bet_won"
        ]
    )

    if bet_count > 0

    else
    np.nan
)


favorite_bets = int(

    (
        bets[
            "bet_type"
        ]
        ==
        "FAVORITE"
    ).sum()
)


underdog_bets = int(

    (
        bets[
            "bet_type"
        ]
        ==
        "UNDERDOG"
    ).sum()
)


# ============================================================
# PRINT RESERVED PROBABILITY TEST
# ============================================================

print()

print(
    "=" * 110
)

print(
    "2025 RESERVED PROBABILITY RESULTS"
)

print(
    "=" * 110
)


print(

    f"{'Model':<20}"
    f"{'Accuracy':>12}"
    f"{'Brier':>12}"
    f"{'LogLoss':>12}"
)


print(
    "-" * 56
)


print(

    f"{'Bovada':<20}"
    f"{market_accuracy:>12.3f}"
    f"{market_brier:>12.3f}"
    f"{market_logloss:>12.3f}"
)


print(

    f"{'NN Ensemble':<20}"
    f"{nn_accuracy:>12.3f}"
    f"{nn_brier:>12.3f}"
    f"{nn_logloss:>12.3f}"
)


print(

    f"{'Tuned Blend':<20}"
    f"{tuned_accuracy:>12.3f}"
    f"{tuned_brier:>12.3f}"
    f"{tuned_logloss:>12.3f}"
)


# ============================================================
# PRINT RESERVED PROFIT TEST
# ============================================================

print()

print(
    "=" * 110
)

print(
    "2025 RESERVED PROFIT RESULT — FROZEN RULE"
)

print(
    "=" * 110
)


print(

    f"Blend:                "
    f"{1.0 - best_weight:.0%} market / "
    f"{best_weight:.0%} NN"
)


print(

    f"Minimum price edge:   "
    f"{MINIMUM_PRICE_EDGE:.0%}"
)


print(

    f"Maximum moneyline:    "
    f"< +{MAXIMUM_MONEYLINE:.0f}"
)


print()

print(

    f"Bets:                 "
    f"{bet_count}"
)


print(

    f"Wins:                 "
    f"{wins}"
)


print(

    f"Losses:               "
    f"{losses}"
)


print(

    f"Win rate:              "
    f"{win_rate:.2%}"
)


print()

print(

    f"Profit:                "
    f"{profit_units:+.2f} units"
)


print(

    f"ROI:                   "
    f"{roi:+.2%}"
)


print()

print(

    f"Max drawdown:          "
    f"{drawdown:.2f} units"
)


print(

    f"Longest losing streak: "
    f"{losing_streak}"
)


print()

print(

    f"Favorite bets:         "
    f"{favorite_bets}"
)


print(

    f"Underdog bets:         "
    f"{underdog_bets}"
)


if bet_count > 0:

    print(

        f"Average moneyline:     "
        f"{bets['selected_moneyline'].mean():+.0f}"
    )


    print(

        f"Average price edge:    "
        f"{bets['selected_price_edge'].mean():.2%}"
    )


# ============================================================
# SAVE OUTPUT
# ============================================================

summary = pd.DataFrame(
    [
        {
            "season":
                TEST_YEAR,

            "market_weight":
                1.0
                -
                best_weight,

            "nn_weight":
                best_weight,

            "minimum_price_edge":
                MINIMUM_PRICE_EDGE,

            "maximum_moneyline_exclusive":
                MAXIMUM_MONEYLINE,

            "market_accuracy":
                market_accuracy,

            "market_brier":
                market_brier,

            "market_logloss":
                market_logloss,

            "nn_accuracy":
                nn_accuracy,

            "nn_brier":
                nn_brier,

            "nn_logloss":
                nn_logloss,

            "tuned_accuracy":
                tuned_accuracy,

            "tuned_brier":
                tuned_brier,

            "tuned_logloss":
                tuned_logloss,

            "bets":
                bet_count,

            "wins":
                wins,

            "losses":
                losses,

            "win_rate":
                win_rate,

            "profit_units":
                profit_units,

            "roi":
                roi,

            "max_drawdown_units":
                drawdown,

            "longest_losing_streak":
                losing_streak,

            "favorite_bets":
                favorite_bets,

            "underdog_bets":
                underdog_bets
        }
    ]
)


os.makedirs(
    "data/processed",
    exist_ok=True
)


games.to_csv(
    BET_OUTPUT_FILE,
    index=False
)


summary.to_csv(
    SUMMARY_OUTPUT_FILE,
    index=False
)


print()

print(
    "=" * 110
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 110
)


print(
    NN_OUTPUT_FILE
)


print(
    BET_OUTPUT_FILE
)


print(
    SUMMARY_OUTPUT_FILE
)


print()

print(
    "IMPORTANT:"
)


print(
    "2025 has now been scored."
)


print(
    "Do NOT change the rule based on this result "
    "and then call the revised rule validated on 2025."
)