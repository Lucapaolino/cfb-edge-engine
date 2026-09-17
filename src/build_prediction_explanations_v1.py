#!/usr/bin/env python3
"""
CFB Edge Engine — Prediction Explanations V1.1

Creates truthful local explanations for the frozen 2026 neural ensemble.

V1.1 safety change: extracts only the FootballNetwork AST definition from the
production predictor so explanation generation never executes/re-writes the
production prediction script as a side effect.

Method
------
For each live game and each of the 54 model inputs:

1. Reproduce the exact frozen ensemble prediction using the production
   feature names, imputation medians, scaler, target normalization, and
   model checkpoints.
2. Verify that the reproduced ensemble output matches the already-written
   production prediction for BOTH:
      - projected home margin
      - raw home win probability
3. Neutralize one standardized feature at a time to 0.0, which corresponds
   to the training-distribution mean after the saved StandardScaler.
4. Re-run all five frozen models.
5. Measure how much the ensemble's projected margin and favored-team win
   probability move.

The reported "impact" is therefore a LOCAL ABLATION IMPACT:
    current prediction - prediction with that input neutralized

It is NOT a causal claim and the individual impacts are NOT additive because
the neural network contains nonlinear interactions.

Group impacts are calculated separately by neutralizing all features in a
football category at once. Group impacts are also not additive.

Safety
------
- NO API calls
- NO retraining
- NO model mutation
- NO prediction-history mutation
- NO wager mutation
- Fails closed if exact ensemble reproduction does not match production
- Uses the FootballNetwork class from the production predictor source so
  activation/architecture details are not re-guessed here.

Outputs
-------
data/processed/prediction_explanations_2026_v1.json
data/processed/prediction_explanations_2026_v1.csv
"""

from __future__ import annotations

import csv
import ast
import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ============================================================================
# PATHS / CONSTANTS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "cfb_nn_2026_v1"
)

PREDICTOR_FILE = (
    PROJECT_ROOT
    / "src"
    / "predict_live_2026_nn_v1.py"
)

BOARD_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_model_market_board_2026_v1.json"
)

FEATURE_NAMES_FILE = (
    MODEL_DIR
    / "feature_names.json"
)

MEDIANS_FILE = (
    MODEL_DIR
    / "imputer_medians.json"
)

METADATA_FILE = (
    MODEL_DIR
    / "metadata.json"
)

SCALER_FILE = (
    MODEL_DIR
    / "scaler.pkl"
)

TARGET_NORMALIZATION_FILE = (
    MODEL_DIR
    / "target_normalization.json"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_explanations_2026_v1.json"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_explanations_2026_v1.csv"
)

EXPECTED_FEATURE_COUNT = 54

MARGIN_REPRODUCTION_TOLERANCE = 0.08
PROBABILITY_REPRODUCTION_TOLERANCE = 0.003


# ============================================================================
# FOOTBALL FEATURE GROUPS
# ============================================================================

FEATURE_GROUPS = {
    "ELO": {
        "label": "Pregame Elo",
        "features": {
            "pregame_elo_diff",
        },
    },
    "TALENT": {
        "label": "Roster Talent",
        "features": {
            "talent_diff",
        },
    },
    "RETURNING_PRODUCTION": {
        "label": "Returning Production",
        "features": {
            "returning_passing_ppa_diff",
            "returning_ppa_diff",
            "returning_receiving_ppa_diff",
            "returning_rushing_ppa_diff",
        },
    },
    "RECENT_RESULTS": {
        "label": "Results & Scoring",
        "features": {
            "diff_pregame_avg_margin",
            "diff_pregame_avg_points_against",
            "diff_pregame_avg_points_for",
            "diff_pregame_avg_win",
            "diff_pregame_last3_margin",
            "diff_pregame_last3_points_against",
            "diff_pregame_last3_points_for",
            "diff_pregame_last3_win",
        },
    },
    "PASSING": {
        "label": "Passing",
        "features": {
            "diff_pregame_avg_netPassingYards",
            "diff_pregame_avg_opponent_netPassingYards",
            "diff_pregame_avg_opponent_yardsPerPass",
            "diff_pregame_avg_yardsPerPass",
            "diff_pregame_last3_netPassingYards",
            "diff_pregame_last3_opponent_netPassingYards",
            "diff_pregame_last3_opponent_yardsPerPass",
            "diff_pregame_last3_yardsPerPass",
        },
    },
    "RUSHING": {
        "label": "Rushing",
        "features": {
            "diff_pregame_avg_opponent_rushingYards",
            "diff_pregame_avg_opponent_yardsPerRushAttempt",
            "diff_pregame_avg_rushingYards",
            "diff_pregame_avg_yardsPerRushAttempt",
            "diff_pregame_last3_opponent_rushingYards",
            "diff_pregame_last3_opponent_yardsPerRushAttempt",
            "diff_pregame_last3_rushingYards",
            "diff_pregame_last3_yardsPerRushAttempt",
        },
    },
    "BALL_MOVEMENT": {
        "label": "Ball Movement & Downs",
        "features": {
            "diff_pregame_avg_firstDowns",
            "diff_pregame_avg_opponent_firstDowns",
            "diff_pregame_avg_opponent_thirdDownPct",
            "diff_pregame_avg_opponent_totalYards",
            "diff_pregame_avg_thirdDownPct",
            "diff_pregame_avg_totalYards",
            "diff_pregame_last3_firstDowns",
            "diff_pregame_last3_opponent_firstDowns",
            "diff_pregame_last3_opponent_thirdDownPct",
            "diff_pregame_last3_opponent_totalYards",
            "diff_pregame_last3_thirdDownPct",
            "diff_pregame_last3_totalYards",
        },
    },
    "TURNOVERS": {
        "label": "Turnovers",
        "features": {
            "diff_pregame_avg_opponent_turnovers",
            "diff_pregame_avg_turnover_margin",
            "diff_pregame_avg_turnovers",
            "diff_pregame_last3_opponent_turnovers",
            "diff_pregame_last3_turnover_margin",
            "diff_pregame_last3_turnovers",
        },
    },
    "PASS_RUSH": {
        "label": "Sacks / Pass Rush",
        "features": {
            "diff_pregame_avg_opponent_sacks",
            "diff_pregame_avg_sacks",
            "diff_pregame_last3_opponent_sacks",
            "diff_pregame_last3_sacks",
        },
    },
    "EXPERIENCE": {
        "label": "Season Sample / Experience",
        "features": {
            "diff_pregame_games_played",
        },
    },
    "GAME_ENVIRONMENT": {
        "label": "Game Environment",
        "features": {
            "neutral_site",
        },
    },
}


# ============================================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================================

SPECIAL_LABELS = {
    "pregame_elo_diff": "Pregame Elo",
    "talent_diff": "Roster talent",
    "returning_passing_ppa_diff": "Returning passing production",
    "returning_ppa_diff": "Overall returning production",
    "returning_receiving_ppa_diff": "Returning receiving production",
    "returning_rushing_ppa_diff": "Returning rushing production",
    "neutral_site": "Neutral-site setting",
    "diff_pregame_games_played": "Games played",
}


METRIC_LABELS = {
    "firstDowns": "first downs",
    "margin": "scoring margin",
    "netPassingYards": "net passing yards",
    "opponent_firstDowns": "opponent first downs",
    "opponent_netPassingYards": "opponent passing yards",
    "opponent_rushingYards": "opponent rushing yards",
    "opponent_sacks": "sacks allowed / opponent sacks",
    "opponent_thirdDownPct": "opponent third-down rate",
    "opponent_totalYards": "opponent total yards",
    "opponent_turnovers": "opponent turnovers",
    "opponent_yardsPerPass": "opponent yards per pass",
    "opponent_yardsPerRushAttempt": "opponent yards per rush",
    "points_against": "points allowed",
    "points_for": "points scored",
    "rushingYards": "rushing yards",
    "sacks": "sacks",
    "thirdDownPct": "third-down rate",
    "totalYards": "total yards",
    "turnover_margin": "turnover margin",
    "turnovers": "turnovers",
    "win": "win rate",
    "yardsPerPass": "yards per pass",
    "yardsPerRushAttempt": "yards per rush",
}


# ============================================================================
# HELPERS
# ============================================================================

def utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def print_rule(
    char: str = "=",
    width: int = 112,
) -> None:
    print(
        char * width
    )


def safe_float(
    value: Any,
) -> float | None:
    try:
        if value is None or value == "":
            return None

        number = float(
            value
        )

        if math.isnan(
            number
        ) or math.isinf(
            number
        ):
            return None

        return number

    except (
        TypeError,
        ValueError,
    ):
        return None


def safe_int(
    value: Any,
) -> int | None:
    try:
        if value is None or value == "":
            return None

        return int(
            float(
                value
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def normalize_team(
    value: Any,
) -> str:
    return " ".join(
        str(
            value
            or
            ""
        )
        .strip()
        .casefold()
        .split()
    )


def same_team(
    left: Any,
    right: Any,
) -> bool:
    left_norm = normalize_team(
        left
    )

    right_norm = normalize_team(
        right
    )

    return bool(
        left_norm
        and
        right_norm
        and
        left_norm
        ==
        right_norm
    )


def rounded(
    value: float | None,
    digits: int = 4,
) -> float | None:
    if value is None:
        return None

    return round(
        value,
        digits,
    )


def human_feature_label(
    feature: str,
) -> str:
    if feature in SPECIAL_LABELS:
        return SPECIAL_LABELS[
            feature
        ]

    prefix = None
    metric = feature

    if feature.startswith(
        "diff_pregame_avg_"
    ):
        prefix = "Season avg"
        metric = feature[
            len(
                "diff_pregame_avg_"
            ):
        ]

    elif feature.startswith(
        "diff_pregame_last3_"
    ):
        prefix = "Last 3"
        metric = feature[
            len(
                "diff_pregame_last3_"
            ):
        ]

    metric_label = METRIC_LABELS.get(
        metric,
        metric
        .replace(
            "_",
            " "
        )
    )

    if prefix:
        return (
            f"{prefix} "
            f"{metric_label}"
        )

    return feature.replace(
        "_",
        " "
    )


def group_for_feature(
    feature: str,
) -> str:
    for group_key, group in FEATURE_GROUPS.items():
        if feature in group[
            "features"
        ]:
            return group_key

    return "OTHER"


def impact_descriptor(
    points: float,
) -> str:
    magnitude = abs(
        points
    )

    if magnitude < 0.15:
        return "minimal"

    if magnitude < 0.50:
        return "small"

    if magnitude < 1.00:
        return "moderate"

    if magnitude < 2.00:
        return "strong"

    return "very strong"


# ============================================================================
# PRODUCTION MODULE / ARTIFACT LOADING
# ============================================================================

def load_production_network_class():
    """
    Load ONLY the FootballNetwork class definition from the production
    predictor source.

    We intentionally do NOT import/execute the predictor module because that
    file is an executable production script with top-level inference/output
    behavior. Executing it here would unnecessarily rewrite live prediction
    artifacts just to obtain the class definition.

    AST extraction lets this explanation builder reuse the exact production
    network class without running any production pipeline code.
    """
    if not PREDICTOR_FILE.exists():
        raise FileNotFoundError(
            f"Production predictor not found: {PREDICTOR_FILE}"
        )

    source_text = PREDICTOR_FILE.read_text(
        encoding="utf-8",
        errors="strict",
    )

    tree = ast.parse(
        source_text,
        filename=str(PREDICTOR_FILE),
    )

    network_node = None

    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and
            node.name == "FootballNetwork"
        ):
            network_node = node
            break

    if network_node is None:
        raise RuntimeError(
            "Production predictor does not define FootballNetwork. "
            "Explanation builder refuses to guess the architecture."
        )

    isolated_module = ast.Module(
        body=[network_node],
        type_ignores=[],
    )

    ast.fix_missing_locations(
        isolated_module
    )

    namespace = {
        "torch": torch,
        "nn": torch.nn,
    }

    exec(
        compile(
            isolated_module,
            filename=str(PREDICTOR_FILE),
            mode="exec",
        ),
        namespace,
    )

    network_class = namespace.get(
        "FootballNetwork"
    )

    if network_class is None:
        raise RuntimeError(
            "FootballNetwork AST extraction failed."
        )

    return network_class


@dataclass
class ArtifactBundle:
    feature_names: list[str]
    medians: dict[str, float]
    scaler: Any
    margin_mean: float
    margin_std: float
    seeds: list[int]
    models: list[Any]
    metadata: dict[str, Any]


def load_json(
    path: Path,
) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def load_artifacts() -> ArtifactBundle:
    FootballNetwork = load_production_network_class()

    feature_names = load_json(
        FEATURE_NAMES_FILE
    )

    if (
        not isinstance(
            feature_names,
            list,
        )
        or
        len(
            feature_names
        )
        !=
        EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            "feature_names.json does not contain the expected 54-feature list."
        )

    feature_names = [
        str(
            feature
        )
        for feature in feature_names
    ]

    medians_raw = load_json(
        MEDIANS_FILE
    )

    if not isinstance(
        medians_raw,
        dict,
    ):
        raise RuntimeError(
            "imputer_medians.json is not a dictionary."
        )

    medians = {}

    for feature in feature_names:
        value = safe_float(
            medians_raw.get(
                feature
            )
        )

        medians[
            feature
        ] = (
            0.0
            if value is None
            else value
        )

    metadata = load_json(
        METADATA_FILE
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise RuntimeError(
            "metadata.json is not a dictionary."
        )

    target_norm = load_json(
        TARGET_NORMALIZATION_FILE
    )

    if not isinstance(
        target_norm,
        dict,
    ):
        raise RuntimeError(
            "target_normalization.json is not a dictionary."
        )

    margin_mean = safe_float(
        target_norm.get(
            "margin_mean"
        )
    )

    margin_std = safe_float(
        target_norm.get(
            "margin_std"
        )
    )

    if (
        margin_mean is None
        or
        margin_std is None
        or
        margin_std
        <=
        0.0
    ):
        raise RuntimeError(
            "Invalid target margin normalization artifact."
        )

    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Scaler artifact missing: {SCALER_FILE}"
        )

    with SCALER_FILE.open(
        "rb"
    ) as handle:
        scaler = pickle.load(
            handle
        )

    seeds_raw = metadata.get(
        "ensemble_seeds"
    )

    if not isinstance(
        seeds_raw,
        list,
    ) or not seeds_raw:
        raise RuntimeError(
            "metadata.json does not contain ensemble_seeds."
        )

    seeds = [
        int(
            seed
        )
        for seed in seeds_raw
    ]

    architecture = metadata.get(
        "architecture"
    )

    if not isinstance(
        architecture,
        dict,
    ):
        architecture = {}

    default_hidden = architecture.get(
        "hidden_sizes",
        [
            64,
            32,
        ],
    )

    default_dropout = safe_float(
        architecture.get(
            "dropout"
        )
    )

    if default_dropout is None:
        default_dropout = 0.2

    models = []

    for seed in seeds:
        checkpoint_path = (
            MODEL_DIR
            /
            f"model_seed_{seed}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing ensemble checkpoint: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise RuntimeError(
                f"Checkpoint {checkpoint_path.name} is not a dictionary."
            )

        hidden_sizes = checkpoint.get(
            "hidden_sizes",
            default_hidden,
        )

        dropout = safe_float(
            checkpoint.get(
                "dropout"
            )
        )

        if dropout is None:
            dropout = default_dropout

        model = FootballNetwork(
            input_size=len(
                feature_names
            ),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

        state_dict = checkpoint.get(
            "state_dict"
        )

        if state_dict is None:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path.name} has no state_dict."
            )

        model.load_state_dict(
            state_dict
        )

        model.eval()

        models.append(
            model
        )

    return ArtifactBundle(
        feature_names=feature_names,
        medians=medians,
        scaler=scaler,
        margin_mean=margin_mean,
        margin_std=margin_std,
        seeds=seeds,
        models=models,
        metadata=metadata,
    )


# ============================================================================
# BOARD / INPUT PREPARATION
# ============================================================================

def load_board() -> list[dict[str, Any]]:
    if not BOARD_FILE.exists():
        raise FileNotFoundError(
            f"Live board not found: {BOARD_FILE}"
        )

    with BOARD_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    if isinstance(
        payload,
        list,
    ):
        rows = [
            row
            for row in payload
            if isinstance(
                row,
                dict,
            )
        ]

    elif isinstance(
        payload,
        dict,
    ):
        rows = []

        for key in (
            "rows",
            "records",
            "games",
            "signals",
            "board",
            "data",
        ):
            value = payload.get(
                key
            )

            if isinstance(
                value,
                list,
            ):
                rows = [
                    row
                    for row in value
                    if isinstance(
                        row,
                        dict,
                    )
                ]

                if rows:
                    break

    else:
        rows = []

    if not rows:
        raise RuntimeError(
            "Could not identify game rows in live board JSON."
        )

    return rows


def prepare_scaled_input(
    row: dict[str, Any],
    artifacts: ArtifactBundle,
) -> tuple[np.ndarray, list[str]]:
    raw = []

    imputed_features = []

    for feature in artifacts.feature_names:
        value = safe_float(
            row.get(
                feature
            )
        )

        if value is None:
            value = artifacts.medians[
                feature
            ]

            imputed_features.append(
                feature
            )

        raw.append(
            value
        )

    raw_array = np.asarray(
        [
            raw
        ],
        dtype=np.float64,
    )

    # Reproduce StandardScaler.transform directly from its saved fitted
    # parameters. This avoids sklearn's feature-name warning while preserving
    # the exact numerical transformation used by production.
    scaler_mean = np.asarray(
        artifacts.scaler.mean_,
        dtype=np.float64,
    )

    scaler_scale = np.asarray(
        artifacts.scaler.scale_,
        dtype=np.float64,
    )

    if (
        scaler_mean.shape[0] != EXPECTED_FEATURE_COUNT
        or
        scaler_scale.shape[0] != EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Saved scaler does not match the 54-feature contract."
        )

    safe_scale = np.where(
        scaler_scale == 0.0,
        1.0,
        scaler_scale,
    )

    scaled = (
        (
            raw_array
            -
            scaler_mean.reshape(1, -1)
        )
        /
        safe_scale.reshape(1, -1)
    ).astype(
        np.float32
    )

    if scaled.shape != (
        1,
        EXPECTED_FEATURE_COUNT,
    ):
        raise RuntimeError(
            f"Unexpected scaled input shape: {scaled.shape}"
        )

    return (
        scaled[
            0
        ],
        imputed_features,
    )


# ============================================================================
# ENSEMBLE INFERENCE
# ============================================================================

def run_ensemble(
    x_scaled: np.ndarray,
    artifacts: ArtifactBundle,
) -> dict[str, Any]:
    tensor = torch.tensor(
        x_scaled.reshape(
            1,
            -1
        ),
        dtype=torch.float32,
    )

    standardized_margins = []

    home_probabilities = []

    with torch.no_grad():
        for model in artifacts.models:
            (
                standardized_margin,
                win_logit,
            ) = model(
                tensor
            )

            standardized_margin_value = float(
                standardized_margin
                .detach()
                .cpu()
                .reshape(
                    -1
                )[
                    0
                ]
                .item()
            )

            win_logit_value = float(
                win_logit
                .detach()
                .cpu()
                .reshape(
                    -1
                )[
                    0
                ]
                .item()
            )

            standardized_margins.append(
                standardized_margin_value
            )

            probability = (
                1.0
                /
                (
                    1.0
                    +
                    math.exp(
                        -max(
                            -50.0,
                            min(
                                50.0,
                                win_logit_value
                            )
                        )
                    )
                )
            )

            home_probabilities.append(
                probability
            )

    raw_margins = [
        (
            value
            *
            artifacts.margin_std
            +
            artifacts.margin_mean
        )
        for value in standardized_margins
    ]

    return {
        "home_margin": float(
            np.mean(
                raw_margins
            )
        ),
        "home_win_probability": float(
            np.mean(
                home_probabilities
            )
        ),
        "seed_home_margins": raw_margins,
        "seed_home_probabilities": home_probabilities,
    }


# ============================================================================
# PRODUCTION REPRODUCTION CHECK
# ============================================================================

def production_home_margin(
    row: dict[str, Any],
) -> float | None:
    for key in (
        "nn_projected_home_margin",
        "model_home_margin",
    ):
        value = safe_float(
            row.get(
                key
            )
        )

        if value is not None:
            return value

    # Last-resort reconstruction from sportsbook-style model spread.
    spread = safe_float(
        row.get(
            "model_spread"
        )
    )

    favorite = str(
        row.get(
            "model_favorite"
        )
        or
        ""
    )

    home = str(
        row.get(
            "home_team"
        )
        or
        ""
    )

    if (
        spread is not None
        and
        favorite
        and
        home
    ):
        magnitude = abs(
            spread
        )

        if same_team(
            favorite,
            home,
        ):
            return magnitude

        return -magnitude

    return None


def production_home_probability(
    row: dict[str, Any],
) -> float | None:
    for key in (
        "nn_home_win_probability_raw",
        "model_home_win_probability",
    ):
        value = safe_float(
            row.get(
                key
            )
        )

        if value is not None:
            if value > 1.0:
                value /= 100.0

            return value

    favorite_probability = safe_float(
        row.get(
            "model_favorite_win_probability"
        )
    )

    if favorite_probability is None:
        return None

    if favorite_probability > 1.0:
        favorite_probability /= 100.0

    favorite = str(
        row.get(
            "model_favorite"
        )
        or
        ""
    )

    home = str(
        row.get(
            "home_team"
        )
        or
        ""
    )

    if same_team(
        favorite,
        home,
    ):
        return favorite_probability

    return (
        1.0
        -
        favorite_probability
    )


def verify_reproduction(
    row: dict[str, Any],
    reproduced: dict[str, Any],
) -> dict[str, Any]:
    expected_margin = production_home_margin(
        row
    )

    expected_probability = production_home_probability(
        row
    )

    if expected_margin is None:
        raise RuntimeError(
            "Board row does not contain a reproducible production home margin."
        )

    if expected_probability is None:
        raise RuntimeError(
            "Board row does not contain a reproducible production home probability."
        )

    margin_difference = (
        reproduced[
            "home_margin"
        ]
        -
        expected_margin
    )

    probability_difference = (
        reproduced[
            "home_win_probability"
        ]
        -
        expected_probability
    )

    passed = (
        abs(
            margin_difference
        )
        <=
        MARGIN_REPRODUCTION_TOLERANCE
        and
        abs(
            probability_difference
        )
        <=
        PROBABILITY_REPRODUCTION_TOLERANCE
    )

    return {
        "passed": passed,
        "expected_home_margin": expected_margin,
        "reproduced_home_margin": reproduced[
            "home_margin"
        ],
        "margin_difference": margin_difference,
        "expected_home_win_probability": expected_probability,
        "reproduced_home_win_probability": reproduced[
            "home_win_probability"
        ],
        "probability_difference": probability_difference,
    }


# ============================================================================
# FAVORITE ORIENTATION
# ============================================================================

def favorite_orientation(
    row: dict[str, Any],
) -> tuple[str, str, float]:
    favorite = str(
        row.get(
            "model_favorite"
        )
        or
        ""
    ).strip()

    home = str(
        row.get(
            "home_team"
        )
        or
        ""
    ).strip()

    away = str(
        row.get(
            "away_team"
        )
        or
        ""
    ).strip()

    if same_team(
        favorite,
        home,
    ):
        return (
            favorite,
            away,
            1.0,
        )

    if same_team(
        favorite,
        away,
    ):
        return (
            favorite,
            home,
            -1.0,
        )

    raise RuntimeError(
        f"Model favorite '{favorite}' does not match game teams."
    )


# ============================================================================
# ABLATION ATTRIBUTION
# ============================================================================

def favorite_probability_from_home_probability(
    home_probability: float,
    orientation: float,
) -> float:
    if orientation > 0:
        return home_probability

    return (
        1.0
        -
        home_probability
    )


def calculate_feature_impacts(
    row: dict[str, Any],
    x_scaled: np.ndarray,
    baseline_prediction: dict[str, Any],
    artifacts: ArtifactBundle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    (
        favorite,
        underdog,
        orientation,
    ) = favorite_orientation(
        row
    )

    baseline_favorite_probability = (
        favorite_probability_from_home_probability(
            baseline_prediction[
                "home_win_probability"
            ],
            orientation,
        )
    )

    feature_results = []

    for index, feature in enumerate(
        artifacts.feature_names
    ):
        ablated = x_scaled.copy()

        # Standardized zero = saved scaler's training mean for this feature.
        ablated[
            index
        ] = 0.0

        prediction = run_ensemble(
            ablated,
            artifacts,
        )

        favorite_margin_impact = (
            (
                baseline_prediction[
                    "home_margin"
                ]
                -
                prediction[
                    "home_margin"
                ]
            )
            *
            orientation
        )

        ablated_favorite_probability = (
            favorite_probability_from_home_probability(
                prediction[
                    "home_win_probability"
                ],
                orientation,
            )
        )

        favorite_probability_impact_pp = (
            baseline_favorite_probability
            -
            ablated_favorite_probability
        ) * 100.0

        raw_value = safe_float(
            row.get(
                feature
            )
        )

        feature_results.append(
            {
                "feature": feature,
                "label": human_feature_label(
                    feature
                ),
                "group": group_for_feature(
                    feature
                ),
                "raw_value": raw_value,
                "standardized_value": rounded(
                    float(
                        x_scaled[
                            index
                        ]
                    ),
                    4,
                ),
                "favorite_margin_impact_points": rounded(
                    favorite_margin_impact,
                    4,
                ),
                "favorite_win_probability_impact_pp": rounded(
                    favorite_probability_impact_pp,
                    4,
                ),
                "supports": (
                    favorite
                    if favorite_margin_impact > 0
                    else
                    underdog
                    if favorite_margin_impact < 0
                    else
                    "NEUTRAL"
                ),
                "impact_descriptor": impact_descriptor(
                    favorite_margin_impact
                ),
                "method": (
                    "single_feature_ablation_to_saved_scaler_training_mean"
                ),
            }
        )

    group_results = []

    feature_index = {
        feature: index
        for index, feature in enumerate(
            artifacts.feature_names
        )
    }

    for group_key, group in FEATURE_GROUPS.items():
        group_features = [
            feature
            for feature in artifacts.feature_names
            if feature in group[
                "features"
            ]
        ]

        if not group_features:
            continue

        ablated = x_scaled.copy()

        for feature in group_features:
            ablated[
                feature_index[
                    feature
                ]
            ] = 0.0

        prediction = run_ensemble(
            ablated,
            artifacts,
        )

        favorite_margin_impact = (
            (
                baseline_prediction[
                    "home_margin"
                ]
                -
                prediction[
                    "home_margin"
                ]
            )
            *
            orientation
        )

        ablated_favorite_probability = (
            favorite_probability_from_home_probability(
                prediction[
                    "home_win_probability"
                ],
                orientation,
            )
        )

        favorite_probability_impact_pp = (
            baseline_favorite_probability
            -
            ablated_favorite_probability
        ) * 100.0

        group_results.append(
            {
                "group": group_key,
                "label": group[
                    "label"
                ],
                "feature_count": len(
                    group_features
                ),
                "features": group_features,
                "favorite_margin_impact_points": rounded(
                    favorite_margin_impact,
                    4,
                ),
                "favorite_win_probability_impact_pp": rounded(
                    favorite_probability_impact_pp,
                    4,
                ),
                "supports": (
                    favorite
                    if favorite_margin_impact > 0
                    else
                    underdog
                    if favorite_margin_impact < 0
                    else
                    "NEUTRAL"
                ),
                "impact_descriptor": impact_descriptor(
                    favorite_margin_impact
                ),
                "method": (
                    "group_ablation_to_saved_scaler_training_mean"
                ),
            }
        )

    return (
        feature_results,
        group_results,
    )


# ============================================================================
# ONE GAME
# ============================================================================

def build_game_explanation(
    row: dict[str, Any],
    artifacts: ArtifactBundle,
) -> dict[str, Any]:
    (
        x_scaled,
        imputed_features,
    ) = prepare_scaled_input(
        row,
        artifacts,
    )

    baseline = run_ensemble(
        x_scaled,
        artifacts,
    )

    reproduction = verify_reproduction(
        row,
        baseline,
    )

    if not reproduction[
        "passed"
    ]:
        game = (
            f"{row.get('away_team', '')} @ "
            f"{row.get('home_team', '')}"
        )

        raise RuntimeError(
            "EXPLANATION REPRODUCTION FAILURE\n"
            f"Game: {game}\n"
            f"Margin expected/reproduced/diff: "
            f"{reproduction['expected_home_margin']:.4f} / "
            f"{reproduction['reproduced_home_margin']:.4f} / "
            f"{reproduction['margin_difference']:+.4f}\n"
            f"Home p expected/reproduced/diff: "
            f"{reproduction['expected_home_win_probability']:.6f} / "
            f"{reproduction['reproduced_home_win_probability']:.6f} / "
            f"{reproduction['probability_difference']:+.6f}\n"
            "No explanation file was written."
        )

    (
        favorite,
        underdog,
        orientation,
    ) = favorite_orientation(
        row
    )

    (
        feature_impacts,
        group_impacts,
    ) = calculate_feature_impacts(
        row,
        x_scaled,
        baseline,
        artifacts,
    )

    strongest_toward_favorite = sorted(
        [
            item
            for item in feature_impacts
            if (
                item[
                    "favorite_margin_impact_points"
                ]
                or
                0.0
            )
            >
            0.0
        ],
        key=lambda item: abs(
            item[
                "favorite_margin_impact_points"
            ]
            or
            0.0
        ),
        reverse=True,
    )[
        :8
    ]

    strongest_against_favorite = sorted(
        [
            item
            for item in feature_impacts
            if (
                item[
                    "favorite_margin_impact_points"
                ]
                or
                0.0
            )
            <
            0.0
        ],
        key=lambda item: abs(
            item[
                "favorite_margin_impact_points"
            ]
            or
            0.0
        ),
        reverse=True,
    )[
        :6
    ]

    strongest_groups = sorted(
        group_impacts,
        key=lambda item: abs(
            item[
                "favorite_margin_impact_points"
            ]
            or
            0.0
        ),
        reverse=True,
    )

    favorite_probability = (
        favorite_probability_from_home_probability(
            baseline[
                "home_win_probability"
            ],
            orientation,
        )
    )

    return {
        "game_id": row.get(
            "game_id"
        ),
        "season": row.get(
            "season"
        ),
        "week": row.get(
            "week"
        ),
        "start_date": row.get(
            "start_date"
        ),
        "away_team": row.get(
            "away_team"
        ),
        "home_team": row.get(
            "home_team"
        ),
        "model_favorite": favorite,
        "model_underdog": underdog,
        "model_spread_display": row.get(
            "model_spread_display"
        ),
        "model_favorite_win_probability": rounded(
            favorite_probability,
            6,
        ),
        "hardrock_display": row.get(
            "hardrock_display"
        ),
        "reference_market_display": row.get(
            "reference_market_display"
        ),
        "strength_label": row.get(
            "strength_label"
        ),
        "signal_warning": row.get(
            "signal_warning"
        ),
        "nn_input_status": row.get(
            "nn_input_status"
        ),
        "nn_heads_agree": row.get(
            "nn_heads_agree"
        ),
        "nn_margin_std": row.get(
            "nn_margin_std"
        ),
        "nn_win_probability_std": row.get(
            "nn_win_probability_std"
        ),
        "imputed_feature_count": len(
            imputed_features
        ),
        "imputed_features": imputed_features,
        "reproduction_check": {
            key: (
                rounded(
                    value,
                    8,
                )
                if isinstance(
                    value,
                    float,
                )
                else
                value
            )
            for key, value in reproduction.items()
        },
        "method": {
            "name": "LOCAL_NEURAL_ABLATION_V1",
            "baseline": (
                "saved StandardScaler standardized value 0.0 "
                "(training-distribution mean after saved imputation)"
            ),
            "ensemble_models": len(
                artifacts.models
            ),
            "ensemble_seeds": artifacts.seeds,
            "causal_claim": False,
            "additive_claim": False,
            "description": (
                "Impact measures how much the frozen ensemble output changes "
                "when the selected input is neutralized while all other inputs "
                "remain exactly as observed."
            ),
        },
        "top_features_toward_favorite": strongest_toward_favorite,
        "top_features_against_favorite": strongest_against_favorite,
        "group_impacts": strongest_groups,
        "all_feature_impacts": sorted(
            feature_impacts,
            key=lambda item: abs(
                item[
                    "favorite_margin_impact_points"
                ]
                or
                0.0
            ),
            reverse=True,
        ),
    }


# ============================================================================
# REPORT BUILD
# ============================================================================

def build_report() -> dict[str, Any]:
    artifacts = load_artifacts()

    board_rows = load_board()

    explanations = []

    reproduction_differences_margin = []

    reproduction_differences_probability = []

    for index, row in enumerate(
        board_rows,
        start=1,
    ):
        explanation = build_game_explanation(
            row,
            artifacts,
        )

        explanations.append(
            explanation
        )

        reproduction = explanation[
            "reproduction_check"
        ]

        reproduction_differences_margin.append(
            abs(
                float(
                    reproduction[
                        "margin_difference"
                    ]
                )
            )
        )

        reproduction_differences_probability.append(
            abs(
                float(
                    reproduction[
                        "probability_difference"
                    ]
                )
            )
        )

        game = (
            f"{row.get('away_team', '')} @ "
            f"{row.get('home_team', '')}"
        )

        print(
            f"[{index:>2}/{len(board_rows)}] "
            f"PASS  {game}"
        )

    return {
        "schema_version": 1,
        "generated_utc": utc_iso(),
        "season": 2026,
        "method": "LOCAL_NEURAL_ABLATION_V1",
        "artifact_version": artifacts.metadata.get(
            "artifact_version"
        ),
        "feature_count": len(
            artifacts.feature_names
        ),
        "ensemble_seeds": artifacts.seeds,
        "games_explained": len(
            explanations
        ),
        "reproduction_validation": {
            "status": "PASS",
            "margin_tolerance_points": MARGIN_REPRODUCTION_TOLERANCE,
            "probability_tolerance": PROBABILITY_REPRODUCTION_TOLERANCE,
            "max_absolute_margin_difference_points": rounded(
                max(
                    reproduction_differences_margin,
                    default=0.0,
                ),
                8,
            ),
            "max_absolute_probability_difference": rounded(
                max(
                    reproduction_differences_probability,
                    default=0.0,
                ),
                8,
            ),
        },
        "interpretation_warning": (
            "Ablation impact is a local model-sensitivity measurement, not a "
            "causal football claim. Feature and group impacts are not additive."
        ),
        "explanations": explanations,
    }


def write_outputs(
    report: dict[str, Any],
) -> None:
    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            default=str,
        )

        handle.write(
            "\n"
        )

    csv_rows = []

    for explanation in report[
        "explanations"
    ]:
        for rank, item in enumerate(
            explanation[
                "all_feature_impacts"
            ],
            start=1,
        ):
            csv_rows.append(
                {
                    "game_id": explanation.get(
                        "game_id"
                    ),
                    "week": explanation.get(
                        "week"
                    ),
                    "away_team": explanation.get(
                        "away_team"
                    ),
                    "home_team": explanation.get(
                        "home_team"
                    ),
                    "model_favorite": explanation.get(
                        "model_favorite"
                    ),
                    "model_spread_display": explanation.get(
                        "model_spread_display"
                    ),
                    "strength_label": explanation.get(
                        "strength_label"
                    ),
                    "rank_by_absolute_margin_impact": rank,
                    "feature": item.get(
                        "feature"
                    ),
                    "label": item.get(
                        "label"
                    ),
                    "group": item.get(
                        "group"
                    ),
                    "raw_value": item.get(
                        "raw_value"
                    ),
                    "standardized_value": item.get(
                        "standardized_value"
                    ),
                    "favorite_margin_impact_points": item.get(
                        "favorite_margin_impact_points"
                    ),
                    "favorite_win_probability_impact_pp": item.get(
                        "favorite_win_probability_impact_pp"
                    ),
                    "supports": item.get(
                        "supports"
                    ),
                }
            )

    fieldnames = [
        "game_id",
        "week",
        "away_team",
        "home_team",
        "model_favorite",
        "model_spread_display",
        "strength_label",
        "rank_by_absolute_margin_impact",
        "feature",
        "label",
        "group",
        "raw_value",
        "standardized_value",
        "favorite_margin_impact_points",
        "favorite_win_probability_impact_pp",
        "supports",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )


# ============================================================================
# CONSOLE SUMMARY
# ============================================================================

def print_summary(
    report: dict[str, Any],
) -> None:
    validation = report[
        "reproduction_validation"
    ]

    print()
    print_rule()
    print(
        "CFB EDGE ENGINE — PREDICTION EXPLANATIONS V1"
    )
    print_rule()

    print(
        f"Method:              {report['method']}"
    )

    print(
        f"Games explained:     {report['games_explained']}"
    )

    print(
        f"Features:            {report['feature_count']}"
    )

    print(
        f"Ensemble seeds:      {report['ensemble_seeds']}"
    )

    print(
        "Production replay:  PASS"
    )

    print(
        f"Max margin mismatch: "
        f"{validation['max_absolute_margin_difference_points']:.6f} pts"
    )

    print(
        f"Max probability mismatch: "
        f"{validation['max_absolute_probability_difference']:.6f}"
    )

    print()
    print(
        "EXAMPLE EXPLANATION"
    )
    print_rule(
        "-"
    )

    if report[
        "explanations"
    ]:
        explanation = report[
            "explanations"
        ][
            0
        ]

        print(
            f"{explanation['away_team']} @ "
            f"{explanation['home_team']}"
        )

        print(
            f"Model: "
            f"{explanation['model_spread_display']}"
        )

        print(
            f"Why the model leans "
            f"{explanation['model_favorite']}:"
        )

        for item in explanation[
            "top_features_toward_favorite"
        ][
            :5
        ]:
            impact = float(
                item[
                    "favorite_margin_impact_points"
                ]
            )

            print(
                f"  + {item['label']:<34} "
                f"{impact:+.2f} pts"
            )

        against = explanation[
            "top_features_against_favorite"
        ]

        if against:
            print()
            print(
                "Inputs pushing the other way:"
            )

            for item in against[
                :3
            ]:
                impact = float(
                    item[
                        "favorite_margin_impact_points"
                    ]
                )

                print(
                    f"  - {item['label']:<34} "
                    f"{impact:+.2f} pts"
                )

    print()
    print(
        "FILES WRITTEN"
    )
    print_rule(
        "-"
    )

    print(
        OUTPUT_JSON
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        "IMPORTANT: These are local neural-network ablation impacts, "
        "not causal claims and not additive point decompositions."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> int:
    print_rule()
    print(
        "CFB EDGE ENGINE — BUILDING PREDICTION EXPLANATIONS V1"
    )
    print_rule()

    print(
        "Mode: LOCAL ONLY / NO API CALLS / NO RETRAINING"
    )

    print(
        f"Predictor: {PREDICTOR_FILE}"
    )

    print(
        f"Artifact:  {MODEL_DIR}"
    )

    print(
        f"Board:     {BOARD_FILE}"
    )

    print()
    print(
        "Reproducing the exact frozen ensemble before calculating any explanation..."
    )
    print()

    try:
        report = build_report()

        # Write only AFTER every game's reproduction check has passed.
        write_outputs(
            report
        )

        print_summary(
            report
        )

        return 0

    except Exception as error:
        print()
        print_rule()
        print(
            "EXPLANATION BUILD FAILED CLOSED"
        )
        print_rule()

        print(
            str(
                error
            )
        )

        print()
        print(
            "No new explanation report was written."
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
