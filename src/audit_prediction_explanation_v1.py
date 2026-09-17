#!/usr/bin/env python3
"""
CFB Edge Engine — Prediction Explanation Readiness Audit V1

Purpose
-------
Before building "WHY THE MODEL LEANS ..." explanations, inventory the exact
frozen 2026 neural-network artifacts and live 54-feature contract.

This prevents us from inventing explanations based on assumed model/scaler
filenames or guessed internals.

This script:
- makes NO API calls
- does NOT load/execute pickle/joblib artifacts
- does NOT retrain anything
- does NOT modify any model, prediction, market, history, or wager data
- reads local source text / JSON / CSV metadata only

Run
---
python src/audit_prediction_explanation_v1.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "cfb_nn_2026_v1"
)

PREDICTOR = (
    PROJECT_ROOT
    / "src"
    / "predict_live_2026_nn_v1.py"
)

FEATURE_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_features_2026_v1.csv"
)

PREDICTION_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_nn_predictions_2026_v1.json"
)

BOARD_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_model_market_board_2026_v1.json"
)

EXPECTED_FEATURE_COUNT = 54

STATIC_FEATURES = {
    "neutral_site",
    "pregame_elo_diff",
    "returning_passing_ppa_diff",
    "returning_ppa_diff",
    "returning_receiving_ppa_diff",
    "returning_rushing_ppa_diff",
    "talent_diff",
}


def print_rule(
    char: str = "=",
    width: int = 108,
) -> None:
    print(
        char * width
    )


def safe_json_load(
    path: Path,
) -> Any:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(
                handle
            )
    except Exception as error:
        return {
            "__read_error__": str(
                error
            )
        }


def human_size(
    byte_count: int,
) -> str:
    value = float(
        byte_count
    )

    for suffix in (
        "B",
        "KB",
        "MB",
        "GB",
    ):
        if value < 1024.0:
            return (
                f"{value:.1f} "
                f"{suffix}"
            )
        value /= 1024.0

    return (
        f"{value:.1f} TB"
    )


def flatten_top_level_keys(
    payload: Any,
) -> list[str]:
    if isinstance(
        payload,
        dict,
    ):
        return sorted(
            str(
                key
            )
            for key in payload.keys()
        )

    if isinstance(
        payload,
        list,
    ):
        return [
            f"<LIST:{len(payload)}>"
        ]

    return [
        f"<{type(payload).__name__}>"
    ]


def first_record(
    payload: Any,
) -> dict[str, Any] | None:
    if isinstance(
        payload,
        list,
    ):
        for item in payload:
            if isinstance(
                item,
                dict,
            ):
                return item
        return None

    if not isinstance(
        payload,
        dict,
    ):
        return None

    for key in (
        "records",
        "rows",
        "games",
        "predictions",
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
            for item in value:
                if isinstance(
                    item,
                    dict,
                ):
                    return item

    candidates = [
        value
        for value in payload.values()
        if isinstance(
            value,
            dict,
        )
    ]

    return (
        candidates[0]
        if candidates
        else None
    )


def inspect_model_directory() -> None:
    print_rule()
    print(
        "FROZEN MODEL DIRECTORY"
    )
    print_rule()

    print(
        f"Path: {MODEL_DIR}"
    )

    if not MODEL_DIR.exists():
        print(
            "STATUS: MISSING"
        )
        return

    files = sorted(
        [
            path
            for path in MODEL_DIR.rglob(
                "*"
            )
            if path.is_file()
        ],
        key=lambda path: str(
            path.relative_to(
                MODEL_DIR
            )
        ).lower(),
    )

    print(
        f"Files: {len(files)}"
    )

    if not files:
        print(
            "(empty model directory)"
        )
        return

    for path in files:
        relative = path.relative_to(
            MODEL_DIR
        )

        print(
            f"  {relative} "
            f"({human_size(path.stat().st_size)})"
        )

    json_files = [
        path
        for path in files
        if path.suffix.lower()
        ==
        ".json"
    ]

    if json_files:
        print()
        print(
            "JSON METADATA"
        )
        print_rule(
            "-"
        )

        for path in json_files:
            relative = path.relative_to(
                MODEL_DIR
            )

            payload = safe_json_load(
                path
            )

            print(
                f"{relative}"
            )

            if (
                isinstance(
                    payload,
                    dict,
                )
                and
                "__read_error__"
                in payload
            ):
                print(
                    f"  READ ERROR: "
                    f"{payload['__read_error__']}"
                )
                continue

            print(
                "  top-level keys: "
                + ", ".join(
                    flatten_top_level_keys(
                        payload
                    )
                )
            )

            if isinstance(
                payload,
                dict,
            ):
                for likely_key in (
                    "feature_names",
                    "features",
                    "model_features",
                    "seeds",
                    "input_dim",
                    "hidden_dims",
                    "architecture",
                    "model_type",
                    "scaler",
                    "imputer",
                ):
                    if likely_key in payload:
                        value = payload[
                            likely_key
                        ]

                        if isinstance(
                            value,
                            list,
                        ):
                            preview = value[
                                :10
                            ]

                            suffix = (
                                " ..."
                                if len(
                                    value
                                )
                                >
                                10
                                else
                                ""
                            )

                            print(
                                f"  {likely_key}: "
                                f"{preview}{suffix} "
                                f"(n={len(value)})"
                            )

                        else:
                            print(
                                f"  {likely_key}: "
                                f"{value}"
                            )


def inspect_predictor_source() -> None:
    print()
    print_rule()
    print(
        "PREDICTOR SOURCE CONTRACT"
    )
    print_rule()

    print(
        f"Path: {PREDICTOR}"
    )

    if not PREDICTOR.exists():
        print(
            "STATUS: MISSING"
        )
        return

    text = PREDICTOR.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    print(
        f"Lines: {len(lines)}"
    )

    patterns = [
        r"torch\.load",
        r"joblib\.load",
        r"pickle\.load",
        r"np\.load",
        r"json\.load",
        r"StandardScaler",
        r"SimpleImputer",
        r"feature_names",
        r"model_features",
        r"hidden",
        r"Linear\(",
        r"Dropout\(",
        r"state_dict",
        r"MODEL_DIR",
        r"ARTIFACT",
        r"seed",
    ]

    combined = re.compile(
        "|".join(
            f"(?:{pattern})"
            for pattern in patterns
        ),
        flags=re.IGNORECASE,
    )

    matches = []

    for number, line in enumerate(
        lines,
        start=1,
    ):
        if combined.search(
            line
        ):
            cleaned = line.strip()

            if cleaned:
                matches.append(
                    (
                        number,
                        cleaned,
                    )
                )

    print()
    print(
        "Relevant loader / architecture / feature lines:"
    )
    print_rule(
        "-"
    )

    if not matches:
        print(
            "  No matching lines detected."
        )
        return

    for number, line in matches[
        :160
    ]:
        print(
            f"  L{number}: {line}"
        )

    if len(
        matches
    ) > 160:
        print(
            f"  ... {len(matches) - 160} "
            f"additional matching lines omitted"
        )


def identify_feature_columns(
    columns: list[str],
) -> list[str]:
    rolling = [
        column
        for column in columns
        if column.startswith(
            "diff_pregame_"
        )
    ]

    static = [
        column
        for column in columns
        if column
        in
        STATIC_FEATURES
    ]

    combined = sorted(
        set(
            rolling
            +
            static
        )
    )

    return combined


def inspect_live_features() -> None:
    print()
    print_rule()
    print(
        "LIVE FEATURE FILE"
    )
    print_rule()

    print(
        f"Path: {FEATURE_CSV}"
    )

    if not FEATURE_CSV.exists():
        print(
            "STATUS: MISSING"
        )
        return

    with FEATURE_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        columns = list(
            reader.fieldnames
            or
            []
        )

        rows = list(
            reader
        )

    print(
        f"Rows: {len(rows)}"
    )

    print(
        f"Columns: {len(columns)}"
    )

    feature_columns = identify_feature_columns(
        columns
    )

    print(
        f"Detected model-like feature columns: "
        f"{len(feature_columns)}"
    )

    if len(
        feature_columns
    ) == EXPECTED_FEATURE_COUNT:
        print(
            "54-feature detection: PASS"
        )
    else:
        print(
            "54-feature detection: REVIEW"
        )

    print()
    print(
        "Detected features:"
    )
    print_rule(
        "-"
    )

    for index, column in enumerate(
        feature_columns,
        start=1,
    ):
        print(
            f"  {index:>2}. {column}"
        )

    print()
    print(
        "Identity / metadata columns:"
    )
    print_rule(
        "-"
    )

    feature_set = set(
        feature_columns
    )

    metadata = [
        column
        for column in columns
        if column
        not in
        feature_set
    ]

    for column in metadata:
        print(
            f"  {column}"
        )

    if rows:
        sample = rows[
            0
        ]

        print()
        print(
            "First live row identity:"
        )
        print_rule(
            "-"
        )

        for key in (
            "game_id",
            "season",
            "week",
            "away_team",
            "home_team",
            "start_date",
            "kickoff_utc",
            "model_non_null_features",
            "context_ready",
        ):
            if key in sample:
                print(
                    f"  {key}: "
                    f"{sample.get(key)}"
                )


def inspect_json_contract(
    title: str,
    path: Path,
) -> None:
    print()
    print_rule()
    print(
        title
    )
    print_rule()

    print(
        f"Path: {path}"
    )

    if not path.exists():
        print(
            "STATUS: MISSING"
        )
        return

    payload = safe_json_load(
        path
    )

    if (
        isinstance(
            payload,
            dict,
        )
        and
        "__read_error__"
        in payload
    ):
        print(
            "READ ERROR: "
            + payload[
                "__read_error__"
            ]
        )
        return

    print(
        "Top-level keys: "
        + ", ".join(
            flatten_top_level_keys(
                payload
            )
        )
    )

    record = first_record(
        payload
    )

    if record is None:
        print(
            "No dictionary record detected."
        )
        return

    print(
        "First-record keys:"
    )

    for key in sorted(
        record.keys()
    ):
        print(
            f"  {key}"
        )


def print_recommendation() -> None:
    print()
    print_rule()
    print(
        "EXPLANATION BUILD DECISION"
    )
    print_rule()

    print(
        "Paste this audit output back into the project chat."
    )

    print()

    print(
        "The next implementation will choose the strongest truthful "
        "explanation method supported by the actual frozen artifact:"
    )

    print(
        "  1. Preferred: neural local attribution using the exact scaler, "
        "imputer, and five frozen model weights."
    )

    print(
        "  2. If the artifact contract makes gradient attribution unsafe, "
        "use deterministic leave-one-feature/group ablation."
    )

    print(
        "  3. Raw feature-direction summaries may supplement the attribution, "
        "but will never be labeled as causal model influence by themselves."
    )

    print()

    print(
        "No explanation code should claim that a feature 'drove' the neural "
        "prediction until this exact artifact contract is confirmed."
    )


def main() -> int:
    print_rule()
    print(
        "CFB EDGE ENGINE — PREDICTION EXPLANATION READINESS AUDIT V1"
    )
    print_rule()

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        "Mode: READ ONLY / NO API CALLS / NO MODEL EXECUTION"
    )

    inspect_model_directory()

    inspect_predictor_source()

    inspect_live_features()

    inspect_json_contract(
        "LIVE NEURAL PREDICTION JSON",
        PREDICTION_JSON,
    )

    inspect_json_contract(
        "LIVE MODEL + MARKET BOARD JSON",
        BOARD_JSON,
    )

    print_recommendation()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
