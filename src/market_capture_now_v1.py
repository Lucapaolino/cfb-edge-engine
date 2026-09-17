#!/usr/bin/env python3
"""
CFB Edge Engine — Market Capture Now V1

Purpose
-------
Take a lightweight, near-kickoff market snapshot WITHOUT rerunning the football
data pipeline or neural model.

This workflow intentionally does ONLY:

    1. Refresh The Odds API reference market
    2. Refresh Hard Rock Bet Florida execution market
    3. Rebuild the live model/market board using the EXISTING frozen prediction
    4. Freeze a NEW immutable Prediction History snapshot
    5. Rebuild line-movement / CLV outputs
    6. Refresh local read-only support outputs (What Changed, Audit, Data Health)

It intentionally does NOT:
- call CFBD
- refresh games/results
- refresh team stats
- refresh player stats
- refresh season context
- rebuild the 54-feature slate
- rerun the neural ensemble
- retrain any model
- score old predictions
- rewrite prior Prediction History rows

Why this exists
---------------
A full RUN GAME DAY is appropriate when football inputs may have changed.
Near kickoff, however, we often only need another market observation so that
line movement and closing-line research have a trustworthy pre-kickoff trail.

The neural prediction file is hashed before the market refresh and checked
again afterward. If it changes, this workflow FAILS CLOSED.

Run
---
    python src/market_capture_now_v1.py

Dry run
-------
    python src/market_capture_now_v1.py --dry-run

Optional
--------
    python src/market_capture_now_v1.py --update-wager-ledger

The optional flag lets market_history_clv_v1.py write derived CLV status/points
into the Actual Wager Ledger. Original ticket fields are not changed by that
downstream script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import sqlite3
import subprocess
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROSPECTIVE_DIR = DATA_DIR / "prospective"

PREDICTIONS_JSON = PROCESSED_DIR / "live_nn_predictions_2026_v1.json"
BOARD_JSON = PROCESSED_DIR / "live_model_market_board_2026_v1.json"
HISTORY_DB = PROSPECTIVE_DIR / "prediction_history.sqlite"

MARKET_HISTORY_JSON = PROCESSED_DIR / "market_history_2026_v1.json"
WAGER_CLV_JSON = PROCESSED_DIR / "wager_clv_2026_v1.json"

SCRIPTS = {
    "reference_market": SRC_DIR / "snapshot_execution_market_2026_v1.py",
    "hardrock": SRC_DIR / "snapshot_hardrock_oddspapi_2026_v1.py",
    "board": SRC_DIR / "build_live_model_market_board_v1.py",
    "history_log": SRC_DIR / "log_predictions_v1.py",
    "market_history": SRC_DIR / "market_history_clv_v1.py",
    "what_changed": SRC_DIR / "what_changed_v1.py",
    "prediction_audit": SRC_DIR / "prediction_audit_v1.py",
    "data_health": SRC_DIR / "data_health_v1.py",
}

# These are the only environment variables this workflow expects API-backed
# market collectors to need. We check PRESENCE only and never print values.
REQUIRED_ENV_KEYS = (
    "THE_ODDS_API_KEY",
    "ODDSPAPI_API_KEY",
)

STAGE_PROGRESS = {
    "SYSTEM CHECK": 5,
    "REFRESHING REFERENCE MARKET": 18,
    "REFRESHING HARD ROCK": 36,
    "ASSEMBLING LIVE BOARD": 52,
    "VERIFYING FROZEN MODEL": 61,
    "FREEZING MARKET SNAPSHOT": 72,
    "UPDATING MARKET HISTORY + CLV": 84,
    "REFRESHING LOCAL SUPPORT OUTPUTS": 94,
    "CAPTURE COMPLETE": 100,
}

WIDTH = 108


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return (
        utc_now()
        .replace(microsecond=0)
        .isoformat()
    )


def print_rule(
    char: str = "=",
) -> None:
    print(
        char * WIDTH,
        flush=True,
    )


def print_stage(
    name: str,
) -> None:
    progress = STAGE_PROGRESS.get(
        name,
        0,
    )

    # Machine-readable line for later GUI integration.
    print(
        f"MARKET_CAPTURE_STAGE|{name}|{progress}",
        flush=True,
    )

    print_rule()
    print(
        name,
        flush=True,
    )
    print_rule()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def normalize_records(
    payload: Any,
) -> list[dict[str, Any]]:
    if isinstance(
        payload,
        list,
    ):
        return [
            row
            for row in payload
            if isinstance(
                row,
                dict,
            )
        ]

    if isinstance(
        payload,
        dict,
    ):
        for key in (
            "records",
            "rows",
            "games",
            "signals",
            "predictions",
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
                    return rows

        # Compatibility fallback for keyed-record payloads.
        rows = []

        for value in payload.values():
            if not isinstance(
                value,
                dict,
            ):
                continue

            if (
                "away_team" in value
                and
                "home_team" in value
            ):
                rows.append(
                    value
                )

        if rows:
            return rows

    return []


def safe_int(
    value: Any,
) -> int | None:
    try:
        if value in (
            None,
            "",
        ):
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


def load_env_presence() -> dict[str, bool]:
    """
    Check key presence only.

    We intentionally do not print, return, or otherwise expose secret values.
    """
    found = {
        key: bool(
            os.environ.get(
                key
            )
        )
        for key in REQUIRED_ENV_KEYS
    }

    env_file = PROJECT_ROOT / ".env"

    if not env_file.exists():
        return found

    try:
        for raw_line in env_file.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            line = raw_line.strip()

            if (
                not line
                or
                line.startswith(
                    "#"
                )
                or
                "=" not in line
            ):
                continue

            key, value = line.split(
                "=",
                1,
            )

            key = key.strip()

            if key not in found:
                continue

            value = value.strip().strip(
                "\"'"
            )

            if value:
                found[
                    key
                ] = True

    except OSError:
        pass

    return found


# =============================================================================
# PROCESS RUNNER
# =============================================================================

def run_script(
    stage_name: str,
    script_key: str,
    extra_args: list[str] | None = None,
) -> None:
    script = SCRIPTS[
        script_key
    ]

    print_stage(
        stage_name
    )

    print(
        f"Script: {script.relative_to(PROJECT_ROOT)}",
        flush=True,
    )

    command = [
        sys.executable,
        str(
            script
        ),
    ]

    if extra_args:
        command.extend(
            extra_args
        )

    print(
        f"Started UTC: {utc_iso()}",
        flush=True,
    )

    print(
        flush=True
    )

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError(
            f"Could not read output from {script.name}."
        )

    for line in process.stdout:
        print(
            line.rstrip(),
            flush=True,
        )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"{stage_name} failed: "
            f"{script.name} exited with code {return_code}."
        )

    print()
    print(
        f"Completed UTC: {utc_iso()}",
        flush=True,
    )


def run_local_support_scripts() -> None:
    print_stage(
        "REFRESHING LOCAL SUPPORT OUTPUTS"
    )

    support_order = (
        (
            "what_changed",
            "What Changed",
        ),
        (
            "prediction_audit",
            "Prediction Audit",
        ),
        (
            "data_health",
            "Data Health",
        ),
    )

    for key, label in support_order:
        script = SCRIPTS[
            key
        ]

        if not script.exists():
            print(
                f"SKIP     {label:<20} "
                f"{script.relative_to(PROJECT_ROOT)} missing",
                flush=True,
            )

            continue

        print()
        print_rule(
            "-"
        )

        print(
            f"{label}: {script.relative_to(PROJECT_ROOT)}",
            flush=True,
        )

        process = subprocess.Popen(
            [
                sys.executable,
                str(
                    script
                ),
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError(
                f"Could not read output from {script.name}."
            )

        for line in process.stdout:
            print(
                line.rstrip(),
                flush=True,
            )

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"Local support refresh failed: "
                f"{script.name} exited with code {return_code}."
            )


# =============================================================================
# PREFLIGHT / VALIDATION
# =============================================================================

def verify_required_files() -> None:
    print(
        "REQUIRED FILES"
    )

    print_rule(
        "-"
    )

    missing: list[Path] = []

    required = [
        PREDICTIONS_JSON,
        *SCRIPTS.values(),
    ]

    for path in required:
        exists = path.exists()

        print(
            f"{'PASS' if exists else 'MISSING':<8} "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

        if not exists:
            missing.append(
                path
            )

    if missing:
        raise RuntimeError(
            "Near-kickoff capture cannot run because required files are missing: "
            +
            ", ".join(
                str(
                    path
                )
                for path in missing
            )
        )


def verify_python_syntax() -> None:
    print()
    print(
        "PYTHON SYNTAX"
    )

    print_rule(
        "-"
    )

    for key, path in SCRIPTS.items():
        py_compile.compile(
            str(
                path
            ),
            doraise=True,
        )

        print(
            f"PASS     {key:<18} {path.name}"
        )


def verify_api_key_presence(
    *,
    dry_run: bool,
) -> None:
    print()
    print(
        "API KEY PRESENCE"
    )

    print_rule(
        "-"
    )

    presence = load_env_presence()

    missing = []

    for key in REQUIRED_ENV_KEYS:
        ok = bool(
            presence.get(
                key
            )
        )

        print(
            f"{'PASS' if ok else 'MISSING':<8} {key}"
        )

        if not ok:
            missing.append(
                key
            )

    if (
        missing
        and
        not dry_run
    ):
        raise RuntimeError(
            "Live market capture requires API key presence for: "
            +
            ", ".join(
                missing
            )
        )


def prediction_file_status() -> tuple[str, int]:
    payload = load_json(
        PREDICTIONS_JSON
    )

    rows = normalize_records(
        payload
    )

    if not rows:
        raise RuntimeError(
            "Frozen prediction JSON exists but contains no identifiable prediction rows."
        )

    digest = sha256_file(
        PREDICTIONS_JSON
    )

    print()
    print(
        "FROZEN MODEL INPUT"
    )

    print_rule(
        "-"
    )

    print(
        f"Prediction file: "
        f"{PREDICTIONS_JSON.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"Prediction rows: {len(rows)}"
    )

    print(
        f"SHA256:          {digest}"
    )

    print(
        "This file MUST remain byte-for-byte unchanged during market-only capture."
    )

    return (
        digest,
        len(
            rows
        ),
    )


def preflight(
    *,
    dry_run: bool,
) -> tuple[str, int]:
    print_stage(
        "SYSTEM CHECK"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Python:       {sys.executable}"
    )

    print(
        "Mode:         "
        +
        (
            "DRY RUN — NO API CALLS / NO HISTORY WRITES"
            if dry_run
            else
            "LIVE MARKET-ONLY CAPTURE"
        )
    )

    print()

    verify_required_files()
    verify_python_syntax()
    verify_api_key_presence(
        dry_run=dry_run
    )

    digest, count = prediction_file_status()

    print()
    print(
        "PREFLIGHT PASS"
    )

    return (
        digest,
        count,
    )


def verify_frozen_prediction(
    before_sha256: str,
    before_count: int,
) -> None:
    print_stage(
        "VERIFYING FROZEN MODEL"
    )

    if not PREDICTIONS_JSON.exists():
        raise RuntimeError(
            "Frozen prediction file disappeared during market capture."
        )

    after_sha256 = sha256_file(
        PREDICTIONS_JSON
    )

    payload = load_json(
        PREDICTIONS_JSON
    )

    rows = normalize_records(
        payload
    )

    after_count = len(
        rows
    )

    print(
        f"Before SHA256: {before_sha256}"
    )

    print(
        f"After SHA256:  {after_sha256}"
    )

    print(
        f"Before rows:   {before_count}"
    )

    print(
        f"After rows:    {after_count}"
    )

    if after_sha256 != before_sha256:
        raise RuntimeError(
            "FAIL CLOSED: live_nn_predictions_2026_v1.json changed during "
            "market-only capture. No immutable market snapshot will be logged."
        )

    if after_count != before_count:
        raise RuntimeError(
            "FAIL CLOSED: frozen prediction row count changed during "
            "market-only capture."
        )

    print()
    print(
        "PASS — neural prediction remained byte-for-byte frozen."
    )


def verify_board() -> int:
    if not BOARD_JSON.exists():
        raise RuntimeError(
            f"Live board was not written: {BOARD_JSON}"
        )

    payload = load_json(
        BOARD_JSON
    )

    rows = normalize_records(
        payload
    )

    if not rows:
        raise RuntimeError(
            "Live board exists but contains no identifiable game rows."
        )

    actionable = 0

    for row in rows:
        explicit = safe_int(
            row.get(
                "game_is_actionable"
            )
        )

        if explicit == 1:
            actionable += 1

    print()
    print(
        f"Live board rows:       {len(rows)}"
    )

    print(
        f"Actionable pregame:    {actionable}"
    )

    if actionable <= 0:
        raise RuntimeError(
            "Live board contains no actionable pregame rows; immutable market "
            "snapshot will not be logged."
        )

    return actionable


def verify_history_run(
    run_id: str,
) -> int:
    if not HISTORY_DB.exists():
        raise RuntimeError(
            f"Prediction History database missing after logging: {HISTORY_DB}"
        )

    connection = sqlite3.connect(
        f"file:{HISTORY_DB}?mode=ro",
        uri=True,
    )

    connection.row_factory = sqlite3.Row
    connection.execute(
        "PRAGMA query_only = ON"
    )

    try:
        run = connection.execute(
            """
            SELECT
                run_id,
                captured_utc,
                record_count,
                batch_file
            FROM prediction_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (
                run_id,
            ),
        ).fetchone()

        if run is None:
            raise RuntimeError(
                f"Prediction History did not contain new run_id {run_id}."
            )

        count_row = connection.execute(
            """
            SELECT COUNT(*)
            FROM prediction_snapshots
            WHERE run_id = ?
            """,
            (
                run_id,
            ),
        ).fetchone()

        snapshot_count = int(
            count_row[
                0
            ]
        )

        expected = int(
            run[
                "record_count"
            ]
        )

        if snapshot_count != expected:
            raise RuntimeError(
                "Prediction History run count mismatch: "
                f"prediction_runs says {expected}, "
                f"prediction_snapshots contains {snapshot_count}."
            )

        if snapshot_count <= 0:
            raise RuntimeError(
                "Prediction History logger created an empty run."
            )

        print()
        print(
            "IMMUTABLE HISTORY VERIFICATION"
        )

        print_rule(
            "-"
        )

        print(
            f"Run ID:          {run_id}"
        )

        print(
            f"Captured UTC:    {run['captured_utc']}"
        )

        print(
            f"Snapshots:       {snapshot_count}"
        )

        print(
            f"Batch file:      {run['batch_file']}"
        )

        print(
            "Status:          PASS"
        )

        return snapshot_count

    finally:
        connection.close()


def verify_market_history_outputs() -> None:
    print()
    print(
        "MARKET HISTORY OUTPUT CHECK"
    )

    print_rule(
        "-"
    )

    missing = []

    for path in (
        MARKET_HISTORY_JSON,
        WAGER_CLV_JSON,
    ):
        exists = path.exists()

        print(
            f"{'PASS' if exists else 'MISSING':<8} "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

        if not exists:
            missing.append(
                path
            )

    if missing:
        raise RuntimeError(
            "Market-history stage completed but expected outputs are missing."
        )


# =============================================================================
# LIVE WORKFLOW
# =============================================================================

def run_live(
    args: argparse.Namespace,
) -> None:
    run_id = (
        "market_"
        +
        utc_now().strftime(
            "%Y%m%dT%H%M%SZ"
        )
        +
        "_"
        +
        uuid.uuid4().hex[
            :8
        ]
    )

    print_rule()
    print(
        "CFB EDGE ENGINE — MARKET CAPTURE NOW V1"
    )
    print_rule()

    print(
        f"Run ID:      {run_id}"
    )

    print(
        f"Started UTC: {utc_iso()}"
    )

    print(
        "Purpose:     fresh market snapshot against EXISTING frozen model"
    )

    print(
        "Execution:   Hard Rock Bet Florida via OddsPapi"
    )

    print(
        "Reference:   The Odds API consensus"
    )

    print(
        "Model:       DO NOT RERUN"
    )

    print_rule()

    frozen_hash, frozen_count = preflight(
        dry_run=False
    )

    run_script(
        "REFRESHING REFERENCE MARKET",
        "reference_market",
    )

    run_script(
        "REFRESHING HARD ROCK",
        "hardrock",
    )

    run_script(
        "ASSEMBLING LIVE BOARD",
        "board",
    )

    actionable = verify_board()

    # IMPORTANT: Verify the model file BEFORE logging the immutable snapshot.
    verify_frozen_prediction(
        frozen_hash,
        frozen_count,
    )

    run_script(
        "FREEZING MARKET SNAPSHOT",
        "history_log",
        [
            "--run-id",
            run_id,
        ],
    )

    snapshot_count = verify_history_run(
        run_id
    )

    market_history_args = []

    if args.update_wager_ledger:
        market_history_args.append(
            "--update-ledger"
        )

    run_script(
        "UPDATING MARKET HISTORY + CLV",
        "market_history",
        market_history_args,
    )

    verify_market_history_outputs()

    if not args.skip_support:
        run_local_support_scripts()

    # Final immutability check: none of the downstream local scripts may have
    # changed the frozen neural predictions either.
    final_hash = sha256_file(
        PREDICTIONS_JSON
    )

    if final_hash != frozen_hash:
        raise RuntimeError(
            "FAIL CLOSED: frozen neural prediction file changed before capture "
            "workflow completed."
        )

    print_stage(
        "CAPTURE COMPLETE"
    )

    print(
        f"Run ID:                   {run_id}"
    )

    print(
        f"Completed UTC:            {utc_iso()}"
    )

    print(
        f"Actionable games:         {actionable}"
    )

    print(
        f"History snapshots frozen: {snapshot_count}"
    )

    print(
        "Neural prediction:        UNCHANGED"
    )

    print(
        "CFBD calls:               NONE"
    )

    print(
        "Feature rebuild:          NONE"
    )

    print(
        "Neural rerun:             NONE"
    )

    print(
        "Model training:           NONE"
    )

    print(
        "Wager ledger CLV update:  "
        +
        (
            "ENABLED"
            if args.update_wager_ledger
            else
            "NOT REQUESTED"
        )
    )

    print()
    print(
        "IMPORTANT"
    )

    print_rule(
        "-"
    )

    print(
        "This capture is a market observation, not a new football-model prediction."
    )

    print(
        "Strength may change because execution/reference pricing changed."
    )

    print(
        "A stored line is only treated as VERIFIED_NEAR_CLOSE by the CLV layer "
        "when captured within its existing 120-minute window."
    )


# =============================================================================
# DRY RUN
# =============================================================================

def run_dry() -> None:
    print_rule()
    print(
        "CFB EDGE ENGINE — MARKET CAPTURE SYSTEM TEST"
    )
    print_rule()

    print(
        "No The Odds API calls."
    )

    print(
        "No OddsPapi calls."
    )

    print(
        "No board rebuild."
    )

    print(
        "No Prediction History writes."
    )

    print(
        "No wager-ledger writes."
    )

    print_rule()

    preflight(
        dry_run=True
    )

    print()
    print(
        "DRY RUN PASS"
    )

    print(
        "The market-only capture workflow is present and syntactically runnable."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Take a lightweight immutable near-kickoff market snapshot without "
            "rerunning football data, features, or the neural model."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Verify files, syntax, API-key presence, and the frozen prediction "
            "artifact without making API calls or writing history."
        ),
    )

    parser.add_argument(
        "--update-wager-ledger",
        action="store_true",
        help=(
            "Allow market_history_clv_v1.py to update derived CLV status/points "
            "in the Actual Wager Ledger. Original ticket fields are unchanged."
        ),
    )

    parser.add_argument(
        "--skip-support",
        action="store_true",
        help=(
            "Skip the local What Changed / Prediction Audit / Data Health refresh."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.dry_run:
            run_dry()

        else:
            run_live(
                args
            )

        return 0

    except KeyboardInterrupt:
        print(
            "\nMARKET CAPTURE CANCELLED BY USER",
            flush=True,
        )

        return 130

    except Exception as error:
        print(
            flush=True
        )

        print_rule(
            "!"
        )

        print(
            "MARKET CAPTURE FAILED",
            flush=True,
        )

        print_rule(
            "!"
        )

        print(
            f"ERROR: {error}",
            flush=True,
        )

        print(
            "No later stages were run after the failure.",
            flush=True,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
