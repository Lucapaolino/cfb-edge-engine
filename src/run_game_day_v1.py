#!/usr/bin/env python3
"""
CFB Edge Engine — canonical 2026 game-day controller.

Purpose
-------
One command owns the entire live workflow used by the GUI:

    1. Preflight checks
    2. Refresh 2026 games/results (quota-aware)
    3. Refresh 2026 team stats (quota-aware)
    4. Refresh 2026 player stats (quota-aware)
    5. Refresh static 2026 context when stale (quota-aware)
    6. Build the live 54-feature slate
    7. Run the frozen 2026 neural ensemble
    8. Snapshot the reference market (The Odds API)
    9. Snapshot Hard Rock Bet Florida execution (OddsPapi)
   10. Build the live model/market board
   11. Log immutable pre-kickoff prediction snapshots
   12. Score completed historical predictions
   13. Rebuild the prospective evaluation report

Important design rules
----------------------
* --dry-run performs NO API calls and does not run the live pipeline.
* The controller never retrains a model.
* Hard Rock is the execution sportsbook. Reference books remain informational.
* Prediction snapshots are immutable. Existing batch files are never overwritten.
* A failed critical stage stops the pipeline instead of silently using stale output.
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path

from cfb_edge_paths import (
    BOARD_JSON,
    DB_FILE,
    ENV_FILE,
    HISTORY_DB,
    HISTORY_REPORT,
    MODEL_DIR,
    PREDICTIONS_JSON,
    PROJECT_ROOT,
    PROCESSED_DIR,
    PROSPECTIVE_DIR,
    SRC_DIR,
    STATE_DIR,
)


SCRIPTS = {
    "games": SRC_DIR / "update_live_2026_games_v1.py",
    "team_stats": SRC_DIR / "update_live_2026_team_stats_v1.py",
    "player_stats": SRC_DIR / "update_live_2026_player_stats_v1.py",
    "context": SRC_DIR / "update_live_2026_context_v1.py",
    "features": SRC_DIR / "build_live_features_2026_v1.py",
    "predict": SRC_DIR / "predict_live_2026_nn_v1.py",
    "reference_market": SRC_DIR / "snapshot_execution_market_2026_v1.py",
    "hardrock": SRC_DIR / "snapshot_hardrock_oddspapi_2026_v1.py",
    "board": SRC_DIR / "build_live_model_market_board_v1.py",
    "history_core": SRC_DIR / "prediction_history_v1.py",
    "history_log": SRC_DIR / "log_predictions_v1.py",
    "history_score": SRC_DIR / "score_prediction_history_v1.py",
    "history_report": SRC_DIR / "report_prediction_history_v1.py",
}


# Minimum interval between quota-bearing CFBD refreshes when this controller
# itself was responsible for the previous successful refresh.
REFRESH_INTERVAL_SECONDS = {
    "games": 30 * 60,          # 30 minutes
    "team_stats": 6 * 60 * 60, # 6 hours
    "player_stats": 6 * 60 * 60,
    "context": 7 * 24 * 60 * 60, # 7 days; primarily static season context
}


REQUIRED_ENV_KEYS = (
    "CFBD_API_KEY",
    "THE_ODDS_API_KEY",
    "ODDSPAPI_API_KEY",
)


STAGE_PROGRESS = {
    "SYSTEM CHECK": 4,
    "REFRESHING GAMES": 12,
    "REFRESHING TEAM STATS": 22,
    "REFRESHING PLAYER STATS": 32,
    "REFRESHING CONTEXT": 40,
    "BUILDING LIVE FEATURES": 50,
    "RUNNING FROZEN NEURAL MODEL": 61,
    "REFRESHING REFERENCE MARKET": 72,
    "REFRESHING HARD ROCK": 82,
    "ASSEMBLING LIVE BOARD": 88,
    "LOGGING PREDICTIONS": 93,
    "SCORING PRIOR PREDICTIONS": 96,
    "UPDATING HISTORY REPORT": 98,
    "ANALYSIS COMPLETE": 100,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def print_rule(char: str = "=", width: int = 104) -> None:
    print(char * width, flush=True)


def print_stage(name: str) -> None:
    progress = STAGE_PROGRESS.get(name, 0)
    # Machine-readable line consumed by the PySide GUI.
    print(f"PIPELINE_STAGE|{name}|{progress}", flush=True)
    print_rule()
    print(name, flush=True)
    print_rule()


def load_env_presence() -> dict[str, bool]:
    """Return presence only; never return/print secret values."""
    values: dict[str, str] = {}

    for key in REQUIRED_ENV_KEYS:
        if os.environ.get(key):
            values[key] = os.environ[key]

    if ENV_FILE.exists():
        try:
            from dotenv import dotenv_values  # type: ignore

            parsed = dotenv_values(ENV_FILE)
            for key in REQUIRED_ENV_KEYS:
                value = parsed.get(key)
                if value and key not in values:
                    values[key] = str(value)
        except Exception:
            # Minimal fallback parser. This is deliberately conservative and
            # only used to determine whether a named key exists.
            for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in REQUIRED_ENV_KEYS and value and key not in values:
                    values[key] = value

    return {key: bool(values.get(key)) for key in REQUIRED_ENV_KEYS}


def compile_script(path: Path) -> tuple[bool, str]:
    try:
        py_compile.compile(str(path), doraise=True)
        return True, "PASS"
    except Exception as exc:
        return False, str(exc)


def preflight(dry_run: bool) -> None:
    print_stage("SYSTEM CHECK")

    errors: list[str] = []

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python:       {sys.executable}")
    print(f"Mode:         {'DRY RUN / NO API CALLS' if dry_run else 'LIVE GAME DAY'}")
    print()

    print("REQUIRED FILES")
    print_rule("-")

    required_paths = [DB_FILE, MODEL_DIR, *SCRIPTS.values()]
    for path in required_paths:
        exists = path.exists()
        label = "PASS" if exists else "MISSING"
        try:
            display = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display = path
        print(f"{label:<8} {display}")
        if not exists:
            errors.append(f"Missing required path: {path}")

    print()
    print("PYTHON SYNTAX")
    print_rule("-")
    for name, path in SCRIPTS.items():
        if not path.exists():
            continue
        ok, message = compile_script(path)
        print(f"{'PASS' if ok else 'FAIL':<8} {name:<18} {path.name}")
        if not ok:
            errors.append(f"Syntax check failed for {path.name}: {message}")

    print()
    print("API KEY PRESENCE")
    print_rule("-")
    env_presence = load_env_presence()
    for key in REQUIRED_ENV_KEYS:
        present = env_presence[key]
        print(f"{'PASS' if present else 'MISSING':<8} {key}")
        if not present:
            errors.append(f"Missing required environment variable: {key}")

    print()
    print("MODEL ARTIFACT")
    print_rule("-")
    if MODEL_DIR.exists():
        artifact_files = [p for p in MODEL_DIR.rglob("*") if p.is_file()]
        print(f"Files present: {len(artifact_files)}")
        if not artifact_files:
            errors.append(f"Model artifact directory is empty: {MODEL_DIR}")
    else:
        print("Files present: 0")

    print()
    print("OUTPUT PATHS")
    print_rule("-")
    print(f"Board JSON:   {BOARD_JSON.relative_to(PROJECT_ROOT)}")
    print(f"History DB:   {HISTORY_DB.relative_to(PROJECT_ROOT)}")
    print(f"History report: {HISTORY_REPORT.relative_to(PROJECT_ROOT)}")

    if errors:
        print()
        print("PREFLIGHT FAILED")
        print_rule("-")
        for error in errors:
            print(f"- {error}")
        raise RuntimeError(f"Preflight failed with {len(errors)} problem(s).")

    print()
    print("PREFLIGHT PASS")
    if dry_run:
        print("No sportsbook or CFBD API calls were made.")


def stamp_path(stage_key: str) -> Path:
    return STATE_DIR / f"{stage_key}.json"


def read_stamp_age_seconds(stage_key: str) -> float | None:
    path = stamp_path(stage_key)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stamp = payload.get("completed_utc")
        if not stamp:
            return None
        dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (utc_now() - dt.astimezone(timezone.utc)).total_seconds())
    except Exception:
        return None


def write_stamp(stage_key: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage_key,
        "completed_utc": utc_iso(),
    }
    stamp_path(stage_key).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def should_refresh(stage_key: str, force: bool) -> tuple[bool, str]:
    if force:
        return True, "forced"

    interval = REFRESH_INTERVAL_SECONDS.get(stage_key)
    if interval is None:
        return True, "not gated"

    age = read_stamp_age_seconds(stage_key)
    if age is None:
        return True, "no controller refresh stamp"

    if age >= interval:
        return True, f"last controller refresh {age / 3600:.2f}h ago"

    remaining = interval - age
    return False, (
        f"recent controller refresh {age / 3600:.2f}h ago; "
        f"next automatic refresh in {remaining / 3600:.2f}h"
    )


def run_script(
    stage_name: str,
    script_key: str,
    extra_args: list[str] | None = None,
) -> None:
    script = SCRIPTS[script_key]
    print_stage(stage_name)
    print(f"Script: {script.relative_to(PROJECT_ROOT)}")

    command = [sys.executable, str(script)]
    if extra_args:
        command.extend(extra_args)

    print(f"Started UTC: {utc_iso()}")
    print()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError(f"Could not read output from {script.name}.")

    for line in process.stdout:
        print(line.rstrip(), flush=True)

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{stage_name} failed: {script.name} exited with code {return_code}."
        )

    print()
    print(f"Completed UTC: {utc_iso()}")


def run_quota_aware_stage(
    stage_name: str,
    script_key: str,
    force: bool,
) -> None:
    refresh, reason = should_refresh(script_key, force)
    if not refresh:
        print_stage(stage_name)
        print(f"SKIPPED API REFRESH: {reason}")
        print("Using the already-refreshed database state, then rebuilding downstream outputs.")
        return

    print(f"Refresh decision for {script_key}: {reason}", flush=True)
    run_script(stage_name, script_key)
    write_stamp(script_key)


def verify_outputs() -> None:
    print()
    print("FINAL OUTPUT CHECK")
    print_rule("-")
    required_outputs = (PREDICTIONS_JSON, BOARD_JSON)
    missing: list[Path] = []
    for path in required_outputs:
        status = "PASS" if path.exists() else "MISSING"
        print(f"{status:<8} {path.relative_to(PROJECT_ROOT)}")
        if not path.exists():
            missing.append(path)
    if missing:
        raise RuntimeError(
            "Pipeline completed commands but required outputs are missing: "
            + ", ".join(str(path) for path in missing)
        )


def run_live(args: argparse.Namespace) -> None:
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

    print_rule()
    print("CFB EDGE ENGINE — 2026 CANONICAL GAME-DAY PIPELINE")
    print_rule()
    print(f"Run ID:      {run_id}")
    print(f"Started UTC: {utc_iso()}")
    print("Model:       frozen 2026 neural ensemble; NO TRAINING")
    print("Execution:   Hard Rock Bet Florida via OddsPapi")
    print("Reference:   The Odds API consensus")
    print("History:     immutable pre-kickoff prediction snapshots + automatic scoring")
    print_rule()

    preflight(dry_run=False)

    run_quota_aware_stage("REFRESHING GAMES", "games", args.force_data)
    run_quota_aware_stage("REFRESHING TEAM STATS", "team_stats", args.force_data)
    run_quota_aware_stage("REFRESHING PLAYER STATS", "player_stats", args.force_data)
    run_quota_aware_stage(
        "REFRESHING CONTEXT",
        "context",
        args.force_context or args.force_data,
    )

    run_script("BUILDING LIVE FEATURES", "features")
    run_script("RUNNING FROZEN NEURAL MODEL", "predict")

    # These are intentionally fresh on every real RUN GAME DAY because the
    # whole point of the board is current executable/reference pricing.
    run_script("REFRESHING REFERENCE MARKET", "reference_market")
    run_script("REFRESHING HARD ROCK", "hardrock")
    run_script("ASSEMBLING LIVE BOARD", "board")

    verify_outputs()
    run_script("LOGGING PREDICTIONS", "history_log", ["--run-id", run_id])
    run_script("SCORING PRIOR PREDICTIONS", "history_score")
    run_script("UPDATING HISTORY REPORT", "history_report", ["--policy", "latest"])

    print_stage("ANALYSIS COMPLETE")
    print(f"Run ID:        {run_id}")
    print(f"Completed UTC: {utc_iso()}")
    print(f"Live board:    {BOARD_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Prediction history: {HISTORY_DB.relative_to(PROJECT_ROOT)}")
    print(f"Evaluation report:  {HISTORY_REPORT.relative_to(PROJECT_ROOT)}")
    print()
    print("IMPORTANT")
    print_rule("-")
    print("Strength is still prospectively unvalidated and is not a cover probability.")
    print("No frozen wagering rule exists yet, so history records predictions without inventing bets.")
    print("Hard Rock execution is never substituted with a reference sportsbook.")


def run_dry() -> None:
    print_rule()
    print("CFB EDGE ENGINE — SYSTEM TEST")
    print_rule()
    print("This mode performs no CFBD, The Odds API, or OddsPapi requests.")
    print("It verifies the canonical pipeline is present and syntactically runnable.")
    print_rule()

    preflight(dry_run=True)

    print_stage("ANALYSIS COMPLETE")
    print("SYSTEM TEST COMPLETE")
    print("No API calls were made and no prediction-history rows were written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical 2026 CFB Edge game-day pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check the full pipeline without making API calls.",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Ignore CFBD refresh intervals and force games/team/player/context refreshes.",
    )
    parser.add_argument(
        "--force-context",
        action="store_true",
        help="Force the normally weekly static-context refresh.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.dry_run:
            run_dry()
        else:
            run_live(args)
        return 0
    except KeyboardInterrupt:
        print("\nPIPELINE CANCELLED BY USER", flush=True)
        return 130
    except Exception as exc:
        print("", flush=True)
        print_rule("!")
        print("PIPELINE FAILED", flush=True)
        print_rule("!")
        print(f"ERROR: {exc}", flush=True)
        print("No later stages were run after the failure.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())