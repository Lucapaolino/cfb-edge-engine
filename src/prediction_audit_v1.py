#!/usr/bin/env python3
"""CFB Edge Engine — prediction provenance / pre-kickoff audit.

Purpose
-------
Create a read-only audit of immutable prediction snapshots so the GUI can show a
compact verification badge such as:

    VERIFIED PRE-KICKOFF SNAPSHOT

The audit does NOT modify prediction history, scores, results, wagers, models,
or market data. It verifies what can be established from the stored evidence:

- snapshot identity exists and is internally coherent;
- prediction capture time is strictly before kickoff;
- model favorite belongs to the stored game;
- probability / feature fields are sane when present;
- duplicate game snapshots within one run are flagged;
- SQLite protection triggers for prediction_snapshots are detected;
- the immutable JSON batch for a run is located and, when possible, the
  snapshot_id is corroborated inside it.

This is an integrity/provenance tool, not a model-performance tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_DB = PROJECT_ROOT / "data" / "prospective" / "prediction_history.sqlite"
BATCH_DIR = PROJECT_ROOT / "data" / "prospective" / "prediction_snapshots"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_JSON = OUTPUT_DIR / "prediction_audit_2026_v1.json"
OUTPUT_CSV = OUTPUT_DIR / "prediction_audit_2026_v1.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "prediction_audit_summary_2026_v1.json"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_iso(value: datetime | None = None) -> str:
    dt = value or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_team(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def same_team(left: Any, right: Any) -> bool:
    left_norm = normalize_team(left)
    right_norm = normalize_team(right)
    return bool(left_norm and right_norm and left_norm == right_norm)


def boolish(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def print_rule(char: str = "=", width: int = 104) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Read-only history access
# ---------------------------------------------------------------------------


def connect_history_read_only(path: Path = HISTORY_DB) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction-history database not found: {path}\n"
            "Run GAME DAY successfully before running the audit."
        )
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        is not None
    )


def validate_schema(connection: sqlite3.Connection) -> set[str]:
    if not table_exists(connection, "prediction_snapshots"):
        raise RuntimeError("prediction_snapshots table does not exist.")

    columns = table_columns(connection, "prediction_snapshots")
    required = {
        "snapshot_id",
        "run_id",
        "captured_utc",
        "game_key",
        "game_id",
        "season",
        "week",
        "kickoff_utc",
        "away_team",
        "home_team",
        "model_favorite",
        "model_spread_display",
        "model_home_margin",
        "model_home_win_probability",
        "strength_label",
    }
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(
            "prediction_snapshots schema is missing required fields: "
            + ", ".join(missing)
        )
    return columns


def detect_snapshot_protection(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='trigger' AND tbl_name='prediction_snapshots' "
        "ORDER BY name"
    ).fetchall()

    triggers = []
    protects_update = False
    protects_delete = False
    for row in rows:
        name = str(row[0])
        sql = str(row[1] or "")
        upper = sql.upper()
        trigger = {
            "name": name,
            "blocks_update": "UPDATE" in upper and "RAISE" in upper,
            "blocks_delete": "DELETE" in upper and "RAISE" in upper,
        }
        protects_update = protects_update or trigger["blocks_update"]
        protects_delete = protects_delete or trigger["blocks_delete"]
        triggers.append(trigger)

    return {
        "protected": bool(protects_update and protects_delete),
        "protects_update": protects_update,
        "protects_delete": protects_delete,
        "triggers": triggers,
    }


# ---------------------------------------------------------------------------
# Batch corroboration
# ---------------------------------------------------------------------------


def candidate_batch_paths(run_id: str) -> list[Path]:
    if not run_id or not BATCH_DIR.exists():
        return []
    exact = BATCH_DIR / f"cfb_2026_predictions_{run_id}.json"
    paths: list[Path] = []
    if exact.exists():
        paths.append(exact)
    for path in BATCH_DIR.glob(f"*{run_id}*.json"):
        if path not in paths:
            paths.append(path)
    return sorted(paths)


def recursively_contains_snapshot_id(value: Any, snapshot_id: str) -> bool:
    if isinstance(value, dict):
        if str(value.get("snapshot_id") or "") == snapshot_id:
            return True
        return any(recursively_contains_snapshot_id(v, snapshot_id) for v in value.values())
    if isinstance(value, list):
        return any(recursively_contains_snapshot_id(v, snapshot_id) for v in value)
    return False


def batch_evidence(run_id: str, snapshot_id: str, cache: dict[Path, Any]) -> dict[str, Any]:
    paths = candidate_batch_paths(run_id)
    if not paths:
        return {
            "batch_file_found": False,
            "batch_file": None,
            "snapshot_id_corroborated": False,
            "batch_parse_error": None,
        }

    # A run should normally have one canonical batch. If several match, the first
    # exact/sorted candidate is used and the audit remains conservative.
    path = paths[0]
    try:
        if path not in cache:
            with path.open("r", encoding="utf-8") as handle:
                cache[path] = json.load(handle)
        payload = cache[path]
        corroborated = recursively_contains_snapshot_id(payload, snapshot_id)
        return {
            "batch_file_found": True,
            "batch_file": str(path.relative_to(PROJECT_ROOT)),
            "snapshot_id_corroborated": corroborated,
            "batch_parse_error": None,
        }
    except Exception as error:
        return {
            "batch_file_found": True,
            "batch_file": str(path.relative_to(PROJECT_ROOT)),
            "snapshot_id_corroborated": False,
            "batch_parse_error": str(error),
        }


# ---------------------------------------------------------------------------
# Snapshot audit
# ---------------------------------------------------------------------------


def identity_status(snapshot: dict[str, Any]) -> tuple[bool, str]:
    game_id = str(snapshot.get("game_id") or "").strip()
    if game_id:
        return True, "EXACT_GAME_ID"

    season = safe_int(snapshot.get("season"))
    week = safe_int(snapshot.get("week"))
    away = str(snapshot.get("away_team") or "").strip()
    home = str(snapshot.get("home_team") or "").strip()
    if season is not None and week is not None and away and home:
        return True, "SEASON_WEEK_TEAMS"

    return False, "INSUFFICIENT_GAME_IDENTITY"


def audit_snapshot(
    snapshot: dict[str, Any],
    *,
    duplicate_run_game: bool,
    protection: dict[str, Any],
    batch_cache: dict[Path, Any],
) -> dict[str, Any]:
    flags: list[str] = []

    snapshot_id = str(snapshot.get("snapshot_id") or "").strip()
    run_id = str(snapshot.get("run_id") or "").strip()
    game_key = str(snapshot.get("game_key") or "").strip()
    away = str(snapshot.get("away_team") or "").strip()
    home = str(snapshot.get("home_team") or "").strip()

    if not snapshot_id:
        flags.append("MISSING_SNAPSHOT_ID")
    if not run_id:
        flags.append("MISSING_RUN_ID")
    if not game_key:
        flags.append("MISSING_GAME_KEY")
    if not away or not home:
        flags.append("MISSING_TEAMS")
    if away and home and same_team(away, home):
        flags.append("SAME_HOME_AND_AWAY_TEAM")
    if duplicate_run_game:
        flags.append("DUPLICATE_GAME_WITHIN_RUN")

    identity_ok, identity_method = identity_status(snapshot)
    if not identity_ok:
        flags.append("INSUFFICIENT_GAME_IDENTITY")

    capture_dt = parse_datetime(snapshot.get("captured_utc"))
    kickoff_dt = parse_datetime(snapshot.get("kickoff_utc"))
    captured_before_kickoff = None
    hours_before_kickoff = None
    if capture_dt is None:
        flags.append("INVALID_CAPTURE_TIME")
    if kickoff_dt is None:
        flags.append("INVALID_KICKOFF_TIME")
    if capture_dt is not None and kickoff_dt is not None:
        seconds = (kickoff_dt - capture_dt).total_seconds()
        hours_before_kickoff = round(seconds / 3600.0, 3)
        captured_before_kickoff = seconds > 0
        if not captured_before_kickoff:
            flags.append("NOT_PREKICKOFF")

    favorite = str(snapshot.get("model_favorite") or "").strip()
    model_side_valid = same_team(favorite, away) or same_team(favorite, home)
    if not model_side_valid:
        flags.append("MODEL_FAVORITE_NOT_IN_GAME")

    model_margin = safe_float(snapshot.get("model_home_margin"))
    if model_margin is None:
        flags.append("INVALID_MODEL_MARGIN")

    home_p = safe_float(snapshot.get("model_home_win_probability"))
    probability_valid = home_p is not None and 0.0 <= home_p <= 1.0
    if not probability_valid:
        flags.append("INVALID_HOME_WIN_PROBABILITY")

    feature_coverage = safe_int(snapshot.get("nn_feature_coverage"))
    if feature_coverage is not None and not (0 <= feature_coverage <= 54):
        flags.append("INVALID_FEATURE_COVERAGE")

    hardrock_available = boolish(snapshot.get("hardrock_available"))
    hr_spread = safe_float(snapshot.get("hardrock_spread_for_model_favorite"))
    hr_price = safe_float(snapshot.get("hardrock_price_for_model_favorite"))
    if hardrock_available is True and hr_spread is None:
        flags.append("HARD_ROCK_MARKED_AVAILABLE_WITHOUT_SPREAD")
    if hardrock_available is True and hr_price is None:
        flags.append("HARD_ROCK_MARKED_AVAILABLE_WITHOUT_PRICE")

    batch = batch_evidence(run_id, snapshot_id, batch_cache) if run_id and snapshot_id else {
        "batch_file_found": False,
        "batch_file": None,
        "snapshot_id_corroborated": False,
        "batch_parse_error": None,
    }
    if batch["batch_parse_error"]:
        flags.append("BATCH_PARSE_ERROR")
    elif batch["batch_file_found"] and not batch["snapshot_id_corroborated"]:
        # This is evidence-quality information, not necessarily proof of a bad DB
        # row, because earlier batch formats may not have stored snapshot_id.
        flags.append("BATCH_ID_NOT_CORROBORATED")

    core_verified = all(
        (
            bool(snapshot_id),
            bool(run_id),
            bool(game_key),
            identity_ok,
            captured_before_kickoff is True,
            model_side_valid,
            model_margin is not None,
            probability_valid,
            not duplicate_run_game,
        )
    )

    if core_verified and batch["snapshot_id_corroborated"] and protection.get("protected"):
        verification_status = "VERIFIED_PREKICKOFF_WITH_BATCH"
    elif core_verified and protection.get("protected"):
        verification_status = "VERIFIED_PREKICKOFF"
    elif core_verified:
        verification_status = "PREKICKOFF_VALID_UNPROTECTED_DB"
    else:
        verification_status = "AUDIT_FAILED"

    return {
        "audit_schema_version": SCHEMA_VERSION,
        "verification_status": verification_status,
        "verified_pre_kickoff": core_verified,
        "snapshot_id": snapshot.get("snapshot_id"),
        "run_id": snapshot.get("run_id"),
        "game_key": snapshot.get("game_key"),
        "game_id": snapshot.get("game_id"),
        "season": safe_int(snapshot.get("season")),
        "week": safe_int(snapshot.get("week")),
        "away_team": snapshot.get("away_team"),
        "home_team": snapshot.get("home_team"),
        "captured_utc": snapshot.get("captured_utc"),
        "kickoff_utc": snapshot.get("kickoff_utc"),
        "captured_before_kickoff": captured_before_kickoff,
        "hours_before_kickoff": hours_before_kickoff,
        "identity_ok": identity_ok,
        "identity_method": identity_method,
        "model_favorite": snapshot.get("model_favorite"),
        "model_side_valid": model_side_valid,
        "model_spread_display": snapshot.get("model_spread_display"),
        "model_home_margin": model_margin,
        "model_home_win_probability": home_p,
        "strength_label": str(snapshot.get("strength_label") or "N/A").upper(),
        "nn_input_status": snapshot.get("nn_input_status"),
        "nn_feature_coverage": feature_coverage,
        "hardrock_available": hardrock_available,
        "hardrock_display": snapshot.get("hardrock_display"),
        "batch_file_found": batch["batch_file_found"],
        "batch_file": batch["batch_file"],
        "batch_snapshot_id_corroborated": batch["snapshot_id_corroborated"],
        "db_immutability_protected": bool(protection.get("protected")),
        "audit_flags": flags,
    }


def load_snapshots(connection: sqlite3.Connection, season: int | None) -> list[dict[str, Any]]:
    if season is None:
        rows = connection.execute(
            "SELECT * FROM prediction_snapshots ORDER BY captured_utc, snapshot_id"
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM prediction_snapshots WHERE season=? "
            "ORDER BY captured_utc, snapshot_id",
            (season,),
        ).fetchall()
    return [dict(row) for row in rows]


def audit_rows(
    snapshots: list[dict[str, Any]],
    protection: dict[str, Any],
) -> list[dict[str, Any]]:
    counts = Counter(
        (
            str(row.get("run_id") or ""),
            str(row.get("game_key") or ""),
        )
        for row in snapshots
        if row.get("run_id") not in (None, "") and row.get("game_key") not in (None, "")
    )
    batch_cache: dict[Path, Any] = {}
    output = []
    for snapshot in snapshots:
        key = (
            str(snapshot.get("run_id") or ""),
            str(snapshot.get("game_key") or ""),
        )
        output.append(
            audit_snapshot(
                snapshot,
                duplicate_run_game=counts.get(key, 0) > 1,
                protection=protection,
                batch_cache=batch_cache,
            )
        )
    return output


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "verification_status": row.get("verification_status"),
        "verified_pre_kickoff": int(bool(row.get("verified_pre_kickoff"))),
        "snapshot_id": row.get("snapshot_id"),
        "run_id": row.get("run_id"),
        "game_key": row.get("game_key"),
        "game_id": row.get("game_id"),
        "season": row.get("season"),
        "week": row.get("week"),
        "away_team": row.get("away_team"),
        "home_team": row.get("home_team"),
        "captured_utc": row.get("captured_utc"),
        "kickoff_utc": row.get("kickoff_utc"),
        "hours_before_kickoff": row.get("hours_before_kickoff"),
        "identity_method": row.get("identity_method"),
        "model_favorite": row.get("model_favorite"),
        "model_spread_display": row.get("model_spread_display"),
        "strength_label": row.get("strength_label"),
        "nn_input_status": row.get("nn_input_status"),
        "nn_feature_coverage": row.get("nn_feature_coverage"),
        "hardrock_display": row.get("hardrock_display"),
        "batch_file_found": int(bool(row.get("batch_file_found"))),
        "batch_snapshot_id_corroborated": int(bool(row.get("batch_snapshot_id_corroborated"))),
        "db_immutability_protected": int(bool(row.get("db_immutability_protected"))),
        "audit_flags": "|".join(row.get("audit_flags") or []),
    }


def build_summary(rows: list[dict[str, Any]], protection: dict[str, Any], season: int | None) -> dict[str, Any]:
    status_counts = Counter(str(row.get("verification_status") or "UNKNOWN") for row in rows)
    flag_counts = Counter(flag for row in rows for flag in (row.get("audit_flags") or []))
    verified = sum(1 for row in rows if row.get("verified_pre_kickoff"))
    with_batch = sum(1 for row in rows if row.get("batch_snapshot_id_corroborated"))
    return {
        "audit_schema_version": SCHEMA_VERSION,
        "generated_utc": utc_iso(),
        "season": season,
        "history_db": str(HISTORY_DB.relative_to(PROJECT_ROOT)),
        "snapshot_count": len(rows),
        "verified_pre_kickoff_count": verified,
        "verified_pre_kickoff_rate": (verified / len(rows)) if rows else None,
        "batch_corroborated_count": with_batch,
        "verification_status_counts": dict(sorted(status_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "database_protection": protection,
        "interpretation": {
            "VERIFIED_PREKICKOFF_WITH_BATCH": (
                "Core snapshot checks passed, database update/delete protection was detected, "
                "and the snapshot_id was corroborated in an immutable run batch."
            ),
            "VERIFIED_PREKICKOFF": (
                "Core snapshot checks passed and database update/delete protection was detected. "
                "Batch corroboration was unavailable or inconclusive."
            ),
            "PREKICKOFF_VALID_UNPROTECTED_DB": (
                "Core snapshot checks passed, but update/delete protection triggers were not detected."
            ),
            "AUDIT_FAILED": "One or more core integrity checks failed; inspect audit_flags.",
        },
    }


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary,
        "records": rows,
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
        handle.write("\n")

    flat_rows = [csv_row(row) for row in rows]
    fieldnames = list(flat_rows[0].keys()) if flat_rows else [
        "verification_status",
        "verified_pre_kickoff",
        "snapshot_id",
        "run_id",
        "game_key",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CFB Edge prediction snapshots for pre-kickoff provenance and integrity."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season to audit (default: 2026).",
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Audit every season in prediction history instead of only 2026.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    season = None if args.all_seasons else args.season

    print_rule()
    print("CFB EDGE ENGINE — PREDICTION AUDIT V1")
    print_rule()
    print(f"History DB: {HISTORY_DB}")
    print(f"Season:     {'ALL' if season is None else season}")
    print("Mode:       READ ONLY")
    print("Purpose:    verify prediction identity, timing, provenance, and immutability evidence")
    print_rule()

    connection = connect_history_read_only()
    try:
        validate_schema(connection)
        protection = detect_snapshot_protection(connection)
        snapshots = load_snapshots(connection, season)
    finally:
        connection.close()

    rows = audit_rows(snapshots, protection)
    summary = build_summary(rows, protection, season)
    write_outputs(rows, summary)

    print()
    print("AUDIT SUMMARY")
    print_rule("-")
    print(f"Snapshots checked:             {summary['snapshot_count']}")
    print(f"Verified pre-kickoff:          {summary['verified_pre_kickoff_count']}")
    print(f"Batch-corroborated snapshots:  {summary['batch_corroborated_count']}")
    print(
        "SQLite immutability protection: "
        + ("PASS" if protection.get("protected") else "NOT VERIFIED")
    )
    if summary["verification_status_counts"]:
        print("Statuses:")
        for status, count in summary["verification_status_counts"].items():
            print(f"  {status:<38} {count}")
    if summary["flag_counts"]:
        print("Flags:")
        for flag, count in summary["flag_counts"].items():
            print(f"  {flag:<38} {count}")
    else:
        print("Flags:                          NONE")

    print()
    print("FILES WRITTEN")
    print_rule("-")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print(OUTPUT_SUMMARY)
    print()
    print("No prediction, score, result, wager, model, or market record was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
