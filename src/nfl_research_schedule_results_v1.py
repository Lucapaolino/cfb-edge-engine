#!/usr/bin/env python3
"""Capture immutable NFL schedule/result source snapshots.

This script uses nflreadpy/nflverse only when invoked with ``--capture``.
It stores a source snapshot and normalized game rows in the separate NFL
research database.  It does not modify CFB records or create NFL forecasts.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
import uuid

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from nfl_research_paths_v1 import (
    NFL_RAW_SCHEDULE_DIR,
    NFL_RESEARCH_DB,
    PROJECT_ROOT,
)
from nfl_research_store_v1 import (
    canonical_json,
    connect_writable,
    insert_schedule_snapshot,
    sha256_json,
)


SOURCE_NAME = "nflverse_nflreadpy"
DEFAULT_SEASON = 2026


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_snapshot_id(captured_at_utc: str) -> str:
    stamp = captured_at_utc.replace("+00:00", "Z").replace("-", "").replace(":", "")
    return f"nfl_schedule_{stamp}_{uuid.uuid4().hex[:10]}"


def jsonable(value: Any) -> Any:
    """Convert dataframe scalar values to stable JSON-compatible values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return jsonable(item_method())
        except (TypeError, ValueError):
            pass

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return numeric_value if math.isfinite(numeric_value) else None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_flag(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return None


def normalize_schedule_rows(
    snapshot_id: str,
    raw_rows: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep a narrow stable schema while retaining every source row as JSON."""
    normalized: list[dict[str, Any]] = []
    skipped_without_game_id = 0

    for raw_row in raw_rows:
        row = jsonable(raw_row)
        if not isinstance(row, Mapping):
            skipped_without_game_id += 1
            continue

        game_id = clean_text(row.get("game_id"))
        if not game_id:
            skipped_without_game_id += 1
            continue

        normalized.append(
            {
                "snapshot_id": snapshot_id,
                "game_id": game_id,
                "season": as_integer(row.get("season")),
                "week": as_integer(row.get("week")),
                "game_type": clean_text(row.get("game_type")),
                "gameday": clean_text(row.get("gameday")),
                "weekday": clean_text(row.get("weekday")),
                "gametime": clean_text(row.get("gametime")),
                "away_team": clean_text(row.get("away_team")),
                "away_score": as_float(row.get("away_score")),
                "home_team": clean_text(row.get("home_team")),
                "home_score": as_float(row.get("home_score")),
                "result": as_float(row.get("result")),
                "total": as_float(row.get("total")),
                "overtime": as_flag(row.get("overtime")),
                "location": clean_text(row.get("location")),
                "roof": clean_text(row.get("roof")),
                "surface": clean_text(row.get("surface")),
                "temperature": as_float(row.get("temp")),
                "wind": as_float(row.get("wind")),
                "raw_game_json": canonical_json(row),
            }
        )

    return normalized, skipped_without_game_id


def write_raw_snapshot(snapshot_id: str, envelope: Mapping[str, Any]) -> Path:
    NFL_RAW_SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    path = NFL_RAW_SCHEDULE_DIR / f"{snapshot_id}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(envelope, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def print_plan(args: argparse.Namespace) -> None:
    mode = "CAPTURE" if args.capture else "DRY RUN / NO PROVIDER CALLS OR WRITES"
    print("=" * 100)
    print("NFL RESEARCH COLLECTION — SCHEDULES + RESULTS V1")
    print("=" * 100)
    print(f"Mode:          {mode}")
    print(f"Requested year:{args.season}")
    print(f"Source:        {SOURCE_NAME}")
    print(f"Database:      {NFL_RESEARCH_DB.relative_to(PROJECT_ROOT)}")
    print("Scope:         source schedule/result snapshots only")
    print("Guardrails:    no CFB reads/writes; no model; no picks; no wager records")
    print()


def run_dry(args: argparse.Namespace) -> int:
    print_plan(args)
    print("DEPENDENCY READINESS")
    print("-" * 100)
    try:
        version = importlib.metadata.version("nflreadpy")
        print(f"PASS     nflreadpy {version}")
    except importlib.metadata.PackageNotFoundError:
        print("MISSING  nflreadpy (install with: pip install -r requirements-nfl-research.txt)")
    print()
    print("SYSTEM TEST COMPLETE")
    print("No nflverse request was made. No NFL database or raw files were written.")
    return 0


def load_schedule_rows(season: int) -> tuple[list[Mapping[str, Any]], str | None]:
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise RuntimeError(
            "nflreadpy is required for schedule capture. Install requirements-nfl-research.txt first."
        ) from exc

    try:
        source_version = importlib.metadata.version("nflreadpy")
    except importlib.metadata.PackageNotFoundError:
        source_version = None

    dataframe = nfl.load_schedules(season)
    to_dicts = getattr(dataframe, "to_dicts", None)
    if not callable(to_dicts):
        raise RuntimeError("nflreadpy returned an unsupported schedule dataframe type.")

    rows = to_dicts()
    if not isinstance(rows, list):
        raise RuntimeError("nflreadpy schedule output was not a list of row objects.")
    filtered = [row for row in rows if isinstance(row, Mapping)]
    return filtered, source_version


def run_capture(args: argparse.Namespace) -> int:
    print_plan(args)
    source_requested_at_utc = utc_iso()
    raw_rows, source_version = load_schedule_rows(args.season)
    captured_at_utc = utc_iso()
    snapshot_id = make_snapshot_id(captured_at_utc)
    json_rows = [jsonable(row) for row in raw_rows]
    normalized_rows, skipped_without_game_id = normalize_schedule_rows(snapshot_id, json_rows)

    raw_envelope = {
        "schema": "nfl_research_schedule_results_v1",
        "snapshot_id": snapshot_id,
        "captured_at_utc": captured_at_utc,
        "source": SOURCE_NAME,
        "source_version": source_version,
        "requested_at_utc": source_requested_at_utc,
        "requested_season": args.season,
        "rows": json_rows,
    }
    raw_path = write_raw_snapshot(snapshot_id, raw_envelope)
    response_json = canonical_json(json_rows)
    snapshot = {
        "snapshot_id": snapshot_id,
        "captured_at_utc": captured_at_utc,
        "source": SOURCE_NAME,
        "source_version": source_version,
        "requested_season": args.season,
        "row_count": len(json_rows),
        "response_sha256": sha256_json(json_rows),
        "raw_file": str(raw_path.relative_to(PROJECT_ROOT)),
        "response_json": response_json,
        "created_at_utc": utc_iso(),
    }

    connection = connect_writable(NFL_RESEARCH_DB)
    try:
        insert_schedule_snapshot(connection, snapshot, normalized_rows)
    finally:
        connection.close()

    print("CAPTURE COMPLETE")
    print("-" * 100)
    print(f"Snapshot ID:   {snapshot_id}")
    print(f"Source rows:   {len(json_rows)}")
    print(f"Stored games:  {len(normalized_rows)}")
    print(f"Skipped rows:  {skipped_without_game_id} (missing game_id)")
    print(f"Raw file:      {raw_path.relative_to(PROJECT_ROOT)}")
    print(f"Database:      {NFL_RESEARCH_DB.relative_to(PROJECT_ROOT)}")
    print()
    print("No CFB file was read or changed. No NFL prediction, signal, or wager was created.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture immutable NFL schedule/result research snapshots.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--capture",
        action="store_true",
        help="Download one source schedule/result snapshot and append it to the NFL ledger.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Show readiness only; this is also the default behavior.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help=f"NFL season to collect (default: {DEFAULT_SEASON}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_capture(args) if args.capture else run_dry(args)
    except KeyboardInterrupt:
        print("\nNFL SCHEDULE CAPTURE CANCELLED", flush=True)
        return 130
    except Exception as exc:
        print("\nNFL SCHEDULE CAPTURE FAILED", flush=True)
        print(f"ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
