#!/usr/bin/env python3
"""Freeze the current pre-kickoff model board into immutable prediction history."""

from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from prediction_history_v1 import (
    BATCH_DIR,
    BOARD_JSON,
    HISTORY_JSONL,
    LEGACY_JSONL,
    PROJECT_ROOT,
    SCHEMA_VERSION,
    actionable_record,
    canonical_json,
    connect_history,
    extract_snapshot_fields,
    insert_prediction_run,
    normalize_board_records,
    sha256_json,
    utc_iso,
    utc_now,
)


def print_rule(char: str = "=", width: int = 104) -> None:
    print(char * width)


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def migrate_legacy_jsonl(connection: sqlite3.Connection) -> int:
    """One-time/idempotent import of the controller's earlier JSONL ledger."""
    if not LEGACY_JSONL.exists():
        return 0

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_line in LEGACY_JSONL.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        run_id = str(row.get("ledger_run_id") or "").strip()
        if run_id:
            groups[run_id].append(row)

    imported = 0
    for run_id, legacy_rows in groups.items():
        exists = connection.execute(
            "SELECT 1 FROM prediction_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if exists:
            continue

        capture = str(
            legacy_rows[0].get("ledger_capture_utc")
            or legacy_rows[0].get("captured_utc")
            or utc_iso()
        )
        rows = [
            extract_snapshot_fields(row, run_id=run_id, captured_utc=capture)
            for row in legacy_rows
        ]
        batch_hash = sha256_json({"legacy_run_id": run_id, "records": legacy_rows})
        try:
            insert_prediction_run(
                connection,
                run_id=run_id,
                captured_utc=capture,
                source_board="legacy:data/prospective/cfb_2026_signal_ledger.jsonl",
                batch_file="legacy-jsonl",
                batch_sha256=batch_hash,
                rows=rows,
            )
            connection.commit()
            imported += len(rows)
        except sqlite3.IntegrityError:
            connection.rollback()

    return imported


def log_current_board(board_path: Path, run_id: str) -> tuple[int, Path]:
    if not board_path.exists():
        raise FileNotFoundError(f"Board file not found: {board_path}")

    with board_path.open("r", encoding="utf-8") as handle:
        board = json.load(handle)

    board_records = normalize_board_records(board)
    pregame_records = [record for record in board_records if actionable_record(record)]
    captured_utc = utc_iso()
    snapshot_rows = [
        extract_snapshot_fields(record, run_id=run_id, captured_utc=captured_utc)
        for record in pregame_records
    ]

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_JSONL.parent.mkdir(parents=True, exist_ok=True)

    batch_payload = {
        "prediction_history_schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "captured_utc": captured_utc,
        "source_board": relative_or_absolute(board_path),
        "board_record_count": len(board_records),
        "pregame_prediction_count": len(snapshot_rows),
        "records": snapshot_rows,
    }
    batch_hash = sha256_json(batch_payload)
    batch_payload["batch_sha256"] = batch_hash

    batch_path = BATCH_DIR / f"cfb_2026_predictions_{run_id}.json"
    if batch_path.exists():
        raise RuntimeError(f"Immutable prediction batch already exists: {batch_path}")

    with batch_path.open("x", encoding="utf-8") as handle:
        json.dump(batch_payload, handle, indent=2, default=str)
        handle.write("\n")

    connection = connect_history()
    try:
        migrated = migrate_legacy_jsonl(connection)
        if migrated:
            print(f"Imported legacy prediction rows: {migrated}")

        existing = connection.execute(
            "SELECT 1 FROM prediction_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if existing:
            raise RuntimeError(f"Prediction run already exists in history DB: {run_id}")

        insert_prediction_run(
            connection,
            run_id=run_id,
            captured_utc=captured_utc,
            source_board=relative_or_absolute(board_path),
            batch_file=relative_or_absolute(batch_path),
            batch_sha256=batch_hash,
            rows=snapshot_rows,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        # The batch file is evidence too. We do not silently delete it after a
        # database failure; the error tells us recovery is needed.
        raise
    finally:
        connection.close()

    with HISTORY_JSONL.open("a", encoding="utf-8") as handle:
        for row in snapshot_rows:
            handle.write(canonical_json(row))
            handle.write("\n")

    return len(snapshot_rows), batch_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append the current live board to immutable prediction history."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Pipeline run ID. If omitted, a unique manual run ID is generated.",
    )
    parser.add_argument(
        "--board",
        default=str(BOARD_JSON),
        help="Path to the live board JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or (
        utc_now().strftime("%Y%m%dT%H%M%SZ") + "_manual_" + uuid.uuid4().hex[:8]
    )
    board_path = Path(args.board).expanduser().resolve()

    print_rule()
    print("PREDICTION HISTORY LOGGER — V1")
    print_rule()
    print(f"Run ID:       {run_id}")
    print(f"Board:        {relative_or_absolute(board_path)}")
    print("Storage:      immutable SQLite + JSON batch + append-only JSONL")
    print("Population:   actionable pre-kickoff games only")
    print_rule()

    count, batch_path = log_current_board(board_path, run_id)

    print()
    print("PREDICTION SNAPSHOT FROZEN")
    print_rule("-")
    print(f"Predictions logged: {count}")
    print(f"Batch file:         {relative_or_absolute(batch_path)}")
    print("History DB:         data/prospective/prediction_history.sqlite")
    print("Append log:         data/prospective/prediction_history.jsonl")
    print()
    print("Existing prediction rows are protected from UPDATE and DELETE in SQLite.")
    print("Repeated runs are preserved as separate time-stamped snapshots of the same game.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
