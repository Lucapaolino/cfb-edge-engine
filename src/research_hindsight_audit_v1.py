#!/usr/bin/env python3
"""CFB Edge Engine — Research Hindsight Audit V1.

Read-only audit for the Three Minute Thesis research study:
"Beyond Hindsight: Building an Auditable Neural Decision-Support System for
Real-Time Forecasting."

The goal is not to prove that the model is profitable or ready for automated
decisions. The goal is to make a narrow, testable claim:

    Forecast evidence was stored before the outcome was known, and the later
    evaluation can be inspected without silently replacing that evidence.

This script:
- makes no API calls;
- opens the prediction-history database in SQLite read-only mode;
- does not retrain or modify the frozen neural model;
- does not modify predictions, scores, results, market data, or wagers;
- verifies stored record hashes against their stored JSON payloads;
- checks that prediction snapshots were captured before kickoff;
- checks that the database's normal immutability triggers are present;
- selects one strictly pre-kickoff snapshot per game using a documented policy;
- reports descriptive outcome metrics after results are attached later; and
- writes derived, non-secret research summaries only.

Run
---
python src/research_hindsight_audit_v1.py --policy latest --season 2026

Use --dry-run to verify inputs and print the audit without writing the derived
JSON/CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfb_edge_paths import HISTORY_DB, MODEL_DIR, PROCESSED_DIR


OUTPUT_JSON = PROCESSED_DIR / "research_hindsight_audit_v1.json"
OUTPUT_CSV = PROCESSED_DIR / "research_hindsight_audit_v1.csv"

REQUIRED_TABLES = {
    "prediction_runs",
    "prediction_snapshots",
    "prediction_scores",
    "game_results",
}

REQUIRED_IMMUTABILITY_TRIGGERS = {
    "prediction_snapshots_no_update",
    "prediction_snapshots_no_delete",
    "prediction_runs_no_update",
    "prediction_runs_no_delete",
}


def rule(char: str = "=", width: int = 108) -> None:
    print(char * width)


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    return int(number) if number is not None else None


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fmt(value: float | None, decimals: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{decimals}f}"


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def connect_read_only() -> sqlite3.Connection:
    if not HISTORY_DB.exists():
        raise FileNotFoundError(
            f"Prediction-history database not found: {HISTORY_DB}"
        )
    connection = sqlite3.connect(
        f"file:{HISTORY_DB}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def existing_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()
    }


def load_audit_data(season: int) -> dict[str, Any]:
    connection = connect_read_only()
    try:
        tables = existing_tables(connection)
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            raise RuntimeError(
                "Prediction-history database is missing required table(s): "
                + ", ".join(missing)
            )

        snapshots = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    p.*,
                    s.scored_utc,
                    s.actual_home_margin,
                    s.margin_error,
                    s.absolute_margin_error,
                    s.squared_margin_error,
                    s.winner_correct,
                    s.brier_score,
                    s.hardrock_implied_home_margin,
                    s.hardrock_absolute_margin_error,
                    s.model_side_ats_result
                FROM prediction_snapshots AS p
                LEFT JOIN prediction_scores AS s
                    ON s.snapshot_id = p.snapshot_id
                WHERE p.season = ?
                ORDER BY p.game_key, p.captured_utc, p.snapshot_id
                """,
                (season,),
            ).fetchall()
        ]

        runs = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    r.run_id,
                    r.captured_utc,
                    r.record_count,
                    r.batch_file,
                    r.batch_sha256,
                    COUNT(p.snapshot_id) AS stored_snapshot_count
                FROM prediction_runs AS r
                LEFT JOIN prediction_snapshots AS p
                    ON p.run_id = r.run_id
                   AND p.season = ?
                GROUP BY
                    r.run_id,
                    r.captured_utc,
                    r.record_count,
                    r.batch_file,
                    r.batch_sha256
                HAVING COUNT(p.snapshot_id) > 0
                ORDER BY r.captured_utc, r.run_id
                """,
                (season,),
            ).fetchall()
        ]

        triggers = {
            str(row["name"]): str(row["sql"] or "")
            for row in connection.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type = 'trigger'
                ORDER BY name
                """
            ).fetchall()
        }
    finally:
        connection.close()

    return {
        "snapshots": snapshots,
        "runs": runs,
        "triggers": triggers,
    }


def audit_timing(rows: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts = {
        "total": len(rows),
        "pre_kickoff": 0,
        "at_or_after_kickoff": 0,
        "missing_or_invalid_time": 0,
    }
    valid: list[dict[str, Any]] = []

    for row in rows:
        captured = parse_datetime(row.get("captured_utc"))
        kickoff = parse_datetime(row.get("kickoff_utc"))
        if captured is None or kickoff is None:
            counts["missing_or_invalid_time"] += 1
            continue
        if captured >= kickoff:
            counts["at_or_after_kickoff"] += 1
            continue
        counts["pre_kickoff"] += 1
        row["minutes_before_kickoff"] = round(
            (kickoff - captured).total_seconds() / 60.0,
            2,
        )
        valid.append(row)

    return counts, valid


def audit_record_hashes(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(rows),
        "verified": 0,
        "missing_hash": 0,
        "missing_payload": 0,
        "invalid_payload_json": 0,
        "hash_mismatch": 0,
    }

    for row in rows:
        stored_hash = str(row.get("record_sha256") or "").strip()
        raw_payload = row.get("record_json")
        if not stored_hash:
            counts["missing_hash"] += 1
            row["record_hash_verified"] = False
            continue
        if raw_payload in (None, ""):
            counts["missing_payload"] += 1
            row["record_hash_verified"] = False
            continue
        try:
            payload = json.loads(str(raw_payload))
        except (TypeError, ValueError, json.JSONDecodeError):
            counts["invalid_payload_json"] += 1
            row["record_hash_verified"] = False
            continue
        if sha256_payload(payload) == stored_hash:
            counts["verified"] += 1
            row["record_hash_verified"] = True
        else:
            counts["hash_mismatch"] += 1
            row["record_hash_verified"] = False

    return counts


def choose_snapshots(
    rows: list[dict[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("game_key") or "")].append(row)

    selected: list[dict[str, Any]] = []
    for game_rows in grouped.values():
        ordered = sorted(
            game_rows,
            key=lambda row: (
                parse_datetime(row.get("captured_utc"))
                or datetime.min.replace(tzinfo=timezone.utc),
                str(row.get("snapshot_id") or ""),
            ),
        )
        selected.append(ordered[0] if policy == "first" else ordered[-1])

    return sorted(
        selected,
        key=lambda row: (
            parse_datetime(row.get("kickoff_utc"))
            or datetime.max.replace(tzinfo=timezone.utc),
            str(row.get("game_key") or ""),
        ),
    )


def outcome_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    winner_values = [
        float(value)
        for row in rows
        if (value := safe_int(row.get("winner_correct"))) in (0, 1)
    ]
    brier_values = [
        value
        for row in rows
        if (value := safe_float(row.get("brier_score"))) is not None
    ]
    model_abs_errors = [
        value
        for row in rows
        if (value := safe_float(row.get("absolute_margin_error"))) is not None
    ]
    hardrock_abs_errors = [
        value
        for row in rows
        if (value := safe_float(row.get("hardrock_absolute_margin_error"))) is not None
    ]

    model_mae = mean(model_abs_errors)
    hardrock_mae = mean(hardrock_abs_errors)

    return {
        "scored_games": len(rows),
        "winner_metric_games": len(winner_values),
        "winner_accuracy": mean(winner_values),
        "brier_metric_games": len(brier_values),
        "brier_score": mean(brier_values),
        "model_margin_metric_games": len(model_abs_errors),
        "model_margin_mae": model_mae,
        "hardrock_margin_metric_games": len(hardrock_abs_errors),
        "hardrock_margin_mae": hardrock_mae,
        "model_minus_hardrock_mae": (
            model_mae - hardrock_mae
            if model_mae is not None and hardrock_mae is not None
            else None
        ),
    }


def audit_run_alignment(runs: list[dict[str, Any]]) -> dict[str, Any]:
    checked = len(runs)
    aligned = [
        row
        for row in runs
        if safe_int(row.get("record_count")) == safe_int(row.get("stored_snapshot_count"))
    ]
    return {
        "runs_with_season_snapshots": checked,
        "row_count_aligned_runs": len(aligned),
        "row_count_mismatch_runs": checked - len(aligned),
        "mismatched_run_ids": [
            str(row.get("run_id") or "")
            for row in runs
            if row not in aligned
        ],
    }


def model_artifact_catalog() -> dict[str, Any]:
    if not MODEL_DIR.exists():
        return {
            "directory_exists": False,
            "file_count": 0,
            "files": [],
            "note": "No model-artifact hash comparison is asserted when the directory is absent.",
        }

    files = [path for path in sorted(MODEL_DIR.rglob("*")) if path.is_file()]
    return {
        "directory_exists": True,
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(MODEL_DIR.parent.parent).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "note": (
            "These are current artifact fingerprints. This audit catalogs them; "
            "it does not claim a comparison against an external baseline manifest."
        ),
    }


def selected_rows_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "snapshot_id": row.get("snapshot_id"),
            "game_key": row.get("game_key"),
            "season": row.get("season"),
            "week": row.get("week"),
            "captured_utc": row.get("captured_utc"),
            "kickoff_utc": row.get("kickoff_utc"),
            "minutes_before_kickoff": row.get("minutes_before_kickoff"),
            "away_team": row.get("away_team"),
            "home_team": row.get("home_team"),
            "model_favorite": row.get("model_favorite"),
            "nn_input_status": row.get("nn_input_status"),
            "nn_heads_agree": row.get("nn_heads_agree"),
            "strength_label": row.get("strength_label"),
            "hardrock_available": row.get("hardrock_available"),
            "record_hash_verified": row.get("record_hash_verified"),
            "scored_utc": row.get("scored_utc"),
            "winner_correct": row.get("winner_correct"),
            "absolute_margin_error": row.get("absolute_margin_error"),
            "hardrock_absolute_margin_error": row.get("hardrock_absolute_margin_error"),
        }
        for row in rows
    ]


def write_outputs(payload: dict[str, Any], selected_rows: list[dict[str, Any]]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    fields = [
        "snapshot_id",
        "game_key",
        "season",
        "week",
        "captured_utc",
        "kickoff_utc",
        "minutes_before_kickoff",
        "away_team",
        "home_team",
        "model_favorite",
        "nn_input_status",
        "nn_heads_agree",
        "strength_label",
        "hardrock_available",
        "record_hash_verified",
        "scored_utc",
        "winner_correct",
        "absolute_margin_error",
        "hardrock_absolute_margin_error",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)


def build_payload(
    *,
    season: int,
    policy: str,
    snapshots: list[dict[str, Any]],
    timing: dict[str, int],
    hash_counts: dict[str, int],
    runs: list[dict[str, Any]],
    triggers: dict[str, str],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    scored = [row for row in selected if row.get("scored_utc")]
    unscored = [row for row in selected if not row.get("scored_utc")]
    artifacts = model_artifact_catalog()
    run_alignment = audit_run_alignment(runs)

    input_statuses = Counter(
        str(row.get("nn_input_status") or "MISSING").strip() or "MISSING"
        for row in selected
    )
    strengths = Counter(
        str(row.get("strength_label") or "N/A").strip().upper() or "N/A"
        for row in selected
    )
    required_trigger_set = set(triggers)

    checks = {
        "all_snapshots_strictly_pre_kickoff": (
            timing["at_or_after_kickoff"] == 0
            and timing["missing_or_invalid_time"] == 0
        ),
        "all_stored_record_hashes_verified": (
            hash_counts["verified"] == hash_counts["total"]
        ),
        "normal_immutability_triggers_present": (
            REQUIRED_IMMUTABILITY_TRIGGERS <= required_trigger_set
        ),
        "run_record_counts_aligned": (
            run_alignment["row_count_mismatch_runs"] == 0
        ),
        "selected_records_have_verified_hashes": all(
            bool(row.get("record_hash_verified")) for row in selected
        ),
    }

    metrics = outcome_metrics(scored)
    evidence_ok = all(checks.values())

    if metrics["scored_games"]:
        finding = (
            "The stored evidence supports an honest post-event evaluation process. "
            f"Among {metrics['scored_games']} {policy}-policy completed games, the model selected "
            f"the winner {pct(metrics['winner_accuracy'])} of the time, while its mean absolute "
            f"margin error was {fmt(metrics['model_margin_mae'], 2)} points. "
            f"The captured Hard Rock margin baseline had a mean absolute error of "
            f"{fmt(metrics['hardrock_margin_mae'], 2)} points. "
            "This is descriptive evidence, not proof of a reliable decision advantage or a wagering strategy."
        )
    else:
        finding = (
            "Pre-event evidence is available, but no selected completed games are currently "
            "available for outcome evaluation."
        )

    return {
        "schema_version": 1,
        "generated_utc": utc_iso(),
        "study_title": (
            "Beyond Hindsight: Building an Auditable Neural Decision-Support "
            "System for Real-Time Forecasting"
        ),
        "mode": "READ_ONLY_RESEARCH_AUDIT_NO_API_CALLS_NO_MODEL_CHANGES",
        "selection_policy": policy,
        "season": season,
        "scope_note": (
            "College football is the live evaluation setting. This report studies evidence timing, "
            "integrity, and later evaluation; it does not recommend wagers or claim profitability."
        ),
        "audit_pass": evidence_ok,
        "audit_checks": checks,
        "snapshot_evidence": {
            "stored_snapshots": len(snapshots),
            "unique_games_with_snapshots": len({str(row.get('game_key') or '') for row in snapshots}),
            "time_integrity": timing,
            "record_integrity": hash_counts,
            "run_alignment": run_alignment,
            "immutability_guard_names_found": sorted(required_trigger_set),
            "required_immutability_guard_names": sorted(REQUIRED_IMMUTABILITY_TRIGGERS),
        },
        "selected_evaluation_set": {
            "strictly_pre_kickoff_games_selected": len(selected),
            "selected_completed_games": len(scored),
            "selected_awaiting_or_unmatched_games": len(unscored),
            "selected_input_statuses": dict(sorted(input_statuses.items())),
            "selected_strength_labels": dict(sorted(strengths.items())),
        },
        "descriptive_outcomes": metrics,
        "model_artifact_catalog": artifacts,
        "research_interpretation": {
            "finding": finding,
            "limits": [
                "This is a prospective evidence audit, not a randomized trial.",
                "A high winner-selection rate alone does not establish a useful decision advantage.",
                "Margin accuracy and comparison to the captured public baseline must be reported alongside winner accuracy.",
                "The current artifact catalog is a fingerprint inventory, not an external cryptographic attestation service.",
                "No profitability, automated-action, or causal claim is made by this report.",
            ],
        },
    }


def print_summary(payload: dict[str, Any]) -> None:
    evidence = payload["snapshot_evidence"]
    timing = evidence["time_integrity"]
    hashes = evidence["record_integrity"]
    selected = payload["selected_evaluation_set"]
    metrics = payload["descriptive_outcomes"]

    rule()
    print("CFB EDGE ENGINE — RESEARCH HINDSIGHT AUDIT V1")
    rule()
    print(f"Mode:   {payload['mode']}")
    print(f"Season: {payload['season']} | Snapshot policy: {payload['selection_policy'].upper()}")
    print()

    print("TIMING INTEGRITY")
    rule("-")
    print(f"Stored prediction snapshots:      {timing['total']}")
    print(f"Captured before kickoff:          {timing['pre_kickoff']}")
    print(f"At/after kickoff:                 {timing['at_or_after_kickoff']}")
    print(f"Missing/invalid timing:           {timing['missing_or_invalid_time']}")
    print()

    print("RECORD INTEGRITY")
    rule("-")
    print(f"Stored record hashes verified:    {hashes['verified']}/{hashes['total']}")
    print(f"Hash mismatches:                  {hashes['hash_mismatch']}")
    print(f"Missing/invalid payloads:         {hashes['missing_payload'] + hashes['invalid_payload_json']}")
    print(f"Immutability guards present:      {payload['audit_checks']['normal_immutability_triggers_present']}")
    print(f"Run row-count alignment:          {payload['audit_checks']['run_record_counts_aligned']}")
    print()

    print("STRICT PRE-KICKOFF EVALUATION SET")
    rule("-")
    print(f"Games selected:                   {selected['strictly_pre_kickoff_games_selected']}")
    print(f"Completed games scored:           {selected['selected_completed_games']}")
    print(f"Awaiting/unmatched games:         {selected['selected_awaiting_or_unmatched_games']}")
    print()

    print("DESCRIPTIVE OUTCOMES")
    rule("-")
    print(f"Winner accuracy:                  {pct(metrics['winner_accuracy'])}")
    print(f"Brier score:                      {fmt(metrics['brier_score'], 3)}")
    print(f"Model margin MAE:                 {fmt(metrics['model_margin_mae'], 2)} pts")
    print(f"Hard Rock margin MAE:             {fmt(metrics['hardrock_margin_mae'], 2)} pts")
    print(f"Model minus Hard Rock MAE:        {fmt(metrics['model_minus_hardrock_mae'], 2)} pts")
    print()

    print("AUDIT CONCLUSION")
    rule("-")
    print("PASS" if payload["audit_pass"] else "REVIEW REQUIRED")
    print(payload["research_interpretation"]["finding"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only prospective evidence audit for the CFB Edge research study."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season to audit (default: 2026).",
    )
    parser.add_argument(
        "--policy",
        choices=("latest", "first"),
        default="latest",
        help="One strictly pre-kickoff snapshot selected per game (default: latest).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the audit without writing derived JSON/CSV outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_audit_data(args.season)
    snapshots = data["snapshots"]

    timing, strictly_pre_kickoff = audit_timing(snapshots)
    hash_counts = audit_record_hashes(snapshots)
    selected = choose_snapshots(strictly_pre_kickoff, args.policy)

    payload = build_payload(
        season=args.season,
        policy=args.policy,
        snapshots=snapshots,
        timing=timing,
        hash_counts=hash_counts,
        runs=data["runs"],
        triggers=data["triggers"],
        selected=selected,
    )
    print_summary(payload)

    if args.dry_run:
        print()
        print("DRY RUN COMPLETE — no derived research files were written.")
    else:
        csv_rows = selected_rows_for_csv(selected)
        write_outputs(payload, csv_rows)
        print()
        print("FILES WRITTEN")
        rule("-")
        print(OUTPUT_JSON)
        print(OUTPUT_CSV)

    return 0 if payload["audit_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
