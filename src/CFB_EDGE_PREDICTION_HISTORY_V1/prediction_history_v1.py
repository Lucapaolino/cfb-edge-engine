#!/usr/bin/env python3
"""Shared helpers for the CFB Edge prospective prediction-history system."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROSPECTIVE_DIR = DATA_DIR / "prospective"
BATCH_DIR = PROSPECTIVE_DIR / "prediction_snapshots"
HISTORY_DB = PROSPECTIVE_DIR / "prediction_history.sqlite"
HISTORY_JSONL = PROSPECTIVE_DIR / "prediction_history.jsonl"
LEGACY_JSONL = PROSPECTIVE_DIR / "cfb_2026_signal_ledger.jsonl"
BOARD_JSON = PROCESSED_DIR / "live_model_market_board_2026_v1.json"
CFB_DB = DATA_DIR / "cfb.db"

SCHEMA_VERSION = 2


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def first_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            value = record.get(key)
            if value is not None and value != "":
                return value
    return None


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def probability01(value: Any) -> float | None:
    p = safe_float(value)
    if p is None:
        return None
    if 1.0 < p <= 100.0:
        p /= 100.0
    if p < 0.0 or p > 1.0:
        return None
    return p


def normalize_team(value: Any) -> str:
    text = clean_text(value) or ""
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_board_records(board: Any) -> list[dict[str, Any]]:
    if isinstance(board, list):
        return [row for row in board if isinstance(row, dict)]
    if isinstance(board, dict):
        for key in ("records", "signals", "games", "rows"):
            value = board.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def actionable_record(record: dict[str, Any]) -> bool:
    value = first_value(record, "game_is_actionable", "is_actionable", "pregame_actionable")
    return safe_int(value) == 1


def kickoff_utc(record: dict[str, Any]) -> str | None:
    return clean_text(
        first_value(
            record,
            "start_date_utc",
            "start_date",
            "commence_time_utc",
            "commence_time",
            "kickoff_utc",
            "kickoff",
        )
    )


def derive_game_key(record: dict[str, Any]) -> str:
    game_id = clean_text(first_value(record, "game_id", "id", "cfbd_game_id"))
    if game_id:
        return f"cfbd:{game_id}"

    season = clean_text(first_value(record, "season")) or "?"
    week = clean_text(first_value(record, "week")) or "?"
    away = normalize_team(first_value(record, "away_team", "away"))
    home = normalize_team(first_value(record, "home_team", "home"))
    start = kickoff_utc(record) or "?"
    return f"match:{season}|{week}|{away}|{home}|{start}"


def derive_model_home_margin(record: dict[str, Any]) -> float | None:
    direct = safe_float(
        first_value(
            record,
            "model_home_margin",
            "nn_home_margin",
            "predicted_home_margin",
            "raw_home_margin",
            "home_margin",
        )
    )
    if direct is not None:
        return direct

    favorite = normalize_team(first_value(record, "model_favorite", "projected_winner"))
    home = normalize_team(first_value(record, "home_team", "home"))
    away = normalize_team(first_value(record, "away_team", "away"))
    spread_value = safe_float(first_value(record, "model_spread", "projected_spread"))
    if spread_value is None or not favorite:
        return None

    magnitude = abs(spread_value)
    if favorite == home:
        return magnitude
    if favorite == away:
        return -magnitude
    return None


def derive_model_home_probability(record: dict[str, Any]) -> float | None:
    direct = probability01(
        first_value(
            record,
            "model_home_win_probability",
            "raw_home_win_probability",
            "home_win_probability",
            "home_win_p",
        )
    )
    if direct is not None:
        return direct

    fav_p = probability01(
        first_value(
            record,
            "model_favorite_win_probability",
            "favorite_win_probability",
            "win_probability",
        )
    )
    if fav_p is None:
        return None

    favorite = normalize_team(first_value(record, "model_favorite", "projected_winner"))
    home = normalize_team(first_value(record, "home_team", "home"))
    away = normalize_team(first_value(record, "away_team", "away"))
    if favorite == home:
        return fav_p
    if favorite == away:
        return 1.0 - fav_p
    return None


def derive_execution_spread_for_model_favorite(record: dict[str, Any]) -> float | None:
    return safe_float(
        first_value(
            record,
            "hardrock_spread",
            "hardrock_spread_for_model_favorite",
            "execution_spread_for_model_favorite",
        )
    )


def derive_execution_price_for_model_favorite(record: dict[str, Any]) -> float | None:
    return safe_float(
        first_value(
            record,
            "hardrock_spread_price",
            "hardrock_price",
            "hardrock_price_for_model_favorite",
            "execution_price_for_model_favorite",
        )
    )


def derive_reference_spread_for_model_favorite(record: dict[str, Any]) -> float | None:
    return safe_float(
        first_value(
            record,
            "reference_market_spread",
            "reference_consensus_spread",
            "reference_spread_for_model_favorite",
            "consensus_spread_for_model_favorite",
        )
    )


def model_favorite(record: dict[str, Any]) -> str | None:
    return clean_text(first_value(record, "model_favorite", "projected_winner"))


def extract_snapshot_fields(record: dict[str, Any], *, run_id: str, captured_utc: str) -> dict[str, Any]:
    away_team = clean_text(first_value(record, "away_team", "away"))
    home_team = clean_text(first_value(record, "home_team", "home"))
    favorite = model_favorite(record)
    game_key = derive_game_key(record)

    model_home_margin = derive_model_home_margin(record)
    model_home_probability = derive_model_home_probability(record)
    model_fav_probability = probability01(
        first_value(
            record,
            "model_favorite_win_probability",
            "favorite_win_probability",
            "win_probability",
        )
    )

    payload_hash = sha256_json(record)
    snapshot_id = hashlib.sha256(
        f"{run_id}|{game_key}|{payload_hash}".encode("utf-8")
    ).hexdigest()

    return {
        "snapshot_id": snapshot_id,
        "run_id": run_id,
        "captured_utc": captured_utc,
        "schema_version": SCHEMA_VERSION,
        "game_key": game_key,
        "game_id": clean_text(first_value(record, "game_id", "id", "cfbd_game_id")),
        "season": safe_int(first_value(record, "season")),
        "week": safe_int(first_value(record, "week")),
        "kickoff_utc": kickoff_utc(record),
        "away_team": away_team,
        "home_team": home_team,
        "model_favorite": favorite,
        "model_spread": safe_float(first_value(record, "model_spread", "projected_spread")),
        "model_spread_display": clean_text(first_value(record, "model_spread_display")),
        "model_home_margin": model_home_margin,
        "model_favorite_win_probability": model_fav_probability,
        "model_home_win_probability": model_home_probability,
        "hardrock_available": safe_int(first_value(record, "hardrock_available")),
        "hardrock_spread_for_model_favorite": derive_execution_spread_for_model_favorite(record),
        "hardrock_price_for_model_favorite": derive_execution_price_for_model_favorite(record),
        "hardrock_display": clean_text(first_value(record, "hardrock_display")),
        "reference_spread_for_model_favorite": derive_reference_spread_for_model_favorite(record),
        "reference_market_display": clean_text(first_value(record, "reference_market_display")),
        "reference_market_books": safe_int(first_value(record, "reference_market_books")),
        "model_edge_points": safe_float(first_value(record, "model_edge_points")),
        "model_vs_consensus_abs_points": safe_float(first_value(record, "model_vs_consensus_abs_points")),
        "hardrock_vs_consensus_points": safe_float(first_value(record, "hardrock_vs_consensus_points")),
        "strength_label": (clean_text(first_value(record, "strength_label")) or "N/A").upper(),
        "strength_score": safe_float(first_value(record, "strength_score")),
        "signal_warning": clean_text(first_value(record, "signal_warning")) or "NONE",
        "nn_input_status": clean_text(first_value(record, "nn_input_status", "input_status")),
        "nn_feature_coverage": safe_int(first_value(record, "nn_feature_coverage", "feature_coverage")),
        "nn_margin_std": safe_float(first_value(record, "nn_margin_std", "margin_std")),
        "nn_win_probability_std": safe_float(first_value(record, "nn_win_probability_std", "win_probability_std", "nn_probability_std")),
        "nn_heads_agree": safe_int(first_value(record, "nn_heads_agree", "heads_agree")),
        "wager_decision": clean_text(first_value(record, "wager_decision")) or "UNSET_NO_FROZEN_WAGER_RULE",
        "wager_side": clean_text(first_value(record, "wager_side")),
        "wager_line": safe_float(first_value(record, "wager_line")),
        "wager_price": safe_float(first_value(record, "wager_price")),
        "record_sha256": payload_hash,
        "record_json": canonical_json(record),
    }


def connect_history() -> sqlite3.Connection:
    PROSPECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_DB)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    ensure_schema(connection)
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS prediction_runs (
            run_id TEXT PRIMARY KEY,
            captured_utc TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            source_board TEXT,
            record_count INTEGER NOT NULL,
            batch_file TEXT,
            batch_sha256 TEXT
        );

        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            captured_utc TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            game_key TEXT NOT NULL,
            game_id TEXT,
            season INTEGER,
            week INTEGER,
            kickoff_utc TEXT,
            away_team TEXT,
            home_team TEXT,
            model_favorite TEXT,
            model_spread REAL,
            model_spread_display TEXT,
            model_home_margin REAL,
            model_favorite_win_probability REAL,
            model_home_win_probability REAL,
            hardrock_available INTEGER,
            hardrock_spread_for_model_favorite REAL,
            hardrock_price_for_model_favorite REAL,
            hardrock_display TEXT,
            reference_spread_for_model_favorite REAL,
            reference_market_display TEXT,
            reference_market_books INTEGER,
            model_edge_points REAL,
            model_vs_consensus_abs_points REAL,
            hardrock_vs_consensus_points REAL,
            strength_label TEXT,
            strength_score REAL,
            signal_warning TEXT,
            nn_input_status TEXT,
            nn_feature_coverage INTEGER,
            nn_margin_std REAL,
            nn_win_probability_std REAL,
            nn_heads_agree INTEGER,
            wager_decision TEXT NOT NULL,
            wager_side TEXT,
            wager_line REAL,
            wager_price REAL,
            record_sha256 TEXT NOT NULL,
            record_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES prediction_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_prediction_game_capture
            ON prediction_snapshots(game_key, captured_utc);
        CREATE INDEX IF NOT EXISTS idx_prediction_kickoff
            ON prediction_snapshots(kickoff_utc);
        CREATE INDEX IF NOT EXISTS idx_prediction_strength
            ON prediction_snapshots(strength_label);

        CREATE TABLE IF NOT EXISTS game_results (
            game_key TEXT PRIMARY KEY,
            game_id TEXT,
            season INTEGER,
            week INTEGER,
            away_team TEXT,
            home_team TEXT,
            away_points REAL NOT NULL,
            home_points REAL NOT NULL,
            actual_home_margin REAL NOT NULL,
            winner TEXT,
            result_source TEXT NOT NULL,
            result_captured_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prediction_scores (
            snapshot_id TEXT PRIMARY KEY,
            game_key TEXT NOT NULL,
            scored_utc TEXT NOT NULL,
            actual_home_margin REAL NOT NULL,
            predicted_home_margin REAL,
            margin_error REAL,
            absolute_margin_error REAL,
            squared_margin_error REAL,
            winner_correct INTEGER,
            brier_score REAL,
            hardrock_implied_home_margin REAL,
            hardrock_absolute_margin_error REAL,
            model_side_ats_result TEXT,
            model_side_flat_units REAL,
            FOREIGN KEY (snapshot_id) REFERENCES prediction_snapshots(snapshot_id),
            FOREIGN KEY (game_key) REFERENCES game_results(game_key)
        );
        """
    )

    # Predictions are evidence. Once written, they cannot be edited or deleted
    # through normal SQL without intentionally dropping these guards first.
    connection.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS prediction_snapshots_no_update
        BEFORE UPDATE ON prediction_snapshots
        BEGIN
            SELECT RAISE(ABORT, 'prediction_snapshots is immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS prediction_snapshots_no_delete
        BEFORE DELETE ON prediction_snapshots
        BEGIN
            SELECT RAISE(ABORT, 'prediction_snapshots is immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS prediction_runs_no_update
        BEFORE UPDATE ON prediction_runs
        BEGIN
            SELECT RAISE(ABORT, 'prediction_runs is immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS prediction_runs_no_delete
        BEFORE DELETE ON prediction_runs
        BEGIN
            SELECT RAISE(ABORT, 'prediction_runs is immutable');
        END;
        """
    )
    connection.commit()


def insert_prediction_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    captured_utc: str,
    source_board: str,
    batch_file: str,
    batch_sha256: str,
    rows: Iterable[dict[str, Any]],
) -> int:
    row_list = list(rows)
    connection.execute(
        """
        INSERT INTO prediction_runs (
            run_id, captured_utc, schema_version, source_board,
            record_count, batch_file, batch_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            captured_utc,
            SCHEMA_VERSION,
            source_board,
            len(row_list),
            batch_file,
            batch_sha256,
        ),
    )

    columns = [
        "snapshot_id", "run_id", "captured_utc", "schema_version",
        "game_key", "game_id", "season", "week", "kickoff_utc",
        "away_team", "home_team", "model_favorite", "model_spread",
        "model_spread_display", "model_home_margin",
        "model_favorite_win_probability", "model_home_win_probability",
        "hardrock_available", "hardrock_spread_for_model_favorite",
        "hardrock_price_for_model_favorite", "hardrock_display",
        "reference_spread_for_model_favorite", "reference_market_display",
        "reference_market_books", "model_edge_points",
        "model_vs_consensus_abs_points", "hardrock_vs_consensus_points",
        "strength_label", "strength_score", "signal_warning",
        "nn_input_status", "nn_feature_coverage", "nn_margin_std",
        "nn_win_probability_std", "nn_heads_agree", "wager_decision",
        "wager_side", "wager_line", "wager_price", "record_sha256",
        "record_json",
    ]
    placeholders = ",".join("?" for _ in columns)
    sql = (
        f"INSERT INTO prediction_snapshots ({','.join(columns)}) "
        f"VALUES ({placeholders})"
    )
    connection.executemany(
        sql,
        [tuple(row.get(column) for column in columns) for row in row_list],
    )
    return len(row_list)


def american_profit_for_one_unit(price: float | None) -> float | None:
    if price is None or price == 0:
        return None
    if price > 0:
        return price / 100.0
    return 100.0 / abs(price)
