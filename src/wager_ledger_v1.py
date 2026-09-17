#!/usr/bin/env python3
"""
CFB Edge Engine — Actual Wager Ledger V1

Purpose
-------
Store wagers the user actually places, separately from model prediction history.
Each wager freezes the model/market context that existed at the time the wager
was logged and, when possible, links to the latest immutable prediction snapshot
captured at or before the wager time.

Safety / integrity
------------------
- This ledger NEVER invents a wager from a model signal.
- Actual wagers are user-entered.
- Wager grading uses only exact game_key results already verified by the strict
  prediction-history scorer. There is no team-pair-only result matching here.
- Model/market context copied into a wager is historical evidence and is never
  refreshed to newer predictions after the ticket is logged.
- Prediction history is opened read-only.

Current scope
-------------
V1 supports spread wagers only. The schema leaves room for future moneyline,
total, CLV, and closing-market fields without changing the meaning of existing
records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROSPECTIVE_DIR = PROJECT_ROOT / "data" / "prospective"
WAGER_DB = PROSPECTIVE_DIR / "wager_ledger.sqlite"
WAGER_EVENTS_JSONL = PROSPECTIVE_DIR / "wager_ledger_events.jsonl"
HISTORY_DB = PROSPECTIVE_DIR / "prediction_history.sqlite"

LEDGER_SCHEMA_VERSION = 1
DEFAULT_BOOK = "Hard Rock Bet Florida"


# ============================================================================
# GENERIC HELPERS
# ============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    value = value or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
        for pattern in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def american_profit(stake: float, odds: int) -> float:
    if stake <= 0:
        raise ValueError("Stake must be greater than zero.")
    if -100 < odds < 100:
        raise ValueError("American odds must be <= -100 or >= +100.")
    if odds > 0:
        return stake * (odds / 100.0)
    return stake * (100.0 / abs(odds))


def make_game_key(signal: dict[str, Any]) -> str:
    game_id = signal.get("game_id")
    if game_id not in (None, ""):
        return f"cfbd:{str(game_id).strip()}"

    season = safe_int(signal.get("season"))
    week = safe_int(signal.get("week"))
    away = str(signal.get("away_team") or "").strip()
    home = str(signal.get("home_team") or "").strip()
    if season is not None and week is not None and away and home:
        # This fallback is only an identifier for the wager record. It is NOT used
        # to attach a result. Result grading still requires exact game_key equality
        # against the strict scorer's verified game_results table.
        return f"fallback:{season}:{week}:{away}:{home}"

    raise ValueError("Could not establish a game identity for this wager.")


def signal_kickoff_utc(signal: dict[str, Any]) -> str | None:
    for field in (
        "kickoff_utc",
        "start_date",
        "start_date_utc",
        "commence_time_utc",
        "start_time",
    ):
        parsed = parse_datetime(signal.get(field))
        if parsed is not None:
            return utc_iso(parsed)
    return None


# ============================================================================
# DATABASE SETUP
# ============================================================================


def connect_wager_db(path: Path = WAGER_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def connect_history_read_only(path: Path = HISTORY_DB) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def initialize_ledger(path: Path = WAGER_DB) -> None:
    connection = connect_wager_db(path)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS wagers (
                wager_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                created_utc TEXT NOT NULL,
                placed_utc TEXT NOT NULL,

                source_book TEXT NOT NULL,
                market_type TEXT NOT NULL,
                status TEXT NOT NULL,

                game_key TEXT NOT NULL,
                game_id TEXT,
                season INTEGER,
                week INTEGER,
                kickoff_utc TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,

                wager_team TEXT NOT NULL,
                wager_line REAL NOT NULL,
                american_odds INTEGER NOT NULL,
                stake REAL NOT NULL,
                potential_profit REAL NOT NULL,
                notes TEXT,

                linked_snapshot_id TEXT,
                snapshot_captured_utc TEXT,
                snapshot_link_status TEXT NOT NULL,

                model_favorite TEXT,
                model_spread_display TEXT,
                model_home_margin REAL,
                model_favorite_win_probability REAL,
                model_home_win_probability REAL,
                hardrock_display TEXT,
                hardrock_spread_for_model_favorite REAL,
                hardrock_price_for_model_favorite REAL,
                reference_market_display TEXT,
                model_edge_points REAL,
                strength_label TEXT,
                strength_score REAL,
                signal_warning TEXT,
                nn_input_status TEXT,
                nn_margin_std REAL,
                nn_win_probability_std REAL,

                final_away_points REAL,
                final_home_points REAL,
                result TEXT,
                profit_loss REAL,
                graded_utc TEXT,
                result_source TEXT,

                closing_reference_display TEXT,
                closing_spread_for_wager_team REAL,
                clv_points REAL,
                clv_status TEXT NOT NULL DEFAULT 'PENDING',

                record_sha256 TEXT NOT NULL UNIQUE
            );

            CREATE INDEX IF NOT EXISTS idx_wagers_game_key
                ON wagers (game_key);

            CREATE INDEX IF NOT EXISTS idx_wagers_status
                ON wagers (status);

            CREATE INDEX IF NOT EXISTS idx_wagers_placed
                ON wagers (placed_utc);

            CREATE INDEX IF NOT EXISTS idx_wagers_snapshot
                ON wagers (linked_snapshot_id);

            CREATE TABLE IF NOT EXISTS wager_events (
                event_id TEXT PRIMARY KEY,
                wager_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_utc TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (wager_id) REFERENCES wagers(wager_id)
            );

            CREATE INDEX IF NOT EXISTS idx_wager_events_wager
                ON wager_events (wager_id, event_utc);
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO ledger_metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(LEDGER_SCHEMA_VERSION)),
        )
        connection.commit()
    finally:
        connection.close()


def append_event(
    connection: sqlite3.Connection,
    wager_id: str,
    event_type: str,
    payload: dict[str, Any],
    *,
    events_jsonl: Path = WAGER_EVENTS_JSONL,
) -> None:
    event = {
        "event_id": uuid.uuid4().hex,
        "wager_id": wager_id,
        "event_type": event_type,
        "event_utc": utc_iso(),
        "payload": payload,
    }
    connection.execute(
        """
        INSERT INTO wager_events (
            event_id, wager_id, event_type, event_utc, payload_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            event["event_id"],
            wager_id,
            event_type,
            event["event_utc"],
            canonical_json(payload),
        ),
    )

    # Append-only human-portable audit trail. The SQLite DB remains authoritative.
    events_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with events_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(event))
        handle.write("\n")


# ============================================================================
# SNAPSHOT LINKING
# ============================================================================


SNAPSHOT_CONTEXT_FIELDS = (
    "snapshot_id",
    "captured_utc",
    "model_favorite",
    "model_spread_display",
    "model_home_margin",
    "model_favorite_win_probability",
    "model_home_win_probability",
    "hardrock_display",
    "hardrock_spread_for_model_favorite",
    "hardrock_price_for_model_favorite",
    "reference_market_display",
    "model_edge_points",
    "strength_label",
    "strength_score",
    "signal_warning",
    "nn_input_status",
    "nn_margin_std",
    "nn_win_probability_std",
)


def latest_snapshot_at_or_before(
    signal: dict[str, Any],
    placed_utc: str,
    *,
    history_db: Path = HISTORY_DB,
) -> dict[str, Any] | None:
    connection = connect_history_read_only(history_db)
    if connection is None:
        return None

    try:
        game_key = make_game_key(signal)
        row = connection.execute(
            """
            SELECT *
            FROM prediction_snapshots
            WHERE game_key = ?
              AND captured_utc <= ?
            ORDER BY captured_utc DESC, snapshot_id DESC
            LIMIT 1
            """,
            (game_key, placed_utc),
        ).fetchone()

        # Compatibility fallback for a history DB whose game_key differs but whose
        # exact CFBD game_id is known. This is identity-safe and never team-only.
        if row is None and signal.get("game_id") not in (None, ""):
            row = connection.execute(
                """
                SELECT *
                FROM prediction_snapshots
                WHERE CAST(game_id AS TEXT) = ?
                  AND captured_utc <= ?
                ORDER BY captured_utc DESC, snapshot_id DESC
                LIMIT 1
                """,
                (str(signal.get("game_id")).strip(), placed_utc),
            ).fetchone()

        if row is None:
            return None
        return dict(row)
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def context_from_signal(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_id": None,
        "captured_utc": None,
        "model_favorite": signal.get("model_favorite"),
        "model_spread_display": signal.get("model_spread_display"),
        "model_home_margin": safe_float(signal.get("model_home_margin")),
        "model_favorite_win_probability": safe_float(
            signal.get("model_favorite_win_probability")
        ),
        "model_home_win_probability": safe_float(signal.get("model_home_win_probability")),
        "hardrock_display": signal.get("hardrock_display"),
        "hardrock_spread_for_model_favorite": safe_float(
            signal.get("hardrock_spread_for_model_favorite")
        ),
        "hardrock_price_for_model_favorite": safe_float(
            signal.get("hardrock_price_for_model_favorite")
        ),
        "reference_market_display": signal.get("reference_market_display"),
        "model_edge_points": safe_float(signal.get("model_edge_points")),
        "strength_label": signal.get("strength_label"),
        "strength_score": safe_float(signal.get("strength_score")),
        "signal_warning": signal.get("signal_warning"),
        "nn_input_status": signal.get("nn_input_status"),
        "nn_margin_std": safe_float(signal.get("nn_margin_std")),
        "nn_win_probability_std": safe_float(signal.get("nn_win_probability_std")),
    }


# ============================================================================
# WAGER CREATION
# ============================================================================


def validate_wager_team(signal: dict[str, Any], wager_team: str) -> str:
    away = str(signal.get("away_team") or "").strip()
    home = str(signal.get("home_team") or "").strip()
    normalized = normalize_team(wager_team)
    if normalized == normalize_team(away):
        return away
    if normalized == normalize_team(home):
        return home
    raise ValueError("Wager team must be one of the two teams in the selected game.")


def log_spread_wager(
    signal: dict[str, Any],
    *,
    wager_team: str,
    wager_line: float,
    american_odds: int,
    stake: float,
    placed_utc: str | None = None,
    notes: str = "",
    source_book: str = DEFAULT_BOOK,
    wager_db: Path = WAGER_DB,
    history_db: Path = HISTORY_DB,
    events_jsonl: Path = WAGER_EVENTS_JSONL,
) -> dict[str, Any]:
    initialize_ledger(wager_db)

    if not isinstance(signal, dict):
        raise ValueError("A selected live-board signal is required.")

    away = str(signal.get("away_team") or "").strip()
    home = str(signal.get("home_team") or "").strip()
    if not away or not home:
        raise ValueError("Selected game is missing team identity.")

    canonical_team = validate_wager_team(signal, wager_team)
    line = safe_float(wager_line)
    if line is None:
        raise ValueError("A valid spread is required.")

    odds = safe_int(american_odds)
    if odds is None or (-100 < odds < 100):
        raise ValueError("American odds must be <= -100 or >= +100.")

    stake_value = safe_float(stake)
    if stake_value is None or stake_value <= 0:
        raise ValueError("Stake must be greater than zero.")

    placed_dt = parse_datetime(placed_utc) if placed_utc else utc_now()
    if placed_dt is None:
        raise ValueError("Could not parse wager placement time.")
    placed_text = utc_iso(placed_dt)

    kickoff_text = signal_kickoff_utc(signal)
    kickoff_dt = parse_datetime(kickoff_text)
    if kickoff_dt is not None and placed_dt >= kickoff_dt:
        raise ValueError("The wager placement time must be before kickoff.")

    snapshot = latest_snapshot_at_or_before(
        signal,
        placed_text,
        history_db=history_db,
    )
    if snapshot is not None:
        context = {field: snapshot.get(field) for field in SNAPSHOT_CONTEXT_FIELDS}
        link_status = "VERIFIED_PRIOR_SNAPSHOT"
    else:
        # The ticket is still real and should be recordable, but model-at-bet
        # performance must not pretend a prior prediction existed.
        context = context_from_signal(signal)
        link_status = "UNLINKED_NO_PRIOR_SNAPSHOT"

    game_key = make_game_key(signal)
    wager_id = "wgr_" + uuid.uuid4().hex
    created_text = utc_iso()
    potential_profit = american_profit(stake_value, odds)

    immutable_payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "wager_id": wager_id,
        "created_utc": created_text,
        "placed_utc": placed_text,
        "source_book": source_book,
        "market_type": "SPREAD",
        "game_key": game_key,
        "game_id": None if signal.get("game_id") in (None, "") else str(signal.get("game_id")),
        "season": safe_int(signal.get("season")),
        "week": safe_int(signal.get("week")),
        "kickoff_utc": kickoff_text,
        "away_team": away,
        "home_team": home,
        "wager_team": canonical_team,
        "wager_line": line,
        "american_odds": odds,
        "stake": stake_value,
        "potential_profit": potential_profit,
        "notes": str(notes or "").strip(),
        "linked_snapshot_id": context.get("snapshot_id"),
        "snapshot_captured_utc": context.get("captured_utc"),
        "snapshot_link_status": link_status,
        "model_favorite": context.get("model_favorite"),
        "model_spread_display": context.get("model_spread_display"),
        "model_home_margin": safe_float(context.get("model_home_margin")),
        "model_favorite_win_probability": safe_float(
            context.get("model_favorite_win_probability")
        ),
        "model_home_win_probability": safe_float(context.get("model_home_win_probability")),
        "hardrock_display": context.get("hardrock_display"),
        "hardrock_spread_for_model_favorite": safe_float(
            context.get("hardrock_spread_for_model_favorite")
        ),
        "hardrock_price_for_model_favorite": safe_float(
            context.get("hardrock_price_for_model_favorite")
        ),
        "reference_market_display": context.get("reference_market_display"),
        "model_edge_points": safe_float(context.get("model_edge_points")),
        "strength_label": context.get("strength_label"),
        "strength_score": safe_float(context.get("strength_score")),
        "signal_warning": context.get("signal_warning"),
        "nn_input_status": context.get("nn_input_status"),
        "nn_margin_std": safe_float(context.get("nn_margin_std")),
        "nn_win_probability_std": safe_float(context.get("nn_win_probability_std")),
    }
    record_sha256 = hashlib.sha256(canonical_json(immutable_payload).encode("utf-8")).hexdigest()

    connection = connect_wager_db(wager_db)
    try:
        connection.execute(
            """
            INSERT INTO wagers (
                wager_id, schema_version, created_utc, placed_utc,
                source_book, market_type, status,
                game_key, game_id, season, week, kickoff_utc, away_team, home_team,
                wager_team, wager_line, american_odds, stake, potential_profit, notes,
                linked_snapshot_id, snapshot_captured_utc, snapshot_link_status,
                model_favorite, model_spread_display, model_home_margin,
                model_favorite_win_probability, model_home_win_probability,
                hardrock_display, hardrock_spread_for_model_favorite,
                hardrock_price_for_model_favorite, reference_market_display,
                model_edge_points, strength_label, strength_score, signal_warning,
                nn_input_status, nn_margin_std, nn_win_probability_std,
                clv_status, record_sha256
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?
            )
            """,
            (
                wager_id,
                LEDGER_SCHEMA_VERSION,
                created_text,
                placed_text,
                source_book,
                "SPREAD",
                "PENDING",
                game_key,
                immutable_payload["game_id"],
                immutable_payload["season"],
                immutable_payload["week"],
                kickoff_text,
                away,
                home,
                canonical_team,
                line,
                odds,
                stake_value,
                potential_profit,
                immutable_payload["notes"],
                context.get("snapshot_id"),
                context.get("captured_utc"),
                link_status,
                context.get("model_favorite"),
                context.get("model_spread_display"),
                safe_float(context.get("model_home_margin")),
                safe_float(context.get("model_favorite_win_probability")),
                safe_float(context.get("model_home_win_probability")),
                context.get("hardrock_display"),
                safe_float(context.get("hardrock_spread_for_model_favorite")),
                safe_float(context.get("hardrock_price_for_model_favorite")),
                context.get("reference_market_display"),
                safe_float(context.get("model_edge_points")),
                context.get("strength_label"),
                safe_float(context.get("strength_score")),
                context.get("signal_warning"),
                context.get("nn_input_status"),
                safe_float(context.get("nn_margin_std")),
                safe_float(context.get("nn_win_probability_std")),
                "PENDING",
                record_sha256,
            ),
        )
        append_event(
            connection,
            wager_id,
            "WAGER_LOGGED",
            immutable_payload,
            events_jsonl=events_jsonl,
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM wagers WHERE wager_id = ?",
            (wager_id,),
        ).fetchone()
        return dict(row)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ============================================================================
# RESULT GRADING
# ============================================================================


def score_spread_ticket(
    *,
    away_team: str,
    home_team: str,
    wager_team: str,
    wager_line: float,
    american_odds: int,
    stake: float,
    away_points: float,
    home_points: float,
) -> tuple[str, float]:
    normalized_wager = normalize_team(wager_team)
    if normalized_wager == normalize_team(home_team):
        team_margin = home_points - away_points
    elif normalized_wager == normalize_team(away_team):
        team_margin = away_points - home_points
    else:
        raise ValueError("Wager team does not match the stored game teams.")

    cover_margin = team_margin + wager_line
    if math.isclose(cover_margin, 0.0, abs_tol=1e-9):
        return "PUSH", 0.0
    if cover_margin > 0:
        return "WON", american_profit(stake, american_odds)
    return "LOST", -stake


def score_pending_wagers(
    *,
    wager_db: Path = WAGER_DB,
    history_db: Path = HISTORY_DB,
    events_jsonl: Path = WAGER_EVENTS_JSONL,
) -> dict[str, int]:
    initialize_ledger(wager_db)
    history = connect_history_read_only(history_db)
    if history is None:
        return {"pending": 0, "graded": 0, "awaiting_result": 0}

    ledger = connect_wager_db(wager_db)
    try:
        pending_rows = ledger.execute(
            "SELECT * FROM wagers WHERE status = 'PENDING' ORDER BY placed_utc, wager_id"
        ).fetchall()
        graded = 0
        awaiting = 0

        for row in pending_rows:
            # Critical safety rule: exact game_key only. game_results itself is
            # populated by the strict result matcher, so this layer never guesses.
            result = history.execute(
                "SELECT * FROM game_results WHERE game_key = ?",
                (row["game_key"],),
            ).fetchone()
            if result is None:
                awaiting += 1
                continue

            away_points = safe_float(result["away_points"])
            home_points = safe_float(result["home_points"])
            if away_points is None or home_points is None:
                awaiting += 1
                continue

            ticket_result, profit_loss = score_spread_ticket(
                away_team=str(row["away_team"]),
                home_team=str(row["home_team"]),
                wager_team=str(row["wager_team"]),
                wager_line=float(row["wager_line"]),
                american_odds=int(row["american_odds"]),
                stake=float(row["stake"]),
                away_points=away_points,
                home_points=home_points,
            )
            graded_utc = utc_iso()
            result_source = str(result["result_source"] or "prediction_history.game_results")

            ledger.execute(
                """
                UPDATE wagers
                SET status = ?,
                    final_away_points = ?,
                    final_home_points = ?,
                    result = ?,
                    profit_loss = ?,
                    graded_utc = ?,
                    result_source = ?
                WHERE wager_id = ?
                  AND status = 'PENDING'
                """,
                (
                    ticket_result,
                    away_points,
                    home_points,
                    ticket_result,
                    profit_loss,
                    graded_utc,
                    result_source,
                    row["wager_id"],
                ),
            )
            append_event(
                ledger,
                str(row["wager_id"]),
                "WAGER_GRADED",
                {
                    "game_key": row["game_key"],
                    "result": ticket_result,
                    "away_points": away_points,
                    "home_points": home_points,
                    "profit_loss": profit_loss,
                    "result_source": result_source,
                },
                events_jsonl=events_jsonl,
            )
            graded += 1

        ledger.commit()
        return {
            "pending": len(pending_rows),
            "graded": graded,
            "awaiting_result": awaiting,
        }
    except Exception:
        ledger.rollback()
        raise
    finally:
        ledger.close()
        history.close()


# ============================================================================
# READ / SUMMARY
# ============================================================================


def list_wagers(*, wager_db: Path = WAGER_DB) -> list[dict[str, Any]]:
    initialize_ledger(wager_db)
    connection = connect_wager_db(wager_db)
    try:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM wagers ORDER BY placed_utc DESC, wager_id DESC"
            ).fetchall()
        ]
    finally:
        connection.close()


def ledger_summary(*, wager_db: Path = WAGER_DB) -> dict[str, Any]:
    rows = list_wagers(wager_db=wager_db)
    settled = [row for row in rows if row.get("status") in {"WON", "LOST", "PUSH"}]
    won = sum(1 for row in settled if row.get("status") == "WON")
    lost = sum(1 for row in settled if row.get("status") == "LOST")
    push = sum(1 for row in settled if row.get("status") == "PUSH")
    pending = sum(1 for row in rows if row.get("status") == "PENDING")
    risked = sum(float(row.get("stake") or 0.0) for row in settled)
    profit = sum(float(row.get("profit_loss") or 0.0) for row in settled)
    roi = (profit / risked) if risked > 0 else None
    linked = sum(1 for row in rows if row.get("snapshot_link_status") == "VERIFIED_PRIOR_SNAPSHOT")

    # Drawdown from realized wager P/L in chronological placement order.
    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    chronological = sorted(
        settled,
        key=lambda row: (str(row.get("placed_utc") or ""), str(row.get("wager_id") or "")),
    )
    for row in chronological:
        running += float(row.get("profit_loss") or 0.0)
        peak = max(peak, running)
        max_drawdown = max(max_drawdown, peak - running)

    return {
        "total_wagers": len(rows),
        "settled": len(settled),
        "pending": pending,
        "won": won,
        "lost": lost,
        "push": push,
        "risked": risked,
        "profit": profit,
        "roi": roi,
        "max_drawdown": max_drawdown,
        "snapshot_linked": linked,
        "snapshot_unlinked": len(rows) - linked,
    }


# ============================================================================
# CLI
# ============================================================================


def print_rule(char: str = "=") -> None:
    print(char * 104)


def command_init() -> int:
    initialize_ledger()
    print_rule()
    print("CFB EDGE ENGINE — ACTUAL WAGER LEDGER V1")
    print_rule()
    print(f"Ledger DB: {WAGER_DB}")
    print(f"Audit log: {WAGER_EVENTS_JSONL}")
    print("Status:    READY")
    print("No wagers were created automatically.")
    return 0


def command_score() -> int:
    stats = score_pending_wagers()
    print_rule()
    print("ACTUAL WAGER GRADER")
    print_rule()
    print(f"Pending before grading: {stats['pending']}")
    print(f"Newly graded:           {stats['graded']}")
    print(f"Awaiting exact result:  {stats['awaiting_result']}")
    print("Result source: strict prediction-history game_results only")
    return 0


def command_summary() -> int:
    summary = ledger_summary()
    print_rule()
    print("ACTUAL WAGER LEDGER SUMMARY")
    print_rule()
    print(f"Total wagers:     {summary['total_wagers']}")
    print(f"Pending:          {summary['pending']}")
    print(
        f"Settled:          {summary['settled']}  "
        f"({summary['won']}-{summary['lost']}-{summary['push']})"
    )
    print(f"Amount risked:    ${summary['risked']:.2f}")
    print(f"Profit / loss:    ${summary['profit']:+.2f}")
    if summary["roi"] is None:
        print("ROI:              N/A")
    else:
        print(f"ROI:              {summary['roi'] * 100:+.2f}%")
    print(f"Max drawdown:     ${summary['max_drawdown']:.2f}")
    print(
        "Snapshot links:   "
        f"{summary['snapshot_linked']} verified / "
        f"{summary['snapshot_unlinked']} unlinked"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CFB Edge Engine actual wager ledger")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("init", "score", "summary"),
        default="summary",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init":
        return command_init()
    if args.command == "score":
        return command_score()
    return command_summary()


if __name__ == "__main__":
    raise SystemExit(main())
