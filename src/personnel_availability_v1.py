#!/usr/bin/env python3
"""
CFB Edge Engine — Personnel Availability V1

Prospective personnel / injury research layer.\n\nOfficial-source status update: supports OUT_FIRST_HALF from conference reports.

This V1 uses the exact local contracts confirmed by the audit:
- live board: data/processed/live_model_market_board_2026_v1.json
- games: data/cfb.db -> games
- player stats: data/cfb.db -> player_game_stat_values

Design rules
------------
1. NO API calls.
2. NO neural-model changes.
3. Availability observations are immutable once inserted.
4. Both capture time and source publication time must be before kickoff.
5. Current games are matched by exact game_id, or by exact
   season + week + team + opponent when game_id is unavailable.
6. Season-to-date player importance uses only games with week < target week.
7. QB availability is tracked separately.
8. personnel_loss_index is research context only. It is NOT points,
   cover probability, or an automatic wagering rule.

First run
---------
python src/personnel_availability_v1.py --init --write-template --build

Import observations
-------------------
python src/personnel_availability_v1.py --import-csv data/manual/personnel.csv --build

Status
------
python src/personnel_availability_v1.py --status
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROSPECTIVE_DIR = DATA_DIR / "prospective"

CFB_DB = DATA_DIR / "cfb.db"
BOARD_JSON = PROCESSED_DIR / "live_model_market_board_2026_v1.json"

PERSONNEL_DB = PROSPECTIVE_DIR / "personnel_availability.sqlite"
PERSONNEL_JSONL = PROSPECTIVE_DIR / "personnel_availability.jsonl"

OUTPUT_JSON = PROCESSED_DIR / "personnel_availability_2026_v1.json"
OUTPUT_CSV = PROCESSED_DIR / "personnel_availability_2026_v1.csv"
IMPORT_TEMPLATE = PROCESSED_DIR / "personnel_import_template_2026.csv"

SEASON = 2026
SCHEMA_VERSION = 1
WIDTH = 112

PLAYER_STATS_TABLE = "player_game_stat_values"
GAMES_TABLE = "games"

REQUIRED_PLAYER_COLUMNS = {
    "game_id",
    "team",
    "athlete_id",
    "athlete_name",
    "category",
    "stat_type",
    "stat_value",
}

REQUIRED_GAME_COLUMNS = {
    "game_id",
    "season",
    "week",
    "start_date",
    "home_team",
    "away_team",
}

STATUS_SEVERITY = {
    "AVAILABLE": 0.00,
    "PROBABLE": 0.15,
    "QUESTIONABLE": 0.40,
    "OUT_FIRST_HALF": 0.50,
    "DOUBTFUL": 0.75,
    "OUT": 1.00,
    "SUSPENDED": 1.00,
    "SEASON_OUT": 1.00,
    "UNKNOWN": 0.35,
}

STATUS_ALIASES = {
    "ACTIVE": "AVAILABLE",
    "AVAILABLE": "AVAILABLE",
    "CLEARED": "AVAILABLE",
    "PROBABLE": "PROBABLE",
    "EXPECTED_TO_PLAY": "PROBABLE",
    "QUESTIONABLE": "QUESTIONABLE",
    "GTD": "QUESTIONABLE",
    "GAME_TIME_DECISION": "QUESTIONABLE",
    "DOUBTFUL": "DOUBTFUL",
    "OUT_1ST_HALF": "OUT_FIRST_HALF",
    "OUT_FIRST_HALF": "OUT_FIRST_HALF",
    "OUT_FIRST_HALF_ONLY": "OUT_FIRST_HALF",
    "OUT": "OUT",
    "INACTIVE": "OUT",
    "WILL_NOT_PLAY": "OUT",
    "SUSPENDED": "SUSPENDED",
    "SEASON_OUT": "SEASON_OUT",
    "OUT_FOR_SEASON": "SEASON_OUT",
}

QB_POSITIONS = {"QB", "QUARTERBACK"}

IMPORT_COLUMNS = [
    "captured_utc",
    "source_published_utc",
    "source_name",
    "source_type",
    "source_url",
    "season",
    "week",
    "game_id",
    "kickoff_utc",
    "team",
    "opponent",
    "player_name",
    "athlete_id",
    "position",
    "status",
    "detail",
    "is_starter",
    "manual_importance_override",
    "notes",
]


# =============================================================================
# HELPERS
# =============================================================================

def rule(char: str = "=") -> None:
    print(char * WIDTH)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def parse_dt(value: Any) -> datetime | None:
    text = clean(value)

    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def norm_team(value: Any) -> str:
    text = clean(value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def norm_player(value: Any) -> str:
    text = clean(value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def same_team(left: Any, right: Any) -> bool:
    a = norm_team(left)
    b = norm_team(right)
    return bool(a and b and a == b)


def normalize_position(value: Any) -> str:
    text = clean(value).upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return text.strip("_")


def normalize_status(value: Any) -> str:
    text = clean(value).upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")

    if not text:
        return "UNKNOWN"

    return STATUS_ALIASES.get(
        text,
        text if text in STATUS_SEVERITY else "UNKNOWN",
    )


def json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def source_confidence(source_type: str) -> str:
    value = clean(source_type).casefold()

    if value in {
        "official",
        "official_conference",
        "official_team",
        "official_school",
    }:
        return "HIGH"

    if value in {
        "provider",
        "licensed_provider",
        "trusted_provider",
    }:
        return "MEDIUM"

    if value in {
        "manual",
        "media",
        "news",
    }:
        return "REVIEW"

    return "UNKNOWN"


# =============================================================================
# LIVE BOARD
# =============================================================================

def normalize_board_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [
            row
            for row in payload
            if isinstance(row, dict)
        ]

    if isinstance(payload, dict):
        for key in (
            "rows",
            "records",
            "games",
            "signals",
            "board",
            "data",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                rows = [
                    row
                    for row in value
                    if isinstance(row, dict)
                ]

                if rows:
                    return rows

    return []


def load_board() -> list[dict[str, Any]]:
    if not BOARD_JSON.exists():
        raise FileNotFoundError(
            f"Live board not found: {BOARD_JSON}"
        )

    payload = json.loads(
        BOARD_JSON.read_text(
            encoding="utf-8"
        )
    )

    rows = normalize_board_records(payload)

    if not rows:
        raise RuntimeError(
            "Live board contains no identifiable game rows."
        )

    return rows


def game_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": clean(row.get("game_id")),
        "season": safe_int(row.get("season")) or SEASON,
        "week": safe_int(row.get("week")),
        "kickoff_utc": clean(
            row.get("kickoff_utc")
            or
            row.get("start_date")
        ),
        "away_team": clean(row.get("away_team")),
        "home_team": clean(row.get("home_team")),
    }


def resolve_current_game(
    raw: dict[str, Any],
    board: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Strict matching:
    1. exact game_id
    2. exact season + week + team + opponent

    There is NO team-pair-only fallback.
    """
    game_id = clean(raw.get("game_id"))

    if game_id:
        matches = [
            game_identity(row)
            for row in board
            if clean(row.get("game_id")) == game_id
        ]

        if len(matches) == 1:
            return matches[0]

        return None

    season = safe_int(raw.get("season"))
    week = safe_int(raw.get("week"))
    team = clean(raw.get("team"))
    opponent = clean(raw.get("opponent"))

    if (
        season is None
        or
        week is None
        or
        not team
        or
        not opponent
    ):
        return None

    matches = []

    for row in board:
        game = game_identity(row)

        if game["season"] != season:
            continue

        if game["week"] != week:
            continue

        teams = (
            game["away_team"],
            game["home_team"],
        )

        if not any(
            same_team(team, candidate)
            for candidate in teams
        ):
            continue

        if not any(
            same_team(opponent, candidate)
            for candidate in teams
        ):
            continue

        if same_team(team, opponent):
            continue

        matches.append(game)

    return matches[0] if len(matches) == 1 else None


# =============================================================================
# EXACT LOCAL DATABASE CONTRACT
# =============================================================================

def connect_cfb() -> sqlite3.Connection:
    if not CFB_DB.exists():
        raise FileNotFoundError(
            f"CFB database not found: {CFB_DB}"
        )

    connection = sqlite3.connect(
        f"file:{CFB_DB}?mode=ro",
        uri=True,
    )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")

    return connection


def table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    }


def verify_local_contract() -> None:
    connection = connect_cfb()

    try:
        player_columns = table_columns(
            connection,
            PLAYER_STATS_TABLE,
        )

        game_columns = table_columns(
            connection,
            GAMES_TABLE,
        )

    finally:
        connection.close()

    missing_player = sorted(
        REQUIRED_PLAYER_COLUMNS
        -
        player_columns
    )

    missing_games = sorted(
        REQUIRED_GAME_COLUMNS
        -
        game_columns
    )

    if missing_player:
        raise RuntimeError(
            "player_game_stat_values contract changed; missing: "
            +
            ", ".join(missing_player)
        )

    if missing_games:
        raise RuntimeError(
            "games contract changed; missing: "
            +
            ", ".join(missing_games)
        )


# =============================================================================
# IMMUTABLE PERSONNEL DATABASE
# =============================================================================

def connect_personnel(
    *,
    read_only: bool = False,
) -> sqlite3.Connection:
    PERSONNEL_DB.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if read_only:
        if not PERSONNEL_DB.exists():
            raise FileNotFoundError(
                f"Personnel DB not found: {PERSONNEL_DB}"
            )

        connection = sqlite3.connect(
            f"file:{PERSONNEL_DB}?mode=ro",
            uri=True,
        )

        connection.execute(
            "PRAGMA query_only = ON"
        )

    else:
        connection = sqlite3.connect(
            str(PERSONNEL_DB)
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")

    return connection


def ensure_personnel_schema(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS personnel_observations (
            observation_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,

            captured_utc TEXT NOT NULL,
            source_published_utc TEXT,
            source_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_url TEXT,
            source_confidence TEXT NOT NULL,

            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            kickoff_utc TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_team TEXT NOT NULL,

            team TEXT NOT NULL,
            opponent TEXT NOT NULL,

            player_name TEXT NOT NULL,
            athlete_id TEXT,
            position TEXT,
            status TEXT NOT NULL,
            detail TEXT,
            is_starter INTEGER,
            manual_importance_override REAL,
            notes TEXT,

            prekickoff_valid INTEGER NOT NULL,
            validation_status TEXT NOT NULL,
            record_sha256 TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_personnel_game_capture
            ON personnel_observations(game_id, captured_utc);

        CREATE INDEX IF NOT EXISTS idx_personnel_player_capture
            ON personnel_observations(
                game_id,
                team,
                player_name,
                captured_utc
            );

        CREATE INDEX IF NOT EXISTS idx_personnel_kickoff
            ON personnel_observations(kickoff_utc);

        CREATE TRIGGER IF NOT EXISTS personnel_observations_no_update
        BEFORE UPDATE ON personnel_observations
        BEGIN
            SELECT RAISE(
                ABORT,
                'personnel_observations is immutable'
            );
        END;

        CREATE TRIGGER IF NOT EXISTS personnel_observations_no_delete
        BEFORE DELETE ON personnel_observations
        BEGIN
            SELECT RAISE(
                ABORT,
                'personnel_observations is immutable'
            );
        END;
        """
    )


# =============================================================================
# IMPORT TEMPLATE / CANONICAL OBSERVATION
# =============================================================================

def write_import_template(
    board: list[dict[str, Any]],
) -> None:
    IMPORT_TEMPLATE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for row in board:
        game = game_identity(row)

        for team, opponent in (
            (
                game["away_team"],
                game["home_team"],
            ),
            (
                game["home_team"],
                game["away_team"],
            ),
        ):
            rows.append(
                {
                    "captured_utc": utc_iso(),
                    "source_published_utc": "",
                    "source_name": "",
                    "source_type": "official",
                    "source_url": "",
                    "season": game["season"],
                    "week": game["week"] or "",
                    "game_id": game["game_id"],
                    "kickoff_utc": game["kickoff_utc"],
                    "team": team,
                    "opponent": opponent,
                    "player_name": "",
                    "athlete_id": "",
                    "position": "",
                    "status": "",
                    "detail": "",
                    "is_starter": "",
                    "manual_importance_override": "",
                    "notes": "",
                }
            )

    with IMPORT_TEMPLATE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=IMPORT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


def canonical_observation(
    raw: dict[str, Any],
    board: list[dict[str, Any]],
) -> dict[str, Any]:
    source_name = clean(raw.get("source_name"))
    player_name = clean(raw.get("player_name"))
    team = clean(raw.get("team"))

    if not source_name:
        raise ValueError("source_name is required")

    if not player_name:
        raise ValueError("player_name is required")

    if not team:
        raise ValueError("team is required")

    game = resolve_current_game(
        raw,
        board,
    )

    if game is None:
        raise ValueError(
            "could not resolve row to exactly one current game"
        )

    if not any(
        same_team(team, candidate)
        for candidate in (
            game["away_team"],
            game["home_team"],
        )
    ):
        raise ValueError(
            "team does not match resolved game"
        )

    canonical_team = (
        game["away_team"]
        if same_team(team, game["away_team"])
        else
        game["home_team"]
    )

    opponent = (
        game["home_team"]
        if same_team(canonical_team, game["away_team"])
        else
        game["away_team"]
    )

    kickoff = parse_dt(game["kickoff_utc"])

    if kickoff is None:
        raise ValueError(
            "resolved game has invalid kickoff timestamp"
        )

    captured = parse_dt(
        raw.get("captured_utc")
    )

    if captured is None:
        captured = utc_now()

    published = parse_dt(
        raw.get("source_published_utc")
    )

    capture_valid = captured < kickoff

    publication_valid = (
        published is None
        or
        published < kickoff
    )

    valid = (
        capture_valid
        and
        publication_valid
    )

    if not capture_valid:
        validation_status = (
            "REJECTED_CAPTURED_AT_OR_AFTER_KICKOFF"
        )

    elif not publication_valid:
        validation_status = (
            "REJECTED_SOURCE_PUBLISHED_AT_OR_AFTER_KICKOFF"
        )

    else:
        validation_status = "VALID_PREKICKOFF"

    starter_text = clean(
        raw.get("is_starter")
    ).casefold()

    if starter_text in {
        "1",
        "true",
        "yes",
        "y",
        "starter",
    }:
        starter = 1

    elif starter_text in {
        "0",
        "false",
        "no",
        "n",
        "backup",
    }:
        starter = 0

    else:
        starter = None

    override = safe_float(
        raw.get(
            "manual_importance_override"
        )
    )

    if override is not None:
        override = max(
            0.0,
            min(
                100.0,
                override,
            ),
        )

    source_type = clean(
        raw.get("source_type")
        or
        "manual"
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "captured_utc": captured.isoformat(
            timespec="seconds"
        ),
        "source_published_utc": (
            published.isoformat(
                timespec="seconds"
            )
            if published is not None
            else
            None
        ),
        "source_name": source_name,
        "source_type": source_type,
        "source_url": clean(
            raw.get("source_url")
        )
        or
        None,
        "source_confidence": source_confidence(
            source_type
        ),
        "game_id": game["game_id"],
        "season": game["season"],
        "week": game["week"],
        "kickoff_utc": game["kickoff_utc"],
        "away_team": game["away_team"],
        "home_team": game["home_team"],
        "team": canonical_team,
        "opponent": opponent,
        "player_name": player_name,
        "athlete_id": clean(
            raw.get("athlete_id")
        )
        or
        None,
        "position": normalize_position(
            raw.get("position")
        )
        or
        None,
        "status": normalize_status(
            raw.get("status")
        ),
        "detail": clean(
            raw.get("detail")
        )
        or
        None,
        "is_starter": starter,
        "manual_importance_override": override,
        "notes": clean(
            raw.get("notes")
        )
        or
        None,
        "prekickoff_valid": 1 if valid else 0,
        "validation_status": validation_status,
    }

    record["record_sha256"] = json_hash(
        record
    )

    record["observation_id"] = hashlib.sha256(
        (
            f"{record['captured_utc']}|"
            f"{record['game_id']}|"
            f"{norm_team(record['team'])}|"
            f"{record['athlete_id'] or norm_player(record['player_name'])}|"
            f"{record['record_sha256']}"
        ).encode("utf-8")
    ).hexdigest()

    return record


def append_jsonl(
    record: dict[str, Any],
) -> None:
    PERSONNEL_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PERSONNEL_JSONL.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                sort_keys=True,
                default=str,
            )
            +
            "\n"
        )


def import_observations(
    csv_path: Path,
    board: list[dict[str, Any]],
) -> dict[str, int]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Import CSV not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        missing = [
            column
            for column in IMPORT_COLUMNS
            if column not in (
                reader.fieldnames
                or
                []
            )
        ]

        if missing:
            raise RuntimeError(
                "Import CSV is missing columns: "
                +
                ", ".join(missing)
            )

        raw_rows = list(reader)

    connection = connect_personnel(
        read_only=False
    )

    ensure_personnel_schema(
        connection
    )

    columns = [
        "observation_id",
        "schema_version",
        "captured_utc",
        "source_published_utc",
        "source_name",
        "source_type",
        "source_url",
        "source_confidence",
        "game_id",
        "season",
        "week",
        "kickoff_utc",
        "away_team",
        "home_team",
        "team",
        "opponent",
        "player_name",
        "athlete_id",
        "position",
        "status",
        "detail",
        "is_starter",
        "manual_importance_override",
        "notes",
        "prekickoff_valid",
        "validation_status",
        "record_sha256",
    ]

    sql = (
        "INSERT INTO personnel_observations ("
        +
        ",".join(columns)
        +
        ") VALUES ("
        +
        ",".join(
            "?"
            for _ in columns
        )
        +
        ")"
    )

    result = {
        "inserted": 0,
        "duplicates": 0,
        "invalid": 0,
        "rejected_pregame": 0,
    }

    try:
        for raw in raw_rows:
            if not clean(
                raw.get("player_name")
            ):
                # Blank template rows are intentionally ignored.
                continue

            try:
                record = canonical_observation(
                    raw,
                    board,
                )

            except Exception as error:
                result["invalid"] += 1

                print(
                    "INVALID  "
                    +
                    clean(
                        raw.get("player_name")
                    )
                    +
                    " | "
                    +
                    str(error)
                )

                continue

            try:
                connection.execute(
                    sql,
                    tuple(
                        record.get(column)
                        for column in columns
                    ),
                )

                append_jsonl(record)

                result["inserted"] += 1

                if record["prekickoff_valid"] != 1:
                    result["rejected_pregame"] += 1

            except sqlite3.IntegrityError:
                result["duplicates"] += 1

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return result


# =============================================================================
# SEASON-TO-DATE PLAYER PRODUCTION
# =============================================================================

def stat_number(value: Any) -> float | None:
    text = clean(value)

    if not text:
        return None

    if "/" in text:
        return None

    text = (
        text
        .replace(",", "")
        .replace("%", "")
    )

    return safe_float(text)


def player_production_before_week(
    target_week: int,
) -> dict[str, dict[str, Any]]:
    """
    Aggregate ONLY prior 2026 games:
        games.week < target_week

    Offensive V1 production:
        passing yards
        rushing yards
        receiving yards

    We intentionally do not invent position information because the audit
    confirmed no roster/position table exists locally.
    """
    connection = connect_cfb()

    try:
        rows = connection.execute(
            """
            SELECT
                p.game_id,
                p.team,
                p.athlete_id,
                p.athlete_name,
                p.category,
                p.stat_type,
                p.stat_value,
                g.week
            FROM player_game_stat_values AS p
            INNER JOIN games AS g
                ON g.game_id = p.game_id
            WHERE g.season = ?
              AND g.week < ?
            """,
            (
                SEASON,
                target_week,
            ),
        ).fetchall()

    finally:
        connection.close()

    players = defaultdict(
        lambda: {
            "team": "",
            "athlete_id": "",
            "player_name": "",
            "passing_yards": 0.0,
            "rushing_yards": 0.0,
            "receiving_yards": 0.0,
            "games_seen": set(),
        }
    )

    for row in rows:
        team = clean(row["team"])
        athlete_id = clean(row["athlete_id"])
        player_name = clean(row["athlete_name"])

        if not team or not player_name:
            continue

        key = (
            norm_team(team)
            +
            "|"
            +
            (
                athlete_id
                or
                norm_player(player_name)
            )
        )

        item = players[key]

        item["team"] = team
        item["athlete_id"] = athlete_id
        item["player_name"] = player_name
        item["games_seen"].add(
            clean(row["game_id"])
        )

        category = clean(
            row["category"]
        ).casefold()

        stat_type = clean(
            row["stat_type"]
        ).upper()

        value = stat_number(
            row["stat_value"]
        )

        if value is None:
            continue

        if stat_type != "YDS":
            continue

        if category == "passing":
            item["passing_yards"] += value

        elif category == "rushing":
            item["rushing_yards"] += value

        elif category == "receiving":
            item["receiving_yards"] += value

    for item in players.values():
        item["games_seen"] = len(
            item["games_seen"]
        )

    return dict(players)


def production_shares(
    players: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    totals = defaultdict(
        lambda: {
            "passing_yards": 0.0,
            "rushing_yards": 0.0,
            "receiving_yards": 0.0,
        }
    )

    for player in players.values():
        team_key = norm_team(
            player["team"]
        )

        for metric in totals[
            team_key
        ]:
            totals[
                team_key
            ][
                metric
            ] += max(
                0.0,
                safe_float(
                    player.get(metric)
                )
                or
                0.0,
            )

    shares = {}

    for key, player in players.items():
        team_key = norm_team(
            player["team"]
        )

        shares[
            key
        ] = {}

        for metric, total in totals[
            team_key
        ].items():
            value = max(
                0.0,
                safe_float(
                    player.get(metric)
                )
                or
                0.0,
            )

            shares[
                key
            ][
                metric
                +
                "_share"
            ] = (
                value / total
                if total > 0.0
                else
                0.0
            )

    return shares


def find_player_production(
    observation: dict[str, Any],
    players: dict[str, dict[str, Any]],
) -> tuple[
    str | None,
    dict[str, Any] | None,
]:
    team = norm_team(
        observation.get("team")
    )

    athlete_id = clean(
        observation.get("athlete_id")
    )

    player_name = norm_player(
        observation.get("player_name")
    )

    if athlete_id:
        matches = [
            (
                key,
                player,
            )
            for key, player in players.items()
            if (
                norm_team(
                    player.get("team")
                )
                ==
                team
                and
                clean(
                    player.get("athlete_id")
                )
                ==
                athlete_id
            )
        ]

        if len(matches) == 1:
            return matches[0]

    matches = [
        (
            key,
            player,
        )
        for key, player in players.items()
        if (
            norm_team(
                player.get("team")
            )
            ==
            team
            and
            norm_player(
                player.get("player_name")
            )
            ==
            player_name
        )
    ]

    if len(matches) == 1:
        return matches[0]

    return (
        None,
        None,
    )


def importance_score(
    observation: dict[str, Any],
    player: dict[str, Any] | None,
    shares: dict[str, float] | None,
) -> tuple[
    float,
    str,
]:
    """
    Transparent 0-100 research importance index.

    This is deliberately NOT "points worth".
    """
    override = safe_float(
        observation.get(
            "manual_importance_override"
        )
    )

    if override is not None:
        return (
            max(
                0.0,
                min(
                    100.0,
                    override,
                ),
            ),
            "MANUAL_OVERRIDE",
        )

    position = normalize_position(
        observation.get("position")
    )

    starter = safe_int(
        observation.get("is_starter")
    )

    if (
        player is None
        or
        shares is None
    ):
        if position in QB_POSITIONS:
            return (
                75.0
                if starter == 1
                else
                55.0,
                "ROLE_FALLBACK",
            )

        if starter == 1:
            return (
                55.0,
                "ROLE_FALLBACK",
            )

        return (
            35.0,
            "ROLE_FALLBACK",
        )

    passing_share = (
        safe_float(
            shares.get(
                "passing_yards_share"
            )
        )
        or
        0.0
    )

    rushing_share = (
        safe_float(
            shares.get(
                "rushing_yards_share"
            )
        )
        or
        0.0
    )

    receiving_share = (
        safe_float(
            shares.get(
                "receiving_yards_share"
            )
        )
        or
        0.0
    )

    if position in QB_POSITIONS:
        production_share = max(
            passing_share,
            0.35 * rushing_share,
        )

    else:
        production_share = max(
            rushing_share,
            receiving_share,
            0.50 * passing_share,
        )

    score = (
        15.0
        +
        75.0
        *
        min(
            1.0,
            production_share,
        )
    )

    if starter == 1:
        score += 7.5

    if position in QB_POSITIONS:
        score = max(
            score,
            70.0
            if starter == 1
            else
            55.0,
        )

    return (
        max(
            0.0,
            min(
                100.0,
                score,
            ),
        ),
        "SEASON_TO_DATE_PRODUCTION_SHARE",
    )


# =============================================================================
# LATEST VALID OBSERVATIONS / REPORT
# =============================================================================

def latest_valid_observations() -> list[dict[str, Any]]:
    if not PERSONNEL_DB.exists():
        return []

    connection = connect_personnel(
        read_only=True
    )

    try:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM personnel_observations
                WHERE season = ?
                  AND prekickoff_valid = 1
                ORDER BY captured_utc, observation_id
                """,
                (
                    SEASON,
                ),
            ).fetchall()
        ]

    finally:
        connection.close()

    latest = {}

    for row in rows:
        key = (
            clean(row.get("game_id")),
            norm_team(row.get("team")),
            (
                clean(
                    row.get("athlete_id")
                )
                or
                norm_player(
                    row.get("player_name")
                )
            ),
        )

        existing = latest.get(key)

        if existing is None:
            latest[key] = row
            continue

        row_time = (
            parse_dt(
                row.get("captured_utc")
            )
            or
            datetime.min.replace(
                tzinfo=timezone.utc
            )
        )

        existing_time = (
            parse_dt(
                existing.get("captured_utc")
            )
            or
            datetime.min.replace(
                tzinfo=timezone.utc
            )
        )

        if row_time > existing_time:
            latest[key] = row

    return list(
        latest.values()
    )


def build_report(
    board: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = latest_valid_observations()

    by_game = defaultdict(list)

    for observation in observations:
        by_game[
            clean(
                observation.get("game_id")
            )
        ].append(
            observation
        )

    production_cache = {}
    share_cache = {}

    games = []

    production_matched = 0
    production_unmatched = 0

    for board_row in board:
        game = game_identity(
            board_row
        )

        week = game["week"]

        if week is None:
            raise RuntimeError(
                f"Current game {game['game_id']} has no week."
            )

        if week not in production_cache:
            production_cache[
                week
            ] = player_production_before_week(
                week
            )

            share_cache[
                week
            ] = production_shares(
                production_cache[
                    week
                ]
            )

        production = production_cache[
            week
        ]

        shares_lookup = share_cache[
            week
        ]

        game_observations = by_game.get(
            game["game_id"],
            [],
        )

        team_results = {}

        for team in (
            game["away_team"],
            game["home_team"],
        ):
            players_out = []

            for observation in game_observations:
                if not same_team(
                    observation.get("team"),
                    team,
                ):
                    continue

                production_key, player = find_player_production(
                    observation,
                    production,
                )

                player_shares = (
                    shares_lookup.get(
                        production_key
                    )
                    if production_key is not None
                    else
                    None
                )

                if player is not None:
                    production_matched += 1

                else:
                    production_unmatched += 1

                importance, method = importance_score(
                    observation,
                    player,
                    player_shares,
                )

                status = normalize_status(
                    observation.get("status")
                )

                severity = STATUS_SEVERITY.get(
                    status,
                    STATUS_SEVERITY["UNKNOWN"],
                )

                loss_index = (
                    importance
                    *
                    severity
                )

                position = normalize_position(
                    observation.get("position")
                )

                players_out.append(
                    {
                        "player_name": observation.get(
                            "player_name"
                        ),
                        "athlete_id": observation.get(
                            "athlete_id"
                        ),
                        "position": position or None,
                        "status": status,
                        "status_severity": severity,
                        "is_starter": observation.get(
                            "is_starter"
                        ),
                        "importance_score": round(
                            importance,
                            2,
                        ),
                        "importance_method": method,
                        "availability_loss_index": round(
                            loss_index,
                            2,
                        ),
                        "production_match": (
                            player is not None
                        ),
                        "prior_passing_yards": (
                            round(
                                safe_float(
                                    player.get(
                                        "passing_yards"
                                    )
                                )
                                or
                                0.0,
                                1,
                            )
                            if player is not None
                            else
                            None
                        ),
                        "prior_rushing_yards": (
                            round(
                                safe_float(
                                    player.get(
                                        "rushing_yards"
                                    )
                                )
                                or
                                0.0,
                                1,
                            )
                            if player is not None
                            else
                            None
                        ),
                        "prior_receiving_yards": (
                            round(
                                safe_float(
                                    player.get(
                                        "receiving_yards"
                                    )
                                )
                                or
                                0.0,
                                1,
                            )
                            if player is not None
                            else
                            None
                        ),
                        "source_name": observation.get(
                            "source_name"
                        ),
                        "source_type": observation.get(
                            "source_type"
                        ),
                        "source_confidence": observation.get(
                            "source_confidence"
                        ),
                        "source_url": observation.get(
                            "source_url"
                        ),
                        "captured_utc": observation.get(
                            "captured_utc"
                        ),
                        "source_published_utc": observation.get(
                            "source_published_utc"
                        ),
                        "detail": observation.get(
                            "detail"
                        ),
                        "notes": observation.get(
                            "notes"
                        ),
                    }
                )

            players_out.sort(
                key=lambda item: (
                    safe_float(
                        item.get(
                            "availability_loss_index"
                        )
                    )
                    or
                    0.0
                ),
                reverse=True,
            )

            total_loss = sum(
                safe_float(
                    item.get(
                        "availability_loss_index"
                    )
                )
                or
                0.0
                for item in players_out
                if (
                    safe_float(
                        item.get(
                            "status_severity"
                        )
                    )
                    or
                    0.0
                )
                >
                0.0
            )

            qb_rows = [
                item
                for item in players_out
                if normalize_position(
                    item.get("position")
                )
                in
                QB_POSITIONS
            ]

            qb_out = any(
                (
                    item.get("status")
                    in
                    {
                        "OUT",
                        "SUSPENDED",
                        "SEASON_OUT",
                    }
                )
                and
                (
                    safe_int(
                        item.get(
                            "is_starter"
                        )
                    )
                    ==
                    1
                    or
                    (
                        safe_float(
                            item.get(
                                "importance_score"
                            )
                        )
                        or
                        0.0
                    )
                    >=
                    70.0
                )
                for item in qb_rows
            )

            qb_uncertain = any(
                (
                    item.get("status")
                    in
                    {
                        "QUESTIONABLE",
                        "OUT_FIRST_HALF",
                        "DOUBTFUL",
                        "UNKNOWN",
                    }
                )
                and
                (
                    safe_int(
                        item.get(
                            "is_starter"
                        )
                    )
                    ==
                    1
                    or
                    (
                        safe_float(
                            item.get(
                                "importance_score"
                            )
                        )
                        or
                        0.0
                    )
                    >=
                    55.0
                )
                for item in qb_rows
            )

            team_results[
                team
            ] = {
                "team": team,
                "player_observations": len(
                    players_out
                ),
                "players_with_availability_concern": sum(
                    1
                    for item in players_out
                    if (
                        safe_float(
                            item.get(
                                "status_severity"
                            )
                        )
                        or
                        0.0
                    )
                    >
                    0.0
                ),
                "personnel_loss_index": round(
                    total_loss,
                    2,
                ),
                "qb_out": qb_out,
                "qb_uncertain": qb_uncertain,
                "players": players_out,
            }

        away = team_results[
            game["away_team"]
        ]

        home = team_results[
            game["home_team"]
        ]

        games.append(
            {
                **game,
                "away_personnel": away,
                "home_personnel": home,
                "home_minus_away_personnel_loss_index": round(
                    (
                        safe_float(
                            home.get(
                                "personnel_loss_index"
                            )
                        )
                        or
                        0.0
                    )
                    -
                    (
                        safe_float(
                            away.get(
                                "personnel_loss_index"
                            )
                        )
                        or
                        0.0
                    ),
                    2,
                ),
                "has_personnel_data": bool(
                    away["player_observations"]
                    or
                    home["player_observations"]
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_iso(),
        "season": SEASON,
        "method": "PROSPECTIVE_AVAILABILITY_CONTEXT_V1",
        "model_usage": "RESEARCH_ONLY_NOT_IN_FROZEN_NEURAL_MODEL",
        "local_contract": {
            "games_table": GAMES_TABLE,
            "player_stats_table": PLAYER_STATS_TABLE,
            "roster_position_table": None,
        },
        "latest_valid_observations": len(
            observations
        ),
        "observations_enriched_with_production": (
            production_matched
        ),
        "observations_without_production_match": (
            production_unmatched
        ),
        "games": games,
        "warning": (
            "personnel_loss_index is a research/context index only. "
            "It is not points, cover probability, or a wagering rule."
        ),
    }


def write_outputs(
    report: dict[str, Any],
) -> None:
    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    rows = []

    for game in report.get(
        "games",
        [],
    ):
        for side in (
            "away_personnel",
            "home_personnel",
        ):
            team_payload = (
                game.get(side)
                or
                {}
            )

            for player in team_payload.get(
                "players",
                [],
            ):
                rows.append(
                    {
                        "game_id": game.get(
                            "game_id"
                        ),
                        "season": game.get(
                            "season"
                        ),
                        "week": game.get(
                            "week"
                        ),
                        "kickoff_utc": game.get(
                            "kickoff_utc"
                        ),
                        "away_team": game.get(
                            "away_team"
                        ),
                        "home_team": game.get(
                            "home_team"
                        ),
                        "team": team_payload.get(
                            "team"
                        ),
                        **player,
                    }
                )

    fields = [
        "game_id",
        "season",
        "week",
        "kickoff_utc",
        "away_team",
        "home_team",
        "team",
        "player_name",
        "athlete_id",
        "position",
        "status",
        "status_severity",
        "is_starter",
        "importance_score",
        "importance_method",
        "availability_loss_index",
        "production_match",
        "prior_passing_yards",
        "prior_rushing_yards",
        "prior_receiving_yards",
        "source_name",
        "source_type",
        "source_confidence",
        "source_url",
        "captured_utc",
        "source_published_utc",
        "detail",
        "notes",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# STATUS / CLI
# =============================================================================

def database_status() -> dict[str, int]:
    if not PERSONNEL_DB.exists():
        return {
            "observations": 0,
            "valid": 0,
            "rejected": 0,
        }

    connection = connect_personnel(
        read_only=True
    )

    try:
        total = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM personnel_observations
                """
            ).fetchone()[0]
        )

        valid = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM personnel_observations
                WHERE prekickoff_valid = 1
                """
            ).fetchone()[0]
        )

    finally:
        connection.close()

    return {
        "observations": total,
        "valid": valid,
        "rejected": (
            total
            -
            valid
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prospective college-football personnel availability research layer."
        )
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="Create/verify immutable personnel database.",
    )

    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write current-slate personnel import template.",
    )

    parser.add_argument(
        "--import-csv",
        type=Path,
        default=None,
        help="Import canonical personnel observations from CSV.",
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Build current personnel research outputs.",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Print personnel database status.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not any(
        (
            args.init,
            args.write_template,
            args.import_csv is not None,
            args.build,
            args.status,
        )
    ):
        args.init = True
        args.write_template = True
        args.build = True

    rule()
    print(
        "CFB EDGE ENGINE — PERSONNEL AVAILABILITY V1"
    )
    rule()

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        "Mode: PROSPECTIVE RESEARCH / NO API CALLS / NO MODEL CHANGES"
    )

    rule()

    verify_local_contract()

    board = load_board()

    print(
        f"Live board rows:             {len(board)}"
    )

    print(
        f"Player-stat table:           {PLAYER_STATS_TABLE}"
    )

    print(
        f"Game/week table:             {GAMES_TABLE}"
    )

    print(
        "Roster/position table:      NONE — position must come from source/import in V1"
    )

    if args.init:
        connection = connect_personnel(
            read_only=False
        )

        try:
            ensure_personnel_schema(
                connection
            )

            connection.commit()

        finally:
            connection.close()

        print()
        print(
            "PERSONNEL DATABASE: PASS"
        )

        print(
            PERSONNEL_DB
        )

    if args.write_template:
        write_import_template(
            board
        )

        print()
        print(
            "IMPORT TEMPLATE WRITTEN"
        )

        print(
            IMPORT_TEMPLATE
        )

    if args.import_csv is not None:
        result = import_observations(
            args.import_csv,
            board,
        )

        print()
        print(
            "IMPORT RESULT"
        )

        rule("-")

        print(
            f"Inserted:             {result['inserted']}"
        )

        print(
            f"Duplicates skipped:   {result['duplicates']}"
        )

        print(
            f"Rejected pre-kickoff: {result['rejected_pregame']}"
        )

        print(
            f"Invalid rows:         {result['invalid']}"
        )

    if args.build:
        report = build_report(
            board
        )

        write_outputs(
            report
        )

        games_with_data = sum(
            1
            for game in report["games"]
            if game.get(
                "has_personnel_data"
            )
        )

        qb_out = 0
        qb_uncertain = 0

        for game in report["games"]:
            for side in (
                "away_personnel",
                "home_personnel",
            ):
                team_payload = (
                    game.get(side)
                    or
                    {}
                )

                qb_out += int(
                    bool(
                        team_payload.get(
                            "qb_out"
                        )
                    )
                )

                qb_uncertain += int(
                    bool(
                        team_payload.get(
                            "qb_uncertain"
                        )
                    )
                )

        print()
        print(
            "PERSONNEL RESEARCH OUTPUT"
        )

        rule("-")

        print(
            f"Current games:                    {len(report['games'])}"
        )

        print(
            f"Games with personnel data:        {games_with_data}"
        )

        print(
            f"Latest valid observations:        {report['latest_valid_observations']}"
        )

        print(
            "Production-matched observations:  "
            f"{report['observations_enriched_with_production']}"
        )

        print(
            "Unmatched observations:           "
            f"{report['observations_without_production_match']}"
        )

        print(
            f"QB OUT teams:                     {qb_out}"
        )

        print(
            f"QB uncertain teams:               {qb_uncertain}"
        )

        print()
        print(
            f"JSON: {OUTPUT_JSON}"
        )

        print(
            f"CSV:  {OUTPUT_CSV}"
        )

    if args.status:
        counts = database_status()

        print()
        print(
            "DATABASE STATUS"
        )

        rule("-")

        print(
            f"Observations:      {counts['observations']}"
        )

        print(
            f"Valid pre-kickoff: {counts['valid']}"
        )

        print(
            f"Rejected:          {counts['rejected']}"
        )

    print()
    print(
        "IMPORTANT"
    )

    rule("-")

    print(
        "personnel_loss_index is research context only."
    )

    print(
        "It is NOT points, cover probability, or an automatic bet signal."
    )

    print(
        "The frozen neural model is unchanged."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
