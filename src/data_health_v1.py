#!/usr/bin/env python3
"""
CFB Edge Engine — Data Health V2

This is a READ-ONLY local health check for the active 2026 production stack.

V2 fixes
--------
The first Data Health version made three incorrect schema assumptions:
1. It required model_home_margin on the live board even though the main board's
   user-facing model contract is model_spread_display + win probability.
2. It expected live nn_feature_coverage values even though the active board can
   represent feature quality primarily through nn_input_status.
3. It expected different field names in prediction_audit_summary_2026_v1.json.

Those mistakes could make a healthy system appear failed. V2 uses the actual
production contracts and fails closed only when the evidence genuinely indicates
a problem.

No API calls are made. No prediction, wager, model, score, result, or market
record is modified.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================================
# PATHS / CONSTANTS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOARD_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_model_market_board_2026_v1.json"
)

HISTORY_DB = (
    PROJECT_ROOT
    / "data"
    / "prospective"
    / "prediction_history.sqlite"
)

WAGER_DB = (
    PROJECT_ROOT
    / "data"
    / "prospective"
    / "wager_ledger.sqlite"
)

AUDIT_SUMMARY_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_audit_summary_2026_v1.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "cfb_nn_2026_v1"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "data_health_2026_v1.json"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "data_health_2026_v1.csv"
)

SEASON = 2026
EXPECTED_FEATURE_COUNT = 54

UNAVAILABLE_TEXT = {
    "",
    "N/A",
    "NA",
    "NONE",
    "NULL",
    "UNAVAILABLE",
    "HEADLINE UNAVAILABLE",
    "HEADLINE_UNAVAILABLE",
}


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    dt = value or utc_now()
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


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


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def available_text(value: Any) -> bool:
    return safe_text(value).upper() not in UNAVAILABLE_TEXT


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None

    text = str(value).strip()
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


def file_age_hours(path: Path) -> float | None:
    if not path.exists():
        return None

    modified = datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )

    return round(
        (utc_now() - modified).total_seconds() / 3600.0,
        2,
    )


def ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator <= 0:
        return None
    return round(
        numerator / denominator,
        4,
    )


def coverage_status(
    numerator: int,
    denominator: int,
    *,
    pass_ratio: float = 0.75,
    partial_ratio: float = 0.35,
) -> str:
    if denominator <= 0:
        return "N/A"

    value = numerator / denominator

    if value >= pass_ratio:
        return "PASS"

    if value >= partial_ratio:
        return "PARTIAL"

    return "WARNING"


def compact(
    parts: list[str],
) -> str:
    return " | ".join(
        part
        for part in parts
        if part
    )


# ============================================================================
# BOARD LOADING
# ============================================================================

def normalize_board_records(
    payload: Any,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [
            row
            for row in payload
            if isinstance(row, dict)
        ]

    if not isinstance(payload, dict):
        return []

    # Known / likely container keys first.
    for key in (
        "records",
        "rows",
        "signals",
        "games",
        "board",
        "data",
        "predictions",
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

    # Compatibility fallback for keyed-record payloads.
    candidates = []

    for value in payload.values():
        if not isinstance(value, dict):
            continue

        if (
            "away_team" in value
            and
            "home_team" in value
        ):
            candidates.append(value)

    return candidates


def load_board() -> list[dict[str, Any]]:
    if not BOARD_JSON.exists():
        raise FileNotFoundError(
            f"Live board not found: {BOARD_JSON}"
        )

    with BOARD_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    rows = normalize_board_records(
        payload
    )

    if not rows:
        raise RuntimeError(
            "Live board exists, but no game rows could be identified."
        )

    return rows


def is_actionable(
    row: dict[str, Any],
) -> bool:
    explicit = safe_int(
        row.get(
            "game_is_actionable"
        )
    )

    if explicit is not None:
        return explicit == 1

    # Compatibility fallback only when the board does not provide
    # game_is_actionable.
    for field in (
        "kickoff_utc",
        "start_date",
        "start_date_utc",
        "commence_time_utc",
    ):
        kickoff = parse_datetime(
            row.get(field)
        )

        if kickoff is not None:
            return kickoff > utc_now()

    return True


def actionable_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if is_actionable(row)
    ]


# ============================================================================
# LIVE MODEL HEALTH
# ============================================================================

def valid_probability(
    value: Any,
) -> bool:
    number = safe_float(value)

    if number is None:
        return False

    # Canonical model probabilities are normally 0-1. Keep support for
    # user-facing 0-100 percentages so the health layer is format tolerant.
    if 0.0 <= number <= 1.0:
        return True

    if 0.0 <= number <= 100.0:
        return True

    return False


def model_health(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)

    spread_ready = 0
    probability_ready = 0
    favorite_ready = 0

    for row in rows:
        if available_text(
            row.get(
                "model_spread_display"
            )
        ):
            spread_ready += 1

        if valid_probability(
            row.get(
                "model_favorite_win_probability"
            )
        ):
            probability_ready += 1

        favorite = safe_text(
            row.get(
                "model_favorite"
            )
        )

        if favorite:
            favorite_ready += 1

    model_dir_exists = MODEL_DIR.exists()

    if (
        total > 0
        and
        spread_ready == total
        and
        probability_ready == total
        and
        favorite_ready == total
        and
        model_dir_exists
    ):
        status = "PASS"

    elif (
        total > 0
        and
        spread_ready > 0
        and
        probability_ready > 0
    ):
        status = "WARNING"

    else:
        status = "FAIL"

    return {
        "status": status,
        "actionable_games": total,
        "spread_ready": spread_ready,
        "valid_probabilities": probability_ready,
        "model_favorites_present": favorite_ready,
        "model_artifact_exists": model_dir_exists,
    }


# ============================================================================
# FEATURE HEALTH
# ============================================================================

def classify_input_status(
    value: Any,
) -> str:
    text = safe_text(
        value
    ).upper()

    if not text:
        return "UNKNOWN"

    if text == "FULL_54":
        return "FULL"

    if "MISSING_STATIC_CONTEXT" in text:
        return "PARTIAL"

    if "HEAVY_IMPUTATION" in text:
        return "HEAVY"

    # Future input statuses containing FULL should still be treated as full
    # only when explicitly indicating the complete contract.
    if (
        "FULL" in text
        and
        "54" in text
    ):
        return "FULL"

    return "OTHER"


def feature_health(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)

    full = 0
    partial = 0
    heavy = 0
    other = 0
    unknown = 0

    numeric_coverages: list[int] = []
    used_numeric_coverage = False

    for row in rows:
        coverage = safe_int(
            row.get(
                "nn_feature_coverage"
            )
        )

        if coverage is not None:
            used_numeric_coverage = True
            numeric_coverages.append(
                coverage
            )

            if coverage >= EXPECTED_FEATURE_COUNT:
                full += 1

            elif coverage >= 40:
                partial += 1

            else:
                heavy += 1

            continue

        classification = classify_input_status(
            row.get(
                "nn_input_status"
            )
        )

        if classification == "FULL":
            full += 1

        elif classification == "PARTIAL":
            partial += 1

        elif classification == "HEAVY":
            heavy += 1

        elif classification == "OTHER":
            other += 1

        else:
            unknown += 1

    if total == 0:
        status = "N/A"

    elif (
        full == total
        and
        heavy == 0
        and
        other == 0
        and
        unknown == 0
    ):
        status = "PASS"

    elif (
        heavy == 0
        and
        unknown == 0
    ):
        # Partial feature context is expected for some FCS/static-context
        # situations and should not imply the model has failed.
        status = "PARTIAL"

    else:
        status = "WARNING"

    return {
        "status": status,
        "actionable_games": total,
        "expected_features": EXPECTED_FEATURE_COUNT,
        "full_games": full,
        "partial_games": partial,
        "heavy_imputation_games": heavy,
        "other_input_status_games": other,
        "unknown_input_status_games": unknown,
        "coverage_source": (
            "nn_feature_coverage"
            if used_numeric_coverage
            else "nn_input_status"
        ),
        "minimum_feature_coverage": (
            min(
                numeric_coverages
            )
            if numeric_coverages
            else None
        ),
        "maximum_feature_coverage": (
            max(
                numeric_coverages
            )
            if numeric_coverages
            else None
        ),
    }


# ============================================================================
# HARD ROCK HEALTH
# ============================================================================

def hardrock_health(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)

    usable = 0
    rejected = 0
    unavailable = 0
    quality_status_counts: dict[str, int] = {}

    for row in rows:
        available = (
            safe_int(
                row.get(
                    "hardrock_available"
                )
            )
            ==
            1
        )

        display = safe_text(
            row.get(
                "hardrock_display"
            )
        )

        quality = safe_text(
            row.get(
                "hardrock_spread_quality_status"
            )
        ).upper()

        if quality:
            quality_status_counts[
                quality
            ] = (
                quality_status_counts.get(
                    quality,
                    0,
                )
                +
                1
            )

        # hardrock_available is the active production board's authoritative
        # "usable headline" flag. Do not require a second internal spread field.
        if (
            available
            and
            available_text(
                display
            )
        ):
            usable += 1
            continue

        if (
            quality
            and
            quality not in {
                "N/A",
                "NA",
                "NONE",
                "HEADLINE_USABLE",
                "USABLE",
                "AVAILABLE",
            }
        ):
            rejected += 1

        else:
            unavailable += 1

    return {
        "status": coverage_status(
            usable,
            total,
            pass_ratio=0.75,
            partial_ratio=0.35,
        ),
        "actionable_games": total,
        "usable_headlines": usable,
        "rejected_or_unusable": rejected,
        "unavailable_or_unmatched": unavailable,
        "coverage_ratio": ratio(
            usable,
            total,
        ),
        "quality_status_counts": quality_status_counts,
    }


# ============================================================================
# REFERENCE MARKET HEALTH
# ============================================================================

def reference_market_health(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)

    available = 0
    multi_book = 0
    book_counts: list[int] = []

    for row in rows:
        display = row.get(
            "reference_market_display"
        )

        books = safe_int(
            row.get(
                "reference_market_books"
            )
        )

        if books is not None:
            book_counts.append(
                books
            )

        if available_text(
            display
        ):
            available += 1

            if (
                books is not None
                and
                books >= 2
            ):
                multi_book += 1

    return {
        "status": coverage_status(
            available,
            total,
            pass_ratio=0.75,
            partial_ratio=0.35,
        ),
        "actionable_games": total,
        "available_games": available,
        "multi_book_games": multi_book,
        "coverage_ratio": ratio(
            available,
            total,
        ),
        "minimum_books": (
            min(
                book_counts
            )
            if book_counts
            else None
        ),
        "maximum_books": (
            max(
                book_counts
            )
            if book_counts
            else None
        ),
    }


# ============================================================================
# BOARD HEALTH
# ============================================================================

def board_health(
    all_rows: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    age = file_age_hours(
        BOARD_JSON
    )

    if age is None:
        status = "FAIL"

    elif age <= 6:
        status = "PASS"

    elif age <= 24:
        status = "PARTIAL"

    else:
        status = "WARNING"

    return {
        "status": status,
        "file": str(
            BOARD_JSON.relative_to(
                PROJECT_ROOT
            )
        ),
        "age_hours": age,
        "rows_total": len(
            all_rows
        ),
        "actionable_games": len(
            active_rows
        ),
    }


# ============================================================================
# SQLITE HELPERS
# ============================================================================

def table_exists(
    connection: sqlite3.Connection,
    table: str,
) -> bool:
    return (
        connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table'
              AND name=?
            LIMIT 1
            """,
            (
                table,
            ),
        ).fetchone()
        is not None
    )


def table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:
    return {
        str(
            row[1]
        )
        for row in connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    }


# ============================================================================
# PREDICTION HISTORY / AUDIT HEALTH
# ============================================================================

def load_audit_summary() -> dict[str, Any] | None:
    if not AUDIT_SUMMARY_JSON.exists():
        return None

    try:
        with AUDIT_SUMMARY_JSON.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            return payload

    except Exception:
        return None

    return None


def history_health() -> dict[str, Any]:
    if not HISTORY_DB.exists():
        return {
            "status": "FAIL",
            "database_exists": False,
            "snapshots": 0,
            "runs": 0,
            "scored_snapshots": 0,
            "audit_verified": 0,
            "audit_total": 0,
            "batch_corroborated": 0,
            "immutability_protection": False,
        }

    connection = sqlite3.connect(
        f"file:{HISTORY_DB}?mode=ro",
        uri=True,
    )

    connection.execute(
        "PRAGMA query_only = ON"
    )

    try:
        snapshots = 0
        runs = 0
        scored = 0
        schema_ok = False

        if table_exists(
            connection,
            "prediction_snapshots",
        ):
            columns = table_columns(
                connection,
                "prediction_snapshots",
            )

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
            }

            schema_ok = (
                required.issubset(
                    columns
                )
            )

            snapshots = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM prediction_snapshots
                    WHERE season = ?
                    """,
                    (
                        SEASON,
                    ),
                ).fetchone()[0]
            )

            runs = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT run_id)
                    FROM prediction_snapshots
                    WHERE season = ?
                    """,
                    (
                        SEASON,
                    ),
                ).fetchone()[0]
            )

        if table_exists(
            connection,
            "prediction_scores",
        ):
            scored = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM prediction_scores
                    """
                ).fetchone()[0]
            )

    finally:
        connection.close()

    audit = load_audit_summary()

    audit_total = 0
    audit_verified = 0
    batch_corroborated = 0
    immutability = False
    verification_status_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}

    if audit:
        # These are the canonical Prediction Audit V1 keys.
        audit_total = (
            safe_int(
                audit.get(
                    "snapshot_count"
                )
            )
            or
            0
        )

        audit_verified = (
            safe_int(
                audit.get(
                    "verified_pre_kickoff_count"
                )
            )
            or
            0
        )

        batch_corroborated = (
            safe_int(
                audit.get(
                    "batch_corroborated_count"
                )
            )
            or
            0
        )

        statuses = audit.get(
            "verification_status_counts"
        )

        if isinstance(
            statuses,
            dict,
        ):
            verification_status_counts = {
                str(
                    key
                ):
                safe_int(
                    value
                )
                or
                0
                for key, value in statuses.items()
            }

        flags = audit.get(
            "flag_counts"
        )

        if isinstance(
            flags,
            dict,
        ):
            flag_counts = {
                str(
                    key
                ):
                safe_int(
                    value
                )
                or
                0
                for key, value in flags.items()
            }

        protection = audit.get(
            "database_protection"
        )

        if isinstance(
            protection,
            dict,
        ):
            immutability = bool(
                protection.get(
                    "protected"
                )
            )

    if (
        snapshots > 0
        and
        schema_ok
        and
        audit_total == snapshots
        and
        audit_verified == snapshots
        and
        batch_corroborated == snapshots
        and
        immutability
        and
        not flag_counts
    ):
        status = "PASS"

    elif (
        snapshots > 0
        and
        schema_ok
    ):
        status = "PARTIAL"

    else:
        status = "WARNING"

    return {
        "status": status,
        "database_exists": True,
        "database_age_hours": file_age_hours(
            HISTORY_DB
        ),
        "schema_ok": schema_ok,
        "snapshots": snapshots,
        "runs": runs,
        "scored_snapshots": scored,
        "audit_file_exists": (
            AUDIT_SUMMARY_JSON.exists()
        ),
        "audit_file_age_hours": file_age_hours(
            AUDIT_SUMMARY_JSON
        ),
        "audit_total": audit_total,
        "audit_verified": audit_verified,
        "batch_corroborated": batch_corroborated,
        "immutability_protection": immutability,
        "verification_status_counts": verification_status_counts,
        "flag_counts": flag_counts,
    }


# ============================================================================
# WAGER LEDGER HEALTH
# ============================================================================

def wager_health() -> dict[str, Any]:
    if not WAGER_DB.exists():
        return {
            "status": "PARTIAL",
            "database_exists": False,
            "wagers": 0,
            "pending": 0,
            "graded": 0,
            "verified_snapshot_links": 0,
            "message": "Ledger has not been initialized yet.",
        }

    connection = sqlite3.connect(
        f"file:{WAGER_DB}?mode=ro",
        uri=True,
    )

    connection.execute(
        "PRAGMA query_only = ON"
    )

    try:
        if not table_exists(
            connection,
            "wagers",
        ):
            return {
                "status": "WARNING",
                "database_exists": True,
                "wagers": 0,
                "pending": 0,
                "graded": 0,
                "verified_snapshot_links": 0,
                "message": "wagers table missing.",
            }

        columns = table_columns(
            connection,
            "wagers",
        )

        required = {
            "wager_id",
            "placed_utc",
            "status",
            "game_key",
            "wager_team",
            "wager_line",
            "american_odds",
            "stake",
            "linked_snapshot_id",
            "snapshot_link_status",
            "result",
            "profit_loss",
            "clv_status",
        }

        schema_ok = required.issubset(
            columns
        )

        total = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM wagers
                """
            ).fetchone()[0]
        )

        pending = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM wagers
                WHERE status = 'PENDING'
                """
            ).fetchone()[0]
        )

        graded = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM wagers
                WHERE status IN (
                    'WON',
                    'LOST',
                    'PUSH'
                )
                """
            ).fetchone()[0]
        )

        verified_links = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM wagers
                WHERE snapshot_link_status =
                      'VERIFIED_PRIOR_SNAPSHOT'
                """
            ).fetchone()[0]
        )

    finally:
        connection.close()

    return {
        "status": (
            "PASS"
            if schema_ok
            else "WARNING"
        ),
        "database_exists": True,
        "database_age_hours": file_age_hours(
            WAGER_DB
        ),
        "schema_ok": schema_ok,
        "wagers": total,
        "pending": pending,
        "graded": graded,
        "verified_snapshot_links": verified_links,
        "message": (
            "Ledger ready."
            if schema_ok
            else "Ledger schema incomplete."
        ),
    }


# ============================================================================
# REPORT BUILDING
# ============================================================================

def build_report() -> dict[str, Any]:
    all_rows = load_board()

    active_rows = actionable_rows(
        all_rows
    )

    board = board_health(
        all_rows,
        active_rows,
    )

    model = model_health(
        active_rows
    )

    features = feature_health(
        active_rows
    )

    hardrock = hardrock_health(
        active_rows
    )

    market = reference_market_health(
        active_rows
    )

    history = history_health()

    wagers = wager_health()

    components = {
        "BOARD": board,
        "MODEL": model,
        "FEATURES": features,
        "HARD_ROCK": hardrock,
        "MARKET": market,
        "HISTORY": history,
        "WAGERS": wagers,
    }

    # HARD_ROCK / MARKET can legitimately be partial early in the week.
    # WAGERS can be empty without being unhealthy.
    critical = [
        board["status"],
        model["status"],
        features["status"],
        history["status"],
    ]

    if "FAIL" in critical:
        overall = "FAIL"

    elif "WARNING" in critical:
        overall = "WARNING"

    else:
        overall = "READY"

    warning_count = sum(
        1
        for row in active_rows
        if safe_text(
            row.get(
                "signal_warning"
            )
        )
    )

    return {
        "health_schema_version": 2,
        "generated_utc": utc_iso(),
        "season": SEASON,
        "overall_status": overall,
        "components": components,
        "live_board_warnings": {
            "games_with_signal_warning": warning_count,
            "actionable_games": len(
                active_rows
            ),
        },
        "notes": [
            (
                "HARD_ROCK and MARKET coverage may be PARTIAL early in the "
                "week without indicating a software failure."
            ),
            (
                "Feature PARTIAL/WARNING describes input completeness; the "
                "model can still produce a prediction because the production "
                "pipeline uses its established imputation rules."
            ),
            (
                "Strength remains prospectively unvalidated and is not a "
                "cover probability."
            ),
            (
                "This health check performs no API calls and modifies no "
                "production data."
            ),
        ],
    }


# ============================================================================
# OUTPUTS
# ============================================================================

def write_outputs(
    report: dict[str, Any],
) -> None:
    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            default=str,
        )
        handle.write(
            "\n"
        )

    csv_rows = []

    for component_name, component in report[
        "components"
    ].items():
        details = {
            key: value
            for key, value in component.items()
            if key != "status"
        }

        csv_rows.append(
            {
                "component": component_name,
                "status": component.get(
                    "status"
                ),
                "details_json": json.dumps(
                    details,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                    default=str,
                ),
            }
        )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component",
                "status",
                "details_json",
            ],
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )


# ============================================================================
# CONSOLE SUMMARY
# ============================================================================

def print_summary(
    report: dict[str, Any],
) -> None:
    print(
        "=" * 104
    )

    print(
        "CFB EDGE ENGINE — DATA HEALTH V2"
    )

    print(
        "=" * 104
    )

    print(
        f"Season:     {report['season']}"
    )

    print(
        f"Generated:  {report['generated_utc']}"
    )

    print(
        "Mode:       READ ONLY / NO API CALLS"
    )

    print(
        f"Overall:    {report['overall_status']}"
    )

    print(
        "=" * 104
    )

    print()

    print(
        "SYSTEM HEALTH"
    )

    print(
        "-" * 104
    )

    components = report[
        "components"
    ]

    board = components[
        "BOARD"
    ]

    print(
        f"{'BOARD':<12} "
        f"{board['status']:<9} "
        f"{board['actionable_games']} actionable | "
        f"age {board['age_hours']}h"
    )

    model = components[
        "MODEL"
    ]

    print(
        f"{'MODEL':<12} "
        f"{model['status']:<9} "
        f"{model['spread_ready']}/{model['actionable_games']} spreads | "
        f"{model['valid_probabilities']} valid probabilities | "
        f"artifact {'YES' if model['model_artifact_exists'] else 'NO'}"
    )

    features = components[
        "FEATURES"
    ]

    coverage_suffix = ""

    if features[
        "coverage_source"
    ] == "nn_feature_coverage":
        coverage_suffix = (
            f" | min "
            f"{features['minimum_feature_coverage']}/"
            f"{features['expected_features']}"
        )

    else:
        coverage_suffix = (
            " | classified from nn_input_status"
        )

    print(
        f"{'FEATURES':<12} "
        f"{features['status']:<9} "
        f"{features['full_games']} full | "
        f"{features['partial_games']} partial | "
        f"{features['heavy_imputation_games']} heavy"
        f"{coverage_suffix}"
    )

    hardrock = components[
        "HARD_ROCK"
    ]

    print(
        f"{'HARD_ROCK':<12} "
        f"{hardrock['status']:<9} "
        f"{hardrock['usable_headlines']}/"
        f"{hardrock['actionable_games']} usable | "
        f"{hardrock['rejected_or_unusable']} rejected/unusable"
    )

    market = components[
        "MARKET"
    ]

    print(
        f"{'MARKET':<12} "
        f"{market['status']:<9} "
        f"{market['available_games']}/"
        f"{market['actionable_games']} available | "
        f"{market['multi_book_games']} multi-book"
    )

    history = components[
        "HISTORY"
    ]

    print(
        f"{'HISTORY':<12} "
        f"{history['status']:<9} "
        f"{history['snapshots']} snapshots | "
        f"{history['audit_verified']}/"
        f"{history['audit_total']} audit verified | "
        f"{history['batch_corroborated']} batch corroborated"
    )

    wagers = components[
        "WAGERS"
    ]

    print(
        f"{'WAGERS':<12} "
        f"{wagers['status']:<9} "
        f"{wagers['wagers']} wagers | "
        f"{wagers['pending']} pending | "
        f"{wagers['graded']} graded"
    )

    print()

    print(
        "FILES WRITTEN"
    )

    print(
        "-" * 104
    )

    print(
        OUTPUT_JSON
    )

    print(
        OUTPUT_CSV
    )

    print()

    print(
        "No API calls were made and no prediction, score, result, wager, "
        "model, or market record was modified."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> int:
    try:
        report = build_report()

        write_outputs(
            report
        )

        print_summary(
            report
        )

        return 0

    except Exception as error:
        print(
            "=" * 104
        )

        print(
            "CFB EDGE ENGINE — DATA HEALTH V2"
        )

        print(
            "=" * 104
        )

        print(
            f"ERROR: {error}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
