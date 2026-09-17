#!/usr/bin/env python3
"""
CFB Edge Engine — Official Availability Collector V1

Production collector for the confirmed public ACC / Big 12 / SEC availability
report endpoint used by the official conference reporting pages.

Data flow
---------
official public report endpoint
    -> compact immutable source capture
    -> strict current-game match
    -> normalize player / position / status
    -> enrich athlete_id from local player stats when possible
    -> canonical personnel import CSV
    -> personnel_availability_v1.py
    -> immutable personnel_availability.sqlite
    -> rebuilt personnel research outputs

Important design choices
------------------------
- Official reports are evidence, not a direct model adjustment.
- The frozen neural model is NOT changed.
- Report Pending rows are ignored.
- Exempt players are ignored as current-game availability signals.
- Available players are normally ignored, EXCEPT when they resolve a player
  previously stored as a concern. That prevents stale injuries from persisting.
- Repeated identical source/report/player/status observations are deduplicated.
- Source publication time is converted to UTC and must precede kickoff.
- Game matching requires BOTH the two teams and the report game date.
  There is no team-pair-only fallback.
- Big Ten is intentionally excluded from V1 because its public transport differs.

Expected companion file
-----------------------
src/personnel_availability_v1.py

The companion file should include support for:
    OUT_FIRST_HALF

Run
---
python src/collect_official_availability_v1.py

Optional dry run
----------------
python src/collect_official_availability_v1.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import time

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROSPECTIVE_DIR = DATA_DIR / "prospective"

BOARD_JSON = PROCESSED_DIR / "live_model_market_board_2026_v1.json"
CFB_DB = DATA_DIR / "cfb.db"
PERSONNEL_DB = PROSPECTIVE_DIR / "personnel_availability.sqlite"
PERSONNEL_SCRIPT = PROJECT_ROOT / "src" / "personnel_availability_v1.py"

CAPTURE_DIR = PROSPECTIVE_DIR / "official_availability_captures"
LATEST_CAPTURE = PROCESSED_DIR / "official_availability_capture_latest.json"
LATEST_IMPORT_CSV = PROCESSED_DIR / "official_personnel_import_2026_latest.csv"
LATEST_SUMMARY = PROCESSED_DIR / "official_availability_collection_2026_v1.json"

SEASON = 2026
WIDTH = 118
TIMEOUT_SECONDS = 30

ENDPOINT = "https://app.hdintelligence.com/api/get-publish-public"

SOURCES = [
    {
        "key": "ACC",
        "label": "ACC",
        "sport": "Football",
        "organization": "ACC",
        "conference": "ACC",
        "official_url": (
            "https://theacc.com/sports/2025/8/28/"
            "availability-reporting-football.aspx"
        ),
        "referer": (
            "https://app.hdintelligence.com/"
            "?source=ACC&sport=Football&conf=ACC&type=report"
        ),
    },
    {
        "key": "BIG12",
        "label": "Big 12",
        "sport": "Football",
        "organization": "B12",
        "conference": "B12",
        "official_url": (
            "https://big12sports.com/sports/2025/8/14/FBreporting.aspx"
        ),
        "referer": (
            "https://app.hdintelligence.com/"
            "?source=B12&sport=Football&conf=B12&type=report"
        ),
    },
    {
        "key": "SEC",
        "label": "SEC",
        "sport": "Football",
        "organization": "SEC",
        "conference": "SEC",
        "official_url": "https://www.secsports.com/fbreports",
        "referer": (
            "https://app.hdintelligence.com/"
            "?source=SEC&sport=Football&conf=SEC&type=report"
        ),
    },
]

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Origin": "https://app.hdintelligence.com",
}

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

TIMEZONE_MAP = {
    "ET": "America/New_York",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CT": "America/Chicago",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "MT": "America/Denver",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "PT": "America/Los_Angeles",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
}

PLAYER_LABEL_RE = re.compile(
    r"^(?P<position>[A-Za-z0-9./-]+)\s+"
    r"#(?P<number>[A-Za-z0-9-]+)\s+"
    r"(?P<name>.+)$"
)

STATUS_MAP = {
    "available": "AVAILABLE",
    "probable": "PROBABLE",
    "questionable": "QUESTIONABLE",
    "game time decision": "QUESTIONABLE",
    "doubtful": "DOUBTFUL",
    "out": "OUT",
    "out - (1st half)": "OUT_FIRST_HALF",
    "out - 1st half": "OUT_FIRST_HALF",
    "out first half": "OUT_FIRST_HALF",
    "out (1st half)": "OUT_FIRST_HALF",
    "suspended": "SUSPENDED",
}

CONCERN_STATUSES = {
    "PROBABLE",
    "QUESTIONABLE",
    "OUT_FIRST_HALF",
    "DOUBTFUL",
    "OUT",
    "SUSPENDED",
    "SEASON_OUT",
    "UNKNOWN",
}

TEAM_ALIASES = {
    "pitt": "pittsburgh",
    "miami fl": "miami",
    "miami florida": "miami",
    "north carolina tar heels": "north carolina",
    "nc state": "nc state",
    "north carolina state": "nc state",
    "ole miss": "ole miss",
    "mississippi": "ole miss",
    "southern methodist": "smu",
    "texas am": "texas a m",
    "texas a and m": "texas a m",
}

PLAYER_TEAM_ALIASES = dict(TEAM_ALIASES)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def rule(char: str = "=") -> None:
    print(char * WIDTH)


def clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(dt: datetime | None = None) -> str:
    value = dt or utc_now()
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(value: Any) -> datetime | None:
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
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())

    if text in TEAM_ALIASES:
        text = TEAM_ALIASES[text]

    return text


def norm_player(value: Any) -> str:
    text = clean(value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def same_team(left: Any, right: Any) -> bool:
    a = norm_team(left)
    b = norm_team(right)
    return bool(a and b and a == b)


def parse_player_label(value: Any) -> dict[str, str]:
    text = clean(value)

    match = PLAYER_LABEL_RE.match(text)

    if not match:
        return {
            "position": "",
            "number": "",
            "player_name": text,
        }

    return {
        "position": clean(match.group("position")).upper(),
        "number": clean(match.group("number")),
        "player_name": clean(match.group("name")),
    }


def canonical_status(
    raw_status: Any,
    exempt_status: Any,
) -> tuple[str | None, str]:
    raw = clean(raw_status)
    exempt = clean(exempt_status)

    if (
        raw.casefold() == "exempt"
        or
        exempt.casefold() == "exempt"
    ):
        return None, "EXEMPT_SKIPPED"

    normalized = STATUS_MAP.get(
        raw.casefold()
    )

    if normalized is None:
        return "UNKNOWN", "UNRECOGNIZED_OFFICIAL_STATUS"

    return normalized, "SUPPORTED"


def source_published_utc(
    report: dict[str, Any],
) -> datetime | None:
    date_text = clean(
        report.get("publishDate")
    )

    time_text = clean(
        report.get("postedTime")
    )

    zone_text = clean(
        report.get("conferenceTimeZone")
    ).upper()

    if not date_text or not time_text:
        return None

    zone_name = TIMEZONE_MAP.get(
        zone_text
    )

    if not zone_name:
        return None

    try:
        local_dt = datetime.fromisoformat(
            f"{date_text}T{time_text}"
        )
    except ValueError:
        return None

    localized = local_dt.replace(
        tzinfo=ZoneInfo(zone_name)
    )

    return localized.astimezone(
        timezone.utc
    )


def sha256_json(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


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

    rows = normalize_board_records(
        payload
    )

    if not rows:
        raise RuntimeError(
            "Live board contains no identifiable game rows."
        )

    return rows


def board_identity(
    row: dict[str, Any],
) -> dict[str, Any]:
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


def report_team_names(
    report: dict[str, Any],
) -> list[str]:
    games = report.get("games")

    if not isinstance(games, list):
        return []

    output = []

    for team in games:
        if not isinstance(team, dict):
            continue

        name = (
            clean(team.get("teamDisplayName"))
            or
            clean(team.get("teamName"))
        )

        if name:
            output.append(name)

    return output


def report_game_date(
    report: dict[str, Any],
) -> str:
    footer = report.get("footer")

    if isinstance(footer, dict):
        return clean(
            footer.get("date")
        )

    return ""


def game_date_in_zone(
    kickoff_utc: str,
    zone_name: str,
) -> str:
    kickoff = parse_iso(
        kickoff_utc
    )

    if kickoff is None:
        return ""

    return kickoff.astimezone(
        ZoneInfo(zone_name)
    ).date().isoformat()


def resolve_report_to_board(
    report: dict[str, Any],
    board: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Strict match on:
    - exact two-team set after normalization
    - report game date

    There is NO team-pair-only fallback.
    """
    teams = report_team_names(
        report
    )

    if len(teams) != 2:
        return None

    report_date = report_game_date(
        report
    )

    if not report_date:
        return None

    zone_text = clean(
        report.get("conferenceTimeZone")
    ).upper()

    zone_name = TIMEZONE_MAP.get(
        zone_text
    )

    if not zone_name:
        return None

    source_team_set = {
        norm_team(team)
        for team in teams
    }

    candidates = []

    for row in board:
        game = board_identity(
            row
        )

        board_team_set = {
            norm_team(game["away_team"]),
            norm_team(game["home_team"]),
        }

        if board_team_set != source_team_set:
            continue

        board_local_date = game_date_in_zone(
            game["kickoff_utc"],
            zone_name,
        )

        if board_local_date != report_date:
            continue

        candidates.append(game)

    return (
        candidates[0]
        if len(candidates) == 1
        else None
    )


# =============================================================================
# LOCAL PLAYER-ID ENRICHMENT
# =============================================================================

def load_athlete_lookup() -> dict[
    tuple[str, str],
    list[str],
]:
    if not CFB_DB.exists():
        return {}

    connection = sqlite3.connect(
        f"file:{CFB_DB}?mode=ro",
        uri=True,
    )

    connection.execute(
        "PRAGMA query_only = ON"
    )

    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(
            """
            SELECT DISTINCT
                p.team,
                p.athlete_id,
                p.athlete_name
            FROM player_game_stat_values AS p
            INNER JOIN games AS g
                ON g.game_id = p.game_id
            WHERE g.season = ?
            """,
            (
                SEASON,
            ),
        ).fetchall()

    finally:
        connection.close()

    lookup = defaultdict(list)

    for row in rows:
        team = norm_team(
            row["team"]
        )

        player = norm_player(
            row["athlete_name"]
        )

        athlete_id = clean(
            row["athlete_id"]
        )

        if (
            team
            and
            player
            and
            athlete_id
            and
            athlete_id
            not in lookup[
                (
                    team,
                    player,
                )
            ]
        ):
            lookup[
                (
                    team,
                    player,
                )
            ].append(
                athlete_id
            )

    return dict(lookup)


def resolve_athlete_id(
    lookup: dict[
        tuple[str, str],
        list[str],
    ],
    team: str,
    player_name: str,
) -> str:
    candidates = lookup.get(
        (
            norm_team(team),
            norm_player(player_name),
        ),
        [],
    )

    return (
        candidates[0]
        if len(candidates) == 1
        else ""
    )


# =============================================================================
# EXISTING PERSONNEL STATE
# =============================================================================

def existing_personnel_state() -> tuple[
    set[tuple[str, str, str, str, str]],
    dict[tuple[str, str, str], str],
]:
    """
    Returns:
    - exact source/status signatures already stored
    - latest canonical status by game/team/player
    """
    if not PERSONNEL_DB.exists():
        return set(), {}

    connection = sqlite3.connect(
        f"file:{PERSONNEL_DB}?mode=ro",
        uri=True,
    )

    connection.row_factory = sqlite3.Row
    connection.execute(
        "PRAGMA query_only = ON"
    )

    try:
        rows = connection.execute(
            """
            SELECT
                game_id,
                team,
                player_name,
                status,
                source_published_utc,
                captured_utc
            FROM personnel_observations
            WHERE season = ?
              AND prekickoff_valid = 1
            ORDER BY captured_utc, observation_id
            """,
            (
                SEASON,
            ),
        ).fetchall()

    finally:
        connection.close()

    exact = set()
    latest = {}

    for row in rows:
        player_key = (
            clean(row["game_id"]),
            norm_team(row["team"]),
            norm_player(row["player_name"]),
        )

        signature = (
            clean(row["game_id"]),
            norm_team(row["team"]),
            norm_player(row["player_name"]),
            clean(row["source_published_utc"]),
            clean(row["status"]),
        )

        exact.add(signature)

        latest[player_key] = clean(
            row["status"]
        )

    return exact, latest


# =============================================================================
# SOURCE FETCHING / COMPACT CAPTURE
# =============================================================================

def fetch_source(
    session: requests.Session,
    source: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    payload = {
        "sport": source["sport"],
        "organization": source[
            "organization"
        ],
        "conference": source[
            "conference"
        ],
    }

    headers = dict(
        HEADERS_BASE
    )

    headers[
        "Referer"
    ] = source[
        "referer"
    ]

    started = time.perf_counter()

    try:
        response = session.post(
            ENDPOINT,
            json=payload,
            headers=headers,
            timeout=TIMEOUT_SECONDS,
            allow_redirects=True,
        )

    except requests.RequestException as error:
        return None, {
            "conference": source["key"],
            "ok": False,
            "error": str(error),
        }

    elapsed = (
        time.perf_counter()
        -
        started
    )

    metadata = {
        "conference": source["key"],
        "ok": response.ok,
        "http_status": response.status_code,
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
        "bytes": len(
            response.content
        ),
        "content_type": clean(
            response.headers.get(
                "content-type"
            )
        ),
    }

    if not response.ok:
        metadata[
            "error"
        ] = (
            "non-200 response"
        )

        return None, metadata

    try:
        parsed = response.json()

    except ValueError:
        metadata[
            "error"
        ] = (
            "response was not JSON"
        )

        return None, metadata

    if not isinstance(
        parsed,
        dict,
    ):
        metadata[
            "error"
        ] = (
            f"expected dict, got "
            f"{type(parsed).__name__}"
        )

        return None, metadata

    metadata[
        "reports"
    ] = len(parsed)

    return parsed, metadata


def compact_source_report(
    conference: str,
    report_id: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    footer = (
        report.get("footer")
        if isinstance(
            report.get("footer"),
            dict,
        )
        else
        {}
    )

    games = (
        report.get("games")
        if isinstance(
            report.get("games"),
            list,
        )
        else
        []
    )

    teams = []

    for team in games:
        if not isinstance(
            team,
            dict,
        ):
            continue

        rows = (
            team.get("rows")
            if isinstance(
                team.get("rows"),
                list,
            )
            else
            []
        )

        compact_rows = []

        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                continue

            compact_rows.append(
                {
                    "name": clean(
                        row.get("name")
                    ),
                    "status": clean(
                        row.get("status")
                    ),
                    "exemptStatus": clean(
                        row.get(
                            "exemptStatus"
                        )
                    ),
                }
            )

        teams.append(
            {
                "teamName": clean(
                    team.get("teamName")
                ),
                "teamDisplayName": clean(
                    team.get(
                        "teamDisplayName"
                    )
                ),
                "rows": compact_rows,
            }
        )

    compact = {
        "conference": conference,
        "report_id": str(
            report_id
        ),
        "ReportType": clean(
            report.get(
                "ReportType"
            )
        ),
        "publishDate": clean(
            report.get(
                "publishDate"
            )
        ),
        "publishDayOfWeek": clean(
            report.get(
                "publishDayOfWeek"
            )
        ),
        "postedTime": clean(
            report.get(
                "postedTime"
            )
        ),
        "conferenceTimeZone": clean(
            report.get(
                "conferenceTimeZone"
            )
        ),
        "gameNeutral": bool(
            report.get(
                "gameNeutral"
            )
        ),
        "footer": {
            "date": clean(
                footer.get("date")
            ),
            "time": clean(
                footer.get("time")
            ),
            "location": clean(
                footer.get("location")
            ),
        },
        "games": teams,
    }

    compact[
        "payload_sha256"
    ] = sha256_json(
        compact
    )

    return compact


# =============================================================================
# NORMALIZATION TO PERSONNEL IMPORT
# =============================================================================

def canonical_team_from_game(
    source_team: str,
    game: dict[str, Any],
) -> str | None:
    if same_team(
        source_team,
        game["away_team"],
    ):
        return game[
            "away_team"
        ]

    if same_team(
        source_team,
        game["home_team"],
    ):
        return game[
            "home_team"
        ]

    return None


def opponent_for(
    team: str,
    game: dict[str, Any],
) -> str:
    if same_team(
        team,
        game["away_team"],
    ):
        return game[
            "home_team"
        ]

    return game[
        "away_team"
    ]


def normalize_report_rows(
    *,
    source: dict[str, str],
    report_id: str,
    report: dict[str, Any],
    board: list[dict[str, Any]],
    athlete_lookup: dict[
        tuple[str, str],
        list[str],
    ],
    existing_exact: set[
        tuple[
            str,
            str,
            str,
            str,
            str,
        ]
    ],
    latest_status: dict[
        tuple[
            str,
            str,
            str,
        ],
        str,
    ],
    captured_utc: datetime,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    result = {
        "report_id": str(
            report_id
        ),
        "report_type": clean(
            report.get(
                "ReportType"
            )
        ),
        "matched": False,
        "reason": "",
        "rows_seen": 0,
        "rows_new": 0,
        "concerns_new": 0,
        "available_resolutions_new": 0,
        "exempt_skipped": 0,
        "available_untracked_skipped": 0,
        "duplicates_skipped": 0,
        "unknown_status_rows": 0,
    }

    if clean(
        report.get(
            "ReportType"
        )
    ).casefold() == "report pending":
        result[
            "reason"
        ] = "REPORT_PENDING"

        return [], result

    games = report.get(
        "games"
    )

    if not isinstance(
        games,
        list,
    ):
        result[
            "reason"
        ] = "NO_GAMES_ARRAY"

        return [], result

    if not any(
        isinstance(team, dict)
        and
        isinstance(
            team.get("rows"),
            list,
        )
        and
        len(
            team.get("rows")
        )
        >
        0
        for team in games
    ):
        result[
            "reason"
        ] = "NO_PLAYER_ROWS"

        return [], result

    game = resolve_report_to_board(
        report,
        board,
    )

    if game is None:
        result[
            "reason"
        ] = "NO_STRICT_BOARD_MATCH"

        return [], result

    result[
        "matched"
    ] = True

    published_utc = source_published_utc(
        report
    )

    if published_utc is None:
        result[
            "reason"
        ] = "INVALID_SOURCE_PUBLICATION_TIME"

        return [], result

    if published_utc > captured_utc:
        result[
            "reason"
        ] = "SOURCE_TIME_IN_FUTURE"

        return [], result

    kickoff = parse_iso(
        game["kickoff_utc"]
    )

    if kickoff is None:
        result[
            "reason"
        ] = "INVALID_BOARD_KICKOFF"

        return [], result

    if published_utc >= kickoff:
        result[
            "reason"
        ] = "SOURCE_NOT_PREKICKOFF"

        return [], result

    output = []

    source_name = (
        f"{source['label']} Official Availability "
        f"Report {report_id}"
    )

    for team_payload in games:
        if not isinstance(
            team_payload,
            dict,
        ):
            continue

        source_team = (
            clean(
                team_payload.get(
                    "teamDisplayName"
                )
            )
            or
            clean(
                team_payload.get(
                    "teamName"
                )
            )
        )

        canonical_team = canonical_team_from_game(
            source_team,
            game,
        )

        if canonical_team is None:
            continue

        opponent = opponent_for(
            canonical_team,
            game,
        )

        rows = (
            team_payload.get("rows")
            if isinstance(
                team_payload.get(
                    "rows"
                ),
                list,
            )
            else
            []
        )

        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                continue

            result[
                "rows_seen"
            ] += 1

            parsed_player = parse_player_label(
                row.get("name")
            )

            player_name = parsed_player[
                "player_name"
            ]

            if not player_name:
                continue

            canonical, status_note = canonical_status(
                row.get("status"),
                row.get(
                    "exemptStatus"
                ),
            )

            if canonical is None:
                result[
                    "exempt_skipped"
                ] += 1

                continue

            if status_note == "UNRECOGNIZED_OFFICIAL_STATUS":
                result[
                    "unknown_status_rows"
                ] += 1

            player_key = (
                game["game_id"],
                norm_team(
                    canonical_team
                ),
                norm_player(
                    player_name
                ),
            )

            previous_status = latest_status.get(
                player_key
            )

            # We do not need to store the entire healthy roster. But if this
            # player was previously a concern, AVAILABLE is an important state
            # transition and must be inserted so stale injuries are cleared.
            if canonical == "AVAILABLE":
                if (
                    previous_status is None
                    or
                    previous_status == "AVAILABLE"
                ):
                    result[
                        "available_untracked_skipped"
                    ] += 1

                    continue

                transition_kind = "AVAILABLE_RESOLUTION"

            else:
                transition_kind = "CONCERN"

            publication_text = utc_iso(
                published_utc
            )

            signature = (
                game["game_id"],
                norm_team(
                    canonical_team
                ),
                norm_player(
                    player_name
                ),
                publication_text,
                canonical,
            )

            if signature in existing_exact:
                result[
                    "duplicates_skipped"
                ] += 1

                continue

            athlete_id = resolve_athlete_id(
                athlete_lookup,
                canonical_team,
                player_name,
            )

            raw_status = clean(
                row.get("status")
            )

            jersey = parsed_player[
                "number"
            ]

            detail_parts = [
                f"Official status: {raw_status or 'UNKNOWN'}",
                f"Report type: {clean(report.get('ReportType')) or 'UNKNOWN'}",
                f"Report ID: {report_id}",
            ]

            if jersey:
                detail_parts.append(
                    f"Jersey: #{jersey}"
                )

            exempt_status = clean(
                row.get(
                    "exemptStatus"
                )
            )

            if exempt_status:
                detail_parts.append(
                    f"Exempt status: {exempt_status}"
                )

            output.append(
                {
                    "captured_utc": utc_iso(
                        captured_utc
                    ),
                    "source_published_utc": publication_text,
                    "source_name": source_name,
                    "source_type": "official_conference",
                    "source_url": source[
                        "official_url"
                    ],
                    "season": game[
                        "season"
                    ],
                    "week": game[
                        "week"
                    ],
                    "game_id": game[
                        "game_id"
                    ],
                    "kickoff_utc": game[
                        "kickoff_utc"
                    ],
                    "team": canonical_team,
                    "opponent": opponent,
                    "player_name": player_name,
                    "athlete_id": athlete_id,
                    "position": parsed_player[
                        "position"
                    ],
                    "status": canonical,
                    "detail": " | ".join(
                        detail_parts
                    ),
                    "is_starter": "",
                    "manual_importance_override": "",
                    "notes": (
                        "Official-source automatic collection; "
                        f"transition={transition_kind}"
                    ),
                }
            )

            result[
                "rows_new"
            ] += 1

            if canonical == "AVAILABLE":
                result[
                    "available_resolutions_new"
                ] += 1

            else:
                result[
                    "concerns_new"
                ] += 1

    result[
        "reason"
    ] = (
        "MATCHED"
        if result[
            "rows_new"
        ]
        >
        0
        else
        "MATCHED_NO_NEW_ROWS"
    )

    return output, result


# =============================================================================
# OUTPUT / BACKEND HANDOFF
# =============================================================================

def write_import_csv(
    rows: list[dict[str, Any]],
) -> None:
    LATEST_IMPORT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LATEST_IMPORT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=IMPORT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_capture(
    *,
    run_id: str,
    captured_utc: datetime,
    source_metadata: list[
        dict[str, Any]
    ],
    compact_reports: list[
        dict[str, Any]
    ],
) -> Path:
    CAPTURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "captured_utc": utc_iso(
            captured_utc
        ),
        "endpoint": ENDPOINT,
        "source_metadata": source_metadata,
        "reports": compact_reports,
    }

    payload[
        "capture_sha256"
    ] = sha256_json(
        payload
    )

    path = (
        CAPTURE_DIR
        /
        f"official_availability_{run_id}.json"
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    LATEST_CAPTURE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LATEST_CAPTURE.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    return path


def run_personnel_backend() -> int:
    if not PERSONNEL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Personnel backend not found: {PERSONNEL_SCRIPT}"
        )

    command = [
        sys.executable,
        str(
            PERSONNEL_SCRIPT
        ),
        "--import-csv",
        str(
            LATEST_IMPORT_CSV
        ),
        "--build",
    ]

    print()
    rule()
    print(
        "PERSONNEL BACKEND"
    )
    rule()

    print(
        "Running canonical immutable import + research rebuild..."
    )

    completed = subprocess.run(
        command,
        cwd=str(
            PROJECT_ROOT
        ),
        check=False,
    )

    return int(
        completed.returncode
    )


def write_summary(
    payload: dict[str, Any],
) -> None:
    LATEST_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LATEST_SUMMARY.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect official ACC / Big 12 / SEC "
            "football availability reports."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fetch, match, and summarize only. "
            "Do not write captures/import files or call personnel backend."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rule()
    print(
        "CFB EDGE ENGINE — OFFICIAL AVAILABILITY COLLECTOR V1"
    )
    rule()

    print(
        "Sources: ACC / Big 12 / SEC official public reports"
    )

    print(
        "Model impact: NONE — prospective research context only"
    )

    print(
        "Big Ten: excluded from V1; different public transport"
    )

    print(
        f"Dry run: {args.dry_run}"
    )

    rule()

    if not BOARD_JSON.exists():
        raise FileNotFoundError(
            f"Live board missing: {BOARD_JSON}"
        )

    if not CFB_DB.exists():
        raise FileNotFoundError(
            f"CFB database missing: {CFB_DB}"
        )

    board = load_board()

    print(
        f"Live board games: {len(board)}"
    )

    athlete_lookup = load_athlete_lookup()

    print(
        f"Local player-name keys: {len(athlete_lookup):,}"
    )

    existing_exact, latest_status = (
        existing_personnel_state()
    )

    print(
        f"Existing exact personnel signatures: {len(existing_exact):,}"
    )

    print(
        f"Previously tracked players: {len(latest_status):,}"
    )

    captured = utc_now()

    run_id = (
        "official_"
        +
        captured.strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    session = requests.Session()

    source_metadata = []
    compact_reports = []
    import_rows = []
    report_results = []

    successful_sources = 0

    for source in SOURCES:
        print()
        rule("-")

        print(
            f"{source['label']} FETCH"
        )

        rule("-")

        parsed, metadata = fetch_source(
            session,
            source,
        )

        source_metadata.append(
            metadata
        )

        if parsed is None:
            print(
                f"FAILED: {metadata.get('error')}"
            )

            continue

        successful_sources += 1

        print(
            f"HTTP {metadata['http_status']} | "
            f"{metadata['reports']} reports | "
            f"{metadata['bytes']:,} bytes | "
            f"{metadata['elapsed_seconds']:.2f}s"
        )

        for report_id, report in parsed.items():
            if not isinstance(
                report,
                dict,
            ):
                continue

            compact_reports.append(
                compact_source_report(
                    source["key"],
                    str(report_id),
                    report,
                )
            )

            rows, result = normalize_report_rows(
                source=source,
                report_id=str(
                    report_id
                ),
                report=report,
                board=board,
                athlete_lookup=athlete_lookup,
                existing_exact=existing_exact,
                latest_status=latest_status,
                captured_utc=captured,
            )

            result[
                "conference"
            ] = source[
                "key"
            ]

            result[
                "teams"
            ] = report_team_names(
                report
            )

            result[
                "game_date"
            ] = report_game_date(
                report
            )

            report_results.append(
                result
            )

            import_rows.extend(
                rows
            )

            print(
                f"Report {report_id} | "
                f"{result['report_type'] or 'UNKNOWN'} | "
                f"{'MATCH' if result['matched'] else 'SKIP'} | "
                f"{result['reason']} | "
                f"new={result['rows_new']}"
            )

    # Within a single run, protect against accidental duplicate rows from
    # overlapping report objects.
    deduped_rows = []
    seen_rows = set()

    for row in import_rows:
        key = (
            clean(row.get("game_id")),
            norm_team(row.get("team")),
            norm_player(
                row.get("player_name")
            ),
            clean(
                row.get(
                    "source_published_utc"
                )
            ),
            clean(row.get("status")),
        )

        if key in seen_rows:
            continue

        seen_rows.add(key)
        deduped_rows.append(row)

    import_rows = deduped_rows

    status_counts = Counter(
        clean(
            row.get("status")
        )
        for row in import_rows
    )

    matched_reports = sum(
        1
        for item in report_results
        if item.get("matched")
    )

    pending_reports = sum(
        1
        for item in report_results
        if item.get("reason")
        ==
        "REPORT_PENDING"
    )

    unmatched_reports = sum(
        1
        for item in report_results
        if item.get("reason")
        ==
        "NO_STRICT_BOARD_MATCH"
    )

    athlete_matched = sum(
        1
        for row in import_rows
        if clean(
            row.get("athlete_id")
        )
    )

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "captured_utc": utc_iso(
            captured
        ),
        "dry_run": bool(
            args.dry_run
        ),
        "successful_sources": successful_sources,
        "sources_attempted": len(
            SOURCES
        ),
        "reports_seen": len(
            report_results
        ),
        "reports_matched": matched_reports,
        "reports_pending": pending_reports,
        "reports_unmatched": unmatched_reports,
        "rows_to_import": len(
            import_rows
        ),
        "athlete_id_matches": athlete_matched,
        "status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "report_results": report_results,
        "backend_returncode": None,
        "capture_path": None,
        "import_csv": str(
            LATEST_IMPORT_CSV
        ),
        "personnel_model_usage": (
            "RESEARCH_ONLY_NOT_IN_FROZEN_NEURAL_MODEL"
        ),
    }

    print()
    rule()
    print(
        "COLLECTION SUMMARY"
    )
    rule()

    print(
        f"Sources successful:       "
        f"{successful_sources}/{len(SOURCES)}"
    )

    print(
        f"Reports seen:             {len(report_results)}"
    )

    print(
        f"Reports strict-matched:   {matched_reports}"
    )

    print(
        f"Reports pending:          {pending_reports}"
    )

    print(
        f"Reports unmatched:        {unmatched_reports}"
    )

    print(
        f"New personnel rows:       {len(import_rows)}"
    )

    print(
        f"Athlete IDs matched:      "
        f"{athlete_matched}/{len(import_rows)}"
    )

    print(
        "New status counts:       "
        +
        json.dumps(
            dict(
                sorted(
                    status_counts.items()
                )
            ),
            sort_keys=True,
        )
    )

    if args.dry_run:
        print()
        print(
            "DRY RUN: no files or databases were changed."
        )

        return 0

    capture_path = write_capture(
        run_id=run_id,
        captured_utc=captured,
        source_metadata=source_metadata,
        compact_reports=compact_reports,
    )

    summary[
        "capture_path"
    ] = str(
        capture_path
    )

    write_import_csv(
        import_rows
    )

    print()
    print(
        f"Immutable compact capture: {capture_path}"
    )

    print(
        f"Canonical import CSV:      {LATEST_IMPORT_CSV}"
    )

    backend_rc = run_personnel_backend()

    summary[
        "backend_returncode"
    ] = backend_rc

    write_summary(
        summary
    )

    print()
    rule()

    if backend_rc == 0:
        print(
            "OFFICIAL AVAILABILITY COLLECTION: PASS"
        )
    else:
        print(
            "OFFICIAL AVAILABILITY COLLECTION: BACKEND FAILED"
        )

    rule()

    print(
        f"Summary: {LATEST_SUMMARY}"
    )

    print(
        "Frozen neural model: UNCHANGED"
    )

    return backend_rc


if __name__ == "__main__":
    raise SystemExit(main())
