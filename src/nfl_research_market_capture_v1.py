#!/usr/bin/env python3
"""Capture append-only NFL reference-market observations.

This is a collection-only workflow.  It deliberately does not import CFB
modules, read or write CFB data, produce a prediction, calculate a signal,
or create a wager record.  A live API request requires ``--capture``; the
default command is a no-write, no-network system check.

Examples
--------
    python src/nfl_research_market_capture_v1.py
    python src/nfl_research_market_capture_v1.py --capture
    python src/nfl_research_market_capture_v1.py --capture --bookmakers hardrockbet_fl,draftkings,fanduel
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from nfl_research_paths_v1 import (
    ENV_FILE,
    NFL_RAW_MARKET_DIR,
    NFL_RESEARCH_DB,
    PROJECT_ROOT,
)
from nfl_research_store_v1 import (
    canonical_json,
    connect_writable,
    insert_market_snapshot,
    sha256_json,
)


API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
SPORT_KEY = "americanfootball_nfl"
MARKETS = ("h2h", "spreads", "totals")
REQUEST_TIMEOUT_SECONDS = 45


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_snapshot_id(captured_at_utc: str) -> str:
    stamp = captured_at_utc.replace("+00:00", "Z").replace("-", "").replace(":", "")
    return f"nfl_market_{stamp}_{uuid.uuid4().hex[:10]}"


def read_secret(name: str) -> str | None:
    """Read a single credential without exposing it to stdout or storage."""
    value = os.environ.get(name)
    if value:
        return value

    if not ENV_FILE.exists():
        return None

    try:
        for raw_line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, candidate = line.split("=", 1)
            if key.strip() != name:
                continue
            candidate = candidate.strip().strip("\"'")
            return candidate or None
    except OSError:
        return None

    return None


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_outcome(outcomes: list[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    for outcome in outcomes:
        if isinstance(outcome, Mapping) and str(outcome.get("name") or "") == name:
            return outcome
    return None


def market_by_key(bookmaker: Mapping[str, Any], market_key: str) -> Mapping[str, Any] | None:
    markets = bookmaker.get("markets")
    if not isinstance(markets, list):
        return None
    for market in markets:
        if isinstance(market, Mapping) and market.get("key") == market_key:
            return market
    return None


def normalize_quotes(snapshot_id: str, payload: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create one normalized row per event/bookmaker without changing raw data."""
    rows: list[dict[str, Any]] = []

    for event in payload:
        event_id = str(event.get("id") or "").strip()
        home_team = str(event.get("home_team") or "").strip()
        away_team = str(event.get("away_team") or "").strip()
        bookmakers = event.get("bookmakers")

        if not event_id or not isinstance(bookmakers, list):
            continue

        for bookmaker in bookmakers:
            if not isinstance(bookmaker, Mapping):
                continue

            bookmaker_key = str(bookmaker.get("key") or "").strip()
            if not bookmaker_key:
                continue

            h2h = market_by_key(bookmaker, "h2h")
            spreads = market_by_key(bookmaker, "spreads")
            totals = market_by_key(bookmaker, "totals")

            h2h_outcomes = h2h.get("outcomes", []) if h2h else []
            spread_outcomes = spreads.get("outcomes", []) if spreads else []
            total_outcomes = totals.get("outcomes", []) if totals else []
            h2h_outcomes = h2h_outcomes if isinstance(h2h_outcomes, list) else []
            spread_outcomes = spread_outcomes if isinstance(spread_outcomes, list) else []
            total_outcomes = total_outcomes if isinstance(total_outcomes, list) else []

            home_h2h = find_outcome(h2h_outcomes, home_team)
            away_h2h = find_outcome(h2h_outcomes, away_team)
            home_spread = find_outcome(spread_outcomes, home_team)
            away_spread = find_outcome(spread_outcomes, away_team)
            over = find_outcome(total_outcomes, "Over")
            under = find_outcome(total_outcomes, "Under")

            total_point = None
            if over:
                total_point = as_float(over.get("point"))
            if total_point is None and under:
                total_point = as_float(under.get("point"))

            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "odds_api_event_id": event_id,
                    "commence_time_utc": event.get("commence_time"),
                    "home_team": home_team or None,
                    "away_team": away_team or None,
                    "bookmaker_key": bookmaker_key,
                    "bookmaker_title": bookmaker.get("title"),
                    "bookmaker_last_update_utc": bookmaker.get("last_update"),
                    "h2h_last_update_utc": h2h.get("last_update") if h2h else None,
                    "spread_last_update_utc": spreads.get("last_update") if spreads else None,
                    "total_last_update_utc": totals.get("last_update") if totals else None,
                    "home_moneyline": as_float(home_h2h.get("price")) if home_h2h else None,
                    "away_moneyline": as_float(away_h2h.get("price")) if away_h2h else None,
                    "home_spread": as_float(home_spread.get("point")) if home_spread else None,
                    "home_spread_price": as_float(home_spread.get("price")) if home_spread else None,
                    "away_spread": as_float(away_spread.get("point")) if away_spread else None,
                    "away_spread_price": as_float(away_spread.get("price")) if away_spread else None,
                    "total": total_point,
                    "over_price": as_float(over.get("price")) if over else None,
                    "under_price": as_float(under.get("price")) if under else None,
                    "raw_bookmaker_json": canonical_json(bookmaker),
                }
            )

    return rows


def request_params(args: argparse.Namespace) -> dict[str, str]:
    params = {
        "markets": ",".join(MARKETS),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    bookmakers = str(args.bookmakers or "").strip()
    if bookmakers:
        params["bookmakers"] = bookmakers
    else:
        params["regions"] = args.regions
    return params


def write_raw_snapshot(snapshot_id: str, envelope: Mapping[str, Any]) -> Path:
    NFL_RAW_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    path = NFL_RAW_MARKET_DIR / f"{snapshot_id}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(envelope, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def print_plan(args: argparse.Namespace) -> None:
    mode = "CAPTURE" if args.capture else "DRY RUN / NO API CALLS OR WRITES"
    params = request_params(args)
    print("=" * 100)
    print("NFL RESEARCH COLLECTION — REFERENCE MARKET V1")
    print("=" * 100)
    print(f"Mode:          {mode}")
    print(f"Sport key:     {SPORT_KEY}")
    print(f"Markets:       {params['markets']}")
    print(f"Source scope:  {params.get('bookmakers') or params.get('regions')}")
    print(f"Database:      {NFL_RESEARCH_DB.relative_to(PROJECT_ROOT)}")
    print("Scope:         raw and normalized market observations only")
    print("Guardrails:    no CFB reads/writes; no model; no picks; no wager records")
    print()


def run_dry(args: argparse.Namespace) -> int:
    print_plan(args)
    print("API KEY PRESENCE")
    print("-" * 100)
    print(f"{'PASS' if read_secret('THE_ODDS_API_KEY') else 'MISSING':<8} THE_ODDS_API_KEY")
    print()
    print("SYSTEM TEST COMPLETE")
    print("No API calls were made. No NFL database or raw files were written.")
    return 0


def run_capture(args: argparse.Namespace) -> int:
    print_plan(args)
    api_key = read_secret("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("THE_ODDS_API_KEY is required for --capture but was not found.")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required; install requirements.txt first.") from exc

    params = request_params(args)
    requested_at_utc = utc_iso()
    response = requests.get(
        API_URL,
        params={**params, "apiKey": api_key},
        timeout=args.timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
        raise RuntimeError("The Odds API response was not the expected list of event objects.")

    captured_at_utc = utc_iso()
    snapshot_id = make_snapshot_id(captured_at_utc)
    request_record = {
        "url": API_URL,
        "params": params,
        "requested_at_utc": requested_at_utc,
    }
    raw_envelope = {
        "schema": "nfl_research_market_capture_v1",
        "snapshot_id": snapshot_id,
        "captured_at_utc": captured_at_utc,
        "source": "the_odds_api_v4",
        "sport_key": SPORT_KEY,
        "request": request_record,
        "response_headers": {
            "x-requests-remaining": response.headers.get("x-requests-remaining"),
            "x-requests-used": response.headers.get("x-requests-used"),
            "x-requests-last": response.headers.get("x-requests-last"),
        },
        "events": payload,
    }
    raw_path = write_raw_snapshot(snapshot_id, raw_envelope)
    quote_rows = normalize_quotes(snapshot_id, payload)

    snapshot = {
        "snapshot_id": snapshot_id,
        "captured_at_utc": captured_at_utc,
        "source": "the_odds_api_v4",
        "sport_key": SPORT_KEY,
        "request_json": canonical_json(request_record),
        "response_sha256": sha256_json(payload),
        "raw_file": str(raw_path.relative_to(PROJECT_ROOT)),
        "response_json": canonical_json(payload),
        "requests_remaining": response.headers.get("x-requests-remaining"),
        "requests_used": response.headers.get("x-requests-used"),
        "requests_last": response.headers.get("x-requests-last"),
        "created_at_utc": utc_iso(),
    }

    connection = connect_writable(NFL_RESEARCH_DB)
    try:
        insert_market_snapshot(connection, snapshot, quote_rows)
    finally:
        connection.close()

    print("CAPTURE COMPLETE")
    print("-" * 100)
    print(f"Snapshot ID:   {snapshot_id}")
    print(f"Events:        {len(payload)}")
    print(f"Bookmaker rows:{len(quote_rows)}")
    print(f"Raw file:      {raw_path.relative_to(PROJECT_ROOT)}")
    print(f"Database:      {NFL_RESEARCH_DB.relative_to(PROJECT_ROOT)}")
    print(f"Quota remain:  {response.headers.get('x-requests-remaining') or 'not returned'}")
    print()
    print("No CFB file was read or changed. No NFL prediction, signal, or wager was created.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture immutable NFL reference-market research snapshots.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--capture",
        action="store_true",
        help="Make one live The Odds API request and append a research snapshot.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Show readiness only; this is also the default behavior.",
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="The Odds API region when --bookmakers is not supplied (default: us).",
    )
    parser.add_argument(
        "--bookmakers",
        default="",
        help="Optional comma-separated bookmaker keys; takes precedence over --regions.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {REQUEST_TIMEOUT_SECONDS}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_capture(args) if args.capture else run_dry(args)
    except KeyboardInterrupt:
        print("\nNFL MARKET CAPTURE CANCELLED", flush=True)
        return 130
    except Exception as exc:
        print("\nNFL MARKET CAPTURE FAILED", flush=True)
        print(f"ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
