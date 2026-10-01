#!/usr/bin/env python3
"""Read-only health report for the NFL research-collection ledger."""

from __future__ import annotations

import argparse
import json
import sys

from nfl_research_paths_v1 import NFL_RESEARCH_DB, PROJECT_ROOT
from nfl_research_store_v1 import IMMUTABLE_TABLES, database_summary


def render_text(summary: dict[str, object]) -> None:
    expected_triggers = len(IMMUTABLE_TABLES) * 2
    print("=" * 100)
    print("NFL RESEARCH COLLECTION — READ-ONLY STATUS")
    print("=" * 100)
    try:
        display_path = NFL_RESEARCH_DB.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = NFL_RESEARCH_DB
    print(f"Database:                 {display_path}")
    print(f"Database initialized:     {'YES' if summary['database_exists'] else 'NO'}")
    print(f"Schedule source snapshots: {summary['schedule_snapshots']}")
    print(f"Schedule game rows:        {summary['schedule_rows']}")
    print(f"Market source snapshots:   {summary['market_snapshots']}")
    print(f"Market bookmaker rows:     {summary['market_quotes']}")
    print(f"Latest schedule capture:   {summary['latest_schedule_capture_utc'] or 'none'}")
    print(f"Latest market capture:     {summary['latest_market_capture_utc'] or 'none'}")
    print(
        "Immutability triggers:    "
        f"{summary['immutability_triggers']}/{expected_triggers} "
        f"({'PASS' if summary['immutability_triggers'] == expected_triggers else 'NOT READY'})"
    )
    print()
    print("This report performs no provider calls and does not modify any record.")
    print("NFL remains collection-only: no model, prediction, signal, or wager output.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show NFL research-collection status without changing data.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the status payload as JSON for a future dashboard or cloud health check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = database_summary(NFL_RESEARCH_DB)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            render_text(summary)
        return 0
    except Exception as exc:
        print("NFL RESEARCH STATUS FAILED", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
