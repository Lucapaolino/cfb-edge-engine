#!/usr/bin/env python3
"""Print a read-only schedule-to-market linkage audit for NFL research."""

from __future__ import annotations

import argparse
import json
import sys

from nfl_research_linkage_v1 import build_linkage_summary
from nfl_research_paths_v1 import NFL_RESEARCH_DB, PROJECT_ROOT


def display_path() -> str:
    try:
        return str(NFL_RESEARCH_DB.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(NFL_RESEARCH_DB)


def render_text(summary: dict[str, object]) -> None:
    print("=" * 100)
    print("NFL RESEARCH — SCHEDULE / MARKET LINKAGE AUDIT")
    print("=" * 100)
    print(f"Database:                  {display_path()}")
    print(f"Audit status:              {summary['linkage_status']}")
    print(f"Detail:                    {summary['message']}")
    print(f"Latest schedule capture:   {summary['latest_schedule_capture_utc'] or 'none'}")
    print(f"Latest market capture:     {summary['latest_market_capture_utc'] or 'none'}")
    print(f"Latest schedule games:     {summary['schedule_game_count']}")
    print(f"Latest market events:      {summary['market_event_count']}")
    print(f"Latest bookmaker rows:     {summary['market_bookmaker_rows']}")
    print(f"Distinct bookmakers:       {summary['market_bookmaker_count']}")
    print(
        "Unambiguous game links:   "
        f"{summary['linked_event_count']}/{summary['market_event_count']}"
    )
    print(f"Unmatched events:          {summary['unmatched_event_count']}")
    print(f"Ambiguous events:          {summary['ambiguous_event_count']}")

    examples = list(summary["unmatched_event_examples"]) + list(
        summary["ambiguous_event_examples"]
    )
    if examples:
        print()
        print("REVIEW EXAMPLES")
        print("-" * 100)
        for item in examples:
            print(
                f"{item['status'].upper():<10} {item['event_label']} "
                f"({item['reason']})"
            )
            matchup_games = item.get("same_matchup_schedule_games") or []
            if matchup_games:
                rendered_games = ", ".join(
                    f"{game.get('gameday') or 'unknown date'} "
                    f"[{game.get('game_id') or 'unknown game'}]"
                    for game in matchup_games
                )
                print(f"           Schedule matchup: {rendered_games}")
            else:
                print("           Schedule matchup: none in the latest source snapshot")

    print()
    print("This audit makes no provider calls and modifies no NFL or CFB record.")
    print("It creates no NFL model, prediction, signal, pick, wager, or player props.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit latest NFL schedule-to-market links without changing data.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the audit payload as JSON for a future dashboard or cloud health check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_linkage_summary(NFL_RESEARCH_DB)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            render_text(summary)
        return 0
    except Exception as exc:
        print("NFL RESEARCH LINKAGE AUDIT FAILED", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
