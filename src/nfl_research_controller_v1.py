#!/usr/bin/env python3
"""One-command controller for separate NFL research collection.

Unlike the CFB game-day controller, this runner has no model stage.  It only
invokes explicit schedule/result and reference-market collection commands plus
a read-only schedule-to-market linkage audit. Nothing runs live unless the
corresponding ``--capture-*`` flag is supplied.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from nfl_research_paths_v1 import PROJECT_ROOT, SRC_DIR


SCRIPTS = {
    "schedule": SRC_DIR / "nfl_research_schedule_results_v1.py",
    "market": SRC_DIR / "nfl_research_market_capture_v1.py",
    "status": SRC_DIR / "nfl_research_status_v1.py",
    "linkage_audit": SRC_DIR / "nfl_research_linkage_audit_v1.py",
}


def print_rule(char: str = "=", width: int = 100) -> None:
    print(char * width, flush=True)


def run_script(name: str, extra_args: list[str]) -> None:
    path = SCRIPTS[name]
    print_rule()
    print(f"NFL RESEARCH STAGE: {name.upper()}", flush=True)
    print_rule()
    command = [sys.executable, str(path), *extra_args]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{path.name} exited with code {completed.returncode}.")


def run_dry(season: int) -> None:
    print_rule()
    print("NFL RESEARCH COLLECTION — SYSTEM TEST", flush=True)
    print_rule()
    print("This mode makes no provider calls and writes no NFL records.", flush=True)
    print("CFB model, data, history, GUI, and wager ledger are not touched.", flush=True)
    print_rule()
    run_script("schedule", ["--dry-run", "--season", str(season)])
    run_script("market", ["--dry-run"])
    run_script("status", [])
    print_rule()
    print("NFL RESEARCH SYSTEM TEST COMPLETE", flush=True)


def run_selected(args: argparse.Namespace) -> None:
    capture_requested = args.capture_schedule or args.capture_market

    print_rule()
    if capture_requested:
        print("NFL RESEARCH COLLECTION — EXPLICIT CAPTURE MODE", flush=True)
    else:
        print("NFL RESEARCH COLLECTION — READ-ONLY AUDIT MODE", flush=True)
    print_rule()
    if capture_requested:
        print("Scope: source schedules/results and reference-market snapshots only.", flush=True)
        print("No NFL model, picks, signals, or wager records will be created.", flush=True)
        print("No CFB file is called by this controller.", flush=True)
    else:
        print("Scope: latest NFL schedule-to-market linkage evidence only.", flush=True)
        print("No provider calls, writes, model, picks, signals, or wager records.", flush=True)
        print("No CFB file is called by this controller.", flush=True)
    print_rule()

    if args.capture_schedule:
        run_script("schedule", ["--capture", "--season", str(args.season)])
    if args.capture_market:
        market_args = ["--capture", "--regions", args.regions]
        if args.bookmakers:
            market_args.extend(["--bookmakers", args.bookmakers])
        run_script("market", market_args)
    if args.linkage_audit:
        run_script("linkage_audit", [])
    run_script("status", [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run explicitly selected NFL research collection stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run no-network/no-write readiness checks (the default with no capture flags).",
    )
    parser.add_argument(
        "--capture-schedule",
        action="store_true",
        help="Capture one immutable schedule/result source snapshot.",
    )
    parser.add_argument(
        "--capture-market",
        action="store_true",
        help="Capture one immutable NFL h2h/spread/total market snapshot.",
    )
    parser.add_argument(
        "--linkage-audit",
        action="store_true",
        help="Read-only audit of latest NFL schedule-to-market links; no provider calls or writes.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season used by --capture-schedule (default: 2026).",
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="The Odds API region used by --capture-market (default: us).",
    )
    parser.add_argument(
        "--bookmakers",
        default="",
        help="Optional comma-separated The Odds API bookmaker keys for --capture-market.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_requested = args.capture_schedule or args.capture_market
    selected_action = capture_requested or args.linkage_audit
    if args.dry_run and selected_action:
        print(
            "ERROR: --dry-run cannot be combined with a --capture-* flag or --linkage-audit.",
            file=sys.stderr,
        )
        return 2

    try:
        if selected_action:
            run_selected(args)
        else:
            run_dry(args.season)
        return 0
    except KeyboardInterrupt:
        print("\nNFL RESEARCH CONTROLLER CANCELLED", flush=True)
        return 130
    except Exception as exc:
        print("\nNFL RESEARCH CONTROLLER FAILED", flush=True)
        print(f"ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
