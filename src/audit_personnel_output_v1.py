#!/usr/bin/env python3
"""
CFB Edge Engine — Personnel Output Audit V1

Read-only quality audit for the current personnel availability layer.

Purpose
-------
Before personnel context is connected to RUN GAME DAY, Strength, or any
decision logic, inspect what the current 12-game official-report layer is
actually producing.

This audit:
- reads data/processed/personnel_availability_2026_v1.json
- does NOT call APIs
- does NOT modify databases
- does NOT modify the neural model
- ranks games by absolute personnel-loss differential
- shows QB uncertainty
- shows top player impacts
- shows how much of the layer is based on local production matching vs
  ROLE_FALLBACK
- flags teams where fallback assumptions dominate the personnel score

Important
---------
personnel_loss_index is a research/context index only.
It is NOT points, cover probability, or an automatic bet signal.

Run
---
python src/audit_personnel_output_v1.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "personnel_availability_2026_v1.json"
)

WIDTH = 118
TOP_PLAYERS_PER_TEAM = 6
FALLBACK_HEAVY_THRESHOLD = 0.50


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def bool_text(value: Any) -> str:
    return "YES" if bool(value) else "NO"


def load_report() -> dict[str, Any]:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Personnel output not found: {INPUT_JSON}"
        )

    payload = json.loads(
        INPUT_JSON.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Personnel output must be a top-level JSON object."
        )

    return payload


def team_quality(team_payload: dict[str, Any]) -> dict[str, Any]:
    players = (
        team_payload.get("players")
        if isinstance(
            team_payload.get("players"),
            list,
        )
        else
        []
    )

    concerns = [
        player
        for player in players
        if safe_float(
            player.get("status_severity")
        )
        > 0.0
    ]

    production_matched = sum(
        1
        for player in concerns
        if bool(
            player.get("production_match")
        )
    )

    fallback = sum(
        1
        for player in concerns
        if clean(
            player.get("importance_method")
        )
        ==
        "ROLE_FALLBACK"
    )

    production_share = sum(
        1
        for player in concerns
        if clean(
            player.get("importance_method")
        )
        ==
        "SEASON_TO_DATE_PRODUCTION_SHARE"
    )

    manual = sum(
        1
        for player in concerns
        if clean(
            player.get("importance_method")
        )
        ==
        "MANUAL_OVERRIDE"
    )

    fallback_ratio = (
        fallback / len(concerns)
        if concerns
        else 0.0
    )

    status_counts = Counter(
        clean(
            player.get("status")
        )
        for player in concerns
    )

    position_counts = Counter(
        clean(
            player.get("position")
        )
        or
        "UNKNOWN"
        for player in concerns
    )

    top_players = sorted(
        concerns,
        key=lambda player: safe_float(
            player.get(
                "availability_loss_index"
            )
        ),
        reverse=True,
    )[:TOP_PLAYERS_PER_TEAM]

    warnings = []

    if fallback_ratio >= FALLBACK_HEAVY_THRESHOLD and concerns:
        warnings.append(
            "FALLBACK_HEAVY"
        )

    if team_payload.get("qb_out"):
        warnings.append(
            "QB_OUT"
        )

    if team_payload.get("qb_uncertain"):
        warnings.append(
            "QB_UNCERTAIN"
        )

    return {
        "team": clean(
            team_payload.get("team")
        ),
        "concerns": len(concerns),
        "loss_index": safe_float(
            team_payload.get(
                "personnel_loss_index"
            )
        ),
        "production_matched": production_matched,
        "fallback": fallback,
        "production_share": production_share,
        "manual": manual,
        "fallback_ratio": fallback_ratio,
        "status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "position_counts": dict(
            sorted(
                position_counts.items()
            )
        ),
        "qb_out": bool(
            team_payload.get("qb_out")
        ),
        "qb_uncertain": bool(
            team_payload.get("qb_uncertain")
        ),
        "top_players": top_players,
        "warnings": warnings,
    }


def print_team(
    label: str,
    quality: dict[str, Any],
) -> None:
    print(
        f"{label:<5} {quality['team']}"
    )

    print(
        f"      loss index:      "
        f"{quality['loss_index']:.2f}"
    )

    print(
        f"      concerns:        "
        f"{quality['concerns']}"
    )

    print(
        f"      production match:"
        f" {quality['production_matched']}/"
        f"{quality['concerns']}"
    )

    print(
        f"      role fallback:   "
        f"{quality['fallback']}/"
        f"{quality['concerns']} "
        f"({quality['fallback_ratio']:.0%})"
    )

    print(
        f"      QB out:          "
        f"{bool_text(quality['qb_out'])}"
    )

    print(
        f"      QB uncertain:    "
        f"{bool_text(quality['qb_uncertain'])}"
    )

    print(
        "      status counts:   "
        +
        json.dumps(
            quality["status_counts"],
            sort_keys=True,
        )
    )

    if quality["warnings"]:
        print(
            "      quality flags:   "
            +
            ", ".join(
                quality["warnings"]
            )
        )

    top_players = quality[
        "top_players"
    ]

    if top_players:
        print(
            "      top impacts:"
        )

        for player in top_players:
            position = (
                clean(
                    player.get("position")
                )
                or
                "?"
            )

            name = (
                clean(
                    player.get("player_name")
                )
                or
                "UNKNOWN"
            )

            status = (
                clean(
                    player.get("status")
                )
                or
                "UNKNOWN"
            )

            importance = safe_float(
                player.get(
                    "importance_score"
                )
            )

            loss = safe_float(
                player.get(
                    "availability_loss_index"
                )
            )

            method = (
                clean(
                    player.get(
                        "importance_method"
                    )
                )
                or
                "UNKNOWN"
            )

            production = (
                "MATCH"
                if player.get(
                    "production_match"
                )
                else
                "NO_MATCH"
            )

            print(
                f"        {position:<4} "
                f"{name:<30} "
                f"{status:<16} "
                f"importance={importance:>6.2f} "
                f"loss={loss:>6.2f} "
                f"{method} "
                f"[{production}]"
            )


def main() -> int:
    report = load_report()

    games = (
        report.get("games")
        if isinstance(
            report.get("games"),
            list,
        )
        else
        []
    )

    games_with_data = [
        game
        for game in games
        if bool(
            game.get("has_personnel_data")
        )
    ]

    audited = []

    aggregate_statuses = Counter()
    aggregate_positions = Counter()

    total_concerns = 0
    total_fallback = 0
    total_production_match = 0

    for game in games_with_data:
        away_payload = (
            game.get("away_personnel")
            if isinstance(
                game.get("away_personnel"),
                dict,
            )
            else
            {}
        )

        home_payload = (
            game.get("home_personnel")
            if isinstance(
                game.get("home_personnel"),
                dict,
            )
            else
            {}
        )

        away = team_quality(
            away_payload
        )

        home = team_quality(
            home_payload
        )

        for quality in (
            away,
            home,
        ):
            total_concerns += quality[
                "concerns"
            ]

            total_fallback += quality[
                "fallback"
            ]

            total_production_match += quality[
                "production_matched"
            ]

            aggregate_statuses.update(
                quality[
                    "status_counts"
                ]
            )

            aggregate_positions.update(
                quality[
                    "position_counts"
                ]
            )

        differential = safe_float(
            game.get(
                "home_minus_away_personnel_loss_index"
            )
        )

        audited.append(
            {
                "game_id": clean(
                    game.get("game_id")
                ),
                "week": game.get("week"),
                "kickoff_utc": clean(
                    game.get("kickoff_utc")
                ),
                "away_team": clean(
                    game.get("away_team")
                ),
                "home_team": clean(
                    game.get("home_team")
                ),
                "away": away,
                "home": home,
                "differential": differential,
                "abs_differential": abs(
                    differential
                ),
            }
        )

    audited.sort(
        key=lambda game: game[
            "abs_differential"
        ],
        reverse=True,
    )

    rule()

    print(
        "CFB EDGE ENGINE — PERSONNEL OUTPUT AUDIT V1"
    )

    rule()

    print(
        f"Input: {INPUT_JSON}"
    )

    print(
        "Mode: READ ONLY / NO API CALLS / NO MODEL CHANGES"
    )

    print()

    print(
        f"Current games in file:          "
        f"{len(games)}"
    )

    print(
        f"Games with personnel data:      "
        f"{len(games_with_data)}"
    )

    print(
        f"Latest valid observations:      "
        f"{safe_int(report.get('latest_valid_observations'))}"
    )

    print(
        f"Production-enriched globally:   "
        f"{safe_int(report.get('observations_enriched_with_production'))}"
    )

    print(
        f"Without production match:       "
        f"{safe_int(report.get('observations_without_production_match'))}"
    )

    print(
        f"Concern players audited:        "
        f"{total_concerns}"
    )

    print(
        f"Role-fallback concerns:         "
        f"{total_fallback}/"
        f"{total_concerns} "
        f"({(total_fallback / total_concerns if total_concerns else 0.0):.0%})"
    )

    print(
        f"Production-matched concerns:    "
        f"{total_production_match}/"
        f"{total_concerns}"
    )

    print()
    print(
        "STATUS TOTALS"
    )

    rule("-")

    print(
        json.dumps(
            dict(
                sorted(
                    aggregate_statuses.items()
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "POSITION TOTALS"
    )

    rule("-")

    print(
        json.dumps(
            dict(
                sorted(
                    aggregate_positions.items()
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )

    print()
    rule()

    print(
        "GAME-BY-GAME — SORTED BY ABSOLUTE PERSONNEL-LOSS DIFFERENTIAL"
    )

    rule()

    for rank, game in enumerate(
        audited,
        start=1,
    ):
        print()
        print(
            f"{rank:>2}. "
            f"{game['away_team']} @ "
            f"{game['home_team']}"
        )

        print(
            f"    game_id:       "
            f"{game['game_id']}"
        )

        print(
            f"    kickoff:       "
            f"{game['kickoff_utc']}"
        )

        print(
            f"    home-away diff:"
            f" {game['differential']:+.2f}"
        )

        print()

        print_team(
            "AWAY",
            game["away"],
        )

        print()

        print_team(
            "HOME",
            game["home"],
        )

    fallback_heavy_teams = []

    qb_flags = []

    for game in audited:
        for side in (
            "away",
            "home",
        ):
            quality = game[
                side
            ]

            if (
                quality[
                    "concerns"
                ]
                and
                quality[
                    "fallback_ratio"
                ]
                >=
                FALLBACK_HEAVY_THRESHOLD
            ):
                fallback_heavy_teams.append(
                    (
                        quality["team"],
                        quality["fallback_ratio"],
                        quality["concerns"],
                    )
                )

            if (
                quality[
                    "qb_out"
                ]
                or
                quality[
                    "qb_uncertain"
                ]
            ):
                qb_flags.append(
                    (
                        quality["team"],
                        quality["qb_out"],
                        quality["qb_uncertain"],
                    )
                )

    print()
    rule()

    print(
        "QUALITY REVIEW"
    )

    rule()

    print(
        f"Fallback-heavy teams "
        f"(>= {FALLBACK_HEAVY_THRESHOLD:.0%}): "
        f"{len(fallback_heavy_teams)}"
    )

    for team, ratio, concerns in sorted(
        fallback_heavy_teams,
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"  {team:<28} "
            f"{ratio:.0%} fallback "
            f"({concerns} concerns)"
        )

    print()
    print(
        f"QB-flagged teams: {len(qb_flags)}"
    )

    for team, qb_out, qb_uncertain in qb_flags:
        print(
            f"  {team:<28} "
            f"QB_OUT={bool_text(qb_out)} "
            f"QB_UNCERTAIN={bool_text(qb_uncertain)}"
        )

    print()
    rule()

    print(
        "INTERPRETATION"
    )

    rule()

    print(
        "This audit is checking whether the personnel layer is structurally "
        "credible enough to integrate into the live pipeline."
    )

    print(
        "A high fallback rate is expected for OL/defensive/back-up players "
        "because player_game_stat_values does not provide useful production "
        "for many of them."
    )

    print(
        "Do NOT interpret personnel_loss_index as sportsbook points."
    )

    print(
        "Do NOT change Strength or the frozen neural model from this audit."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
