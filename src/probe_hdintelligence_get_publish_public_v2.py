#!/usr/bin/env python3
"""
CFB Edge Engine — HD Intelligence get-publish-public Probe V2

Compact replacement for V1.

Why V2 exists
-------------
V1 successfully reached the confirmed public report endpoint, but ACC returned
~5.5 MB of JSON containing large base64 images and full rosters. Printing that
much data makes Terminal look frozen even though the HTTP request already
finished.

V2 keeps the SAME confirmed public endpoint and SAME payload, but prints only:
- response size and timing
- report IDs
- report type / publish date / posted time
- team names
- roster row counts
- counts by status
- ONLY players whose status is not Available
- a compact union of field names

It never prints embedded images or full roster dumps.

Safety
------
- confirmed public endpoint only
- no API key
- no login
- no auth token
- no cookies required
- no database writes
- no project-data writes
- no CFBD calls
- no sportsbook calls
- Big Ten intentionally excluded because it uses a different transport

Run
---
python src/probe_hdintelligence_get_publish_public_v2.py
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

import requests


WIDTH = 118
TIMEOUT_SECONDS = 30

ENDPOINT = "https://app.hdintelligence.com/api/get-publish-public"

SOURCES = [
    {
        "label": "ACC",
        "sport": "Football",
        "organization": "ACC",
        "conference": "ACC",
        "referer": (
            "https://app.hdintelligence.com/"
            "?source=ACC&sport=Football&conf=ACC&type=report"
        ),
    },
    {
        "label": "BIG12",
        "sport": "Football",
        "organization": "B12",
        "conference": "B12",
        "referer": (
            "https://app.hdintelligence.com/"
            "?source=B12&sport=Football&conf=B12&type=report"
        ),
    },
    {
        "label": "SEC",
        "sport": "Football",
        "organization": "SEC",
        "conference": "SEC",
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

PLAYER_NAME_RE = re.compile(
    r"^(?P<position>[A-Za-z0-9./-]+)\s+"
    r"#(?P<number>[A-Za-z0-9-]+)\s+"
    r"(?P<name>.+)$"
)


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def parse_player_label(value: Any) -> dict[str, str]:
    text = clean(value)

    match = PLAYER_NAME_RE.match(text)

    if not match:
        return {
            "position": "",
            "number": "",
            "name": text,
        }

    return {
        "position": clean(match.group("position")).upper(),
        "number": clean(match.group("number")),
        "name": clean(match.group("name")),
    }


def collect_field_names(
    value: Any,
    output: set[str] | None = None,
    *,
    depth: int = 0,
    max_depth: int = 5,
) -> set[str]:
    if output is None:
        output = set()

    if depth > max_depth:
        return output

    if isinstance(value, dict):
        for key, item in value.items():
            output.add(str(key))
            collect_field_names(
                item,
                output,
                depth=depth + 1,
                max_depth=max_depth,
            )

    elif isinstance(value, list):
        for item in value[:20]:
            collect_field_names(
                item,
                output,
                depth=depth + 1,
                max_depth=max_depth,
            )

    return output


def status_counter(
    rows: list[dict[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()

    for row in rows:
        status = clean(row.get("status")) or "UNKNOWN"
        counter[status] += 1

    return counter


def concern_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for row in rows:
        status = clean(row.get("status"))

        if status.casefold() == "available":
            continue

        parsed = parse_player_label(
            row.get("name")
        )

        output.append(
            {
                "position": parsed["position"],
                "number": parsed["number"],
                "player_name": parsed["name"],
                "status": status or "UNKNOWN",
                "exempt_status": clean(
                    row.get("exemptStatus")
                ),
            }
        )

    return output


def summarize_team(
    team: dict[str, Any],
) -> dict[str, Any]:
    rows_raw = team.get("rows")

    rows = (
        [
            row
            for row in rows_raw
            if isinstance(row, dict)
        ]
        if isinstance(rows_raw, list)
        else
        []
    )

    counts = status_counter(rows)

    return {
        "team_name": clean(
            team.get("teamName")
        ),
        "team_display_name": clean(
            team.get("teamDisplayName")
        ),
        "roster_rows": len(rows),
        "status_counts": dict(
            sorted(
                counts.items(),
                key=lambda item: item[0].casefold(),
            )
        ),
        "concerns": concern_rows(rows),
    }


def summarize_report(
    report_id: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    games_raw = report.get("games")

    teams = (
        [
            item
            for item in games_raw
            if isinstance(item, dict)
        ]
        if isinstance(games_raw, list)
        else
        []
    )

    footer = (
        report.get("footer")
        if isinstance(
            report.get("footer"),
            dict,
        )
        else
        {}
    )

    return {
        "report_id": report_id,
        "report_type": clean(
            report.get("ReportType")
        ),
        "publish_date": clean(
            report.get("publishDate")
        ),
        "publish_day": clean(
            report.get("publishDayOfWeek")
        ),
        "posted_time": clean(
            report.get("postedTime")
        ),
        "conference_time_zone": clean(
            report.get("conferenceTimeZone")
        ),
        "game_neutral": bool(
            report.get("gameNeutral")
        ),
        "footer_date": clean(
            footer.get("date")
        ),
        "footer_time": clean(
            footer.get("time")
        ),
        "location": clean(
            footer.get("location")
        ),
        "teams": [
            summarize_team(team)
            for team in teams
        ],
    }


def print_report_summary(
    summary: dict[str, Any],
) -> None:
    print()
    rule("-")

    print(
        f"REPORT {summary['report_id']}"
    )

    rule("-")

    print(
        f"Type:          {summary['report_type'] or 'N/A'}"
    )

    print(
        f"Publish date:  {summary['publish_date'] or 'N/A'}"
    )

    print(
        f"Publish day:   {summary['publish_day'] or 'N/A'}"
    )

    print(
        f"Posted time:   {summary['posted_time'] or 'N/A'}"
    )

    print(
        f"Time zone:     {summary['conference_time_zone'] or 'N/A'}"
    )

    print(
        f"Footer date:   {summary['footer_date'] or 'N/A'}"
    )

    print(
        f"Footer time:   {summary['footer_time'] or 'N/A'}"
    )

    print(
        f"Location:      {summary['location'] or 'N/A'}"
    )

    print(
        f"Neutral:       {summary['game_neutral']}"
    )

    teams = summary.get("teams") or []

    print(
        f"Teams:         {len(teams)}"
    )

    for team in teams:
        print()

        display = (
            team.get("team_display_name")
            or
            team.get("team_name")
            or
            "UNKNOWN TEAM"
        )

        print(
            f"  {display}"
        )

        print(
            f"    roster rows:   {team.get('roster_rows', 0)}"
        )

        print(
            "    status counts: "
            +
            json.dumps(
                team.get("status_counts") or {},
                sort_keys=True,
            )
        )

        concerns = team.get("concerns") or []

        print(
            f"    concerns:      {len(concerns)}"
        )

        for player in concerns:
            number = (
                f" #{player['number']}"
                if player.get("number")
                else
                ""
            )

            position = (
                f"{player['position']} "
                if player.get("position")
                else
                ""
            )

            exempt = (
                f" | exempt={player['exempt_status']}"
                if player.get("exempt_status")
                else
                ""
            )

            print(
                "      "
                f"{position}{number} {player['player_name']}"
                f" — {player['status']}"
                f"{exempt}"
            )


def probe_source(
    session: requests.Session,
    source: dict[str, str],
) -> None:
    rule()
    print(source["label"])
    rule()

    payload = {
        "sport": source["sport"],
        "organization": source["organization"],
        "conference": source["conference"],
    }

    headers = dict(HEADERS_BASE)
    headers["Referer"] = source["referer"]

    print(
        f"Endpoint: {ENDPOINT}"
    )

    print(
        "Method:   POST"
    )

    print(
        "Auth:     NONE"
    )

    print(
        "API key:  NONE"
    )

    print(
        "Cookies:  NONE"
    )

    print(
        "Payload:  "
        +
        json.dumps(
            payload,
            sort_keys=True,
        )
    )

    print()
    print(
        "Requesting report payload..."
    )

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
        print(
            f"REQUEST FAILED: {error}"
        )
        return

    elapsed = (
        time.perf_counter()
        -
        started
    )

    print(
        f"HTTP status:  {response.status_code}"
    )

    print(
        f"Elapsed:      {elapsed:.2f}s"
    )

    print(
        f"Bytes:        {len(response.content):,}"
    )

    print(
        "Content-Type: "
        +
        (
            clean(
                response.headers.get(
                    "content-type"
                )
            )
            or
            "UNKNOWN"
        )
    )

    try:
        parsed = response.json()

    except ValueError:
        print(
            "ERROR: response was not JSON."
        )

        return

    if not isinstance(parsed, dict):
        print(
            "ERROR: expected a top-level JSON object."
        )

        print(
            f"Actual type: {type(parsed).__name__}"
        )

        return

    field_names = sorted(
        collect_field_names(parsed)
    )

    print()
    print(
        f"Reports returned: {len(parsed)}"
    )

    print(
        "Report IDs:      "
        +
        (
            ", ".join(
                sorted(
                    str(key)
                    for key in parsed.keys()
                )
            )
            if parsed
            else
            "NONE"
        )
    )

    print()
    print(
        "FIELD NAMES"
    )

    rule("-")

    print(
        ", ".join(field_names)
        if field_names
        else
        "NONE"
    )

    summaries = []

    for report_id, report in parsed.items():
        if not isinstance(report, dict):
            continue

        summaries.append(
            summarize_report(
                str(report_id),
                report,
            )
        )

    for summary in summaries:
        print_report_summary(summary)

    total_teams = sum(
        len(
            summary.get("teams")
            or
            []
        )
        for summary in summaries
    )

    total_roster_rows = sum(
        int(
            team.get("roster_rows")
            or
            0
        )
        for summary in summaries
        for team in (
            summary.get("teams")
            or
            []
        )
    )

    total_concerns = sum(
        len(
            team.get("concerns")
            or
            []
        )
        for summary in summaries
        for team in (
            summary.get("teams")
            or
            []
        )
    )

    print()
    rule("-")

    print(
        f"{source['label']} TOTALS"
    )

    rule("-")

    print(
        f"Reports:       {len(summaries)}"
    )

    print(
        f"Team entries:  {total_teams}"
    )

    print(
        f"Roster rows:   {total_roster_rows}"
    )

    print(
        f"Concerns:      {total_concerns}"
    )


def main() -> int:
    rule()

    print(
        "CFB EDGE ENGINE — HD INTELLIGENCE GET-PUBLISH-PUBLIC PROBE V2"
    )

    rule()

    print(
        "Mode: CONFIRMED PUBLIC REPORT ENDPOINT / COMPACT OUTPUT / NO WRITES"
    )

    print(
        "V2 deliberately does NOT print base64 images or full Available rosters."
    )

    print()

    session = requests.Session()

    completed = 0

    for source in SOURCES:
        try:
            probe_source(
                session,
                source,
            )

            completed += 1

        except KeyboardInterrupt:
            print(
                "\nCancelled by user."
            )

            return 130

        except Exception as error:
            rule("!")

            print(
                f"{source['label']} ERROR: {error}"
            )

            rule("!")

    print()
    rule()

    print(
        "PROBE COMPLETE"
    )

    rule()

    print(
        f"Conferences attempted: {completed}/{len(SOURCES)}"
    )

    print(
        "Big Ten intentionally excluded because its report transport is different."
    )

    print()
    print(
        "Paste this compact output back into the project chat."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
