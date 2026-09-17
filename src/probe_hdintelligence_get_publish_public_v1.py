#!/usr/bin/env python3
"""
CFB Edge Engine — HD Intelligence get-publish-public Probe V1

Purpose
-------
Probe the CONFIRMED public published-report endpoint used by the shared
ACC / Big 12 / SEC availability application:

    POST https://app.hdintelligence.com/api/get-publish-public

The exact public-mode call was confirmed from the current PublishScreen bundle:

    g.post(
        "/api/get-publish-public",
        {
            sport: t,
            organization: r,
            conference: o
        },
        {
            withCredentials: false,
            withXSRFToken: false
        }
    )

This probe is intentionally narrow. It exists only to learn the response
schema so the next file can be the production personnel collector.

Safety
------
- confirmed PUBLIC endpoint only
- no API key
- no login
- no authentication token
- no cookies required
- no auth bypass
- no admin/private endpoints
- no database writes
- no project-data writes
- no CFBD calls
- no sportsbook calls
- Big Ten is NOT included because it uses a different system

Run
---
python src/probe_hdintelligence_get_publish_public_v1.py
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests


WIDTH = 118
TIMEOUT_SECONDS = 25
MAX_SAMPLE_CHARS = 16000
MAX_SCHEMA_DEPTH = 6
MAX_SCHEMA_KEYS = 100
MAX_LIST_SAMPLE = 3

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

SENSITIVE_KEY_RE = re.compile(
    r"(token|secret|password|authorization|cookie|email|phone)",
    re.IGNORECASE,
)

PLAYERISH_KEYS = {
    "player",
    "playername",
    "player_name",
    "name",
    "status",
    "position",
    "team",
    "teamname",
    "team_name",
    "teamdisplayname",
    "rows",
    "games",
    "publishdate",
    "reporttype",
    "postedtime",
    "footer",
}


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def redact(value: Any, key_hint: str = "") -> Any:
    """
    Keep report/player fields readable, but redact likely credentials/contact
    information if any appear unexpectedly.
    """
    if SENSITIVE_KEY_RE.search(key_hint):
        if value in (None, "", [], {}):
            return value
        return "<REDACTED>"

    if isinstance(value, dict):
        return {
            str(key): redact(item, str(key))
            for key, item in list(value.items())[:200]
        }

    if isinstance(value, list):
        return [
            redact(item, key_hint)
            for item in value[:50]
        ]

    if isinstance(value, str):
        if re.search(
            r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b",
            value,
        ):
            return "<REDACTED_EMAIL>"

        if len(value) > 3000:
            return (
                value[:3000]
                +
                " ... <TRUNCATED>"
            )

    return value


def schema_of(
    value: Any,
    depth: int = 0,
) -> Any:
    if depth >= MAX_SCHEMA_DEPTH:
        return type(value).__name__

    if isinstance(value, dict):
        keys = list(value.keys())[:MAX_SCHEMA_KEYS]

        return {
            str(key): schema_of(
                value[key],
                depth + 1,
            )
            for key in keys
        }

    if isinstance(value, list):
        result = {
            "__type__": "list",
            "__length__": len(value),
        }

        samples = []

        for item in value[:MAX_LIST_SAMPLE]:
            samples.append(
                schema_of(
                    item,
                    depth + 1,
                )
            )

        if samples:
            result[
                "__sample_item_schemas__"
            ] = samples

        return result

    if value is None:
        return "null"

    return type(value).__name__


def collect_key_names(
    value: Any,
    names: set[str] | None = None,
) -> set[str]:
    if names is None:
        names = set()

    if isinstance(value, dict):
        for key, item in value.items():
            names.add(
                str(key)
            )

            collect_key_names(
                item,
                names,
            )

    elif isinstance(value, list):
        for item in value[:200]:
            collect_key_names(
                item,
                names,
            )

    return names


def find_report_like_objects(
    value: Any,
    path: str = "$",
    output: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    if output is None:
        output = []

    if isinstance(value, dict):
        normalized_keys = {
            str(key).replace("-", "").replace("_", "").casefold()
            for key in value.keys()
        }

        score = sum(
            1
            for key in PLAYERISH_KEYS
            if key.replace("_", "").casefold()
            in
            normalized_keys
        )

        if score >= 2:
            output.append(
                (
                    path,
                    value,
                )
            )

        for key, item in value.items():
            find_report_like_objects(
                item,
                f"{path}.{key}",
                output,
            )

    elif isinstance(value, list):
        for index, item in enumerate(
            value[:100]
        ):
            find_report_like_objects(
                item,
                f"{path}[{index}]",
                output,
            )

    return output


def compact_sample(
    value: Any,
) -> str:
    rendered = json.dumps(
        redact(value),
        indent=2,
        sort_keys=True,
        default=str,
    )

    if len(rendered) > MAX_SAMPLE_CHARS:
        return (
            rendered[:MAX_SAMPLE_CHARS]
            +
            "\n... <OUTPUT TRUNCATED>"
        )

    return rendered


def probe_source(
    session: requests.Session,
    source: dict[str, str],
) -> None:
    rule()
    print(source["label"])
    rule()

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
        "Payload:"
    )

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )

    print()

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

    print(
        f"HTTP status:  {response.status_code}"
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

    print(
        f"Bytes:        {len(response.content):,}"
    )

    try:
        parsed = response.json()

    except ValueError:
        print()
        print(
            "RESPONSE IS NOT JSON"
        )

        rule("-")

        print(
            response.text[
                :MAX_SAMPLE_CHARS
            ]
        )

        return

    print()
    print(
        "TOP-LEVEL TYPE"
    )

    rule("-")

    print(
        type(parsed).__name__
    )

    if isinstance(
        parsed,
        dict,
    ):
        print(
            "Top-level keys: "
            +
            ", ".join(
                sorted(
                    str(key)
                    for key in parsed.keys()
                )
            )
        )

    elif isinstance(
        parsed,
        list,
    ):
        print(
            f"Top-level list length: {len(parsed)}"
        )

    all_keys = sorted(
        collect_key_names(
            parsed
        )
    )

    print()
    print(
        "ALL DISCOVERED FIELD NAMES"
    )

    rule("-")

    if all_keys:
        for key in all_keys:
            print(
                f"  {key}"
            )
    else:
        print(
            "  NONE"
        )

    print()
    print(
        "SCHEMA"
    )

    rule("-")

    schema_text = json.dumps(
        schema_of(
            parsed
        ),
        indent=2,
        sort_keys=True,
        default=str,
    )

    print(
        schema_text[
            :MAX_SAMPLE_CHARS
        ]
    )

    report_like = find_report_like_objects(
        parsed
    )

    print()
    print(
        "REPORT-LIKE OBJECTS"
    )

    rule("-")

    print(
        f"Detected: {len(report_like)}"
    )

    for index, (
        path,
        obj,
    ) in enumerate(
        report_like[:10],
        start=1,
    ):
        print()
        print(
            f"[{index}] PATH: {path}"
        )

        print(
            compact_sample(
                obj
            )[:5000]
        )

    print()
    print(
        "REDACTED RESPONSE SAMPLE"
    )

    rule("-")

    print(
        compact_sample(
            parsed
        )
    )


def main() -> int:
    rule()

    print(
        "CFB EDGE ENGINE — HD INTELLIGENCE GET-PUBLISH-PUBLIC PROBE V1"
    )

    rule()

    print(
        "Mode: CONFIRMED PUBLIC REPORT ENDPOINT / NO AUTH / NO WRITES"
    )

    print(
        "Purpose: learn the real response schema before building the production collector."
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
        "Big Ten intentionally excluded because its public report transport is different."
    )

    print()
    print(
        "Paste this output back into the project chat."
    )

    print()
    print(
        "If this returns the report/player structure shown by PublishScreen, "
        "the next file can be the actual immutable personnel collector."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
