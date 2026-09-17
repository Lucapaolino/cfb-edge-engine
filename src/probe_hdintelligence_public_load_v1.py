#!/usr/bin/env python3
"""
CFB Edge Engine — HD Intelligence Public-Load Probe V1

Purpose
-------
Probe the CONFIRMED public endpoint used by the HD Intelligence conference
availability app for ACC, Big 12, and SEC:

    POST https://app.hdintelligence.com/api/public-load

The endpoint and payload contract were discovered directly in the public
JavaScript bundle. This probe does not guess endpoints.

Safety
------
- public POST requests only
- no credentials
- no API key
- no login
- no auth bypass
- no database writes
- no project-data writes
- no sportsbook calls
- no CFBD calls

Big Ten is intentionally NOT probed here. Its visible /api/V$3rs1 call is tied
to userEmail lookup and is not a confirmed public report-data endpoint.

Run
---
python src/probe_hdintelligence_public_load_v1.py
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests


WIDTH = 118
TIMEOUT_SECONDS = 20
MAX_RENDER_CHARS = 12000

ENDPOINT = 'https://app.hdintelligence.com/api/public-load'
APP_ORIGIN = 'https://app.hdintelligence.com'

SOURCES = [
    {
        'label': 'ACC',
        'conference_param': 'ACC',
        'sport_param': 'Football',
        'type_param': 'report',
        'source': 'ACC',
        'official_referrer': (
            'https://theacc.com/sports/2025/8/28/'
            'availability-reporting-football.aspx'
        ),
        'app_url': (
            'https://app.hdintelligence.com/'
            '?source=ACC&sport=Football&conf=ACC&type=report'
        ),
    },
    {
        'label': 'BIG12',
        'conference_param': 'B12',
        'sport_param': 'Football',
        'type_param': 'report',
        'source': 'B12',
        'official_referrer': (
            'https://big12sports.com/sports/2025/8/14/FBreporting.aspx'
        ),
        'app_url': (
            'https://app.hdintelligence.com/'
            '?source=B12&sport=Football&conf=B12&type=report'
        ),
    },
    {
        'label': 'SEC',
        'conference_param': 'SEC',
        'sport_param': 'Football',
        'type_param': 'report',
        'source': 'SEC',
        'official_referrer': 'https://www.secsports.com/fbreports',
        'app_url': (
            'https://app.hdintelligence.com/'
            '?source=SEC&sport=Football&conf=SEC&type=report'
        ),
    },
]

SENSITIVE_KEY_RE = re.compile(
    r'(token|secret|password|authorization|cookie|email|phone)',
    re.IGNORECASE,
)


def rule(char: str = '=') -> None:
    print(char * WIDTH)


def clean(value: object) -> str:
    return ' '.join(str(value or '').strip().split())


def redact(value: Any, key_hint: str = '') -> Any:
    if SENSITIVE_KEY_RE.search(key_hint):
        if value in (None, '', [], {}):
            return value
        return '<REDACTED>'

    if isinstance(value, dict):
        return {
            str(key): redact(item, str(key))
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            redact(item, key_hint)
            for item in value[:50]
        ]

    if isinstance(value, str):
        # Redact obvious email addresses even when the field name is generic.
        if re.search(r'\b[^\s@]+@[^\s@]+\.[^\s@]+\b', value):
            return '<REDACTED_EMAIL>'

        if len(value) > 2000:
            return value[:2000] + ' ... <TRUNCATED>'

    return value


def describe(value: Any, depth: int = 0) -> Any:
    """Return a structural schema summary without dumping huge payloads."""
    if depth >= 4:
        return type(value).__name__

    if isinstance(value, dict):
        return {
            str(key): describe(item, depth + 1)
            for key, item in list(value.items())[:80]
        }

    if isinstance(value, list):
        result = {
            '__type__': 'list',
            '__length__': len(value),
        }

        if value:
            result['__first_item_schema__'] = describe(
                value[0],
                depth + 1,
            )

        return result

    if value is None:
        return 'null'

    return type(value).__name__


def probe_source(
    session: requests.Session,
    source: dict[str, str],
) -> None:
    rule()
    print(source['label'])
    rule()

    payload = {
        'conference_param': source['conference_param'],
        'sport_param': source['sport_param'],
        'type_param': source['type_param'],
        'referrer': source['official_referrer'],
        'source': source['source'],
    }

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/152.0 Safari/537.36'
        ),
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json',
        'Origin': APP_ORIGIN,
        'Referer': source['app_url'],
    }

    print(f"Endpoint: {ENDPOINT}")
    print('Method:   POST')
    print('Auth:     NONE')
    print('API key:  NONE')
    print('Payload:')
    print(json.dumps(payload, indent=2, sort_keys=True))
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
        print(f'REQUEST FAILED: {error}')
        return

    print(f'HTTP status:  {response.status_code}')
    print(f'Content-Type: {clean(response.headers.get("content-type")) or "UNKNOWN"}')
    print(f'Bytes:        {len(response.content):,}')

    text = response.text

    try:
        parsed = response.json()
    except ValueError:
        parsed = None

    if parsed is None:
        print()
        print('RESPONSE IS NOT JSON')
        print('-' * WIDTH)
        print(text[:MAX_RENDER_CHARS])
        return

    print()
    print('TOP-LEVEL RESPONSE')
    print('-' * WIDTH)

    if isinstance(parsed, dict):
        print('Keys: ' + ', '.join(sorted(str(key) for key in parsed.keys())))
    elif isinstance(parsed, list):
        print(f'List length: {len(parsed)}')
    else:
        print(f'Type: {type(parsed).__name__}')

    print()
    print('SCHEMA')
    print('-' * WIDTH)
    print(
        json.dumps(
            describe(parsed),
            indent=2,
            sort_keys=True,
            default=str,
        )[:MAX_RENDER_CHARS]
    )

    print()
    print('REDACTED SAMPLE')
    print('-' * WIDTH)
    print(
        json.dumps(
            redact(parsed),
            indent=2,
            sort_keys=True,
            default=str,
        )[:MAX_RENDER_CHARS]
    )


def main() -> int:
    rule()
    print('CFB EDGE ENGINE — HD INTELLIGENCE PUBLIC-LOAD PROBE V1')
    rule()

    print('Mode: PUBLIC ENDPOINT PROBE / NO AUTH / NO DATABASE WRITES')
    print('Endpoint contract: confirmed from the public availability-app bundle')
    print()

    session = requests.Session()

    for source in SOURCES:
        try:
            probe_source(
                session,
                source,
            )
        except KeyboardInterrupt:
            print('\nCancelled by user.')
            return 130
        except Exception as error:
            rule('!')
            print(f"{source['label']} ERROR: {error}")
            rule('!')

    print()
    rule()
    print('PROBE COMPLETE')
    rule()

    print('Big Ten intentionally skipped: no confirmed public report-data endpoint yet.')
    print()
    print('Paste this output back into the project chat.')
    print()
    print(
        'If public-load returns report records or a stable report-data contract, '
        'the next file will be the production collector. If it only returns a '
        'public-view authorization flag, we will inspect the now-confirmed public '
        'view flow one final time without using private/admin endpoints.'
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
