#!/usr/bin/env python3
"""
CFB Edge Engine — HD Intelligence Public Report Flow Audit V3

Purpose
-------
Final narrow transport audit for the shared ACC / Big 12 / SEC public
availability-report application.

V2 confirmed that the public app first calls:

    POST /api/public-load

and that this returns only:

    {"public": true}

This V3 does NOT call any application API endpoints. It only downloads the
public app shell and its current JavaScript bundle, then statically inspects
the bundle to identify what code executes after public mode is enabled and
which literal /api/... paths are referenced by report-facing code.

Safety
------
- GET requests only
- no application API POSTs
- no login
- no credentials
- no auth bypass
- no database writes
- no project-data writes
- no CFBD calls
- no sportsbook calls
- any x-api-key / authorization-like values found in JavaScript are REDACTED

Run
---
python src/audit_hdintelligence_public_report_flow_v3.py
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests


WIDTH = 118
TIMEOUT_SECONDS = 25
MAX_BUNDLE_BYTES = 15_000_000
CONTEXT_RADIUS = 1800

APP_URL = (
    "https://app.hdintelligence.com/"
    "?source=ACC&sport=Football&conf=ACC&type=report"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

KNOWN_NON_REPORT_ENDPOINTS = {
    "/api/auth/session",
    "/api/auth/logout",
    "/api/getSelectOptions",
    "/api/get-all-conferences-and-teams",
    "/api/get-all-conferences",
    "/api/resolve-conference",
}

PUBLIC_BOOTSTRAP_ENDPOINTS = {
    "/api/public-load",
    "/api/get-conference-teams",
    "/api/get-team-display",
    "/api/get-team-timezone",
}

MARKERS = [
    '"/api/public-load"',
    "public check=",
    'c("Public")',
    'P==="report"',
    "type_param",
    "conference_param",
    "sport_param",
    "withCredentials:!1",
    "withXSRFToken:!1",
    "availability",
    "injury",
    "published",
    "player",
    "position",
    "status",
]


class ScriptParser(HTMLParser):

    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() != "script":
            return

        mapping = {
            key.casefold(): value
            for key, value in attrs
            if key
        }

        src = mapping.get("src")

        if src:
            self.scripts.append(src)


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def sanitize(text: str) -> str:
    """
    Redact values that look like secrets/tokens/authorization material while
    leaving endpoint names and surrounding application logic readable.
    """
    patterns = [
        (
            re.compile(
                r'(?i)(x-api-key["\']?\s*[:=]\s*["\'])[^"\']+(["\'])'
            ),
            r'\1<REDACTED>\2',
        ),
        (
            re.compile(
                r'(?i)(authorization["\']?\s*[:=]\s*["\']?)Bearer\s+[^"\'}\],;)]+'
            ),
            r'\1Bearer <REDACTED>',
        ),
        (
            re.compile(
                r'(?i)(api[_-]?key["\']?\s*[:=]\s*["\'])[^"\']+(["\'])'
            ),
            r'\1<REDACTED>\2',
        ),
        (
            re.compile(
                r'(?i)(token["\']?\s*[:=]\s*["\'])[^"\']{12,}(["\'])'
            ),
            r'\1<REDACTED>\2',
        ),
    ]

    result = text

    for pattern, replacement in patterns:
        result = pattern.sub(
            replacement,
            result,
        )

    return result


def get_text(
    session: requests.Session,
    url: str,
    max_bytes: int,
) -> tuple[requests.Response | None, str | None]:
    try:
        response = session.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT_SECONDS,
            allow_redirects=True,
            stream=True,
        )

        chunks = []
        total = 0

        for chunk in response.iter_content(
            chunk_size=128 * 1024,
        ):
            if not chunk:
                continue

            remaining = max_bytes - total

            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunks.append(
                    chunk[:remaining]
                )
                total += remaining
                break

            chunks.append(chunk)
            total += len(chunk)

        raw = b"".join(chunks)

        encoding = response.encoding or "utf-8"

        try:
            text = raw.decode(
                encoding,
                errors="replace",
            )

        except LookupError:
            text = raw.decode(
                "utf-8",
                errors="replace",
            )

        return response, text

    except requests.RequestException as error:
        return None, str(error)


def discover_bundle(
    page_url: str,
    page_text: str,
) -> str | None:
    parser = ScriptParser()

    try:
        parser.feed(
            page_text
        )
    except Exception:
        pass

    candidates = []

    page_host = urlparse(
        page_url
    ).netloc.casefold()

    for src in parser.scripts:
        full = urljoin(
            page_url,
            html.unescape(src),
        )

        host = urlparse(
            full
        ).netloc.casefold()

        if host == page_host:
            candidates.append(full)

    if not candidates:
        return None

    # Prefer the app's own compiled asset instead of tiny runtime snippets.
    candidates.sort(
        key=lambda url: (
            "/assets/" not in url
            and
            "/static/js/" not in url,
            len(url),
        )
    )

    return candidates[0]


def literal_api_paths(
    bundle: str,
) -> list[str]:
    paths = set()

    for match in re.finditer(
        r'["\'](/api/[A-Za-z0-9_$./?&=%:{}-]+)["\']',
        bundle,
    ):
        paths.add(
            match.group(1)
        )

    return sorted(paths)


def literal_http_calls(
    bundle: str,
) -> list[tuple[str, str]]:
    """
    Find literal .get("/api/..."), .post("/api/..."), etc.
    The object name is deliberately ignored because minification changes it.
    """
    calls = set()

    pattern = re.compile(
        r'\.(get|post|put|patch|delete)\(\s*["\']'
        r'(/api/[A-Za-z0-9_$./?&=%:{}-]+)["\']',
        re.IGNORECASE,
    )

    for match in pattern.finditer(bundle):
        calls.add(
            (
                match.group(1).upper(),
                match.group(2),
            )
        )

    return sorted(
        calls,
        key=lambda item: (
            item[1],
            item[0],
        ),
    )


def contexts(
    text: str,
    needle: str,
    radius: int = CONTEXT_RADIUS,
    limit: int = 4,
) -> list[str]:
    result = []
    search_from = 0

    while len(result) < limit:
        index = text.find(
            needle,
            search_from,
        )

        if index < 0:
            break

        start = max(
            0,
            index - radius,
        )

        end = min(
            len(text),
            index + len(needle) + radius,
        )

        snippet = text[
            start:end
        ]

        snippet = sanitize(
            snippet
        )

        if snippet not in result:
            result.append(snippet)

        search_from = (
            index
            +
            len(needle)
        )

    return result


def interesting_string_literals(
    bundle: str,
) -> list[str]:
    result = set()

    keywords = (
        "availability",
        "injury",
        "report",
        "published",
        "questionable",
        "doubtful",
        "out",
        "position",
        "player",
        "school",
        "team",
    )

    pattern = re.compile(
        r'["\']([^"\']{3,140})["\']'
    )

    for match in pattern.finditer(bundle):
        value = clean(
            match.group(1)
        )

        lowered = value.casefold()

        if any(
            keyword in lowered
            for keyword in keywords
        ):
            # Skip obvious library/docs noise.
            if (
                "react" in lowered
                or
                "three" in lowered
                or
                "node_modules" in lowered
            ):
                continue

            result.add(
                sanitize(value)
            )

    return sorted(result)[:250]


def print_context_group(
    bundle: str,
    marker: str,
) -> None:
    found = contexts(
        bundle,
        marker,
    )

    if not found:
        return

    print()
    rule("-")
    print(
        f"MARKER: {marker}"
    )
    rule("-")

    for index, snippet in enumerate(
        found,
        start=1,
    ):
        print(
            f"[{index}]"
        )

        print(snippet)

        print()


def main() -> int:
    rule()
    print(
        "CFB EDGE ENGINE — HD INTELLIGENCE PUBLIC REPORT FLOW AUDIT V3"
    )
    rule()

    print(
        "Mode: STATIC PUBLIC-BUNDLE INSPECTION / GET ONLY / NO API POSTS"
    )

    print(
        "Goal: identify the report-data call that follows public-load."
    )

    print()

    session = requests.Session()

    page_response, page_text = get_text(
        session,
        APP_URL,
        2_000_000,
    )

    if page_response is None:
        print(
            f"APP PAGE FAILED: {page_text}"
        )
        return 1

    print(
        f"App page HTTP: {page_response.status_code}"
    )

    print(
        f"App final URL: {page_response.url}"
    )

    bundle_url = discover_bundle(
        page_response.url,
        page_text or "",
    )

    if not bundle_url:
        print(
            "No same-origin application bundle discovered."
        )
        return 1

    print(
        f"Bundle: {bundle_url}"
    )

    bundle_response, bundle = get_text(
        session,
        bundle_url,
        MAX_BUNDLE_BYTES,
    )

    if bundle_response is None:
        print(
            f"BUNDLE FAILED: {bundle}"
        )
        return 1

    bundle = bundle or ""

    print(
        f"Bundle HTTP: {bundle_response.status_code}"
    )

    print(
        f"Bundle characters scanned: {len(bundle):,}"
    )

    paths = literal_api_paths(
        bundle
    )

    calls = literal_http_calls(
        bundle
    )

    print()
    rule()
    print(
        "ALL LITERAL /api/... PATHS"
    )
    rule()

    for path in paths:
        if path in KNOWN_NON_REPORT_ENDPOINTS:
            label = "KNOWN ADMIN/AUTH — DO NOT USE"
        elif path in PUBLIC_BOOTSTRAP_ENDPOINTS:
            label = "KNOWN PUBLIC/BOOTSTRAP"
        else:
            label = "UNCLASSIFIED — INSPECT"

        print(
            f"{path:<54} {label}"
        )

    print()
    rule()
    print(
        "LITERAL HTTP CALLS"
    )
    rule()

    if calls:
        for method, path in calls:
            print(
                f"{method:<7} {path}"
            )
    else:
        print(
            "NONE FOUND"
        )

    # Print substantial sanitized contexts around every unclassified endpoint.
    unclassified = [
        path
        for path in paths
        if (
            path not in KNOWN_NON_REPORT_ENDPOINTS
            and
            path not in PUBLIC_BOOTSTRAP_ENDPOINTS
        )
    ]

    if unclassified:
        print()
        rule()
        print(
            "UNCLASSIFIED ENDPOINT CONTEXTS"
        )
        rule()

        for path in unclassified:
            print_context_group(
                bundle,
                f'"{path}"',
            )

            print_context_group(
                bundle,
                f"'{path}'",
            )

    print()
    rule()
    print(
        "PUBLIC / REPORT FLOW CONTEXTS"
    )
    rule()

    for marker in MARKERS:
        print_context_group(
            bundle,
            marker,
        )

    strings = interesting_string_literals(
        bundle
    )

    print()
    rule()
    print(
        "REPORT-RELATED STRING LITERALS"
    )
    rule()

    for value in strings:
        print(
            value
        )

    print()
    rule()
    print(
        "AUDIT COMPLETE"
    )
    rule()

    print(
        "No application API endpoint was called by this audit."
    )

    print(
        "Any key/token-like value found in the JavaScript output was redacted."
    )

    print()
    print(
        "Paste this output back into the project chat."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
