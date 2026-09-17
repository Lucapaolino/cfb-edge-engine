#!/usr/bin/env python3
"""
CFB Edge Engine — PublishScreen Transport Audit V4

Purpose
-------
The V3 audit proved that the main application bundle does NOT contain the
report-data endpoint itself. Instead, the public report view lazy-loads a
separate Vite chunk named PublishScreen-*.js.

This audit follows only that confirmed static path:

    public app shell
        -> current main bundle
        -> current PublishScreen-*.js chunk
        -> directly imported same-origin JS chunks

It then statically inspects those JavaScript files for literal /api/... paths
and HTTP calls.

Safety
------
- GET requests only
- no application API POST/GET calls
- no login
- no credentials
- no auth bypass
- no database writes
- no project-data writes
- no CFBD calls
- no sportsbook calls
- token/key-like values in printed contexts are redacted

Run
---
python src/audit_publishscreen_transport_v4.py
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests


WIDTH = 118
TIMEOUT_SECONDS = 25
MAX_FILE_BYTES = 12_000_000
MAX_IMPORTED_CHUNKS = 12
CONTEXT_RADIUS = 1400

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

ENDPOINT_HINTS = (
    "report",
    "publish",
    "public",
    "availability",
    "injury",
    "player",
    "team",
    "status",
    "game",
    "window",
    "archive",
)

MARKERS = (
    "/api/",
    ".post(",
    ".get(",
    "axios",
    "fetch(",
    "published",
    "availability",
    "injury",
    "player",
    "status",
    "conference",
    "sport",
)


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


def same_origin(left: str, right: str) -> bool:
    try:
        return (
            urlparse(left).netloc.casefold()
            ==
            urlparse(right).netloc.casefold()
        )
    except Exception:
        return False


def sanitize(text: str) -> str:
    patterns = (
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
    )

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
    *,
    max_bytes: int = MAX_FILE_BYTES,
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


def discover_main_bundle(
    page_url: str,
    page_text: str,
) -> str | None:
    parser = ScriptParser()

    try:
        parser.feed(page_text)
    except Exception:
        pass

    candidates = []

    for src in parser.scripts:
        full = urljoin(
            page_url,
            html.unescape(src),
        )

        if same_origin(
            full,
            page_url,
        ):
            candidates.append(full)

    if not candidates:
        return None

    candidates.sort(
        key=lambda url: (
            "/assets/" not in url,
            len(url),
        )
    )

    return candidates[0]


def discover_publish_chunk(
    main_bundle_url: str,
    main_bundle: str,
) -> str | None:
    patterns = (
        r'["\'](\./PublishScreen-[A-Za-z0-9_-]+\.js)["\']',
        r'["\'](PublishScreen-[A-Za-z0-9_-]+\.js)["\']',
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            main_bundle,
        )

        if match:
            return urljoin(
                main_bundle_url,
                match.group(1),
            )

    return None


def imported_js_chunks(
    base_url: str,
    text: str,
) -> list[str]:
    candidates = []

    patterns = (
        r'from\s*["\'](\./[^"\']+\.js)["\']',
        r'import\(\s*["\'](\./[^"\']+\.js)["\']\s*\)',
        r'["\'](\./[A-Za-z0-9_.-]+\.js)["\']',
    )

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
        ):
            full = urljoin(
                base_url,
                match.group(1),
            )

            if not same_origin(
                full,
                base_url,
            ):
                continue

            if full not in candidates:
                candidates.append(full)

    return candidates[:MAX_IMPORTED_CHUNKS]


def literal_api_paths(
    text: str,
) -> list[str]:
    paths = set()

    patterns = (
        r'["\'](/api/[A-Za-z0-9_$./?&=%:{}-]+)["\']',
        r'`(/api/[A-Za-z0-9_$./?&=%:{}-]+)`',
    )

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
        ):
            paths.add(
                match.group(1)
            )

    return sorted(paths)


def http_calls(
    text: str,
) -> list[tuple[str, str]]:
    calls = set()

    # axios-like .post("/api/foo", ...)
    pattern = re.compile(
        r'\.(get|post|put|patch|delete)\(\s*["\']'
        r'(/api/[A-Za-z0-9_$./?&=%:{}-]+)["\']',
        re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        calls.add(
            (
                match.group(1).upper(),
                match.group(2),
            )
        )

    # fetch("/api/foo", ...)
    fetch_pattern = re.compile(
        r'fetch\(\s*["\']'
        r'(/api/[A-Za-z0-9_$./?&=%:{}-]+)["\']',
        re.IGNORECASE,
    )

    for match in fetch_pattern.finditer(text):
        calls.add(
            (
                "FETCH",
                match.group(1),
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
    *,
    radius: int = CONTEXT_RADIUS,
    limit: int = 6,
) -> list[str]:
    output = []
    start_at = 0

    while len(output) < limit:
        index = text.find(
            needle,
            start_at,
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

        snippet = sanitize(
            text[start:end]
        )

        if snippet not in output:
            output.append(snippet)

        start_at = (
            index
            +
            len(needle)
        )

    return output


def print_file_analysis(
    label: str,
    url: str,
    text: str,
) -> None:
    rule()
    print(label)
    rule()

    print(
        f"URL: {url}"
    )

    print(
        f"Characters scanned: {len(text):,}"
    )

    paths = literal_api_paths(
        text
    )

    calls = http_calls(
        text
    )

    print()
    print(
        "LITERAL API PATHS"
    )

    rule("-")

    if paths:
        for path in paths:
            print(
                f"  {path}"
            )
    else:
        print(
            "  NONE"
        )

    print()
    print(
        "LITERAL HTTP CALLS"
    )

    rule("-")

    if calls:
        for method, path in calls:
            print(
                f"  {method:<7} {path}"
            )
    else:
        print(
            "  NONE"
        )

    # Show endpoint contexts first.
    for path in paths:
        print()
        print(
            f"CONTEXT: {path}"
        )

        rule("-")

        for index, snippet in enumerate(
            contexts(
                text,
                path,
            ),
            start=1,
        ):
            print(
                f"[{index}]"
            )
            print(snippet)
            print()

    # Then selected semantic contexts.
    for marker in MARKERS:
        found = contexts(
            text,
            marker,
            radius=900,
            limit=3,
        )

        if not found:
            continue

        # Avoid dumping generic framework content unless it sits near something
        # report-related.
        if marker in {
            ".get(",
            ".post(",
            "axios",
            "fetch(",
        }:
            interesting = any(
                any(
                    hint in snippet.casefold()
                    for hint in ENDPOINT_HINTS
                )
                for snippet in found
            )

            if not interesting:
                continue

        print()
        print(
            f"MARKER CONTEXT: {marker}"
        )

        rule("-")

        for index, snippet in enumerate(
            found,
            start=1,
        ):
            print(
                f"[{index}] {snippet}"
            )


def main() -> int:
    rule()

    print(
        "CFB EDGE ENGINE — PUBLISHSCREEN TRANSPORT AUDIT V4"
    )

    rule()

    print(
        "Mode: STATIC JS INSPECTION / GET ONLY / NO APPLICATION API CALLS"
    )

    print(
        "Goal: inspect the lazy-loaded public Published Report component."
    )

    print()

    session = requests.Session()

    page_response, page_text = get_text(
        session,
        APP_URL,
        max_bytes=2_000_000,
    )

    if page_response is None:
        print(
            f"APP PAGE FAILED: {page_text}"
        )
        return 1

    print(
        f"App page HTTP: {page_response.status_code}"
    )

    main_bundle_url = discover_main_bundle(
        page_response.url,
        page_text or "",
    )

    if not main_bundle_url:
        print(
            "Main bundle not found."
        )
        return 1

    print(
        f"Main bundle: {main_bundle_url}"
    )

    bundle_response, main_bundle = get_text(
        session,
        main_bundle_url,
    )

    if bundle_response is None:
        print(
            f"MAIN BUNDLE FAILED: {main_bundle}"
        )
        return 1

    publish_url = discover_publish_chunk(
        bundle_response.url,
        main_bundle or "",
    )

    if not publish_url:
        print(
            "PublishScreen chunk not found in current main bundle."
        )
        return 1

    print(
        f"PublishScreen chunk: {publish_url}"
    )

    publish_response, publish_text = get_text(
        session,
        publish_url,
    )

    if publish_response is None:
        print(
            f"PUBLISH CHUNK FAILED: {publish_text}"
        )
        return 1

    publish_text = publish_text or ""

    print_file_analysis(
        "PUBLISHSCREEN CHUNK",
        publish_response.url,
        publish_text,
    )

    imported = imported_js_chunks(
        publish_response.url,
        publish_text,
    )

    if imported:
        print()
        rule()
        print(
            "DIRECTLY IMPORTED JS CHUNKS"
        )
        rule()

        for url in imported:
            print(
                f"  {url}"
            )

        for index, imported_url in enumerate(
            imported,
            start=1,
        ):
            response, text = get_text(
                session,
                imported_url,
            )

            if response is None:
                print()
                print(
                    f"IMPORT {index} FAILED: {imported_url}"
                )

                print(
                    f"  {text}"
                )

                continue

            text = text or ""

            # Only print full analysis if the imported chunk appears relevant.
            lowered = text.casefold()

            relevant = (
                "/api/" in lowered
                or
                any(
                    hint in lowered
                    for hint in ENDPOINT_HINTS
                )
            )

            if relevant:
                print_file_analysis(
                    f"IMPORTED CHUNK {index}",
                    response.url,
                    text,
                )

    print()
    rule()

    print(
        "AUDIT COMPLETE"
    )

    rule()

    print(
        "No application API endpoint was called."
    )

    print(
        "Only the public app shell and static JavaScript assets were downloaded."
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
