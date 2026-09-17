#!/usr/bin/env python3
"""
CFB Edge Engine — Official Availability Source Audit V1

Read-only discovery audit for official conference availability-report pages.

It:
- downloads public conference pages
- finds public iframe and JavaScript asset URLs
- downloads those public assets
- scans for likely JSON/API/data endpoints and service domains
- prints a compact discovery report

It does NOT:
- log in
- bypass authentication
- submit forms
- modify local databases
- write project data
- make sportsbook or CFBD calls
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests


WIDTH = 118
TIMEOUT_SECONDS = 20
MAX_ASSET_BYTES = 4_000_000
MAX_SCRIPTS_PER_PAGE = 8
MAX_HINTS_PER_SOURCE = 40

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
}

SOURCES = [
    {
        "conference": "ACC",
        "kind": "REPORTING_PAGE",
        "url": "https://theacc.com/sports/2025/8/28/availability-reporting-football.aspx",
    },
    {
        "conference": "BIG12",
        "kind": "REPORTING_PAGE",
        "url": "https://big12sports.com/sports/2025/8/14/FBreporting.aspx",
    },
    {
        "conference": "BIGTEN",
        "kind": "REPORTING_POLICY_2026",
        "url": "https://bigten.org/fb/article/60284/",
    },
    {
        "conference": "BIGTEN",
        "kind": "KNOWN_REPORT_APP_HOST_PAGE",
        "url": "https://bigten.org/fb/article/37919/",
    },
    {
        "conference": "SEC",
        "kind": "REPORTING_PAGE",
        "url": "https://www.secsports.com/fbreports",
    },
]

ABSOLUTE_URL_PATTERN = re.compile(
    r"""https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"""
)

QUOTED_ENDPOINT_PATTERN = re.compile(
    r"""["'](
        /(?:api|graphql|reports?|availability|injur(?:y|ies)|players?|games?|teams?|data)
        [A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%-]*
    )["']""",
    re.IGNORECASE | re.VERBOSE,
)

SERVICE_KEYWORDS = (
    "api",
    "graphql",
    "firebase",
    "firestore",
    "supabase",
    "amazonaws",
    "cloudfront",
    "azure",
    "hdintelligence",
    "availability",
    "injury",
    "report",
)


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.casefold()
    except Exception:
        return ""


class AssetParser(HTMLParser):

    def __init__(self) -> None:
        super().__init__()
        self.iframes: list[str] = []
        self.scripts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attrs_map = {
            key.casefold(): value
            for key, value in attrs
            if key
        }

        tag = tag.casefold()

        if tag == "iframe":
            src = attrs_map.get("src")
            if src:
                self.iframes.append(src)

        elif tag == "script":
            src = attrs_map.get("src")
            if src:
                self.scripts.append(src)

        elif tag == "link":
            href = attrs_map.get("href")
            rel = clean(attrs_map.get("rel")).casefold()

            if href and (
                "preload" in rel
                or
                "modulepreload" in rel
                or
                "script" in rel
            ):
                self.links.append(href)


def bounded_get(
    session: requests.Session,
    url: str,
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
            chunk_size=64 * 1024,
        ):
            if not chunk:
                continue

            remaining = MAX_ASSET_BYTES - total

            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
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


def parse_assets(
    base_url: str,
    text: str,
) -> tuple[list[str], list[str]]:
    parser = AssetParser()

    try:
        parser.feed(text)
    except Exception:
        pass

    iframes = []

    for value in parser.iframes:
        url = urljoin(
            base_url,
            html.unescape(value),
        )

        if url not in iframes:
            iframes.append(url)

    scripts = []

    for value in parser.scripts + parser.links:
        url = urljoin(
            base_url,
            html.unescape(value),
        )

        if url not in scripts:
            scripts.append(url)

    return iframes, scripts


def endpoint_hints(
    text: str,
) -> list[str]:
    hints = []

    for match in ABSOLUTE_URL_PATTERN.findall(text):
        candidate = match.rstrip("\"'`,);]}>")

        lowered = candidate.casefold()

        if any(
            keyword in lowered
            for keyword in SERVICE_KEYWORDS
        ):
            if candidate not in hints:
                hints.append(candidate)

    for match in QUOTED_ENDPOINT_PATTERN.finditer(text):
        candidate = match.group(1)

        if candidate not in hints:
            hints.append(candidate)

    config_patterns = (
        r"\b(?:API|GRAPHQL|BASE|BACKEND|SERVICE|REPORT)[A-Z0-9_]{2,}\b",
        r"\bREACT_APP_[A-Z0-9_]+\b",
        r"\bVITE_[A-Z0-9_]+\b",
    )

    for pattern in config_patterns:
        for match in re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            candidate = f"CONFIG_TOKEN:{match}"

            if candidate not in hints:
                hints.append(candidate)

    return hints[:MAX_HINTS_PER_SOURCE]


def inspect_asset(
    session: requests.Session,
    url: str,
    label: str,
) -> dict[str, object]:
    response, content = bounded_get(
        session,
        url,
    )

    if response is None:
        return {
            "label": label,
            "url": url,
            "ok": False,
            "error": content,
            "status": None,
            "content_type": None,
            "bytes_scanned": 0,
            "iframes": [],
            "scripts": [],
            "hints": [],
        }

    text = content or ""

    content_type = clean(
        response.headers.get("content-type")
    )

    iframes = []
    scripts = []

    if (
        "html" in content_type.casefold()
        or
        "<html" in text[:2000].casefold()
        or
        "<script" in text[:5000].casefold()
    ):
        iframes, scripts = parse_assets(
            response.url,
            text,
        )

    return {
        "label": label,
        "url": url,
        "final_url": response.url,
        "ok": response.ok,
        "status": response.status_code,
        "content_type": content_type,
        "bytes_scanned": len(
            text.encode(
                "utf-8",
                errors="ignore",
            )
        ),
        "iframes": iframes,
        "scripts": scripts,
        "hints": endpoint_hints(text),
    }


def print_result(
    result: dict[str, object],
    indent: str = "",
) -> None:
    print(
        f"{indent}{result.get('label')}:"
    )

    print(
        f"{indent}  URL:       {result.get('url')}"
    )

    if not result.get("ok"):
        print(
            f"{indent}  STATUS:    FAILED"
        )

        print(
            f"{indent}  ERROR:     {result.get('error')}"
        )

        return

    print(
        f"{indent}  STATUS:    HTTP {result.get('status')}"
    )

    final_url = clean(
        result.get("final_url")
    )

    if (
        final_url
        and
        final_url
        !=
        clean(result.get("url"))
    ):
        print(
            f"{indent}  FINAL URL: {final_url}"
        )

    print(
        f"{indent}  TYPE:      {result.get('content_type') or 'UNKNOWN'}"
    )

    print(
        f"{indent}  SCANNED:   {result.get('bytes_scanned')} bytes"
    )

    iframes = result.get("iframes") or []
    scripts = result.get("scripts") or []
    hints = result.get("hints") or []

    if iframes:
        print(
            f"{indent}  IFRAMES:"
        )

        for url in iframes:
            print(
                f"{indent}    {url}"
            )

    if scripts:
        print(
            f"{indent}  SCRIPT ASSETS: {len(scripts)}"
        )

        for url in scripts[:MAX_SCRIPTS_PER_PAGE]:
            print(
                f"{indent}    {url}"
            )

        if len(scripts) > MAX_SCRIPTS_PER_PAGE:
            print(
                f"{indent}    ... "
                f"{len(scripts) - MAX_SCRIPTS_PER_PAGE} more"
            )

    if hints:
        print(
            f"{indent}  DATA/API HINTS:"
        )

        for hint in hints:
            print(
                f"{indent}    {hint}"
            )


def inspect_source(
    session: requests.Session,
    source: dict[str, str],
) -> None:
    rule()
    print(
        f"{source['conference']} — {source['kind']}"
    )
    rule()

    page = inspect_asset(
        session,
        source["url"],
        "OFFICIAL PAGE",
    )

    print_result(page)

    if not page.get("ok"):
        return

    iframe_urls = list(
        page.get("iframes")
        or
        []
    )

    script_urls = list(
        page.get("scripts")
        or
        []
    )

    for index, iframe_url in enumerate(
        iframe_urls,
        start=1,
    ):
        print()

        iframe_result = inspect_asset(
            session,
            iframe_url,
            f"IFRAME {index}",
        )

        print_result(
            iframe_result,
            indent="  ",
        )

        for script_url in (
            iframe_result.get("scripts")
            or
            []
        ):
            if script_url not in script_urls:
                script_urls.append(script_url)

    if script_urls:
        print()
        print(
            "  JAVASCRIPT DISCOVERY"
        )

        print(
            "  "
            +
            "-" * (WIDTH - 2)
        )

    service_domains = set()

    for index, script_url in enumerate(
        script_urls[:MAX_SCRIPTS_PER_PAGE],
        start=1,
    ):
        result = inspect_asset(
            session,
            script_url,
            f"JS {index}",
        )

        if not result.get("ok"):
            print(
                f"  JS {index}: FAILED {script_url}"
            )
            continue

        hints = result.get("hints") or []

        if not hints:
            continue

        print(
            f"  JS {index}: {script_url}"
        )

        for hint in hints:
            print(
                f"    {hint}"
            )

            if hint.startswith("http"):
                domain = domain_of(hint)

                if domain:
                    service_domains.add(domain)

    if service_domains:
        print()
        print(
            "  DISCOVERED SERVICE DOMAINS:"
        )

        for domain in sorted(service_domains):
            print(
                f"    {domain}"
            )


def main() -> int:
    rule()
    print(
        "CFB EDGE ENGINE — OFFICIAL AVAILABILITY SOURCE AUDIT V1"
    )
    rule()

    print(
        "Mode: READ ONLY / PUBLIC WEB DISCOVERY / NO DATABASE WRITES"
    )

    print(
        "Purpose: identify the public report-data transport used by official conference pages."
    )

    print(
        f"Timeout: {TIMEOUT_SECONDS}s/request | max {MAX_ASSET_BYTES:,} bytes/asset"
    )

    print()

    session = requests.Session()

    completed = 0

    for source in SOURCES:
        try:
            inspect_source(
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
                f"{source['conference']} AUDIT ERROR: {error}"
            )
            rule("!")

    print()
    rule()
    print(
        "AUDIT COMPLETE"
    )
    rule()

    print(
        f"Source pages attempted: {len(SOURCES)}"
    )

    print(
        f"Audit blocks completed: {completed}"
    )

    print()
    print(
        "Paste this output back into the project chat."
    )

    print()
    print(
        "The next build will use only confirmed public endpoints/structures."
    )

    print(
        "If an official site exposes only a browser-rendered application with "
        "no stable public data contract, that conference will remain behind a "
        "manual/provider adapter instead of a brittle guessed scraper."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
