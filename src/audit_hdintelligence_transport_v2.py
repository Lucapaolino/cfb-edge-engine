#!/usr/bin/env python3
"""
CFB Edge Engine — HD Intelligence Transport Audit V2

Purpose
-------
Focused read-only discovery of the PUBLIC data transport used by conference
availability-report applications.

The first source audit confirmed:
- ACC -> app.hdintelligence.com
- Big 12 -> app.hdintelligence.com
- SEC page references app.hdintelligence.com
- Big Ten historical report page -> confinjrepxyz.hdintelligence-app.com

This V2 directly inspects those public app shells and their CURRENT JavaScript
bundles instead of spending the scan budget on conference-site analytics assets.

Safety
------
- GET requests only
- no login
- no auth bypass
- no form submission
- no database writes
- no project data writes
- no CFBD calls
- no sportsbook calls
- no endpoint guessing/brute force

Run
---
python src/audit_hdintelligence_transport_v2.py
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests


WIDTH = 118
TIMEOUT_SECONDS = 25
MAX_BUNDLE_BYTES = 12_000_000
MAX_CONTEXTS_PER_TERM = 12

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

APPS = [
    {
        "label": "ACC",
        "url": (
            "https://app.hdintelligence.com/"
            "?source=ACC&sport=Football&conf=ACC&type=report"
        ),
    },
    {
        "label": "BIG12",
        "url": (
            "https://app.hdintelligence.com/"
            "?source=B12&sport=Football&conf=B12&type=report"
        ),
    },
    {
        "label": "SEC",
        "url": (
            "https://app.hdintelligence.com/"
            "?source=SEC&sport=Football&conf=SEC&type=report"
        ),
    },
    {
        "label": "BIGTEN_CURRENT_HOST",
        "url": (
            "https://confinjrepxyz.hdintelligence-app.com/"
            "?source=B10reports"
        ),
    },
    {
        "label": "BIGTEN_ARCHIVE_HOST",
        "url": (
            "https://confinjrepxyz.hdintelligence-app.com/"
            "?source=B10archive"
        ),
    },
]

ABSOLUTE_URL_RE = re.compile(
    r"""https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"""
)

RELATIVE_ENDPOINT_RE = re.compile(
    r"""["'](
        /[A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%-]{2,}
    )["']""",
    re.VERBOSE,
)

HOSTLIKE_RE = re.compile(
    r"""(?:[A-Za-z0-9-]+\.)+
        (?:com|net|org|io|app|dev|ai|cloud|co)
    """,
    re.IGNORECASE | re.VERBOSE,
)

INTERESTING_TERMS = (
    "baseURL",
    "baseUrl",
    "axios.create",
    "axios.get",
    "axios.post",
    "fetch(",
    "XMLHttpRequest",
    "/api/",
    "/report",
    "report/",
    "availability",
    "injury",
    "status",
    "source=",
    "conf=",
    "sport=",
    "B10reports",
    "B10archive",
)

ENDPOINT_KEYWORDS = (
    "api",
    "report",
    "availability",
    "injury",
    "player",
    "team",
    "game",
    "conference",
    "football",
    "hdintelligence",
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

        attr_map = {
            key.casefold(): value
            for key, value in attrs
            if key
        }

        src = attr_map.get("src")

        if src:
            self.scripts.append(src)


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def clean(value: object) -> str:
    return " ".join(
        str(value or "").strip().split()
    )


def origin(url: str) -> str:
    parsed = urlparse(url)

    return (
        f"{parsed.scheme}://{parsed.netloc}"
        if parsed.scheme and parsed.netloc
        else
        ""
    )


def same_origin(left: str, right: str) -> bool:
    try:
        return (
            urlparse(left).netloc.casefold()
            ==
            urlparse(right).netloc.casefold()
        )
    except Exception:
        return False


def get_text(
    session: requests.Session,
    url: str,
    *,
    max_bytes: int,
) -> tuple[
    requests.Response | None,
    str | None,
]:
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


def script_urls(
    page_url: str,
    html_text: str,
) -> list[str]:
    parser = ScriptParser()

    try:
        parser.feed(
            html_text
        )
    except Exception:
        pass

    result = []

    for src in parser.scripts:
        full = urljoin(
            page_url,
            html.unescape(src),
        )

        if full not in result:
            result.append(full)

    return result


def context_snippets(
    text: str,
    term: str,
    *,
    radius: int = 220,
) -> list[str]:
    snippets = []

    lowered = text.casefold()
    needle = term.casefold()
    offset = 0

    while True:
        index = lowered.find(
            needle,
            offset,
        )

        if index < 0:
            break

        start = max(
            0,
            index - radius,
        )

        end = min(
            len(text),
            index + len(term) + radius,
        )

        snippet = text[
            start:end
        ]

        snippet = re.sub(
            r"\s+",
            " ",
            snippet,
        ).strip()

        if snippet not in snippets:
            snippets.append(
                snippet
            )

        if len(snippets) >= MAX_CONTEXTS_PER_TERM:
            break

        offset = index + len(term)

    return snippets


def likely_urls(
    text: str,
    app_url: str,
) -> list[str]:
    found = []

    for match in ABSOLUTE_URL_RE.findall(
        text
    ):
        candidate = match.rstrip(
            "\"'`,);]}>"
        )

        lowered = candidate.casefold()

        if any(
            keyword in lowered
            for keyword in ENDPOINT_KEYWORDS
        ):
            if candidate not in found:
                found.append(
                    candidate
                )

    app_origin = origin(
        app_url
    )

    for match in RELATIVE_ENDPOINT_RE.finditer(
        text
    ):
        relative = match.group(
            1
        )

        lowered = relative.casefold()

        if not any(
            keyword in lowered
            for keyword in ENDPOINT_KEYWORDS
        ):
            continue

        absolute = urljoin(
            app_origin + "/",
            relative,
        )

        if absolute not in found:
            found.append(
                absolute
            )

    return found


def host_hints(
    text: str,
) -> list[str]:
    results = []

    for match in HOSTLIKE_RE.findall(
        text
    ):
        host = match.casefold()

        if any(
            keyword in host
            for keyword in (
                "hdintelligence",
                "amazonaws",
                "cloudfront",
                "firebase",
                "supabase",
                "azure",
                "api",
            )
        ):
            if host not in results:
                results.append(
                    host
                )

    return results


def print_bundle_analysis(
    app_label: str,
    app_url: str,
    bundle_url: str,
    text: str,
) -> None:
    rule("-")

    print(
        f"{app_label} BUNDLE"
    )

    rule("-")

    print(
        f"Bundle: {bundle_url}"
    )

    print(
        f"Characters scanned: {len(text):,}"
    )

    urls = likely_urls(
        text,
        app_url,
    )

    hosts = host_hints(
        text
    )

    if urls:
        print()
        print(
            "LIKELY DATA / API URLS"
        )

        for url in urls[
            :80
        ]:
            print(
                f"  {url}"
            )

    else:
        print()
        print(
            "LIKELY DATA / API URLS: NONE FOUND"
        )

    if hosts:
        print()
        print(
            "SERVICE HOST HINTS"
        )

        for host in hosts:
            print(
                f"  {host}"
            )

    print()
    print(
        "TRANSPORT CONTEXTS"
    )

    any_context = False

    for term in INTERESTING_TERMS:
        snippets = context_snippets(
            text,
            term,
        )

        if not snippets:
            continue

        any_context = True

        print()
        print(
            f"TERM: {term}"
        )

        for index, snippet in enumerate(
            snippets,
            start=1,
        ):
            print(
                f"  [{index}] {snippet}"
            )

    if not any_context:
        print(
            "  No selected transport terms found."
        )


def inspect_app(
    session: requests.Session,
    app: dict[str, str],
) -> None:
    rule()
    print(
        app["label"]
    )
    rule()

    print(
        f"App URL: {app['url']}"
    )

    response, page_text = get_text(
        session,
        app["url"],
        max_bytes=2_000_000,
    )

    if response is None:
        print(
            f"FAILED: {page_text}"
        )
        return

    print(
        f"HTTP: {response.status_code}"
    )

    print(
        f"Final URL: {response.url}"
    )

    print(
        f"Content-Type: {clean(response.headers.get('content-type'))}"
    )

    scripts = script_urls(
        response.url,
        page_text or "",
    )

    same_origin_scripts = [
        script
        for script in scripts
        if same_origin(
            script,
            response.url,
        )
    ]

    print(
        f"Script assets: {len(scripts)} "
        f"({len(same_origin_scripts)} same-origin)"
    )

    for script in scripts:
        print(
            f"  {script}"
        )

    targets = (
        same_origin_scripts
        if same_origin_scripts
        else
        scripts
    )

    if not targets:
        print(
            "No JavaScript bundle discovered."
        )
        return

    for bundle_url in targets:
        bundle_response, bundle_text = get_text(
            session,
            bundle_url,
            max_bytes=MAX_BUNDLE_BYTES,
        )

        if bundle_response is None:
            print()
            print(
                f"BUNDLE FAILED: {bundle_url}"
            )

            print(
                f"  {bundle_text}"
            )

            continue

        print_bundle_analysis(
            app["label"],
            response.url,
            bundle_response.url,
            bundle_text or "",
        )


def main() -> int:
    rule()

    print(
        "CFB EDGE ENGINE — HD INTELLIGENCE TRANSPORT AUDIT V2"
    )

    rule()

    print(
        "Mode: READ ONLY / PUBLIC GET REQUESTS / NO DATABASE WRITES"
    )

    print(
        "Focus: inspect the availability-report app bundles directly."
    )

    print(
        "No guessed endpoints will be requested."
    )

    print()

    session = requests.Session()

    completed = 0

    for app in APPS:
        try:
            inspect_app(
                session,
                app,
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
                f"{app['label']} ERROR: {error}"
            )
            rule("!")

    print()
    rule()

    print(
        "AUDIT COMPLETE"
    )

    rule()

    print(
        f"Apps inspected: {completed}/{len(APPS)}"
    )

    print()
    print(
        "Paste this output back into the project chat."
    )

    print()
    print(
        "If the bundle exposes a stable public read endpoint, the next file "
        "will be the actual official availability collector."
    )

    print(
        "If it does not, we will keep the immutable personnel backend and use "
        "a provider/manual adapter rather than fabricate a brittle scraper."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
