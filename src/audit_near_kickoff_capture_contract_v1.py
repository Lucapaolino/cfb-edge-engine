#!/usr/bin/env python3
"""
CFB Edge Engine — Near-Kickoff Capture Contract Audit V1

Purpose
-------
Inspect the CURRENT local production files that govern:
- the canonical Game Day controller
- prediction-history logging
- prediction-history storage
- market snapshots
- live board assembly

This audit exists so the next lightweight near-kickoff market-capture script
can reuse the exact production contracts without guessing logger arguments or
reimplementing immutable-history behavior incorrectly.

Safety
------
- NO API calls
- NO subprocesses
- NO imports of project modules
- NO database writes
- NO file writes except this script's normal terminal output
- only parses local Python source as text / AST

Run
---
python src/audit_near_kickoff_capture_contract_v1.py
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

FILES = {
    "controller": SRC / "run_game_day_v1.py",
    "logger": SRC / "log_predictions_v1.py",
    "history_core": SRC / "prediction_history_v1.py",
    "reference_market": SRC / "snapshot_execution_market_2026_v1.py",
    "hardrock": SRC / "snapshot_hardrock_oddspapi_2026_v1.py",
    "board": SRC / "build_live_model_market_board_v1.py",
}

WIDTH = 112


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_source(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def line_number(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def source_excerpt(
    source: str,
    *,
    start_line: int,
    end_line: int,
) -> str:
    lines = source.splitlines()
    start = max(1, start_line)
    end = min(len(lines), end_line)

    rendered = []

    for number in range(start, end + 1):
        rendered.append(
            f"{number:>5}: {lines[number - 1]}"
        )

    return "\n".join(rendered)


def find_excerpt(
    source: str,
    pattern: str,
    *,
    before: int = 5,
    after: int = 12,
    regex: bool = False,
) -> list[tuple[int, str]]:
    matches = []

    if regex:
        iterator = re.finditer(
            pattern,
            source,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    else:
        iterator = re.finditer(
            re.escape(pattern),
            source,
            flags=re.IGNORECASE,
        )

    for match in iterator:
        number = line_number(
            source,
            match.start(),
        )

        matches.append(
            (
                number,
                source_excerpt(
                    source,
                    start_line=number - before,
                    end_line=number + after,
                ),
            )
        )

    return matches


def parse_tree(source: str, path: Path) -> ast.AST:
    try:
        return ast.parse(
            source,
            filename=str(path),
        )
    except SyntaxError as error:
        raise RuntimeError(
            f"Could not parse {path.name}: {error}"
        ) from error


def function_names(tree: ast.AST) -> list[str]:
    names = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            names.append(
                node.name
            )

    return sorted(
        set(
            names
        )
    )


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        left = call_name(
            node.value
        )

        if left:
            return f"{left}.{node.attr}"

        return node.attr

    return ""


def literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(
            node
        )
    except Exception:
        return None


def argparse_arguments(tree: ast.AST) -> list[dict[str, Any]]:
    results = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        name = call_name(
            node.func
        )

        if not name.endswith(
            ".add_argument"
        ):
            continue

        positional = []

        for argument in node.args:
            value = literal_value(
                argument
            )

            if value is not None:
                positional.append(
                    value
                )

        keywords = {}

        for keyword in node.keywords:
            if keyword.arg is None:
                continue

            value = literal_value(
                keyword.value
            )

            if value is not None:
                keywords[
                    keyword.arg
                ] = value

            else:
                keywords[
                    keyword.arg
                ] = ast.unparse(
                    keyword.value
                )

        results.append(
            {
                "args": positional,
                "kwargs": keywords,
                "line": getattr(
                    node,
                    "lineno",
                    None,
                ),
            }
        )

    return sorted(
        results,
        key=lambda item: (
            item.get("line")
            or
            0
        ),
    )


def subprocess_calls(tree: ast.AST) -> list[dict[str, Any]]:
    results = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        name = call_name(
            node.func
        )

        if name not in {
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
        }:
            continue

        rendered_args = []

        for arg in node.args:
            try:
                rendered_args.append(
                    ast.unparse(
                        arg
                    )
                )
            except Exception:
                rendered_args.append(
                    "<unparseable>"
                )

        results.append(
            {
                "call": name,
                "line": getattr(
                    node,
                    "lineno",
                    None,
                ),
                "args": rendered_args,
            }
        )

    return sorted(
        results,
        key=lambda item: (
            item.get("line")
            or
            0
        ),
    )


def assigned_paths(
    tree: ast.AST,
    source: str,
) -> list[tuple[str, str, int]]:
    interesting = re.compile(
        r"(BOARD|HISTORY|SNAPSHOT|BATCH|JSONL|LEDGER|DB|MARKET|ODDS|HARDROCK)",
        re.IGNORECASE,
    )

    results = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue

        targets = []

        if isinstance(
            node,
            ast.Assign,
        ):
            targets = node.targets
            value_node = node.value

        else:
            targets = [
                node.target
            ]
            value_node = node.value

        if value_node is None:
            continue

        try:
            value_text = ast.get_source_segment(
                source,
                value_node,
            ) or ast.unparse(
                value_node
            )
        except Exception:
            continue

        for target in targets:
            if isinstance(
                target,
                ast.Name,
            ):
                name = target.id

                if interesting.search(
                    name
                ):
                    results.append(
                        (
                            name,
                            " ".join(
                                value_text.split()
                            ),
                            getattr(
                                node,
                                "lineno",
                                0,
                            ),
                        )
                    )

    return sorted(
        results,
        key=lambda item: item[
            2
        ],
    )


def print_file_inventory() -> dict[str, tuple[Path, str, ast.AST]]:
    rule()
    print(
        "CURRENT LOCAL PRODUCTION FILES"
    )
    rule()

    loaded = {}

    for key, path in FILES.items():
        if not path.exists():
            print(
                f"MISSING  {key:<18} {path}"
            )
            continue

        source = read_source(
            path
        )

        tree = parse_tree(
            source,
            path,
        )

        loaded[
            key
        ] = (
            path,
            source,
            tree,
        )

        print(
            f"PASS     {key:<18} "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

        print(
            f"         lines={len(source.splitlines())} "
            f"sha256={sha256(path)}"
        )

    return loaded


def print_controller_contract(
    loaded: dict[str, tuple[Path, str, ast.AST]],
) -> None:
    rule()
    print(
        "CANONICAL CONTROLLER CONTRACT"
    )
    rule()

    item = loaded.get(
        "controller"
    )

    if item is None:
        print(
            "Controller unavailable."
        )
        return

    path, source, tree = item

    print(
        f"File: {path.relative_to(PROJECT_ROOT)}"
    )

    print()
    print(
        "Relevant functions:"
    )

    functions = function_names(
        tree
    )

    for name in functions:
        if any(
            token in name.casefold()
            for token in (
                "run",
                "log",
                "history",
                "ledger",
                "market",
                "snapshot",
                "preflight",
            )
        ):
            print(
                f"  {name}"
            )

    print()
    print(
        "CLI arguments:"
    )

    args = argparse_arguments(
        tree
    )

    if args:
        for item in args:
            print(
                f"  L{item['line']}: "
                f"{item['args']} {item['kwargs']}"
            )
    else:
        print(
            "  None detected."
        )

    for token in (
        "log_predictions",
        "LOGGING PREDICTIONS",
        "score_prediction_history",
        "report_prediction_history",
        "reference_market",
        "hardrock",
        "ASSEMBLING LIVE BOARD",
    ):
        hits = find_excerpt(
            source,
            token,
            before=5,
            after=12,
        )

        if not hits:
            continue

        print()
        print(
            f"Excerpt(s) for: {token}"
        )
        rule(
            "-"
        )

        for number, excerpt in hits[
            :4
        ]:
            print(
                excerpt
            )
            print()


def print_logger_contract(
    loaded: dict[str, tuple[Path, str, ast.AST]],
) -> None:
    rule()
    print(
        "PREDICTION LOGGER CONTRACT"
    )
    rule()

    item = loaded.get(
        "logger"
    )

    if item is None:
        print(
            "Logger unavailable."
        )
        return

    path, source, tree = item

    print(
        f"File: {path.relative_to(PROJECT_ROOT)}"
    )

    print()
    print(
        "CLI arguments:"
    )

    args = argparse_arguments(
        tree
    )

    if args:
        for item in args:
            print(
                f"  L{item['line']}: "
                f"{item['args']} {item['kwargs']}"
            )
    else:
        print(
            "  None detected."
        )

    print()
    print(
        "Relevant functions:"
    )

    for name in function_names(
        tree
    ):
        if any(
            token in name.casefold()
            for token in (
                "main",
                "log",
                "run",
                "snapshot",
                "history",
                "load",
                "board",
            )
        ):
            print(
                f"  {name}"
            )

    print()
    print(
        "Relevant path/config assignments:"
    )

    for name, value, line in assigned_paths(
        tree,
        source,
    ):
        print(
            f"  L{line}: {name} = {value}"
        )

    for token in (
        "run_id",
        "sys.argv",
        "os.environ",
        "BOARD",
        "prediction_snapshots",
        "jsonl",
        "batch",
        "insert",
    ):
        hits = find_excerpt(
            source,
            token,
            before=5,
            after=14,
        )

        if not hits:
            continue

        print()
        print(
            f"Excerpt(s) for: {token}"
        )
        rule(
            "-"
        )

        for _number, excerpt in hits[
            :5
        ]:
            print(
                excerpt
            )
            print()


def print_history_contract(
    loaded: dict[str, tuple[Path, str, ast.AST]],
) -> None:
    rule()
    print(
        "PREDICTION HISTORY CORE CONTRACT"
    )
    rule()

    item = loaded.get(
        "history_core"
    )

    if item is None:
        print(
            "History core unavailable."
        )
        return

    path, source, tree = item

    print(
        f"File: {path.relative_to(PROJECT_ROOT)}"
    )

    print()
    print(
        "Relevant functions:"
    )

    for name in function_names(
        tree
    ):
        if any(
            token in name.casefold()
            for token in (
                "init",
                "insert",
                "snapshot",
                "history",
                "run",
                "write",
                "append",
                "record",
                "protect",
            )
        ):
            print(
                f"  {name}"
            )

    for token in (
        "CREATE TABLE",
        "prediction_snapshots",
        "CREATE TRIGGER",
        "INSERT INTO",
        "snapshot_id",
        "run_id",
    ):
        hits = find_excerpt(
            source,
            token,
            before=4,
            after=18,
        )

        if not hits:
            continue

        print()
        print(
            f"Excerpt(s) for: {token}"
        )
        rule(
            "-"
        )

        for _number, excerpt in hits[
            :4
        ]:
            print(
                excerpt
            )
            print()


def print_market_contracts(
    loaded: dict[str, tuple[Path, str, ast.AST]],
) -> None:
    rule()
    print(
        "MARKET + BOARD CONTRACTS"
    )
    rule()

    for key in (
        "reference_market",
        "hardrock",
        "board",
    ):
        item = loaded.get(
            key
        )

        if item is None:
            print(
                f"{key}: unavailable"
            )
            continue

        path, source, tree = item

        print()
        print(
            f"{key.upper()}: "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

        args = argparse_arguments(
            tree
        )

        if args:
            print(
                "  CLI arguments:"
            )

            for item in args:
                print(
                    f"    L{item['line']}: "
                    f"{item['args']} "
                    f"{item['kwargs']}"
                )

        else:
            print(
                "  CLI arguments: none detected"
            )

        subprocesses = subprocess_calls(
            tree
        )

        if subprocesses:
            print(
                "  Subprocess calls:"
            )

            for call in subprocesses:
                print(
                    f"    L{call['line']}: "
                    f"{call['call']} "
                    f"{call['args']}"
                )

        for token in (
            "if __name__",
            "def main",
            "write",
            "latest",
            "snapshot",
            "batch",
        ):
            hits = find_excerpt(
                source,
                token,
                before=3,
                after=8,
            )

            if hits:
                print(
                    f"  First excerpt for '{token}':"
                )

                print(
                    hits[
                        0
                    ][
                        1
                    ]
                )

                break


def print_decision() -> None:
    rule()
    print(
        "NEXT IMPLEMENTATION DECISION"
    )
    rule()

    print(
        "Paste this audit output back into the project chat."
    )

    print()

    print(
        "The next whole-file build will be a lightweight near-kickoff market "
        "capture workflow that:"
    )

    print(
        "  1. refreshes ONLY reference-market and Hard Rock prices,"
    )

    print(
        "  2. rebuilds the board against the already-frozen model prediction,"
    )

    print(
        "  3. writes a new immutable Prediction History snapshot using the "
        "CURRENT logger contract,"
    )

    print(
        "  4. refreshes Market History / CLV,"
    )

    print(
        "  5. never refreshes CFBD, team stats, player stats, context, features, "
        "or retrains/reruns the neural model."
    )

    print()

    print(
        "No lightweight capture script should be written until the exact current "
        "logger/controller contract above is confirmed."
    )


def main() -> int:
    rule()
    print(
        "CFB EDGE ENGINE — NEAR-KICKOFF CAPTURE CONTRACT AUDIT V1"
    )
    rule()

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        "Mode: READ ONLY / NO APIS / NO SUBPROCESSES / NO DATABASE WRITES"
    )

    loaded = print_file_inventory()

    print_controller_contract(
        loaded
    )

    print_logger_contract(
        loaded
    )

    print_history_contract(
        loaded
    )

    print_market_contracts(
        loaded
    )

    print_decision()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
