# NFL Research Collection Protocol V1

## Status

Collection-only foundation. No NFL model, prediction, signal, wager rule, or customer output exists in V1.

## Purpose

Create an auditable NFL source ledger while the CFB study remains frozen. The NFL track begins with two independent source captures:

- Schedule and results from `nflreadpy` / nflverse.
- Reference market observations for moneyline, spread, and total from The Odds API.

No player props are collected in this V1 scope.

## Separation rule

NFL V1 must not read or write any CFB database, prediction history, model artifact, board, GUI state, actual wager ledger, or CFB research result. CFB remains the authoritative frozen 2026 study environment.

## Evidence record

Each successful collection action stores:

- A UTC capture timestamp and unique snapshot ID.
- The unmodified source payload in a new NFL raw-snapshot file.
- A SHA-256 fingerprint of the source payload.
- A normalized row set in `data/nfl/prospective/nfl_research.sqlite`.
- Database triggers that reject updates and deletes for stored source records.

The raw source artifact and normalized database row are both retained. A later schedule correction or market change becomes a new snapshot; it does not rewrite the older one.

## Guardrails

- Live source calls require an explicit `--capture` or `--capture-*` flag.
- Default commands are dry runs with no source call and no file/database write.
- The NFL market source is labelled reference market data. It is not an execution-book substitute.
- NFL collection cannot produce picks, signals, probability claims, or wagers.
- Any future NFL model must be developed and validated as a separate model, never copied from the CFB 54-feature ensemble.

## Commands

```bash
pip install -r requirements-nfl-research.txt

python src/nfl_research_controller_v1.py --dry-run
python src/test_nfl_research_contract_v1.py
python src/nfl_research_controller_v1.py --capture-schedule --season 2026
python src/nfl_research_controller_v1.py --capture-market --regions us
python src/nfl_research_status_v1.py
```

## Promotion conditions

NFL moves beyond collection mode only after a separate feature contract, data-quality checks, historical validation plan, and prospective-record policy are documented. The first model must be benchmarked against appropriate simple and market baselines before any public or betting-related claim.
