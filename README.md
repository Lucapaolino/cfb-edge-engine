# CFB Edge Engine

Private college-football prediction and market-analysis project.

## Current status

- Frozen 2026 neural-network model
- Live game-day board and desktop GUI
- Market snapshots, prediction history, audit tools, and wager ledger
- Research only — not financial advice and not an automated betting system

## Requirements

- Python 3
- A CollegeFootballData API key
- A The Odds API key for reference-market data
- An OddsPAPI API key for Hard Rock market data

## Setup

```bash
git clone https://github.com/Lucapaolino/cfb-edge-engine.git
cd cfb-edge-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your personal API keys to `.env`:

```env
CFBD_API_KEY=your_key_here
THE_ODDS_API_KEY=your_key_here
ODDSPAPI_API_KEY=your_key_here
```

## NFL research collection (separate from CFB)

NFL V1 is collection-only. It records source schedule/results and reference-market
snapshots in `data/nfl/`; it does not read or write the frozen CFB workflow,
create NFL predictions, signals, picks, or wagers.

```bash
pip install -r requirements-nfl-research.txt

# No provider calls and no writes.
python src/nfl_research_controller_v1.py --dry-run
python src/test_nfl_research_contract_v1.py
python src/test_nfl_research_linkage_v1.py

# Explicit, append-only source captures.
python src/nfl_research_controller_v1.py --capture-schedule --season 2026
python src/nfl_research_controller_v1.py --capture-market --regions us

# Read-only collection status.
python src/nfl_research_status_v1.py

# Read-only proof that latest market events link to latest schedule games.
python src/nfl_research_controller_v1.py --linkage-audit
```

## Container readiness

The NFL research worker can be built locally without moving CFB or creating a
cloud account:

```bash
docker compose -f docker-compose.nfl-research.yml build
docker compose -f docker-compose.nfl-research.yml run --rm nfl-research --dry-run

# Headless CFB portability check; this only runs the existing canonical dry run.
docker compose -f docker-compose.cfb-runner.yml build
docker compose -f docker-compose.cfb-runner.yml run --rm cfb-runner --dry-run
```

See `research/NFL_RESEARCH_PROTOCOL_V1.md` and
`research/CLOUD_OFFLOAD_READINESS_V1.md` for the scope and migration rules.
