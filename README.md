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
- An OddsPAPI API key for Hard Rock market data

## Setup

```bash
git clone https://github.com/Lucapaolino/cfb-edge-engine.git
cd cfb-edge-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env