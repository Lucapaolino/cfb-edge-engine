"""Filesystem paths for the NFL research-collection track.

This module is intentionally separate from ``cfb_edge_paths.py``.  It does
not read credentials, call providers, load a model, or modify CFB files.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env"

# NFL is a distinct research namespace.  Nothing in these paths points at the
# frozen CFB database, prediction history, model artifacts, or GUI outputs.
NFL_DIR = DATA_DIR / "nfl"
NFL_RAW_DIR = NFL_DIR / "raw"
NFL_RAW_MARKET_DIR = NFL_RAW_DIR / "market_snapshots"
NFL_RAW_SCHEDULE_DIR = NFL_RAW_DIR / "schedule_snapshots"
NFL_PROSPECTIVE_DIR = NFL_DIR / "prospective"
NFL_PROCESSED_DIR = NFL_DIR / "processed"

NFL_RESEARCH_DB = NFL_PROSPECTIVE_DIR / "nfl_research.sqlite"
