"""Shared project paths for CFB Edge Engine.

This module contains filesystem locations only. It does not load API keys,
call providers, run the model, or modify any data.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROSPECTIVE_DIR = DATA_DIR / "prospective"
MODEL_DIR = PROJECT_ROOT / "models" / "cfb_nn_2026_v1"
DB_FILE = DATA_DIR / "cfb.db"
ENV_FILE = PROJECT_ROOT / ".env"

CONTROLLER = SRC_DIR / "run_game_day_v1.py"

BOARD_JSON = PROCESSED_DIR / "live_model_market_board_2026_v1.json"
BOARD_FILE = BOARD_JSON
PREDICTIONS_JSON = PROCESSED_DIR / "live_nn_predictions_2026_v1.json"
STATE_DIR = PROCESSED_DIR / ".game_day_state"

HISTORY_DB = PROSPECTIVE_DIR / "prediction_history.sqlite"
HISTORY_REPORT = PROCESSED_DIR / "prediction_history_summary_latest.json"
WAGER_DB = PROSPECTIVE_DIR / "wager_ledger.sqlite"

WHAT_CHANGED_FILE = PROCESSED_DIR / "what_changed_2026_v1.json"
AUDIT_FILE = PROCESSED_DIR / "prediction_audit_2026_v1.json"
HEALTH_FILE = PROCESSED_DIR / "data_health_2026_v1.json"
PERFORMANCE_FILE = PROCESSED_DIR / "model_market_performance_2026_v1.json"
WEEKLY_RECAP_FILE = PROCESSED_DIR / "weekly_recap_2026_v1.json"
EXPLANATION_FILE = PROCESSED_DIR / "prediction_explanations_2026_v1.json"
MARKET_HISTORY_FILE = PROCESSED_DIR / "market_history_2026_v1.json"
WAGER_CLV_FILE = PROCESSED_DIR / "wager_clv_2026_v1.json"
STRATEGY_LAB_FILE = PROCESSED_DIR / "strategy_lab_v1.json"
STRATEGY_LAB_CSV = PROCESSED_DIR / "strategy_lab_v1.csv"

WHAT_CHANGED_SCRIPT = SRC_DIR / "what_changed_v1.py"
AUDIT_SCRIPT = SRC_DIR / "prediction_audit_v1.py"
HEALTH_SCRIPT = SRC_DIR / "data_health_v1.py"
PERFORMANCE_SCRIPT = SRC_DIR / "model_market_performance_v1.py"
WEEKLY_RECAP_SCRIPT = SRC_DIR / "weekly_recap_v1.py"
EXPLANATION_SCRIPT = SRC_DIR / "build_prediction_explanations_v1.py"
MARKET_HISTORY_SCRIPT = SRC_DIR / "market_history_clv_v1.py"
MARKET_CAPTURE_SCRIPT = SRC_DIR / "market_capture_now_v1.py"
STRATEGY_LAB_SCRIPT = SRC_DIR / "strategy_lab_v1.py"
