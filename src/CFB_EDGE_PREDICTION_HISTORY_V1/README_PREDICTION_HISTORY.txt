CFB EDGE ENGINE — PREDICTION HISTORY V1

WHAT THIS DOES
--------------
Every real RUN GAME DAY now freezes the model's pre-kickoff predictions before results are known.
Those predictions are stored immutably, then later scored automatically when final scores exist in data/cfb.db.

FILES TO COPY INTO ~/Desktop/Neural/src/
---------------------------------------
run_game_day_v1.py                 REPLACE existing controller
prediction_history_v1.py           NEW shared history module
log_predictions_v1.py              NEW immutable prediction logger
score_prediction_history_v1.py     NEW result scorer
report_prediction_history_v1.py    NEW evaluation/report generator

NO GUI REPLACEMENT IS NEEDED.
The current GUI already reads PIPELINE_STAGE messages from run_game_day_v1.py.

STORAGE CREATED AUTOMATICALLY
-----------------------------
data/prospective/prediction_history.sqlite
    Canonical history database.
    prediction_snapshots and prediction_runs are protected from UPDATE/DELETE.

data/prospective/prediction_history.jsonl
    Append-only text backup of every pre-kickoff prediction snapshot.

data/prospective/prediction_snapshots/cfb_2026_predictions_<run_id>.json
    One immutable human-readable batch file for each RUN GAME DAY.

data/processed/prediction_history_evaluation_latest.csv
    Latest pre-kickoff snapshot per completed game, with result/scoring fields.

data/processed/prediction_history_summary_latest.json
    Machine-readable aggregate performance summary.

WHAT GETS SAVED
---------------
Run ID and capture time
Game ID / teams / week / kickoff
Model favorite and projected spread
Model win probability
Internal model home-margin representation for objective scoring
Hard Rock line and juice for the model side
Reference-market spread
Model edge
LOW / MEDIUM / HIGH strength and score
Warnings and NN diagnostics
Full original board record as JSON
SHA-256 record hash
Wager decision fields (currently explicitly unset because there is no validated betting rule yet)

WHAT GETS SCORED LATER
----------------------
Winner accuracy
Brier score
Model spread MAE
Model spread RMSE
Hard Rock implied-margin MAE
Model MAE minus Hard Rock MAE
Hypothetical ATS result of backing the model favorite at the captured Hard Rock line
Strength-bucket performance
Warning vs no-warning performance

IMPORTANT
---------
The ATS diagnostic is NOT a betting strategy and is not treated as one.
The system preserves every run, so we can evaluate the earliest prediction, latest pre-kickoff prediction, or every snapshot without hindsight rewriting.

INSTALL
-------
Copy all five .py files into ~/Desktop/Neural/src/.

Then:

cd ~/Desktop/Neural
source .venv/bin/activate
python -m py_compile src/run_game_day_v1.py src/prediction_history_v1.py src/log_predictions_v1.py src/score_prediction_history_v1.py src/report_prediction_history_v1.py
python src/run_game_day_v1.py --dry-run

The dry run makes no API calls and writes no prediction rows.

MANUAL REPORT COMMANDS
----------------------
Latest pre-kickoff prediction per game:
python src/report_prediction_history_v1.py --policy latest

First recorded prediction per game:
python src/report_prediction_history_v1.py --policy first

Every recorded snapshot:
python src/report_prediction_history_v1.py --policy all

Score history again after game results are updated:
python src/score_prediction_history_v1.py
