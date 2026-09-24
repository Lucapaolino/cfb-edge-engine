# The Cost of Hindsight: Auditing Real-Time Neural Forecasts Before the Outcome

## Status

Protocol version 1.0  
Study status: prospective data collection  
System: CFB Edge Engine  
Model status: frozen 2026 neural ensemble  
Product status: separate from this research study

## Research Purpose

This study tests whether conclusions about a real-time neural forecasting system change when the system is evaluated from records captured before an event versus records reconstructed after the outcome is known.

College football is the live test environment. The research focus is trustworthy AI evaluation: timing, data provenance, auditability, and resistance to hindsight bias.

This is not a study claiming that the system is profitable for wagering.

## Research Question

How different are model-performance conclusions when a real-time forecasting system is evaluated from immutable pre-event records versus reconstructed after outcomes are known?

## System Under Study

CFB Edge Engine contains:

- A frozen five-seed neural-network ensemble
- A fixed 54-feature forecasting pipeline
- Pre-kickoff prediction snapshots
- Sportsbook market capture and comparison
- Immutable prediction-history records
- Postgame scoring and audit tools
- Data-health and provenance checks
- A separate personnel-availability research layer

The frozen model will not be retrained, replaced, or tuned during this study.

Personnel availability remains research context only. It will not change the frozen model, projected spreads, or a wagering rule during the study period.

## Study Conditions

### Condition A: Prospective Evaluation

The authoritative prospective record is the snapshot captured before kickoff.

Each eligible record must preserve, when available:

- Prediction timestamp
- Kickoff timestamp
- Model version and model artifact identity
- Projected margin
- Win probability
- Captured market line
- Input/data-quality status
- Game identity
- Final game result after completion

### Condition B: Retrospective Evaluation

The same games will be evaluated using information reconstructed after results are known, including later available data, later market information, revised records, or hindsight-selected subsets.

Every retrospective result must be clearly labeled as retrospective. It must never overwrite the prospective record.

## Eligibility Rules

A game is eligible for the primary prospective analysis when:

1. A prediction was captured before scheduled kickoff.
2. The prediction can be safely matched to a completed game result.
3. The record has a valid prediction timestamp and game identity.
4. The record is retained regardless of whether the result helps or hurts measured performance.

Missing or partial-input games will not be silently removed. They will be reported as a separate data-quality group.

## Pre-Specified Metrics

### Primary Metrics

- Winner accuracy
- Projected-margin mean absolute error
- Brier score for win probabilities
- Calibration of stated win probabilities

### Secondary Metrics

- Error relative to the captured sportsbook market
- Difference between model margin and market spread
- Near-close market movement when a verified capture exists
- Data completeness versus forecast error
- Frequency and type of data-quality failures

Spread-cover performance, flat-bet units, return on investment, and strategy thresholds are diagnostics only. They are not primary evidence of forecasting value or profitability.

## Baselines

The frozen neural ensemble will be compared with:

1. The captured market-implied forecast, when available.
2. A simple non-neural benchmark defined before final analysis.
3. The retrospective reconstruction condition.

Hard Rock execution data and reference-market data remain distinct. A reference sportsbook must not be substituted for Hard Rock execution data.

## Anti-Hindsight Rules

- The 2026 neural model remains frozen.
- No game will be removed because it was a poor prediction.
- No threshold or strategy will be changed because of early results.
- Strength labels are not validated cover probabilities.
- The original pre-kickoff snapshot is authoritative for prospective evaluation.
- Retrospective calculations must be stored separately.
- Failed, missing, partial, or late data will be documented.
- No betting rule will be claimed without adequate prospective evidence and verified near-close market data.

## Sample Plan

The study will evaluate the full available 2026 season.

The target is at least 250 eligible prospective game records. Weekly reports may be reviewed operationally, but they are descriptive only and will not support broad claims about model quality or profitability.

## Expected Contribution

Neural forecasting, sports models, market comparison, and audit logs already exist. This study does not claim to invent them.

Its contribution is an empirical test of whether hindsight changes the apparent performance of a live neural forecasting system, and whether immutable pre-event records provide a more trustworthy basis for evaluation.

A meaningful result may show either:

- Retrospective reconstruction materially overstates performance; or
- Prospective and retrospective evaluation remain similar, indicating that the audit design successfully limits hindsight distortion.

Both outcomes are valid findings.

## Limitations

- College football is the study environment, so results may not generalize to every domain.
- Market data may be missing, delayed, or unavailable for some games.
- This is one frozen system evaluated over one season.
- Forecast accuracy does not prove wagering profitability.
- Model explanations represent sensitivity and are not causal explanations of game outcomes.

## Separation From the Commercial Product

This protocol, research data, retrospective analysis, raw audit records, strategy experiments, and private operational data remain separate from the eventual customer product.

Customers may eventually receive approved published forecasts and explanations. They will not receive API keys, raw databases, private wager records, experimental thresholds, research controls, or model-pipeline access.