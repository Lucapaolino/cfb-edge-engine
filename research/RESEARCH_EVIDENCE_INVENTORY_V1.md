# Research Evidence Inventory V1

## Study

**Title:** The Cost of Hindsight: Auditing Real-Time Neural Forecasts Before the Outcome  
**System:** CFB Edge Engine  
**Study status:** Prospective evidence validated  
**Scope:** Read-only inventory of existing research evidence  
**Product impact:** None

## Purpose

This document records what CFB Edge Engine already preserves for the prospective evaluation study.

The study asks:

> How different are model-performance conclusions when a real-time forecasting system is evaluated from immutable pre-event records versus reconstructed after outcomes are known?

This inventory does not alter the frozen model, product GUI, customer-facing output, market data, prediction history, or wagering behavior.

## Evidence Sources Reviewed

- `data/prospective/prediction_history.sqlite`
- `models/cfb_nn_2026_v1/`

## Prediction-History Tables

| Table | Research Role |
|---|---|
| `prediction_runs` | Identifies capture batches, timestamps, source board, record count, and batch SHA-256 hash |
| `prediction_snapshots` | Stores immutable pre-kickoff forecast, market, quality, uncertainty, and integrity fields |
| `game_results` | Stores matched completed-game results and result-capture timing |
| `prediction_scores` | Stores derived error, accuracy, Brier, market-error, and ATS diagnostic calculations |

## Prospective Timing Check

| Measure | Result |
|---|---:|
| Total prediction snapshots | 933 |
| Captured before kickoff | 933 |
| Late or unknown timing | 0 |

All inspected prediction snapshots were captured before their scheduled kickoff timestamps.

This confirms pre-event timing at the snapshot-table level. The count represents snapshots, not necessarily unique games, because a game may have multiple pre-kickoff captures.

## Input and Data-Quality Evidence

| Input Status | Snapshots |
|---|---:|
| `FULL_54` | 898 |
| `MISSING_STATIC_CONTEXT` | 19 |
| `EARLY_SEASON_HEAVY_IMPUTATION` | 16 |

The snapshot record preserves:

- Feature values used by the model
- Feature count and non-null feature count
- Missing-feature state
- Input-quality status
- Team history, returning-production, talent, Elo, and pregame feature fields
- Context-readiness indicators

This allows later analysis to distinguish full-input forecasts from forecasts made with partial or heavily imputed information. Partial-input records will not be silently removed from research analysis.

## Forecast and Uncertainty Evidence

Each snapshot includes:

- Model favorite and underdog
- Model spread and projected home margin
- Home and favorite win probabilities
- Ensemble margin standard deviation
- Ensemble win-probability standard deviation
- Ensemble vote fields and head-agreement status
- Strength label and strength score
- Signal warnings
- Neural input status and feature coverage

Ensemble-head agreement was recorded for all snapshots:

| Head Agreement | Snapshots |
|---|---:|
| Agree | 929 |
| Disagree | 4 |

## Market Evidence

| Hard Rock Availability | Snapshots |
|---|---:|
| Available | 823 |
| Unavailable | 110 |

The immutable snapshot payload preserves, when available:

- Hard Rock spread, price, display, fixture identity, freshness, and quality status
- Reference-market spread, book count, event identity, batch identity, freshness, and matching information
- Model-to-market edge and model-to-consensus measures
- Market-quality and signal-warning information

Hard Rock execution information and reference-market information remain distinct research fields. Reference-market data will not be substituted for Hard Rock execution data.

## Outcome and Derived-Score Evidence

Completed-game records contain:

- Game identity, teams, season, and week
- Final points and actual home margin
- Winner
- Result source and result-capture timestamp

Derived score records contain:

- Predicted and actual margins
- Margin error, absolute margin error, and squared error
- Winner-correct indicator
- Brier score
- Hard Rock implied-margin error
- Model-side ATS diagnostic result and flat units

ATS and flat-unit fields remain secondary diagnostics only. They are not evidence of a validated wagering strategy or profitability.

## Integrity Evidence

The history layer contains:

- Snapshot ID
- Run ID
- Record SHA-256 hash
- Batch SHA-256 hash
- Immutable JSON payload
- Capture timestamps
- Kickoff timestamps

The current frozen model artifact directory contains:

- Five neural model checkpoints
- Feature-name contract
- Imputer medians
- Scaler
- Target normalization
- Model metadata

SHA-256 hashes were generated for all current artifact files during the evidence check. Those hashes will be formalized in a separate study-baseline manifest before final analysis.

## Assessment

The prospective side of the study is sufficiently supported to proceed.

The system already preserves the essential evidence needed to evaluate a forecast according to what was known before the outcome:

1. Pre-kickoff timing  
2. Game identity  
3. Inputs and input quality  
4. Forecast output and uncertainty  
5. Market state  
6. Immutable integrity records  
7. Completed outcomes  
8. Derived performance metrics  

## Remaining Research Work

The following items remain separate research tasks:

1. Create a versioned study-baseline manifest linking model hashes and Git commit identity.
2. Define and build a retrospective-reconstruction dataset.
3. Predefine the simple non-neural benchmark.
4. Analyze unique games and define handling for multiple snapshots of the same game.
5. Add verified near-close market analysis when sufficient observations exist.
6. Complete the full-season prospective evaluation.

## Research/Product Separation

This evidence inventory is private research documentation.

It does not change the CFB Edge product, customer GUI, frozen model, customer board, API keys, raw databases, or any future subscription offering.