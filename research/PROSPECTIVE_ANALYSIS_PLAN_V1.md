# Prospective Analysis Plan V1

**Project:** Beyond Hindsight: Building an Auditable Neural Decision-Support System for Real-Time Forecasting  
**Date recorded:** October 1, 2026  
**Version:** 1.0  
**Test environment:** Live 2026 college-football forecasting

## 1. Purpose

This plan defines the collection and analysis rules for the next prospective
collection window. Its purpose is to separate what the system knew before an
outcome from what becomes known after the outcome.

The research question is:

> Can a live neural decision-support workflow preserve a forecast and its
> decision-time evidence before the outcome is known, then support a later
> evaluation of that same saved forecast?

College football is the live test environment. The study does **not** ask
whether the system is a profitable wagering system, whether it beats a public
market, or whether it is ready to make automated decisions in other fields.

## 2. Scope and claim boundary

The primary claim under study is operational:

> Within this system, a forecast record can be captured before kickoff,
> preserved with a timestamp and record hash, and scored separately after the
> game result becomes available.

This is evidence of an internally auditable, tamper-evident workflow. It is
not independent third-party certification, a guarantee that no administrator
could ever alter a database, or proof that the workflow generalizes to medicine,
cybersecurity, finance, or any other domain.

The preliminary records created before this plan are useful descriptive
evidence, but they are not retroactively preregistered. The prospective
confirmation window begins after this plan is committed to version control and
continues through the next college-football collection cycle.

## 3. Units and definitions

- **Prediction snapshot:** one stored forecast record for one game at one
  capture time. Multiple snapshots can exist for the same game. Snapshots are
  not independent games.
- **Resolved game:** one unique game identified by season, game identity,
  teams, and kickoff time.
- **Strictly pre-kickoff snapshot:** a record for which
  `captured_utc < kickoff_utc`.
- **Primary snapshot policy:** for each resolved game, select the **latest**
  strictly pre-kickoff snapshot using capture time only. The game result must
  never influence selection.
- **Robustness snapshot policy:** for the same resolved games, select the
  **first** strictly pre-kickoff snapshot using capture time only.
- **Hash check:** recompute and compare the stored `record_sha256` against the
  stored record payload. A mismatch is an integrity failure that must be
  reported. A match supports internal tamper evidence for the stored record; it
  is not an outside timestamping service.
- **Scoring:** add or rebuild a derived result and score after the game ends.
  The original prediction snapshot remains separate from derived scores.

## 4. Pre-specified data-collection rules

1. Use the frozen CFB model and the canonical game-day workflow. Do not retrain
   the model, alter features, or change signal thresholds during this window.
2. Run the game-day workflow before the main kickoff slate so forecasts are
   captured prospectively.
3. Capture market context at planned pre-kickoff times when available. These
   captures support later timing analysis; they do not change the primary
   hindsight-audit question.
4. Do not rerun the workflow solely to manufacture a preferred historical
   forecast after a result is known.
5. Do not edit prior prediction snapshots. If an operational problem occurs,
   record it as a failure or limitation rather than replacing the old record.
6. Preserve unavailable, unmatched, late, or malformed records in reporting.
   Do not silently exclude them.

## 5. Primary analysis

For the latest strictly pre-kickoff snapshot of each resolved game, report:

1. Number of stored snapshots.
2. Number and percentage captured before kickoff.
3. Number of invalid, missing, or at/after-kickoff timestamps.
4. Number of record-hash matches, mismatches, and invalid payloads.
5. Number of distinct resolved games selected.
6. Number of selected games with completed outcomes, and number still awaiting
   or unmatched.
7. Confirmation that original forecast records and later derived scores remain
   separate in the evidence store.

The primary endpoint is whether the workflow preserves and later evaluates
decision-time records without result-dependent snapshot selection. A PASS does
not indicate that the forecasts are accurate, profitable, or superior to any
other source.

## 6. Secondary descriptive outcomes

For completed, correctly matched games only, report the following as
descriptive context:

- winner accuracy;
- Brier score for forecast probability;
- model margin mean absolute error and root mean squared error;
- any available captured public-market margin baseline;
- model-side against-the-spread diagnostic, clearly labeled as diagnostic only.

These quantities describe the currently available sample. They will not be used
to choose a new strategy, declare a commercial advantage, or support a claim
that the system outperforms a market.

## 7. Robustness analysis

Run the same integrity checks under the first strictly pre-kickoff snapshot
policy. Compare first and latest selections only by the pre-specified rules
above. Report:

- number of games with the same versus different selected snapshot;
- capture lead time before kickoff under each policy;
- differences in the descriptive outcomes on the common completed-game set.

This sensitivity check tests whether a result-neutral choice of pre-kickoff
snapshot materially changes the description. It does not establish that either
capture time is a better forecasting or wagering policy.

## 8. Failure handling and exclusions

No record will be excluded because its forecast was good or bad. The only
permitted exclusions from outcome metrics are records that cannot be linked to
one exact completed game, records that fail the pre-kickoff timestamp rule, or
records with invalid payloads. Each exclusion count and reason must be shown.

If a data source is unavailable, an API call fails, a result is delayed, or an
integrity mismatch occurs, the report will state that directly. Missing evidence
will not be treated as passing evidence.

## 9. Planned operational sequence

### Before and during the next Saturday slate

1. Run `python src/run_game_day_v1.py` before the main kickoff windows.
2. Run the existing near-kickoff market capture workflow at the planned
   pre-kickoff times.
3. Keep the frozen-model configuration unchanged.
4. Do not alter or delete any resulting prospective records.

### After outcomes become available

1. Run `python src/score_prediction_history_v1.py`.
2. Run `python src/research_hindsight_audit_v1.py --policy latest --season 2026`.
3. Run `python src/research_snapshot_policy_robustness_v1.py --season 2026`.
4. Retain the generated JSON and CSV reports in `data/processed/`.
5. Update any 3MT slide numbers literally, including the reporting date and the
   distinction between forecast records and completed games.

## 10. Reporting language

The presentation and any written report will use language consistent with the
evidence. Suitable wording includes:

> The internal audit found that the saved forecast records met the stated
> pre-kickoff timing rule and that the stored hash check found no mismatches.
> Outcomes were later scored separately from the saved forecasts.

The report will not say that the records are impossible to change, independently
certified, clinically validated, universally generalizable, or proven to create
a decision advantage.

## 11. Amendments

This document must not be silently revised after collection begins. Any change
to the model, selection policy, metrics, or scope requires a new dated plan
version that states what changed and why. The original version remains in the
version-control history.
