# Research & Operations Panel Replacement V1

## Decision

Replace the animated AI Core only after the current CFB presentation/research milestone. The existing desktop operator GUI stays; the customer viewer is not a replacement for it.

The new left-side panel should be called **Research & Operations**. It uses the same top-dashboard space as the AI Core and keeps the fixed System Control panel, the lower Signal Board, and persisted splitter behavior.

## What belongs in the space

1. **Current Board** — selected game, model/market timestamp, and data-quality state. It shows provenance rather than a decorative animation.
2. **Research Queue** — CFB live research status, NFL collection-only status, and cloud-staging readiness as separate tracks.
3. **Audit Pulse** — pre-kickoff timing checks, frozen-model identity, immutable-history health, and missing-data flags.
4. **Pipeline State** — the active command stage from capture through postgame review, with no implication that the system can place a bet.

## Compatibility contract

The replacement widget should preserve these existing GUI callback methods so the canonical controller wiring does not change:

- `set_stage(stage, progress)`
- `set_signal_activity(strength, warning_text="")`
- `activate_game(game_key, strength="N/A", warning_text="")`

Those methods should update labels and badges only. They must not rerun the model, alter a frozen prediction, or write a CFB research record.

## Implementation boundary

When the CFB presentation milestone is complete, add the replacement as a new widget module first, test it against the current GUI callbacks, then swap the panel in a single reviewed GUI batch. Do not delete the existing neural widget or the archived GUI variants until the replacement has been reviewed on the Mac.
