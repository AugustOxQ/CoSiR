# Experiment 13 symmetric-shared training sweep

## Goal

Run the required GPU smoke test, launch the full detached six-run sweep only after the smoke gate passes, and monitor it without modifying training implementation.

## Phases

1. **Smoke gate** — in progress: run the two-arm, two-epoch GPU smoke test and inspect logs for completion, finite losses, and non-degenerate retrieval.
2. **Detached launch** — pending: start the 3-seed x 2-arm, 100-epoch sweep after a passing smoke gate.
3. **Monitoring** — pending: periodically inspect the detached process and logs to all terminal states.
4. **Handoff/closeout** — pending: record final or continuation status in the Experiment 13 SDD ledger and CCG task record.

## Next Step

Run the exact user-provided smoke command with the CoSiR conda environment active.

## Errors Encountered

| Error | Attempt | Resolution |
| --- | --- | --- |
| None | — | — |
