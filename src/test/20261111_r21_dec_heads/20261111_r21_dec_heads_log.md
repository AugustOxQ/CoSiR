# r21: DEC loss on every head of c*: round log

Spec: `docs/superpowers/specs/2026-11-11-r21-dec-heads-design.md`. Decision lines: `decisions.md`.

## Build (2026-11-11, build chat)

- 2026-11-11 06:37: tickets 01 to 06 written (`.scratch/r21-dec-heads/issues/`); integration branch `r21-dec-heads`.
  Codex had no room for implementer jobs (five-hour window 61% left, 80% needed), so Claude workers build.

## Ticket 06 (build, `r21/t06`)

- 2026-11-11 07:05: `r21_table.py` (CLI `table`, `diag`, `regress`) and `test_r21_table.py` (37 tests: word rule at the
  interval edges, pairing, cells, silhouette against scikit-learn, DEC statistics, two real-file checks on r10's tree).
  No existing file edited. Held (test) exports are read by design (stage 2); nothing here tunes or selects on them.
