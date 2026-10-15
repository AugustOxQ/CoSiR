# Handoff: the Leiden and k-shot side checks are closed; nothing to continue here

## Brief
- Done: the Leiden and k-shot side checks closed, both readings recorded, reports updated and checked.
- State: Leiden ties k-means on the bar margin (−0.20 [−0.44, +0.05]), 0.56 lower on AFF's R@1; kept as a record.
  AFF's recipe does not change. AFF's whole lead is in emotion episodes.
- Next: none in this line. The user takes the labels question to the weekly meeting (2026-10-14).
- Blocked: nothing.

> Written 2026-10-14 by the "Leiden groupings 2" side chat (outside the r9 loop). No next chat opened: no work moves
> on. Direction log entries: "2026-10-14 · Leiden groupings side check" and "2026-10-14 · K-shot side check".

## 1. Read in this order

1. The direction log's "Now" paragraph and the two entries above.
2. Only if needed: the reports, [Leiden](../../reports/auto/v2/2026-12-04_leiden_groupings.md) and
   [k-shot](../../reports/auto/v2/2026-12-03_kshot_supports.md), with their user-read reports in `docs/user_read/`.

## 2. Decided by the user, not to be reopened

- The grouping method matters less than the labels question; Leiden in AFF and the H1 one-grouping failure are kept as
  a record. Leiden is a minor open direction, in "Parked ideas" (user, 2026-10-14, before seeing ours).
- The k-shot reading: "emotion is so different from style and genre that it needs fewer examples; style × genre stays
  hard at any k" (user, after seeing ours; the user notes it may hold mistakes).
- Storage: the rebuild's intermediates and check runs deleted; r9 folders and the pending-deletion rows left to the
  r9 run chat (user).

## 3. Where things stand

- AFF's fixed recipe keeps k-means in r9 and the paper; no Leiden held read. That "keep this just as a record" means
  "no recipe change" is our reading, stated to the user.
- The k-shot split by the aspect the pairs show (`src/test/20261203_kshot_supports/ks_by_condition.py`): AFF − B′(A0)
  at 4 pairs is +2.88 on emotion, −0.09 on style, −0.69 on genre; at 1 pair +4.41, −1.15, −8.44.

## 4. Open points for the next step

- **Key, the user's (weekly meeting, 2026-10-14):** are aspect labels needed in training to separate style from genre?
  Options and our recommendation are in the k-shot entry of the direction log ("Open").

## 5. Code and pitfalls

- `lg_rebuild.py --step result` still recomputes the Leiden table from `results/rebuild/` (kept: checkpoint,
  per-episode arrays, JSON records). The frozen worktree `/project/CoSiR-lg-r9` was removed; recreate it at b84db1e
  before any other rebuild step.
- `/project/CoSiR` is on main and shared with the r9 run chat: stage only your own files.

## 6. State at handoff

- Running: nothing. Uncommitted: nothing of this chat's.
- Codex: two jobs (report update and its fix round), in `/project/claude-config/codex/jobs.md`.
