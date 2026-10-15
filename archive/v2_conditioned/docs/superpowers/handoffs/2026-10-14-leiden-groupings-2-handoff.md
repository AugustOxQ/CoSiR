# Handoff: show the user the Leiden AFF rebuild result, get their reading, then finish the reports and the wrap-up

## Brief
- Done: AFF rebuilt on Leiden at 64 groups; H1 retrained on scorer-train and screened.
- State: AFF over B′(A0): Leiden +0.50 [+0.30, +0.71] against k-means +0.70; paired −0.20 [−0.44, +0.05]. AFF R@1 paired −0.56 [−0.80, −0.30]. Retrained H1 fails the screen (−0.27).
- Next: ask the user's reading, then update both reports, run the checks, wrap up.
- Blocked: the user's reading, and whether "go with Leiden" changes AFF's fixed recipe.

> Written 2026-10-14 02:56 by the "Leiden groupings" side chat (outside the r9 loop). Cut at the context meter (about
> 205k tokens), as the user asked. Direction log entry: `docs/superpowers/direction_log.md`, "2026-10-14 · Leiden
> groupings side check".

## 1. Read in this order

1. This handoff.
2. The run log `src/test/20261204_leiden_groupings/20261204_leiden_groupings_log.md` (decisions, all numbers, the last
   section holds the follow-ups).
3. The direction log entry above (the user's reading of the first screens, and the open question).
4. Only when you update them: the auto report `docs/reports/auto/v2/2026-12-04_leiden_groupings.md` and the user-read
   report `docs/user_read/2026-10-14_leiden_groupings.md` (both describe the first screens only).

## 2. Decided by the user, not to be reopened

- Side experiments outside the research loop, no loop protocol, "cheap is best": reuse cached data and code, minimal
  tests (user, 2026-10-14).
- H1: only the one-grouping slot (H1 alone), as in buddy vs PercepT (user, after seeing the recommendation).
- After the first screens: rebuild AFF on Leiden and retrain H1 (user). "If leiden ties with k-means we just go with
  leiden, because this makes the design coherent" (user).
- No more AFF tuning (user, earlier).

## 3. Where things stand

Seed-42 development episodes (12,288), R@1 points. The k-means reference reproduces AFF's stated margin.

| AFF | k-means (today) | Leiden at 64 | Leiden minus k-means (paired) |
|---|---|---|---|
| margin over B′(A0) (the bar) | +0.70 [+0.46, +0.94] | +0.50 [+0.30, +0.71] | −0.20 [−0.44, +0.05] |
| AFF's own R@1 | 19.14 | 18.58 | −0.56 [−0.80, −0.30] |
| over its matched control | — | +0.53 [+0.34, +0.72] | — |

- The rebuild replaced image and caption everywhere, B and B′(A0) included (agent default). So Leiden's B′(A0) is also
  lower: AFF's R@1 drops more (−0.56) than its margin (−0.20).
- Retrained H1 (20 communities, AMI 0.49 with the old labels): alone −0.27 [−0.48, −0.07] over its matched control,
  −0.63 [−0.94, −0.32] against the reference. Cross-fit weight 0 again. No rebuild (cheap rule).
- Leiden rebuild's τ_0..3: 0.0001, 0.142, 0.342, 0.611. Gate open share in the cells step: 40.4%.

## 4. Open points for the next step

1. **Key, ask first (decision C):** show the user the table above without our reading; ask whether this counts as the
   tie they meant. The margin difference is within the noise; AFF's own R@1 is lower with an interval below zero.
   Our reading to give after theirs: on the margin, a tie within the noise (point −0.20); on absolute R@1, k-means is
   better, because the weaker Leiden groupings also weaken B and B′(A0). Agent default recommendation: keep k-means
   in AFF; record Leiden as a coherent alternative that costs about 0.2 margin points and 0.6 R@1 points.
2. **Key:** if the user still takes Leiden, does that change AFF's fixed recipe (r9 is building AFF with k-means for
   two more backbones; the paper; AFF's held read in round 6 used k-means, so a Leiden AFF would need its own held
   read and a ledger entry), or only the model card? Ask; no safe default.
3. **Minor:** update the auto report and the user-read report with the follow-ups (H1 retrained, the rebuild), then the
   two checks (cold read on Haiku, fidelity on Sonnet medium) and one fix wave. Add the user's reading to the direction
   log entry.
4. **Wrap-up** (`research-loop.md`, "Wrap-up and handoff"): storage summary, `wrapup-check`, `disk-check`,
   `/project/claude-config/sync.sh`, the memory pointer.

## 5. Code and pitfalls

- `lg_rebuild.py --step result` recomputes `results/rebuild/lg_result.json` from the saved arrays in seconds; no
  rescoring needed.
- The rebuild imports r9 code from the frozen worktree `/project/CoSiR-lg-r9` (detached at b84db1e). Never write under
  any copy of `20261202_r9_scope`.
- `*.json` and `run_*.log` are gitignored; numbers live in the run log and `results/lg_eval.txt`.
- Shell: the Bash tool runs zsh-like; `$VAR` holding several paths does not split. Pass paths explicitly.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of this chat's (`src/test/20261129_r8_grouping/results*` are not ours).
- On disk, for the storage summary (none over 1 GB): `src/test/20261204_leiden_groupings/results/rebuild/` 440 MB
  (A3 checkpoint, banks, posteriors; recommend keep until the user decides on Leiden); its `_check/` 5.7 MB and
  `_check2/` 31 MB (the builder's check runs; recommend delete); `results/` otherwise 44 MB plus `h1r_labels.npz`
  (keep); the worktree `/project/CoSiR-lg-r9` 82 MB (recommend `git worktree remove` once the reports are final).
