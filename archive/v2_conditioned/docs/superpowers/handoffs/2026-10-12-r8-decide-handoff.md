# Handoff: r8 decide: step 1 with the user on round 7's results (the L3 departure first), then the wrap-up and step 2

## Brief
- Done: round 7 ran. K1 CLEAR (+3.92 R@1, bar +1.0); the seed-42 gate found no pass (n = 0), so design L ends.
- State: Friday verdict NO-GO: the paper is the ArtELingo paper with AFF, which stays the current best.
- Next: step 1, decision C, on the auto report's results brief; then the wrap-up and step 2.
- Blocked: the user decides the L3 departure. The run went on with an agent fix where rule §8 said stop. It cannot change n = 0.

> Written 2026-10-12 07:18 by the `r7 run` chat. Loop step reached: step 9 done (both reports committed); step 1 of the
> next loop next. Direction log: round 7's entry is written at this chat's wrap-up.

## 1. Read in this order

1. The auto report `docs/reports/auto/v2/2026-11-26_design_l_style_source.md`: show the user its results brief
   without the collapsed block, ask for their reading (decision C), then open ours and say where the two differ.
2. The user-read report `docs/user_read/2026-10-12_design_l_style_source.md` (its advice folded away the same way).
3. The run log `src/test/20261126_design_l_style_source/20261126_design_l_style_source_log.md`, sections on the L3
   stop and "Process departure", before the L3 question below.
4. `docs/superpowers/direction_log.md` ("Now", "Parked ideas") and `~/.claude/templates/direction_log_entry.md` for
   the wrap-up.

## 2. Decided by the user, not to be reopened

- Spec D1 to D27 and the rule (`2026-10-12-r7-build-handoff.md` §2), approved 2026-10-12 02:34. The run made no
  user decision.

## 3. Where things stand

- K1 (told true styles, style × genre, fused minus matched counterpart): Σ(4d) 643 against the CLEAR bar 164; +3.92
  [+3.35, +4.49] R@1 points.
- Gate (12 scorers on seed 42): S1 fails 12 of 12 (best R1 on V3 +0.25 [+0.05, +0.46] against the bar +0.5; AFF's
  own +0.33 is below it too); S3 emotion × genre fails 12 of 12 (−0.93 to −1.61 against AFF, bar −0.3); S2 passes 5
  of 12. n = 0, so FV NO-GO (rule §6, §7.7, §8). No sensitivity, smoke, seed build or test ran; the seed ledger is
  unchanged (55 and later free); no held row touched.
- "Promising" holds for R1 on V2, R1 on V3 and extended AFF on V3; the extended-AFF references on raw CSD and N meet
  it too, and the random controls beat the candidates on margin and gain (descriptive, cause not traced).
- Checks: each verdict file passed its separate re-derivation; the auto report's claim review found 0 critical
  findings (about 175 numbers, 174 at once, 1 rounding fixed); the user-read report had its cold read and fidelity
  check; one fix wave applied to both.

## 4. Open points for the next step

- **Key, L3 (ask answer-first):** rule L3's own recipe failed its assertion on the CSD vectors (V1, V4). Rule §8
  said stop and let the user decide; the run sent a phone note (05:55) and went on with an agent-default fallback
  (the same projector from the SVD basis, rank 64). The user decides whether K1 and the gate stand on it: keep the
  fallback, drop V1 and V4 (with Z1, Z4), or a rank-63 projector (a rerun of V1, V4, Z1, Z4; a change of n is very
  unlikely). Keep or drop leaves n = 0 for certain.
- **Key, decision C:** the user's reading of the results brief, recorded before ours.
- **Key, decision A (step 2):** take the NO-GO (the ArtELingo paper with AFF; Friday 2026-10-16 go or no-go; CVPR
  abstract 2026-11-10), or resume design L after Friday (rule §8: only on the user's choice; new spec and rule). The
  "promising" note goes to "Parked ideas" either way (spec §6: no dig budget before Friday).
- Minor, wrap-up: round 7's direction-log entry and "Now"; the storage summary (below); `wrapup-check`; `disk-check`.

## 5. Code and pitfalls

- `results/` is write-once. A corrected run writes `<name>_fix<n>` beside the verdict and the descriptive pass then
  refuses until the user decides which stands (rule §8). A rank-63 rerun of L3 would go that way.
- `r7_groupings.orth` keeps the literal pinv form where it passes and falls back only where it fails (commit e3776ca).
- `subagent-watch --watch` printed "no running subagents" while a worker was running in this chat; check the
  transcript's time by hand when in doubt.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing in CoSiR's git (`.scratch/` and `results/` are outside git; `.claude/20261012_log.md` is
  gitignored and holds the L3 change log).
- On disk over 1 GB: `src/test/20261126_design_l_style_source/results/` about 4.7 GB (the verdicts, arrays, heads,
  readers and banks the reports rest on: recommend keep); plus the build's items already on
  `.scratch/pending_deletions.md` (`results/smoke/chain/` 1.2 GB, a session scratchpad `smoke_fix/` 1.2 GB, earlier
  trial outputs about 7 GB, worktrees `/project/CoSiR-wt/r7-*` and merged ticket branches); the token cache on SSD2
  (1.1 GB, keep); `res/cluster_jobs/` 3.1 GB (round 6, keep).
