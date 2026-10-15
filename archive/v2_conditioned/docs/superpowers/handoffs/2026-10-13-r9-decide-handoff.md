# Handoff: r9 decide: step 1 with the user on round 8's results brief, then the wrap-up and step 2

## Brief
- Done: round 8's two reports are checked and committed. No pick in either track, so AFF stays the current best.
- State: all 20 main-line scorers are below AFF (best −0.47 R@1 points). The DINOv2 reader is +0.29 [−0.01, +0.59] and misses the bar.
- Next: step 1, decision C, on the auto report's results brief. Then the wrap-up and step 2.
- Blocked: nothing. The user's reading comes first.

> Written 2026-10-13 11:54 by the `r8 run 2` chat. Loop step reached: step 9 done (both reports committed, f94e836);
> step 1 of the next loop next. Direction log: round 8's entry is written at this chat's wrap-up
> ([2026-10-12 · Round 8](../direction_log.md) holds its decisions).

## 1. Read in this order

1. The auto report `docs/reports/auto/v2/2026-11-29_learned_grouping.md`: show the user its results brief without the
   collapsed block, ask for their reading (decision C), then open ours and say where the two differ.
2. The user-read report `docs/user_read/2026-10-13_learned_grouping.md` (its advice is folded away the same way).
3. The run log `src/test/20261129_r8_grouping/20261129_r8_grouping_log.md`: decisions table, timeline, "For the report".
4. `docs/superpowers/direction_log.md` ("Now", "Parked ideas") and `~/.claude/templates/direction_log_entry.md` for
   the wrap-up.

## 2. Decided by the user, not to be reopened

- Spec A and spec B and the rule, approved 2026-10-12 (`2026-10-12-r8-build-handoff.md` §2).
- Amendment 1 (2026-10-13 01:13): M4's view-check threshold 0.9999 → 0.999 and a full restart. Lowering: user; 0.999:
  user, after seeing the recommendation (flagged). Recorded in spec A §4 "Amendments" and the rule header (fdaee39).

## 3. Where things stand

- **Gate** (`results/gate_verdict.json`, 11:20): no main pick and not promising; no screen pick. Rule §8: round 8 ends
  at the gate, AFF stays current best, no held read, no `HELD_RULE.md`. Both C13 re-derivations agree (0
  disagreements). NBC triggered and ran; it did not pass.
- **Main line:** 20 scored scorers, SC3 from −0.466 to −0.802 R@1 points against AFF, every upper bound below 0. The best
  is extended AFF on FCMI weight 0.20: −0.466 [−0.651, −0.288]. FC4 to FC6 were dropped by the 60-minute cap (D21), so
  only weights 0.05 to 0.20 were scored.
- **Screen:** the DINOv2 reader +0.285 [−0.012, +0.592]; emotion × style Σ(4d) −63 against the bar −49 (about −0.38
  against −0.30 R@1 points). The SigLIP 2 reader +0.059, style −35 (passes), genre −64 (fails).
- **Amendment 1** does not change the verdict: all six view-grouping scorers fail the gate.
- **Checks:** the claim review re-derived every number of the auto report from the result files: no wrong number; 3
  must-fix and 17 minor items, fixed (`.scratch/r8-grouping/notes/claim_review.md`). The user-read report had its cold
  read and fidelity check, with one fix wave.

## 4. Open points for the next step

- **Key, decision C:** the user's reading of the results brief, recorded before ours.
- **Key, decision A (step 2):** the brief's three options: close the grouping line and write the CVPR paper with AFF; a
  new loop on the DINOv2 reader (own spec, rule, fresh seeds); a fair-clustering follow-up with a longer cap. Round
  8's held-read cut-off is Fri 2026-10-23 12:00 (rule §8); the CVPR abstract is due 2026-11-10. The options not chosen
  go to "Parked ideas".
- **Minor, wrap-up:** round 8's direction-log entry (the 0.999 value flagged as chosen after the recommendation) and a
  new "Now"; the storage summary (§6; 31 rows on `.scratch/pending_deletions.md`); `wrapup-check`, `disk-check`,
  `sync.sh`. The held ledger gets no H6 row, since there was no held read.
- **Minor, process points for the wrap-up:** (a) build 4's fixer read the context meter's message as untrusted
  output; the `worker` body could say that the context meter is ours. (b) The GPU queue logs a D21 drop as "failed"
  (exit 247), and its summary line says "3 failed"; a rule drop could be logged as "dropped under the rule".

## 5. Code and pitfalls

- `src/test/20261129_r8_grouping/results/` and `results_m4_09999/` stay untracked and write-once; do not commit them.
- The auto report's links into `results/` and `.scratch/` work only on this machine.

## 6. State at handoff

- Running: nothing (no queue, subagent or Codex job).
- Uncommitted: only the two untracked results trees above.
- On disk over 1 GB, all on `.scratch/pending_deletions.md` for the storage summary: `results/` 7.2 GB and
  `results_m4_09999/` 2.1 GB (system disk); on `/data/SSD2/cosir_r8/fcmi/`, FC1 to FC6 about 27 GB, the archived
  `FC1_m4_09999` and `FC3_m4_09999` 8 GB, and the smoke folder 66 GB; the build rows of builds 2 to 5 (trial outputs,
  worktrees, ticket branches). Free: 236 GB on `/`, 1.8 TB on `/data/SSD2`.
