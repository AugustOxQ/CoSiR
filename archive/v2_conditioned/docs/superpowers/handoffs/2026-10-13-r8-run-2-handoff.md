# Handoff: r8 run 2: review the two report drafts, commit them, then open the r9 decide chat

## Brief
- Done: the r8 run ended at 11:31. The gate verdict says no pick in either track, so AFF stays the current best.
- State: all 20 main-line scorers are below AFF. The best screen scorer is R1_DI at +0.29 R@1 points [−0.01, +0.59], which misses the bar.
- Next: claim-review the auto report and check the user-read report. Then commit both and open r9 decide.
- Blocked: nothing.

> Written 2026-10-13 11:46 by the `r8 run` chat (cut at the context meter's 200k). Loop step reached: step 9, report
> drafts written, reviews not yet run. Direction log entry: [2026-10-12 · Round 8](../direction_log.md).

## 1. Read in this order

1. This handoff, §3 to §5.
2. The run log `src/test/20261129_r8_grouping/20261129_r8_grouping_log.md`: decisions, timeline, "For the report".
3. The drafts: `docs/reports/auto/v2/2026-11-29_learned_grouping.md` (15.0 KB) and
   `docs/user_read/2026-10-13_learned_grouping.md`, with Codex's number-source list
   `.scratch/r8-grouping/notes/report_numbers.md`.

## 2. Decided by the user, not to be reopened

- Amendment 1 (2026-10-13 01:13): M4's view-check threshold 0.9999 → 0.999 and a full restart. Lowering: user; 0.999:
  user, after seeing the recommendation (flagged). Recorded in spec A §4 "Amendments" and the rule header (fdaee39).

## 3. Where things stand

- **Run:** first start 2026-10-12 23:29, stopped 01:14 for amendment 1, restarted 01:27 (stage 0 pass under the new
  rule SHA-256 18b4e063…). GPU queue done 09:06; CPU queue done 11:31, 81 of 81 steps ok. It ended 31 min past the
  11:00 preference (the choice and why: run log, 01:27 row).
- **Gate** (`results/gate_verdict.json`, 11:20): no main pick, not promising, so the main line fails at the gate (D31).
  The screen has no pick. Rule §8: round 8 ends at the gate, AFF stays the current best, no held read and no
  `HELD_RULE.md`. Both C13 re-derivations agree (0 disagreements). NBC triggered and ran; it did not pass.
- **Drops:** FC4, FC5, FC6 by D21 (killed at about 3,601 s, 297 to 299 of 300 epochs; run time grows with the FCMI
  weight). V100 to V33 stayed in under amendment 1.
- **Phone notes sent:** FC4's drop (06:15) and the gate verdict (11:33).
- **Reports:** Codex drafts (gpt-6.1-sol, medium; job ledger row added), committed with the index row in
  `reports_sum.md` (7f01217); `check_reports_sum.py` OK, `docs-check` silent. Codex says all 42 gate rows match the JSON.

## 4. Open points for the next step

- **Claim review of the auto report** (Sonnet, `high`, worker): re-derive every number in the report from the result
  files (`report_numbers.md` maps each to its source; do not trust it), check the results brief against the template
  (our reading and recommendation only in the collapsed block), and check that amendment 1 and the D21 drops are in
  the first lines. Fix, then update the index row's last sentence ("Draft by Codex; …").
- **User-read report checks**, per `~/.claude/rules/user-read-reports.md` and `~/.claude/references/user-read-report.md`:
  a cold read on `haiku` and a fidelity check on `sonnet`, in parallel. Fix what they find.
- Then commit, write the r9 decide handoff (it opens with step 1: show the results brief without our reading), and run
  `loop-next <handoff> --label "r9 decide" --new-loop --notify` (`~/.claude/references/loop-chats.md`).
- **Minor, for the r9 wrap-up:** (a) the build's process point: build 4's fixer read the context meter's message as
  untrusted output; the `worker` body could carry "the context meter is ours". (b) The queue logs a D21 drop as
  "failed" (exit 247), not as "dropped under the rule"; the summary line says "3 failed". (c) The held ledger gets no
  H6 row, since there was no held read.

## 5. Code and pitfalls

- The results tree `src/test/20261129_r8_grouping/results/` and the archive `results_m4_09999/` stay untracked. Do not
  commit them.
- Nothing is deleted in this chat (unattended). New rows on `.scratch/pending_deletions.md` (2026-10-13 01:20 and 11:45).

## 6. State at handoff

- Running: nothing (both queues ended; no subagent, no Codex job).
- Uncommitted: only the untracked results trees above.
- On disk over 1 GB: real FCMI checkpoints `/data/SSD2/cosir_r8/fcmi/FC1` to `FC6` (about 27 GB), the archived
  `FC1_m4_09999` and `FC3_m4_09999` (8 GB), `results/` (7.2 GB on the system disk, 6.1 GB of it readers and heads),
  `results_m4_09999/` (2.1 GB), the smoke folder `/data/SSD2/cosir_r8/fcmi/smoke/` (66 GB, listed in rows of builds 2 to 5).
  All are on `.scratch/pending_deletions.md` for the r9 wrap-up's storage summary.
