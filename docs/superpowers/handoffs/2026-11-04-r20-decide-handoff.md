# Handoff: r20 decide: read r17's results with the user (decision C), wrap up r17, then decide what comes next

## Brief
- Done: r17 ran every stage, and the independent re-derivation passed for all 412 cells. The auto and user-read reports are committed.
- State: c* WINs all 96 plain-image clustering cells. Against the teacher, AMI and NMI WIN 46 of 48, ARI and ACC 22 of 48, with 5 BELOW, all emotion.
- Next: step 1: show the results brief without our reading and record the user's reading. Then wrap up r17 and go to step 2.
- Blocked: nothing. r18 and r19 run in their own chats.

> Written 2026-11-04 09:32 by tab "r17 run". Loop step reached: r17 step 9 done; step 1 next, with decision C. Direction log entry: none
> yet for r17; the wrap-up writes it.

## 1. Read in this order

1. The results brief of the auto report, [2027-01-04_new_scores.md](../../reports/auto/percept/2027-01-04_new_scores.md).
   The collapsed block holds our reading: keep it closed until the user gives theirs.
2. The user-read report, [2026-11-04_new_scores.md](../../user_read/2026-11-04_new_scores.md). Its advice is folded away the
   same way.
3. The direction log's "Now" and the plan in the
   [experiment overview](../../user_read/2026-11-03_experiment_overview.md) (lines near 262 to 268: r20, r21, r22).
4. The r17 spec, [2026-11-03-new-scores-design.md](../specs/2026-11-03-new-scores-design.md), only for a question on the rule.

Skills: none for steps 1 and 2; `grilling` and `domain-modeling` at step 5 if the loop has a grill.

## 2. Decided by the user, not to be reopened

- The six-loop plan, r17 to r22 (decision A, 2026-11-03 18:30, direction log).
- r17's decisions D1 to D20 (spec §4, approved 2026-11-03 20:45), with the who-decided column there.
- Bars and picks of r17 stay as they are: ranges are not changed after results (spec §8).

## 3. Where things stand

**Result.** There is no overall verdict: each metric cell is its own claim. 373 WIN, 32 LEVEL, 7 BELOW; 398 give a claim.

|Comparison|Cells|WIN|LEVEL|BELOW|Claims|
|---|---:|---:|---:|---:|---:|
|c* / plain image, clustering|96|96|0|0|96|
|c* / PercepT, clustering|96|93|3|0|93|
|c* / teacher, clustering on paintings|96|68|23|5|91|
|Photo model / plain image and teacher|40|38|2|0|40|
|c* / CRL, clustering (CLIP L/14)|30|26|3|1|26|
|c* / CRL, retrieval (CLIP L/14)|6|6|0|0|6|
|Untold c* / plain image, retrieval|24|22|1|1|22|
|c* / PercepT, retrieval|24|24|0|0|24|

- **Checks:** 16 stage 0 gates pass; 161 status items ok, 0 failed, 0 missing; 0 void cells. The re-derivation compared 22,615
  items, with no failure and a largest difference of 3.5e-14.
- **Prior against result:** the named surprise (an ACC or ARI BELOW against the plain image where AMI wins) did not happen.
  Untold emotion beat its prior: 8 of 8 WIN against 50%. Teacher cells kept their AMI-at-20 outcome in 51 of 84 cells, against 70%.
- **Time:** the run chose to go on past 11:00 if needed (CPU only, §3.6's 12:00 cut applies). It was not needed: stage 3 took
  about 1 h 53 min and all CPU use ended at 09:02. Nothing was cut.
- **Records:** `src/test/20261103_new_scores/records/` (`verdicts.json`, `c13.json`, `status.json`, picks, ranges), round log
  in the same folder. Diagnostics: `/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/`. All on main, ebb036cf.

**Other loops at this time:** r19's run holds the local GPU. r18's TPIPS stores end about Thursday 23:00, so its stage 1 runs
after that. Both have their own run chats and reports.

## 4. Open points for the next step

For the user:

1. Decision C: what do these results mean? Show the brief first, without our reading.
2. Decision A for r20: the report's options are the paper tables now, intervals at the other cluster counts, or the planned
   loops (r20 figure, r21 head variants, r22 text queries). Abstract 2026-11-10, paper 2026-11-16.

For the agent, at the wrap-up:

- Direction log entry for r17 and a new "Now" (`~/.claude/templates/direction_log_entry.md`).
- Storage summary: r17 left 183 MB on SSD2 and nothing of 1 GB or more. Read every row of `.scratch/pending_deletions.md`,
  r17's included (c13_smoke, c13_round_smoke, the scratchpad, the build's worktrees and branches, worktree-agent-a3a6eb4dd7fdff1b9).
- `wrapup-check`, then `/project/claude-config/sync.sh`. `check_reports_sum.py` prints 4 problems: all are weekly drafts
  of another chat, not r17's.

**For the paper:** §2.3's closing paragraph claims a student that exceeds its teacher. It must now name the metric. The
report's section "How much this changes the paper draft" lists what changes; Sec. 4 does not exist yet.

**Not done, on purpose:** the teacher prior for ArtELingo Qwen's 12 cells (their AMI-at-20 outcomes are not in r14's or r15's
records; older rounds not checked). The auto report is about 18 KB against a target of about 15 KB, because of the items that
no code writes.

## 5. Code and pitfalls

- `ns_score` is frozen for r18 and r19 (spec §3.7). The one code change after the build is in `ns_diag.qwen_crl` (bd8717c2): it
  changes no signature, and no verdict reads it.
- The re-derivation's source fingerprint is an AST hash of `ns_c13.py` without `compare`, `certify` and `main`. Any other edit
  there makes all 52 sections stale, about 10 slot-hours.
- Review notes kept, not fixed: `compare` would give a false FAIL only if a stage 1 range were not ok (all 116 are ok); hand edits
  to `sentence`, `present` or `r_source` in the records would not be seen.

## 6. State at handoff

- Running: nothing of this chat (no subagent, queue job or Codex job).
- Uncommitted: nothing of this chat's.
- On disk over 1 GB: nothing new.
