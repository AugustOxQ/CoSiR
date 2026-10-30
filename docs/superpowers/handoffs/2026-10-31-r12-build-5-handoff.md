# Handoff: finish the r12 build (review the Codex fix wave, fix vh stage 1, a fresh smoke, re-review, merge to main), then open the run chat

## Brief
- Done: the end-to-end smoke ran to vh stage 1 (4 contact fixes); the final Opus `xhigh` review: pass after fixes.
- State: the fix wave (diag assert, c* clustering on stored float16 rows) is built by Codex on `r12-fixwave`
  (`a0c8038`, 10 tests, 4 mutations caught), not yet reviewed; vh stage 1 has 14 open contact mismatches.
- Next: review and merge `r12-fixwave`; fix vh stage 1's ArtELingo comparison; a fresh smoke from stage 0b to stage 4;
  a Sonnet `high` re-review; merge to main; the run handoff and the run chat.
- Blocked: nothing.

> Written 2026-10-31 00:08 by the r12 build chat 4 (cut at the context meter, 200k). Loop step reached: step 7 (build), all
> tickets merged; the final review is done, its fixes and the smoke are left. Direction log entry:
> [2026-10-30](../direction_log.md).

## 1. Read in this order

1. The round log's "Build chat 4" part, on `r12-affection`:
   `src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md` (smoke times, contact fixes, the
   review's findings, the reviewer's re-derivation).
2. The ticket index `.scratch/r12-affection/issues/00-index.md` and ticket 12 (`12-queue.md`; its smoke box is open).
3. The spec, from §1 on: [2026-10-30-affection-confirmation-design.md](../specs/2026-10-30-affection-confirmation-design.md).
4. `~/.claude/references/matt-chain.md` ("The final whole-branch review": one fix wave, then a scoped re-review on
   Sonnet `high`), and `~/.claude/references/codex-jobs.md` before a Codex job. The earlier build handoffs' pitfalls
   still hold: [build 4](2026-10-30-r12-build-4-handoff.md) §5 and the ones it names.

Skills: none.

## 2. Decided by the user, not to be reopened

All in the spec's §4 and the direction log, 2026-10-30; the spec was approved from its brief at 21:17.

## 3. Where things stand

- `r12-affection` (pushed) holds all 13 tickets, the ticket 13 review fixes (`931fae7`) and four smoke contact fixes
  (`d3e054a`, `3f4fd5d`, `84f8e11`, `910e527`), each with a test.
- **The smoke** (`ac_queue.sh smoke`, 2,100 keys per split) passed every step up to and including stage 1 and vh 0b;
  it stopped at vh 1 (below). Its folder:
  `/tmp/claude-0/-project-CoSiR/877a08b8-2b87-407d-ae98-84283a075758/scratchpad/r12_smoke/` (step times in the round
  log; GPU steps about 12 minutes).
- **The final whole-branch review** (Opus `xhigh` worker at `910e527`): pass after fixes.
  - Must-fix: `ac_diag.py:187` asserts six stage 2 arms; stage 2 writes ten, so diag always fails.
  - Should-fix: no test for `ac_diag.py`; c* clustering in stages 0b and 2 uses re-normalized float32 rows, while r10,
    the 0a(iii) replay and vh use the stored float16 rows; C13 (vh) runs after a verdict file is written.
  - 28 mutations, all caught. SC table, bootstraps, outcomes, verdict logic, stage 1 rule, seed ranges, held-data
    order, stage 0a and joins traced as matching the spec.
- **The reviewer's own re-derivation on the smoke** (numpy and scikit-learn only): every stage 0b and stage 1 value equals the smoke:
  r_clu, r_map, r_nDCG per seed (diff 0); the teacher AMIs; the clustering and retrieval wording picks; the read-out
  picks of both teachers and of c* on ArtELingo (`center+k10`, diff 0); label counts equal the spec gate. Read-out mAPs
  agree within 5e-8 (scores in units of r then differ by about 5e-5; same picks). Output and script:
  `.scratch/r12-affection/build4/review/` (`stage01.txt`, `rederive.py`).
- **The fix wave** (Codex `gpt-6.1-sol` high, branch `r12-fixwave`, worktree
  `/project/CoSiR/.claude/worktrees/r12-fixwave`): the diag assert with a new `test_ac_diag.py`, and c* clustering on
  its stored float16 rows in stage 0b (r_clu), stage 2 (SC1, SC3) and diag. Done in about 5 minutes (`a0c8038`): `ac_data.stored()` returns c*'s checked float16 rows; `ac_run.py:355`,
  `ac_stats.py:52` (r_clu only; mAP and nDCG keep unit rows) and `ac_diag.py:129` use it; the diag assert takes ten
  arms (Codex audited every later diag field for the CRL arms); `test_ac_diag.py` runs diag on runner-made stages 0b
  to 3 and stage 4 on a small fake Impressions store. 512 fast tests pass; four mutations caught (its report).
  Not yet read by Claude. Brief, log and the review brief: `.scratch/r12-affection/build4/`.

## 4. Open points for the next step

None for the user. For the controller, in order:

1. **Review the fix wave's diff** (`git diff r12-affection...r12-fixwave`), run its tests, and merge it into
   `r12-affection` if it holds. Verdict code (c* clustering): read the diff closely. Codex brief and log:
   `.scratch/r12-affection/build4/codex_fixwave.txt` and `.log` (session `01a18f0b-b4d9-7332-9e4a-e672be11239a`, for
   one fix round by resume).
2. **Fix vh stage 1** (`ac_vh.py`, `compare_final`, stage 1). vh's 14 mismatches (`vh/1.json` in the smoke folder) are
   all rows the runner does not hold: for ArtELingo's fixed-pick arms vh scores the full read-out menu (c* and CLIP CRL),
   the runner only what spec §6 needs (CLIP CRL at R0 and `none+k10`; c* none, since its pick and stage 5 fit need no
   cross-fit row). Every shared row agrees. Fix: vh compares only the rule's rows for those two arms. Better also: vh
   skips computing the rest; they are most of vh's 20 minutes per stage, and vh 2 and vh 3 recompute stage 1 inside.
   Add a test; it is verdict-check code (Sonnet `high`, or Codex if the fix wave's session can take it by resume).
3. **A fresh smoke from stage 0b on**, since the c* clustering change alters stage 0b and stage 2 inputs. The old
   folder holds sealed stage 0b and 1 outputs: copy everything except `stage0b/`, `stage1/`, `records/`, `vh/`,
   `partitions/` and later stages into a new scratch folder (the GPU-step outputs are reused), or rerun the whole smoke
   (about 12 minutes of light GPU). Delete nothing. Expect vh 2 and vh 3 to take about 25 minutes each. Record each
   step's time in the round log.
4. **The re-derivation of SC values**: once the fresh smoke has `stage2/verdict.json` and `stage3/verdict.json`, run
   the reviewer's script: `cd /project/CoSiR/.scratch/r12-affection/build4/review && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
   /root/miniconda3/envs/CoSiR/bin/python -u rederive.py sc <smoke folder> > sc.txt 2>&1` (SC1, SC2, SC8 and the
   verdict line; bar 1e-6). Check that it reads the stored float16 c* rows for clustering after the fix wave.
5. **A scoped re-review** of every change after `910e527` (Sonnet `high` worker), then merge `r12-affection` into main
   and push (`matt-chain.md`); list the worktrees and branches on `.scratch/pending_deletions.md`.
6. **The run handoff** with the ticket list (index table) and the run chat (`loop-next`), after the GPU check (night
   mode: the check only). Plan the run for 2026-10-31 (Saturday, spec §9). In the run handoff: a verdict counts only
   once its `vh/<stage>.json` passes (the queue stops on a failed vh); the full run's vh steps are long (estimate them
   from the smoke's times with 4,251 validation rows), so estimate the end time against 11:00 on Sunday if it runs
   into the night.

## 5. Code and pitfalls

- Integration worktree: `/project/CoSiR/.claude/worktrees/r12-int` on `r12-affection`. Main is untouched.
- The queue skips a step whose seals exist; stages 0b and 1 are now sealed by the outputs file and the record copy.
  A changed input does not unseal a step: use a fresh folder when code before a step changed.
- `ac_cluster.cached_partitions` is keyed by the hash of its float32 input rows, so a changed c* input is recomputed.
- Workers cannot run `bash <script>` here (the sandbox guard refuses a script that contains `git`): the controller runs
  the queue and its dry-runs. Workers must not run CPU jobs over about 4 minutes; slow tests run under the controller.
- The worker of the ticket 13 fixes could not stop its own long pytest (the kill was refused); it ended by itself. Say
  "no command over 4 minutes" plainly in each brief.
- Unattended: delete nothing; this chat's worktrees (`agent-aeb053cf77ef63c53`, `r12-fixwave`) and branches
  (`r12-t13fix`, `worktree-agent-aeb053cf77ef63c53`, `r12-fixwave`) and the smoke folder go on
  `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing (no subagent, no Codex job, no GPU lock, no background command). GPU check at 00:07: driver
  595.99.02 answers, module and library agree, GPU free. Run mode: night. Codex: 94% of the 5-hour window left.
- Uncommitted: nothing of this chat's in CoSiR (`.scratch/` is outside git).
- On disk over 1 GB: nothing new (the smoke folder is about 280 MB, on the system disk under the scratchpad).
