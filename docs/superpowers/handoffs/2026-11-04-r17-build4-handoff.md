# Handoff: r17 build 4: fix wave, scoped re-review, merge into main, run handoff

## Brief
- Done: the final whole-branch review ran. It found no blocker and re-derived the key numbers. The stage 3 smoke passes.
- State: three should-fix items and small notes are open; the code is on `r17-new-scores` (c615e8d), not yet on main.
- Next: fix wave (Codex has room), scoped re-review, merge into main, write the run handoff, open "r17 run".
- Blocked: nothing. The run should start by about 01:30; stage 3 will likely end after 11:00.

> Written 2026-11-04 00:05 by tab "r17 build 3" (context meter past 200k). Loop step reached: step 7 (build), final
> review done, fix wave not yet run. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet
> "r17 spec approved".

## 1. Read in this order

1. The controller notes, `/project/CoSiR/.scratch/r17-new-scores/notes/controller_notes.md`, section "r17 build 3"
   only: the review records and the smoke record.
2. The build 3 handoff, [2026-11-03-r17-build3-handoff.md](2026-11-03-r17-build3-handoff.md): §3 (replay table,
   costs), §4 item 4 (what the run handoff holds), §5 (pitfalls), §6 (deletion list).
3. `~/.claude/references/codex-jobs.md` before the Codex job.
4. The spec only where a fix needs it: [2026-11-03-new-scores-design.md](../specs/2026-11-03-new-scores-design.md)
   (§3.6 for R0).

Skills: none.

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D20), approved 2026-11-03 20:45 from its brief; no container restart (user).

## 3. Where things stand

**Integration branch** `r17-new-scores` (worktree `/project/CoSiR/.claude/worktrees/r17-int`) at c615e8d, pushed, clean.
Main has moved past its base (r19's commits).

**Final whole-branch review (Opus `xhigh`, three workers).** The first stopped at the 200k meter after reading; two
fresh ones with narrow scopes finished it. No blocker. Records: controller notes; scripts and logs in
`/tmp/claude-0/-project-CoSiR/705bc8f1-cdf9-45fc-8840-a1bbf082633a/scratchpad/review/` (`a/`, `b/`).

- Scope A (claim cells, clustering, outcomes): PASS. 412 cells equal on every field; bars WIN 332, LEVEL or WIN 80;
  class counts; one class-count cell with 200 draws on MMArt validation, plain and matched PercepT, diff 0.0;
  `stage1_ranges` diff 0.0; `decide` on 325 void scenarios, 0 differences; `build_status` outside smoke 35 of 35.
- The real row route (`_cand_parts`, `_comp_parts`): equal to r14, r10 and r12 stored partitions on ArtELingo, EmoSet
  and Affection; MMArt AMI at 20 diff 0.0. This closes the first worker's gate-coverage finding as a one-off check, so
  no gate item is added.
- Scope B (retrieval, gate records, tests): PASS with fixes. `paired_ret` none, apart and shared agree to about 1e-16;
  untold join under 1.6e-8; 251 stored expected values read back from source with 0 mismatches; every item of §3.5
  (i), (ii), (vi) has a record; "Amendments: none" holds. 9 of 11 sabotages caught (F2, F8 below).

**Smoke through the runner, real inputs (`scratchpad/smoke3`, all exit 0):** `r0`, `per-class` (100 s), `sweep`
(CLIP L/14 emotion k 9 and 20, 72 k-means runs, 448 s, so about 48 min per painting item at load 60), `figure`,
`tables`, `told-untold` (741 s), gates `i_a_r14` (49 s, smoke draws) and `ii_scorer` (881 s), `cells --task ret` on
MMArt SigLIP 2 with stub picks (SC7, SC8, 1,158 s at smoke draws and load 60; the plan's 12 min is too low),
`qwen-crl --aspects emotion` (236 s for one aspect, 6 clustering rows and 1 retrieval row).

**Tests:** the full round suite has not run on c615e8d (queue busy). Direct: `test_ns_outcome` 113, `test_ns_ret` 42,
`test_ns_clu_cells` 29 pass. `test_spec_fragments_in_spec_text` fails only on copies without `docs/`.

**The ticket list** (for the run handoff): the table in `/project/CoSiR/.scratch/r17-new-scores/issues/00-index.md`,
all 11 `Status: done`.

## 4. Open points for the next step

None for the user. Agent steps, in order:

1. **Fix wave.** `codex-usage --room implementer` said room at 23:52 (5h 96% left). One Codex job (`gpt-6.1-sol`,
   high, verdict code in F1) in a worktree on a branch from `r17-new-scores`; else a Sonnet `high` worker. Items:
   - F1 should-fix: `ns_ret.score_cells` (`ns_ret.py:201-205`, `:267-273`, `:298-306`) raises KeyError on an absent new
     pick and loses the whole `cells --task ret` step. Skip only the SC of that pick and go on (§3.5: "a pick missing:
     that arm's cells"). Test with one pick absent.
   - F2 should-fix, test only: `test_ns_ret.py:265-266`, the fake `recorded_range` ignores the encoder, so
     `ns_ret.py:296`'s encoder argument is not pinned. Key the fake on `enc`; assert each cell's `r` per encoder.
   - F3 should-fix: `ns_diag.r0_rows` gives validation cross-fit values. §3.6's R0 rows follow `mt_diag.task_ret_r0`
     (`mt_diag.py:250`): `none+k0` on the held or test split through `ns_ret.arm_ap` for the new arms, with the
     validation value beside it. A missing pick stays listed.
   - F4 note: in `ns_run._step_status`, outside smoke, a step with `smoke: true` counts as `failed`.
   - F5 note: in `ns_diag.main`, outside smoke, assert that `records/verdicts.json` exists (§3.6, after the verdicts).
   - F6 note: `ns_run.cmd_gate` returns 1 when any status entry is `failed`, not only on an exception.
   - F7 ours: `ns_paths.write_json` writes atomically (temp file beside it, `os.replace`), since parallel chains write
     step files and `status.json`.
   - F8 note: a test that stubs `paired_ret` and asserts the mode: `shared` in `told_untold`, `none` in `per_class`
     and `qwen_crl`.
   - F9 ours: `ns_run_cpu.sh`: `cells_ret` est 25m, its comment and the header's total; `told_untold` stays 15m.
   No change for: the rounding note on identical arms; over-voiding when `gate_v` or `gate_iv` raises (safe).
2. **Scoped re-review** (Sonnet `high` worker) of the fix diff only, then the round's tests through `res-queue cpu`
   (`pytest -q src/test/20261103_new_scores`, about 2 minutes).
3. **Merge:** merge main into `r17-new-scores`, test again, then in the main checkout
   `git -C /project/CoSiR merge --ff-only r17-new-scores` and push. Use this form, not `update-ref`: the run imports
   the files of `/project/CoSiR/src/test/20261103_new_scores/`, so the checkout itself must move.
4. **Run handoff** (`2026-11-04-r17-run-handoff.md`) and `loop-next --label "r17 run"` (unattended), as build 3's §4
   item 4 says, plus:
   - **Three parallel chains:** one `ns_run_cpu.sh` line waits for its slot, so a stage alone runs in one slot. Use
     `bash ns_run_cpu.sh <stage> --dry | grep -v <what must come last> | xargs -d '\n' -P 3 -I{} bash -c '{}'`. Stage 0:
     `gate iv` after `iv_export`. Stage 1: ranges and picks first, `stage1-records` last, then commit. Check the form
     on one short step first.
   - **Read `ns_run.py status` after every stage;** `cmd_outcomes` lets `missing` rows through and only prints a count,
     and nothing stops `verdicts` on them: the run chat looks at that count itself.
   - **Records** are gitignored by `*.json`: `git add -f`.
   - **The round log** `src/test/20261103_new_scores/20261103_new_scores_log.md` (§9) has no owner yet: the run chat
     opens it at stage 0.
   - **Report items without code:** D4's "the ArtELingo retrieval values were seen before"; D5's "lower point value";
     the "not run" list in the text; the EmoSet sweep is Qwen only (SC4's scope, r14 has rows for the other three);
     CRL clustering has no stored partitions, so its route is proved only by the run and C13; the notes of build 3's
     §5.
   - **Time.** Slot-hours at load 30 to 60: stage 0 2.1, ranges 1.2, picks 4, clustering cells 5, retrieval cells 5,
     stage 3 about 11, C13 not known (r15: 5,178 s for 60 cells). In three chains stages 0 to 2 take about 6 h, so about
     07:30 from a 01:30 start; C13 then about 2 to 3 h; stage 3 about 4 h, so past 11:00. Weigh it as
     `shared-resources.md` says; CPU use past 11:00 is a strong preference, not a hard limit; §3.6's rule cuts the
     photo sets of the sweep first (from 12:00).
5. **Delete nothing:** worktrees and ticket branches go on `.scratch/pending_deletions.md` at the end (build 3's §6
   list, plus the fix wave's worktree, this chat's scratchpad and the review scratch).

## 5. Code and pitfalls

Build 3's §5 holds. Two more:

- A smoke writes new test numbers. Nobody reads them before stage 1 is committed: check structure and exit codes only.
- `subagent-watch --watch` exits at once when no subagent is live; start it after the dispatch. A worker idle on its own
  queue job counts as not running.

## 6. State at handoff

- Running: nothing of this chat (no subagent, no queue job, no Codex job). GPU check: driver 595.99.02 answers; run mode night.
- Uncommitted: nothing of this chat's. Notes, tickets and smoke results in `.scratch` are outside git by design.
- On disk over 1 GB: nothing new. This chat's scratchpad is 31 MB.
