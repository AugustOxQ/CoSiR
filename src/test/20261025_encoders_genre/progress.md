# r10 build 21 final fix i-ii handoff

Single-agent work on `r10-fix-i-ii`, from `b4e11f0`, within EG. Read all three supplied reviews, EG rule §§2.1, 2.2, 4.1, 4.2 and RB §4.3. No GPU, push, CCG records or stage iii edits. Real outputs were read only; all runners and simulations used scratch outputs. No project/output files were deleted.

## Per-item changes and evidence

Paths below are relative to `src/test/20261025_encoders_genre/`.

| Item | Change (file:line) | Red/green or queue dry-run proof |
|---|---|---|
| 1 | `eg_queue_stage_ii.sh:24` distinguishes an existing failed X0/X1 from an absent gate. An actual X0 failure permits the old-arm path even when X1 never ran. `eg_run.py:314`, called by non-smoke `eg_pinfer.py:263`, refuses a plan lock until every non-deferred candidate has a stage 3 block, unless a recorded X0/X1 failed. | `test_eg_final_i_ii.py::test_non_smoke_refuses_premature_plan_lock`: absent, partial and missing-X1 cases red before fix, now green. Completion, recorded deferral, failed X0 and failed X1 cases green. `test_vl4_refuses_added_arm_or_seed_even_when_forced` covers both changes with and without `--forced-vl3`; deleting the VL4 call caused 4 failures, deleting the readiness call caused 3. Both restored. Queue A/B/C stop before pinfer. |
| 2 | `eg_pinfer.py:84`, `eg_lift.py:287`, shared reason lookup `eg_run.py:322`: a new encoder without a stage 3 outcome is missing even when weights/exports exist; the block's `v_failure` is preserved. Deferred encoders and failed-X0/X1 paths also omit new arms. Weight-missing paths are recorded relative to the outputs root. | `test_weights_do_not_rescue_a_missing_stage3_outcome` (no outcome, absent block, deferral, failed X0) and `test_eg_lift.py::test_stage3_no_outcome_is_missing_even_with_all_p_exports` were red, now green. |
| 3 | `eg_matched.py:74`, `:109`, `:258` add selectable inference kinds in this round, preserving RB's training/resume guard. `eg_queue_stage_i.sh:31` and `:36` pass `--kinds held` to both arms at sample and full scales after an A28-only V5 failure. `eg_v5.py:203`, `:212`, `:218` atomically save every completed dataset check, catch per-dataset exceptions and retain their causes. `eg_score.py:317` blocks A with the A28 V5 cause. RB sources are unchanged. | `test_matched_held_gate_does_not_need_failed_a28_store`, `test_v5_records_pass_before_a28_exception_and_keeps_its_cause`, `test_failed_a28_v5_cause_precedes_outputs` were red, now green. The two real held-only matched driver tests never touch A28 data or write A28 outputs. The stage i stub run asserts both held paths at both scales, retained S1/S2 progression, and an A no-outcome block. Copied reviewer sim_f1 still refuses the default held+A28 path, while the new held-only call passes. |
| 4 | Removed the environment deferral route from `eg_queue_common.sh:81`, `eg_queue_stores.sh:7`, stage i/ii queues and the allowed documentation parts. Lists, including dry-run lists, now come from the driver and `deferral.json`. `eg_run.py:261` also supplies stage 5's reporting list and `--include-done`. The existing `EG_DEFER` output label of the `defer` command is retained; it is not an environment deferral. | `test_recorded_deferral_controls_candidate_lists_and_check3` confirms the recorded list and successful check3. Store and stage i dry runs create a real estimate-based K5 deferral, set the obsolete environment value to K6, then prove K6 runs and K5 does not. Copied sim_f2 confirms an env value alone does not satisfy check3; the recorded deferral does. |
| 5 | `eg_score.py:118`, `:132` check the full-store V5 result before reading held outputs, preserving the original cause. `eg_v2.py:125` and `eg_run.py:151` explicitly say no candidate has an outcome when nothing can be re-derived; the stop remains. Queue stop messages describe the V2 stop without claiming a disagreement in that case. | `test_failed_v5_cause_precedes_missing_held_output` and `test_v2_no_outcomes_message_keeps_the_stop` were red, now green. |
| 6 | `eg_run.py:332` omits no-outcome candidates from the A28 inference plan. `eg_score.py:317` writes their stage 5 block with `v_failure = "no stage 3 outcome: <cause>"` before any A28 statistic. `eg_queue_stage_i.sh:100` scores/checks every non-deferred stage 3 block, including candidates excluded from inference, so their causes remain reportable. | `test_no_stage3_outcome_is_left_out_of_a28_plan_and_gets_a_no_outcome_block` was red, now green; its statistic stub fails if called. Stage i dry run includes the blocked candidate in stage 5 scoring/check5. |
| 7 | `eg_queue_stage_ii.sh:15` skips the impossible stage i report wait after an actual X0/X1 failure. `:35` passes `--n-rows 29232` with smoke inference. | Dry runs cover failed X0 with no X1 files, failed X1, the standalone held command with/without a failure, and full-P smoke. The smoke stub derives the wrong hash unless the row argument is present; the real VL1 gate passes and the queue reaches lift-score, VL2, stats, then held. |
| 8 | `eg_pinfer.py:230` verifies the saved export sidecar's weights hash and code commit before either student or own-arm re-inference. Moved-aside exports, including after `complete: true`, are logged in `vl4_log.jsonl`. Detailed failed-VL3 records remain in forced-rerun log entries. | `test_forced_export_rerun_requires_same_weights_and_code` was red for changed weights/code in each export kind, now green; unchanged identity permits re-inference and logging. Both `test_moved_export_after_complete_is_logged_as_a_rerun` cases were red, now green. |
| 9 | `eg_report_i.py:95`, `:255`, `:293`, `:324`, `:385` render no-outcome causes, accept failed exception entries, refresh descriptive stores on either source hash change, and write Markdown even when no candidate has an outcome. `eg_queue_cpu_i.sh:13` permits cause/store reporting when V2 has no outcome to re-derive; reported outcomes still require V2. | `test_no_outcome_markdown_lists_each_cause`, both `test_no_outcome_report_writes_stores_and_refreshes_changed_hashes` cases and `test_stores_report_accepts_exception_entry_with_its_cause` were red, now green. An additional cpu_i stub test reproduced the queue's V2 stop red, then went green with the cause-reporting path. |
| 10 | `test_eg_score.py:548` runs `eg_score.run_stage -> stage_block -> rb_score.candidate_block -> outcome` at exactly `d=.01` with positive lower bound. Only the statistic is stubbed. | The original code is green. Temporarily changing RB's `d >= .01` to `d > .01` caused this stage i test to fail; restored exactly, then green in the final stage i run. |
| 11 | No CPU implementation change needed: RB score/report pool initializer, lift pool initializer and VL2 chunk scope already limit threads to one. | `test_every_cpu_worker_limits_blas_and_omp_to_one_thread` inspects all eight distinct worker PIDs in each of cpu_i, lift and VL2; every loaded BLAS/OMP runtime reports one thread. All three cases green. Two concurrent eight-worker pools therefore have at most sixteen active BLAS/OMP threads. |

## Test and mutation records

All logs and scratch trees are retained under `/tmp/claude-0/codex-fix12/`.

- `red.log`: initial focused baseline, 24 failed / 11 passed (4.56 s).
- `green2.log`: focused regressions, 35 passed (4.40 s).
- `cpu_queue_red.log`: additional no-outcome cpu_i queue reproduction, 1 failed before the queue fix.
- `focused_final1.log`: 48 focused verdict, matched-driver, worker and queue checks passed (9.26 s).
- `mutation_vl4_guard.log`: deleted VL4 call, 4 failures.
- `mutation_ready_guard.log`: deleted readiness call, 3 failures.
- `mutation_s1_threshold.log`: `>=` changed to `>`, 1 threshold failure.
- All three mutations were restored byte-for-byte. There is no RB source diff.
- `final_stage_i.log`: **69 passed**, 136.07 s, for `test_eg_score.py`, `test_eg_v2.py`, `test_eg_report_i.py`.
- `final_stage_ii_queues.log`: **132 passed**, 81.54 s, for `test_eg_lift.py`, `test_eg_vl2.py`, `test_eg_final_i_ii.py`, `test_eg_queue_final.py`, and a repeat of the 10 reporter tests after its final message change.
- **Final total: 191 unique passing tests** (201 executions in the final split runs, including the 10 reporter repeats). The 12 queue tests use stub trees and real gate/list decisions; heavy runners are faked. Each bounded final command used `timeout 230s` and the requested environment/options.
- `bash -n` passes for common, stores, stage i, stage ii and cpu_i queues; `git diff --check` passes.

Existing A28/lift test fixtures now explicitly provide successful stage 3 prerequisites; the separate no-outcome regressions exercise the refusal. The smoke-path test explicitly sets an allowed root, because the requested pytest temp root itself lies within `/tmp/claude-0/`. Statistical expectations were retained.

## Reviewer reproductions

Original reviewer scripts were copied verbatim into `reviewer/stage_i/` and `reviewer/stage_ii/` beneath the scratch folder. `replay_reviewers.py` adapts checkout/scratch paths in memory without editing those copies or the originals. `reviewer_replay.log` records:

- sim_f1's held+A28 refusal and the added successful matched held-only call.
- sim_f2's ignored env value and successful recorded-deferral check3.
- Stage ii A/C stopping on absent gates and B stopping on pending stage 3 before pinfer.
- The first non-smoke lock reproduction refusing before writing an index.
- Original case D reaching the scorer call; its old fake Python deliberately writes no VL2 result and therefore stops at that gate. The enhanced queue tests separately verify the no-outcome encoder, smoke progression and held-after-failure paths through successful completion.

## Files that could be removed later

Nothing was deliberately deleted. Optional cleanup after review: `/tmp/claude-0/codex-fix12/` (pytest trees, stub outputs, copied review scripts and logs), and the EG planning notes `task_plan.md` and `findings.md`. This handoff is retained in `progress.md`. No real-output deletion is proposed.
