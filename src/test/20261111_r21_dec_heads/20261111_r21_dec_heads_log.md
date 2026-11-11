# r21: DEC loss on every head of c*: round log

Spec: `docs/superpowers/specs/2026-11-11-r21-dec-heads-design.md`. Decision lines: `decisions.md`.

## Build (2026-11-11, build chat)

- 2026-11-11 06:37: tickets 01 to 06 written (`.scratch/r21-dec-heads/issues/`); integration branch `r21-dec-heads`.
  Codex had no room for implementer jobs (five-hour window 61% left, 80% needed), so Claude workers build.

## Build chat 2 (2026-11-11, from 06:51)

- 06:54: **gate 0a PASS** (full run, `r21_score.py gate0a` at `r21/t05` a282ae62, 155 s on one CPU slot). Test AMI at
  the class count, 5-seed c* means equal r17's file: 0.102413 / 0.410926 / 0.510070; teacher 0.088875 / 0.406315 /
  0.501466; all diffs 0.0. c* mAP per seed equals r20's `ablation_ret.json` for all five seeds, max |diff| 0.0.
  Record: `/data/SSD2/cosir_r21/20261111_dec_heads/scores/gate0a.json`.
- 06:55: reviews of 02 and 05 (Opus `high` workers) and builds of 03 and 06 (Sonnet `high` workers) started in
  parallel; 03 and 06 branch from the integration tip with the unreviewed 02 and 05 merged in (review fixes merge later).
- 07:08: ticket 05 reviewed (Opus `high`: F1 to F9; F1 to F5 should-fix) and fixed (86095ff9, Sonnet `high`
  fixer; F9 kept); 72 tests pass; `val` on saved c* seed 0 still equals `stage1_ranges.json` (diff 0.0). Merged.
- 07:15: ticket 02 reviewed (Opus `high`: no code defect on the DEC-off path, which matches `ct2_d2.run` line by
  line; 7 of 16 hand mutations survived the tests) and fixed (ce46aa8e, Sonnet `high` fixer): new tests kill all 7;
  the teacher form now asserts its feature rows are in the head's fit order (C6); the parity test covers a distilled
  row and a patience stop. Merged; the three test files pass on the integration branch, each in its own process (07:19).
- 07:19: the three test files in one pytest process in the worktree give 1 failure and 11 errors in
  `test_r21_score.py` (each passes alone: 72, 29, 27). Cause: frozen `20261021_d2_ablation/ab_paths.py` (via r17's
  `ns_paths.add_rounds()`) pins the main checkout's round folders and asserts imports come from there, while
  `r21_train.py` adds the worktree's own folders. In a worktree the two copies clash; on main they are the same files.
  Rule for this build: run test files one per process in worktrees; the final check runs them together on main.
- 07:25: ticket 06 built (fb9ecc6c, Sonnet `high`; 37 tests). **Regression exact**: `table` on r10's removal arm
  `k11=40-no_buddy` gives AMI at 20 clusters (change, interval, both absolute values) equal to r10's `held.json` and mAP
  equal to `ablation_ret.json`, all 30 cells diff 0.0; c* against itself gives zero changes, zero intervals and "no
  clear change". 1,402 s on one CPU slot under load about 20 (two arms). Records in this chat's scratchpad
  (`t06/reg/reg_table.json`, `regress.log`). Review (Opus `high`) started.
- 07:42: ticket 03 built (05a5017c, f02b4257; Sonnet `high`). **Gate 0b PASS on CPU** (424 s; record
  `r21_gate0b_20261111_072007.json` in this chat's scratchpad `r21_smoke/`, written from an uncommitted state, so rerun on
  merged code for the permanent record): `dec-off` against the frozen chain, 11 fixed batches and a 2-epoch smoke (20
  steps, histories, best epochs, exports), all diffs 0. Mutations: loss × (1 + 1e-5) and λ = 1e-3 in the λ = 0 run both
  give FAIL and a non-zero exit. CPU smoke of all four forms (`dec-off`, plain S1, start S2, teacher S1 and S2): every head
  reaches DEC epoch 2, histories finite, sidecars carry c*'s config fields, rerun skips, r10's tree unchanged. **GPU smoke
  not run** (GPU held by r25:extract). Merged (0ded9542).
- 07:47: ticket 06 reviewed (Opus `high`: core right; F1 `complete` flag and default output, F2 row order resting on
  the sidecar's claim; F3 to F7 minor) and fixed (2d3917d9, Sonnet `high`): `complete` needs the r21 and r10 trees and
  the picked arms, the paper-facing stem is refused otherwise; held pairing also needs `image_sha256` equal to c*'s;
  DEC statistics through `r21_score.head_dec`; 53 tests; 11 mutations killed; emotion regression at full draws still
  exact (10 cells, diff 0). Merged.

## Ticket 06 (build, `r21/t06`)

- 2026-11-11 07:05: `r21_table.py` (CLI `table`, `diag`, `regress`) and `test_r21_table.py` (37 tests: word rule at the
  interval edges, pairing, cells, silhouette against scikit-learn, DEC statistics, two real-file checks on r10's tree).
  No existing file edited. Held (test) exports are read by design (stage 2); nothing here tunes or selects on them.

## Build chat 3 (2026-11-11, from 07:48)

- 07:55: Codex has no implementer room (five-hour window 60% left, 80% needed; resets 10:09), so Claude workers again.
  Started in parallel: review of `r21_gate0b.py` (Opus `high`) and ticket 04 (Sonnet `medium`, worktree
  `/project/CoSiR-r21-t04`, branch `r21/t04`). GPU smoke of ticket 03 queued (`res-queue gpu`, label `r21:smoke`,
  behind r25 and r26; estimated start 08:48).

## Ticket 04 (build, `r21/t04`)

- 2026-11-11: `r21_run.py` gained `held` and `teacher` (module docstring has the CLI); new `r21_queue_lib.sh` and
  `r21_queue_stage{0c,1a,1b,2}.sh` (each with `--dry-run`; 1b and 2 also `--picks FILE`). Plumbing check: held on r10's saved
  c* seed 0 weights, CPU inference, differs from r10's GPU export by at most 6.1e-5 (one float16 step) on 0.13 to 0.15 % of
  entries. Smoke: `train --arm dec-off --smoke` then `held`, sidecar fields equal r10's. Smoke output: scratchpad `t04/`.
- 08:05: gate 0b script reviewed (Opus `high`): sound for the spec's sentence, no must-fix. Its own mutation (best state
  only at epoch 1) gives FAIL. Should-fix: S1 the λ = 0 runs stop before step 11's backward, so a gradient-only DEC leak
  passes (its mutation m3 passed); S2 a one-part run prints PASS; S3 the saved-weights load is not counted. Minor: M1
  histories compared on three keys only, M2 the export glob takes any match, M5 `--mutate` exit codes, M6 the record
  only under `/tmp/claude-0`. All seven sent to one Codex fix job (`gpt-6.1-sol` high, worktree
  `/project/CoSiR-r21-fix0b`, branch `r21/fix0b`).
- 08:06: ticket 04 built (86f8e100, Sonnet `medium`); controller diff read and dry runs (1a: the 8 arms of §3.5; a
  failing pick read stops the script under `set -e`). Merged.
- 08:20: gate 0b fix (72414bc9, Codex `gpt-6.1-sol` high; S1 to S3, M1, M2, M5, M6) merged (84ce3c64); scoped re-review
  (Sonnet `high`): no must-fix. **Gate 0b PASS on the merged code** (525 s, one CPU slot): i-a, i-b1, i-b2, i-c over steps
  1 to 12 and ii-a to ii-d, all diffs 0; i-info (λ = 0 as the trainer is, step 11): style 0.0083, genre 0.0101 (dropout
  stream shift, kept). Record: `/data/SSD2/cosir_r21/20261111_dec_heads/scores/r21_gate0b_20261111_080857.json`.
  Mutations, each FAIL at its own check and "mutation detected": `grad_leak` (a gradient-only DEC term) at i-c only, 0.0421;
  `lam_leak` (λ = 1e-3) at i-b2, i-c and the λ = 0 provenance; `loss_scale` at i-a to i-c. The re-review's two
  should-fix points (exact history equality may give a false FAIL; a mutation counts as detected on any FAIL) are answered
  by these records and left as they are.
- 08:22: final whole-branch review started (Opus `xhigh`).
- 08:30: **final whole-branch review** (Opus `xhigh`; handed back at its 200k context limit, its two last checks read by
  the controller). Verdict: the claim holds with fixes; no must-fix; no defect that gives a wrong pick or a wrong word.
  Re-derived: gate 0a by its own route (scikit-learn MiniBatchKMeans and AMI, `ct_held` labels, `ct_diag` teacher rows),
  all diffs 0; gate 0c's saved values (best epochs 4, 1, 18; validation AMI 0.100984, 0.420558, 0.529839); flat q (equal
  to the decision line); 18 stage 2 regression values by hand and an independent per-seed recompute of AMI at 20 and mAP
  for `k11=40-no_buddy` and c* (equal; mAP to 1e-9); the selection rule on four constructed sheets. Hard rule 6: no
  missing or contradicting item; partial: §3.6's collapse check on the stage 1 sheet.
  Records: `/tmp/claude-0/-project-CoSiR/9ad8ae03-6854-4c8f-8421-1435e89de566/scratchpad/final_review/`.
  - **Gradient size** (`grad/grad_ratio.json`, saved c* seed 0, first DEC batch, image view parameters): the DEC term's
    gradient norm over c*'s loss's is 0.017 / 0.075 / 0.023 at λ = 1 and 0.17 / 0.75 / 0.23 at λ = 10 (emotion, style,
    genre; cosine with c*'s gradient 0.05 / 0.31 / 0.16). The balance term (×1000) is 6.3 / 45 / 27 times c*'s gradient at
    λ = 1. So λ = 10 is not weak, although q is flat; the balance-on arms are dominated by the balance term.
  - Flags for the run handoff (no code change): S1's stop rule keeps the last state after up to 50 DEC epochs with c*'s
    losses still on, so an S1 arm can train much longer than c* (19, 16, 33 epochs), and `dec-off` does not control for
    that length; the teacher pull's target is as flat as q.
  - Fix wave (Codex `gpt-6.1-sol` high, worktree `/project/CoSiR-r21-fixwave`): F1 the stage 1 sheet shows DEC's own
    cluster shares (last DEC epoch) beside the k-means shares; F2 `collect_rows` refuses an arm without an `ok` run record.
- 08:27: fix wave (0f05d2f5, Codex) re-reviewed (Sonnet `high`): no must-fix; reverting the fix fails 7 new tests;
  should-fix (for S2 the sheet's DEC shares are the last epoch's, not the scored best epoch's) fixed in the sheet note
  (b05aaa00). Merged (c3c6a3bb); `r21-dec-heads` merged into main (7704f331) and pushed.
- 08:28: all r21 tests on main in one process: 186 passed (124 s, one CPU slot).
- 09:16: GPU held check: `r21_run.py held` on saved c* seed 0 (CUDA) equals r10's saved held export exactly (diff 0,
  three aspects).
- 09:22: **GPU smoke** (ticket 03's command, CUDA, seed 0, 2 epochs; `dec-off`, plain S1, start S2, teacher S1): all `ok`;
  every DEC head starts and stops at epoch 2; histories finite; `config.rg.arm` equals the folder; stop reasons agree;
  exports 6,152 × 2,048 float16. DEC start 18 to 49 s per head (k-means), DEC epoch end 1.4 to 3.3 s per head. The
  rerun skip and r10's unchanged tree were checked on CPU (tickets 03, 04; `train` asserts the tree on every run).
- 09:23: run handoff `docs/superpowers/handoffs/2026-11-11-r21-run-handoff.md`; GPU check passes (driver 595.99.02).

## Run chat (2026-11-11, from 09:24)

- 09:25: GPU check passes (driver 595.99.02 in the kernel module and the library; container up since 2026-11-10
  16:08). Run mode: night. The GPU holds r26's `extract_patch` (project CoSiR-r26) until about 09:43; stage 0c waits
  behind it in `res-queue`. Queue logs: `/data/SSD2/cosir_r21/20261111_dec_heads/logs/`.
- Timing plan (D17, `run-planning.md`): stage 0c is one short job and runs now. Stage 0c's power draw decides stage 1:
  under 200 W, stages 1 and 2 run in the day; else they start at 19:00. Today is a Wednesday.
- 09:35: stage 0c training done (r26's job ended early, so 0c ran 09:28 to 09:35, 6.5 min in the queue, 319 s of
  training): `dec-off` seed 0 best epochs 4 / 1 / 18 (emotion, style, genre), equal to the saved c*'s. GPU power over the
  run (77 samples at 5 s, `logs/power_0c.txt`): mean 260 W, median 307 W, max 347 W. Heavy (over 200 W), so per D17 the
  long stages 1 and 2 start at 19:00, not in the day.
- 09:36: **gate 0c PASS** (`r21_score.py sheet`): best epochs 4 / 1 / 18 against 4 / 1 / 18; validation AMI differences
  0.0 / 0.0 / 0.0 (tolerance 0.004). The CUDA rerun is bit-exact: the `dec-off` validation export has the same file hash
  as the saved c* export (`c2fe3d15…` for emotion; the scorer checks the sidecar hash against the file and the sidecar's
  `rg.arm` against the folder). Picks file written with both levels empty (`stage1a`, `stage1b` null); the levels are
  write-once only once filled, so stage 1a can still write its pick.
- Consequence for the report: with a bit-exact rerun, the `dec-off` row is expected to be zero on every seed, so it
  measures no rerun noise. A DEC arm also draws a shifted dropout stream (decision line 07:42), which the `dec-off` row
  does not cover; the seed bootstrap of the intervals is the only noise term for that. Flag in the report.
- 09:40: the teacher-cluster cache (stage 1b's first step, needs no pick) runs now on one CPU slot, so the spec's
  open assumption (the teacher's training features exist) is checked before the night.
- 09:39: teacher-cluster cache written (`teacher_clusters/{emotion,style,genre}.npz`, k 9 / 27 / 10 on 42,969 training
  paintings each; 2.5 min on one CPU slot). The spec's open assumption holds: the teacher's training features exist.
- 09:40: **the user said "run now"**: stages 1 and 2 start in the day, not from 19:00 (user, overrides D17's evening
  start for this run). The 19:00 wait was stopped. Stages 1a, 1b and 2 run in order in one background chain; each starts
  only when the previous script exits 0 (`logs/stage{1a,1b,2}.log`).
- 09:42: D16 code job to Codex (`gpt-6.1-sol` high, config default; `codex-usage --room rederivation`: room, five-hour
  window 41% left, reset in 28 min). Brief `.scratch/r21-dec-heads/d16/task.txt`: separate `r21_d16.py` and tests; no
  r21_* imports or function bodies; own k-means, AMI and mAP per seed; the frozen bootstraps of reports 2026-12-28 and
  2027-01-13 allowed; a smoke on r10's removal arm `k11=40-no_buddy` against r10's and r20's frozen records. The full
  derive and the compare run after stage 2.
- 09:52: D16 code back from Codex (10 min): `r21_d16.py` (373 lines) and `test_r21_d16.py` (21 tests pass). Imports
  only numpy, sklearn, threadpoolctl and frozen older-round modules (`ns_paths`, `ns_data`, `ns_metrics`,
  `rs_ablation`, `r20_ret`); no r21_* import. Smoke `regress` (r10's `k11=40-no_buddy` against saved c*, emotion, full
  draws, 228 s): `D16 REGRESS PASS cells=3 max_abs=3.8e-17` against r10's `ablation/held.json` (AMI at 20) and r20's
  `ablation_ret.json` (mAP); `d16_smoke/regress.json`. Codex's one rule choice: r17 keeps no per-seed test record, so
  c*'s class-count values per seed are checked by r17's recorded mean and a replay of its frozen scorer.
