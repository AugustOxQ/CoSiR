# Handoff: r12 run: run `ac_queue.sh full` from main (GPU steps first, then CPU stages with C13), then verify and write the reports

## Brief
- Done: the r12 build: 13 tickets, the final review (pass after fixes), the fix wave, the vh stage 1 fix, a scoped re-review (pass); merged into main.
- State: fresh smoke at 2,100 keys: every step exit 0; vh agrees on all 1,013 items; the reviewer's own re-derivation equals it (SC8 d within 1e-8).
- Next: `bash ac_queue.sh full` from `/project/CoSiR` on main, about 5 to 8 hours (GPU about 1.5 h first), then step 9: the auto report and the user-read report.
- Blocked: nothing.

> Written 2026-10-31 01:00 by "r12 build 5". Loop step reached: step 7 done (build merged); step 8 (run) next.
> Direction log entry: [2026-10-30](../direction_log.md).

## 1. Read in this order

1. The spec, [2026-10-30-affection-confirmation-design.md](../specs/2026-10-30-affection-confirmation-design.md),
   from §1 on: §3.4 (measurement, C13), §3.5 (stages and gates), §3.6 (diagnostics), §6 (decision rule), §9
   (deliverables: the auto report `docs/reports/auto/percept/2026-12-31_affection_confirmation.md` with its results
   brief and index row, and a user-read report; time-box 2026-11-02 23:00).
2. The round log, `src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md`: "Build
   decisions", "Readings settled in the build", "Build chat 4" and "Build chat 5" (smoke step times).
3. `ac_queue.sh` in the round folder: the step order, the seals, the two commit steps.
4. For step 9: `~/.claude/rules/report-kinds.md`, `report-writing.md`, `user-read-reports.md`, and the results brief
   template `~/.claude/templates/results_brief.md`.

Skills: none (`diagnosing-bugs` only if a step fails).

## 2. Decided by the user, not to be reopened

- The whole protocol, the arms, the bars and the decision rule: spec §4 (D1 to D14) and the direction log,
  2026-10-30; approved from its brief at 21:17 (r12 decide).
- Night mode at this cut: the GPU check only, no restart question (§6).

## 3. Where things stand

Code: main at `fd6a4b5` (merge of `r12-affection`, pushed), then the round log (`47b170f`) and this handoff. Round folder `src/test/20261030_affection_confirmation/`,
prefix `ac_`. Outputs: `/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs` (does not exist yet; the queue
makes it). The Affection store `/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b/` has no FI yet and no
instructed views; the queue builds both.

**The ticket list** (the record, `matt-chain.md`; for information):

| # | Title | Spec IDs | Verdict code | Built by | Check |
|---|---|---|---|---|---|
| 01 | FI in the Affection store: adapter, FI-only extraction, merge and hash gate | D2, D11, §3.5 0a(i) | no | Sonnet `medium` | controller diff read; gate 0a(i) in the run |
| 02 | Round paths, Affection keys, labels and arm loaders | D2, D9, D12, §3.3, C6 | yes | Codex | Opus `high`: pass |
| 03 | Key-list image extractor: instructed Qwen wordings and CLIP L/14 | D3, D11, 0a(ii), 0a(vi) | no | Sonnet `medium` | controller diff read; smoke |
| 04 | Frozen exports: c* heads, PercepT core, Impressions c* | D1, D8, D13, 0a(iv), 0a(v) | no | Sonnet `medium` | controller diff read; smoke |
| 05 | CRL texts and vectors | D5, 0a(vii) | no | Sonnet `medium` | controller diff read; smoke |
| 06 | Clustering: r10's partitions, AMI, paired bootstrap | SC1, SC3, D9 | yes | Sonnet `high` | Opus `high`: pass |
| 07 | Stage 1 selection: read-outs, teacher wording, CRL version | SC2, SC4-SC8, D3, D6, D10, D12 | yes | Sonnet `high` | Opus `high`: pass |
| 08 | Seed ranges, SC table, outcomes, verdict | SC1-SC8, D4, D7, D10 | yes | Sonnet `high` | Opus `high`: pass; four should-fix test gaps fixed |
| 09 | Stage runner: stages 0a to 3, guards, verdict files | SC1-SC8, D4, D7, D10, D14 | no | Sonnet `medium` | controller diff read; smoke |
| 10 | Diagnostics and the Impressions figure | D1, D8, D13, §3.6 | no | Sonnet `medium` | controller diff read; smoke; fix wave (diag) |
| 11 | C13 re-derivation (`ac_vh.py`) | §3.4 C13, SC1-SC8 | yes (a check) | Codex | Claude review: pass after fixes; vh stage 1 fix (build 5) |
| 12 | Queue script and end-to-end smoke | §3.5 order, §9, D9 | no | Sonnet `medium` | controller diff read; two smokes (build 4, 5) |
| 13 | Smoke mode for the loaders | §3.3, C6 | yes | Sonnet `high` | Opus `high`: pass; review nits fixed (`931fae7`) |
| fix wave | diag ten arms; c* clustering on stored float16 rows | §3.4, §3.6 | yes | Codex | controller diff read; re-review |
| vh fix | vh stage 1 scores ArtELingo's fixed-pick arms as the runner | §6, C13 | yes (a check) | Codex | controller diff read; re-review |

**The final whole-branch review** (Opus `xhigh`, at `910e527`): pass after fixes. It traced the SC table, bootstraps,
outcomes, verdict logic, stage 1 rule, seed ranges, held-data order, stage 0a and joins as matching the spec;
28 mutations, all caught. Its re-derivation script (numpy and scikit-learn only) is
`.scratch/r12-affection/build4/review/rederive.py`.

**The scoped re-review** (Sonnet `high`, every change after `910e527`): pass, no must-fix or should-fix. Each fix's new test fails when
that fix is reverted; vh imports no `ac_*` module; the restricted cross-fit is bit-equal to the full menu. Note: the
smoke gives the same r_clu with stored and re-normalized c* rows, so only the unit tests show that change. Report:
`.scratch/r12-affection/build5/rereview_report.md`.

**The fresh smoke** (build chat 5, 2,100 keys per split, draws 200 and 500; round log "Build chat 5"): every step exit
0. vh agrees on every item: stage 0b 41, stage 1 598, stage 2 186, stage 3 188. The reviewer's script on it: stage
0b and 1 values diff 0 (read-out mAPs within 5e-8, same picks); SC1 and SC2 d diff 0, SC8 d diff 9.7e-9 (bar 1e-6);
same outcomes and verdict. Smoke numbers are not results.

## 4. Open points for the next step

None for the user.

- **A verdict counts only once its `vh/<stage>.json` passes.** The queue stops at a failed vh step; "All steps done"
  means every C13 check passed. C13 (vh) runs after the runner writes a verdict file; the report reads a stage only
  after its vh passed (the final review's reading).
- Spec §3.5: if a stage 0a gate fails, stop, trace and note it; stage 0b fails, stop and note it; later stages report.

## 5. Code and pitfalls

- **Start:** check `nvidia-smi` and `pgrep -af gpu0.lock` first; then `bash ac_queue.sh full --dry-run`; then, in a
  background Bash (`run_in_background: true`): `cd /project/CoSiR/src/test/20261030_affection_confirmation && bash
  ac_queue.sh full`. Status lines: `<outputs>/queue_status.txt`; step seconds: `<outputs>/step_seconds.txt`; logs:
  `<outputs>/logs/`.
- **Full mode** refuses unless the checkout is `/project/CoSiR` on main with no tracked changes in the round folder.
  It commits and pushes only `stage0b_ranges.json` (after vh 0b) and `stage1_picks.json` (after vh 1), with `git add
  -f` (the repo ignores `*.json`). Commit nothing else in the round folder while the queue runs.
- **GPU:** the GPU steps wait for the lock (`flock -o -w 3600`). Spec §9 budget: about 1.4 h, light (the smoke drew
  well under 200 W at about 8.5 images/s). Saturday: no working-hours limit.
- **The store swap** (`merge_swap`): `ac_store.py merge` writes the 85,007-image store into a new folder, checks gate
  0a(i), then renames the old store to `qwen3vl_emb_2b_pre_fi/` and the merged one to `qwen3vl_emb_2b/`. The old
  folder goes on `.scratch/pending_deletions.md` (about 1.8 GB; a disk-ledger row at the wrap-up).
- **Time estimate** (not measured at full scale; from the smoke at 2,100 keys, with val 4,251 and test 8,501 rows,
  full ranking cost growing with the square of the rows, and 10,000 / 2,000 draws instead of 500 / 200): GPU steps
  about 1.5 h; stage 0a with its real-data replays about 25 min (the slow replay tests took 24 min); stage 0b and vh
  0b a few minutes; stage 1 about 20 to 25 min; vh 1 about 30 min; stage 2 about 30 to 45 min (clustering
  bootstraps); vh 2 about 1 h (it re-derives stage 1 inside); stage 3 about 15 min; vh 3 about 45 min; diag about
  20 to 30 min. In all about 5 to 8 hours. Started at about 01:15, it ends between about 06:15 and 09:15, before 11:00.
  Re-estimate after stage 1 and vh 1 (their real times scale the rest), and tell the main chat the expected end.
- **A rerun** skips steps whose seals exist; stages 0b and 1 are sealed by the outputs file and the record copy. A
  changed input does not unseal a step. A committed record must be reproduced byte for byte, or the queue stops at
  its commit step.
- **Workers cannot run `bash <script>`** here (the sandbox guard refuses a script that contains `git`): the main
  session runs the queue and its dry-runs. Workers run no command over about 4 minutes.
- `.claude/` is gitignored; the day logs `.claude/20261030_log.md` and `.claude/20261031_log.md` are force-added.

## 6. State at handoff

- GPU check (night mode, check only, 00:52): driver 595.99.02 in all three places; the GPU idle, no lock.
- Running: nothing of this chat.
- Uncommitted: nothing in the project. `claude-config`: two Codex ledger rows (sync at the wrap-up).
- On disk over 1 GB: nothing new. Listed on `.scratch/pending_deletions.md` (nothing deleted): the build's worktrees
  and branches, the two smoke folders (about 280 MB each), the build scratchpads.
