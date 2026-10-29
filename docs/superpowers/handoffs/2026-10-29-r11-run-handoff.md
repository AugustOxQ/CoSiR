# Handoff: r11 run: run `rt_queue.sh full` from main (CPU only), then verify and write the reports

## Brief
- Done: the r11 retrieval build: 9 tickets; the final review (no must-fix); fixes F1 to F3 (re-review pass) and F4 (a must-fix found after it); merged into main.
- State: full-scale validation check: stage 4 19 min; `rt_vh` agrees on all 790 values (largest difference 1.8e-7, tolerance 1e-6). Queue smoke at 2,100 rows after the fixes: all steps exit 0, 0 disagreements.
- Next: run `rt_queue.sh full` from `/project/CoSiR` on main, about 1.5 to 2.5 hours, then step 9: the auto report and the user-read report.
- Blocked: nothing.

> Written 2026-10-29 22:01 by "r11 build 4". Loop step reached: step 7 done (build merged); step 8 (run) next.
> Direction log entry: [2026-10-29](../direction_log.md).

## 1. Read in this order

1. The spec, `docs/superpowers/specs/2026-10-29-retrieval-design.md`: §3.5 (the stages and gates), §6 (the decision
   rule), §3.6 (diagnostics), §9 (deliverables: the auto report `docs/reports/auto/percept/2026-12-29_retrieval.md`
   with its results brief and index row, and a user-read report; time-box 2026-10-31 23:00). Amendment A1 (§4, §5).
2. The round log, `src/test/20261029_retrieval/20261029_retrieval_log.md`: "C13 stage 4 numeric alignment" (the one
   numeric convention), "Build 3: the queue smoke", "Final review fixes (r11 build 4)".
3. `rt_queue.sh` in the round folder: the step order and the full mode's commit step.
4. For step 9: `~/.claude/rules/report-kinds.md`, `report-writing.md`, `user-read-reports.md`, and the results brief
   template `~/.claude/templates/results_brief.md`.

Skills: none (`diagnosing-bugs` only if a step fails).

## 2. Decided by the user, not to be reopened

- The whole protocol, the bar, the baselines and the read-out optimization: spec §4, D1 to D8 and D15; approved from
  its brief on 2026-10-29 18:30 (r11 decide 2).
- No container restart before the build (user, r11 decide 2). Night mode at this cut: the GPU check only (§6 below).

## 3. Where things stand

Code: main at `938565e` (merge of `r11-retrieval`, pushed). Round folder `src/test/20261029_retrieval/`, prefix
`rt_`. Inputs and outputs: `/data/SSD2/cosir_r11/20261029_retrieval/outputs` (holds only `percept_val/` and `crl/`).

**The ticket list** (the record, `matt-chain.md`; for information):

| # | Title | Blocked by | Spec IDs | Verdict code | Built by | Check |
|---|---|---|---|---|---|---|
| 01 | Round folder, labels and arm loaders, `rt_percept_val.py` | none | D5, D9, D11, §3.3, C6 | yes (joins) | Codex | Opus `high`: pass (4 of 4 planted wrong joins caught) |
| 02 | Retrieval metrics: AP, P@10, nDCG, query expansion | none | D1, D3, D9, §3.4 | yes | Sonnet | Opus `high`: pass after fixes |
| 03 | Statistics, outcomes and the SC table | none | SC1-SC15, D4, D7, D10, D15, §3.4, §6 | yes | Sonnet | Opus `high`: pass (15 of 15 mutations caught) |
| 04 | Read-out menu, cross-fit and the selection rule | 02 | SC10-SC15, D8, D13, §6 | yes | Sonnet | Opus `high`: pass, 3 should-fix applied |
| 05 | CRL value lists, text embeddings and the CRL arm | 01 | D5, D6, D12 | no | Sonnet `medium` | controller diff read |
| 06 | Stage 0: seed ranges and the regression check | 01, 02, 03 | D10, §3.5 0a and 0b | partly | Codex | controller diff read; smoke |
| 07 | Stage runner: stages 1 to 5, diagnostics, verdict files | 01-06 | SC1-SC15, D2, D14, D15, §3.5, §3.6 | no | Sonnet `high` | controller diff read; smoke |
| 08 | C13 re-derivation (`rt_vh.py`) | 03, 04 | §3.4 C13, SC1-SC15 | yes (a check) | Sonnet | Sonnet `high`: pass after one must-fix |
| 09 | Queue script and smoke run | 06, 07 | §3.5 order, §9 | no | Codex | controller diff read; two fixes; smoke |
| fix-c13 | One numeric convention in `rt_data`, `rt_readout`, `rt_vh` | 08 | C13 | yes | Codex | controller diff read; smoke |
| fix2 | The final review's fixes F1 to F4 (below) | all | §3.4, §3.5 | no (guards) | Sonnet `high`; F4 controller | Sonnet `high` re-review: pass; 5 of 5 mutations caught |

**The final whole-branch review** (Opus `xhigh`, 2026-10-29 21:40): no must-fix. It re-derived on the 2,100-row smoke,
with its own code: c*'s r_a, SC1's d, one SC7 cell, one CRL vector and two stage 4 scores; all match within 1.5e-13.
Joins (C6), A1 and `rt_vh`'s independence hold. The fixes (round log, "Final review fixes"):
- F1: the committed `stage0a_ranges.json` held wall-clock times, so a rerun of the full queue always failed at
  `commit_ranges`. It now holds only `map`, `ndcg`, `inputs`, `smoke`. A real-data rerun of 0a gave identical values.
- F2: `test_rt_guards.py` pins the held-data guards (243 tests in the round, all pass).
- F3: the queue runs `rt_vh stage4` right after stage 4, so C13 confirms the picks before stage 5 reads them.
- F4 (the controller, after the re-review): the repo's `*.json` ignore rule would have made the queue's `git add` of
  `stage0a_ranges.json` fail at the first full run. `.gitignore` now un-ignores that one path; a test pins it.

**The queue smoke after the fixes** (21:38 to 22:01, 2,100 rows, from the fix worktree): every step exit 0, in the
new order; `rt_vh` agrees on all 995 values (0a 24, stage 1 68, stage 2 47, stage 3 19, stage 4 790, stage 5 47).

**The full-scale validation check** (no held data; outputs in this chat's scratchpad, `fullcheck/`):
- Stage 0a: 35 s. r_a for mAP: emotion 0.0014116, style 0.0022293, genre 0.0038238; emotion nDCG r 0.00030588.
  `rt_vh stage0a` agrees.
- Stage 4: 1,141 s (19 min, at a CPU load of about 33). Picks: `center+k10` for c*, plain, instructed, CLIP, SigLIP 2
  and PercepT; `none+k10` for CRL. `rt_vh stage4` (643 s): 790 values, 0 disagreements, picks equal; largest
  difference 1.8e-7 against the 1e-6 tolerance (about 5× margin; at 2,100 rows it was 5e-8).
- The run must reproduce these values (same data, same code): a cheap cross-check, not a gate.

## 4. Open points for the next step

None for the user.

- Notes from the re-review (test gaps, not defects): the call sites that pick the validation-only gate
  (`Loader(args.smoke, ('val',))` in stage 4, `Inputs(..., validation_only=True)` in 0a) are not pinned by a test. C13
  covers both at run time: `rt_vh` re-derives 0a and stage 4 on validation, so a held read would disagree.
- §3.4 says C13 must agree "before the verdict is written": the report reads a stage only when its `rt_vh` check passed.
  The queue stops at the first failure, so "All steps done." means every check passed.

## 5. Code and pitfalls

- Start: `cd /project/CoSiR/src/test/20261029_retrieval && bash rt_queue.sh full`, in a background Bash
  (`run_in_background: true`). Check `bash rt_queue.sh full --dry-run` first. Status lines: `<outputs>/queue_status.txt`;
  logs: `<outputs>/logs/`.
- The full mode refuses unless the checkout is `/project/CoSiR` on main with no tracked changes in the round folder.
  After 0a it commits and pushes only `stage0a_ranges.json`. Commit nothing else in the round folder before that step.
- The GPU steps are skipped (their files exist), so the queue takes no GPU lock. CPU: 8 threads and 8 workers.
- A rerun keeps the committed r_a: 0a must reproduce it byte for byte, or the queue stops at `commit_ranges` (F1 made
  this possible; checked on real data).
- Times (full scale where measured): 0a 35 s; 0b 3 to 5 min (estimate); stage 4 19 min; `rt_vh stage0a` 36 s; `rt_vh stage4` 11 min.
  Stages 1 to 3 and 5 at full scale, with 10,000 bootstrap draws, are not measured. Spec budget: about 1 hour of CPU.
  Estimate for the whole queue: 1.5 to 2.5 hours. Started before about 23:30, it ends well before 11:00.
- `rt_run.py --arms` is for smoke runs only (asserted). In this shell a bare `cluster` is Graphviz; this loop needs no DAS6.
- `.claude/` is gitignored: the day's change log is `/project/CoSiR/.claude/20261029_log.md` (local file).

## 6. State at handoff

- GPU check (night mode, check only, 21:50): driver 595.99.02 in all three places; the GPU idle, no lock. The run
  needs no GPU (its GPU steps are skipped).
- Running: nothing of this chat.
- Uncommitted: nothing in the project. `claude-config`: Codex ledger rows (sync at the wrap-up).
- On disk over 1 GB: the Affection store, 1.8 GB at `/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b/` (made by
  the data side job; it needs a disk-ledger row at the wrap-up).
- Listed on `.scratch/pending_deletions.md` (nothing deleted): the 13 r11 worktrees, the r11 ticket branches, two
  scratchpads, one stray `.pyc`.
