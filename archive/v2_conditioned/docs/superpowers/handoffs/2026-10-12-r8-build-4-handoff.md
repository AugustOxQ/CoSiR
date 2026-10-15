# Handoff: r8 build 4: review 08, fix 11, apply the FCMI loader speedup, check 12, then build 13 and the final review

## Brief
- Done: stage 0 (SC1) passed, every difference 0.0. Tickets 04, 07 and 09 are merged; 11 of 13 tickets are built.
- State: FCMI projects to about 58.6 min against D21's 60 as built. A loader change gives identical batches and saves about 3 min.
- Next: Opus review of 08, the fixes for 11 (6 blockers, all interfaces), the FCMI loader change, a check of 12, then ticket 13.
- Blocked: nothing. The view-check threshold (M4) still waits for the user in the morning.

> Written 2026-10-12 21:55 by the `r8 build 3` chat (cut at 200k tokens of context). Loop step reached: step 7, the
> build. Direction log entry: [2026-10-12 · Round 7 design L](../direction_log.md).

## 1. Read in this order

1. `.scratch/r8-grouping/notes/build_log.md`, section "r8 build 3": stage 0, the FCMI profile and the loader proof.
2. `.scratch/r8-grouping/notes/review_11.md`: 6 blockers, 3 major, 7 minor. The interface decisions are in §4 below.
3. `.scratch/r8-grouping/notes/handback_08.md` and `handback_12.md`: built, not yet checked. Seven and six rule gaps.
4. `.scratch/r8-grouping/issues/00-index.md`: this chat added "Encoder files" (ticket 04's formats).
5. The previous handoff, [2026-10-12-r8-build-3-handoff.md](2026-10-12-r8-build-3-handoff.md), §3 to §5. Its
   view-check facts and pitfalls still apply.

## 2. Decided by the user, not to be reopened

As in the r8 build handoff (`2026-10-12-r8-build-handoff.md` §2) and both specs' §4 rows marked "user".

## 3. Where things stand

Integration branch `r8-grouping` at ad601e6, pushed; worktree `/project/CoSiR-wt/r8-grouping`. On the merged tip,
164 tests pass and 1 is skipped (groupings, gate, stage 0, common, head, FCMI; run from the round folder with
`--rootdir=.`).

| # | State | Commits | Check |
|---|---|---|---|
| 01, 02, 03, 05, 06, 10 | merged | see the build 3 handoff | as there |
| 04 | merged | 8e352f2, ba8e7af | plumbing: controller read; the controller narrowed the fail record (only a failed weight load drops, B:ED4); GPU smoke 256 images: AL, DI, SG pass ESC1 |
| 07 | merged | 5ff945f (fixes) | Opus: 4 major, 9 minor, all fixed by a Sonnet fixer; both surviving mutations now caught; 44 tests |
| 09 | merged | 280d254 | plumbing: controller read; its smoke fed ticket 10's `load_batches` |
| 08 | built, branch `r8-08` | 76ed6aa | verdict code: needs its Opus review. It merged 07's first version: merge `r8-grouping` before the review |
| 11 | built, branch `r8-11` | 19d6988 | Opus review: fix first (6 blockers, 3 major, 7 minor) |
| 12 | built, branch `r8-12` | 76dacff | plumbing: controller read and alignment still to do (below) |
| 13 | not started | | after 08, 11 and 12 merge |

- **Stage 0:** `results/stage0_regression.json` in the main checkout: pass, 758 s, §10 hashes unchanged at both ends.
- **FCMI time (D21, 60 min, strict):** the profile is in the build log. As built, about 11.7 s per clean epoch, about
  58.6 min for 300 epochs on a fairly quiet machine, and over 60 min under CPU load (the loop is GPU-launch bound on
  one CPU thread). faiss threads gain nothing. The loader change is proven identical over 300 epochs (0.026 s
  against 0.615 s per epoch; `notes/runs/fcmi_loader_check.py`). Apply it in `r8_fcmi.py` `painting_loaders` (a
  `TensorDataset` subclass with `__getitems__`, `collate_fn` identity; keep the same `DataLoader`, generator, shuffle
  and `drop_last`). Add a test that the batches equal the old loader's over a few epochs. Rerun `--time-epochs 6` on
  the GPU, and confirm identical labels on the CPU smoke. It is verdict code (ticket 06): Sonnet fixer, then the
  controller checks.
- **Codex:** 18% of the five-hour window left at 21:31 (reset 01:37), 72% weekly. After 01:37, Codex can take the 11
  fix (it wrote 11: `resume <thread_id>` from `notes/briefs/codex_11.log`), if it fits.

## 4. Open points for the next step

Agent defaults for the interfaces (from review 11; for the report):

- **11 reads 10's pending layout** as written (stats at top level, `trigger` under `pending["trigger"]`, 10's reason
  strings, `nb_records`, `source_scorer`). 10 does not change.
- **`--stage-evidence` and `--tp-map` go.** 11 reads `stageE_<X>.json` `evidence` and `esc1` (index, "Encoder files")
  and `stage1_fcmi.json`'s stored labels and loss log. Check the exact key names in 06's `r8_fcmi.py`. TP goes through
  the normal space path on 07's one-hot `spaces/TP.npz`.
- **A0:** 11 reads R7's `post_{affect,image,caption}.npz` and R7's `results/seed42/reader_A0.npz` (§10 inputs).
- **Taus and readers:** 11 reads 09's merged files (`seed42/taus_<batch>.json`, `seed42/reader_<X>.npz`, named by
  each index entry), not the review's `results/taus.json`. xheads are in R7's format (no `rows` key).
- **Late drops** (M12 drops in 08's `heads/head_<X>.json`; reader drops in 09): ticket 13's queue (or 09's `score`)
  writes one grouping-format file per batch, `groupings/r8_drops_<batch>.json` (`groupings: {}`, `dropped: {X:
  reason}`). It goes to 10 and 11 through `--groupings`. 10 already merges several files; 11 must too (review M2: no
  NPZ needed beside a drops-only JSON).
- **Smoke bars:** 11 scales its integer bars as 10 does (⌈a·N/25⌉) in smoke (review M1).
- **Ticket 12 alignment:** H<X> labels. `r8_scoring_io.load_groupings` has no `H<X>` key in `r8_groupings.npz`; H<X>
  uses X's labels. Ticket 12's `--smoke` root for `r8_place.py` differs from `run_r8.py --smoke`
  (`results/smoke/gate/`); give `r8_place.py` a `--root` for smoke, or make 12 pass the same root.
- **Ticket 13** handles smoke stage records (stage 0's smoke has `pass: false, wiring_pass: true`, while 10's smoke
  reader wants `pass` true).
- For the morning, key, unchanged: the M4 view check fails by 0.00007 on 1 of 256 paintings. V100 to V33 drop under
  the rule (07 now writes that drop with the cosines). Amending the threshold is the user's call (C7).

## 5. Code and pitfalls

- Run tests in a worktree from the round folder with `--rootdir=.` and `--basetemp` in the session scratchpad.
- Don't `py_compile` in a worktree: it writes `__pycache__` despite `PYTHONDONTWRITEBYTECODE` (use `ast.parse`).
- Don't run CPU benchmarks with 32 threads while subagents run tests: the load reached 55 and faiss's OpenMP threads
  stalled.
- After 08's review, check 09's `split_dropped` against 08's real `head_<X>.json` (`dropped`, `reason`) and the late-drop
  file above.
- Ticket 13's queue: run FCMI stage 1 alone on the CPU (no reader, head or grouping jobs beside it).
- Timings for the run chat: encoders about 18 to 46 min each at the smoke speeds (46, 32, 18 img/s under load); stage 0
  took 12.6 min. Ticket 08: about 2 to 5 min per ID (estimate). Ticket 09's score: 20 to 40 min (estimate).

## 6. State at handoff

- Running: nothing. All subagents have finished.
- Uncommitted: nothing of this chat in git. On main, `palette.md`, `reports_sum.md` and the stage-report files belong
  to another session; do not commit them.
- Worktrees: `/project/CoSiR-wt/r8-{grouping,01,02,03,04,05,06,07,08,09,10,11,12}`.
- On disk over 1 GB: the FCMI smoke and timing checkpoints in `/data/SSD2/cosir_r8/fcmi/smoke/` (16 GB in five
  runs) and `RF/results/smoke/gpu_wiring_06/` (1.6 GB). All are on `.scratch/pending_deletions.md`.
