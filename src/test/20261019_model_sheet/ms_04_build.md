# Ticket 04 — Stage 0, number sheet and GPU queues

Implemented on `r10-ms-04-stage0`. No trainer, rows, shared scorer or earlier-round files changed.
No agents, CCG records, unit tests, GPU execution, push or merge.

## Definitions settled (agent defaults)

- Stage 0 per-aspect workers only train; run `ms_stage0.py --check` after all three exit, because a worker cannot gate arrays another worker is still writing. With no flags the runner trains all aspects and checks.
- Stage 0 calls the trainer through its CLI with `--row A1 --set graphs.<aspect>=<stage1 pick> --name stage0 --aspects <aspect> --device cpu`; genre uses seed 0, emotion/style use 0/1/2. This writes exactly the flat `outputs/stage0` files consumed by the scorer.
- Gate (b) uses `ms_score.measures`, taking its AMI mean, because both it and `ags_head.ami_scores` convert to float32, use `pav_score.partitions(x, 20)` (three MiniBatchKMeans seeds, n_init=3, batch_size=4096), mask labels after clustering, and use sklearn adjusted mutual information with population SD. The direct smoke comparison below additionally checks exact API equality. `ags_head.aggregate` averages those per-head means over model seeds; gate (b) compares individual heads instead.
- `stage0.json` records missing paths, uint16-array identity/equality for gate (a), every actual/expected AMI for gate (b), actual/reference best epochs, and the complete PercepT gate result. Missing arrays fail the check while available gates still execute.
- All earlier-round artifact constants come from `ms_score`'s absolute main-checkout paths; earlier code imports come from the worktree, preserving the ignored-data boundary.
- `--outputs` controls every sheet artifact and cache; stage0 and cache are the only excluded directories. NPZ is the completion marker, so metadata-only and pending writes are ignored.
- Head caches bind output SHA, aspect target, label bytes, scorer protocol and sklearn version; metadata, sample fingerprint, filename, checksum, shapes, float16 dtype, finite values and norms are revalidated on replay. Config/epoch are read fresh, allowing identical arrays to share scores across row names.
- Diagnostic target is exactly `prompted(s, key)[0]` in frozen subset order, asserted `(5000, 2048)` and unit norm; no second indexing or normalization changes its values.
- Learned rows average per-model-seed metric means; population SD appears only for three seeds, because one k-means SD is not model-seed uncertainty. T1's three-seed own-label image AMI SD is the tie band.
- Bars precede ranked rows in each aspect table; probe has AMI only, unavailable stage0 network bars say pending, and D1 caption AMI says untrained. All four measures on both sides and all labels remain in results/cache.
- Seed manifest is `{rows, seeds: [1,2], top_by_aspect}`; rows are the union of A1, T1 and top two seed-0 default-input rows per aspect, because one trunk run produces all aspects. D17 is excluded to retain its seed-0-only rule on later replays. Stable score ties use row-name order.
- D17 needs a strict improvement beyond T1 SD in at least two aspects; a missing three-seed T1 SD cannot trigger it. If both forms qualify, choose the larger sum of positive margins beyond T1 SD, preferring b on an exact tie. This supplies one reproducible rerun when the spec permits either winning form.
- Best D is the complete D2–D5 row with highest mean over the three aspect own-label image AMIs; each aspect first averages its available model seeds, as specified. Exact D-row ties use deterministic row-name order.
- Derived D17 and I1 definitions use exactly `{name, base, set: ["k=v", ...]}`. **D17 and I1 train seed 0 only**; three seeds are reserved for A1, T1 and seed picks (§3.4).
- Stage 3 CLI is `ms_queue_stage3.sh <base row> [k=v ...]`; it records `outputs/i1.json`, enforces instructed inputs, and accepts form/share overrides so the selected derived D17 form can be reproduced from its canonical base.
- Queues split training by row and continue after failures; every attempted step has an exit-code line and each run has one final `MS_STAGE<n>_DONE ...` line. Shared shell functions keep lock/status behavior identical.
- GPU subprocesses run under `flock -o -w 7200 /tmp/gpu0.lock`; CPU scoring starts after that process exits, with CUDA hidden and eight threads. Stage 2 validates the seed manifest, trains seeds 1/2, scores, selects/runs D17, then scores again.

## Commands run

Read the index, ticket, spec from §1, log defaults, prior build notes, scorer/trainer CLI, stage 1 scorer/queues and partition implementation in the requested order. Used read-only `cat`, `sed`, `rg`, `find`, and git status/branch inspection. All writes stayed in the round folder or ticket 04 scratch folder.

Scratch root (`S` below):
`/tmp/claude-0/-project-CoSiR/56b9b751-6946-4c76-b52d-3761e520be8f/scratchpad/ms_smoke/04`.
Copied ticket 02's `outputs/T1` and `full/A1` to `S/outputs/` with `cp -a`.

Python execution prefix:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' timeout 235 /root/miniconda3/envs/CoSiR/bin/python
```

With that prefix, from the worktree:

```text
-u src/test/20261019_model_sheet/ms_sheet.py --outputs S/outputs --score --sheet --pick-seeds --d17
    sheet.log (initial), replay.log (final head cache identity), sheet_final.log (final code)
-u src/test/20261019_model_sheet/ms_stage0.py --check --outputs S/outputs
    stage0_check.log, stage0_check_final.log (expected exit 1)
-u - (inline smoke verification; verify.log):
    compare ags_head.ami_scores with both cached A1 genre sides;
    replace ms_score.partitions with a raising function;
    replay all four sheet actions against the real caches.
```

`bash -n` on each of the four shell files, `git diff --check`, staged diff review, and the ticket commit. Queue GPU steps were syntax checked, not executed. No full stage0 training was requested in this CPU smoke.

## Smoke numbers

Four complete heads, eight float16 `(5000, 2048)` arrays; all checks passed. First uncached scoring timings include both sides, all labels and all four metrics:

| Row/head | Image own-label AMI | Caption own-label AMI | Image/instructed cosine | Best epoch | Both-side scoring seconds |
|---|---:|---:|---:|---:|---:|
| A1 genre R0o | 0.465055 | 0.395375 | -0.005363 | 1 | 6.50 |
| T1 emotion Gres | 0.094452 | 0.146906 | -0.005059 | 2 | 6.68 |
| T1 style Fres05 | 0.274877 | 0.153526 | -0.015105 | 2 | 7.71 |
| T1 genre R0o | 0.444795 | 0.364556 | -0.015719 | 2 | 6.21 |

- Both-side head scoring: **6.21–7.71 s**, mean **6.78 s** on this CPU; descriptive timing, not a throughput guarantee.
- Cached per-head metric retrieval during head-cache revision: **0.20 s** each. Final replay: **zero k-means calls**, enforced by a raising partition function.
- Direct stage 1 API comparison: **6/6 label-side AMI means and SDs exactly equal** on A1 genre.
- Three aspect tables, five bars each, ranked trained rows, diagnostic cosine/epochs, JSON scores, seed manifest and D17 manifest written. Emotion/style stage 1 bars are pending, genre is populated.
- Seed picks: **A1, T1**; top rows: emotion T1, style T1, genre A1/T1. D17: **none**; T1 SD is pending (seed 0 only). No masks exist in these inputs, so mask overlap is empty.
- Section 6 line: smoke T1 emotion **0.094452 ≥ 0.087**. These truncated-training smoke numbers are not results of the full research run.
- Stage0 empty-directory check: **seven full missing NPZ paths**, `STAGE0 FAIL missing,a,b`, **exit 1**, JSON written; PercepT gate **PASS 45/45**, direct scorer API equal.
- Shell syntax: **4/4 pass** (three queues plus shared helper).

## Open points and removal candidates

Full stage0 regeneration/gates (a)/(b), GPU training, three-model-seed tie bands, positive D17 execution, masks from T5 and I1 training remain for the controller after merge with ticket 03. The required ticket smoke passes; the empty stage0 failure is intentional. To refresh the separate K40 report back-fill after stage0, use the existing `ms_score.py --backfill --stage0-dir <outputs>/stage0 --outdir <outputs>` CLI.

Deleted nothing. No repository removal candidates. After controller validation, the entire ticket 04 scratch directory (copied heads, logs, generated reports and caches, including superseded `head_<sha>.json` cache files) may be removed.
