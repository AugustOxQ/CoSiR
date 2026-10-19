# Ticket 01 — Shared scorer

Implemented `ms_score.py`; earlier-round code and artifacts, including the user-read report, are read only.

## Definitions settled

- `measures(x, labels, cache_dir=None)` returns label → metric → `{mean, sd}`; the optional argument lets CLI callers persist scores without changing the binding two-argument API.
- Inputs become float32 before validation, hashing and clustering; this matches `pc_score.ami_scores` exactly.
- One `pav_score.partitions(x, 20)` call supplies three MiniBatchKMeans partitions for every label and all four measures; this makes comparisons use identical clusters.
- Each label masks `y >= 0` after clustering all 5,000 paintings; this preserves the frozen protocol and genre's 4,053 labelled paintings.
- Single-array SD is population SD over three k-means seeds; learned-row mean and population SD use the three model-seed means, matching the original table.
- Cache identity binds SHA256 of float32 source bytes (with shape/dtype), label bytes, protocol and sklearn version; changed inputs cannot reuse stale scores, and CLI reruns require no k-means.
- PercepT readers bind NPZ checksums to recorded scores, stage-1 checksums/sample identity and `subset_pid` evaluation order; this keeps the saved probabilities tied to the frozen evaluation sample.
- Network readers require sample identity, aspect/graph/seed, default inputs, NPZ checksum and exactly two finite float16 `(5000, 2048)` unit-norm arrays; malformed or partial existing output sets fail loudly.
- `bar_vectors(s, stage0_dir=None)` returns bar → aspect → array or three-array list; this gives the runner unscored image-side vectors for PercepT P3/P2/P3, encoder Qwen/CLIP/Qwen, instructed targets and stage-1 heads.
- The linear-probe bar uses `None` for vectors and `probe_ami()` for recorded default-Qwen AMI; its probability arrays were not saved, so V/ARI/FMI are unavailable.
- CLIP encoder vectors follow `pc_score.row1` normalization; Qwen vectors use already-normalized `default_images`, avoiding an extra normalization pass.
- Stage-1 emotion Fres1 and style Fres05 use flat `<aspect>__<graph row>__s<seed>.npz` filenames under `--stage0-dir`; absent sets stay pending because stage 0 has not run. Genre R0o always comes from main checkout's `outputs_fu`.
- An incomplete network row keeps available genre values and marks emotion/style pending; no recorded AMI is substituted for missing four-measure scores.
- Back-fill parses only the report's first two-bar table and asserts available AMI cells at three decimals; the report is never edited.
- CLI defaults to this round's `outputs`; all smoke artifacts are isolated under the ticket's scratch folder.

## Commands and checks

Read the index, ticket and design from section 1 in that order, then inspected the requested earlier code and report lines 18–35 using `cat`, `sed` and `rg`. Checked branch/status with `git branch --show-current` and `git status --short`.

All Python commands used this CPU-only prefix:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' timeout 235 /root/miniconda3/envs/CoSiR/bin/python
```

Final smoke invocation (prefix above):

```bash
src/test/20261019_model_sheet/ms_score.py --check-percept --backfill \
  --stage0-dir /tmp/claude-0/-project-CoSiR/56b9b751-6946-4c76-b52d-3761e520be8f/scratchpad/ms_smoke/01/outputs/stage0_not_run \
  --outdir /tmp/claude-0/-project-CoSiR/56b9b751-6946-4c76-b52d-3761e520be8f/scratchpad/ms_smoke/01/outputs
```

Earlier attempts ran the same scorer without `--stage0-dir`; their logs are `smoke.log` and `smoke_fixed.log`. They exposed evaluation selection before shape validation and ambiguous report matching; both were corrected. The successful CLI log is `smoke_final.log`.

Ran the scratch-only `replay.py` with the same prefix, logging to `replay.log`: it replaces both scorer partition functions with raising functions, replays the regression/back-fill, then loads and validates every bar and the recorded probe. This checks actual cache reuse without allowing any clustering. No unit tests were added.

Ran `git diff --check` and reviewed the new scorer/build notes before the ticket commit.

## Smoke numbers

- PercepT regression: **PASS**, 45/45 comparisons (P1–P5 × three seeds × three labels) equal at six decimals.
- Direct `pc_score.ami_scores` comparison: all three label means and SDs exactly equal on P1 seed 0.
- Back-fill: **22/24 complete** aspect cells, all available AMI cells equal to the report at three decimals; two network cells pending.
- Label counts: emotion 5,000; style 5,000; genre 4,053.
- Cached replay: **zero k-means calls**; all five bars × three aspects validated, with the documented pending/AMI-only entries.
- Genre network: AMI **0.458275 ± 0.005048**, V **0.462929 ± 0.005005**, ARI **0.265925 ± 0.006170**, FMI **0.363774 ± 0.006195** (three model seeds).
- Best PercepT bar means: emotion P3 **0.077174**, style P2 **0.228560**, genre P3 **0.435880**.
- Default-Qwen probe AMI: emotion **0.121467**, style **0.527451**, genre **0.656639**.
- Default image ID join: cosine mean **0.99992734**, minimum **0.99856031**.

Generated artifacts: scratch `outputs/stage0_scorer.json`, `outputs/pc_ami_reference.json`, `outputs/backfill.json`, `outputs/backfill.md` and `outputs/cache/*.json`. The controller can rerun with the default output folder for integration artifacts.

## Open points and removal candidates

Stage 0 must generate emotion Fres1 and style Fres05 seeds 0–2; rerun `--backfill --stage0-dir DIR` afterward. Their numbers have not been verified here. The probe has AMI only by design.

Nothing deleted. Once the controller has retained the successful smoke evidence, removal candidates are this ticket's scratch logs, `replay.py`, scratch reports and score caches. No earlier-round or repository files are proposed for removal.
