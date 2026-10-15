# Lightweight CLIP fine-tuning comparator: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** train CLIP ViT-B/32 on ArtELingo scorer-train image–caption pairs in three lightweight variants (linear
probe, last block, LoRA; 3 learning rates each, 9 DAS6 runs), select each variant on val retrieval, and score the
selected models' cosine on the aspect episodes beside AFF and its comparators.

**Architecture:** new modules in `src/test/20261124_clip_lightweight_ft/` (prefix `ft_`): a data module (rows, captions,
the 224-pixel image cache), a trainer (one script, three variants), an evaluation script; a launcher
`scripts/run_clipft.sh` and a data-sync script `scripts/das6_sync_clipft.py`. Training runs on DAS6 through the cluster
CLI; the image cache is built locally once and synced; evaluation runs locally on CPU.

**Tech stack:** torch 2.11 (+cu130 locally), transformers 5.6.2, peft 0.19.1, numpy, the CoSiR env locally and
`/var/scratch/wding/conda-envs/CoSiR` on DAS6.

**Spec:** `docs/superpowers/specs/2026-10-07-clip-lightweight-ft-design.md` (commit f472cc3). It fixes the data use,
variants, learning-rate grids and the val-only selection; this plan does not change them.

## Global constraints

- Held rows are never loaded: no image, caption or feature of a held row is read, cached or extracted.
- No evaluation label (emotion, style, genre) is read by the data cache, the trainer or selection.
- Selection uses val retrieval only (spec §4); the aspect episodes are read only by the evaluation script.
- Rows follow the feature-row order (`data.sample_ids`); captions are `annotations[sample_ids[i]]["caption"]`.
- Features are the projection outputs (`visual_projection(vision_model(pixel_values=...).pooler_output)`,
  `text_projection(text_model(...).pooler_output)`), as the existing cache; never `get_image_features` (changed in
  transformers 5.x). Do not import `src.model` (broken llvmlite via cuml); `src.data`, `src.eval`, `src.dataset` are safe.
- Local CPU work: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`, every Python
  call. The local GPU is shared: check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and use
  `flock -n -o -E 75 /tmp/gpu0.lock` if a step uses it.
- Cluster: only `cluster launch --node <node> -- bash scripts/run_clipft.sh ...`; never raw ssh; the data-sync script
  runs with `/usr/bin/python3` (conda's OpenSSL breaks ssh). One GPU per job.
- Smoke runs write to `results/smoke/` (local) and are deleted after their assertions; they never print an episode
  metric.
- Commits to `main` by explicit path, never pushed; a commit that carries a cluster run's code has `cluster run` in its
  subject.

## Review focus

1. Same-painting negatives: a batch must never hold two captions of one painting (sampler test on a toy index with
   many captions per painting).
2. Row alignment: features written for evaluation are in feature-row order with NaN outside selection, and the caption
   of row i is the caption of `sample_ids[i]` (test on a toy with shuffled `sample_ids`).
3. Frozen parameters: in LB only the last layer, final layer norm and projection of each encoder (and the temperature)
   require grad; in LoRA only the adapters (and the temperature); LP at step 0 reproduces the frozen features exactly.
4. The selection metric and its tie rule (smaller learning rate, then earlier epoch) on a hand-made table.
5. No held row anywhere: the cache builder and trainer refuse a row whose split is held (test).

---

### Task 1 (data, Sonnet): rows, captions, image cache

**Files:** create `src/test/20261124_clip_lightweight_ft/ft_data.py`, `test_ft_data.py`.

**Produces:** `split_index(data) -> {"scorer_train", "val", "selection": np.ndarray rows}` (from `artelingo_splits`;
held excluded); `captions(sample_ids, annotations) -> list[str]` in feature-row order; `painting_table(data, rows)` (the
painting of each row, unique paintings, row → painting position); `build_image_cache(out_dir, data, annotations,
wikiart_dir)` writing `images_uint8.npy` (N, 224, 224, 3), `paintings.json` (painting → cache index, image path) and
`cache_record.json` (SHA-256s, counts, processor settings), using the CLIP processor's resize and centre crop without
normalisation, refusing to overwrite; `load_image_cache(dir)` (memmap, SHA check).

- [ ] Tests first (toy data): split rows exclude held; captions follow `sample_ids`; cache shape, dtype and painting
  index on 3 toy images; refuse-overwrite; a held row raises.
- [ ] Implement; commit.

### Task 2 (trainer, Opus): three variants, sampler, val retrieval, outputs

**Files:** create `ft_train.py`, `test_ft_train.py`.

**Produces:** `python ft_train.py --variant {LP,LB,LoRA} --lr <float> --epochs 10 --out <dir> [--smoke]` with:
- the painting-unique sampler (one caption per painting per epoch, shuffled by seed 0 and the epoch);
- LP on the cached frozen features (`load_artelingo` features: the map, initialised to the identity, acts on the raw
  projection features, and its output is L2-normalised), LB and LoRA on the image cache and tokenised captions
  (`max_length 77`, `padding="max_length"`, `truncation=True`);
- symmetric InfoNCE with a learnable temperature initialised from `logit_scale`; AdamW, 5% linear warm-up, cosine decay,
  bf16 autocast when supported; weight decay 0.1 for LB weight matrices (none on biases and layer norms), 0 for LP
  and LoRA;
- after each epoch: val retrieval (image→caption R@1 over all val captions, caption→image R@1 over the val paintings'
  images, their mean = the selection metric), written to `metrics.json`; the best epoch's trained parameters saved;
- at the end: the best epoch's features for val and selection rows (`features.npz`: `rows`, `img`, `txt`, float32,
  projection outputs), and `run_record.json` (args, commit, host, GPU name, times, best epoch).

- [ ] Tests first (tiny, CPU, real CLIP weights from the local HF cache, 32 rows): sampler uniqueness; LP identity at
  step 0; the trainable-parameter sets of LB and LoRA; one training step lowers the loss on a fixed toy batch; val
  retrieval on a hand-made similarity matrix; the output files and their row order.
- [ ] Implement; `--smoke` run locally on 64 scorer-train rows and 32 val rows (CPU) passes; commit.

### Task 3 (cluster scripts, Sonnet): launcher and data sync

**Files:** create `scripts/run_clipft.sh`, `scripts/das6_sync_clipft.py`.

- `run_clipft.sh <variant> <lr> [--check-only|--smoke]`: like `scripts/run_mllm_probe_8b.sh`: node paths with
  overrides (features `/local/wding/pre_extract/artelingo/features`, annotations
  `/local/wding/Dataset/artelingo/artelingo_train.json`, image cache `/local/wding/Dataset/pre_extract/artelingo_clip224`,
  HF cache `/var/scratch/wding/cache/hub`, offline), checks every input (cache SHA from `cache_record.json`, the CLIP
  snapshot, a visible GPU, `import peft`), prints `inputs ok`, then runs `ft_train.py` with the output under
  `outputs/clipft/<variant>_lr<lr>`.
- `das6_sync_clipft.py [--node N] [--run]`: on the pattern of `scripts/das6_sync_mllm_probe_8b.py` (cluster.py as a
  library, `plan_data_sync` / `run_data_sync`): the image cache, the annotations, the feature cache and the HF repo
  `openai/clip-vit-base-patch32`; prints the plan without `--run`.

- [ ] Implement; `bash scripts/run_clipft.sh LP 1e-3 --check-only` passes locally with local path overrides; the sync
  script prints a plan; commit with `cluster run` in the subject.

### Task 4 (evaluation, Sonnet): episode scoring and comparisons

**Files:** create `ft_eval.py`, `test_ft_eval.py`.

**Produces:** `python ft_eval.py --features <variant>=<features.npz> ... --out results/eval.json`: per variant, features
placed in feature-row order with NaN outside selection; for seeds 42, 49, 50, 51 the episodes from
`src/test/20261030_aspect_baselines/results/episodes_seed{s}.npz` (SHA-256 asserted against `baselines_seed{s}.json`),
`cosine_scores` and `per_anchor`; plain cosine from the same path on the frozen features (asserted equal to
`per_anchor_seed{s}.npz`'s `cosine__*`); AFF, B and B′(A0) per-anchor arrays from round 3's stored results (seeds 49 to
51) and round 3's or round 4's seed-42 records; B′(A1) on seed 42 only. Outputs: per seed, pooled over 49 to 51 (one
cluster per painting), and per aspect pair: R@1 and either rate of each scorer, and the paired differences of spec §5
with 95% intervals (`cluster_bootstrap`, 5,000, seed 42).

- [ ] Tests first (synthetic arrays): row placement and NaN mask; pooled clustering across seeds; the difference
  signs.
- [ ] Implement; commit.

### Task 5 (main session): runs

- [ ] Build the image cache locally (background; CPU; `/data/SSD2/pre_extract/artelingo_clip224/`); record SHA-256s.
- [ ] Sync to nodes 404, 405, 411 (`/usr/bin/python3 scripts/das6_sync_clipft.py --node <n> --run`; node404 needs
  only features and annotations, but all three get the full set for flexibility only if the transfer is quick).
- [ ] `cluster status` on each node; a `--check-only` job on each node (real in-job data check).
- [ ] One cluster smoke job (`--smoke`) per variant on its node; read only pass/fail.
- [ ] The nine runs: LP × 3 on node404, LB × 3 on node405, LoRA × 3 on node411; `cluster watch` each in the background;
  `cluster pull` on success.
- [ ] Selection per variant (spec §4) from the pulled `metrics.json`; `ft_eval.py` on the three selected models.
- [ ] An independent agent recomputes the evaluation numbers from the pulled features with its own code.

### Task 6: final review and report

- [ ] Whole-branch final review (most capable model; re-derives the evaluation numbers and the selection); one fix
  wave; a scoped re-review.
- [ ] The report `docs/reports/auto/v2/2026-11-24_clip_lightweight_ft.md` (AFF and the comparators beside every number;
  figures: val retrieval per epoch; R@1 of plain, fine-tuned and AFF per pair), its `reports_sum.md` row,
  `scripts/check_reports_sum.py`; update of the CVPR memo's §2 with the outcome; storage report.
