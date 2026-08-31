# Buddy-init beats generic init at RedCaps-300k — but neither beats raw CLIP

**Date:** 2026-08-19 · **Dataset:** RedCaps, first 300,000 rows of `redcaps_train.json` (a scale slice toward the full 3,106,894-row corpus) · **Branch:** `experiment/buddy_init_ablation`
**Code:** `scripts/run_init_ablation_redcaps_300k.sh`, `scripts/run_init_ablation.sh`, `scripts/analyze_init_ablation.py`
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 1
**Prior result:** `docs/reports/2026-08-16_buddy_init_ablation.md` (RedCaps-150k, same question)

---

## TL;DR

**Buddy-graph initialization beats generic (`imgtxt`) initialization on RedCaps-300k, cleanly, in both retrieval directions, all 3 seeds:**

- **t2i R@1:** mean Δ = **+7.43 ± 0.93**, mean/SEM = **+13.9** (buddies wins 3/3 seeds)
- **i2t R@1:** mean Δ = **+5.00 ± 1.20**, mean/SEM = **+7.2** (buddies wins 3/3 seeds)

This is a materially *cleaner* result than the 150k-scale finding (`2026-08-16_buddy_init_ablation.md`), which was direction-and-dataset-dependent (RedCaps-150k won clean, but Impressions-150k split — i2t won, t2i lost). At 300k, both directions win with consistent, low-variance effects across seeds — no split, no outlier seed.

**But the win is relative to a weak baseline, not to CLIP.** `imgtxt` is a within-model alternative initialization, not an external baseline. Checked against the pipeline's own `test_diff`/`test_pre_diff` metrics (conditioned model − raw frozen-CLIP retrieval on the same embeddings, no conditioning at all), **both `imgtxt` and `buddies` retrieve worse than plain CLIP**, at every seed, in both directions, under both the oracle upper bound and the realistic single-forward-pass predictor. Buddy-init closes most (t2i) or some (i2t) of that gap relative to `imgtxt`, but never closes it fully — CLIP zero-shot remains the best retriever measured in this sweep. See "Buddy-init vs. raw CLIP" below — this is the load-bearing caveat for how this result should be read and framed.

**What this report does not cover:** results at 500k, 1M, or the full 3.1M scale. Getting the pipeline to run correctly at those scales — not just fast, but *correct* — consumed the large majority of this work session; see "The scaling engineering work" below. A 500k real sweep was in progress when a hard time constraint (external reporting deadline) forced a stop; only the 300k result above is a completed, real experiment.

---

## Method

Same operating point as every prior Experiment 1 report, for direct comparability:

| | value |
|---|---|
| `optimizer.lr` / `lr_label` | 1e-3 / 1e-4 |
| `model.embedding_dim` | 16 |
| `train.buddies.alpha` | 0.5 |
| `model.num_layers` | 6 |
| training-time buddy terms | **off** — `lambda_buddy=0`, `lambda_buddy_con=0`, `buddy_refresh=False` |
| seeds | 1, 2, 3 |
| epochs / eval interval | 100 / 10 |
| test set | RedCaps' own in-domain test set (`redcaps_test.json`, 25k pairs, `test_ratio=0.2` → 5k evaluated), same test set used at every RedCaps scale so far (150k, 300k, full) |

Training annotation is the first 300,000 rows of `redcaps_train.json` (arbitrary prefix, not a random sample — same slicing convention used for the 500k/750k/1M slices prepared alongside this). Own feature store at `/data/SSD2/pre_extract/redcaps_300k/features`, fully extracted before this run (not timed as part of the sweep).

## Results

**`test_oracle/t2i_R1`**

| seed | imgtxt | buddies | Δ (buddies − imgtxt) |
|---:|---:|---:|---:|
| 1 | 18.90 | 25.30 | +6.40 |
| 2 | 17.30 | 25.50 | +8.20 |
| 3 | 17.60 | 25.30 | +7.70 |

3/3 seeds: buddies wins. Mean Δ = **+7.43 ± 0.93** R1 pts, mean/SEM = **+13.9**.

**`test_oracle/i2t_R1`**

| seed | imgtxt | buddies | Δ (buddies − imgtxt) |
|---:|---:|---:|---:|
| 1 | 8.00 | 13.00 | +5.00 |
| 2 | 9.20 | 13.00 | +3.80 |
| 3 | 7.30 | 13.50 | +6.20 |

3/3 seeds: buddies wins. Mean Δ = **+5.00 ± 1.20** R1 pts, mean/SEM = **+7.2**.

Both effects are far above the noise floor (~0.1-0.7 R1, `2026-06-24_buddy_progress_report.md` §8a) and consistent seed-to-seed — no split by direction the way Impressions-150k showed.

## Buddy-init vs. raw CLIP

Every eval call in this pipeline also logs `test_raw/*` (frozen-CLIP embeddings, no label conditioning at all — the model's own eval harness computes this as its baseline) and two derived diffs already wired into `src/eval/pipeline.py`: `test_diff/*` = oracle-conditioned − raw (best case: search over all label conditions per query, an upper bound no real deployment gets), and `test_pre_diff/*` = predictor-conditioned − raw (the model's actual `condition_predictor`, single forward pass — the realistic, deployable comparison). Both are `ours − CLIP`; negative means CLIP wins.

`test_raw` is identical across all 6 runs (same frozen backbone, same test set): **t2i_R1 = 28.1, i2t_R1 = 29.7.**

| metric (mean over 3 seeds) | imgtxt | buddies | gap closed by buddies |
|---|---:|---:|---:|
| `test_diff/t2i_R1` (oracle − CLIP) | −10.17 | **−2.73** | ~73% |
| `test_diff/i2t_R1` (oracle − CLIP) | −21.53 | **−16.53** | ~23% |
| `test_pre_diff/t2i_R1` (predictor − CLIP, deployable) | −13.97 | **−3.90** | ~72% |
| `test_pre_diff/i2t_R1` (predictor − CLIP, deployable) | −16.47 | **−6.77** | ~59% |

Every cell is negative. Under the realistic predictor setting (the one that matters for a deployable claim), `buddies` still trails raw CLIP by 3.9 R1 pts on t2i and 6.8 pts on i2t — even though it's a large, consistent improvement over `imgtxt` (which trails by 14.0 and 16.5 pts respectively). Read together with the paired result above: **buddy-init is the better of the two initializations tested, and meaningfully narrows this model's gap to its own frozen-CLIP backbone, but neither initialization strategy makes the conditioning approach worth using over plain CLIP retrieval on this test set at this scale.** This was not checked in the 150k report (`2026-08-16_buddy_init_ablation.md` does not report `test_raw`/`test_diff` at all) — whether the same shortfall holds there is unverified, not "no", since those runs' wandb data was not available to re-query for this report (see Reproduction).

This matters for how the paired win (C5/C6 in the spec) should be framed: it is evidence that buddy-graph structure is a better initializer than the generic alternative *within this training approach*, not evidence that this training approach beats the untouched CLIP baseline it started from.

### A data-integrity issue found and fixed while producing this table

The first pass at this analysis (before a GPU driver outage mid-sweep, see below) showed an inflated, noisier i2t result (mean Δ = +9.70 ± 8.23, driven by a seed-1 outlier of +19.10). The cause: the driver outage killed the original seed-1 `buddies` run right after it logged its epoch-0 metrics, and when the seed was re-run to completion, `scripts/analyze_init_ablation.py`'s per-cell aggregation (`groupby("strategy")[metric].max()`) silently preferred whichever of the two same-cell runs had the numerically higher value — which was the killed run's near-random epoch-0 number, not the real, converged 99-epoch result. `fetch()` now excludes any wandb run with `state != "finished"` before that aggregation ever runs (commit `a3abeab`). The table above is post-fix. This is a pre-existing latent bug in code that predates this session (Task 4, originally TDD-verified) — its test suite never exercised a same-cell duplicate-run scenario, since that only arises from a crash-and-resume, which hadn't happened before today.

## The scaling engineering work

Most of this session was spent making the pipeline work *correctly* at scales beyond the previously-validated 150k, in preparation for a full-3.1M run. Five real, independent bug fixes (one of them itself a two-step correction) landed across the commits below, all only manifesting above roughly 500k-1M samples (invisible at 150k):

1. **GPU OOM in isolated-node fallback search** (`buddy_graph.py`) — an unbatched `[n_isolated, N]` matmul; ~300GB attempted at N~3M. Fixed by batching (commit `e03a54c`).
2. **`spectral_embedding`'s OOM at N~3M** (commit `5bc7c56`, the largest single fix) — sklearn's own `SpectralEmbedding(eigen_solver="amg")` has an internal dtype leak (`sparse.eye()` defaulting to float64 regardless of the Laplacian's dtype) that silently doubles memory through pyamg's whole multigrid hierarchy. Fixing it naively (forcing float32) broke *correctness* instead — a second, independent bug in `scipy.sparse.linalg.lobpcg`'s default tolerance, which scales with the array's dtype epsilon and becomes meaningless at float32/N~3M scale (returns instantly "converged" on garbage). Both fixed via targeted monkeypatches of sklearn's own already-validated code path, plus using pyamg's unsmoothed aggregation (`smooth=None`) for a further ~3x memory cut with no correctness cost. Verified against sklearn's own unmodified output via cluster-recovery ARI on a synthetic graph (0.75 vs 0.76/0.75 baseline). The same commit bundles three more scaling bugs hit while chasing this one down to a real 3.1M end-to-end run: **(a)** `sparse_cosine_distance`'s edge-endpoint gather materializing the full `(nnz, D)` array in one shot (~102GB attempted at nnz~54M) — fixed by batching; **(b)** `ensure_connected`'s mix-weighted concat silently upcast float32→float64 via a `np.sqrt(alpha)` NumPy scalar (NEP 50 promotion), a ~51GB transient spike at N~3M from one line — fixed by forcing float32 scalars; **(c)** three "dead generation" copies of the ~3M-row feature arrays (shard-list → raw → normalized) held alive simultaneously instead of freed as superseded, ~38GB of pure waste — fixed with in-place normalization and explicit `del` (`buddy_graph.py`, `compute_buddies.py`, `embedding_manager_nocache.py`). Net result of this one commit: full 3.1M-sample buddy-graph construction, ~15 min, peak RSS 39.7GB (was killing the process near 60GB on a 64GB machine).
3. **`buddy_knn_preservation`'s fixed chunk size (2048)** didn't scale with N — fine at 150k (~1.2GB/chunk), ~25GB in one allocation at N~3M (commit `5bbc25c`); made N-adaptive.
4. **`buddy_knn_preservation`'s inner loop was an unvectorized Python `for`** over every node — 3.1 million individual iterations at full scale — replaced with a vectorized `torch.repeat_interleave`/`scatter_add_` formulation (commit `d5d1992`).
5. **The actual root cause of a 5-hour hang** — a two-step story. `compute_comb_all_eval` first OOM'd at million-sample scale; the first fix (commit `1208a38`) moved its output to CPU to dodge the GPU memory spike, which "worked" in the sense that the OOM went away, but silently wrecked the performance of its only consumer, `buddy_knn_preservation`, whose similarity computation is O(N²·D): moving from GPU to CPU combined ~44x more work (at N=1M vs the N~150k the code was tuned for) with a compute substrate ~50-100x slower per-FLOP than GPU. No exception, no OOM — just a computation that looked identical to a hang for hours (confirmed via process inspection: GPU idle, ~14 CPU cores pegged, zero disk I/O, frozen RSS — pure unproductive-looking CPU compute with no progress signal). The real fix (commit `a5233a4`) pre-allocates the output tensor once on GPU instead of list-then-`torch.cat` (which is what caused the original memory spike in the first place), avoiding the CPU move entirely — the OOM-safe fix that didn't need to sacrifice GPU residency. Verified: a 1M-scale isolated benchmark of this one function dropped from "doesn't finish in 5+ hours" to 90.7s.

Additionally, a **cuVS (approximate nearest-neighbor) backend was added** for `mutual_knn` (commit `30bbc43`, threshold corrected in `ad01e41`), auto-selected above `CUVS_MIN_N=1,000,000` (measured crossover: exact still wins at 500k — 17.9s vs 31.1s cuvs — cuVS wins 2.2x by 1.5M — 37.3s vs 83.7s exact, single modality). Below the threshold, behavior is byte-identical to before (confirmed by the full existing test suite). This targets the O(N²) graph-construction cost specifically, separate from the memory/correctness bugs above.

Also found and fixed in passing: `SMOKE=1` silently ran full epoch counts instead of the intended 2-epoch smoke test in both dataset wrapper scripts (commit `7bf9e9d`).

All of the above were exercised and confirmed working via a full, real end-to-end run at 1M scale (~8 minutes total, buddy-graph construction through training and eval) before the environment's GPU driver dropped out for unrelated infrastructure reasons.

## Scope / what's not done

- **500k**: a real sweep (matching this report's protocol) was in progress — `imgtxt`'s 3 seeds completed and logged; `buddies` had not started — when a hard reporting deadline forced a stop. Not reported here.
- **1M**: validated for pipeline *correctness and speed* only (a single-seed, throwaway/offline run completed cleanly in ~8 minutes after the fixes above), not run as a real 3-seed sweep.
- **3.1M (full)**: same — validated up through buddy-graph construction and a partial SMOKE run (2 strategies, 2 epochs) with no further crashes after the fixes above, but not completed as a full real sweep. Rough time budgeting suggests a full 3.1M sweep (2 strategies × 3 seeds × 100 epochs) is a multi-hour-to-day-scale commitment even after all the fixes in this report, dominated by `buddy_knn_preservation`'s O(N²) eval cost at that N (extrapolated ~15 min/eval-call × 10 calls/run × 6 runs ≈ 15 hours just for that one diagnostic metric, on top of training time) — worth a follow-up look at reducing its own cost (e.g. subsampling the preservation check, or applying the same cuVS approach used for `mutual_knn`) before attempting it.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_init_ablation_redcaps_300k.sh   # 2 strategies x 3 seeds, 100 epochs each
python scripts/analyze_init_ablation.py --tag init-ablation-redcaps-300k
```
