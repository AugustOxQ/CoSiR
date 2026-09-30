# Does mutual-KNN K matter, and does it need to scale with N? — Experiment 16.1 (Stage A)

**Date:** 2026-09-01 · **Dataset:** RedCaps at three independently-drawn, uniform-random, all-350-subreddit scales — 150k (`redcaps_150k.json`), 300k (`redcaps_300k_diverse.json`), 500k (`redcaps_500k_diverse.json`), the same three stores used by Experiment 9/C1a · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `src/test/20260901_buddy_k_scaling/buddy_k_sweep.py`
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 16, subsection 16.1
**Precursor:** `src/test/20260609_conditional_buddy/dim_hparam_study.py` (the project's only earlier K sweep — Impressions only, statistics-only, never checked against training, and predates the strict/union distinction this report depends on)

---

## TL;DR

Every RedCaps finding to date (C5 through C15) fixes mutual-KNN **K=30** as part of a shared "matched operating point," inherited from a single early sweep on a *different* dataset that never isolated K's own effect. This experiment is Stage A of a staged check on that assumption: a training-free graph diagnostic sweeping **K ∈ {10, 20, 30, 50, 75, 100} × N ∈ {150k, 300k, 500k}** (18 cells), asking (1) how K reshapes strict buddy (`A_img ∩ A_txt`, the real cross-modal signal) vs. union buddy (`E`, what training actually consumes), and (2) whether K should scale with N.

**Strict buddy is sparse everywhere in this grid — the median RedCaps node has zero strict buddies at every K and every N tested**, from 60.9% (150k, K=100) to 94.8% (500k, K=10) of nodes having no strict buddy at all. This is the concrete version of the sparsity concern motivating this experiment: even at K=100, strict buddy alone cannot supervise most of the dataset, which is exactly why every production path falls back to the noisier union graph.

**Strict-buddy average degree rises with K but *falls* with N at fixed K** — e.g. at K=30: 0.371 (150k) → 0.305 (300k) → 0.266 (500k). Holding the 150k/K=30 value as an invariant, in-grid interpolation predicts **K≈35 at 300k** and **K≈39 at 500k** — a much gentler scaling than linear-in-N (K+5 and K+9 off the 150k baseline, not K+30/K+90).

**Union buddy's quality proxy (subreddit lift) does not degrade anywhere in the grid** — it stays in a tight 22.7–23.0× band across all 18 cells, so the K(N) question is a pure structural-sparsity effect, not a signal-quality collapse. But the union graph's *reliance* on repair/single-modality edges is K-dependent: at K=10, only 1.2–1.4% of union edges are genuine strict ("both") overlap and 2.0–2.4% are synthetic repair edges (added because a node had no mutual-KNN neighbor in either modality at all); by K=100 the "both" fraction only rises to ~2.0–2.8% while repair drops to near zero. **Strict buddy stays a small minority of the union graph across the entire tested range** — increasing K does not make union buddy meaningfully "more strict," it mostly grows the single-modality majority alongside it.

No training was run. Stage B (the gated 3-seed training ablation testing whether any of this moves retrieval) is scoped but not launched, pending review of this report.

---

## Method

### What's measured, per (N, K) cell

For each of the 3×6 = 18 cells, `A_img`/`A_txt` are built via `mutual_knn(features, K, device)` (`src/conditional_buddy/buddy_graph.py`, exact GPU brute-force at these scales — `CUVS_MIN_N=1,000,000` keeps the approximate CAGRA backend out of range). Three groups of statistics are recorded per cell:

- **Strict buddy** (`B = A_img.multiply(A_txt)`, the mutual-in-both-modalities intersection — the signal Experiment 16's context calls the "real" cross-modal evidence): edge count, average degree, the fraction of nodes with zero strict-buddy edges, and median/P90 degree (reported because the median is uninformative on its own here — see Results).
- **Union buddy** (`E = ensure_min_degree(union_graph(A_img, A_txt), img, txt, device)`, what `compute_buddy_init` and every training-time buddy term actually consume): edge count, average degree, and edge-type composition (`img_only`/`txt_only`/`both`/`repair`, via `classify_edges` — reusing Experiment 15.1's provenance code, not reimplementing it). `union_both_edge_count` is asserted equal to the strict edge count at every cell as an internal-consistency check (strict buddy is definitionally the "both" slice of union buddy).
- **Subreddit-lift quality proxy** (`subreddit_lift(data, e, top_k=None)`, reused as-is from Experiment 9/C1a's `redcaps_buddy.py`, computed on the repaired union graph `E`): `overall_lift`, `obs_same_frac`, `exp_same_frac`, and the count of subreddits clearing the reliability filter.

### Dataset-variant correction

The spec's first draft named the plain `redcaps_300k`/`redcaps_500k` feature stores. Pre-implementation research (two parallel Codex feasibility passes) found these are subreddit-grouped training-prefix stores covering only 15 and 28 of RedCaps' 350 subreddits respectively — unusable for a cross-scale subreddit-lift comparison. This run uses `redcaps_300k_diverse`/`redcaps_500k_diverse` instead (with their matching annotation files), the same independently-drawn, all-350-subreddit samples Experiment 9/C1a already validated and used for its own 300k/500k scale check. 150k needs no substitution — `redcaps_150k` was never a subreddit-grouped prefix.

### K(N) inversion procedure

Given the 150k/K=30 strict average degree as the target invariant, and each N's own six measured `(K, strict_avg_degree)` points: violations of monotonicity are checked (exact mutual-KNN neighborhoods are nested in K, so degree should be nondecreasing — none were observed in this run), a monotone PCHIP interpolator is fit over the six points, and bisection search finds the smallest K meeting the target. Predictions are clamped and flagged, not extrapolated, if the target falls outside `[10, 100]`'s attainable range (it did not, for either 300k or 500k).

### Deterministic Stage-B short-list rule

`{anchor K=30} ∪ {round(K*) if it differs from 30 by more than 1} ∪ {one bracketing tested-grid point, on the opposite side of K* from the anchor, nearest by log-distance}` — chosen so a rounding-noise prediction near 30 doesn't manufacture a spurious third run, and so the three selected values straddle the predicted operating point rather than clustering on one side of it.

---

## Results

**Command run:**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260901_buddy_k_scaling/buddy_k_sweep.py --selftest   # K(N)/shortlist algorithm check, offline
python src/test/20260901_buddy_k_scaling/buddy_k_sweep.py               # full 18-cell GPU sweep
```

### Full 18-cell table

| N | K | strict edges | strict avg deg | strict zero-deg frac | union edges | union avg deg | union both frac | union repair frac | subreddit lift | qualifying subreddits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 150k | 10 | 6,654 | 0.089 | 93.5% | 462,839 | 6.171 | 1.44% | 1.98% | 22.730× | 99 |
| 150k | 20 | 16,236 | 0.216 | 87.6% | 941,595 | 12.555 | 1.72% | 0.35% | 22.742× | 139 |
| 150k | 30 | 27,797 | 0.371 | 82.5% | 1,429,418 | 19.059 | 1.94% | 0.12% | 22.788× | 159 |
| 150k | 50 | 54,662 | 0.729 | 74.5% | 2,413,639 | 32.182 | 2.26% | 0.03% | 22.883× | 189 |
| 150k | 75 | 93,941 | 1.253 | 66.8% | 3,644,570 | 48.594 | 2.58% | 0.01% | 22.912× | 208 |
| 150k | 100 | 137,947 | 1.839 | 60.9% | 4,880,238 | 65.070 | 2.83% | 0.005% | 22.878× | 217 |
| 300k | 10 | 11,632 | 0.078 | 94.3% | 906,038 | 6.040 | 1.28% | 2.19% | 22.833× | 137 |
| 300k | 20 | 27,051 | 0.180 | 89.4% | 1,845,758 | 12.305 | 1.47% | 0.39% | 22.787× | 173 |
| 300k | 30 | 45,741 | 0.305 | 85.1% | 2,811,003 | 18.740 | 1.63% | 0.13% | 22.735× | 197 |
| 300k | 50 | 89,538 | 0.597 | 78.1% | 4,763,878 | 31.759 | 1.88% | 0.03% | 22.763× | 217 |
| 300k | 75 | 152,549 | 1.017 | 71.3% | 7,221,895 | 48.146 | 2.11% | 0.01% | 22.802× | 233 |
| 300k | 100 | 222,689 | 1.485 | 65.9% | 9,684,016 | 64.560 | 2.30% | 0.005% | 22.831× | 250 |
| 500k | 10 | 18,457 | 0.074 | 94.8% | 1,485,243 | 5.941 | 1.24% | 2.36% | 22.965× | 160 |
| 500k | 20 | 40,501 | 0.162 | 90.3% | 3,021,891 | 12.088 | 1.34% | 0.42% | 22.859× | 201 |
| 500k | 30 | 66,539 | 0.266 | 86.6% | 4,603,486 | 18.414 | 1.45% | 0.14% | 22.785× | 215 |
| 500k | 50 | 127,302 | 0.509 | 80.5% | 7,823,655 | 31.295 | 1.63% | 0.04% | 22.714× | 241 |
| 500k | 75 | 215,785 | 0.863 | 74.3% | 11,893,043 | 47.572 | 1.81% | 0.01% | 22.731× | 259 |
| 500k | 100 | 314,480 | 1.258 | 69.3% | 15,981,763 | 63.927 | 1.97% | 0.006% | 22.748× | 267 |

Median and P90 strict degree are 0 and ≤5 at every single cell — omitted from the table since the median column would read `0.0` eighteen times in a row; the average-degree and zero-fraction columns already carry that information. `union_both_edge_count == strict_edge_count` and the four union edge-type fractions summed to `1 ± 1e-6` at all 18 cells (script-enforced assertions, independently re-verified from the raw artifacts after the run).

### Strict buddy: sparse at every setting tested

The zero-strict-degree fraction never drops below 60.9% in this grid (150k, K=100 — the single densest cell tested) and reaches as high as 94.8% (500k, K=10 — the sparsest). Median strict degree is 0 everywhere; P90 only exceeds 1 once K≥50. **This is the quantitative version of the premise motivating this experiment**: strict buddy is not a marginal minority signal that a slightly larger K would fix — even an aggressive 3–10× increase over the project's default K=30 leaves the majority of nodes with zero strict buddies at every scale tested.

### Union buddy: quality-stable, but strict overlap stays a small minority

Subreddit lift — the training-free proxy for whether the union graph's edges are still real, content-specific structure rather than noise — is essentially flat across the entire grid (22.71×–22.97×, a 1.1% relative spread over 18 cells that vary K by 10× and N by 3.3×). This rules out "K just needs to be big enough that the graph stops being noisy" as an explanation for why K might matter: by this proxy, it already isn't noisy, at every K tested.

What *does* move with K is the union graph's own composition. The `both` fraction (edges that are also strict buddies) rises modestly with K — from ~1.2–1.4% at K=10 to ~2.0–2.8% at K=100 — while the `repair` fraction (synthetic edges added by `ensure_min_degree` for nodes with zero mutual-KNN neighbors in either modality) falls sharply, from ~2.0–2.4% at K=10 to effectively zero by K=75. Both trends make mechanical sense — a larger K means fewer totally-isolated nodes needing repair, and a slightly higher chance that a node's top-K neighbor sets overlap across modalities — but neither moves the `both` fraction out of low single digits anywhere in the grid. **The union graph is, and stays, overwhelmingly single-modality edges at every K tested**; K controls how many single-modality edges exist, much more than it controls what fraction of them are corroborated by the other modality.

### K(N): a sublinear, in-grid prediction

Strict average degree at fixed K falls as N grows (visible directly in the table: every column pair 150k→300k→500k decreases at matched K). Holding the 150k/K=30 value (`0.37063`) as the target invariant and inverting each target N's own measured curve (isotonic-checked — zero monotonicity violations at either target N — then PCHIP-interpolated, then bisected):

| Target N | Predicted K* | Attainable range at target N | Rounded training K | Deterministic Stage-B shortlist |
|---|---:|---:|---:|---|
| 300k | 34.800 | [0.078, 1.485] | 35 | {30, 35, 50} |
| 500k | 38.990 | [0.074, 1.258] | 39 | {30, 39, 50} |

Both predictions land comfortably inside the tested `[10, 100]` range (not a boundary clamp), and both are far below what naive proportional scaling would suggest (2×N → 2×K would predict K=60 at 300k; the actual invariant-preserving prediction is K≈35, an 8.7% increase over the 150k baseline for a 2× increase in N). **The relationship is real but sublinear** — N growing 2×/3.3× only calls for a ~17%/30% increase in K to hold strict-graph density constant, not a proportional one.

### Figure

No figure was generated for this diagnostic — all 18 cells are shown in full in the table above, and the two headline trends (strict sparsity vs. K/N, and the flat subreddit-lift proxy) are each fully captured by two of the table's columns; a scatter/line plot was judged to add rendering overhead without showing anything the table doesn't already state directly at this cell count (unlike Experiment 9/C1a's per-subreddit report, which plots 159–267 individual points per panel where a table would be unreadable).

---

## Caveats

- **This report establishes structure, not a retrieval effect.** Everything above is a graph-statistics and training-free-quality-proxy claim. Whether any of it moves actual retrieval numbers — whether K=35/K=39 outperform K=30 at 300k/500k, and whether the freeze-vs-trainable asymmetry from C9 holds across K — is exactly what the still-gated Stage B (16.2) is for. Do not read the K(N) prediction here as a retrieval-optimal K; it is a structural-invariant-preserving K.
- **The invariant choice (average strict-buddy degree) is one defensible choice, not the only one.** It was picked because it most directly targets the sparsity concern that motivated this experiment. The table above reports enough columns (zero-fraction, union composition, lift) to re-derive a K(N) prediction under a different invariant if Stage B's results suggest average degree isn't the property that matters for training.
- **Interpolation is over only 6 measured points per N.** The K* predictions (34.8, 39.0) fall between measured grid points (30 and 50 in both cases) — real values, not measured ones. The deterministic short-list rule accounts for this by always including a measured bracket point (K=50 in both cases here) alongside the interpolated prediction, so Stage B's runs are not solely at unmeasured-structure K values.
- **`_diverse` sample composition is not identical in kind to the plain training-prefix stores.** The 300k/500k `_diverse` stores are independently-drawn, uniform-random, all-350-subreddit samples (Experiment 9/C1a's construction), not literal supersets of `redcaps_150k` or of the plain `redcaps_300k`/`redcaps_500k` training stores used elsewhere in this project's C5/C6 training ablations. If Stage B trains on these `_diverse` stores, its results are not a strict apples-to-apples continuation of C5/C6 (which used the plain stores) — this is flagged for Stage B's design, not resolved here.
- **No training-code path was touched or exercised.** This report and its script are entirely read-only with respect to `src/hook/train_cosir.py` and the training pipeline; nothing here changes what today's default K=30 pipeline does.

---

## Reproduce

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260901_buddy_k_scaling/buddy_k_sweep.py --selftest   # offline K(N)/shortlist algorithm check
python src/test/20260901_buddy_k_scaling/buddy_k_sweep.py               # full 18-cell sweep, writes stage_a_cells.{csv,json}
```

CUDA is used automatically if available (`torch.cuda.is_available()`); the run underlying this report used the exact GPU backend at all three scales (`CUVS_MIN_N=1,000,000` keeps N=500k below the approximate-search crossover). No GPU training is involved — this is graph construction and statistics only, on the order of tens of seconds per `mutual_knn` call even at the largest scale tested.

Raw per-cell output: `src/test/20260901_buddy_k_scaling/stage_a_cells.csv` / `.json` (18 rows, schema documented in-script).
