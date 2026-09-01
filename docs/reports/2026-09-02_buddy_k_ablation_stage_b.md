# Does mutual-KNN K move real retrieval numbers? — Experiment 16.2 (Stage B)

**Date:** 2026-09-02 · **Dataset:** RedCaps 300k/500k, `redcaps_300k_diverse`/`redcaps_500k_diverse` (same independently-drawn, all-350-subreddit stores as Stage A/Experiment 9/C1a) · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `scripts/run_buddy_k_ablation.sh`, `scripts/run_buddy_k_ablation_redcaps_{300k,500k}.sh`, `scripts/analyze_buddy_k_ablation.py`
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 16, subsection 16.2
**Precursor:** `docs/reports/2026-09-01_buddy_k_scaling_stage_a.md` (Experiment 16.1 — the graph-diagnostic sweep that derived this experiment's K shortlists and the K(N) scaling prediction being tested here)

---

## TL;DR

Stage A (16.1) found that mutual-KNN K reshapes strict/union buddy-graph composition, and derived a K(N) prediction — holding 150k/K=30's strict average degree constant implies **K≈35 at 300k** and **K≈39 at 500k**, not the K=30 used everywhere in this project's prior RedCaps work. Stage B trains at those shortlists (`{30,35,50}` at 300k, `{30,39,50}` at 500k, trained-arm only — the freeze-vs-trainable crossing originally planned was dropped) to check whether any of that structure actually moves retrieval.

**It does, at exactly the K(N)-predicted value: 500k/K=39 moves `test_oracle` i2t R1 by +0.97 over K=30, mean/SEM=+14.5** — the clearest, best-supported result in the whole sweep, clearing this plan's `mean/SEM ≥ 2` significance bar and its ~0.1–0.7 R1 measured noise floor by a wide margin. K=30 (today's project-wide default) is measurably suboptimal for i2t retrieval at 500k, and the value that fixes it is the one 16.1's structural invariant predicted, not the K=50 bracket point.

The picture is not uniform, though: 500k/t2i shows no comparable K=39 effect (+0.10, mean/SEM+0.9 — noise), and 300k shows only a smaller, borderline-to-modest, **t2i-only** signal (K=35 +0.20 mean/SEM+2.0, K=50 +0.33 mean/SEM+2.8), with 300k/i2t flat. So K materially matters, but which axis (t2i vs. i2t) it moves and at what N is not the same story at every scale — worth treating as an open question for Stage C rather than a settled asymmetry.

The user stopped the sweep at 16 of the originally-planned 18 runs once the signal above was already visible; 500k/K=50 has only 1 seed (not a 3-seed paired claim) and is reported as a directional data point only.

---

## Method

### Design (per spec §4 Experiment 16.2, as scoped this session)

For each K in each scale's Stage-A shortlist, 3 seeds at the C5/C6/C9 matched operating point: `combine_side=img`, `dim=16`, `α=0.5`, `lr=1e-3`/`lr_label=1e-4`, 100 epochs, `train.initialization_strategy=buddies`. **Trained arm only** — the freeze-vs-trainable crossing (C9's `em_interval` axis) originally in the spec's first draft was dropped by explicit user decision before launch, narrowing this experiment to "does K matter for the training result," not "does the freeze/trainable asymmetry hold across K."

K is a template-compatibility key (`src/hook/train_cosir.py`'s buddy-init `_extra` guard, alongside `alpha`/`method`): a template built at one K is rejected and rebuilt at another, so — exactly like `initialization_strategy` in the project's existing `run_init_ablation.sh` — K is a bash-loop axis, each value getting its own `results_dir`/`template_embeddings/`, with seed as the inner Hydra multirun axis (`seed=1,2,3`) reusing that K's template.

### Reuse decision: 150k yes, 300k no

150k needed zero new runs — its shortlist is just the K=30 anchor, already covered by C5's completed trained cell. **300k's existing K=30 cell (C6) was checked and rejected as a reuse candidate**: it was trained via `scripts/run_init_ablation_redcaps_300k.sh`, which points at the *plain* `redcaps_300k` feature store (15/28-of-350 subreddit coverage, per `docs/reports/2026-08-19_buddy_init_ablation_redcaps_300k.md`), not `redcaps_300k_diverse` — the same plain-vs-diverse mismatch Stage A had to correct for its own diagnostic, recurring here at the training-reuse layer. Reusing C6 would have silently confounded the K effect with a subreddit-coverage difference. All 300k and 500k Stage-B cells are therefore fresh runs on the `_diverse` stores.

### Scope reduction: 16 of 18 planned runs

Both new launcher scripts (`run_buddy_k_ablation_redcaps_{300k,500k}.sh`) were SMOKE-tested locally (2 epochs, seed=1, K=30) before the full sweep, confirming correct `_diverse` paths and results-dir wiring. The full sweep then ran sequentially on a single local GPU (no SLURM cluster automation — explored but dropped for lack of a quick automation path this session), starting 2026-09-01 15:14 and running for roughly 9 hours. Once 300k's full `{30,35,50}`×3-seed grid and 500k's `{30,39}`×3-seed grid were complete and 500k/K=50's first seed had just finished (16 runs total), the user judged the result already clear enough ("we have enough on that one") and stopped the sweep rather than running the remaining 2 seeds. A watcher polling the run log killed the process the instant the 16th run's completion line appeared, catching Hydra's multirun before it did any real work on seed=2 (only a CLIP model re-download — no `results_dir`, no logged wandb data; `analyze_buddy_k_ablation.py`'s finished-state filter excludes the resulting empty run automatically). **500k/K=50 is therefore n=1** — reported below as a single directional data point, not a 3-seed paired claim under this plan's §5 methodology standard.

### Metrics

`test_oracle`/`test_pre_diff` t2i/i2t R1 (`scripts/analyze_buddy_k_ablation.py`, mirroring `analyze_init_ablation.py`'s pattern): per-(N,K) mean ± std and mean/SEM across seeds, plus the paired within-seed delta from that N's own K=30 cell (K=30 is the common comparison point present at every N).

---

## Results

**Commands run:**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SMOKE=1 K_SWEEP="30" bash scripts/run_buddy_k_ablation_redcaps_300k.sh   # pipeline sanity check
SMOKE=1 K_SWEEP="30" bash scripts/run_buddy_k_ablation_redcaps_500k.sh   # pipeline sanity check
bash scripts/run_buddy_k_ablation_redcaps_300k.sh                       # full sweep (300k), then 500k — stopped at 16/18 runs
python scripts/analyze_buddy_k_ablation.py --tag buddy-k-ablation-redcaps_300k_diverse
python scripts/analyze_buddy_k_ablation.py --tag buddy-k-ablation-redcaps_500k_diverse
```

### 300k (`redcaps_300k_diverse`), `test_oracle`, all cells n=3

| K | t2i R1 mean | i2t R1 mean | t2i Δ vs K=30 | mean/SEM | i2t Δ vs K=30 | mean/SEM |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 26.47 ± 0.12 | 11.67 ± 0.29 | — | — | — | — |
| 35 | 26.67 ± 0.25 | 11.37 ± 0.40 | +0.20 | +2.0 | −0.30 | −1.0 |
| 50 | 26.80 ± 0.10 | 12.17 ± 0.45 | +0.33 | **+2.8** | +0.50 | +1.2 |

### 500k (`redcaps_500k_diverse`), `test_oracle`

| K | t2i R1 mean (n) | i2t R1 mean (n) | t2i Δ vs K=30 | mean/SEM | i2t Δ vs K=30 | mean/SEM |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 26.63 ± 0.06 (3) | 10.60 ± 0.30 (3) | — | — | — | — |
| 39 | 26.73 ± 0.21 (3) | 11.57 ± 0.32 (3) | +0.10 | +0.9 | **+0.97** | **+14.5** |
| 50 | 27.10 (1) | 11.00 (1) | +0.50 (n=1) | — | +0.70 (n=1) | — |

*(`test_pre_diff` t2i/i2t R1 — the same metrics relative to raw CLIP rather than to K=30 — were also collected and show no additional signal beyond what's summarized above; omitted from the table for space, full numbers reproducible via the commands above.)*

### Reading the two scales together

**500k/K=39 is the one clean, well-supported result: a +0.97 i2t R1 gain over K=30, mean/SEM=+14.5, far past both the significance bar and the noise floor.** This is exactly the K(N)-invariant-matched prediction from Stage A (16.1 predicted K≈38.99, rounded to 39 for training) — not the K=30 anchor everyone else in this project's RedCaps work has used, and not the K=50 bracket point either. K=30 is measurably leaving retrieval on the table at 500k, specifically on the harder i2t direction, and the structural K(N) rule found the fix.

That said, the effect is **direction- and scale-specific, not a uniform "bigger K is better" story**:
- 500k/t2i shows essentially nothing at K=39 (Δ+0.10, mean/SEM+0.9) — the win is i2t-only at this scale.
- 300k shows the opposite pattern in miniature: a small **t2i-only** signal (K=35 borderline at the mean/SEM≥2 bar, K=50 clearing it at +2.8, but both magnitudes — 0.20/0.33 R1 — sitting at the noise floor's low end), with i2t flat-to-noisy (−1.0 to +1.2 mean/SEM).
- 500k/K=50's single-seed numbers (t2i +0.50, i2t +0.70) point in the same helpful direction as K=39, suggestive that the effect may continue to grow past the invariant-matched prediction, but with n=1 this cannot be treated as evidence — only as a reason a full 3-seed K=50 cell would be worth completing later.

No training-time buddy structural stats (edge composition, strict-degree) were re-examined against these deltas in this pass — Stage A's structural numbers explain *why* the K(N) prediction is where it is, but explicitly cross-referencing which structural property predicts the i2t-specific 500k effect (vs. the t2i-specific 300k effect) is unresolved and would need its own look.

---

## Caveats

- **300k and 500k tell different-shaped stories, not a single consistent K effect.** Do not generalize "K should be increased for i2t" or "K should be increased for t2i" from this report alone — at 300k the (small) win is t2i, at 500k it's i2t, and neither scale shows both. Any paper claim should state the effect per-(N, metric) cell, not as a blanket K-helps-retrieval statement.
- **500k/K=50 is not evidence.** Its single seed is reported for direction only; treating +0.50/+0.70 as comparable to the 3-seed cells' mean/SEM figures would violate this plan's own §5 standard (≥3 seeds per claim). A future session could cheaply complete K=50's remaining 2 seeds if this cell turns out to matter for the paper's final K(N) story.
- **The freeze-vs-trainable question from the original 16.2 design was dropped, not answered.** This report only speaks to "does K matter under trained conditions"; whether C9's freeze-vs-trainable i2t asymmetry holds, weakens, or strengthens across K remains open and out of scope here.
- **`_diverse` stores are not identical in kind to the plain training-prefix stores** used by this project's earlier C5/C6 training ablations (independently-drawn all-350-subreddit sample vs. an arbitrary training-file prefix) — this report's 300k/500k numbers are not a strict apples-to-apples continuation of C5/C6, by design (see Method's reuse-decision section), but this means any direct numeric comparison against pre-16.1 300k/500k results should account for the store difference, not just the K difference.
- **No structural cross-reference performed yet.** Stage A's per-cell structural stats (strict-degree, union composition) were not formally correlated against these retrieval deltas in this pass; the *why* behind the i2t-specific 500k win vs. the t2i-specific 300k win is not explained here.

---

## Reproduce

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR

# Pipeline sanity check (2 epochs, seed=1, K=30) before committing GPU time to a full sweep:
SMOKE=1 K_SWEEP="30" bash scripts/run_buddy_k_ablation_redcaps_300k.sh
SMOKE=1 K_SWEEP="30" bash scripts/run_buddy_k_ablation_redcaps_500k.sh

# Full sweep (100 epochs, 3 seeds per K) — this report's 500k/K=50 stopped after 1 seed:
K_SWEEP="30 35 50" bash scripts/run_buddy_k_ablation_redcaps_300k.sh
K_SWEEP="30 39 50" bash scripts/run_buddy_k_ablation_redcaps_500k.sh

# Analysis (paired mean/SEM vs. K=30, per scale):
python scripts/analyze_buddy_k_ablation.py --tag buddy-k-ablation-redcaps_300k_diverse
python scripts/analyze_buddy_k_ablation.py --tag buddy-k-ablation-redcaps_500k_diverse
```

Each K value gets its own `results_dir`/template under `res/CoSiR_buddy_k_ablation/<dataset>/k_<K>/` (K is a template-compatibility key, so different K's cannot safely share a template — see Method). Wall-clock for the 16 completed runs in this report was approximately 9 hours on a single local GPU, sequential (no cluster parallelism was set up this session).
