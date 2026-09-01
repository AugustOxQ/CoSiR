# Findings & Decisions

## Requirements

- Test whether mutual-KNN K materially matters for real training results, not just the graph-statistics-only conclusion from the earliest K/α sweep (`dim_hparam_study.py`, Impressions, no training).
- At a fixed RedCaps scale, characterize how K changes strict buddy (`A_img ∩ A_txt`) vs. union buddy (`E`) composition, and how that propagates to training result, holding freeze-vs-trainable (C9's axis) as a crossed dimension.
- Determine whether K should scale with N: given a K that works at 150k, what K is needed at 300k and 500k to preserve equivalent buddy-graph quality?
- Stage the work: cheap graph-only diagnostic first (no training), gate the expensive 3-seed training ablation on its results — same discipline this plan already applies in Experiment 11/15.



## Research Findings

- The project's only prior K sweep (`src/test/20260609_conditional_buddy/dim_hparam_study.py`) ran on Impressions, not RedCaps, was purely statistical (`knn_preservation`/`buddy_ratio`/`participation_ratio`, no training), and used a single fixed K=30 while sweeping embedding dimensionality — it never varied K against a training outcome, and its "K not very important" reading is exactly the untested prior this experiment checks.
- Every RedCaps finding since (C5 through C15) fixes K=30 as part of a project-wide "matched operating point," so no completed experiment isolates K's own effect on retrieval.
- Feature stores for all three target scales already exist and are training-ready: `/data/SSD2/pre_extract/redcaps_{150k,300k,500k}` (confirmed via `find`), so no new feature extraction is needed.
- `src/conditional_buddy/buddy_graph.py` already exposes every primitive Stage A needs unchanged: `mutual_knn(features, K, ...)`, `union_graph(A_img, A_txt)`, `ensure_min_degree(E, ...)`, and (per Experiment 15.1) `classify_edges()` for `img_only`/`txt_only`/`both`/`repair` provenance.
- C1a's subreddit-lift signal-strength metric (`src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py`) is a training-free proxy for buddy-graph "quality" already validated at all three scales (150k/197/214 of 350 subreddits covered per C1a) — reusable as-is for Stage A rather than inventing a new quality metric.
- Default K=30 in the production pipeline is set at `src/conditional_buddy/init_conditions.py:44`.
- Stage A completed its 18-cell RedCaps graph sweep on 2026-09-01 with the corrected dataset variants (`150k`, `300k_diverse`, `500k_diverse`). Strict average degree rose monotonically with K at every scale: 150k `0.089→0.216→0.371→0.729→1.253→1.839`; 300k `0.078→0.180→0.305→0.597→1.017→1.485`; 500k `0.074→0.162→0.266→0.509→0.863→1.258` for K `{10,20,30,50,75,100}`. Union average degree followed the expected near-linear K trend (150k `6.171→65.070`, 300k `6.040→64.560`, 500k `5.941→63.927`), while strict overlap decreased with N at fixed K.
- Union subreddit lift was stable and high across the whole grid (150k `22.730–22.912×`, 300k `22.735–22.833×`, 500k `22.714–22.965×`), so the structural K/N comparison is not confounded by a collapse of the subreddit signal proxy. All cells passed `union_both_edge_count == strict_edge_count` and the four-type fraction-sum check.
- Holding the 150k/K=30 strict average degree target (`0.37064`) constant gives interpolated K(N) predictions of `K=34.800` for 300k and `K=38.990` for 500k; both were in-grid and had zero monotonicity violations. The deterministic Stage-B shortlists are 300k `{30,35,50}` and 500k `{30,39,50}`. The 150k anchor remains K=30.



## Technical Decisions


| Decision                                                                                                                                   | Rationale                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Fixed absolute K grid `{10,20,30,50,75,100}` swept at all three N's, rather than a grid scaled per N                                       | The N(K) scaling question can only be read honestly off a fixed-K, varying-N comparison; a grid that shifts with N would confound the two effects.                                                                                                                                                                                                                                   |
| Stage A holds "quality" as three parallel signals (graph structure, edge-type composition, subreddit-lift) rather than picking one upfront | Keeps the diagnostic informative even if the graph-structure invariant alone turns out not to predict the training result — cross-referencing is the point of pairing 16.1 with 16.2.                                                                                                                                                                                                |
| Stage B reuses C5 (150k K=30 trained) and C6 (300k K=30 trained) cells rather than rerunning them                                          | Both already ran at the exact matched operating point 16.2 needs; rerunning would be pure waste.                                                                                                                                                                                                                                                                                     |
| K(N) candidate uses 150k/K=30 strict average degree `0.37063` as the invariant                                                             | Isotonic/PCHIP inversion of the measured fixed-K curves produces in-grid predictions K=34.800 (300k) and K=38.990 (500k); integer training candidates are 35 and 39, respectively.                                                                                                                                                                                                   |
| Stage-B shortlist is anchor + invariant prediction + opposite-side bracket                                                                 | The applied deterministic rule yields 300k `{30,35,50}` and 500k `{30,39,50}`, retaining K=30 for comparability while bracketing the predicted increase.                                                                                                                                                                                                                             |
| Stage B drops the freeze-vs-trainable crossing (trained arm only)                                                                          | User's explicit scope call, 2026-09-01, after reviewing 16.1's results. Reduces 16.2 to "does K matter for the training result" only.                                                                                                                                                                                                                                                |
| Stage B does not reuse C6's 300k K=30 trained cell                                                                                         | `scripts/run_init_ablation_redcaps_300k.sh` points at the plain `redcaps_300k` store (15/28 subreddit coverage), not `redcaps_300k_diverse` — same plain-vs-diverse mismatch 16.1 corrected, caught again at the reuse layer during Stage B scoping. All 300k cells are fresh runs on `_diverse`. C5's 150k K=30 (single-store scale, no plain/diverse split) is still reused as-is. |
| Final Stage B scope: 18 new training runs                                                                                                  | 150k: 0 new (C5 reuse covers the only shortlisted K). 300k: 9 new (`{30,35,50}`×3 seeds). 500k: 9 new (`{30,39,50}`×3 seeds). 21 logical (N,K) cells reported in total (18 new + 3 reused).                                                                                                                                                                                          |




## Issues Encountered


| Issue | Resolution |
| ----- | ---------- |




## Resources

- `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 16 — full staged design (16.1/16.2), written and approved this session.
- `docs/reports/2026-09-01_buddy_k_scaling_stage_a.md` — Stage A full write-up (18-cell table, method, caveats, reproduce steps), the canonical report for these results.


## Stage B scope reduction & results (2026-09-02)

| Decision | Rationale |
|----------|-----------|
| Stopped the sweep at 16/18 runs, not the full 18 | User's explicit call ("we have enough on that one"), once 300k K∈{30,35,50} and 500k K∈{30,39} each had their full 3 seeds and 500k K=50's first seed had just finished. A watcher killed the process the instant the 16th run's `Training Complete!` landed, catching Hydra's multirun before seed=2 of 500k/K=50 did any real work (only CLIP re-download; no results_dir, no logged data — `analyze_buddy_k_ablation.py`'s finished-state filter excludes it automatically). 500k/K=50 is therefore n=1, reportable only as a single directional data point, not a 3-seed paired claim under this plan's §5 standard. |

### Stage B Results (`scripts/analyze_buddy_k_ablation.py`)

**300k (`redcaps_300k_diverse`), test_oracle, paired Δ vs K=30 (n=3 all cells):**

| K | t2i R1 Δ | mean/SEM | i2t R1 Δ | mean/SEM |
|---|---|---|---|---|
| 35 | +0.20 | +2.0 | −0.30 | −1.0 |
| 50 | +0.33 | +2.8 | +0.50 | +1.2 |

**500k (`redcaps_500k_diverse`), test_oracle, paired Δ vs K=30:**

| K | t2i R1 Δ (n) | mean/SEM | i2t R1 Δ (n) | mean/SEM |
|---|---|---|---|---|
| 39 | +0.10 (n=3) | +0.9 | **+0.97 (n=3)** | **+14.5** |
| 50 | +0.50 (n=1, single seed) | — | +0.70 (n=1, single seed) | — |

- The clearest, best-supported signal in the whole sweep: **500k/K=39 (the K(N)-invariant-matched prediction from 16.1) moves i2t R1 by +0.97, mean/SEM=+14.5** — well past this plan's `mean/SEM ≥ 2` bar and its measured noise floor (~0.1–0.7 R1). K materially changed the training result at exactly the K(N) rule's own predicted value, not at K=30 or the K=50 bracket.
- 300k shows a smaller, borderline-to-modest t2i-only signal (K=35 mean/SEM=+2.0 right at the bar, K=50 mean/SEM=+2.8 clears it; magnitudes 0.20–0.33 R1, inside the noise floor's lower end); i2t at 300k stays flat/noise.
- 500k/t2i shows no significant K=39 effect (mean/SEM +0.9) — the significant 500k signal is i2t-specific.
- 500k/K=50's single-seed deltas (+0.50/+0.70) are directionally consistent with more K helping further but are not evidence on their own.
- Report: `docs/reports/2026-09-02_buddy_k_ablation_stage_b.md`.
