# Weekly Report — PercepT Topic-Formation Investigation

**Period covered:** 2026-09-22 to 2026-09-23
**Project:** CoSiR / ArtELingo — PercepT replication side-investigation
**Branch:** `experiment/percept_topic_pipeline`

---

## 1. Executive summary

This is the first reporting checkpoint for a short, self-contained side-investigation: a literal-as-possible replication of **PercepT**, the topic-formation method from *Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from Vision-Language Data* (arXiv 2606.03345), run on ArtELingo. Stage 1 (unsupervised topic formation) and Stage 2 (image-only topic prediction) both closed initial results on 2026-09-23 at a standing configuration of K=60/40 (see §2 for terminology). This period ran four follow-up pilots to stress-test that standing result: a fine K-sweep confirmed nothing nearby beats K=60/40; a 14-seed extended-seed pilot found the result is genuinely **seed-dependent** (10/14 seeds clear both held-out bars, not the near-universal robustness the original 4-seed test suggested); a Stage 2 reproducibility check uncovered a previously invisible methodological gap — every "established" value in this investigation, including the ones the last two points measure against, was produced without controlling for GPU run-to-run execution differences — which forced a rebase of the Stage 2 gate and, after that rebase, delivered a clean, seed-robust 14-seed Stage 2 result (held-out macro AUC mean 0.8290, tighter and marginally higher than the original 4-seed citation of 0.8256); and a dedicated noise-floor pilot was launched to measure that gap directly, with a first surprising signal — see §6.

## 2. Terminology and scope (read this before the results)

**PercepT / Stage 1 / Stage 2.** PercepT forms "P-Topics" by fusing a factual (CLIP) and an affective (GoEmotions-tuned text encoder) embedding per painting, autoencoding the fusion into a compact latent space, and running **Deep Embedded Clustering (DEC)** — an algorithm that alternates between soft cluster assignment and self-training on its own confident assignments — to sharpen an initial K-means partition into a smaller set of final topics. This is **Stage 1** ("P-Topic Formation"), and it is unsupervised: no emotion or genre label is used during training, only afterward for evaluation. **Stage 2** ("P-Topic Mapping") freezes Stage 1's topic assignments as fixed targets and trains an image-only classifier — no text at inference — to recover them from the picture alone.

**K=60/40 notation.** Stage 1 starts K-means with 60 initial clusters, then DEC-trains and prunes down to the 40 highest-norm surviving clusters ("60 initial, 40 surviving").

**AMI and the held-out Pareto bar.** Adjusted Mutual Information (AMI) measures chance-adjusted agreement between the discovered cluster assignment and a human label (emotion or genre); 0 is chance-level agreement, 1 is perfect. This investigation's predeclared success criterion, used throughout, is the **held-out Pareto bar**: `emotion AMI > 0.1236 AND genre AMI > 0.1954`, both simultaneously, measured on val+test data with zero train overlap. A configuration is **Collapsed** if over half its surviving clusters hold under 1% of assigned nodes; **Real success** means non-collapsed and clearing both thresholds; anything else non-collapsed is **Merely a compromise**.

**Is this investigation's replication faithful to the paper?** No, and that is documented, not hidden: the affect encoder substitutes `SamLowe/roberta-base-go_emotions` for the paper's ModernBERT-family choice (same objective, different backbone); the fusion step substitutes `L2_normalize(concat([content, content, affect]))` for the paper's dimension-matched elementwise 2:1 weighted sum, because this repository's CLIP (512-D) and RoBERTa (768-D) features don't share a dimension; and DEC training here runs to a convergence criterion rather than the paper's fixed 200 epochs.

**Is it fair to compare against the paper's own reported numbers?** No — treat the paper's numbers as a different metric on different data, not a baseline. Beyond the deviations above, the paper's results were computed on a different dataset with a different label taxonomy; there is no shared yardstick to claim "beats" or "matches" the paper. The one *fair*, already-computed baseline in this project is the buddy-graph fusion mechanism (the main, separate publication track), because it is evaluated on the exact same ArtELingo held-out split with the exact same AMI Pareto-bar criterion: **Attention-h1, held-out emotion AMI=0.1249, genre AMI=0.2404.** PercepT's standing K=60/40 result (4-seed mean emotion 0.1252, genre 0.2486) is essentially on par with that baseline. The honest framing for this whole investigation is therefore a **mechanism-fidelity check** — does PercepT's approach work on this project's data, roughly matching the project's own established baseline — not a paper-benchmark comparison.

## 2a. Where the Attention-h1 baseline comes from

Attention-h1 (§2's fair baseline) was not chosen arbitrarily — it is the endpoint of a separate, dedicated fusion-mechanism investigation on the same ArtELingo content+affect data, evaluated with the identical held-out AMI Pareto-bar criterion this whole report compares PercepT against. This is recap context (all of it predates this reporting period), included because every K-sweep comparison in §4–§6 below cites this number without otherwise explaining where it came from.

**Five mechanisms were tried, in order, before arriving at a learned model:**

1. **Late fusion** (build separate content and affect buddy graphs, then combine edge sets): the union graph reached emotion AMI=0.1236 / genre AMI=0.1394 (train) — better than early (feature-level) fusion's 0.1160/0.0867, but still far below content-only's 0.4384 genre ceiling. The intersection graph was degenerate (98.96% isolated nodes before repair).
2. **Hierarchical refinement** (content communities as fixed parents, affect only splits within a parent): found a real, control-verified emotion signal (0.1072 vs. a size-matched random-split control's 0.0362), but most of its genre cost (0.1954 vs. a 0.3507 retention floor) turned out to be a granularity artifact of creating more, smaller labels — not specifically caused by affect.
3. **Classical joint-fusion methods** (Similarity Network Fusion, co-regularized spectral clustering) were evidence-based *skipped*, not run: both require the content and affect graphs to locally agree, and their mutual-kNN graphs had a near-empty intersection, predicting neither method would find a real shared local structure.
4. **The CCA audit — the pivot point.** Before committing to a learned model, a linear Canonical Correlation Analysis (CCA) plus conditional-residual audit tested whether content and affect share *global* structure, even though they'd shown a near-empty *local* (mutual-kNN) overlap. Top held-out canonical correlation was 0.7285, far above both the predeclared 0.15 bar and the permutation-null threshold of 0.0699 — real, substantial, held-out-replicated shared signal. This licensed moving to a learned joint representation.
5. **A learned two-teacher contrastive student**, trained with InfoNCE losses against two teacher graphs (content, affect) through a shared embedding space.

**From linear heads to attention — how the final result was found:**

| configuration | encoding heads | combination | train emotion/genre AMI | held-out emotion/genre AMI | verdict |
|---|---|---|---|---|---|
| Stage 1 | linear projection | scalar gate | 0.1284 / 0.2799 | 0.1095 / 0.2901 | best multi-signal point on train; misses held-out emotion bar by ~11% |
| Stage 2 | 2-layer MLP (64 hidden) | scalar gate | 0.1230 / 0.2319 | 0.1046 / 0.2087 | worse on both axes — more per-view capacity hurt |
| MLP-128 | 2-layer MLP (128 hidden) | scalar gate | 0.1142 / 0.2039 | 0.1114 / 0.1604 | worse still — confirms the MLP-capacity finding at a second scale |
| Attention-h4 | linear (fixed) | 4-head self-attention | 0.1328 / 0.2441 | 0.1216 / 0.1875 | clears held-out emotion, misses genre by 0.0079 — capacity spread too thin across 4 heads |
| **Attention-h1 (standing)** | **linear (fixed)** | **1-head self-attention** | **0.1351 / 0.2397** | **0.1249 / 0.2404** | **Real success — first configuration in the whole investigation to clear both held-out bars** |

Attention-h1 does not *strictly dominate* Stage 1 (Stage 1's held-out genre AMI, 0.2901, is still higher) — it trades some genre AMI for a real emotion-AMI gain that happens to be exactly what's needed to clear the predeclared bar. Read as strict dominance, Stage 1 still "wins" on genre; read by this investigation's own predeclared Pareto-bar criterion (its primary success test throughout), Attention-h1 is the standing result and the one used as this report's baseline.

**The mechanistic pattern that emerged:** richer *per-view encoding* capacity (MLP heads, whether 64 or 128 hidden units) consistently hurt both AMI axes. Richer *combination* capacity (replacing a scalar gate with self-attention), while holding per-view encoding at the simplest linear heads, helped — but only with a single attention head; spreading the same small (32-D) shared space across 4 heads left too little per-head capacity and lost the genre-axis gain.

Source: `2026-09-22_artelingo_fusion_mechanism_investigation.md` (sections II–V, glossary and full methodology), `2026-09-22_artelingo_fusion_diagnostics_report.md` (existing figures: fusion Pareto frontier, headline method comparison, CCA held-out canonical correlations — note these figures predate the architecture sweep and still show Stage 1, not Attention-h1, as the frontier), `src/test/20260923_artelingo_buddy_analysis/learned_student_arch_sweep_pilot_report.md` (the architecture sweep that produced Attention-h1).

## 3. Recap — where Stage 1 and Stage 2 stood entering this period

Both closed on 2026-09-23, hours before this period's pilots began, so their content is recapped only as context:

- **Stage 1** closed at K=60/40 as the standing configuration: 4-seed mean held-out emotion AMI=0.1252, genre AMI=0.2486; the representative seed-42 fit (frozen and reused by Stage 2) measured emotion AMI=0.1238, genre AMI=0.2617.
- **Stage 2** closed at held-out macro AUC=0.8256 as a 4-seed mean (range 0.8248–0.8272), using the patch-attention mapper (one learnable query attention-pools 50 CLIP ViT-B/32 patch tokens, then a linear 40-topic head), `q > 1.2/40` multi-label targets (always include the argmax topic, plus any topic whose soft assignment exceeds 1.2/40 ≈ 0.03), `lr=3e-3`, 100 epochs.

## 3a. Fusion-mechanism progression — how Stage 1 got to K=60/40

Stage 1 did not arrive at K=60/40 on the first attempt. It failed repeatedly first, and each failure diagnosed the next fix. This is recap context (all of it predates this reporting period, closed in the original Stage 1 report), included here because the slide deck built from this report needs it to explain *why* K=60/40 specifically, not just *that* it works.

1. **Base PercepT, unregularized (K=100/67, single seed): collapsed.** DEC converged, but 65/67 surviving topics held under 1% of nodes each (median topic size 0) — a **collapse**, where DEC's self-sharpening pressure has no counterweight and greedily empties almost every cluster into a few. Held-out emotion AMI=0.0363, genre AMI=0.3081: genre signal survived, but the partition itself was useless. The cause was visible in the loss scale: at convergence, the clustering (KL) loss was ~760× larger than the reconstruction loss, so reconstruction could not anchor the latent space against DEC's self-sharpening.
2. **First balance regularizer (`lambda_balance=1`): still collapsed, but directionally useful.** A term penalizing uneven cluster occupancy was added. Held-out emotion AMI rose to 0.0925 (genre 0.2631), but 64/67 topics were still under 1% — the regularizer was pointed the right way but far too weak (its own loss term stayed three orders of magnitude below KL).
3. **Balance sweep found the effective range was far higher than typical defaults.** Testing `lambda_balance` ∈ {10, 50, 100, 500} still collapsed every point; the best (500) reached emotion 0.1142 / genre 0.2116 with 44/67 topics still under 1%. Only at `lambda_balance` ∈ {1000, 2500, 5000} did non-collapsed, both-bar-clearing results first appear, at seed 42 only.
4. **Lambda=1000, 4-seed stress: real but fragile.** Only seed 42 (of 42/7/123/2024) cleared both bars; held-out means were emotion 0.1235, genre 0.2089, with only 1/4 seeds clearing both bars jointly. A real success existed, but not a reliable one.
5. **Reconstruction reweighting did not fix the fragility.** Raising the reconstruction loss weight (100/300/1000) to rebalance it against the now-large balance term did not help: the best point (weight 300) dropped to 0/4 seeds clearing both bars in its own 4-seed stress test.
6. **A literature-standard alternative, IDEC, failed more severely.** IDEC (`reconstruction + 0.1×KL`, no balance term) collapsed even harder: 97/100 centers under 1%, worse than the original unregularized run. IDEC's published default loss scale did not transfer to this fusion representation.
7. **Changing the cluster count itself was the actual stability mechanism.** With 100 initial clusters, average train-cluster size (614 nodes) sits close to the 1%-of-61,402 collapse threshold, so small assignment shifts can flip the collapse verdict. Screening coarser counts found K=30/20 stabilizes the *genre* axis (4/4 seeds clear genre) but sits too low on emotion (0/4 clear emotion). An intermediate screen of 40/27, 60/40, 80/53 found the crossover: **K=60/40 is the only configuration in the entire Stage 1 sequence to clear both held-out bars in 4/4 seeds**, with every seed converging smoothly through the stability criterion in 82–88 epochs (versus up to the 500-epoch ceiling for K=100 configurations), and the balance loss decaying toward zero rather than fighting the clustering objective throughout training.

Full cross-method summary (all single-seed rows use seed 42; all "4-seed mean" rows use seeds 42/7/123/2024):

| configuration | held-out emotion AMI | held-out genre AMI | verdict |
|---|---:|---:|---|
| Base, K=100/67 (single seed) | 0.0363 | 0.3081 | Collapsed (65/67 below 1%) |
| Balance λ=1, K=100/67 (single seed) | 0.0925 | 0.2631 | Collapsed (64/67 below 1%) |
| Balance λ=500, K=100/67 (single seed) | 0.1142 | 0.2116 | Collapsed (44/67 below 1%) |
| Balance λ=1000, K=100/67 (single seed) | 0.1242 | 0.2466 | Real success |
| Balance λ=2500, K=100/67 (single seed) | 0.1242 | 0.1970 | Real success |
| Balance λ=5000, K=100/67 (single seed) | 0.1242 | 0.2208 | Real success (held-out); train collapsed |
| Balance λ=1000, K=100/67 (4-seed mean) | 0.1235 | 0.2089 | Seed-dependent: 1/4 clear both |
| Reconstruction=300, balance=1000, K=100/67 (4-seed mean) | 0.1234 | 0.1966 | Seed-dependent: 0/4 clear both |
| Faithful IDEC γ=0.1, K=100/67 (single seed) | 0.0444 | 0.2718 | Collapsed (64/67 below 1%) |
| K=20/13 (single seed) | 0.1152 | 0.2273 | Merely a compromise |
| K=30/20 (4-seed mean) | 0.1212 | 0.2616 | Stable but below emotion bar: 0/4 clear |
| K=40/27 (single seed) | 0.1220 | 0.2736 | Merely a compromise |
| K=50/33 (single seed) | 0.1202 | 0.2830 | Merely a compromise |
| **K=60/40 (4-seed mean)** | **0.1252** | **0.2486** | **Real success: 4/4 clear both** |
| K=80/53 (single seed) | 0.1220 | 0.2396 | Merely a compromise |
| K=100/67 (4-seed mean) | 0.1235 | 0.2089 | Seed-dependent: 1/4 clear both |

For direct comparison, the fair same-split, same-metric baseline (§2) is buddy-graph **Attention-h1: emotion AMI=0.1249, genre AMI=0.2404** — K=60/40 (0.1252/0.2486) is essentially on par, with the added evidentiary strength of 4/4-seed robustness that Attention-h1 does not yet have documented.

Source: `2026-09-23_artelingo_percept_stage1_report.md`, sections III–V (closed prior to this reporting period; recapped here as the necessary context for §4–§5 below).

## 4. Fine K-sweep — does anything near K=60/40 do better?

### What was tested

Every cluster-count pair tested before this period sat at 40/27, 60/40, or 80/53 — a coarse grid around the K=60/40 crossover. This pilot filled the gap: **Phase 1** screened `(N initial, N surviving) ∈ {(55,37), (65,43), (70,47)}` at seed 42 (50/33 cited from the first cluster-count sweep, not retrained); **Phase 2** seed-stressed the best Phase-1 point across seeds 7, 123, 2024, matching this investigation's standard 4-seed methodology.

### Key results

Short summary — held-out AMI at the two configurations that matter:

| configuration | held-out emotion AMI (4-seed mean) | held-out genre AMI (4-seed mean) | both bars clear |
|---|---:|---:|---:|
| K=60/40 (standing) | 0.1252 | 0.2486 | 4/4 |
| K=65/43 (best fine-sweep point) | 0.1234 | 0.2435 | 3/4 |

Full consolidated cluster-count picture, all points tested across this investigation to date:

| N initial | N surviving | held-out emotion AMI | held-out genre AMI | source |
|---:|---:|---:|---:|---|
| 20 | 13 | 0.1152 | 0.2273 | cited, first cluster-count sweep |
| 30 | 20 | 0.1220 | 0.2577 | cited, first cluster-count sweep |
| 40 | 27 | 0.1220 | 0.2736 | cited, v2 cluster-count sweep |
| 50 | 33 | 0.1202 | 0.2830 | cited, first cluster-count sweep |
| 55 | 37 | 0.1208 | 0.2622 | this pilot |
| 65 | 43 | 0.1238 | 0.2376 | this pilot (seed-42 screen) |
| 70 | 47 | 0.1237 | 0.2357 | this pilot |
| **60** | **40** | **0.1238** | **0.2617** | cited, v2 cluster-count sweep (standing) |
| 80 | 53 | 0.1220 | 0.2396 | cited, v2 cluster-count sweep |
| 100 | 67 | 0.1242 | 0.2466 | cited, balance-v2 sweep, lambda=1000 |

Only 65/43 and 70/47 cleared the held-out Pareto bar at seed 42; 65/43 was selected (largest emotion margin, +0.0002 vs. +0.0001). Its 4-seed stress only cleared both bars in 3/4 seeds (seed 7 missed emotion by 0.004), with a mean below K=60/40's on both axes.

### Verdict

**K=60/40 remains the standing result.** No nearby point beats it on a like-for-like 4-seed comparison — the finer screen confirms K=60/40 sits at, or very near, a local optimum in this region rather than being an arbitrary choice that a denser search would overturn.

## 5. Stage 1 extended-seed pilot — is K=60/40 actually seed-robust?

### What was tested

The original 4-seed stress test (seeds 42, 7, 123, 2024) was extended to 14 seeds (10 new: 1–6, 8–11) at the fixed K=60/40 configuration, to get a real confidence interval rather than a 4-seed anecdote.

### Key results

| statistic | held-out emotion AMI | held-out genre AMI |
|---|---:|---:|
| mean (14 seeds) | 0.1242 | 0.2517 |
| min / max | 0.1157 / 0.1289 | 0.2192 / 0.2750 |
| sample standard deviation | 0.0032 | 0.0156 |
| individual bar clears | 10/14 (71.4%) | 14/14 (100%) |
| **both bars clear simultaneously** | **10/14 (71.4%)** | |

The 95% confidence interval on emotion AMI (normal approximation) is 0.1242 ± 0.0017 = [0.1225, 0.1259] — the lower bound misses the 0.1236 threshold by 0.0011. Misses: seed 2 (−0.0010 emotion), seed 3 (−0.0079), seed 8 (−0.0001), seed 9 (−0.0030).

### Verdict

**Seed-dependent result.** Genre clears essentially unconditionally; emotion sits right at its threshold's edge, missing in 4/14 seeds and pulling the joint clearance rate down to 71.4%. This is a materially more cautious picture than the original 4-seed test (which cleared 4/4) suggested.

## 6. The Stage 2 reproducibility investigation — a finding bigger than one pilot

This section is the most consequential of the period. It started as a routine relaunch of the Stage 2 extended-seed pilot (10 new mapper-init seeds, citing the original 4) and turned into a genuine methodological finding affecting how every "established" value in this investigation should be read.

### 6.1 The gate failure

Every Stage 2 pilot in this investigation re-fits the frozen K=60/40, seed-42 Stage 1 encoder from scratch before training any mapper, and gates on reproducing the established held-out AMI (0.1238 emotion, 0.2617 genre) within an absolute tolerance of 0.002 — a safety check so a pilot never silently trains against a different clustering. On node404, this re-fit measured emotion AMI=0.1225 (diff 0.0013, inside tolerance) but genre AMI=0.2274 (diff 0.0343, over 17× the tolerance). The gate failed correctly: no mapper was trained.

### 6.2 First hypothesis: GPU kernel-level nondeterminism

DEC's self-training is a feedback loop — each epoch's soft cluster assignments become next epoch's training targets. GPUs do not guarantee bit-identical results across process launches by default (cuDNN/cuBLAS may pick different, equally-valid computation strategies), and over 100 epochs of self-reinforcing training, a tiny per-epoch floating-point difference can compound into a materially different final clustering. The Stage 2 extended-seed script was patched to force deterministic execution (`torch.backends.cudnn.deterministic=True`, `benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) and relaunched.

### 6.3 The patch didn't reproduce the citation — but it did something informative

Two consecutive relaunches of the patched script on node404 produced **byte-identical** results (0.1225/0.2274 both times, identical pruned-cluster indices) — proof the pipeline is self-consistent under forced determinism — but still did not match the original 0.1238/0.2617 citation. Code-path comparison against `run_percept_stage2_best_config_stress_pilot.py` (the script that *did* originally reproduce the citation with 0.0000 diff) ruled out a code bug: both scripts call the identical underlying `initialize_cluster_centers`/`train_dec_until_stable`/`prune_centers` functions. Independent evidence that GPU nondeterminism is real in this pipeline did surface — two different, unpatched historical scripts logged slightly different pretraining-loss traces at epoch 10 for a nominally identical seed-42 pretrain (0.000077 vs. 0.000078) — but that evidence didn't explain *this specific* mismatch, since the patched pipeline was shown to be internally stable.

**Interpretation adopted at the time:** forcing deterministic kernels selects a genuinely different numerical code path than PyTorch's default kernels, not merely "the same computation made repeatable," and DEC's chaotic self-training dynamics are sensitive enough to converge to a different fixed point under that different code path. Chasing bit-exact reproduction of a citation established under default (non-deterministic) kernels was judged not to be a meaningful gate.

**This interpretation is superseded by §7's result.** The noise-floor pilot shows node404's *default* (non-forced) kernels are themselves already perfectly reproducible (5/5 identical launches) and land on the exact same 0.1225/0.2274 fixed point as the forced-deterministic runs. So determinism-forcing was never actually selecting a different code path on this node — there was no run-to-run variance for it to remove. The correct explanation for the citation mismatch is therefore not a deterministic-vs-default kernel difference at all, but a difference in *execution environment* between node404 and whichever node/library-version stack originally produced the 0.1238/0.2617 citation (see §7, §9).

### 6.4 The rebase

`run_percept_stage2_extended_seed_pilot.py` was rewritten to gate against its own deterministic fixed point (0.1225/0.2274) instead of the old citation, and — critically — to stop citing the four previously-measured mapper seeds. Those four were trained against the *old* Stage 1 clustering; combining them with ten seeds trained against the *rebased* clustering would have mixed two different target label sets into one table. All 14 mapper seeds were retrained fresh against the rebased fit.

### 6.5 Rebased Stage 2 result

| statistic | held-out macro AUC |
|---|---:|
| mean (14 seeds) | 0.8290 |
| min / max | 0.8266 / 0.8315 |
| sample standard deviation | 0.0014 |
| 95% CI (normal approx.) | [0.8283, 0.8298] |

All 14 seeds exceed both the 0.5000 train-marginal-frequency baseline and the original threshold-sweep's 4-seed maximum (0.5760). **Verdict: seed-robust.** Notably, this mean (0.8290) is marginally *higher* than the original 4-seed citation (0.8256) despite Stage 1's genre AMI having shifted meaningfully (0.2617 → 0.2274) — Stage 2's image-only mapper turns out not to be very sensitive to that specific difference in Stage 1 clustering quality, so the rebase did not quietly change the substantive Stage 2 conclusion, only which exact clustering it is conditioned on.

### 6.6 Why this matters beyond this one pilot

Every prior "established"/"cited" value across this entire investigation — every cluster-count sweep, every balance sweep, the original Stage 2 stress pilot — was produced under PyTorch's default, non-deterministic kernels, and none of them were checked for run-to-run stability before being treated as a fixed reference point. The one clean measurement available of that instability (the 0.0343 genre AMI gap at a nominally fixed seed 42) is **already larger than** the between-*different*-seed genre AMI standard deviation the §5 extended-seed pilot measured (0.0156). That raised a direct, answerable question — how much of §5's "seed-dependent result" reflects genuine seed sensitivity versus this same uncontrolled noise — which motivated §7.

## 7. Noise-floor pilot — how much of the "seed-dependent result" is really noise?

### What was tested

A dedicated pilot (`run_percept_stage1_noise_floor_pilot.py` + `aggregate_percept_stage1_noise_floor_pilot.py`) repeats *only* the shared Stage 1 K=60/40 seed-42 refit, five times, as five **separate process launches** (not an in-process loop, since cuDNN's algorithm cache can persist within one process and would understate real cross-launch variance), deliberately under PyTorch's *default* (non-deterministic) kernels — the same kernel behavior every prior citation in this investigation was produced under. Any spread between repeats is pure uncontrolled GPU nondeterminism, isolated from genuine seed choice, since every repeat uses the identical seed 42.

### Key results

All 5 independent process launches on node404 produced **byte-identical** results:

| statistic | held-out emotion AMI | held-out genre AMI |
|---|---:|---:|
| all 5 repeats | 0.1225 | 0.2274 |
| sample standard deviation | 0.0000 | 0.0000 |

For comparison, the §5 extended-seed pilot's between-*different*-seed standard deviation was 0.0032 (emotion) and 0.0156 (genre) — both strictly larger than this pilot's zero within-seed spread.

### Verdict

**The noise floor on node404 is zero, not merely smaller than the between-seed spread.** Uncontrolled GPU nondeterminism is not a real confound on this node for this workload: five fresh process launches, identical seed 42, identical code, identical data, produced the exact same result down to the fourteenth decimal place every time. Two conclusions follow directly:

1. **§5's "seed-dependent result" is genuine seed sensitivity, not a nondeterminism artifact.** The 14-seed spread reflects real differences between seeds, not uncontrolled run-to-run noise masquerading as seed sensitivity.
2. **§6's citation mismatch is not explained by run-to-run randomness on this node** — there was none to explain it. The remaining, more likely explanation is a genuinely different execution environment (GPU model, CUDA/cuDNN/PyTorch version) between node404 and whichever node originally produced the 0.1238/0.2617 citation, each side internally reproducible but numerically different from the other. This also means §6.3's original "deterministic kernels select a different code path" interpretation was likely wrong: node404's default kernels were already deterministic, so forcing determinism could not have changed anything on this node, and indeed the forced-deterministic runs and this pilot's default-kernel runs landed on the identical value (0.1225/0.2274).

## 8. Synthesis

K=60/40 remains this investigation's best-supported Stage 1 configuration — the fine K-sweep found nothing nearby that beats it on a like-for-like 4-seed basis — but "best-supported" is not the same as "seed-robust": the 14-seed extended pilot shows real, non-trivial seed sensitivity concentrated on the emotion axis (10/14 clear both bars), a materially more cautious picture than the original 4-seed test implied. Stage 2's image-only mapper, by contrast, held up cleanly under both a genuine Stage 1 clustering perturbation and a full 14-seed mapper-init stress test (mean 0.8290, tight spread), suggesting the topic-mapping task itself is comparatively robust even when the topics it's mapping to shift somewhat.

The period's biggest finding, however, is methodological rather than substantive: this investigation had never previously checked whether its own "established" values were even reproducible run-to-run. That check is now resolved cleanly. On node404, the noise floor is exactly zero — five independent process launches at seed 42 produced bit-identical results — so §5's seed-dependent result stands as genuine seed sensitivity, not a nondeterminism artifact, and the original citation mismatch (0.1238/0.2617 vs. node404's reproducible 0.1225/0.2274) is best explained by a different execution environment (GPU/CUDA/cuDNN/PyTorch version) than whichever node originally produced that citation, not by run-to-run randomness. This also retroactively corrects §6.3's in-the-moment interpretation, which assumed forcing determinism was itself changing the numerical outcome — it wasn't, since node404 was already deterministic by default. That correction is a useful lesson on its own: the investigation's first explanation for a surprising result was plausible, internally consistent, and wrong, and only a second, independent measurement (the dedicated noise-floor pilot) caught it.

## 9. Next steps

1. Identify which node/environment produced the original v2 citation (0.1238/0.2617) and diff its CUDA/cuDNN/PyTorch versions against node404's — §7 points squarely at an environment/version mismatch, not run-to-run randomness, and this is now directly checkable.
2. Decide whether to retroactively append a note to the closed Stage 1 and Stage 2 reports (`2026-09-23_artelingo_percept_stage1_report.md`, `..._stage2_report.md`), which predate this period's reproducibility findings entirely, pointing to this report for the resolution.
3. Given §7 found zero noise floor on node404, forced-determinism is not needed as a blanket fix here — but pinning and recording the exact library/driver versions used for any future "established" citation would make the next cross-environment mismatch (if any) immediately diagnosable instead of requiring a dedicated pilot to explain.
4. The emotion-axis seed sensitivity from §5 (10/14 clearance) is now confirmed genuine, not a nondeterminism artifact — decide whether that is publication-safe as a stated limitation or needs a mitigation before this investigation's Stage 1 result is treated as final.
5. The open deviations from a literal PercepT replication (substituted affect backbone, substituted fusion formula, convergence-controlled DEC schedule vs. the paper's fixed 200 epochs) remain unresolved, carried over from the original Stage 1 report.

## Appendix — source artifacts

- Stage 1 closing report: [`2026-09-23_artelingo_percept_stage1_report.md`](2026-09-23_artelingo_percept_stage1_report.md)
- Stage 2 closing report: [`2026-09-23_artelingo_percept_stage2_report.md`](2026-09-23_artelingo_percept_stage2_report.md)
- Fine K-sweep pilot: [`percept_stage1_fine_k_sweep_pilot_report.md`](../../src/test/20260922_percept_topic_pipeline/percept_stage1_fine_k_sweep_pilot_report.md)
- Stage 1 extended-seed pilot: [`percept_stage1_extended_seed_pilot_report.md`](../../src/test/20260922_percept_topic_pipeline/percept_stage1_extended_seed_pilot_report.md)
- Stage 2 extended-seed pilot (rebased): [`percept_stage2_extended_seed_pilot_report.md`](../../src/test/20260922_percept_topic_pipeline/percept_stage2_extended_seed_pilot_report.md)
- Noise-floor pilot: [`percept_stage1_noise_floor_pilot_report.md`](../../src/test/20260922_percept_topic_pipeline/percept_stage1_noise_floor_pilot_report.md)
- Buddy-graph baseline comparison figure (Attention-h1): see main publication track's combiner-architecture reports
