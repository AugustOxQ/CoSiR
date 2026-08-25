# Conditional Buddies — Progress Update

**From:** cross-VLM signal validation → **To:** does it actually help retrieval, at scale?

---

## What is a "conditional buddy"?

CoSiR gives every sample a trainable **condition vector** `z` (dim 16) that gets combined with frozen CLIP features. It needs a starting value. A **buddy** is a content-aware way to pick that starting value instead of a generic transform.

**Definition:** for each sample, find its mutual nearest neighbor in CLIP *image* space and its mutual nearest neighbor in CLIP *text* space (K=30 each). Two graphs result:

| Graph | Rule | Property |
|---|---|---|
| **B** (strict) | mutual-NN in **both** image AND text space | sparse, but very precise |
| **E** (union) | mutual-NN in **either** space | dense, fully connects the data — this is the one that actually seeds `z` |

Everything below asks one of two questions: **(1) is this graph meaningful?** and later **(2) does using it actually improve retrieval?**

---

## Recap: is the buddy signal real? (not a CLIP artifact)

Tested on two datasets with opposite structure: **Impressions** (12,123 records but only 814 *unique source photos* — lots of near-duplicates) and **RedCaps** (genuinely 1 image : 1 caption, no duplicates at all, 350 topical subreddits as free ground truth).

Three independent checks, each answering "do buddy pairs share real content?":

| Metric | What it measures | What "good" looks like |
|---|---|---|
| **Lift** | `P(buddy pair shares a label) / P(random pair shares that label)`. 1.0× = chance. | Impressions type lift: 1.5–3× · RedCaps subreddit lift: **~20×** |
| **Held-out encoder distance** | Feed buddy pairs through an encoder that never built the graph (DINOv2); compare buddy-pair distance to random-pair distance. Buddies closer = real, not a CLIP quirk. | Impressions cross-photo: 0.39 vs 0.95 (random) · RedCaps: 0.39 vs 0.97 |
| **VLM judge** | Ask Qwen2.5-VL (uninvolved in graph-building) "does this buddy's caption actually describe this image?" | Impressions: 74% GOOD vs 1% random · RedCaps: 81% GOOD vs 7% same-topic-random |

**Conclusion:** buddies connect samples with real shared content on both a near-duplicate-heavy dataset and a fully deduplicated one. Not a CLIP artifact, not a near-duplicate artifact.

---

## Does it survive held-out encoders? (validation, 24/24)

Prior check above used one held-out encoder (DINOv2, vision only). This widened it to **3 vision encoders** (self-supervised DINOv2, sigmoid-VLM SigLIP, supervised ViT) **× 3 language encoders** (MiniLM, BGE, E5) — none of which touched the graph — × 2 graphs (B, E) × 2 datasets = **24 cells**.

**Metric:** same held-out distance ratio as above (buddy-pair distance ÷ random-pair distance, in the held-out encoder's own space). Ratio < 1 = buddies are closer than random = signal confirmed in that representation.

| | Impressions (mean ratio) | RedCaps (mean ratio) |
|---|---:|---:|
| Vision encoders, B / E | 0.38 / 0.66 | 0.40 / 0.58 |
| Text encoders, B / E | 0.51 / 0.73 | 0.48 / 0.70 |

**Result: 24/24 cells confirm buddies closer than random.** Not a single-encoder artifact, not a single-modality artifact. Strict **B** is consistently tighter than union **E** in every cell — a recurring pattern (E buys full connectivity at the cost of noisier neighbors).

---

## Does it survive rebuilding the graph with a different VLM entirely? (the strongest test)

Everything above kept the graph fixed (built from CLIP) and only *scored* it with other encoders. This experiment **rebuilds the buddy graph from scratch** with 16 different (vision × text) encoder combinations — 4 vision towers × 4 text towers, RedCaps-150k — and asks: are the *same edges* found again?

**Metric 1 — edge-recurrence lift:** `observed Jaccard overlap between two graphs / expected overlap under random chance`. 1× = the two graphs share edges no more than random relabeling would.

**Metric 2 — consensus core:** edges that appear in **all 16** of the 16 encoder combinations — the edges that don't depend on which VLM you used at all.

| Result | Value |
|---|---|
| Median edge overlap vs chance (lift) | **B: ~176,000× · E: ~2,650×** |
| Exact edges recurring between any two arbitrary encoder pairs | ~20% |
| Core surviving all 16 combinations | 2,915 edges (B) / 174,161 edges (E) |
| Subreddit-coherence lift of that core | **12–23×** (still real content, not noise both graphs happen to share) |

**A secondary finding:** the **vision** encoder drives most of the variation — swapping the text tower barely moves agreement (Jaccard ~0.49–0.52), swapping the vision tower moves it much more (~0.13–0.41). The buddy relation is mostly anchored in image geometry.

**Conclusion: buddies are not a property of CLIP specifically — they're a property of the underlying image-text data**, recoverable from 16 very different encoder choices, with a real, semantically-coherent consensus core.

---

## Does using buddies *during* training help? (three attempts, mostly null)

Everything so far only used the buddy graph to **initialize** `z`. This tested three ways to keep using it *during* training (Impressions, then cross-checked on RedCaps):

| Family | Mechanism | Result |
|---|---|---|
| **#1 — Smoothness regularizer** | penalize `‖z_i − z_j‖²` for buddy pairs every step, keeping the init structure alive | **Null** — effect didn't survive seed replication |
| **#2 — Contrastive supervision** | treat buddies as extra positive pairs in the retrieval loss | Real win on Impressions (+2.3 t2i / +3.2 i2t R1 at peak strength) — **but see below** |
| **#3 — Self-refreshing graph** | rebuild the buddy graph periodically as `z` evolves during training | **Null** — no different from a static graph |

**The catch on #2:** transplanting the same setting to RedCaps (no near-duplicates) gave **+0.4 t2i / −0.9 i2t** — a wash-to-negative, opposite of the Impressions win. Measuring *why*: **40.6% of the Impressions buddy edges the term optimizes connect records of the same source photo** (279× enriched over chance) — i.e., much of the "win" was the term tightening near-identical images, not learning transferable retrieval structure. RedCaps has 0% same-photo edges by construction, and the win disappears there.

**Conclusion:** of three ways to exploit the signal during training, two are inert and the one that looked promising was substantially a near-duplicate artifact. This motivated going back to the simplest, cleanest use of the signal — **initialization only** (below).

---

## Does buddy-graph initialization beat generic initialization? — 150k result

The most basic possible use of the signal: does starting `z` from the buddy graph (vs. the pre-existing generic `imgtxt` init) beat it on retrieval, with every training-time buddy term (above) turned **off**?

**Metric:** paired Recall@1 delta (`buddies − imgtxt`) at matched seed, same learning rate / dim / operating point. **`mean/SEM`** is the significance read used throughout this project — the mean delta divided by its standard error across 3 seeds; `|mean/SEM| ≥ 2` is the bar for "reliably different from noise" (the measured seed-to-seed noise floor is ~0.1–0.7 R1).

| Dataset | Δ t2i R1 | Δ i2t R1 | Verdict |
|---|---:|---:|---|
| **RedCaps-150k** | **+4.00 ± 0.26** (mean/SEM +26.2) | **+4.57 ± 0.72** (mean/SEM +10.9) | clean win, 3/3 seeds, both directions |
| **Impressions** | −1.60 ± 0.62 (mean/SEM −4.4) | +3.00 ± 1.18 (mean/SEM +4.4) | **split** — i2t wins, t2i loses, both real |

**Reading:** buddy-init is a genuine, seed-replicated improvement over the generic baseline on RedCaps (both directions) and on Impressions' i2t direction — but a genuine, seed-replicated *regression* on Impressions' t2i direction. Net positive (3 of 4 cells), not a clean sweep.

---

## Does the win hold at 2× scale? — RedCaps 300k result

Repeated the RedCaps comparison at double the training data (150k → 300k rows) to check the win isn't a small-scale artifact.

| Dataset | Δ t2i R1 | Δ i2t R1 | Verdict |
|---|---:|---:|---|
| RedCaps-150k (for reference) | +4.00 ± 0.26 | +4.57 ± 0.72 | clean win |
| **RedCaps-300k** | **+7.43 ± 0.93** (mean/SEM +13.9) | **+5.00 ± 1.20** (mean/SEM +7.2) | clean win, **larger and still tight**, 3/3 seeds |

**Conclusion: the RedCaps win is not a 150k-specific fluke — it holds, if anything strengthened, at double the data.** (Impressions was not re-run at scale this round — out of scope given this week's time budget.)

---

## The catch: neither initialization beats plain CLIP retrieval

This is the sobering finding from this week, and it changes how the above should be read. `imgtxt` is a **within-model** alternative, not an external baseline — it's still our own conditioning pipeline, just seeded differently. We also have `test_raw`, the model's own logged **frozen-CLIP-embeddings, no conditioning at all** baseline, on the identical test set.

**Metric — `test_pre_diff`:** `(our conditioned model's retrieval) − (raw CLIP retrieval)`, using the model's actual deployable prediction path (a single forward pass through its condition predictor — not a cheating best-of-all-labels search). Negative = CLIP wins.

RedCaps-300k, raw CLIP baseline: **t2i R1 = 28.1, i2t R1 = 29.7.**

| `test_pre_diff` (mean, 3 seeds) | imgtxt-init | buddies-init | gap buddies closes vs imgtxt |
|---|---:|---:|---:|
| t2i R1 | −13.97 | **−3.90** | ~72% |
| i2t R1 | −16.47 | **−6.77** | ~59% |

**Every cell is negative.** Buddy-init is clearly the better of the two initializations — it closes most of the gap — but neither strategy makes the conditioning approach beat simply using frozen CLIP embeddings directly for retrieval, at this scale. This was not checked in the 150k result above; whether it holds there too is unverified (not "no" — just not yet re-checked).

**What this means:** the paper's honest claim right now is *"buddy-graph structure is a better initializer than the generic alternative"* (a real, robust, useful finding) — not yet *"this conditioning approach beats CLIP retrieval."* Those are different claims, and only the first one is currently supported.

---

## Scaling this pipeline: what it took

Getting real experiments to run past 150k (up to 300k, with 1M/3.1M validated for correctness) surfaced and fixed **five independent memory/correctness bugs** invisible at 150k scale — most notably: an OOM in the spectral-embedding step at N~3M traced to a dtype leak inside scikit-learn's own solver (fixed via a targeted, re-validated monkeypatch, not a risky reimplementation), and a "5-hour hang" that turned out to be a self-inflicted regression (an earlier OOM fix had silently moved an O(N²) computation from GPU to CPU). Also added an optional approximate-nearest-neighbor backend (cuVS) for the graph-construction step, auto-enabled above 1M samples where it's measurably faster.

*(Full detail: `docs/reports/2026-08-19_buddy_init_ablation_redcaps_300k.md`.)*

---

## Where this leaves things

- **Robustness of the signal itself (C1–C3): very strong.** Real, content-specific, survives 24 held-out-encoder cells and 16 from-scratch graph rebuilds with a semantically coherent consensus core.
- **Training-time exploitation (C4): mostly negative**, and the one apparent win was substantially a measured confound.
- **As an initializer (C5/C6): real, and the RedCaps win reproduces and strengthens at scale** — but it's a win over a weaker in-house baseline, not over raw CLIP, which both strategies still trail.
- **Framing:** still supports the primary target (TMLR, an honest rigorous-analysis paper: real robust signal, mostly-null exploitation, one real init-level win with a clear scope). Does **not** yet support a stretch-tier "decisive win" claim (ICLR/NeurIPS), which would need a result that actually beats CLIP.

---

## Next steps

- 500k RedCaps sweep was started, not finished — resume and complete.
- 1M / 3.1M real sweeps: pipeline is now validated correct and fast enough; not yet run for real (only smoke/diagnostic-scale so far).
- Check whether the CLIP-baseline shortfall also holds at 150k (not yet re-verified retroactively).
- Open question flagged earlier, still unresolved: Impressions' t2i/i2t split under buddy-init — near-duplicate structure, or a genuine trade-off?
