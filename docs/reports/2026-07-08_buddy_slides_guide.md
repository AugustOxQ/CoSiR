# Slide guide — Conditional Buddies: what worked, what didn't, and why

**Purpose:** a ready-to-build storyboard for the meeting deck. One slide per page below;
each gives the **layout**, the **left-column text** (speaker-ready bullets), and the
**right-column figure** (path + what it shows). Figures are tagged **[EXISTS]** (already in
`docs/reports/assets/`) or **[READY]** (freshly generated into `docs/reports/assets/slides/`
by `scripts/make_slide_figs.py` — regenerate with `python scripts/make_slide_figs.py` if the
final numbers shift). All four new result figures are already rendered and verified.

Suggested deck length: **13 content slides in two acts** (+ title). Aim ~40 sec/slide →
~9–10 min segment (trim Part I if you're tight — Slides 2–3 can merge).

- **Part I — Where buddies came from (origin & validation)**, Slides 1–6: the idea, the graph,
  the initialization, and the evidence that the buddy *signal* is real, content-specific, and
  generalizes across datasets *and* across 6 held-out encoders.
- **Part II — Using buddies during training**, Slides 7–13: the three families, the one that
  works, and why its win doesn't cleanly transfer.

The full arc: *the idea → the signal is real & generalizes → but using it in training only
partly helps → and the Impressions win was largely a near-duplicate artifact → what's next.*
Lead with the nuance; don't oversell. The deliberate tension: Part I shows the per-edge signal
generalizes off near-duplicates; Part II shows the training-term *benefit* did not — because
the term used the same-photo edges Part I's controls had removed. Keep that distinction crisp.

Figures live under `docs/reports/assets/` (existing, from the init/validation work) and
`docs/reports/assets/slides/` (new §8 results). Source numbers: the progress report
`docs/reports/2026-06-24_buddy_progress_report.md` (§1–4 for Part I, §8 for Part II).

---

## Slide 0 — Title

- **Layout:** centered title, no figure.
- **Title:** *Conditional Buddies during training — a dose-dependent win, and a near-duplicate confound*
- **Subtitle:** one line — "Only contrastive supervision (#2) moves retrieval on Impressions; it does not cleanly transfer to a 1:1 dataset, and we measured why."
- **Footer:** your name · date · branch `experiment/conditional_buddy_train_family3`

---

# Part I — Where buddies came from (origin & validation)

## Slide 1 — The problem: condition vectors started from nowhere (30 sec)

- **Layout:** text LEFT · figure RIGHT.
- **Left text:**
  - CoSiR fuses frozen CLIP image/text features with a per-sample **trainable** condition vector `z` (D=16) — the only per-sample knob, and it needs an initial value.
  - Old init (`imgtxt`): `z` from a generic `(image − text)` PCA transform — **no notion of which samples are related**; conditions scattered arbitrarily.
  - **Hypothesis:** if samples sharing cross-modal context *start close* in condition space, training has an easier landscape. So seed `z` from the data's own neighborhood structure.
- **Right figure:** `assets/fixed_rank.png` **[EXISTS]** — the buddy-seeded condition space (structured spread) vs an arbitrary generic init. (Or a hand-drawn before/after scatter schematic. **[optional]**)

---

## Slide 2 — What is a "buddy"? (40 sec)

- **Layout:** text LEFT · figure/schematic RIGHT.
- **Left text:**
  - A **buddy** = a pair that are mutual K-NN (K=30) in **both** CLIP image space *and* CLIP text space. Mutuality drops hub samples that don't point back.
  - Two graphs: **B = img ∩ txt** (strict, precise, sparse) · **E = img ∪ txt** (broad, well-connected). **E is used for init** (few isolated nodes).
  - Edges weighted by rank-normalized cosine distance, image/text mixed at **α=0.5**.
  - Connectivity fix `ensure_connected`: bridge E's disconnected components via a medoid MST (RedCaps: 54 components → 1) — else the spectral init wastes dimensions on "which component am I in?".
- **Right figure:** `assets/slides/buddy_venn.png` **[READY]** — B = A_img ∩ A_txt (strict lens) vs E = A_img ∪ A_txt (union, used for init). Alt: `assets/buddy_degree_hist.png` **[EXISTS]** (E's degree distribution / connectivity).

---

## Slide 3 — How the init works: Laplacian Eigenmaps (50 sec)

- **Layout:** figure TOP or RIGHT (wide 2-panel — give it room) · text below/left.
- **Left text (walk the figure L→R):**
  - **Input = just the graph E.** Who is whose buddy — *no coordinates yet* (left panel: a coordinate-free hairball).
  - **Goal:** give every node a position so that **graph-neighbors land close**. Formally, find coordinates `z` minimizing the *smoothness energy*
    > `Σ_(i,j)∈E  w_ij · ‖z_i − z_j‖²`
    — "connected nodes should have similar coordinates" — subject to a spread/normalization constraint (otherwise everything collapses to a single point).
  - **How it's solved:** that objective is an **eigenvalue problem on the graph Laplacian** `L = D − W`. The smoothest layouts are `L`'s lowest non-trivial eigenvectors; we discard the trivial constant (0th) one and take the next **16** → the `[N, 16]` initialization (matches the model's `embedding_dim`).
  - **Result (right panel):** related samples fall into a **smooth manifold** — clusters at the corners, neighbors adjacent — rather than scattering. That structured layout is written into the condition store as the `buddies` init. Computed once per dataset, cached as a template.
  - *(Two practical fixes made it usable at scale — rank-norm vs a numerical collapse, and a `pyamg` solver vs a stalled eigensolver on 150k nodes; details in report §3. Mention only if asked.)*
- **Right/top figure:** `assets/slides/laplacian_eigenmaps.png` **[READY]** — left: graph E as a coordinate-free hairball; right: its Laplacian-Eigenmaps embedding, neighbors close, communities separating. Header carries the smoothness-energy equation.

---

## Slide 4 — Is the signal real? (Impressions, 45 sec)

- **Layout:** text LEFT · figure RIGHT.
- **Left text:**
  - The confound, handled up front: 814 photos behind 12k records → every metric split into within- vs **cross-photo** edges.
  - Three independent probes all agree the signal is **content-specific**:
    - **Type lift:** 1.5× overall → **2–3× on cross-photo** edges (signal gets *stronger* once same-photo pairs are removed).
    - **Held-out DINOv2** (an encoder the graph never saw): buddies **0.39 (B) / 0.65 (E)** vs **0.95 random**.
    - **VLM judge:** a strict buddy's caption describes the anchor image **74 %** of the time vs ~1 % random.
  - Also: **n_dim=16** is the sweet spot (kNN preservation 6.7 % @2-D → 72 % @16-D).
- **Right figure:** `assets/buddy_analysis/phase2_vlm.png` **[EXISTS]** (VLM 74 %) — or `assets/buddy_analysis/identity_heldout.png` (DINOv2 held-out).

---

## Slide 5 — Does it generalize off near-duplicates? (RedCaps, 40 sec)

- **Layout:** text LEFT · figure RIGHT.
- **Left text:**
  - RedCaps is genuinely **1 image : 1 caption**, 350 subreddits — every buddy edge cross-content by construction.
  - **Subreddit lift ~20×** (order of magnitude above Impressions); DINOv2 **0.39 / 0.59** vs 0.97; VLM **81 %** vs **7 %** for a *same-subreddit* hard negative vs 1 % random — a buddy ≫ same-topic ≫ random gradient = **specific** content, not just broad topic.
  - **Verdict: the buddy *signal* is not a near-duplicate artifact — it generalizes.**
  - ⚠️ **Foreshadow (sets up Part II):** that's the *per-edge signal*, measured on cross-photo/clean edges. Whether *using* buddies as a **training term** helps retrieval is a separate question — and the answer turns out to be more nuanced.
- **Right figure:** `assets/redcaps_buddy/lift_and_dino.png` **[EXISTS]**.

---

## Slide 6 — Not a CLIP / encoder / modality artifact: the held-out grid (45 sec)

- **Layout:** figure TOP or RIGHT (wide 2-panel grid) · text below/left.
- **Left text:**
  - Toughest validation: score buddies with **6 encoders that never built the graph** — 3 vision (DINOv2 self-sup · SigLIP VLM · ImageNet-supervised ViT) + 3 language (MiniLM · BGE · E5) — on **both datasets**, **both graphs** = **24 cells**.
  - **All 24 cells: buddies closer than random** (ratio < 1, every one). Vision separates harder (mean B ratio ~0.38–0.40) than text (~0.48–0.51), but all unambiguous.
  - Kills three "artifact" worries at once:
    - not a **CLIP** artifact — 3 unrelated vision paradigms agree;
    - not a **modality** artifact — independent *language* models confirm captions too;
    - not a **near-duplicate** artifact — RedCaps (1:1) is if anything *tighter* (mean vision E 0.58 vs Impressions 0.66).
  - Recurring theme: **B ≪ E in every cell** — the strict graph is the cleaner signal (→ motivates B-lean, Slide 12).
- **Right figure:** `assets/heldout_grid/impressions_grid.png` **[EXISTS]** (primary; 2 panels B | E, red=vision / blue=text buddy vs grey random) — pair with `assets/heldout_grid/redcaps_grid.png` **[EXISTS]** as the "even cleaner on genuinely 1:1 data" companion.

---

# Part II — Using buddies during training

## Slide 7 — Three ways to use buddies in training (30 sec)

- **Layout:** figure TOP (wide 3-panel) · text below — or text LEFT / figure RIGHT.
- **Left text:**
  - **#1 Smoothness** — keep buddies' `z` close (regularizer on condition space).
  - **#2 Contrastive** — buddies as extra retrieval positives in **combined/retrieval space** (where R1 lives).
  - **#3 Self-refresh** — rebuild the graph from the model's *own* evolving features.
  - All three: implemented, unit-tested, gated default-off; share **one** buddy init → clean ablation.
- **Figure:** `assets/slides/training_families.png` **[READY]** — 3 panels: #1 pull buddies' `z` together (z-space); #2 InfoNCE pulls buddy images to the anchor / pushes others (retrieval space); #3 frozen CLIP graph → refreshed model graph. Neutral colors on purpose (the "#2 wins" reveal is Slide 7).

---

## Slide 8 — Only #2 survives replication (45 sec) — **the pivot slide**

- **Layout:** the **table is the slide** (full-width); 2–3 lines of setup/takeaway above or below.
- **Setup line:** single-seed grid put all three near the **noise floor** (~0.1–0.7 R1); replicating across 3 seeds and reading the **paired** Δ resolves it.
- **Table** — `assets/slides/seed_replication_table.png` **[READY]** (rendered, drop-in), or build it natively from this:

  | Family | Metric | seed 1 | seed 2 | seed 3 | mean Δ ± std | mean/SEM | verdict |
  | ------ | :----: | :----: | :----: | :----: | :----------: | :------: | :-----: |
  | **#1 Smoothness** | t2i | −0.30 | +0.10 | −0.10 | −0.10 ± 0.20 | −0.9 | null |
  | **#1 Smoothness** | i2t | +0.00 | +1.70 | +1.00 | +0.90 ± 0.85 | +1.8 | n.s. |
  | **#2 Contrastive** | t2i | +0.90 | +1.50 | +1.30 | **+1.23 ± 0.31** | **+7.0** ∗ | **WIN** |
  | **#2 Contrastive** | i2t | +1.90 | +0.30 | +1.00 | **+1.07 ± 0.80** | **+2.3** ∗ | **WIN** |

  *#3 self-refresh: grid-confirmed null (blend 0 vs 1: t2i −0.03, i2t +0.03) — dropped from replication.*
- **Takeaway line (say it):** *replication demoted two families and promoted one — #2 is 3/3 seeds positive in both directions, `mean/SEM` up to 7; #1 is noise, #3 is null.*
- **Alt figure:** `assets/slides/seed_replication.png` **[READY]** — the bar version (mean Δ ± SEM per family), if you prefer a chart to a table.

---

## Slide 9 — #2 is dose-dependent, peaks at λ_con = 1.0 (45 sec)

- **Layout:** text LEFT · figure RIGHT (the figure carries this slide).
- **Left text:**
  - Sweeping `λ_con`: **monotonic up to 1.0**, then **rolls over into harm**.
  - Peak `λ_con=1.0`: **+2.3 R1 t2i / +3.2 R1 i2t** (`mean/SEM ≈ 40`).
  - `λ_con=4` significantly **hurts** i2t (−1.8) — the term overwhelms the retrieval loss.
  - Tell: `alignment` & `buddy_preservation` keep *rising* past the peak while R1 *falls* → the aux objective "succeeds" by distorting retrieval geometry. Preservation is a mechanism indicator, **not** a thing to maximize.
- **Right figure:** `assets/slides/peak_curve.png` **[READY]** — x=`λ_con` {0.3,0.5,1.0,2.0,4.0}, two lines Δt2i & Δi2t, horizontal zero line, vertical marker at peak=1.0, shaded "harm" region past 1.0.

---

## Slide 10 — The critical test: does it generalize? (45 sec)

- **Layout:** text LEFT · figure RIGHT.
- **Left text:**
  - Impressions has **near-duplicate structure** (814 photos → 12k records). Does the win depend on it?
  - **RedCaps-150k**: genuinely **1 image : 1 caption**, no near-dups. Transplant `λ_con=1.0`.
  - Result — **it does not transfer:**
    - t2i **+0.4** (clean but tiny), i2t **−0.9** (a *significant loss*).
  - Yet the term is **provably active** (alignment 0.41; preservation move *larger* than on Impressions) — it acts, it just no longer helps.
- **Right figure:** `assets/slides/transfer_bars.png` **[READY]** — grouped bars: Impressions vs RedCaps, Δt2i & Δi2t at λ_con=1.0; Impressions bars tall + green, RedCaps t2i tiny, RedCaps i2t below zero + red.

---

## Slide 11 — Why: 40.6 % of buddy edges are the *same photo* (45 sec) — **the payoff slide**

- **Layout:** text LEFT · figure RIGHT.
- **Left text:**
  - Measured the **actual training edge set** (`buddy_edges.npy` = E union):
    - **E: 40.6 %** of edges are the *same source photo* → **279× enriched** over chance (0.145 %).
    - **B: 79.9 %** same-photo → **550×**.
    - **RedCaps: 0 %** (unique `image_id`, by construction).
  - So ~2 in 5 edges the term tightens are *literally the same photo* — trivially-correct neighbors it gets "for free."
  - **Reconciles with Slide 5** (say this out loud): the per-edge *signal* genuinely generalizes — but validation measured it on *cross-photo* edges, whereas the training term used the **full** graph. Its retrieval benefit rode on exactly the same-photo edges validation had controlled away. Both results are true; they're about different things.
  - **Headline:** the Impressions win was largely the term exploiting **dataset redundancy**, not transferable image–text signal.
- **Right figure:** `assets/slides/nearup_enrichment.png` **[READY]** — bar of within-same-photo % for E (40.6), B (79.9), RedCaps (0); annotate 279×/550×/0; dashed line at random 0.145 %.

---

## Slide 12 — Cross-thread caution: B-lean init (30 sec, optional)

- **Layout:** text LEFT · figure RIGHT (reuse existing).
- **Left text:**
  - Held-out grid found **B "cleaner" than E** → motivated the new B-lean init (`b_weight`).
  - But **B is 79.9 % same-photo** — part of what makes B "clean" is that it's *more* near-dup-dominated.
  - On Impressions B-lean will likely look good (more free near-dup tightening); on 1:1 data it has nothing to lean on.
  - **Action:** validate B-lean on RedCaps *before* crediting it as a general improvement.
- **Right figure:** reuse `assets/heldout_grid/impressions_grid.png` from Slide 6 (the B≫E grid — visibly lower red/blue buddy bars on B than E) — or the Slide 11 enrichment bar highlighting the B column.

---

## Slide 13 — Takeaways & next steps (30 sec)

- **Layout:** text full-width, 2 columns (Findings | Next), no figure.
- **Findings:**
  - #1 smoothness & #3 self-refresh: **null**.
  - #2 contrastive: **works on Impressions** (dose-dependent, peak +2.3/+3.2), **does not cleanly transfer** to 1:1 data.
  - Quantified cause: **40.6 % same-photo buddy edges** (279×) — near-duplicate artifact.
- **Next:**
  - Decide #2's fate on clean data: a **gentler retuned `λ_con` ∈ {0,0.1,0.3}** on RedCaps — net-positive both directions, or not the lever?
  - Re-scope **B-lean** — validate off near-dups.
  - Fallback: if #2 doesn't survive, buddies remain a **validated init-only** contribution.

---

## Appendix — the eight generated figures

All eight **[READY]** figures are already rendered into `docs/reports/assets/slides/` by
`scripts/make_slide_figs.py` (result numbers hard-coded from report §8; Slides 2–3, 6 are
synthetic schematics — no wandb needed):

| file | slide | shows |
| ---- | ----- | ----- |
| `buddy_venn.png` | 2 | B = img ∩ txt (strict) vs E = img ∪ txt (union, used for init) |
| `laplacian_eigenmaps.png` | 3 | coordinate-free graph → LE embedding (neighbors land close) |
| `training_families.png` | 7 | the three training-time uses: z-smoothness / contrastive / self-refresh |
| `seed_replication_table.png` | 8 | per-seed paired ΔR1 table — #1/#2 detail, verdicts (primary) |
| `seed_replication.png` | 8 (alt) | mean Δ R1 ± SEM per family bar chart — only #2 above zero |
| `peak_curve.png` | 8 | Δ vs `λ_con`, peak at 1.0, harm zone shaded past it |
| `transfer_bars.png` | 9 | Impressions vs RedCaps Δ — tall green vs tiny/negative |
| `nearup_enrichment.png` | 10 | within-same-photo %: E 40.6 / B 79.9 / RedCaps 0 (279×/550×) |

**To regenerate** (e.g. if the last RedCaps run nudges the numbers): edit the tuples at the top
of each function in `scripts/make_slide_figs.py`, then `python scripts/make_slide_figs.py`.
Style: green = positive, red = negative, grey = neutral, zero line always drawn, log-x on the
dose curve. Export tips: PNGs are 140 dpi (fine for projected slides); for print/PDF bump
`figure.dpi` to 200+ or add `fig.savefig(..., format="pdf")`.
