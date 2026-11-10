# Ablation in image retrieval: the parts that carry held clustering also carry held retrieval; two paper figures built and kept

> 2026-11-09 (sequence name 2027-01-13) · Spec: none, small loop; design in the [round log](../../../../src/test/20261109_r20_figures/20261109_r20_figures_log.md) · Decision rule: none (exploration, no bar) · Extends the held ablation of [report 2026-12-28](2026-12-28_recipe_sweep.md), "Ablation"

## Results brief

**What we tried.** Two things in one small loop (r20).

1. **Retrieval for the ablation.** The eight removals of c* had held clustering AMI only (report 12-28). We scored image-to-image retrieval for c* and the eight removals: label mAP, plain cosine, held ArtELingo (12,281 queries; genre 9,949), seeds 0 to 4. Each removal takes out one part of c*'s recipe; nothing is retuned.
2. **Two paper figures** from saved outputs: a cluster figure (c* against the plain image) and a page-1 teaser (one painting, its neighbours in each aspect).

**Outcome.** No bar. Δ is removal minus c*, in label mAP, with its 95% interval. The full table is in Results.

- **The same parts carry retrieval as clustering.** Without the instructed graphs, style changes by -0.1223 [-0.1246, -0.1200] and emotion by -0.0208 [-0.0223, -0.0195]. The buddy terms and the content correction act the same way, smaller. Without the relational term, genre changes by -0.1287 [-0.1328, -0.1241].
- **Two removals take c* back to the plain image or below.**
  - Without the instructed graphs: emotion 0.2433 and style 0.1887, against the plain image's 0.2424 and 0.1801 (c* 0.2642, 0.3110).
  - Without the relational term: genre 0.4610, against the plain image's 0.4760 (c* 0.5896).
- **Small changes** (under 0.01).
  - Without the caption endpoints, style rises by +0.0044 [+0.0034, +0.0055], as in clustering.
  - Without the sharper teacher row, genre falls by -0.0077 [-0.0099, -0.0047], as in clustering.
  - Without the direct teacher, no aspect shows a clear change (all three intervals across zero). In clustering, emotion and style rose.
- **The two tasks agree on 18 of 24 cells** (below zero, across zero or above zero). The 6 that differ are 0.0072 or smaller in both tasks.
- **Read-out check.** c*'s three values equal r11's recorded held mAP exactly.
- **Figures.** Both are built from rules applied to every candidate; you saw all candidates and kept both figures as they are. You picked the teaser painting; the cluster design is our pick, at your request. The cluster figure's class rule favours c*, and its caption says so.

**What it does not show.**

- **Other cells.** Image to image only: no text or cross-modal retrieval for the removals. ArtELingo held, Qwen only; no MMArt and no other encoder.
- **Limits of the type.** An exploration: 24 cells, no multiplicity correction. The plain-image and teacher rows are fixed embeddings with no interval.
- **Small effects.** The three small changes above are about two seed ranges or less (validation seed range: emotion 0.0014, style 0.0022, genre 0.0038).
- **The figures.** Each is a qualitative example. It shows what one rule picks; it measures nothing. The teaser's query passed a filter first (at least 3 of 5 label matches in each aspect).

**Options for next.**

a. **Retrieval in the paper's ablation table.** Add the mAP Δ column beside held AMI. Writing only, in the paper chat. It answers whether the ablation also holds in retrieval, the paper's second task.
b. **A teacher version of the figures** for the supplementary, with the all-classes figure, before 11-23. CPU minutes from saved outputs. It shows where c* and its teacher differ in clusters and neighbours.
c. **Keep the paper's ablation in clustering only.** The retrieval table stays in this report. No cost.

**Your reading.** (user, 2026-11-10) "So seems content correction and instructed graphs means a lot to style and possibly emotion, but not so much for genre. buddy term also help style and emotion. relational term seems really lower the genre. So all four parts actually contribute and no redundency."

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** Retrieval confirms the clustering ablation for every large part. It also shows that c*'s retrieval gain over the plain image rests on two parts: the instructed graphs (emotion, style) and the relational term (genre). We are confident in the large effects: intervals far from zero, five seeds, and the same pattern in two tasks. We read nothing into the small effects, and the two tasks disagree only on small cells.

Two follow-ups of report 12-28 get an answer. The direct teacher's raise in clustering does not carry over to retrieval. The caption endpoints cost a little style in both tasks. c* keeps them: on 2026-10-29 you decided that c* freezes and the variant without them is a later analysis item (after seeing our recommendation; direction log).

**Our recommendation.** Option a, then option b with the supplementary work. Option a costs no compute and answers a likely reviewer question. For the main figures, the comparison with the plain image stays, since it has the same input as c*.

</details>

## What ran

- **Stages and types:** one exploration (the retrieval check) and two descriptive figures. No spec; the design was set in chat (round log, "Decisions").
- **Data and seeds:** held ArtELingo, 12,281 paintings (genre labels on 9,949). Seeds 0 to 4 of c* and of each removal, from r10's saved held exports. The plain image and the teacher (the instructed Qwen image) are fixed embeddings. Figures: c* seed 0, held ArtELingo, paintings by artists who died in 1955 or earlier (Wikidata).
- **Code:** `src/test/20261109_r20_figures/` (commit 6e7950ed): `r20_ret.py` (Codex), `r20_figs.py` and its step modules (Sonnet worker). Figure scripts in the paper repo, `figures/src/` (Opus worker, commit eb4bb74). All on CPU through `res-queue`; the retrieval check took 7.75 minutes.
- **Read-out:** told retrieval (the aspect's head ranks the gallery), plain cosine on unit rows, no map or query expansion; the query itself is excluded. Interval: shared query and seed bootstrap, 10,000 draws.
- **Checks:** c*'s mAP equals r11's stage 1 record in all three aspects; tests 4 passed (`test_r20_ret.py`) and 13 passed (`test_r20_figs.py`); the review page rendered in headless screenshots; each painting in the figures is asserted public domain in its script.

## Results

Label mAP, held ArtELingo; Δ = removal minus c*, mean over seeds 0 to 4, [95%]. Queries: 12,281 for emotion and style, 9,949 for genre.

| Removal | Emotion Δ | Style Δ | Genre Δ |
|---|---|---|---|
| instructed graphs | -0.0208 [-0.0223, -0.0195] | -0.1223 [-0.1246, -0.1200] | +0.0010 [+0.0000, +0.0018] |
| buddy terms | -0.0181 [-0.0190, -0.0171] | -0.0981 [-0.1002, -0.0961] | +0.0000 [-0.0012, +0.0011] |
| content correction | -0.0068 [-0.0077, -0.0061] | -0.0577 [-0.0607, -0.0550] | -0.0011 [-0.0019, -0.0000] |
| relational term | +0.0004 [-0.0007, +0.0014] | -0.0013 [-0.0031, +0.0001] | -0.1287 [-0.1328, -0.1241] |
| sharper teacher row | +0.0001 [-0.0009, +0.0010] | -0.0004 [-0.0023, +0.0011] | -0.0077 [-0.0099, -0.0047] |
| caption endpoints | -0.0010 [-0.0029, +0.0009] | +0.0044 [+0.0034, +0.0055] | -0.0017 [-0.0038, +0.0004] |
| direct teacher | +0.0006 [-0.0005, +0.0017] | +0.0005 [-0.0007, +0.0016] | -0.0013 [-0.0053, +0.0017] |
| shared trunk | -0.0009 [-0.0013, -0.0006] | +0.0001 [-0.0012, +0.0010] | -0.0007 [-0.0013, -0.0002] |

Reference rows, mAP: c* 0.2642 / 0.3110 / 0.5896; plain image 0.2424 / 0.1801 / 0.4760; teacher 0.2665 / 0.3091 / 0.5860 (emotion / style / genre). Full table with each removal's mAP: [`ablation_ret.md`](../../../../src/test/20261109_r20_figures/ret_full/ablation_ret.md); per-seed values and input hashes in the JSON beside it.

The two boundary cells: content correction on genre has its upper bound at -0.00003, instructed graphs on genre its lower bound at +0.00003.

**Retrieval against held clustering AMI** (report 12-28, held, seeds 0 to 4). Each cell is the sign class of the interval: below zero (−), across zero (0), above zero (+).

| Removal | Emotion AMI / mAP | Style AMI / mAP | Genre AMI / mAP |
|---|---|---|---|
| instructed graphs | − / − | − / − | 0 / + |
| buddy terms | − / − | − / − | 0 / 0 |
| content correction | − / − | − / − | + / − |
| relational term | 0 / 0 | 0 / 0 | − / − |
| sharper teacher row | 0 / 0 | 0 / 0 | − / − |
| caption endpoints | 0 / 0 | + / + | 0 / 0 |
| direct teacher | + / 0 | + / 0 | 0 / 0 |
| shared trunk | 0 / − | 0 / 0 | 0 / − |

18 of 24 cells agree. The 6 that differ: the direct teacher on emotion and style (AMI +0.0035, +0.0041), content correction on genre (AMI +0.0072, mAP -0.0011), instructed graphs on genre (mAP +0.0010), shared trunk on emotion and genre (mAP -0.0009, -0.0007). The clustering intervals used 2,000 draws per aspect, the retrieval intervals 10,000.

**How much of c*'s gain over the plain image each large removal loses** (point values; the plain image has no interval):

| Removal | Emotion (gain 0.0217) | Style (gain 0.1309) | Genre (gain 0.1136) |
|---|---|---|---|
| instructed graphs | 96% | 93% | none |
| buddy terms | 83% | 75% | none |
| content correction | 31% | 44% | 1% |
| relational term | none | 1% | 113% |

**Figures** (paper repo, `figures/fig_clusters.png` and `fig_teaser.png`; draft captions in `captions.md`, with trace blocks):

- **Cluster figure** (6.875 x 1.2 in; our pick among the candidates, at your request). Held ArtELingo, k-means with 20 clusters in each embedding. Per aspect, the class whose share gains most from the plain image to c*. For that class, the cluster of each model with the largest share, and its 4 paintings nearest the centre. Sadness 45% against 25%, Ukiyo-e 93% against 14%, sketch and study 51% against 28%.
- **Teaser** (6.875 x 2.3 in). Childe Hassam, "Old Lyme Bridge" (candidate 2 of 12, your pick). The candidates are queries with at least 3 of 5 label matches in each aspect of c*, ranked by distinct neighbours over the three aspects. Five neighbours per row, from the public-domain paintings, near copies skipped. Each aspect row of c* has 5 of 5 neighbours with the query's label, and the three rows share no painting. The plain image's row matches emotion 1 of 5, style 3, genre 4.

## Analysis

Our reading: the ablation's story is the same in both tasks. The instructed graphs give each aspect its own neighbours, and the buddy terms pull paintings toward them. Without either part, emotion and style retrieval falls to about the plain image's level. Without the content correction, c* loses about a third of its emotion gain and almost half of its style gain. Genre does not use these parts; the relational term carries it.

Without the relational term, genre falls about 0.015 below the plain image. The genre head then keeps its buddy terms on the instructed genre graph and the direct teacher. Our reading, not checked: these two alone do not keep the plain image's genre neighbours, and the relational term, which copies the teacher's similarity pattern, restores them and more.

The direct teacher raised emotion and style in held clustering (+0.0035, +0.0041); in retrieval no aspect shows a clear change. Our reading: its effect on the cluster structure does not reach the ranking of near neighbours. The caption endpoints cost style in both tasks, by about two seed ranges in retrieval.

Removing the shared trunk changes retrieval by less than one seed range in each aspect; two of the intervals are below zero. We read this as no change of practical size.

## Deviations

- **Figure designs picked after the candidates were seen.** The decide handoff asked for rules fixed before the paintings were seen. You chose to see every candidate first (round log, 00:20). Then you picked the teaser painting, and we picked the cluster design at your request. The captions state each rule; they do not say it was fixed in advance.
- **The cluster figure's class rule favours c*.** It picks, per aspect, the class with the largest share gain of c* over the plain image. The caption says so. The plain image's Ukiyo-e cluster (14% Ukiyo-e) is mostly Art Nouveau. That cluster and the plain sketch-and-study cluster also serve another class (trace block only, not in the caption).
- **A launch error.** The controller's first launch of the full figure steps failed at once on a shell quoting error. The rerun was unchanged.
- **Screenshots from another project's env.** The review-page check used the Playwright of the DustDevilVis env, run isolated, for screenshots only.
- **A second artist-name match.** The figures worker added a match pass that reads dots as dashes (3 ArtELingo artists). You kept it.

## Files and storage

- **Outputs:** `src/test/20261109_r20_figures/ret_full/ablation_ret.{md,json}` (in git); figure candidates and the review page under `outputs/` → `/data/SSD2/cosir_r20/20261109_r20_figures/`; the paper figures in `docs/paper/cvpr2027_aspect_spaces/figures/`.
- **Storage:** `/data/SSD2/cosir_r20/20261109_r20_figures/` is 32 MB (review page 17 MB, smoke outputs 12 MB); the untracked `ret_smoke/` is 120 KB. Nothing of 1 GB or more. At the wrap-up (user, 2026-11-10) the smoke outputs were deleted; the review page and the figure inputs stay until the supplementary figures.
