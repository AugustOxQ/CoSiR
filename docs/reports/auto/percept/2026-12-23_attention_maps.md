# Qwen attention maps under the four views: the instructions move attention, partly to border tokens

> 2026-10-21 · Spec: none (small loop, design in chat; [handoff](../../../superpowers/handoffs/2026-10-21-r10-attention-maps-handoff.md)) · Decision rule: none (descriptive) · Run log: [round log](../../../../src/test/20261021_attention_maps/20261021_attention_maps_log.md)

## Results brief

**What we tried.** We recorded where the pooled token of Qwen3-VL-Embedding-2B looks, on the 5,000 development
paintings. We did this under the four instructions (default, emotion, style, genre) and three inputs: image only,
caption only (all 25,160 captions), and image with caption. A local viewer shows each map over its painting. Seven
summary measures cover all 5,000 paintings. Side task of r10 decide 17.

**Outcome against the bar.** No bar: a descriptive exploration. Main results for layers 16 to 27 (the default band):

- The instructions change the image map. L1 distance to the default map: 0.68 (emotion), 0.79 (style), 0.81 (genre).
  Same values at shuffled positions: 1.16 to 1.23. bf16 rounding noise: 0.09. Style and genre are the closest pair (0.49).
- The instructions lower the image's share of the pooled token's attention: 0.22 (default) against 0.12 (each
  instruction). The first token (an attention sink) takes 0.62 to 0.67 in every view, but passes on little.
- Instructed maps are more peaked: the top 1% of image tokens hold 0.14 to 0.15 of the mass, against 0.06. But the
  hottest token sits on the image border in 34 to 63% of paintings, against 29% for the default and 21% by chance.
  In the last layer and over all layers every view, the default included, peaks on the border in 61 to 98%.
- Added 2026-10-22: the maps are speckled, and our checks found no bug behind it. A fresh recomputation (4 paintings,
  2 views) matches the stored maps up to rounding, and the token order is right. Neighbouring tokens are only weakly
  alike: median Moran's I 0.14 (default) and 0.06 to 0.09 (instructions), about 0 for shuffled maps (5,000 paintings,
  layers 16 to 27). A plain grey image gives the same speckle. So "more peaked" means more mass on a few scattered
  tokens, not on one region ("Spatial structure check" below).
- Added 2026-10-22: replacing each attention weight by the size of what the token adds to the attention output
  (Kobayashi et al. 2020) keeps the checked image-only and caption-only results in the same direction (304 paintings,
  all captions; joint input and word maps not re-checked). It cuts the first token's share to 0.12 to 0.13 (image
  only) and leaves the speckle (Spearman 0.96 to 0.97 with the raw maps). The hottest token sits on the border less
  often: 0.41 (emotion) and 0.15 (default), against 0.60 and 0.25 raw ("Value-weighted check" below).
- On captions, the emotion instruction moves attention to emotion words (bored, anger, lonely). Style and genre
  shift less cleanly: style raises cartoon, abstract, realistic; genre raises abstract, portrait, adventure. The
  default favours scene nouns (horses, statue, beach).
- With the caption added, the image map moves by L1 0.24 to 0.39 (floor 0.09). Under the emotion instruction the
  pooled token then weights the caption (0.071) above the image (0.065).
- Label classes differ in 61 of 63 Kruskal–Wallis tests (Bonferroni p < 0.001), but the differences are small: the
  emotion classes' image share spans 0.217 to 0.223 under the default.

**What it does not show.** Attention weight is not importance for the embedding: no gradient or masking check. Part of
the border hot spots is weight on tokens that pass on little; whether the rest are vision-encoder artifact tokens is
not checked. One model (2B), development paintings
only, one bf16 pass per input. The numbers depend on the band; the last layer and all layers are in the tables below.

**Options for next.**
1. Check the border tokens: mask them and re-measure, or test them as artifact tokens (about an hour, mostly CPU).
   Tells us whether the sharper instructed maps show content.
2. Test the word maps against regions a person names, on a small hand-labelled set (days). Tells us whether
   "anger" lands on the angry part of the painting.
3. Stop here: keep the viewer for figures and intuition, and go back to decide 17 (no cost).

**Your reading.** <left for the user>

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The instructions act as a clear redirection inside the language model. The pooled token reads less of
the image, and the image map changes far above the noise floor in all three bands. We are fairly sure of this. On
captions the move to the aspect's words is clear for emotion and partial for style and genre. We are unsure what the
sharper instructed image maps mean: much of their peak sits on the border. Value weighting removes part of it, but
the emotion map still peaks there more often than the default. The views separate from layer 16 on; layers 0 to 15 give the image nearly the same share in every view.

**Our recommendation.** Option 3. The value-weighted check answered part of option 1; a paper figure should use
value-weighted maps or a gradient method. Our view.

</details>

## What ran

- **Stages and types:** extraction (GPU), then summary analysis (CPU); a small exploration.
- **Data and seeds:** development only: the 5,000 evaluation paintings in `s['subset_pid']` order and their 25,160
  captions (3 to 12 per painting). No held data. No seeds (deterministic forward passes).
- **Code:** `src/test/20261021_attention_maps/` (`att_extract.py`, `att_analysis.py`, `att_viewer.py`), commits
  3ec09b4, 33eab24, d3fccaa. Local RTX 3090, 19:00 to 20:08 (4,116 s, 0.82 s per painting, 300 W, peak 5.6 GB);
  analysis 29 s.
- **Map:** the raw attention of the pooled last token (its CLS role), mean over the 16 heads, summed over a band of
  layers and renormalised over the image tokens (or caption tokens). Bands: layers 16 to 27 (default), the last
  layer, all 28 layers. Word maps: each caption token's attention over the image, in the joint input.
- **Checks:** gate a (embeddings match the stored ones: mean cosine at least 0.998, min 0.9960; each image's nearest
  stored row is its own); gate b (the shared-prefix shortcut matches plain passes: cosine min 0.9971, raw map L1 max
  0.074). All passed.

## Results

Image-only input, mean over 5,000 paintings. "Shuffled" is view B's map with its token positions permuted.

| Band | Measure | default | emotion | style | genre |
|---|---|---|---|---|---|
| 16–27 | image share | 0.219 | 0.125 | 0.120 | 0.119 |
| last | image share | 0.076 | 0.037 | 0.039 | 0.021 |
| all | image share | 0.141 | 0.095 | 0.095 | 0.094 |
| 16–27 | first-token share | 0.623 | 0.647 | 0.670 | 0.673 |
| 16–27 | L1 to default (shuffled; floor 0.09) | | 0.683 (1.162) | 0.790 (1.171) | 0.805 (1.230) |
| last | L1 to default (shuffled; floor 0.14) | | 0.519 (1.411) | 0.462 (1.359) | 0.543 (1.332) |
| all | L1 to default (shuffled; floor 0.06) | | 0.431 (0.999) | 0.504 (1.023) | 0.492 (1.040) |
| 16–27 | normalised entropy | 0.932 | 0.851 | 0.851 | 0.833 |
| 16–27 | top-1% mass | 0.055 | 0.153 | 0.142 | 0.140 |
| 16–27 | hottest token on border (chance 0.21) | 0.29 | 0.63 | 0.42 | 0.34 |
| last | hottest token on border | 0.63 | 0.71 | 0.64 | 0.61 |
| all | hottest token on border | 0.98 | 0.85 | 0.72 | 0.73 |

Captions and the joint input, mean over 5,000 paintings (caption values first averaged within a painting).

| Band | Measure | default | emotion | style | genre |
|---|---|---|---|---|---|
| 16–27 | caption only: caption share | 0.135 | 0.079 | 0.060 | 0.057 |
| 16–27 | joint: image share | 0.170 | 0.065 | 0.095 | 0.106 |
| 16–27 | joint: caption share | 0.065 | 0.071 | 0.052 | 0.045 |
| 16–27 | image map, image only vs joint (L1) | 0.393 | 0.253 | 0.307 | 0.240 |
| last | image map, image only vs joint (L1) | 0.247 | 0.368 | 0.337 | 0.431 |
| all | image map, image only vs joint (L1) | 0.278 | 0.225 | 0.198 | 0.165 |
| 16–27 | word map vs pooled map (correlation) | 0.48 | 0.59 | 0.48 | 0.46 |
| last | word map vs pooled map (correlation) | 0.84 | 0.81 | 0.79 | 0.76 |
| all | word map vs pooled map (correlation) | 0.58 | 0.59 | 0.58 | 0.56 |

**Verdict under the rule:** none (descriptive).

## Analysis

**The views separate in the upper layers.** Per layer, the image share is nearly the same under all four
instructions up to layer 15 (0.03 to 0.21). From layer 17 the default gives the image 0.18 to 0.35 in most layers
(17 and 20 to 26); the instructed views stay below it from layer 16 to 26, at most 0.24 (figure
`fig_image_share_by_layer.png`). The one exception is layer 19, where style (0.106) equals the default (0.103). This
supports layers 16 to 27 as the default band. The last layer gives the image
little weight in every view (0.02 to 0.08).

**Where the attention goes instead.** The instructions move weight from the image to the first token (+0.02 to +0.05),
the instruction tokens (+0.01 to +0.02) and the other template tokens (0.157 default against 0.185 to 0.207). So
the instructed pooled token reads less of the image, and reads it in a more peaked way.

**The border caveat.** The ring of border tokens holds 21% of the image tokens on average. The hottest token is on it
in 29% of default maps (layers 16 to 27), a little above chance, and in 63% of emotion maps. In the last layer and all layers the
share is 0.61 to 0.98 in every view, so the early and last layers carry strong edge spots. Value weighting (below) shows that part of
these spots is weight on tokens that pass on little; whether they are the artifact tokens known from ViTs (Darcet et
al. 2024) is not checked. The viewer shows them as bright edge cells.

**Caption words.** The weight of a caption token is the pooled token's attention on it, renormalised over the
caption and averaged per occurrence (tokens with at least 50 occurrences). Layers 16 to 27, caption only:

- default: horses 0.236, statue 0.227, beach 0.221 (scene nouns);
- emotion: bored 0.404, anger 0.347, lonely 0.335 (emotion words);
- style: cartoon 0.257, abstract 0.255, rainbow 0.245; largest rise over the default: abstract, cartoon, realistic;
- genre: portrait 0.320, abstract 0.295, statue 0.274; largest rise: abstract, portrait, adventure.

So the emotion view picks out emotion words cleanly; style and genre share words (abstract) and keep scene nouns
(statue, garden). In the joint input the style and genre views also raise emotion words (anger, sadness, evil). In
the last layer, punctuation (".", "?", "!") takes the top places in every view.

**Word maps.** A caption word's own map over the image is more peaked than the pooled token's map under the default
(normalised entropy 0.879 against 0.921) and about as peaked under the instructions. It correlates 0.46 to 0.59 with
the pooled map in layers 16 to 27, and 0.76 to 0.84 in the last layer.

**Labels.** With 5,000 paintings, small differences test significant. The largest effects in layers 16 to 27 (the
largest overall: style classes' image share, last layer, style instruction, H 1573):

- genre under the emotion instruction (H 1484): image share 0.137 (portrait), 0.134 (nude) against 0.111
  (cityscape), 0.113 (landscape);
- style under the style instruction, L1 to default (H 895): 0.676 (Minimalism), 0.685 (Color Field) against 0.854
  (Impressionism), 0.846 (Ukiyo-e);
- emotion classes: image share 0.217 to 0.223 (default), 0.120 to 0.133 (emotion instruction).

The 2 tests that do not pass are emotion classes' L1 to default in the last layer (style view) and all layers (genre
view). Full tables: `analysis.md`.

**Spatial structure check (added 2026-10-22).** The scattered look of the image maps is real; the checks found no
bug behind it (the user asked after seeing the viewer). Scripts `att_diag_spatial.py`, `att_diag_order_real.py`,
`att_diag_pos_pattern.py`; full numbers in `outputs/spatial_diag/` and the round log.

- Stored maps: a plain GPU pass (4 paintings, default and emotion views) gives L1 0.02 to 0.03 to the stored maps
  (layers 16 to 27); the viewer serves the stored values to within 5e-7.
- Token order: with a 96 px square painted black on 6 paintings, the changed tokens are found in the map's order
  (median AUC 0.98 to 1.00 at the LLM input and layers 1, 4, 8 and 16; transposed or shuffled 0.33 to 0.55).
- Spatial structure: Moran's I measures how alike neighbouring tokens are (about 0 for random positions, near 1 for a
  smooth map). Median over the 5,000 paintings, layers 16 to 27: 0.138 (default), 0.065 to 0.088 (instructions),
  about 0 shuffled. On 16 paintings, single heads are no smoother.
- Source: a grey canvas with one object gives the same speckle. A pattern shared by position explains little: a
  painting and the leave-one-out mean map of its grid size have median Spearman 0.15 to 0.21 (29 grid sizes, 3,163 paintings).
- Coarse content shows when maps are coloured by rank (8 examples): frames, margins, sky or ground bands. A red disc
  gets 1.4 to 2.3 times its token share (default view, layers 16 to 27, four positions).

Our reading: raw attention of the pooled token is a weak picture of where the model looks. A paper figure would need
another attribution method (for example gradient-based; not tried).

**Value-weighted check (added 2026-10-22).** The first token is a standard attention sink, and discounting it changes
none of the checked results (joint input, word maps and label tests not re-checked). Each attention weight of the pooled token is replaced by the size of what that token adds to the
layer's attention output, ||sum_h a_h W_O^h v_h|| (Kobayashi et al. 2020). Image only on 304 paintings, caption only
on all 25,160 captions; layers 16 to 27 unless named. Script `att_diag_value_weighted.py`, tables in
`outputs/value_weighted/`; reading rules fixed before the run (round log).

- The sink: at the outputs of layers 2 to 8 the first token's hidden state is about 200 to 370 times the median
  token's (norm and largest value); the size of what it passes on (||W_O^h v||, mean over heads) is 0.07 times the median
  image token's. Weighted, its share falls from 0.62–0.67 to 0.12–0.13 (image only; caption only 0.65–0.71 to
  0.15–0.17).
- The speckle stays: weighted and raw maps have Spearman 0.96–0.97. Moran's I misses our bar (a median rise of at least
  0.05, and a rise in at least 75% of paintings); it rises only for emotion, by 0.03 in median.
- The checked results hold: weighted image share 0.64 (default) against 0.36–0.37 (raw 0.22 against 0.12); L1 to the default
  0.65–0.89 (shuffled 1.18–1.31); emotion caption words stay on top (bored, awe, amused, excited).
- Border: the hottest token is on the border in 0.15 (default) to 0.41 (emotion) of weighted maps, against 0.25 to
  0.60 raw; over all layers 0.15–0.42 against 0.73–0.99. So the early layers' edge spots are mostly parked weight.

## Deviations

- **Map type.** The handoff asked for attention maps; we first used rollout (Abnar and Zuidema). It failed on this
  causal model: 99.999% of the rolled-out mass goes to the first token, and the image remainder is top-heavy from the
  causal accumulation. Removing the sink made it worse. Replaced by the pooled token's raw attention (its CLS role),
  layers 16 to 27 as the default, the last layer and all layers as choices, single layers in the viewer. Who decided:
  user, after seeing the recommendation (the user first chose the last layer, then layers 16 to 27). Rollout is still
  stored in the files. Gradient-weighted maps were not made.
- **Gate limits.** Eager bf16 attention does not reproduce the stored sdpa embeddings to cosine 0.999, so gate a uses
  mean 0.998, min 0.99 and a nearest-row check. The raw-map limits of gate b are 0.10 (image) and 0.20 (word maps),
  set from a float32 test: bf16 alone moves a map by L1 0.06 to 0.14. In float32 the shortcut matches the plain pass
  to 1e-6. Who decided: agent.
- **Other choices.** All captions (user). Word maps plus the pooled split between image and caption (user, after
  seeing the recommendation). Local GPU from 19:00, a heavy run under the weekday rule (user, after seeing the
  recommendation). Every measure for all three bands (agent default).

## Files and storage

- **Checks of 2026-10-22:** `outputs/spatial_diag/`, `outputs/value_weighted/` (in the round folder).
- **Outputs:** `src/test/20261021_attention_maps/outputs/qwen3vl_emb_2b_attention/` (`analysis.md`, `analysis.json`,
  8 figures; `per_painting.csv.gz` not committed), run log `outputs/full_run.log`.
- **Viewer:** `PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python src/test/20261021_attention_maps/att_viewer.py --root /data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_attention --port 8765`, then http://localhost:8765/.
- **Store:** `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_attention/`, 8.1 GB, 5,000 `p####.npz` files and
  `run_0_all.json`. Storage: see the disk ledger rows dated 2026-10-21.
