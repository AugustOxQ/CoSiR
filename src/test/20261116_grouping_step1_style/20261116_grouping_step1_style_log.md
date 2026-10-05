# 2026-11-16 grouping redesign, step 1: a style grouping beside affect, image and caption (EXPLORATORY: decides nothing; seed 42 development episodes only)

## Problem

The image grouping (E2 image k-means 64) is dominated by genre: its pair lift is 9.75 for genre against 4.59 for style.
As a result the told oracle gains exactly nothing on style × genre, and the reader picks the image grouping in only 17% of
style-conditioned style × genre rankings. Step 1 (synthesis `src/test/20261114_grouping_research/synthesis.md` §8) asks
whether a fourth grouping in which style dominates makes the style slot readable. Two style sources were tried: CSD,
which is hand-matched to style, and VGG-19 Gram statistics, a generic image-appearance source. Each was compared with a
random-slot control that has the same group sizes. The run also completed P0's "Leiden for every source" by adding
descriptive Leiden versions of the image and caption groupings (arm A3).

## What was run

`PLAN.md` (SHA-256 `b00a4a7aa796957addc232dd92e1b1579306f8b312748a36ac1479331f8b0520`, written 2026-10-05T19:26:09+02:00,
before any number of this folder existed) fixed the groupings, heads, arms, diagnostics and readings. `run_step1.py`
has four stages (`group`, `heads`, `eval`, `describe`) and two sets. `--set descriptive` covers image_leiden,
caption_leiden and arms A0 and A3. `--set style` covers style_csd, style_gram, style_rand and arms A0, AR, A1, A1s, A2
and A2s; there A0 is recomputed as the paired baseline. The script asserts PLAN.md's SHA-256 and refuses to overwrite
non-smoke outputs. It reuses by import `run_told_oracle.py` (`fit_one_head`, `evaluate_arm`, `merge_small`,
`group_stats`, `per_pair_margin`, `margin_arrays`, `dpairs`), `run_sweep.py` (`knn_union_graph`, `leiden`, `setup`) and
step 0's `run_checks.py` (`e2_partitions`, `placeability`, `load_cell`, `now_ams`, loaded under another module name).
Nothing outside this folder changed. All work ran on the CPU (`CUDA_VISIBLE_DEVICES=`, 8 threads per process, at most 2
processes at a time).

The four measurement stages:

- **Groupings.** style_csd, style_gram and image_leiden use one node per scorer-train painting (36,518). The graph is
  `detect_communities`' kNN union graph (k = 20, Euclidean on unit vectors), partitioned by Leiden
  RBConfigurationVertexPartition at resolution 1.0 with seed 42. Rows inherit their painting's community, and
  communities under 200 rows are merged with `merge_small`. caption_leiden runs `detect_communities` defaults on unit CLIP
  caption features at row level (183,694 rows), followed by the same merge. style_rand is style_csd's painting labels
  permuted across paintings (seed 0). Leiden seeds 43 and 44 were run on the same graphs for the stability diagnostic.
- **Heads.** Every new grouping gets `fit_one_head` (CLIP ViT-B/32 features, the 60,000-row draw, `LogisticRegression(C=1,
  max_iter=300)`). style_csd and style_gram also get a secondary image head on their own source features (same draw,
  same classifier); the caption head stays the CLIP one.
- **Arms.** The told term and N6's hard reader (argmax Δ over the arm's groupings) are each fused on B (the stored C2,
  R@1 18.34 [17.97, 18.70]) and compared with their matched counterpart T_cf, also fused on B. Each arm also gets B′
  (B rebuilt with the arm's averaged-heads term) and the bar margin: the fused reader minus whichever of B′ and the
  counterpart has the larger mean R@1, paired. Results are reported per aspect pair, with pick accuracy under the arm's
  told mapping and paired differences against A0 and AR. 95% intervals come from 5,000 painting resamples. The
  told-oracle `evaluate_arm` assumes three groupings and one told mapping, so a generalised copy (any tuple of
  groupings, any mapping) was written and proven equal to the original on A0.
- **Diagnostics.** These are the label-free measures of PLAN.md §5. After both eval runs, the one disclosed label
  description was computed.

Runs (Amsterdam time; `TZ=Europe/Amsterdam`):

| Batch | Script SHA-256 | Stages (runtime) | Started → finished |
|---|---|---|---|
| Smoke (stand-in features, 3,000-row heads, `results/smoke/`) | `78e887e9164b…` (as descriptive) | all 8 stage × set runs | 19:38:45 → 19:46:23 |
| Descriptive | `78e887e9164b032afb97bb0766180d498c152ee2455de2520e4046bcf7aae4d0` | group 325 s (caption_leiden 297 s), heads 39 s, eval 74 s, describe 6 s | 19:46:36 → 19:54:08 |
| Style | `f4bea509a6ca4fbb48e60823b4add1ba89570b626416127aaf57db66d63018c0` | group 61 s, heads 86 s, eval 129 s, describe 10 s | 20:44:40 → 20:49:52 |

The script changed once between the batches. The style-feature loader had been a stub until the features existed, and
it was then written against `meta.json`. The change touches only `real_style_features` and its call. The final file's
SHA-256 is `f4bea509a6ca4fbb48e60823b4add1ba89570b626416127aaf57db66d63018c0`, as recorded in each style JSON's provenance. A first style `group`
attempt at 20:44:12 stopped inside the loader before writing anything. It had compared `meta.json`'s mapping hashes
with file hashes, while the extractor hashes the arrays' bytes (`sha256_array`). The comparison was corrected and the
stage rerun.

The style features are the extraction agent's full pass (meta `created` 2026-10-05T20:30:40+02:00, script
`1d166babe2f0…`). CSD ViT-L comes from `tomg-group-umd/CSD-ViT-L` (file SHA-256 `40e92fad63a3…`): a 768-d style head,
L2-normalised. VGG-Gram comes from torchvision VGG-19 `IMAGENET1K_V1`: 5 Gram blocks, each PCA'd to 128 and
L2-normalised, giving 640 dimensions. Both are stored as 61,402 image rows plus `row_to_image.npy`, so the
load_artelingo-order features are `embeddings[row_to_image]`.

### Checks (all passed)

| Check | Result |
|---|---|
| PLAN.md SHA-256 equals the dispatched version (every stage) | yes |
| `run_sweep.setup`: B equals stored C2; R0 told +1.14 [0.90, 1.41], reader +0.14 [−0.04, 0.32] reproduce `diagnose_counterparts.json`, `diagnose_fixes.json`, `told_oracle.json`; stored L and K arrays reproduce `told_oracle.json` (both eval runs) | exact |
| `partition_L` equals the sweep's `leiden_k20_r1.0` partition | exact |
| Refit affect L head equals the told oracle's arm L head (both eval runs) | exact |
| **Generalised `evaluate_arm` equals the original on A0**: told, reader and pick blocks, all per-anchor arrays (5 metrics × fused and counterpart), T_6u, told indices (smoke and both full runs) | bit identical |
| **A0 reproduces told-oracle arm L** (both full eval runs): blocks, arrays, B′, told +1.64 [1.37, 1.92], reader +0.35 [0.15, 0.57], pick 54.7, bar +0.31 = 18.75 − 18.44 (B′) | exact |
| Seeded Leiden wrapper at seed 42 equals `run_sweep.leiden` (painting-level groupings); graph + Modularity at seed 42 equals `detect_communities` (caption_leiden) | exact |
| Style features (both): `meta.json` is the full pass under this PLAN.md; annotation file SHA-256 equal; mapping-array SHA-256s equal `meta.json`; 61,402 distinct images, all used; `image_paths[row_to_image_annotation_order]` equals every annotation row's `image` field; `row_to_image` equals the annotation map at `data.sample_ids`; every load_artelingo row maps to its own annotation image; painting ↔ image one to one; rows sharing an image path carry identical vectors | all true |
| Painting vectors: paintings whose rows differ in the source vector | CSD 0, Gram 0; CLIP image 309 (max 1.05e−5, float noise) |
| `fit_image_head` (source-feature head) run with CLIP image features equals `fit_one_head`'s image head | bit identical |
| Every 60,000-row draw contains every group; 0 convergence warnings | yes |
| Style × genre contrast copy reproduces profile check 3 (E2 image, rtol 1e−9) and `pair_stats_groups` (L, emotion × style) | yes |

After the runs, the implementer re-derived several numbers with separate code in the session scratchpad. The told,
reader and bar margins, R1's style × genre difference, R3's difference and pick accuracy for every arm were recomputed
from `step1_eval_*.npz` with a separate bootstrap (seed 777). All points were equal, and every interval was within
±0.02 of the JSON. style_csd and style_gram were also rebuilt from `embeddings.npy` and `row_to_image.npy` with an
independently written kNN and Leiden call (ARI 1.0 against the stored partitions, 17 and 11 communities).

## Results

### Groupings (label free; PLAN.md §5)

| Grouping | Groups | Rows min / median / max | effN | AMI vs E2 image / E2 caption / affect L | Leiden-seed AMI (42–43, 42–44, 43–44; mean) | Held-out img / txt (majority) | P_ami (P_lift) |
|---|---|---|---|---|---|---|---|
| style_csd | 17 | 532 / 9,287 / 23,979 | 11.7 | 0.4029 / 0.1110 / 0.0197 | 0.827, 0.799, 0.796; **0.807** (seeds 43, 44: 14, 15 groups) | 85.10 / 40.41 (13.08) | 0.2084 (2.511) |
| style_csd, CSD image head | | | | | | 91.67 / (CLIP) | 0.1981 (2.548) |
| style_gram | 11 | 3,153 / 17,461 / 24,951 | 10.3 | 0.2199 / 0.0750 / 0.0128 | 0.695, 0.687, 0.637; **0.673** (11, 13 groups) | 71.53 / 31.92 (13.12) | 0.1497 (1.863) |
| style_gram, Gram image head | | | | | | 89.90 / (CLIP) | 0.1242 (1.891) |
| style_rand | 17 | 567 / 9,343 / 23,374 | 11.7 | 0.0038 / 0.0005 / 0.0000 (vs style_csd 0.0011) | n/a | 15.64 / 13.20 (13.18) | 0.0022 (1.004) |
| image_leiden | 17 | 2,697 / 10,328 / 22,009 | 14.4 | **0.5702** / 0.1651 / 0.0225 | 0.732, 0.712, 0.785; 0.743 | 90.35 / 46.46 (11.92) | 0.2818 (3.664) |
| caption_leiden | 30 (32 raw, 217 rows merged) | 631 / 4,064 / 20,726 | 21.2 | 0.1648 / **0.5120** / 0.0329 | 0.843, 0.848, 0.835; 0.842 | 37.34 / 84.60 (11.68) | 0.2226 (3.346) |

References: E2 image and caption from the stored N6 posteriors have P_ami 0.2416 and 0.1959 and P_lift 5.896 and 4.392;
AMI(E2 image, E2 caption) is 0.1663.

At resolution 1.0, painting-level Leiden gives coarse groupings: 17 groups for CSD and for CLIP image, 11 for Gram, all
far below the 64 of the stored k-means groupings. No style or image community fell under 200 rows. style_csd shares a
fair amount with E2 image (AMI 0.40) and little with caption or affect, so it is not a copy of the image grouping.
style_gram is weaker on every label-free count: AMI with image 0.22, seed stability 0.67, held-out CLIP image accuracy
71.53, P_ami 0.150. The source-feature image heads fit their own groupings better than the CLIP head (CSD 91.67 against
85.10, Gram 89.90 against 71.53), but they place no better against the caption head (P_ami 0.198 against 0.208 for CSD,
0.124 against 0.150 for Gram). image_leiden and caption_leiden agree with their k-means counterparts at AMI 0.57 and 0.51,
so the algorithm change alone moves about half the partition.

### Arms (R@1 in pp; B = C2, R@1 18.34 [17.97, 18.70]; baselines A0 and AR)

| Arm | Told margin | Reader margin | Reader gain margin | Reader − B (T_cf − B) | B′ (B′ − B) | Bar margin (comparator) | Pick accuracy |
|---|---|---|---|---|---|---|---|
| **A0** (baseline) | +1.64 [1.37, 1.92] | +0.35 [0.15, 0.57] | +1.34 [1.03, 1.65] | +0.41 (+0.05) | 18.44 (+0.10) | **+0.31 [0.10, 0.53]** (B′) | 54.7 [54.1, 55.3] |
| **AR** (random slot) | +0.67 [0.40, 0.95] | +0.25 [0.09, 0.42] | +0.71 [0.46, 0.97] | +0.27 (+0.02) | 18.45 (+0.11) | +0.16 [−0.02, 0.35] (B′) | 41.9 [41.3, 42.5] |
| A1 (CSD, CLIP heads) | **+2.23 [1.93, 2.56]** | +0.06 [−0.15, 0.28] | +1.30 [0.96, 1.63] | +0.47 (+0.41) | 18.80 (+0.46) | +0.01 [−0.23, 0.24] (B′) | 43.2 [42.5, 43.9] |
| A1s (CSD, CSD image head) | **+2.26 [1.94, 2.61]** | −0.01 [−0.24, 0.22] | +1.52 [1.16, 1.87] | +0.53 (+0.54) | 18.88 (+0.54) | −0.01 [−0.26, 0.25] (B′) | 43.3 [42.6, 43.9] |
| A2 (Gram, CLIP heads) | +1.76 [1.45, 2.08] | −0.00 [−0.18, 0.17] | +0.56 [0.29, 0.82] | +0.13 (+0.14) | 18.51 (+0.17) | −0.04 [−0.25, 0.18] (B′) | 42.0 [41.3, 42.6] |
| A2s (Gram, Gram image head) | +1.60 [1.26, 1.93] | +0.10 [−0.04, 0.23] | +0.28 [0.08, 0.49] | +0.22 (+0.12) | 18.24 (−0.10) | +0.10 [−0.04, 0.23] (counterpart) | 41.4 [40.8, 42.1] |
| A3 (descriptive: affect L, image Leiden, caption Leiden) | +1.39 [1.09, 1.69] | +0.04 [−0.20, 0.27] | +1.57 [1.22, 1.93] | +0.29 (+0.25) | 18.35 (+0.01) | +0.04 [−0.20, 0.27] (counterpart) | 57.0 [56.3, 57.6] |

Pick accuracy counts a pick as correct when the reader's argmax-Δ grouping equals the arm's told grouping for that
condition. With four groupings the chance level is 25% against 33% for three, so the pick accuracies of A1 to AR are
not directly comparable with A0's or A3's.

Per aspect pair (margin R@1; told | reader):

| Arm | Emotion × style | Emotion × genre | Style × genre |
|---|---|---|---|
| A0 | +2.50 / +0.49 | +3.03 / +0.67 | −0.61 [−0.89, −0.35] / −0.10 |
| AR | −0.27 / +0.09 | +2.81 / +0.56 | −0.54 / +0.11 |
| A1 | +3.27 / +0.47 | +3.04 / +0.45 | **+0.39 [−0.16, 0.91]** / −0.73 [−1.11, −0.34] |
| A1s | +3.45 / +0.56 | +2.98 / +0.19 | +0.35 [−0.20, 0.89] / −0.77 [−1.19, −0.35] |
| A2 | +2.15 / +0.22 | +3.14 / −0.01 | +0.00 [−0.54, 0.52] / −0.21 |
| A2s | +1.64 / −0.11 | +3.12 / +0.20 | +0.04 [−0.53, 0.59] / +0.20 |
| A3 | +2.77 / +0.29 | +2.20 / +0.60 | −0.81 / −0.77 |

Paired differences (per anchor; R@1):

| Arm | Told − A0 | Told s×g − A0 | Reader − A0 | Bar − A0 | Told − AR | Reader − AR | Bar − AR |
|---|---|---|---|---|---|---|---|
| AR | −0.97 [−1.25, −0.68] | +0.07 [−0.42, 0.57] | −0.10 [−0.29, 0.09] | −0.15 [−0.33, 0.02] | | | |
| A1 | +0.59 [0.35, 0.85] | **+1.00 [0.49, 1.51]** | −0.29 [−0.52, −0.06] | −0.31 [−0.55, −0.05] | +1.56 [1.24, 1.89] | **−0.19 [−0.44, 0.04]** | −0.15 [−0.41, 0.11] |
| A1s | +0.62 [0.37, 0.89] | **+0.96 [0.42, 1.47]** | −0.36 [−0.61, −0.12] | −0.32 [−0.59, −0.04] | +1.59 [1.25, 1.94] | **−0.26 [−0.52, −0.01]** | −0.17 [−0.45, 0.12] |
| A2 | +0.12 [−0.14, 0.39] | **+0.61 [0.08, 1.15]** | −0.36 [−0.56, −0.15] | −0.35 [−0.61, −0.09] | +1.09 [0.78, 1.41] | **−0.26 [−0.47, −0.05]** | −0.20 [−0.45, 0.06] |
| A2s | −0.04 [−0.32, 0.24] | **+0.65 [0.09, 1.21]** | −0.26 [−0.47, −0.04] | −0.22 [−0.44, 0.01] | +0.93 [0.61, 1.25] | **−0.16 [−0.34, 0.02]** | −0.06 [−0.27, 0.14] |
| A3 | −0.25 [−0.54, 0.03] | −0.20 [−0.65, 0.28] | −0.32 [−0.57, −0.06] | −0.27 [−0.53, −0.01] | | | |

**What moved and why.** Telling the model "style → style_csd" works. The told margin rose from +1.64 to +2.23 (A1)
and +2.26 (A1s). On style × genre it went from −0.61 to +0.39 and +0.35, a paired gain of about one point. The told
term rose on emotion × style as well (+3.27 against +2.50), while emotion × genre did not move. Gram helped the told
term on style × genre too (+0.61, +0.65 paired), though less than CSD, and the gain did not reach the overall told
margin (A2 +0.12 paired, A2s −0.04). The random slot lowered the told term by 0.97, as expected when style is told to
read a grouping that carries no style.

The reader did not follow. Every style arm's reader margin fell to about zero, between −0.01 and +0.10, against A0's
+0.35, and it fell below AR's +0.25 as well. Measured against B, the A1 reader barely moved (+0.47 against A0's +0.41).
Its condition-free counterpart, however, rose from +0.05 to +0.41, and B′ rose by the same amount (B′ − B +0.46): the
fourth grouping's heads are useful as an unconditioned similarity, and the matched comparators absorb that use.

The pick tables show where the conditioned use fails. Under A1 the reader chose style_csd for 73.5% of emotion × style
style conditions, but also for 70.5% of emotion × genre genre conditions, where the told grouping is image (image was
picked 15.8%). In style × genre it picked style_csd for only 30.2% of style conditions (affect 45.8%) and for 55.0% of
genre conditions (image 20.4%). The reader treats style_csd as the image-appearance slot for both style and genre, so
the style × genre reader margin fell from −0.10 to −0.73. The label description below agrees: style_csd still carries
genre about as much as style.

A3 (descriptive) swaps both k-means 64 groupings for Leiden ones (17 and 30 groups). The told term fell by 0.25
(paired, interval reaching +0.03) and the reader by 0.32 [−0.57, −0.06]. The counterpart rose again (+0.25 against B,
against +0.05 for A0), so the reader's margin fell to +0.04.

## Readings of PLAN.md §6, applied literally

| Reading | Rule | A1 | A1s | A2 | A2s |
|---|---|---|---|---|---|
| R1, style slot readable (told) | told s×g margin, arm − A0 (paired), lower bound > 0 | +1.00 [0.49, 1.51] **met** | +0.96 [0.42, 1.47] **met** | +0.61 [0.08, 1.15] **met** | +0.65 [0.09, 1.21] **met** |
| R2, development bar | bar margin ≥ +0.5 with lower bound > 0, and reader gain margin lower bound > 0 | bar +0.01 [−0.23, 0.24]; gain +1.30 [0.96, 1.63]: **not met** | bar −0.01 [−0.26, 0.25]; gain +1.52 [1.16, 1.87]: **not met** | bar −0.04 [−0.25, 0.18]; gain +0.56 [0.29, 0.82]: **not met** | bar +0.10 [−0.04, 0.23]; gain +0.28 [0.08, 0.49]: **not met** |
| R3, not just a fourth option | reader margin, arm − AR (paired), lower bound > 0 | −0.19 [−0.44, 0.04] **not met** | −0.26 [−0.52, −0.01] **not met** | −0.26 [−0.47, −0.05] **not met** | −0.16 [−0.34, 0.02] **not met** |

R2 for reference and controls: A0's bar margin is +0.31 [0.10, 0.53] (reference; not met). AR's is +0.16 [−0.02, 0.35]
(not met). A3, descriptive, is at +0.04 [−0.20, 0.27] (not met). Every arm's reader gain margin has its lower bound
above 0; every R2 failure comes from the bar clause.

**Default for 9 October (PLAN.md):** no arm among A1, A1s, A2 and A2s meets R2, so no fresh-seed test is built, and
9 October decides between continuing (design L, step 3) and changing course. The user decides.

## Label description (disclosed; computed after both eval runs; chooses nothing)

AMI with style uses all 183,694 scorer-train rows and AMI with genre the 148,956 rows with a genre label. The style ×
genre contrast is profile check 3's measure on different-painting pairs: s_AB = P(same group | same style, different
genre), s_BA = P(same group | same genre, different style), and ratio = s_AB / s_BA (above 1: style dominates).

| Grouping | AMI style | AMI genre | s_AB | s_BA | Ratio |
|---|---|---|---|---|---|
| style_csd | **0.3413** | 0.3302 | 0.1859 | 0.2229 | **0.834** |
| style_gram | 0.1435 | 0.1871 | 0.1354 | 0.1756 | 0.771 |
| style_rand | 0.0018 | 0.0011 | 0.0975 | 0.0976 | 0.999 |
| image_leiden | 0.2844 | 0.4358 | 0.1028 | 0.2384 | 0.431 |
| caption_leiden | 0.0554 | 0.1636 | 0.0540 | 0.1082 | 0.499 |
| E2 image (reference) | 0.3176 | 0.3970 | 0.0257 | 0.0585 | 0.439 |
| E2 caption (reference) | 0.0582 | 0.1613 | 0.0161 | 0.0374 | 0.430 |
| affect L (reference) | 0.0159 | 0.0236 | 0.0376 | 0.0388 | 0.970 |

CSD moves the balance toward style: the ratio rose from 0.439 (E2 image) to 0.834, and style_csd is the only grouping
whose AMI with style exceeds its AMI with genre. Style still does not dominate (ratio below 1). That fits the reader
choosing style_csd for genre conditions. Gram moves the ratio about as much (0.771) but carries little of either label
(AMI 0.14 and 0.19). The random slot sits at 1.0, as it must.

## Choices not fixed by PLAN.md

- **Two sets, one script.** The style features arrived after the descriptive arms ran, so every stage runs per set and
  writes `results/step1_<stage>_<set>.{json,txt[,npz]}` (the brief named `step1_<stage>`). The style eval run recomputes
  A0 as its paired baseline and asserts the reproduction again.
- **Painting vectors.** A painting's vector is that of its first scorer-train row. CLIP image vectors differ inside 309
  paintings by at most 1.05e−5; CSD and Gram vectors never differ. Vectors were unit-normalised again (`run_checks.unit`,
  float32) before the kNN, including the already-unit CSD and Gram vectors.
- **Merge.** Under-200 is counted in rows. Centroids are row-weighted means of the unit painting vectors (rows inherit
  their painting's vector); for caption_leiden they are row means of the unit caption features. No style or image
  community needed merging. caption_leiden merged 2 communities holding 217 rows.
- **Stability.** Seeds 43 and 44 ran on the same graph and were merged the same way. For caption_leiden they used
  ModularityVertexPartition (`detect_communities`' call), with seed 42 asserted equal to `detect_communities`. AMI was
  computed on scorer-train rows (sklearn default, arithmetic). It is not applicable to style_rand.
- **style_rand.** `default_rng(0).permutation` over the 36,518 merged painting labels of style_csd. Painting counts per
  group are kept exactly and row counts approximately (min 567 against 532). No merge was applied.
- **Secondary heads.** These use a copy of `fit_one_head`'s image branch with the source features (verified bit identical
  to `fit_one_head` when given CLIP features). The caption posterior is the same grouping's CLIP caption head, not refit.
- **Bar margin.** The comparator is whichever of B′ and the counterpart has the larger mean R@1 over all episodes (ties
  go to B′); the same comparator is used per aspect pair.
- **Readings.** R2's "the reader's gain over its counterpart" was read as the gain metric of the fused reader minus the
  fused counterpart (`fusedT_vs_fusedTcf.gain`), and "≥ +0.5" was applied to the full-precision point. R1 and R3 were
  applied to A1, A1s, A2 and A2s, and R2 to those four plus AR (A0 as reference, A3 as description). "No arm meets R2"
  in the default was read over A1, A1s, A2 and A2s.
- **Pick accuracy.** A pick is correct when the reader's argmax-Δ grouping equals the arm's told grouping for that
  condition. For AR the told style grouping is style_rand.
- **Label description.** Style AMI on all rows (every row has a style), genre AMI on genre-labelled rows, and the
  contrast on rows labelled for both with exact pair counts. Values for E2 image, E2 caption and affect L were added as
  references.
- **Placeability.** P_ami and P_lift were computed on the 32,413 selection rows, as in step 0b, with references from the
  stored N6 posteriors.

## Caveats

- This is exploratory work on the seed 42 development episodes, which earlier runs have read many times. The groupings,
  heads, arms, told mappings and readings were fixed in PLAN.md before any number.
- Each grouping comes from one Leiden seed and one head draw. Seed stability is moderate for style_csd (0.807) and lower
  for style_gram (0.673). For example, seeds 43 and 44 give 14 and 15 CSD groups instead of 17.
- **Granularity is confounded with source.** Painting-level Leiden at resolution 1.0 gives 11 to 17 groups, against 64 for
  the k-means groupings they sit beside (or replace, in A3). A coarse grouping makes a strong condition-free similarity,
  which raises the counterpart and B′; that is part of why the bar margins fall. PLAN.md fixed the recipe, and no other
  resolution was tried.
- Pick accuracies over four groupings (chance 25%) are not comparable with those over three (33%).
- The told mapping keeps genre → image. The reader's preference for style_csd under genre conditions counts as a wrong
  pick even where style_csd separates genre well, so pick accuracy understates how usable the grouping is for genre. The
  fused R@1 margins do not depend on that mapping.
- B's cross-fit picks were tuned on the same parity halves that the fusion and counterpart cross-fits reuse. This is a
  small second-order leak shared by every arm.
- The script's SHA-256 differs between the descriptive (`78e887e9…`) and style (`f4bea509…`) batches, because of the
  loader change described above. The descriptive outputs do not touch the changed function.
- `results/smoke/` holds the smoke runs (DINOv2-small stand-in for CSD and Gram, 3,000-row heads); their numbers are not
  results.

## Files

- `PLAN.md`: the fixed design (unchanged).
- `run_step1.py`: stages `group`, `heads`, `eval`, `describe`; `--set descriptive|style`; `--smoke`.
- `results/step1_group_{descriptive,style}.{json,txt,npz}`: grouping settings, sizes, merges, graph and Leiden
  statistics, feature provenance and alignment checks; the npz holds every partition (merged, raw, seeds 43/44).
- `results/step1_heads_{descriptive,style}.{json,txt,npz}`: head provenance and held-out accuracies; selection-row
  posteriors (CLIP heads and source-feature image heads).
- `results/step1_eval_{descriptive,style}.{json,txt,npz}`: checks, per-arm blocks (overall, per pair, pick, B′, bar,
  paired differences), readings; the npz holds per-anchor fused, counterpart and B′ arrays for all metrics, reader picks
  and told indices, so every margin can be re-derived.
- `results/step1_describe_{descriptive,style}.{json,txt}`: label-free diagnostics; the style file also holds the label
  description.
- `results/run_*.log`: run logs. `results/extract_full.log` belongs to the GPU extraction, not to this script.
- Everything in `results/` is gitignored: 56 MB in total, including 27 MB of smoke outputs. The largest file is 13.7 MB,
  and nothing over 100 MB was written. The style features (`/data/SSD2/pre_extract/artelingo/style_*`, about 0.5 GB) are
  the extraction's outputs and were only read.

## Controller review (2026-10-05 20:55)

The main session re-derived with its own code (`rederive_step1.py` in the session scratchpad): the CSD style grouping
rebuilt from the raw embeddings with its own kNN graph and Leiden call (17 communities, identical to the stored
partition, AMI 1.0; AMI with E2 image 0.4029); and, from `step1_eval_style.npz` with its own painting bootstrap, the
told, reader and bar margins of A0, AR, A1, A1s, A2 and A2s, R1 on the style × genre episodes and R3. Every point estimate
equals the table above and every interval agrees to within 0.03 (bootstrap draws differ). The readings stand: R1 met by
all four style arms, R2 and R3 by none.
