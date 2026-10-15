# Grouping redesign, step 1: a style grouping beside affect, image and caption (EXPLORATORY, seed 42 only)

Written 2026-10-05, before any number of this folder exists. Development only; no fresh episode seed is read. Context:
`src/test/20261114_grouping_research/synthesis.md` §5 and §8 (step 1), step 0 in `src/test/20261115_grouping_step0_checks/`.
Decisions by the user (2026-10-05): hand-matched style source **CSD**; generic image-appearance source **VGG-19 Gram
statistics**; affect grouping **Leiden default** (L, 41 groups); GoEmotions stays the affect source for now; the decision
point is Friday 9 October (the CVPR plan's go/no-go date).

**Why.** The image grouping is dominated by genre (pair lift 9.75 against 4.59 for style), so the told oracle's gain on
style × genre is exactly 0 and the reader picks the image grouping in only 17% of style-conditioned style × genre
rankings. A grouping in which style dominates is the only route to reading style × genre.

## 1. Features (GPU, one pass per painting image)

The 61,402 painting images (`/data/PDD/wikiart_proj/wikiart`, paths as in
`src/test/20260929_cross_encoder_stage1/extract_features.py`), one vector per image, mapped to rows by image path.

- **CSD.** The authors' released CSD ViT-L checkpoint (Somepalli et al. 2024, arXiv 2404.01292, official repository) and
  their preprocessing; the style embedding, L2-normalised. If the official weights cannot be obtained, stop and report;
  no substitute model.
- **VGG-Gram.** torchvision VGG-19 with ImageNet weights (`IMAGENET1K_V1`). Shorter side resized to 384, centre crop
  384 × 384, ImageNet normalisation. Layers relu1_1, relu2_1, relu3_1, relu4_1, relu5_1; per layer the Gram matrix
  G = F Fᵀ / (H W), upper triangle with diagonal; per layer a PCA to 128 dimensions fitted on the images of 8,000
  scorer-train paintings (seed 0, randomized solver, seed 0); each 128-dimensional block L2-normalised; the five blocks
  concatenated (640) and L2-normalised.
- Stored under `/data/SSD2/pre_extract/artelingo/style_csd_vitl/` and `/data/SSD2/pre_extract/artelingo/style_vgg19_gram/`
  with a metadata file (model id and weight SHA-256, preprocessing, image order, alignment to the annotation file).
- Local RTX 3090 under `flock -n -o -E 75 /tmp/gpu0.lock`; DAS6 node404 if the local GPU is busy.

## 2. Groupings (CPU, no labels)

- **Style groupings** `style_csd` and `style_gram`: one node per scorer-train painting (36,518; a painting's rows share
  its image), the kNN union graph of `src/model/communities.py::detect_communities` (k = 20, Euclidean on the unit
  vectors), Leiden `RBConfigurationVertexPartition` with resolution 1.0, seed 42 (equal to the modularity default,
  verified in the sweep). Rows inherit their painting's community; communities under 200 rows are merged into the
  nearest-centroid community (`run_told_oracle.merge_small`, centroids in the source space).
- **Random-slot control** `style_rand`: `style_csd`'s painting labels permuted across paintings (seed 0), sizes kept.
- **Descriptive Leiden versions of image and caption** (completing P0's "Leiden for every source"): image on CLIP image
  features with the painting-level recipe above; caption on unit CLIP caption features with `detect_communities`
  defaults at row level plus the same merge.
- Affect L, image k-means 64 and caption k-means 64 are the stored groupings.

## 3. Heads

- **Primary:** `run_told_oracle.fit_one_head` (CLIP ViT-B/32 features, same 60,000-row draw, `LogisticRegression(C=1,
  max_iter=300)`) for every new grouping, image and caption.
- **Secondary, style groupings only:** the image head fitted on the grouping's own source features (CSD or Gram, same draw,
  same classifier); the caption head stays on CLIP caption features. At test time the image is available, so the source
  can place it directly.

## 4. Arms (seed 42 development episodes; B = stored C2, R@1 18.34)

| Arm | Groupings the reader chooses among | Told mapping (fixed now) |
|---|---|---|
| A0 | affect L, image k-means 64, caption k-means 64 (the told-oracle arm L; must reproduce told +1.64, reader +0.35 exactly) | emotion → affect, style → image, genre → image |
| A1 | A0 + `style_csd`, CLIP heads | emotion → affect, style → `style_csd`, genre → image |
| A1s | as A1, `style_csd` image head on CSD features | as A1 |
| A2 | A0 + `style_gram`, CLIP heads | emotion → affect, style → `style_gram`, genre → image |
| A2s | as A2, `style_gram` image head on Gram features | as A2 |
| AR | A0 + `style_rand`, CLIP heads | emotion → affect, style → `style_rand`, genre → image |
| A3 | affect L, image Leiden, caption Leiden (descriptive) | emotion → affect, style → image, genre → image |

For every arm, as in the told-oracle and step-0 runs: the told term and N6's hard reader (argmax Δ over the arm's
groupings) fused on B; the matched counterpart T_cf (mean of the two conditions) fused on B; B′ (B rebuilt with the arm's
averaged-heads term over all its groupings). Reported: R@1, gain and either of fused T against its counterpart (the
margin) and against B; the **bar margin** = fused reader minus whichever of B′ and the counterpart has the larger mean
R@1 (paired); per aspect pair; pick accuracy per pair and condition under the arm's told mapping; paired differences
against A0 and, for A1 to A2s, against AR. 95% intervals from 5,000 painting resamples.

## 5. Label-free diagnostics of the new groupings (reported, decide nothing)

Group count and sizes; AMI with the E2 image, E2 caption and affect L groupings (redundancy), beside the AMI between
image Leiden and E2 image k-means (same source, different algorithm); Leiden-seed stability (seeds 42, 43, 44, mean
pairwise AMI); held-out head accuracy against the majority share; placeability P_ami (step 0b's measure).

**One disclosed label description** of each new grouping, computed after the arms and choosing nothing: AMI with style
and with genre on scorer-train rows, and the pair-level style-versus-genre contrast ratio (profile check 3's measure).

## 6. Readings

- **R1, the style slot is readable (told):** the told margin on style × genre, arm minus A0 (paired), has a lower bound
  above 0 (applied to A1, A1s, A2, A2s).
- **R2, development bar** (handoff Step 0): reader bar margin ≥ +0.5 with a lower bound above 0, and the reader's gain over
  its counterpart with a lower bound above 0. Reference: A0's bar margin is +0.31 (18.75 − 18.44).
- **R3, not just a fourth option:** reader margin over the counterpart, arm minus AR (paired), lower bound above 0.
- **Default for 9 October (the user decides):** if a hand-matched arm (A1, A1s) meets R2 and R3, the one with the larger
  bar margin is proposed for a pre-registered fresh-seed test (49 to 51) with its own decision rule; the generic arms
  (A2, A2s) are reported beside it as the generalisation result. If no arm meets R2, no test is built and 9 October
  decides between continuing (design L, step 3) and changing course.

## 7. Execution

- `extract_style_features.py` (GPU; smoke on 256 images first; the main session launches the full pass in the
  background under the GPU lock) and `run_step1.py` (CPU; stages `group`, `heads`, `eval`, `describe`; `--smoke` with
  stand-in features). CPU work with `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, at most 3
  processes. Scripts assert this file's SHA-256 and refuse to overwrite results.
- Results in `results/` (gitignored), features in `/data/SSD2/pre_extract/artelingo/`, log
  `20261116_grouping_step1_style_log.md`.
- Not done here: design L, the multiplex diagnostic (step 2), any fresh seed, any change to B.
