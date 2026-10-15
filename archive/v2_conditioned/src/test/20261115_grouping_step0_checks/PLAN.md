# Grouping redesign, step 0: three label-light checks (EXPLORATORY, seed 42 only)

Written 2026-10-05, before any number of this folder exists. Decides nothing about the GO; no fresh episode seed is
read. Context: `src/test/20261114_grouping_research/research_brief.md` and `synthesis.md` §6 to §8. The user approved
the order (step 0 first) and the two defaults below (2026-10-05): check 0b may read the already computed told margins
once, disclosed, only to validate a criterion; check 0c uses the centred-centroid soft cosine F1 with γ = 1, fixed.

## Shared inputs (all from existing files; nothing is re-clustered)

- Rows: `artelingo_splits(load_artelingo())`; scorer-train 183,694 rows, selection 32,413 rows. Painting id of a row =
  its split group (`groups`); local scorer-train painting ids `np.unique(groups[scorer_train], return_inverse=True)[1]`.
- Groupings on scorer-train rows (local indices):
  - **R0**: E2 affect k-means 64 (`src/test/20261031_pseudo_partitions/results/partitions.npz`, key `affect`).
  - **L**: Leiden default on GoEmotions probabilities (graph k 20, resolution 1.0, merged, 41 groups;
    `src/test/20261111_community_told_oracle/results/per_anchor_told_oracle.npz`, key `partition_L`; must equal the
    sweep's `leiden_k20_r1.0` partition).
  - **Sweep cells**: the 9 Leiden cells and 8 k-means controls in `src/test/20261112_community_sweep/results/cells/*.npz`
    (key `partition`).
  - **image** and **caption**: E2 k-means 64 (`partitions.npz`, keys `image`, `caption`), used as references.
- Heads: `run_told_oracle.fit_one_head` (same 60,000-row draw, same 10,000 check rows, `LogisticRegression(C=1,
  max_iter=300)` on unit-normalised CLIP features); stored N6 posteriors
  (`src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz`) for R0, image and caption.
- Episodes, B and the fusion: the told-oracle machinery (`run_told_oracle.evaluate_arm`, B = stored C2, R@1 18.34),
  seed 42 development episodes, 95% intervals from 5,000 painting resamples.

## Check 0a: same-painting ceiling of the affect grouping (no labels)

**Question.** Can an image ever tell which affect group a row is in? A painting's rows share one image; each row's
affect group comes from its own caption.

**Measures**, for R0, L, every sweep cell (descriptive), caption and image (references; image should be close to 1):

- p_same = share of unordered row pairs within one painting that share a group (exact counts).
- p_diff = share of unordered row pairs from different paintings that share a group (exact counts).
- R = p_same / p_diff.
- A_loo = leave-one-out painting-majority accuracy: each row of a painting with at least 2 rows is predicted by the most
  frequent group among the painting's other rows; a tie between t groups that contains the true group counts 1/t
  (expected accuracy under uniform tie-breaking). Number of rows used is reported.
- Majority share (largest group's share of rows) beside A_loo.
- Control: 20 random relabellings that keep group sizes (permute labels across rows, seeds 0 to 19): mean and range of
  p_same, p_diff, R and A_loo.

**Reading** (applied to R0 and to L separately; A_head = the grouping's image-head held-out accuracy, R0 13.47%,
L 9.81%):

- **Room**: A_loo ≥ 1.5 × A_head and R ≥ 1.5. Image-side head work for affect (synthesis §8 step 3, arm b) is worthwhile.
- **No room**: A_loo < 1.2 × A_head or R < 1.2. Skip image-side head work for affect; consider placing captions by
  GoEmotions directly.
- **Limited room**: otherwise.

Caveat stated in advance: A_head's 10,000 check rows were drawn by row, so some of their paintings are in the 60,000-row
fitting draw; A_head may be optimistic for unseen paintings.

## Check 0b: is placeability a usable label-free criterion?

**Question.** Does a label-free score rank groupings the way their (already computed) told margins do?

**Groupings.** The 9 Leiden cells and the 8 k-means controls (affect heads refit with `fit_one_head`, which must
reproduce each cell's stored held-out accuracies in `cells/*.json`), plus R0 (stored posteriors). Image and caption
k-means are reported for reference only; comparing placeability across sources is known to favour content (synthesis §6).

**Measures** (selection rows, image-head and caption-head posteriors of the same row):

- **Primary, P_ami**: adjusted mutual information between the argmax of the image head and the argmax of the caption
  head, over all selection rows.
- Secondary, P_lift: mean same-row agreement p_img(i) · p_txt(i) divided by the mean agreement over all ordered pairs of
  rows from different paintings (exact, from per-painting sums).
- Head held-out accuracies beside them.

**Matched pairs** (sweep §8): k10_r0.25 ↔ k-means 15, k10_r1.0 ↔ 44, k10_r4.0 ↔ 118, k20_r0.25 ↔ 14, k20_r1.0 ↔ 41,
k20_r4.0 ↔ 95, k40_r0.25 ↔ 15, k40_r1.0 ↔ 31, k40_r4.0 ↔ 88.

**Validation target (label read, disclosed).** The told margins in `src/test/20261112_community_sweep/results/sweep.json`,
already computed: Leiden beat its matched k-means in all 9 pairs (+0.48 to +0.88, every lower bound above 0). They are
read once here, to validate the criterion; no grouping is chosen with them.

**Reading.** Placeability is adopted as the within-source criterion for later steps if P_ami(Leiden) > P_ami(matched
k-means) in **at least 8 of the 9 pairs**. Otherwise it is not adopted, and later label-free choices fall back to
stability and non-redundancy (not computed here). P_lift's count and a Spearman correlation between P_ami and the told
margin over the 17 cells are reported, descriptive only.

## Check 0c: sibling-aware agreement

**Question.** Does counting sibling groups as related (agreement p_imgᵀ S p_txt) help the told and reader margins
without inflating agreement for unrelated pairs?

**Setting.** As the told-oracle arm L: affect = L (41 groups), image and caption = E2 k-means 64; affect heads refit with
`fit_one_head`, image and caption posteriors stored. B unchanged (stored C2).

**S (F1, primary, fixed).** For a grouping with source features X (affect: the 28 GoEmotions probabilities,
`affect_prepare.npz` `affect_probs`, raw; image and caption: unit-normalised CLIP features), on scorer-train rows:
centroid μ_g = mean of X over group g; global mean μ̄ = mean of X over all rows; c_g = μ_g − μ̄;
S_gk = max(0, cos(c_g, c_k)) (γ = 1), S_gg = 1. S is symmetric, so agreement p_imgᵀ S p_txt is applied by replacing every
caption posterior q with S q; Δ (reader), the told and reader scores and the condition-free counterpart all use it.

**Arms.**

| Arm | S per grouping | Role |
|---|---|---|
| I | identity everywhere | matched control; must reproduce told-oracle arm L exactly (told +1.64 [1.37, 1.92], reader +0.35 [0.15, 0.57]); otherwise stop |
| F1-all | F1 on affect, image and caption | **primary** |
| F1-affect | F1 on affect, identity on image and caption | secondary |
| F3-all | hierarchy kernel: average-linkage agglomerative clustering of the centred centroids (cosine distance), cut at K, ⌈K/2⌉, ⌈K/4⌉, ⌈K/8⌉ clusters; S_gk = share of the 4 cuts in which g and k share a cluster | secondary |
| F2-all | co-membership: C_gk = Σ_i p_img(i, g) p_txt(i, k) over the 10,000 check rows (not used for fitting), symmetrised; S_gk = clip(C_gk / sqrt(C_gg C_kk), 0, 1), S_gg = 1 | descriptive only |

**Label-free gate** (per arm, for each grouping whose S is not the identity; selection rows): AUC of agreement scores,
same-row pairs (p_img(i)ᵀ S p_txt(i), all selection rows) against 200,000 random ordered pairs from different paintings
(seed 0), ties counted half. The gate passes if AUC_S ≥ AUC_I − 0.005 for every such grouping. The ratio of mean same-row
to mean random-pair agreement is reported beside it, not gated: smoothing shrinks that ratio toward 1 by construction
even when ranking improves, so a rank measure is the operational form of "relative to shuffled pairs must not drop".

**Measures per arm.** Told and reader terms fused on B against their matched counterparts (R@1, gain, either; overall and
per aspect pair), pick accuracy, and paired per-anchor differences of each margin against arm I.

**Reading (primary arm F1-all).**

- **Adopt S** for later steps if the gate passes, at least one of the two paired differences (told margin, reader
  margin; F1-all minus I) has a 95% lower bound above 0, and the other's point estimate is at least 0.
- **Do not adopt** if the gate fails or both lower bounds are at or below 0.
- **Mixed** otherwise: reported for the user to decide.

The margins read evaluation labels through the development episodes, as every development run does; S's formula and γ
were fixed above before any number.

## Execution and outputs

- One script, `run_checks.py`, with stages `a`, `b`, `c` and `--smoke`; CPU only (`CUDA_VISIBLE_DEVICES=`,
  `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`); refuses to overwrite results; checks this file's SHA-256.
- Results in `results/` (gitignored); log `20261115_grouping_step0_checks_log.md` at the end.
- Not done here: stability across Leiden seeds, non-redundancy, any new grouping or source, any test seed.
