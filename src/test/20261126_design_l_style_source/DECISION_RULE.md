# Decision rule: round 7, design L from the style source (ceiling check, seed-42 screen, fresh-seed test), committed before any code

> Spec: `docs/superpowers/specs/2026-10-12-r7-design-l-design.md` @ (spec commit) · Written 2026-10-12 from 02:20
> (Amsterdam) before any implementation script of this round exists; the review's fix wave applied from 02:24, the
> re-review's fixes and the NrKmeans reference (N) from 02:32 · Reviewers: a fresh Opus reviewer
> (`rule_check/opus_rule_check.md`) and a fresh Sonnet re-reviewer (`rule_check/rereview.md`)

**Precedence.** Where this file differs from the spec, from round 3's rule or from round 6's rule, this file governs.
Round 3's rule (`src/test/20261121_round3_affect_gate/DECISION_RULE.md`, SHA-256
2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925; "R3" below) supplies D2 to D15, its §2 metrics and
bootstrap, its §5 threshold procedure and its §6.1 to §6.3 test machinery, unchanged except where this file says.
Round 6's rule (`src/test/20261125_artelingo_held_test/DECISION_RULE.md`, SHA-256
7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724; "R6") supplies the bootstrap draws and the Holm
integer form of its §3. Round 1's rule (`src/test/20261117_reader_fix_csd/DECISION_RULE.md`, SHA-256
613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c; "R1rule") §4.2 items 1 to 6 is the reader recipe.
Statistics, reading a failure, the regression check and re-derivation follow the constitution (C9, C11, C12, C13)
except where §11 says otherwise. Folder and report dates (`20261126`) are sequence numbers; times are Amsterdam.

**Authorisation.** The user's decisions in the `r7 decide` grill of 2026-10-11 and 2026-10-12 (handoff
`docs/superpowers/handoffs/2026-10-12-r7-decide-2-handoff.md` §2; spec §4) and the spec's approval. After its commit
this file changes only with the user's approval, as an amendment row in the spec (hard rule 5), marked before or after
results. Commits go to `main` and are pushed.

**Status.** Seed 42 is development data (C4). Seeds 55, 56, 57 are the test, read once. No held row is read (C5).

## 1. Prior (C8)

Written before any number of this round.
- **Ceiling (K1):** told true styles on style × genre about +0.9 R@1 over its matched control; about 60% that the
  point reaches +0.5 with its lower bound above 0; about 30% that it reaches +1.0. Reference: told CSD +0.134
  [−0.350, +0.595] (spike, `told_A1_224`).
- **Groupings:** removal lowers the AMI with the content grouping and with genre (raw CSD 0.33); the projection (V1)
  and the cuts keep more style than the ridge residual (V5). CLIP heads place the new groups worse than CSD's (85.1%
  image, 40.4% caption), captions above all.
- **Seed-42 gate (S1 to S3):** about 25% overall that at least one of the twelve scorers passes, that is about 40%
  given K1 passes. AFF is only +0.332 over B′(A1) pooled on seed 42, so S1 needs about +0.17 pooled over AFF, which
  in practice means about +0.6 on style × genre while S3 keeps AFF's emotion pairs. Extended AFF more likely than R1 to pass S3; R1 more likely to pass S2.
- **Fresh seeds:** given a pick, about 50% that F1 passes (seed-42 margins shrink after selection); F2 about 30%; F3
  about 25%. Friday GO about 12% overall (0.6 × 0.4 × 0.5).
- **Disclosure:** on seed 42, about 50 label-free variants were read before AFF (R3 §6.11), about 10 in rounds 4 and 5,
  and this round adds 12 candidate scorers (4 reference scorers and 12 random-control scorers pick nothing). Seed-42 margins of a
  pick are inflated; the fresh seeds are the protection.

## 2. Candidates and comparators

| ID | What | Role |
|---|---|---|
| T | true-style grouping, told (§5) | ceiling, decides only K1 |
| R1[Vk], X-AFF[Vk], k = 1 to 6 | R1 and extended AFF on configuration A0+Vk (L13) | candidates (12) |
| R1[R0], X-AFF[R0] | the same on raw CSD (A1) | references, never picked |
| R1[N], X-AFF[N] | the same on N, an off-the-shelf non-redundant clustering of CSD (L7) | grouping-step comparator, reference, never picked |
| R1[Zk], X-AFF[Zk] | the same on Vk's random control | controls, never picked |
| B, B0, B1, BX | B, B′(A0), B′(A1), the scorer's own B′ (L14) | condition-free comparators |
| CF[s] | scorer s's matched counterpart (R3 D9) | condition-free comparator (C2) |
| AFF | R3's fused reader | current best (C3) |

## 3. Definitions (L1 to L16; R3's D-items keep their meaning)

**L1. Paintings and joins.** The 36,518 scorer-train paintings, sorted ids (R1rule §4.2 item 1). A painting's vector
in a source is the vector of its first scorer-train row, as `run_step1.painting_grouping` takes it. One image per
painting is asserted by `run_step1.real_style_features`'s `painting_to_image_one_to_one`, re-asserted for every
source; asserted also: `paintings_with_differing_rows` = 0 for CSD and for the token statistics. Image-level sources
join to rows only through `row_to_image.npy` with the hash checks of `run_step1.real_style_features` (C6), never
`range(n)`. All fits in L3 to L10 use scorer-train rows or paintings only.

**L2. Content grouping.** E2's `image` partition (`src/test/20261031_pseudo_partitions/results/partitions.npz`, SHA-256
cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa), 64 groups; asserted constant within each painting;
c(p) is painting p's group.

**L3. Orth** (Cui, Fern and Dy, ICDM 2007, method 2, on centred vectors). For painting vectors x_p in float64:
x̃_p = x_p − mean over the 36,518 paintings; μ_g = the mean of x̃_p over paintings with c(p) = g; M = [μ_1 … μ_64];
Orth(x)_p = x̃_p − M·pinv(M)·x̃_p (`numpy.linalg.pinv`, default rcond); cast to float32. Asserted:
max |Mᵀ·Orth(x)| ≤ 1e-6 · max |Mᵀ·x̃|. The singular values of M and the rank `pinv` used are recorded in the
grouping record (descriptive).

**L4. Ridge residual.** `sklearn.linear_model.RidgeCV(alphas=numpy.logspace(-2, 3, 6))`, other arguments default,
fitted from the unit-normalised cached CLIP ViT-B/32 image vectors of the paintings (512 values) to the source
vectors; residual = source − prediction. The chosen alpha is recorded.

**L5. CLIP token statistics.** For each of the 61,402 images of `style_csd_vitl/meta.json` (same order and joins), the
CLIP ViT-B/32 weights and preprocessing of the cached `clip_b32` features; asserted before extraction: the image
embeddings recomputed for the first 256 images match the cache with cosine at least 0.9999 each. Hidden states after
encoder blocks 4, 6 and 8 of 12 (the model's `output_hidden_states` at indices 4, 6, 8), the 49 patch tokens (class
token excluded). Per image and block: tokens with L2 norm above 3 × the median of the 49 norms are dropped; then the
per-channel mean and population standard deviation (ddof 0) of the kept tokens. Concatenated in block order, means
before standard deviations: 4,608 float32 values per image, written once to
`/data/SSD2/pre_extract/artelingo/clip_b32_tokstats/` with a `meta.json` holding the joins, their hashes and the
dropped share per block; the dropped shares cover all 61,402 images (features, not labels). Painting preprocessing U: each dimension z-scored over the paintings (population standard
deviation; a dimension with deviation 0 becomes 0), then `sklearn.decomposition.PCA(n_components=256,
svd_solver="full")` fitted on the paintings.

**L6. Grouping recipe** ("Leiden at the defaults"): `run_step1.painting_grouping`'s settings: unit-normalised
painting vectors, `run_sweep.knn_union_graph` with k = 20, `leidenalg.RBConfigurationVertexPartition` at resolution
1.0 and seed 42, then `run_told_oracle.merge_small` at 200 rows in the vector space of the graph; seeds 43 and 44 for
stability (descriptive). **Cut graph with weight w:** the same kNN union graph, each edge (a, b) weighted w if
c(a) = c(b), else 1, and Leiden with `weights="weight"`; otherwise the same.

**L7. Variants.** CSD is `style_csd_vitl/embeddings.npy` (SHA-256
7eefe522383b753bad32f404651103471f9d5b6c41926ee57bfc7386c84818ad).

| ID | Painting vectors | Graph | Family |
|---|---|---|---|
| V1 | Orth(CSD) | L6 | CSD |
| V2 | CSD | cut, w = 0 | CSD |
| V3 | CSD | cut, w = 0.5 | CSD |
| V4 | Orth(CSD) | cut, w = 0 | CSD |
| V5 | ridge residual of U (L4) | L6 | generic |
| V6 | Orth(U) | L6 | generic |

R0 is round 1's `style_csd` grouping (`src/test/20261116_grouping_step1_style/results/step1_group_style.npz`, SHA-256
b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2, 17 groups). A variant with fewer than 2 groups after
merging is dropped and reported; its scorers count as not passing.

N, the grouping-step comparator (reference, descriptive, seed 42 only): ClustPy's NrKmeans (Mautz et al., KDD 2018)
on the CSD painting vectors with two clustered subspaces of 17 clusters each (R0's count), `random_state=0`, other
arguments at ClustPy's defaults. Of its two clusterings, N is the one with the lower AMI with the content grouping
(label-free; ties to the first). ClustPy is installed into the CoSiR env with `pip install --no-deps` at a version
recorded in the grouping record, so torch is untouched. Then L6's merge at 200 rows, L10's heads, L11's reader and
both scorers; no random control. If ClustPy cannot run, or NrKmeans takes over 30 minutes, N is dropped and reported.

**L8. Random controls.** Zk (k = 1 to 6) has Vk's group sizes in paintings. The sorted painting ids are permuted with
`numpy.random.default_rng(7000 + k)`; consecutive blocks of the permutation form groups 0, 1, … with Vk's sizes in
Vk's group order. Rows inherit their painting's group; no merge. T and R0 get no random control; R0's is round 1's
`style_rand` (configuration AR), not rerun.

**L9. True-style grouping T** (ceiling only). `artelingo_aspect_labels(data)["style"]` on the scorer-train rows;
asserted: no code missing, constant within each painting, 27 codes, smallest at least 200 rows (no merge).

**L10. Standard heads.** For each grouping of L7 to L9: an image and a caption head with `fit_one_head`'s recipe
(R3 D2); every group must appear in the 60,000-row draw, otherwise that grouping is dropped and reported. Held-out
accuracies on its 10,000 check rows are recorded. Affect, image and caption keep R3 D2's heads; csd keeps round 6's
`r6_heads` csd head.

**L11. Configuration and reader.** A0+X is round 1's A1 with X in csd's place: the same grouping order, bank blocks,
bank seeds (11700, 11800), bank sizes (98,304 per half), 24 features and model selection (R1rule §4.2 items 1 to 6),
with X's cross-fitted heads per item 2. One reader (two half-readers with their scalers) per configuration. T has no
reader.
- In `build_episode_bank`'s partition dict, X has the key `csd` (any key that sorts between `caption` and `image`
  gives the same result). The builder's blocks must equal round 1's A1 list in `rb_build.BLOCKS["A1"]` (asserted, as
  `stage_bank` does). Class index = position in (affect, image, caption, X).
- If a cross-fitted head of X misses a group, or a bank block of A0+X cannot be filled, X is dropped and reported, as
  in L7; its scorers count as not passing. If X is some Vk, Zk is still built and reported. If X is some Zk, the flag
  of §6 cannot be evaluated for Vk; the report says so. G3 (on A1) still stops the round.

**L12. Thresholds.** τ^X = `numpy.percentile` (linear) at 0, 25, 50, 75 of X's reader's 24,576 seed-42 top-two margins
(both conditions), float64 (`rc_core.thresholds`, R3 §5); frozen afterwards.

**L13. Scorers.** R1[X]: R3 D5 on A0+X with X's reader and the standard heads; R3 D6's R1 gate with τ^X; R3 D8's 224
cells and min-margin cross-fit; counterpart CF per R3 D9. X-AFF[X], extended AFF: the same with gate
g_t^c = 1[m^c ≥ τ^X_t] · 1[π^c ∈ {affect, X}]; its counterpart keeps these gates (R3 D9). AFF: R3's, unchanged.

**L14. Condition-free comparators.** B: R3 D10. B0: R3 D11. B1: `crossfit_condition_free(cos, T_N1u, T_6u(A1),
parity)` with T_6u over affect, image, caption and csd with the standard heads, as round 6 computes it. BX: the same
over A0+X's groupings.

**L15. Bar comparator, bar margin, gain statistic.** For scorer s over a scope: the bar comparator is whichever of
B1, BX, CF[s], B0, B has the largest mean R@1 over the scope (full precision), ties to the earliest in that order;
the bar margin is s minus the bar comparator, paired per anchor; the gain statistic is R3 D12's.

**L16. Integer form.** Per-episode R@1 and gain are multiples of 0.25 (R3 D8.5). For paired per-episode differences d
(as fractions) over N episodes, the point in R@1 points is 25 · Σ(4d) / N, and "point ≥ a" is decided as
Σ(4d) ≥ ⌈a · N / 25⌉. Used below: N = 12,288 and a = 0.5 → 246; N = 4,096 and a = 0.5 → 82, a = 1.0 → 164, a = 0.3
→ 50, a = 0 → 0, a = −0.3 → −49. Intervals are C9's (`cluster_bootstrap`, anchor-painting clusters over the scope's
episodes, 5,000 resamples, seed 42, percentile 95%).

## 4. Stage 0: regression checks (C12)

Before any number of stages 1 and 2. Each must hold exactly; a failure stops the round (§9).

- **G1.** AFF on seed 42, produced by the per-seed cross-fit path of §7.3 (or by the round-7 X-AFF function with
  allowed set {affect}, A0's reader and R3's τ), must choose fused cells 39 and 119, counterpart cells 149 and 10,
  and σ* = 0 on both halves, and equal `aff_fused__*` and `aff_cf__*` of R3's `results/seed42_arrays.npz` per anchor;
  pooled 19.137 (+0.700 over B0, +0.332 over B1).
- **G2.** Told A1 by the reused told route (the spike's one-hot probabilities through round 2's fusion, 224 cells)
  equals `fused_r1` and `margin_r1` of `sp_cand_told_A1_224.npz` per anchor (style × genre 21.82, margin +0.134).
- **G3.** The round-7 reader path run on A1 gives round 1's A1 half-readers (the same chosen C per half; on the
  seed-42 features `predict_proba` within 1e-12 absolute of `rb_reader_A1.pkl`'s), and R1[R0] by the round-7 scorer path equals
  `fused_r1` and `margin_r1` of `sp_cand_R1_A1_224.npz` per anchor. Its intermediate outputs are also compared
  exactly with round 1's (`20261117_reader_fix_csd/results/`): the cross-fitted posteriors with
  `rb_heads_{affect,image,caption,csd}.npz`, the banks with `rb_bank_A1_half{0,1}.npz`, and the chosen C per half
  with `rb_reader_A1.json`'s `halves.<h>.chosen_C`. "Exactly" in G1 to G4 means `numpy.array_equal` for integer and per-anchor R@1 arrays
  (bank episode indices, chosen C and cells, picks, fused and counterpart R@1, margins); float arrays (posteriors, bank
  features, probabilities) agree within 1e-12 absolute. Runs use 8 threads and the pinned scikit-learn 1.6.1 and
  numpy 2.2.6.
- **G4.** B, B0 and B1 come from the per-seed `crossfit_condition_free` path (§7.3), not from `picks_seed42.json`. B
  and B0 on seed 42 equal R3 D10 and D11 (18.341064453125, 18.436686197916664); AFF minus B1 per episode equals
  `diff__S1` of round 6's `results/seed42_per_episode.npz`.

## 5. Stage 1: the ceiling check

- **Setting.** Configuration A0+T with T's standard heads; told map emotion → affect, style → T, genre → image; the
  told route of G2 (one-hot probabilities, so every gate is open).
- **Statistic.** On the 4,096 style × genre episodes of seed 42: d = fused minus counterpart R@1, paired per anchor;
  point and 95% interval (L16).
- **K1.** STOP if Σ(4d) < 82 or the lower bound is at or below 0. WEAK if 82 ≤ Σ(4d) < 164 and the lower bound is
  above 0. CLEAR if Σ(4d) ≥ 164 and the lower bound is above 0.
  - STOP: design L ends. No stage 2 number, no seed built; FV is NO-GO; the run writes its report.
  - WEAK: the run goes on; the report's first line flags it, and the run chat sends a note to the user's phone.
- **Written** to `results/ceiling_verdict.json` (point, interval, Σ(4d), zone, this file's SHA-256, the time) before
  any stage 2 number on seed 42. Building groupings, heads and readers on scorer-train rows may run before it.
- **Descriptive beside it:** told-T fused R@1 on style × genre minus B1 and minus BT (B′ of A0+T); T's head
  accuracies; the told margins and gains on the other pairs; T*, the same
  statistic with each seed-42 item's T posterior replaced by the one-hot of its true style code (label placement).

## 6. Stage 2: the seed-42 screen

For each of the twelve candidates s (§2), with the comparators of L14 and L15 on seed 42:
- **S1** (C10, pooled over 12,288 episodes): bar margin Σ(4d) ≥ 246; bar-margin lower bound above 0; gain-statistic
  lower bound above 0.
- **S2** (style × genre, 4,096 episodes): d = s − CF[s], Σ(4d) ≥ 0.
- **S3** (emotion × style and emotion × genre, 4,096 each): d = s − AFF, Σ(4d) ≥ −49 on each pair.
- **Pass** = S1 and S2 and S3.
- **P1, the pick.** Per family, among the passing scorers, the largest pooled bar margin (compared on Σ(4d) over the
  same 12,288 episodes); ties to the larger integer sum of the gain statistic, then to the earlier in the order V1 to
  V6, R1 before X-AFF. Labels and AMIs never enter S1 to S3 or P1; label AMIs are computed only after
  `gate_verdict.json` is written.
- **Chains.** n = the number of families with a pick. n = 2: two chains, each at one-sided α = 0.0125. n = 1: one
  chain at α = 0.025. n = 0: design L ends; FV is NO-GO. The report states "promising" (spec §6) if some candidate
  has, on style × genre, Σ(4·(s − AFF)) ≥ 50 and Σ(4·(s − CF[s])) ≥ 0, and a pooled bar margin with Σ(4d) > 0.
- **Random-control flag.** If the random control of a pick's grouping passes S1 to S3 with the pick's scorer kind, the
  report's first lines flag it (spec D23). The pick stands.
- **Written** to `results/gate_verdict.json` (every candidate's S1 to S3 quantities and integer sums, passes, picks,
  n, α, this file's SHA-256, the time) before any fresh seed is built. References and controls get the same quantities
  in a separate key; they pick nothing.

## 7. Stage 3: the fresh-seed test

1. **Sensitivity,** after `gate_verdict.json` and before any build. For each check of each chain, R3 §6.1's
   decomposition on the pick's seed-42 per-episode differences, with n the seed-42 episodes of the check's scope
   (12,288 for F1 and F2; 4,096 for F3). Detectable margin x = (z(1 − a) + 0.8416) · SE, with a = α/5 for F1 and
   a = α for F2 and F3: multipliers 3.649 and 3.083 at α = 0.0125, 3.417 and 2.802 at α = 0.025. Written to
   `results/sensitivity.json`; used only to read a failure.
2. **Build.** Seeds 55, 56, 57 as R3 §6.2 (`run_baselines.py --episodes-seed <s>`, no `--overwrite`, one repeat after
   a crash, build logs not opened before the verdict). The 9 new per-pair episode SHA-256s must differ from each other,
   from every per-pair SHA-256 in `baselines_seed{42,43,45,47,48,49,50,51}.json` and from held ledger row H5's nine;
   `codes_provenance.json` keeps its SHA-256 across each build. A failure stops the test. After the check the seed
   ledger gets 55, 56, 57 as "test, round 7" and "58 and later" as free.
3. **Frozen** from seed 42: L3 to L13's fitted objects (transforms, groupings, heads, readers and scalers, τ^X), the
   gates, the 224-cell family, the picks, and AFF's R3 parts. **Rerun** on each seed's own parity halves: B, B0, B1,
   BX, the readers' probabilities, the gates against the frozen thresholds, σ*, the fused min-margin and the
   counterpart max-R@1 cross-fits (AFF as R3 §6.3).
   - **Code path.** Round 3's per-seed pipeline (`run_gonogo.EvalContext`, `r3_bundle.build_bundle`,
     `r3_fusion.run_family`), used through `r7_` copies that change the seed guard (allow 42, 55, 56, 57 and smoke
     9001 to 9003) and add the csd and X posteriors, and nothing else. Round 6's scoring modules are used only where their guard
     admits seed 42 (e.g. `r6_heads` for the csd head), never for fresh-seed scoring.
4. **One look.** Until `results/test_verdict.json` is written (with this file's SHA-256 and the time), only these are
   computed: per-seed bundles and reader arrays; the per-anchor arrays of each pick, its counterpart, B, B0, B1, BX
   and AFF; chosen cells and σ*; the pooled counts and points of F1 to F3. No per-seed or per-pair number before it.
5. **Checks,** per chain with pick s and level α, pooled over seeds 55, 56, 57. Draws: R6 §3's (clusters = anchor
   paintings over the 36,864 pooled episodes, seeds in order 55, 56, 57; `default_rng(42)`; chunks of 250), the same
   draws for every F1 check and F2; F3 uses the same procedure over the 12,288 pooled style × genre episodes. n_j =
   the number of the 5,000 resample means at or below 0, decided on the integer 4 · Σ cluster sums.
   - **F1:** on R@1, five checks: s − B, s − B0, s − B1, s − BX, s − CF[s]. Holm: order by n_j ascending, ties in that order; the
     k-th (k = 1 to 5) passes if every earlier one passes and 80 · (n_(k) + 1) · (6 − k) ≤ 5,001 at α = 0.0125, or
     40 · (n_(k) + 1) · (6 − k) ≤ 5,001 at α = 0.025. F1 passes if all five pass.
   - **F2,** tested only if F1 passes: R@1, s − AFF; passes if 80 · (n + 1) ≤ 5,001 (α = 0.0125) or
     40 · (n + 1) ≤ 5,001 (α = 0.025).
   - **F3,** tested only if F2 passes: R@1 on style × genre episodes, s − B1; the same criterion as F2.
   - Each check's 95% interval and its interval at its own level are reported; the integer form decides.
6. **Readings (C11).** A failed check with point above 0: "inconclusive at a detectable margin of x" (x from item 1,
   the realised half-width beside it). Point at or below 0: "<pick> did not beat <name> on fresh episodes". An F1
   check that Holm stopped is reported as "not reached: Holm stopped at <check>", with its C11 reading. F2 and F3 not
   reached: "not tested", point and 95% interval only.
7. **FV, the Friday verdict.** GO if F1 passes in at least one chain: the method paper with design L; the report names
   each chain's last stage passed. NO-GO otherwise, and after a K1 stop or n = 0: the ArtELingo paper with AFF. A pick
   replaces AFF as current best only if its F2 passes (C3). If both chains reach GO, the user chooses which one the
   paper carries, at the next decide chat.
8. **Claim licensed** (C14). F1: "<pick> beats B, B′(A0), B′(A1), its own B′ and its matched control on R@1, pooled
   over the three aspect pairs, on new episodes from the same selection paintings." F2 adds AFF; F3 adds B′(A1) on
   style × genre episodes. Not transfer to new paintings and not a margin on each pair; a paper claim needs a held
   read (C5). §1's disclosure accompanies every number of a pick.
9. **After the verdict** (descriptive): per-seed and per-pair numbers, gain statistics, the frozen-cell line (the cell
   chosen on seed-42 tune half h scores parity 1 − h), pick accuracy, gate-open shares, the pick's AMIs.

## 8. Outcome-to-action table

| Outcome | Action |
|---|---|
| G1 to G4 fail | Stop; trace the cause; report and phone note; nothing improvised |
| K1 STOP | Design L ends; FV NO-GO; report |
| K1 WEAK | Go on; flag at the top of the report; phone note |
| n = 0 | Design L ends; FV NO-GO; report, with "promising" if it holds |
| n = 1 or 2 | Sensitivity, builds, F1 to F3 per chain |
| F1 passes in a chain | FV GO; F2 and F3 decide the claim's extent |
| F1 fails in every chain | FV NO-GO; C11 readings |
| A bug found after a verdict file exists | Correcting code so that it matches this file is not a change of the rule. The verdict file is never overwritten: the corrected run writes `<name>_fix<n>.json` beside it, the old file is kept, the log records the cause and the numbers before and after, both are reported, and the user decides which stands |
| Not complete by Thu 2026-10-15 23:59 | No partial verdict; FV is NO-GO for Friday (the ArtELingo paper with AFF); the user is told; design L resumes after Friday only with the user's approval |
| Anything this file does not cover | The run stops with its report and a phone note; the user decides |

## 9. Order of work, re-derivation and process

- **Order:** this file committed → G1 to G4 → K1 (groupings, heads and readers may be built alongside) → S1 to S3
  and P1 → sensitivity → smoke → builds → F1 to F3 and FV → descriptive.
- **Re-derivation (C13):** before each of `ceiling_verdict.json`, `gate_verdict.json` and `test_verdict.json` is
  written, code written separately from the pipeline (a Codex job or a fresh subagent) recomputes from the stored
  per-anchor arrays: K1's Σ(4d) and interval; S1 to S3 and P1 for all twelve candidates; F1 to F3's n_j, Holm order
  and passes, with its own bootstrap loop written from this file. Integer sums and counts must agree exactly,
  intervals within 1e-9; a disagreement stops that step until traced. The per-anchor arrays themselves are
  re-derived in three phases; each must equal the pipeline's exactly:
  - **Phase K,** before `ceiling_verdict.json`: separate code recomputes T's told per-anchor arrays (style × genre
    fused and counterpart) from the stored T head posteriors with the 224 cells and both cross-fits.
  - **Phase A,** before `gate_verdict.json`: separate code recomputes, for all 12 candidates on seed 42, the gates
    from the stored reader probabilities and τ^X, the 224-cell integer statistics, σ*, both cross-fits, and the
    per-anchor arrays of s, CF[s] and BX. It may import the fitted objects, `zscore_rows`, `crossfit_condition_free`
    and `cluster_bootstrap`, as R3 §8 lists.
  - **Phase B,** before `test_verdict.json`: the same for each pick, its CF, B, B0, B1, BX and AFF on seeds 55 to 57.
  - Groupings and transforms are not re-derived; their checks are L3's assertion, the C6 hash checks and the
    recorded group sizes.
- **Read only.** Reused folders are read only. Their functions that write files (`sp_run.run_one`, `sp_run.main`,
  `run_r2_fusion.run_candidate`, the `rb_build.py` stages, `common.save_candidate`, `common.write_json_once`) are not
  called. The round-7 code calls the pure functions (`RF.terms_from_probs`, `K.thresholds`, `RF.fusion_pass`,
  `per_anchor`, `C.evaluate_fused`). G2 and G3 load their references before any round-7 write, and the SHA-256s of
  §10 are asserted again at the end of stage 0. One exception: `run_baselines.py` writes the seed builds and the smoke
  episodes into `20261030_aspect_baselines/results/` (and its `smoke/`): new files only, never `--overwrite`, with
  §7.2's SHA-256 checks.
- **Modules** of this folder start with `r7_` (or `run_r7_`, `test_r7_`).
- **Smoke,** after the sensitivity and before any build: an end-to-end run of the stage-3 code on smoke seeds 9001 to
  9003 (64 episodes per pair, `run_baselines.py --smoke --episodes-seed <s>`, whose episodes land in its own
  `results/smoke/` as in rounds 3 and 6; the round's smoke outputs go under the round's `results/smoke/`). It prints no metric: only shapes, counts, file names and the pass or fail of each assertion. It
  also runs one deliberate wiring mutation, chosen in the ticket, that must make a named assertion fail; if it does
  not, the run stops before any build. Smoke seeds are never results (C4).
- **Unattended:** build and run chats do not wait for the user. A stop ends the run with its report and a phone note.
  Correcting code so that it matches this file is not a change; it is logged with the numbers before and after.
- **Resources:** extraction on the local GPU under `flock -n -o -E 75 /tmp/gpu0.lock`; CPU work at 8 threads; every
  Python call with `PYTHONDONTWRITEBYTECODE=1`.

## 10. Inputs (read only; SHA-256 asserted before use)

| File (under `src/test/` unless absolute) | SHA-256 | Used for |
|---|---|---|
| `20261121_round3_affect_gate/results/seed42_arrays.npz` | 5ea4b09a4161a5eac6ca78942cf0f4b99b9c634edca651fba4c2689c7c24ab8a | G1; AFF in S3 |
| `20261119_no_caption_csd_spike/results/sp_cand_told_A1_224.npz` | 08ac1ee86b46375c13c08420e192877eb8c67b54f14304f0a2f0db5e96cd3417 | G2 |
| `20261119_no_caption_csd_spike/results/sp_cand_R1_A1_224.npz` | 778a769bbdc55705669bf1f5c0bbd6db7ec5c541dac7abf713301dd5c6478d66 | G3 |
| `20261117_reader_fix_csd/results/rb_reader_A1.pkl` | 4e1e4e23c20333b839aa1f92942891bdef51955317640d1014db5d097b7060f6 | G3 |
| `20261125_artelingo_held_test/results/seed42_per_episode.npz` | e93fd7b2b18e788d80605caaa854afa8a25ebda735728f5db956dc322a81b531 | G4 |
| `20261031_pseudo_partitions/results/partitions.npz` | cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa | L2 |
| `20261116_grouping_step1_style/results/step1_group_style.npz` | b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2 | R0 |
| `/data/SSD2/pre_extract/artelingo/style_csd_vitl/embeddings.npy` | 7eefe522383b753bad32f404651103471f9d5b6c41926ee57bfc7386c84818ad | CSD |
| `/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json` | 6013b2f801eb1e6ae104e9fb3c71a377890bb9aab58e4cb27991c269febdf13b | joins (C6) |
| `20261030_aspect_baselines/results/codes_provenance.json` | 8e6a517b73610d4d21b42e2eb39f6fd4118dc4132049364c8e89eecb7716bfaf | build check (§7.2) |

Reused code (read only, §9; SHA-256 asserted before use and again at the end of stage 0). Round 6's modules serve
seed 42 only; round 3's per-seed pipeline is used through `r7_` copies whose only change is the seed guard (§7.3).

| File (under `src/test/` unless stated) | SHA-256 | Used for |
|---|---|---|
| `20261125_artelingo_held_test/r6_common.py` | cacbe204add6a6141978af20599f8b2431da78c14040127a373be2a1fd77f3f3 | imported by round 6's modules |
| `20261125_artelingo_held_test/r6_stats.py` | fde578723d7e3e7e8eeacb3a4b0e1902c81cc0e95b4ba154b9ab9ba975ae8aa1 | R6 §3's draws and Holm counts, if imported (else copied into `r7_`) |
| `20261121_round3_affect_gate/r3_stats.py` | 846a4f5b3175302280c20fa415cfff4b0652522d06aa4fc46290f1a1db5de54d | round 3's bootstrap helpers, if imported (else copied into `r7_`) |
| `20261125_artelingo_held_test/r6_heads.py` | f157e2eda66350047b24c09e06844dc825d3e4061c0f334b76b8d5309fc4e2af | csd head (L10), all seeds (no seed guard; B1 needs its posteriors on 55 to 57) |
| `20261125_artelingo_held_test/r6_bundle.py` | 6b638e5bc2f582358216aea4104ac79de9a78e440560276bae7c42646f030cd9 | seed-42 reference only |
| `20261125_artelingo_held_test/r6_score.py` | d44bbf150cc8773df9c792baaa232a2825eed4d7605eb95cdf3efe7e3a1f4948 | seed-42 reference only |
| `20261125_artelingo_held_test/r6_context.py` | 0ce2086a2cefa9e75b1c6b8b5733e5052130df0e0cd6f012b3f874de220ba7d7 | seed-42 context of the round 6 modules |
| `20261121_round3_affect_gate/r3_common.py` | b1a60b1fd801bf9147d0bd58ae6da806d12fbcc982796135454f31dce3836f86 | paths, round 1 and 2 imports (§7.3) |
| `20261121_round3_affect_gate/r3_bundle.py` | bef50cbbef17f6c40d53c705a060967950d0690168c4b046b1cc3f0dd9fa0e9d | source of the `r7_` per-seed bundle copy (§7.3) |
| `20261121_round3_affect_gate/r3_fusion.py` | ce51a819157842035fdfde488f042c25fa4764ed1b1c363728d5e3837e314637 | source of the `r7_` family copy (§7.3) |
| `20261101_aspect_factor_gonogo/run_gonogo.py` | 8353dbc118cf619494a8e8e5d83bee4f2034ebbaa52946be0bfce92e67ea63dc | `EvalContext` (§7.3) |
| `20261108_new_method_quick_checks/run_checks.py` | 5dee3bdf44e526dd6b22580bb5302b41e133e76d4d458224011ca3107dbc4610 | `model_inputs` (R3 §4) |
| `20261108_new_method_quick_checks/run_n6.py` | 57046023af8f4352dc90b563def5e5e27ba35e12489bf7f858a6d0580902f4d0 | posteriors, B (R3 §4) |
| `20261119_no_caption_csd_spike/sp_run.py` | d2bf443b21f5af4ad6a343e2b24eac997aeb0e651f6f0d9128919dfd660c995f | told route (G2, K1); writers not called |
| `20261118_reader_fix_round2/run_r2_fusion.py` | fe6fb9ff870c3942bef98327ff23894782522e516fc100cb1635aeec6e7e5b65 | `terms_from_probs`, `fusion_pass` |
| `20261118_reader_fix_round2/r2_fusion.py` | 1406e8e71629e74c468f8b009a110e49aa05c481ed52cc1b8d99b59f87016027 | cell statistics, cross-fits |
| `20261118_reader_fix_round2/r2_common.py` | 402c722cc8a94f056ed04b513a1b751f54ac1a8d2f04e91bd0a1226f0b629878 | round 2 helpers (spike import) |
| `20261117_reader_fix_csd/common.py` | 99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7 | terms, `evaluate_fused`; writers not called |
| `20261117_reader_fix_csd/rc_core.py` | e649fff5d8253c8ba7aae0dbfff7a68321cef5fe3caa06d7b325a27fe904185c | `thresholds` (L12), gates |
| `20261117_reader_fix_csd/rb_build.py` | 63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b | reader recipe, `BLOCKS` (L11); stages not called |
| `20261117_reader_fix_csd/rb_eval.py` | 1826e65fd73b8c6cecae33b5df48503bf2cf33e067fc291d5c078c51184e0b14 | reader features, half-reader probabilities |
| `20261117_reader_fix_csd/rb_features.py` | 1e758f6d5de1b1dd246592f4045e0ae1dcc520c1d6012540a9e69b3314fb6caf | features, `bank_blocks` |
| `20261116_grouping_step1_style/run_step1.py` | f4bea509a6ca4fbb48e60823b4add1ba89570b626416127aaf57db66d63018c0 | painting vectors, joins, grouping (L1, L6) |
| `20261111_community_told_oracle/run_told_oracle.py` | b7ae64175f259bf17edf503a57c5befd47edb938bd4d241aa732abf14ffe464d | `merge_small`, `fit_one_head` (L6, L10) |
| `20261112_community_sweep/run_sweep.py` | 13314f6f17edc4fe8665e229608300f6ff38a50025ce321823260f3fab0681b1 | `knn_union_graph` (L6; `rsw` in `run_step1`) |
| `20261030_aspect_baselines/run_baselines.py` | 26508dde35f77850c4a80e98d08b0ea6576638577c84d3e96c39660ef02e63c4 | seed builds and smoke (§7.2, §9) |
| `src/train/pseudo_partitions.py` | 8901cc668fe3fd1834dc2f0a00851030dabb1135bd034a26646629c3621e71f8 | `build_episode_bank` (L11) |
| `src/eval/aspect_metrics.py` | ad95485ef887089d74bb685e2cdb1aab7057b87ad62d926f5c252219f4c6c2f1 | `per_anchor`, `cluster_bootstrap` |
| `src/eval/aspect_quick_checks.py` | fcd116f529082f81e163092921a91909bd358eb3347695a959c68f814dc7d985 | `crossfit_condition_free`, `centered_term` |
| `src/eval/aspect_nested.py` | fadd1fd5d6fe8920a19ff8ccddd91b0e487f286bf908428c31183c503b4e7fa3 | `_zdict` (`r3_fusion`) |
| `src/model/aspect_rule.py` | f1d83bd9a7eb535a054c3221d565b8b38514f1e85590bbde0fd2be8cd502fe34 | `zscore_rows` (§9 phase A) |

R3 D15's inputs keep their asserted SHA-256s.

## 11. Exceptions to the constitution

| Principle | Exception | Why | Simpler option rejected |
|---|---|---|---|
| C9 | On the fresh seeds, Holm across F1's five checks and a fixed sequence F1 → F2 → F3 per chain, at one-sided 0.0125 per chain (0.025 for a single chain), decided by R6 §3's integer count form, in place of each 95% lower bound; GO needs F1 in at least one chain, and F2 and F3 set only the claim's extent | the user's choice (spec D10, D11; Q6 and Q7): every "beats" claim shares one family-wise one-sided 0.025 | each check at 95%: up to 14 claims uncorrected |

## 12. ID glossary

| ID | Meaning |
|---|---|
| R1rule, R3, R6 | rounds 1, 3 and 6 decision rules (precedence line) |
| L1 to L16 | this round's definitions (§3) |
| V1 to V6, R0, Z1 to Z6, T | variants, raw CSD reference, random controls, true-style grouping |
| A0+X | round 1's A1 configuration with grouping X in csd's place |
| R1[X], X-AFF[X], CF[s] | R1 and extended AFF on A0+X; scorer s's matched counterpart |
| B, B0, B1, BX | B, B′(A0), B′(A1), B′(A0+X) |
| G1 to G4 | regression checks (§4) |
| K1, T* | the ceiling's zones; the label-placement ceiling (descriptive) |
| S1 to S3, P1, n, α | seed-42 gate items, the pick, the number of chains, the chain level |
| F1 to F3, FV | fresh-seed stages, the Friday verdict |
| x | detectable margin (§7.1) |
| N | NrKmeans on CSD, the grouping-step comparator (L7) |
| Phase K, A, B | the C13 re-derivations before `ceiling_verdict.json`, `gate_verdict.json` and `test_verdict.json` (§9) |
