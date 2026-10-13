# Decision rule: round 9, AFF's scope (two more backbones on ArtELingo, ArtELingo-28 on three backbones), committed before any scoring code

> Spec: `docs/superpowers/specs/2026-10-13-r9-aff-scope-design.md` @ b1319d8 · Written 2026-10-13 15:33 before any implementation script exists · Reviewer: Opus fresh review (rule_check/rule_review.md), fixes applied 2026-10-13 15:56; scoped re-review (Sonnet) and its fixes 2026-10-13 16:03

**Precedence.** Where this file differs from the spec, from an earlier rule or from a ticket, this file governs (C7).
It cites two earlier rules by section instead of restating them:
- "R3 rule": `src/test/20261121_round3_affect_gate/DECISION_RULE.md` (SHA-256
  2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925). It defines AFF, CF, B, B′ and their parts (D1 to
  D15), the metrics and the bootstrap (§2), and the agreement of a re-derivation (§8).
- "R6 rule": `src/test/20261125_artelingo_held_test/DECISION_RULE.md` (SHA-256
  7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724). It defines the seven checks and their Holm integer
  form (§3), the NO-GO reading (§4), the held pipeline (§5), the pre-read checks (§6), the read and its refusals (§8)
  and the detectable margin (§8.2).

Statistics, failure reading, the regression check and re-derivation follow C9, C11, C12 and C13, except where §4
says otherwise. Folder and report dates (`20261202`, `2026-12-02`) are sequence numbers. Times are Amsterdam.

**Authorisation.** The user's decisions in the grill of 2026-10-13 (spec §4, D1 to D34) and the spec's approval.
After its commit this file changes only with the user's approval.

**Status.** AFF is frozen (R3 rule; H5 GO on CLIP ViT-B/32). This round reads it five more times, one read per
method, backbone and split (C5 amendment 2; D17): H6 and H7 on ArtELingo's held rows (Qwen3-VL-Embedding-2B, SigLIP 2
so400m/14-384), H8 to H10 on the ArtELingo-28 read set (CLIP, Qwen, SigLIP 2). Seed 42 on selection rows is the only
development data. Nothing is fitted, picked or tuned on held rows or on ArtELingo-28, except the label-only reads disclosed in §1.

## 1. Prior (C8)

Written before any score of this round, restating spec §3.1. AFF's mechanism is not tied to CLIP. The affect
grouping comes from GoEmotions on English captions. The heads, the reader and the fusion are re-fitted on any
backbone. We expect a positive margin over B′(A0) on both new backbones, likely +0.2 to +0.6 R@1, because a stronger
encoder lifts B as well. A GO on each new backbone is about even odds. On ArtELingo-28 the small claim is about even
odds per backbone, and the full claim is less likely, from power. A surprise would be AFF below B′(A0) on a backbone
whose gate opens as often as on CLIP (54.5% at τ_0 on held), or the recipe switching AFF off.

**Disclosure** (spec §8). AFF was found among about 50 variants on CLIP's seed 42. On each new backbone only the
automatic seed-42 pick runs: 224 cells plus the condition-free and baseline picks, with no human choice. ArtELingo's
held rows were read five times (H1 to H5). 248 of the 746 ArtELingo-28 read paintings are among them. Before this
rule, its writers read ArtELingo-28's labels (no feature, no score) to set the value sets and to choose the episode
design (§5.1 items 7 and 8). The round 9 recipe and episode design were made after H5. Some languages are thin
(§5.1); every per-language number carries that note.

## 2. Arms (per backbone, spec §3.2)

Every arm is built from that backbone's features by the fixed recipe (D3). CLIP's arms are round 6's pinned files.
`pins_clip.json` lists them (R6 rule §6 item 1's inputs, its §5 item 4 picks) with their SHA-256s; it is
committed before H8's ledger row.

| ID | What | Role |
|---|---|---|
| AFF | R3 rule D5, D6, D8; cells by the min-margin cross-fit pick on seed 42 | candidate |
| CF | R3 rule D9, cells picked on seed 42 | matched control (C2) |
| COS | cosine on the backbone's features | comparator |
| RCA | RCA fused with cosine, λ by `crossfit_lambda` on seed 42 | comparator (C1) |
| B, B0 | B and B′(A0) (R3 rule D10, D11), `crossfit_condition_free` picks on seed 42 | comparators (C1) |
| B1, R1, PM | B′(A1) (R6 rule §2), R1, the other scorers of `run_baselines.py` | reported, no check (D27) |
| DTS, FT, MLLM | round 6's external comparators | CLIP only, H5's numbers; not run (D27) |

## 3. Stages before the reads

### 3.1 Stage 0a: regression (C12)

The generic chain (`r9_*` modules, generic over backbone and dataset) loads CLIP's pinned files and runs round 6's
seed-42 regression.
- Source: `src/test/20261125_artelingo_held_test/results/regression_seed42.json` (SHA-256
  ca168eb628f6a51e9afe779b0194088d05b293417bac04617a0d8a428d3984be; 131 items, 0 failed). It holds 39 seed42_arrays
  items, 57 per_anchor_seed42 items and 35 scalar or hash items.
- Also `seed42_pass_counts.json` (SHA-256 5618c59a3907af67bbcced0f29e0db8462b12ca4fb2e40a69e4be537de29d5e5): P1 to P7
  n_j, points and intervals.
- **Gate.** Every item equal: scalars exactly; arrays by shape, dtype and `np.array_equal` (round 6's
  `array_item`). Every field of `seed42_pass_counts.json` equal, except the run-specific fields `rule_sha256`, `runner_sha256`,
  `module_sha256`, `time` and `input_sha256` (they name the code and the clock). `coef_sha256`, `episodes_sha256`,
  `mode`, `seeds`, `n_episodes`, `n_clusters`, `checks`, `holm_order` and `secondary` must be equal. `.npz` file hashes are not compared (`np.savez`
  writes zip timestamps).
- **Fail.** Stop, trace, ask the user (D34). Nothing that uses the chain runs until 0a passes.

### 3.2 Stage 0b: recipe replay on CLIP (D32)

The fixed recipe re-fits every part above CLIP's encoder from scratch.
1. **Parts bit-equal** to CLIP's pinned files (all CPU-made except the content graph, which `src/model/graph.py`
   builds in fp16 on the GPU: it runs locally on the RTX 3090 and is bit-equal only there; user, 2026-10-13): image k-means 64 (`prepare.npz` `clip_image`), caption
   k-means (E2 `partitions.npz`), the content graph (`20261016_factor_learning_grid/cache/graph.npz`, SHA-256
   485cb7fe12b44bd70bf45ae9c131fc347306c76fc7c6cf75704b95edbb393de8), bank AIC
   (`20261031_pseudo_partitions/results/bank_AIC.npz`, SHA-256
   b6c662ef4b79684bc86ee422056782b8cfec278c8c6c164044b9cffbcd59fc41), every head (`n6_posteriors`, the affect heads
   with arm-L identity, `step1_heads_style`), the reader (`rb_heads_*`, `rb_bank_A0_half{0,1}`, `rb_reader_A0.pkl`,
   with the C picked by 5-fold CV: 1.0 and 100.0), τ (`rc_tau.json`), the RCA and PM fits and λ picks.
   Arrays are compared element by element, not by file hash. The head replay runs locally at `OMP_NUM_THREADS=8` and
   `MKL_NUM_THREADS=8`.
2. **A3** (`train_factors.py`, its chosen settings, D18). Local run L1.
   - Identity gates in every case: the config equals the stored one with `aspect_tau_fixed: False` added; the inputs
     are equal; the step-1 loss is within 1e-5 relative of 20.68084716796875.
   - Case 1: L1 is bit-equal to the pinned checkpoint: every state-dict tensor equal (`torch.equal`, same dtype and
     shape), the config as the identity gate says. Every downstream item must then be exact (131 items).
   - Otherwise a second local run L2. Case 2: L1 ≠ L2 (GPU non-determinism shown). L1 and L2 each pass item 3.
   - Case 3: L1 = L2 ≠ pinned. The cause is traced and the user decides.
   - If the new backbones' A3 will train on DAS6, a cluster run C1 passes the identity gates and item 3 against the
     pinned files. If C1 fails, those A3 runs train locally.
3. **Tolerance (case 2).** What A3 feeds through T_N1u: B, B′(A0), B′(A1), σ*, and the AFF, CF and R1 cells.
   - Picks exact: AFF cells 39 and 119, CF 149 and 10, R1 cells 116 and 119 (fused) and 58 and 123 (counterpart),
     σ* 0 and 0, B, B′(A0) and B′(A1) picks as `picks_seed42.json` (SHA-256
     df1fff781136c001343ec88264b014ca6a2448d71e8c4f33e230e9821deb47c6). τ, RCA and PM exact (CPU-made).
   - Seed-42 R@1 of B, B0, B1, AFF and CF, and the P1 to P7 seed-42 points, each within 0.05 R@1 points of the
     stored value. That is about 25 net ranking flips of 49,152, about 1/4 of the narrowest seed-42 half-width (P3,
     0.212).
   - Episodes with a changed per-anchor R@1 are counted and reported. This tolerance is a C12 exception (§12). C1 is held to it too, because the cluster GPU differs.
   - Stage 0b records the SHA-256s of the recipe scripts it ran.
4. **Fail.** Any 0b failure: stop, trace, ask the user. The recipe is not used until it replays. Reads it would feed
   (H6, H7, H9, H10) are "dropped under the rule" (D25) if the user so decides. H8 uses CLIP's pinned files, never
   the replay, so a failed 0b does not block H8.

### 3.3 Stage 1: preparation

D31 allows extraction and translation before this rule. They read no label. The label mapping, the read-set checks
and the value sets read ArtELingo-28 labels only, never a feature or a score. Nothing is scored.
- **Stores.** SigLIP 2 on all of ArtELingo (D19) and on SemArt (spec §3.5; counts and finite values asserted).
  SemArt's official val split is development data (D23). The ArtELingo-28 stores per backbone: one image per
  painting (the zip path `images/` + `image_name`; 746 of 746 found) and one caption feature per read row. Row ID =
  the CSV data-row position. Counts, finite values and the `sample_id` joins are asserted (C6). Qwen's
  `artelingo_full` joins `load_artelingo()` by `sample_id` (308,723 ids).
- **Spot check.** 20 rows per store, drawn by `numpy.random.default_rng(0)`, re-encoded locally: cosine ≥ 0.99.
- **Painting join.** For each of the 248 held paintings, the zip image's nearest CLIP image among all paintings of our
  ArtELingo store (61,402) is the same painting. All 248 cosines and each margin to the next painting are reported.
  (User, 2026-10-13: the zip images are re-encoded copies, 600 px, so the same painting scores 0.92 to 0.99; the
  first wording, cosine ≥ 0.99, failed on 196 of 248.)
- **Near-duplicates.** The read paintings whose zip image has a train-side image (the train side is every non-held ArtELingo painting: train and val, 790 + 122 = 912;
  user, 2026-10-13) at CLIP cosine ≥ 0.99 (and ≥ 0.98)
  are counted. The counts are disclosed with every ArtELingo-28 claim; none is excluded (D7). Because of the version
  gap above, the stage-1 record also counts ≥ 0.92, and the report states it with that gap (user, 2026-10-13).
- **Read set.** 746 paintings: ArtELingo-28 paintings minus its 912 paintings that lie on our train side (248 held
  and 498 new). Asserted: no painting is both held and train-side; 74,569 non-English rows in 27 languages; the
  2,000 English rows on these paintings dropped (D21).
- **Labels**, mapped by name to our codes (never re-encoded from the CSV): emotion and style from the CSV; genre from
  `load_wikiart_genre`, asserted equal to the CSV genre on 746 of 746. `something else` and `other` map to −1. The 6
  held paintings whose ArtELingo-28 style differs from ours keep the CSV style; they are listed in the report.
- **Translation.** NLLB-200 3.3B (D20). IsiNdebele and Emakhuwa use proxy codes `zul_Latn` and `swh_Latn`.
  Rows with an empty output are dropped and counted before the episodes.
- **Language flags**, written to `language_flags_a28.json` before any read. A language is flagged if: it uses a
  proxy code; failed outputs, (`empty_outputs` + `outputs_equal_to_source` + `hit_max_new_tokens`) / `rows` of
  `coverage.json` over all its translated rows, exceed 5%; its NLLB source script is not Latin (the script suffix of its `code` field in `coverage.json` is not `Latn`) and
  `outputs_still_in_source_script` / `rows` exceeds 5% (`letters()`: over 50% non-Latin letters); or its median
  output word count on read rows (`len(str.split())` on `translations.parquet`) is below 0.5× or above 2× that of the
  ArtELingo-28 English CSV rows on the 746 paintings (the median is taken after the empty-output drops). Every language stays in the pooled read (D21), except a language set
  aside by §5.1 item 6 (up to two can be). Flagged languages are named in the claim.
- **Pins.** Stage 1 ends by writing V (§5 item 2) and `pins_a28.json`: the SHA-256 of every translation shard's
  `translations.parquet` and `coverage.json`, of each backbone's ArtELingo-28 image and caption stores, of
  `value_sets_a28.json`, `language_flags_a28.json` and the read-set file `read_set_a28.parquet` (746 painting names,
  row IDs, languages, mapped labels, empty-output drops). It is committed in this folder before H8's ledger row.
- **Fail.** Fix and rerun.

### 3.4 Stage 2: the recipe per backbone (D3, D18)

- The split is CLIP's: its leakage groups are passed in and the sizes asserted (D15).
- One A3 run per backbone, seed 42. A crashed run is rerun with the same seed. A completed run is never replaced.
- Every re-fitted file and every seed-42 pick (AFF, CF, B, B0, B1, R1, RCA, PM; τ) goes to `pins_<backbone>.json`
  with its SHA-256, written when the file is made. The pins file is committed before the backbone's ledger row.
  Files pulled from DAS6 are hashed against the node-side pins.
- **Provenance.** For every file of §11's re-fitted list, the pins file also records the SHA-256s of the
  feature-store files it was fitted from and of the recipe scripts that made it. It also lists, per file, the keys
  tested as made and as copied. Three assertions follow:
  - every re-fitted file differs from CLIP's counterpart in each array its fit makes (labels, coefficients,
    posteriors, reader features); arrays it copies from frozen parts equal CLIP's;
  - every frozen file of §11 (split and leakage groups, halves, row draws, affect partition and grouping, CSD
    grouping) equals CLIP's (as arrays);
  - the recipe script SHA-256s equal those of the run that passed 0b. A changed recipe script reruns 0b.
  The chain's fixed CLIP paths (`rb_build.py` `SOURCES`, `run_n6.py` `PARTITIONS`, `run_checks` A3 folder and
  `A3_SHA`) become backbone parameters; the first assertion catches a missed one.
- **Heads.** Every head that a read refits (§6 item 2) is pinned from a local fit at `OMP_NUM_THREADS=8` and
  `MKL_NUM_THREADS=8`. A head fitted on DAS6 is refitted locally before it is pinned. Everything that consumes a head's posteriors (reader
  and heads `rb_*`, practice banks, τ, the seed-42 picks) is made after that head is pinned.
- The pins file also records the backbone's seed-42 numbers that the pre-read regression (§6 item 3) reproduces: the
  mean R@1 of B, B0, B1, AFF and CF, the P1 to P7 points and n_j, and the SHA-256 of the per-anchor arrays (the
  keys of stage 0a's 96 array items where they exist; C-contiguous bytes, dtype and shape recorded).
- **Switched off (D4, D30).** If AFF's min-margin cells chosen on tune halves 0 and 1 both have λ_a = 0, the
  backbone's ArtELingo and ArtELingo-28 rows are recorded as "not run: switched off". No held data is read.
- The seed-42 gate share at the chosen cells is recorded in the pins file.

## 4. The bars

One cluster bootstrap per read, as R6 rule §3 ("Draws"): 5,000 resamples of anchor paintings, `default_rng(42)`,
the same indices for every check. Three seeds are concatenated in order (36,864 episodes). Counts n_j and the
integer comparison are as R6 rule §3 ("Counts").

### 4.1 Stage 3: H6 (Qwen), H7 (SigLIP 2) on ArtELingo held rows (D5)

- **Checks.** P1 to P7 exactly as R6 rule §3, with the backbone's arms.
- **Holm** over the seven, one family per backbone, R6 rule §3's integer form: order by n_j, ties P1 to P7; the k-th
  passes if it and every earlier one pass and 40·(n_(k) + 1)·(8 − k) ≤ 5,001. The largest passing counts are
  n* = 16, 19, 24, 30, 40, 61, 124.
- **Episodes.** H5's: seeds 52, 53, 54, built by R6 rule §5 item 3 on held rows with `value_sets.json`. The rebuilt
  per-pair SHA-256s must equal the nine of ledger row H5 (D16). Intervals as R6 rule §3 ("Intervals").
- **GO** (SC1, SC2): §6 passed and all seven pass. **NO-GO** otherwise, read as R6 rule §4 with x of §7.4.
- No secondary checks. B1, R1 and PM are reported, not tested.

### 4.2 Stage 4: H8 (CLIP), H9 (Qwen), H10 (SigLIP 2) on ArtELingo-28 (D8)

The same checks P1 to P7 on the read set, in two levels, one family per backbone:
- **Level 1:** P4 and P6. Holm over two at one-sided α = 0.025: order by n_j, ties P4 first; the k-th passes if it
  and the earlier one pass and 40·(n_(k) + 1)·(3 − k) ≤ 5,001. n* = 61, 124.
- **Level 2**, tested only if both level-1 checks pass: P1, P2, P3, P5, P7. Holm over five at α = 0.025: ties in the
  order P1, P2, P3, P5, P7; 40·(n_(k) + 1)·(6 − k) ≤ 5,001. n* = 24, 30, 40, 61, 124.
- Serial gatekeeping keeps the family-wise one-sided error at 0.025.
- **Full claim:** both levels pass. **Small claim:** level 1 passes, level 2 does not. **NO-GO:** level 1 fails.
- If level 1 fails, the level-2 checks are reported (point, 95% interval), not tested and not read.
- Every failed tested check is read as R6 rule §4 (C11), with x of §7.4.

### 4.3 Claims licensed (spec §6)

- **H6 or H7 GO:** "on ArtELingo's held paintings, AFF re-fitted by its fixed recipe on frozen Qwen (or SigLIP 2)
  features beats cosine, RCA, B, B′(A0) and its matched control on R@1, with a positive condition gain over its
  control and RCA, pooled over three aspect pairs." Not each pair, not another dataset, not a margin size.
- **N** is the number of languages that entered all nine builds; a set-aside language is named with its pair and seed.
- **Full claim** on a backbone: "with captions from the N languages that entered the read (27 if none is set
  aside), machine-translated to English, AFF trained on English ArtELingo keeps the same sentence on that backbone."
  **Small claim:** "beats B′(A0) on R@1 with a positive condition gain over its matched control". Not native-language
  text, not any single language.
- Flagged languages (§3.3) and set-aside languages (§5.1 item 6) are named beside every ArtELingo-28 claim.

## 5. The ArtELingo-28 read pipeline (what is new beyond R6 rule §5)

1. **Rows and clusters.** The read rows of §3.3 after empty-output drops (the label index of §5.1 item 3 is built
   before the drops). Clusters are anchor paintings (painting name), across languages and seeds.
2. **Value sets.** V = the development set (`value_sets.json`) ∩ `eligible_values(·, groups, pool, 30)` on the pooled
   read pool (all three labels known; groups = paintings). Asserted: emotion 8 (all), style 7 (Expressionism 9,
   Impressionism 12, Northern_Renaissance 17, Post_Impressionism 20, Realism 21, Romanticism 23, Symbolism 24), genre
   8 (codes 0, 1, 2, 4, 6, 7, 8, 9; illustration and nude_painting out). Every eligible value is a development value
   (asserted). V is written to `results/value_sets_a28.json` in stage 1 and pinned in `pins_a28.json` (§3.3).
3. **Episodes.** §5.1. The same episodes serve H8, H9 and H10. H8 builds them; H9 and H10 rebuild them and assert H8's
   per-pair SHA-256s.
4. **Bundle.** As R6 rule §5 items 4 and 5, with the backbone's pinned files (CLIP: round 6's), its ArtELingo-28
   image and caption features, and ArtELingo-28 rows in place of held rows. Every episode row is a read row; features,
   A3 codes and posteriors are NaN outside read rows and finite on them (asserted).
5. **Picks.** Frozen from seed 42 as R6 rule §5 item 4, from the backbone's pins (CLIP: R6 rule §5 item 4's table).
   Tune half h scores read parity 1 − h, by episode index of the concatenated 4,096 per pair.

### 5.1 ArtELingo-28 episodes: language-homogeneous (option B)

<!-- Controller note: option B was taken as the agent default at 15:59 on 2026-10-13 (the user was asked at 15:29 and gave no
answer in 30 minutes). The user may switch to A by an amendment before H8's ledger row is committed. If the user picks
option A, this subsection is replaced: the R6 builder with values=V, min_paintings 30 and one PaintingValueIndex over all read rows (before the
empty-output drops), integer seeds 55 to 57, the anchor-emotion skew of item 8 disclosed. Outside §5.1, option A
changes (fresh review, rule_check/rule_review.md, "If the user picks option A"):
- §1 and §4.3: §1 discloses the skew (in the emotion pairs every anchor's emotion is anger, disgust, fear or
  sadness); every ArtELingo-28 claim carries it. §4.3's claim reads "27 other languages"; set-aside wording goes.
- §3.3 language flags and §5 item 1: the set-aside exception goes; the index is one index over all read rows.
- §6 item 5: the synthetic language column goes; the smoke runs the plain builder (the bundle smoke stays).
- §7.5: "its own §5.1 split, seeds and assembly" becomes its own builder call (values=V, min_paintings 30).
- §8 item 5: "per language" and "without the flagged languages" need a definition for mixed-language episodes
  (e.g. episodes whose members all come from non-flagged languages); the set-aside item goes.
- §10: the "more than two languages set aside" row and the set-aside exception in the assertion row go.
- §12: the second C4 row goes (plain seeds 55 to 57); the C4 smoke seeds become 9001 to 9003.
- §13: Pool_ℓ, n_ℓ and R_ℓ go. -->

Every episode is built from one language's rows. Then `lacks` sees only that language's rows, not the about 100
rows a painting has over 27 languages. The median rows per painting is 1 to 6 by language (English ArtELingo: 5): 1
for Emakhuwa and IsiNdebele, 2 for IsiXhosa, Kazakh and Urdu. Some languages hold a value of V on very few
paintings (Kyrgyz excitement on 2, Kazakh still_life on 3). The spec's "≥ 30 paintings" (§3.4) holds for the pooled
pool only (item 3); this is disclosed with the per-language numbers (§8 item 5).

1. **Languages.** The 27 CSV names in Python sort order, index ℓ = 1 to 27: Burmese, Darija, Emakhuwa, Hausa, Hindi,
   Igbo, Indonesian, IsiNdebele, IsiXhosa, IsiZulu, Kazakh, Kinyarwanda, Korean, Kyrgyz, Malay, Setswana, Swahili,
   Tagalog, Tamil, Thai, Turkish, Urdu, Uzbek, Vietnamese, Yoruba, arabic, chinese.
2. **Shares.** Pool_ℓ = language ℓ's read rows after the drops, with all three labels known. n_ℓ =
   ⌊4,096·|Pool_ℓ| / Σ|Pool|⌋. The remaining episodes go one each to the largest remainders, ties to the lower ℓ.
   Σ n_ℓ = 4,096 (asserted). Every pair has a third aspect, so the shares are the same for the three pairs and the
   three seeds.
3. **Builder.** An `r9_` copy of R6 rule §5 item 3's builder (`values=`), unchanged, called once per language, pair
   and seed on sub-arrays: R_ℓ = ℓ's read rows before the empty-output drops; labels[a][R_ℓ]; groups[R_ℓ] (global
   painting index); `PaintingValueIndex` over these sub-arrays only, so a dropped row's labels still count for
   `lacks` (as round 6's index over all rows); rows = the positions in R_ℓ of the rows kept after the drops;
   n_episodes = n_ℓ; values = (V_a, V_b); min_paintings = 1. Episode arrays map back to global row IDs by R_ℓ[·].
   - Eligibility (≥ 30 paintings) is applied once, on the pooled pool (§5 item 2). With min_paintings = 1 the
     builder's subset guard asserts that every value of V occurs in Pool_ℓ.
   - Each part passes `validate_aspect_episodes` on its sub-arrays. Every member row lies in R_ℓ (asserted).
4. **Seeds.** Builder seed for read seed s and language ℓ: 1000·s + ℓ (55001 to 55027, 56001 to 56027, 57001 to
   57027), the same for the three pairs, as R6 used s for every pair. A tuple seed is not used:
   `default_rng([55, 0])` equals `default_rng(55)` (checked).
   - The 81 integers are asserted absent from every seed in `episode_seed_ledger.md` and are entered there under
     55 to 57.
   - The smoke uses 1000·s + ℓ for s = 9001 to 9003 (9001001 to 9003027), entered in the ledger's smoke row.
5. **Assembly.** Per seed and pair, the 27 parts are concatenated in ℓ order (`concat_episodes`). The SHA-256 of the
   concatenation is the per-pair hash of the ledger. The 27 part hashes are recorded in `held_started_H8.json`.
   - Hash guard as R6 rule §5 item 3: the nine per-pair hashes differ from each other, from H5's nine and from every
     recorded non-smoke hash.
6. **A language that cannot reach its share.** Per pair and seed, a pass builds every remaining language in ℓ
   order, each with its item-4 seed.
   - A language is set aside if its call raises the values-subset guard (`AssertionError`) or the builder's
     failed-draw `RuntimeError` (more than 10·n_ℓ failed draws). These two are caught only here. Any other
     exception, and any validation failure, stops the step (§10).
   - After a pass with set-asides, the 4,096 are re-split by item 2 over the remaining languages and a new pass
     starts. Stop when a pass has no set-aside. Labels only, before any score.
   - Set-aside languages are named in the report. More than two set aside for one pair and seed: the read stops
     before any score and the user decides.
7. **Feasibility** (written with this rule, smoke seeds 9001 to 9003, read rows before the drops, labels only).
   - All 27 languages reached their shares in all three pairs on seeds 9001, 9002 and 9003. Shares range from 35
     (Emakhuwa) to 215 (Hindi).
   - The worst failed-draw ratio was Kyrgyz (excitement on 2, anger on 4 paintings), emotion × genre: 6.9 to 7.3
     per episode against the limit of 10.
   - Anchor paintings: 498, 711 and 489 per pair (the pooled builder: 455, 608, 460). Building took about 1 s per
     pair.
8. **Anchor-emotion distribution.** Share of anchor emotions on seed 9001 (style × genre: the anchor's emotion as
   the third aspect):

   | Builder, pair | amus. | anger | awe | content. | disgust | excit. | fear | sadness |
   |---|---|---|---|---|---|---|---|---|
   | Pooled, emotion × style | 0 | .137 | 0 | 0 | .231 | 0 | .342 | .290 |
   | Pooled, emotion × genre | 0 | .188 | 0 | 0 | .072 | 0 | .306 | .434 |
   | Pooled, style × genre | .110 | .044 | 0 | 0 | .117 | .180 | .214 | .336 |
   | Option B, emotion × style | .115 | .022 | .112 | .319 | .047 | .124 | .101 | .160 |
   | Option B, emotion × genre | .109 | .022 | .129 | .269 | .067 | .138 | .100 | .165 |
   | Option B, style × genre | .100 | .018 | .116 | .340 | .053 | .129 | .092 | .152 |

   After the verdicts, the read's distribution is reported beside H5's (descriptive).

## 6. Before each read

As R6 rule §6, per read, in this order. Each failure stops the work before the read, except the smoke's, which is
fixed and rerun. Round 6's arm-L head identity (its §6 item 2) applies to CLIP only. Its describe-then-score stop
(its §6 item 6 and §7) and the refusals tied to that stop do not apply.
1. **Inputs.** SHA-256 of this file, the R3 and R6 rules, `pins_<backbone>.json` (CLIP: `pins_clip.json`) and every
   file it lists; for H8 to H10 also `pins_a28.json` and every file it lists. Stage 0a has passed. For H6, H7, H9
   and H10, stage 0b and the backbone's stage 2 have passed and the backbone is not switched off.
2. **Head refits.** As R6 rule §6 item 2, with the backbone's pins: every refit posterior on selection rows equals
   the pinned array bit for bit.
3. **Seed-42 regression.** The runner's scoring code in selection mode on seed 42, with the frozen picks, reproduces
   exactly the backbone's pinned seed-42 numbers and per-anchor array hashes (§3.4). For CLIP this is stage 0a. The
   episode code reproduces the per-pair hashes of seeds 42 and 9001 to 9003 on selection rows (R6 rule §6 item 4).
4. **Sensitivity input.** The backbone's seed-42 per-episode differences of P1 to P7, split into σ_a² and σ_ε² as R6
   rule §6 item 5 (CLIP: round 6's `sensitivity_seed42.json`), written to `results/sensitivity_seed42_<backbone>.json`.
5. **Smoke.** Last. The same script bytes run end to end on smoke seeds 9001 to 9003, selection rows, 64 episodes
   per pair, printing no metric, as R6 rule §6 item 7. One wiring mutation must fire an assertion.
   - The ArtELingo-28 path runs on selection rows with a synthetic language column (row ID mod 27), values V,
     shares and seeds by §5.1.
   - The smoke also builds the ArtELingo-28 bundle inputs on read rows with the ArtELingo-28 loader: features, A3
     codes, posteriors and the NaN masks of §5 item 4. No label, no episode and no score. No ArtELingo-28 row is
     scored before its read.
   - SHA-256s of the runner and every `r9_` module go to `results/smoke_record.json`; a copy is committed in this
     folder. Any later change to these files needs a new smoke.
6. **Ledger row first.** The run chat commits and pushes the read's ledger row (§7) before launch.

## 7. The read: ledger first, one runner

1. **Ledger.** `docs/superpowers/held_ledger.md` gains a backbone column (D17). Rows H1 to H3 and H5 read "CLIP ViT-B/32"; H4 reads "CLIP ViT-B/32,
   SigLIP 2, PE-Core, Qwen3-VL-Embedding-2B (backbone check)" (user, 2026-10-13).
   New rows:

   | Row | Dataset and split | Backbone | Seeds | Bar |
   |---|---|---|---|---|
   | H6 | ArtELingo held rows | Qwen3-VL-Embedding-2B | 52 to 54 (H5's episodes) | §4.1 |
   | H7 | ArtELingo held rows | SigLIP 2 so400m/14-384 | 52 to 54 (H5's episodes) | §4.1 |
   | H8 | ArtELingo-28 read set, 746 paintings | CLIP ViT-B/32 | 55 to 57 (§5.1) | §4.2 |
   | H9 | ArtELingo-28 read set | Qwen3-VL-Embedding-2B | 55 to 57, H8's episodes | §4.2 |
   | H10 | ArtELingo-28 read set | SigLIP 2 so400m/14-384 | 55 to 57, H8's episodes | §4.2 |

   Each row records the date, the runner's SHA-256, the backbone's pins-file SHA-256 (H8 to H10: also
   `pins_a28.json`'s) and "(pending)" in the episode and report cells. Switched-off rows read "not run: switched off".
2. **Per-read files**, in this folder's `results/`: `held_started_<Hn>.json` (a copy committed in the folder),
   `held_episodes_<Hn>_seed<s>.npz`, `sensitivity_<Hn>.json`, `held_arrays_<Hn>.npz`, `held_pass_<Hn>.json`,
   `held_verdict_<Hn>.json`. A fix pass writes `_fix1` names; a reserve read `_reserve` names (ledger row `<Hn>-R`).
3. **Refusals**, as R6 rule §8.1 with `<Hn>` names: `--after-crash` (one rerun, same episode hashes), `--fix 1` (one
   corrected pass after a traced runner bug, a new smoke first), `--reserve` (C5's reserve read). In addition the
   runner refuses if:
   - row `<Hn>` is missing, names another backbone, or has another runner or pins SHA-256 (either pins file);
   - the backbone is switched off;
   - the read is H8, H9 or H10 and `held_verdict_H6.json` or `held_verdict_H7.json` does not exist, unless that row
     reads "not run: switched off" or "dropped under the rule" (D9: stage 4 after stage 3);
   - the read is H9 or H10 and `held_verdict_H8.json` does not exist (the episodes are H8's).
4. **Order of computation**, as R6 rule §8.3. Its first write is `held_started_<Hn>.json`. Then the episodes, the
   sensitivity, the pooled P1 to P7 and `held_pass_<Hn>.json`. No per-seed, per-pair, per-language or subset number,
   and no B1, R1 or PM metric, is computed before `held_verdict_<Hn>.json` exists.
   - **Detectable margins** (R6 rule §8.2): SE² = (σ_a²·Σ_p M_p² + σ_ε²·N) / N², σ² from §6 item 4, M_p from the
     read's episodes, N = 36,864. ArtELingo reads: x = 3.532·SE. ArtELingo-28: level 1 x = 3.083·SE, level 2
     x = 3.417·SE. x₉₅ = 2.80·SE beside each.
5. **Re-derivation (C13).** As R6 rule §8.4, per read, by an agent that has not read the implementation. It also
   writes its own §5.1 split, seeds, set-aside passes and assembly.
   - Phase 1 of H6, H7, H9 and H10 also re-derives from the pinned files (reader margins, posteriors, A3 codes): τ,
     σ* and the B and B0 picks; AFF's min-margin cells and CF's max-R@1 cells per tune half; RCA's λ; the
     switched-off outcome.
   - Phase 2 checks the episode hashes, the frozen-pick assembly, P1 to P7, the n_j, the Holm orders and every pass
     or fail. Agreement as R3 rule §8.
6. **Boundaries.** A count within one of its n* is reported to the user with both counts (R6 rule §8.5). The integer
   rule decides.
7. **Reads run locally** from pinned files (D12). Re-fits may run on DAS6, except that the heads a read refits are
   pinned from a local fit (§3.4).

## 8. Reported after the verdicts, no check (spec §3.4, §3.6)

None of these changes a verdict.
1. Per pair and per direction; per seed.
2. Per backbone: the gate share and the reader's affect-pick rate on the read (CLIP on held: 54.5% at τ_0).
3. B1, R1 and PM, each with AFF minus it.
4. AFF on one backbone against AFF on another, paired on the same episodes (H5, H6, H7; and H8, H9, H10).
5. ArtELingo-28: per language (27), each from its own episodes, with its median rows per painting and the values
   it holds on fewer than 30 paintings (§5.1); the pooled read without the flagged languages; the 498-painting new
   subset against the 248 held (episodes by anchor painting); the anchor-emotion distribution beside H5's; set-aside
   languages (§5.1 item 6); translation coverage per language.
6. Stage 0b's count of episodes with a changed per-anchor R@1.

## 9. Stage 5: probes (exploration, D23)

No bar, no claim (SC4: a direction only). Fixed before any probe score:
- development splits only: CUB 120 training species to fit and 30 development species to score; SemArt official
  train to fit and official val to score. CUB test, SemArt test and Affection are not touched;
- per dataset, the plan's three aspects and their three pairs; values with ≥ 30 units on the scoring split;
- the value and pair choices are recorded in the log before any score;
- a new affect grouping made by the recipe from that dataset's fit-split captions;
- seed 42 only; COS, RCA, B and CLIP AFF, the gate share and the episode counts are read.

## 10. Outcome to action

| Outcome | Action |
|---|---|
| Stage 0a item differs | stop; trace; the user decides (C12) |
| Stage 0b fails (any case or gate) | stop; trace; the recipe is not used; the user decides; H8 is not blocked (§3.2 item 4) |
| An assertion fails (SHA-256, join, split, value set, hash guard, NaN mask, condition-free, provenance) | stop that step; report; nothing improvised. Only exception: the two builder errors that set a language aside (§5.1 item 6) |
| Stage 2 switched off | "switched off": that backbone's rows read "not run: switched off"; a result, no read |
| A read not run because the rule stopped it | "dropped under the rule" (D25), not "failed"; listed in the paper table |
| H6 or H7: all seven pass | GO (SC1, SC2), claim of §4.3 |
| H6 or H7: any check fails | NO-GO, read as R6 rule §4 (inconclusive or did not beat) |
| H8 to H10: both levels pass | full claim |
| H8 to H10: level 1 passes, level 2 does not | small claim; failed level-2 checks read by C11 |
| H8 to H10: level 1 fails | NO-GO, read as R6 rule §4 |
| More than two languages set aside (§5.1 item 6) | stop before any score; the user decides |
| Re-derivation disagrees, a crash, or a bug before a verdict | as R6 rule §9, per `<Hn>` |
| A bug found after a verdict | C5's reserve read `<Hn>-R`, as R6 rule §9 |
| A read not started by 2026-10-18 (Amsterdam) | it does not start; the user decides (spec §9; D14: no cap drops it) |
| Anything this file does not cover | the user decides |

Every read is reported, in one main table with every backbone (D13, C5).

## 11. Frozen and rerun

- **Frozen:** AFF's method and recipe (R3 rule D1 to D15; spec §3.7): the 224-cell grid, the pick rules, the head
  settings, k = 64, the reader features, A3's chosen settings; the affect partition and affect-km grouping; the split
  and leakage groups; the painting halves, head row draws and the 60,000 PM fit rows (`fit_rows_sha256`
  0a0d3b77…, asserted against `baselines_seed42.json`); the CSD style grouping (`step1_group_style.npz`
  `style_csd`, made from CSD features, not the backbone); H5's episodes; CLIP's whole AFF (H8).
- **Re-fitted per new backbone** (stage 2): the image and caption groupings, the content graph, bank AIC, all heads (the style heads included: CSD labels, the backbone's features),
  A3, the practice banks and the reader, τ, every seed-42 pick.
- **New this round:** the ArtELingo-28 value sets and episodes (§5).

## 12. Exceptions to the constitution

| Principle | Exception | Why | Simpler option rejected |
|---|---|---|---|
| C4 | H6 and H7 reuse H5's seeds and episodes (52 to 54). Their hashes are still checked against every other earlier seed | backbones compare on the same episodes; the held rows are the same either way, so fresh seeds add no protection (D16) | fresh seeds 55 to 57 for H6 and H7 |
| C4 | ArtELingo-28 builder seeds are 1000·s + ℓ, not s (§5.1 item 4), all entered in the ledger; smoke builder seeds 9001001 to 9003027, not 9001 to 9003 | one stream per language; a tuple seed collides with the integer seed | one pooled build per seed (option A), whose anchors cover only four emotions |
| C9 | ArtELingo-28: a two-level test, not "every check must pass" | 746 paintings would fail seven checks from power alone; serial gatekeeping keeps the family-wise error at α (D8) | seven checks; two checks alone |
| C9 | Holm decided by the integer count form (R6 rule §12) | as round 6 | each check at 95%, uncorrected |
| C12 | stage 0b case 2 and the cluster run C1: picks exact, seed-42 R@1 and P1 to P7 points within 0.05 R@1 points (§3.2 item 3) | A3's GPU non-determinism, shown by L1 ≠ L2 (D32), or another GPU (C1); 0.05 is about 1/4 of the narrowest seed-42 half-width | an exact replay, which would stop the round on GPU noise |

## 13. ID glossary

| ID | Meaning |
|---|---|
| AFF, CF, COS, RCA, B, B0, B1, R1, PM | §2 |
| P1 to P7 | the seven checks (R6 rule §3) |
| n_j, n* | resample means ≤ 0 for check j; the largest passing count at a Holm step (§4) |
| x, x₉₅ | detectable margins (§7.4) |
| H6 to H10; `<Hn>-R` | the reads of §7.1; a reserve read |
| L1, L2, C1 | stage 0b's local A3 runs and the cluster run (§3.2) |
| V; Pool_ℓ; n_ℓ; R_ℓ | the ArtELingo-28 value sets; language ℓ's pool, share and rows (§5) |
| `pins_<backbone>`, `pins_clip`, `pins_a28` | pinned inputs with SHA-256s (§2, §3.3, §3.4) |
| SC1 to SC4 | spec §6's success criteria |
| R3 rule, R6 rule | see Precedence |
