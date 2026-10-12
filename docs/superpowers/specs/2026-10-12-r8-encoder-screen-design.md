# Round 8, a screen of fixed style encoders (own track): design

> Date: 2026-10-12 19:50 · Status: draft · Kind: experiment
> Builds on: [spec A, the main line](2026-10-12-r8-learned-grouping-design.md) (cited as "A:" plus its ID, for example
> A:SC3); the grill's record ([direction log, entry 2026-10-12](../direction_log.md)); [round 7
> report](../../reports/auto/v2/2026-11-26_design_l_style_source.md) · Constitution: [version 2](../constitution.md)
> Decision rule: none of its own. A's `DECISION_RULE.md` and `HELD_RULE.md` cover this track (A:D33) and govern where
> they differ from this spec. IDs of this spec: ED for decisions, ESC for criteria.

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** Round 7 showed that CSD, our style encoder, mixes style with genre. You said we might need another
encoder. This spec tests three other fixed encoders in CSD's place: ALADIN, DINOv2 and SigLIP 2. You said that even if
it works it adds no novelty. So we treat it as an engineering finding, not a method claim.

**Plan.**
1. Regression check, shared with the main-line spec: the reused code gives its seed-42 numbers again, exactly.
2. Extraction (agent default): each encoder embeds the 49,121 WikiArt images that are not held. An encoder whose
   public weights do not load drops out (user).
3. Seed-42 sweep (agent default): six settings, three encoders with two scorers each. Each encoder gives 17 groups by
   raw CSD's recipe. A gallery image goes to the nearest group centre in the encoder's space. Captions keep CLIP
   heads. The bar is the main line's gate, unchanged (user).
4. Held read (claim test; fallback only): the pick, once, only if the main line fails. Before it, the 12,281 held
   images get embedded, and not earlier.

**Compared against.** AFF, our current best method (gate and held read). Beside it: B′(A1), the strongest
condition-free scorer, and each scorer's matched control. References that decide nothing: raw CSD with the same
placement, and one random grouping per encoder.

**Outcomes.**
- Gate pass: beat AFF pooled with the lower bound above 0, and lose at most 0.3 R@1 to AFF per emotion pair. The
  largest gain over AFF is the pick. It waits as the fallback.
- No pass: the screen ends. We report that none of the three encoders helped.
- Held read: AFF's seven checks plus "beats AFF", at one-sided 0.0125. The tiers are the main line's. Tier 1 replaces
  AFF. Tiers 2 and 3 go into the paper's analysis.
- Expected: a gate pass about 8 to 10%, Tier 1 given a pick about 30%, so Tier 1 overall about 3%. In round 7, raw CSD
  lost more than one R@1 point to AFF on emotion × genre.

**Cost.** The same overnight run as the main line: at most 3 GPU hours on the local card and about 1 CPU hour. The
fallback read adds about 15 GPU minutes. You accepted our default of the local card in place of the cluster. The
cut-off is the main line's: a held verdict after Friday 23 October stays out of the CVPR paper.

**Check these.**
1. The stage type is a sweep, not a new idea. It picks among settings of a method we have, and its bar is borrowed.
2. The checkpoints: ALADIN's official ICCV 2021 weights, DINOv2 ViT-L/14 (CLS token), SigLIP 2 so400m (pooled).
3. The 17 groups come from raw CSD's recipe with the resolution searched. The recipe was built for CSD's space.
4. Held images are embedded only at the start of a held read, and only for the pick's encoder.
5. The fallback read needs the main line's rule (its D31): no main-line pick, or its read without Tier 1. Its α is
   halved to 0.0125 (you accepted this after our recommendation).

## 1. Context and goal

Asked: "even it works it does not add novelty" (user, decision A, 2026-10-12, on this screen). Direction log: "a screen
of fixed encoders (ALADIN, DINOv2, SigLIP 2) … an engineering finding, not a method claim".

Round 7 ended NO-GO. Told true styles gave +3.92 R@1 on style × genre, label-free groupings about +0.38, and raw CSD
lost 1.11 (R1) and 1.28 (extended AFF) to AFF on emotion × genre. The user reads the cause as CSD, which treats style
and genre almost as one aspect (A §1). The main line learns a grouping on frozen CLIP. This track keeps the grouping
recipe and changes only the encoder that feeds it.

Goal: find out whether another fixed encoder gives a grouping with which a reader-steered scorer beats AFF on held
paintings by Fri 2026-10-23.
Question this step answers: does an encoder other than CSD, in CSD's place and with the same recipe, let R1 or
extended AFF beat AFF on seed 42?

## 2. Scope

In: extraction of three encoders on the non-held images; a 17-group grouping per encoder; own assignment; a reader and
two scorers per grouping (six candidates); one random control per encoder; the seed-42 gate and pick; the held read of
the pick as the main line's fallback; held-ledger row H7; descriptive AMIs.

Out: caption placement and any learned head (A:D4, A:D1); other encoders or checkpoints (a larger SigLIP 2, CLIP
variants); fine-tuning an encoder; token statistics of the encoders (user); a wider grid of groups or recipe settings;
the CLIP-head placement arm; a promising band and dig budget (a sweep has none; the track ends at a failed gate);
fresh seeds on selection rows; a second dataset; the second backbone.

## 3. Design

### 3.1 Hypothesis and prior

Claim: an encoder that places paintings by style apart from subject gives groups that cut across the content groups.
The reader then picks that grouping on style episodes, and extended AFF or R1 beats AFF.

Mechanism: ALADIN is trained for style-aware retrieval, so it may separate style from subject better than CSD. DINOv2
and SigLIP 2 are general encoders; SigLIP 2 matches by what the picture shows, so it may follow content.

Prior, before any number exists (agent default):
- Gate pass of any of the six: about 8 to 10%. No encoder is cut free of content here. Raw CSD with a CLIP-trained
  reader failed A:SC4 on emotion × genre (−1.11 and −1.28 against the bar −0.3). Every round 7 scorer with a fourth
  grouping lost 0.31 to 1.61 to AFF on that pair. DINOv2-L reaches 40.0 mAP@1 on WikiArt style retrieval against
  CSD's 63.1.
- Tier 1 given a pick: about 30%. Overall about 3%, because the read happens only if the main line fails.
- Surprises: a random control passing; DINOv2 or SigLIP 2 passing where ALADIN fails; raw CSD with own assignment
  passing (A:3.1).

### 3.2 Arms and controls

Every grouping E has 17 groups, joins affect, image and caption in CSD's place, and gets R1[E] and extended AFF XA[E]
(A:3.2; round 7 L11, L13). Thresholds, gates and cells follow A:D25 and round 7.

| Arm | What it is | Differs from R0 by | Role |
|-----|------------|--------------------|------|
| AL, DI, SG | ALADIN, DINOv2-L, SigLIP 2 embeddings → 17 groups; each with R1 and XA (six scorers) | the encoder | candidates |
| Z(AL), Z(DI), Z(SG) | random grouping with the encoder's group sizes (A:D18) | no map | control, decides nothing |
| R0 | raw CSD, own assignment in CSD space, from A | none | reference, shared with A (run once) |

Comparators of every scorer s: AFF (gate, held; C3), B, B′(A0), B′(A1), B′(E) and its matched control CF[s] (C2), as
A:3.2. A random control that passes with the same scorer kind is flagged (A:D27).

**Grouping (ED8).** For each encoder, the recipe that made raw CSD's 17 groups: painting vectors (each painting's first
scorer-train row), unit-normalised; k-nearest-neighbour union graph, k = 20; Leiden (RBConfiguration), seed 42; groups
under 200 rows merged. Raw CSD gave 17 communities at resolution 1.0. For an encoder, the resolution is searched until
the result has 17 groups, else the nearest count is kept and reported (A:D15).

**Placement.** Own assignment (A:D4, D16, D17): an image goes to the nearest group centre in the encoder's space, the
posterior is one-hot, the reader's banks use the cross-fitted form, captions keep CLIP text heads, and a random
control draws by group-size shares.

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---------------|------|---------------|
| 49,121 non-held WikiArt images (images of the 224 cache), from the original files | encoder inputs; groupings and centres fit on the scorer-train paintings (36,518) | extraction, 1 |
| Seed 42 on selection rows (12,288 episodes) | development: gate and pick | 0, 1 |
| 12,281 held images | held: embedded for the pick's encoder only, at the start of its held read | 2 |
| Held seeds 52 to 54 (H5's episodes, 36,864) | held: one read (H7) | 2 |
| Style and genre labels | AMIs after `gate_verdict.json` | diagnostics |
| Fresh seeds 55 and later | not touched | none |

### 3.4 Measurement

As A:3.4 and C9. Seed 42: paired per anchor, pooled over 12,288 episodes, 95% cluster intervals; the gate has no
correction over the six candidates (as A's gate over 24; disclosed in §8). Held read: pooled over seeds 52 to 54, the
eight checks P1 to P8 of A:SC6 under Holm at family-wise one-sided 0.0125, one look.

Embedding: each encoder with its own preprocessing, from the original files in `/data/PDD/wikiart_proj/wikiart`, in
CSD's image order and joins (`style_csd_vitl/row_to_image.npy`; C6). Checkpoints are in ED5; weights are recorded by
SHA-256.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|-------|------|-------|--------------|-------------|
| 0 | regression check | A's stage 0, seed 42 (shared) | A:SC1 | stop; trace the cause; a note for the user |
| E | extraction (engineering step, no type) | 3 encoders × 49,121 images | ESC1 per encoder | the encoder drops out (ED4); all three out ends the track |
| 1 | sweep | 3 groupings × 2 scorers on seed 42 | ESC2; pick by §6 | no pass: the track ends; reported |
| 2 | claim test, fallback only | the pick on held seeds 52 to 54 | ESC1 repeat on 200 non-held images, A:SC5, then ESC3 | AFF stays |

Stage 2 starts only if A:D31 holds, the gate has a pick, and the read can start before Fri 2026-10-23 12:00 (A:D32).
It begins with the held images' embeddings for the pick's encoder. The embeddings join rows by CSD's mapping.

### 3.6 Diagnostics

Measured, decides nothing. Label-based ones only after `gate_verdict.json`; held ones only after the held verdict.
- Per grouping: group sizes and the resolution found; AMI with the content grouping, with R0 and between encoders
  (label-free); AMI with style and genre; own-map agreement on held-out halves. Tells whether an encoder cut content.
- Per scorer: per-pair R@1 and condition gain; margins over B′(A1), B′(E) and CF; gate-open shares; the random control
  beside it. After the held verdict: per seed and per pair.

### 3.7 Frozen vs rerun

Frozen from seed 42: encoder checkpoints and preprocessing, groupings, centres, caption heads, readers, thresholds,
gates, the pick and its cells (A:3.7). Embeddings of the non-held images are read from storage. Rerun at the read: the
held images' embeddings, their own-map posteriors, held captions' posteriors, the reader's probabilities, gates, B′(E)
and its CF. AFF's held arrays, COS, RCA, B, B′(A0) and B′(A1) follow A:D34.

## 4. Decisions

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| ED1 | Track, loop size | The screen is its own track, spec, report and pick (A:D5); medium loop | one pool with the main line | user; loop size: user, after seeing the recommendation | 2026-10-12 |
| ED2 | Stage types | 0 regression (shared); extraction, an engineering step; 1 sweep; 2 claim test, fallback | new idea needs a method change; a held read can only be a claim test | agent default | 2026-10-12 19:50 |
| ED3 | Encoders | ALADIN, DINOv2, SigLIP 2 | a wider screen is out | user | 2026-10-12 |
| ED4 | Weights that do not load | the encoder drops out | stopping the track for one encoder | user | 2026-10-12 |
| ED5 | Checkpoints | ALADIN: official ICCV 2021 weights from its README (not cached; download). DINOv2: `facebook/dinov2-large`, CLS after final norm (only dinov2-small is cached; download). SigLIP 2: `google/siglip2-so400m-patch14-384`, pooled embedding (cached) | the largest public checkpoints that keep the cost within the overnight run; patch tokens are out | agent default | 2026-10-12 19:50 |
| ED6 | Token statistics | none for the encoders | | user | 2026-10-12 |
| ED7 | Held images | 49,121 non-held images now; the 12,281 held images only for the pick's encoder, at the start of its held read | embedding all held images up front reads held paintings for encoders that never get a read | agent default | 2026-10-12 19:50 |
| ED8 | 17 groups | R0's recipe (round 1 Leiden on painting vectors, k = 20, merge under 200 rows), resolution searched to 17 | one recipe for R0, the main line (A:D15) and the screen; k-means differs from R0's | agent default | 2026-10-12 19:50 |
| ED9 | Placement | Own assignment by the nearest centre; captions by the CLIP text head | CLIP heads copy the grouping and lose it (A:D4) | user (A:D4); details agent default (A:D16, D17) | 2026-10-12 |
| ED10 | Arms | One random control per encoder; R0 with own assignment shared with A as the reference | a CLIP-head arm for the screen: descriptive only, and A has one | agent default | 2026-10-12 19:50 |
| ED11 | Scorers | R1 and extended AFF per encoder: six candidates | fewer scorers per encoder | user, after seeing the recommendation | 2026-10-12 19:50 |
| ED12 | Gate | A:SC3 and A:SC4 unchanged: the same gate as the main line | an easier gate for a fallback | user | 2026-10-12 |
| ED13 | Pick | A:D26: largest pooled Σ(4d) of s − AFF among passers; ties by gain-statistic sum, then AL, DI, SG; R1 before XA | | agent default | 2026-10-12 19:50 |
| ED14 | Fallback condition | The read happens only if the main line fails by A:D31 and the screen has a pick; the main line goes first (A:D12) | a joint pool (A:D5) | user (priority); condition agent default | 2026-10-12 |
| ED15 | Held read, α | A:SC5 to SC7 at family-wise one-sided 0.0125; ledger row H7; tiers as A:SC7 | the full 0.025 (A:D12) | user, after seeing the recommendation | 2026-10-12 |
| ED16 | Claim | Engineering finding, not a method claim; the Tier 1 sentence of A:§6 in §6 | | user (the framing); wording agent default | 2026-10-12 |
| ED17 | Prior | Gate pass 8 to 10%; Tier 1 given a pick 30% | | agent default | 2026-10-12 19:50 |
| ED18 | Where it runs | The local GPU under the shared lock, at most 3 h; the cluster is not used | the cluster, as first planned | user, after seeing the recommendation | 2026-10-12 |
| ED19 | Build | The same build chat and the same overnight run as A (A:D13) | a separate run | user, after seeing the recommendation | 2026-10-12 |

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: all three public weights load. DINOv2-L is reachable on the hub (config fetched 2026-10-12) and SigLIP 2
  is cached. ALADIN's README link was not opened. Checked: partly; a failed load drops the encoder (ED4).
- Assumption: the original files cover all 61,402 images in CSD's order. Checked: CSD's extraction read them from the
  same folder (its `meta.json`).
- Assumption: Leiden reaches 17 groups for each encoder at some resolution. Not checked; else the nearest count.
- Assumption: A's pipeline is unchanged (A:SC1) and its file format takes a new grouping ID with no other edit.
- Assumption: the DINOv2 and CSD WikiArt figures in §3.1 are right. Not checked here; the report cites the table.

No open questions: the grill settled the bar, the data and the scope; the rest are agent defaults (§4).

## 6. Success criteria

**Decision mode.** Bars are A's; no rule of its own (A:D33).

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| A:SC1 | stage 0 | round 3 and 7 arrays | exactly equal | seed 42 |
| ESC1 | per encoder: embedding rows, finite values, the row-to-image join, the same image twice, weights SHA | CSD's order file; the encoder's own repeat | all hold: 49,121 rows (61,402 with held), finite, join hash equal, cosine ≥ 0.9999 on 200 repeats | non-held images; at the read |
| ESC2 | the six scorers: A:SC3 (R@1, s − AFF, pooled, lower bound above 0) and A:SC4 (Σ(4d) ≥ −49 on emotion × style and on emotion × genre) | AFF | as A:SC3 and A:SC4 | seed 42 |
| ESC3 | the pick: A:SC5 to SC7 (eight checks P1 to P8, tiers T1 to T3) | as A:SC6 | Holm, family-wise one-sided 0.0125 | held seeds 52 to 54 |

- Carry (seed 42): A:SC1 and ESC1 hold for at least one encoder, and a scorer passes ESC2. The pick waits as the
  fallback.
- GO (held): Tier 1; the pick replaces AFF as current best (C3). The user decides how the CVPR paper carries it.
- Tier 2 or 3: AFF stays; the grouping enters the paper's analysis.
- Kill: no scorer passes ESC2, or all three encoders drop out; or a held read in no tier.
- Inconclusive: a held no-go whose failed checks all have points above 0 (C11); AFF stays; no second read (C5). This
  takes precedence over the last clause of Kill.
- No read (the main line wins, or A:D32 applies): the seed-42 pick is a development finding only.
- A failure of A:SC1 stops the stage. A failure of ESC1 drops its encoder.

**Licensed claim.** An engineering finding, not a method claim. Tier 1: "AFF's reader with <encoder>'s grouping in
CSD's place (<R1 or extended AFF>) beats cosine, RCA, B, B′(A0), its matched control and AFF on aspect R@1, and its
condition gain exceeds its matched control's and RCA's, pooled over three aspect pairs, on held paintings never used to
fit, select or tune it." Not: that the encoder separates style from genre; other encoders or checkpoints; a margin per
pair; B′(A1) (untested). Tiers 2 and 3 license only an analysis statement. A failed gate licenses "none of these three
checkpoints, with this recipe, beat AFF on seed 42".

**Selection rule** (ED13, from A:D26). Among passing scorers, the largest pooled Σ(4d) of s − AFF; ties to the larger
gain-statistic sum, then AL, DI, SG; R1 before XA. Labels and AMIs never enter it.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1, C2, C3 | pass | B′(A1) beside every number, CF[s] for every scorer, AFF the bar (A:§7) |
| C4 | exception | as A: development on seed 42; test episodes are H5's |
| C5 | pass | one read (H7), only as the fallback; held images embedded inside it; frozen; ledgered; D32 timing |
| C6 | pass | joins through CSD's row-to-image file, hash-checked (ESC1) |
| C7 | pass | A's two rules cover this track (A:D33) |
| C8 | pass | prior §3.1; disclosure §8 |
| C9 | exception | as A: Holm at 0.0125 over eight checks |
| C10 | pass | the bar is A's gate (A:D10) |
| C11, C12, C13 | pass | §6; A:SC1 and A:SC5; A's re-derivation covers this track |
| C14 | pass | the claim comes from held data |

Exceptions: C4 and C9 · same cause, reason and rejected option as A §7; the track shares A's family-wise α with the
main line, which is why the read is a fallback (A:D12).

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|------------------------------------------------|---------------------|-----------------|
| A checkpoint loads but gives poor features (wrong preprocessing or head) | ESC1 passes but the AMI with R0 is near zero and the grouping near the random control's | reported; the gate decides |
| The encoder follows content, not style | AMI with the content grouping at or above R0's 0.40 | reported; the gate decides |
| The recipe, built for CSD, gives odd groups in another space | resolution search misses 17; very uneven sizes | nearest count; sizes reported |
| A chance pass over six looks | a random control passes too (A:D27) | flagged; the held read protects |
| Held embeddings differ from the stored ones (environment drift) | ESC1 repeat at the read | stop the read; trace the cause |
| Late against the cut-off, or the GPU lock is held | gate verdict after 16 October; exit 75 | A:D32: no read after Fri 2026-10-23 12:00 |
| Reused code changed silently | A:SC1 | stop and trace |

**Selection disclosure.** On seed 42: about 72 variants before round 8 (A §8), the main line's 24 (26) candidates,
and these six. The gate has no correction. Six independent looks at a 2.5% one-sided lower bound pass by chance about
14% of the time before A:SC4, so the seed-42 margin of the pick is inflated. The held read protects the verdict and the
report gives both margins. Held rows were read before (H1 to H3, H5); the paper discloses this (C5).

## 9. Deliverables and review

- At the end: embeddings under `/data/SSD2/pre_extract/artelingo/` (three encoders, about 0.6 GB for 49,121 images);
  code and results in A's folder `src/test/20261129_r8_grouping/` (prefix `r8_`, screen files marked); the screen's
  section in A's rules; its auto report with a results brief in `docs/reports/auto/v2/` and a user-read report; held
  ledger row H7 if read; a direction log entry.
- Reviews: A's rule reviews cover this track (A:D33); ticket reviews of the verdict code are A's; C13
  re-derivation of the gate and the pick; the final whole-branch review with A's.
- Stop points: none between approval and the reports. A stage-0 failure ends the run with its report and a phone note.
- Budget: local GPU at most 3 h under the shared lock (extraction, 3 × 49,121 images); about 1 CPU hour (groupings,
  three readers, scoring), in the same overnight run as A; the fallback read adds about 15 GPU minutes (12,281
  images) and under 1 CPU hour. No cluster. Time-box: Fri 2026-10-23 23:59 (A:D3, D32).

Body length: about 235 lines against a target near 180 (claim test about 150 plus one further stage); the overrun
is §4's 19 decision rows and §6's criteria table, which cite A instead of repeating it.
