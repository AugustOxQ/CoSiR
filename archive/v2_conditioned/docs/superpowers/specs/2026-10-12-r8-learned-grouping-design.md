# Round 8, a learned grouping across content (main line): design

> Date: 2026-10-12 19:35 · Status: approved (2026-10-12 19:47, user) · Kind: experiment
> Builds on: [round 7 report](../../reports/auto/v2/2026-11-26_design_l_style_source.md); the grill's record
> ([direction log, entry 2026-10-12](../direction_log.md), bullets "Meaning (C)" to "Idea (B) and the grill");
> [facts brief](../../../src/test/20261129_r8_grouping/step3_facts.md);
> [paired-image reference](../../../src/test/20261129_r8_grouping/step3_caption_ref/step3_caption_ref.md);
> [literature note](../../../../../docs/reports/literature/2026-11-30_r8_learned_grouping.md) · Constitution:
> [version 2](../constitution.md) · Companion: [encoder screen spec](2026-10-12-r8-encoder-screen-design.md)
> Decision rules (D33): `src/test/20261129_r8_grouping/DECISION_RULE.md` (stages 0 to 2 of both tracks; before any
> code) and `src/test/20261129_r8_grouping/HELD_RULE.md` (stage 3; after the gate verdict, before any held-read code);
> neither written yet; each governs where it differs from this spec

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** Round 7 showed a large gain within reach on style × genre. Told the true styles, the scorer gained
+3.92 R@1 over its matched control (the same score with the condition removed). Our label-free groupings reached about
+0.38. You read the cause as CSD, our style encoder, which mixes style with genre. So we learn a new grouping on frozen
CLIP that cuts across content, and put it in CSD's place.

**Plan.**
1. Regression check: the reused round 7 code gives its seed-42 numbers again, exactly.
2. Seed-42 screen (new idea; agent default): two builders, six settings each, 17 groups each (user).
   (a) A contrastive head on CLIP image features. Its negatives come from the query's own content group (64 CLIP image
   clusters). Its positives are views of the painting or a neighbour in another content group. (b) FCMI, a clustering
   made blind to the content groups. Gallery images go to the grouping's own map. Captions keep CLIP heads (user).
   Each grouping gets a reader and two scorers: 24 candidates, one pick.
3. Held read (claim test; agent default): the pick, once, on the episodes of AFF's held read.

**Compared against.** AFF, the current best (user). AFF steers the score only when the reader (our model that picks a
grouping from the example pairs) picks affect. Beside it: B′(A1), the strongest condition-free scorer, and each matched
control. References that decide nothing (user): raw CSD, random groupings, NrKmeans, told prompts (privileged).

**Outcomes.**
- Seed-42 pass (user): beat AFF pooled with the lower bound above 0, and lose at most 0.3 R@1 to AFF per emotion
  pair. The largest gain over AFF is the pick (agent default). Promising (user): only round 7's weaker gate passes.
- Held read (user, after seeing the recommendation): AFF's seven checks plus "beats AFF", Holm at one-sided 0.0125.
  Tier 1, all eight pass: the pick replaces AFF. Tier 2: beats AFF, fails a condition-gain check. Tier 3: the seven
  pass, "beats AFF" fails. In Tiers 2 and 3, AFF stays and the grouping goes into the paper's analysis.
- If the main line fails and the encoder screen passed its gate, the screen's pick takes the read (user).
- Expected: a seed-42 pass about 15%, Tier 1 about 5%. In round 7, every scorer with a fourth grouping lost 0.3 to
  1.6 R@1 to AFF on emotion × genre.

**Cost.** Build on 13 October, then one overnight run for both tracks: about 7.5 CPU hours and 1 GPU hour (3 more for
the screen). Gate verdict 14 October, held read about 16 October. Hard cut-off (user): a held verdict after Friday 23
October stays out of the CVPR paper.

**Check these.**
1. The tiers are yours, set after seeing our recommendation. You first had "beats AFF" alone replace AFF. We added
   that Tier 2 also needs the five R@1 checks (agent default). The halved α of 0.0125 is also yours, accepted after
   our recommendation.
2. Own assignment (agent default): an image goes to the nearest group centre in the builder's space. The head saw all
   training paintings. So the reader trains on cleaner image inputs than it scores.
3. Settings (agent default): FCMI weights 0.25 to 8 times its code's default. The 17 groups follow raw CSD's Leiden
   recipe. NrKmeans runs on CLIP features. Told prompts are CLIP zero-shot with the 27 style names.
4. Timing (agent default): no held read starts after Friday 23 October 12:00. An unreported read breaks
   constitution C5. "The main line fails" means no pick, or a held read without Tier 1.
5. Most likely wrong: that a grouping which ignores content also fixes the loss on emotion × genre.

## 1. Context and goal

Asked: "go back to the parked idea to gain more on the grouping" (user, decision C, 2026-10-12); on the main line, "I
think this is a good one" (user, decision A, before any recommendation of ours).

AFF is the current best: held GO in round 6 (+0.60 R@1 over B′(A0)), inconclusive against B′(A1) on held paintings
(+0.14 [−0.03, +0.30]), and a loss on style × genre. Round 7 ended NO-GO at its seed-42 gate: told true styles gave
+3.92 [+3.35, +4.49] on style × genre, the best label-free grouping +0.38, and every scorer with a fourth grouping
(candidates, raw CSD, NrKmeans, random controls) lost 0.3 to 1.6 R@1 to AFF on emotion × genre. Step 3 found placement matters too: with captions
placed by their painting's image head (a privileged reference), told true styles reach +10.87, raw CSD +0.59. Images
come first (user). MCL (ECCV 2024) is the closest prior for same-cluster negatives, so the mechanism is not ours to
claim; default augmentation positives keep content, so our positives drop colour change.

Goal: a label-free grouping that cuts across content, placed by its own assignment, with which a reader-steered scorer
beats AFF on held paintings by Fri 2026-10-23.
Question this step answers: does a grouping learned on frozen CLIP to ignore the content groups let R1 or extended AFF
beat AFF?

## 2. Scope

In: stage 0; builders (ii) and (iii) at six settings each, and the CSD-space neighbour probe if triggered; own
assignment for gallery images, with its cross-fitted path for the readers; a reader and two scorers per grouping; the
references and random controls of §3.2; the CLIP-head arm for the pick; the seed-42 gate and pick; the held read of the
pick, and of the screen's pick as fallback (spec B); held-ledger rows; descriptive AMIs.

Out: caption placement (a later loop; user); a wider grid (user); X-Cluster (paper stage; user); running MCL (cited
only; user); fresh seeds on selection rows (the pick goes from seed 42 to the held read); the encoder screen (spec B);
a second dataset (COCO-4c); inferred prompts (N5, parked); changes to the reader recipe, scorers or 224-cell family;
the second backbone.

## 3. Design

### 3.1 Hypothesis and prior

Claim: a grouping learned to ignore content keeps style apart from genre. The reader then picks it on style episodes
and not on genre ones, so extended AFF beats AFF pooled and keeps AFF's emotion pairs.

Mechanism: with negatives from the query's own content group, the features that separate content cannot separate the
pairs, so the head must use other directions (MCL's argument). Views without colour change share palette, texture and
brushwork; corner crops cut layout. A neighbour in another content group shares what CLIP sees beyond content. FCMI
makes the clusters independent of content directly. A grouping less like content should also be confused less with
the image grouping on emotion × genre, the pair that failed round 7's S3. Own assignment removes the CLIP head's
relearning of the grouping on the image side (CLIP heads placed raw CSD at 85.1%, true styles at 63.7%).

Prior, before any number (the rule repeats it, C8):
- Groupings: AMI with the content grouping below raw CSD's 0.40, lowest at 100% own-group negatives and at the
  largest FCMI weights; at those settings AMI with style falls too (round 7's projection: style 0.10). No source says
  whether views or neighbours keep more style.
- Seed-42 gate: about 15% that one of the 24 passes. AFF is only +0.33 over B′(A1), and every round 7 scorer with a
  fourth grouping, random controls included, failed S3 on emotion × genre. Extended AFF is more likely than R1 to pass.
- Promising (round 7's gate) about 30%.
- Given a pick: Tier 1 about 35%, Tier 3 about 20%, Tier 2 about 10%; Tier 1 overall about 5%.
- Surprises: a random control passing the gate; raw CSD with own assignment passing; FCMI far above (ii).

### 3.2 Arms and controls

Every grouping X has 17 groups (FCMI natively; the others by raw CSD's recipe, D15), joins affect, image and caption in
CSD's place (round 7 L11), and gets R1[X] and extended AFF X-AFF[X] (round 7 L13). The content grouping is round 7's L2.

| Arm | What it is | Role |
|-----|------------|------|
| V100, V67, V33 | (ii) contrastive head, view positives, 100 / 67 / 33% of negatives from the query's content group | candidates |
| NB100, NB67, NB33 | (ii) with positive = nearest CLIP neighbour in another content group | candidates |
| FC1 to FC6 | (iii) FCMI on CLIP features, content groups protected, six fairness weights (D20) | candidates; also (ii)'s comparator (D2) |
| NBC | (ii) with the nearest CSD neighbour in another content group; only if all six NB scorers fail the gate (D22) | extra candidate |
| R0 | raw CSD (round 1's 17 groups), own assignment in CSD space | reference |
| NRK | NrKmeans on CLIP features (D23) | grouping-step reference |
| TP | told prompts: CLIP zero-shot with the 27 ArtELingo style names (D24) | privileged reference |
| Z(X) | random grouping with X's group sizes, one per candidate grouping | control, decides nothing |
| X-H | the pick's grouping with round 7's CLIP image heads and its own reader, the pick's scorer kind (D28) | descriptive arm |

Comparators of every scorer s: AFF (gate, held; C3), B, B′(A0), B′(A1), B′(X) and its matched control CF[s] (C2), all
with the same 224-cell family and per-half cross-fit. Round 7's R0 numbers with CLIP heads stand beside R0.

**Placement (D16 to D18).** The own map of a grouping is the nearest group centre in the builder's feature space (head
output for (ii), FCMI's hidden features, CSD for R0, the kept subspace for NRK; TP's map is its zero-shot argmax). A
centre is the normalised mean of the group's member paintings, unit vectors. A gallery image's posterior is the one-hot
of its group. Captions keep round 7's CLIP text heads. The reader's banks need cross-fitted inputs on scorer-train rows
(round 1's two painting halves): an image in half h takes the nearest centre computed from half 1 − h, labels as in the
full grouping. The learned map is not refit per half. A random control has no map: each gallery painting draws a group
by the group-size shares from a fixed seed. All posteriors use round 7's file format, so readers and scorers run
unchanged; only the grouping IDs and the image-side source are new.

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---------------|------|---------------|
| Scorer-train rows (183,694 rows, 36,518 paintings) and their CLIP features | fit builders, groupings, centres, caption heads, banks, readers | 1, 2 |
| Images of the scorer-train paintings in the 224-pixel cache | views for V100 to V33 (GPU, D19) | 2 |
| Seed 42 on selection rows (12,288 episodes) | development: thresholds, gate, probe trigger, pick | 0, 2 |
| Style and genre labels of scorer-train rows | AMIs after the gate verdict; TP uses the 27 style names, no label of any item | diagnostics |
| Held rows: H5's episodes, seeds 52 to 54 (36,864) | held: one read per pick (H6; H7 for the screen's pick) | 3 |
| Fresh seeds 55 and later | not touched | none |

### 3.4 Measurement

Metrics, cross-fitting and the bootstrap follow C9 and round 7's rule (L13 to L16, integer form). Seed 42: paired per
anchor, pooled over 12,288 episodes, 95% cluster intervals. Held read: pooled over seeds 52 to 54 (anchor-painting
clusters across seeds), round 6's draws and Holm integer form (its §3) over the eight checks P1 to P8 at family-wise
one-sided 0.0125; each check's 95% interval and its interval at its Holm level are reported; one look. Detectable
margins for C11 readings come from the pick's seed-42 differences before the read (round 7 §7.1).

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|-------|------|-------|--------------|-------------|
| 0 | regression check | reused round 3 and round 7 code, seed 42 | SC1 | stop; trace the cause; a note for the user |
| 1 | engineering step (no type) | FCMI at its default weight, 64 protected groups | SC2 | FCMI drops out; (ii)'s 12 scorers go on |
| 2 | new idea | 12 groupings × 2 scorers on seed 42; the probe if triggered | SC3 and SC4; pick by §6 | no pass: promising or kill; screen fallback (D31) |
| 3 | claim test | the pick on held seeds 52 to 54 | SC5, then SC6 and SC7 | AFF stays; screen fallback (D31) |

### 3.6 Diagnostics

Measured, decides nothing. Label-based ones only after `gate_verdict.json`; held ones only after the held verdict.
- Per grouping: group sizes; AMI with the content grouping and between settings (label-free); AMI with style and genre;
  own-map agreement with the grouping on held-out halves; caption-head accuracy; FCMI's spread of content-group shares
  per cluster; (ii)'s final loss. Tells whether a builder cut content, and at what cost to style.
- Per scorer: per-pair R@1 and condition gain; margins over B′(A1), B′(X) and CF; round 7's S1 and S2 (promising);
  gate-open shares; pick accuracy under the told map; the random control beside it.
- Placement: R0 with own assignment against round 7's R0, and the pick against X-H. Tells what own assignment adds.
- After the held verdict: per seed and per pair; s − B′(A1), s − B′(X), s − R1 (C1, beside, untested).

### 3.7 Frozen vs rerun

Frozen from seed 42: builders and their fitted models, groupings, centres, caption heads, readers and scalers, τ^X,
gates, the 224-cell family, the pick, its chosen cells and σ*. Loaded from H5 (SHA-256 asserted): the held episodes,
AFF's held per-episode arrays (not recomputed: that would be a second AFF read), and COS, RCA, B, B′(A0), B′(A1).
Rerun on held rows: the own-map posteriors of held images (centres from all scorer-train paintings), CLIP text-head
posteriors of held captions, the reader's probabilities, gates against the frozen τ^X, the pick and its CF with the
seed-42 cells of the other parity half (round 6 §5.4), B′(X), and COS, RCA, B, B′(A0) as SC5's check.

## 4. Decisions

Rows D1 to D13 are the grill's (`r8 decide` and `r8 decide 2`, 2026-10-12; direction log). Accepted agent defaults
count as "user, after seeing the recommendation".

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Idea | (ii) on frozen CLIP, negatives from the query's content group, as the fourth grouping in CSD's place | the grouping is the limit (decision C) | user | 2026-10-12 |
| D2 | FCMI's role | A builder beside (ii), same budget fixed before seed 42; also reported as a comparator | ours: comparator only | user | 2026-10-12 |
| D3 | Paper plan, cut-off | Beside the ArtELingo paper with AFF; the pick's held verdict by Fri 2026-10-23, later stays out of the CVPR paper | other dates | user | 2026-10-12 |
| D4 | Placement | Gallery images by own assignment; captions keep the CLIP text head; caption placement later | images can gain more | user | 2026-10-12 |
| D5 | Tracks | The encoder screen is its own track, spec, report and pick | ours: one pool | user | 2026-10-12 |
| D6 | Type, loop size | Main line: new idea, full loop | the work adds a grouping and a scorer, so a new idea needs the full loop | user, after seeing the recommendation | 2026-10-12 |
| D7 | Positives | Both: views (a flip and four 160-pixel corner crops resized to 224, from the 224 cache, no colour change) and the nearest CLIP neighbour in another content group | no prior knowledge settles it | user | 2026-10-12 |
| D8 | CSD neighbours | A CSD-space probe only if the CLIP neighbour form fails, an extra candidate | the CLIP neighbour form is tried first; CSD mixes style with genre, so it is the fallback | user | 2026-10-12 |
| D9 | Budget | Six settings per builder; (ii) two forms × strengths 100, 67, 33%; FCMI six weights; 12 groupings of 17 groups; R1 and extended AFF; 24 candidates; one overnight run; grid not extended | ours: four each | user | 2026-10-12 |
| D10 | Seed-42 gate | Beats AFF pooled (lower bound above 0) and round 7's S3; round 7's gate is the weaker goal ("promising") | AFF is the current best (C3); round 7's weaker gate stays as a lesser goal | user | 2026-10-12 |
| D11 | Held read | Seeds 52 to 54 (H5's episodes); AFF's seven checks plus "beats AFF", Holm; tiers as SC7 | the user first had "beats AFF" alone replace AFF | user, after seeing the recommendation | 2026-10-12 |
| D12 | Priority, α | Main line first; the screen's pick is read only if the main line fails and the screen passes; family-wise one-sided 0.0125 per track | priority: user; halving α: ours, accepted | user; α: user, after seeing the recommendation | 2026-10-12 |
| D13 | Defaults accepted | 17 groups; references raw CSD, random groupings, NrKmeans, told prompts (privileged); X-Cluster at the paper stage; MCL cited; a CLIP-head arm, descriptive; groupings on scorer-train rows; FCMI drops out if its smoke run fails at 64 groups; one build chat and one overnight run for both tracks; no plan page | the defaults keep the build to one overnight run; the user accepted them at the grill | user, after seeing the recommendation | 2026-10-12 |
| D14 | Stage types | 0 regression; 1 engineering step (no type); 2 new idea; 3 claim test (C5 paper test) | a held read can only be a claim test | agent default | 2026-10-12 19:35 |
| D15 | 17 groups | Raw CSD's recipe (round 1 Leiden on painting vectors, k = 20 graph, merge under 200 rows), resolution searched to give 17; FCMI with 17 clusters | one recipe for (ii), R0 and the screen; k-means differs from R0's | agent default | 2026-10-12 19:35 |
| D16 | Own map | One-hot nearest group centre in the builder's space; TP its zero-shot argmax | soft posteriors add a temperature knob | agent default | 2026-10-12 19:35 |
| D17 | Readers' training input | Centres from the other painting half; learned maps not refit per half | a refit per half doubles builder cost and needs group matching | agent default | 2026-10-12 19:35 |
| D18 | Random control placement | Per painting, a seeded draw with the group-size shares | nearest centre of random groups is noise dressed as a map | agent default | 2026-10-12 19:35 |
| D19 | Head recipe (ii) | Painting level; MLP 512 → 512 → 128, unit output; InfoNCE, temperature 0.1, 128 negatives per query; Adam 1e-3, batch 512, 30 epochs, seed 0. Query = cached CLIP feature; positive = one of its five views, drawn per step, or its fixed neighbour. Views are CLIP B/32 features of the 36,518 scorer-train images, on the local GPU | standard small-head settings; the user fixed the grid on the two axes that matter | agent default | 2026-10-12 19:35 |
| D20 | FCMI setup | Official code with a loader for 512-d painting vectors; one shared encoder (gallery images need no content group); fairness weight × 0.25, 0.5, 1, 2, 4, 8 of the code's default for its feature-vector data; other settings at that default | 64 groups dilute the balance term, so the grid leans strong; weight 0 is not fair clustering | agent default | 2026-10-12 19:35 |
| D21 | FCMI smoke failure | The default-weight run does not finish, has a non-finite loss, ends with fewer than 2 groups, or takes over 60 min | 64 protected groups are untested (literature note, Q4), so a fixed failure test keeps a broken builder out of the pick | agent default | 2026-10-12 19:35 |
| D22 | Probe NBC | Triggered when none of the six NB scorers passes; one grouping at the strength of the NB grouping with the largest pooled s − AFF; two scorers, its random control; joins the pick | the CSD neighbour is the fallback form (D8), so it runs only when the CLIP form fails | agent default | 2026-10-12 19:35 |
| D23 | NrKmeans | On CLIP painting features (the builders' input), two subspaces of 17, `random_state` 0, the one less like the content grouping; own map in its subspace; dropped past 30 min | on CSD (round 7's N), whose numbers stand beside | agent default | 2026-10-12 19:35 |
| D24 | Told prompts | Zero-shot CLIP B/32 on cached features, "a {style} painting" for the 27 style names, groups under 200 rows merged; same reader and scorers | Multi-MaP's proxy learning: a GPU optimisation per image | agent default | 2026-10-12 19:35 |
| D25 | Thresholds | τ^X = percentiles 0, 25, 50, 75 of X's reader's seed-42 top-two margins (round 7 L12) | round 7's rule, so the thresholds stay comparable with round 7 | agent default | 2026-10-12 19:35 |
| D26 | Pick | One pick over both builders and NBC (§6) | round 7's pick per family: one main-line read only | agent default | 2026-10-12 19:35 |
| D27 | Random-control flag | If the pick's random control passes the gate with the same scorer kind, flagged at the top; the pick stands | a gate item (the user fixed the gate) | agent default | 2026-10-12 19:35 |
| D28 | CLIP-head arm | For the pick; with no pick, for the candidate with the largest pooled s − AFF | the arm shows what own assignment adds over round 7's CLIP heads (§3.6) | agent default | 2026-10-12 19:35 |
| D29 | Tier reading | Tier 2 also needs P1 to P5; any pattern outside the three tiers is a no-go | a win over AFF without one over CF would be a condition-free gain (C2) | agent default | 2026-10-12 19:35 |
| D30 | Promising and dig | Round 7's S1 to S3 pass (comparators B, B′(A0), B′(A1), B′(X), CF), this gate fails; no dig before the cut-off; after it, the next decision A may take it up | the user set round 7's gate as the weaker goal; the cut-off leaves no time to dig before 23 October | agent default | 2026-10-12 19:35 |
| D31 | "Main line fails" | No main-line pick, or its held read without Tier 1 | "no pick" only would make the halved α pointless | agent default | 2026-10-12 19:35 |
| D32 | Late read | No held read starts after Fri 2026-10-23 12:00; an unstarted read waits for the next decision A | a read after the cut-off would be left out of the paper; C5 forbids that | agent default | 2026-10-12 19:35 |
| D33 | Rules | The gate rule (both tracks) before any code; the held rule after the gate verdict, before held code, mechanics only | C7; the bar is fixed here | agent default | 2026-10-12 19:35 |
| D34 | Held regression | H5's nine episode SHA-256s; COS, RCA, B, B′(A0) recomputed equal H5's arrays; AFF loaded | recomputing AFF is a second read (C5) | agent default | 2026-10-12 19:35 |
| D35 | Timeline | Rules on 12 and 13 October; build 13 October; overnight run; gate verdict 14 October; held rule, smoke and read by 16 October | from the facts' timings | agent default | 2026-10-12 19:35 |
| D36 | Stop before the held read | None: the gate verdict goes as a phone note, and the held read follows under its rule and D34 | the read is guarded by the committed rule and the regression check; the cut-off leaves room | user | 2026-10-12 19:45 |

**Amendments** (the rule's header: after its commit, the rule changes only with the user's approval, as a row here):

| # | Before or after results | Change | Why | Who decided | When |
|---|---|---|---|---|---|
| 1 | after results (the real M4 view check) | `DECISION_RULE.md` M4: view-check threshold 0.9999 → 0.999; the run restarts from stage 0 under the new rule SHA-256; old records kept in `src/test/20261129_r8_grouping/results_m4_09999/` | the failure looks like noise: 255 of 256 cosines are ≥ 0.9999 and the minimum is 0.99983, where a real fault would give about 0.9 or lower on most paintings; 0.999 is a round value with a margin, not fitted to 0.99983. The same failure showed on the build smoke; no V grouping had a seed-42 number when the change was made | lowering: user; the value 0.999: user, after seeing the recommendation (flagged as a key decision made after the recommendation) | 2026-10-13 01:13 |

## 5. Assumptions and open questions

- Assumption: the 224 cache holds one image per scorer-train painting, and CLIP on an unchanged cached image gives its
  cached feature (cosine at least 0.999; amendment 1, was 0.9999). Checked: the build smoke and the real check both
  failed 0.9999 at min cosine 0.99983 (1 of 256 paintings); hence amendment 1.
- Assumption: views of the cached centre crop are enough; the border lost to the crop is never seen. Not checked.
- Assumption: round 7's pipeline is unchanged (stage 0) and H5's stored arrays are intact (held-ledger SHA-256s).
- Assumption: FCMI's code runs on 512-d vectors with 64 groups. Checked: stage 1.
- Assumption: Leiden reaches 17 groups for each (ii) setting at some resolution. Not checked; else the nearest count
  (rule).
- Assumption: one image per painting; the content grouping is constant within a painting. Checked: round 7 asserts.

No open questions: the grill settled the bar, the data and the scope; the rest are agent defaults (§4).

## 6. Success criteria

**Decision mode.** Bars fixed here; the two rules (D33) add mechanics only.

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| SC1 | AFF, R1[R0] and X-AFF[R0] (round 7's CLIP-head posteriors through the round 8 path), B, B′(A0), B′(A1), per anchor | round 3's and round 7's stored arrays | exactly equal (C12) | seed 42 |
| SC2 | FCMI default-weight run | D21 | passes D21 | scorer-train |
| SC3 | R@1, s − AFF, pooled, 12,288 episodes | AFF | 95% lower bound above 0 | seed 42 |
| SC4 | R@1, s − AFF, on emotion × style and on emotion × genre | AFF | Σ(4d) ≥ −49 on each (−0.3) | seed 42 |
| SC5 | H5's episode SHA-256s; COS, RCA, B, B′(A0) recomputed | H5's stored values | exactly equal | held, in the read |
| SC6 | P1 to P5: R@1, s − COS, RCA, B, B′(A0), CF[s]; P6: condition gain, s − CF[s]; P7: condition gain, s − RCA; P8: R@1, s − AFF; pooled over 36,864 episodes | as named | Holm over eight, family-wise one-sided 0.0125 | held seeds 52 to 54 |
| SC7 | tier, read from Holm's pass flags (a check not reached counts as failed) | | T1: all eight pass; T2: P1 to P5 and P8 pass, P6 or P7 fails; T3: P1 to P7 pass, P8 fails | held |

- Carry (seed 42): SC1 holds and a candidate passes SC3 and SC4; the pick goes to stage 3.
- GO (held): Tier 1; the pick replaces AFF as current best (C3); the user decides at the next decide chat how the
  CVPR paper carries it.
- Tier 2 or 3: AFF stays; the grouping enters the paper's analysis.
- Kill: no candidate passes and none is promising; or a held read in no tier. AFF stays.
- Inconclusive: a held no-go whose failed checks all have points above 0 (C11); AFF stays; no second read (C5). This
  takes precedence over the last clause of Kill.
- A failure of SC1 or SC5 stops the stage before any candidate number.

**Promising and dig budget.** D30.

**Licensed claim.** Tier 1: "<pick> beats cosine, RCA, B, B′(A0), its matched control and AFF on aspect R@1, and its
condition gain exceeds its matched control's and RCA's, pooled over three aspect pairs, on held paintings never used to
fit, select or tune it." Not: a margin per pair, other datasets or backbones, B′(A1) (untested), or the negative
sampling idea (MCL's). Tiers 2 and 3 license only an analysis statement, for example "passes AFF's seven held checks
but is not shown better than AFF".

**Selection rule** (D26). Among passing candidates, the largest pooled Σ(4d) of s − AFF; ties to the larger
gain-statistic sum, then V100, V67, V33, NB100, NB67, NB33, FC1 to FC6 (weights ascending), NBC; R1 before X-AFF. Labels and
AMIs never enter it.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 | pass | B′(A1), the strongest condition-free scorer, stands beside every number; AFF is the bar |
| C2 | pass | CF[s] for every scorer; P5 and P6 on held |
| C3 | pass | the gate and P8; only Tier 1 replaces AFF |
| C4 | exception | development on seed 42; test episodes are H5's (below) |
| C5 | pass | one read per pick (H6; H7 for the screen); all frozen; every read reported, H5 before this design disclosed; D32 |
| C6 | pass | views and centres join rows through image index and extraction sample IDs, hash-checked |
| C7 | pass | two rules, each reviewed by a fresh Opus reviewer and committed before its code (D33) |
| C8 | pass | prior §3.1; disclosure §8 |
| C9 | exception | Holm at 0.0125 over eight checks on held (below) |
| C10 | pass | the spec says otherwise (D10); C10's bar is reported within "promising" |
| C11 | pass | §6 |
| C12 | pass | SC1 and SC5 |
| C13 | pass | separate code re-derives SC3, SC4, the pick and SC6 before each verdict file |
| C14 | pass | the claim comes from held data; no fresh-seed claim |

Exceptions:
- C4 · the test episodes are H5's held episodes, so they are not fresh to the ledgers · P8 must pair with AFF's one
  held read · new held seeds would need AFF scored again on held rows, a second AFF read under C5.
- C9 · Holm at family-wise one-sided 0.0125 over eight checks in place of each 95% lower bound · the user's design
  (D11, D12): both tracks share one family-wise one-sided 0.025 · each check at 95% leaves eight claims uncorrected.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|------------------------------------------------|---------------------|-----------------|
| The head learns colour, medium or genre in place of style | AMIs with style and genre | reported; the gate decides |
| Ignoring content also removes style (round 7's projection) | AMI with style near the random control's | reported |
| The reader trains on cleaner image inputs than it scores (D17) | own-map agreement high on held-out halves, gate-open shares lower on seed 42 than in the banks | reported; a later loop refits per half |
| Four-grouping readers still lose emotion × genre | SC4 | gate fails; the promising reading |
| A pass comes from retraining the reader, not the grouping | the random control passes too | flagged (D27) |
| FCMI collapses at 64 groups | SC2 | dropped and reported |
| Late against the cut-off | gate verdict after 16 October | the screen's fallback read may not fit; D32 |
| Reused code changed silently | SC1, SC5 | stop and trace |

**Selection disclosure.** On seed 42: about 50 variants before AFF, about 10 in rounds 4 and 5, round 7's 12
candidates, and now 24 (26 with the probe) here plus 6 in the screen; references and controls pick nothing. The pick's
seed-42 margin is inflated. The held read protects the verdict; the report gives both margins side by side. Held rows
were read before (H1 to H3, value episodes; H5, AFF on these episodes), and this design follows H5's result; the paper
discloses both (C5).

## 9. Deliverables and review

- At the end: code and results in `src/test/20261129_r8_grouping/` (prefix `r8_`); view features under
  `/data/SSD2/pre_extract/artelingo/` (about 0.4 GB); the auto report with its results brief in
  `docs/reports/auto/v2/` and a user-read report; held-ledger row H6 (H7 if read); the direction log entry.
- Reviews: each rule by a fresh Opus reviewer before its commit (C7); ticket reviews of verdict code (gate, pick,
  Holm, tiers); C13 re-derivation before `gate_verdict.json` and the held verdict; an end-to-end smoke of the held-read
  code on smoke seeds 9001 to 9003 with one deliberate wiring mutation; the final whole-branch review.
- Stop points: none between approval and the reports. A stage-0 or SC5 failure ends the run with its report and a
  phone note; the gate verdict is sent as a phone note before the held read (no stop there: user, D36).
- Budget: local CPU about 7.5 hours at 8 threads (builders, groupings, about 30 readers three at a time, scoring);
  the local GPU about 1 hour under its lock (five views of 36,518 images); the probe about 1 CPU hour more; the held
  read under 1 hour. No cluster. Time-box: Fri 2026-10-23 23:59 (D3, D32).

Body length: about 280 lines against a target near 240 (claim test about 150 plus three further stages); the overrun
is §4's 35 decision rows, 13 of them the grill's.
