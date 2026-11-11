# r21: a DEC loss on every head of c*: design

> Date: 2026-11-11 06:30 · Status: approved 2026-11-11 06:31 · Kind: experiment
> Builds on: to-do item 4 of the [experiment overview](../../user_read/2026-11-03_experiment_overview.md); the
> [r21 decide handoff](../handoffs/2026-11-11-r21-decide-handoff.md); the ablation table of reports 2026-12-28 and
> 2027-01-13 · Constitution: `docs/superpowers/constitution.md` (amended 2026-11-10)
> Decision rule: none (descriptive: a sweep, then ablation rows; no claim ladder)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** We add a DEC loss (Xie et al. 2016: each image's soft assignment to cluster centres is trained toward
a sharpened copy of itself) to all three heads of c*, inside c*'s training, the way PercepT trains DEC. A reviewer can
ask whether PercepT's clustering loss would help our model. The answer becomes rows in the paper's ablation table.

**Plan.**
1. Stage 0, regression check: the new training code with DEC off reproduces c*; the scoring code reproduces Table 2's
   c* and teacher numbers.
2. Stage 1, sweep (validation, seed 0): DEC's settings, about 13 short runs, then one setting per form. The three
   forms: plain DEC; DEC with a pull toward the start clusters; DEC with a pull toward the teacher's clusters.
3. Stage 2, ablation (test set, seeds 0 to 4): each form at its setting, against c*. Stage 2 waits for stage 1's picks.

**Compared against.** c* itself (the saved models of the paper's tables), as every ablation row is. A rerun of c* by
the new code, with DEC off, shows how much a rerun alone moves the numbers.

**Outcomes.** No bar. Each form gives one row: its change from c* in test-set AMI at the class count (9, 27, 10) and
in image retrieval mAP, per aspect, each with a 95% interval, worded "raises" (interval above zero), "lowers" (below)
or "no clear change" (across zero). We expect (literature check of 10-20; our September runs): plain DEC at best
level with c* and possibly below; the pulls closer to c*; the teacher pull below c* on emotion, where the teacher's
clusters are weaker (AMI 0.089 against c*'s 0.102). Silhouette (cluster tightness) should rise in every form.

**Cost.** GPU about 3 to 4 h in short jobs (one c* training takes 4 to 5 min), CPU about 1 to 2 h for scoring. Build
today; runs in the day if the GPU draws under 200 W, else from 19:00. Results to read on 11-12.

**Check these.**
- The stop rule (our default, changed after the grill): c*'s heads reach their best validation loss early (epochs 4,
  1 and 18 for seed 0). Stopping on that loss alone would end DEC after one or two epochs, so the sweep also runs
  DEC's own stop rule (fewer than 0.1% of paintings change cluster).
- The teacher-pull form was your "maybe", then a yes after our recommendation. Its row mixes DEC with a second
  teacher signal and carries its own label.
- Settings are picked on one seed of validation; with about 13 settings, the picked rows' validation numbers are
  optimistic. The test rows are not affected.
- DEC trains on the image side only; the caption side of c* is unchanged.

**Morning checks.**
- `src/test/20261111_r21_dec_heads/` log: stage 0 gates passed.
- The stage 1 number sheet: did a DEC run collapse (one cluster with most paintings)?
- The stage 2 table: AMI and mAP change per form and aspect, with intervals.
- The rerun-noise row (c* rerun with DEC off) against the saved c*.
- Silhouette per form beside AMI.

## 1. Context and goal

Body about 150 lines, over the 90 to 110 target: two stages, three regression gates and 17 decision rows.

Asked: "r21 runs now, beside the paper writing, not after 11-16. Scope: DEC only, to-do item 4: a DEC loss on every
head, in its two planned forms (plain, and with a pull toward the start clusters), with retrieval reported too."
(user, 2026-11-11 05:40, recorded in the decide handoff). The grill's answers are in §4.

The ablation table (reports 2026-12-28, 2027-01-13) has eight removals of c*'s parts and no row that adds one. PercepT
uses DEC; its own ablation reports DEC +0.02 AMI on ArtELingo and −0.03 on Affection (literature note
`2026-12-17_cluster_level_losses.md`, Q3), which found no published gain from DEC on strong features.

Goal: measure what a DEC loss on every head does to c*'s clustering and retrieval on the test set.
Question this step answers: does PercepT's clustering loss, added to c*, change c*'s AMI and mAP, and does a pull
toward fixed clusters change the answer?

## 2. Scope

In: ArtELingo, Qwen3-VL-Embedding-2B features, c*'s recipe (`k11=40`), three DEC forms, validation sweep on seed 0,
test set on seeds 0 to 4, AMI at the class count and at 20 clusters, image-to-image label mAP, silhouette.
Out: the deeper head (to-do 18, waits); other encoders, MMArt, photos; DEC on the caption side; one DEC arm per
aspect; the silhouette figure (decided at the read); any change of c* in the paper's other tables. Parked ideas go to
the handoff.

## 3. Design

### 3.1 Hypothesis and prior

Prior as the brief's "Outcomes": the field reports drift from DEC on strong features (literature note Q3), so plain
DEC should leave AMI level or lower it, most on genre (our September runs); the start pull should stay near c*; the
teacher pull should move each head toward a partition weaker than c*'s on emotion and close on style and genre.
A surprise: plain DEC raising AMI on all three aspects by more than one validation seed range (0.004, 0.009, 0.014).

### 3.2 Arms and controls

| Arm | What it is | Differs from c* by | Role |
|-----|------------|--------------------|------|
| c* | saved models, seeds 0 to 4 | – | comparator (the ablation table's reference) |
| c* rerun | new code, DEC off, seeds 0 to 4 | rerun only | control for rerun noise |
| DEC plain | c*'s losses, then from warm-up end also λ·KL(P‖q) on each head | the DEC term | ablation arm (addition) |
| DEC + start pull | as plain, plus λ·KL(q₀‖q), q₀ the soft assignments at DEC start, fixed | DEC term and the pull | ablation arm |
| DEC + teacher pull | as plain, centres started at the means of the teacher's clusters, plus λ·KL(t‖q), t the teacher's soft assignments | DEC term and a teacher pull | ablation arm |

DEC follows our PercepT build (`src/test/20261018_percept_core/pc_sept.py`): Student-t soft assignments (α = 1) of the
unit head outputs to learnable centres, k = the class count, and a target P recomputed once per epoch from a full pass
over the training paintings (the EM step). c*'s own losses stay on throughout, in the place of PercepT's reconstruction
loss. All arms share c*'s settings, data, graphs and seeds.

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---------------|------|---------------|
| ArtELingo train paintings (42,969) | training; DEC targets, k-means starts, the teacher's clusters | 0, 1, 2 |
| ArtELingo validation (6,152), seed 0 | development: early stop (as c*), DEC settings picked | 0, 1 |
| ArtELingo test set (12,281; genre labels on 9,949), seeds 0 to 4 | test: the ablation rows | 0 (scoring gate), 2 |

Seed 0 on validation was already used by c*'s own picks; the test set shapes no choice in r21.

### 3.4 Measurement

- AMI: as Table 2 (r17's procedure): unit rows, MiniBatchKMeans (3 inits, batch 4,096, states 0, 1, 2), averaged over
  the states, at the class count; also at 20 clusters (report 2026-12-28's count). All test paintings are clustered;
  unlabelled genre rows are masked only in scoring.
- mAP: the ablation table's read-out (report 2027-01-13): told image-to-image retrieval, label mAP, plain cosine, the
  query excluded.
- Changes: arm minus c*, paired by seed. Intervals as in the ablation table: AMI by a painting and seed bootstrap with
  2,000 draws (report 2026-12-28), mAP by a shared query and seed bootstrap with 10,000 draws (report 2027-01-13).
  No correction across cells, as in the existing table; the report says so.
- Five seeds, the ablation table's count.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|-------|------|-------|--------------|-------------|
| 0a | regression (scoring) | saved c* and teacher test embeddings | c* and teacher AMI at the class count equal r17's file to 6 decimals (0.102413, 0.410926, 0.510070; 0.088875, 0.406315, 0.501466); c* mAP equals report 2027-01-13's | stop, trace, tell the user |
| 0b | regression (training, CPU) | saved c* seed 0 weights, a fixed batch | new code with DEC off gives the frozen code's loss to 1e-6 | as above |
| 0c | regression (training, GPU) | c* seed 0 | DEC-off rerun: best epochs within 2 of 4, 1, 18; validation AMI within 0.004 of the saved run's (0.1010, 0.4206, 0.5298) | as above |
| 1 | sweep | validation, seed 0 | every run finishes; no NaN | fix bugs only (constitution, I-decide row 4) |
| 2 | ablation | test set, seeds 0 to 4 | every arm has 5 seeds | as above |

The validation AMI for 0c is computed by the same code as stage 1 and compared with
`src/test/20261103_new_scores/records/stage1_ranges.json`.

**Stage 1 grid** (agent defaults; λ is the DEC weight, W the warm-up in epochs of c*'s losses only):
- 1a, plain form: λ ∈ {1, 10} × W ∈ {1, 5} × stop rule ∈ {S1, S2}: 8 runs.
  - S1, DEC's own rule: a head stops when fewer than 0.1% of training paintings change cluster between epochs, at
    most 50 DEC epochs; its last state is kept.
  - S2, c*'s own rule: c*'s validation loss without the DEC term, patience 15 counted from DEC start; the best state
    among DEC epochs only.
- 1b, at 1a's picked λ, W and stop rule: the three forms × PercepT's balance term (off, on): 6 runs, one shared with
  1a. The pulls use the weight λ.

**Selection rule** (per form): the highest mean over the three aspects of validation AMI at the class count minus c*'s
seed 0 validation AMI. Tie band 0.002; ties go to the higher mean validation mAP, then to the smaller λ.

### 3.6 Diagnostics

Measured after the stage 2 table is written; they decide nothing.
- Silhouette (cosine) of each arm's test embeddings against its own k-means partition at the class count (state 0),
  per aspect; also for c*, the plain Qwen image and the teacher. Tells whether "tighter" goes with AMI.
- Cluster sizes: the largest and smallest cluster share per head (collapse check).
- DEC epochs run and the stop epoch per head; the fraction of paintings that left their start cluster.
- Validation AMI and mAP of the picked settings on seeds 0 to 4.

### 3.7 Frozen vs rerun

Frozen: c*'s recipe and saved models, the scoring code of r17 and r20, the test set, and stage 1's picks once written.
Rerun: every DEC arm and the DEC-off control, seeds 0 to 4, by the new code.

## 4. Decisions

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | When and what | r21 now, DEC only (item 4); the deeper head waits | paper deadline; item 18 later | user (deeper head: user, after seeing the recommendation) | 2026-11-11 05:40 |
| D2 | What the result is for | Ablation rows beside the eight removals; c* stays the paper's model | a claim test (1b) or validation only (1c) | user | 2026-11-11 06:15 |
| D3 | What DEC starts from | Inside c*'s training after a warm-up, target recomputed each epoch, c*'s losses kept, as PercepT | fine-tuning saved c* (2a, our recommendation) | user | 2026-11-11 06:15 |
| D4 | The forms | Plain, start pull, teacher pull | the two planned forms only | user, after seeing the recommendation (first "maybe") | 2026-11-11 06:22 |
| D5 | Silhouette | Measured beside AMI and mAP; the figure decided at the read | – | user, after seeing the recommendation | 2026-11-11 06:22 |
| D6 | Arms per aspect | One joint row on all three heads; none per aspect | per-aspect arms | user, after seeing the recommendation | 2026-11-11 06:22 |
| D7 | Retrieval, data, k, timing | The ablation table's mAP; ArtELingo only; k = class count; build today, run today or tonight | – | user, after seeing the recommendation | 2026-11-11 06:15 |
| D8 | Stage types | Stage 1 sweep, stage 2 ablation | claim test: excluded by D2 | agent default | 2026-11-11 06:30 |
| D9 | Where DEC acts | Image-side heads only | caption side too: clustering and the mAP column read the image side | agent default | 2026-11-11 06:30 |
| D10 | Grid and stop rules | §3.5: λ, W, S1/S2, then forms × balance; pull weight = λ | S2 alone: ends DEC after 1 to 2 epochs (best epochs 4, 1, 18) | agent default | 2026-11-11 06:30 |
| D11 | Selection rule | §3.5 | mAP first: DEC is a clustering loss | agent default | 2026-11-11 06:30 |
| D12 | Teacher's clusters | k-means (k = class count, 10 inits, state 0) on the unit instructed Qwen image of the training paintings, per aspect, Table 2's teacher view; t = Student-t (α = 1) soft assignments to those centres in the teacher's space | content-corrected views: Table 2's teacher is uncorrected | agent default | 2026-11-11 06:30 |
| D13 | Comparator and intervals | Saved c*, paired by seed; intervals as reports 2026-12-28 and 2027-01-13, no correction across cells; AMI at the class count and at 20 clusters | the rerun as reference: the table's other rows use saved c* | agent default | 2026-11-11 06:30 |
| D14 | Words per cell | "raises", "lowers", "no clear change" by the interval's sign | a bar: excluded by D2 | agent default | 2026-11-11 06:30 |
| D15 | Gates | §3.5 stage 0; 0c tolerances: best epochs within 2, AMI within 0.004 (the emotion seed range) | exact GPU rerun: CUDA training is not bit-exact (r19 rule review) | agent default | 2026-11-11 06:30 |
| D16 | Re-derivation | Codex re-derives the stage 2 change table from the saved embeddings and the rules of §3.4, without our numbers | none: the rows go into the paper | agent default | 2026-11-11 06:30 |
| D17 | Device | Local GPU through `res-queue`, short jobs; in the day if under 200 W, else from 19:00; DAS6 not needed | DAS6: a few hours of small jobs | agent default | 2026-11-11 06:30 |

**Amendments**: none.

## 5. Assumptions and open questions

- Assumption: c*'s early stop reads the validation split ("validation is dev" in `ct2_data.py`). Checked: code comment
  only; the sidecar's dev size of 1,489 is not explained (not checked).
- Assumption: a training run with DEC takes under 10 minutes. Checked: not checked (c* alone 3.5 to 5.3 min).
- Assumption: the teacher's instructed image features of the training paintings exist in the feature cache. Checked:
  not checked; stage 0a's teacher gate covers the test set only.
- No open questions.

## 6. Success criteria

**Descriptive mode.** No decision rides on these numbers: D2 makes them ablation rows, and c* stays the model.

| ID | What is measured | Compared against | Level 1 | Level 2 | Level 3 | On which data |
|----|------------------|------------------|---------|---------|---------|---------------|
| SC1 | Stage 1 number sheet: validation AMI and mAP per run; one pick per form by §3.5's rule | c* seed 0 validation | every run and the picks reported | – | – | validation |
| SC2 | Per form and aspect: test AMI change at the class count, with its interval and word (D14) | saved c* | every cell reported | – | – | test |
| SC3 | As SC2, mAP | saved c* | every cell reported | – | – | test |
| SC4 | As SC2 at 20 clusters; the c* rerun row | saved c* | every cell reported | – | – | test |

Licensed claim: "On the ArtELingo test set with Qwen3-VL-Embedding-2B, adding a DEC loss <form> to every head of c*,
at settings picked on validation, changes AMI by Δ [interval] and mAP by Δ [interval]." Not licensed: other encoders
or datasets, other DEC settings, "DEC never helps", or any cause.
Reading: a form that raises AMI on all three aspects without lowering mAP would make DEC worth a later loop (decision
A at the read); a form that lowers AMI supports the method section's choice not to use a cluster loss.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | c*, the model the rows modify; Table 2 gives the outside comparators |
| C6 sample IDs | pass | joins by extraction IDs; gate 0a covers the teacher join on the test set |
| C7 rule before code | pass | no claim ladder |
| C8 prior and disclosure | pass | prior §3.1; disclosure §8 |
| C12 regression first | exception | 0a and 0b are exact; 0c has a tolerance |
| C13 re-derivation | pass | D16 |
| `held-data.md` | pass | the test set shapes no choice |

Exception (C12): a CUDA rerun of c* is not bit-exact (r19's rule review), so the GPU gate 0c uses a tolerance. The
exact checks 0a and 0b still cover the scoring code and the DEC-off loss. A bit-exact deterministic mode was rejected:
it changes c*'s training from the saved runs.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|------------------------------------------------|---------------------|-----------------|
| DEC collapses one head to few clusters | cluster sizes (§3.6) | report it as the result of that form |
| The stop rule cuts DEC so early that the arm equals c* | DEC epochs run (§3.6); changes near zero | report; the S1 runs in stage 1 show the longer case |
| Rerun noise is read as a DEC effect | the c* rerun row (SC4) | read each change beside the rerun row |
| Higher silhouette is read as better clusters | silhouette beside AMI (§3.6) | the report states both |
| Selection disclosure: about 13 settings on validation seed 0, so the picks' validation numbers are optimistic | – | stage 2 on the test set, five seeds; the report gives the count |

## 9. Deliverables and review

- At the end: code in `src/test/20261111_r21_dec_heads/` (module prefix `r21_`); outputs in
  `/data/SSD2/cosir_r21/20261111_dec_heads/`; the auto report in `docs/reports/auto/percept/` and a user-read report.
- Reviews: the spec self-review; the build per `matt-chain.md` (the DEC loss and the scoring are verdict-like code:
  ticket reviews and unit tests); the final whole-branch review; D16's re-derivation before the report.
- Stop points: none between approval and the reports. Time-box: results by 2026-11-12 12:00.
- Budget: GPU about 3 to 4 h of short jobs on the local GPU (stage 0c 5 min, stage 1 about 13 runs, stage 2 20 runs);
  CPU about 1 to 2 h; disk under 5 GB.
