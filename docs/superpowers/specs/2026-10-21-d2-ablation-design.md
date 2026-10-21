# D2 building-part ablation and the four Qwen views: design

> Date: 2026-10-21 03:35 · Status: approved 2026-10-21 03:30 (closing summary confirmed by the user; brief check in
> the morning) · Kind: experiment
> Builds on: [model sheet](../../reports/auto/percept/2026-12-16_model_sheet.md), [claim test stage 2](../../reports/auto/percept/2026-12-19_claim_test_stage2.md) · Constitution: `docs/superpowers/constitution.md`
> Decision rule: none (descriptive)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** We remove D2's parts one at a time to see what each part brings. We also measure how the four Qwen
image embeddings (default and the instructed emotion, style and genre views) differ. You asked for this because "we
did not really understand our current design". It comes before more datasets and backbones.

**Plan.**
1. A regression check (stage 0): D2 and T1 (D2 without the teacher) rerun and must match the model sheet bit for bit.
2. The ablation (stage 1): 13 rows, R0 (D2) to R12, each changing one part. Development data, 3 seeds, local 3090.
3. The analysis (stage 2, exploration): what each row changes, plus the four-view analysis. CPU only.
4. Beside it, in the run chat: a literature check for ways to improve, a note on our advantages and framing, and one
   fact for the morning: how ArtELingo-28's paintings overlap our train and held paintings.

**Compared against.** D2 itself (R0). The tables also show the instructed Qwen image (the target, 0.093 / 0.395 /
0.496 emotion / style / genre) and PercepT core (0.077 / 0.229 / 0.436), development numbers from the model sheet.

**Outcomes.** No bar: an attribution table, with each part's effect and its interval per aspect. Then a list of ways
to improve, for the next decision. We expect the teacher and the graphs to carry most of the gain. If R11 (no
instruction anywhere) stays near PercepT core, the "any backbone" story needs the Qwen teacher.

**Cost.** About 25 min of GPU and 1 h of CPU, local. The build runs tonight, then the run; reports by morning.

**Check these.**
- Three rows need new switches: R3, the default teacher; R9 and R11, a default-view graph; R10, random pairs with the
  same degree. Their exact forms are agent defaults (§4 D13 to D17).
- The rows are yours, "probably right, I can't tell now"; add rows in the morning if the table shows a gap.
- The 5,000 evaluation paintings already chose D2, so removals may look worse than they are (§8).

## 1. Context and goal

Asked: "(1) A (quick) building part ablation on our best method, understand exactly what these building part bring to
us. (on cluster, 3 seeds) (2) After the ablation/sweep, do deep analysis on them, find potential way to improve more
(cause now only emotion exceeds qwen test time instructed). Also do deep analysis on four kinds of embeddings from qwen
(default + three prompts) and see how they really differ and correlate. (3) Then list ways to improve more [...] (4)
Think about our advantages and how to show our advantages [...] These are sweep and analysis work so no bar to set."

D2 passed both claim tests on held emotion (direction log, 2026-10-21) but stays below the instructed target on style
and genre. Its recipe came from development exploration; no clean leave-one-out exists (f1 facts §2: only T1 is
exactly D2 minus one part).

Goal: an attribution of D2's development result to its parts, and a map of the four Qwen views.
Question this step answers: which parts carry D2's gain per aspect, and where the room to improve lies.

## 2. Scope

In: stages 0 to 2 below; the run chat's side work (items 3 and 4 drafts, the ArtELingo-28 overlap fact).
Out: other backbones, datasets and the ArtELingo-28 build (item 5, decided in the morning decide chat); held data;
any change to D2's recipe; new methods (they enter the improvement list only).

## 3. Design

Inherits the model sheet spec (`2026-10-19-stage2-model-sheet-design.md` §3) for the head, losses, settings, early
stopping and scorer. Changes below.

### 3.2 Arms and controls

Each row changes one part of R0; "switch" means new code, default-off, that leaves R0 bit for bit.

| Row | Change from D2 | What it tests | How |
|---|---|---|---|
| R0 | none (D2: trunk, form a, graphs Gres / Fres05 / R0o, direct distillation 0.5) | reference | existing |
| R1 | separate heads, no shared trunk | the trunk | existing flag |
| R2 | no distillation (= T1) | the teacher | existing flag |
| R3 | teacher = the default Qwen image of the same painting | whether the teacher needs the instruction | switch |
| R4 | distillation only, no buddy and no own-caption terms | the graph terms | existing flag (D13) |
| R5 | no buddy terms; own-caption terms and teacher kept | the buddy pairs | switch |
| R6 | no own-caption terms; buddy terms and teacher kept | the own caption | switch |
| R7 | image endpoints only: image–image buddy term and teacher | the caption side | switch (D16) |
| R8 | plain instructed graphs: emotion R0, style R0 (no content subtraction; genre R0o as D2; amendment A1) | the content correction | existing graphs |
| R9 | one default-view graph for all three heads | whether the graphs need the instruction | switch (D14) |
| R10 | random pairs with each node's degree kept, per aspect | whether graph structure matters | switch (D15) |
| R11 | R9 and R2: default-view graph, no teacher | D2 with no instruction anywhere | R9 + R2 |
| R12 | instructed inputs on both sides (three passes) | the input ceiling | existing flag |

References, no training (model sheet numbers): the default Qwen image, the instructed Qwen image, PercepT core.

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---|---|---|
| Model sheet sample: 13,400 fit, 1,489 early-stopping, 5,000 evaluation paintings | development | 0, 1, 2 |
| Model seeds 0, 1, 2; k-means random states 0, 1, 2 | development | 0, 1 |
| Fit-side instructed images (14,889) | development (four-view analysis) | 2 |
| The 12,281 held paintings | not touched | none |

### 3.4 Measurement

- Main: image-side AMI with the own label per aspect, k-means k = 20, mean over random states 0 to 2, then over model
  seeds. Effect of a row: its AMI minus R0's, with a 95% bootstrap interval over evaluation paintings (paired across
  rows) and model seeds (resampled per row), 2,000 draws, the same draws for every row (D19, as the claim test).
- Beside it: caption-side AMI; a Leiden reading (union 20-NN graph, resolution 1.0, 3 seeds, as the claim test); a
  linear probe (D18); NMI and ARI in the JSON.
- Seeds: 3, as the model sheet; the effects we look for (0.005 and up) sit above its seed spread (0.001 to 0.008).

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check (C12) | R0 and R2 through the new code, all switches off, seeds 0 to 2 | exported vectors equal the model sheet's D2 and T1 outputs bit for bit | stop the ablation, trace the cause, report it; stage 2's four-view part still runs |
| 1 | ablation | R1 and R3 to R12, seeds 0 to 2 | all rows trained and scored | a failed row is reported as missing; the rest go on |
| 2 | exploration | stage 1 outputs; the four Qwen views | none | — |

### 3.6 Diagnostics (stage 2; they decide nothing)

Per row (ablation analysis):
- linear CKA and top-20 neighbour overlap of each head output with its aspect's teacher and with the default input:
  does a part pull the head toward the teacher;
- cluster size entropy of the k-means partition: balance;
- the gap between probe accuracy and AMI: is information present but not clustered;
- per-class recall from the probe and the class-to-cluster table: which classes separate.

The four Qwen views (5,000 evaluation paintings; fit side where it exists):
- pairwise linear CKA and top-20 neighbour overlap, image side and caption side (6 pairs each);
- each view against each label: AMI (k-means 20) and the probe, a 4 × 3 table per side;
- the instruction offset (instructed minus default, per painting): its effective rank, the principal angles between the
  three offsets' top subspaces, and how much of each label's probe accuracy the offset alone carries;
- image against caption agreement per view (CKA, neighbour overlap).

### 3.7 Frozen vs rerun

Frozen: the sample, graphs Gres, Fres05, R0, R0o, the feature stores, the scorer. Rerun: every row, R0 included.
New: the default-view graph (R9, R11) and the random graphs (R10), built on fit and early-stopping paintings only.

## 4. Decisions

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Direction (A) | the five-part plan; this spec covers parts 1, 2 and the run chat's drafts of 3 and 4 | understand the design before more datasets | user | 2026-10-21 02:50 |
| D2 | Stage types | 0 regression, 1 ablation, 2 exploration | 1 removes parts; 2 steers direction | agent default | 2026-10-21 03:35 |
| D3 | Device | local RTX 3090 (option a) | short run; bit-for-bit with the model sheet (node408 does not reproduce) | user | 2026-10-21 03:10 |
| D4 | Rows | R0 to R12 as §3.2 | "probably right, I can't tell now" | user, after seeing the proposal | 2026-10-21 03:10 |
| D5 | Data | development only; held not touched | the ablation shapes the next design (`held-data.md`) | user, after seeing the recommendation | 2026-10-21 03:10 |
| D6 | Seeds | model seeds 0 to 2; k-means k = 20, random states 0 to 2 | as the model sheet | user, after seeing the recommendation | 2026-10-21 03:10 |
| D7 | Scores | §3.4 | as the claim test's readings | user, after seeing the recommendation | 2026-10-21 03:10 |
| D8 | Analysis lists | §3.6 | answers "how they differ and correlate" | user, after seeing the recommendation | 2026-10-21 03:10 |
| D9 | Output | no bar; an auto report and a user-read report | sweep and analysis work | user | 2026-10-21 02:50 |
| D10 | Plan page | none; summary in chat | | user | 2026-10-21 03:10 |
| D11 | Item 5 | grilled in the morning decide chat | the user goes to sleep | user, after seeing the recommendation | 2026-10-21 03:20 |
| D12 | Restart before the build | none | the container restarted at 02:56; GPU check passes | user | 2026-10-21 03:20 |
| D13 | R4's batches | 1,024 paintings per batch (D1's batch), distillation weight 0.5 | without edges there is no edge batch; weight as D2 | agent default | 2026-10-21 03:35 |
| D14 | R9's graph | reciprocal top-20 on the content score (mean of standardized default caption and default image cosine), one graph for all heads | the default-view analogue of the aspect graphs, built the same way | agent default | 2026-10-21 03:35 |
| D15 | R10's graph | per aspect and partition, a degree-preserving random pairing of D2's graph (no self-pairs), seeded by the model seed | keeps batch composition; a uniform random graph would change degrees too | agent default | 2026-10-21 03:35 |
| D16 | R7 | image–image buddy term and the teacher; the caption head is not trained; R7 has no caption AMI | the clean "no caption side" row | agent default | 2026-10-21 03:35 |
| D17 | R3's teacher | the default Qwen image of the same painting, same weight and temperature | the matched control for the instruction in the teacher | agent default | 2026-10-21 03:35 |
| D18 | Probe | 5-fold stratified logistic regression on the 5,000 evaluation outputs, balanced accuracy | the sheet exports evaluation outputs only | agent default | 2026-10-21 03:35 |
| D19 | Interval | paired bootstrap over paintings and seeds, 2,000 draws, percentile | as the claim test | agent default | 2026-10-21 03:35 |
| D20 | Side work | literature check on the `deep-research` agent (Opus, deep: the paper's central gap); framing note by an Opus `xhigh` worker; the ArtELingo-28 overlap check on CPU | design judgment; the overlap is a fact item 5 needs | agent default | 2026-10-21 03:35 |

**Amendments:**
- A1 (2026-10-21 04:45, agent default in night mode, from the final review; check it in the morning): R8's style graph
  is R0, not R0o. In the aspect store, style R0 is the reciprocal top-20 graph of the instructed style *image* view
  (the view Fres05 corrects) and R0o that of the style *caption* view; Fres05 shares 13.5% of its fit edges with R0 and
  0.07% with R0o. With R0o, R8 would change the view as well as the content correction. Emotion R0 (the caption view,
  as Gres) and genre R0o (D2's own graph) stay. Also: R4's caption side is not scored, since no loss reaches its
  caption network (as R7, D16). Wording only: D19's draws pair paintings across rows and resample model seeds per
  row, the claim test's scheme. Places changed: §3.2, §3.4, §5, `ab_rows.py`, `ab_train.py`, `ab_score.py`, tests.

## 5. Assumptions and open questions

- Assumption: the model sheet's saved D2 and T1 outputs reproduce bit for bit on today's 3090 and driver. Checked: the
  claim test's stage 2a reproduced stage 1 bit for bit on 2026-10-20; the model sheet itself is not rechecked yet.
- Assumption: graphs R0 (emotion) and R0 (style) exist in the aspect graph store. Checked 2026-10-21 04:40: both load
  with `ags_head.load_graph` (amendment A1).
- Open questions: none.

## 6. Success criteria

**Descriptive mode.** No decision rides on these numbers; the type is ablation and exploration (D9).
Reading: a part whose removal costs more than its interval on an aspect carries that aspect's gain. R3 and R9 tell
whether the instruction acts through the teacher or the graphs. R11 near PercepT core would mean "any backbone" needs
the Qwen teacher; R11 well above it would make a backbone-free story possible. R12 far above D2 would point to the
input as the limit for style and genre.

**Licensed claim.** "On 5,000 development ArtELingo paintings (3 seeds), removing <part> from D2 changes <aspect>
image AMI by <d> [interval]." It is an attribution on development data only, as strong as the model sheet result. It
says nothing about held data, other datasets or backbones.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | D2 and the instructed target beside every row |
| C6 sample IDs | pass | joins by the sample's painting IDs; new graphs index each partition's node order |
| C7, C13 rule and re-derivation | not applicable | no verdict |
| C8 disclosure | pass | §8 |
| C12 regression first | pass | stage 0 |
| `held-data.md` | pass | held not touched |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|------------------------------------------------|---------------------|-----------------|
| Selection: the 5,000 evaluation paintings chose D2 from 13 settings and earlier sheets, so R0 is optimistic and removals look costly | effects near the seed spread | report effects beside R0's own spread; read small effects as none |
| A switch changes R0's path | stage 0 fails | stop the ablation; report the cause |
| Separate early stopping per aspect hides joint effects in trunk rows | best epochs differ a lot between rows | report best epochs per row |
| Three seeds miss small effects | intervals cross 0 | say "not resolved", not "no effect" |

## 9. Deliverables and review

- At the end: round folder `src/test/20261021_d2_ablation/` (new module prefix), outputs gitignored; the auto report
  in `docs/reports/auto/percept/` with a results brief, the user-read report in `docs/user_read/`; the literature note
  in `docs/reports/literature/`; the framing note in `docs/reports/auto/percept/`; a facts note on the ArtELingo-28
  overlap for the morning decide chat.
- Reviews: build per `matt-chain.md` (no verdict code: the controller's diff read, unit tests for the new switches
  and graphs, one end-to-end smoke, the final whole-branch review); the auto report gets Claude's claim review.
- Stop points: none between approval and the reports. Budget: about 25 min GPU (39 row-seed runs at about 35 s, plus
  R12's instructed inputs), about 1 h CPU, local 3090 under the GPU lock. Time-box: reports by 2026-10-21 10:00.
- Length: the body is about 170 lines against a target of about 110, because of the 13-row arm table, the 20-row
  decision table and two analysis lists.
