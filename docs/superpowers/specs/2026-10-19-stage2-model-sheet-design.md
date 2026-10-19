# Stage 2 model sheet: one trunk, three heads, toward the instructed image in one pass: design

> Date: 2026-10-19 20:10 · Status: draft · Kind: experiment
> Builds on: [aspect graph spec](2026-10-18-aspect-graph-sweep-design.md) (base; its stage 2 moves here), the grill in
> the [direction log entry of 2026-10-19](../direction_log.md), the
> [ceiling check](../../../src/test/20261019_emotion_ceiling/20261019_emotion_ceiling_log.md) · Constitution:
> `docs/superpowers/constitution.md`
> Decision rule: none (descriptive; every stage is an exploration)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** Stage 2 of the aspect graph sweep, with your new target: one pass on the default Qwen image that
reaches the instructed Qwen image clustered directly (image-side AMI 0.093 / 0.395 / 0.496 for emotion, style,
genre). Today our network reaches 0.081 / 0.348 / 0.458. "Instructed" means Qwen reads the image with an aspect
instruction, so it needs three passes.

**Plan.** All stages are explorations on development data, one number sheet.
0. Regression check: the new training code reproduces stage 1's heads exactly. If it fails, the run waits for you.
1. Sheet on one seed: one shared trunk with three heads, under the three loss forms (a, b, c); graphs picked on the
   image side; a pull from each head toward the instructed image of its aspect (distillation), at three weights,
   alone and on top of the graphs.
2. Seeds 1 and 2 for the top rows, the matched control and the plain trunk.
3. Second result: the best row with instructed inputs.
Then V, ARI and FMI (beside AMI) for every row, and once for the two-bar table of the K = 40 report.

**Compared against.** In every table: PercepT core, best arm per aspect (0.077 / 0.229 / 0.436); the image encoder
clustered directly (0.070 / 0.300 / 0.449); the target; the linear probe as a reference ceiling (0.121 / 0.527 /
0.657). The matched control for the trunk: three separate heads on the same graphs.

**Outcomes.** No go or kill. What changes direction: (a) a row reaches the target on all three aspects: it becomes
the model of the 10-29 claim test; (b) only on some: a second stage 2 loop on the others (your call); (c)
distillation alone is level with the best row: the graphs add nothing on the image side, and we talk before the
claim test; (d) the trunk below separate heads everywhere: we bring the one-trunk choice back to you. We expect
distillation to lift style and genre most, emotion to stay near 0.09, and forms b and c to tie with a.

**Cost.** Build a few hours, unattended, from tonight. Run right after: about 1 h GPU and 2 h CPU; all ends by about
08:00, before 11:00. You hear back with the reports on 2026-10-20.

**Check these.**
- Taken after seeing our recommendation: decision A, the target's timing (10-29 gate, the instructed bar judged with
  retrieval on 11-10), the extra measures, minor m1 to m6.
- Agent default: a new spec for stage 2 rather than an amendment of the aspect graph spec.
- Agent default: distillation pulls the default-image head output toward the instructed image embedding through the
  shared trunk; weights 0.5, 1, 2; one row without graphs.
- Agent default: graphs picked on image-side AMI. Emotion takes the residual graph with a per-painting weight
  (0.091), not the purified graph (0.089) that the last handoff named; both lie within the seed spread.
- Agent default: the trunk is the first layer of stage 1's head, shared; each head keeps its own second layer.

## 1. Context and goal

Asked (the grill, 2026-10-19; the user's answer as the direction log records it, not verbatim): reach "the
instructed Qwen image clustered directly (0.093 / 0.395 / 0.496) with one pass on the default image; may take one or
two loops". The direction log entry of 2026-10-19 holds every grill answer.

Stage 1 (base spec) trained one cross-modal head per aspect on buddy graphs (pairs of paintings that are near
neighbours in an aspect view). The best default-input heads beat PercepT core clearly only on style; the instructed
image beats them on all three aspects. The ceiling check shows that the default image carries most of the label
signal (probe 0.121 / 0.527 / 0.657 against 0.130 / 0.569 / 0.658 instructed).

Goal: a number sheet that shows which parts bring a one-pass model to the instructed image.
Question this step answers: which trunk, loss form, graphs and distillation to carry into the 10-29 claim test.

## 2. Scope

In: emotion, style and genre heads; default Qwen image and caption inputs (instructed inputs as a second result);
the base spec's frozen train-side sample; V, ARI and FMI in the shared scorer and the two-bar back-fill.
Out: CLIP inputs (m4); retrieval (stage 3 of the base spec); the held-out claim test (its own spec after this sheet);
learned aspects and Affection (base spec §2).

## 3. Design

Inherits the base spec's head (§3.4: two-layer MLP per side, buddy InfoNCE terms plus own-caption terms, its
settings and early stopping on dev loss). Changes below.

### 3.2 Arms

The trunk: the first layer (2048 to 1024) of each side's MLP is shared by the three aspects; each aspect keeps its
own second layer (1024 to 2048). Graph rows by image-side AMI (D13): emotion Gres, style Fres05, genre R0o; second
picks R1, Gres, Fside025; stage 1 picks Fres1, Fres05, R0o.

| Row | What it is | Differs from T1 by | Role |
|---|---|---|---|
| A1 | three separate heads (stage 1 code), image-side picks | no shared layer | matched control |
| T1 | trunk, form a, image-side picks | — | candidate (base row) |
| T2 | T1 on stage 1 picks | emotion graph (Fres1) | candidate |
| T3 | T1 on second picks | all three graphs | candidate |
| T4 | form b: T1 plus cross-aspect negatives | loss form | candidate |
| T5 | form c: one embedding, one learned mask per aspect | loss form | candidate |
| D1 | trunk, distillation only, no graph or caption terms | graph terms removed | control for the graphs |
| D2 to D4 | T1 plus direct distillation, weight 0.5, 1, 2 | distillation term | candidate |
| D5 | T1 plus relational distillation, weight 1 | distillation term | candidate |
| I1 | the best default-input row, instructed inputs | input (three passes) | second result |

Definitions (D14 to D16): form b adds, for each anchor in head h, its buddies under the other two aspects' graphs
that are not pairs of h as extra negatives. Form c follows conditional similarity networks (Veit et al., 2017): a
non-negative mask per aspect multiplies one shared output, with an L1 penalty on the masks. Direct distillation is
InfoNCE from the image head output to the instructed image embedding of its aspect, other paintings in the batch as
negatives, temperature 0.05; relational distillation is the KL between the batch similarity rows of the two.
Distillation acts on the image side only. If T4 or T5 beats T1 by more than T1's seed sd in at least two aspects,
the best D row is rerun on that form (D17).

### 3.3 Data roles

Inherits base §3.3: 13,400 fit paintings train, 1,489 dev paintings stop early, the fixed 5,000 evaluation
paintings score every row. New: the instructed images of fit and dev paintings are distillation targets (training
only). The evaluation paintings' instructed images only score the target bar. ArtELingo held test: not touched.

### 3.4 Measurement

Base §3.4 scoring: AMI at k = 20, three k-means seeds, image and caption side, all three labels (genre on its 4,053
labelled paintings). New (D7, D18): V, ARI and FMI on the same partitions; every table carries the four bars of the
brief and the stage 1 network; rows rank on the image-side AMI of each head's own label. Seed 0 for all rows; seeds
1 and 2 for A1, T1 and the top two rows per aspect (D11). Tie band: T1's population sd over its three seeds; form a
stays on a tie (base D5). Back-fill: V, ARI and FMI for each row of the two-bar table of the K = 40 user-read report,
from stored outputs and stage 0's regenerated heads.

### 3.5 Stages

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check (CPU, 8 threads) | new trainer set to separate heads; stored outputs | genre R0o seed 0 output equals the stored array; emotion Fres1 and style Fres05, seeds 0 to 2, both-side AMI equal to the recorded values (6 decimals); the scorer's AMI of the stored PercepT core outputs equal to the recorded values | stop, trace, note it for the user; the run waits |
| 1 | exploration | A1, T1 to T5, D1 to D5, seed 0 | none | — |
| 2 | exploration | seeds 1 and 2 (§3.4); the D17 rerun | none | — |
| 3 | exploration | I1 | none | — |

Stage 0 runs at the end of the build, so a failure is traced before the run chat opens.

### 3.6 Diagnostics

Cosine of each image head output to its instructed image embedding (evaluation paintings); each head's AMI on the
other two labels; mask overlap of form c; best epoch per head. They explain the sheet and decide nothing.

## 4. Decisions

Inherited from the base spec: D3 (one trunk, one head per aspect), D4 (side per aspect), D5 (loss forms a, b, c;
a stays on a tie).

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Spec form | new spec, base = aspect graph spec | target and inputs changed (an amendment would rewrite most of it) | agent default | 2026-10-19 20:10 |
| D2 | Next step and type | stage 2, exploration, medium loop | after the ceiling check (decision A) | user, after seeing the recommendation | 2026-10-19 |
| D3 | Target | instructed Qwen image clustered directly, one pass on the default image; one or two loops | the reviewer's first attack is "just prompt Qwen" | user | 2026-10-19 |
| D4 | Gate timing | 10-29: emotion at least +0.01 over PercepT core, clearly above the other bars; the instructed bar judged with retrieval on 11-10 | gate kept apart from the harder target | user, after seeing the recommendation | 2026-10-19 |
| D5 | Inputs | default input carries the claim; instructed input a second result | one pass is the point | user | 2026-10-19 |
| D6 | Pick side | image-side AMI; retrieval's pick waits for stage 3 | stage 1 picked on both sides | user | 2026-10-19 |
| D7 | Extra measures | V, ARI, FMI beside AMI; H and C dropped; SI, CHI, DBI left out | "without DEC I don't think we gain a lot on these" | user, after seeing the recommendation | 2026-10-19 |
| D8 | Which measure decides | AMI | one measure for every pick | agent default, shown to the user | 2026-10-19 |
| D9 | Minor m1 to m6 | four bars incl. the probe; V, ARI, FMI; distillation and image-side picks as new rows; no CLIP inputs; one seed, three for the top rows; build after the restart | grill | user, after seeing the recommendation | 2026-10-19 |
| D10 | Trunk form | first layer shared, second per aspect | matched to stage 1's head, so A1 differs only in sharing | agent default | 2026-10-19 20:10 |
| D11 | Seeds | §3.4 | tie rule needs T1 on three seeds; caps runs near 30 | agent default | 2026-10-19 20:10 |
| D12 | Batching and stopping | one batch per aspect per step; each head keeps its own best dev epoch; stop when every head passes patience 15 | heads peak at different epochs (stage 1: 1 to 4) | agent default | 2026-10-19 20:10 |
| D13 | Graph rows | top two by image-side AMI from the follow-up sheet (3 seeds), plus stage 1's picks | D6 | agent default | 2026-10-19 20:10 |
| D14 | Form b | other aspects' buddies as extra negatives | the base spec named the form, not the negatives | agent default | 2026-10-19 20:10 |
| D15 | Form c | learned masks, L1 penalty at the paper's weight | fixed disjoint masks would fix the split in advance | agent default | 2026-10-19 20:10 |
| D16 | Distillation | direct InfoNCE (weights 0.5, 1, 2) and relational (1); through the trunk; image side only; one row alone | a separate branch would be three models (cosine regression rejected: the batch loss is the head's own form) | agent default | 2026-10-19 20:10 |
| D17 | Distillation on form b or c | rerun the best D row if b or c wins (§3.2) | keeps form a as default unless beaten | agent default | 2026-10-19 20:10 |
| D18 | Device | sheet on the GPU under the lock; stage 0 on CPU | stage 1 ran on CPU; A1 is retrained on the GPU so rows share a device | agent default | 2026-10-19 20:10 |

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: stage 1's head training is bitwise repeatable on CPU at 8 threads. Checked: not checked (stage 0 tests it).
- Assumption: instructed images exist for all fit and dev paintings under the three instructions. Checked: the
  loader validates the 14,889 training images per instruction; not run in this chat.
- No open questions.

## 6. Success criteria

**Descriptive mode.** No decision rides on these numbers: every stage is an exploration on development data, never
quoted as a result. Reading: the outcomes (a) to (d) of the brief. Also: if no row keeps image-side emotion at 0.087
or above (PercepT core + 0.01), the 10-29 emotion gate is at risk already on development data, and we say so first.

## 7. Constitution check

| Principle | Pass or exception | Note |
|---|---|---|
| C1 strongest comparator | pass | bars 1 and 2, the target, the stage 1 network, A1 |
| C6 sample IDs | pass | frozen sample; instructed images joined by extraction IDs |
| C7, C8 rule and prior | not applicable | no go bar; disclosure in §8 |
| C12 regression check | pass | stage 0 |
| C13 re-derivation | not applicable | no verdict |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| One seed picks a lucky row | top rows within T1's seed sd | three seeds for the top rows; the simplest tied row goes on |
| Distillation learns artists seen in training (painting-grouped split) | not measured here | the held split shares this; say so in the report |
| GPU runs do not repeat bitwise | seed sd | the tie band comes from three seeds |

Selection disclosure: the 5,000 evaluation paintings were read by the premise check, three pilots, stage 1 (63
graphs, 59 head runs), PercepT core and the ceiling check; this sheet adds about 30 runs. Development numbers are
optimistic; the claim test on held data protects.

## 9. Deliverables and review

- At the end: a round folder `src/test/2026102X_model_sheet/` (new module prefix); an auto report in
  `docs/reports/auto/percept/` and a user-read report; V, ARI and FMI filled into the K = 40 two-bar table.
- Reviews: the controller's diff read and one smoke run per ticket (no verdict code); the final whole-branch review.
- Stop points: none between approval and the reports, except a failed stage 0.
- Budget: GPU about 1 h under the lock; CPU about 2 h at 8 threads (stage 0, scoring). Time-box 2026-10-23, so the
  claim test spec fits before 10-29.

Body over the exploration target (about 160 lines): the arms, the regression check and the carried grill decisions
each need their own table.
