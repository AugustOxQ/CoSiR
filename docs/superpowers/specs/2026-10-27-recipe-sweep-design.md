# Model G's held emotion test, a staged recipe sweep, an ablation and an analysis: design

> Date: 2026-10-27 03:42 · Status: approved 2026-10-27 03:48 (from the brief, r10 decide 27) · Kind: experiment
> Builds on: [report 12-27](../../reports/auto/percept/2026-12-27_relational_genre.md), its
> [spec](2026-10-26-relational-genre-design.md) and [rule](../../../src/test/20261027_relational_genre/DECISION_RULE.md);
> the grill in the r10 decide 26 chat (its record: the [decide 26 handoff](../handoffs/2026-10-27-r10-decide-27-handoff.md)
> §2 and the [direction log](../direction_log.md) entry of 2026-10-27) · Constitution: [constitution.md](../constitution.md)
> Decision rule: `src/test/20261027_recipe_sweep/DECISION_RULE.md` @ <first commit, after the review>

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** Model G (the recipe plus the relational genre term with a sharper teacher row) wins on emotion and is
level on style and genre against the instructed Qwen image on held paintings (report 12-27). This loop tests G's held
emotion claim, sweeps G's settings on development paintings for the strongest recipe on all three aspects, and adds an
ablation and an analysis.

**Plan.**
1. Stage 0 (claim test, held, local): G's emotion against PercepT core at both training scales, 5 seeds.
2. Stage 1 (sweep, development, seed 0, DAS6): 13 settings of G, each changed alone to 3 values (39 runs, and G).
3. Stage 2 (sweep, seed 0, DAS6): a joint grid over the 4 settings with the largest gain: G's value and their 2 best
   values (at most 72 new runs).
4. Stage 3 (confirmation with a pick, seeds 1 to 3, local): the top 3 runs of stages 1 and 2 against G.
5. Stage 4 (claim test, held, local): the pick at full scale, 5 seeds; skipped when G stays.
6. Beside them (local): an analysis of G now and of the final model later; an ablation of the final model.

**Compared against.** Stage 0: the strongest PercepT core entry (0.0874 at full scale, 0.0855 at the sample scale).
Stages 1 to 3: G on the same machine and seeds. Stage 4: the instructed Qwen image of each aspect (held: emotion
0.0921, style 0.4026, genre 0.5062).

**Outcomes.** The mean gain is the gain over G in seed ranges, averaged over the three aspects (seed ranges: emotion
0.0046, style 0.0082, genre 0.0128).
- Stage 0: GO when G gains at least 0.0100 with the interval above zero at full scale, and the sample scale is not NO-GO.
- Stages 1 and 2: a run counts when no aspect sits more than half a seed range below G; runs are ranked by mean gain.
- Stage 3: a run qualifies when each aspect is level with G (the interval's lower end above minus one seed range) and
  the mean gain's interval is above zero. The highest mean gain wins; if none qualifies, G stays.
- Stage 4, per aspect: a win when the interval is above zero; level when its lower end is above minus one seed range.
- We expect (guesses): stage 0 GO 0.85; a pick that replaces G 0.5; given a pick, held wins on emotion 0.9, style 0.4,
  genre 0.15.

**Cost.** DAS6 node402: about 3.5 hours on 3 GPUs. Local: about 2.5 GPU hours and 8 CPU hours. Light work runs by
day; the full-scale ablation (about 80 GPU minutes, then its CPU scoring) runs from 19:00 to about 23:00. If the node
fails, stages 1 and 2 add about 6 local hours. Reports by 2026-10-28 03:00.

**Check these.**
- Taken after our recommendation: the sweep before the encoders; "level" with the teacher as an accepted end state;
  stages 1 and 2 on the node, all else local, node numbers never compared with local; the per-stage form of your
  "each aspect at least level with G" rule.
- Agent defaults: the stage types (0 and 4 claim tests, 1 and 2 sweeps, 3 a confirmation); the 39 stage 1 values (40
  runs, below the "about 45"), graph neighbours kept; the grid rule; one joint bootstrap for the mean gain.
- Agent default, an exception to the regression rule (C12): the node's agreement with local is recorded, not a gate.
  The node must repeat its own G run within half a seed range; if not, or if the route fails, stages 1 and 2 run
  locally.
- Agent default: the full-scale ablation ends about 23:00 (its CPU scoring), later than the 21:30 you saw.
- Most likely wrong: about 110 runs on one seed inflate the top gains. On fresh seeds the mean gain's interval may
  reach zero, so G stays (about 0.5); with three seeds its lower end sits near the worst seed's gain.

## 1. Context and goal

Asked (decision C, r10 decide 26, 2026-10-27): "I also think so, so we now have two levels and one win (emotion), and
hopefully we can get all three as win (higher-than-noise)." Decisions A and B and the grill (same chat, 03:00 to
03:30) set the loop: the held emotion test with G, then a staged sweep, with an ablation and an analysis beside it.

Report 12-27 picked model G (code name GS-0.02: the recipe R6-5 plus the relational term on the genre head at weight 8,
the teacher's row at temperature 0.02) on development paintings and tested its genre on held paintings: 0.5001 against
the instructed Qwen genre image's 0.5062, −0.0060 [−0.0105, −0.0014], level within one seed range. G's held readings
against the instructed Qwen image of the aspect: emotion +0.0118 [+0.0083, +0.0150], style +0.0032 [−0.0019,
+0.0085]. G's emotion margin over PercepT core, the emotion claim's bar, was never computed. Most of G's settings were
never varied (facts brief §A): learning rate, weight decay, batch, trunk width, dropout, graph neighbours, the buddy
temperature, patience, and the relational term on emotion and style.

Goal: carry one model for the paper's three aspect claims, as strong as a sweep on development data can make it.
Questions this step answers: does G hold the emotion claim on held paintings; is there a recipe that beats G on
development data on all three aspects together; does it win against the instructed Qwen image on held paintings; and
which parts of the final model carry its result.

## 2. Scope

In: stage 0 (G's held emotion claim, both scales); stages 1 to 3 (the sweep on development data); stage 4 (one held
test of the pick); the ablation (§3.2, on the final model: development and held); the analysis (§3.6, on G and on the
final model); one auto report with its user-read report.
Out: the wide random search over all knobs (a later follow-up with spare node time, development paintings only); the
other encoders and aspect-named retrieval (later loops, m7); k-means K (stays 20, m4); the paper text.

## 3. Design

### 3.1 Hypothesis and prior

G was built one change at a time on top of settings fixed in the first model sheet (12-16). The knobs never varied may
hold small gains, and two never-tried terms (the relational term on emotion and on style) may lift those aspects the
way the genre term lifted genre (+0.0356 on fresh seeds). If the gains of single knobs add up, a joint grid over the
best few gives a recipe above G on all three aspects. We expect single-knob gains under one seed range per aspect, and
the style relational term as the likeliest large one. Prior (rule §1): stage 0 GO 0.85; a pick that replaces G 0.5;
given a pick, held wins on emotion 0.9, style 0.4, genre 0.15. A surprise: a stage 3 pick with a mean gain above one
seed range, or a held genre win.

### 3.2 Arms and controls

**Stage 0.** G at the sample scale (seeds 0 to 3 from 12-27, seed 4 new) and at full scale (12-27's five seeds)
against 12-24's PercepT entries B1 to B6 at the same scale (the RB rule, part 2; G takes K4's place).

**Stages 1 and 2.** Every run is G with the knobs named changed and all else fixed (K = 20 k-means readout, inputs,
teachers, AdamW, 200 epochs at most, output width 2048). The comparator is G's own run at seed 0 on the machine of stages 1 and 2.

The number sheet (stage 1; each knob changed alone; 39 runs and G's reference):

| ID | Knob | G | Stage 1 values |
|----|------|---|----------------|
| k1 | relational term on the emotion head: weight (12-27's form: in-batch, the teacher row at k4's temperature, the student row at 0.05, beside the direct term; teacher: the instructed Qwen emotion image) | off | 2, 8, 16 |
| k2 | relational term on the style head: weight (the same form; the instructed style image) | off | 2, 8, 16 |
| k3 | the genre relational weight | 8 | 4, 12, 16 |
| k4 | the relational teacher row's temperature (every relational term in the run) | 0.02 | 0.015, 0.025, 0.035 |
| k5 | the direct teacher weight, all three heads | 0.25 | 0.0625, 0.125, 0.5 |
| k6 | learning rate | 1e-3 | 3e-4, 5e-4, 2e-3 |
| k7 | weight decay (AdamW) | 0.05 | 0, 0.01, 0.2 |
| k8 | batch, in graph pairs (rows are twice the pairs) | 512 | 256, 1024, 2048 |
| k9 | shared trunk width | 1024 | 512, 1536, 2048 |
| k10 | dropout | 0.1 | 0, 0.2, 0.3 |
| k11 | graph neighbours: mutual top-k in each aspect graph, rebuilt with the same code | 20 | 10, 30, 40 |
| k12 | the buddy terms' temperature (the direct and relational temperatures stay) | 0.05 | 0.03, 0.07, 0.1 |
| k13 | early stopping patience, epochs | 15 | 5, 10, 30 |

Stage 2's grid: the (up to) 4 knobs with the largest positive knob gain, each at G's value and its (up to) 2 best
counting values with a positive mean gain; at most 3⁴ = 81 points, of which up to 9 ran in stage 1 (rule §4.2).

**Stage 3.** The top 3 configurations of stages 1 and 2 at seeds 1 to 3 against G's 12-27 runs at the same seeds.

**Stage 4.** The pick at full scale, seeds 0 to 4, against T, the instructed Qwen image of each aspect on held
paintings (12-24's frozen partitions); against B1 to B6 on emotion; against G on held (reported).

**Ablation.** On the final model (the pick, or G when G stays); each arm removes one part, nothing is retuned:

| Removal | What changes |
|---------|--------------|
| relational term | weight 0 on every head that has one |
| sharper teacher row | every teacher row at 0.05, the student's temperature |
| direct teacher | weight 0 on all three heads |
| buddy terms | the four buddy terms off |
| caption endpoints | only the image–image buddy term (D2's image-only endpoints) |
| content correction | emotion and style graphs built without it (genre's has none) |
| instructed graphs | one default-view graph for all three aspects |
| shared trunk | separate heads, one trunk per aspect |

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---------------|------|---------------|
| the frozen sample: 13,400 fit and 1,489 stopping paintings | training; label-free early stopping | 0 (sample scale), 1, 2, 3, ablation (development) |
| 29,232 development paintings P (genre labels on 23,682) | development: every gain, guard, rank and pick | 1, 2, 3, ablation (development) |
| model seed 0 | development: stages 1 and 2 | 1, 2 |
| model seeds 1 to 3 | development: the confirmation and the pick; the development ablation | 3, ablation |
| 42,969 train and 6,152 validation paintings | training and stopping at full scale | 0, 4, ablation (held) |
| 12,281 held paintings H (genre 9,949) | held: images for inference, captions for labels only | 0, 4, ablation (held), analysis |
| ArtELingo-28 | not touched | — |

P, seed 0 and seeds 1 to 3 chose 12-24's recipe, 12-25's weight and 12-27's G; they stay development data. No held
number enters any choice (`held-data.md`): the pick is written to the stage 3 verdict file before its first full-scale
run. G itself became the recipe after its held genre reading; the report discloses it.

### 3.4 Measurement

- **Metric.** AMI of k-means partitions (20 clusters, three random states, mean) against each aspect's labels; image side
  only. Leiden is reported beside at stage 4.
- **Gains.** On P, gain_a = AMI_a(run) − AMI_a(G) over aspects emotion, style, genre, in units of the seed range r_a;
  the mean gain M = the mean of gain_a / r_a. Stages 1 and 2 use point values at seed 0; stage 3 three-seed means with
  one paired bootstrap per configuration (2,000 draws over paintings and seeds, the same draws for all three aspects
  and M; rule §3.3).
- **Stage 0.** The RB rule's part 2 statistic, at each scale (rule §3.1).
- **Stage 4.** Per aspect, d_T = AMI(pick) − AMI(T) on H, 5 seeds, a paired bootstrap over paintings and seeds (as
  12-27's stage C); the emotion margin over B1 to B6 at full scale (the RB rule).
- **Ablation.** Each removal minus the final model, per aspect: development seeds 1 to 3 on P, and held at full scale
  with 5 seeds; paired bootstraps over paintings and seeds (rule §3.5).

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|-------|------|-------|--------------|-------------|
| R0 | regression | the stage 0 scorer on 12-24's K4 outputs; G's frozen files | K4's verdicts at both scales within 1e-6; G's hashes | stop stage 0; trace |
| R1 | regression | this round's trainer, every knob at G's value, seeds 0 and 1, locally | 12-27's G exports bit for bit | no new training run (stage 0's seed 4, stages 1 to 4, the ablation); trace |
| R3 | regression | the stage 3 scorer on 12-27's stage B files | 12-27's stage B values within 1e-6 | stop stage 3; trace |
| R4 | regression | the full-scale trainer with G's knobs, seed 0 | G's full-scale weights and held outputs bit for bit | no held output of a pick; trace |
| N | node route | node check, smoke job, G's reference run twice | runs complete, exports valid, the two G runs within half a seed range; agreement with local recorded | stages 1 and 2 both run locally |
| 0 | claim test, held | G, both scales | the emotion bar at each scale | reported; the sweep goes on |
| 1 | sweep, seed 0 | 39 runs and G (node) | knob gains; the grid | no knob gains: G stays |
| 2 | sweep, seed 0 | the grid (node) | the top 3 | — (after a grid the pool is never empty) |
| 3 | confirmation with a pick, seeds 1 to 3 | the top 3 and G (local) | a qualifying run | G stays; stage 4 skipped |
| 4 | claim test, held | the pick at full scale | per aspect against T; emotion against PercepT | — |
| AB | ablation | the final model | effects with intervals | — |

R0, R1, R3 and R4 run in the build. Order in the run: stage 1 on the node and stage 0 locally at the same time, G's
analysis after stage 0's verdict; then stage 2; then stage 3, stage 4 and the final model's analysis; the development ablation by day; the
held ablation from 19:00. Re-derivations by separate code (C13) run before each stage's outcome is stated.

### 3.6 Diagnostics: the analysis (decides nothing)

Measured on held paintings at full scale, 5 seeds (galleries on seed 0); on G after stage 0's verdict and again on the
final model when it is not G; for comparison, on T and on the frozen default Qwen image where the item applies.

- **Heads against label sets:** the AMI of each head's partition against each aspect's labels (3 by 3).
- **Per class:** per label class, the F1 of its best-matching cluster, the model minus T; the cluster composition
  (label shares per cluster) with a gallery of the 16 paintings nearest each cluster centre.
- **Agreement between heads:** the AMI between the three heads' partitions of the same paintings.
- **Geometry:** effective rank, hubness (skew of the 10-nearest-neighbour counts), alignment (image and caption
  outputs of the same painting) and uniformity of each head's outputs.
- **Neighbours against the teacher:** the overlap of each painting's 20 nearest neighbours under the head and under
  T, per class; galleries of the paintings with the lowest overlap.

Also reported, deciding nothing: each stage's best epochs, the node's agreement with local (scorer and trainer), the
stage 1 knob curves, and stage 4's margins against G and against PercepT on style and genre.

### 3.7 Frozen vs rerun

Rule §5. Frozen: G's 12-27 runs (sample seeds 0 to 3, full seeds 0 to 4 with held outputs), 12-24's K4 verdicts, the
PercepT entries, T's partitions, P, H and their hashes. Rerun: G's sample seed 4 and held inference of its sample
seeds; G's reference run on the node; every sweep, pick and ablation run; all scoring.

## 4. Decisions

From the r10 decide 26 chat, 2026-10-27 (the direction log entry of that date), and this chat.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Direction (A) | one medium loop, type sweep: stage 0 first, the sweep with the ablation and analysis beside it; encoders and retrieval later | the encoders first (the user's first order) | user; the order after seeing the recommendation | decide 26 |
| D2 | The aim | the strongest recipe for all three aspects; "level" with the teacher is an accepted end state | a win on all three as the only aim | user, after seeing the recommendation | decide 26 |
| D3 | Idea (B) | the staged sweep: stages 1 to 4; the wide random search later, not here | a narrow sweep of the loss knobs alone | user | decide 26 |
| D4 | Where (Q1) | stages 1 and 2 on node402 with G's reference run; stages 0, 3, 4, the ablation and the analysis local; no comparison across machines | everything local | user (DAS6); the split after seeing the recommendation (flag) | decide 26 |
| D5 | The pick (Q2) | rank by mean gain in seed ranges (option d); each aspect at least level with G: half a seed range below G at most at seed 0; at stage 3, level as 12-27 and the mean gain's interval above zero; else G stays | genre alone; a sum of raw gains | user (the rank and the rule); its per-stage form after seeing the recommendation (flag) | decide 26 |
| D6 | A held win (Q3) | the interval of the pick minus T above zero; level as 12-27; each win's size in seed ranges; the paper says "above the teacher (interval above zero)" | a point win | user | decide 26 |
| D7 | Stage 0 (m1) | the earlier emotion claim test: both scales, 5 seeds, +0.0100 over PercepT core with the interval above zero; G's full-scale files reused after checks; sample seed 4 and held inference of the sample seeds new; G's adoption after its held reading disclosed | full scale only | user, after seeing the recommendation | decide 26 |
| D8 | Stage 1 knobs (m2) | the 13 knobs of §3.2 | a smaller list | user, after seeing the recommendation | decide 26 |
| D9 | Stage 1 values | 3 values per knob (§3.2); 39 runs and G | 2 values for most knobs | agent default | 03:42 |
| D10 | The grid | the up to 4 knobs with the largest positive knob gain; each at G's value and its up to 2 best counting values with M > 0; reused stage 1 points | 3 knobs at 4 levels | agent default | 03:42 |
| D11 | Seeds (m3) | stages 1 and 2 on seed 0; the top 3 on seeds 1 to 3; the pick at full scale, 5 seeds, held | more seeds in the sweep | user, after seeing the recommendation | decide 26 |
| D12 | K (m4) | k-means K stays 20 | K as a knob | user, after seeing the recommendation | decide 26 |
| D13 | The ablation (m5) | on the final model; development with seeds 1 to 3 and held at full scale with 5 seeds; the eight removals of §3.2 | on G only | user, after seeing the recommendation | decide 26 |
| D14 | The analysis (m6) | §3.6, on G now and on the pick | none | user, after seeing the recommendation | decide 26 |
| D15 | Later loops (m7) | the encoders, then retrieval | in this loop | user, after seeing the recommendation | decide 26 |
| D16 | A plan page | none ("plain") | a page | user | decide 26 |
| D17 | Stage types | 0 and 4 claim tests; 1 and 2 sweeps; 3 a confirmation with a pick; the ablation; the analysis descriptive | stage 3 as a sweep | agent default | 03:42 |
| D18 | The mean gain's interval | one bootstrap per configuration, the same draws for the three aspects and M, paintings drawn over all of P with unlabelled rows masked | per-aspect bootstraps (no joint interval for M) | agent default | 03:42 |
| D19 | The node route | inputs in one small folder under a new mapping in the shared cluster config, checked by the smoke job; scoring on the node by both scorers; G's node run repeated at once, within half a seed range; the node's agreement with local recorded, not a gate (a C12 exception); if the route fails before stage 2 ends, stages 1 and 2 both run locally; the kept node runs copied home before 2026-10-30 | a gate on agreement with local (the 10-20 gap of up to 0.0168 would likely close the route) | agent default (flag) | 03:42 |
| D20 | Stage 0's sentence | GO at full scale and the sample scale not NO-GO, as 12-24's flow | full scale alone | agent default | 03:42 |
| D21 | Graph neighbours | stays in stage 1: a rebuild takes about 30 s at the sample scale and about 200 s at full scale | dropped for cost | agent default | 03:42 |
| D22 | Timing | the development ablation by day (about 17 GPU minutes); the held ablation from 19:00 (about 80 GPU minutes, then its CPU scoring to about 23:00; the weekday rule) | all of it from 19:00 | agent default | 03:42 |
| D23 | Names and places | round folder `src/test/20261027_recipe_sweep/` (prefix `rs_`), report 12-28; "mean gain" in plain words, no glossary entry | a new term | agent default | 03:42 |

**Amendments** (changes after approval; "none" until there is one)

| ID | When | Change | Reason | Decided by | Before or after results | Places updated |
|----|------|--------|--------|------------|-------------------------|----------------|
| A1 | 2026-10-27 03:59 | the review's fixes (C12 exception, the k11 and k8 notes, the sample held inference check, E0, the node's repeat check and deadline, the ablation's graph check, tie-breaks, wording) | the rule review, run before approval and finished after it; none changes a user decision or a bar (user's instruction at approval) | agent default | before | brief (Cost, Check these), §3.2, §3.4, §3.5, §3.6, §4 D10 D19 D22, §6, §7, §9; rule |

## 5. Assumptions and open questions

- Assumption: this round's trainer wraps 12-27's unchanged, so R1 and R4 pass bit for bit. Checked: R1, R4.
- Assumption: the node's runs are valid within the node, even if they differ from local (10-20: up to 0.0168). Not
  checked; the parity is recorded, and no number crosses machines.
- Assumption: node402's 3 GPUs are free by 10:30 (another project used 2 of them from 03:15 for 2 to 3 hours). Not
  checked; the run chat waits for free GPUs (`shared-resources.md`).
- Assumption: the node's local disk holds about 60 GB of exports and weights. Checked: the node check.
- Open questions: none; the grill settled the key and minor questions (§4).

## 6. Success criteria

**Decision mode.** Binding rule: `src/test/20261027_recipe_sweep/DECISION_RULE.md` at the commit in the header,
committed before any code; it governs where it differs from this spec. Its criteria:
- Stage 0: G-ES-GO, -MM, -IN, -NO (sample scale), G-EF-GO, -MM, -IN, -NO (full scale), the sentence E0; validity V0.
- Stages 1 and 2: the guard, knob gains, GRID or SWEEP-NONE, TOP or TOP-NONE; validity V1, the node route N.
- Stage 3: qualifies, PICK or KEEP-G; validity V3.
- Stage 4: per aspect T-WIN, T-LEVEL, T-BELOW; the emotion claim PE-GO, -MM, -IN, -NO; validity V4.

**Selection rules.** Stages 1 and 2 (rule §4.2): guarded configurations ranked by M; ties to fewer changed knobs,
then the number sheet's order. Stage 3 (rule §4.3): the qualifier with the largest three-seed M; ties to TOP's
order.

**Licensed claims.**
- Stage 0, E0: "On 12,281 held ArtELingo paintings, model G gives emotion clusters (k-means, 20 clusters) above the
  strongest PercepT core entry by <d> [interval] at full scale and <d> [interval] at the sample scale, 5 seeds each,
  paired 95% intervals over paintings and seeds." With the disclosure that G became the recipe after its held genre
  reading.
- Stages 1 to 3: "On 29,232 development ArtELingo paintings (Qwen; picked on one seed, confirmed and picked on three
  fresh seeds), <pick> gives a mean gain of <M> seed ranges over G, each aspect level with G." At most "promising on
  development data".
- Stage 4, per aspect, T-WIN: "On <n> held ArtELingo paintings, <model> gives <aspect> clusters above the instructed
  Qwen <aspect> image (interval above zero): <d> [interval], <d / r> seed ranges, 5 seeds." T-LEVEL: "level with the
  instructed Qwen <aspect> image within one seed range". T-BELOW: no claim. The emotion claim over PercepT as E0, at
  full scale only.
- Ablation: "On <data>, removing <part> from <model> changes <aspect> AMI by <Δ> [interval]", as strong as the result
  it explains. The analysis: descriptive, no claim.
- Not covered: other encoders, other datasets, other readouts, retrieval.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest real comparator | pass | stage 0: PercepT core's strongest entry; stages 1 to 3: G, the current best; stage 4: T, the strongest reading per aspect |
| C6 sample IDs | pass | exports keep P's and H's orders; the format checks hash them |
| C7 rule before code | pass | the rule reviewed by a fresh Opus `xhigh` worker and committed before any code |
| C8 prior and disclosure | pass | rule §1; §8 below |
| C12 regression first | exception (rule §6) | R0, R1, R3, R4 and the ablation's graph check locally; on the node the parity with local is recorded, not a gate |
| C13 independent re-derivation | pass | separate code for every deciding number, on the node for its partitions and locally for every aggregate |

Exceptions: C12 on the node (rule §6): the node's trainer and scorer need not reproduce 12-27's local numbers · stages
1 and 2 compare node runs only with node runs (D4) · a parity gate was rejected: the 10-20 gap (up to 0.0168) would
likely close the route.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| The node gives numbers unlike local | the parity readings | none cross machines; the report gives the size |
| The node route fails (env, data, GPUs taken) | the node check, the smoke job, the queue | stages 1 and 2 run locally, under the weekday rule if the runs draw over 200 W |
| Single-knob gains do not add up in the grid | stage 2's ranks against stage 1's | the top 3 may be single knobs; reported |
| A knob's best value sits at the end of its range | the knob curves | reported; the later random search can extend it |
| The seed-0 pick is noise | stage 3 | G stays when nothing qualifies |
| A removal arm fails to train | its loss | reported as failed, no effect |

Selection disclosure. P and seed 0 chose 12-24's recipe among six, 12-25's weight among seven and 12-27's seed-0
winner among eight arms; this loop adds about 110 configurations on the same paintings and seed, so seed-0 gains are
inflated by the pick. Seeds 1 to 3 protect, but they also picked in 12-25 and 12-27, and here they pick among up to
three. The held paintings were read by the claim tests, 12-24 and 12-27; T's and G's held scores are known, and G's
held emotion AMI (0.1039) puts stage 0's outcome close to known. G became the recipe after its held genre reading.
No choice is made on held data here: the pick is written down before its first full-scale run, and only it is tested.

## 9. Deliverables and review

- At the end: round folder `src/test/20261027_recipe_sweep/` with the rule, a verdict file per stage, the knob gains,
  the grid and the pick, the re-derivation files (outputs on `/data/SSD2/cosir_r10/20261027_recipe_sweep/`, gitignored;
  node exports stay on the node until the wrap-up, the kept node runs copied home before 2026-10-30); one auto report (12-28) in `docs/reports/auto/percept/` with a
  results brief, and a user-read report; a row in `docs/superpowers/held_ledger.md` per held read.
- Reviews: the rule and this spec by a fresh Opus `xhigh` worker before the commit; the build per `matt-chain.md`, with
  ticket reviews and unit tests for verdict code (the new switches, the stage 0 adapter, the knob gains and grid, the
  stage 3 statistic and pick, the stage 4 scorer, the re-derivations); the final whole-branch review; Claude's claim
  review of the auto report.
- Stop points: none between approval and the reports, except a failed gate.
- Budget: node402, about 110 runs of about 5 minutes on 3 A6000 (about 3.5 hours, with scoring on the node's CPU);
  local 3090 under the lock, about 2.5 GPU hours (stage 0: 1 run; stage 3: up to 9; stage 4: 5; ablation: 24
  development and 40 full-scale runs; the gates), and about 8 CPU hours at 8 threads (k-means and bootstraps). Weekday
  rule: only the held ablation is a long run, from 19:00; light work by day. Time-box: reports by 2026-10-28 03:00.
- Length: the body is about 270 lines against about 210 for two claim tests plus stages, because of the number sheet
  and the 23 decisions carried from the grill.
