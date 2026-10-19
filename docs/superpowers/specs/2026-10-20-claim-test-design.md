# Claim test of D2 against the strongest PercepT on held paintings, in two stages: design

> Date: 2026-10-20 01:35, revised 01:43 after the rule review · Status: draft; built under the user's overnight
> instruction of 2026-10-20 01:27 ("Change the mode to overnight run now, so you can automate and hand me the
> results in the morning"; D29); approval from the brief is pending · Kind: experiment
> Builds on: the [stage 2 sheet spec](2026-10-19-stage2-model-sheet-design.md) (D2) and its
> [report](../../reports/auto/percept/2026-12-16_model_sheet.md); the [PercepT core spec](2026-10-18-percept-core-design.md)
> (arms, A1 to A3); the grill in the [direction log entry of 2026-10-20](../direction_log.md) ("Grill, claim test");
> the facts brief `.scratch/r10-claim-test/facts.md` (not in git); the
> [development Leiden reading](../../../src/test/20261020_leiden_dev_reading/20261020_leiden_dev_reading_log.md) ·
> Constitution: [constitution.md](../constitution.md)
> Decision rule: `src/test/20261020_claim_test/DECISION_RULE.md` @ 7d82e91 (first commit; the rule text is unchanged since, except dates)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** The 10-29 gate asks for held-out emotion clusters above PercepT. D2 (one shared trunk, three
heads, a pull toward the instructed image) leads PercepT core by +0.019 on development data. This claim test
measures that lead on the 12,281 held paintings, which this line never read.

**Plan.**
0. Regression check: D2 and PercepT core retrain with weights saved and must reproduce their saved development
   outputs. If not, the run stops before any held read and the morning report gives the cause.
1. Stage 1 (the 10-29 gate): D2, PercepT core and two "paper settings" PercepT versions retrain on the developed
   sample (13,400 paintings), 5 seeds each, and are read on the held paintings.
2. Stage 2: the same on the full train side (42,969 paintings), in its own build and run, only after a GO or an
   inconclusive stage 1. Its outcome replaces stage 1's.

**Compared against.** The strongest PercepT that did not collapse: core, or a paper settings version ("strict":
the paper's topic cut; "K = 40": our cut). Each is read two ways (k-means at 20 clusters, or its own top topic); the
stronger counts. Today's development bar: about 0.092 (core's own reading 0.082 + 0.01); D2 has 0.096.

**Outcomes.** d = D2's emotion AMI minus that comparator's, means over 5 seeds; stage 2 uses the same bar.
- GO: d at least 0.01 and the paired 95% interval above 0. Stage 2 runs.
- Inconclusive, margin met (d at least 0.01, interval reaches 0), or inconclusive (0 < d < 0.01): stage 2 runs.
- NO-GO: d at 0 or below. Back to step 2.
After stage 2: GO gives the paper's sentence; margin met: improve the margin on development data, test again;
inconclusive: we talk; NO-GO: back to step 2. Leiden is reported beside. We expect d near +0.005 to +0.015.

**Cost.** Build now, unattended; the stage 1 run follows: about 1.5 to 3 GPU h locally, 1 to 1.5 CPU h; node408
takes PercepT training if the local run would end after 11:00. You get the stage 1 reports this morning.

**Check these.**
- You did not read this brief before the build, and the held read happens before you do. A change in the morning
  to a default that touches the verdict becomes an amendment "after results".
- Agent default: the interval resamples model seeds as well as paintings, and the comparator maximum is taken again
  in each resample. Both make the test stricter.
- Agent default: a paper settings version counts when at least 1 of its 5 seeds did not collapse (mean over those).
- Agent default: the paper leaves DEC details open: the target is recomputed each epoch; training stops when under
  0.1% of pairs change topic, or at 200 epochs. Only S1-GO passes the 10-29 gate.
- Please confirm: after stage 2, its outcome alone sets the next step (your S1), so "margin met" in stage 1 does not
  by itself start a margin round (your S6).

## 1. Context and goal

Asked (the grill, r10 decide 13; the direction log's record): "develop on same samples now, but be careful of the
result claim", then the full ArtELingo train side as a second claim test (Q1, Q4). The direction log's bullet "Grill,
claim test" holds every answer; §4 carries them.

The stage 2 sheet picked D2 on the 5,000 development evaluation paintings: image-side AMI 0.0962 / 0.3697 / 0.4540
(emotion, style, genre; 3 seeds) against PercepT core 0.0772 / 0.2286 / 0.4359. No weights were saved for D2 or
PercepT core (facts §3, §5), so every arm retrains.

Goal: the paper claim on emotion clusters from the default image, judged on held paintings.
Question this step answers: does D2's held emotion AMI exceed the strongest uncollapsed PercepT by at least 0.01,
with a paired interval above 0, first on the developed sample and then on the full train side?

## 2. Scope

In: stage 0, stage 1 and stage 2 (§3.5); emotion decides; style and genre margins as diagnostics; the instructed Qwen
image of the held paintings for the target row; the two paper settings versions (arms P1, P4).
Out: ArtELingo-28 (its own facts brief and spec after the gate); retrieval and the instructed Qwen bar (11-10);
TURTLE and TEMI (track 2 or 11-10); cluster-level losses (track 2); any change of D2's recipe.

## 3. Design

### 3.1 Hypothesis and prior

Claim: a one-pass model trained on buddy graphs with a pull toward the instructed image gives emotion clusters on
unseen paintings that PercepT does not reach on the same features. Mechanism: the emotion graph and the distillation
carry emotion structure into the default image head. Prior: d on held near +0.005 to +0.015. Reasons: the
development lead (+0.019 under k-means, +0.014 against core's own reading) is optimistic after many variants on the
same 5,000 paintings (§8); the paper settings versions and the max over readings can only raise the bar. A d above
0.025 or below −0.005 would surprise us. The rule states the prior per outcome (C8).

### 3.2 Arms and controls

| Arm | What it is | Differs from PercepT core by | Role |
|-----|------------|------------------------------|------|
| D2 | stage 2 sheet row D2 (sheet §3.2): trunk + direct distillation 0.5, graphs Gres / Fres05 / R0o; read from the default Qwen image | our model | candidate |
| PercepT core | core spec with A3 (K = 40); arm per aspect fixed from development data: P3 emotion, P2 style, P3 genre (m5) | — | comparator |
| Paper strict, P1 and P4 | "PercepT, paper settings, strict" (`GLOSSARY.md`): paper training settings, Algorithm 1 cut (γ = 2, start at 100 topics) | training settings and topic cut | comparator when not collapsed |
| Paper K = 40, P1 and P4 | "PercepT, paper settings, K = 40": as strict, with core's A3 cut | training settings | comparator when not collapsed |
| Target | instructed Qwen image clustered directly | reference (three passes) | diagnostic |
| Encoder | default Qwen image clustered directly | reference | diagnostic |

Strict and K = 40 share each Stage 1 training run and differ only in the cut; each cut trains its own Stage 2. P1:
CLIP + RoBERTa Stage 1, CLIP patch Stage 2. P4: Qwen Stage 1 (emotion-instructed caption as E), CLIP patch Stage 2.
Budgets: no arm is tuned on held data. D2's recipe and core's arms were chosen on development data; the paper
settings come from the paper, so they get no tuning. Unstated paper details are D21.

### 3.3 Data roles

| Data | Role | Stage |
|------|------|-------|
| 13,400 fit paintings (67,327 pairs) of the frozen sample | training | 0, 1 |
| 1,489 dev paintings | D2's early stopping (label-free loss) | 0, 1 |
| 5,000 development evaluation paintings | development: stage 0 only; training data in stage 2 | 0, 2 |
| Other 23,080 train paintings | not touched in stage 1; training in stage 2 | 2 |
| Validation split, 6,152 paintings | not touched in stage 1; D2's early stopping in stage 2 | 2 |
| Held split, 12,281 paintings, 61,744 captions | held: images for inference, captions for labels only | 1, 2 |

Held instructed images serve only the target row (m4). No held caption enters any model.

### 3.4 Measurement

- **Metric:** image-side AMI of each head on its own label; emotion decides. k-means as the sheet: k = 20,
  MiniBatchKMeans seeds 0 to 2, n_init 3, batch 4,096, fit on all 12,281 held paintings, labels < 0 masked; mean
  over k-means seeds, then over model seeds 0 to 4 (m1). Five seeds: the sheet's seed sd (about 0.002 for D2)
  gives a seed-mean error near 0.001, small against the 0.01 margin.
- **PercepT's second reading:** its own assignment, the argmax of its K Stage 2 probabilities (as `native_img`).
  The k-means reading L2-normalizes the K probabilities first (as `pc_score`).
- **Labels (m2):** per held painting, emotion = mode of its captions' emotions ("something else" included, ties to
  the smallest code); style and genre from the first caption; genre missing on 2,332 paintings, masked.
- **Painting order (C6):** one list of the 12,281 held paintings, sorted by name, saved with its SHA-256. Every arm's
  output rows follow it; joins use extraction IDs and painting names, never row positions of another store.
- **Normalization:** model outputs are saved in float16, cast to float32 and checked for unit norm (tolerance 1e-3),
  with no re-normalization (as the sheet; the probe's re-normalization moved one value by 0.0037).
- **Statistic:** d = mean AMI of D2 minus the maximum over the uncollapsed PercepT entries (version, arm, reading);
  the interval is a paired bootstrap over held paintings and model seeds (rule §3).
- **Leiden (3a), beside the verdict:** for each vector set, a union 20-NN graph by cosine (undirected, unweighted),
  `ModularityVertexPartition` at resolution 1.0, Leiden seeds 0 to 2, as the development probe; d under Leiden with
  the same bootstrap. It decides nothing (Q7).

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|-------|------|-------|--------------|-------------|
| 0 | regression check | D2 and core P1 to P4, seed 0, through the new code with the paper switches off, weights saved; outputs on the 5,000 evaluation paintings; on every device that trains a stage 1 arm | output hash equals the saved one; if CUDA does not repeat bitwise, image-side AMI within 1e-4 of the recorded value on all three aspects, with the cause named. Also: the held scorer, given the saved 5,000-row outputs, returns the sheet's recorded AMI to 6 decimals | stop before any held read; trace the cause; report it in the morning |
| 1 | claim test | all arms of §3.2, seeds 0 to 4, held inference | rule, stage 1 | rule outcomes |
| 2a | regression check | stage 2 code set to the sample sizes, seed 0, on every device that trains a stage 2 arm (node408 included) | reproduces stage 1's seed 0 outputs (hash, or AMI within 1e-4 with the cause named) | stop; trace; report |
| 2 | claim test | all arms retrained on the full train side, seeds 0 to 4 | rule, stage 2 | rule outcomes |

The paper settings code runs core's path when its switches are off, so stage 0 covers its shared parts, the patch
Stage 2 of P1 and P4 included. Its new parts (the Algorithm 1 cut, the collapse rule, the DEC refresh) are verdict
code and get unit tests.

### 3.6 Diagnostics (computed only after the verdict is written)

V, ARI and FMI with a random baseline at the same count; k-means at Leiden's count (S3); Leiden 3b (communities on
the fit paintings' outputs, mutual 20-NN with repair, held paintings join by a 20-NN vote); an in-sample reading on
the fit paintings (PercepT's paper scores its training set); style and genre margins with their intervals; the
target and encoder rows; per paper settings run: topic count, largest topic share, collapse flag, Stage 2 dev AUC.

### 3.7 Frozen vs rerun

- **Frozen:** D2's recipe and graphs on the sample (sheet; `AGS/graphs`, checked by manifest hash); core's recipe,
  K = 40 and arms per aspect; the paper settings and D21; the scorer protocol and label rule; the held painting list.
- **Rerun in stage 1:** all training (seeds 0 to 4, weights saved), held inference, held instructed images, scoring.
- **Stage 2:** the same recipes, unchanged (D27); rerun: the missing features, graphs on 42,969 nodes, all training.

## 4. Decisions

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Stages | two claim tests: the developed sample for the 10-29 gate (careful claim wording), then full data | gate date; a sample claim is weaker (one stage) | user (Q1) | 2026-10-20 01:20 |
| D2 | Paper settings in the bar | enters only when not collapsed; also runs at full scale (its collapse may come from the sample size) | core alone | user (Q2) | 2026-10-20 01:20 |
| D3 | Leiden protocol | 3a beside the verdict; 3b as a diagnostic | September's vote alone | user (Q3) | 2026-10-20 01:20 |
| D4 | Spec scope | this spec = stage 1 + stage 2 (full train side, English captions); ArtELingo-28 after the gate | ArtELingo-28 in this spec | user (Q4) | 2026-10-20 01:20 |
| D5 | Paper settings versions | strict (Algorithm 1, γ = 2) and K = 40; arms P1, P4; collapse thresholds of `GLOSSARY.md` | one version | user (5c); arms and thresholds: user, after seeing the recommendation | 2026-10-20 01:20 |
| D6 | GO bar | d ≥ 0.01 and the paired 95% interval of d above 0 | margin alone | user (6b) | 2026-10-20 01:20 |
| D7 | Leiden's role | reported beside, decides nothing | Leiden as a second gate | user (7a; given after the development Leiden reading) | 2026-10-20 01:20 |
| D8 | When stage 2 runs | only after a GO or an inconclusive stage 1; after a NO-GO, back to step 2 | stage 2 regardless | user (8b) | 2026-10-20 01:20 |
| D9 | Stage 2's weight | its result counts over stage 1's; an uncollapsed paper settings PercepT that beats D2 there becomes the comparator to improve against | stage 1 final | user (S1) | 2026-10-20 01:20 |
| D10 | Inconclusive, margin met | d ≥ 0.01, interval reaches 0: improve the margin, choosing the change on development data, and test again; stage 2 still runs (D12) | treat as NO-GO | user (S6; "half-go", no new name) | 2026-10-20 01:20 |
| D11 | Bars by stage; translation timing; K = 40 raising the bar | the bar can differ by stage; a long ArtELingo-28 translation moves past 11-10; a K = 40 PercepT may raise the 10-29 bar above D2's development score | — | user (S2, S4, S5) | 2026-10-20 01:20 |
| D12 | After an inconclusive stage 1 | stage 2 runs and its GO is the paper's sentence | stop at stage 1 | user (S8) | 2026-10-20 01:20 |
| D13 | k-means at Leiden's count | a diagnostic that decides nothing | in the verdict | agent default (user deferred S3) | 2026-10-20 01:20 |
| D14 | Minors m1 to m10 | §3.4 (m1, m2), §3.5 (m3), §3.2 and §3.3 (m4, m5, m8), §3.6 (m6), §2 (m7), `GLOSSARY.md` (m9), §9 (m10) | — | user, after seeing the recommendation | 2026-10-20 01:20 |
| D15 | Plan page | none | — | user ("yes to all"; default) | 2026-10-20 01:20 |
| D16 | Stage types | 0 and 2a regression checks; 1 and 2 claim tests | claim test is the first type that fits | agent default | 2026-10-20 01:35 |
| D17 | Bootstrap | 2,000 resamples of the 12,281 held paintings and of each arm's seeds, seed 0; percentile interval (rule §3) | paintings only (too narrow: ignores training noise, rule review); 1,000 resamples (coarser tails) | agent default | 2026-10-20 01:43 |
| D18 | Comparator inside the bootstrap | the maximum over entries is taken again in each resample | a comparator fixed at the point estimate (ignores the selection on held) | agent default | 2026-10-20 01:35 |
| D19 | A partly collapsed entry | counts when ≥ 1 of 5 seeds is uncollapsed; score = mean over those seeds; collapse is judged per Stage 1 run and cut | ≥ 3 of 5 (could drop a real PercepT result) | agent default | 2026-10-20 01:35 |
| D20 | Collapse rule for core | none; core is the established comparator and always counts | the glossary rule targets the paper versions | agent default | 2026-10-20 01:35 |
| D21 | Unstated paper details | DEC target P refreshed on all fit pairs at each epoch start; stop when < 0.1% of fit pairs change hard topic between epochs (DEC's tolerance), or 200 epochs; cosine per epoch over each phase's maximum; Algorithm 1 starts at 100 topics as core | per-minibatch targets (unstable), core's full-batch rule | agent default | 2026-10-20 01:35 |
| D22 | Held painting order and normalization | §3.4 | re-normalize (differs from the sheet) | agent default | 2026-10-20 01:35 |
| D23 | Device | local GPU under the lock; node408 (3 GPUs, reserved) takes PercepT training when the local estimate ends after 11:00, or for stage 2 | all local (misses 11:00 in stage 2) | agent default; node408 offered by the user | 2026-10-20 01:35 |
| D24 | Re-derivation (C13) | d, its interval and each entry's mean re-derived by separate code from the outputs and the rule only | — | agent default | 2026-10-20 01:35 |
| D25 | Stage 2 data | train = all 42,969 train paintings; D2 stops early on the validation split (6,152); held unchanged | keep the 1,489 dev paintings for stopping (they are train paintings) | user (m8); split details agent default | 2026-10-20 01:35 |
| D26 | Stage 2 features | extract instructed Qwen images (3 instructions) for the train and validation paintings not covered, emotion-instructed captions for their captions, RoBERTa for P1's missing captions; in the shared cache | rebuild all stores | agent default | 2026-10-20 01:35 |
| D27 | Stage 2 recipes | frozen: only the data grows (same epochs, patience, batch sizes, K, γ) | retune at scale (would be selection on new data) | agent default | 2026-10-20 01:35 |
| D28 | Chat order | stage 1 builds and runs tonight; stage 2 builds after stage 1's verdict, in its own build and run chats | one build for both (delays the morning result) | agent default | 2026-10-20 01:35 |
| D29 | Approval | build and run go on under the user's overnight instruction; approval from the brief is pending; a later change to a verdict default is an amendment after results | wait for the morning (loses the night) | user's instruction; going on before approval: agent default | 2026-10-20 01:27 |
| D30 | Candidate | row D2 | other sheet rows | user, after seeing the recommendation (r10 decide 12) | 2026-10-20 |
| D31 | PercepT readings | own assignment (top topic) and k-means k = 20; the stronger reading counts | one reading | user, after seeing the recommendation (r10 decide 12) | 2026-10-20 |
| D32 | Leiden | a Leiden reading in all major tests beside k-means | k-means alone | user (r10 decide 12) | 2026-10-20 |
| D33 | Paper settings PercepT | runs beside core, with a collapse rule ("it can collapse easily, but I tend to give a try") | core alone | user, after seeing the recommendation (r10 decide 12) | 2026-10-20 |
| D34 | 10-29 gate | only S1-GO passes; S1-MM and S1-IN leave it open until stage 2 (rule §4) | — | agent default | 2026-10-20 01:43 |
| D35 | After stage 2 | its outcome alone sets the next step; S2-MM starts a margin round (rule §4) | stage 1's margin met starts one too (conflicts with S1) | agent default, from the user's S1 and S6; to confirm | 2026-10-20 01:43 |

Grill rows (D1 to D15) carry 01:20, when the user confirmed the closing summary.

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: the held default Qwen, CLIP global and CLIP patch rows are finite and correct. Checked: coverage by
  IDs and headers (facts §2); values not checked.
- Assumption: this line never used the validation split. Checked: a code search of the new line's round folders
  found no use; the conditioned line's archive was not searched (its models do not enter here).
- Assumption: the paper's settings are as facts §6 reads them. Checked: by a Codex comparison, spot-checked.
- Assumption: the saved `AGS/graphs` are the graphs D2 trained on. Checked: the sheet's loader checks the manifest.
- No open questions.

## 6. Success criteria

**Decision mode.** Binding rule: `src/test/20261020_claim_test/DECISION_RULE.md` at the commit in the header,
committed before any code; it governs where it differs from this spec. Its criteria:
- S1-GO, S1-MM (inconclusive, margin met), S1-IN (inconclusive), S1-NO (NO-GO): stage 1 outcomes from d and its
  interval.
- S2-GO, S2-MM, S2-IN, S2-NO: the same at full scale, against the strongest uncollapsed PercepT at full scale.
- V1 to V4 and R-COL: validity (stage 0 or 2a passed; the re-derivation agrees; output, label and order checks; no
  seed change after held outputs exist) and the collapse rule.

**Licensed claim.** Stage 1 GO: "On 12,281 held ArtELingo paintings, a model trained on 13,400 paintings and read
from the default Qwen image gives emotion clusters (k-means, 20 clusters) with AMI at least 0.01 above the strongest
uncollapsed PercepT reimplementation trained on the same paintings, each with its own recipe on our feature stores
(point estimate over 5 seeds; paired 95% interval of the difference, over paintings and seeds, above 0)." Stage 2 GO: the same with "trained on the 42,969 train paintings". Neither covers style or genre, other
datasets, other backbones, the instructed Qwen bar, retrieval, or PercepT with its own encoders.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | the maximum over three PercepT versions, two arms and two readings, taken on held data |
| C6 sample IDs | pass | one held painting list with its hash; joins by extraction IDs |
| C7 rule before code | pass | rule reviewed by a fresh opus `xhigh` reviewer, committed before any code |
| C8 prior and disclosure | pass | prior in rule §1; disclosure in §8 |
| C12 regression check | pass | stages 0 and 2a |
| C13 re-derivation | pass | D24 |
| `held-data.md` | pass | no tuning, selection or early stopping reads held data; the comparator's maximum on held works against D2 |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| Paper settings collapse in every seed | collapse table | they leave the bar; the report says so; stage 2 retries at full scale (D2) |
| CUDA does not repeat bitwise | stage 0 hash differs | AMI within 1e-4 with the cause named; else stop |
| Held labels differ in kind from development (more paintings with few captions) | label counts per class | report them; the verdict stands |
| Artists shared between train and held (painting-grouped split) | not measured | say so in the report |
| The seed resample has only 5 seeds per arm (coarse) | seed spread per arm | report each arm's seed sd beside d |

Selection disclosure: the 5,000 development evaluation paintings chose D2 among 13 sheet rows, after the premise
checks, three pilots, stage 1 (63 graphs, 59 head runs), PercepT core (5 arms, 2 runs) and the ceiling check. D2's
development lead is optimistic. The held paintings were never read by this line; older lines read them (held
ledger H1 to H3, H5 to H7). Protection: held data, 5 fresh seeds (3 and 4 are new), a comparator maximum taken on
held.

## 9. Deliverables and review

- At the end: round folder `src/test/20261020_claim_test/` (module prefix `ct_`), with the rule, saved weights,
  held outputs, `verdict_stage1.json` and the re-derivation; an auto report in `docs/reports/auto/percept/` with a
  results brief and a user-read report; one plain row per held read in `docs/superpowers/held_ledger.md` (no
  script checks the ledger or refuses to run).
- Reviews: the rule by a fresh opus `xhigh` reviewer before commit; ticket reviews and unit tests for verdict code
  (scorer, bootstrap, collapse rule, Algorithm 1 cut, held labels); the final whole-branch review.
- Stop points: none between approval and the reports, except a failed stage 0 or 2a.
- Budget, stage 1: GPU about 1.5 to 3 h (held instructed images about 0.5 h; training about 1 to 2 h), CPU about 1
  to 1.5 h (scoring, bootstrap, Leiden) at 8 threads. Build from about 02:00; run from about 07:00; the end
  estimate is checked at the start against 11:00 (D23). Stage 2: extraction about 1 h, graphs about 0.5 h CPU,
  training about 3 to 6 GPU h, mostly on node408. Time-box: stage 1 report 2026-10-21; stage 2 before 2026-10-29.

Body over the claim test target (about 225 lines against about 180 for two claim stages): the 35 carried and new
decisions need their rows.
