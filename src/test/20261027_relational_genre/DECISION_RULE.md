# Decision rule: the relational genre term's sweep on seed 0 (stage A), the pick on fresh seeds (stage B) and the held genre claim test of the selected variant (stage C)

> Spec: [2026-10-26-relational-genre-design.md](../../../docs/superpowers/specs/2026-10-26-relational-genre-design.md)
> @ 2e9529a (first commit) · Written 2026-10-26 19:35 (Amsterdam) before any implementation script exists; revised
> 19:49 after the review · Reviewer: a fresh Opus `xhigh` worker, 2026-10-26 19:55 (pass after fixes: four
> must-fix, eleven should-fix and ten nit findings, all applied)

The trimmed constitution sets no statistics, development bar or failure reading, so this rule states or inherits them.
**Stages A and B** inherit 12-25's rule, `src/test/20261025_encoders_genre/DECISION_RULE.md` (below "the EG rule"),
stage iii text: §2.3 (the base R6-5, the GR form, "Every run", the references), §3.3 (the statistics on P, T_g, the
fresh-seed statistic) and §4.3 (VG1 to VG4, failed runs, the guard, the weights), which in turn inherit 12-24's rule
§3.1 (`src/test/20261025_recipe_backbones/DECISION_RULE.md`, "the RB rule"). The changes are written here: the arms,
the ladder's climb, the seed-0 winner, the pick on fresh seeds. **Stage C** inherits the RB rule's part 2 (§2 "Part
2", §3.2, §4.2), which inherits the claim test's rule §3 (`src/test/20261020_claim_test/DECISION_RULE.md`, "the CT
rule"); the changes: the aspect is genre, the deciding comparator is the instructed Qwen genre image, a second level
against the current recipe. C7, C8, C12 and C13 apply as written. Where this rule and the spec differ, this rule
governs. Paths below are relative to `src/test/20261025_encoders_genre/` ("EG/"), `src/test/20261025_recipe_backbones/`
("RB/") or `src/test/20261020_claim_test/` ("CT/") unless they start with `/` or `src/`.

## 1. Prior (C8)

Written before any new arm is trained and before any new number on P or on held paintings exists. Known: 12-25's
stage iii values on P (seed 0: R6-5 0.4674, GR-8 0.5018, T_g 0.4989; fresh seeds 1 to 3: GR-8 0.5022 against R6-5's
0.4694); 12-24's held values on 9,949 labelled paintings (K4, the recipe at full scale, 0.4534; T 0.5062; F4 0.4521;
the strongest PercepT entry 0.4269), with K4's margin under T, −0.0527 [−0.0588, −0.0467].

**Stage A** (genre AMI on P, seed 0; the guard on emotion at −0.0046 and style at −0.0082). Point guesses: GR-16
0.497 to 0.507, emotion effect −0.003 to −0.008 (the guard at risk); GR-32 0.493 to 0.510, guard failure 0.5; the
climb past 32 runs: 0.3; GN-8 0.490 to 0.505 (in CLIP-KD the direct term alone beat the relational term alone); GS-0.02
0.497 to 0.508; GS-0.01 0.493 to 0.510 (a sharper target can help or make training unstable); GQ-8 0.495 to 0.508
(4,096 anchors against up to 2,047 in-batch rows). A new arm beats GR-8 on seed 0 and is guarded: 0.6. No published
sweep shows a gain that rises steadily with the relational weight (literature note 12-26, Q2).

**Stage B.** GR-2 and GR-4 within one seed range (0.0128) of GR-8's three-seed mean: 0.7. A new seed-0 winner, when
there is one, is confirmed on fresh seeds: 0.7. B-NONE: 0.02. The held variant is not GR-8: 0.4.

**Stage C** (held, 9,949 labelled paintings). The selected variant's genre AMI 0.480 to 0.500; d_T −0.026 to −0.006.
G-GO 0.15, G-MM 0.15, G-PART 0.50, G-NO 0.20. A held lift the size of the fresh-seed lift (+0.033) would leave the
student about 0.020 under T: G-PART.

Disclosure: spec §8.

## 2. Candidates, arms and comparators

### 2.1 Stage A (seed 0, P)

**Base:** R6-5 as the EG rule §2.3 (its seed-0 P exports `RB/outputs/part1/train/R6-5`, frozen). **The relational
term** as the EG rule §2.3's GR form: per row of the batch (the 2,048 endpoints of 1,024 pairs; a painting can appear
in several rows), the KL divergence from the teacher's row to the student's, each row a softmax over the row's
similarities to the other rows of the batch (rows of the same painting excluded), temperature 0.05 on both sides, the
teacher rows from the instructed Qwen genre image, the student rows from the genre head's image output; mean over the
rows; weight λ. Every arm changes only the genre head's loss
and keeps R6-5's settings (the EG rule §2.3, "Every run": early stopping per aspect on that aspect's loss with the
new terms included, exports of the image head outputs on P and on the old 5,000).

| ID | Change to the genre loss (the direct term at weight 0.25 unless said) |
|---|---|
| GR-8 | λ = 8, in-batch, beside the direct term: 12-25's run, frozen (`EG/outputs/stage_iii/train/GR-8/s0`) |
| GR-16, GR-32 | λ = 16, 32; run first |
| GR-64, GR-128, GR-256 | the climb (§4.1): at most three more rungs, each double the last |
| GN-8 | λ = 8, in-batch; the direct term removed (weight 0) on the genre head only |
| GS-0.02, GS-0.01 | λ = 8, in-batch; the teacher's row at temperature 0.02 or 0.01, the student's row at 0.05; the direct term unchanged (0.05) |
| GQ-8 | λ = 8; teacher anchors in place of the in-batch rows (below); the direct term stays |

**Teacher anchors (GQ-8).** The anchors loss takes two anchor sets: A_s for the student rows and A_t for the teacher
rows. In GQ-8 both are A: the teacher rows (the instructed genre image, as stored, unit norm) of 4,096 fit paintings,
drawn without replacement by a `torch.Generator` seeded with the model seed after the model is built, so the model's
random stream is unchanged (as the EG rule's GA adaptor); fixed for the run. Per row i of the batch: the student row
is softmax(z_i A_sᵀ / 0.05), the teacher row softmax(t_i A_tᵀ / 0.05); anchors of row i's own painting are excluded
from both rows; the loss is the KL divergence from the teacher row to the student row, mean over the rows. **EQ
(equivalence):** with A_s = the batch's own student rows and A_t = its own teacher rows, and the same exclusion, the
loss equals `ms_train.distill_loss`'s relational loss of the same batch within 1e-6, on three real fit batches (the
trainer's batch draw at seeds 0, 1, 2).

**References**, without training: T_g, the instructed Qwen genre image on P (the EG rule §3.3, 0.4989161752444698),
and the ceiling probe's ridge prediction (0.504009), reported only.

### 2.2 Stage B (fresh seeds 1 to 3, P)

| ID | Runs | Source |
|---|---|---|
| GR-2, GR-4 | seeds 1, 2, 3, new | trained as their seed-0 runs (the EG rule §3.3, "Fresh seeds") |
| GR-8 | seeds 1, 2, 3 | 12-25's runs, frozen (`EG/outputs/stage_iii/train/GR-8/s{1,2,3}`) |
| v*_A, when it is not GR-8 | seeds 1, 2, 3, new | trained as its seed-0 run |
| R6-5 | seeds 1, 2, 3 | 12-25's runs, frozen (`EG/outputs/stage_iii/train/R6-5/s{1,2,3}`); the comparator |

### 2.3 Stage C (held)

**Candidate G** = v_held (§4.2) trained at full scale as the RB rule's part 2 stage 4 (the 42,969 train paintings,
the CT's 42,969-node graphs, early stopping on the 6,152 validation paintings, the instructed teachers of those
paintings; GQ-8's anchors drawn from the 42,969 train paintings' teacher rows), seeds 0 to 4; held outputs of the
three heads on H (float16, 2048 wide) from each head's best state, as the RB rule.

| ID | What | Role |
|---|---|---|
| T | the instructed Qwen genre image on H: the CT's held store, its three r20 partitions (12-24's T) | the deciding comparator |
| R | the current recipe at full scale: 12-24's saved held genre outputs of K4, seeds 0 to 4 (`RB/outputs/part2/full/held/K4/genre__s{0..4}.npz`) | the partial level's comparator |
| B1 to B6, F4 | the RB rule's PercepT entries and the frozen default Qwen image | reported only |

## 3. The statistics

### 3.1 Stages A and B

The EG rule §3.3 as written: P, L_a, the partitions (MiniBatchKMeans, k = 20, states 0, 1, 2), A_a(v, I), e_a(v) =
A_a(v) − A_a(R6-5) at seed 0, the per-aspect bootstrap of the RB rule §3.1 (2,000 draws, the same I_b for every arm
of an aspect), r = 0.0046 / 0.0082 / 0.0128, T_g unrounded, the promising level T_g − r_genre. Stage B: for each arm v
of §2.2, Ā_a(v) over seeds 1 to 3, ē_a(v) = Ā_a(v) − Ā_a(R6-5), the paired interval of the EG rule §3.3 ("Fresh
seeds"); each arm's bootstrap uses its own `numpy.random.default_rng(0)` per aspect, so every arm gets the same draws
(I_b, S_b). Decisions use unrounded values; reports give 4 decimals.

### 3.2 Stage C

The CT rule §3 and the RB rule §3.2 (H in sorted order, its SHA-256
`a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876`; partitions r20 on the float32 cast of the saved
float16 outputs, unit norm checked; AMI with default arguments), with these changes:
- **Labels:** genre, as the RB rule's reported statistic (the first caption's label; labels < 0 masked; 9,949
  labelled paintings, L_g). The AMI is computed on the labelled rows; the held bootstrap draws I_b over all 12,281
  rows and masks the unlabelled rows inside the AMI, as 12-24's margin "T" did.
- **A_G(I, S):** the mean over the seeds in S of the mean over G's three r20 partitions. **A_T(I):** the mean over
  T's three frozen partitions (`RB/outputs/part2/partitions/full/T/genre__s0__r20.npz`, 12-24's, computed from the
  CT's held store of the instructed genre image; no seeds). **A_R(I, S_R):** as A_G from R's saved outputs with K4's
  frozen partitions (`RB/outputs/part2/partitions/full/K4/genre__s{0..4}__r20.npz`), seeds 0 to 4. VC2 recomputes
  both sets of partitions from the store and the outputs.
- **d_T(I, S) = A_G(I, S) − A_T(I).** **d_R(I, S, S_R) = A_G(I, S) − A_R(I, S_R).** Point estimates at I₀ = 0, 1, …,
  12,280 and the full seed lists.
- **Intervals:** one bootstrap per statistic, as the RB rule's reported margins. For d_T: `rng = default_rng(0)`; per
  draw b, I_b = `rng.integers(0, 12281, size=12281)`, then S_b = `rng.choice([0, 1, 2, 3, 4], size=5, replace=True)`
  for G; T draws nothing. For d_R: its own `default_rng(0)`; I_b, then S_b for G, then S_R,b for R, in that order.
  2,000 draws; the 95% interval is `numpy.percentile(·, [2.5, 97.5])` (default linear method).
- **The margin X = r_genre = 0.0128.** **The half gap G_half = (A_T(I₀) − A_R(I₀, {0, …, 4})) / 2**, recomputed from
  the frozen files; its value from 12-24's `RB/outputs/part2/reported_stage4.json` (block K4, genre, margin "T":
  A_T = `A_e."T-r20"` 0.5061513009972589, A_R = `A_cand` 0.4534172670570083) is 0.0263670169701253, and the scorer's
  value must match within 1e-6.
- **The scorer check:** on 12-24's saved K4 outputs with K4 in G's place, the scorer gives `reported_stage4.json`'s
  genre margin "T" (d −0.05273403394025061, interval [−0.05876959743462086, −0.04667470103045529]), its margin
  "a_B1_to_B6" values, and its margin "b_matched" (K4 against B1, with B1 in R's place: d 0.03041045992232999,
  interval [0.020721646966819336, 0.03962273041085061], which exercises d_R's draw order), each within 1e-6.

## 4. Rules

### 4.1 Stage A

Validity first; no outcome is stated until VA1 to VA4 hold.
- **VA1** (C12) This round's trainer with every switch off (λ = 0, the direct term at 0.25, temperature 0.05, no
  anchors) trains R6-5 at seed 0 and its P exports equal `RB/outputs/part1/train/R6-5`'s seed-0 exports bit for bit;
  with the GR-8 switches (λ = 8, in-batch, 0.05, the direct term at 0.25) its P exports equal
  `EG/outputs/stage_iii/train/GR-8/s0`'s bit for bit; at seed 1 with every switch off, its P exports equal
  `EG/outputs/stage_iii/train/R6-5/s1`'s bit for bit; its scorer gives 12-25's `EG/outputs/stage_iii/score_all.json`
  values of R6-5 and GR-8 (`results`: each A, e and bound) and T_g within 1e-6. Otherwise nothing runs in stages A, B
  or C; the cause goes in the report.
- **EQ** (§2.1) passes before GQ-8 trains. If it fails, GQ-8 still runs and is reported with its guard, but it is
  excluded from the seed-0 winner pool, and the cause goes in the report.
- **VA2** (C13) Separate code, given this rule, P's names, the captions and the saved exports (never our numbers),
  recomputes the labels, the partitions, each A, e and interval, the guard, the climb, v*_A, the stage B means,
  effects, intervals, confirmations and v_held, and the outcomes. Equal partitions; values within 1e-6; the same
  climb, v*_A, v_held and outcomes. Before each climb decision, the separate code recomputes the newest rung's A_genre
  and guard; the decision uses agreeing values, and a disagreement is settled before the next rung runs. A
  disagreement is traced and fixed before an outcome is stated, settled by this rule's text; both values and the fix
  go in the report.
- **VA3** Every export on P has 29,232 finite rows in P's order with unit norm; P's SHA-256 equals 12-25's
  (`EG/outputs/stage_iii/reported.json`, `p_sha256`).
- **VA4** (seed policy) After any new arm's export on P exists, no arm is added, dropped or retrained beyond the climb,
  stage B and the loss-scale run (§5) as this rule defines them, except a rerun forced by a failed VA3 (same seed and
  code), which the report lists.
- **Failed runs.** A run that stops on a non-finite loss, or still fails VA3 after its rerun, has no A_genre: it is
  not guarded, cannot be v*_A, and counts as run for the climb (it ends the climb when it is the newest rung). The
  report gives the cause.

Conditions:
- **Guard:** e_emotion(v) ≥ −0.0046 and e_style(v) ≥ −0.0082 (point estimates against R6-5 at seed 0). GR-8's
  values are 12-25's, re-derived under VA2.
- **The climb.** GR-16 and GR-32 run first. After them: while the newest rung run is guarded and its unrounded
  A_genre is strictly above that of every other guarded ladder rung run (GR-8 included), the next rung, double the
  newest, runs. The climb stops when that condition fails (the newest rung unguarded, failed, tied or lower) or after
  GR-256. So GR-64 runs if and only if GR-32 is guarded and above every guarded rung among GR-8 and GR-16; GR-128 if
  and only if GR-64 is guarded and above every other guarded rung run; GR-256 likewise. A rung that still rises at
  256 is reported; nothing more runs.
- **Seed-0 winner v*_A:** among GR-8 and every new arm run (GQ-8 only when EQ passed), the guarded one with the
  largest unrounded A_genre; on equal values, GR-8 first, then the order GR-16, GR-32, GR-64, GR-128, GR-256, GN-8,
  GS-0.02, GS-0.01, GQ-8. GR-8 is guarded (12-25's `score_all.json`: e_emotion −0.0032, e_style −0.0052), so a winner
  always exists.

Outcomes, seed 0:
- **A-NEW** v*_A is not GR-8: v*_A trains on seeds 1 to 3 (§2.2). **A-SAME** v*_A is GR-8: stage B runs GR-2 and
  GR-4 only.
- Reported, deciding nothing: each arm's level against T_g (at target when A_genre ≥ T_g; promising when
  T_g − r_genre ≤ A_genre < T_g; neither otherwise), its three aspects with intervals, the old 5,000, Leiden, best
  epochs, the loss values of §3.6 of the spec.

### 4.2 Stage B

Validity first; no outcome is stated until VB1 to VB4 hold.
- **VB1** (C12) The frozen fresh-seed exports of GR-8 and R6-5 are 12-25's files: their SHA-256 equal
  `EG/outputs/stage_iii/score_fresh.json`'s `identity.img_sha256` per arm, seed and aspect; and this round's
  fresh-seed scorer, on those files, gives that file's `effects` (Ā, ē and both bounds per aspect) within 1e-6.
- **VB2** = VA2. **VB3** = VA3 per export. **VB4** No run beyond GR-2, GR-4 and v*_A on seeds 1 to 3.
- A failed VB1 or VB2 stops stages B and C; the cause is traced and goes in the report.
- **Failed runs.** A fresh-seed run that stops on a non-finite loss, or still fails VB3 after its rerun (same seed
  and code, listed), leaves its arm not confirmed; the report gives the cause.

Per arm v in {GR-2, GR-4, GR-8, v*_A when A-NEW}: Ā_a(v), ē_a(v), intervals (§3.1).
- **Confirmed** (the EG rule's C-HOLDS): ē_genre(v) ≥ r_genre with its lower bound > 0, and ē_emotion(v) ≥ −r_emotion
  and ē_style(v) ≥ −r_style.
- **v_held:** among the confirmed arms, the one with the largest unrounded Ā_genre; on equal values, GR-8 first, then
  GR-2, GR-4, then v*_A.
- **B-PICK** a v_held exists: it is the held candidate G, recorded in the stage B verdict file before any full-scale
  run of a candidate. **B-NONE** no arm is confirmed: no held test; every arm is reported; the next decide chat
  decides.
- Reported: each arm's C level (C-HOLDS, C-QUAL, C-FAILS as the EG rule) and Ā_genre against T_g; when v_held's
  Ā_genre is within r_genre of another confirmed arm's, the report says the arms are within one seed range and the
  pick follows this rule (user, K2b).

### 4.3 Stage C

Validity first; no outcome is stated until VC1 to VC4 hold.
- **VC1** (C12) This round's full-scale trainer with every switch off trains R6-5 at full scale at seed 0, and its
  weights equal `RB/outputs/part2/full/weights/K4/{emotion,style,genre}__s0.pt` bit for bit; the held outputs of the
  three heads computed from those weights equal `RB/outputs/part2/full/held/K4/{emotion,style,genre}__s0.npz` bit
  for bit. The held scorer, on 12-24's saved K4 outputs, passes the scorer check of §3.2. Otherwise no held output of
  G is computed; the cause goes in the report.
- **VC2** (C13) Separate code, given this rule, the CT and RB rules, the held list, the captions, T's and R's saved
  outputs and G's saved outputs (never our numbers), recomputes the labels, the partitions, A_G, A_T, A_R, d_T, d_R,
  both intervals, G_half and the outcome. Equal partitions; values within 1e-6; the same outcome. A disagreement is
  traced and fixed before an outcome is stated.
- **VC3** Each of G's output sets (three aspects, five seeds) has 12,281 finite rows of unit norm in H's order; H's
  SHA-256 equals the CT's.
- **VC4** (seed policy) G's five seeds run once; a seed that fails VC3 is rerun once (same seed and code), listed;
  no other rerun, no second candidate, no change to G after its first held output exists. G's held outputs are
  computed only after this rule is committed and v_held is written to the stage B verdict file. A seed that stops on
  a non-finite loss, or still fails VC3 after its rerun, leaves no stage C outcome; the cause goes in the report.

Outcomes (X = 0.0128; PC, the partial condition: d_R ≥ G_half and the lower bound of d_R's interval > 0):
- **G-GO** the lower bound of d_T's interval > −X. When that lower bound is also > 0, the licensed sentence says
  "matches or exceeds".
- **G-MM** (margin met, inconclusive) d_T ≥ −X and the lower bound ≤ −X. Reported with PC: "margin met, inconclusive;
  the partial condition holds" or "does not hold". The partial sentence applies when PC holds.
- **G-PART** (partial) not G-GO, not G-MM, and PC holds.
- **G-NO** otherwise. No genre claim; the held row is a reading.

What the outcomes lead to: nothing automatically; every outcome goes to the next decide chat. A variant chosen or
changed after reading the held outcome is reported as chosen after reading held data (the RB rule §4.2). The licensed
sentences are in the spec §6. The diagnostics (spec §3.6) and the reported margins (B1 to B6, F4, Leiden, G's emotion and style
with the RB rule's margins) are computed only after the outcome is written to the verdict file. The held read gets a
row in `docs/superpowers/held_ledger.md`.

**Weights.** Every run saves its weights. Kept: each stage A arm with A_genre ≥ T_g − r_genre (the EG rule's user
rule), every stage B run, and G's five full-scale seeds; the others go on the pending deletions list for the
wrap-up.

## 5. Frozen and rerun

- **Frozen:** the EG rule's frozen list; 12-25's stage iii exports and weights of GR-8 (seeds 0 to 3) and R6-5
  (seeds 1 to 3), its `score_first.json`, `score_all.json`, `score_fresh.json` and `reported.json`, P and its hash,
  the labels, the seed ranges, T_g; 12-24's K4 full-scale weights and held outputs, its `reported_stage4.json`, T's
  held store and the frozen partitions of T and K4 (§3.2), the RB rule's entries B1 to B6 and F4 on H, H and its
  hash; this rule.
- **Rerun:** every new stage A arm at seed 0; GR-2 and GR-4 at seeds 1 to 3; v*_A at seeds 1 to 3 when A-NEW; G at
  full scale, seeds 0 to 4, with its held inference; every scoring and bootstrap; the re-derivations VA2, VB2, VC2;
  the loss-scale run (spec §3.6): R6-5's settings at seed 0 with λ = 1, the direct and relational terms' values
  logged per step over the first epoch, no export on P, deciding nothing.
- **New:** the trainer's three switches (the direct term off, the teacher temperature, the anchors) and the anchors
  loss with its equivalence test.

## 6. Exceptions to the constitution

| Principle | Exception | Why | Simpler option rejected |
|---|---|---|---|
| none | — | — | — |

## 7. ID glossary

| ID | Meaning |
|---|---|
| EG rule, RB rule, CT rule | 12-25's rule (encoders and genre); 12-24's rule (recipe and backbones); the claim test's rule (2026-10-20) |
| R6-5 | the current recipe: no own-caption terms, teacher share 1/5 (12-24's pick) |
| K4 | R6-5 on the default Qwen image at full scale, 12-24's held candidate; R in this rule |
| GR-λ | R6-5 plus the in-batch relational term at weight λ, beside the direct term |
| GN-8 | GR-8 with the direct term removed from the genre head |
| GS-τ | GR-8 with the teacher's row at temperature τ (0.02 or 0.01); the student's row at 0.05 |
| GQ-8 | GR-8 with teacher anchors (4,096 fixed teacher rows) in place of the in-batch rows |
| A_s, A_t; EQ | the anchors loss's student-side and teacher-side anchor sets (both A in GQ-8); the equivalence test, with A_s and A_t set to the batch's own rows |
| P, L_a, r_a, T_g | the 29,232 development paintings; those labelled for aspect a; the seed ranges; the instructed Qwen genre image's AMI on P |
| v*_A, v_held, G | the seed-0 winner; the arm picked on fresh seeds; the held candidate (v_held at full scale) |
| H, L_g, T, R | the 12,281 held paintings; the 9,949 with a genre label; the instructed Qwen genre image on H; the recipe's saved held genre outputs |
| d_T, d_R, X, G_half, PC | G minus T; G minus R; the margin 0.0128; half the held gap, 0.0263670169701253; the partial condition |
| A-NEW, A-SAME; B-PICK, B-NONE; G-GO, G-MM, G-PART, G-NO | the outcomes of stages A, B and C |
| VA1 to VA4, VB1 to VB4, VC1 to VC4 | the validity checks of stages A, B and C |
