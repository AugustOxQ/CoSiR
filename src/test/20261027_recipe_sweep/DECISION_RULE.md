# Decision rule: model G's held emotion claim test (stage 0), the staged recipe sweep (stages 1 to 3), the held test of the pick (stage 4), and the ablation

> Spec: [2026-10-27-recipe-sweep-design.md](../../../docs/superpowers/specs/2026-10-27-recipe-sweep-design.md)
> @ 35fcf68 (approved spec, first commit with the rule) · Written 2026-10-27 03:45 (Amsterdam) before any implementation script exists; revised 03:59 after the
> review · Reviewer: a fresh Opus `xhigh` worker, 2026-10-27 03:59 (pass after fixes: four must-fix, nine should-fix and
> twelve nit findings, all applied)

The trimmed constitution sets no statistics, development bar or failure reading, so this rule states or inherits them.
**Stage 0** inherits the RB rule's part 2 (`src/test/20261025_recipe_backbones/DECISION_RULE.md`, §2 "Part 2", §3.2,
§4.2), which inherits the CT rule's §3 (`src/test/20261020_claim_test/DECISION_RULE.md`); the change: G takes K4's
place at both scales. **Stages 1 to 3 and the development ablation** use the EG rule's definitions on P
(`src/test/20261025_encoders_genre/DECISION_RULE.md` §3.3: P, L_a, the partitions, A_a(v, I), the seed ranges r_a),
as the RG rule did (`src/test/20261027_relational_genre/DECISION_RULE.md` §3.1); new here: the mean gain, its guard,
the knob gains, the grid and a joint bootstrap. **Stage 4 and the held ablation** use the RG rule's §3.2 statistic
d_T, per aspect, and the RB rule's emotion margin over PercepT. C7, C8, C12 and C13 apply as written. Where this rule
and the spec differ, this rule governs. Paths start at `src/test/`: "RG/" is `20261027_relational_genre/`, "RB/"
`20261025_recipe_backbones/`, "EG/" `20261025_encoders_genre/`, "CT/" `20261020_claim_test/`. A batch is 512 graph
pairs (1,024 endpoint rows) unless knob k8 changes it.

## 1. Prior (C8)

Written before any new run and before any new number on P or H exists. Known: G's 12-27 values on P (seed 0: emotion,
style and genre in RG/outputs/stage_a/verdict.json; genre 0.5074) and on seeds 1 to 3 (genre 0.5050); G's held AMIs
at full scale (emotion 0.1039, style 0.4057, genre 0.5001) and its margins against T (emotion +0.0118, style +0.0032,
genre −0.0060); K4's emotion margins over PercepT (sample +0.0190 [+0.0137, +0.0201], full +0.0154 [+0.0100,
+0.0169]); the bars 0.0855 (sample) and 0.0874 (full).

- **Stage 0.** Sample d +0.015 to +0.021, full d +0.014 to +0.018. G-ES-GO 0.85, G-EF-GO 0.9, E0 0.85. G's full-scale
  held emotion AMI is known, so the full-scale point estimate is close to known (about +0.0164); the interval is not.
- **Stage 1.** Knobs with a positive knob gain: 6 of 13 (3 to 9). The largest knob gain +0.6 seed ranges (+0.3 to
  +1.2), likeliest from k2 (the style relational term), k6 (a lower learning rate) or k10 (more dropout). k1 (the
  emotion relational term) is the least known: −0.5 to +0.8. k13 (patience) most likely gives M = 0: G's best epochs
  are 3, 2 and 4, and its stopping losses rise after them. GRID-4: 0.7; SWEEP-NONE: 0.05.
- **Stage 2.** The best M +1.0 (+0.5 to +1.6); a grid run first in TOP: 0.6.
- **Stage 3.** PICK 0.5; given PICK, M̄(c*) +0.3 to +0.7.
- **Stage 4**, given PICK: T-WIN on emotion 0.9, style 0.4, genre 0.15; all three T-WIN 0.07; a T-BELOW on any
  aspect 0.15; PE-GO 0.9.
- **Ablation** (held, the final model): the buddy terms and the instructed graphs carry the largest drops (as D2's
  ablation); removing the relational term lowers genre by 0.03 to 0.05; the sharper row, the direct teacher and the
  shared trunk each move every aspect by less than one seed range.

Disclosure: spec §8.

## 2. Candidates and comparators

### 2.1 Stage 0 (held, both scales)

| ID | What | Role |
|---|---|---|
| G (sample) | GS-0.02 trained on the frozen sample: seeds 0 to 3 are 12-27's weights (`RG/outputs/train/GS-0.02/s{0..3}/{aspect}.pt`); seed 4 new, by this round's trainer with G's settings after V1a; held outputs of the three heads on H from each head's best state, float16, 2048 wide | candidate |
| G (full) | 12-27's full-scale seeds 0 to 4: weights `RG/outputs/stage_c/full/weights/GS-0.02/`, held outputs `RG/outputs/stage_c/full/held/GS-0.02/{aspect}__s{0..4}.npz` | candidate |
| B1 to B6 | the RB rule's PercepT entries at the same scale (saved held outputs; eligibility frozen as the RB rule) | comparators |

G reads the default Qwen image, so its matched arm is B1, as K4's; no other matched arm.

### 2.2 Stages 1 and 2 (seed 0, P)

Every run is G's sample-scale training (the RG rule's GS-0.02 at λ = 8, 512 pairs, the frozen sample, early stopping
per aspect on the 1,489 stopping paintings) with the named knobs changed, exports of the three image heads on P from
each head's best state. **The knobs and their values:** the number sheet of the spec, §3.2, at this rule's spec
commit (k1 to k13, three values each, G's value beside). Knob notes: k1 and k2 add the RG rule's relational term to
the emotion or style head (in-batch, rows of the same painting excluded, teacher row at the run's k4 temperature,
student row at 0.05, teacher the instructed Qwen image of that aspect, beside the direct term); k4 sets the teacher
row's temperature of every relational term in the run; k5 sets the direct term's weight on all three heads; k8 sets
the batch of training and of the stopping loss; k11 replaces 20 in every mutual top-k of the CT's graph code (the
aspect graphs and the graphs that set the per-node λ), all else unchanged; before any k11 run, the rebuild at k = 20
equals the frozen sample graphs bit for bit, otherwise k11's runs do not run, are not counting, and the cause is
reported; k12 sets the temperature of the four buddy terms only; k13 sets
the patience of every aspect's stopping clock.

| ID | Runs | Role |
|---|---|---|
| G₀ | G at seed 0 on the machine of stages 1 and 2 | the reference |
| c(k, v) | G with knob k at value v, seed 0: 39 runs | stage 1 candidates |
| grid points | G with the grid knobs at their levels (§4.2), seed 0 | stage 2 candidates |
| G₀′ | G at seed 0 again, right after G₀ | the node's repeat check (N3) |

### 2.3 Stage 3 (seeds 1 to 3, P, local)

| ID | Runs | Source |
|---|---|---|
| TOP's configurations | seeds 1, 2, 3, new, local | trained as their seed-0 runs |
| G | seeds 1, 2, 3 | 12-27's stage B runs, frozen (`RG/outputs/train/GS-0.02/s{1,2,3}`); the comparator |

### 2.4 Stage 4 (held, full scale, local)

| ID | What | Role |
|---|---|---|
| c* | the stage 3 pick trained as G's full-scale run (the 42,969 train paintings, the CT's 42,969-node graphs, rebuilt with c*'s k when k11 ≠ 20, stopping on the 6,152 validation paintings), seeds 0 to 4; held outputs of the three heads | candidate |
| T_a | the instructed Qwen image of aspect a on H: 12-24's three frozen r20 partitions, `RB/outputs/part2/partitions/full/T/{a}__s0__r20.npz` | the deciding comparator per aspect |
| B1 to B6 | the RB rule's full-scale PercepT entries | emotion comparators |
| G | 12-27's full-scale held outputs (§2.1) | reported only |

### 2.5 Ablation (local)

F, the final model: c* under PICK, else G. Each removal x of the spec's §3.2 table is F's settings with that one
part removed and nothing retuned. Graphs without the content correction and the default-view graph are built by the
CT's graph code with F's graph settings, at each scale. Development: each x at seeds 1 to 3 on the frozen sample,
exports on P; F's runs from stage 3 (or 12-27's G runs). Before these graphs are used, the code reproduces at the
sample scale the D2 ablation's frozen default-view graph (`20261021_d2_ablation/graphs`) and the aspect graph
sheet's uncorrected graphs (`20261018_aspect_graph_sheet/graphs`) bit for bit; otherwise those two removals do not run
and the cause is reported. Held: each x at full scale, seeds 0 to 4, held outputs; F's
from stage 4 (or 12-27's G).

## 3. The statistics

Decisions use unrounded values; reports give 4 decimals. r = (r_emotion, r_style, r_genre) = (0.0046, 0.0082,
0.0128). Every bootstrap has 2,000 draws, and its 95% interval is `numpy.percentile(·, [2.5, 97.5])`.

### 3.1 Stage 0

The RB rule §3.2 as written at each scale, with G in K4's place: A_cand from G's r20 partitions (mean over the three
states, then the seeds); the entries' two readings (r20, own) of B1 to B6; d = A_G(I₀) minus the maximum over the
eligible entries; the bootstrap with its own `default_rng(0)` per scale, draws in the RB order (I_b, then S_b for G,
then for B1 to B6 with eligible seeds), the maximum retaken in each draw. **The scorer check:** on 12-24's saved K4
outputs with K4 in G's place, the scorer gives the K4 blocks of `RB/outputs/part2/verdict_stage3.json` (sample) and
`verdict_stage4.json` (full): d, A_cand, each A_e and both bounds, within 1e-6.

### 3.2 Stages 1 and 2

For a run c at seed 0: A_a(c) = the EG rule's A_a(c, L_a) on P (MiniBatchKMeans, k = 20, states 0, 1, 2, mean of the
AMIs against L_a). gain_a(c) = A_a(c) − A_a(G₀). **The mean gain** M(c) = (1/3) Σ_a gain_a(c) / r_a. **The guard:**
gain_a(c) ≥ −r_a / 2 for every aspect a. A run is **counting** when it passed V1b and the guard.

### 3.3 Stage 3

For a configuration c: Ā_a(c) = the mean over s in {1, 2, 3} of A_a(c, L_a, s); ē_a(c) = Ā_a(c) − Ā_a(G); M̄(c) =
(1/3) Σ_a ē_a(c) / r_a. **The joint bootstrap:** per configuration its own `rng = numpy.random.default_rng(0)`; for b
= 1 to 2,000, in this order: I_b = `rng.integers(0, 29232, size=29232)` (rows of P in its order); S_b =
`rng.choice([1, 2, 3], size=3, replace=True)`, shared by c and G and by all three aspects. A_a(v, I_b, s) is the AMI
of v's frozen seed-s partitions (the three states, mean) over the rows of I_b that carry an a-label (repeats kept,
unlabelled rows dropped). ē_a,b = the mean over S_b of A_a(c, I_b, s) − A_a(G, I_b, s); M_b = (1/3) Σ_a ē_a,b / r_a.
Intervals for each ē_a and for M̄ from the same draws. **The scorer check:** on 12-27's stage B files with R6-5 in G's
place and GS-0.02 in c's, the point values Ā_a and ē_a equal `RG/outputs/stage_b/verdict.json`'s within 1e-6, in the joint
mode the emotion and style intervals equal them too (all 29,232 P rows carry both labels), and in the EG rule's
per-aspect draw mode (§3.3 "Fresh seeds") all three do.

### 3.4 Stage 4

Per aspect a, the RG rule §3.2 with c* in G's place and T_a in T's: labels L_a on H (emotion: the CT's
`painting_labels` in annotation order; style and genre: the first caption's label, labels < 0 masked), A_c*(I, S) over
c*'s three r20 partitions per seed, A_T,a(I) over T_a's three partitions; **d_T,a = A_c*(I₀, S₀) − A_T,a(I₀)**; its
own `default_rng(0)` per aspect: I_b = `rng.integers(0, 12281, size=12281)`, then S_b =
`rng.choice([0, 1, 2, 3, 4], size=5, replace=True)` for c*; T draws nothing. Its size in seed ranges is d_T,a / r_a.
**The emotion margin over PercepT:** §3.1 at full scale with c* in G's place. **Reported:** d_G,a = c* minus G per
aspect (its own `default_rng(0)`; I_b, then one S_b shared by c* and G); style and genre against B1 to B6 as the RB
rule's reported margins; Leiden on genre, as 12-27's readings. **The scorer check:** on G's saved held outputs with G in c*'s place, d_T,genre and its
interval equal `RG/outputs/stage_c/verdict.json`'s, and d_T,emotion and d_T,style with their intervals equal the
margins against the instructed image in `RG/outputs/stage_c/reported.json`, within 1e-6.

### 3.5 Ablation

Development: for each removal x, ē_a(x) = Ā_a(x) − Ā_a(F) over seeds 1 to 3 on P, with §3.3's joint bootstrap (own
`default_rng(0)` per removal). Held: Δ_a(x) = A_x(I₀, S₀) − A_F(I₀, S₀) on H at full scale, per aspect its own
`default_rng(0)`: I_b, then one S_b over seeds 0 to 4 shared by x and F. Each arm's absolute AMI per aspect is
reported beside.

## 4. Rules

### 4.0 Machines and order

- Stages 1 and 2 run on one machine: node402, or both locally after the node route fails (§4.2). Stages 0, 3 and 4,
  the ablation and the analysis run locally. No decision compares a node number with a local number.
- Order: the gates V0a, V1a, V3a and V4a in the build; then stage 1 on the node while stage 0 and G's analysis run
  locally; stage 2; stage 3; stage 4; the final model's analysis; the development ablation by day; the held ablation
  from 19:00 (the weekday rule). A failed gate stops only the stages behind it.

### 4.1 Stage 0

Validity first; no outcome is stated until V0a to V0d hold.
- **V0a** (C12) The scorer check of §3.1 at both scales. G's full-scale held outputs have the SHA-256 recorded in
  12-27's stage C metadata, and held inference from G's full-scale seed-0 weights equals the saved seed-0 outputs bit
  for bit. G's sample-scale weights at seeds 0 to 3 are 12-27's files (SHA-256 recorded in the build), and V1a passed
  before seed 4 trains. Before G's sample-scale held inference, this round's inference from 12-25's R6-5 sample
  seed-1 weights (`EG/outputs/stage_iii/train/R6-5/s1`, equal to K4's by the EG rule's fresh-seed check) equals
  `RB/outputs/part2/sample/held/K4/{aspect}__s1.npz` bit for bit.
- **V0b** (C13) Separate code, given this rule, the RB and CT rules, the held list, the captions, the saved outputs of
  G and of B1 to B6 and the frozen eligibility (never our numbers), recomputes the labels, the partitions, each A_e,
  d, both bounds and the outcome at both scales. Equal partitions; values within 1e-6; the same outcomes.
- **V0c** Each G output set (three aspects, five seeds, each scale) has 12,281 finite rows of unit norm in H's order;
  H's SHA-256 equals the CT's.
- **V0d** (seed policy) Seed 4 trains once; no sample seed is retrained or replaced after its held output exists; a
  seed that fails V0c is rerun once (same seed and code), listed. A seed that still fails leaves that scale with no outcome.

Outcomes per scale (the RB rule's thresholds): **G-ES-GO** (sample) d ≥ 0.01 and lower bound > 0; **G-ES-MM** d ≥ 0.01
and lower bound ≤ 0; **G-ES-IN** 0 < d < 0.01; **G-ES-NO** d ≤ 0. **G-EF-GO, -MM, -IN, -NO** the same at full scale.
**E0** (the licensed sentence, spec §6): G-EF-GO, and the sample scale has an outcome other than G-ES-NO. Stage 0 leads to nothing automatically: the sweep runs
whatever it gives. The analysis of G is computed after stage 0's verdict file is written.

### 4.2 Stages 1 and 2

Validity first; no outcome is stated until V1a to V1d hold.
- **V1a** (C12, local, in the build) This round's sample-scale trainer with every knob at G's value gives exports on P
  equal to 12-27's GS-0.02 exports bit for bit at seed 0 and at seed 1 (`RG/outputs/train/GS-0.02/s{0,1}`), and its
  scorer gives G's seed-0 A_a of `RG/outputs/stage_a/verdict.json` within 1e-6. Otherwise no run of stages 0 (seed
  4), 1 to 4 or of the ablation starts; the cause goes in the report.
- **V1b** Every export on P has 29,232 finite rows of unit norm in P's order; P's SHA-256 equals 12-25's
  (`EG/outputs/stage_iii/reported.json`, `p_sha256`).
- **V1c** (C13) On the machine of the runs, separate code, given this rule, P's names, the labels and the exports,
  recomputes every partition and A_a: equal partitions, A within 1e-6. Locally, separate code, given this rule and both
  sets of A values, recomputes every gain, M, guard, knob gain, the grid, the pool and TOP: the same outcome. A
  disagreement is traced and fixed before the next stage starts; both values and the fix go in the report.
- **V1d** (seed policy) Only G₀, G₀′, the 39 stage 1 runs and the grid's new points run. A run that fails V1b is rerun
  once (same seed and code), listed. A run that stops on a non-finite loss or still fails V1b is not counting; the
  report gives the cause.

**The node route (N).** N1, the node check: the env's package versions, the Qwen stores, the inputs folder's files
equal to local by SHA-256, and at least 60 GB free on the node's local disk. N2, a smoke job: a short run of G that
completes and whose export passes V1b. N3: G₀ and then G₀′ (the same run repeated at once) complete and pass V1b, and |A_a(G₀′) − A_a(G₀)| ≤ r_a / 2 for
every aspect. Readings, no gate: G₀ minus 12-27's local
seed-0 run per aspect (trainer parity); the node scorer on 12-27's local seed-0 exports minus 12-27's A values
(scorer parity); whether G₀′ equals G₀ bit for bit. **Fallback:** when N1, N2 or N3 fails, when
stage 1 cannot start on the node by 14:00, when stage 2 has not ended by 22:00, or when the node is lost before stage 2 ends and one rerun does not fix it,
stages 1 and 2 run locally from the start, with 12-27's local GS-0.02 seed-0 run as G₀ (frozen; V1a shows the trainer
reproduces it), and the node results are reported only. The run chat measures the power draw of the first local run:
under 200 W the runs go on by day; otherwise they wait for 19:00, and the held ablation follows them.
**Exception, 2026-10-27 15:31 (user, after seeing the recommendation): "keep the node route and shift the
deadlines".** The build ended after 14:00, so stage 1 starts on the node later in the evening. Both deadlines move by
the same shift, the real start of stage 1 on the node minus 14:00: stage 1's start deadline is its real start (the run
chat sets `RS_START_BY` to it), and stage 2 must end by that start plus 8 hours (22:00 plus the shift; `RS_END_BY`).
The other fallback triggers stay as written: N1, N2 or N3 fails, or the node is lost and one rerun does not fix it.

**Stage 1.** For each knob k: its knob gain g_k = the largest M over its counting values; none when no value counts.
- **Grid knobs:** the knobs with g_k > 0; the 4 with the largest g_k when more qualify; ties to the number sheet's
  order (k1 first).
- **GRID-n** (n = 1 to 4 grid knobs): stage 2 runs. **SWEEP-NONE** (no knob with g_k > 0): stages 2, 3 and 4 do not
  run; G stays the recipe.

**Stage 2.** The levels of a grid knob: G's value and its counting values with M > 0, at most two (the largest M; ties
to the listed order of values).
- **The grid:** every combination of the grid knobs' levels, all other knobs at G's value. The all-G point is G₀; the
  points with one changed knob are stage 1 runs, reused; every other point is a new run at seed 0 on the same machine.
- **The pool:** every counting run of stages 1 and 2 with M > 0, G₀ and G₀′ excluded.
- **TOP:** the 3 runs of the pool with the largest M (all of them when fewer); ties to fewer changed knobs, then the
  sorted tuple of the changed knobs' numbers, compared lexicographically, then the tuple of their values' positions
  in the sheet. **TOP-NONE** (the pool is empty; after
  GRID-n it never is, stated for completeness): stages 3 and 4 do not run; G stays.
- Reported, deciding nothing: each knob's curve (M and each gain_a against its values), each run's guard, the grid's
  table, best epochs.

### 4.3 Stage 3

Validity first; no outcome is stated until V3a to V3d hold.
- **V3a** (C12, in the build) The scorer check of §3.3, run on G's frozen exports at seeds 1 to 3; the build records
  their SHA-256, and stage 3 reads only files with those hashes.
- **V3b** = V1b per export. **V3c** (C13) Separate code, given this rule, P's names, the labels and the exports,
  recomputes the partitions, Ā, ē, M̄, the intervals, the qualifiers and c*: equal partitions, values within 1e-6, the
  same c*.
- **V3d** (seed policy) Only TOP's configurations at seeds 1 to 3, trained locally; a run that fails V3b is rerun once
  (same seed and code), listed. A configuration with a seed that still fails, or stops on a non-finite loss, does not
  qualify.

Conditions:
- **Qualifies:** for every aspect a, the lower bound of ē_a(c)'s interval > −r_a; and the lower bound of M̄(c)'s
  interval > 0.
- **PICK:** c* = the qualifier with the largest unrounded M̄; ties to TOP's order. c* is written to the stage 3
  verdict file before any full-scale run of c*. **KEEP-G:** no configuration qualifies: G stays the recipe; stage 4
  does not run.
- Worked example (the user's accepted scenario): ē = emotion −0.0010 [−0.0040, +0.0020], style +0.0100, genre +0.0080.
  Emotion passes (−0.0040 > −0.0046); M̄ = (−0.217 + 1.220 + 0.625) / 3 = 0.542; the configuration qualifies when
  style's and genre's lower bounds are above −0.0082 and −0.0128 and M̄'s lower bound is above 0.
- Reported: each configuration's Ā_a, ē_a and M̄ with intervals; whether c* leads the next qualifier by less than its
  interval's half width.

### 4.4 Stage 4

Runs only under PICK. Validity first; no outcome is stated until V4a to V4d hold.
- **V4a** (C12, in the build) This round's full-scale trainer with G's knobs at seed 0 gives weights equal to
  `RG/outputs/stage_c/full/weights/GS-0.02/{aspect}__s0.pt` bit for bit, and held outputs equal to G's saved seed-0
  held outputs bit for bit. Every full-scale run records the SHA-256 of the frozen trainer modules it imports (the
  `ms_` and `ct2_` modules) and of RG's `rg_loss` and `rg_full`; they equal their values at commit 68e8081. The scorer
  check of §3.4 within 1e-6. Otherwise no held output of c* is computed, and the held ablation does not run.
- **V4b** (C13) Separate code, given this rule, the RG, RB and CT rules, the held list, the captions and the saved
  outputs (never our numbers), recomputes the labels, the partitions, each d_T,a, the emotion margin, their intervals
  and the outcomes: equal partitions, values within 1e-6, the same outcomes.
- **V4c** = V0c for c*'s output sets. When k11 ≠ 20, the full-scale graph rebuild at k = 20 equals the CT's
  42,969-node graphs bit for bit before c*'s graphs are built.
- **V4d** (seed policy) c*'s five seeds run once; a seed that fails V4c is rerun once (same seed and code), listed; no
  second candidate and no change to c* after its first held output exists. A seed that still fails leaves stage 4 with
  no outcome.

Outcomes per aspect a: **T-WIN** the lower bound of d_T,a > 0; **T-LEVEL** not T-WIN and the lower bound > −r_a;
**T-BELOW** otherwise. Emotion over PercepT: **PE-GO, PE-MM, PE-IN, PE-NO** with the thresholds of §4.1. Under KEEP-G
or SWEEP-NONE or TOP-NONE, G's outcomes come from this round's stage 4 scorer on G's saved held outputs (equal to
12-27's by V4a): genre and style T-LEVEL and emotion T-WIN at the time of writing, with no new held run.

What the outcomes lead to: nothing automatically; every outcome goes to the next decide chat. The paper's recipe is
the stage 3 outcome (c* or G); the held outcomes test it and do not choose. A recipe changed after reading a held
outcome is reported as chosen after reading held data. The reported margins and the analysis of c* are computed only
after the outcomes are written to the verdict file. Each held read gets a row in `docs/superpowers/held_ledger.md`.

### 4.5 Ablation and analysis

- They decide nothing. The analysis (spec §3.6) runs on G after stage 0's verdict and on c* after stage 4's verdict.
- Each removal arm passes V1b (development) or V0c (held) per export, with one rerun on failure (same seed and code),
  listed. An arm that stops on a non-finite loss is reported as failed, with no effect.
- (C13) Separate code recomputes every number of the held ablation table (each Δ_a and its interval) and the
  development point values ē_a, within 1e-6.
- The removal switches have unit tests in the build: each removes only its part (the loss terms present, the graphs
  read, the trunk count).

**Weights and outputs.** Every run saves its weights. Kept: G₀, TOP's runs, every stage 3, stage 4 and ablation run.
G₀'s and TOP's weights and exports are copied to `/data/SSD2/cosir_r10/20261027_recipe_sweep/` before 2026-10-30
(the reservation ends about 2026-10-31 00:40). The node's outputs stay on the node, with their size on
`.scratch/pending_deletions.md` for the wrap-up; the other local runs go there too.

## 5. Frozen and rerun

- **Frozen:** 12-27's GS-0.02 runs (sample seeds 0 to 3: weights and exports; full seeds 0 to 4: weights, held outputs,
  metadata), its stage A, B and C verdicts and readings, P and its hash, the labels, the seed ranges; 12-24's K4 held
  outputs and verdicts of stages 3 and 4, its T partitions, the PercepT entries B1 to B6 at both scales with their
  eligibility, H and its hash; the CT's sample and full-scale graphs; the spec's number sheet; this rule.
- **Rerun:** G's sample seed 4 and the held inference of G's sample seeds 0 to 4; G₀ and G₀′; every stage 1 run and
  grid point; TOP's configurations at seeds 1 to 3; c* at full scale with its held inference; every ablation run; the
  rebuilt graphs (k11, the ablation's graphs); all scoring, bootstraps and re-derivations.
- **New:** the trainer's switches for k1, k2 (the relational term on emotion and style), k12 (the buddy temperature
  apart from the others), k11 (graphs with another k), and the eight removal switches; the joint bootstrap; the stage 0
  adapter for G.

## 6. Exceptions to the constitution

| Principle | Exception | Why | Simpler option rejected |
|---|---|---|---|
| C12 | On node402 the trainer and scorer need not reproduce 12-27's local numbers; their parity is recorded, not a gate (N). C12 holds locally (V0a, V1a, V3a, V4a, the ablation's graph check). | Stages 1 and 2 compare node runs only with node runs (user, Q1). | A parity gate: the 10-20 gap (up to 0.0168) would likely close the route. |

## 7. ID glossary

| ID | Meaning |
|---|---|
| CT, RB, EG, RG rule | the claim test's rule (2026-10-20); 12-24's rule (recipe and backbones); 12-25's rule (encoders and genre); 12-27's rule (relational genre) |
| G, GS-0.02 | model G: the recipe R6-5 plus the relational genre term at weight 8 with the teacher row at 0.02 (12-27's pick) |
| G₀, G₀′ | G at seed 0 on the machine of stages 1 and 2; the same run repeated at once (the repeat check) |
| k1 to k13; c(k, v) | the knobs of the spec's number sheet; G with knob k at value v |
| M, g_k, the guard | the mean gain in seed ranges over G₀; knob k's best counting M; no aspect more than half a seed range below G₀ |
| M̄, ē_a | the three-seed mean gain and gains over G (seeds 1 to 3) |
| c*, F | the stage 3 pick; the final model (c*, or G) |
| T_a, d_T,a | the instructed Qwen image of aspect a on H; c* minus T_a |
| B1 to B6, K4 | the RB rule's PercepT entries; 12-24's recipe on the default Qwen image |
| P, H, L_a, r_a | the 29,232 development paintings; the 12,281 held paintings; the paintings with an a-label; the seed ranges |
| G-ES-*, G-EF-*, E0 | stage 0's outcomes at the sample and full scale; its licensed sentence |
| GRID-n, SWEEP-NONE; TOP, TOP-NONE; PICK, KEEP-G | the outcomes of stages 1, 2 and 3 |
| T-WIN, T-LEVEL, T-BELOW; PE-* | stage 4's outcomes per aspect against T_a; its emotion claim over PercepT |
| V0a to V4d, N1 to N3 | the validity checks of each stage; the node route's checks |
