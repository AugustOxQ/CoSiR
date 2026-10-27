# A sharper teacher in the relational genre term brings genre level with the instructed Qwen genre image on held paintings (G-GO)

> 2026-10-27 (sequence name 12-27) · Spec: [design](../../../superpowers/specs/2026-10-26-relational-genre-design.md) · Decision rule: [binding](../../../../src/test/20261027_relational_genre/DECISION_RULE.md) @ 2e9529a · Run log: [round log](../../../../src/test/20261027_relational_genre/20261027_relational_genre_log.md) · Previous round: [12-25](2026-12-25_encoders_genre.md)

## Results brief

**What we tried.** We changed the relational genre term of the current recipe in six ways on development paintings,
picked the best change on fresh seeds, and tested it once on held paintings at full scale
([spec](../../../superpowers/specs/2026-10-26-relational-genre-design.md)). The relational term makes each painting's
similarities to the other paintings of the batch copy the teacher's.

**Outcome against the bar.** The picked change is a sharper teacher row (temperature 0.02 in place of 0.05) at weight 8.
On 9,949 labelled held paintings, 5 seeds, the outcome is GO: genre level with the instructed Qwen genre image within
one seed range (0.0128).

| Held genre AMI (k-means) | Value | The picked model minus it |
|---|---|---|
| The picked model | 0.5001 | |
| Instructed Qwen genre image (the bar) | 0.5062 | −0.0060 [−0.0105, −0.0014] |
| Current recipe | 0.4534 | +0.0467 [+0.0402, +0.0531] |
| Strongest PercepT core entry | 0.4269 | +0.0732 [+0.0622, +0.0795] |

On development data the sharper teacher won seed 0 (0.5074 against weight 8's 0.5018) and was confirmed on seeds 1 to
3 (0.5050 against the recipe's 0.4694). Every gate and all three independent re-derivations agree.

**What it does not show.** The model is still below the instructed image: the whole interval is under zero, so
"matches or exceeds" is not licensed. The fresh-seed pick is by 0.0028 over weight 8, inside one seed range. Weight 8
was not tested on held paintings. The claim is k-means, Qwen and ArtELingo only. The emotion margin against PercepT
core, the emotion claim's bar, was not computed for this model.

**Options for next.**
1. Make the sharper-teacher model the recipe and re-run the held emotion claim test with it (about 1 hour; chosen
   after reading held data, so disclosed). It tells us whether one model carries all three aspects.
2. Test the change on the other passing encoders (SigLIP 2, SigLIP 1, CLIP L/14): a development check, then held
   tests (about half a day). It tells us whether the genre gain is Qwen's alone.
3. Write up the encoder and genre claims for the 2026-11-10 abstract (writing time, no compute).

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The genre gap of the recipe is almost closed: the held lift (+0.0467) is larger than the development
lift (+0.0356), and 0.0467 of the 0.0527 gap is gone. We expected a partial outcome (prior: GO 0.15, partial 0.50).
We are fairly sure the relational term drives the lift, since every confirmed arm gains +0.027 to +0.036 on
development data. We are not sure the sharper teacher matters: its fresh-seed lead over weight 8 is inside one seed range. A
likely reason for the larger held lift (not checked): at full scale the recipe's genre head stops at epoch 1 in every
seed, while this model's genre head trains to epochs 4 to 7.

**Our recommendation.** Option 1 first: it is cheap, and a paper needs one model for its emotion, style and genre
claims. Then option 3.

</details>

## What ran

All three stages ran in one queue chain with no stop, no failed run and no rerun.

- **Stages and types:** stage A, a sweep at seed 0 (weights 16 and 32 and the climb, the relational term alone, teacher
  temperatures 0.02 and 0.01, teacher anchors); stage B, a confirmation with a pick on seeds 1 to 3 (weights 2, 4, 8
  and the seed-0 winner); stage C, a claim test on held paintings (the pick at full scale, seeds 0 to 4).
- **Data:** stages A and B: 29,232 development paintings (genre labels on 23,682), training on 13,400 fit paintings;
  stage C: 42,969 train paintings, early stopping on 6,152 validation paintings, 12,281 held paintings (genre labels
  on 9,949). Held data chose nothing: the stage B verdict file named the candidate before any full-scale run.
- **Code:** main at 68e8081 (the round's code as merged at b8310b1, not dirty), local RTX 3090 under the GPU lock and
  8 CPU threads; 2026-10-27 00:33 to 02:16 (stage A 23 min, B 47 min, C 33 min; the estimate was 210 min).
- **Gates:** VA1 (both halves), EQ, VB1 and VC1 (both halves) passed in the build on 2026-10-26 at the same code; the
  queue kept those files after checking their code hashes. EQ passed again in the queue. The interrupted-run log (rule VA4)
  has no line.

## Results

### Stage A: seed 0, development paintings

The sharper teacher at 0.02 wins seed 0 within the guard (A-NEW). Weight 32 is below weight 16, so the ladder stops
at 32. The teacher temperature 0.01 fails the style guard.

| Arm | What changes | Genre AMI | Genre effect [95%] | Emotion effect | Style effect | Guard |
|---|---|---|---|---|---|---|
| R6-5 | the recipe | 0.4674 | | | | |
| GR-8 | weight 8 (12-25, frozen) | 0.5018 | +0.0345 [+0.0315, +0.0377] | −0.0032 | −0.0052 | pass |
| GR-16 | weight 16 | 0.5022 | +0.0348 [+0.0317, +0.0382] | −0.0008 | −0.0030 | pass |
| GR-32 | weight 32 | 0.4964 | +0.0291 [+0.0258, +0.0324] | −0.0013 | +0.0027 | pass |
| GN-8 | weight 8, no direct term | 0.4972 | +0.0298 [+0.0267, +0.0332] | −0.0011 | +0.0024 | pass |
| GS-0.02 | weight 8, teacher row at 0.02 | **0.5074** | +0.0401 [+0.0369, +0.0433] | −0.0016 | −0.0016 | pass |
| GS-0.01 | weight 8, teacher row at 0.01 | 0.4882 | +0.0208 [+0.0178, +0.0239] | −0.0009 | −0.0094 | **fail** |
| GQ-8 | weight 8, 4,096 teacher anchors | 0.4900 | +0.0227 [+0.0199, +0.0253] | −0.0007 | +0.0000 | pass |

The guard allows at most −0.0046 on emotion and −0.0082 on style. The instructed Qwen genre image gives 0.4989 on
these paintings, so GR-8, GR-16 and GS-0.02 are at the target level and the others are promising (at least 0.4861).
The anchors loss equals the in-batch loss on three real batches (EQ, difference 0 at 1e-6), so the teacher-anchors arm
could win; it is the weakest guarded arm.

### Stage B: seeds 1 to 3, development paintings

All four arms are confirmed, and the sharper teacher has the highest three-seed mean (B-PICK). The four means span
0.0085, less than one genre seed range (0.0128), so the pick follows the rule's order, not a clear gap.

| Arm | Genre AMI, mean of 3 seeds | Genre effect over the recipe (0.4694) | Emotion effect | Style effect | Level |
|---|---|---|---|---|---|
| GR-2 | 0.4965 | +0.0272 [+0.0180, +0.0387] | −0.0003 [−0.0022, +0.0013] | −0.0036 [−0.0110, +0.0014] | C-HOLDS |
| GR-4 | 0.5005 | +0.0311 [+0.0216, +0.0387] | −0.0000 [−0.0023, +0.0020] | +0.0009 [−0.0017, +0.0033] | C-HOLDS |
| GR-8 | 0.5022 | +0.0328 [+0.0235, +0.0442] | +0.0004 [−0.0012, +0.0022] | −0.0010 [−0.0056, +0.0028] | C-HOLDS |
| GS-0.02 | **0.5050** | +0.0356 [+0.0263, +0.0464] | +0.0017 [−0.0003, +0.0039] | −0.0010 [−0.0101, +0.0057] | C-HOLDS |

### Stage C: held paintings, full scale, 5 seeds

The sharper-teacher model (G) is level with the instructed Qwen genre image within one seed range (G-GO). Its genre
AMI per seed is 0.4967, 0.5051, 0.4957, 0.5014 and 0.5018 (recipe: 0.4463 to 0.4604).

| ID | Measured | Number | Compared against | Bar | Outcome |
|---|---|---|---|---|---|
| d_T | genre AMI, G minus T | −0.0060 [−0.0105, −0.0014] | instructed Qwen genre image, 0.5062 | lower bound > −0.0128 | pass (G-GO) |
| matches or exceeds | lower bound of d_T | −0.0105 | 0 | > 0 | not met |
| PC | genre AMI, G minus the recipe | +0.0467 [+0.0402, +0.0531] | recipe K4, 0.4534 | ≥ 0.0264, lower bound > 0 | holds (reported) |

**Verdict under the rule:** G-GO. The licensed sentence (spec §6): on 9,949 held ArtELingo paintings with a genre
label, a model whose image side reads the default Qwen image, trained on the 42,969 train paintings with
Qwen3-VL-Embedding's instructed graphs, instructed teacher and default captions, gives genre clusters (k-means, 20
clusters) level with the instructed Qwen genre image within one seed range (0.0128): −0.0060 [−0.0105, −0.0014], 5
seeds, paired 95% interval over paintings and seeds.

## Analysis

**Re-derivations agree everywhere.** Separate code, given the rule and the exports, recomputed the climb rung (0
differences), stage A (0), stage B (0) and stage C (largest difference 6.9e-18).

**The ladder peaks between weights 8 and 16.** Weight 16 adds 0.0003 over weight 8 at seed 0, and weight 32 is
0.0054 below weight 8. The fresh seeds show the same shape below: weights 2, 4 and 8 give 0.4965, 0.5005 and 0.5022.

**The teacher temperature has a narrow best value.** At 0.02 the arm gains 0.0056 over weight 8 at seed 0; at 0.01
it is 0.0136 below weight 8 and its style effect is −0.0094. The relational term alone (no direct term) and the teacher anchors both lose
genre against weight 8, so the direct term and the in-batch rows both help.

**Loss scale** (one epoch at weight 1, seed 0, reported only): the direct term's mean is 3.72 and the relational
term's 0.30. With the direct term at 0.25 and the relational term at 8, the relational term carries about 2.6 times
the direct term's weighted value in the first epoch.

**Held readings beside the verdict** (decide nothing; 12-24's margins, G against the largest entry):

| Aspect | G | Recipe K4 | G against the instructed image of the aspect | G against PercepT core |
|---|---|---|---|---|
| emotion | 0.1039 | 0.1029 | +0.0118 [+0.0083, +0.0150] (0.0921) | not computed |
| style | 0.4057 | 0.3996 | +0.0032 [−0.0019, +0.0085] (0.4026) | +0.1760 [+0.1645, +0.1808] (B6 own, 0.2297) |
| genre | 0.5001 | 0.4534 | −0.0060 [−0.0105, −0.0014] (0.5062) | +0.0732 [+0.0622, +0.0795] (B5, 0.4269) |

- Against the frozen default Qwen image, G's genre gains +0.0481 [+0.0422, +0.0541] (0.4521); the recipe's gain was
  +0.0014 [−0.0050, +0.0074].
- Leiden genre: 0.5072 against the strongest PercepT entry's 0.4454, +0.0618 [+0.0527, +0.0718]. Leiden against the
  instructed image was not computed.
- Seed spread of G's genre: standard deviation 0.0039 (recipe 0.0050).
- Best epochs: the recipe's genre head stops at epoch 1 in all five seeds; G's at epochs 4 to 7. Emotion and style
  stop at the same epochs as the recipe except style in seeds 3 and 4.

**Development readings** (old 5,000 and Leiden, seed-0 arms only): on the old 5,000, GS-0.02 gives genre 0.5032
against GR-8's 0.5119 and the recipe's 0.4794; Leiden genre on development paintings gives GS-0.02 0.5005 against
GR-8's 0.4968 and GR-16's 0.5099. Full tables: `reported_ab.md` (below).

**Against the prior (rule §1).** A new seed-0 winner within the guard (0.6): yes. The held variant not weight 8
(0.4): yes. Held genre 0.480 to 0.500 and d_T −0.026 to −0.006: both at the top edge (0.5001, −0.0060). GO had 0.15;
partial 0.50. Below the guessed ranges: the temperature 0.01 arm (0.4882) and the anchors arm (0.4900).

## Deviations

- **None from the rule or the spec.** No amendment, no failed run, no rerun, no stop.
- **The rule's batch size is wrong in its text.** The rule's §2.1 speaks of 1,024 pairs per batch. The frozen trainer
  uses 512 pairs (1,024 rows), which VA1's bit-for-bit match needs; EQ's batches have 512 pairs. The code is right.
- **No old-5,000 or Leiden reading for weights 2 and 4.** They have no seed-0 run; the rule's stage B list has no such
  reading, and the rule governs over spec §3.6.
- **The held emotion margin against PercepT core is missing.** The stage C readings reuse 12-24's reported margins,
  which give emotion only against the frozen image, the instructed image and Leiden. G's held emotion AMI is 0.1039
  against the recipe's 0.1029, whose margin over PercepT core's own reading was +0.0154 (12-24). Not recomputed here.
- **Build defaults** (handoff of build 6): the full-scale held guard does not hash the frozen `ms_train` and `ct`
  modules; the teacher-anchors arm trained after EQ whether it passed or not (it passed).

## Files and storage

- **Outputs:** `/data/SSD2/cosir_r10/20261027_relational_genre/outputs/` (linked as the round folder's `outputs/`):
  `stage_a/climb.json`, `stage_a/verdict.json`, `stage_b/verdict.json`, `stage_c/verdict.json` (SHA-256
  a6fd488e…5361), the re-derivations `stage_a/rederive_va2_{rung,a}.json`, `stage_b/rederive_vb2.json`,
  `stage_c/rederive_vc2.json`, the readings `reported_ab.md` and `stage_c/reported.md`, queue status files and logs.
- **Held ledger:** row H19 in `docs/superpowers/held_ledger.md`.
- **Storage:** 12 GB on `/data/SSD2` (training runs 7.3 GB, stage C 2.3 GB, regression runs 1.5 GB). Kept by the rule:
  every stage A arm (all at or above 0.4861), every stage B run, G's five full-scale seeds. On
  `.scratch/pending_deletions.md` for the next wrap-up: the three VA1 runs, the loss-scale run and the VC1 run (97 MB
  of weights each), with the build's smoke outputs and worktrees.
