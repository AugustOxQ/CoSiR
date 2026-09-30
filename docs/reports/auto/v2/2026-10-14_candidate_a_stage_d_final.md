# CoSiR v2 Candidate A: stage (d) final held-out test, trained scorer G3 vs the naive rule

## Verdict

**On held-out paintings, the trained scorer G3 does not use the condition better than the naive rule
in both retrieval directions, so criterion 1 is NOT MET. It does pass the human emotion-vs-style swap
test, so criterion 2 is MET.** Seeds 43 and 44 of the same recipe give the same two verdicts.

The two questions, in plain terms:

- **Did the trained scorer use the condition better than naive, in both directions, on held? No.**
  - The condition-use gain Δ is positive in both directions: +0.83 R@1 points in image→text and +1.61
    in text→image. But both 95% CIs include 0: [−1.12, +2.78] and [−0.54, +3.81].
  - On the selection set the same model scored +1.83 and +3.25, with both CIs above 0. On held rows the
    gain is about half that size and no longer distinguishable from 0.
  - All of the remaining gain comes from **art style**, where Δ is +2.59 [+0.54, +4.69] (driven by text→image: +3.71 [+0.68, +6.74]). For **emotion** the gain is zero (−0.15 [−2.29, +2.05]).
  - Plain ranking is not better either. G3's held R@1 is 16.70 / 21.34 against naive's 17.63 / 20.90. The paired
    differences are −0.93 [−2.44, +0.59] and +0.44 [−1.17, +2.10].
- **Did it pass the human swap test? Yes.**
  - In the swap test, the anchor and the 13 candidates stay the same and only the condition changes,
    from the anchor's emotion to its art style. Success means the ranking flips the right way.
  - G3 succeeds in **25.4% / 26.6%** of held episodes (i2t / t2i). Naive succeeds in **18.8% / 20.3%**.
  - The difference is **+6.40 [+4.20, +8.54]** points pooled, and it is above 0 in each direction. All
    three seeds pass (+5.9 to +8.1).

**How to read the two results together.** Training cut G3's CLIP weight β from 0.30 to 0.05 and
concentrated its condition weights on about 4 factors instead of naive's 16. So G3 relies more on
the condition:
- A wrong condition hurts it more. Its wrong-condition R@1 is 10.45 / 11.57, against naive's 12.21 / 12.74.
- Its ranking changes more when the condition changes, and that is what the swap test rewards.
- Each half of the swap test barely moves, though. "The same-emotion item ranks above the same-style
  item under the emotion condition" rises only +2.4 / +0.3 points, and the style half +0.5 / +2.7.
  The +6.5 / +6.3 gain in the conjunction comes mostly from both halves holding in the same episode
  more often.

On held rows, the evidence therefore shows a scorer that is **more condition-sensitive** than naive.
It does not show a scorer that **ranks better** given the right condition. This reading is an
interpretation, not a separate test.

**The stage (e) decision is the user's.** This report states only which pre-registered criteria hold.

## Criterion 1 (primary): condition-use gain on the held label episodes, NOT MET

The criterion's definition:
- Δ = [R@1(G3) − R@1(G3 | wrong c)] − [R@1(naive) − R@1(naive | wrong c)], in R@1 points.
- It is pooled over 1,024 emotion and 1,024 art-style held episodes.
- The CI is a paired bootstrap over the 2,048 episodes (5,000 resamples, seed 42).
- **Met iff the CI's lower bound is > 0 in image→text AND in text→image, judged on the seed-42 model.**

| Model | Δ image→text | Δ text→image | Δ mean of directions | Same test passes? |
|---|---:|---:|---:|---|
| **G3 seed 42 (judged)** | **+0.83 [−1.12, +2.78]** | **+1.61 [−0.54, +3.81]** | +1.22 [−0.29, +2.76] | **no: NOT MET** |
| G3 seed 43 | +1.17 [−0.73, +3.12] | +1.37 [−0.73, +3.56] | +1.27 [−0.17, +2.78] | no |
| G3 seed 44 | +0.78 [−1.22, +2.78] | +2.05 [+0.00, +4.25] | +1.42 [−0.07, +2.91] | no |

Seed 44's text→image lower bound is exactly 0.00, which fails the strict "> 0" test.

**Per label type** (context, not criteria):

| Model | Emotion i2t | Emotion t2i | Emotion mean | Art style i2t | Art style t2i | Art style mean |
|---|---:|---:|---:|---:|---:|---:|
| G3 seed 42 | +0.20 [−2.54, +3.03] | −0.49 [−3.42, +2.44] | −0.15 [−2.29, +2.05] | +1.46 [−1.27, +4.10] | +3.71 [+0.68, +6.74] | +2.59 [+0.54, +4.69] |
| G3 seed 43 | +0.59 [−2.15, +3.42] | −0.88 [−3.91, +2.05] | −0.15 [−2.29, +2.05] | +1.76 [−0.98, +4.39] | +3.61 [+0.68, +6.64] | +2.69 [+0.73, +4.69] |
| G3 seed 44 | −0.39 [−3.22, +2.64] | −1.07 [−4.10, +1.86] | −0.73 [−2.93, +1.46] | +1.95 [−0.69, +4.59] | +5.18 [+2.25, +8.20] | +3.56 [+1.61, +5.57] |

**Own condition use and plain R@1 against naive** (pooled, paired, R@1 points):
- "Own use" is R@1(right c) − R@1(wrong c).
- "R@1 − naive" is the plain difference on the same episodes.

| Model | Own use, i2t | Own use, t2i | R@1 − naive, i2t | R@1 − naive, t2i |
|---|---:|---:|---:|---:|
| naive | +5.42 [+3.71, +7.13] | +8.15 [+6.25, +10.06] | 0 | 0 |
| G3 seed 42 | +6.25 [+4.44, +8.06] | +9.77 [+7.76, +11.77] | −0.93 [−2.44, +0.59] | +0.44 [−1.17, +2.10] |
| G3 seed 43 | +6.59 [+4.74, +8.45] | +9.52 [+7.42, +11.57] | −0.54 [−2.05, +0.98] | +0.29 [−1.32, +1.95] |
| G3 seed 44 | +6.20 [+4.35, +8.01] | +10.21 [+8.20, +12.26] | −0.68 [−2.20, +0.88] | +0.83 [−0.78, +2.49] |

G3 seed 42's R@1 − naive per label type:
- emotion: −0.59 [−2.83, +1.76] (i2t) and −1.07 [−3.22, +0.98] (t2i);
- art style: −1.27 [−3.32, +0.68] (i2t) and +1.95 [−0.49, +4.49] (t2i).

Only seed 44's art-style text→image difference, +3.03 [+0.68, +5.47], has a CI above 0.

## Criterion 2: the human swap test on held rows, MET

**The episodes.** There are 1,024 new held episodes from `build_human_swap_episodes` (seed 42),
covering 8 anchor emotions and 24 anchor styles. Each has one anchor, two conditions and one pool of
13 candidates:
- **c_emo:** 4 supports with the anchor's emotion and a different style, and 4 contrasts from paintings
  with no annotation of that emotion.
- **c_style:** 4 supports of the anchor's style from paintings with no annotation of the anchor's
  emotion, and 4 contrasts of other styles.
- **The pool:**
  - `p_emo`: same emotion, different style;
  - `p_style`: same style, from a painting with no annotation of the anchor's emotion;
  - 11 clean negatives.

**Success** means `p_emo` ranks above `p_style` under c_emo AND `p_style` ranks above `p_emo` under
c_style. **The criterion is met iff G3's success rate minus naive's has a pooled paired-bootstrap 95%
CI above 0.**

| Model | Success i2t | Success t2i | Diff vs naive, pooled | Diff i2t | Diff t2i | Passes? |
|---|---:|---:|---:|---:|---:|---|
| naive | 18.85% (193) | 20.31% (208) | — | — | — | — |
| **G3 seed 42 (judged)** | **25.39% (260)** | **26.56% (272)** | **+6.40 [+4.20, +8.54]** | +6.54 [+3.71, +9.48] | +6.25 [+3.42, +9.08] | **yes: MET** |
| G3 seed 43 | 26.76% | 28.61% | +8.11 [+6.01, +10.16] | +7.91 [+4.98, +10.84] | +8.30 [+5.56, +11.04] | yes |
| G3 seed 44 | 25.20% | 25.78% | +5.91 [+3.81, +7.96] | +6.35 [+3.52, +9.28] | +5.47 [+2.54, +8.30] | yes |
| CLIP-only (β 0.3) | 0% | 0% | | | | |
| uniform (β 0.3) | 0% | 0% | | | | |

CLIP-only and uniform weights ignore the condition, so they score 0% by construction: the same
ranking cannot put each item above the other.

**The two halves of the swap test** are descriptive. Each cell is the share of episodes; the "both"
column is the success rate.

| Scorer | i2t: emotion half | i2t: style half | i2t: both | t2i: emotion half | t2i: style half | t2i: both |
|---|---:|---:|---:|---:|---:|---:|
| naive | 53.12 | 58.50 | 18.85 | 53.32 | 58.50 | 20.31 |
| G3 seed 42 | 55.57 | 58.98 | 25.39 | 53.61 | 61.23 | 26.56 |
| G3 seed 43 | 56.15 | 59.57 | 26.76 | 53.71 | 62.50 | 28.61 |
| G3 seed 44 | 55.57 | 58.98 | 25.20 | 52.34 | 60.64 | 25.78 |
| CLIP-only | 47.46 | 52.54 | 0.00 | 49.02 | 50.98 | 0.00 |
| uniform | 46.97 | 53.03 | 0.00 | 48.54 | 51.46 | 0.00 |

- The emotion half is "`p_emo` ranks above `p_style` under c_emo"; the style half is the reverse
  under c_style.
- Naive and G3 get each half right in 52-63% of episodes, only a little more often than a coin flip.
- For naive, the two halves rarely hold together, because its rankings under the two conditions are
  similar. G3's rankings differ more between conditions, so the conjunction rises.

## Replication: seeds 43 and 44

Both seeds use G3's recipe with only the seed changed: CLIP-cluster conditions, no swap term,
`ScorerTrainingConfig()` defaults otherwise. They were trained on the same CLIP-cluster source; a
k-means refit on scorer-train rows reproduces Task 6's cached labels exactly.

**On the selection set,** scored as in Task 6 Step 4 (Task 6's selection episodes, SHA-256s asserted):

| Model | Selection score (mean) | i2t | t2i | Emotion mean | Art style mean | β | R@1 i2t / t2i |
|---|---:|---:|---:|---:|---:|---:|---|
| naive | 0 | 0 | 0 | 0 | 0 | 0.300 | 18.36 / 20.36 |
| G3 seed 42 (Task 6) | +2.54 [+1.44, +3.67] | +1.83 [+0.32, +3.32] | +3.25 [+1.76, +4.79] | +0.98 [−0.54, +2.51] | +4.10 [+2.44, +5.76] | 0.050 | 18.87 / 21.66 |
| G3 seed 43 | +3.00 [+1.95, +4.08] | +2.49 [+1.07, +3.96] | +3.52 [+2.10, +4.93] | +1.17 [−0.27, +2.66] | +4.83 [+3.22, +6.45] | 0.042 | 19.19 / 21.73 |
| G3 seed 44 | +2.32 [+1.26, +3.41] | +1.83 [+0.34, +3.32] | +2.81 [+1.39, +4.27] | +0.46 [−1.03, +2.00] | +4.17 [+2.56, +5.76] | 0.056 | 18.68 / 21.29 |

- **The recipe is stable across seeds.** All three clear the selection bar, with both directions'
  CIs above 0. All three drive β from 0.30 to 0.04-0.06 over the same trajectory: about 0.20 at step
  500, 0.14 at step 1,000, 0.08 at step 2,000, and 0.04-0.06 at step 3,000.
- **Seed 42 reproduces Task 6 exactly.** Re-scored here, its selection gain equals Task 6's recorded
  value bit for bit in every scope.
- **On held rows, all three seeds lose the same way.** Δ shrinks to +0.8 to +1.2 in i2t and +1.4 to
  +2.1 in t2i, and no seed has both CIs above 0. The held miss is therefore not a seed-42 accident.

## Context: held R@1 and R@3 on the same label episodes

Values are percentages (i2t / t2i). "Wrong c" scores the same episodes with the deranged condition.
Chance is 1/13 = 7.7%.

| Model | R@1 | R@3 | R@1, wrong c |
|---|---|---|---|
| naive (step 0, β 0.3) | 17.63 / 20.90 | 40.09 / 44.53 | 12.21 / 12.74 |
| **G3 seed 42** | **16.70 / 21.34** | 38.67 / 45.75 | 10.45 / 11.57 |
| G3 seed 43 | 17.09 / 21.19 | 39.75 / 44.73 | 10.50 / 11.67 |
| G3 seed 44 | 16.94 / 21.73 | 38.43 / 45.75 | 10.74 / 11.52 |
| CLIP-only (zero weights, β 0.3) | 11.52 / 14.89 | 29.05 / 37.16 | = R@1 |
| uniform (1/32, β 0.3) | 15.43 / 16.26 | 34.42 / 38.96 | = R@1 |

| Model | Emotion R@1 | Art style R@1 |
|---|---|---|
| naive | 15.23 / 16.21 | 20.02 / 25.59 |
| G3 seed 42 | 14.65 / 15.14 | 18.75 / 27.54 |
| G3 seed 43 | 15.33 / 14.65 | 18.85 / 27.73 |
| G3 seed 44 | 14.36 / 14.84 | 19.53 / 28.61 |
| CLIP-only | 9.67 / 12.30 | 13.38 / 17.48 |
| uniform | 12.60 / 13.77 | 18.26 / 18.75 |

G3 seed 42 minus the condition-blind rows (pooled, paired):
- G3 − CLIP-only: +5.18 [+3.32, +7.03] (i2t) and +6.45 [+4.49, +8.35] (t2i);
- G3 − uniform: +1.27 [−0.59, +3.12] (i2t) and +5.08 [+3.17, +7.03] (t2i).

**R3 naive's numbers from the repair plan's held evaluation, cited and not recomputed**
([2026-10-12 report](2026-10-12_candidate_a_condition_eval_repaired_factors.md), on the same 2,048
held label episodes):
- R3 naive (β 0.3): pooled **17.6 / 20.9**; emotion 15.2 / 16.2; art style 20.0 / 25.6.
- R3 uniform at its own β (0.1): 15.6 / 15.7.
- CLIP alone: 11.5 / 14.9.
- R3 naive − R3 uniform: +2.0 [+0.5, +3.6] / +5.2 [+3.5, +6.8].
- The post-hoc interaction there showed better use of the condition than R0 in text→image only.

**Consistency.** The naive baseline used here is the step-0 model. Its held R@1 and R@3 equal the 2026-10-12 report's
recorded R3 naive values in every cell (pooled and per label type, both directions; max |diff| 0).
CLIP-only equals that report's CLIP-only row in every cell as well.

## Learned β and the condition weights on the held label episodes

| Model | β | τ | Mean active factors (of 32) | All-zero weight rows | Mean largest weight |
|---|---:|---:|---:|---:|---:|
| naive | 0.300 | 0.094 | 15.56 | 0.0% | 0.206 |
| G3 seed 42 | 0.050 | 0.241 | 3.68 | 2.1% | 0.603 |
| G3 seed 43 | 0.042 | 0.239 | 5.63 | 0.1% | 0.521 |
| G3 seed 44 | 0.056 | 0.242 | 4.33 | 0.9% | 0.556 |

τ is the logit temperature and does not change rankings.

## What was run

- **Script:** `src/test/20261014_stage_d_final/run_final.py`. It imports Task 6's `run_selection.py` and
  reuses its cache, source rebuild, masking and helpers rather than copying them.
- **Rebuild checks (`--prepare`),** all asserted:
  - the split (216,107 / 30,872 / 61,744), the leakage groups, scorer-train (183,694 rows) and
    selection (32,413 rows) equal Task 6's cache;
  - the R3 SHA-256 (`1c299fc0…e453f`) and config are checked;
  - re-encoded train-part codes and `factor_scale` are bit-identical to Task 6's;
  - the CLIP k-means refit on scorer-train rows equals Task 6's labels;
  - the G3 checkpoint SHA-256 is `8d9317c2…fd63`.
- **Seeds 43 and 44** ran as two parallel processes on one RTX 3090, 2.9 min each, with finite
  histories. Checkpoint SHA-256s: `6bce4fc8…4081` and `5d6021ee…f408`.
- **Naive** is built exactly as in Task 6: the step-0 `train_scorer` with β 0.3. Its τ equals Task
  6's, and its weights `torch.equal` `label_episode_weights` on the first 16 held episodes of each
  label. Its ranks equal `label_episode_recall(rule, β 0.3)` on 100% of held episodes.
- **Held label episodes** are `standard_label_episodes(data, groups, split.held, label, 1024, seed=42)`:
  - The SHA-256s equal the recorded ones: emotion `e62ab41f…8c85` and art style `3a58cf9d…422e167f`
    (asserted).
  - There are 8 emotion targets and 24 art-style targets.
  - The wrong condition is `wrong_condition(seed=42)`, applied per label type. Δ is pooled by
    concatenating the two label types' episodes.
- **Human swap episodes** were checked in the script:
  - every row is a held row;
  - no painting (leakage group) repeats inside an episode;
  - every support, contrast, `p_emo`, `p_style` and negative is one-aspect clean as defined.
  - SHA-256 `f1195dcc…0062`.
- **Row-scope guards.** Every input outside the rows a phase may read is NaN. Training reads
  scorer-train rows only, the selection phase reads the train part, and the final phase reads held rows
  only. The held phase ran **once** (23 s) and refuses to run a second time.
- **Smoke run.** Before the held run, a smoke run exercised the whole final code path with selection
  rows in place of held rows. Its numbers were discarded.

## Caveats

- **The β collapse, and what it means.** G3 cut β from 0.30 to 0.050 (seeds 43/44: 0.042, 0.056). It
  also concentrated its weights on 3.7-5.6 active factors, against naive's 15.6.
  - Δ rewards a wrong condition hurting more. Part of any Δ therefore means "leans harder on the
    condition" rather than "ranks better". On held, G3's wrong-condition R@1 falls 1.2-1.8 points below
    naive's, while its right-condition R@1 does not rise (−0.93 / +0.44).
  - The swap test rewards a ranking that changes with the condition. A lower β makes that easier
    without better one-sided judgments, as the two-halves table shows.
  - Criterion 2's pass is real by its definition. It is best read as "more condition-sensitive than
    naive", not as "understands the condition better".
  - Task 6 attributed the collapse, as an untested interpretation, to CLIP-nearest hard negatives in
    training episodes, where the CLIP term misleads. β had not converged at 3,000 steps in any seed.
- **Same-label wrong conditions.** `wrong_condition` sometimes hands an episode the condition of
  another episode with the same target label. That "wrong" condition is effectively right.
  - Measured on the held episodes: **12.9%** of emotion episodes and **4.6%** of art-style episodes
    (8.7% pooled).
  - This attenuates Δ toward 0 for G3 and naive alike.
  - Rescaling by 1/(1 − 0.087) as a rough correction leaves both criterion-1 lower bounds below 0, at
    about −1.2 and −0.6. So this effect does not explain the miss.
- **Selection set vs held** (spec §10).
  - The selection set contains paintings whose features R3's encoders saw, never their labels. That
    was acceptable for choosing a run but not for final numbers.
  - The held Δ is about half the selection Δ in both directions. Part of the drop may be this
    in-sample effect, and part may be ordinary selection optimism. G3 was one of five runs, though
    not the top scorer.
- **Emotion labels are per annotation** (spec §10). Clean negatives mitigate, but do not remove, the
  ambiguity. Emotion's Δ is about 0 on held, as it was within noise on the selection set.
- **Label coverage** (spec §10). Only two human condition types exist. Neither result says anything yet
  about arbitrary stated conditions; that is stage (e)'s job.
- **Source confound** (spec §10). The selected source, CLIP clusters, offers 128 distinct training
  conditions. This limits what G3's result says about self-generated conditions in general.
- **A single held split** (spec §10).
  - These 2,048 held label episodes were used before: by the repair plan's Task 7 (the 2026-10-12
    report) and by its final review's post-hoc re-scoring. Those uses measured R0/R3 naive, uniform and
    CLIP-only only.
  - No stage-(d) choice used held rows. The human swap episodes are new.
  - The bootstrap CIs resample episodes under fixed models. They do not cover the choice of split.
- **The tie-break chose G3 over G5.** On the selection set the two could not be told apart: G5 − G3 =
  +0.46 [−0.40, +1.32]. The rule picked G3 by its pre-registered tie-break, not by a measured
  advantage.
  - G5 (communities) was not evaluated on held, and this report says nothing about how it would have
    done.
  - Evaluating it now would be a post-hoc use of held rows.
- **Training randomness.** Seeds 43 and 44 cover it only partly: three seeds, one split. GPU training is
  not bit-deterministic (about 1e-6, per Task 6). All three seeds agree on both verdicts.
- **Criterion 2's reference point.** Condition-blind scorers (CLIP-only, uniform) score 0% on the swap
  test by construction. Naive's 18.8-20.3% is the only meaningful baseline, and the test compares
  against it.

## Files

- **Script:** `src/test/20261014_stage_d_final/run_final.py`. `--tables` reprints every table above
  from the saved JSON.
- **Log:** `src/test/20261014_stage_d_final/20261014_stage_d_final_log.md`.
- **Gitignored, local only:**
  - `cache/prepare_checks.json` and `cache/held_codes.npz`;
  - `checkpoints/G3_seed{43,44}.pt`;
  - `results/history_G3_seed{43,44}.json`, `results/selection_seeds.json`, `results/final_results.json`,
    `results/final_ranks.npz`, and the discarded `results/smoke_final*.{json,npz}`;
  - `run_*.log`.
- **G3 seed 42** is Task 6's `src/test/20261013_stage_d_selection/checkpoints/G3.pt`.

**The stage (e) decision is the user's.**
