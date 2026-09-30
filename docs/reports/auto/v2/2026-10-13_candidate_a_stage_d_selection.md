# CoSiR v2 Candidate A: stage (d) selection, self-generated-condition scorers vs the naive rule

## Verdict

**A trained scorer uses the condition better than the naive rule on the selection set when its
training conditions come from CLIP clusters or from Stage-1 communities.** Three runs clear the bar:
G3, G4 and G5 gain +2.4 to +3.0 R@1 points over naive, with 95% CIs above 0 in both retrieval
directions. **Factor-combination conditions do not work** (G1, G2). **The swap term does not help.**
The pre-registered rule selects **G3 (CLIP clusters, no swap)**. The stop point (best score ≤ +0.5)
is not reached, so the final held-out test (Task 7) can go ahead.

What the numbers say, in plain terms:

- **Which condition source works.**
  - CLIP clusters (G3 +2.54, G4 +2.36) and communities (G5 +3.00) all beat naive at using the
    condition.
  - Factor combinations do not: G1 scores +0.46 and G2 +0.79, and both CIs span 0. They also rank
    worse than naive overall, by 1.3 to 1.6 R@1 points.
  - Factor-combination conditions are defined in the factor space itself, so they are easy to fit.
    G1's training loss falls from 0.69 to 0.13, yet what it learns does not carry over to human
    conditions.
- **Whether the swap term helps.** It does not. Within each source, the paired difference is small
  and its CI spans 0:
  - factor combinations, G2 − G1: +0.33 [−0.24, +0.85];
  - CLIP clusters, G4 − G3: −0.18 [−0.59, +0.23].
  - For CLIP clusters the swap loss did not move at all (1.45 → 1.46). The scorer could not learn to
    separate an anchor's image cluster from its caption cluster in R3 space.
- **Whether the scorer uses the condition better than naive.** Yes on this set, in both directions.
  For G3, the selected run: **+1.83 [+0.32, +3.32] in image→text and +3.25 [+1.76, +4.79] in
  text→image.**
  - Most of the gain comes from art-style conditions: +4.1 to +4.7 points for G3-G5.
  - For emotion the gain is smaller (+0.6 to +1.3), and every CI touches or spans 0.
- **How the scorers get there.** Every run did the same two things:
  - It cut the CLIP weight β from 0.30 to 0.04-0.05, 6-7.5× less.
  - It concentrated the condition weights on fewer factors: 3.7-7.5 active factors instead of naive's
    15.7.
  - As a result, a wrong condition hurts more: wrong-condition R@1 is 1.3-2.3 points below naive's
    (1.3-2.0 for G3-G5).
  - The right condition also helps somewhat more. G3's R@1 rises +0.51 [−0.66, +1.68] in i2t (CI
    spans 0) and +1.29 [+0.12, +2.47] in t2i. G5's rises +1.32 [+0.34, +2.32] and +1.29 [+0.27, +2.34].
  - So part of the gain means "leans harder on the condition" rather than "ranks better". Both are
    reported below.
- **Why G3 and not G5.** G5 has the highest score. G3 and G4 lie within 1.0 point of it, so the three
  tie. The pre-registered tie-break prefers a no-swap run (G3 or G5), then the earlier run in the
  table, which is G3. The two cannot be told apart: G5 − G3 = +0.46 [−0.40, +1.32].
- **Headroom.** The ceiling (oracle per-episode weights on the frozen factors) reaches 84.7 / 86.7 R@1,
  against naive's 18.4 / 20.4. The oracle is fitted to the answer, so this is an optimistic bound. Even
  so, it says the frozen factors are not the limit yet.

This is the **selection** set. R3's encoders saw these paintings' features during training (never
their labels), so these numbers are for choosing a run, not final results. The final, pre-registered
test is Task 7 on the untouched held split.

## What was run

- **Data and split.**
  - ArtELingo with the painting-grouped split (seed 42): 216,107 / 30,872 / 61,744 rows (asserted).
  - The train part was split again by leakage group (seed 42):
    - scorer-train: 183,694 rows, 36,518 groups;
    - selection: 32,413 rows, 6,451 groups (15.0%).
  - Val and held rows were not used. Mechanical guards NaN out every row a phase may not read.
- **Frozen factors.** The R3 checkpoint (SHA-256 `1c299fc0…e453f`, asserted), encoded once; codes
  finite.
- **Condition sources,** fitted on scorer-train rows only:
  - **Factor combinations:** a group is the top 10% of 1-3 Dirichlet-weighted factors (18,370 rows).
    Effectively unlimited distinct conditions; all 1,000 test draws were valid.
  - **CLIP clusters:** k-means with k=64 on image features and, separately, on caption features. 128
    groups, all valid, of 635-7,431 rows.
  - **Communities:** content graph (1.87M edges) → Stage 1 → Leiden. 19 communities, all valid, of
    2,875-17,410 rows.
- **Training.** `ScorerTrainingConfig()` defaults exactly: 3,000 steps, 64 fresh episodes per step
  (plus 64 swap episodes for swap runs), Adam at lr 1e-3, seed 42, no early stopping.
  - The five runs ran as parallel processes on one RTX 3090.
  - Train times: G1 13.1 min, G2 31.5, G3 5.6, G4 14.9, G5 5.4.
- **Naive.** The step-0 model (β = 0.3). Its weights equal `label_episode_weights` under
  `torch.equal` on 16 + 16 selection episodes, and its ranks equal the plain rule's on 100% of
  episodes.
- **Selection episodes.** `standard_label_episodes` on selection rows, seed 42:
  - 2,048 emotion episodes over 8 targets;
  - 2,048 art-style episodes over 23 targets.
  - Each episode has 13 candidates, so chance R@1 is 7.7%.
- **Wrong condition.** `wrong_condition(seed=42)`, a derangement within each label type's episodes.
  Δ is pooled by concatenating the two label types' episodes.
- **Selection score.** Δ = [R@1(run) − R@1(run | wrong c)] − [R@1(naive) − R@1(naive | wrong c)],
  averaged over i2t and t2i. Units are R@1 points, with a paired bootstrap over episodes (5,000
  resamples, seed 42).

## Selection score and the rule

Condition-use gain over naive, pooled over emotion and art style (R@1 points, 95% CI):

| Run | Source | Swap | **Selection score** (mean of directions) | image→text | text→image |
|---|---|---|---:|---:|---:|
| G1 | factor combinations | no | +0.46 [−0.54, +1.51] | +0.54 [−0.90, +1.95] | +0.39 [−1.00, +1.76] |
| G2 | factor combinations | yes | +0.79 [−0.23, +1.81] | +0.61 [−0.76, +2.00] | +0.98 [−0.37, +2.34] |
| G3 | CLIP clusters | no | **+2.54 [+1.44, +3.67]** | +1.83 [+0.32, +3.32] | +3.25 [+1.76, +4.79] |
| G4 | CLIP clusters | yes | +2.36 [+1.29, +3.49] | +2.08 [+0.59, +3.59] | +2.64 [+1.12, +4.15] |
| G5 | communities | no | **+3.00 [+2.05, +3.99]** | +2.69 [+1.34, +4.03] | +3.32 [+2.00, +4.69] |

**The pre-registered rule, step by step:**
1. **Highest score:** G5, at +3.00.
2. **Ties** (within 1.0 point of the best, so ≥ +2.00): G3 (+2.54), G4 (+2.36) and G5.
3. **Tie-break:** prefer a no-swap run, leaving G3 and G5. Then take the earlier run in the table:
   **G3**.
4. **Stop point:** the best score +3.00 is above +0.5, so there is no stop.

**Selected: G3** (CLIP clusters, no swap, seed 42).

### Per label type

| Run | Emotion: mean | Emotion: i2t | Emotion: t2i | Art style: mean | Art style: i2t | Art style: t2i |
|---|---:|---:|---:|---:|---:|---:|
| G1 | −1.03 [−2.39, +0.39] | −0.83 [−2.78, +1.12] | −1.22 [−3.08, +0.63] | +1.95 [+0.44, +3.47] | +1.90 [−0.15, +3.96] | +2.00 [−0.05, +4.05] |
| G2 | −0.71 [−2.05, +0.63] | −0.83 [−2.73, +1.03] | −0.59 [−2.34, +1.17] | +2.29 [+0.81, +3.78] | +2.05 [+0.10, +4.10] | +2.54 [+0.54, +4.59] |
| G3 | +0.98 [−0.54, +2.51] | +0.88 [−1.22, +2.98] | +1.07 [−0.78, +3.08] | +4.10 [+2.44, +5.76] | +2.78 [+0.59, +4.98] | +5.42 [+3.17, +7.67] |
| G4 | +0.56 [−0.98, +2.12] | +0.68 [−1.51, +2.73] | +0.44 [−1.42, +2.44] | +4.15 [+2.54, +5.76] | +3.47 [+1.37, +5.62] | +4.83 [+2.64, +7.03] |
| G5 | +1.29 [−0.02, +2.71] | +0.98 [−0.98, +2.83] | +1.61 [+0.00, +3.37] | +4.71 [+3.27, +6.18] | +4.39 [+2.39, +6.35] | +5.03 [+2.98, +7.03] |

### Swap term and tie-break contrasts

These are descriptive paired differences in condition-use gain. They are **not** part of the rule,
and were added after the first evaluation; a re-run reproduced every rule number bit for bit.

| Contrast | Pooled, mean | Pooled, i2t | Pooled, t2i | Emotion, mean | Art style, mean |
|---|---:|---:|---:|---:|---:|
| G2 − G1 (swap term, factor combinations) | +0.33 [−0.24, +0.85] | +0.07 [−0.68, +0.81] | +0.59 [−0.17, +1.34] | +0.32 [−0.46, +1.12] | +0.34 [−0.42, +1.07] |
| G4 − G3 (swap term, CLIP clusters) | −0.18 [−0.59, +0.23] | +0.24 [−0.29, +0.78] | −0.61 [−1.22, +0.02] | −0.42 [−1.00, +0.17] | +0.05 [−0.56, +0.66] |
| G5 − G3 (communities vs CLIP clusters) | +0.46 [−0.40, +1.32] | +0.85 [−0.34, +2.00] | +0.07 [−1.10, +1.22] | +0.32 [−0.95, +1.54] | +0.61 [−0.59, +1.81] |

## R@1 and R@3

Values are percentages (image→text / text→image). The "wrong" columns score the same episodes under
the deranged condition.

- CLIP-only uses all-zero condition weights; uniform uses weights of 1/32. Both use β = 0.3 and
  ignore the condition, so their wrong-condition R@1 equals their right-condition R@1.
- The ceiling uses per-episode oracle weights, optimized on the episode's own positive (β = 0.3).
- Chance is 7.7%.

**Pooled (4,096 episodes)**

| Model | R@1 i2t | R@1 t2i | R@3 i2t | R@3 t2i | R@1 wrong c, i2t | R@1 wrong c, t2i |
|---|---:|---:|---:|---:|---:|---:|
| naive (step 0, β 0.3) | 18.36 | 20.36 | 40.75 | 42.90 | 13.18 | 13.16 |
| G1 | 16.85 | 18.75 | 39.11 | 40.77 | 11.13 | 11.16 |
| G2 | 17.09 | 19.09 | 39.18 | 40.87 | 11.30 | 10.91 |
| **G3 (selected)** | **18.87** | **21.66** | 40.43 | 43.58 | 11.87 | 11.21 |
| G4 | 18.75 | 21.12 | 40.45 | 43.33 | 11.50 | 11.28 |
| G5 | 19.68 | 21.66 | 41.67 | 44.07 | 11.82 | 11.13 |
| CLIP-only | 12.18 | 14.67 | 31.01 | 34.72 | 12.18 | 14.67 |
| uniform | 14.62 | 15.53 | 35.21 | 37.18 | 14.62 | 15.53 |
| ceiling (diagnostic) | 84.72 | 86.65 | 91.33 | 92.26 | n/a | n/a |

**Emotion (2,048 episodes)**

| Model | R@1 i2t | R@1 t2i | R@3 i2t | R@3 t2i | R@1 wrong c, i2t | R@1 wrong c, t2i |
|---|---:|---:|---:|---:|---:|---:|
| naive | 16.50 | 15.14 | 35.64 | 35.60 | 10.64 | 11.08 |
| G1 | 15.09 | 12.94 | 34.72 | 32.96 | 10.06 | 10.11 |
| G2 | 15.19 | 13.33 | 35.30 | 33.06 | 10.16 | 9.86 |
| G3 | 17.09 | 14.79 | 36.52 | 33.64 | 10.35 | 9.67 |
| G4 | 16.65 | 14.45 | 35.94 | 33.54 | 10.11 | 9.96 |
| G5 | 17.48 | 15.43 | 37.26 | 34.47 | 10.64 | 9.77 |
| CLIP-only | 9.57 | 11.77 | 25.59 | 29.25 | 9.57 | 11.77 |
| uniform | 10.94 | 12.30 | 28.91 | 31.59 | 10.94 | 12.30 |
| ceiling | 81.98 | 84.23 | 89.79 | 90.43 | n/a | n/a |

**Art style (2,048 episodes)**

| Model | R@1 i2t | R@1 t2i | R@3 i2t | R@3 t2i | R@1 wrong c, i2t | R@1 wrong c, t2i |
|---|---:|---:|---:|---:|---:|---:|
| naive | 20.21 | 25.59 | 45.85 | 50.20 | 15.72 | 15.23 |
| G1 | 18.60 | 24.56 | 43.51 | 48.58 | 12.21 | 12.21 |
| G2 | 18.99 | 24.85 | 43.07 | 48.68 | 12.45 | 11.96 |
| G3 | 20.65 | 28.52 | 44.34 | 53.52 | 13.38 | 12.74 |
| G4 | 20.85 | 27.78 | 44.97 | 53.12 | 12.89 | 12.60 |
| G5 | 21.88 | 27.88 | 46.09 | 53.66 | 12.99 | 12.50 |
| CLIP-only | 14.79 | 17.58 | 36.43 | 40.19 | 14.79 | 17.58 |
| uniform | 18.31 | 18.75 | 41.50 | 42.77 | 18.31 | 18.75 |
| ceiling | 87.45 | 89.06 | 92.87 | 94.09 | n/a | n/a |

**Condition use and R@1 against naive**

Both columns are pooled paired differences in R@1 points (95% CI):
- "Own condition use" is R@1(right c) − R@1(wrong c).
- "R@1 − naive" is the plain R@1 difference on the same episodes.

| Model | own use, i2t | own use, t2i | R@1 − naive, i2t | R@1 − naive, t2i |
|---|---:|---:|---:|---:|
| naive | +5.18 [+3.88, +6.47] | +7.20 [+5.88, +8.52] | 0 | 0 |
| G1 | +5.71 [+4.37, +7.08] | +7.59 [+6.20, +9.03] | −1.51 [−2.59, −0.39] | −1.61 [−2.71, −0.51] |
| G2 | +5.79 [+4.39, +7.18] | +8.18 [+6.76, +9.62] | −1.27 [−2.32, −0.20] | −1.27 [−2.34, −0.22] |
| G3 | +7.01 [+5.64, +8.42] | +10.45 [+9.01, +11.89] | +0.51 [−0.66, +1.68] | +1.29 [+0.12, +2.47] |
| G4 | +7.25 [+5.91, +8.67] | +9.84 [+8.40, +11.25] | +0.39 [−0.76, +1.56] | +0.76 [−0.42, +1.93] |
| G5 | +7.86 [+6.42, +9.30] | +10.52 [+9.11, +11.96] | +1.32 [+0.34, +2.32] | +1.29 [+0.27, +2.34] |

Per label type, G3's R@1 − naive is:
- emotion: +0.59 [−1.07, +2.20] (i2t) and −0.34 [−1.90, +1.22] (t2i);
- art style: +0.44 [−1.17, +1.95] (i2t) and +2.93 [+1.22, +4.59] (t2i).

## Learned β and τ, and the condition weights

The weight columns are measured on the 4,096 selection episodes. "Active factors" counts factors with
a weight above 0.

| Model | β | τ | mean active factors (of 32) | all-zero weight rows | mean largest weight |
|---|---:|---:|---:|---:|---:|
| naive | 0.3000 | 0.0936 | 15.66 | 0.0% | 0.205 |
| G1 | 0.0463 | 0.0327 | 3.82 | 1.1% | 0.587 |
| G2 | 0.0403 | 0.0340 | 5.11 | 0.2% | 0.510 |
| G3 | 0.0496 | 0.2405 | 3.67 | 2.0% | 0.603 |
| G4 | 0.0497 | 0.2496 | 4.08 | 1.4% | 0.575 |
| G5 | 0.0402 | 0.1422 | 7.47 | 0.0% | 0.372 |

τ is the logit temperature. It does not change rankings, and naive's τ comes only from its first
mined batch.

**β was still falling when training stopped.** In every run it went from 0.30 to 0.20-0.22 at step
500, 0.13-0.16 at step 1,000 and 0.07-0.08 at step 2,000, ending at 0.04-0.05 at step 3,000. The
recipe fixes 3,000 steps with no early stopping, so this is the pre-registered model.

## Loss curves

Each logged value is the loss on one fresh batch of 64 mined episodes, so single values are noisy.
The table gives the step-0 value and block means of the logged points (every 100 steps).

| Run | Minutes | Step 0 | Steps 100-900 | Steps 1,000-1,900 | Steps 2,000-2,999 | Swap loss: step 0 → steps 2,000-2,999 |
|---|---:|---:|---:|---:|---:|---|
| G1 | 13.1 | 0.689 | 0.217 | 0.161 | 0.132 | n/a |
| G2 | 31.5 | 0.689 | 0.212 | 0.130 | 0.129 | 0.904 → 0.239 |
| G3 | 5.6 | 1.228 | 1.102 | 1.197 | 1.123 | n/a |
| G4 | 14.9 | 1.228 | 1.173 | 1.141 | 1.155 | 1.454 → 1.464 (flat) |
| G5 | 5.4 | 1.106 | 1.050 | 1.068 | 1.036 | n/a |

The columns are the multi-positive ranking loss. For swap runs the total loss is ranking + swap.
Runs on the same source share their step-0 batch, so G1 and G2 match at step 0, as do G3 and G4.

The factor-combination runs fit their own training task well. The CLIP-cluster and community runs
barely lower their training loss after the first 100 steps. How well a run fits its training task
does not predict transfer to human conditions: the source that fits best (factor combinations)
transfers worst.

## Caveats

- **The selection set is in-sample for R3's features** (spec §10). R3's encoders saw these paintings'
  features, though never their labels. This affects every run and naive alike, so it is fine for
  choosing a run, but these are not final numbers. Task 7's held test is the pre-registered result.
- **Sources differ in how many distinct conditions they offer** (spec §3 and §10). Factor
  combinations are effectively unlimited (1,000/1,000 random draws were valid). CLIP clusters offer
  128 (64 image + 64 caption). Communities offer 19. The source comparison therefore mixes "what kind
  of condition" with "how many distinct conditions".
- **Emotion is per annotation** (spec §10). Clean negatives mitigate the ambiguity but do not remove
  it. This may be one reason every emotion-only gain has a CI that touches or spans 0.
- **Only two human condition types** (spec §10). Success on emotion and art style says nothing yet
  about arbitrary stated conditions; that is stage (e)'s job.
- **Same-label wrong conditions.** `wrong_condition` hands an episode another episode's supports and
  contrasts, and sometimes that episode has the same target label. Measured by tracing the
  derangement:
  - 11.9% of emotion episodes;
  - 4.1% of art-style episodes.
  - Those episodes get a "wrong" condition that is effectively right. This attenuates Δ toward 0 for
    every run and for naive alike. It does not change the ranking of runs or the sign of any gain.
- **Part of the gain is heavier reliance on the condition, not better ranking.** All five runs cut β
  6-7.5× and sparsified the weights. That makes the wrong condition hurt more, and Δ rewards it.
  So we report plain R@1 − naive too: G3 is +0.51 (CI spans 0) in i2t and +1.29 (CI above 0) in t2i.
- **A likely cause of the β collapse** (an interpretation, not tested): half of each training
  episode's negatives are the CLIP-nearest outside items. On training episodes the CLIP term actively
  misleads, so training pushes β down. On human-label episodes the negatives are random and CLIP
  helps: CLIP-only gets 12.2 / 14.7 against 7.7% chance. β had not converged at 3,000 steps.
- **The swap term had no usable signal for CLIP clusters:** the swap loss stayed flat. For factor
  combinations it learned the swap task (0.90 → 0.24), but that did not transfer.
- **Single seed, single selection set.** Task 7 retrains G3 with seeds 43 and 44. GPU training is not
  bit-deterministic (about 1e-6 differences). G5 cannot be told apart from G3 here (+0.46
  [−0.40, +1.32]); the rule chose G3 by its tie-break, not by a measured advantage.
- **Pilot runs.** A 30-step timing pilot and a code-path smoke test ran before the real runs. Their
  numbers were discarded and informed no choice.

## Files

- Script: `src/test/20261013_stage_d_selection/run_selection.py`. `--tables` reprints every table
  above from `results/selection_results.json`.
- Log: `src/test/20261013_stage_d_selection/20261013_stage_d_selection_log.md`.
- Gitignored, local only:
  - `cache/prepare.{npz,json}` (split, codes, source labels);
  - `checkpoints/G{1..5}.pt`;
  - `results/history_G{k}.json`, `results/selection_results.json`, `results/selection_ranks.npz`;
  - `run_*.log`.
- Selection episode SHA-256s:
  - emotion `8405632159883eebd627401724895e67cebca9f8326139aa2c84058033439ea0`;
  - art style `e1cfe1ba301ece6a4e3902ea75f2d57a78de896103a2c60be218f8c64cca4c46`.
- G3 checkpoint SHA-256: `8d9317c29a224f4cba8c5c1eca1e0bb9c59cf8877a74a673626182ed53fcfd63`.
