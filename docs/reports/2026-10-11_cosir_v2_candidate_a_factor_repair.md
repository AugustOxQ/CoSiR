# CoSiR v2 Candidate A: factor-repair grid, pre-registered gates and goal-based selection

## Verdict

**No configuration passes all nine gates on `val`. Under the pre-registered rule (Step 4.4) no recipe is selected, and the task stops here for the user to decide.** No runs were added and no thresholds were changed. Three things were therefore not run:

- seed replication (Step 5);
- the held check (Step 6); held rows were never encoded;
- the two checkpoints for Task 7.

There is no selected `FactorTrainingConfig`.

What the nine runs show, in plain language:

- **The collapse itself is repaired.** Two mechanisms repair it: InfoNCE agreement in place of the cosine term (R1, R3, R6, R7), and removing the agreement term altogether (R8).
  - The smaller participation ratio goes from 1.31 (R0) to 17-22.
  - Max |r| goes from 0.9993 to 0.43-0.80, with no factor pair at or above 0.9.
  - The first principal component's share of code variance drops from 0.86-0.87 to 0.10-0.13.
  - Decorrelation on top of the cosine term (R2) only partly lifts the collapse (PR 2.86 / 3.02, max |r| 0.950), and fails both geometry gates.
- **Pair-specificity is repaired only by InfoNCE.** Every InfoNCE run reaches a pair-retrieval ratio of 0.91-1.13. R1, R3 and R6 reach code R@10 of 0.386-0.387 against CLIP's 0.341 on the same painting-disjoint val pools, so the 32-factor code matches image-caption pairs better than raw CLIP does. Without an agreement term (R8) the ratio is 0.397 and fails, exactly as in Task 3's D3.
- **No mechanism in the grid meets pair-specificity and the readout/sparsity floors at the same time.**
  - The best InfoNCE runs (R1, R3, R6) fail exactly two gates. One is the text readout: 0.4589-0.4600 against the CLIP PCA-10 floor of 0.4441, a miss of 0.015-0.016. The other is sparsity: 44-49% of factors are active per row, against a gate of 37.5%.
  - Stronger L1 (R7) reaches image sparsity but not text (0.405). It also adds 3 modality-private factors and worsens the readout.
  - TopK (R4, R5) meets sparsity by construction and keeps pair retrieval. It fails the readout on both modalities and leaves 9 and 14 modality-private factors.
  - R8 passes readout and sparsity but not pair retrieval (8/9).
- **Condition-specific benefit over R0.** The selection score is mean naive-minus-uniform R@1 on val label episodes: emotion and art style, both directions, beta=0. R0 scores +0.55 R@1 points. Every repair mechanism scores higher:
  - R6 +4.99, the highest;
  - R3 +4.17, R7 +3.91, R1 +3.85;
  - R8 +3.38, R2 +3.31;
  - R5 +2.70, R4 +2.37.

  All of these runs fail at least one gate, so none is selected.
- **Cost to reconstruction relative to R0: none for the ReLU repair runs; TopK costs some on the image side.**
  - Every InfoNCE ReLU run reads out CLIP features better than R0 in both modalities: image 0.4806-0.4912 against 0.4935, text 0.4589-0.4650 against 0.4666. R8 (0.4700 / 0.4249) and R2 (0.4492 / 0.4248) are better still.
  - The readout gate fails because the absolute text floor (CLIP PCA-10, 0.4441) is stricter than anything R0 reached. R0 fails that floor too. The repair itself does not lose information.
  - Only TopK loses image readout against R0 (0.5002 / 0.4999 vs 0.4935).
- **Gate sanity check (Step 4.5): not applicable, so no flag.** The check compares gate-failing runs against the best passing run, and there is none. Descriptively only, and not part of the rule: the highest-scoring run (R6, 7/9) exceeds the 8/9 run (R8) by 1.61 R@1 points.

**Options for the user.** None of these was run. Each is a gate or configuration decision that the plan reserves for the user.

1. **Waive a named gate for a near-passing run.**
   - R6 or R3 miss only the text readout floor, by about 0.015, and sparsity.
   - R8 misses only pair retrieval, at 0.397 against 0.5.
   - Such a run would then go through replication and the held check as pre-registered.
2. **Register new configurations aimed at the joint binding set.** The InfoNCE runs clear pair retrieval with a wide margin (ratio 1.13 against 0.5). A weaker agreement weight or temperature might trade some of that margin for readout and sparsity. R8 sits at the other end of that same axis.
3. **Revisit the text readout floor** (Ruling 4 kept it at PCA-10 as an information floor). Every InfoNCE ReLU run is closer to it than R0 is.

## Evidence

All numbers come from one run of `src/test/20261011_factor_repair_grid/run_grid.py`: seed 42, RTX 3090, torch 2.11.0+cu130, 387.4 s total, of which the train-side graph, Stage 1 and communities took 54.8 s. Each fit took about 24-30 s of training plus about 10 s of evaluation.

Setup:

- **Split:** the painting-grouped split (seed 42) gives train/val/held = 216,107 / 30,872 / 61,744 rows, with zero painting and image-vector leakage (asserted).
- **Train-side graph:** 2,198,162 edges and 22 Leiden communities, rebuilt here and identical to Task 3.
- **Gates:** fit = train rows, eval = val rows, community codes and labels = train rows, default thresholds.
- **Every run:** base `lambda_usage_balance=0.1`, seed 42, 32 factors, 2,000 epochs, `group_ids = leakage_groups(...)[train]`, other fields at their defaults. All codes and losses were finite in all nine runs.

Validation label episodes were built once and shared by every run:

- 2,048 emotion episodes over 8 target labels. "something else" is excluded as a target but still appears as a negative.
- 2,048 art-style episodes over 23 styles. The 4 styles with fewer than 30 val paintings are not targets.
- Painting key = leakage group. Clean negatives: contrasts and distractors never come from a painting that carries the target label. The flag has no effect for art style, which is per painting.
- Each episode scores 13 candidates, so chance R@1 is 0.077.
- The script asserts that the episodes use only val rows and are unchanged when built from val-only arrays.

| ID | Config beyond the base |
|---|---|
| R0 | none (the collapsed recipe) |
| R1 | `agreement="infonce"` |
| R2 | `lambda_decorrelation=1.0` |
| R3 | `agreement="infonce", lambda_decorrelation=1.0` |
| R4 | `agreement="infonce", activation="topk", topk=8, lambda_sparsity=0.0` |
| R5 | R4 + `lambda_decorrelation=1.0` |
| R6 | R3 + `center_inputs=True` |
| R7 | R3 + `lambda_sparsity=0.1` |
| R8 | `lambda_paired=0.0` (Task 3 D3; added with user approval) |

### Gate pass/fail and selection score (val)

| run | PR | redundancy | readout | sparsity | dead | private | concentration | spanning | pair retrieval | passed | emotion lift | art-style lift | selection score | mean readout |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| R0 | FAIL | FAIL | FAIL | FAIL | PASS | PASS | PASS | PASS | FAIL | 4/9 | -0.0012 | +0.0122 | +0.0055 | 0.4801 |
| R1 | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | 7/9 | +0.0286 | +0.0483 | +0.0385 | 0.4708 |
| R2 | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | 6/9 | +0.0076 | +0.0586 | +0.0331 | 0.4370 |
| R3 | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | 7/9 | +0.0339 | +0.0496 | +0.0417 | 0.4698 |
| R4 | PASS | PASS | FAIL | PASS | PASS | FAIL | PASS | PASS | PASS | 7/9 | +0.0203 | +0.0271 | +0.0237 | 0.4829 |
| R5 | PASS | PASS | FAIL | PASS | PASS | FAIL | PASS | PASS | PASS | 7/9 | +0.0198 | +0.0342 | +0.0270 | 0.4788 |
| R6 | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | 7/9 | +0.0435 | +0.0564 | +0.0499 | 0.4709 |
| R7 | PASS | PASS | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | 6/9 | +0.0315 | +0.0466 | +0.0391 | 0.4781 |
| R8 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | 8/9 | +0.0134 | +0.0542 | +0.0338 | 0.4474 |

The lifts and the selection score are R@1 fractions: +0.0499 means 4.99 R@1 points. Mean readout is the mean of the image and text readout errors, which would have been the tie-break.

### Selection, applied step by step

1. **Necessary (all gates on val):** no run qualifies. The passing set is empty.
2. **Choose / tie-break:** not reached.
3. **No run passes (Step 4.4):** the binding gates and scores are below. **STOP.**
4. **Sanity check (Step 4.5):** not applicable because there is no passing run. No flag.

| run | binding gates (value vs threshold) | score |
|---|---|---:|
| R0 | PR min 1.309 (>= 8); max abs r 0.99934 (<= 0.9); text readout 0.4666 vs 0.4441; active 0.7351 (<= 0.375); retrieval ratio 0.3732 (>= 0.5) | +0.55 |
| R1 | text readout 0.4600 vs 0.4441; active 0.4839 | +3.85 |
| R2 | PR min 2.859; max abs r 0.9499; active 0.8387 | +3.31 |
| R3 | text readout 0.4589 vs 0.4441; active 0.4874 | +4.17 |
| R4 | image readout 0.5002 vs 0.4951, text 0.4655 vs 0.4441; 9 modality-private factors (<= 1) | +2.37 |
| R5 | image readout 0.4999 vs 0.4951, text 0.4577 vs 0.4441; 14 modality-private factors | +2.70 |
| R6 | text readout 0.4596 vs 0.4441; active 0.4858 | +4.99 |
| R7 | text readout 0.4650 vs 0.4441; active 0.4048 (text; image 0.3667 passes); 3 modality-private factors | +3.91 |
| R8 | retrieval ratio 0.3974 (>= 0.5) | +3.38 |

Across the grid, readout binds 7 of 9 runs (every run except R2 and R8), and in all 7 it is the text side that fails. Sparsity binds 6 runs, modality-private factors 3, pair retrieval 2, and PR and redundancy 2.

### Gate values (val)

| run | PR img | PR txt | max abs r (pairs >= .9) | PC1 img/txt | readout img (PCA-10) | readout txt (PCA-10) | active img | active txt | dead | private | top-2 | spanning | code R@10 | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 1.342 | 1.309 | 0.9993 (371/496) | 0.859/0.871 | 0.4935 (0.4951) | 0.4666 (0.4441) | 0.7191 | 0.7351 | 0 | 0 | 0.0790 | 1.000 | 0.1274 | 0.3732 |
| R1 | 20.809 | 19.932 | 0.4758 (0/496) | 0.112/0.127 | 0.4816 (0.4951) | 0.4600 (0.4441) | 0.4358 | 0.4839 | 0 | 0 | 0.0770 | 1.000 | 0.3861 | 1.1307 |
| R2 | 2.859 | 3.021 | 0.9499 (1/496) | 0.546/0.520 | 0.4492 (0.4951) | 0.4248 (0.4441) | 0.8322 | 0.8387 | 0 | 0 | 0.1783 | 0.875 | 0.1837 | 0.5380 |
| R3 | 21.701 | 20.786 | 0.4302 (0/496) | 0.104/0.119 | 0.4806 (0.4951) | 0.4589 (0.4441) | 0.4534 | 0.4874 | 0 | 0 | 0.0797 | 1.000 | 0.3858 | 1.1301 |
| R4 | 14.174 | 15.023 | 0.7877 (0/496) | 0.154/0.153 | 0.5002 (0.4951) | 0.4655 (0.4441) | 0.2497 | 0.2494 | 0 | 9 | 0.1367 | 0.969 | 0.3354 | 0.9824 |
| R5 | 14.383 | 14.390 | 0.4355 (0/496) | 0.136/0.142 | 0.4999 (0.4951) | 0.4577 (0.4441) | 0.2500 | 0.2499 | 0 | 14 | 0.1469 | 1.000 | 0.3093 | 0.9060 |
| R6 | 21.772 | 21.022 | 0.4525 (0/496) | 0.104/0.116 | 0.4822 (0.4951) | 0.4596 (0.4441) | 0.4413 | 0.4858 | 0 | 0 | 0.0802 | 1.000 | 0.3874 | 1.1348 |
| R7 | 19.295 | 19.731 | 0.5876 (0/496) | 0.118/0.117 | 0.4912 (0.4951) | 0.4650 (0.4441) | 0.3667 | 0.4048 | 0 | 3 | 0.0916 | 0.938 | 0.3561 | 1.0431 |
| R8 | 20.629 | 17.074 | 0.7996 (0/496) | 0.109/0.116 | 0.4700 (0.4951) | 0.4249 (0.4441) | 0.2082 | 0.2918 | 0 | 0 | 0.0736 | 0.812 | 0.1357 | 0.3974 |

How the columns are defined:

- The thresholds are: PR >= 8 per modality; max |r| <= 0.9 with no constant factor; readout <= CLIP PCA-10 per modality; active fraction <= 0.375; dead = 0; private <= 1; top-2 mass <= 0.20; spanning >= 0.75; ratio >= 0.5.
- CLIP R@10 is 0.3414 in every run: the same 30 pools of 1,000 val rows.
- PC1 is the first principal component's share of code variance. It is a diagnostic, not a gate.

### Condition lift detail (val, beta=0): naive / uniform R@1

| run | emotion i2t | emotion t2i | art style i2t | art style t2i |
|---|---|---|---|---|
| R0 | 0.1035 / 0.1050 | 0.1001 / 0.1011 | 0.1187 / 0.1045 | 0.1558 / 0.1455 |
| R1 | 0.1611 / 0.1216 | 0.1562 / 0.1387 | 0.2100 / 0.1875 | 0.2524 / 0.1782 |
| R2 | 0.1089 / 0.0967 | 0.1128 / 0.1099 | 0.1670 / 0.1157 | 0.1914 / 0.1255 |
| R3 | 0.1636 / 0.1221 | 0.1616 / 0.1353 | 0.2061 / 0.1826 | 0.2554 / 0.1797 |
| R4 | 0.1440 / 0.1230 | 0.1406 / 0.1211 | 0.1826 / 0.1792 | 0.2275 / 0.1768 |
| R5 | 0.1455 / 0.1245 | 0.1484 / 0.1299 | 0.1875 / 0.1807 | 0.2305 / 0.1689 |
| R6 | 0.1680 / 0.1143 | 0.1646 / 0.1313 | 0.2197 / 0.2002 | 0.2788 / 0.1855 |
| R7 | 0.1572 / 0.1094 | 0.1431 / 0.1279 | 0.2080 / 0.1948 | 0.2568 / 0.1768 |
| R8 | 0.1104 / 0.0972 | 0.1226 / 0.1089 | 0.1851 / 0.1714 | 0.2754 / 0.1807 |

The benefit is uneven across directions. For art style it sits mostly in text-to-image: R6 lift +9.33 points in t2i against +1.95 in i2t, and R8 +9.47 against +1.37. For emotion it sits in image-to-text: R6 +5.37 against +3.32. R0's emotion lift is negative in both directions (-0.15 and -0.10 points).

### Before / after against Task 1's collapsed codes

No model was selected, so the "after" columns are the two runs nearest to passing. They are descriptive only.

| | Task 1 collapsed (row split, held) | R0 (painting split, val) | R8 (val, 8/9) | R6 (val, 7/9, top score) |
|---|---:|---:|---:|---:|
| PR img / txt | 1.3242 / 1.3262 | 1.342 / 1.309 | 20.629 / 17.074 | 21.772 / 21.022 |
| max abs r | 0.99977 | 0.99934 | 0.79963 | 0.45248 |
| readout img (PCA-10) | 0.4936 (0.4943) | 0.4935 (0.4951) | 0.4700 (0.4951) | 0.4822 (0.4951) |
| readout txt (PCA-10) | 0.4669 (0.4452) | 0.4666 (0.4441) | 0.4249 (0.4441) | 0.4596 (0.4441) |
| active img / txt | 0.7133 / 0.7176 | 0.7191 / 0.7351 | 0.2082 / 0.2918 | 0.4413 / 0.4858 |
| retrieval ratio | 0.3709 | 0.3732 | 0.3974 | 1.1348 |
| gates passed | 4/9 | 4/9 | 8/9 | 7/9 |
| selection score (R@1 points) | n/a | +0.55 | +3.38 | +4.99 |

### Reproduction check

R0 reproduces Task 3's D0, and R8 reproduces D3, to every printed decimal. R0: PR 1.342 / 1.309, max |r| 0.99934, retrieval 0.3732. R8: PR 20.629 / 17.074, max |r| 0.79963, readout 0.4700 / 0.4249, retrieval 0.3974. Training on this setup is deterministic, and the rebuilt train-side graph matches Task 3's.

### Not run because of the stop

- Step 5: seed replication (seeds 43 and 44), and Hungarian factor alignment.
- Step 6: gates on held rows for the selected model and for R0, the checkpoints `checkpoints/selected_seed42.pt` and `checkpoints/R0_seed42.pt`, and the reload check.

The code paths for both steps were exercised once in a 3-epoch smoke run whose numbers were discarded. That smoke run used val rows as a stand-in for held rows. Its reload of 1,000 rows was bit-identical, and its checkpoints were deleted afterwards.

## Caveats

- **Single seed, single run per configuration, and no confidence intervals on lift.** Each label type has 2,048 episodes. Score differences of about 1 R@1 point between runs are not resolvable; the pre-registered rule treats them as ties. Among the InfoNCE ReLU runs the scores span 3.85-4.99 points. Only R6 vs R1 and R6 vs R7 differ by more than 1 point (1.14 and 1.08).
- **Ties at beta=0 in sparse codes.** Ties count half against the positive. They are frequent where codes are sparse or collapsed:
  - R8: 701 of 2,048 emotion i2t naive episodes have a candidate tied with the positive;
  - R4 and R5: 425-488;
  - R0: 436;
  - R1, R3 and R6: 25-46.

  This lowers R@1 for both naive and uniform in those runs. Lift is a difference, so the effect partly cancels, but it may understate the benefit available from sparse codes.
- **A retrieval ratio above 1 is not leakage.** Val rows are painting- and image-disjoint from the train rows the model was fit on (asserted). The ratio uses 30 pools of 1,000, so differences of a few hundredths are within pool noise.
- **The readout gate is linear.** It measures an affine readout of CLIP features from the codes, not the model's own decoder reconstruction. The text floor (PCA-10, 0.4441) is the single number that binds every InfoNCE ReLU run.
- **Held rows and integrity checks.** Held rows were never encoded or scored. Their metadata (painting ids, image hashes, art styles) was read only by integrity checks: the split-leakage assertion and the one-style-per-painting assertion. Neither can influence selection.
- **What the selection score measures.** It averages two label types and both directions. The episodes come from val paintings, and the labels were not used in training. The score says nothing yet about arbitrary human-stated conditions.

## Reproduction

```
/root/miniconda3/envs/CoSiR/bin/python src/test/20261011_factor_repair_grid/run_grid.py
/root/miniconda3/envs/CoSiR/bin/python src/test/20261011_factor_repair_grid/run_grid.py --tables
```

The first command takes about 6.5 minutes on an RTX 3090. The second reprints the tables from `results/summary.json`. Per-run JSON and logs are gitignored in the same folder. Log: `src/test/20261011_factor_repair_grid/20261011_factor_repair_grid_log.md`.
