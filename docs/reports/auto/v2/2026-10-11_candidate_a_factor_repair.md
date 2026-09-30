# CoSiR v2 Candidate A: factor-repair grid, pre-registered gates and goal-based selection

## Verdict (final state, after the user's amendment)

**Under amended gates, the factor space is repaired.**

The selected recipe is **R3**: InfoNCE agreement in place of the cosine term, plus the decorrelation penalty (`agreement="infonce", lambda_decorrelation=1.0`). It passes all nine amended gates in three places:
- on `val` at seed 42;
- on `val` at replication seeds 43 and 44;
- on `held` rows.

**The gates were amended after the fact.** The user changed two thresholds after the pre-registered rule had stopped with no passing run. That stop is documented below as the historical record. The two changes:
- readout must be no worse than R0's, instead of the absolute CLIP PCA-10 floor;
- the sparsity cap is 50% active, instead of 37.5%.

Under the original pre-registered gates, R3 passes 7 of 9 on `val` and on `held`. It fails the absolute text-readout floor and the 37.5% sparsity cap.

- **Is the space repaired?** Yes, in geometry and in pair-specificity, on held rows as on val. Held rows, R3 against R0:

  | Measure (held rows) | R3 | R0 |
  |---|---:|---:|
  | Participation ratio img / txt | 21.8 / 20.7 | 1.34 / 1.31 |
  | Max \|r\| between factors | 0.451, no pair ≥ 0.9 | 0.99936, 371/496 pairs ≥ 0.9 |
  | First-PC share of code variance, img / txt | 0.105 / 0.121 | 0.860 / 0.872 |
  | Pair-retrieval ratio | 1.13 | 0.368 |

  - On the same held pools, R3's code R@10 is 0.387 against CLIP's 0.343.
  - R3 has no dead factors and no modality-private factors, and all 32 factors span communities.
  - **What the "repaired" claim rests on.** The amended 50% sparsity cap sits only about 1.3 points above R3's seed-42 val text active fraction (0.487). The claim therefore rests mainly on the pre-registered geometry gates: participation ratio, redundancy and pair retrieval. R3 passes those by wide margins under the ORIGINAL thresholds. On val, its PR is 20.8 (gate ≥ 8), its max |r| is 0.43 (gate ≤ 0.90) and its pair-retrieval ratio is 1.13 (gate ≥ 0.5). On held, the same values are 20.7, 0.45 and 1.13.
- **Which mechanism did it?** Removing the cosine agreement term escapes the collapse. InfoNCE then adds pair-specificity.
  - The collapse is caused by the cosine agreement term. Removing it (R8, equivalently Task 3's D3) already escapes the collapse: PR 20.6 / 17.1, max |r| 0.80, and 8/9 original gates.
  - InfoNCE's specific contribution is pair-specificity: the pair-retrieval ratio rises from 0.40 (R8) to 1.13 (R3). It also adds about 0.8 selection-score points over R8 (R3 +4.17, R8 +3.38). All four InfoNCE ReLU runs (R1, R3, R6, R7) escape the collapse and reach a ratio of 1.04-1.13.
  - Decorrelation on top of the cosine term (R2) only partly lifts the collapse (PR 2.86 / 3.02, max |r| 0.950). It passes pair retrieval only narrowly (0.538).
  - Decorrelation adds little on top of InfoNCE. From R1 to R3 the selection score goes from +3.85 to +4.17, and max |r| from 0.476 to 0.430.
  - R3 was chosen over R6 (R3 plus input centering) by the tie-break. R6 scored 0.82 points higher, which is inside the pre-registered 1-point tie band. The tie-break prefers the lower mean readout, and R3's is 0.4698 against R6's 0.4709.
- **Condition-specific benefit over R0.** Measured on val, beta=0, over 2,048 emotion plus 2,048 art-style episodes.
  - R3 scores **+4.17** R@1 points, against R0's **+0.55**.
  - Replication seeds score +3.71 (seed 43) and +4.28 (seed 44). Neither is at or below R0.
  - The benefit is uneven across directions. For art style it sits mainly in text-to-image (+7.6 to +8.3 points against +2.3 to +2.7 for image-to-text). For emotion it sits mainly in image-to-text (+3.8 to +4.2 points against +0.8 to +2.6 for text-to-image).
  - The held-episode test of this benefit is Task 7's job. No held episodes were built or scored here.
- **Cost in linear-readout error: none relative to R0.** This is an affine readout of CLIP features from the codes, which is the readout gate's metric. The model's own decoder reconstruction was not measured.
  - On held, R3 reads out 0.4800 (image) and 0.4600 (text), against R0's 0.4939 and 0.4680.
  - Against CLIP PCA-10 (0.4949 / 0.4454), R3 is better on image and 0.0146 worse on text.
- **Cost in sparsity.** On held, 45% of image factors and 49% of text factors are active per row. The original cap was 37.5% and the amended cap is 50%.
- **Seed stability: the gates and the score replicate. About 24 directions are shared across seeds and about 7 are seed-specific; individual factor indices are not stable.**
  - Hungarian matching of individual seed-43 and seed-44 factors to seed 42 gives mean |r| 0.747 / 0.742 and median 0.837 / 0.786. The minimum is only 0.143 / 0.195. Of the 32 factors, 13 and 10 match at |r| ≥ 0.9, while 5 and 4 match below 0.5.
  - Final-review measurements (post hoc) of the spanned directions:
    - The canonical correlations from seed 42 to seeds 43/44 have median 0.94.
    - Of the 32, 19–20 are ≥ 0.9 and 24–25 are ≥ 0.8. The last few are only 0.06–0.09.
    - Every seed-43/44 factor is linearly predictable from the seed-42 codes with R² ≥ 0.55 (median about 0.86).
  - The plan counts this as a caveat, not a failure.
- **Checkpoints for Task 7.** `checkpoints/selected_seed42.pt` (R3) and `checkpoints/R0_seed42.pt` both come from deterministic seed-42 retrains, which reproduced all 24 compared stored grid metrics exactly. Reloading either checkpoint re-encodes 1,000 held rows bit-identically. Their SHA-256 hashes are listed under "Checkpoints and reload verification".

## Amended thresholds (user decision, after the fact)

### The two changes and their stated reasons

The plan amendment is in commit `d490b96` (2026-09-29). It was decided after the Step 4.4 stop, on `val` results only; held rows had not been encoded. All other gates, the selection rule, the tie-break, replication and the held check are unchanged.

1. **Readout becomes "no worse than R0".** A run passes if readout_img ≤ R0's readout_img and readout_txt ≤ R0's readout_txt, on the same rows.
   - Stated reason: the floor exists so that a repair must not lose information (Ruling 4), and R0 itself never met the absolute PCA-10 text floor.
   - Reference for the grid and for seeds 43/44: R0's seed-42 `val` readout, the exact stored values 0.493539 (image) and 0.466609 (text).
   - Reference for the held check: R0's `held` readout, 0.493931 and 0.467955.
   - R0 is its own reference, so it passes this gate trivially.
2. **Sparsity cap becomes ≤ 50% active** (`FactorGateThresholds(max_active_fraction=0.5)`).
   - Stated reason: runs made sparser, TopK at 25% active and R7, scored lower on the selection metric than the InfoNCE runs at 44-49% active.

**How the change was implemented.** `run_grid.py --amended` re-derives both the original and the amended flags from each run's stored gate values, so the grid was re-gated without retraining. For all nine runs, the re-derived original flags equal the stored ones (asserted). New evaluations (seeds 43/44 and held) call `evaluate_factor_gates` with `max_active_fraction=0.5`, then apply the R0-reference readout rule to the values it returns. Nothing under `src/` changed at the time. Since the final review, `src/eval/factor_gates.py` exposes the same rule: `evaluate_factor_gates(..., thresholds=AMENDED_2026_09_29_THRESHOLDS, readout_reference=(R0 readout_img, R0 readout_txt))`, with R0's readout measured on the same rows. Its defaults remain the pre-registered rule, under which R3 passes 7/9. The numbers in this report were produced by `run_grid.py`.

**What this means for interpretation.**
- The thresholds were changed after the val results were known. They are exactly the two gates the best runs had failed.
- The val selection is therefore no longer a pre-registered test. What remains honest is:
  - the replication seeds;
  - the held gates, computed once after the decision and not informing it;
  - Task 7's held label episodes, which are the independent test of the condition benefit.

### Re-gated grid (val): original and amended flags side by side

Only the readout and sparsity columns can change. All other flags are identical under both rules (asserted).

| run | PR | redundancy | readout (orig -> amended) | sparsity (orig -> amended) | dead | private | concentration | spanning | pair retrieval | original passed | amended passed | selection score | mean readout |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| R0 | FAIL | FAIL | FAIL -> **PASS** | FAIL -> **FAIL** | PASS | PASS | PASS | PASS | FAIL | 4/9 | **5/9** | +0.0055 | 0.4801 |
| R1 | PASS | PASS | FAIL -> **PASS** | FAIL -> **PASS** | PASS | PASS | PASS | PASS | PASS | 7/9 | **9/9** | +0.0385 | 0.4708 |
| R2 | FAIL | FAIL | PASS -> **PASS** | FAIL -> **FAIL** | PASS | PASS | PASS | PASS | PASS | 6/9 | **6/9** | +0.0331 | 0.4370 |
| R3 | PASS | PASS | FAIL -> **PASS** | FAIL -> **PASS** | PASS | PASS | PASS | PASS | PASS | 7/9 | **9/9** | +0.0417 | 0.4698 |
| R4 | PASS | PASS | FAIL -> **FAIL** | PASS -> **PASS** | PASS | FAIL | PASS | PASS | PASS | 7/9 | **7/9** | +0.0237 | 0.4829 |
| R5 | PASS | PASS | FAIL -> **FAIL** | PASS -> **PASS** | PASS | FAIL | PASS | PASS | PASS | 7/9 | **7/9** | +0.0270 | 0.4788 |
| R6 | PASS | PASS | FAIL -> **PASS** | FAIL -> **PASS** | PASS | PASS | PASS | PASS | PASS | 7/9 | **9/9** | +0.0499 | 0.4709 |
| R7 | PASS | PASS | FAIL -> **PASS** | FAIL -> **PASS** | PASS | FAIL | PASS | PASS | PASS | 6/9 | **8/9** | +0.0391 | 0.4781 |
| R8 | PASS | PASS | PASS -> **PASS** | PASS -> **PASS** | PASS | PASS | PASS | PASS | FAIL | 8/9 | **8/9** | +0.0338 | 0.4474 |

- R4 and R5 still fail readout: their image readout (0.5002 / 0.4999) is worse than R0's 0.4935.
- R0 and R2 still fail sparsity: 0.735 and 0.839 active.

### Selection under the amended gates (pre-registered rule, applied literally)

1. **Necessary (all amended gates pass):** R1, R3 and R6.
2. **Highest score:** R6, +0.04993.
3. **Tie band (within 1.0 R@1 point of the best):** R6 and R3. R3 is 0.82 points below R6. R1 (+0.03845) is 1.15 points below, so it is not tied.
4. **Tie-break (lower mean readout):** R3 at 0.46975, against R6 at 0.47093. **R3 is selected.**
5. **Gate sanity check (Step 4.5):** no gate-failing run beats +4.99 by more than 2 points (the best failing run is R7 at +3.91). **No flag.**

These results agree with the controller's expectation. They were computed from the stored JSON by `select()`, not assumed.

**The selected `FactorTrainingConfig`:**

```
FactorTrainingConfig(num_factors=32, lr=0.001, epochs=2000, batch_size=1024,
    lambda_reconstruction=1.0, lambda_paired=1.0, lambda_graph=1.0, lambda_sparsity=0.01,
    lambda_anti_split=0.1, lambda_usage_balance=0.1, seed=42, agreement="infonce",
    infonce_temperature=0.1, lambda_decorrelation=1.0, activation="relu", topk=None,
    center_inputs=False)
```

Training call: `train_factors(train_img, train_txt, train_graph, config, group_ids=leakage_groups(...)[train])`.

### Replication (Step 5, val, amended gates, readout reference = R0 seed-42 val)

| seed | amended gates | original gates (failed) | PR img / txt | max abs r (pairs >= .9) | PC1 img / txt | active img / txt | readout img / txt | dead / private | spanning | code R@10 / ratio | emotion lift | art-style lift | selection score | Hungarian abs r to seed 42: mean / median / min (n >= .9, n < .5, of 32) |
|---|---:|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---|
| 42 | 9/9 | 7/9 (readout, sparsity) | 21.701 / 20.786 | 0.4302 (0/496) | 0.104 / 0.119 | 0.4534 / 0.4874 | 0.4806 / 0.4589 | 0 / 0 | 1.000 | 0.3858 / 1.1301 | +0.0339 | +0.0496 | +0.0417 | reference |
| 43 | 9/9 | 7/9 (readout, sparsity) | 21.970 / 21.156 | 0.3617 (0/496) | 0.105 / 0.116 | 0.4286 / 0.4850 | 0.4807 / 0.4594 | 0 / 0 | 1.000 | 0.3847 / 1.1268 | +0.0227 | +0.0515 | +0.0371 | 0.747 / 0.837 / 0.143 (13, 5) |
| 44 | 9/9 | 7/9 (readout, sparsity) | 21.752 / 20.785 | 0.4404 (0/496) | 0.104 / 0.119 | 0.4357 / 0.4768 | 0.4810 / 0.4617 | 0 / 0 | 1.000 | 0.3839 / 1.1244 | +0.0305 | +0.0552 | +0.0428 | 0.742 / 0.786 / 0.195 (10, 4) |

| seed | emotion lift i2t / t2i | art-style lift i2t / t2i |
|---|---|---|
| 42 | +0.0415 / +0.0264 | +0.0234 / +0.0757 |
| 43 | +0.0376 / +0.0078 | +0.0264 / +0.0767 |
| 44 | +0.0396 / +0.0215 | +0.0273 / +0.0830 |

- Both replication seeds pass all nine amended gates, so the recipe stays selected.
- The scores (+3.71, +4.28) are within ±0.6 points of seed 42's +4.17, and far above R0's +0.55.
- The weakest cell is seed 43's emotion text-to-image lift, at +0.78 points.
- The Hungarian alignment matches each replication seed's factors one-to-one to seed 42's factors. It maximizes the total |r| by running `scipy.optimize.linear_sum_assignment` on −|r|, where r is the Pearson correlation between factor columns of the `val` pair codes, 0.5·(img + txt). No factor is constant in any seed.

### Held check (Step 6): the first and only time held rows were encoded

Setup:
- Eval rows are the 61,744 held rows. Fit and community rows are train rows, as on val.
- The amended thresholds apply. The readout reference is **R0's held readout (0.493931 / 0.467955)**.
- R0 is the reference itself, so its amended readout flag passes trivially. Its values are reported for comparison.

| model | PR | redundancy | readout (orig -> amended) | sparsity (orig -> amended) | dead | private | concentration | spanning | pair retrieval | original passed | amended passed |
|---|---|---|---|---|---|---|---|---|---|---:|---:|
| selected R3 | PASS | PASS | FAIL -> **PASS** | FAIL -> **PASS** | PASS | PASS | PASS | PASS | PASS | 7/9 | **9/9** |
| R0 | FAIL | FAIL | FAIL -> **PASS** (self-reference) | FAIL -> **FAIL** | PASS | PASS | PASS | PASS | FAIL | 4/9 | **5/9** |

| model | PR img / txt | max abs r (pairs >= .9) | PC1 img / txt | readout img (PCA-10) | readout txt (PCA-10) | active img / txt | dead / private | top-2 | spanning | code R@10 / CLIP R@10 / ratio |
|---|---|---|---|---|---|---|---|---:|---:|---|
| selected R3 | 21.775 / 20.676 | 0.45057 (0/496) | 0.105 / 0.121 | 0.4800 (0.4949) | 0.4600 (0.4454) | 0.4526 / 0.4857 | 0 / 0 | 0.0803 | 1.000 | 0.3871 / 0.3427 / 1.1295 |
| R0 | 1.338 / 1.305 | 0.99936 (371/496) | 0.860 / 0.872 | 0.4939 (0.4949) | 0.4680 (0.4454) | 0.7168 / 0.7314 | 0 / 0 | 0.0784 | 1.000 | 0.1261 / 0.3427 / 0.3680 |

Held values track the val values closely for both models:
- readout within 0.0014;
- active fraction within 0.004;
- retrieval ratio within 0.006;
- max |r| within 0.021;
- participation ratio within 0.12.

### Checkpoints and reload verification

- **Where the models came from.** Both checkpointed models are **deterministic seed-42 retrains**. The in-memory grid models were not kept after the stop.
  - The R0 and R3 retrains were checked against the stored grid results on 24 values each: gate values, PC1 shares, first and last loss, mean code, both lift means and the selection score.
  - Both matched **exactly**.
  - The retrains were used only to obtain the models and the seed-42 val pair codes for the alignment. They are not new configurations.
- **Saved** with `save_factor_checkpoint`:
  - `src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt` (R3);
  - `src/test/20261011_factor_repair_grid/checkpoints/R0_seed42.pt`.

  Both files are gitignored.
- **SHA-256**, computed with `sha256sum` during the final-review fix wave. They match the prefixes the final reviewer recorded.
  - `checkpoints/selected_seed42.pt`: `1c299fc008ca006ffd2de37315ac3557523bf06e0b755d4dcbf70b999b4e453f`
  - `checkpoints/R0_seed42.pt`: `4229dfe55f735bc7e9849c8d7af623b5872a9de940f616969ef477fb00a253a7`
- **Reloaded** with `load_factor_checkpoint(path, device="cuda:0")`. The same 1,000 held rows (chosen with `default_rng(42)`) were encoded as one batch by the in-session model and by the reloaded model.
  - **Bit-identical** for both models: max abs difference 0.0, and the config round-trip is equal.
  - For information, the same rows inside the full 8,192-row-batch held encoding differ by at most 8.3e-7 (R3) and 3.0e-7 (R0). That difference comes from the batch shape (cuBLAS), not from the checkpoint.

### Before / after against Task 1's collapsed codes

| | Task 1 collapsed (row split, held) | R0 (painting split, held) | selected R3 (painting split, held) | selected R3 (val) |
|---|---:|---:|---:|---:|
| PR img / txt | 1.3242 / 1.3262 | 1.338 / 1.305 | 21.775 / 20.676 | 21.701 / 20.786 |
| max abs r | 0.99977 | 0.99936 | 0.45057 | 0.43019 |
| readout img (PCA-10) | 0.4936 (0.4943) | 0.4939 (0.4949) | 0.4800 (0.4949) | 0.4806 (0.4951) |
| readout txt (PCA-10) | 0.4669 (0.4452) | 0.4680 (0.4454) | 0.4600 (0.4454) | 0.4589 (0.4441) |
| active img / txt | 0.7133 / 0.7176 | 0.7168 / 0.7314 | 0.4526 / 0.4857 | 0.4534 / 0.4874 |
| retrieval ratio | 0.3709 | 0.3680 | 1.1295 | 1.1301 |
| gates passed, original / amended | 4/9 / n/a | 4/9 / 5/9 | 7/9 / 9/9 | 7/9 / 9/9 |

## Original pre-registered result (historical record): the Step 4.4 stop

The rest of this section is the verdict as written at the stop. Five reviewer wording fixes are folded in:
- pair-specificity;
- linear-readout cost;
- ties;
- the loss table;
- the non-finite ruling.

### Original verdict

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
- **Only InfoNCE brings the pair-retrieval ratio to about 0.9 or above** (0.91-1.13 in every InfoNCE run). Cosine plus decorrelation (R2) clears the 0.5 gate only narrowly, at 0.538. It stays collapsed, so this is not pair-specificity of a healthy space. R1, R3 and R6 reach code R@10 of 0.386-0.387 against CLIP's 0.341 on the same painting-disjoint val pools, so the 32-factor code matches image-caption pairs better than raw CLIP does. Without an agreement term (R8) the ratio is 0.397 and fails, exactly as in Task 3's D3.
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
- **Cost in linear-readout error relative to R0: none for the ReLU repair runs; TopK costs some on the image side.** This is the readout gate's metric: an affine readout of the CLIP features from the codes. It is not the model's own decoder reconstruction, which this task did not measure.
  - Every InfoNCE ReLU run reads out CLIP features better than R0 in both modalities: image 0.4806-0.4912 against 0.4935, text 0.4589-0.4650 against 0.4666. R8 (0.4700 / 0.4249) and R2 (0.4492 / 0.4248) are better still.
  - The readout gate fails because the absolute text floor (CLIP PCA-10, 0.4441) is stricter than anything R0 reached. R0 fails that floor too. The repair itself does not lose information.
  - Only TopK loses image readout against R0 (0.5002 / 0.4999 vs 0.4935).
- **Gate sanity check (Step 4.5): not applicable, so no flag.** The check compares gate-failing runs against the best passing run, and there is none. Descriptively only, and not part of the rule: the highest-scoring run (R6, 7/9) exceeds the 8/9 run (R8) by 1.61 R@1 points.

**Options for the user, as offered at the stop.** None of these was run. Each is a gate or configuration decision that the plan reserves for the user. *Resolved 2026-09-29:* the user chose a variant of option 3 (readout no worse than R0) together with a 50% sparsity cap. See "Amended thresholds" above.

1. **Waive a named gate for a near-passing run.**
   - R6 or R3 miss only the text readout floor, by about 0.015, and sparsity.
   - R8 misses only pair retrieval, at 0.397 against 0.5.
   - Such a run would then go through replication and the held check as pre-registered.
2. **Register new configurations aimed at the joint binding set.** The InfoNCE runs clear pair retrieval with a wide margin (ratio 1.13 against 0.5). A weaker agreement weight or temperature might trade some of that margin for readout and sparsity. R8 sits at the other end of that same axis.
3. **Revisit the text readout floor** (Ruling 4 kept it at PCA-10 as an information floor). Every InfoNCE ReLU run is closer to it than R0 is.

## Evidence for the original grid (val, original gates)

All numbers in this section come from one run of `src/test/20261011_factor_repair_grid/run_grid.py`: seed 42, RTX 3090, torch 2.11.0+cu130, 387.4 s total, of which the train-side graph, Stage 1 and communities took 54.8 s. Each fit took about 24-30 s of training plus about 10 s of evaluation.

Setup:

- **Split:** the painting-grouped split (seed 42) gives train/val/held = 216,107 / 30,872 / 61,744 rows, with zero painting and image-vector leakage (asserted).
- **Train-side graph:** 2,198,162 edges and 22 Leiden communities, rebuilt here and identical to Task 3.
- **Gates:** fit = train rows, eval = val rows, community codes and labels = train rows, default thresholds.
- **Every run:** base `lambda_usage_balance=0.1`, seed 42, 32 factors, 2,000 epochs, `group_ids = leakage_groups(...)[train]`, other fields at their defaults. All codes and losses were finite in all nine runs.
- **Finite codes.** The dispatch ruling "assert all codes are finite before gating/scoring" was implemented as *record a failed run*. A run with non-finite train or val codes would be kept in the table with every gate FAIL instead of aborting the script or being dropped. The branch never triggered: every run in the grid, the replication and the retrains was finite.

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

### Loss and runtime per run (seed 42)

| run | loss first | loss last | mean of last 100 | mean code img / txt (val) | train s |
|---|---:|---:|---:|---|---:|
| R0 | 0.93165 | -0.14783 | -0.14612 | 0.1018 / 0.0989 | 24 |
| R1 | 8.35477 | 5.22943 | 5.15636 | 0.1581 / 0.1545 | 25 |
| R2 | 0.94369 | -0.11012 | -0.10792 | 0.1118 / 0.1049 | 28 |
| R3 | 8.36682 | 5.25179 | 5.17578 | 0.1621 / 0.1542 | 29 |
| R4 | 8.53089 | 5.43908 | 5.39825 | 0.2226 / 0.2266 | 26 |
| R5 | 8.53774 | 5.58101 | 5.54510 | 0.2310 / 0.2234 | 30 |
| R6 | 8.56187 | 5.19589 | 5.14714 | 0.1448 / 0.1411 | 30 |
| R7 | 8.67352 | 5.47569 | 5.41184 | 0.0531 / 0.0514 | 29 |
| R8 | 0.30742 | -0.27202 | -0.27092 | 0.0785 / 0.0767 | 24 |

Totals differ in scale between agreement types. InfoNCE's log-softmax term is about log(batch) at the start, so the InfoNCE totals start near 8.4. They are not comparable with the cosine totals. All losses were finite at every epoch.

### Before / after against Task 1's collapsed codes (at the stop, descriptive)

At the stop no model was selected, so the "after" columns are the two runs nearest to passing. They are descriptive only. The final before/after table, with the selected R3 on held rows, is in the amendment section above.

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

### Not run at the time of the stop (later run under the amendment, see above)

- Step 5: seed replication (seeds 43 and 44), and Hungarian factor alignment.
- Step 6: gates on held rows for the selected model and for R0, the checkpoints `checkpoints/selected_seed42.pt` and `checkpoints/R0_seed42.pt`, and the reload check.

The code paths for both steps were exercised once in a 3-epoch smoke run whose numbers were discarded. That smoke run used val rows as a stand-in for held rows. Its reload of 1,000 rows was bit-identical, and its checkpoints were deleted afterwards.

## Caveats

- **After-the-fact thresholds.** The two amended gates (readout no worse than R0, and sparsity ≤ 50%) were set after the val results were known. They are exactly the gates the best runs had failed.
  - The val-based selection is therefore post hoc, even though the selection rule and tie-break were applied unchanged.
  - The evidence not shaped by the decision is: the replication seeds, the held gates (computed once, after the decision) and Task 7's held label episodes.
  - Under the original pre-registered gates, no run passes (see the historical section).
- **Tie-break over R6.** R3 was selected over R6 by a mean-readout difference of 0.0012. R6 scored 0.82 points higher, inside the 1-point band. The two recipes differ only in input centering.
- **Seed-specific directions and unstable factor indices.** The replication seeds pass every gate and score +3.71 and +4.28.
  - The final review measured the spanned directions (post hoc). About 24 directions are shared across seeds: the canonical correlations to seed 42 have median 0.94, and 19–20 of 32 are ≥ 0.9, 24–25 ≥ 0.8. About 7 are seed-specific: the last canonical correlations are 0.06–0.09. Every seed-43/44 factor is linearly predictable from the seed-42 codes with R² ≥ 0.55 (median about 0.86).
  - Individual factor indices are less stable. Only 13 (seed 43) and 10 (seed 44) of the 32 factors match a seed-42 factor one-to-one at |r| ≥ 0.9, and the worst matches are 0.143 and 0.195.
  - Anything that reads meaning into a specific factor index should be checked across seeds.
- **Single seed per grid configuration, and no confidence intervals on lift.** Each label type has 2,048 episodes. Score differences of about 1 R@1 point between runs are not resolvable, which is why the pre-registered rule treats them as ties. Among the InfoNCE ReLU runs the scores span 3.85-4.99 points. Only R6 vs R1 and R6 vs R7 differ by more than 1 point (1.14 and 1.08).
- **Ties at beta=0.** The rank is tie-aware: rank = 1 + (number of candidates scored above the positive) + 0.5 × (number tied with it).
  - A single tie with the positive therefore gives rank 1.5. That is a **full miss at R@1**, which needs rank ≤ 1.
  - The half-rank convention only matters at R@3 and deeper; rank 1.5 counts as a hit at R@3.
  - Ties are frequent where codes are sparse or collapsed, and they lower R@1 for both naive and uniform weights. Lift is a difference, so the effect partly cancels, but it may understate what sparse codes could deliver.

  Episodes (of 2,048 per label type) in which at least one candidate ties the positive, per direction:

  | run | emotion naive i2t / t2i | emotion uniform i2t / t2i | art naive i2t / t2i | art uniform i2t / t2i |
  |---|---|---|---|---|
  | R0 | 436 / 411 | 104 / 97 | 307 / 317 | 89 / 75 |
  | R1 | 32 / 43 | 0 / 3 | 16 / 29 | 0 / 1 |
  | R2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
  | R3 | 25 / 36 | 0 / 0 | 20 / 27 | 0 / 0 |
  | R4 | 425 / 434 | 54 / 64 | 264 / 319 | 36 / 39 |
  | R5 | 488 / 473 | 111 / 103 | 312 / 340 | 60 / 63 |
  | R6 | 32 / 46 | 0 / 0 | 21 / 16 | 1 / 1 |
  | R7 | 86 / 104 | 1 / 9 | 65 / 65 | 3 / 2 |
  | R8 | 701 / 689 | 318 / 331 | 429 / 424 | 216 / 209 |

- **A retrieval ratio above 1 is not leakage.** Val and held rows are painting- and image-disjoint from the train rows the model was fit on (asserted). Val uses 30 pools of 1,000 rows and held uses 61 pools, so differences of a few hundredths are within pool noise.
- **The readout gate is linear.** It measures an affine readout of CLIP features from the codes, not the model's own decoder reconstruction, which was not measured here. Under the original rule, the text floor (PCA-10, 0.4441 on val) is the single number that binds every InfoNCE ReLU run.
- **Held rows were used once, in Step 6.** Held rows were encoded and gated once, for R3 and R0, after the selection and the replication had finished. Before that, their metadata (painting ids, image hashes, art styles) was read only by two integrity checks: the split-leakage assertion and the one-style-per-painting assertion. Neither can influence selection. No held label episodes were built.
- **What the selection score measures.** It averages two label types and both directions. The episodes come from val paintings, and the labels were not used in training. The score says nothing yet about arbitrary human-stated conditions.

## Reproduction

```
# original grid (Steps 1-4; about 6.5 min on an RTX 3090)
/root/miniconda3/envs/CoSiR/bin/python src/test/20261011_factor_repair_grid/run_grid.py
# amendment continuation: re-gate + select (Step A), R0/R3 seed-42 retrains, seeds 43/44, held check, checkpoints (about 4 min)
/root/miniconda3/envs/CoSiR/bin/python src/test/20261011_factor_repair_grid/run_grid.py --amended
# tables only, from results/summary.json and results/amended_summary.json
/root/miniconda3/envs/CoSiR/bin/python src/test/20261011_factor_repair_grid/run_grid.py --tables
/root/miniconda3/envs/CoSiR/bin/python src/test/20261011_factor_repair_grid/run_grid.py --amended --tables
```

- Per-run JSON, logs, val pair codes and checkpoints are gitignored in the same folder.
- Re-running `--amended` re-encodes held rows. Its results are only reproductions of the single held check reported here.
- Log: `src/test/20261011_factor_repair_grid/20261011_factor_repair_grid_log.md`.
