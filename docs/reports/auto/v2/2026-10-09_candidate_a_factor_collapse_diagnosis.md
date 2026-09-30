# CoSiR v2 Candidate A: where the factor-space collapse comes from

## Verdict

**One term is implicated: the cosine paired-agreement loss (`lambda_paired`).** With it in the loss, the 32 factors collapse onto one axis (smaller participation ratio 1.31, first principal component holding 86-87% of code variance, 371 of 496 factor pairs with |r| >= 0.9). With only that term removed (D3) the same code, seed, split and everything else gives participation ratio 20.6 (img) / 17.1 (txt), PC1 share 11-12%, max |r| 0.80 with no pair above 0.9, and 8 of the 9 gates pass. No other single removal helps: taking out the balance loss (D4) makes things worse, taking out the graph loss (D5) leaves the collapse in place, and centering the encoder input (D6) changes nothing that matters.

Under the pre-registered rules:

- **Implicated:** cosine agreement only (D3: min PR 17.07, 13.0 times D0's 1.31, and max |r| crossed from 0.9993 to 0.7996).
- **Not implicated:** usage balance (D4: min PR 1.19, max |r| 0.922), graph consistency (D5: min PR 1.29, max |r| 0.9994).
- **D1 (reconstruction only) is not collapsed** by the rule (min PR 3.20 >= 3, img 5.98), though only just on text. So the collapse does not exist before the auxiliary losses: they create it. D1 is nevertheless far from a usable shared space (PR gate 8 not met, text codes 97% active, 7 modality-private factors, pair retrieval ratio 0.03), so a repair must add pressure, not only remove a term.
- **It is not a split artifact.** D0 on the painting-grouped split reproduces Task 1's collapsed numbers on the old row split almost to the third decimal (table below).

**Which Task 4 mechanisms the evidence supports:**

| Mechanism | Verdict | Why |
|---|---|---|
| InfoNCE agreement (replacing cosine) | **Supported** | The cosine agreement term is the only implicated cause. Dropping it fixes geometry but no variant gets pair retrieval to the 0.5 gate (best D3 0.397, D0 0.373), so some pair-specific agreement is still needed and a contrastive one does not reward a shared single direction. That InfoNCE reaches 0.5 is untested here. |
| Decorrelation penalty | **Not needed to escape the collapse** | D3 has no decorrelation term and reaches max |r| 0.80 and 0/496 pairs >= 0.9. It might still be worth keeping as insurance for the InfoNCE recipe, but nothing here requires it. |
| TopK activation | **Not needed for the sparsity gate; mild support as a safer sparsity mechanism** | D3 passes sparsity with the existing L1 (active 0.21 / 0.29 vs 0.375). But L1 alone (D2) leaves 31 of 32 factors dead and every image code at zero on val, while no sparsity pressure (D1) leaves text 97% active, so the L1 weight sits on a cliff. TopK would remove that dependence. |
| Input centering | **Not supported** | D6: PR 1.63 / 1.66 (1.24 times D0), max |r| 0.99994, still collapsed. |

**Does the evidence point to a cause none of the four addresses?** Not for the D0 collapse itself: every variant that keeps the cosine term collapses (D0, D4, D5, D6: min PR between 1.19 and 1.63 whatever else is removed or changed), and removing it (D3) is the only change that lifts it. Two observations belong on the controller's checkpoint list, but neither is a new cause of the collapse:

1. **Dying factors are a live failure mode once the agreement term is gone or the balance loss is absent.** D2 (L1 without balance) and D4 (balance removed) both lose factors (D2: 31 dead; D4: 1 dead, top-2 mass share 0.886, 20 of 32 factors concentrated in one community). The usage-balance loss is what keeps D0 and D3 alive, so Task 4/6 should keep `lambda_usage_balance > 0` (or TopK) rather than treat balance as a candidate to remove.
2. **Pair retrieval fails in all seven variants** (ratio 0.00 to 0.40, gate 0.5). Geometry can be repaired by removal; pair-specificity cannot. That is squarely InfoNCE's job, but it is the gate to watch.

I recommend no amendment to Task 4's mechanism list on this evidence, with the two points above carried into Task 6's grid. The controller decides.

## Evidence

All numbers are from one run of `src/test/20261009_factor_collapse_diagnosis/run_diagnosis.py` (seed 42, 32 factors, 2,000 epochs, 33 s per variant, whole run 288.6 s). Fit rows = 216,107 train rows, eval rows = 30,872 val rows, community codes and labels = train rows (22 train communities, 2,198,162 graph edges). Painting-grouped split with zero painting or image-vector leakage (asserted at run time). Tables are regenerated from the per-variant JSON by `make_tables.py`.

Variants (change from D0 = `FactorTrainingConfig` defaults with `lambda_usage_balance=0.1`):

| ID | Change | Question |
|---|---|---|
| D0 | `lambda_usage_balance=0.1` (Task 6 / 7 recipe) | Does the collapse reproduce on the painting split? |
| D1 | reconstruction only (paired, graph, sparsity, anti-split, balance all 0) | Does a plain ReLU autoencoder collapse? |
| D2 | D1 + `lambda_sparsity=0.01` | Does L1 alone cause it? |
| D3 | D0, `lambda_paired=0` | Is the cosine agreement the cause? |
| D4 | D0, `lambda_usage_balance=0` | Is the balance loss the cause? |
| D5 | D0, `lambda_graph=0` | Is the graph term the cause? |
| D6 | D0, encoder input centered by train-row means (gates on original features) | Does centering change the optimum? |

### Geometry (val rows)

| ID | PR img | PR txt | max abs r | pairs >= 0.9 | PC1 share img | PC1 share txt | active img | active txt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 1.3420 | 1.3090 | 0.99934 | 371/496 | 0.8586 | 0.8707 | 0.7191 | 0.7351 |
| D1 | 5.9750 | 3.2002 | 0.80330 | 0/496 | 0.3478 | 0.5434 | 0.7808 | 0.9665 |
| D2 | 1.5343 | 1.0000 | 0.94607 | 2/55 | 0.7820 | 1.0000 | 0.0000 | 0.0313 |
| D3 | 20.6290 | 17.0744 | 0.79963 | 0/496 | 0.1085 | 0.1164 | 0.2082 | 0.2918 |
| D4 | 1.1910 | 1.1933 | 0.92206 | 1/496 | 0.9145 | 0.9134 | 0.1043 | 0.1150 |
| D5 | 1.3113 | 1.2857 | 0.99942 | 248/496 | 0.8706 | 0.8790 | 0.9135 | 0.9158 |
| D6 | 1.6262 | 1.6641 | 0.99994 | 201/496 | 0.7643 | 0.7566 | 0.6946 | 0.7055 |

Gate thresholds: PR >= 8 per modality, max |r| <= 0.9, active fraction <= 0.375. PC1 share and PR agree everywhere: a PR near 1.3 goes with a PC1 share near 0.86, i.e. one dominant axis rather than "a few equal axes". In D2 only 11 factors vary at all (pairs total 55), so its max |r| and PC1 share describe a nearly empty code.

### Remaining values

| ID | readout img (PCA-10) | readout txt (PCA-10) | dead | private | top-2 mass | spanning | code R@10 | CLIP R@10 | ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0.4935 (0.4951) | 0.4666 (0.4441) | 0 | 0 | 0.0790 | 1.000 | 0.1274 | 0.3414 | 0.3732 |
| D1 | 0.4247 (0.4951) | 0.3803 (0.4441) | 0 | 7 | 0.1182 | 1.000 | 0.0103 | 0.3414 | 0.0303 |
| D2 | 0.6397 (0.4951) | 0.5162 (0.4441) | 31 | 1 | 1.0000 | 0.031 | 0.0000 | 0.3414 | 0.0000 |
| D3 | 0.4700 (0.4951) | 0.4249 (0.4441) | 0 | 0 | 0.0736 | 0.812 | 0.1357 | 0.3414 | 0.3974 |
| D4 | 0.5352 (0.4951) | 0.4916 (0.4441) | 1 | 2 | 0.8859 | 0.344 | 0.0927 | 0.3414 | 0.2716 |
| D5 | 0.4857 (0.4951) | 0.4401 (0.4441) | 1 | 1 | 0.0829 | 0.906 | 0.0708 | 0.3414 | 0.2073 |
| D6 | 0.5157 (0.4951) | 0.4782 (0.4441) | 0 | 0 | 0.0681 | 1.000 | 0.0895 | 0.3414 | 0.2621 |

Readout is relative L2 of the best affine map from codes to the original features (lower is better); the gate needs it at or below the CLIP PCA-10 value in both modalities. "Dead" and "private" are counts of factors.

### Gate pass/fail

| ID | PR | redundancy | readout | sparsity | dead | private | concentration | spanning | pair retrieval | passed |
|---|---|---|---|---|---|---|---|---|---|---:|
| D0 | FAIL | FAIL | FAIL | FAIL | PASS | PASS | PASS | PASS | FAIL | 4/9 |
| D1 | FAIL | PASS | PASS | FAIL | PASS | FAIL | PASS | PASS | FAIL | 5/9 |
| D2 | FAIL | FAIL | FAIL | PASS | FAIL | PASS | FAIL | FAIL | FAIL | 2/9 |
| D3 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | 8/9 |
| D4 | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 1/9 |
| D5 | FAIL | FAIL | PASS | FAIL | FAIL | PASS | PASS | PASS | FAIL | 4/9 |
| D6 | FAIL | FAIL | FAIL | FAIL | PASS | PASS | PASS | PASS | FAIL | 4/9 |

D2's sparsity PASS is a false pass: the codes are nearly all zero because the units died, not because they are sparse.

### Pre-registered rule applied

A term is implicated if removing it at least doubles the smaller per-modality PR (D0's is 1.3090, so the bar is 2.618) or brings max |r| from above 0.9 to below 0.9 (D0's is 0.99934).

| Removed | Variant | min PR | times D0 min PR | max abs r | doubled PR? | r crossed 0.9? | Implicated |
|---|---|---:|---:|---:|---|---|---|
| cosine agreement | D3 | 17.0744 | 13.04 | 0.79963 | yes | yes | **YES** |
| usage balance | D4 | 1.1910 | 0.91 | 0.92206 | no | no | no |
| graph consistency | D5 | 1.2857 | 0.98 | 0.99942 | no | no | no |

D1 collapse check: min PR = min(5.9750, 3.2002) = 3.2002 >= 3, so D1 is **not collapsed** under the pre-registered definition. D6 (centering) is a different kind of test, not part of the removal rule; it moves min PR from 1.309 to 1.626 (1.24 times) and max |r| to 0.99994, so it does not change the optimum in any way that matters.

### Variant by variant

- **D0 reproduces the collapse** on the painting split (below).
- **D1, plain autoencoder:** PR 5.98 / 3.20, no correlated pairs (max |r| 0.80), readout beats PCA-10 in both modalities (0.4247 vs 0.4951, 0.3803 vs 0.4441). Not collapsed, but low-dimensional in text, dense (text 97% active), and the two modalities do not share factors (7 modality-private, retrieval ratio 0.03), which is expected with no agreement term.
- **D2, L1 alone:** not the D0 mechanism. The image codes are all zero on val (active fraction 0.0000, mean image code 0.0000), text keeps essentially one always-on factor (active fraction 0.0313, about 1/32 of code entries), 31 of 32 factors have mean activation below 1e-4, and the final training loss is 0.090 vs 0.031 for D1. PR 1.53 / 1.00 is a dead-code collapse, not copies of one axis. Since D3 keeps the same L1 weight and is healthy, L1 by itself does not cause the D0 collapse.
- **D3, no cosine agreement:** the only removal that moves anything. All gates pass except pair retrieval (0.3974 < 0.5). Spanning 0.812 is close to its 0.75 gate (6 topic-like factors), and text active fraction 0.2918 is under its 0.375 gate.
- **D4, no balance loss:** collapses harder and differently: 1 dead, 2 modality-private, top-2 mass 0.886, active fraction 0.10 / 0.12, PC1 0.91. Its max |r| 0.922 rests on only one pair because most factors carry almost no signal. Balance is not the cause; it is what keeps D0's copies alive and spread.
- **D5, no graph term:** same collapse (PR 1.31 / 1.29, 248 of 496 pairs >= 0.9), but codes become denser (active 0.91 / 0.92) and pair retrieval drops from 0.373 to 0.207. The graph term contributes sparsity and retrieval, not collapse.
- **D6, centered input:** PR 1.63 / 1.66, PC1 0.76, 201/496 pairs >= 0.9, retrieval 0.262 (below D0's 0.373), readout worse than D0. Not a fix. For context, 59% (image) and 71% (text) of the mean squared feature norm on the train rows is one common mean vector (row norms about 10, mean-vector norms 7.75 and 8.37), so the centering was a large transformation, not a no-op, and still changed nothing structural.

### Split-effect comparison: D0 (painting split) vs Task 1's collapsed codes (row split)

| | Task 1 (row split, eval = held 61,745) | D0 (painting split, eval = val 30,872) |
|---|---:|---:|
| PR img / txt | 1.3242 / 1.3262 | 1.3420 / 1.3090 |
| max abs r | 0.99977 | 0.99934 |
| pairs >= 0.9 | 374/496 | 371/496 |
| readout img (PCA-10) | 0.4936 (0.4943) | 0.4935 (0.4951) |
| readout txt (PCA-10) | 0.4669 (0.4452) | 0.4666 (0.4441) |
| active img / txt | 0.7133 / 0.7176 | 0.7191 / 0.7351 |
| top-2 mass share | 0.0782 | 0.0790 |
| spanning | 1.000 | 1.000 |
| code R@10 / CLIP R@10 / ratio | 0.1276 / 0.3440 / 0.3709 | 0.1274 / 0.3414 / 0.3732 |

The collapse is the same on both splits, so the row-split leakage did not cause it and moving to the painting split does not cure it.

## Caveats

- **Single seed, single run per variant.** The rule is applied literally, but D3's margin is large (13 times) while D4's and D5's "no effect" readings are small differences around a collapsed baseline. Nothing here estimates seed variance.
- **D1's "not collapsed" is a close call on text.** Min PR 3.2002 vs the pre-registered 3.0; the image side is 5.98. The reading follows the rule, but a different seed could land under 3.
- **Removal is not a repair.** D3 fails pair retrieval and has only a modest margin on community spanning (0.812 vs 0.75). It shows which term to replace, not what to replace it with.
- **Retrieval noise.** Val has 30,872 rows, so retrieval uses 30 pools of 1,000 (the last 872 rows unused). Differences of a few hundredths in the ratio (D0 0.373 vs D3 0.397) are within what pool sampling could produce; the split-effect comparison uses 30 vs 61 pools for the same reason.
- **Why the cosine term collapses is inferred, not tested.** Cosine agreement is scale-free and non-negative codes make it easiest to satisfy by putting every row on one shared direction. The data are consistent with that (PC1 share 0.86-0.87 with the term, 0.11-0.12 without) but no variant isolates the geometric mechanism itself.
- **Interactions are only partly covered.** Each removal is from D0. D3 was not repeated without the balance loss, and D2 was run without balance, so the dying-factor readings in D2 and D4 are single-variable but confounded by what else was absent.

## Reproduction

`/root/miniconda3/envs/CoSiR/bin/python src/test/20261009_factor_collapse_diagnosis/run_diagnosis.py` (about 5 minutes on an RTX 3090), then `make_tables.py` for the tables. Per-variant JSON, the cached train-row graph and labels, and logs are gitignored under the same folder. Log: `src/test/20261009_factor_collapse_diagnosis/20261009_factor_collapse_diagnosis_log.md`.
