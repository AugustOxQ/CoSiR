# r28 final review, part 2: settings check and Table 5 ablation on SigLIP 1

Written 2026-11-12 23:28 (reviewer: Claude, read only). Brief: `briefs/review_part2.md`.

## Verdict: PASS

No finding is blocking. All four removal rows trained the removal that the Qwen3 ablation applied, and nothing else
changed. The settings rows used their setting. Every value I re-derived equals the record exactly (difference 0.0,
under the 1e-6 bar). The numbers sheet's Table 5 values are right, and the paper's `ablation.tex` already holds the
same 36 numbers. One check now fails, but only because the sheet's Table 5 snapshot is out of date (finding 1).

Tests: `pytest -q test_r28_trainer.py test_r28_settings.py test_r28_ablation.py test_r28_clay_crl.py`, run from the
round folder: 162 passed in 18.5 s. No test was skipped.

## Findings, by severity

| # | Where | What | Blocking | Changes a Table 5 number or the settings outcome |
|---|---|---|---|---|
| 1 | `r28_cells_check.py:150`; `paper_cells.md` Table 5 `now` cells | `--values-only` now gives FAIL: 12 mismatches, all of them Table 5 `now` cells; 0 of 152 value cells fail | no | no |
| 2 | `r28_cells_check.py:43-48, 145-146` (`norm`) | the checker no longer checks how signs are written: a missing `+` and a `-0.000` both pass | no | no |
| 3 | run records and `run_trainer.log` | no loss term is logged by name; only the total train and stopping loss per epoch | no | no |
| 4 | `r28_settings.py:9-14` (docstring), decision line 19:58 | the reason for the fallback is right, but only P makes stage 3 impossible; the r_a and seed reasons are weaker | no | no |
| 5 | `r28_clay_crl.py:266` | the narrowed guard is correct and covers every rule that names the CLIP L/14 unit; it has no final check | no | no |
| 6 | run log line `23:13` | the buddy genre mAP change is written `-0.000`; the sheet and the paper write `0.000` | no | no |
| 7 | `train/k11=10/run__s0.json` | `code_dirty: true` (commit d423780e) | no | no |

**1. The Table 5 snapshot is out of date, so `--values-only` fails.** At 23:15:45 the paper chat put the SigLIP 1
values into `Tables/ablation.tex` (paper commit a616b3b). That was one minute after e1ab6fdb, whose log line says PASS.
The sheet's Table 5 `now` cells still hold the Qwen3 rows. Every mismatch is a `now` cell, and every value cell passes.
My own parse of `ablation.tex` lines 11 to 18 found all 36 numbers equal to the records, rounded to 3 decimals
(`tex_t5.py` in the scratchpad). The fix: refresh the Table 5 `now` cells, or skip them as for the other tables.

**2. The sign style is no longer checked.** `norm` drops a leading `+` and turns `-0.000` into `0.000` on both sides.
I planted errors in a scratch copy of the sheet. The checker caught a wrong digit (`-0.211` for `-0.210`) and two wrong
signs (`+0.143` for `-0.143`, `-0.001` for `+0.001`). It let through `0.006` for `+0.006` and `-0.000` for `0.000`.
Both are only a matter of how the number is written, so the checker is still strict on value and sign.

**3. No per-term loss log, but other evidence shows each removal took effect.** The training loop logs only the total
loss. In its place:
- The run record's `config` and `knobs` (the patch description) equal r10's Qwen3 `k11=40-no_<removal>` records for
  all 20 runs.
- The `code_sha256` of every module the runs imported equals r14's c* run. The only differences are the entry modules
  (`bs_node.py` in r14; `r28_trainer.py` and `r28_trainer_loss.py` in r28, and these equal the files now on disk).
- The epoch-1 training loss moves the way r10's Qwen3 runs moved (table below).
- The graph removals give the same steps per epoch as Qwen3 (graphs are on Qwen3 in both).

| Row (seed 0) | Steps per epoch, E / S / G | Epoch-1 train loss, E / S / G | Qwen3 r10, same row |
|---|---|---|---|
| c* (r14) | 446 / 244 / 774 | 5.28 / 5.58 / 6.86 | 446 / 244 / 774; 5.24 / 5.64 / 5.13 |
| buddy | 446 / 244 / 774 | 0.49 / 0.49 / 3.90 | same steps; 0.38 / 0.46 / 2.27 |
| relational | 446 / 244 / 774 | 5.24 / 5.56 / 3.30 | same steps; 5.19 / 5.60 / 3.10 |
| content_correction | 691 / 832 / 774 | 4.16 / 3.57 / 6.87 | 691 / 832 / 774 |
| instructed_graphs | 775 / 775 / 775 | 2.62 / 2.61 / 6.35 | 775 / 775 / 775 |

The trainer tests check the loss term by term against r10's loss on the same tensors.

**4. P alone makes r10's stage 3 criterion impossible.** I confirmed that P (29,232 paintings) holds the 6,152
validation paintings plus 23,080 others. The full-scale fit set has 42,969 paintings (the identity line of the k = 10
graph store). That count matches 19,889 + 23,080, so the other P paintings are training rows at full scale. No held-out
export on P can exist, so the fallback is right (DECISION_RULE §3.3 and §4.3 need P, the k = 20 partitions, and seeds
1 to 3). The two other reasons are weaker. r_a is a fixed constant of the rule, not a missing input, and the brief
itself allowed the seed set {0}.

**5. The CLAY guard is correct.** Every picks file has the same keys. Only `clay_textset/ret/artelingo` and
`clay_textset/ret/mmart` name the CLIP L/14 unit, and the new key test covers both. `picks_siglip1.json` no longer names
CLIP anywhere. Suggestion: after the loop, assert that no rule names `CLIP L/14` when the encoder is not CLIP L/14. This
would also catch a future r18 rule that uses the same wording.

**6, 7. Small.** Finding 6 is a run-log line only. For finding 7: all 24 run records have one identical set of code
hashes, so the dirty files were not code that the run imported.

## The four checks the brief asked for

- **Removal rows.** Five seeds each, status `ok`, `smoke: false`. The 60 held exports carry the held-list hash
  a0b1139c..., and each export's `weights_sha256` equals both the run record and the saved `.pt` file. The scorer binds
  rows to painting names through the list hash (C6) and checks all 150 input files.
- **Settings rows.** The steps per epoch match the stores' pair counts (pairs / 512, rounded up). For k = 10: emotion
  47,732 pairs gives 94 steps, style 17,249 gives 34, genre 91,431 gives 179. For k = 80: 977 / 644 / 1,604 steps,
  also matching. The k = 20 run uses the original graph code with no graph patch, as r10 and r14 did. r10's gate
  `k11_full.json` shows that the k = 20 rebuild equals the original graphs, and the steps (202 / 92 / 373) fall
  between k = 10 and k = 40. For `k6=0.0005`, the patch description and each head's export config both give
  `lr` 0.0005.
- **The k = 20 rerun.** The rerun repeated all 77 epoch lines of the failed run exactly (the training is deterministic),
  then finished: three heads saved, record `ok`, 194.8 s. The settings check started after the rerun (21:12:34, against
  21:12:19), and its recorded file hashes equal the files on disk.
- **No test data used to choose anything.** `r28_settings.py` reads only the validation names, labels and exports, plus
  `valpick_siglip1.json`. The c* used for seed 0 is r14's validation export (path, a loader equality assert, and a gate
  against valpick at 0.0). `r28_ablation.py` reads the test split only to compute values and its gates. The c* gate is
  exact: difference 0.0 against r17's sweep record and the verdict cells for all three aspects.

Context, not a finding: the validation exports cover the paintings that each run's early stopping used. That is r14's
protocol, and the encoder pick used it too. As on Qwen3, the relational removal stops the genre head at epoch 1 in all
five seeds.

## Re-derived values

My own code did the loading, the list-hash and label binding (labels via `rt_data`, not the scorers' `ct_held` path),
the masking and the means. It imported only `ns_metrics.partitions`, sklearn's AMI and `rt_metrics.retrieval`. A
separate numpy AP gave the same result as `rt_metrics.retrieval` to the last bit on 3 exports.

| Value | Mine | Record | Abs. difference |
|---|---|---|---|
| Settings: c* (seed 0) mean of the six cells | 0.38489083 | 0.38489083 | 0.0 |
| Settings: `k6=0.0005` (seed 0) mean of the six cells | 0.38559290 | 0.38559290 | 0.0 |
| Settings: `k6=0.0005` minus c* | +0.00070207 | +0.00070207 | 0.0 |
| Settings: each of the 12 cells | | | max 0.0 |
| Held: c* style AMI at k = 27, five-seed mean | 0.44079073 | 0.44079073 | 0.0 |
| Held: instructed_graphs style AMI at k = 27 | 0.23064682 | 0.23064682 | 0.0 (per seed too) |
| Held: instructed_graphs minus c*, style AMI | -0.21014391 (-0.210) | -0.21014391 | 0.0 |
| Held: c* genre plain-cosine mAP, five-seed mean | 0.59862013 | 0.59862013 | 0.0 |
| Held: relational genre mAP | 0.48432801 | 0.48432801 | 0.0 (per seed too) |
| Held: relational minus c*, genre mAP | -0.11429212 (-0.114) | -0.11429212 | 0.0 |
| Held: buddy minus c*, genre mAP (sign check) | -0.00039186 (written 0.000) | -0.00039186 | 0.0 |

## Left out

- I did not re-run the trainer on CPU to watch each patch reach the loss. I relied on the tests, the records and the
  loss levels above.
- I re-derived 3 of the 36 Table 5 cells end to end (plus c*'s style AMI and genre mAP). I did not re-derive the other
  removal cells or the plain row.
- I did not count the k = 20 graph pairs directly; I relied on r10's gate.
- I did not run the paired bootstrap for the settings margin.

Nothing in the project needs removing. Scratch files are under
`/tmp/claude-0/-project-CoSiR/dae700bf-e7a2-4618-ae57-fa086b1e35b9/scratchpad/review2/`.
