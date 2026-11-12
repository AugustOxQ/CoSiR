# SigLIP 1 main result: paper cells complete, and the user kept the reference settings

> 2027-01-16 (sequence date) · Run: 2026-11-12, 19:30 to 23:31 · Spec: [r28 handoff](../../../superpowers/handoffs/2026-11-12-r28-run-handoff.md) · Decision rule: handoff §4, descriptive jobs and validation picks, no claim ladder · [Run log](../../../../src/test/20261112_siglip_main/20261112_siglip_main_log.md)

## Results brief

**Flags before the results.**

- **Settings (settled):** the user kept `c*`'s settings for SigLIP 1 in the paper chat at 22:50, after seeing the recommendation. These are 40 graph neighbours (`k11=40`) and learning rate 0.001.
  - Under the fallback rule, the learning rate 0.0005 (`k6`) beat `c*` by +0.0007 on validation. The six-cell mean used seed 0.
  - 80 graph neighbours (`k11=80`) led by +0.0005.
  - Both margins lie inside the reference's seed spread, 0.3849 to 0.3879. Seed 0 is its lowest seed.
  - Neither winning row was scored on test. The ablation used `c*`. No retraining follows.
- **Replay:** the Table 3 caption-tower replay was accepted at float16 resolution, rather than the brief's `1e-5` tolerance. Constitution row 4 does not let an agent loosen that tolerance alone. An independent reviewer confirmed the inference path. This confirms the path, without authorizing the changed tolerance.
- **Close calls:** SigLIP 1's validation mean was 0.3863 against SigLIP 2's 0.3856, ahead in 3 of 5 seeds. Style AMI against CLAY reverses on one seed.

| Seeds of ours above CLAY, out of 5 | Emotion | Style | Genre |
|---|---:|---:|---:|
| AMI | 5 | 4 | 5 |
| mAP | 5 | 5 | 5 |

**What we tried.** We completed the paper's SigLIP 1 cells after the user moved the main result to that image encoder. The caption tower and Qwen3 training targets stayed as before. We scored saved models, matched outside methods, checked settings, and trained four removals.

**The level reached.** All twelve jobs are complete, with no claim ladder. SigLIP 1 won the validation pick. The sheet contains 152 paper value cells.

| Job outcome | SigLIP 1 result | Comparator |
|---|---|---|
| Encoder pick, six-cell validation mean | 0.3863 | SigLIP 2 0.3856; CLIP L/14 0.3756 |
| ArtELingo test AMI, emotion / style / genre | 0.099 / 0.441 / 0.528 | Instructed Qwen3 0.089 / 0.406 / 0.501 |
| Image mAP, same order | 0.269 / 0.355 / 0.623 | Instructed Qwen3 0.268 / 0.322 / 0.612 |
| Settings, seed 0 six-cell validation mean | `k6` (learning rate 0.0005): 0.3856; `k11=80`: 0.3854 | `c*`: 0.3849 |
| Without instructed graphs, style AMI | change -0.210 | Full SigLIP 1 model 0.441 |
| Without relational term, genre plain-cosine mAP | change -0.114 | Full SigLIP 1 model 0.599 |

Text, MMArt, photos, matched comparators, per-seed values and times are complete. Figure candidates await the user. Settings stay at `c*`, as the user decided.

**What it does not show.** These descriptive scores have no new statistical test. Settings use one seed and the early-stopping validation split. They do not establish a better recipe across seeds. Text emotion remains below instructed Qwen3. Speed covers test time only.

**Options for next.**

1. Close the round with `c*`'s settings, as the user decided. Cost: no runs. The tables stay as filled.
2. Raise the learning rate question in the spec of the next round (r29), as the paper chat allows. Cost: no runs now.
3. Train both leading settings on the other four seeds and score validation. Cost: eight runs of about six GPU minutes, plus scoring. This tests whether the margins repeat. It is optional, because the user kept `c*`.

**Your reading.** Left for the user.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The requested evidence is complete. The settings lead is smaller than the reference's seed spread, so keeping `c*` fits the evidence. Instructed graphs carry style; the relational term carries genre.

**Choices that could change a verdict.** The fallback winner depends on that rule; original-rule inputs are absent. Mean and per-cell vote both pick SigLIP 1. CRL's alternative tolerance keeps its pick. Unresolved alternatives follow below. No decision line is marked “yes”.

**Our recommendation.** Keep `c*`'s settings, as the user decided. Skip the seed check unless the learning rate matters in the next round. Resolve the replay tolerance explicitly with the user.

</details>

## What ran

The round completed the twelve jobs in the handoff, with validation alone used for picks.

- **Stages and types:** descriptive scoring, validation selection, timing, a settings check and a four-removal ablation. Jobs are listed below.
- **Data and seeds:** five model seeds; settings use seed 0. Plain embeddings are deterministic. MMArt picks use the ArtELingo validation paintings it labels. Photos use each dataset's validation split.

| Scored split | Images |
|---|---:|
| ArtELingo validation | 6,152 |
| ArtELingo test | 12,281; genre 9,949 |
| Affection test | 8,501 |
| EmoSet test | 17,348 |

- **Code:** [round folder](../../../../src/test/20261112_siglip_main/), local RTX 3090. Trainer extension merged at `f8753886`; later fixes are recorded in the run log. One unattended chat continued through context compaction. No report-stage training or scoring was run.

## Results

All requested cells have sources in the [numbers sheet](../../../../src/test/20261112_siglip_main/paper_cells.md). Triples use emotion / style / genre. AMI uses class counts; retrieval uses validation-picked read-outs unless marked plain cosine. Seeded rows use five-seed means.

| Job | Result | Comparator / outcome |
|---|---|---|
| 1. Encoder pick | Six-cell validation mean 0.3863 | SigLIP 2 0.3856; CLIP L/14 0.3756 |
| 2. Text directions | All nine mAP cells filled, plus plain SigLIP 1 | Instructed Qwen3: emotion below, style and genre above |
| 3. MMArt AMI | 0.053 / 0.455 / 0.539 | Instructed Qwen3 0.049 / 0.416 / 0.526 |
| 3. MMArt image mAP | 0.202 / 0.383 / 0.628 | Instructed Qwen3 0.202 / 0.348 / 0.617 |
| 4. EmoSet photo model AMI / mAP | 0.432 / 0.483 | Plain SigLIP 1 0.267 / 0.364 |
| 4. Paintings-only photos AMI / mAP | Affection 0.150 / 0.288; EmoSet 0.374 / 0.446 | Qwen3 counterpart 0.123 / 0.276; 0.385 / 0.441 |
| 5. Seeds above instructed Qwen3 | AMI 5 / 5 / 5; image mAP 4 / 5 / 5, of 5 | Fixed instructed Qwen3 row; means in brief |
| 6. Text times | Qwen3 pass 3.616 ms; three caption heads 0.017 ms per caption | Retained image costs: SigLIP 1 19.5; instructed Qwen3 266.8 ms per image |
| 7. CLAY, style AMI / mAP | 0.437 / 0.351 | Ours 0.441 / 0.355; matched CLAY and CRL cells complete |
| 8. MMArt TURTLE AMI | 0.030 / 0.349 / 0.372 | Plain SigLIP 1 0.035 / 0.391 / 0.410; r18 values reused |
| 9–10. Settings and ablation | Tables below | Fallback winner flagged; removals scored on current reference |
| 11. Figure candidates | Figure 3 panels: 11 and 18 | Qwen3 panel 3a: 9; overlap with each old panel: 0 |
| 12. Numbers sheet | 152 value cells | Source checks passed in run log |

Text timing used one RTX 3090, batch 32, a generic instruction, 2,000 captions and three repeats. It includes tokenization, excluding model load and disk I/O. Figure 4's leading classes are amusement, Cubism and still life. Candidate row order and viewer checks passed. The user picks later.

**Settings:** the lower learning rate wins under the fallback by a small margin, and the user kept `c*` anyway.

| Setting | Six-cell validation mean, seed 0 | Change from `c*` |
|---|---:|---:|
| `c*` (`k11=40`, learning rate 0.001) | 0.3849 | 0.0000 |
| `k11=10` (10 neighbours) | 0.3797 | -0.0052 |
| `k11=20` (20 neighbours) | 0.3838 | -0.0011 |
| `k11=80` (80 neighbours) | 0.3854 | +0.0005 |
| `k6` (learning rate 0.0005) | 0.3856 | +0.0007 |

**Ablation:** graph removals chiefly hurt style; the relational removal chiefly hurts genre. Changes subtract unrounded means. Changes below half the displayed unit appear as `0.000`.

| Removal | AMI change, SigLIP 1 | AMI change, Qwen3 | Plain-cosine mAP change, SigLIP 1 | Plain-cosine mAP change, Qwen3 |
|---|---|---|---|---|
| Instructed graphs | -0.031 / -0.210 / -0.007 | -0.033 / -0.186 / +0.010 | -0.020 / -0.143 / -0.001 | -0.021 / -0.122 / +0.001 |
| Content correction | -0.011 / -0.079 / -0.004 | -0.016 / -0.068 / 0.000 | -0.006 / -0.069 / 0.000 | -0.007 / -0.058 / -0.001 |
| Graph-positive loss | -0.033 / -0.164 / -0.006 | -0.029 / -0.140 / +0.008 | -0.018 / -0.118 / 0.000 | -0.018 / -0.098 / 0.000 |
| Relational term | +0.001 / +0.006 / -0.067 | +0.001 / -0.002 / -0.071 | 0.000 / -0.001 / -0.114 | 0.000 / -0.001 / -0.129 |
| Full reference, absolute score | 0.099 / 0.441 / 0.528 | 0.102 / 0.411 / 0.510 | 0.263 / 0.340 / 0.599 | 0.264 / 0.311 / 0.590 |
| Plain embedding, absolute score | 0.061 / 0.382 / 0.402 | 0.070 / 0.235 / 0.425 | 0.244 / 0.268 / 0.455 | 0.242 / 0.180 / 0.476 |

The first three removals lower emotion and style in every seed. The relational genre loss holds in every seed. That removal stops genre at epoch 1 throughout, as on Qwen3. Its result includes early stopping.

**Level reached under the rule:** descriptive completion. The encoder pick is confirmed. The rule's settings winner is the lower learning rate, but the user kept `c*`.

## Agent choices checked

The settings rule affects the outcome, but the user kept `c*`. Checks use recorded alternatives. The [decision lines](../../../../src/test/20261112_siglip_main/decisions.md) contain no “yes” entries. All “maybe” entries follow.

| Priority / choice | Recorded main alternative and result | Limit |
|---|---|---|
| 1. Settings fallback | Six-cell mean picks the learning rate 0.0005 (`k6`), 0.3856 against `c*` 0.3849. Original stage 3 cannot be evaluated without its P exports. | Outcome is conditional on fallback; no alternative score exists. The user kept `c*`, so no table depends on it |
| 2. Encoder mean versus per-cell vote | SigLIP 1 wins five of six cells and mean 0.3863 against SigLIP 2 0.3856 | Both rules pick SigLIP 1; only three seed means favor it |
| 3. CLAY / CRL tolerance units and read-outs | CRL keeps `center+k10` under Qwen3 units and SigLIP 1 units; CLAY picks `none+k10` | CRL alternative checked for ArtELingo; no blanket equivalence claim for CLAY |
| Photo tolerance choices | Half and double the chosen tolerance retain `center+k10` on all recorded photo rows | Each arm's own EmoSet range and the fixed Qwen3 Affection range were not separately scored |
| Instructed-graph removal | Keeps one graph on plain Qwen3, per the paper-chat default | A graph on SigLIP 1 images was not trained; attribution depends on this definition |
| Text timing includes tokenizer | Recorded 3.616 ms per caption includes tokenizer | Forward-only alternative has no recorded value |
| Figure neighbours and cluster candidates | Plain SigLIP 1 comparison, seed 0, unchanged rules; new panel counts 11 and 18, each with no old-candidate overlap | Figure choices change with the encoder; user has not selected them |
| Table 5 rounding | Difference of unrounded five-seed means, then three decimals | Difference of rounded means can differ by 0.001; it does not replace the recorded rule |

The replay tolerance is marked “no” in the log. It remains a protocol deviation that an agent cannot authorize alone.

## Analysis

**Our reading:** the main image means stay ahead of instructed Qwen3, while text emotion remains a limit. Training and the caption input still depend on Qwen3. The image speed claim does not cover every modality.

**Our reading:** the ablation pattern carries over. Style depends on graphs; genre depends on the relational term. Small gains after removal do not establish that a part is dispensable.

## Deviations

The fixes preserved test cells. The fallback is settled by the user's choice. The replay tolerance remains a caveat.

- **Settings fallback:** the user kept `c*`'s settings after the check (paper README, 2026-11-12). The handoff allowed the six-cell mean when inputs were missing. P exports are missing: 29,232 paintings, including 23,080 training paintings. The corrected decision and scorer docstring rest on P alone. The fixed constant and seed count are not missing inputs. The JSON retains the earlier explanation.
- **Replay tolerance:** the brief required `1e-5`. The worker accepted at least 99.5% equality, the rest one float16 step apart, maximum `6.1e-5`. Part 1 independently confirmed the inference path. Constitution row 4 reserves this tolerance decision for the user.
- **CUDA failure:** `k11=20` failed at epoch 34 with an unspecified launch failure. A rerun with identical seed and code completed before settings scoring. It reproduced the earlier epoch lines. The stage summary retains the original failure.
- **CLAY overwrite and guard:** part 1 found shared pick tables overwritten by the CLIP L/14 replay. Tables became per encoder. The guard then crashed on a clustering rule without units. It was narrowed to retrieval rules and tested. All 18 picks stayed unchanged; CLIP L/14 replay matched all 24 cells exactly. Test rescoring was unnecessary.
- **Sheet drift:** the paper chat changed the tables after the sheet was written. The checker gained `--values-only`.

| Snapshot | Change and fix |
|---|---|
| Tables 2, 3, 4, 6, 7 | Moved to SigLIP 1; skip old snapshots |
| Table 5, part 2 finding | Also stale; now skip every snapshot and check exact sign style. Planted missing-plus, negative-zero and wrong-digit errors were caught. |
- **Part 2 follow-up:** a tested final CLAY check rejects CLIP L/14 unit rules for other encoders. The fallback explanation was corrected. Named loss logs remain absent; no retraining was done. Removal evidence is configs, code hashes, step counts and total losses. Identical imported-code hashes resolve the dirty-record flag. The log's negative zero appears as `0.000` here.
- **Review limits:** part 2 passed, with 162 tests and exact selected-value checks. It did not replay every removal cell, observe each loss term, or bootstrap settings.
- **Context:** the chat compacted after the GPU stage. It continued as one chat; there was no chat cut.

Source: [final review part 2](../../../../src/test/20261112_siglip_main/logs/review_part2.md), with part 1 and both fix waves summarized in the run log.

## Claim review

Review on 2026-11-12, against the round's records. Values below come from the record files, not from this report's tables.

| Brief value | Recomputed value | File |
|---|---|---|
| Encoder pick: SigLIP 1 0.3863, SigLIP 2 0.3856, CLIP L/14 0.3756 | Mean of the six cell means: 0.3863, 0.3856, 0.3756 (margin 0.0007) | `records/valpick.json` |
| SigLIP 1 ahead of SigLIP 2 in 3 of 5 seeds | Per-seed differences +0.0023, +0.0005, -0.0013, -0.0005, +0.0027: 3 of 5 | `records/close_calls.json` |
| Seeds of ours above CLAY, AMI 5 / 4 / 5 | 5 / 4 / 5 (style seed 3 below by 0.00001); image mAP 5 / 5 / 5 | `records/close_calls.json` |
| Settings: `k6` +0.0007, `k11=80` +0.0005; seed spread 0.3849 to 0.3879 | +0.0007, +0.0005; c* seed means 0.38489 to 0.38787 | `records/settings_check.json` |
| Instructed graphs removed: style AMI -0.210 | -0.2101 | `records/ablation_siglip1.json`, `deltas_vs_cstar` |
| Relational term removed: genre plain-cosine mAP -0.114 | -0.1143 | `records/ablation_siglip1.json`, `deltas_vs_cstar` |
| Qwen3 text pass 3.616 ms per caption; three caption heads 0.017 ms | 3.6162 and 0.0167 (mean of three repeats) | `records/time_text.json` |
| Seeds above instructed Qwen3: AMI 5 / 5 / 5, image mAP 4 / 5 / 5 | 5 / 5 / 5 and 4 / 5 / 5 | `records/per_seed.json` |

Other results (test AMI and mAP, MMArt, photos, ablation table, the 152 sheet cells) agree with the records and with `paper_cells.md`. The checker `r28_cells_check.py --values-only` passes.

**Decision lines.** `decisions.md` has no line marked "yes", in either of its last two columns. The folded block needs no such line. Its "No decision line is marked yes" is correct. Every "maybe" line appears in the table "Agent choices checked".

**Corrected in this review.**

- The settings flag is closed: the user kept `c*`'s settings at 22:50 in the paper chat (paper README, row of 2026-11-12; run log, 23:31). The report no longer asks the user to decide about retraining.
- The run end time read "about 23:59". The run log ends at 23:31.
- The Qwen3 text time lacked its unit in the results table.
- The settings rows now say what `k11` (graph neighbours) and `k6` (learning rate) are.

## Files and storage

Records and models remain. Nothing was deleted.

- **Run folder:** [scripts, tests, briefs and logs](../../../../src/test/20261112_siglip_main/).
- **Records:** [all job records](../../../../src/test/20261112_siglip_main/records/), including picks, settings, ablation replays and figure candidates. The numbers sheet maps cells to record keys.
- **Large outputs:** `/data/SSD2/cosir_r28/20261112_siglip_main/` is **10G**, from `du -sh` during this report. It holds models, exports, scores and `qual/viewer/index.html`.
- **Would remove, pending user storage decision:** the r28 [pending-deletion rows](../../../../.scratch/pending_deletions.md) list these items and sizes. The merged trainer worktree (136 MB) and its branch are also listed there.

| Item under the large-output folder | Size | Reason it could go |
|---|---:|---|
| `clay_crl/stage1/` | 48 KB | Overwritten shared pick tables, superseded by per-encoder tables |
| `clay_crl/picks_{siglip1,clip_l14}.before_fix1.json` | 16 KB | Before-fix records; all picks unchanged |
| `smoke/` | 834 MB | Smoke outputs; round logs retain the checks |
