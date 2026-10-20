# Claim test, stage 1: D2 passes the held emotion bar; full-scale testing follows

> 2026-10-20 (Amsterdam; sequence-date filename) · Spec: [claim test](../../../superpowers/specs/2026-10-20-claim-test-design.md) · Rule: [binding @ 7d82e91](../../../../src/test/20261020_claim_test/DECISION_RULE.md) · Run log: [runs 5–6](../../../../src/test/20261020_claim_test/20261020_claim_test_run_log.md) · Held ledger: [H11](../../../superpowers/held_ledger.md)

## Results brief

**What we tried.** We tested D2 against the strongest uncollapsed PercepT on 12,281 held paintings. Both trained on the developed sample (13,400 fit paintings), with five model seeds. D2 is the stage 2 sheet's model: a shared first layer (the trunk), one head per aspect, graphs, and distillation toward the instructed Qwen image.

**Outcome against the bar.** Stage 1 passes. Adjusted mutual information (AMI) measures cluster agreement with labels. D2 scores **0.1001** against the strongest entry, core's own assignment, **0.0855**. Intervals use the paired 95% bootstrap.

The difference is **+0.0146 [+0.0093, +0.0157]**. The bar requires at least **0.0100** and a lower bound above **0.0000**.

**Rule consequence.** The 10-29 gate passes. Stage 2 runs with the same recipes on the full train side (42,969 paintings). After stage 2, its own outcome alone sets the next step.

**What it does not show.** Full-scale training, style and genre claims, other datasets and backbones, retrieval, and the instructed Qwen bar are untested. So is PercepT with its own encoders (CLIP L/14, ModernBERT tuned on GoEmotions). Our PercepT arms read our feature stores. The split groups by painting and identical image vectors, not by artist.

**Options for next.**
1. Run stage 2 locally overnight or on node408. Cost: a build, device regression checks, missing features, graphs, and full-scale training. It tests the margin at scale; local work uses the evening GPU window.
2. Choose a cluster-level loss for track 2. Cost: a decide chat and development experiment. It tests whether that objective helps; this track waits for your choice of idea.
3. Prepare the 11-10 gate work. Cost: development work and separate tests of the instructed Qwen bar and aspect-named retrieval. They tell whether D2 reaches the instructed Qwen image's scores on style and genre, and whether retrieval by aspect name works.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The held emotion margin fits the prior, near its upper end. It depends on the clustering method: under Leiden the lead is +0.0019 and its interval crosses 0. D2 is above the instructed Qwen image on emotion; style and genre stay below it.

**Our recommendation.** Build stage 2 with fixed recipes. Choose track 2's idea separately; develop the 11-10 work on development data.

</details>

## What ran

Stage 0 passed twice; stage 1 completed locally; the independent check agreed.

| Work | Type and scope | Recorded result |
|---|---|---|
| Stage 0 | Regression check before training and after the fix | Both passes: 24/24, bitwise equal |
| Stage 1 | Claim test, resumed across runs 5 and 6 | Saved training verified; held inference completed |
| CPU queue | Verdict, then diagnostics | 118 s / 967 s in run log; finished 06:34 |
| Independent check | Separate code, rule and artifacts; no main scores | Partitions, eligibility, scores, bounds and 2,000 draws equal |

| Data or code | Value |
|---|---|
| Fit paintings / image-caption pairs | 13,400 / 67,327 |
| Development early stopping / evaluation | 1,489, label-free loss / 5,000, stage 0 replay only |
| Held paintings / captions used for labels | 12,281 / 61,744; no held caption enters a model |
| Model seeds | 0 to 4 |
| k-means | 20 clusters; random states 0, 1, 2; mean within each model seed |
| Code at verdict | `81113a1e2cebf61ee6d8d4350f5c0e33f6a203b4`; `dirty: false` |
| Location | Main checkout, local RTX 3090 for GPU work; CPU for scoring and diagnostics |

The [model sheet](2026-12-16_model_sheet.md) defines D2; the [glossary](../../../../GLOSSARY.md) defines the PercepT versions and collapse. D2 reads the default image once; the target needs one pass per aspect.

## Results

D2 exceeds every eligible entry; core's own assignment sets the bar. An entry is one comparator read one way. “Own” takes the largest topic probability; k-means clusters normalized probability vectors. Values are held emotion AMI means.

| Entry ID | Comparator | Reading | Eligible seeds | Comparator AMI | D2 AMI | D2 minus entry |
|---|---|---|---|---:|---:|---:|
| B1-r20 | Core, P3 | k-means | All five | 0.0779 | 0.1001 | +0.0222 |
| B1-own | Core, P3 | Own assignment | All five | **0.0855** | **0.1001** | **+0.0146** |
| B4-r20 | Paper K = 40, P1 | k-means | All five | 0.0693 | 0.1001 | +0.0308 |
| B4-own | Paper K = 40, P1 | Own assignment | All five | 0.0707 | 0.1001 | +0.0294 |
| B5-r20 | Paper K = 40, P4 | k-means | All five | 0.0652 | 0.1001 | +0.0349 |
| B5-own | Paper K = 40, P4 | Own assignment | All five | 0.0687 | 0.1001 | +0.0314 |

All ten strict runs collapsed, so both strict comparators are out. Kept counts were 1, 2, 2, 98, 98 (CLIP + RoBERTa) and 98, 99, 99, 99, 99 (Qwen); the rule collapses a run under 10 or over 90. No fixed-count run collapsed.

| Rule outcome | Statistic | Bar | Result |
|---|---|---|---|
| S1-GO | d = +0.0146 [+0.0093, +0.0157] | d ≥ 0.0100; lower bound > 0.0000 | GO; 10-29 gate passes; stage 2 runs |

**Verdict under the rule:** GO; all validity checks pass. The unrounded [verdict](../../../../src/test/20261020_claim_test/verdict_stage1.json) values license the spec's stage 1 claim (§6). On 12,281 held paintings, emotion clusters (k-means, 20 clusters) from the default Qwen image, trained on 13,400 paintings, score AMI at least 0.01 above the strongest uncollapsed PercepT reimplementation trained on the same paintings. Each uses its own recipe on our feature stores, and the paired 95% interval is above 0.

| Check | Evidence |
|---|---|
| V1 | Stage 0 passed twice; the second pass covers the fixed code and ran before any held output |
| V2 | Partitions, eligibility, means, d, bounds and bootstrap draws equal at full precision |
| V3 | Required finite candidate/core rows in held order; candidate norms pass; held-list SHA matches pre-inference record; counts below |
| V4 | No seed added, dropped or rerun after held outputs; no non-finite paper output |

Emotion labels use caption modes, including “something else”; ties take the smallest code in label-name order. None are missing.

| Code | Emotion | Held paintings |
|---|---|---:|
| 0 | amusement | 1,818 |
| 1 | anger | 153 |
| 2 | awe | 2,448 |
| 3 | contentment | 4,244 |
| 4 | disgust | 356 |
| 5 | excitement | 543 |
| 6 | fear | 967 |
| 7 | sadness | 1,021 |
| 8 | something else | 731 |

ID glossary: K1 = D2; B1 = core; B2/B3 = paper strict; B4/B5 = paper K = 40; P1/P4 = CLIP + RoBERTa / Qwen topics, both CLIP patch classifiers; r20/own = k-means/own assignment; d = candidate minus largest eligible mean; S1-GO = stage 1 GO; V1–V4 = validity; R-COL = collapse; V2 also names the independent-check folder.

## Analysis

**Our reading:** the sample-trained emotion claim holds, with dependence on clustering method. Post-verdict [diagnostics](../../../../src/test/20261020_claim_test/diag_stage1.json) decide nothing.

The paired bootstrap resamples paintings and model seeds and retakes the maximum over eligible entries each time. The mean of the 2,000 resampled differences is **+0.0125**. The point **+0.0146** lies near the interval's upper end and exceeds 1,795 of the 2,000 draws. **Our reading:** the downward placement is consistent with the rule's stricter design: each draw must beat the strongest resampled entry.

Seed means first average the k-means readings; sd is the sample sd over model seeds.

| Model / reading | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Sample sd |
|---|---:|---:|---:|---:|---:|---:|
| D2, k-means | 0.1007 | 0.1007 | 0.0993 | 0.0982 | 0.1015 | 0.0013 |
| Core, own | 0.0855 | 0.0846 | 0.0859 | 0.0873 | 0.0842 | 0.0012 |

Leiden finds communities in a nearest-neighbor graph. The verdict-side reading uses a union cosine graph on held outputs; its small lead crosses zero. A second reading learns fit communities and assigns held paintings by neighbor vote. Neither decides anything.

| Held emotion reading | D2 AMI | Core AMI | Difference / interval where computed |
|---|---:|---:|---|
| Verdict, k-means versus core own | 0.1001 | 0.0855 | +0.0146 [+0.0093, +0.0157] |
| Leiden on held graph | 0.0766 | 0.0748 | +0.0019 [-0.0030, +0.0053] |
| k-means at each output's mean Leiden count | 0.1002 | 0.0778 | +0.0223 |
| Fit communities, held neighbor vote | 0.0727 | 0.0745 | -0.0018 |

**Our reading:** count alone does not explain weaker Leiden results. The fit-community vote reverses the ordering. Retain k-means in the claim.

Style and genre compare against core's stronger reading; paper versions ran emotion only. Missing genre labels are masked.

| Aspect | Labelled held paintings | D2 AMI | Stronger core AMI | D2 minus core, paired 95% interval |
|---|---:|---:|---:|---|
| Style | 12,281 | 0.3719 | 0.2283 | +0.1436 [+0.1339, +0.1491] |
| Genre | 9,949 | 0.4490 | 0.4411 | +0.0079 [+0.0002, +0.0153] |

Target and encoder cluster the instructed and default Qwen images directly. Their repeats are k-means seeds, with no model training seeds.

| Aspect | D2 | Target | D2 minus target | Encoder | D2 minus encoder |
|---|---:|---:|---:|---:|---:|
| Emotion | 0.1001 | 0.0921 | +0.0080 | 0.0708 | +0.0293 |
| Style | 0.3719 | 0.4026 | -0.0307 | 0.2233 | +0.1486 |
| Genre | 0.4490 | 0.5062 | -0.0572 | 0.4521 | -0.0031 |

**Our reading:** D2 exceeds the emotion target, misses style and genre, and falls below the raw genre encoder. The 11-10 gate needs separate tests.

The fit-set reading is in-sample; core's own assignment remains the strongest eligible entry there.

| Emotion AMI | D2, k-means | Core, k-means | Core, own | D2 minus core own |
|---|---:|---:|---:|---:|
| Fit set | 0.1080 | 0.0810 | 0.0890 | +0.0190 |
| Held set | 0.1001 | 0.0779 | 0.0855 | +0.0146 |
| Development sheet | 0.0962 | 0.0772 | — | — |

**Our reading:** the held lead is smaller than the fit lead. Development favored D2 but selected it on those paintings; paintings and seed totals differ.

The rule's prior placed d near +0.0050 to +0.0150. The observed +0.0146 is near its upper end. The interval's half-width is 0.0032; the prior guessed 0.004 to 0.008. **Our reading:** the point fits the prior, and the interval is narrower than we guessed.

V is V-measure (homogeneity and completeness); ARI is adjusted Rand index; FMI is Fowlkes–Mallows pairwise agreement. Random rows permute assignments with exact cluster sizes preserved.

| Held emotion, k-means | AMI | V | ARI | FMI |
|---|---:|---:|---:|---:|
| D2 | 0.1001 | 0.1024 | 0.0394 | 0.1471 |
| D2, random | -0.0001 | 0.0025 | -0.0001 | 0.1025 |
| Core | 0.0779 | 0.0803 | 0.0233 | 0.1327 |
| Core, random | 0.0002 | 0.0028 | 0.0001 | 0.1074 |

The strict cut kept almost none or almost all topics, and the rule excludes both extremes. Share is the largest hard-topic fraction of fit pairs. Development AUC is the macro-average ROC AUC of the image classifier (PercepT's Stage 2) on the development paintings.

| Paper arm | Seed | Strict kept | Strict share | Strict dev AUC | K = 40 share | K = 40 dev AUC |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 1 | 1.0000 | — | 0.0480 | 0.8818 |
| P1 | 1 | 2 | 0.5074 | 0.7404 | 0.0588 | 0.8949 |
| P1 | 2 | 2 | 0.6696 | 0.9588 | 0.0603 | 0.8830 |
| P1 | 3 | 98 | 0.0218 | 0.9011 | 0.0487 | 0.8730 |
| P1 | 4 | 98 | 0.0260 | 0.9040 | 0.0428 | 0.8897 |
| P4 | 0 | 98 | 0.0221 | 0.9355 | 0.0590 | 0.9271 |
| P4 | 1 | 99 | 0.0303 | 0.9337 | 0.0498 | 0.9194 |
| P4 | 2 | 99 | 0.0508 | 0.8675 | 0.1782 | 0.8619 |
| P4 | 3 | 99 | 0.0243 | 0.9281 | 0.0801 | 0.9175 |
| P4 | 4 | 99 | 0.0262 | 0.9288 | 0.0643 | 0.9163 |

All strict rows collapsed; fixed-count rows stayed eligible. The one-topic run trained no classifier; other collapsed runs trained for diagnostics. **Our reading:** high AUC cannot rescue collapsed topics; collapse could change at full scale.

## Deviations

There was **no amendment to the rule**. The pre-inference recheck fix preserved collapse judgments. See the run log and [handoff](../../../superpowers/handoffs/2026-10-20-r10-run-6-handoff.md).

| Deviation | Cause, action and effect |
|---|---|
| Run 5 R-COL failure | Duplicate deep embedded clustering (DEC) centers in `qwen_qwen`, seed 2: float64 recheck versus saved float32 counts gave L1 difference 8,892, limit 6. Stopped before held inference. |
| Fix and review | `95a8a57`: relative tolerance 1e-6 recognizes tied maxima, checks saved topics against them, sums counts by tie group. Artifacts: 10,002 tied pairs, three groups. Masks, shares and collapse logic unchanged. Sonnet fix; Opus review PASS. Tests: 67 focused; 321 suite passed, 2 skipped. |
| Second stage 0 | The code identity changed; the seed-zero retraining passed bitwise before held inference. Old artifacts moved to `outputs/stale/`. The review's optional per-center bound was left out. |
| Two chats | Run 6 verified and reused every saved artifact without retraining, then completed classifier and held work. |
| V2 substitution | Codex lacked room; a Claude Sonnet worker used the independent brief and data without main scores. |
| V2 ties | Float64 tie choices change its K = 40 kept set for `qwen_qwen`, seed 2. Strict share ours/check: 0.0508/0.0475; fixed-count: 0.1782/0.1828. Eligibility unchanged. Held scores come from our saved Stage 2 outputs. |
| Extraction time | Instructed held images: 57 minutes versus 30 estimated. Stage 1 still finished at 06:16. |

The first diagnosis named two duplicate-center pairs; the final artifacts show three tie groups, which we use.

The independent worker's `choices` say it would drop any seed with non-finite or sidecar-mismatched held output. The rule is narrower: a non-finite paper seed counts as collapsed, a non-finite candidate or core row fails V3, and a sidecar mismatch is no drop. None occurred. **Our reading:** this unused path has no effect here; future validity failures must follow the rule.

## Files and storage

The next wrap-up decide chat makes storage decisions ([handoff](../../../superpowers/handoffs/2026-10-20-r10-run-5-handoff.md)). Nothing was deleted.

| Location | Contents / size |
|---|---|
| [Round outputs](../../../../src/test/20261020_claim_test/outputs/) | Weights, artifacts, held arrays, partitions, logs, stale copies; **3.1G**, `du -sh` |
| Shared cache | `emotion_focus_held`, `style_word_held`, `genre_word_held`; **54 MB each**, run log |
| [Verdict](../../../../src/test/20261020_claim_test/verdict_stage1.json) | Scores, seeds, interval, draws, eligibility, labels, code identity |
| [Diagnostics](../../../../src/test/20261020_claim_test/diag_stage1.json) / [sheet](../../../../src/test/20261020_claim_test/diag_stage1.md) | Post-verdict readings and paper-run details |
| [Independent check](../../../../.scratch/r10-claim-test/v2/) | Code, `choices`, partitions, bootstrap; **7.4M**, `du -sh` |
| [Comparison log](../../../../.scratch/r10-claim-test/v2/compare_v2.log) | Full-precision `AGREE` |

**Would remove after the storage decision:** stale regression copies, merged worktrees and branches, smoke outputs, and fixer/reviewer scratch trees ([pending deletions](../../../../.scratch/pending_deletions.md)). Keep claim-test artifacts and the independent evidence.
