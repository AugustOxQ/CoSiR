# Claim test, stage 2: D2 passes the full-scale held emotion bar

> 2026-10-20 (Amsterdam; sequence-date filename) · Spec: [claim test](../../../superpowers/specs/2026-10-20-claim-test-design.md) · Rule: [binding @ 7d82e91](../../../../src/test/20261020_claim_test/DECISION_RULE.md) · Run log: [runs 9–10](../../../../src/test/20261020_claim_test/20261020_claim_test_run_log.md) · Held ledger: [H12](../../../superpowers/held_ledger.md) · Previous: [stage 1](2026-12-18_claim_test_stage1.md)

## Results brief

**What we tried.** We tested D2 against the strongest uncollapsed PercepT on 12,281 held paintings, with stage 1’s rule.
Every arm retrained on the 42,969 train paintings with five seeds.
D2 uses a shared first layer, one head per aspect, graphs, and distillation toward the instructed Qwen image.

**Outcome against the bar.** Stage 2 passes. Adjusted mutual information (AMI) measures cluster agreement with labels.
D2 scores **0.1008**, against **0.0874** for core's own assignment, the strongest entry. Intervals use the paired 95% bootstrap.

The difference is **+0.0133 [+0.0081, +0.0146]**.
The bar requires d at least **0.0100** and a lower bound above **0.0000**.

**Rule consequence.** The rule requires the paper's stage 2 sentence, then ArtELingo-28's facts brief and spec.
No paper settings entry reaches D2, so none becomes the comparator to improve against.

**What it does not show.** Spec §6 excludes style, genre, other datasets, other backbones, the instructed Qwen bar, and retrieval.
It also excludes PercepT with its own encoders.
The split is grouped by painting; whether artists are shared between train and held was not measured (spec §8).

**Options for next.** These can sit beside the required next step.

1. Choose a cluster-level loss for track 2. Cost: decision B and development work. It tests whether that objective helps.
2. Prepare the 11-10 gate items. Cost: separate development and tests. They test the instructed Qwen bar and retrieval by aspect name.
3. Investigate Leiden. Cost: development graph diagnostics. It could explain the dependence on clustering method.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The emotion claim passes with a smaller lead than stage 1. Leiden does not support a lead across clustering methods.
D2 misses the style and genre targets.

**Our recommendation.** Follow the rule: write the paper sentence and prepare ArtELingo-28’s facts brief and spec.
Prepare the 11-10 work beside it. Track 2 waits for your decision B.

</details>

## What ran

The local run passed; Codex agreed. See [model sheet](2026-12-16_model_sheet.md) and [glossary](../../../../GLOSSARY.md).

| Data or code | Value |
|---|---|
| Train paintings / pairs | 42,969 / 216,107 |
| Validation paintings / captions | 6,152 / 30,872; D2 label-free early stopping |
| Held paintings / label captions | 12,281 / 61,744; no held caption enters a model |
| Model seeds | 0 to 4; fixed recipes, all retrained |
| k-means | 20 clusters, random states 0, 1, 2; average within model seed |
| Code at verdict | `c2eabfb81e91313944a2ea8bfe555c9826f8bb85`; dirty run log only |
| Run root / device | `outputs/stage2/`; main checkout; local RTX 3090; CPU scoring |


| Step / type | Device and timing | Result |
|---|---|---|
| Stage 2a regression | RTX 3090, 16:44–16:57, about 13 min | Hash, code, collapse and replay checks pass |
| GPU queue / claim test | 19:00:08–22:28:34 | Exit 0 |
| Feature check | CPU, 3 s | Stores reused |
| Graphs | CPU, 197 s | 42,969 nodes |
| Regression require-pass | 4 s | Earlier pass reused |
| D2 | RTX 3090, 598 s | All seeds and aspects |
| PercepT core | RTX 3090 + CPU, 3,558 s | All arms/seeds |
| Paper settings | RTX 3090 + CPU, 8,014 s | Topics, both cuts, classifiers |
| Collapse check | CPU, 13 s | Saved before held scoring |
| Held inference | RTX 3090, 119 s | Five one-topic skips |
| CPU queue | 22:28:46–23:02:06 | Require-pass 4 s; verdict 126 s; diagnostics 1,870 s |
| Independent check | Codex, about 7 min; compute 198 s; compared 22:38 | Exact agreement |

Queue timings are Amsterdam time. The GPU queue includes CPU work; extraction finished earlier with user authorization.
Diagnostics followed the verdict. Codex received no main scores. Its `choices` abort on file/order errors and use exact ties (none occurred).
Regression and seed history rely on the run log.

## Results

D2 exceeds every eligible entry (comparator plus reading). Core's own assignment sets the bar.
“Own” takes the largest topic probability; k-means uses normalized probability vectors. Scores are held emotion AMI means.

| Entry ID | Comparator | Reading | Eligible seeds | Comparator AMI | D2 AMI | D2 minus entry |
|---|---|---|---|---:|---:|---:|
| B1-r20 | Core, P3 | k-means | All five | 0.0799 | 0.1008 | +0.0209 |
| B1-own | Core, P3 | Own assignment | All five | 0.0874 | 0.1008 | +0.0133 |
| B4-r20 | Paper K = 40, P1 | k-means | All five | 0.0699 | 0.1008 | +0.0308 |
| B4-own | Paper K = 40, P1 | Own assignment | All five | 0.0711 | 0.1008 | +0.0297 |
| B5-r20 | Paper K = 40, P4 | k-means | All five | 0.0644 | 0.1008 | +0.0364 |
| B5-own | Paper K = 40, P4 | Own assignment | All five | 0.0707 | 0.1008 | +0.0301 |

All ten strict runs collapsed; both strict comparators are out. Counts/shares are below; fixed-count seeds remain eligible.

| Outcome | Statistic | Bar | Result |
|---|---|---|---|
| S2-GO | d = +0.0133 [+0.0081, +0.0146] | d ≥ 0.0100; lower bound > 0.0000 | GO; full-scale claim licensed |

**Verdict under the rule:** GO; all validity checks hold. Unrounded verdict governs (linked below).

Paper sentence (spec §6): On 12,281 held ArtELingo paintings, a model trained on the 42,969 train paintings and read from the default Qwen image gives emotion clusters (k-means, 20 clusters) with AMI at least 0.01 above the strongest uncollapsed PercepT reimplementation trained on the same paintings, each with its own recipe on our feature stores (point estimate over 5 seeds; paired 95% interval of the difference, over paintings and seeds, above 0).

| Emotion comparison | Stage 1 AMI | Stage 2 AMI | Stage 2 minus stage 1 |
|---|---:|---:|---:|
| D2 | 0.1001 | 0.1008 | +0.0007 |
| B1-r20 | 0.0779 | 0.0799 | +0.0020 |
| B1-own | 0.0855 | 0.0874 | +0.0019 |
| B4-r20 | 0.0693 | 0.0699 | +0.0007 |
| B4-own | 0.0707 | 0.0711 | +0.0004 |
| B5-r20 | 0.0652 | 0.0644 | -0.0008 |
| B5-own | 0.0687 | 0.0707 | +0.0020 |
| d, paired 95% interval | +0.0146 [+0.0093, +0.0157] | +0.0133 [+0.0081, +0.0146] | -0.0013 |
| Strict entries, both readings | Ineligible | Ineligible | All collapsed |

| Check | Evidence |
|---|---|
| V1 | Local stage 2a passed before full training/held inference; regression hashes equal |
| V2 | Independent partitions, eligibility, means, d, bounds and 2,000 draws equal; AGREE |
| V3 | 12,281 finite rows per candidate/core set, held order; D2 norm error ≤ 3.7e-5; list SHA matches; counts below |
| V4 | No seed changed after held outputs; no non-finite paper output |

Emotion labels are caption modes, with ties to the smallest code in label-name order; none are missing.

| Emotion | Held paintings | Emotion | Held paintings | Emotion | Held paintings |
|---|---:|---|---:|---|---:|
| amusement | 1,818 | anger | 153 | awe | 2,448 |
| contentment | 4,244 | disgust | 356 | excitement | 543 |
| fear | 967 | sadness | 1,021 | something else | 731 |

ID glossary: K1=D2; B1=core; B2/B3=strict; B4/B5=paper K=40; P1/P4=CLIP+RoBERTa/Qwen topics, CLIP patch classifiers; P2/P3=CLIP global/Qwen classifiers; r20/own=k-means/top topic; d=D2 minus maximum entry; S2-GO=stage 2 GO; R-COL=collapse; V1–V4=validity; V2 also names the check folder.

## Analysis

**Our reading:** the emotion claim holds under k-means; the lead depends on clustering method.
Sources: stage 2 diagnostics (linked below), [stage 1 diagnostics](../../../../src/test/20261020_claim_test/diag_stage1.json). Diagnostics decide nothing.

The bootstrap resamples paintings and seeds, retaking the eligible-entry maximum each time.
Its mean is **+0.0113**; the point **+0.0133** lies in the upper part of the interval, above 1,776 of 2,000 draws.
**Our reading:** the maximum helps explain this downward placement.

| Emotion seed means | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Sample sd |
|---|---:|---:|---:|---:|---:|---:|
| D2, k-means | 0.0994 | 0.0997 | 0.1012 | 0.1019 | 0.1017 | 0.0011 |
| Core, own | 0.0844 | 0.0876 | 0.0895 | 0.0880 | 0.0877 | 0.0018 |

**Our reading:** core gained more than D2. CLIP + RoBERTa fixed-count scores rose; Qwen k-means fell and own assignment rose.
The prior expected both sides to improve and strict settings possibly to recover. Collapse persisted.
The point fits the rule's prior for d: +0.0050 to +0.0150. Interval half-width is 0.0032, below the guessed 0.0040 to 0.0080.

Leiden finds communities in a union cosine nearest-neighbor graph on held outputs. The fit-community reading assigns held paintings by neighbor vote. Neither decides anything.

| Held emotion reading | D2 AMI | Core AMI | Difference / 95% interval |
|---|---:|---:|---|
| Verdict | 0.1008 | 0.0874 | +0.0133 [+0.0081, +0.0146] |
| Leiden on held graph | 0.0757 | 0.0762 | -0.0005 [-0.0048, +0.0023] |
| k-means at mean Leiden count | 0.0980 | 0.0789 | +0.0190 |
| Fit communities, held neighbor vote | 0.0678 | 0.0748 | -0.0070 |
| Stage 1 held Leiden | 0.0766 | 0.0748 | +0.0019 [-0.0030, +0.0053] |

Style/genre use core’s stronger reading and mask missing labels; paper versions ran emotion only.

| Aspect | Labelled held paintings | D2 AMI | Stronger core AMI | Stage 2 margin, 95% interval | Stage 1 margin |
|---|---:|---:|---:|---|---:|
| Style | 12,281 | 0.3685 | 0.2297 | +0.1388 [+0.1283, +0.1427] | +0.1436 |
| Genre | 9,949 | 0.4438 | 0.4230 | +0.0208 [+0.0103, +0.0310] | +0.0079 |

**Our reading:** D2 fell on style/genre; core fell further on genre. The claim remains emotion only.

Target and encoder cluster instructed and default Qwen images directly, with k-means seeds only.

| Aspect | D2 | Target | D2 minus target | Encoder | D2 minus encoder |
|---|---:|---:|---:|---:|---:|
| Emotion | 0.1008 | 0.0921 | +0.0087 | 0.0708 | +0.0300 |
| Style | 0.3685 | 0.4026 | -0.0340 | 0.2233 | +0.1452 |
| Genre | 0.4438 | 0.5062 | -0.0623 | 0.4521 | -0.0082 |

| Emotion AMI | D2, k-means | Core, k-means | Core, own | D2 minus core own |
|---|---:|---:|---:|---:|
| Full train set | 0.1047 | 0.0802 | 0.0865 | +0.0182 |
| Held set | 0.1008 | 0.0799 | 0.0874 | +0.0133 |
| Stage 1 fit set | 0.1080 | 0.0810 | 0.0890 | +0.0190 |

Strict runs failed the topic-count rule: under 10 or over 90. Share is the largest hard-topic fraction of train pairs.
Share above 0.5 also collapses strict runs and alone decides fixed-count collapse. Dev AUC is the classifier’s macro-average ROC AUC (PercepT’s Stage 2).

| Arm | Seed | Strict kept | Strict share | Strict dev AUC | K = 40 share | K = 40 dev AUC |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 1 | 1.0000 | — | 0.0603 | 0.9116 |
| P1 | 1 | 1 | 1.0000 | — | 0.0650 | 0.9063 |
| P1 | 2 | 1 | 1.0000 | — | 0.0471 | 0.8982 |
| P1 | 3 | 8 | 0.2384 | 0.8479 | 0.0504 | 0.9059 |
| P1 | 4 | 3 | 0.4217 | 0.9091 | 0.0672 | 0.9037 |
| P4 | 0 | 99 | 0.0193 | 0.9477 | 0.0520 | 0.9427 |
| P4 | 1 | 98 | 0.0207 | 0.9474 | 0.0566 | 0.9433 |
| P4 | 2 | 5 | 0.3167 | 0.8916 | 0.0506 | 0.9434 |
| P4 | 3 | 1 | 1.0000 | — | 0.0651 | 0.9456 |
| P4 | 4 | 1 | 1.0000 | — | 0.0478 | 0.9450 |


## Deviations

There was **no amendment to the rule**. Decisions: [run 8](../../../superpowers/handoffs/2026-10-20-r10-run-8-handoff.md), [run 9](../../../superpowers/handoffs/2026-10-20-r10-run-9-handoff.md), [run 10 handoffs](../../../superpowers/handoffs/2026-10-20-r10-run-10-handoff.md).

The spec assigned node408 PercepT training; the build moved stage 2 locally. The user chose node408 about 14:55, with local fallback.
Node stage 2a hit a guard bug: it refused its default folder because stage 1 and this round share the outputs parent.
This was a guard, not verdict code. Fix `de45b87`, developed test-first, passed 376 tests and merged at `042744b`.

CPU checks found D2/core hash drift; paper artifacts matched bit for bit.
Absolute AMI differences against stage 1 follow; escape: 0.0001. Core topic-stage checks repeat classifier readings.

| Node regression output | Emotion difference | Style difference | Genre difference |
|---|---:|---:|---:|
| D2 emotion head | 0.0019 | 0.0065 | 0.0014 |
| D2 style head | 0.0001 | 0.0016 | 0.0095 |
| D2 genre head | 0.0009 | 0.0063 | 0.0168 |
| Core P1 | 0.0004 | 0.0025 | 0.0041 |
| Core P2 | 0.0022 | 0.0005 | 0.0149 |
| Core P3 | 0.0020 | 0.0004 | 0.0039 |
| Core P4 | 0.0006 | 0.0041 | 0.0054 |

[Node check](../../../../.scratch/r10-claim-test-s2/node/run1_check/stage2a_nodeartifacts.json).
Smallest error: 5.8258e-5; others exceed the escape. The node route closed before held inference.
CPU replay failures reflect unavailable CUDA; regression drift blocks the route.
The RTX 3090 reproduced stage 1 bit for bit. **Our reading:** different GPU kernels and rounding amplified by training likely caused drift (unproven).

Full-scale deep embedded clustering (DEC) uses 1,024-row chunks; stage 2a gates only the unchunked sample path.
The [non-gating check](../../../../src/test/20261020_claim_test/stage2a.json) differs in norms/probabilities; both keep and use 40 topics. Scientific notation preserves tiny differences.

| Chunked core | Emotion absolute AMI difference | Style | Genre |
|---|---:|---:|---:|
| CLIP + RoBERTa | 3.7844e-5 | 1.8259e-5 | 6.0028e-6 |
| Qwen | 2.0874e-5 | 2.0157e-7 | 3.7022e-6 |

The verdict's `dirty` flag is true. Its only dirty file was the run log, edited during the run; no code changed.

Held inference skipped five strict paper settings runs with only one topic ([held log](../../../../src/test/20261020_claim_test/outputs/stage2/logs/gpu_held.log)): CLIP + RoBERTa seeds 0–2 and Qwen seeds 3–4.
All five were collapsed under the collapse rule, so eligibility is unchanged.

## Files and storage

Nothing was deleted. The wrap-up decide chat decides storage; see [pending deletions](../../../../.scratch/pending_deletions.md).

| Location | Contents / size |
|---|---|
| [Stage 2 outputs](../../../../src/test/20261020_claim_test/outputs/stage2/) | Weights, regression, held arrays, partitions, logs; 4.6 GB (`du -sh`: 4.6G) |
| [Verdict](../../../../src/test/20261020_claim_test/verdict_stage2.json) | Scores, seeds, interval, draws, eligibility, labels, code |
| [Diagnostics](../../../../src/test/20261020_claim_test/diag_stage2.json) | Post-verdict readings and paper runs |
| [Independent check](../../../../.scratch/r10-claim-test-s2/v2/) | Code, `choices`, partitions, bootstrap; 194 MB (`du -sh`: 194M) |
| [Comparison log](../../../../.scratch/r10-claim-test-s2/v2_compare/compare_v2.log) | Full-precision AGREE |
| [Node fetch](../../../../.scratch/r10-claim-test-s2/node/run1_fetch/) | Failed regression and graphs; 486 MB (`du -sh`: 486M); pending deletion |

Candidates for the storage summary: the node fetch, the node smoke scratch, merged worktrees and branches, and the
node's inputs, environment and job folders (about 33 GB on node408). Keep the stage 2 outputs and the verdict files:
this report rests on them.
