# CoSiR v2 code review: does each component serve Candidate A's goal?

**Date:** 2026-09-29 · **Reviewed:** `cosir-v2` @ `5d0b47b`, plus the staged, uncommitted Task 9 work · **Reviewer:** Claude (separate session from the SDD controller)

**Verdict.** Most of the code does what its docstrings say, and all 51 unit tests pass. But several components don't serve the goal, which is `s(I,T|c)` over a self-discovered factor space. The most serious problem is that **the 32-factor space is effectively one-dimensional**, so a condition has almost nothing to select. Second, **the held-out split reuses 99.5% of training images**. Recommendation: **do not start stage (d) yet.** Fix factor discovery, with effective rank as a pass/fail check, and move to a painting-level split first. Every downstream number depends on both.

Reproduce all numbers below (no training, about 1 min):

```bash
/root/miniconda3/envs/CoSiR/bin/python src/test/20260929_factor_collapse_probe/probe.py
```

The probe reads the cached seed-42 codes from Task 9 (`src/test/20261007_naive_rule_mechanism_analysis/cache/factor42_*.npy`). These are the train-item fit used by Tasks 7–9, not Task 6's full-data fit. Saved output: `src/test/20260929_factor_collapse_probe/probe_output.txt`.

## Critical

### 1. The factor space has collapsed to about one dimension

Files: `src/train/factors.py`, `src/train/train_factors.py`

| Measurement (held rows) | Image | Text |
|---|---:|---:|
| Share of centered code variance on principal component 1 (top 3 / top 5) | **86.5%** (96.9 / 98.8) | **86.5%** (96.8 / 99.1) |
| Participation ratio (effective dimensionality) | **1.32** | **1.33** |
| Active factors per row (of 32) | 22.8 | 23.0 |
| Relative L2 reconstruction, factor model (Task 6 report) | 0.561 | 0.515 |
| Mean-only baseline | 0.639 | 0.551 |
| PCA rank 1 / rank 4 / rank 32 | 0.607 / 0.546 / 0.397 | 0.517 / 0.480 / 0.374 |

Other measurements:
- 374 of 496 factor pairs have |r| ≥ 0.9 on pair codes (mean |r| is 0.865).
- Image and text codes of the same item have cosine 0.920. Two **random, unrelated** items already have 0.589.

**Reading.** Of the 32 factors, one axis carries almost all the variance. The codes are dense, not sparse. Reconstruction is only about as good as PCA rank 1 (text) to rank 4 (image). The loading signs of principal component 1 are mixed: 62.5% of factors load with the same sign. So the axis is bipolar, with blocks of factors that are near-copies or near-anti-copies of each other, rather than a single overall intensity axis.

**Probable causes in the code:**
- **Agreement loss has a trivial solution.** `paired_agreement_loss` (`factors.py:32`) is 1 − cosine to the item's *own* pair, with no negatives. A single code direction shared by every item satisfies it.
- **No term decorrelates the factors.**
- **The balance loss can be satisfied by copies.** `usage_balance_penalty` (`factors.py:93`) equalizes each factor's *mean* activation mass, and 32 copies of one axis satisfy it. Task 6's "top-2 mass 87.7% → 7.90%" fix therefore probably spread one axis across all factors rather than creating new axes.
- **Sparsity pressure is weak** (`λ_sparsity=0.01`), and the encoder input is uncentered CLIP, which carries a large shared mean. The mean alone already gives 0.64 / 0.55 relative L2.

**Why Tasks 3–6 didn't catch it.** Top-2 mass share, dead factors, modality-private factors and community-spanning can all pass on a collapsed dictionary.

**Consistent with Task 9:**
- The naive-over-uniform lift shrinks to about 3 R@1 points once weights are L1-normalized (Q1).
- The median target has a 12-factor correlated cluster (Q2).
- Seed 43 reproduces seed 42's factors at median r = 0.995 (Q3a). A near-one-dimensional solution is easy to reproduce.

**Candidate fixes:**
- Replace the agreement loss with cross-modal InfoNCE, so matched pairs must beat in-batch negatives.
- Add a covariance or decorrelation penalty (VICReg- or Barlow-Twins-style).
- Or replace the ReLU + L1 encoder with a TopK sparse autoencoder.
- Center the input features.
- **Add participation ratio / effective rank and max pairwise |r| to the factor-discovery pass/fail checks.**

### 2. The evaluation cannot yet measure the goal

- **Held-out images were seen in training.** `split_items` (`src/test/20261005_condition_ranking_evaluation/run_ranking_eval.py:42`) splits by annotation row. ArtELingo has 61,901 unique image vectors across 308,723 rows, and **99.54% of held rows have their exact image vector in the training split.** The graph, Stage 1 and the factor encoder therefore saw every held i2t anchor image. The Tasks 7–9 "item-disjoint" numbers should be read as text-disjoint, image-seen. Fix: group the split by the annotation `painting` field (or by image-vector hash).
  - Within episodes the roles are clean: 0% of held episodes reuse the anchor's or the positive's image in any other role.
- **Circularity.** Episodes are mined from the same factor codes the scorer uses. The reports acknowledge this. Only the emotion-label episodes (Task 9 Q3b) break it, and their lift is small.
- **The spec's stage (e) doesn't exist.** There is no human-judged cross-item evaluation set, so nothing yet measures the actual target: cross-item aspect matching under a condition.

## High

### 3. Candidate A doesn't use Block 1's trained model

`train_factors` receives the raw buddy content graph (`build_content_graph`). The Stage 1 student's embeddings and communities feed only a diagnostic (community-spanning). The spec says factor discovery sits "on top of Block 1's frozen output". Also, `AttentionFusionStudent.forward` (`src/model/student.py:22`) needs image **and** text together, so it cannot embed a single-modality query for i2t or t2i scoring.

Whether and how Block 1 feeds the scorer is an open design decision. Until it is made, Candidate B's question ("is the factor layer doing real work, or is Stage 1's graph carrying everything?") has no clean framing.

### 4. The working pieces live in test scripts; the module in `src` is the broken one

- `ConditionEncoder` (`src/model/condition.py`), as trained by recovery CE, ignores the condition. It matches its shuffled-condition control and reverses 0/61 swap pairs (Tasks 7–9). Its unnormalized softplus output also causes the scale mismatch with β.
- The naive rule the ledger proposes to adopt, and the `s(I,T|c)` scorer (`score_pool`, `run_ranking_eval.py:213`), exist only in dated test scripts. This goes against the spec's "function/class-formal" constraint.
- Before stage (d), move the L1-normalized naive weights and the conditional score into `src/model/`, with tests. Then either deprecate `ConditionEncoder` or keep it only with a ranking objective. Task 9 Q4c: InfoNCE from scratch gave 41/44 swaps.

## Medium

### 5. Episode roles don't mean what `mine_episodes` claims

File: `src/train/episodes.py`

- `_cosine_without_factor` (line 54) zeroes only the target factor. With a median 12-factor correlated cluster (#1), "similar on every other factor" still includes copies of the target. So the hard-negative and anchor-distractor definitions assume factors are independent, which they are not.
- The positive is a random draw from the target's high pool. Condition-only distractors come from the *same* pool, so under the condition they are as correct as the positive. Ranking the positive above them only rewards general similarity to the anchor. That is why roughly 40–60% of episodes, across every weighting in Task 9's per-role table, the oracle included, have a condition-only distractor outranking the positive.
- The positive needs a definition that actually separates it from condition-only distractors, or those distractors should not count as negatives.

### 6. No checkpoints are saved

Every validation script retrains inline. That is fine so far, but stage (d)'s "frozen/lagging snapshot" for mining needs saved models.

## Low

- `ConditionEncoder.forward` raises a `TypeError` if only one of `contrast_img_codes` / `contrast_txt_codes` is passed (`condition.py:40-46`).
- `src/model/whitening.py` is dead code after Task 6. Remove it, or mark it as a documented negative result.
- Stage 1 draws one 1024-edge batch per epoch, about 205k edge draws against 2.46M edges. Repeated or neighboring nodes in a batch act as false negatives in the in-batch InfoNCE. The existing epoch-sensitivity check covers the practical impact.
- `anti_split_penalty` and the "0/32 modality-private" check both use batch-mean definitions, so the check largely confirms the loss rather than testing it independently. Per-item splits are covered by the paired loss, but see #1.
- `train_factors` slices the local graph twice (`train_factors.py:81`, then `factors.py:52`). This is harmless.

## Per-component verdict

| Component | Works as coded | Serves the goal |
|---|---|---|
| `src/model/graph.py`: content buddy graph | ✅ | ✅ |
| `src/model/student.py` + `src/train/stage1.py` | ✅ | ⚠️ nothing downstream uses its output (#3) |
| `src/model/communities.py` | ✅ | diagnostic only |
| `SharedFactorEncoder` + factor losses | ✅ | ❌ collapses to about 1 dimension (#1) |
| `src/model/whitening.py` | ✅ | ❌ dead code |
| `ConditionEncoder` | ✅ | ❌ ignores the condition (#4) |
| Naive rule / `s(I,T\|c)` scorer | only in test scripts | ✅ small real effect; move into `src` (#4) |
| `src/train/episodes.py` | ✅ | ⚠️ role definitions (#5) |
| Swap-loss training, human-judged eval set | missing | stages (d) and (e) |

## Suggested next steps, in order

1. Treat factor discovery as **not validated**. Rerun it with a contrastive agreement loss plus decorrelation or TopK sparsity, and centered inputs. Gate it on participation ratio / effective rank and max |r| in addition to the existing checks.
2. Replace `split_items` with a painting-grouped split and rerun the Task 7/9 ranking evaluation on it.
3. Decide explicitly how, or whether, Block 1's output enters the scorer.
4. Only then: move the naive rule and scorer into `src`, fix the episode-role definitions, and plan stage (d).
