# Handoff: turn the NO-GO into a GO by fixing the aspect reader (and the style/genre split)

Written 2026-10-04 for a fresh agent. Read this first; it points to everything else. The user drives the decisions.

## The job

Make the example-conditioned method pass a fresh-seed GO test before the CVPR abstract (10 November 2026). The last
three days (`docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`) ended with every attempt at or
below its matched condition-free control. Exploratory diagnostics run afterwards (Section 3 below) located the
blocking part: **the reader that decides which aspect the example pairs show**. With the aspect told, the same
label-free heads already clear the strongest condition-free score by **+1.50 [1.22, 1.79] R@1** on the development
episodes; with the label-free reader they clear it by only +0.19. The plan below fixes the reader, then the one aspect
pair the representation cannot separate, then tests once on fresh seeds.

Decided with the user and not to be reopened:

- **Branch decision not taken.** The user wants a working method before choosing between the method paper and the
  analysis paper. Paper writing stays paused.
- **Seed handling light** (project memory `feedback_seed-handling-light.md`): develop on seed 42; test the final
  configuration on fresh episode seeds (next free: **49, 50, 51**), each reported and pooled. No single-look ceremony.
- **Write the decision rule before the test seeds are built**, and every rule before the numbers it governs.
- **Matched controls** (project memory `v2-matched-control-lesson.md`): every configuration is compared with the same
  score with only the condition removed. Two false passes came from controls that removed more than the condition.

## Read in this order

1. The stage report above (self-contained; Sections 2, 9 to 11 and 14 matter most).
2. `docs/reports/auto/v2/2026-11-08_new_method_quick_checks.md` (D0, N1, N2, N6, N6c, matched controls).
3. `src/test/20261109_fix_diagnostics/20261109_fix_diagnostics_log.md` and `results/diagnose_fixes.txt` (this
   handoff's evidence).
4. `src/test/20261108_new_method_quick_checks/` rule files (`DECISION_RULE.md`, `ADDENDUM_1.md` to `ADDENDUM_3_N6C.md`)
   as templates for the next rule file.
5. Project memory: `project_v2-publication-plan-pending.md` (top entries), `v2-matched-control-lesson.md`.

## 1. Where things stand (seed 42 development episodes unless noted)

| Scorer | R@1 | Gain | Either | Note |
|---|---|---|---|---|
| CLIP cosine | 12.96 | 0 | 25.92 | backbone only |
| RCA (GO bar, best raw pair metric) | 13.38 | 0.10 | 26.66 | |
| E3's uniform factor control | 16.55 | 0 | 33.09 | condition-free |
| **B = best condition-free score** | **18.34** [17.97, 18.70] | 0 | 36.68 | z(cos) + λ_u·z(T_N1u) + λ_a·z(T_6u), max-R@1 cross-fit |
| N6 reader term alone (label-free) | 16.51 | 4.41 [3.96, 4.87] | 28.61 | reads the condition |
| N6 reader on B (N6c-like) | 18.53 | 0.79 | 36.27 | +0.19 [0.01, 0.38] over B |
| **Same heads, aspect told, on B** | **19.84** | **4.56** | 35.12 | **+1.50 [1.22, 1.79] over B** |
| Label probes told (diagnostic) | 30.66 | 21.06 | 40.27 | evaluation labels |

T_N1u is A3's centered uniform factor term; T_6u the mean over the three partition heads; T_6 the reader term. All
from `src/eval/aspect_quick_checks.py`.

## 2. Why the method fails

R@1 = (either + gain) / 2. A conditioned term passes only if it adds more gain than it costs either rate. Each fusion so
far traded them almost evenly (N6c: +1.19 gain, −0.88 either, net +0.15 R@1). The diagnostics show where the either
loss comes from: **wrong aspect picks**.

## 3. The evidence (exploratory, seed 42, decides nothing; `diagnose_fixes.txt`)

- **Pick accuracy.** N6's hard reader picks the told partition (emotion to affect, style and genre to image) in 52.4%
  of rankings and under both conditions of an episode in only 28.1% of episodes. Per pair and condition: emotion ×
  style 59.5 / 54.3, emotion × genre 67.9 / 61.9, **style × genre 17.1** / 53.6.
- **Right picks are free; wrong picks cost.** On episodes where both picks were right, the reader on B gained +1.16
  [0.80, 1.54] R@1 and +2.21 gain with no either loss (+0.12 [−0.43, 0.66]). Elsewhere it lost −0.61 [−0.94, −0.29]
  either for −0.19 R@1.
- **The heads are good enough.** Told the partition, the same heads add +1.50 R@1 and 4.56 gain over B.
- **Style and genre share one partition.** The told mapping sends both to the image partition, so its gain on
  style × genre is exactly 0. The E2 AMI of the image partition is 0.397 with genre and 0.318 with style.
- **What did not help:** contrastive terms (picked partition minus the contrast-like partition: +0.07 over B), top-k
  cascades with the reader (−0.55 to −2.59 R@1), soft weighting (+0.12).
- **Why the reader errs (from the final review of the quick checks):** Δ for the affect partition averages about 0
  (|Δ| ≤ 0.003) while Δ for the image partition swings ±0.02 to 0.03, and Δ under condition b is exactly −Δ under a. The
  reader therefore mostly reads the sign of the image partition's Δ, and the affect and caption partitions win by
  default. Partitions differ in head sharpness (held-out head accuracy, image / caption heads: affect 13.5 / 34.6,
  image 92.7 / 22.9, caption 21.6 / 89.7), so raw Δ values are not comparable across partitions.

## 4. The plan

Development on seed 42 throughout; pick accuracy is measured against the told mapping as a diagnostic (it reads only
which aspect pair and condition an episode has, never a row's label). Each step: implement, report pick accuracy, the
reader on B (R@1, gain and either against B with intervals) and per pair; continue only if it beats the previous step.

### Step 0. Commit the decision rule (before any code)

`src/test/20261110_reader_fix/DECISION_RULE.md` (next free sequence date; copy `.gitignore` from
`src/test/20261023_aspect_episode_spike/`): the candidate list below, the development pass rule (R@1 and gain against
B, both lower bounds above 0, plus R@1 against the configuration's own matched control), the single configuration to
carry forward (largest min(R@1 margin, gain margin) over B among passing ones), the test (Section 5) and what follows
each outcome. Copy the structure of `src/test/20261108_new_method_quick_checks/DECISION_RULE.md` and `ADDENDUM_3_N6C.md`.

### Step 1. A calibrated reader (cheapest, highest expected value)

The reader compares Δ_h across partitions whose scales differ. Two fixes, in order:

1. **Standardised Δ (minutes).** Divide each partition's Δ_h by its spread, estimated **label-free** on pseudo-aspect
   episodes from E2's bank `src/test/20261031_pseudo_partitions/results/bank_AIC.npz` (65,536 episodes on scorer-train
   rows; fields `anchor`, `candidates`, `pairs_*`, `block_pairs`; rows are scorer-train rows, check the local-to-global
   mapping the way `src/test/20261105_method_repair_diagnostics/common.py` does). Pick argmax of the standardised Δ.
2. **A learned reader (an hour).** A multinomial logistic regression over a few per-episode features (for each
   partition: mean support agreement S_h, mean contrast agreement C_h, their per-pair spread, and the share of support
   pairs whose image and caption arg-max clusters coincide), trained on the AIC bank to predict which partition the
   supports share (and, as a second head, which one the contrasts share). Labels are the bank's own pseudo-aspects, so
   no evaluation label is read. Output: partition probabilities per episode and condition. Enforce the swap structure
   (condition b's probabilities from swapped inputs), and score with the expected term Σ_h P(h)·s_h or the arg-max.

**Look at:** pick accuracy against the told mapping (now 52.4%; 28.1% both conditions), the reader on B against B, and
the either change. **Kill:** neither variant raises pick accuracy by at least 10 points or the R@1 margin over B by at
least 0.3.

Reuse: `aspect_deltas`, `probe_dots`, `inferred_scores` in `src/eval/aspect_quick_checks.py`; head posteriors from
`src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz` (SHA-256 2ad75026…; selection rows only).
Training the learned reader needs head posteriors on scorer-train rows: refit the heads with `run_n6.py`'s
`fit_heads` and predict on scorer-train rows (CPU, about 2 minutes per head), or keep the posteriors fitted on the D0
row draw and predict the rest.

### Step 2. A partition that separates style from genre (representation)

Style × genre cannot be read from three partitions in which both aspects live in the image clusters (told gain 0
there). Add a fourth label-free partition aimed at style:

- **Residual image partition (cheap):** fit a ridge map from caption features to image features on scorer-train rows,
  take each image's residual (what the image shows that its caption does not say, typically style and technique),
  k-means it (k between 16 and 32), and train cross-modal heads on it as N6 did. Check its AMI with style and genre once,
  as E2 did, as a description.
- **Alternatives if the residual fails:** low-level statistics (colour and texture summaries of the image) or CLIP patch
  token statistics, k-means on those.

Rerun Step 1's reader with four partitions. **Look at:** pick accuracy on style × genre (now 17.1% under the style
condition) and the per-pair gain there. **Kill:** style × genre gain stays below 0.5.

### Step 3. Confidence gating (if Steps 1 and 2 leave wrong picks)

Wrong picks cost either; right picks are free. Add the reader term only where the reader is confident:
s = z(B) + λ·g_e·z(T), with g_e in [0, 1] from the reader's probability margin (learned reader) or the standardised Δ
margin. **Note:** the per-episode z-score removes any per-episode scale, so the gate must multiply after z-scoring, not
the raw term. Choose λ and the gate threshold by A′'s min-margin cross-fit against B.

### Step 4. The fused configuration and its matched control

Final score: z(B) + λ·z(T_reader) (with the gate if Step 3 is used), min-margin cross-fit against B
(`crossfit_nested(B, B, T, parity)`, as `diagnose_fixes.py` does). Its **matched control** is B itself extended by every
new condition-free ingredient the configuration uses (for Step 2: the fourth partition's averaged head term), built with
`crossfit_condition_free`. If the matched control rises above 18.34, the configuration must beat the higher number.

### Step 5. Contingency if Steps 1 to 4 do not pass on seed 42

- A learned fusion: a small ranker over [z(cos), z(T_N1u), the head terms, the reader's probabilities] trained on the
  AIC bank (label-free), checked against the same matched control.
- If nothing passes, the evidence still supports a strong analysis paper (branch 3): the task, the either/gain
  arithmetic, the matched-control lesson, D0, N6's reading, and the told-partition result. Report to the user.

## 5. The test (write it in the decision rule before building the seeds)

- Build seeds 49, 50, 51 with `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>` (about 200 s
  each on CPU; also scores cosine and RCA). Check that the 3 new episode SHA-256s differ from all earlier ones.
- Score the one carried configuration with its cross-fit rerun on each seed's own halves (pattern:
  `src/test/20261108_new_method_quick_checks/run_n6c.py` and `test_seeds.py`).
- **GO** = pooled over the three seeds (one cluster per painting across seeds), R@1 and condition gain both have 95%
  lower bounds above 0 against cosine, RCA, and the matched control (B plus any new condition-free ingredient). Report
  each seed alone and against C2 explicitly. Gain against cosine, B and the matched control is one number (all have
  gain 0); do not count it three times.

## 6. Pitfalls already paid for

- A control that drops more than the condition fakes a pass (N1 on 3 October; caught by the final review).
- The pick rule mean(R@1, gain) rewards trading R@1 for gain; use A′'s min-margin rule.
- Per-episode z-scoring removes per-episode scale (Step 3).
- Δ under condition b is exactly −Δ under a; a condition-antisymmetric term flips B's top two in a k = 2 cascade.
- Ties count as misses; zero-weight episodes give constant rows (`flat_share` in the runners).
- The best condition-free score uses centering on the episode's own example items. A probe found centering on 8
  random items gives 16.85 and on the global mean 17.14, against 17.95. Part of B's lift may come from the value-disjoint
  episode construction. Before the paper, rerun B with a fixed reference set and report both.
- Exploratory looks on seed 42 are not evidence; only the fresh-seed test is.

## 7. Code, data and environment

| What | Where |
|---|---|
| Episodes, metrics, bootstrap | `src/eval/aspect_episodes.py`, `src/eval/aspect_metrics.py` |
| Nested score, cross-fit, controls, readers, decision functions | `src/eval/aspect_nested.py`, `src/eval/aspect_quick_checks.py` (tests `src/test/test_aspect_quick_checks.py`, 28) |
| Seed-42 context, codes, probes, heads | `src/test/20261108_new_method_quick_checks/run_checks.py` (`EvalContext` via `run_gonogo`, `model_inputs`, `fit_probes`), `run_n6.py` (`fit_heads`, `partition_labels`, `load_posteriors`, `n6_terms`) |
| Base B and the fusion pattern | `src/test/20261109_fix_diagnostics/diagnose_fixes.py` |
| Partitions and banks | `src/test/20261031_pseudo_partitions/results/` (`partitions.npz` SHA-256 cfd57dbb…, `bank_AIC.npz`) |
| A3 checkpoint | `src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt` (SHA-256 dadfef1b…) |
| Seed ledger | `docs/superpowers/episode_seed_ledger.md` (42 development; 43, 45, 47, 48 spent; 44, 46 MLLM probes) |

- **Python:** `/root/miniconda3/envs/CoSiR/bin/python`; tests `... -m pytest <file> -q`; prefix CPU work with
  `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. Never install into the env.
- **GPU:** the local RTX 3090 is shared with other projects; all steps above run on CPU. DAS6 node404 is available to
  this project (cluster-run skill; `watch`, `logs`, `pull` need `--node node404`).
- **Git:** main; stage by explicit path; `bin/` and `docs/paper/` stay untracked; push only when the user asks or
  `cluster sync` needs it.
- **Process the user uses:** superpowers writing-plans plus subagent-driven development for implementation; ARS for
  reviews and literature; a whole-branch final review on the most capable model that re-derives every load-bearing
  number, then one fix wave and a scoped re-review (`~/.claude/rules/final-review.md`).
- **Reports:** `docs/reports/auto/v2/<sequence date>_<topic>.md` (next free sequence date 2026-11-10), one row in
  `docs/reports/reports_sum.md`, then `python scripts/check_reports_sum.py`.

## 8. Open items (not blocking)

1. Reports not yet written: the 8B MLLM probe (`2026-11-06`, numbers in its log), the candidates write-up
   (`2026-11-07`), the fix diagnostics (`2026-11-09`; the log has the numbers).
2. The ARS paper intake draft (`docs/paper/`, untracked) is paused.
3. Privileged references (CRL and Qwen3-VL-Embedding told the aspect name), needed for the K3 claim in any branch, are not
   run.
