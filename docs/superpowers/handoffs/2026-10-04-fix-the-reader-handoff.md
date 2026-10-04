# Handoff: turn the NO-GO into a GO by fixing the aspect reader (and the style/genre split)

Written 2026-10-04 for a fresh agent; revised after a final review the same day. Read this first; it points to everything
else. The user drives the decisions.

## The job

Make the example-conditioned method pass a fresh-seed GO test before the CVPR abstract (10 November 2026). The last three
days (`docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`) ended with every attempt at or below its
matched condition-free control. Exploratory diagnostics run afterwards (Section 3) located the blocking part: **the
reader that decides which aspect the example pairs show**. Told the right partition, the same heads (trained without
ArtELingo labels) beat their own condition-free counterpart by **+1.14 [0.90, 1.41] R@1** on the development episodes;
with the current reader the margin is +0.14 [−0.04, 0.32]. The plan fixes the reader, then the one aspect pair the
representation cannot separate, then tests once on fresh seeds.

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
3. `src/test/20261109_fix_diagnostics/20261109_fix_diagnostics_log.md`, `results/diagnose_fixes.txt` and
   `results/diagnose_counterparts.txt` (this handoff's evidence).
4. `src/test/20261108_new_method_quick_checks/` rule files (`DECISION_RULE.md`, `ADDENDUM_1.md` to `ADDENDUM_3_N6C.md`)
   as templates for the next rule file.
5. Project memory: `project_v2-publication-plan-pending.md` (top entries), `v2-matched-control-lesson.md`.

## 1. Where things stand (seed 42 development episodes)

| Scorer | R@1 | Gain | Either | Note |
|---|---|---|---|---|
| CLIP cosine | 12.96 | 0 | 25.92 | backbone only |
| RCA (GO bar, best raw pair metric) | 13.38 | 0.10 | 26.66 | |
| **B = best condition-free score** | **18.34** [17.97, 18.70] | 0 | 36.68 | z(cos) + λ_u·z(T_N1u) + λ_a·z(T_6u), max-R@1 cross-fit (N6c's matched control) |
| N6 reader term alone | 16.51 | 4.41 [3.96, 4.87] | 28.61 | reads the condition |
| N6 reader on B | 18.53 | 0.79 | 36.27 | +0.19 [0.01, 0.38] over B; **+0.14 [−0.04, 0.32] over its counterpart** |
| Same heads, partition told, on B | 19.84 | 4.56 | 35.12 | +1.50 [1.22, 1.79] over B; **+1.14 [0.90, 1.41] over its counterpart** |
| Label-probe reference, told, on B (diagnostic) | 30.62 | 20.55 | 40.69 | +6.66 [6.31, 7.00] over its counterpart |

T_N1u is A3's centered uniform factor term; T_6u the mean over the three partition heads; T_6 the reader term (all in
`src/eval/aspect_quick_checks.py`). The *counterpart* of a conditioned term T is T_cf = (T under condition a + T under
condition b) / 2, the same term with the condition removed, fused on B with a max-R@1 cross-fit
(`crossfit_condition_free(B, B, T_cf, parity)`). The told term's counterpart alone already adds +0.35 [0.16, 0.55] to B.

## 2. Why the method fails

R@1 = (either + gain) / 2. A conditioned term passes only if its condition gain outruns the either rate it costs. Even
correct readers pay: the told-partition term loses 2.27 [1.88, 2.66] either against its counterpart, and label probes
told the aspect lose 4.81 against their own condition-free mean (either 40.27 against 45.08). They pass because their
gain is large (4.56; 21.06). N6's reader gains too little (0.79) for what it costs (0.51 either), because it often picks
the wrong partition.

## 3. The evidence (exploratory, seed 42, decides nothing)

- **Pick accuracy** (a label-informed development diagnostic: it reads each episode's aspect identity, which the method
  must infer, and the told mapping emotion to affect, style and genre to image was itself chosen from AMI with the
  labels; never compute it on test seeds before their verdict). N6's hard reader picks the told partition in 52.4% of
  rankings and under both conditions in 28.1% of episodes. Per pair and condition: emotion × style 59.5 / 54.3, emotion ×
  genre 67.9 / 61.9, **style × genre 17.1** / 53.6. Because Δ under condition b is exactly −Δ under a and the told mapping
  sends style and genre to the same partition, the ceiling for an arg-max reader is 83.3% of rankings and 66.7% of
  episodes with both picks right (style × genre can never have both).
- **The told partition's margin is the target.** Over its counterpart: +1.14 [0.90, 1.41] R@1, gain 4.56, either −2.27.
  Most of it comes from emotion × style and emotion × genre (its gain on style × genre is exactly 0).
- **Right picks still cost either.** On the 3,459 episodes where N6's picks were right under both conditions (emotion ×
  style 1,576, emotion × genre 1,883, style × genre 0), the told fusion lost 3.01 [2.09, 3.93] either and still gained
  1.95 R@1; N6's own fusion showed no detectable either loss there (+0.12 [−0.43, 0.66]) only because the cross-fit gave
  it a small weight. A confidence gate can reduce the cost of wrong picks but not remove the cost of reading.
- **What did not help:** contrastive terms (picked partition minus the contrast-like partition: +0.07 over B), top-k
  cascades with the reader (−0.55 to −2.59 R@1), soft weighting (+0.12).
- **Why the reader errs.** Partitions differ in head sharpness (held-out head accuracy, image / caption heads: affect
  13.5 / 34.6, image 92.7 / 22.9, caption 21.6 / 89.7), so raw Δ values are not comparable across partitions. Per aspect
  pair and condition, the affect partition's mean Δ is near 0 (|mean| ≤ 0.003; per-episode sd 0.017) while the image
  partition's mean Δ swings by about ±0.02 to 0.03; the reader mostly follows the image partition's sign (an observation
  from the quick-checks final review, not tested causally).

## 4. The plan

Development on seed 42 throughout. For every variant report: pick accuracy (diagnostic, above), the fused score on B
against B **and against its counterpart** (R@1, gain, either, with intervals), and per pair.

### Step 0. Commit the decision rule (before any code)

`src/test/20261110_reader_fix/DECISION_RULE.md` (next free sequence date; copy `.gitignore` from
`src/test/20261023_aspect_episode_spike/`), following `src/test/20261108_new_method_quick_checks/DECISION_RULE.md` and
`ADDENDUM_3_N6C.md`. It must fix:

- the candidate list (Steps 1 to 3) and, for Step 2, the four-partition told mapping used by the pick-accuracy
  diagnostic (which partition counts as correct for style), before any pick accuracy is computed;
- the **matched counterpart** of every configuration: the same fused score with T replaced by T_cf, and B extended by
  every new condition-free ingredient (for Step 2, the new partition's averaged head term; for a learned or soft reader,
  its probabilities averaged over the two conditions; for a gated score, the gate applied to T_cf);
- a **development bar** before any test seed is built: R@1 over max(B, the matched counterpart) of at least **0.5 with a
  lower bound above 0**, and gain lower bound above 0. N6's current reader on B (+0.19 over B, +0.14 over its
  counterpart) is the reference to beat, not a candidate. Every pick so far roughly halved on fresh seeds (gains 0.52 to 0.26
  and 0.26 to 0.15), so a smaller development margin is unlikely to survive;
- the single configuration carried forward (largest R@1 margin over its counterpart among those that clear the bar), the
  test (Section 5) and what follows each outcome.

### Step 1. A calibrated reader (cheapest, highest expected value)

1. **Standardised Δ (minutes).** Divide each partition's Δ_h by its spread and pick the arg-max. Estimate the spreads
   without labels on the seed-42 selection episodes themselves (both conditions pooled; this reads no label), or on bank
   episodes built from rows outside the heads' training draw. Do not estimate them on bank episodes whose rows the heads
   were trained on (those posteriors are in-sample and sharper).
2. **A learned reader (an hour or two).** A multinomial logistic regression over a few per-episode features (for each
   partition: mean support agreement S_h, mean contrast agreement C_h, their per-pair spread, and the share of support
   pairs whose image and caption arg-max clusters coincide), trained on E2's bank `bank_AIC.npz` to predict which
   partition the supports share. Labels are the bank's own pseudo-aspects, so no ArtELingo label is read. Two
   requirements:
   - **Cross-fitted heads.** The current heads were trained on 60,000 scorer-train rows (D0's draw), about a third of the
     rows the bank uses. Refit heads on half of the scorer-train paintings and build the reader's training features only
     from bank episodes on the other half (or swap halves and average), so the reader learns from out-of-sample
     posteriors as it will meet them on selection rows.
   - **Domain-shift check.** Bank supports share a partition cluster exactly; real supports share an ArtELingo value that
     the partitions only partly track (AMI 0.20 to 0.40). Published evidence against the dropped meta-conditioner N4 (a
     pseudo-task adapter that did not help real tasks) points at this gap; E3 never tested it, because method A barely
     fitted the bank. Report the reader's accuracy on held-out bank episodes beside its seed-42 pick accuracy; kill it if the
     first is high and the second does not move.
   Score with the arg-max partition or the expected term Σ_h P(h)·s_h; enforce the swap structure (condition b's
   probabilities computed from the swapped inputs).

**Kill:** neither variant raises the seed-42 pick accuracy by at least 10 points (current 52.4%, ceiling 83.3%) or the R@1
margin over the counterpart to at least 0.5.

Reuse: `aspect_deltas`, `probe_dots`, `inferred_scores` in `src/eval/aspect_quick_checks.py`; `fit_heads`,
`partition_labels`, `load_posteriors`, `n6_terms` in `src/test/20261108_new_method_quick_checks/run_n6.py`; current head
posteriors `src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz` (SHA-256 2ad75026…; selection rows only).
The bank's rows are local scorer-train indices: map them as `src/test/20261105_method_repair_diagnostics/common.py` does.

### Step 2. A partition that separates style from genre (representation)

Style × genre cannot be read from partitions in which both aspects live in the image clusters (told gain 0 there; E2 AMI
of the image partition 0.397 with genre, 0.318 with style). Add a fourth partition trained without ArtELingo labels and
aimed at style, **readable by both modalities**:

- **First kill (cheap): caption readability.** A partition the caption head cannot predict cannot show agreement in
  cross-modal pairs. Fit its caption head first; drop the partition if its held-out caption-head accuracy is near chance.
  This rules out a pure residual-of-caption image partition (caption-unpredictable by construction).
- **Candidates:** clusters of caption-predictable image directions with the genre-heavy component of the image partition
  removed (for example, project out the directions that predict the existing image clusters, then k-means what the
  caption head can still read); or colour and technique statistics of the image if captions describe them.
- **No label-guided choice.** Do not choose among partition variants by AMI with the style or genre labels. If AMI is
  computed (as E2 did, as a description), compute it once for the variant already chosen, and disclose it.

Rerun Step 1's reader with four partitions. **Look at:** style × genre pick accuracy (now 17.1% under the style condition)
and style × genre gain. **Kill:** style × genre gain stays below 0.5.

### Step 3. Confidence gating (only if Steps 1 and 2 leave many wrong picks)

s = z(B) + λ·g_e·z(T), with g_e in [0, 1] from the reader's probability margin. The per-episode z-score removes any
per-episode scale, so the gate must multiply after z-scoring. Choose λ and the threshold by A′'s min-margin cross-fit;
the matched counterpart applies the same gate to T_cf. A gate reduces the either cost of wrong picks; it cannot remove
the cost of correct reading (Section 3).

### Step 4. The fused configuration

Final score: z(B') + λ·z(T_reader) (B' = B plus any new condition-free ingredient), min-margin cross-fit against B'
(`crossfit_nested(B', B', T, parity)`, as `src/test/20261109_fix_diagnostics/diagnose_fixes.py` does), with the matched
counterpart from `diagnose_counterparts.py`.

### Step 5. Contingency if nothing clears the development bar

- An MLLM as a three-way partition reader (show the 4 + 4 pairs, ask which of three described partitions the supports
  share; a variant of candidate N5, with its privileged-reference caveat).
- A learned fusion: a small ranker over [z(cos), z(T_N1u), the head terms, the reader's probabilities] trained on the
  bank, judged against the same matched counterpart.
- If nothing passes, the evidence still supports an analysis paper (branch 3): the task, the either/gain arithmetic, the
  matched-control lesson, D0, N6's reading, the told-partition margin. Report to the user.

## 5. The test (write it in the decision rule before building the seeds)

- Build seeds 49, 50, 51 with `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>` (about 200 s
  each on CPU; also scores cosine and RCA). Check that the 9 new episode SHA-256s (3 pairs × 3 seeds) differ from all
  earlier ones.
- Score the one carried configuration with its cross-fit rerun on each seed's own halves (patterns:
  `src/test/20261108_new_method_quick_checks/run_n6c.py`, `test_seeds.py`).
- **GO** = pooled over the three seeds (one cluster per painting across seeds), R@1 and condition gain both have 95%
  lower bounds above 0 against **each of** cosine, RCA, B (or B') and the matched counterpart. All of these except RCA
  have gain 0, so gain against them is one number; do not count it several times. Report each seed alone.

## 6. Pitfalls already paid for

- A control that drops more than the condition fakes a pass (N1 on 3 October; caught by the final review).
- The pick rule mean(R@1, gain) rewards trading R@1 for gain; use A′'s min-margin rule.
- Per-episode z-scoring removes per-episode scale (Step 3).
- Δ under condition b is exactly −Δ under a; a condition-antisymmetric term flips B's top two in a k = 2 cascade.
- Ties count as misses; zero-weight episodes give constant rows (`flat_share` in the runners).
- B's cross-fit picks are tuned on the same parity halves a fusion reuses (a small second-order leak).
- B uses centering on the episode's own example items. A reviewer probe found centering on 8 random items gives 16.85
  and on the global mean 17.14, against 17.95. Part of B's lift may come from the value-disjoint episode construction.
  Before the paper, rerun B with a fixed reference set and report both.
- Exploratory looks on seed 42 are not evidence; only the fresh-seed test is. Seed 42 has been reused many times.

## 7. Code, data and environment

| What | Where |
|---|---|
| Episodes, metrics, bootstrap | `src/eval/aspect_episodes.py`, `src/eval/aspect_metrics.py` |
| Nested score, cross-fits, controls, readers, decision functions | `src/eval/aspect_nested.py`, `src/eval/aspect_quick_checks.py` (tests `src/test/test_aspect_quick_checks.py`, 28) |
| Seed-42 context, codes, probes, heads | `src/test/20261108_new_method_quick_checks/run_checks.py` (`EvalContext` via `run_gonogo`, `model_inputs`, `fit_probes`), `run_n6.py` (`fit_heads`, `partition_labels`, `load_posteriors`, `n6_terms`) |
| Base B, fusion and counterpart patterns | `src/test/20261109_fix_diagnostics/diagnose_fixes.py`, `diagnose_counterparts.py` |
| Partitions and banks | `src/test/20261031_pseudo_partitions/results/` (`partitions.npz` SHA-256 cfd57dbb…, `bank_AIC.npz`) |
| A3 checkpoint | `src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt` (SHA-256 dadfef1b…) |
| Seed ledger | `docs/superpowers/episode_seed_ledger.md` (42 development; 43, 45, 47, 48 spent; 44, 46 MLLM probes) |

- **Python:** `/root/miniconda3/envs/CoSiR/bin/python`; tests `... -m pytest <file> -q`; prefix CPU work with
  `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. Never install into the env.
- **GPU:** the local RTX 3090 is shared with other projects; all steps above run on CPU. DAS6 node404 is available to
  this project (cluster-run skill; `watch`, `logs`, `pull` need `--node node404`), for example for the MLLM reader.
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
