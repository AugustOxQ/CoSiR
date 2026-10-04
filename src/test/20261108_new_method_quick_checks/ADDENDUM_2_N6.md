# Addendum 2: N6 pre-specification (committed before any N6 code exists or any N6 number is read)

**Date:** 2026-10-04 (overnight, fully automated at the user's request). **Why N6 now:** the fresh-seed test of
N1-nested-A3 was NO-GO under `DECISION_RULE.md` §6 as committed (pooled R@1 minus its declared control +0.07
[−0.04, 0.17]; gain minus RCA's +0.14 [−0.02, 0.29]), and the final review showed its seed-42 pass came from
centering, not from reading the condition (`ADDENDUM_1.md` §1). By `ADDENDUM_1.md` R4 the re-applied table governs:
no N1 or N2 configuration passes and D0 read "close", so **row 2: build N6** (spec §5).

## 1. N6

- **Partitions (label-free):** E2's frozen pseudo-partitions `src/test/20261031_pseudo_partitions/results/partitions.npz`
  (SHA-256 cfd57dbb…): affect (k-means on the GoEmotions probabilities of the captions), image (k-means on CLIP image
  features), caption (k-means on CLIP caption features), 64 clusters each, one label per scorer-train row in
  `artelingo_splits().scorer_train` order (alignment asserted through `local_groups`, as the A′ stage did). No
  evaluation label is read anywhere in N6.
- **Heads:** for each partition h and modality m (image, caption), `LogisticRegression(C=1.0, max_iter=300)` on
  unit-normalised CLIP features of the same 60,000 scorer-train rows D0 used
  (`np.random.default_rng(0).choice(scorer_train, 60000, replace=False)`), target = the row's cluster under h. The two
  heads of one partition must have the same classes. Posteriors on selection rows, NaN elsewhere.
- **Reader (D0's rule, its decision variant "hard"):** Δ_h = S_h − C_h, the mean within-pair posterior agreement
  p_h(image)·p_h(caption) over the 4 support pairs minus the same over the 4 contrast pairs; pick the partition with the
  largest Δ_h. **T_6** = p_h(query)·p_h(candidate) on the picked partition, each item with its own modality's head.
  T_6soft (D0's soft weighting with uniform fallback) is descriptive.
- **Condition-free version:** **T_6u** = the mean over the three partitions of p_h(query)·p_h(candidate). It is T_6 with
  the condition removed (same heads, uniform partition weights) and identical under both conditions.
- **Configuration N6-nested:** the cross-fitted nested score z(cos) + λ_u·z(T_6u) + λ_a·z(T_6) with A′'s grid and
  min-margin parity cross-fit (`crossfit_nested(cos, T_6u, T_6, parity)`). Its **own condition-free control** is the
  returned nested uniform control z(cos) + σ·z(T_6u); because T_6u is T_6's own condition-removed version, this control
  is already matched in the sense of `ADDENDUM_1.md` R1.

## 2. Seed-42 check (development)

Reported: term-only R@1, gain and either rate of T_6, T_6soft and T_6u; N6-nested and its control; paired differences
against the control, cosine and RCA; the partition picked per aspect pair and condition; head accuracies on the heads'
own targets. Descriptive reference: the strongest condition-free score measured on these episodes (A3's matched
control, `ADDENDUM_1.md`) and D0.

**Pass:** N6-nested passes if its paired R@1 and condition-gain differences against its own control both have 95%
lower bounds above 0 (painting-clustered bootstrap, 5,000 resamples, seed 42).

## 3. Test (only if the seed-42 check passes)

Seeds 45, 47 and 48 (already built; they were used to test N1-nested-A3, never to develop or pick anything for N6),
the same configuration with its cross-fit rerun on each seed's own halves. **GO** = on the three seeds pooled (one
cluster per painting across seeds), the paired R@1 and gain differences against cosine, RCA and N6's own control all
have 95% lower bounds above 0. Each seed is reported on its own. The gap to the strongest condition-free score is
reported beside the verdict.

## 4. What follows

- N6 fails at seed 42, or the test is NO-GO: method work under this spec ends; D0, N1, N2 and N6 are reported as
  analysis results and the branch decision goes to the user. No further method is built overnight.
- GO: report N6 to the user with the gap to the strongest condition-free score; nothing further is built.
