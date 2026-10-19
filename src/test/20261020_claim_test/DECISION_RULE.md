# Decision rule: claim test of D2 against the strongest uncollapsed PercepT, stages 1 and 2

> Spec: [2026-10-20-claim-test-design.md](../../../docs/superpowers/specs/2026-10-20-claim-test-design.md) @ (commit
> filled at commit) · Written 2026-10-20 01:55 (Amsterdam) before any implementation script exists; revised 02:25
> after the review · Reviewer: a fresh Opus `xhigh` worker, 2026-10-20 02:10 (pass after fixes; all twelve findings
> applied)

The trimmed constitution has no statistics, development-bar or failure-reading principles (old C9 to C11), so this
rule states its own statistics (§3). C7, C8, C12 and C13 apply as written. Where this rule and the spec differ, this
rule governs.

## 1. Prior (C8)

Before any held number exists: d (§3) near +0.005 to +0.015 in stage 1. The development lead of D2 over PercepT core
is +0.019 (k-means k = 20) and +0.014 against core's own assignment, on the 5,000 development evaluation paintings,
which chose D2 among 13 rows after many earlier variants (spec §8), so it is optimistic. The paper settings versions
and the maximum over entries (§3) can only raise the bar. The interval half-width is not measured; we guess 0.004 to
0.008.

| Outcome | Prior, stage 1 | Prior, stage 2 |
|---|---|---|
| GO | 0.30 | 0.35 |
| Inconclusive, margin met | 0.10 | 0.10 |
| Inconclusive | 0.40 | 0.35 |
| NO-GO | 0.20 | 0.20 |

Stage 2's prior: more training data helps both sides; a paper settings PercepT that collapses on the sample may not
collapse at full scale.

## 2. Candidates and comparators

| ID | What | Role |
|---|---|---|
| K1 | D2 (stage 2 sheet row D2), seeds 0 to 4: its emotion head output on the default Qwen image, from the emotion head's best dev-loss state (trunk and head) | candidate |
| B1 | PercepT core, arm P3 (emotion), K = 40, seeds 0 to 4 | comparator, always eligible |
| B2, B3 | PercepT, paper settings, strict, arms P1 and P4 | comparator when eligible (R-COL) |
| B4, B5 | PercepT, paper settings, K = 40, arms P1 and P4 | comparator when eligible (R-COL) |

Each comparator has two readings: k-means k = 20 (r20) and its own assignment (own). An **entry** is one comparator
with one reading: B1-r20, B1-own, …, B5-own (up to 10). A PercepT seed s uses seed s in Stage 1 and in Stage 2.
Stage 1 trains every arm on the frozen sample's 13,400 fit paintings; stage 2 on the 42,969 train paintings (spec
§3.3).

## 3. The bar and the statistics

**Data.** H = the 12,281 held paintings in Python `sorted()` order of their names; its SHA-256 is the hash of
`'\n'.join(names).encode('utf-8')`, saved before any held inference and recorded in the verdict. Emotion label per
painting: the mode of its captions' emotion codes, "something else" included, ties to the smallest code (codes
sorted by label name). All 12,281 have one; any label < 0 is masked all the same.

**Partitions, fixed once.** For each output set (arm, model seed) the partitions are computed once on all of H and
then fixed:
- r20 (K1 and every r20 entry): MiniBatchKMeans, k = 20, random states 0, 1, 2, n_init 3, batch size 4,096, on
  float32 vectors, under the sheet scorer's thread limit. K1 vectors: the saved float16 outputs cast to float32,
  unit norm checked (|norm − 1| < 1e-3), not re-normalized. PercepT vectors: the K Stage 2 sigmoid probabilities in
  float32, divided by their row norm in float32 (`apc_data.l2`, as the core scorer).
- own: argmax over the K Stage 2 probabilities (lowest index on a tie).

**AMI.** `sklearn.metrics.adjusted_mutual_info_score(labels, partition)` with default arguments, on the masked rows,
as the sheet's scorer.

**Scores.** For an index vector I over H (with repeats) and a seed list S per arm:
- A_K1(I, S) = mean over the seeds in S of the mean over the three r20 partitions of AMI(labels[I], π[I]).
- A_e(I, S) for entry e: the same mean (three partitions for r20, one for own) over the seeds in S, drawn from the
  entry's eligible seeds (R-COL). The two readings of one comparator share its seed list.
- d(I) = A_K1 − max over eligible entries e of A_e.

**Point estimate.** d = d(I₀) with I₀ = 0, 1, …, 12,280 and each arm's full eligible seed list.

**Interval.** A paired bootstrap over held paintings and model seeds. `rng = numpy.random.default_rng(0)`; for b = 1
to 2,000, in this order: I_b = `rng.integers(0, 12281, size=12281)`; then for each arm with eligible seeds, in the
order K1, B1, B2, B3, B4, B5, S_b = `rng.choice(seeds, size=len(seeds), replace=True)` from its eligible seeds in
ascending order. The same I_b serves every arm, seed, partition and entry. The maximum over entries is taken again
in each resample. The 95% interval is `numpy.percentile(d_b, [2.5, 97.5])` (default linear method). Decisions use
unrounded values; reports give 4 decimals.

**The bar.** d ≥ 0.01 and the interval's lower bound > 0 (user, 6b).

## 4. Rules

Validity first; no outcome is stated until V1 to V4 hold.

- **V1** Stage 1 needs a passed stage 0, stage 2 a passed stage 2a (spec §3.5), on every device that trains an arm
  of that stage. Otherwise no held output is computed or read; the cause goes in the report.
- **V2** (C13) Separate code, given this rule, the held painting list, the captions, the saved Stage 1 artifacts and
  the saved outputs (never our scores), recomputes the labels, the partitions, the eligibility, each entry's
  A_e(I₀), d and the interval. It must agree with the main code: equal partitions and eligibility, and d, each A_e
  and both bounds within 1e-6. A disagreement is traced to its cause and fixed before an outcome is stated. It is
  settled by this rule's text, never by which d is larger; both values and the fix go in the report.
- **V3** Every K1 and B1 output set has 12,281 finite rows in the saved order, and K1 rows have unit norm (§3);
  otherwise V3 fails. The label counts per class are reported. The list's SHA-256 matches the one saved before
  inference.
- **V4** (seed policy) After any held output exists, no seed is added, dropped or rerun, except a rerun forced by a
  failed V check (same seed and code), which the report lists. A paper settings seed with non-finite held outputs
  counts as collapsed and is reported.
- **R-COL** (collapse; `GLOSSARY.md`) Computed from the Stage 1 artifacts and saved before any held output is
  scored. Judged per Stage 1 run (Stage 1 version, seed) and cut.
  - Strict cut (γ = 2, Algorithm 1 as `pc_fixes.kept_centers`): sort the 100 center norms in ascending order and
    take the consecutive gaps. If the largest gap (first on a tie) exceeds 2 × the mean gap, keep the centers with
    norm ≤ the midpoint of that gap; otherwise keep all 100.
  - K = 40 cut: the 40 centers with the most fit pairs (core's A3, `pc_fixes.top_used`).
  - Kept count = the number of kept centers. Hard topic of a fit pair = the argmax of its Student-t Q renormalized
    over the kept centers (lowest index on a tie), as core assigns captions. Share = the largest hard-topic count
    divided by the number of fit pairs. Fit pairs: the 67,327 pairs of stage 1; all train pairs of the 42,969
    paintings in stage 2.
  - Strict (B2, B3) collapses when the kept count is under 10 or over 90, or the share is over 0.5. K = 40 (B4, B5)
    collapses only when the share is over 0.5. B1 has no collapse rule.
  - A paper settings entry is eligible when at least one of its 5 seeds is not collapsed; its score uses those seeds
    only. A collapsed run with at least 2 kept topics still trains Stage 2, for the diagnostics only.

Stage 1 (the 10-29 gate):
- **S1-GO** d ≥ 0.01 and lower bound > 0.
- **S1-MM** (inconclusive, margin met) d ≥ 0.01 and lower bound ≤ 0.
- **S1-IN** (inconclusive) 0 < d < 0.01.
- **S1-NO** (NO-GO) d ≤ 0.

Stage 2 (full train side; K1 and B1 to B5 retrained at full scale; same statistics, margin and held paintings; the
bar's value differs because the comparator is retrained, as the user allowed, S2):
- **S2-GO**, **S2-MM**, **S2-IN**, **S2-NO** as S1-GO to S1-NO, with every arm at full scale.

What each outcome leads to:
- **The 10-29 gate** (agent default): S1-GO passes it; S1-MM and S1-IN leave it open until stage 2; S1-NO fails it.
  An S2-GO before 10-29 also passes it.
- **After stage 1:** S1-GO, S1-MM and S1-IN: stage 2 runs (user, 8b, S8). S1-NO: stage 2 does not run; back to step
  2, we talk (user, 8b).
- **After stage 2**, its outcome alone sets the next step (user, S1):
  - S2-GO: the paper's sentence (spec §6, stage 2). Next: ArtELingo-28's facts brief and spec.
  - S2-MM: improve the margin, with the change chosen on development data, then test again under a new committed
    rule (user, S6; C7).
  - S2-IN: a decide chat on what to change (agent default).
  - S2-NO: back to step 2; we talk (user, 8b).
  - In any stage 2 outcome, an eligible paper settings entry with A_e(I₀) ≥ A_K1(I₀) becomes the comparator to
    improve against (the largest, if several) (user, S1).

Leiden (spec §3.4) is reported beside every outcome and decides nothing (user, 7a). The diagnostics (spec §3.6) are
computed only after the outcome is written to the verdict file.

## 5. Frozen and rerun

- **Frozen:** D2's recipe and its sample graphs (stage 2 sheet); PercepT core's recipe, K = 40 and arm per aspect
  (core spec, A3; m5); the paper settings with spec D21; the scorer protocol and the label rule (§3); the held
  painting list; this rule. Stage 2 keeps every recipe unchanged (spec D27).
- **Rerun:** all training, seeds 0 to 4, weights saved; held inference; the scoring and the bootstrap. Stage 2 also
  reruns graphs on 42,969 nodes and extracts the missing features (spec D26).

## 6. Exceptions to the constitution

| Principle | Exception | Why | Simpler option rejected |
|---|---|---|---|
| none | — | — | — |

## 7. ID glossary

| ID | Meaning |
|---|---|
| K1 | D2, the candidate |
| B1 to B5 | PercepT core P3; paper strict P1, P4; paper K = 40 P1, P4 |
| r20, own | the two readings: k-means k = 20, and the top topic |
| d | K1's mean emotion AMI minus the maximum over eligible entries |
| S1-*, S2-* | outcomes of stage 1 and stage 2: GO, MM (inconclusive, margin met), IN (inconclusive), NO (NO-GO) |
| V1 to V4, R-COL | validity checks and the seed policy; the collapse rule |
