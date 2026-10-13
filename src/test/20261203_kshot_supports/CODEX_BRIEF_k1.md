Follow-up to your k-shot sweep in /project/CoSiR/src/test/20261203_kshot_supports/ (your earlier brief is saved there as CODEX_BRIEF.md; its rules still hold: write only inside that folder, no git, no .ccg records, CPU only, the same Python env and variables, short printed output). Your result passed review and is committed. The user now wants one more value: k = 1 (one support pair and one contrast pair). Add it; change nothing else.

## The one design rule for k = 1 (decided; do not change it)
The reader's two spread features per grouping (sample spread, ddof 1, of the support agreements and of the contrast agreements; 6 of the 18 features) are undefined with one pair. At k = 1, set each of these 6 features, for each half-reader separately, to that half-reader's StandardScaler mean for that feature, so that after scaling the value is exactly 0 and the feature adds nothing to that half-reader's logits. Do this inside the reader call (the two half-readers have different scalers, so one raw value cannot serve both). All other features stay as defined: S, C and Δ over the one pair, and the share of support pairs whose image and caption top groups coincide (0 or 1 at k = 1). Everything else stays frozen as before (readers, heads, thresholds, picks, cells, RCA basis). RCA's covariance (`_cov` in src/eval/pair_metric_baselines.py divides by the pair count and adds I) and the factor centring (2 items) work at k = 1 as they are.

## Checks (stop and report if one fails)
- C1 and C2 as before; C2 now includes k = 1.
- C3 at k = 1: every example-dependent array has exactly 1 support and 1 contrast pair.
- C4: at k = 1, for each half-reader and each condition, the 6 spread columns after that half-reader's scaling are exactly 0; and with k = 2, 3, 4 the reader path gives the same features and probabilities as before (the k = 1 rule must not touch k >= 2).
- C5: every k = 2, 3, 4 number in the new results JSON equals the committed one in /tmp/claude-0/-project-CoSiR/f8e5aee2-8d30-4a4a-9b25-7044a814a226/scratchpad/k1/kshot_seed42_k234_committed.json exactly (bootstrap draws are shared, so the intervals must match too).

## Outputs
- `results/kshot_seed42.json` rewritten with k = 1 added in the same fields as the other k (pooled, per pair, AFF − B0 and AFF − CF with intervals, X(1) − X(4) for AFF and B0, gate-open shares at τ_0 and τ_2), plus the k = 1 rule in a short field.
- `results/ks_run.log`: the new run's output appended below the old.
- `20261203_kshot_supports_log.md`: add k = 1 to every table, the k = 1 rule, C4 and C5 and their results, and the new runtime. Keep the rest.

## Final message (short)
The checks' results, the k = 1 pooled row (R@1 and condition gain per scorer), AFF − B0 and AFF − CF at k = 1 with intervals, AFF(1) − AFF(4), the k = 1 gate-open shares, the runtime, and anything you were unsure of.
