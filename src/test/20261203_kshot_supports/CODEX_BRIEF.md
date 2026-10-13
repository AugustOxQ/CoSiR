Build and run one small, cheap exploratory check in /project/CoSiR: AFF and its comparators with fewer example pairs
("k-shot"). Exploratory: it decides nothing, but its numbers will be read by the user, so correctness matters. Keep it
lean: the smallest code that answers the question, no extras beyond the outputs listed.

## Rules for this job
- Write only new files under /project/CoSiR/src/test/20261203_kshot_supports/ (create it). Read anything else. Do not
  edit, move or delete any existing file; do not run git; do not create .ccg task records.
- Python: /root/miniconda3/envs/CoSiR/bin/python with PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  on every call. CPU only; no GPU. Never touch held rows (this check uses only seed-42 development episodes).
- Round folders are put on sys.path, so every new module in this folder gets the prefix `ks_` (e.g. ks_episodes.py,
  run_ks.py is fine for the runner). Reuse code from earlier folders by import; do not copy-edit them.
- Keep printed output short; long logs go to a file in the folder.

## The question
An aspect episode today has a query, 4 support pairs (share the query's value of aspect A), 4 contrast pairs (share
its value of aspect B) and 13 candidates (target, other-aspect candidate, 11 negatives). How do AFF and the baselines
change when the method sees only k support pairs and k contrast pairs, k in {2, 3, 4}? The 13 candidates stay as they
are.

## Design (fixed; do not change it)
1. **Episodes:** the existing seed-42 development episodes (12,288 = 4,096 per aspect pair; the same ones round 6's
   picks were made on; `src/test/20261125_artelingo_held_test/prep_episodes_seed42.py` and its output
   `results/dts_inputs/episodes_seed42.npz`, or wherever r6 code loads seed-42 episodes from; check the per-pair
   SHA-256s the r6 code checks). For k < 4, keep **the first k** support pairs and **the first k** contrast pairs of
   each episode; the query, the 13 candidates and the target stay identical. So the k values are nested and paired.
   Condition b swaps supports and contrasts as usual.
2. **Scorers, all frozen at their seed-42 picks exactly as round 6's held read applied them** (the pick made on tune
   half h scores parity 1 − h): cosine, RCA, B, B′(A0) ("B0"), AFF (cells 39 / 119), AFF's matched control ("CF",
   cells 149 / 10). No other scorer. Picks: `src/test/20261125_artelingo_held_test/results/picks_seed42.json`,
   `src/test/20261125_artelingo_held_test/DECISION_RULE.md` §4, `r6_picks.py` (load_picks, frozen_lambdas). Readers,
   heads, posteriors, the A3 factor checkpoint and τ_0 to τ_3 are all frozen; nothing is re-fitted or re-picked.
3. **Reuse the round 6 scoring path:** `src/test/20261125_artelingo_held_test/` (r6_context.py, r6_bundle.py
   build_bundle_r6, r6_score.py score_seed / frozen_cells, r6_stats.py, run_r6_held.py and run_r6_picks.py show how
   they are called). Find every place that assumes 4 pairs (episode arrays, the reader's 18 features: means, sample
   spreads with ddof 1 and the share of support pairs whose image and caption top groups coincide, the factor term
   centred on the episode's example items (2k items instead of 8), RCA's example-pair statistics, any validate or
   assert on NUM_PAIRS in src/eval/aspect_episodes.py or src/eval/aspect_quick_checks.py). Where code hard-codes 4,
   wrap or re-implement that piece in a ks_ module with k as a parameter; never edit the original.

## Checks (stop and report if one fails)
- C1 **k = 4 reproduces round 6's seed-42 frozen scores exactly**: per-scorer pooled R@1 equal to the recorded
  seed-42 values (r6's frozen-equals-crossfit check, `r6_picks.frozen_equals_crossfit`, and the recorded means: AFF
  19.136555989583336, B 18.341064453125, B′(A0) 18.436686197916664; find the CF, RCA and cosine values in the r6
  results and check them too).
- C2 **cosine is identical at every k** (it does not use the examples).
- C3 at k = 2 and 3, assert that every example-dependent array has exactly k support and k contrast pairs.

## Outputs (all in the new folder)
- `results/kshot_seed42.json`: per k and per scorer, pooled over the three pairs: R@1 and condition gain; per aspect
  pair, point values only. With 95% bootstrap intervals over anchor paintings (5,000 resamples, the r6 stats code and
  its clustering), only these: per k, AFF − B′(A0) and AFF − CF in R@1, and AFF − CF in condition gain; paired across
  k, X(k) − X(4) in R@1 and in condition gain for AFF and B′(A0). Diagnostics per k (point values): AFF's gate-open
  share per condition (a, b) at τ_0 and τ_2. Runtime.
- `20261203_kshot_supports_log.md`, short: purpose, design, how to run, the three checks and their results, the
  results tables, runtime, and any deviation from this brief. No figure.

## Final message (short)
The checks' results, the pooled table (R@1 and condition gain per scorer per k; AFF − B′(A0) and AFF − CF with
intervals), the runtime, every place you had to re-implement because of a hard-coded 4, and anything you were unsure of.
