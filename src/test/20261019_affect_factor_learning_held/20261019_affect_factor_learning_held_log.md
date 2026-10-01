# 20261019 affect factor learning: held test (plan Task 5, spec §7)

## Problem
Selection picked SE (affect clusters + CLIP image clusters as condition sources) over C0: D_emo +1.31 [+0.54, +2.06],
D_style +1.12 [+0.29, +1.93] on 4,096 selection episodes per label. Spec §7 confirms the pick once on fresh held
episodes under a criterion fixed in advance: confirmed iff the D_emo,held lower bound > 0 and the D_style,held lower
bound > -1.5 (SE seed 42 vs C0 seed 42, naive beta 0.3, mean of directions, paired bootstrap 5,000 resamples seed 42).

## Steps
1. `run_held.py` (new): imports `run_affect.py` (and through it `run_grid.py`, the probe and stage (d) helpers) by
   file path. Phases `--power`, `--smoke`, `--run` (once), `--tables [--smoke]`. Guards: the GoEmotions functions are
   replaced by a raising stub at import; `--run` refuses if `results/held_results.json` or `results/held_started.json`
   exists (the latter unless `--after-crash`, recorded), or unless `power.json` predates it and `smoke_held.json`
   records a passing smoke of the same script SHA-256.
2. `--power` (01:17 local): `held_episode_count` verbatim from the brief, hand check asserted (d 2.0, CI [1, 3] ->
   SE 0.5102, power .547 / .836, n 4096). From SE's selection CI: SE_sel 0.3894, effect 0.9796, power 42.8% / 71.1% /
   94.5% at 2,048 / 4,096 / 8,192 -> **n = 8,192 per label**. Style guard at n: SE 0.295, pass if point > -0.92;
   P(pass | true 0 / -0.5 / -1.0) = 99.9% / 92.4% / 39.5%.
3. `--smoke` (twice; the second after adding beta-0 rows for the replication seeds, so the script SHA matches the
   run): selection rows in place of held rows, seed 43, 8,192 per label; 154 s; all outputs finite; numbers
   discarded. Code check inside the smoke: the same code on the selection episodes (seed 42, 4,096) reproduced the
   stored selection D_emo / D_style exactly (identical-rank share 1.000 for SE and C0; episode SHA-256s equal).
   R3 encoded on CPU from its checkpoint is bit-identical to stage (d)'s cached codes on selection rows.
4. `--run` (01:23 local, 151 s, one attempt): split recomputed and equal to stage (d)'s cache; held rows / paintings
   disjoint from train and val; 7 checkpoints SHA- and config-asserted (SE and C0 seeds 42/43/44, R3); only the 61,744
   held rows encoded; R3 held codes bit-identical to stage (d)'s `cache/held_codes.npz`; episodes
   `standard_label_episodes(data, groups, held, label, 8192, seed=43)`, every row a held row (asserted); SHA-256
   emotion `abd1ca38e4e2daaf0aca850aafa2498af4b998cb25f678c6672a0a3273187b05`, art style
   `ee87686c0637e3c3833cd56cbc56f496b04898c7d89a4d8f1d1eebe8608db543` (differ from the earlier seed-42 held episodes;
   asserted). 8 emotion targets, 24 style targets (New_Realism appears on held only).

## Result (held, seed 42, naive beta 0.3)
| | point | 95% CI | bar |
|---|---:|---|---|
| D_emo,held | +2.08 | [+1.51, +2.62] | > 0: met |
| D_style,held | +0.65 | [+0.07, +1.21] | > -1.5: met |
| pooled | +1.36 | [+0.96, +1.76] | context |

**CONFIRMED.** Recomputed from `held_ranks.npz` with a separate bootstrap: points identical; CIs agree within 0.03
across 20 other bootstrap seeds (corrected in the final fix wave; the earlier "same numbers" held only for seed 42).
Context: naive R@1 SE 21.24 / C0 19.88 / R3 19.45 / CLIP-only 13.16 pooled; SE - R3 emotion +1.56 [+1.01, +2.09].
Seeds 43 / 44 (same-seed C0): D_emo +1.85 [+1.29, +2.39] / +1.64 [+1.12, +2.17]; D_style +2.00 / +1.18.
beta 0: D_emo +2.45 [+1.84, +3.05], D_style +0.62 [-0.01, +1.25]. Oracle beta 0 SE - C0: emotion +3.02, style +1.78.

## Report and figures
docs/reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md; figures `held_criterion.png`,
`held_per_target_emotion.png` in docs/reports/assets/2026-10-18_affect_factor_learning/ (added to
build_2026-10-18_affect_factor_learning_figures.py; the six selection PNGs rebuild byte-identical).

## Files (gitignored)
results/power.json, results/smoke_held.json + smoke_held_ranks.npz (discarded), results/held_started.json,
results/held_results.json, results/held_ranks.npz, run_power.log, run_smoke.log, run_held.log.

## Final fix wave (2026-10-01, after the whole-branch final review)
Verdict unchanged (CONFIRMED). No held row read again; nothing retrained.
- Held run time stated as 2026-10-01 01:23 CEST (2026-09-30 23:23:29 UTC, from held_started.json; system TZ
  Europe/Amsterdam).
- Held CIs re-bootstrapped from the stored ranks with seeds 1-20 (`run_posthoc_affect.py`): D_emo lower +1.514 to
  +1.544, upper +2.600 to +2.637; D_style lower +0.049 to +0.085, upper +1.215 to +1.245; max deviation 0.031.
- Report corrected: distant-supervision disclosure with scorer-train numbers; "first" claims (R3's repair was
  confirmed on held against R0 first); trained-scorer ceiling (+0.74 [+0.33, +1.16] pooled at beta 0.3, at most
  +1.78, of which +0.34 is the beta effect); smoke-twice clause; sparsity amendment; C0-recipe share of the style
  edge over R3 (+1.36 of +2.01) and C0's emotion deficit to R3 per seed (-0.51 / -0.66 / -1.03); awe/admiration note;
  next-step options (a)-(d) with costs; design-effect caveat (selection anchor-painting bootstrap 1.02).

