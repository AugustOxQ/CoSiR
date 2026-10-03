# Method-repair diagnostics (CoSiR v2, method A′, spec §15): folder log

Report: `docs/reports/auto/v2/2026-11-05_method_repair_diagnostics.md`. Figures and re-derivation:
`docs/reports/assets/2026-11-05_method_repair_diagnostics/build_figures.py`. All runs on 2026-10-03 (local time,
UTC+2).

## Problem

E3 (aspect-factor go/no-go) ended NO-GO. The user chose a method repair before branch 3: method A′ keeps E3's factor
model and agreement rule and scores with the nested score z(cos) + λ_u·z(T_u) + λ_a·z(T_a). Following the ARS
repair-order review (Major Revision, R1 to R12), this stage had to decide, before any A′ pre-registration, between
(a) pre-registering A′ on A3, (b) running the pre-registered H2 training grid, or (c) stopping the repair (branch 3).
It ran two diagnostics under `PREREGISTRATION.md`:

- **H1 pilot:** the cross-fitted nested score on E3's checkpoints, seed-42 selection episodes, A3 primary.
- **H3 learnability diagnostic:** E3's architecture and loss trained on episodes from the evaluation labels of
  scorer-train rows (bank LAB: runs L3, L5, LT) plus a label-free matched-granularity bank (MK: run MK3).

## Steps

| Time | Commit / file | What |
|---|---|---|
| 15:30:22 | eb58116 | spec rev 3 (§15, method A′), ARS repair-order review record, `PREREGISTRATION.md`, episode-seed ledger; committed before any script of the stage existed |
| 15:33:37 | a6a4bb2 | `src/eval/aspect_nested.py`: nested score, nested uniform control, min-margin cross-fit, pre-registered readings |
| 15:39:28 | 4208c7b | `aspect_tau_fixed` option in `train_factors` (run LT) |
| 15:50:00, 15:55:57 | 2dd43d5, 4d9e714 | test fixes for the nested-score module (input validation, exact criterion and tie-breaking tests) |
| 15:52 to 16:00 | `results/smoke/`, `results/bank_*.npz`, `build_record.json` | smoke banks, then the real LAB (15:56:19) and MK (16:00:28) banks, all validated |
| 16:00:33 | b044925 | `build_banks.py` (LAB and MK banks) |
| 16:02:34, 16:05:04 | 527e85e, c24464b | `train_runs.py` and `run_day1.sh`; runner checks, atomic label-checkpoint registry, launcher propagates failures |
| 16:04:15 | 83e7027 | `score_pilot.py` (H1 pilot) |
| 16:07 to 16:08 | `pilot_seed42.json`, `per_anchor_pilot_seed42.npz`, `pilot.log` | H1 pilot run (55 s, CPU): A3 not promising, g* = 0.218 |
| about 16:07 | `run_day1.sh` | L3, L5, LT, MK3 launched in parallel |
| 16:08:16 | `train_L3_seed42_oom_attempt1.log`, `train_L5_seed42_oom_attempt1.log`, `day1.log` | L3 and L5 crashed with CUDA out-of-memory after 158 of 2,000 steps (no checkpoint) |
| 16:08:36, 16:12:32 | 896029c, 3afe1bd | `score_h3.py`; failed LAB or MK3 runs count as no fit or a negative matched-k reading |
| 16:16:54, 16:21:16, 16:26:02 | f7245b2, c0d7879, e824ad1 | `decide.py` (pre-registered joint table), its tests and an atomic write that refuses to overwrite |
| 16:17:24 | `LT_seed42.pt`, `MK3_seed42.pt` | LT and MK3 finished (622 s each) |
| 16:17 to 16:27:47 | `L3_seed42.pt`, `L5_seed42.pt` | L3 and L5 rerun with identical settings, two at a time (598 s each) |
| 16:29 | `h3_pass1_needs_seed43.json`, `h3_pass1.log` | first H3 pass: L3 and LT fit, L5 inconclusive, exit 3 (needs seed 43) |
| 16:29 to 16:39:10 | `L5_seed43.pt` | L5 retrained at model seed 43 (569 s), as §7 prescribes |
| 16:40 | `h3.json`, `per_anchor_h3.npz`, `h3.log` | final H3 pass: L5 no fit on the two-seed mean, ceiling too low |
| 16:40:38 | `decision.json` | joint decision: branch 3 |

`git log --oneline eb58116..HEAD` before the report commit:

```
e824ad1 fix(v2): race-safe atomic write using os.link() instead of os.replace()
c0d7879 fix(v2): comprehensive decision tests (all 9 h1-h3 pairs) and atomic write for decide.py
b299393 docs(v2): ARS repair-order review report
f7245b2 feat(v2): joint decision of the method-repair diagnostics (pre-registered table)
3afe1bd fix(v2): H3 scorer treats failed LAB/MK3 runs as no fit / negative matched-k, failed seed-43 reruns as no fit
896029c feat(v2): H3 learnability scorer (paired fit vs A3 on fresh label episodes, seed-42 ceiling, matched-k)
c24464b fix(v2): diagnostics runner checks, atomic label registry, launcher propagates failures
83e7027 feat(v2): H1 nested-score pilot scorer (seed 42, A3 primary, pre-registered reading)
527e85e feat(v2): diagnostics training runner (L3, L5, LT, MK3) and day-1 launcher
b044925 feat(v2): LAB (scorer-train labels, H3 only) and MK (matched-k, label-free) episode banks
4d9e714 fix(v2): I1 and I3 test replacement with exact criterion and tie-breaking tests
2dd43d5 fix(v2): I1/I2/I3 test gaps and Ruling 4 input validation for nested scores
4208c7b feat(v2): aspect_tau_fixed option for factor training (default off; spec §15 run LT)
a6a4bb2 feat(v2): nested score, its uniform control, min-margin cross-fit and pre-registered readings (spec §15)
```

## Results

- **H1 pilot (seed 42, 12,288 episodes, 4,602 paintings): not promising.** A3 nested R@1 16.52 [16.18, 16.87], gain
  −0.01 [−0.10, 0.07]; nested uniform control 16.55. m_R = −0.022 (SE 0.039), m_g = −0.012 (SE 0.043). Picks: tuned on
  the even half (4, 0.25), on the odd half (8, 0), which is the control at σ 8. g* = max(5.6·SE_R, 2.8·SE_g) = 0.218.
  Baselines: cosine 12.96, RCA 13.38, A3 under E3's score 13.39 / gain 0.52.
- **H3 fit (12,288 fresh label episodes on scorer-train rows, 9,900 paintings), term-only gain minus A3:** L3 +1.19
  [0.60, 1.76] fits; LT +1.34 [0.77, 1.90] fits; L5 +0.15 [−0.41, 0.70] inconclusive, two-seed mean +0.08
  [−0.41, 0.56] no fit.
- **H3 ceiling: too low.** Best fitting run LT, cross-fitted nested gain on seed 42 0.05 [−0.11, 0.22] < g* 0.218
  (L3 0.01).
- **Matched-k: not a lever.** MK3 minus A3 term-only gain on seed 42 −0.88 [−1.44, −0.34].
- **Decision (§8): branch 3, stop the repair.** Seed 45 not built; H2 grid not run.
- Descriptive: label training raised the seed-42 term-only gain from 0.99 (A3) to 1.85 (L3) and 1.82 (LT); the aspect
  loss stayed within 2% of its constant-score value (L3 1.8% below, L5 1.0% below, LT 0.4% above); term-only either
  rates (21.0 to 22.9) stayed below the cosine's 25.92.

## Issues

1. **CUDA out-of-memory with four parallel trainings.** `run_day1.sh` started four runs at once; the four processes
   held about 23 GiB of the 23.55 GiB GPU, and L3 and L5 crashed in the backward pass after 158 steps without writing
   a checkpoint, history or failure record. `score_h3.py` would have refused to run (neither checkpoint nor failure
   record).
2. **L5 inconclusive at seed 42.** The first H3 pass stopped with exit 3 and `needs_seed43 = ["L5"]`.
3. **g* far below the ARS panel's estimate** (0.218 against about 0.94): the odd-tuned nested pick equalled the
   control, so half the anchors contribute a zero paired difference and the SEs shrink.
4. **`build_record.json` has no script SHA-256**, although PREREGISTRATION §10 asks every result JSON for it;
   `build_banks.py` was committed 33 s after the record was written and is unchanged since.

## Resolution

1. L3 and L5 were rerun with identical settings, two at a time, as PREREGISTRATION §4 allows after an infrastructure
   failure; the crashed logs were kept as `*_oom_attempt1.log`. LT and MK3 had completed and were not rerun.
2. L5 was retrained at model seed 43 and `score_h3.py` was rerun; the pass-1 JSON was kept as
   `h3_pass1_needs_seed43.json`, and the two passes differ only in the seed-43 resolution (asserted by the report's
   build script).
3. g* was kept, since the rule was frozen; the ceiling failed it anyway.
4. Disclosed in the report (§7); bank and partition SHA-256s are recorded and verified.

The report's `build_figures.py` re-derived every quoted summary and paired comparison from the stored arrays,
recomputed A3's 56-cell profile and cross-fit from its checkpoint, rebuilt the fresh label episodes (SHA-256 checked)
and reapplied the readings and the decision. Storage: `results/` holds about 54 MB and `checkpoints/` about 2.4 MB
(all gitignored); nothing over 1 GB was written.
