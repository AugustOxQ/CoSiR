# Round 5 (idea 3) independent re-derivation: phase-1 agreement report

Written 2026-10-07 18:33 (Amsterdam), after the implementation's seed-42 run had written `results/carry.json`
(18:25, SHA-256 9291e3a8…141d) and the controller lifted blindness for the comparison. The rule is
`../DECISION_RULE.md` (19e59fc7…735e), §8 "Agreement". The stage-A record is in `rd5_stageA_report.md`.

## 1. Verdict

**Agreement: yes.** I compared 492 quantities, and all of them agree. Every compared difference is exactly 0,
including the largest numerical one and the largest array difference. No quantity sits at a boundary.

**My own decision on seed 42: KILL.** E is empty. Neither candidate clears D10: both fail clause 1 (bar-margin
point at least +0.5) and pass clauses 2 and 3. Both Δ_k are negative: G-T −169, G-TF −134. With no carried candidate,
§6.1's x is not computed, so the `sensitivity` field of my record is null. The implementation reached the same kill.

| Kind | Compared | Failed | Tolerance (rule §8) | Largest difference |
|---|---|---|---|---|
| discrete (classes, n_iter_, fallback, cells, σ*, bar comparator, Δ_k, D10 clauses, carry, flags, open counts, hashes) | 222 | 0 | identical | — |
| pp (R@1 means, margins, gain statistics, points and bounds, either change, per-pair margins, comparator means) | 142 | 0 | 1e-9 pp | 0.0 |
| τ, τ′ (elementwise, including the cell records' τ) | 24 | 0 | 1e-15 absolute or 1e-9 relative | 0.0 |
| arrays (Q_GE, per-anchor arrays, gates, P, margins, picks, cells, σ*, τ, τ′, comparators) | 104 | 0 | exact | 0.0 |

## 2. Inputs (hashes asserted before anything of the implementation's was read)

| Side | File | SHA-256 |
|---|---|---|
| mine | `results/rd5_stageB.json` | 07e92ebdda993d6ed87e2d8571068704d24c750954c7c18d7f6dfbfb1f743ebc |
| mine | `results/rd5_stageB_arrays.npz` | 165f4de944486e330fcc1d2b4802ced7e287aefc5a63af30d760d92098a8a2cc |
| mine | `results/rd5_stageA_fix1.json` | 01ba2ecb7ba5888afc87b1f017ded9373436660765e340e2fcbd91e8714dc9df |
| theirs | `results/carry.json` | 9291e3a862e84a587178cd31e4cfcfb8f3a9c6ef4ddeb7432fcdee140bcf141d |
| theirs | `results/dev_seed42.json` | 66225589bf5e8659bda21755e4db22301b6cbb871f887adbc0806b625dfb2b7c |
| theirs | `results/seed42_arrays.npz` | 8dd1dd83327cd04e0f3f135c9cd3ba0fa55bf53e6e818a64d4ac5119f6eb9a14 |
| theirs | `results/placement.json` | ab09071369e6db8746ac4142d53e5f898b8cfa7d7642f69a2007fdeb69bd0ef0 |
| theirs | `results/regression_check.json` | cc8dea679aa8e882edbba8292701fe8832fce2ba98ed02521c54dc2a44342edd |
| theirs | `cache/r5_ge_posterior.npz` | 081bb19e2a9b23cc5d97612f83950f77e2c9a503919e48d42fb72d94bbd7fbf7 |

The comparison program is `rd5_compare.py`, and its output is `agreement_phase1.json`
(SHA-256 3ef4e730c284ca29ee68d0065cca5933a95537253ee75940d34d2b8bfd03abc4). That file holds every compared row with
both values, every leaf without a counterpart and its reason, and the perturbation tests. Two earlier drafts of the
file were deleted before this report. One had 178 leaves without a stated reason; the other had an inaccurate label on
one perturbation test. Neither changed a compared value, and neither hash was given to anyone.

## 3. My stage-B run (17:49 to 17:50, CPU, one process)

- **Spot check.** I reran 1,024 selection captions on CPU, at positions from `rng(6)`, joined the same way as D2.
  - Largest absolute difference from the stored file: 3.28e-7, against the 1e-4 tolerance.
  - Entries above 1e-5: none.
  - Rows and sample ids: equal to the join and the splits.
- **GE head.**
  - Held-out accuracy: 86.17, against 35.72 for the CLIP caption head and 9.81 for the image head.
  - Classes 0 to 40; lbfgs stopped after 139 iterations, so the fallback was not needed.
  - Draw SHA-256 7be956c0…; the scorer-train mapping held.
  - Q_GE is identical to the implementation's file: float32 values, rows and the NaN pattern. I checked the pattern by
    hashing their selection rows scattered into the full array.
- **D5's positive check** passed. The bundle's post dict was untouched (fingerprints taken before and after).
- **B′_G.** Mean R@1 18.398030598958336. Against B′(A0) it is −0.0387 [−0.2057, +0.1220].

Development numbers on seed 42, identical to the implementation's (R@1 in percentage points):

| | G-T | G-TF |
|---|---|---|
| fused R@1 / counterpart R@1 | 18.7927 / 18.3289 | 18.8639 / 18.2353 |
| bar comparator (B′_G 18.3980, B′(A0) 18.4367, counterpart, B 18.3411) | B′(A0) | B′(A0) |
| bar margin | +0.3560 [+0.1240, +0.5830] | +0.4272 [+0.1848, +0.6668] |
| margin against the counterpart | +0.4639 [+0.2674, +0.6625] | +0.6287 [+0.4229, +0.8371] |
| gain statistic | 2.5309 [2.2320, 2.8371] | 2.8727 [2.5587, 3.1995] |
| either change against the counterpart | −1.6032 | −1.6154 |
| D10 clauses 1 / 2 / 3 | no / yes / yes | no / yes / yes |
| Δ_k against AFF (integer; points) | −169; −0.3438 [−0.5222, −0.1630] | −134; −0.2726 [−0.4565, −0.0924] |
| τ (G-T) or τ′ (G-TF) | AFF's τ | 2.4383e-05, 0.21725, 0.47370, 0.74248 |
| τ₀ open counts, conditions a / b | 9,941 / 3,627 (AFF's) | 9,825 / 3,431 |
| fused cells / counterpart cells, σ* | 46, 117 / 101, 25; 0, 0 | 12, 117 / 67, 164; 0, 0 |
| minus B′(A1) (18.8049), beside only | −0.0122 [−0.2896, +0.2699] | +0.0590 [−0.2227, +0.3510] |

AFF's own record (stage A, and again in this process) gives fused R@1 19.1366 and bar margin +0.6999. The carry
follows directly: E is empty, there is no M and no tied set, and the outcome is KILL.

## 4. What was compared

- **`placement.json`.** GE-head accuracy, classes, n_iter_, n_iter_first, fallback and its iteration count, majority
  share, uniform rate, the posterior SHA-256 it records, and all item-3 fields: CLIP n_iter_, accuracies and
  equality flags.
- **`cache/r5_ge_posterior.npz`.** `post_sel` element by element, `rows`, `classes`, the dtype, and the full array with
  its NaN pattern.
- **`dev_seed42.json`.** Every development field of G-T, G-TF and AFF:
  - R@1 means, cells (fpick, cpick and the cell texts with τ index, τ, λ_u, λ_a and k_top), σ*;
  - the bar comparator, the comparator means, the bar margin, the margin against the counterpart and the gain
    statistic with their intervals and n_clusters;
  - the either change, the per-pair bar margins (points and intervals), B′_G minus B′(A0), the B′(A1) line beside;
  - Δ_k (the integer, its point and interval), the D10 clauses, the τ₀ open counts, τ′ and AFF's τ;
  - the comparators block, the beside block, the D5 positive-check flags and the file hashes it records.
- **`seed42_arrays.npz`.** All 104 arrays; none is left without a counterpart.
  - G-T, G-TF and B′_G arrays, G-TF's P, margins and picks, τ and τ′: against my stage-B arrays.
  - AFF's P, margins and picks: against my G-T reader outputs, which are AFF's.
  - AFF, B, B′(A0), B′(A1), cosine, RCA, cl, pair_index and parity: against round 4's stored seed-42 arrays. My stage A
    asserted those equal to my own, and stage B asserted AFF equal again in-process. Cosine and RCA were also checked
    against `per_anchor_seed42.npz`.
- **`carry.json`.** E, M, tied, carried, kill, the order, the tie band, each candidate's Δ_k and D10 clauses, the
  boundary list (empty on both sides), `boundary_reported`, and the hashes it records.
- **`regression_check.json`.** The pass flags of items 1, 3 and 4 against my stage-A checks, and item 2's against my
  spot check, the analogous step. Also `all_passed`, the GoEmotions-file hashes, and every comparison whose value has
  a counterpart in my records: R1's and AFF's numbers, cells and rule-text cells, τ, open counts, B′(A1),
  AFF − B′(A1), item-3 accuracies, and my pass flags for the same equalities with stored arrays.

Two counterparts were computed during the comparison rather than taken from my records:

1. **Per-pair interval bounds.** My records kept per-pair points only. I computed the bounds from my stage-B arrays,
   and for AFF from the stored arrays that stage A proved equal to mine. They agree to 0.
2. **R1's mean R@1.** Taken from `cand_Rc_Rb_expected_A0.npz`, which stage A proved equal to mine.

## 5. Leaves of the implementation's files without a counterpart (1,935; each listed in `agreement_phase1.json`)

| Count | File | Reason |
|---|---|---|
| 804 | regression | Item-4 self-check flags for G-T and G-TF. My counterparts are my stage-A item-4 checks, all passed |
| 354 | regression | Round 3's bundle-equality self-checks. Mine: the bundle equal to round 1's `load_bundle` and the stored arrays |
| 108 | regression | Round 4's A1-extension self-checks (CSD posteriors, A1 features, B′(A1) scores). Not re-derived; my B′(A1) per anchor is round 4's stored array, which equals round 1's `pBp["A1"]` |
| 84 + 6 | regression | Other item-4 self-check flags (CLIP-extension equality) |
| 120 | regression | Their constants checked against round 3's or the brainstorm's files; no computation (10 more against round 4's file are in the last row) |
| 48 | regression | D7 redundancy values: not in the re-derivation's phase-1 list |
| 40 + 12 | regression | Item 4's pair-lift and AUC parts: outside the re-derivation's list. The final review re-derives the diagnostics |
| 48 | regression | Item-3 self-check flags (mine: the item3 checks of stage A) |
| 30 | regression | Their D3 scorer-train sample (rng 5, 2,048 captions). My analogue is the selection spot check |
| 34 | dev | Their gate self-check flags. The gates themselves were compared array by array |
| remaining | all | Record-presence and failure-record self-checks, the implementation's input and module hash tables, the rule's SHA-256, provenance, write times, descriptions, check counts, run time and run mode |

No leaf is unexplained.

## 6. Perturbation test (the comparator must catch each change)

Each case copies the implementation's data, changes one thing and reruns the whole comparison. In every case the new
failures are exactly the targeted quantity.

| Perturbation | New failures | Caught |
|---|---|---|
| G-T bar-margin point + 2e-9 pp | `G-T.bar_margin.point` | yes |
| τ′[1] × (1 + 2e-9) | `tau_prime[1]` | yes |
| G-TF Δ_k + 1 (dev_seed42.json) | `G-TF.delta_int` | yes |
| G-T Δ_k − 1 (carry.json) | `carry.G-T.delta_int` | yes |
| one element of `gtf_fused__r1` ± 0.25 | `arrays.gtf_fused__r1` | yes |
| G-T D10 clause 2 flipped | `G-T.D10.c2` | yes |
| carry `kill` flipped | `carry.kill` | yes |

## 7. Boundaries and process

- **Boundaries.** None on either side: no lower bound and no D10 clause within 1e-12 of its threshold, no Δ_k of 0,
  no tie gap of 24.
- **What I read.** Only the implementation's result and cache files named above. I never opened
  `diagnostics_seed42.json`, `run_r5_*.log`, the implementation's code or `.superpowers/`.
- **CPU and processes.** Every Python call ran on CPU with the full env prefix, never more than one process of mine
  at a time.
- **`__pycache__`.** None under `rederive/`.
