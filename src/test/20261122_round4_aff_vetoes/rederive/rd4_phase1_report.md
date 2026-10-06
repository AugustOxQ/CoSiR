# Reader fix round 4 (vetoes on AFF's gate): independent re-derivation, phase 1 (seed 42)

Written 2026-10-07 01:50 (Amsterdam) by the independent re-derivation agent, under `../DECISION_RULE.md` (SHA-256
cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b) and round 3's rule, which it cites
(`../../20261121_round3_affect_gate/DECISION_RULE.md`, SHA-256
2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925). Every script asserts both. We never opened or
imported the round-4 implementation (`r4_*.py`, `run_r4_*.py`, `test_r4_*.py`, `results/`), and never imported
`r3_fusion`, `r3_stats`, `r3_bundle`, `r3_common`, `r2_fusion` or `rc_core`. Every target below is copied from the
rules' text.

**Result: items 1 to 5 match, and the carry is a kill.**

- Rule §5 items 1 to 5 match their targets bit for bit: absolute difference 0.0 on every number, every per-anchor array
  identical, every discrete quantity identical. No difference needed tracing.
- We computed no candidate result until all five items had passed (the script exits before item 6 otherwise).
- **Item 8 kill:** E is empty. V4 clears the development bar but has Δ_V4 = −18 (it does not beat AFF). V2 and V24 fail
  D10 against their floor B′(A1). No candidate is carried, so there is no sensitivity projection (§6.1 applies only to a
  carried candidate).
- No boundary case of rule §8: no D10 clause lies within 1e-12 of its threshold, no Δ_k is 0, and there is no tie gap.
- `out/rd4_phase1.json` has SHA-256 **13e30ebc5f42d3f3e464c44bc9dd8494179fcca10b7cbd522543dad281ff5ff5**. It was written
  to `out/rd4_phase1.sha256` at 01:40, before any comparison with the implementation. Rule §8 says phase 1 must agree
  with the implementation before a kill is reported. We have not run that comparison: `rd4_compare_phase1.py` is ready
  for the controller (§9).

## 1. What we computed, and with what

| Script (`rederive/`) | What | Runtime |
|---|---|---|
| `rd4_core.py` | Round-4 pieces in our own code: SHA-256 checks of both rules and of every D11 input we read; the A1 reader's probabilities (half-reader mean); v and the gate factors (D5, D6) with their float32 0/1 and closed-where-AFF-closed asserts; R1's and AFF's gates; the four-way bar comparator (D8, decided on integer sums); evaluation (bar margin, margin, gain statistic, either change, D10 clauses, per pair); Δ_k from integer per-episode values (D9); the carry (§5 item 8) | |
| `rd4_bundle.py` | Seed-42 bundle in round 3's §4 item 1 call sequence, then the A1 extension (D2: csd posteriors via `run_step1.full_post`; D4: B′(A1)); our own A0 and A1 grouping scores and 18/24 features; v | 44 s |
| `rd4_ref42.py` | Round 1's `common.load_bundle()` with stdout discarded (18 lines). Saves only the reference arrays item 1 compares against. Runs as a separate process, so its imports never mix with ours | 85 s |
| `rd4_phase1.py` | §5 items 1 to 9 in order (it stops before item 6 if any of items 1 to 5 fails), and §6.1 for a carried candidate | 20 s |
| `rd4_selftest.py` | Synthetic checks: the carry (tie band 24 inclusive at gap 24, excluded at 25; order V4, V2, V24; Δ_k = 0 excluded), the four-way comparator ties, gate-factor asserts, `as_int4`, the rule's named cells (116, 119, 58, 123, 39, 149, 10, 117, 67) decoding to their (τ index, λ_u, λ_a), first-grouping arg-max ties | passed |
| `rd4_compare_phase1.py` | The §8 agreement with the implementation's files (§9) | not yet run on real files |

We reused by import round 3's independent re-derivation (`rd3_core`, `rd3_family`), as the brief allows. That code is
independent of the implementation path, and it reproduced round 3's targets bit for bit. From it we took the 6-per-grouping
features, grouping scores, z-scores, the float32 combine, integer rank counts and per-anchor metrics (R@1 strict, ties
miss), the painting bootstrap wrapper, the one-way sensitivity projection, and the 224-cell family: z before the gate,
gated term, G_cf, integer ρ and γ, σ*, both cross-fits, assembly, and a literal re-scoring check of the assembled scores.
Allowed loaders imported: `EvalContext`, `run_checks.model_inputs` with `centered_term`, `run_n6.load_posteriors`,
`run_told_oracle.fit_one_head` and `global_labels`, `rb_build.load_readers("A0"/"A1", False)`, `run_step1.full_post`,
`crossfit_condition_free`, `uniform_probe_scores`, `zscore_rows`, `cluster_bootstrap`, and round 1's
`common.load_bundle` (seed 42, separate process). Module files are asserted.

Run conditions: CPU only, `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`
(`sys.dont_write_bytecode` is also set), scikit-learn 1.6.1, numpy 2.2.6, at most two processes at once. Only seed 42
was touched (the guard refuses any other seed). No `build_seed*.log` was opened.

## 2. Item 1: bundle (passed)

| Piece | Reference | Result |
|---|---|---|
| anchors, candidates, parity, anchor paintings (4,602), pair index, selection | episodes file; step 1; rd2 cache; `load_bundle` | identical |
| cosine scores and per-anchor arrays | `load_bundle`; `per_anchor_seed42.npz` | identical |
| B scores (float32) and per anchor | `load_bundle`; rd2 cache; step 1 | identical; R@1 18.341064453125 = round 3 D10 |
| B cross-fit picks | `told_oracle.json` `B_picks` | identical |
| B′(A0) scores and per anchor | `load_bundle`; step 1; rd2 cache | identical; R@1 18.436686197916664 = round 3 D11 |
| B′(A0) picks | `told_oracle.json` arm L | identical |
| affect, image, caption posteriors (selection rows) | `load_bundle`; `n6_posteriors.npz` (image, caption) | identical; affect head = arm L's record |
| **csd posteriors** (image and caption, 308,723 × 17 float32, NaN off selection) | `load_bundle` (full-array SHA-256) | identical (362cf827…, 050bff6b…) |
| **B′(A1) scores and per anchor** | `load_bundle` `Bp["A1"]`, `pBp["A1"]`; step 1 `A1__Bprime__*`; rd2 cache | identical; mean R@1 **18.804931640625** = D4 exactly |
| grouping scores s_h, A0 and A1 | rd2 cache `stack__*` | identical |
| 18 A0 features, both conditions | rd2 cache `F_A0__*` | identical; Δ^b = −Δ^a |
| **24 A1 features**, both conditions | rd2 cache `F_A1__*` (checked bit-equal to `rb_eval.seed42_features` in round 2's re-derivation) | identical; first 18 columns = A0 features |
| D7 redundancy (six values) | round 3's rule | all six equal exactly; affect smallest in both directions |

The rd2 cache is `20261118_reader_fix_round2/rederive/out/rd2_seed42_cache.npz`, whose SHA-256 a173a7c4… matches its
`rd2_prep.json` record. `rb_eval.seed42_features` itself is not on the allowed list, so it was not called.

## 3. Item 2: R1 = round-1 R-c, AFF = round 3's targets (passed)

- **R1:** T^c, margins and picks equal `cand_Rc_Rb_expected_A0.npz`. τ_0..τ_3 recomputed equal `rc_tau.json` and the
  rule text exactly. R1's gates equal the stored `extra__gate_*`. Fused cells 116 / 119, counterpart 58 / 123, σ* 0 / 0.
  All per-anchor arrays and `bar_v` are identical. The bar comparator is the counterpart. Bar margin
  0.4435221354166667 [0.21646171563312194, 0.6735669710776852] and gain statistic 2.667236328125 [2.325087836946873,
  3.012361650695922] are exact.
- **AFF through the candidates' gate function:** `apply_factors(AFF gates, ones, ones)` returns AFF's gates exactly
  (float32), and AFF's family is run from them. The following all match exactly:
  - fused R@1 19.136555989583336 and counterpart 18.39599609375;
  - bar comparator B′(A0);
  - bar margin 0.6998697916666667 [0.4598852740816973, 0.9371680126852968];
  - margin 0.7405598958333333 [0.5196896694963071, 0.9598857494832738];
  - gain statistic 3.110758463541667 [2.780005709854805, 3.4559584315470384];
  - either change −1.629638671875;
  - per-pair bar margins 0.9765625, 1.45263671875 and −0.32958984375;
  - cells 39 / 119 and 149 / 10, σ* 0 / 0;
  - AFF minus R1: fused 0.21769205729166666 [0.06425880757348419, 0.3709597330984391] and bar 0.25634765625
    [0.04280778303598444, 0.46195041633015954];
  - τ_0 open counts 9,941 / 3,627.

## 4. Item 3: R1 × a_v = `IMGABST_q75` (passed)

v = min(S_image^a, C_image^a), from A0 feature columns 6 and 7 of condition a. v^b = v^a exactly.
`numpy.percentile(v, 75)` = **0.021043562795966864**, exactly the rule's v₇₅. a_v = 1 on **9,216** episodes, the same in
both conditions.

R1's gates × a_v come from the same `apply_factors` that builds V4's gate. Through the float32 path, everything matches
the rule's text exactly:

| Quantity | Value |
|---|---|
| fused R@1 | 19.059244791666664 |
| counterpart R@1 | 18.49365234375 |
| bar comparator | counterpart |
| bar margin | 0.5655924479166667 [0.3448683992591827, 0.79821625538382] |
| gain statistic | 2.878824869791667 [2.554983173204304, 3.2105685950938248] |
| either change | −1.7476399739583333 |
| per-pair bar margins | 0.677490234375, 1.28173828125, −0.262451171875 |
| fused cells | 117 (τ_2, 0, 4) and 119 (τ_2, 0, 16) |
| counterpart cells | 58 (τ_1, 0, 0.5) and 67 (τ_1, 0.5, 1) |
| σ* | 0 / 0 |

The brainstorm's own `bs_04_readers.json` record (`results.IMGABST_q75`) matches too: every number exactly, including
the either interval and the per-pair gains, and the cells by (τ index, λ_u, λ_a). The float64 brainstorm and our
float32 path did not differ.

## 5. Item 4: the A1 reader = `cand_R1_A1.npz` (passed)

P_A1^c (the mean of round 1's two A1 half-readers on our 24 features; C = 1.0 and 100.0) equals `probs__a` and
`probs__b` exactly (maximum absolute difference 0.0). π_A1^c equals the stored int8 `pick__a` and `pick__b` exactly.
There are no exact arg-max ties in either condition (the tie rule is covered by `rd4_selftest.py`).

| Condition | affect | image | caption | csd |
|---|---|---|---|---|
| a | 8,742 | 421 | 1,083 | 2,042 |
| b | 1,814 | 3,527 | 1,887 | 5,060 |

## 6. Item 5: gate algebra (passed)

At every τ index and in both conditions:

- every gate is a float32 array of 0s and 1s;
- each candidate's gate is 0 wherever AFF's is 0;
- V24 = V4 × V2 exactly;
- V2 = AFF × 1[stored `pick__c` = 0], with `pick__c` from `cand_R1_A1.npz` as an independent source;
- V4 = AFF × 1[v < 0.021043562795966864], with v from the A0 features.

Open counts (condition a / b), gate statistics only:

| Gate | τ_0 | τ_1 | τ_2 | τ_3 |
|---|---|---|---|---|
| AFF | 9,941 / 3,627 | 9,005 / 2,573 | 7,326 / 1,368 | 4,308 / 336 |
| **V4** | **8,245 / 3,066** | 7,699 / 2,255 | 6,541 / 1,260 | 4,055 / 324 |
| **V2** | **8,593 / 1,762** | 8,121 / 1,558 | 6,938 / 1,077 | 4,261 / 318 |
| **V24** | **7,504 / 1,572** | 7,186 / 1,412 | 6,315 / 1,014 | 4,032 / 309 |
| R1 | 12,288 / 12,288 | 10,294 / 8,138 | 7,913 / 4,375 | 4,502 / 1,642 |
| R1 × a_v | 9,216 / 9,216 | 8,184 / 6,176 | 6,731 / 3,343 | 4,109 / 1,220 |

## 7. Item 6: development numbers (seed 42)

The integer sums (Σ 4·R@1 over 12,288 episodes) behind the means:

| Scorer | Sum |
|---|---|
| AFF | 9,406 |
| V4 | 9,388 |
| V2 | 9,388 |
| V24 | 9,372 |
| B′(A1) | 9,243 |
| B′(A0) | 9,062 |
| B | 9,015 |

| | V4 | V2 | V24 |
|---|---|---|---|
| Floor and comparator order | B′(A0), cf, B | B′(A1), B′(A0), cf, B | B′(A1), B′(A0), cf, B |
| Fused R@1 | 19.099934895833336 | 19.099934895833336 | 19.0673828125 |
| Counterpart R@1 | 18.391927083333336 | 18.34716796875 | 18.391927083333336 |
| Bar comparator | **B′(A0)** (18.4367) | **B′(A1)** (18.8049) | **B′(A1)** (18.8049) |
| Bar margin | +0.66324869791666674 [+0.42921625314611117, +0.89445055765880288] | +0.29500325520833337 [+0.010178842628223385, +0.57841110778525695] | +0.262451171875 [−0.016448639942734416, +0.54087577359005001] |
| Margin vs counterpart | +0.7080078125 [+0.49975103998915654, +0.92119043390945732] | +0.75276692708333326 [+0.54148264406116675, +0.96455524436398465] | +0.67545572916666674 [+0.473481170763905, +0.87796282978981599] |
| Gain statistic | +2.797444661458333 [+2.4854376869391626, +3.1170386067788969] | +2.952067057291667 [+2.6295553013908282, +3.2848781581949842] | +2.748616536458333 [+2.4376065646148186, +3.0677596402654079] |
| Either change vs counterpart | −1.3814290364583335 [−1.6799890009163325, −1.0789894791788144] | −1.446533203125 [−1.7361479052844329, −1.1512858635641638] | −1.397705078125 [−1.6774664510002653, −1.1168742814045742] |
| **Δ_k (integer, D9)** | **−18** | **−18** | **−34** |
| Δ_k point (pp) and 95% interval | −0.03662109375 [−0.10506836425355538, +0.03271528780460466] | −0.03662109375 [−0.11495968701561131, +0.038888209184998086] | −0.06917317708333333 [−0.16515849108866412, +0.026392838403383488] |
| Episodes up / down against AFF | 124 / 137 | 151 / 165 | 236 / 264 |
| D10 (1) bar point ≥ 0.5 | pass | **fail** | **fail** |
| D10 (2) bar lower > 0 | pass | pass | **fail** |
| D10 (3) gain lower > 0 | pass | pass | pass |
| Clears the bar | **yes** | no | no |
| Fused cells (half 0 / half 1) | 39 (τ_0, 4, 16) / 119 (τ_2, 0, 16) | 39 / 119 | 39 / 119 |
| Counterpart cells | 167 (τ_2, 16, 16) / 54 (τ_0, 16, 8) | 93 (τ_1, 4, 4) / 166 (τ_2, 16, 8) | 149 (τ_2, 4, 4) / 166 (τ_2, 16, 8) |
| σ* | 0 / 0 | 0 / 0 | 0 / 0 |
| Per-pair bar margin, emotion × style | +0.921630859375 | +0.128173828125 | +0.103759765625 |
| emotion × genre | +1.324462890625 | +1.3671875 | +1.2451171875 |
| style × genre | −0.25634765625 | −0.6103515625 | −0.5615234375 |

Per-pair intervals are in `rd4_phase1.json` (`item6_dev.<k>.evaluation.per_pair`); per-pair results are descriptive.

**Why the vetoes did not help.** Every candidate's fused cross-fit chose AFF's own cells (39 and 119), so each
candidate is AFF with fewer gate openings in the same cells. The episodes that the vetoes switch off were slightly more
often helped by steering than hurt: V4 loses on 137 episodes and gains on 124. V4 and V2 reach the same total (Σ 4·R@1
9,388) on different episodes. Their per-anchor arrays differ on 426 episodes, so the tie is a coincidence and not a
wiring error.

V2 and V24 also face the higher floor B′(A1) (18.805). To clear it they needed a fused R@1 near 19.31, as the rule's
prior said, and they reached 19.10 and 19.07, for bar margins of +0.295 and +0.262. V4 kept the bar against B′(A0)
(+0.663), but its Δ_V4 interval contains 0, with the point on the negative side. The prior expected V4 to be the likely
carry with a gain over AFF of at most about +0.1, and said that a kill at the carry would not surprise us.

## 8. Items 8 and 9: carry, kill

E = {k : D10 cleared and Δ_k > 0} = **∅**:

- V4 cleared D10, but Δ_V4 = −18 is not above 0;
- V2 failed D10 (Δ_V2 = −18);
- V24 failed D10 (Δ_V24 = −34).

So M is undefined, the tied set is empty, and no candidate is carried: **kill (rule §5 item 9)**. No test seed is built.
The seed-42 results go to the user, and AFF stays the current best. The boundary list of §8 is empty.

**Sensitivity (§6.1): not computed.** The rule computes it only for the carried candidate, after the carry, and nothing
was carried. So there is no x for an AFF check.

## 9. The comparison script, `rd4_compare_phase1.py` (not yet run on the implementation's files)

It asserts both rules and checks that `rd4_phase1.json` still has the recorded SHA-256. It then loads whichever of
`../results/regression_check.json`, `dev_seed42.json`, `carry.json` and `sensitivity.json` exist; `sensitivity.json` is
not expected after a kill. It compares 169 of our quantities under §8's tolerances:

- discrete quantities identical;
- v₇₅ within 1e-15 absolute or 1e-9 relative;
- numbers within 1e-9 pp;
- every per-anchor array exactly: each fused and counterpart array of AFF, V4, V2 and V24 (5 metrics each) must have a
  SHA-256-identical twin, in a `results/*.npz` file, under a key that names its scorer.

The output is `out/agreement_phase1.json`, with `all_agree`, every compared quantity (ours, theirs, path, difference),
value hints for unlocated quantities, and the implementation's leaf paths that no quantity used.

**Key mapping needs adapting.** We wrote the script without reading the implementation, so it locates quantities with
a token heuristic over the implementation's key paths. The `IMPL_MAP` dict (dotted paths) overrides the heuristic and is
empty for now. Once the implementation's files exist, the mapping may need a small adaptation: unmatched or ambiguous
quantities show up as such, and `all_agree` stays false until they are mapped.

We tested the script on a synthetic layout built from our own JSON with different key names, nested intervals and an
npz of arrays:

- faithful copy: 169 of 169 agree, `all_agree` true;
- a 2e-9 shift in one bound: caught as "disagree";
- one chosen cell changed: caught as "disagree";
- one per-anchor element flipped: caught as an array without a twin.

These tests wrote only to the session scratchpad.

## 10. Files

All outputs are in `rederive/out/`, which the repository's `.gitignore` (`out/`) excludes:

| File | Size | Contents |
|---|---|---|
| `rd4_phase1.json` | 68 KB | every quantity at full precision; SHA-256 13e30ebc…5ff5, in `rd4_phase1.sha256` |
| `rd4_phase1_arrays.npz` | 14 MB | per-anchor arrays of R1, AFF, R1 × a_v, V4, V2, V24 (fused and counterpart), B, B′(A0), B′(A1), cosine, RCA; gates; picks; P; P_A1; v |
| `rd4_bundle_seed42.{npz,json}` | 78 MB | our bundle cache |
| `rd4_ref42.{npz,json}` | 62 MB | `load_bundle` references |
| logs | | |

Nothing exceeds 100 MB, and the total is 147 MB. The two caches can be deleted once the agreement is settled.
Runtime: 44 s for the bundle, 85 s for the references (run in parallel), and 20 s for phase 1.
