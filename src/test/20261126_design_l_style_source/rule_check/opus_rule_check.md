# Rule check: round 7 decision rule (design L), fresh Opus reviewer

> Reviewed 2026-10-12 02:20 (Amsterdam). Rule: `src/test/20261126_design_l_style_source/DECISION_RULE.md` (uncommitted,
> 308 lines). Spec: `docs/superpowers/specs/2026-10-12-r7-design-l-design.md`. Source of decisions: handoff
> `docs/superpowers/handoffs/2026-10-12-r7-decide-2-handoff.md` §2 and §4. Nothing was edited except this file.

## Verdict

**Commit after the fixes below.** No finding is blocking. The rule keeps every user decision of handoff §2 with the
same meaning, its integer thresholds, Holm integer forms and detectable-margin multipliers are all correct, and the
family-wise one-sided 0.025 is controlled. Six findings are important. All six are about the build: two places where
the reused code cannot do what the rule says without unstated changes, a C13 re-derivation narrower than C13 asks, a
write into a pinned input, an unattended stop the rule does not resolve, and a missing time-box. Fix them before the
commit so that `r7 build` needs no decision.

## Important

**I1. §7, §4 G1/G4, §10: the reused round 3 and round 6 modules refuse seeds 55 to 57 and score with frozen seed-42
picks. So the fresh-seed path is new code that stage 0 does not check.**
- Evidence: `20261125_artelingo_held_test/r6_context.py:53` `ADMITTED` allows only (selection, 42), smoke 9001 to
  9003 and held 52 to 54. `r6_score.py:170` and `r6_bundle.py:150` assert it. `r6_score.score_seed` scores B, B0 and B1
  with frozen seed-42 picks (`frozen_nested`) and AFF with the frozen cells 39/119 and 149/10 (`CELLS`, `frozen_cells`).
  `20261121_round3_affect_gate/r3_bundle.py:133` `_check_seed` allows only 42 and 49 to 51. The rule says these modules
  are "imported unchanged" (§10). It also requires per-seed reruns of B, B0, B1, BX, σ* and both cross-fits (§7.3).
  G1 checks AFF "by the reused round 6 code ... with the chosen cells", which is the frozen-cell path. That is not the
  path that §7 runs on seeds 55 to 57.
- Fix:
  1. §7.3: state the fresh-seed code path. Round 3's per-seed pipeline (`EvalContext`, `r3_bundle.build_bundle`,
     `r3_fusion.run_family`) in an `r7_` copy whose only change is the seed guard (42, 55, 56, 57), extended with the
     csd and X posteriors.
  2. G1: AFF must be produced by that cross-fit path, or by the round-7 X-AFF function with the allowed set {affect},
     A0's reader and R3's τ. It must choose fused cells 39 and 119, counterpart cells 149 and 10, and σ* = 0 on both
     halves, and then equal `aff_fused__*` and `aff_cf__*`.
  3. G4: B, B0 and B1 come from the per-seed `crossfit_condition_free` path, not from `picks_seed42.json`.

**I2. L11: "the same bank blocks, bank seeds" holds only if X has the dictionary key `csd`. The rule does not say
so.**
- Evidence: `src/train/pseudo_partitions.py:24-30` sorts the partition keys and gives block i the seed `seed + i`.
  `rb_features.bank_blocks` (line 56) does the same. Round 1's A1 record (`rb_bank_A1_half0.json`) has keys `['affect',
  'caption', 'csd', 'image']` and blocks (affect, caption) 11700, (affect, csd) 11701, (affect, image) 11702, (caption,
  csd) 11703, (caption, image) 11704, (csd, image) 11705. Take a key such as `v1` or `style_v1`, which sorts after
  `image`. Then block 2 becomes (affect, image) instead of (affect, csd), the seeds move to other blocks, and block 5
  becomes (image, X) with image as the condition-a class. Two honest implementers would train different readers, and
  every candidate's verdict could differ.
- Fix: add to L11: "In `build_episode_bank`'s partition dict, X has the key `csd` (any key that sorts between
  `caption` and `image` gives the same result). The builder's blocks must equal round 1's A1 list in `rb_build.BLOCKS`
  (asserted, as `stage_bank` does). Class index = position in (affect, image, caption, X)."

**I3. §9 Re-derivation: C13 asks for more than recomputing statistics from stored per-anchor arrays.**
- Evidence: §9 re-derives only K1's Σ(4d) and interval, S1 to S3 and P1, and F1 to F3's counts, all "from the stored
  per-anchor arrays". The per-anchor arrays are the numbers that decide the verdict (C13: "Every number that decides a
  verdict MUST be re-derived by separate code"). The new parts that produce them are not covered: Orth, the cut graphs,
  τ^X, the X-AFF gate, G_cf, both cross-fits and the frozen-object assembly on fresh seeds. R3 §8 re-derived features,
  P, T, margins, gates, G_cf, the 224 cells, σ*, both cross-fits and the assembly with its own code. R6 §8.4 did the same
  for the frozen-pick assembly.
- Fix: add two phases.
  - Phase A, before `gate_verdict.json`: separate code recomputes, for all 12 candidates on seed 42, the gates from
    the stored reader probabilities and τ^X, the 224-cell integer statistics, σ*, both cross-fits, and the per-anchor
    arrays of s, CF[s] and BX. They must equal the pipeline's exactly. It may import the fitted objects, `zscore_rows`,
    `crossfit_condition_free` and `cluster_bootstrap`, as R3 §8 lists.
  - Phase B, before `test_verdict.json`: the same for each pick, its CF, B, B0, B1, BX and AFF on seeds 55 to 57.

**I4. §10 "imported unchanged": some reused functions write into read-only folders, including the G2 reference.**
- Evidence: `20261119_no_caption_csd_spike/sp_run.py:92-94` `run_one` writes `sp_cand_{name}.json` and `.npz` into the
  spike's `results/`. For the told A1 run this overwrites `sp_cand_told_A1_224.npz`, which is G2's pinned input.
  `np.savez_compressed` output bytes can change, so the pinned SHA-256 can break. If the file is reloaded after the
  write, G2 compares the new arrays with themselves. `run_r2_fusion.run_candidate`, the `rb_build` stages,
  `common.save_candidate` and `common.write_json_once` also write.
- Fix: copy R3 §10's clause into §9: "Reused folders are read only. Their functions that write files
  (`sp_run.run_one`, `sp_run.main`, `run_r2_fusion.run_candidate`, the `rb_build.py` stages, `common.save_candidate`,
  `common.write_json_once`) are not called. The round-7 code calls the pure functions (`RF.terms_from_probs`,
  `K.thresholds`, `RF.fusion_pass`, `per_anchor`, `C.evaluate_fused`). G2 and G3 load their references before any
  round-7 write, and the SHA-256s of §10 are asserted again at the end of stage 0."

**I5. L10, L11, L7: an unattended stop that the rule does not resolve.**
- Evidence: R1rule §4.2 item 2 says that if a group is missing from a cross-fitted head's 60,000-row draw, "the step
  stops and goes to the user". Item 3 says the same when a bank block cannot be filled; `rb_build.stage_bank` raises
  `SystemExit` there. Both can happen for a new variant: small Leiden groups, or a random control Zk. `min_paintings=30`
  per half drops small groups from the eligible values. L10 drops a grouping only for the standard heads. The run is
  unattended (§9), so one bad variant would stop the whole round.
- Fix: add to L11: "If a cross-fitted head of X misses a group, or a bank block of A0+X cannot be filled, X is dropped
  and reported, as in L7. Its scorers count as not passing. If X is some Vk, Zk is still built and reported. If X is
  some Zk, the flag of §6 cannot be evaluated for Vk; the report says so." G3 (A1) still stops the round.

**I6. §8 and §9: no time-box and no catch-all row.**
- Evidence: Q3 (handoff §2): "done by Thursday 2026-10-15". The spec (§9) says "report by Thursday 2026-10-15;
  Friday 2026-10-16 go or no-go". The rule has no time-box and no "anything this file does not cover" row. R3 §8, R3 §9
  and R6 §9 have both. The rule does not say what FV is if the run is not complete by Thursday.
- Fix:
  - Add to §8: "Not complete by Thu 2026-10-15 23:59: no partial verdict; FV is NO-GO for Friday (ArtELingo paper with
    AFF), the user is told, and design L resumes after Friday only with the user's approval."
  - Add to §8: "Anything this file does not cover: the run stops with its report and a phone note; the user decides."

## Minor

**M1. §7.6: checks that Holm stopped get no C11 reading.** A check that Holm did not reach has not been rejected, so
it fails. C11 is a MUST. R6 §4 reads it as "not reached: the Holm procedure stopped at <check>", together with its C11
reading. Fix: "An F1 check that Holm stopped is reported as 'not reached: Holm stopped at <check>', with its C11
reading. F2 and F3 not reached: 'not tested', point and 95% interval only."

**M2. §11: the GO definition departs from C9.** C9 says "a GO needs every check to pass". Under §7.7, FV is GO when F1
passes in one chain. F2, F3 and the second chain do not enter GO. This is the user's Q6 and Q7. Fix: add it to the
§11 C9 row: "GO needs F1 in at least one chain; F2 and F3 set only the claim (user, Q6, Q7)".

**M3. L1: wrong attribution.** `run_step1.painting_grouping` (lines 322 to 370) only counts paintings whose rows
differ. It asserts nothing. The one-image-per-painting assertion is `run_step1.real_style_features` (line 264,
`painting_to_image_one_to_one`). Fix: "(asserted by `run_step1.real_style_features`'s `painting_to_image_one_to_one`,
re-asserted for every source)". Also assert `paintings_with_differing_rows == 0` for CSD and for the token statistics.

**M4. §10: code pinned by 8-character prefixes only, and the list is incomplete.** The rule imports or names modules
that are not listed: `rb_eval.py` 1826e65f, `rb_features.py` 1e758f6d, `r2_fusion.py` 1406e8e7, `r2_common.py`
402c722c, `run_sweep.py` 13314f6f (named in L6), `src/train/pseudo_partitions.py` and the `src/eval/` modules. Fix: give
full SHA-256s now. All 14 listed prefixes match the files today.

**M5. §9: no wiring smoke test, and no rule for a bug found after `test_verdict.json` exists.** R3 §10 and R6 §6.7 ran
an end-to-end smoke on seeds 9001 to 9003, including one wiring mutation, before any test seed was built. They also
had a `_fix<n>` rule that never overwrites a written file. With one look, a bug found after the verdict has no defined
path. Fix: add R3 §10's smoke, with no metric printed, before the builds, and R3 §9's `_fix<n>` row.

**M6. §1 prior: the numbers do not multiply.** 60% (K1 not stopping) × 25% (gate) × 50% (F1) = 7.5%, not 12%. 12%
follows only if the 25% already includes K1, which would mean about 42% given K1 passes. Fix: say which reading is
meant, for example "25% overall, that is about 40% given K1 passes".

**M7. §2 and L8: "every arm gets a size-matched random grouping" (handoff §2, deep check).** T and R0 have no Zk.
Fix: one sentence. "T (the ceiling) and R0 (reference) get no random control; R0's is round 1's `style_rand` (AR),
not rerun."

**M8. §5 and C1: the ceiling has no strongest-comparator number beside it.** Fix: add as descriptive beside K1 the
told-T fused R@1 on style × genre minus B1 and minus BT (B′ of A0+T), as the facts brief did for told A1.

**M9. L3: rank and cut-off.** The centred group means satisfy Σ_g n_g·μ_g = 0, so M has rank at most 63. The 64th
singular value is near the default `pinv` cut-off (1e-15 · σ_max). If it is kept, one direction made of rounding
noise is also removed. The assertion still holds, and the result is deterministic on one machine. I did not test this
numerically. Fix: record the singular values and the rank that `pinv` used in the grouping record (descriptive).

**M10. G3: make a failure easy to trace.** The versions match round 1's (scikit-learn 1.6.1, numpy 2.2.6). Round 1
trained at 8 threads on this machine (`rb_reader_A1.json` provenance). Fix: in G3, also compare the intermediate
outputs exactly: the cross-fitted posteriors with `rb_heads_*.npz`, the banks with `rb_bank_A1_half{0,1}.npz`, and
the chosen C. A failure then points to its step.

**M11. Process items from the spec that are missing from the rule.**
- The module prefix `r7_` (m6).
- When the label AMIs are computed. Suggest after `gate_verdict.json`, to keep D3 clean.
- That the token-statistics `meta.json` dropped shares are computed on scorer-train paintings only, or say that they
  cover all 61,402 images. They are features, not labels, so this is not a C5 issue.

## Spec and rule differences (the rule governs; fix the spec)

- **Spec §4 D10 and §3.5, F3: "lower bound above 0".** The rule decides at one-sided α (0.0125 or 0.025) in the integer
  form. That is stricter than a 95% lower bound. Fix the spec: "one-sided α of its chain, rule §7.5".
- **Spec Brief and §9: "15 readers".** The rule needs 13: V1 to V6, Z1 to Z6, and A1 for G3 and R0. T has none. Fix
  the count, or say what the other 2 are.
- **Spec §3.5 stage 0: "B′(A1) as round 6 stored it".** The rule's G4 adds B and B0 and checks per episode. Align the
  spec text with G1 to G4.
- **Spec §5: "round 1's reader recipe is deterministic ... Checked: stage 0".** Add "and that X keyed `csd` reproduces
  A1's block layout" (I2).

## Checks run

| Check | Result |
|---|---|
| Handoff §2 items (Q1 to Q7, deep-check bullets, m1 to m12, s1 to s3) against the rule | all present with the same meaning; m6 prefix and Q3's time-box missing (I6, M11); "every arm" control read narrowly (M7) |
| SHA-256 of the 10 inputs in §10 and of the 3 cited rules | all 13 match |
| SHA-256 prefixes of the 14 reused code files | all match |
| L16 thresholds ⌈a·N/25⌉ | 246, 82, 164, 50, 0, −49: all correct; every quantity they apply to is a difference of per-episode values in multiples of 0.25 (4 rankings × {0, 1}), and the gain statistic is gain − 0 |
| Holm integer forms | 80·(n+1)·(6−k) ≤ 5,001 ⟺ p ≤ 0.0125/(6−k); 40·… ⟺ 0.025/(6−k); largest passing n_(k) for k = 1 to 5: 11, 14, 19, 30, 61 at 0.0125 and 24, 30, 40, 61, 124 at 0.025; F2/F3: 61 and 124 |
| Detectable-margin multipliers | z(1−α/5)+0.8416 = 3.6487, 3.4175; z(1−α)+0.8416 = 3.0830, 2.8016: the rule's 3.649, 3.417, 3.083, 2.802 are correct |
| Family-wise error | Holm in F1, then serial gatekeeping (F2 only if all of F1 pass, F3 only if F2 passes) at the chain's full α controls the chain's FWER; Bonferroni across two chains gives ≤ 0.025; giving one chain 0.025 is valid because n is fixed on seed 42 before any fresh episode exists |
| F3 bootstrap on the 12,288 style × genre episodes | coherent: its own `default_rng(42)` draw over that subset's anchor-painting clusters, the same procedure as per-pair intervals |
| `diff__S1` of round 6's `seed42_per_episode.npz` | equals AFF − B1 per episode: implied B1 mean 18.804931640625, B0 18.436686197916664, B 18.341064453125; `cl` and `pair_index` equal R3's |
| `aff_fused__r1`, `aff_cf__r1` in R3's `seed42_arrays.npz` | present; pooled 19.136555989583336 and 18.39599609375; cells [39, 119] and [149, 10], σ* 0 |
| Spike told route and R1 route | τ from the configuration's own margins (`K.thresholds(margins)` in `sp_run.run_one`, `run_r2_fusion.run_candidate`), so L12 matches; one-hot margins are all 1.0, so τ = 1 and every told gate is open |
| Bank block order | sorted keys (`pseudo_partitions.py:24`); A1's record confirms (I2) |
| Seeds 55 to 57 | free in `episode_seed_ledger.md`; `baselines_seed{42,43,45,47,48,49,50,51}.json` are all the non-smoke builds |
| L5 sizes | 3 blocks × 768 channels × 2 = 4,608 values; 61,402 × 4,608 × 4 B ≈ 1.13 GB; the 3 × median rule keeps at least 25 of 49 tokens |
