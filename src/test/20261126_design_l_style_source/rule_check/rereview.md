# Re-review of the fix wave: round 7 decision rule (design L)

> Reviewed 2026-10-12 (Amsterdam). Rule: `DECISION_RULE.md` (386 lines). Spec: `docs/superpowers/specs/2026-10-12-r7-design-l-design.md`. First review: `rule_check/opus_rule_check.md`. Nothing was edited except this file.

## Verdict

**Commit after the listed fixes (N1 to N4).** All 21 findings are applied and each applied text does what the finding asked. The fix wave added no contradiction in numbers, IDs, file names, order of work, the time-box, the 13 readers, the 12% / 25% / 40% priors or F3's level. Four small new points remain; each is a one-line edit, and none changes a verdict path.

## Findings

| Finding | Status | Note |
|---|---|---|
| I1 fresh-seed code path | applied | §7.3 Code path names `EvalContext`, `r3_bundle.build_bundle`, `r3_fusion.run_family` as `r7_` copies with the seed guard (42, 55 to 57, 9001 to 9003). G1 and G4 use that path. See N2 for wording. |
| I2 key `csd` and block layout | applied | L11 first bullet. Checked: sorted keys give blocks as in `rb_build.BLOCKS["A1"]`. Class order matches `rb_reader_A1.json` `classes` (affect, image, caption, csd). |
| I3 C13 re-derivation of the arrays | applied | §9 phases K, A, B, in the glossary and in the file-writing order. Phase K goes beyond the finding (covers T). |
| I4 writes into read-only folders | applied | §9 "Read only" copies the clause. See N1: it now conflicts with the run_baselines builds. |
| I5 unattended stop on a missing group or block | applied | L11 second bullet; G3 still stops the round. |
| I6 time-box and catch-all | applied | Two §8 rows. Matches spec D26 and the Brief. 2026-10-15 is a Thursday. |
| M1 Holm-stopped reading | applied | §7.6. |
| M2 GO versus C9 | applied | §11 row and §7.7 agree. |
| M3 L1 attribution | applied | `real_style_features` / `painting_to_image_one_to_one` (run_step1.py:264); `paintings_with_differing_rows` asserted 0. |
| M4 full SHA-256s and a complete list | applied | All 38 rows of §10 recomputed with sha256sum: 0 mismatches, 0 missing. R3, R6, R1rule SHAs in the header also match. See N5. |
| M5 smoke and `_fix<n>` | applied | §9 Smoke bullet (with a wiring mutation) and a §8 row. `run_baselines.py` has `--smoke` and `--episodes-seed`. |
| M6 prior arithmetic | applied | "25% overall, about 40% given K1 passes"; 0.6 × 0.4 × 0.5 = 12% in rule, spec §3.1 and Brief. |
| M7 controls for T and R0 | applied | L8 last sentence. See N6 for the spec. |
| M8 strongest comparator beside the ceiling | applied | §5 descriptive line (told-T minus B1 and minus BT). |
| M9 rank of M | applied | L3 records singular values and the `pinv` rank. |
| M10 G3 intermediates | applied | Files exist: `rb_heads_{affect,image,caption,csd}.npz`, `rb_bank_A1_half{0,1}.npz`, `rb_reader_A1.json` `halves.{0,1}.chosen_C` = 1.0 and 100.0. See N3. |
| M11 prefix, AMI timing, dropped shares | applied | `r7_` in §9, AMIs after `gate_verdict.json` in P1, dropped shares over all 61,402 images in L5. |
| Spec diff: F3 "lower bound above 0" | applied | Spec D10 and §3.5 row 3 now say "one-sided α of its chain (rule §7.5)". |
| Spec diff: 15 readers | applied | 13 in Brief and §9, with the list. |
| Spec diff: stage 0 text | applied | Spec §3.5 row 0 matches G1 to G4. |
| Spec diff: §5 key `csd` | applied | Added to the first assumption. |

Feasibility checks (all pass): `rb_build.BLOCKS["A1"]` exists (line 53); `EvalContext` in `run_gonogo.py:207`; `r3_bundle.build_bundle` and `_check_seed`; `r3_fusion.run_family`; `RF.terms_from_probs`, `RF.fusion_pass`, `rc_core.thresholds`, `C.evaluate_fused`, `per_anchor`, `cluster_bootstrap`, `crossfit_condition_free`, `zscore_rows`, `_zdict`, `merge_small`, `fit_one_head`, `knn_union_graph` all exist. Keys `fused_r1`, `margin_r1` (both spike files), `diff__S1` (round 6), `aff_fused__r1`, `aff_cf__r1` (round 3) exist.

## New issues

**N1. §9 "Reused folders are read only" contradicts the builds of §7.2 and the smoke.** `run_baselines.py` writes `baselines_seed{55,56,57}.json`, its per-anchor files and `results/smoke/` into `20261030_aspect_baselines/results/`, a reused folder. Read literally, no seed can be built. Fix: add to the Read-only bullet: "Exception: `run_baselines.py` (§7.2, smoke) writes only new files for seeds 55 to 57 and 9001 to 9003 into its own `results/`, never with `--overwrite`; the SHA-256 of `codes_provenance.json` and of every earlier `baselines_seed*.json` is asserted unchanged after each build."

**N2. §7.3 wording and the csd head.** (a) "Whose only change is the seed guard" and "extended with the csd and X posteriors" disagree. Fix: "whose only changes are the seed guard and the added csd and X posteriors and readers (a diff against the pinned originals shows nothing else)". (b) "Round 6's modules are used only where their guard admits seed 42 (e.g. `r6_heads` ...), never for fresh-seed scoring" can be read as: no csd-head posteriors on seeds 55 to 57, but B1 needs them. `r6_heads.predict(heads, data, rows)` has no seed guard (it takes rows). Fix: "`r6_heads` (no seed guard) gives the csd-head posteriors on each seed's selection rows; `r6_context`, `r6_score` and `r6_bundle` are never called for seeds 55 to 57." Also change the §10 `r6_heads` row "seed 42" to "all seeds (no guard)".

**N3. G3 "exactly" has no definition for float arrays.** G3 gives 1e-12 for `predict_proba` but "exactly" for posteriors and banks. Fix: "exactly = `numpy.array_equal`, bitwise, at 8 threads, scikit-learn 1.6.1 and numpy 2.2.6 asserted; a difference stops the round and the report gives the largest absolute difference".

**N4. §7.5 F1 lists six items for a five-check Holm.** "F1: R@1, s − B, s − B0, s − B1, s − BX, s − CF[s]" reads as six. Fix: "F1, on R@1, five checks: s − B, s − B0, s − B1, s − BX, s − CF[s]".

**N5 (minor). Unpinned imports.** `r6_common.py` (imported by `r6_heads.py`) is not in the §10 code table. If round-7 code reuses `r6_stats.py` or `r3_stats.py` for the draws or the Holm form, those are not pinned either. Fix: pin them, or say the round-7 code copies the loop into an `r7_` module.

**N6 (minor, spec only). Spec still says each grouping has a random control.** Brief ("each grouping has a random size-matched control") and D13 ("per arm") do not mention that T and R0 have none (L8). Fix: "each candidate grouping" in the Brief and "per candidate arm" in D13.
