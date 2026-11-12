# r29 decision rule re-review

> 2026-11-12 22:04 · fresh Sonnet `high` worker, read-only · Spec: `docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md`
> (uncommitted diff vs HEAD, "revised 22:00") · First review: `r29_rule_review.md`

**Verdict: pass after fixes.** M1, M2 and M3 are applied. All recomputed numbers match to 10 decimals. One must-fix
remains: the table names real sources, but none of them stores per-unit values for the bootstrap, and no gate replays
the rebuilt values. Four should-fix items stop a night chat from applying the rule with no judgment.

## Must-fix

**M-A. §3.4 table, §3.5 (v) to (vii), §5 lines 281 to 283: comparator per-unit values are rebuilt, not stored, and nothing checks the rebuild.**
- Sources checked, all exist and hold what the spec says (point values and picks):
  - `cells.json` `photo/affection/*`, `photo/emoset/*` sources are r17 SC4, r14 `verdicts.json` SC5 / SC8 / SC10 and r13 `rows.teacher_artwork` (0.2945474692).
  - `photos.json` cell c: plain SigLIP 1 EmoSet, mAP 0.3638545779, AMI 0.2672607188. Its `r_source` is r14 `stage2_emoset.json` `ranges.map.r` = 0.0069460867 (the Qwen photo model's range; D19 should say so).
  - `picks_teacher.json`: one pick (center+k10) and a validation table for MMArt, n_val 5769.
- None gives per-unit values for a paired bootstrap:
  - r17 cell files are 6-cell summaries, with no per-query data. r13 `stage3/perquery` holds only photo arms. r14's EmoSet teacher (SC10, `teacher_emotion_focus`) is scored in `bs_emoset_score.py`. `picks_teacher.json` has no test values.
  - So P1 (plain) and Q1 (teacher) values for MMArt mAP, Affection mAP and EmoSet mAP must be rebuilt from the stores at the stated pick. The spec says "frozen per-unit values" (gate (v)) and §5 says gates (v) to (vii) "find out". They do not: no gate compares a rebuilt comparator with its recorded point value.
- Fix 1, §3.4, after the table: "Where a source holds only point values and a pick, the per-unit values (per-query AP, per-row labels) are rebuilt from the stores at that pick. The rebuilt point value must equal the recorded one within 1e-9 (gate (v))."
- Fix 2, replace the last sentence of (v): "Every rebuilt comparator point value equals its record within 1e-9: r13 `teacher_artwork` Affection mAP 0.2945474692; r14 SC10 EmoSet teacher mAP 0.4518753521; r14 SC5 Affection plain SigLIP 1 mAP 0.2545391742; `photos.json` cell c (mAP 0.3638545779, AMI 0.2672607188); r15's MMArt teacher and plain mAP at the `picks_teacher.json` pick (from r17's MMArt cells); r17's ArtELingo and MMArt plain and teacher cells. Failure parks the cells that use that comparator."
- Fix 3, replace §5 lines 281 to 283: "Per-unit values are rebuilt from the stores for P1 and Q1 in MMArt, Affection and EmoSet mAP (no stored per-query file exists); gate (v) checks the rebuild against the recorded point values."

## Should-fix

- **S-A. §3.5 and §6: no rule when gate (ii) fails.** It parks SC1 and SC2, so there is no K*. Add to §6 Inconclusive: "If SC1 or SC2 is parked, stage 1 is inconclusive; 1c and stage 2 do not run; Table 8 runs."
- **S-B. §3.4 Order and C13: no rule when Codex disagrees or is unavailable.** Add: "A re-derived value that differs by more than 1e-6 makes that pick, range or cell inconclusive until fixed; if Codex cannot run, the stage waits and the report says so. The night chat never decides a mismatch."
- **S-C. D20: the P1 caption-form pick has no stage or record.** Add to the 1a row and §3.4: "If K2 wins SC1, the P1 form is picked in stage 1a on validation and written to `prefix_pick.json` before 1b."
- **S-D. §9 time-box: no cut-off rule for stage 2.** Add: "Stage 2 pieces not finished by Saturday 11-14 12:00 are reported as not run (inconclusive cells, D22)."

## S and N tally

- S0 applied (2a / 2b, §3.3 rows). Stage column of ArtELingo validation omits 1c, though the text names it (trivial).
- S1 applied (unrounded bounds in §6 and `pass_rule.json`).
- S2 applied (1e-6 for mAP, float16 rule for towers, D23).
- S3 applied (empty-prefix replay, stored prefix, index hashes in (iv)).
- S4 applied, with a departure: (v) parks SC3 to SC6, since SC3 is stage 1c. Sound.
- S5 applied (§3.4 Crashes, D21, §6).
- S6 applied (§6, D22).
- S7 applied (D20, §3.2); the pick has no stage or record (S-C).
- S8 applied (§3.4 Order and C13).
- S9 applied (§3.1 priors; pass 0.4 stated with its assumption).
- S10 applied as in the review.
- S11 applied (§3.2 F1 paragraph).
- S12 applied (§6 level sentences).
- S13 applied (§3.6 (d) and (e)).
- S14 applied (D17 to D25 cover the listed choices).
- S15 applied (§6 Pick).
- S16 applied: times are now 21:30, 21:34, 21:38 and 22:00. I cannot check the grill times.
- N1 applied. N2 applied (the brief no longer gives the rounded number).
- N3 applied (§6 uses "N of 15"); the brief keeps "most cells" with the counts.
- N4 applied (SC1, SC7, SC8 sentences).
- N5 applied: no A1 to A4 left, R1-R4 used in §3.2, §3.5, §3.7, §6, D10. Nit: R0 is also r14's read-out name.
- N6 applied (187 s, two prefixed extractions, truncation count).
- N7 applied (1a row, `prefix_pick.json`).
- N8 partly applied: D6 is in "Check these", but the brief is still about 55 lines (template: about 40). Item 5 says "stage 2 are claim tests", while 2a is now a sweep.
- N9 applied.

## Consistency (steps 1 to 4)

- M1 applied as in the review, minus the sha256. M3 applied; (v) and (vi) wording differs slightly, same content. Gate (viii) also covers the SigLIP 2 and CLIP L/14 text stores. Sound.
- Brief against §3 to §9: stages, 12 of 15 / 5 of 6 / 3 of 4, cell counts (15, 6, 4, 6 each), cost (1.5 and 6 GPU-hours; §9 sums to about that), priors (0.8, 0.4, 0.6, 0.3) all agree.
- Gate letters (i) to (viii) are the same in §3.5, §7 and §8. Parks list: (i) (iii) all training; (ii) SC1 SC2; (iv) K2; (v) SC3 to SC6; (vi) SC5; (vii) SC6 SC7; (viii) its piece. Consistent with §6 Inconclusive.
- D17 to D25 match the text that uses them (§3.4, §3.5, §3.7, §6). D10 "R1-R4 ablation" is a stage type in name only (it is part of 2b).
- Table detail: "r17 SC7" for P1 image mAP is r17's plain side only (its candidate is the untold model); say "plain side of r17 SC7".

## Recomputed numbers (r28 records, CoSiR env)

| Group | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| Image (6 cells) | 0.3848908314 | 0.3857966619 | 0.3859741878 | 0.3869271680 | 0.3878696916 |
| Text (9 cells, r0) | 0.2595700735 | 0.2587331093 | 0.2606346548 | 0.2591946633 | 0.2586239413 |

| Group | m | r | m − r | Best seed |
|---|---|---|---|---|
| Image | 0.3862917081 | 0.0029788602 | 0.3833128479 | 0.3878696916 |
| Text | 0.2593512884 | 0.0020107136 | 0.2573405749 | 0.2606346548 |

Every value in §3.4 and §6 matches to 10 decimals.

## Can a night chat apply the rule with no judgment?

Yes for SC1 and SC2, and for the outcome words and ladders, once M-A and S-A to S-D are in. Before that, it cannot:
(1) check a rebuilt comparator, (2) handle a parked gate (ii), (3) handle a Codex mismatch, (4) place the P1 form pick, (5) stop stage 2.

Not done: no per-query files opened; no test run; the spec was not edited. Nothing to remove.
