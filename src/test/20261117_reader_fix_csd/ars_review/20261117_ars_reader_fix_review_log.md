# ARS methodology-focus review of plan (a), the reader fix with the CSD grouping (log)

Date: 2026-10-06, 00:30 to 01:25 (Amsterdam; folder sequence date 2026-11-17). No code, episode, experiment, GPU job,
`DECISION_RULE.md` or commit was made. The review folder sits inside the reader-fix folder that the handoff reserved
(`src/test/20261117_reader_fix_csd/`, `.gitignore` copied from `20261023_aspect_episode_spike/`).

## Problem

The handoff `docs/superpowers/handoffs/2026-10-06-reader-fix-with-csd-handoff.md` (uncommitted) sets plan (a): reader
candidates R-a (scaled Δ), R-b (learned reader on a pseudo-aspect bank), R-c (confidence gate), selected on seed 42 by a
draft rule (§5.4) and tested once on fresh seeds 49 to 51 before the 9 October go/no-go. The user ordered an ARS
methodology-focus review of §5 (including §5.4 and the targets in §5.6) before any code.

## Steps

1. Read the handoff and its §2 reading list, and the code the plan names (`build_episode_bank`, `aspect_deltas`,
   `inferred_scores`, `crossfit_condition_free`, `crossfit_nested`, step 1's `evaluate_general` and `bar_margin`).
2. Wrote the cover memo `memo.md` (decision requested, six questions, fixed constraints, code-level facts the plan relies
   on, §5.6 verbatim). `manuscript.md` = memo + verbatim appendices A (the handoff), B (the grouping report), C (step-1
   log, Results to Controller review), D (the 2026-10-04 reader handoff), E (stage report §2 and §14), F (the
   matched-control lesson and the seed-handling preference); 19,092 words.
3. Contract `contract.json`: frozen `reviewer/reviewer_methodology_focus/v2` plus `generated_at` and a stage note (the
   first note exceeded the schema's 500 characters and was shortened); `check_sprint_contract.py` OK (SC-12 warning
   inherent to the two-seat template).
4. Phase 0 (`phase0_reviewer_configuration.md`): Journal-Fit Reviewer (D2) and Peer Reviewer 1, methodology (D1). The
   user confirmed both cards unchanged.
5. Phase 1, paper-content-blind, both seats, run in parallel with Phase 0 (it reads only contract and metadata):
   `check_phase_conformance.py --phase1-only` PASS for both.
6. Phase 2 (`phase2_eic.md`, `phase2_methodology.md`): conformance PASS and `check_panel_synthesis.py --layer1-only`
   PASS for both.
7. Synthesis (`editorial_decision.md`): `check_panel_synthesis.py` PASS (exit 0); acronym check appended as the last
   write. Plain-text provenance (one model family, fresh contexts, blind Phase 1, no peer visibility, no human seat).

## Outcome

- `dimension_verdicts: [D1=block, D2=block]`, `fired_conditions: [F2, F4]`, **editorial_decision=major_revision**.
  D1's block is repairable; the fatal trigger (test reusing development material) was considered and judged not met,
  because fresh seeds draw new episodes and nothing selected on seed 42 is fitted per painting; the shared paintings are
  a scope limit on what a GO licenses.
- Must fix (R1 to R5): the development bar alone decides the kill, pick accuracy and R-b's bank accuracy become
  diagnostics, and only GO quantities are computed on test seeds before the verdict (R1); one definitions block (fusion
  on B, counterpart per reader, B′ rebuilt, the gain statistic, the bar comparator) (R2); the rule file governs (R3);
  R-b's domain-shift check becomes a stated diagnostic plus a label-free shift report (R4; arbitrated against the
  Journal-Fit seat's numeric kill); R-c fully specified with the condition-free counterpart
  G_cf = (g_a·z(T_a) + g_b·z(T_b)) / 2 on 224 cells (R5; the methodology seat's Critical finding).
- Should fix S1 to S17, of which the two Major ones are S9 (R-a's spread from the pooled within-episode standard error
  instead of the RMS, which mixes signal into the noise scale) and S10 (a frozen R-b specification: 60,000-row
  cross-fitted heads per half, fixed bank, C by cross-validation on the bank, A0 bank, no change after seed-42 numbers).
  The Journal-Fit seat needs the outcome table (S2) for D2 to reach pass.

## Controller checks

Re-derived by the main session (by hand and from the code; no new run):
- R5 / Meth W1: `crossfit_condition_free` calls `_require_condition_free`, which raises unless each term is identical
  under both conditions (`src/eval/aspect_quick_checks.py`); the top-two margin of Δ under b is the bottom-two gap of Δ
  under a, so g_a ≠ g_b and g_c·z(T_cf) fails the check. Confirmed.
- S9 / Meth W3: RMS² = noise variance + mean squared signal when the two conditions are pooled (Δ_b = −Δ_a). With the
  card's inputs, RMS = sqrt(0.017² + (0.0055² + 0.0236² + 0.0205²)/3) = 0.025; scaled image Δ in emotion × genre
  0.0236/0.025 = 0.94 against 0.0236/0.017 = 1.39; P(an uninformative grouping outranks it) = Φ(−0.94/1.21) = 0.22
  against Φ(−1.39/1.41) = 0.16. Confirmed.
- S12 / Meth W6: SE from the seed-42 half-widths 0.215 and 0.235 is 0.11 to 0.12; pooled half-width 0.12 to 0.14
  (episode noise, ×1/√3) or 0.18 to 0.20 (painting noise, ×√(4602/6451)); power at +0.25 Φ(0.25/0.063 − 1.96) ≈ 0.98
  and Φ(0.25/0.10 − 1.96) ≈ 0.71. Agrees with the card's 0.96 and 0.73 within rounding.
- S4 / Meth W9: E[max of 7 standard normals] ≈ 1.35. Confirmed.
- Cross-fit halves are episode index mod 2 (`src/test/20261101_aspect_factor_gonogo/run_gonogo.py:242`), as the
  memo states.
- S17 / Meth Q4 answered from the record: the two halvings behind the +0.5 bar were condition gains, not R@1 margins
  against a matched control (E3 gain 0.52 to 0.26, stage report §5.2; N1 gain 0.26 to 0.15 pooled, quick-checks report
  line 164). No reader margin against a matched counterpart has yet been measured on fresh seeds.

Not raised by the panel (controller addition): the random-slot control AR (A0 plus a random 17-group grouping) is
dropped from §5. Running every reader on AR as a label-free sanity check (the share of rankings that pick the random
grouping should fall toward 0 under R-a) costs minutes and shows whether the noise scaling does what it claims.

## Open

- The plan is not revised yet. The user decides which R and S items to adopt (and the cutoff time, whether an A0 GO
  counts for plan (a), and whether any R-b kill is wanted), then §5 is rewritten as `DECISION_RULE.md`.
- No report in `docs/reports/` yet (the precedent wrote it after the user ruled on the changes).
- Files: everything in this folder is small (largest 119 KB); nothing over 100 MB.
