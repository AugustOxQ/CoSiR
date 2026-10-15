# ARS methodology-focus review of the method-repair order (log)

Date: 2026-10-03 (folder sequence date 2026-11-04). No experiment, training, GPU job or commit was made.

## Problem

After the E3 NO-GO the user chose a method repair (A′). Three candidate first steps exist: H1 (nested test-time score
pilot), H2 (fit repair grid), H3 (label-trained learnability diagnostic). The user leaned toward H3 first and asked for
an ARS check, methodology-focus mode, before deciding.

## Steps

1. Wrote the decision memo `memo.md` (about 3,700 words: decision, task and GO test, E3 evidence, GO arithmetic, H1 to
   H4, concrete H3 and H1 designs with outcome rules, orders O1 to O4, fixed constraints, known weaknesses, five
   questions). `manuscript.md` = memo + verbatim appendices (E3 report, repair handoff, spec §3 and §6); 19,424 words.
2. Contract `contract.json`: frozen `reviewer/reviewer_methodology_focus/v2` template plus `generated_at` and a stage
   note; `check_sprint_contract.py` OK (SC-12 warning inherent to the two-seat template).
3. Phase 0 (`phase0_reviewer_configuration.md`): two cards, Journal-Fit Reviewer (D2) and Peer Reviewer 1 methodology
   (D1). The user confirmed them unchanged.
4. Phase 1, paper-content-blind, both seats (`phase1_eic.md`, `phase1_methodology.md`): `check_phase_conformance.py
   --phase1-only` PASS for both.
5. Phase 2 (`phase2_eic.md`, `phase2_methodology.md`): conformance PASS and `check_panel_synthesis.py --layer1-only`
   PASS for both.
6. Synthesis (`editorial_decision.md`): `check_panel_synthesis.py` PASS (exit 0); acronym check appended as the last
   write. The typed provenance artifact is closed to reviewer_full, so the letter carries a plain-text provenance note
   (one model family, fresh contexts, blind Phase 1, no peer visibility, no human seat).

## Outcome

- `dimension_verdicts: [D1=block, D2=warn]`, `fired_conditions: [F2, F4]`, **editorial_decision=major_revision**. D1's
  block is repairable; nothing shows seed 45 or the held rows spent.
- Order: neither seat supports H3 first and alone. The methodology seat recommends a modified O3 (commit revised rules
  first; day 1 the H1 pilot on A3 plus the LAB runs L3, L5, a fixed-τ run and a matched-k run in one GPU lock; day 2
  paired scoring against A3 and C0 and a joint decision table that stops the repair if both readings are negative).
  If only one code stream fits per day: H1 then H3, never H3 first.
- Controller checks: re-derived the methodology seat's W2 algebra (at equal mean(R@1, gain), R@1 changes by minus the
  gain change; +1.0 gain and −2.9 either gives criterion +0.025 and R@1 −0.95) and its W1 example (A3 on E3's fresh
  pseudo episodes: agreement rule [−0.12, 0.86] reads "no fit", training score [0.28, 1.62] reads "fits"). The
  journal-fit seat's W4 is correct: the memo's AMI row came from E2's `build_record.json`, not Appendix A.

## Open

- The memo is not revised yet; the required changes R1 to R12 and the optional S items are listed in the letter.
- No report in `docs/reports/` yet (the 2026-10-27 precedent wrote its report after the user ruled on the changes).
