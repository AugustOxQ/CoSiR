# Reader fix, round 4: vetoes on AFF's gate (brainstorm ideas 2 and 4), developed on seed 42, tested on fresh seeds (design)

**Date:** 2026-10-07 (Amsterdam). **Status:** the user settled the open points of the handoff
(`docs/superpowers/handoffs/2026-10-07-method-improvements-handoff.md` §5) one at a time in chat between 00:20 and 01:00
and approved both design sections; this file is the written spec. The binding rule will be
`src/test/20261122_round4_aff_vetoes/DECISION_RULE.md`, written from this spec, checked by a fresh Opus reviewer and
committed before any code. Where the rule and this spec differ, the rule governs.

## 1. Why this round

Round 3 (rule `src/test/20261121_round3_affect_gate/DECISION_RULE.md`, report
`docs/reports/auto/v2/2026-11-21_round3_affect_gate.md`) gave GO for AFF: R1 (round 1's learned reader with the
confidence gate on A0) with its gate opened only when the reader picks the affect grouping. Pooled over the fresh seeds
49 to 51, AFF's bar margin against B′(A0) was +0.591 [+0.462, +0.729] and it beat R1 by +0.202 [+0.093, +0.309]. Two
weaknesses remained: on style × genre AFF fell below B′(A0) (−0.580 [−0.793, −0.363]), because the reader picks affect on
the style side 77.6% of the time; and the random-share control matched AFF, so the gain comes from steering mostly one
side, not from which episodes within a condition get steered.

User decisions (2026-10-07, not reopened): round 3's GO stands and AFF, frozen as tested, is the current best; option B,
improve the method first with the brainstorm's remaining ideas (`docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md`
§3.2 to §3.4) before the held-split paper test, which is deferred, not dropped.

Settled for this round (00:20 to 01:00):

1. **Candidates:** the handoff's proposal. Idea 4 (style × genre abstention), idea 2 (CSD as detector evidence) and
   their combination, each built on AFF with its own matched counterpart, in one development family. Idea 3 (GoEmotions
   placement of captions) is not part of this round; it follows later as its own measured step (detector AUC first),
   and the question whether it counts as the parked grouping redesign is asked then.
2. **Development data:** seed 42, with round 2's development bar and a carry rule; the carried candidate is tested on the
   fresh seeds 52, 53 and 54.
3. **AFF in the bar:** on seed 42 a candidate is carried only if it beats AFF in paired fused R@1 (point above 0); on the
   test seeds the paired check against AFF is a GO check.
4. **The floor when CSD enters:** a candidate that reads CSD anywhere has B′(A1) as its floor; an A0-only candidate has
   B′(A0). B′(A1) is also reported, descriptively, beside AFF.

## 2. The candidates

All three keep every ingredient of AFF: round 1's two A0 half-readers (frozen), the term T^c = Σ_h P^c(h)·s_h on A0,
R1's thresholds τ_0..τ_3 (round 1's `results/rc_tau.json`), AFF's gate g^c = 1[m^c ≥ τ] · 1[π^c = affect], the 224 cells
(k_top 13 only), z-scoring before the gate, round 2's integer min-margin cross-fit. Each multiplies AFF's gate by one or
two extra factors, so each **only closes AFF's gate**, never opens it where AFF does not.

| Candidate | Gate | Reads CSD | Floor |
|---|---|---|---|
| **V4** (idea 4) | AFF's gate × 1[v < v₇₅] | no | B′(A0) |
| **V2** (idea 2) | AFF's gate × 1[π_A1^c = affect] | yes | B′(A1) |
| **V24** (ideas 2 and 4) | AFF's gate × 1[π_A1^c = affect] × 1[v < v₇₅] | yes | B′(A1) |

- **v, the abstention signal (idea 4):** v = min(S_image, C_image), the support and contrast agreements of the image
  grouping from the standard heads (A0 features 6 and 7 of condition a). Since condition b swaps supports and contrasts,
  v is the same in both conditions, so the factor shuts both conditions of an episode. It is high when both the
  supports and the contrasts agree visually, the style × genre signature (in the emotion pairs one side is non-visual).
  **v₇₅** = the 75th percentile of v over seed 42's 12,288 episodes, frozen for every seed, like τ. This is the
  brainstorm's best abstention (image signal, 75th percentile: +0.566 on R1, seed 42, exploratory; the CSD signal gave
  +0.48).
- **π_A1, the CSD-informed pick (idea 2):** the arg max of the mean probabilities of round 1's two frozen A1
  half-readers (`src/test/20261117_reader_fix_csd/results/rb_reader_A1.pkl`; groupings affect, image, caption, csd; 24
  features including the CSD agreements), ties to affect. V2 steers only when this reader also picks affect. The AND
  form was chosen over replacing AFF's pick: if the A1 reader picked affect while the A0 reader picked image, AFF's term
  would be image-dominated and opening the gate would steer with image. The brainstorm measured the A1 reader's
  P(affect) as an emotion-condition detector at AUC 0.824 against R1's 0.787 (seed 42, exploratory); this AND gate itself
  was not tried there.
- **Matched counterpart:** G_cf = the two-condition mean of the gated term under the candidate's own gates, round 2's
  integer max-R@1 cross-fit, as in round 3.
- **Comparators:** B, the candidate's floor (B′(A0) or B′(A1)) and its matched counterpart; the bar comparator is the
  strongest of the three. A candidate that reads CSD also keeps B′(A0) as a comparator.

## 3. Seed 42: regression checks, then the development step

There is one pipeline, parameterised by the episode seed: round 3's, reused by import, extended with the CSD
posteriors, the 24 A1 features, the A1 reader and B′(A1). Run on seed 42, before any candidate number, it must
reproduce:

1. round 3's seed-42 bundle and round 1's seed-42 A1 arrays (CSD posteriors, the 24 features, B′(A1)) exactly;
2. R1 = round-1 R-c and AFF = round 3's seed-42 targets, exactly as round 3's rule §5 items 2 and 3 state them;
3. R1 times the image abstention at v₇₅ = the brainstorm's recorded `IMGABST_q75` (`bs_04_readers.json`): fused R@1
   19.059244791666664, bar margin 0.5655924479166667 [0.3448683992591827, 0.79821625538382], at full precision. This
   checks the abstention code path;
4. the A1 reader's probabilities on seed 42 equal the ones stored by round 2 (`cand_R1_A1.npz`), if the rule confirms
   that they are the same half-reader mean.

Any difference stops the work and goes to the user with its cause traced.

**Development.** On seed 42 each candidate is evaluated against its own comparators: bar margin, gain statistic, and
fused R@1 minus AFF's fused R@1, paired per anchor.

- **Development bar** (round 2's D12): bar margin point at least +0.5, its lower bound above 0, and the gain
  statistic's lower bound above 0.
- **Carry:** among the candidates that clear the bar and whose paired fused R@1 point over AFF is above 0, carry the one
  with the largest paired gain over AFF; candidates within 0.05 of the largest are tied, and a tie goes to the simplest,
  in the order V4, V2, V24.
- **Kill:** if no candidate qualifies, no test seed is built, the seed-42 results go to the user, and AFF stays the
  current best.

**Expectation, stated now.** Against B′(A1) (18.805 on seed 42), V2 and V24 need a fused R@1 of about 19.31 to clear
the +0.5 bar, about +0.17 above AFF's 19.137. V4 is the likeliest to be carried, and its ceiling is small: style × genre
costs AFF about 0.19 of the pooled bar margin on the fresh seeds, and a full abstention there still leaves B's deficit
to B′(A0) on that pair.

## 4. The test on seeds 52, 53 and 54

- **Sensitivity projection, after the carry and before the build:** round 3's formula on the carried candidate's
  seed-42 per-episode differences, for every GO check including the one against AFF; logged, used only to read a failed
  check, never stops the round. Three test seeds, as the user chose, though power against AFF is a known risk.
- **Build:** seeds 52, 53 and 54, once each, with `run_baselines.py --episodes-seed <s>`; the per-pair episode hashes
  must differ from each other and from seeds 42, 43, 45, 47, 48, 49, 50 and 51; a ledger row.
- **Frozen from seed 42:** the A0 and A1 half-readers, τ_0..τ_3, v₇₅, the affect restriction, the 224-cell family and
  its tie rules, the heads, the recipes of B, B′(A0) and B′(A1), the method-A checkpoint. **Rerun on each seed's own
  parity halves:** B, B′(A0), B′(A1), the gates, σ*, both cross-fits and the counterpart.
- **Order:** until the verdict is written, only the GO quantities are computed: the per-anchor arrays of the carried
  candidate's fused reader and counterpart, AFF's fused reader, B, B′(A0), B′(A1) where it is a GO comparator, cosine
  and RCA, the chosen cells, and the pooled checks. Candidates that were not carried never run on the test seeds.
- **GO** if, pooled over the three seeds (one cluster per anchor painting across seeds; 5,000 resamples, seed 42), every
  check has a 95% lower bound above 0:
  - R@1, the candidate's fused reader minus each of cosine, RCA, B, B′(A0), B′(A1) (only for a candidate that reads
    CSD) and its matched counterpart;
  - the gain statistic;
  - condition gain, the candidate minus RCA;
  - **R@1, the candidate's fused reader minus AFF's fused reader, paired per anchor.**

  Eight checks for V4, nine for V2 or V24. A GO means the candidate replaces AFF as the current best for the held-split
  test.
- **Reading a failure:** a failed check with a pooled point above 0 is "inconclusive at a detectable margin of x"; at or
  below 0 it is "the candidate did not beat <comparator>". If the AFF check is the only one that fails, the reading is
  "the candidate works, but no improvement over AFF was shown", and AFF stays the current best.
- **Claim licensed:** the candidate beats AFF and each comparator, pooled over the three aspect pairs, on new episodes
  from the same 6,451 selection paintings; not transfer to new paintings, not a margin on each pair.

## 5. After the verdict (descriptive; decides nothing)

- Per-seed and per-pair results for the candidate and for AFF.
- **AFF's own seven checks** on seeds 52 to 54 (a replication of round 3), with **B′(A1) beside AFF**.
- Gate open shares per condition and pair, and the share of AFF's open values each veto closes, per pair and condition
  (does V4 close the style side of style × genre more than the emotion sides?).
- **Random-share control** at the carried candidate's per-condition τ_0 open shares, two draws, as round 3 defined it
  (a mechanism control that reads which condition is a, never a method).
- The frozen-cell line (seed 42's chosen cells applied to the test seeds) for the candidate and for AFF.

## 6. Disclosure and stated prior

- **Disclosure:** AFF was selected on seed 42 among about 50 variants; ideas 2 and 4 were explored on seed 42 on R1 (not
  on AFF), and V4's percentile and signal are the brainstorm's best of four abstention variants. The development numbers
  are therefore inflated; the fresh seeds 52 to 54 are the protection.
- **Prior,** written into the rule before any number: we expect the carried candidate's gain over AFF to be small, near
  or below its detectable margin; a NO-GO on the AFF check would not surprise us, and a kill at the carry would not
  either.

## 7. Process and timeline

Round 3's process. The rule is written from this spec with round 3's rule as the template, checked by a fresh Opus
reviewer (not an ARS round), fixed, committed before any code and sent to the user. A short implementation plan follows,
and the user chooses the execution method.

- **Before reusing round 3's code** (handoff §6, round 3's final review N11 and the deferred minors): a GO-pass assertion
  that `cl == groups[anchor]` and that AFF's family sees AFF's τ_0 open counts; a test that no candidate number is
  written before the regression checks pass; a criterion test for ρ_ctrl; `--boundary-reported` bound to the current
  `go_pooled.json`; a wiring-test leak check that also catches one-decimal numbers.
- Subagents implement; modules are prefixed `r4_` (`run_r4_`, `test_r4_`) so that none shares a name with a module of
  rounds 1 to 3. The main session launches every real run, CPU only, at most three processes.
- **Independent re-derivation** with its own code, in two phases: phase 1 on seed 42 (the regression checks, every
  candidate's development numbers and the carry), before any test seed is built; phase 2 on the test seeds (the GO
  quantities), before the rule is applied.
- The end-to-end wiring smoke test on seeds 9001 to 9003, after the sensitivity projection and before the build.
- The whole-branch final review on the most capable model, one fix wave and a scoped re-review, **before** the report is
  committed; the full report `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md` with its `reports_sum.md` row.
- Folder `src/test/20261122_round4_aff_vetoes/` (folder and report dates are sequence numbers). Commits go to `main`,
  scoped by explicit path, never pushed.
- Target: the verdict on Wednesday 7 October if the runs allow; the time-box ends Wednesday 14 October.
