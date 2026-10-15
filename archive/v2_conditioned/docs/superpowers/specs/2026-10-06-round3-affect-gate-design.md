# Reader fix, round 3: one-sided affect steering on R1, tested on fresh seeds (design)

**Date:** 2026-10-06 (Amsterdam). **Status:** the user settled the open points of the handoff
(`docs/superpowers/handoffs/2026-10-06-round3-affect-gate-handoff.md` §4) one at a time in chat between 19:11 and 19:21
and approved this design; this file is the written spec. The binding rule will be
`src/test/20261121_round3_affect_gate/DECISION_RULE.md`, written from this spec, checked by a fresh Opus reviewer and
committed before any code. Where the rule and this spec differ, the rule governs.

## 1. Why this round

Round 2 (rule `src/test/20261118_reader_fix_round2/DECISION_RULE.md`, report
`docs/reports/auto/v2/2026-11-19_reader_fix_round2.md`) found no candidate that cleared the development bar. Its best
reader, R1 (round 1's learned reader with the confidence gate on A0), had a bar margin of +0.472 in round 2's 896-cell
family and +0.444 (round-1 R-c exactly) in the 224-cell family. The brainstorm that followed
(`docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md`, exploratory) traced R1's margin to one place: lifting the
emotion candidate in the condition whose supports show emotion. Image and caption picks never paid, because B already
ranks what those groupings carry. Its rank-1 idea, one-sided affect steering (AFF), keeps R1 and opens the gate only
when the reader picks affect.

User decisions (2026-10-06, not reopened): keep improving R1, no design L now; round 3 tests AFF as a pre-registered
round with R1 beside it, straight on the fresh seeds 49, 50 and 51. AFF was found by a search of about 50 variants on
seed 42, so seed 42 cannot serve as its development data.

## 2. The candidate

AFF keeps every ingredient of R1 and changes one factor of the gate:

- **Reader:** round 1's two A0 half-readers (`src/test/20261117_reader_fix_csd/results/rb_reader_A0.pkl`), frozen,
  probabilities averaged over the halves. A0 = (affect, image, caption), the standard heads of rounds 1 and 2.
- **Term:** T^c = Σ_h P^c(h)·s_h on A0, unchanged.
- **Thresholds:** R1's τ_0..τ_3 from round 1's `results/rc_tau.json`, frozen.
- **Gate:** g^c = 1[m^c ≥ τ] · 1[arg max_h P^c(h) = affect]. Ties in the arg max go to the first grouping in A0 order,
  which is affect. R1's gate is the first factor alone.
- **Fusion:** R1's 224 cells (k_top 13 only; the top-k restriction is dropped because it never helped), z-scoring before
  the gate, round 2's integer min-margin cross-fit (round 2's rule §4.5 item 8).
- **Matched counterpart:** G_cf = the two-condition mean of the gated term under the same gates, with round 2's integer
  max-R@1 cross-fit (§4.6).
- **Comparators:** B, B′(A0) and the matched counterpart; the bar comparator is the strongest of the three (round 2's
  D10).

**Affect is frozen by name.** Its label-free reason is recorded in the rule: affect is the A0 grouping least redundant
with B, measured as the mean per-row Pearson correlation over the 13 candidates of z(s_h) with z(B) on seed 42 (affect
0.353 i2t and 0.383 t2i, caption 0.618 and 0.665, image 0.715 and 0.709). Round-3 code recomputes this on seed 42, and
it must still name affect. The criterion is not re-chosen on the test seeds; its test-seed values are reported after the
verdict. It was stated after affect was seen to pay, so it explains the choice but does not protect it.

**R1 beside it:** R1 runs through the same pipeline on the same 224 cells (round-1 R-c).

## 3. Seed 42: regression checks only

There is one pipeline, parameterised by the episode seed. Run on seed 42, before any seed is built, it must reproduce:

1. round 1's seed-42 bundle (B, B′(A0), the grouping scores s_h and the 18 reader features), exactly;
2. R1 = round-1 R-c's stored arrays (`cand_Rc_Rb_expected_A0.{npz,json}`, `rc_tau.json`) exactly, as round 2's rule §4.7
   specifies for the k_top 13 cells;
3. AFF = the brainstorm's recorded numbers, at full precision: fused R@1 19.136555989583336, counterpart
   18.39599609375, bar margin 0.6998697916666667 [0.4598852740816973, 0.9371680126852968] against B′, gain statistic
   3.110758463541667 [2.780005709854805, 3.4559584315470384], and the chosen cells (fused τ_0 (4, 16) and τ_2 (0, 16),
   counterpart τ_2 (4, 4) and τ_0 (0.5, 0.5), as (τ index, λ_u, λ_a) for parity halves 0 and 1).

This is also the end-to-end wiring test of the run scripts that round 2's final review asked for (its N8). Any
difference stops the work and goes to the user with its cause traced. The development bar (round 2's D12) is evaluated
for AFF on seed 42 and recorded with the disclosure of §6; if it fails, which only a code difference could cause, the
work stops. No other variant is computed on seed 42.

## 4. The test on seeds 49, 50 and 51

- **Sensitivity projection, before the seeds are built:** round 2's rule §6.1 formula on AFF's seed-42 per-episode
  differences, for each of the seven GO checks and for the secondary check, written to the log (SE, half-width,
  detectable margin x = 2.80·SE). It is used only to read a failed check; it never stops the round.
- **Build:** seeds 49, 50 and 51, once each, with `run_baselines.py --episodes-seed <s>`; the per-pair episode hashes
  must differ from each other and from seeds 42, 43, 45, 47 and 48; a ledger row.
- **Frozen from seed 42:** the readers, τ, the affect restriction, the 224-cell family and its tie rules, the heads, the
  recipes of B and B′(A0), the method-A checkpoint. **Rerun on each seed's own parity halves:** B, B′(A0), the gates
  (that seed's margins and picks against the frozen τ), both cross-fits and the counterpart.
- **Order:** only the GO quantities and the secondary check's inputs are computed until the verdict is written.
- **GO** if, pooled over the three seeds (one cluster per anchor painting across seeds; 5,000 resamples, seed 42), all
  seven of round 2's checks have a 95% lower bound above 0: R@1 of AFF's fused reader minus each of cosine, RCA, B, B′
  and its matched counterpart; the gain statistic; the condition gain minus RCA's.
- **Secondary check (pre-registered, never changes GO):** fused R@1, AFF minus R1, paired per anchor, pooled; it passes
  if its 95% lower bound is above 0, and its pass or fail is recorded with the verdict.
- **Reading a failure** (round 2's §6.6): a failed check with a pooled point estimate above 0 is "inconclusive at a
  detectable margin of x"; one at or below 0 is "AFF did not beat <comparator>". The secondary check is read the same
  way.
- **Claim licensed** (round 2's §6.9): AFF beats each comparator, pooled over the three aspect pairs, on new episodes
  from the same 6,451 selection paintings; not transfer to new paintings, not a margin on each pair.

## 5. After the verdict (descriptive; decides nothing)

- Per-seed and per-pair results; the frozen-cell line (seed 42's chosen cells applied to the test seeds, as round 2's
  §6.8); gate-open shares per condition and pair; chosen cells; pick accuracy (told mapping).
- **R1's own seven checks** on the test seeds, labelled descriptive. R1 never gives a second verdict, even if it passes
  and AFF fails; the next step is then the user's call.
- The redundancy criterion's values on the test seeds.
- **Random-share control** (a mechanism control that reads which condition is a, so never a method): R1's gates times a
  Bernoulli mask with AFF's τ_0 open share in each condition on that seed, two draws, the 224 cells, its own cross-fit
  and its own counterpart. Reported: its bar margin, and AFF minus the control (fused R@1 and bar margin, paired). On
  seed 42 (exploratory) the two draws gave +0.665 and +0.564 against AFF's +0.700. It tells whether the reader's choice
  within a condition adds anything beyond steering one side.
- The brainstorm's ideas 2 to 4 are not part of this round.

## 6. Disclosure and stated prior

- **Disclosure** (reported with every AFF number): AFF was found among about 50 label-free variants read on seed 42; the
  median of its one-sided cluster (+0.63, range +0.42 to +0.72) is a better guide than its +0.700; the fresh seeds 49 to
  51 are the protection.
- **Prior, written before any test number:** we expect a fresh-seed bar margin of about +0.3 (about half the cluster
  median, since earlier fresh-seed tests roughly halved development effects), above a projected detectable margin of
  about 0.2 (the round-2 tab's exploratory projection for R1 against its counterpart). Our chance of a GO is moderate:
  a NO-GO would not be a surprise, and a large fresh-seed margin would be.

## 7. Process and timeline

Round 2's process. The rule is written from this spec and checked by a fresh Opus reviewer (not an ARS round). Its
findings are fixed and it is committed before any code, then sent to the user. A short implementation plan follows,
and the user chooses the execution method. Subagents implement: the new piece is the seed-parameterised pipeline, with
the AFF gate on round 2's fusion code, reused by import. The main session launches every real run on CPU only. An
independent agent re-derives every decision number with its own code before the verdict is recorded. The whole-branch
final review runs on the most capable model, with one fix wave and a scoped re-review. The full report goes to
`docs/reports/auto/v2/2026-11-21_round3_affect_gate.md` with its `reports_sum.md` row. Folder:
`src/test/20261121_round3_affect_gate/` (folder and report dates are sequence numbers). Commits go to `main`, scoped,
never pushed. Target: the verdict on Tuesday 6 October if the runs allow; the time-box ends Tuesday 13 October.

Carried from round 2's final review: an end-to-end wiring test (here §3); a +1e-12 tolerance at the 0.05 carry-tie
boundary (no carry step in this round, so only where a tie boundary arises); an explicit τ tolerance in the
re-derivation's agreement (absolute 1e-15 or relative 1e-9). Smoke runs print no candidate number before the
regression checks of §3 pass.
