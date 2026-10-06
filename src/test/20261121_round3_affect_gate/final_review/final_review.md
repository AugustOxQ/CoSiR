# Final whole-branch review: reader-fix round 3 (3b71e2f..02215b8)

Written 2026-10-06 21:49 to 22:30 (Amsterdam) by the whole-branch final reviewer (fresh context). Binding rule
`../DECISION_RULE.md` (SHA-256 2d311dbe…5925, committed fab5ae1, re-hashed now: unchanged). Scope: the 13 commits
728f5d7, fab5ae1, 83b51f4, a4b1af9, f1d6d25, 137d46f, 05fca40, 7ece674, d38f1f9, 2f8c784, e48cd9c, 66d68a0, 02215b8.
Everything here was computed on CPU with this review's own scripts in this folder (`fr3_*.py`). `results/` and
`rederive/` were only read (never `build_seed{s}.log`, never a held row); no committed file was changed; the mutation
checks of §4 ran on scratch copies in `final_review/mut/w/` (deleted afterwards). Nothing was committed.

## Verdict: CONFIRMED WITH FIXES

**The GO stands.** I rebuilt the three test seeds' bundles myself (from `EvalContext`, `model_inputs` +
`centered_term`, `load_posteriors`, `fit_one_head`, `load_readers`, `crossfit_condition_free`,
`uniform_probe_scores`), and wrote my own features, grouping scores, reader arithmetic, gates, z-score fusion (only
`zscore_rows` imported), integer cell statistics, control σ*, both cross-fits, assembly, per-episode metrics,
comparators, pooling and checks (only `cluster_bootstrap` imported). All 862 automated comparisons with the stored
results agree exactly (bit for bit for arrays, `==` for floats): 691 for the test seeds (`out/fr3_run_comparisons.json`)
and 171 for seed 42 and the bundle caches (`out/fr3_seed42_comparisons.json`). About 250 further numbers of the report
were checked against my outputs at the precision shown; two brainstorm quotes are off in the last digit (N5), none of
the round's own numbers is wrong.

The fixes are report text and one labelling decision for the user (3 should-fix, 11 nits). No number, verdict or code
changes. The mutation checks found six guards or choices that no test protects (§4: M06, M09, M20, M27, M28, M31). For
this round each is covered by the run's own records, the seed-42 regression, the phase-2 re-derivation or this review,
so they are recorded for the next reuse of the code (N11), not fixed now.

## 1. What was re-derived and how

| Script | What it does (own code unless stated) |
|---|---|
| `fr3_lib.py` | From the rule text: D5 features (float32 agreements as D3, ddof-1 spreads, match share), D4 s_h, P (mean of the two half-readers' `predict_proba`), T, π (np.argmax), m; D6 gates of R1 and AFF; §7.7 random gates; D8 fusion in `_combine`'s order, float32; integer ρ, γ per cell and episode; D8.5 control σ* over the 30 sums; min-margin and max-ρ cross-fits with ties to the lowest cell; D9 G_cf in float64 then float32 with the per-cell condition-free assertion; assembly; per-episode r1, gain, other, swap, strict; D12 bar comparator; §6.5 checks; §6.1 sensitivity; D7 redundancy. Imports only `zscore_rows` and `cluster_bootstrap` |
| `fr3_build.py` | Seeds 49, 50, 51 and 42 through rule §4 item 1's call sequence with the allowed loaders; my own `global_labels`; D2 head identity with told_oracle.json arm L checked (true on every seed); B's T_6u via `uniform_probe_scores(post_E2, ep, (affect, image, caption))`, which is `run_n6.n6_terms(...)[2]` by its definition |
| `fr3_seed42.py` | My seed-42 bundle against round 1's stored arrays (T, margins, picks, gates, fused and counterpart arrays, bar_v), τ recomputed, §5 items 2 to 4, D7, the sensitivity projection; then my test-seed bundles against `results/cache_seed{49,50,51}.npz` |
| `fr3_run.py` | AFF, R1 and the two random-share families per test seed; frozen-cell line; GO checks and secondary; every §7 quantity and the report's diagnostics; comparison with `go_seed{s}.npz`, `cache_reader_seed{s}.npz`, `go_pooled.json`, `test_verdict.json`, `descriptive.json` |
| `fr3_seed49_equality.py` | Seed 49 against seed 42: episode overlap and per-pair net counts |
| `fr3_side.py`, `fr3_cites.py` | Review diagnostics for the report's mechanism reading (§3, S1) and its brainstorm quotes (N5) |
| `fr3_mutate.py` | The mutation checks of §4 on scratch copies |

The caches were not used as a source: I built all three test seeds myself and then compared (cos, B, B′, s_h, F,
anchors, parity, pair index, episode SHA-256s: all identical).

## 2. Re-derived numbers

"Agree" means identical at full precision unless stated. Arrays: bit-identical.

### 2.1 Decision numbers (rule §6.5, §6.6; pooled over 49, 50, 51; 36,864 episodes, 5,195 painting clusters)

| Check | Reported (`go_pooled.json` = `test_verdict.json`) | Re-derived | Agree |
|---|---|---|---|
| R@1, AFF − cosine | 5.839708116319445 [5.6113334691723, 6.068553623829037] | same | yes |
| R@1, AFF − RCA | 5.721706814236112 [5.495126735024904, 5.9456186249255225] | same | yes |
| R@1, AFF − B | 0.8063422309027778 [0.6862297310541294, 0.9302913961004797] | same | yes |
| R@1, AFF − B′(A0) | 0.5913628472222222 [0.46150280307786573, 0.7290529246365278] | same | yes |
| R@1, AFF − its counterpart | 0.7961697048611112 [0.669718134664606, 0.9201667199227945] | same | yes |
| Gain statistic | 3.3189561631944446 [3.1301212087937755, 3.5178707882265416] | same | yes |
| Gain, AFF − RCA | 3.2857259114583335 [3.0764180203333327, 3.494111586858046] | same | yes |
| Secondary, AFF − R1 fused | 0.2020941840277778 [0.09320136111224805, 0.3087028210199184] | same | yes |
| Pass flags; verdict | all pass; GO; secondary passes | same | yes |
| Smallest lower bound; within 1e-12 of 0 | +0.0932 (secondary), +0.4615 (GO); none | same | yes |
| Clusters | 5,195 global painting ids (`groups[anchor]`), shared across seeds | same | yes |
| Per seed: AFF fused / cf cells, R1 fused cells, σ* | 49: 119, 63 / 158, 156 / 127, 93; 50: 62, 127 / 114, 137 / 14, 151; 51: 118, 119 / 115, 170 / 117, 68; σ* 0, 0 | same, with the same tie counts as the phase-2 report | yes |
| Per-anchor arrays (7 scorers × 5 metrics × 3 seeds) | `go_seed{s}.npz` | bit-identical | yes |
| P, m, π, T (both conditions, 3 seeds) | `cache_reader_seed{s}.npz` | bit-identical | yes |
| Bundles (cos, B, B′, s_h, 18 features, anchors, parity, pair index, episode SHA-256s) | `cache_seed{s}.npz` | bit-identical | yes |
| cosine per anchor = `per_anchor_seed{s}.npz` cosine; anchor groups and pair index aligned | asserted | same | yes |

### 2.2 Seed 42 (rule §5, D7, D13) and the sensitivity projection (§6.1)

| Quantity | Rule / stored | Re-derived | Agree |
|---|---|---|---|
| B, B′(A0) R@1 | 18.341064453125, 18.436686197916664 | same | yes |
| R1's T, margins, picks; gates at τ_0..3 | `cand_Rc_Rb_expected_A0.npz` | bit-identical | yes |
| τ = percentile(m, [0, 25, 50, 75]) of the 24,576 margins | rc_tau.json | equal | yes |
| R1 cells fused / cf; σ* | 116, 119 / 58, 123; 0, 0 | same | yes |
| R1 fused and cf arrays (10), bar_v | stored | bit-identical | yes |
| R1 bar margin (vs counterpart); gain statistic | 0.4435221354166667 [0.21646171563312194, 0.6735669710776852]; 2.667236328125 [2.325087836946873, 3.012361650695922] | same | yes |
| AFF fused / cf R@1; comparator | 19.136555989583336 / 18.39599609375; B′(A0) | same | yes |
| AFF bar margin; margin; gain statistic; either | 0.6998697916666667 [0.4598852740816973, 0.9371680126852968]; 0.7405598958333333 [0.5196896694963071, 0.9598857494832738]; 3.110758463541667 [2.780005709854805, 3.4559584315470384]; −1.629638671875 | same | yes |
| AFF per-pair bar margins | 0.9765625, 1.45263671875, −0.32958984375 | same | yes |
| AFF cells fused / cf; σ* | 39, 119 / 149, 10; 0, 0 | same | yes |
| AFF − R1 fused; bar | 0.21769205729166666 [0.06425880757348419, 0.3709597330984391]; 0.25634765625 [0.04280778303598444, 0.46195041633015954] | same | yes |
| AFF τ_0 open counts | 9,941 and 3,627 | same | yes |
| D13 clauses | all hold | all hold | yes |
| D7 six values; affect smallest | rule values | equal (exact) | yes |
| Sensitivity x (pp): cosine, RCA, B, B′, cf, gain, gain−RCA, secondary | 0.32446, 0.31969, 0.17973, 0.19330, 0.18120, 0.27953, 0.31141, 0.13070 | same within 1e-9 (SE, x, 1.96 SE, seed-42 half-width all checked) | yes |
| Seed-42 points (Table 4) | +6.175, +5.756, +0.795, +0.700, +0.741, +3.111, +3.007, +0.218 | same | yes |
| R1 − B′(A0), R1 − B on seed 42 | +0.482 [+0.236, +0.727]; +0.578 | +0.4822 [+0.2357, +0.7270]; +0.5778 | yes |

### 2.3 Descriptive pass (rule §7) and the report's tables

| Quantity | Reported | Re-derived | Agree |
|---|---|---|---|
| Per seed, seven checks and secondary (24 + 3 × 3 bounds) | `descriptive.json` item 1, Table 5 | same; all seven pass on every seed; secondary fails on 51 alone (+0.159 [−0.022, +0.342]) | yes |
| Per seed B′ / AFF R@1; bar; margin; gain; either | 49: 18.231 / 18.931, +0.700, +0.850, +3.593, −1.892; 50: 18.337 / 18.789, +0.452, +0.702, +3.267, −1.864; 51: 18.296 / 18.919, +0.623, +0.836, +3.097, −1.424 | same | yes |
| Bar comparator | B′(A0) on every seed and pooled (pooled B′ 18.288, cf 18.083, B 18.073) | same | yes |
| Per pair, seven checks + secondary (Table 8) | e×s B′ +0.810, e×g +1.544, s×g −0.580 [−0.793, −0.363]; s×g vs B −0.336 [−0.526, −0.149], vs cf −0.149 [−0.348, +0.051] | same | yes |
| AFF vs counterpart, pooled: margin, gain, either | +0.796, +3.319, −1.727 [−1.907, −1.546] | same | yes |
| AFF − R1 bar margin, pooled | +0.202 [+0.093, +0.309] | same | yes |
| R1's seven checks pooled; bar margin; per seed vs B′ | Table 7; +0.389 [+0.254, +0.532]; +0.515, +0.189 [−0.079, +0.452], +0.464 | same | yes |
| R1 vs its counterpart: margin, gain, either | +0.519, +2.521, −1.482 | same (either −1.482 [−1.679, −1.287]) | yes |
| Frozen-cell line: AFF / R1 bar pooled; per seed; AFF − R1 | +0.585 [+0.455, +0.719] / +0.495 [+0.362, +0.633]; +0.730, +0.450, +0.576 / +0.602, +0.460, +0.423; +0.090 [+0.005, +0.177] (+0.128, −0.010, +0.153) | same | yes |
| Frozen AFF gain statistic; per-pair bar | +3.162; +0.960, +1.390, −0.594 | same | yes |
| Gate shares pooled (AFF, R1, 4 τ × overall/a/b) | Table 12 | same within 1e-12 pp (integer counts) | yes |
| R1 τ_0 closures on the test seeds | 2 of 36,864 condition-b values | 0 in a, 2 in b | yes |
| Pick accuracy; both correct; per pair a/b; picks by condition | 51.4 [51.1, 51.8]; 25.1; 78.4/37.3, 85.8/52.0, 9.1/45.8; a 80.6/6.0/13.4, b 29.7/45.0/25.3 | same | yes |
| AFF τ_0 open share per pair a/b | 78.4/41.9, 85.8/22.0, 77.6/25.2 | same | yes |
| D7 per test seed (Table 13) | 18 values | same within 1e-9 | yes |
| Random-share control: shares per seed; bar per draw; gain; AFF − control | 80.1 to 80.9 / 29.5 to 30.0; +0.633 [+0.504, +0.769], +0.562 [+0.426, +0.699]; +3.240, +3.096; −0.042 [−0.138, +0.048], +0.029 [−0.076, +0.132] | same | yes |
| Control and R1 chosen cells per seed | `descriptive.json` | same | yes |
| Per-pair AFF − control | +0.037/+0.031, +0.157/+0.279, −0.319/−0.222 | same | yes |
| Table 9 (R@1, gain, other, either vs B′, per pair, AFF and R1, 24 intervals) | report | same | yes |
| Pooled means cosine, RCA, R1 fused | 13.040, 13.158, 18.677 | same | yes |
| Kept share 0.591 / 0.700; either per unit gain AFF test / seed 42 / R1 | 84%; 0.52 / 0.52 / 0.59 | 0.845; 0.520 / 0.524 / 0.588 | yes |
| Realised vs projected half-widths | within 5% except secondary (+18%) | 0.229/0.227, 0.225/0.224, 0.122/0.126, 0.134/0.135, 0.125/0.127, 0.194/0.196, 0.209/0.218, 0.108/0.091 | yes |
| Point / x (B′, cf, secondary) | 3.1, 4.4, 1.5 | 3.06, 4.39, 1.55 | yes |
| Seed 49's bar margin = seed 42's | coincidence: net 344 of 49,152 rankings on both | 9,305 − 8,961 = 344 (seed 49) and 9,406 − 9,062 = 344 (seed 42); per-pair nets differ (171, 288, −115 vs 160, 238, −54); 0 of 12,288 anchors equal at the same index | yes |
| Selection paintings; pooled anchor paintings | 6,451; 5,195 | 6,451; 5,195 | yes |
| Brainstorm quotes (§2, §8): R1 fused − cf on e×s a, e×g a, e×s b, e×g b (seed 42) | +1.41, +2.69, +0.01, −0.25 | +1.404, +2.698, +0.012, −0.256 (115, 221, 1, −21 of 8,192 rankings) | last digit off (N5) |

## 3. Findings

### Blocking

None. The verdict, all eight pooled checks, every chosen cell and every per-anchor array were re-derived from my own
bundles and agree exactly; the verdict follows the rule's text (§6.5 strict lower bounds, §8 agreement, no boundary).

### Should-fix

**S1. The mechanism wording turns a visual-contrast rule into "the side a label-free method needs".**
`docs/reports/auto/v2/2026-11-21_round3_affect_gate.md:72-76` (Summary: "the reader's affect pick pays as a label-free
choice of the side to steer"), `:569-572` (§8: "The side decision is what a label-free method needs … AFF's
contribution is that it finds the side without labels"), `:447-449` ("AFF is the better reader of the two"),
`:766-767` and `:773` (§12 item 3 and "Our view": "one-sided affect steering whose label-free part is the choice of
side"), and the `reports_sum.md` row (line 98, "with the affect pick choosing the side"). The side AFF finds is the
right side to steer only where the other side's shared aspect is emotion:
- In style × genre the affect pick opens AFF's gate on 77.6% of condition a (the style side), as often as on the
  emotion side of emotion × style (78.4%), and AFF loses there: −0.580 [−0.793, −0.363] against B′(A0) and −0.319 and
  −0.222 against the two random-share draws.
- A label-free check of the report's own "not tested" reading supports a visual-contrast rule, not an affect detector
  (`fr3_side.py`, pooled 49 to 51; a review diagnostic, not one of §7's quantities). When both visual groupings have
  Δ < 0 (the contrasts agree more than the supports), the reader picks affect in 95.1%, 96.1% and 93.4% of condition a
  and 89.7%, 74.0% and 76.8% of condition b. Otherwise it picks affect in 57.0%, 60.7% and 50.2% (a) and 32.1%, 18.1%
  and 18.4% (b). The mean Δ of affect itself in condition a is +0.003, +0.002 and −0.001, against −0.021, −0.033 and
  −0.026 for image.
- So whether the chosen side pays depends on the benchmark's pair order: condition a is the emotion side in two of three
  pairs. The control shows that the per-condition open rate carries the gain, and "the reader" adds nothing measurable beyond it, so
  "the better reader" names the wrong part.

*Fix:* replace the §8 sentences with "AFF's label-free part is a visual-contrast rule: it steers when the visual
groupings agree more on the contrasts than on the supports. On this benchmark that is the emotion side of the two
emotion pairs, where steering pays, and the style side of style × genre, where it costs R@1 (−0.580 against
B′(A0)). The control shows that this per-condition asymmetry, not the choice of episodes within a condition, carries
the gain." Use the same wording in the Summary bullet, §12 item 3, "Our view" and the reports_sum row. At `:448` write
"AFF's gate is the better of the two". If the Δ shares above go into the report, label them as a final-review
diagnostic outside §7's list.

**S2. The outcome-table row for the control applies, and the report does not name it.** `:753-755` (§12, "Under the
rule's outcome table, what follows is the user's decision"). Rule §9 has a row "… or the random-share control matches
or beats AFF → Reported only; no second verdict; the user decides what follows". Draw 0's point (+0.633) is above
AFF's (+0.591), and both paired differences straddle 0 (−0.042 [−0.138, +0.048], +0.029 [−0.076, +0.132]), so the row
is triggered. The GO row on its own does not hand the next step to the user, so the report's sentence has no named
source. A reader of the paper plan needs to see that the rule flagged this case.
*Fix:* "Two rows of the outcome table apply: the GO row (all seven checks pass) and 'the random-share control matches
or beats AFF' (draw 0 +0.633 against AFF's +0.591; both paired differences straddle 0), which is reported only and
leaves what follows to the user." Add "(rule §9: reported only; the user decides what follows)" to the Summary's
control bullet (`:72-77`). Add the same line to the run log.

**S3. The +0.090 that sizes the secondary check is outside the rule's descriptive list and has not been accepted.**
`:65-67` (Summary), `:432-451` (§6.3), `:599` (Table 11), `:663-664` (§10), and `run_r3_test.py:345, 349`. Rule §6.9
asks for the frozen-cell line "with the same comparators as item 5" (the seven checks), and §7 item 8 says "nothing else
is computed on the test seeds". Frozen AFF minus frozen R1 is neither. Task 3's review flagged it (minor 4: "drop it, or
have the user accept it"), and the ledger deferred it without either. Its use is conservative, since it shrinks the
pre-registered +0.202, and the number is right (+0.090 [+0.005, +0.177]). But the Summary leans on it, and §10's
"Report diagnostics" bullet does not list it.
*Fix:* keep it, label it "outside rule §6.9 and §7 (descriptive; accepted by the user on <date>)" in §6.3 and §10, and
ask the user to accept it before the round is closed. If the user declines, drop it from the Summary and §6.3 and keep
it only in a limitations line.

### Nits

- **N1.** `:69` and `reports_sum.md:98` "where AFF also fell below B and its counterpart" (style × genre): the
  difference from the counterpart is −0.149 [−0.348, +0.051], so its interval includes 0. Say "below B and B′(A0)
  (intervals excluding 0) and, in its point, below its counterpart (−0.149 [−0.348, +0.051])".
- **N2.** `:39-42`, `:343-345` and Figure 2's title ("each point at least 1.5 times its detectable margin"): rule §6.1
  defines x only to read a failed check. The ratio of an observed point to x has no inferential meaning, and putting it
  beside the verdict reads like extra strength. Drop the ratios or label them "descriptive; x is for reading a failed
  check".
- **N3.** `:316` "built in one invocation of `run_baselines.py` (21:00 to 21:08)" and Table 1 `:199`: one invocation
  of `run_r3_build.py`, started at 20:58 (`run_r3_build.log`), ran `run_baselines.py` once per seed. Seed 49 finished
  at 21:01:48 (197 s), seed 50 at 21:04:56 and seed 51 at 21:08:14. Write "20:58 to 21:08".
- **N4.** `:735` "Smoke runs never printed a metric value": under the ledger's pre-flight ruling, the smoke builds of
  9001 to 9003 (`run_baselines.py --smoke`) wrote its scorer R@1 tables to `results/smoke/build_900{1,2,3}.log`. I
  counted 29, 31 and 28 lines with decimal numbers there without reading the values. Rule §10 says smoke runs never
  print *or log* a metric value. The values are baseline scorers on 64-episode smoke draws, with no AFF number and no
  test seed, so the impact is nil. The fix's red-test log `test_r3_bundle_fix1_red.log` also holds round 1's printed
  seed-42 values (so they were logged twice, not "once"). Write: "Smoke runs printed no metric value to the console.
  Under a pre-flight ruling, run_baselines' own smoke-build logs hold its scorer tables for the smoke draws, read only
  with grep. Round 1's known seed-42 values appear in the dry-check output and in the red-test log of its fix
  (7ece674)."
- **N5.** `:166` "+1.41 on emotion × style a and +2.69 on emotion × genre a" and `:560` "+0.01 and −0.25": these come
  from the brainstorm (`2026-11-20_r1_levers_brainstorm.md:78-79`, which has the same slip). From round-1 R-c's arrays
  they are 115, 221, 1 and −21 of 8,192 rankings: +1.404, +2.698, +0.012 and −0.256. The brainstorm's s×g b −1.02 is
  −1.013. Write +1.40, +2.70, +0.01 and −0.26 (`out/fr3_cites.json`).
- **N6.** Order of work: rule §8 (steps 10 and 11), the plan's Task 6 and the spec all put the final review, its fix
  wave and the scoped re-review before the report. The report was committed first (02215b8, placeholder §11.6), and
  the ledger has no ruling for it. No harm, because this review covers the report. Add a ruling line to the ledger
  and the log.
- **N7.** Run log rows 20:05 and 20:50 give the cosine x as 0.325. `sensitivity.json` has 0.32446 (0.324, as the
  report says).
- **N8.** Figure 4's title "R1 trailed both" and `:548` "Both draws were above R1's +0.389" compare points only. No
  paired control − R1 interval exists. Write "R1's point lay below both".
- **N9.** Table 7 `:413-424` leaves R1's seed-42 cosine, RCA and gain-vs-RCA cells blank. From the stored seed-42
  arrays (equal to round-1 R-c's, §2.2) they are +5.957 [+5.589, +6.330], +5.538 [+5.162, +5.907] and +2.563
  [+2.173, +2.954].
- **N10.** Phase 2 was authorised at 21:08:40 (`rederive/out/PHASE2_AUTHORISED`), while the GO pass was still running
  (`go_pooled.json` 21:09:59). Rule §8 places phase 2 "after the GO pass". Its first step (the hash check) ran at
  21:11 and its bundles at 21:11 to 21:13, so it had no effect. One line in the log would record it.
- **N11.** For reuse, not this round: the GO pass has no pre-verdict guard of its own against a wiring swap of the
  gates or clusters (M09, M10). Add an assertion in `go_phase` that `cl == groups[anchor]` and that AFF's family sees
  AFF's τ_0 open counts. Add the §5-order test (M20) and a criterion test for ρ_ctrl (M06).

## 4. Mutation checks

Each mutation changed one guard in a fresh scratch copy of the round-3 folder (`final_review/mut/w/`, with r3_common's
ROOT pinned to the repository so that the copy's `results/` stayed inside the copy; round 2's `r2_fusion.py` was
copied in for M05 to M07). The named test files ran with `pytest -x --basetemp final_review/mut/tmp/…`. Unmutated, the
copy passes all four suites: 22 + 32 + 21 tests plus the wiring test (76). The committed files were never touched
(`git status` of the folder shows only `final_review/` untracked and the controller's uncommitted log row). Record:
`out/fr3_mutate.json`, logs in `mut/`.

| Id | Guard / mutation | Tests run | Result |
|---|---|---|---|
| M01 | AFF gate: the affect factor dropped (AFF becomes R1) | fusion, wiring | **caught** by `test_aff_gate_is_r1_gate_times_affect_pick` (wiring passes) |
| M02 | AFF gate opens on image picks | fusion | **caught** |
| M03 | Counterpart built from all-open gates, not the family's own | fusion, wiring | **caught** by `test_counterpart_from_aff_gates_is_condition_free_and_differs_from_r1s` (wiring passes) |
| M04 | Counterpart = the condition-dependent gated term | fusion | **caught** (condition-free assertion) |
| M05 | Integer cross-fit: fused ties to the highest cell (round 2's `select_fused`, copy) | fusion | **caught** |
| M06 | Integer cross-fit: fused criterion min(ρ, γ), ρ_ctrl ignored (copy of round 2) | fusion | **survives** all 22 tests. At runtime the seed-42 regression would stop it: my code gives cells 7/63 (R1) and 7/7 (AFF) under this criterion, not 116/119 and 39/119 |
| M07 | Integer cross-fit: counterpart ties to the highest cell (copy) | fusion | **caught** |
| M08 | Pooled clustering: seed-specific clusters in `pooled_check` | fusion | **caught** by `test_pooled_check_shared_painting_is_one_cluster` |
| M09 | Pooled clustering: the GO pass stores per-seed local painting ids (`np.unique(cl, return_inverse=True)[1]`) | runners, wiring | **survives** both. Only the phase-2 re-derivation (which compares `cl`) or this review would catch it |
| M10 | GO pass: AFF's family run on R1's gates | runners, wiring | **caught** only by the wiring test, through the descriptive pass's cache-reproduces-GO check (the M27 guard), which runs after the verdict. Before the verdict only phase 2 catches it |
| M11 | §6.4: GO pass cross-fits R1's counterpart | runners, wiring | **caught** by the wiring test (runtime assertion `run_r3_test.py:116`); no unit test |
| M12 | §6.4: `fused_only` still selects and reports the counterpart | fusion | **caught** |
| M13 | Strict inequality: `pooled_check` pass on lower bound ≥ 0 | fusion | **caught** by `test_pooled_check_pass_is_strict` |
| M14 | Strict inequality: `read_check` pass on lower bound ≥ 0 | runners | **caught** by `test_go_iff_all_seven_lower_bounds_above_zero` |
| M15 | Secondary check changes `go` in `go_checks` | fusion | **caught** |
| M15b | Secondary check changes the verdict in `decide` | runners | **caught** by `test_secondary_never_changes_go` |
| M16 | Hash check skips the earlier seeds | runners | **caught** |
| M17 | Hash check skips the other new seeds | runners | **caught** |
| M18 | Hash check allows two pairs of one seed to share a SHA-256 | runners | **caught** |
| M19 | `verify_build_record` no longer compares file SHA-256s | runners | **caught** |
| M20 | §5 order: `require(2)` deleted before item 3 | runners, bundle | **survives** both (as task 3's minor 2 said). For this round the order is verified from `regression_check.json` (items in order, all pass) |
| M21 | Descriptive pass accepts a verdict written under another rule | runners | **caught** |
| M22 | Random-share control draws condition b first | fusion | **caught** |
| M23 | Counterpart gain-zero assertion removed (`go_checks`) | fusion | **caught** |
| M24 | Boundary epsilon set to 0 | runners | **caught** |
| M25 | Rule applied without the phase-2 agreement record | runners | **caught** |
| M26 | Sensitivity formula with 9 Σ m² + 3n | fusion | **caught** |
| M27 | Descriptive pass's cache-reproduces-GO check removed | runners, wiring | **survives** (alone it changes no number; it is what catches M10) |
| M28 | Pick ties go to the last grouping | fusion | **survives**; no effect here: no exact arg-max tie exists on seed 42 or on any test seed |
| M29 | Gate opens on m > τ instead of m ≥ τ | fusion | **caught** |
| M30 | Never-overwrite guard (`refuse_existing`) disabled | runners, fusion | **caught** |
| M31 | GO pass no longer calls the episode-hash cross check of the build records | runners, wiring | **survives**; the same check ran at build time (`build_seed{s}.json` hash_check pass) and was re-done by phase 2 and by this review |

Verdict on the tests: every guard the brief named binds to a test or to a runtime check. That covers the AFF gate
(M01, M02), the counterpart from AFF's own gates (M03, M04), the integer tie rules (M05, M07), the pooled clustering
inside the statistics (M08), §6.4 (M11, M12), the strict inequality (M13, M14) and the hash check (M16 to M19). The
survivors are in the runners' wiring and ordering (M09, M20, M27, M31), plus two arithmetic choices that unit tests miss
but that the seed-42 regression stops (M06) or that cannot bite on these data (M28). For this round each survivor is
covered by the run's own record, the phase-2 re-derivation or this review. Before the code is reused, add a GO-pass
assertion that `cl == groups[anchor]`, and add the §5-order test (task 3's minor 2) (N11).

## 5. Process against the rule

| Check | Evidence (Amsterdam time; git times are +02:00, the same) | Result |
|---|---|---|
| Rule committed before any code | fab5ae1 19:51:00; first code 83b51f4 19:52:48 | ok |
| §5 order in the seed-42 run | `run_r3_seed42.run`: `require(1)`, `require(2)`, then item 3; `regression_check.json` (20:52:03) holds 66, 40, 24 and 4 rows in item order, all pass | ok (guard untested, M20) |
| No AFF number before items 1 and 2 | dry runs (results/smoke, 20:17 and 20:38) ran the same order and printed PASS/FAIL only; phase 1 of the re-derivation computed AFF's seed-42 numbers at 20:05, which rule §8 allows once the rule is committed (the numbers are the rule's own) | ok |
| Sensitivity before any build (§6.1) | `sensitivity.json` 20:52:04; build started 20:58 | ok |
| Wiring smoke after §6.1, smoke seeds only, mutation fired | wiring logs 20:52:30 to 20:54:03, seeds 9001 to 9003; `wiring_mutant_go.log` fired the condition-free assertion; wiring logs have no decimal number (also none with `\d\.\d`) | ok; smoke-build logs hold scorer tables by ruling (N4) |
| Seed-42 real run during Task 3's review | launched 20:50 at d38f1f9 (ledger ruling); review approved 20:58 with no change; `git diff d38f1f9 02215b8 -- '*.py'` of the folder is empty; `test_verdict.json` script SHA-256 = committed `r3_apply_rule.py` | ok |
| Build (§6.2) | `run_r3_build.py` passes `--episodes-seed` only (no `--overwrite`); one invocation 20:58 to 21:08, exit 0, one attempt per seed; `codes_provenance.json` 8e6a517b… before and after each build and now (rewritten at 21:05:04, same bytes); the 24 per-pair SHA-256s of seeds 42, 43, 45, 47, 48, 49, 50, 51 are all distinct (recomputed); build records compare each new seed with the earlier five and with the new seeds built before it | ok |
| Ledger rows | 49 to 51 test (spent), 52 and later free, 9001 to 9003 smoke | ok |
| §6.4: only GO quantities before the verdict | `test_verdict.json` 21:16:19. Before it: build records (21:01 to 21:08), caches and `go_seed{s}.npz` (21:09), `go_pooled.json` 21:09:59, phase 2 (21:11 to 21:14; `phase2.json` holds cells, σ*, equality flags and the pooled checks only), the phase-1 rerun at 21:15:17 (seed 42 only, explained in the phase-2 report). After it: `descriptive.json` 21:17:38, figures 21:32 | ok |
| R1's counterpart not cross-fitted before the verdict | `run_family(..., fused_only=True)` and the runtime assertion at `run_r3_test.py:116`; phase 2 skipped it | ok (see M11, M12) |
| Re-derivation agreed before the rule was applied (§8) | `phase2_agreement.json` 21:14:42, `all_agree` true, and its go_pooled SHA-256 equals the current file (closes T3-10 for this round); rule applied 21:16:19 | ok |
| Boundary (§8) | no lower bound within 1e-12 of 0; no `verdict_boundary.json`; `--boundary-reported` not used | ok |
| D15 inputs | all 36 SHA-256s of `r3_common.INPUTS` and the rule's table re-hashed now: unchanged | ok |
| Report after the final review (§8) | report first (02215b8) | N6 |
| Commit hygiene | 13 commits, each with both attribution lines; each touches only this round's files, the spec, plan, ledger, report, figures and `reports_sum.md`; `main` is 77 ahead of `origin/main`, nothing pushed; `check_reports_sum.py`: OK | ok |
| Held rows, GPU | none read (episodes via `EvalContext` selection rows only); CPU only | ok |

**Disclosures in the report.** Selection on seed 42 (header, §4, §10, Summary), the style × genre loss (Summary, §7, §10,
§12), R1 also passing (Summary, §6.2, §10, §12), the control matching AFF (Summary, §8), the either cost (Summary, §10)
and the prior (Summary, §10) are all present. The claim stays within rule §6.10 (`:271-277` repeats it). The gaps are
the triggered §9 row (S2), the status of the +0.090 (S3) and the mechanism wording (S1).

## 6. Triage of the deferred minor findings

Must be resolved before the round is closed: **T3-4** (as S3: label the frozen-cell AFF − R1 and get the user's
acceptance, or drop it from the Summary). None of the others changes a number or the verdict.

| Finding | Triage |
|---|---|
| T2-M4 bar-comparator test repeats the code under test | Closed for this round: the comparator is B′(A0) on every seed and pooled, re-derived independently. Add a test where per-seed and pooled comparators differ before the code is reused |
| T2-M6 code written before tests | Process only; this review's mutation checks (§4) stand in. No fix |
| T2-M7 naming `F` vs `bundle.F` | Cosmetic. No fix |
| T1-1 untriggered bundle guards (D2 head identity, `compare_with_round1` mismatch path, row, parity, finiteness checks) | Closed for this round: my own builds confirmed the D2 identity on every seed and reproduced every bundle piece from round 1's stored seed-42 arrays and from the test-seed caches. Add tests before reuse |
| T1-2 inputs hashed once per process | Closed: ledger ruling; all 36 inputs re-hashed now are unchanged |
| T1-3, T1-4, T1-5 (test strength, warnings, open npz handle) | Cosmetic. No fix |
| T3-1 `--boundary-reported` not bound to the boundary file | Moot: no boundary, flag not used. Bind it before reuse |
| T3-2 guards without a test (§5 order, `load_pooled` checks, `decide`'s go flag, go_pooled-vs-verdict, episode SHA in the GO phase, cache reproduction, R1-no-counterpart) | Confirmed by mutation (§4). For this round each is covered by the run's own record or by the phase-2 re-derivation and this review. Add tests before reuse; the §5-order test first |
| T3-3 §5 item 4 disclosure not in `regression_check.json` | Closed: the report carries it (`:297-299`) |
| T3-4 frozen-cell line computes an extra AFF − R1 quantity | **Must fix: S3** |
| T3-5 wiring leak regex misses one-decimal numbers | Closed for this round: no wiring log has a match of `\d\.\d`. Widen the regex before reuse |
| T3-6 dry outputs left in `results/smoke/` | Closed: deleted at 20:58; only logs remain |
| T3-7, T3-8, T3-9 (temp-file cleanup, truncated partial file, operational gaps after a crash) | No crash happened. No fix |
| T3-10 agreement record not bound to a GO pass | Closed for this round: its recorded go_pooled SHA-256 equals the current file's. Bind it in code before reuse |

## 7. Files and storage

Scripts in `final_review/`: `fr3_lib.py`, `fr3_build.py`, `fr3_seed42.py`, `fr3_run.py`, `fr3_seed49_equality.py`,
`fr3_side.py`, `fr3_cites.py`, `fr3_mutate.py` and this file. Outputs in `final_review/out/` (gitignored, about
61 MB in all; the largest files are my four bundles, about 15 MB each): `fr3_seed{42,49,50,51}.npz`,
`fr3_perepisode.npz`, `fr3_run.json` and `fr3_run_comparisons.json` (691 comparisons), `fr3_seed42.json` and
`fr3_seed42_comparisons.json` (171), `fr3_seed49_equality.json`, `fr3_side.json`, `fr3_cites.json`,
`fr3_mutate.json`, and the logs. Mutation logs are in `final_review/mut/*.log`; the scratch copy and pytest temp
folders were deleted. Nothing over 1 GB is left behind. No committed file was changed and nothing was committed.
