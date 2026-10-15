# Final whole-branch review: reader-fix round 2 (afef374..e50da1e)

Written 2026-10-06 17:48 (Amsterdam) by the whole-branch final reviewer (fresh context). Binding rule
`../DECISION_RULE.md` (SHA-256 368bec11…265c, committed dc9fac6). Scope: the 8 commits e78f380, dc9fac6, a221b64,
cdf9b4d, ee48015, e0981ce, 9fc1c6d, e50da1e. Everything here was computed on CPU with this review's own scripts in this
folder (`fr_*.py`); `results/` was only read, no committed file was changed except the temporary mutations of §4, which
were reverted (git status of the round's tracked files is clean).

## Verdict: CONFIRMED WITH FIXES

The decision stands: **no candidate clears the development bar, no test is built, and the A1 ablation runs on R1.**
I rebuilt every candidate from scratch (R1, R2 and R3 on A0, R1 on A1), including R3's replacement draws, impure banks,
D(k), k* and the retrained half-readers, with my own code for everything except the bundle loader, the D6 z-score and
the bootstrap. All 371 automated comparisons with the stored results agree exactly (bit for bit for arrays, `==` for
floats), and every report diagnostic I re-derived (about 90 values) agrees at the precision shown. About 300 numbers
of the report were checked against these sources; none is wrong.

The fixes are report text only (3 should-fix, 9 nits); no number, verdict or code changes. No blocking finding.

## 1. What was re-derived and how

| Script | What it does (own code unless stated) |
|---|---|
| `fr_prep.py` | Loads the seed-42 bundle with round 1's `common.load_bundle` (the only shared loader; it re-runs the step-1 regression check). Own: the 18 (A0) and 24 (A1) reader features from the posteriors and episodes, the grouping scores s_h. Cache in `out/fr_cache.npz` |
| `fr_core.py` | Own fusion family from the rule text: T, picks, margins, τ, gates, G_cf, `_combine`'s float32 arithmetic, hit counting **without building the restricted score** (j is first under the restriction iff j ∈ K and S(j) beats every other member of K), integer ρ/γ, control σ*, min-margin and max-ρ picks, assembly with an explicit S′ for `swap`, own per-episode r1/gain/other/swap/strict, comparator (D10), clauses (D12). Imports only `src.eval.aspect_nested._zdict` (D6) and `src.eval.aspect_metrics.cluster_bootstrap` |
| `fr_cands.py` | R1 and R1/A1 from round 1's pickles on my features; R2 with my μ42, σ42, EM and adaptation; R3 from the stored pickle and from my own retrained readers. Then T → τ → 896 cells → cross-fits → assembly → bootstrap; compares 342 quantities with the stored files, plus the regression check of §4.7 |
| `fr_r3.py` | R3 from scratch: draws per §4.4 b, nested impure banks, features from the cross-fitted posteriors, purity-4 checks, SMD, D(k), k*, five-fold CV over episodes and refit at k* (own loop, sklearn LR), seed-42 probabilities; 29 comparisons |
| `fr_diag.py` | The report's own diagnostics (§4.3 to §5.4) from my per-cell matrices |
| `fr_mutate.py` | The mutation checks of §4 |

Two bugs in my own first pass were found by the comparison and fixed before the numbers below (a numpy bool `+` that
acts as OR in my hit counter; and the EM/adaptation evaluated as P·(π̂/π_train) instead of the rule's left-to-right
P·π̂/π_train, which moved R2's τ_0 by 1.5e-12 relative with identical gates, picks and cells; the implementation
follows the rule's text; see nit N9).

## 2. Re-derived numbers

"Agree" means identical at full precision unless stated. Arrays: bit-identical.

### 2.1 Decision numbers (rule §5 items 3 to 5, D10 to D12)

| Quantity | Reported (stored) | Re-derived | Agree |
|---|---|---|---|
| R1/A0 bar margin [95%] | 0.4720052083333333 [0.24030286496949704, 0.7028163072942406] | same | yes |
| R1/A0 comparator; fused / counterpart / B′ / B R@1 | counterpart; 18.918863932291664 / 18.446858723958336 / 18.437 / 18.341 | same | yes |
| R1/A0 gain statistic [95%] | 2.667236328125 [2.325087836946873, 3.012361650695922] | same | yes |
| R1/A0 D12 clauses 1/2/3 | no / yes / yes | no / yes / yes | yes |
| R2/A0 bar margin [95%] | 0.115966796875 [−0.07289118013657525, 0.30106240852848065] | same | yes |
| R2/A0 comparator; fused / counterpart | B′; 18.552652994791664 / 18.290201822916664 | same | yes |
| R2/A0 gain statistic [95%] | 0.9175618489583334 [0.6617526421912794, 1.1691833632202837] | same | yes |
| R2/A0 clauses | no / no / yes | no / no / yes | yes |
| R3/A0 bar margin [95%] (stored pickle and my retrained readers) | 0.07731119791666666 [−0.11125362620492793, 0.2709495572650762] | same, both routes | yes |
| R3/A0 comparator; fused / counterpart | B′; 18.513997395833336 / 18.426513671875 | same | yes |
| R3/A0 gain statistic [95%] | 0.9541829427083334 [0.7021970530564084, 1.1988815133025827] | same | yes |
| R3/A0 clauses | no / no / yes | no / no / yes | yes |
| Verdict (item 5) | no candidate clears; carried none; ablation subject R1 | same (E empty; largest bar margin R1) | yes |
| Margins (fused − counterpart) R1 / R2 / R3 | +0.4720 / +0.2625 / +0.0875 | same, with intervals | yes |
| Either (fused − cf) R1 / R2 / R3 | −1.723 [−2.048, −1.389] / −0.393 [−0.640, −0.140] / −0.779 [−1.076, −0.494] | same | yes |
| Fused − B R1 / R2 / R3 | +0.578 [0.358, 0.811] / +0.212 [0.029, 0.394] / +0.173 [−0.026, 0.381] | same | yes |
| Per-pair bar margin and gain (9 + 9) | Table 4 | same | yes |
| Bar margin − step-1 arg-max reader's, R1 / R2 / R3 | +0.159 [−0.118, 0.438] / −0.197 [−0.387, −0.008] / −0.236 [−0.436, −0.026] | same | yes |
| Per-anchor arrays fused/cf × r1/gain/other/swap/strict, bar_v (11 per candidate, 5 candidates) | stored npz | bit-identical (two routes: per-cell matrices and assembled S′) | yes |

### 2.2 Readers, thresholds and cells

| Quantity | Reported | Re-derived | Agree |
|---|---|---|---|
| R1 probabilities, T (4), picks, margins | stored | bit-identical | yes |
| τ_0..3 R1 | 3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211 | same | yes |
| τ_0..3 R2 | 3.771746264230602e-05, 0.2907081713292494, 0.5994124414690893, 0.861841000813007 | same | yes |
| τ_0..3 R3 | 3.399954748600775e-05, 0.12456908995572917, 0.2854599358396749, 0.4974364340057035 | same | yes |
| σ*, ρ_ctrl per half (all candidates) | 0, 0; 4,532 and 4,483 | same | yes |
| Fused cells (h0, h1) R1 / R2 / R3 / R1-A1 | 116, 119 / 67, 615 / 55, 574 / 12, 45 | same | yes |
| Counterpart cells R1 / R2 / R3 / R1-A1 | 58, 571 / 278, 155 / 157, 156 / 11, 11 | same | yes |
| Integer criteria at the picks (ρ, γ fused; ρ cf) | stored `integer_criteria` | same | yes |
| Exact ties at the max (phase-2 report) | R1 fused h0 4; R1 cf h0 4, h1 2; R2 fused h0 2; R2 cf h1 2; R3 fused h1 2; R3 cf h0 4, h1 2 | same cell sets (e.g. R1 fused h0 116, 125, 133, 142) | yes |
| R2 μ42 (18), σ42 (18), no zero σ | stored | bit-identical | yes |
| R2 code check (own scaler, no EM) | exact, picks identical | max diff ≤ 1e-12, picks identical | yes |
| R2 π̂; EM updates; cap | (0.33954936858584167, 0.31214248949301643, 0.348308141921142); 39; not reached | same | yes |
| R2 top probability R1 / before EM / after | 0.694 / 0.756 / 0.757 | 0.69426 / 0.75590 / 0.75694 | yes |
| R2 picks differing from R1, before / after EM | 21.2% / 21.6% | 21.18% / 21.60% | yes |
| R2 shift report: S, C offsets (affect / image / caption) | −0.20 / −0.36 to −0.37 / −0.48 to −0.49 | −0.195 to −0.196 / −0.363 to −0.367 / −0.482 to −0.489 | yes |
| R2 σ42/scale (affect / image / caption); match-share offsets | 0.93–1.13 / 0.62–0.74 / 0.48–0.62; −0.27 to −0.38 | 0.934–1.134 / 0.624–0.742 / 0.479–0.616; −0.273 to −0.384 | yes |
| R3 SHA-256 of order, img, cap (both halves) | r3_k_A0.json | same (draws written from the rule text) | yes |
| R3 SHA-256 of purity 1–4 feature matrices (8) | r3_k_A0.json | same | yes |
| R3 purity-4 features = round 1's half{0,1}__X; labels = half{0,1}__y; SMDs = rb_diag_A0 | exact | exact | yes |
| R3 D(1..4) | 0.10945540348228223, 0.08964658887594207, 0.18506833918606608, 0.27487353947498383 | same; all 72 SMDs bit-identical | yes |
| R3 k* | 2 | 2 | yes |
| R3 mean abs Δ seed 42 / bank k=1..4 (affect, image, caption) | 0.0081, 0.0332, 0.0212 / … | same (affect 0.0074 to 0.0080; image k=2 0.0331; caption k=1 0.0208) | yes |
| R3 chosen C (half 0, half 1); CV log-loss gap of best two | 100, 10; 3.8e-6, 3.1e-6 | 100, 10; 3.777e-6, 3.135e-6; all 10 CV losses identical | yes |
| R3 OOF accuracy at chosen C | 62.0%, 61.5% | 62.02%, 61.54% | yes |
| R3 probabilities on seed 42 (my retrained readers) | probs_R3_A0.npz | bit-identical | yes |
| R3 top probability seed 42 / bank OOF; half-reader pick agreement a / b | 0.600 / 0.600; 98.5% / 98.1% | 0.5997 / 0.6003; 98.49% / 98.12% | yes |
| Pick accuracy R1 / R2 / R3 / R1-A1 [95%] | 51.3 [50.7, 51.8] / 47.2 [46.7, 47.8] / 55.7 [55.1, 56.3] / 48.7 | same; per pair × condition (6 each) same | yes |
| Gate open share at τ_0..3 (overall, a, b) | stored (float32 means) | same within 2e-6 pp (stored are float32 means) | yes |

### 2.3 Regression check (§4.7) and round-1 identities

| Quantity | Reported | Re-derived | Agree |
|---|---|---|---|
| R1 on cells 0–223: T, margins, picks = round-1 R-c's | exact | exact | yes |
| τ = rc_tau.json | exact | exact | yes |
| Cells fused 116/119, cf 58/123, σ* 0/0 | as stated | same | yes |
| 10 per-anchor arrays and bar_v = round-1 R-c's | exact | bit-identical | yes |
| Bar margin; gain statistic = rule literals | 0.4435221354166667 [0.21646171563312194, 0.6735669710776852]; 2.667236328125 [2.325087836946873, 3.012361650695922] | same | yes |
| R1 (896) fused arrays = round-1 R-c's on every episode (r1 and gain) | "paired difference 0.000" | identical on all 12,288 episodes | yes |
| R1 − round-1 R-c: fused / margin / bar | +0.000 [0, 0] / +0.028 [−0.002, +0.059] / +0.028 [−0.002, +0.059] | same | yes |
| R2 − R1: fused / margin / bar | −0.366 [−0.581, −0.149] / −0.210 [−0.450, +0.032] / −0.356 [−0.612, −0.106] | same | yes |
| R3 − R1: fused / margin / bar | −0.405 [−0.624, −0.186] / −0.385 [−0.633, −0.123] / −0.395 [−0.641, −0.146] | same | yes |
| R1 cf tune-1 move: ρ_tune 571 vs 123; other-half R@1 | 4,509 vs 4,502; 18.656 vs 18.713 | same | yes |
| Ranking counts: R1 bar 232; +0.5 needs 246; Rc 218 | 232 / 246 / 14 short; 28 short | 232.0 / 218.0 | yes |

### 2.4 A1 ablation

| Quantity | Reported | Re-derived | Agree |
|---|---|---|---|
| R1/A1 fused / cf / B′(A1) | 18.970 / 18.980 / 18.805 | 18.9697265625 / 18.979899088541664 / 18.805 | yes |
| R1/A1 bar margin [95%] (comparator) | −0.010 [−0.169, +0.156] (counterpart) | −0.010172526041666668 [−0.16932314051371594, 0.1558926398406146] | yes |
| R1/A1 gain statistic | +0.535 [+0.336, +0.729] | same | yes |
| A1 − A0 fused; bar margin | +0.051 [−0.236, +0.336]; −0.482 [−0.747, −0.214] | same | yes |
| R1/A1 fused arrays = round 1's R-b expected A1 (r1, gain, other) | identical | identical | yes |
| R1/A1 cf − round 1's R-b expected A1 cf | +0.108 [+0.024, +0.192] | same | yes |
| R1/A1 cf − R1/A0 cf (means) | 0.533 | 0.5330 | yes |

### 2.5 The report's own diagnostics (§5)

| Quantity | Reported | Re-derived | Agree |
|---|---|---|---|
| Table 6: R1 cf h1 571 vs 123 (tune / other) | 4,509 / 18.656 vs 4,502 / 18.713 | same | yes |
| Table 6: R2 fused h1 615 vs best k13 167 | 86 / 18.604 vs 75 / 18.717 | same (best k13 by criterion = 167) | yes |
| Table 6: R2 cf h0 278 vs best k13 36 | 4,602 / 18.119 vs 4,600 / 18.180 | same | yes |
| Table 6: R3 fused h1 574 vs best k13 115 | 96 / 18.632 vs 86 / 18.937 | same | yes |
| Top-k effect on bar margin R1 / R2 / R3 | +0.028 / −0.057 [−0.121, +0.006] / −0.153 [−0.240, −0.062] | same | yes |
| k13-only bar margins R2 / R3 (comparator B′) | +0.173 [−0.022, +0.369] / +0.230 [+0.047, +0.413] | same; k13-only cells R2 67/167, 36/155; R3 55/115, 157/156 | yes |
| R1 rank analysis: first places outside B's top 3 | 11.1% (5,472); 12.3% hits; 18.9% all | 11.13% (5,472); 12.34%; 18.92% | yes |
| Restriction flips at k 3 / 5 / 2 | +614 −675 = −61 (−0.124) / −43 / −139 | same (311/354, 874/1,013); all 614 inside the 5,472 | yes |
| Targets outside B's top 3 | 59.9% | 59.89% | yes |
| Table 7 (each reader at R-c's cells): fused / cf / margin / bar / gain / either / cost per gain | R1 18.919/18.475/+0.444/+0.444/2.667/−1.780/0.67; R2 18.429/18.431/−0.002/−0.008 (B′)/2.222/−2.226/1.00; R3 18.589/18.406/+0.183/+0.153 (B′)/1.404/−1.038/0.74 | same (incl. intervals [0.216, 0.674], [−0.256, 0.232], [−0.066, 0.375]) | yes |
| Curve τ_2, λ_u 0, k13: peaks | R1 19.116 (λ_a 4); R2 18.628 (1); R3 18.764 (1); gains R1 to 3.41, R3 to 1.71 | same | yes |
| In-sample best cell of 896 | 19.116, 18.703, 18.766 | same; cells 117, 67, 167 (all k_top 13) | yes |
| Total variation P^a vs P^b R1 / R2 / R3; on emotion pairs R1 / R2 | 0.591 / 0.732 / 0.506; 0.589 / 0.734 | 0.5910 / 0.7317 / 0.5058; 0.5892 / 0.7338 | yes |
| Told contrast, emotion pairs R1 / R2 / R3 | 0.361 / 0.376 / 0.315 | 0.3613 / 0.3763 / 0.3151 | yes |
| Row correlation z(T^a), z(T^b) (i2t, t2i) R1 / R3 | 0.628, 0.687 / 0.753, 0.799 | 0.6278, 0.6872 / 0.7527, 0.7989 | yes |
| Pick accuracy gate open / closed at τ_2 R1 / R3 | 60.6 / 41.9; 66.2 / 45.2 | 60.63 / 41.89; 66.18 / 45.21 | yes |
| R2 pick shares: caption all / a / b; affect in a | 19.6→33.9 / 13.2→27.6 / 26.0→40.3; 80.9→61.4 | 19.58→33.94 / 13.15→27.62 / 26.02→40.27; 80.90→61.37 | yes |
| R3 per pair × condition accuracy vs R1 | e×s b 36.8→52.6, e×g b 50.8→65.1, s×g b 45.8→59.6; e×s a 79.8→65.9, e×g a 85.6→75.1 | same | yes |
| "R3 picks best of any A0 reader so far" | 55.7 vs 54.7 (step-1), 51.3 | round-1 A0 readers: R-a 47.1, R-b 51.3, R-c 51.3, step-1 54.7 | yes |

## 3. Findings

### Blocking

None. The verdict, every decision number and every chosen cell were re-derived independently and agree exactly.

### Should-fix

**S1. The R2/R3 − R1 comparison lacks the selection caveat, and one §8 "fact" generalises from one reader.**
`docs/reports/auto/v2/2026-11-19_reader_fix_round2.md:311-312` (§4.3, "Both reader fixes lowered…"), `:588-590` (§8,
"*The picks fix made things worse* … so better arg-max picks are not the lever for the weighted term") and the
Summary at `:40-41`. R1's fused reader is round-1 R-c, the winner of round 1's seven candidates on this same seed 42;
round 1 itself put the best-of-seven inflation at about 0.1 to 0.15 R@1 (rule D12). R2 and R3 were each run once and
not selected on seed 42. The paired intervals cover sampling noise only, so the −0.356 and −0.395 bar-margin
differences are biased against the fixes by about that inflation. The sign is safe: the fused-R@1 differences
(−0.366, −0.405) exceed it, and the in-sample best of all 896 cells gives the same order with equal freedom (19.116,
18.703, 18.766). But the size is overstated as written. Separately, "better arg-max picks are not the lever" rests on a
single retrained reader (R3) whose higher pick accuracy came with flatter probabilities, on one seed.
*Fix:* in §4.3 and §8 add one sentence: "R1 is round 1's seed-42 winner, so part of its lead (round 1 estimated 0.1 to
0.15 R@1 for its best-of-seven choice) may be selection inflation; the sign holds in the in-sample comparison at equal
freedom." Reword the last clause to "R3 raised pick accuracy to 55.7% and still lost, so higher arg-max accuracy alone
did not raise the bar margin."

**S2. Smoke runs computed 896-cell numbers before the regression check passed (and an A1 number before the rule was
applied), and the report does not say so.** `results/smoke/cand_R1_A0.json` (written 16:19, HEAD a221b64) and
`results/smoke/cand_R1_A1.json` (16:21) hold full 896-cell cross-fits on 600 seed-42 episodes (fused cells 284/172 and
counterpart 484/146 on A0, i.e. cells 224 to 895; bar margins −0.083 and −0.292), and `task-1-report.md` printed them.
The regression check passed at 16:39:32; the rule was applied at 16:57:19. Rule §4.7 says "no number from cells 224 to
895 is written or printed before the check passes"; §9 allows smoke runs as "not results", so the rule is ambiguous,
and the impact is nil: the fusion code did not change after the smoke runs (`r2_fusion.py` and `run_r2_fusion.py`
last modified 16:17 and 16:18; the SHA-256s recorded in every `cand_*.json` equal the committed files at cdf9b4d,
ee48015 and HEAD), the smoke sample is 4.9% of seed 42 and its numbers do not resemble the full ones, and the verdict
was re-derived twice. But the report states at `:219` (§3.5) "Before any round-2 candidate number existed, R1 restricted
to cells 0 to 223 had to reproduce round-1 R-c", and the order-of-work disclosure (`:556-561`) omits it.
*Fix:* add to §7 "Order of work": "During implementation, smoke runs on 600 seed-42 episodes (results/smoke/, not
results under rule §9) computed R1's 896-cell cross-fit on A0 (16:19) and A1 (16:21), before the regression check and
before the rule was applied; their numbers appeared in the task report. The fusion code did not change afterwards."
Change §3.5's opening to "Before any full-data round-2 candidate number existed". Add the same line to the run log.

**S3. The report quotes the run log as it was before its corrections in the same commit.** e50da1e corrected the log
(16:15 → 16:13 for the rule commit, 27/28 → 29/30 mutations, HEAD cdf9b4d → ee48015 for the regression check), but the
report still says:
- `:129` and `:135` "the rule was committed at 16:15 (dc9fac6)" / "16:15 | rule committed": git time of dc9fac6 is
  16:13:07; 16:15 is when both streams were dispatched (progress.md). *Fix:* "16:13 rule committed; 16:15 fusion and
  reader streams dispatched".
- `:523` "the run log counts this as 27 of 28": the log says 29 of 30 (M1 to M28 plus M6b and M6c); 27/28 is
  progress.md's count. *Fix:* "29 of 30 (progress.md's ledger says 27 of 28)".
- `:562-564` "The log records the regression check at HEAD cdf9b4d, while regression_check.json records ee48015":
  the corrected log records ee48015; cdf9b4d appears in progress.md ("regression exact at cdf9b4d") and in the
  re-derivation's phase-2 report. *Fix:* name those two sources. (The substance holds: ee48015 added only reader-stream
  files; I confirmed r2_fusion.py and run_r2_fusion.py are byte-identical at cdf9b4d, ee48015 and HEAD and equal the
  SHA-256s recorded in each cand json.)

### Nits

- **N1.** `:365` "(0.008 to 0.045 R@1)": for the two fused picks the tune advantage (11 and 10) is in units of the
  criterion min(ρ − ρ_ctrl, γ), not R@1. Say "rankings of the tune-half criterion".
- **N2.** `:285` "R1's per-pair bar margins differ from round-1 R-c's only on the two emotion pairs": on style × genre
  the point is equal but 14 episodes differ (interval −0.972 vs −0.966 in Table 4). Say "point estimates".
- **N3.** `:413-414` "R2's counterpart fell below B (18.290), so B′ set the bar": B′ (18.437) exceeds B regardless; B′
  is the comparator because the counterpart fell below B′.
- **N4.** Figure 1 (`what_changed.png`) colours the matched counterpart orange ("replaced"), but G_cf is round-1 R-c's
  counterpart definition. New in round 2 are its 896 cells, the shared top-k sets and the integer criterion. Colour it
  purple with a teal note, or say "extended".
- **N5.** Figure 7 title "At every term weight R1 stays above R2 and R3": the three coincide at λ_a = 0; the text says
  "every positive weight".
- **N6.** `:140` "16:41 | … R2 and R3 reader stages done": R3 finished at 16:42:15 (`probs_R3_A0.json`); the log row
  has the same rounding.
- **N7.** §5.1 omits its strongest evidence: over all 896 cells, in sample, the best fused cell is a k_top 13 cell for
  every reader (cells 117, 67, 167), and the best restricted cell is lower (18.976, 18.683, 18.728 against 19.116,
  18.703, 18.766). One sentence would close the "did not help" claim beyond the four cross-fit picks.
- **N8.** `test_r2_fusion.py:206-209` docstring says the test fails "with K taken from the fused score"; with the
  counterpart built from the condition-dependent term and the per-cell assertion deleted (mutation F1b) this test
  passes, and only `test_cell_statistics_equal_naive_reference` catches it. The guard set holds; the docstring
  overstates this one test. Separately, the runners' wiring has no unit test (F6 and R4 survive every test; see §4):
  for this round the runtime per-cell assertion, the regression check and two independent re-derivations cover it.
  Before any test-seed run under this code, add one end-to-end smoke assertion of the wiring.
- **N9.** For future rules: §7's 1e-12 relative tolerance on τ is at the level of float64 operation-order noise for
  τ_0 (a near-tie margin of 3.8e-5). Evaluating P·(π̂/π_train) instead of (P·π̂)/π_train moved R2's τ_0 by 1.5e-12
  relative with identical gates, picks and cells. An absolute 1e-15 (or relative 1e-9) on τ would separate real
  differences from rounding.

## 4. Mutation checks

Each mutation edited one committed file, ran the named test(s) with
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 … -m pytest <file>::<test> -q`, then restored the file with
`git checkout -- <file>`; `git status --porcelain` was empty for the file afterwards and for the round's tracked files
at the end (`fr_mutate.py`, record `out/fr_mutate.json`). Unmutated, both suites pass (43 tests).

| Id | Guard / mutation | Test(s) run | Result |
|---|---|---|---|
| F1 | Counterpart condition-free: the counterpart is built from the gated, condition-dependent term (`G[t]` → `gated[t]` in `cell_statistics`) | `test_counterpart_is_condition_free_and_K_comes_from_B` | **fails** (caught) |
| F1b | F1 plus the per-cell `_require_condition_free` deleted | same test; `test_cell_statistics_equal_naive_reference` | same test **passes** (not caught); naive reference **fails** (caught). See N8 |
| F2 | Integer tie-break: the counterpart's max-ρ pick takes the last tied cell | `test_integer_crossfit_ties_go_to_lowest_cell` | **fails** (caught) |
| F3 | Integer criterion: the fused pick compares round 1's float means | `test_integer_criterion_exact_where_float_means_misorder` | **fails** (caught) |
| F4 | Restriction: outside candidates placed below the highest score in K (min → max; R@1 unaffected, `swap` affected) | `test_restriction_keeps_S_inside_K_and_B_order_outside`; naive reference | restriction test **fails** (caught); naive reference passes, as expected (first place unchanged) |
| F5 | D12 clause 3 on the gain statistic's point instead of its lower bound (`r2_apply_rule.py`) | `test_apply_rule_carry_tie_and_kill` | **fails** (caught) |
| F6 | Wiring: `run_r2_fusion.fusion_pass` passes the gated term as G_cf | whole `test_r2_fusion.py` | 22 pass (no unit test; on real data the per-cell assertion and the regression check fire). See N8 |
| R1 | R2 code check never compares condition b | `test_code_check_tolerance_is_1e12_absolute` | **fails** (caught) |
| R2 | R3 draws: a replacement caption may come from the anchor's painting | `test_draws_respect_the_painting_constraints` | **fails** (caught) |
| R3 | R3 nested banks: replaced positions depend on k (order rolled by k) | `test_impure_banks_follow_the_rule_and_are_nested` | **fails** (caught) |
| R4 | Wiring: `run_r2_readers.stage_r2` runs EM on condition a only | whole `test_r2_readers.py` | 21 pass (no unit test; the real π̂ was re-derived on both conditions by two independent codes). See N8 |

Verdict on the tests: every pure-function guard I removed made its test fail; the guards of the four tests the brief
named (counterpart condition-free, integer tie-break, restriction, R3 nesting, R2 code check) all bind. The gaps are
the runners' wiring, which unit tests do not reach.

## 5. Process against the rule

| Check | Evidence (Amsterdam time; git times are +02:00 = Amsterdam) | Result |
|---|---|---|
| Rule committed before any code | dc9fac6 16:13:07; first code a221b64 16:13:24; fusion code first modified 16:17 | ok |
| No candidate number before the regression check (§4.7, §7) | `regression_check.json` 16:39:32; first full-data candidate output `tau_R1_A0.json` 16:41:30, `cand_R1_A0` 16:41:52; R2/R3 thresholds 16:43:55, candidates 16:44:16 | ok for full-data numbers; smoke exception, S2 |
| R2 order: μ42, σ42, code check, π̂ before thresholds and scores | `probs_R2_A0.json` 16:41:55 (code check runs before any R2 probability in `stage_r2`); `tau_R2_A0.json` 16:43:55 | ok |
| R3 order: D(k) and k* written before training | `r3_k_A0.json` 16:42:01; pickle and probabilities 16:42:15 | ok |
| Reader stages in parallel with their task review | launched 16:40 at ee48015; `task-2-review.md` 16:41:30 (approved, no code change); outputs 16:41:55 and 16:42:15 | ok, disclosed |
| Code that ran = code committed | SHA-256 in each cand json = `r2_fusion.py` 1406e8e7…, `run_r2_fusion.py` fe6fb9ff…, identical at cdf9b4d, ee48015, HEAD; reader stream f9937a34… / 4c135af6… = HEAD | ok (also closes Task-1 minor 3) |
| Re-derivation before the rule (§7) | phase 1 16:30 (no candidate number), phase 2 authorised 16:53:11, outputs 16:54:12, comparison 16:55:50; rule applied 16:57:19 | ok |
| Outcome table (§8) | no candidate clears → item 5: no test, seeds 49 to 51 unused, results to the user; A1 ablation on the largest bar margin (R1), labelled "best development candidate, not carried" | ok |
| A1 ablation after the rule; thresholds before scores | `tau_R1_A1.json` 16:58:53, `cand_R1_A1` 16:59:14, `a1_ablation_R1.json` 17:01:11 (written once; correct values, so neither slip produced a written run) | ok (smoke A1 at 16:21, S2) |
| Disclosures (§6.10 multiplicity, development data, A1 descriptive, diagnostics after the rule) | report §7 | ok; selection asymmetry R1 vs R2/R3 missing (S1) |
| Write scopes (§9) | all round-2 outputs in `results/` (gitignored) or `results/smoke/`; diagnostics in `docs/reports/assets/…`; no round-1 file changed; no existing source file edited (diff adds files only, plus one row in `reports_sum.md`; `check_reports_sum.py` OK) | ok |
| Run log times vs git and file metadata | all rows agree to the minute (after e50da1e's corrections) except 16:41 for R3 (16:42:15) | ok (N6) |
| Report times vs git and metadata | rule commit 16:15 vs 16:13:07 (S3); others agree | S3 |

## 6. Triage of the deferred minor findings

None must be fixed for this round's verdict or report.

| Finding | Triage |
|---|---|
| T1-1 float comparison at the 0.05 tie boundary (`r2_apply_rule.py:94`) | Moot: no candidate clears, so E is empty and the tie rule never ran. Fix (`M − bar ≤ 0.05 + 1e-12`) before any future carry |
| T1-2 dead assignment in `save_candidate` | Cosmetic; no fix |
| T1-3 regression gate checks no code hash | Closed for this round: the code SHA-256s recorded in every cand json equal the committed fusion code at cdf9b4d and ee48015 (§5) |
| T1-4 tau written before cand files; manual recovery | No crash happened; no fix |
| T1-5 failed regression blocks reruns (usage text) | Not triggered; no fix |
| T1-6 docstring promises "section 8 rows" | Cosmetic; no fix |
| T1-7 test gaps (refuse_existing, D14 mismatch, apply_rule end to end, non-tie control) | Not needed for this round (outputs verified twice independently); same class as N8. Worth one end-to-end smoke test before any test-seed run |
| T1-8 per-half array copy | Performance only; no fix |
| T2-1 runner ordering untested (r3_k before training, k* = 4 branch, overwrite refusal) | Ordering verified from file times (§5); k* = 4 branch not exercised (k* = 2). No fix now |
| T2-2 draw tests use a same-author reference | Closed: my draw code, written separately from the rule text, reproduces the stored SHA-256s of order, img and cap for both halves, as did the re-derivation |
| T2-3 real-data tests skip silently | Nit; no fix |
| T2-4 pure helpers raise SystemExit | Cosmetic; no fix |
| T2-5 pickle not written atomically | No crash; no fix |
| T2-6 predict_proba called twice for a diagnostic | Cosmetic; no fix |

## 7. Files and storage

Scripts: `final_review/fr_prep.py`, `fr_core.py`, `fr_cands.py`, `fr_r3.py`, `fr_diag.py`, `fr_mutate.py`. Outputs in
`final_review/out/` (gitignored, 63 MB in all; the largest file is the 19 MB cache): `fr_cands_*.json` (342
comparisons), `fr_r3.json` (29), `fr_diag.json`, `fr_mutate.json`, logs. Nothing over 1 GB is left behind.
