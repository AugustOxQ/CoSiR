# Opus check of the round-2 decision rule (before its commit)

**Checked:** `src/test/20261118_reader_fix_round2/DECISION_RULE.md`, 569 lines, SHA-256
fc94ec8d3a47f5db616f27c565316047f70b00adbf96ee5f79630d785316dd01 (uncommitted draft), on 2026-10-06 between 15:47 and
16:10 (Amsterdam), against the approved spec (SHA-256 ceecc515…11c8), round 1's rule (613d8c9d…d23c), round 1's code
(`common.py`, `rc_core.py`, `run_rc.py`, `rb_build.py`, `rb_eval.py`, `rb_features.py`, `apply_rule.py`), round 1's
stored results and the handoff `docs/superpowers/handoffs/2026-10-06-reader-fix-round2-handoff.md`.

**No round-2 number was computed.** The scripts in this folder read round 1's files only: `check_sha.py` (every SHA-256
in the draft), `check_numbers.py` (round-1 numbers the draft quotes), `check_bank.py` (bank, halves and reader
structures), `check_B_ties.py` (dtype, condition-freeness and ties of B), `check_crossfit_arith.py` and
`check_crossfit_examples.py` (round-1 R-c's 224 cells recomputed with `rc_core`, float against exact criteria),
`check_draws.py` (the §4.4(b) draw procedure executed as written, no features, SMDs or k). CPU only
(`CUDA_VISIBLE_DEVICES=`, 8 threads).

## Counts

| Class | Count | Findings |
|---|---|---|
| Blocking | 2 | B1 cross-fit criterion arithmetic; B2 bank array names |
| Should-fix | 7 | S1 re-derivation agreement; S2 numeric recipes; S3 code fixes, crashes, stopped steps; S4 write scope vs run_baselines; S5 top-k reading made explicit; S6 bank records; S7 D(k) disclosure |
| Nit | 9 | N1 to N9 |

Verdicts on the drafter's four resolutions: (1) top-k reading: **must be made more explicit** (S5), no decision number
changes; (2) R3's impure banks: **must change** the array names (B2), otherwise fully specified and bit-reproducible,
records to add (S6); (3) A1 ablation subject: **acceptable as written** (details in N5); (4) R2 shift report and
k* = 4: **acceptable as written**.

---

## Blocking

### B1. The cross-fit criterion is not implementation-independent: floating-point evaluation breaks exact ties against the "lowest cell number" rule

- **Where:** §4.5 item 8 (lines 378 to 383), §4.6 (line 391), and through them §4.7, §6.3, §6.8 and the re-derivation of §7.
- **Problem.** The rule defines the fused pick as "the cell with the largest min(mean R@1 − r_ctrl, mean gain) on the
  tune half (ties to the lowest cell number)". Every per-episode R@1 and gain is a multiple of 0.25, so exact ties
  between cells are common, but the value depends on how the difference is evaluated in floating point:
  `rc_core.select_fused` computes `float(mean R@1) − float(mean control R@1)` and compares it with `float(mean gain)`;
  an independent implementation computing `(ΣR@1 − Σctrl)/n`, or comparing an R@1-limited with a gain-limited cell, can
  order mathematically tied cells differently. The independent re-derivation of §7 is required to use its own code, so
  two implementations can pick different cells and produce different decision numbers. With 896 cells and three
  candidates the chance of a tie at the maximum is higher than in round 1.
- **Evidence** (`check_crossfit_arith.py`, `check_crossfit_examples.py`, round-1 R-c's 224 cells recomputed with
  `rc_core`; the stored per-anchor arrays are reproduced exactly):
  - half 0: four cells tie exactly at the maximum (4·Σ criterion = 175: cells 116, 125, 133, 142); 12 cell pairs are
    ordered differently by the float expression than by exact arithmetic; half 1: 3 pairs;
  - example, half 0: cells 6 (τ_0, 0, 8) and 43 (τ_0, 8, 1) are exactly tied (47 / 24,576), yet the float criteria are
    0.0019124348958333148 and 0.0019124348958333333, so a float comparison would prefer the higher cell number;
  - round 1 was lucky: at the maximum the float pick equals the exact first-of-ties pick on both halves (fused cells
    116 and 119, counterpart 58 and 123, control σ = 0), so an exact rule reproduces round 1 and the regression check.
- **Replacement text** for §4.5 item 8:
  > 8. **Cross-fit, min-margin, compared exactly** (round 1's `rc_core.control_choice` and `select_fused`, extended to
  > 896 cells). Every per-episode R@1 and gain of `per_anchor` is a multiple of 0.25. For a score x and a tune half
  > (the episodes with parity h), ρ(x) = Σ 4·R@1 and γ(x) = Σ 4·gain over the tune half's episodes, both integers. The
  > control takes σ* = the smallest of the 30 control sums with the largest ρ((1 + σ)·z(B)), unrestricted;
  > ρ_ctrl = ρ((1 + σ*)·z(B)). The fused reader takes the cell with the largest min(ρ(S′) − ρ_ctrl, γ(S′)), compared as
  > integers, ties to the lowest cell number; that cell's S′ scores the episodes of the other half (parity ≠ h).
  > Means and their differences are never compared as floating-point numbers. On round 1's 224 cells this gives
  > `rc_core.select_fused`'s picks (cells 116 and 119, σ* = 0 on both halves).

  and in §4.6: "on each tune half the cell with the largest ρ (§4.5 item 8), ties to the lowest cell number, scores the
  other half (on round 1's 224 cells: 58 and 123, as `rc_core.select_cf`)". §6.3 and §6.8 then inherit it.

### B2. R3's draw procedure names bank arrays that do not exist

- **Where:** §4.4(b) line 303 ("side 0 is the `pairs_a` arrays, side 1 the `pairs_b` arrays") and §4.4(c) lines 315 to
  318.
- **Problem.** The bank npz has no `pairs_a` or `pairs_b`. Its per-pair arrays are `pairs_a_img`, `pairs_a_txt`,
  `pairs_b_img`, `pairs_b_txt`; a pair is the image of one row and the caption of another row
  (`src/eval/aspect_episodes.py:22`). An implementer must guess which arrays "the `pairs_a` arrays" are and that
  "position p" is the column index of both.
- **Evidence** (`check_bank.py`): `rb_bank_A0_half{0,1}.npz` keys `anchor` (49,152,) int64, `candidates` (49,152, 13),
  `pairs_a_img`, `pairs_a_txt`, `pairs_b_img`, `pairs_b_txt` (49,152, 4) int64, `block_pairs`, `block_size`, `seed`,
  `half`; A1: the same with 98,304. `load_bank` (`rb_build.py:522`) builds `AspectEpisodes` from exactly these.
  In both A0 banks no pair has its image and caption from the same row or painting (share 0.0), and every episode's
  paintings are all distinct (share 1.0).
- **Replacement text** for the parenthesis of §4.4(b):
  > (N = 49,152; A = the bank's `anchor` array, (N,) int64 local rows; R = `local_rows_half{j}` of `rb_halves.npz`;
  > paint = `painting_of_local_row`; side 0 = the arrays `pairs_a_img` and `pairs_a_txt`, side 1 = `pairs_b_img` and
  > `pairs_b_txt`, each (N, 4) int64 local rows; position p = column p of both arrays of a side)

  and for §4.4(c): "… are replaced: `pairs_{a|b}_img[n, p]` becomes img[n, s, p] and `pairs_{a|b}_txt[n, p]` becomes
  cap[n, s, p]. These four arrays are the bank's only per-pair arrays; `anchor`, `candidates`, `block_pairs`,
  `block_size`, `seed` and `half` are unchanged."

---

## Should-fix

### S1. The re-derivation "disagrees" outcome has no tolerance and no rule for which computation settles it

- **Where:** §7 line 526 and §8 line 553.
- **Problem.** Two correct implementations differ at the last bit in μ42 and σ42 (numpy against `StandardScaler`'s
  incremental formula), in π̂ (EM summation order) and in R3's readers (row order of the training matrix). Taken
  literally, every such difference is a "disagreement", and "the number re-derived from the data settles it" names both
  computations. Round 1's re-derivation met exactly this (its log, 03:11: "Top-two margins differ by at most 1.6e-7 when
  σ is computed in float64 instead of float32"). The intervals depend on the bootstrap's random stream, so "own code"
  for the bootstrap would change every bound.
- **Replacement text** (append to the §7 re-derivation sentence and use in the §8 row):
  > The re-derivation writes its own code for everything except the bootstrap, which is
  > `src.eval.aspect_metrics.cluster_bootstrap` itself (5,000 resamples, seed 42, chunk 250, clusters the anchor
  > paintings). It also re-derives R3's chosen C per half and the chosen cells of both cross-fits. **Agreement** means:
  > every discrete quantity identical (picks, gates, top-k sets, chosen cells, σ*, k*, chosen C, the bar comparator,
  > each clause of D12); μ42, σ42, π̂, τ and D(k) equal to within 1e-12 (relative); every bar margin, gain statistic
  > and bound equal to within 1e-9 percentage points. A difference beyond these is traced to its cause before the rule
  > is applied; the computation that follows this file's text settles it, and the user is told.

### S2. Numeric recipes that two implementations could compute differently are not written out

- **Where:** §4.3(a) lines 263 to 268; §4.4(g) lines 336 to 343; §4.1 line 246.
- **Problem.** (i) "per-feature mean and standard deviation (ddof 0, as `StandardScaler` uses)" admits both
  `X.mean(0)`/`X.std(0)` and `StandardScaler().fit(X)`, which differ at the last bit. (ii) §4.4(g) does not state the
  training matrix's row layout or the fold order; `KFold` folds depend on episode order and lbfgs on row order.
- **Replacement text.** §4.3(a): "μ42 = `X.mean(axis=0)` and σ42 = `X.std(axis=0)` (numpy, float64, ddof 0), X =
  `numpy.vstack([F_a, F_b])` with F_c the float64 arrays of `rb_eval.seed42_features`; an entry of σ42 exactly 0 is
  replaced by 1 and reported." §4.4(g): "per half j, X, y and the episode index are
  `rf.stack_conditions(F_a, F_b, y_a, y_b)` of the impure bank (rows 0 to N − 1 condition a, N to 2N − 1 condition b,
  episodes in the bank's order), folds `rf.episode_folds(N)`, then `rb_build.fit_half_reader(X, y, episode, N, 3)`, as
  `rb_build.stage_train` calls it."

### S3. No clause for code corrections, crashed runs, or a candidate stopped by a failed check

- **Where:** §8 (no row); §9 line 559 ("refuse to overwrite non-smoke results"); §7 line 525.
- **Problem.** Round 1's rule (§4.2 lines 195 to 197) said "Correcting code so that it matches this specification is not
  a change; such a correction is logged with the numbers before and after." The draft drops it. With write-once results,
  a bug found after a results file exists, or a crash midway, leads only to the catch-all "the user decides", which
  stalls a run meant to finish today. A step stopped by a failed check (R2's code check, R3's purity-4 checks) leaves
  open whether the others' rule application waits for the Thursday cutoff.
- **Replacement text** (two §8 rows, and the sentence in §4):
  > | A run crashes, or a bug is found before the rule is applied (by the implementer, a test or the re-derivation) |
  > Correcting code so that it matches this file is not a change. A run that crashed before its results file was
  > complete is repeated after its partial outputs are deleted. A results file already written is not overwritten: the
  > corrected run writes beside it with the suffix `_fix<n>`, the log records the cause and the numbers before and
  > after, and the rule uses the corrected file. |
  >
  > | A step stops on a failed check | The other candidates continue. The stopped candidate has no development numbers
  > until the user rules; the user may drop it at once, after which the rule is applied as soon as the others have
  > numbers; otherwise it is applied at the cutoff. |

### S4. §9's write scope contradicts §6.2

- **Where:** §9 line 559 ("write only to this folder's `results/`") against §6.2 line 473 and §6.4 line 488.
- **Problem.** `run_baselines.py --episodes-seed <s>` writes `episodes_seed<s>.npz` and `baselines_seed<s>.json` into
  `src/test/20261030_aspect_baselines/results/`, and the ledger gets a row; it also has an `--overwrite` flag
  (`run_baselines.py:86`).
- **Replacement text:** "… and write only to this folder's `results/`, except that `run_baselines.py` (§6.2), run
  without `--overwrite`, writes its own outputs for seeds 49, 50 and 51 in `src/test/20261030_aspect_baselines/results/`,
  and `docs/superpowers/episode_seed_ledger.md` gets its row."

### S5. Top-k reading: state that it departs from the spec's literal words only where no decision depends on it

- **Where:** §4.5 item 6 (lines 368 to 373).
- **Problem.** The spec says "first place must come from B's top k_top candidates; below it, B's order"; the draft keeps
  the fused order inside K and applies B's order outside K. The spec lets the rule govern where they differ, but the rule
  does not say it reads the spec this way, or why it is harmless.
- **Evidence.** `src/eval/aspect_metrics.py`: `first_place` counts a target only if it scores strictly above every other
  candidate (ties miss); R@1, `other`, `gain` and `strict` (= hit_aa·hit_bb) depend only on first place. `swap`
  (p_A above p_B under a and p_B above p_A under b) is the only stored per-anchor metric that depends on ranks below
  first. No rule item uses `swap`: `evaluate_fused` uses r1 and gain (and either), the cross-fit uses R@1 and gain, the
  GO checks use R@1 and gain. The regression check compares `swap` only at k_top = 13, where both readings give S
  unchanged.
- **Replacement text** (append to item 6):
  > This reads the spec's "below it, B's order" as "outside K, B's order". Both readings put the same candidate first
  > and count a tie for first as a miss, so R@1, the other-aspect rate, the gain, `strict`, either, every cross-fit
  > pick, bar margin, gain statistic and GO check are the same under both. Only `swap` can differ, for k_top < 13, and
  > `swap` enters no rule. At k_top = 13 both readings leave S unchanged.

### S6. R3's banks: records for a bit-level re-derivation

- **Where:** §4.4(b) to (d).
- **Problem.** The procedure is complete and reproducible (dry run: one rejection round for images and one for captions
  on each half, identical output on a second run), but nothing is recorded that lets the re-derivation confirm it built
  the same banks before comparing D(k).
- **Replacement text** (append to §4.4(d)): "The SHA-256 of `order`, `img` and `cap` (int64, C order) of each half and of
  each purity's feature matrix of each half are written to the results with the D(k) table."

### S7. D(k) is a level-matching criterion; disclose it and add one diagnostic

- **Where:** §4.4(f) and (i).
- **Problem.** Three of the 18 inputs (the Δ features) give an SMD of exactly 0 for every k, because Δ^b = −Δ^a makes
  their pooled mean 0 on seed 42 and on every bank; S and C, and sd_support and sd_contrast, mirror each other. D(k) so
  measures shifts in the levels of S, C, their spread and the match share, not the strength of the support-contrast
  difference that purity changes most directly. Round 1's purity-4 SMDs (S and C −0.19, −0.44, −0.62 for affect, image
  and caption) show level gaps that may come partly from the heads (cross-fitted half heads on the bank against the
  standard heads on seed 42), so a small k* would not by itself show realistic purity. The spec fixed the criterion; this
  is a disclosure, not a change.
- **Evidence:** `rb_diag_A0.json` `c_shift_report.smd`: `affect__Delta`, `image__Delta`, `caption__Delta` = 0.0;
  `*__S` = `*__C` and `*__sd_support` = `*__sd_contrast` to about 1e-15.
- **Replacement text** (append to §4.4(i)): "D(k) matches levels: the three Δ features have SMD 0 for every k by
  construction, and S/C and the two standard deviations mirror each other. Reported beside it, entering no rule: the
  mean of |Δ_h| per grouping on seed 42 and on the bank of each purity."

---

## Nits

- **N1.** §4.1 line 246: "condition a's 12,288 rows stacked on condition b's" can be read either way; write "followed by
  condition b's" (as §4.5 item 2 and §4.4(f) do; `rb_eval.shift_report` uses `vstack([F_a, F_b])`).
- **N2.** §4.5 item 5: name the array: "b = `bundle.B[c][d]`, float32", and assert `B[a] == B[b]` before computing K.
  Verified on seed 42: B is float32 and identical under both conditions; no row has any tie among its 13 B scores, and
  the top-2, top-3 and top-5 sets from B and from z(B) coincide in every row, so K is unambiguous on seed 42.
- **N3.** §4.5 items 4 and 6: say that S′ may be held in float32 or float64 with identical ranks: a z-score over 13
  candidates is at most √12 ≈ 3.464, so |S| ≤ 33·√12 ≈ 114.3 < 128, where float32 spacing is at most 7.6e-6, and
  `per_anchor` casts to float64; members of K keep their float32 values exactly, and every candidate outside K stays at
  least 1 below them. Gates compare float64 margins with float64 τ (no NumPy 2 scalar-promotion effect).
- **N4.** D6 and §4.5 item 8: list the 30 control sums (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, 3, 4, 4.25, 4.5,
  5, 6, 8, 8.25, 8.5, 9, 10, 12, 16, 16.25, 16.5, 17, 18, 20, 24, 32), per "every number written out".
- **N5.** §4.8: write the A1 block order ((affect, caption), (affect, csd), (affect, image), (caption, csd),
  (caption, image), (csd, image); 16,384 each, no third grouping), the A1 SMD count (24,576 seed-42 rows against 393,216
  bank rows), that k*_A1 = 4 makes R3/A1 = R1/A1, that R2/A1 runs the code check of §4.3(e), and that A1 thresholds are
  written before any A1 score.
- **N6.** §4.4: replacement rows are drawn without regard to paintings already in the episode other than the anchor's;
  in the dry run 0.31% (half 0) and 0.34% (half 1) of replacement slots reuse a painting of the episode, while round 1's
  banks and the seed-42 episodes have all-distinct paintings. Say so in one sentence (negligible; fully specified).
  Also say why a replacement pair is cross-painting: every pair in the banks and in the real episodes is the image of
  one row and the caption of another row from a different painting.
- **N7.** D14 line 195: round 1's rule is not "asserted by round 1's code on import". `common.py` and `rb_build.py`
  assert nothing at import; `common.load_bundle` asserts it (`assert_rule` and `verify_inputs`), and
  `rb_build.load_json_npz`/`load_readers` compare the stored JSON's `rule_sha256` string. Write "asserted by
  `common.load_bundle`".
- **N8.** §4.7: the stored `pick__a/b` are int8 and R1's picks int64; say "equal in value". Also say that R1's 896-cell
  statistics may be computed in the same pass as the check, but no number from cells 224 to 895 is written or printed
  before the check passes.
- **N9.** §4.3(a): `StandardScaler` replaces near-zero scales too (`_handle_zeros_in_scale`); with S2's text ("exactly
  0") the parenthesis "as `StandardScaler` does" can go. None is expected (smallest bank variance 5.6e-5).

---

## The four drafter resolutions

1. **Top-k: fused order inside K, B's order outside.** *Must be made more explicit* (S5). It is consistent with the
   spec's intent and changes no decision number: R@1, other, gain, either and `strict` depend only on which candidate
   is strictly first, and the restriction puts the same candidate first under either reading (ties for first miss under
   both). Only `swap` differs, only for k_top < 13; no rule item uses it, and the regression check compares `swap` at
   k_top = 13 only.
2. **R3's impure banks.** *Must change* the array names (B2); otherwise *acceptable*. The RNG call order, shapes, the
   rejection loops (masked assignment fills in row-major order) and the two painting constraints are fully specified;
   my dry run of the text as written (`check_draws.py`) terminates after one rejection round per loop on both halves and
   is bit-identical on repeat. Nesting holds by construction. "Replacing a pair" covers every per-pair array the bank
   stores (image row and caption row); labels are per block, posteriors are looked up per row from `rb_heads_*.npz`
   (replacement rows are half-j rows, so cross-fitted). The cross-painting replacement pair matches what a "pair" is in
   this codebase (N6). Add S6's records.
3. **A1 ablation subject.** *Acceptable as written.* The spec names only the carried candidate; using the best
   development candidate, labelled "not carried", when nothing is carried adds a descriptive line without touching a
   decision. Exact ties go to R1, which also covers R3 = R1. Details in N5.
4. **R2 shift report; k* = 4.** *Acceptable as written.* The spec's diagnostics list names "the shift report for R2 and
   R3" without defining R2's; the draft's (feature offsets against each scaler, π̂ and iterations, top-probability
   distribution against round 1's 0.694 and 0.781, share of changed picks) is diagnostic only. k* = 4 gives R3 = R1
   with identical bar margins; item 4's order (R1 first among ties within 0.05) then carries R1, as the spec's "k = 4 is
   round 1's reader" and its carry order imply.

---

## Checklist results

**A. Matched controls: pass.** For every candidate the counterpart keeps z(B) and z(T^c) z-scored over all 13
candidates per condition, the candidate's own τ and gates, the same 896 cells and order, the same K, and replaces only
the gated term by G_cf (float64 mean, float32 cast, asserted condition-free; `rc_core.g_cf`). K depends on B alone;
B is identical under both conditions on seed 42 (verified), so the restricted counterpart is condition-free (outside
K it depends on min over K of a condition-free score), and the per-cell assertion guards test seeds. R2's μ42, σ42 and
π̂ and R3's k* are pooled over both conditions and are shared by the fused reader and its counterpart. The counterpart
chooses among the same 896 cells with the most favourable criterion (max R@1), so its freedom is at least the fused
reader's. The control in the min-margin criterion is the unrestricted (1 + σ)·z(B); restricting it to its own top-k
would not change its first place.

**B. Outcomes: one action each, after S3.** Walked: SHA mismatch, bundle check, regression-check failure (stop, no
candidate number), R2 code check, EM cap (π̂ = π^(cap), reported), σ42 zero (1, reported), purity-4 feature and SMD
checks, non-finite SMD, D(k) ties (larger k), k* = 4, C ties (smaller C), cell ties (with B1's exact rule), bar
comparator ties, carry ties within 0.05 (a tie at exactly M − 0.05 cannot occur: per-anchor R@1 differences are multiples of 0.25, so
bar margins and their differences are multiples of 25/12,288 ≈ 0.00203 pp, and 0.05 pp is 24.576 such steps), a bar
point of exactly 0.5 (245.76 steps, likewise impossible), cutoff, no clear, one clear, several
clear, test hash collision, GO, partial pass, NO-GO readings, time-box, re-derivation disagreement (needs S1's
tolerance), A1 ablation failures (stop that step). Gaps: crash or bug fix after write-once results, and a stopped step's
effect on the others (S3). No outcome has two actions once S4 removes the write-scope contradiction.

**C. Other ambiguities.** Cell order: `rc_core.rc_cells()` is τ outer, then `nested_cells()` (λ_u outer, λ_a inner), so
cell t·56 + u·8 + a, which is the draft's formula with κ = 0; cells 0 to 223 are round 1's in round 1's order. Thresholds:
`rc_core.thresholds` concatenates a then b in float64 and takes `numpy.percentile` (linear) at 0, 25, 50, 75, as
written; one margin equals τ_0 exactly and passes the ≥ gate. Margins are float64 (`rf.picks_and_margins`). The EM,
SMD, D(k) tie rule, C grid, KFold and log-loss criterion are written out and match `rb_features`/`rb_build`. The
24,576 seed-42 rows are `vstack([F_a, F_b])` (N1). Dtypes: no ranking or tie changes (N3). Tie handling in R@1 under the
restriction: ties for first inside K miss; outside K never ties with K. K with ties in B: stable rule written; no ties on
seed 42 (N2). Test seeds: τ, μ42, σ42, π̂, k*, the half-readers and the family frozen; B, K, gates and both cross-fits
rerun per seed. The only implementation-dependent quantity found is the cross-fit criterion (B1).

**D. Numbers: all verified.** All 44 SHA-256 values in the draft (36 in D14, the spec, round 1's rule, the six inputs of
§2, D1, D2 and D8) equal `sha256sum`. Round-1 R-c: fused 18.918863932291664, counterpart 18.475341796875, bar margin
0.4435221354166667 [0.21646171563312194, 0.6735669710776852] against the counterpart, gain statistic 2.667236328125
[2.325087836946873, 3.012361650695922]; τ 3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526,
0.7502585816077211 (`rc_tau.json`, 24,576 values, parent Rb_expected_A0); chosen cells fused τ_2 (0, 2) and τ_2 (0, 16),
counterpart τ_1 (0, 0.5) and τ_2 (0.5, 1), control σ 0 on both halves. B 18.341064453125 [17.9745, 18.6972];
B′(A0) 18.436686197916664; B′(A1) 18.804931640625. Reference rows of D13 (both learned-reader rows, the arg-max row
copied from round 1's rule) and pick accuracies 51.261393229166664 and 48.69 equal the stored JSONs; told ceiling +1.64
[1.37, 1.92] (report line 132); cosine 12.96, RCA 13.38; 0.5 − 0.4435 = 0.056. Counts: 12,288 episodes, 4,602
clusters, 4,096 per pair, parity 6,144/6,144; banks 49,152 (A0) and 98,304 (A1) per half in the stated block order;
rows per half 91,949 and 91,745 (sum 183,694); readers C 1.0 and 100.0, 98,304 examples per half (A0), scikit-learn
1.6.1 and numpy 2.2.6 (the env has the same); `half{j}__X` (98,304, 18) float64; diag top probability 0.6943 and 0.7814.
`cand_Rc_Rb_expected_A0.npz` holds every key §4.7 names. `rc_core` reproduces round-1 R-c's stored per-anchor arrays
bit for bit (so the regression check is feasible).

**E. Spec consistency: no contradiction** of the spec's §2 and §3 decisions. The added details (EM stopping rule, R3
tie to the larger k, A1 subject when nothing is carried, R2 shift report, k* = 4 handling) are consistent with them.
The only literal departure is the top-k reading (S5), which the spec's precedence clause allows and which changes no
decision number.

**F. Feasibility: fine on CPU today.** Bundle 72 s and about 5.3 GB per process. Round-1 R-c's 224 cells (fused and
counterpart) took 2.8 s, so 896 cells take about 12 s per candidate plus the restriction; R3's draws take seconds,
features for three purities a few seconds per half, training at k* about 45 s (round 1's A0 training 42.6 s); A1 about
twice that. Seeds: `run_baselines.py` took 207 s per seed (seed 48). Nothing needs the GPU.
