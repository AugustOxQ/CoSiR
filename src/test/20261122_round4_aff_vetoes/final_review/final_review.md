# Round 4 (vetoes on AFF's gate): whole-branch final review

**Verdict: CONFIRMED WITH FIXES.** The kill is true. A third derivation with my own code reproduces every decision
quantity of seed 42 exactly: all gates, cells, σ*, per-anchor arrays, bar comparators, D10 clauses, Δ_k (−18, −18, −34)
and the empty carry set. The fixes are in the draft report's provenance and process statements, plus two test gaps that
must be closed before this code is reused. No finding is blocking. Counts: 0 blocking, 6 should-fix, 13 nits.

Reviewer: Opus (fresh context), 2026-10-07 02:56 to 03:20 (Amsterdam). Brief:
`.superpowers/sdd/2026-10-07-round4-aff-vetoes/final-review-brief.md`. Binding rule `DECISION_RULE.md` (SHA-256
cf11a873…911b, asserted by every script below). CPU only (`CUDA_VISIBLE_DEVICES=`, 8 threads, no bytecode), at most 2
processes. I wrote only under `final_review/`, made no commit and dispatched no subagent. I never built or read a seed
from 49 to 54, never opened a `build_seed*.log`, and found no seed-52 to 54 file anywhere. One disclosure: verifying
D11 hashed `baselines_seed{49,50,51}.json`, which are D11 rows (SHA-256 only, content never parsed). The copied
`test_r4_common.py::test_inputs_pass` in my mutation runs did the same.

## 1. What I re-derived and how

**Third derivation** (`fr_derive.py`, 91 s; results hashed before any comparison: `out/fr_results.json`
8c2d7cd7…, `out/fr_arrays.npz` a9cf66bd…, written 03:02). Imports: only what rule §8 allows a re-derivation, namely round
1's `common.load_bundle()` (seed 42: episodes, parity, clusters, posteriors including csd, B, B′(A0), B′(A1)),
`rb_build.load_readers("A0"/"A1")`, `zscore_rows` and `cluster_bootstrap`. I imported nothing from `r4_*`, `r3_*`,
`r2_fusion`, `rc_core`, `rd3_*` or `rederive/`. My own code computes:

- the 18 and 24 features, P and T, the margins and both readers' picks;
- v, v₇₅, a_v, and the R1, AFF, IMGABST, V4, V2 and V24 gates (float32);
- z(B), z(T), the gated terms and G_cf;
- the integer ρ and γ of all 224 cells, fused and counterpart, from my own strict-first-place hit counter;
- σ*, both integer cross-fits and the parity assembly;
- per-anchor r1, gain, other, swap and strict;
- the D8 comparator sets and tie orders, the bar margins, margins, gain statistics, either changes and per-pair bar
  margins, all with intervals;
- the D10 clauses, Δ_k from integers with its interval, and the carry.

Before computing any candidate number, my code reproduced the stored round-1 R-c T, margins and picks, the A1 reader's
stored probabilities and picks, τ, v₇₅ and its count of 9,216, all exactly. It also reproduced the rule's regression
targets for R1 (cells 116/119, 58/123; bar 0.4435221354166667 [0.21646171563312194, 0.6735669710776852]), AFF (cells
39/119, 149/10; every round-3 item-3 number; AFF minus R1 0.21769205729166666 […]) and IMGABST_q75 (cells 117/119,
58/67; every §5 item 3 number), exactly.

**Comparison** (`fr_compare.py` → `out/fr_agreement.json`). There were 222 comparisons against `dev_seed42.json`,
`carry.json`, `seed42_arrays.npz` and the rule's stated targets, under rule §8's tolerances:

- **Discrete quantities and arrays:** all identical, including 110 arrays (every fused and counterpart per-anchor array
  of R1, IMGABST, AFF, V4, V2 and V24, every gate array with its dtype, B, B′(A0), B′(A1), v, keep, both readers' picks
  and the margins).
- **Continuous quantities:** the largest absolute difference was 3.6e-15 pp. It comes from the either change, which I
  form as a difference of means and the implementation forms as a mean of differences. The one row that I had set to
  demand exact equality (IMGABST's either change against the rule's text) differs by 1.8e-15, for the same reason. That
  is far inside rule §8's 1e-9.

| Quantity (seed 42, pp) | V4 | V2 | V24 | AFF |
|---|---|---|---|---|
| fused / counterpart R@1 | 19.099934895833336 / 18.391927083333336 | 19.099934895833336 / 18.34716796875 | 19.0673828125 / 18.391927083333336 | 19.136555989583336 / 18.39599609375 |
| fused cells, counterpart cells, σ* | 39, 119; 167, 54; 0, 0 | 39, 119; 93, 166; 0, 0 | 39, 119; 149, 166; 0, 0 | 39, 119; 149, 10; 0, 0 |
| bar comparator (means) | B′(A0) 18.4367 > cf 18.3919 > B 18.3411 | B′(A1) 18.8049 > B′(A0) > cf 18.3472 > B | B′(A1) > B′(A0) > cf 18.3919 > B | B′(A0) |
| bar margin | 0.6632486979166667 [0.42921625314611117, 0.8944505576588029] | 0.29500325520833337 [0.010178842628223385, 0.578411107785257] | 0.262451171875 [−0.016448639942734416, 0.54087577359005] | 0.6998697916666667 […] |
| gain statistic lower bound | 2.4854376869391626 | 2.629555301390828 | 2.4376065646148186 | 2.780005709854805 |
| D10 c1 / c2 / c3 | yes / yes / yes | **no** / yes / yes | **no** / **no** / yes | yes / yes / yes |
| Δ_k (integer); point [95%] | −18; −0.03662109375 [−0.10506836425355538, 0.03271528780460466] | −18; −0.03662109375 [−0.11495968701561131, 0.038888209184998086] | −34; −0.06917317708333333 [−0.16515849108866412, 0.026392838403383488] | |

E = ∅, M = none, no tied set, nothing carried: **kill**. No boundary of rule §8 applies. The closest any D10 value comes
to its threshold is V2's bar-margin lower bound, +0.0102 against 0. No Δ_k is 0.

**Descriptive analyses** (`fr_descriptive.py`, from my own arrays). Every number in `figure_data.json` that the report's
§6 and Summary use matches my recomputation to 1e-12 (0 differences): the closure shares (τ_0 and as scored, all
pair × condition cells), the net rankings by closure class, better and worse counts, per-pair candidate minus AFF, the
gain, other and either changes against AFF, V4 closed against kept, the comparator gaps, Table 7 and V4 against V2. I
also checked every number quoted in the report's text against my files (§5 lists the few that need words changed).

**`build_figures.py`.** It runs on a scratch copy with only `ROOT` patched (`bf/`) and passes all its assertions. The
`figure_data.json` it writes is byte-identical to the stored one. The figures carry legends, labelled axes and a title
that states the finding.

## 2. The specific doubts

- **(a) V4 and V2 (same Δ −18, same fused R@1 19.0999).** Their gates differ on 1,830 condition-a and 1,684 condition-b
  values at τ_0, and on 7,385 gate entries over all τ. Their fused R@1 arrays differ on 426 episodes and their
  other-aspect arrays on 489. Their gain statistics (2.797, 2.952), counterpart cells (167/54, 93/166) and counterpart
  R@1 (18.392, 18.347) all differ. Neither has AFF's gates (both differ at every τ index) or AFF's R@1 array. My
  derivation built each family from its own gates and reproduced the implementation's arrays exactly, so no wiring
  explains the coincidence. In integer units both are 9,388 hit rankings against AFF's 9,406.
- **(b) Fused cells and counterparts.** All three fused cross-fits chose 39 and 119, AFF's cells, each as a unique
  maximum. The margins over the runner-up were:

  | Scorer | Tune half 0 | Tune half 1 |
  |---|---|---|
  | AFF | 3 | 5 |
  | V4 | **1** (runner-up cell 13) | 6 |
  | V2 | 5 | 7 |
  | V24 | 5 | 7 |

  Had the runner-up won on tune half 0, Δ would have been −25, −11 and −43: still a kill (`out/fr_extra.json`). Each
  counterpart is built from its candidate's own gates: my G_cf from each candidate's gates reproduces the
  implementation's counterpart cells and arrays exactly, and they differ from AFF's (149/10, 18.396). Mutation C1
  (`run_candidate` given AFF's gates) is killed by a test.
- **(c) The floor and the D10 values.** The comparator order and the means are in the table above. B′(A1) wins for V2
  and V24 by 0.368 over B′(A0), so the inclusion of B′(A0) in their set changed nothing. The D10 clause values are
  confirmed, and none lies within 1e-12 of its threshold.
- **(d) Nothing of a test seed.**
  - `src/test/20261030_aspect_baselines/results/` has no `episodes_seed5{2,3,4}`, `per_anchor_seed5*` or
    `baselines_seed5*` file (directory listing only).
  - `codes_provenance.json` still has round 3's SHA-256 (8e6a517b…bfaf).
  - The seed ledger last changed in round 3 (2f8c784): "52 and later | free".
  - This round's `results/` has no `build_seed*.json` and no `sensitivity.json`, and nothing was run on the smoke seeds.
  - The branch diff touches only new files (spec, plan, rule, this folder). No round 1 to 3 file changed.

## 3. Process against the rule

- **Order (§8).** The steps ran in this order:
  1. implementation (01:31 to 02:23);
  2. the regression items 1 to 5 (02:25, 262 comparisons, all pass);
  3. the development step (02:27);
  4. phase 1, which was hashed at 01:40, before the implementation ran, and compared at 02:35;
  5. the kill recorded at 02:36, after agreement, as §8 requires.

  The runner's console printed "KILL" at 02:27. The log marked it pending agreement, which is acceptable. **One
  departure:** step (1) was cut short by the Task 3 split (S3).
- **Rulings.** The ones within the rule, or process-only: the model choices, the parallel tasks, Task 3 owning items 4
  and 5, the promoted minors, Task 3a dispatched during Task 1's review, the boundary stop placed before `carry.json`
  (which matches §8's "before the step that depends on it is recorded"), the real run launched during Task 3a's review,
  no 3b, phase 2 or test seeds after the kill (§5 item 9), the workspace kept, and the report drafted early but
  uncommitted (§10).

  The `--dry` ruling created a run type the rule does not define. It computed seed-42 candidate results, unseen and then
  deleted, before the real run (N12). Two departures are not recorded as rulings: phase 1's reuse of round 3's
  re-derivation code (S1) and the outcome reaching the implementers through the ledger (S2). None of these is a change
  of the rule's text, and none changed a number. S1 and S3 depart from §8's text, so the user should be told.
- **Timestamps.** Every time in the log, the ledger and the draft is Amsterdam local time, without an offset. The commit
  times in the draft match `git log` in Amsterdam time.
- **Environment.** The real run wrote `.pyc` files into round 1's and round 3's read-only folders at 02:25:26 to 02:25:27,
  11 s before the run log's first line. So it was launched without `PYTHONDONTWRITEBYTECODE=1` (§10; N11). There was no
  effect on any number.

## 4. Findings

### Blocking

None.

### Should-fix

**S1. The draft misstates how independent phase 1 was** (report §8.3, and §4's paragraph "The independent
re-derivation"). The draft says phase 1 "used its own code, imported only the loaders and frozen components the rule
lists". It did not:

- `rederive/rd4_core.py:42-43` imports `rd3_core` and `rd3_family`, round 3's re-derivation. Phase 1 took from them the
  features, grouping scores, z-scores, the float32 combine, rank counts, per-anchor metrics, the bootstrap wrapper and
  the whole 224-cell family with σ* and both cross-fits (`rd4_phase1_report.md` §1).
- It also used round 2's re-derivation cache as a reference.

Rule §8 says the re-derivation "writes its own code for the features, P, T, … the 224 cells' integer statistics, σ*,
both cross-fits, the assembly, the per-anchor metrics", and lists what it may import. `rd3_*` is not on that list.
`rd4_core.py`'s docstring says "round 4's brief and rule §8 allow it". The rule does not, and no ruling in the ledger
records the permission. The code is independent of the implementation, so the agreement still means something, and my
third derivation (own code, none of rd3, rd4 or the implementation) closes the gap. The text must say what happened.

Replace §8.3's second sentence with:

> It reused round 3's independent re-derivation code by import (`rd3_core`, `rd3_family`: features, z-scores, the
> 224-cell family, σ*, both cross-fits, assembly and per-anchor metrics) and round 2's re-derivation cache as a
> reference, besides the loaders the rule lists. That code is independent of the implementation, but rule §8 asked for
> new code, and the reuse was not recorded as a ruling. The whole-branch final review closed the gap with a third
> derivation in its own code (§8.5), which reproduced every decision quantity exactly. Phase 1 found the kill on its
> own.

In §4, after "written by an agent that never read the implementation", add:

> (it reused round 3's re-derivation code for the fusion family; see §8.3)

Add a ledger line recording the deviation.

**S2. The implementation was not blind to the outcome, and the draft does not say so.** At 01:48 the ledger and the run
log recorded phase 1's candidate results: "V4 clears D10, Δ=−18; V2 fails D10 c1, Δ=−18; V24 fails c1,c2, Δ=−34; E
empty → kill". Task 3a (the seed-42 runner) was dispatched at 01:58, and its report says: "The Task 3 progress ledger
mentions the re-derivation's phase-1 outcome" (`task-3a-report.md:234`). Context.md forbade implementers from opening
candidate results, yet the shared ledger carried them. Exact agreement on 110 arrays cannot be tuned toward, so the
numbers stand (and the third derivation confirms them). Still, "two independent derivations agree" is weaker than the
draft implies. Add to §8.4 "Other notes":

> The controller recorded phase 1's development numbers and its kill in the ledger and the run log at 01:48, before
> the seed-42 runner was written (Task 3a, dispatched 01:58). Its implementer saw the outcome there (its report says
> so), so the implementation was not blind to the result it then reproduced. The agreement is exact on every array,
> which code cannot be steered toward, and the final review's own derivation reproduced it; but later rounds should
> keep candidate results out of the shared ledger until the implementation's run is done.

**S3. The Task 3 split departs from rule §8's order and §10's test list, and the draft presents it as a routine
ruling** (§8.4 bullet "Task 3 split"). Rule §8 step (1), "implementation with unit tests, including the tests of §10 that
round 3's final review asked for before reuse", comes before step (2). Several §10 tests were never written:

- the GO-pass assertions (a) to (d), each shown to fire under a mutation;
- (e) M27;
- T3-1 and T3-10;
- the wiring smoke test's leak check.

The controller decided this after it knew phase 1's kill. It changed no seed-42 number, because these tests guard
test-seed code that was never written or run. It is still a departure from the rule's stated order, made without the
user. Replace the bullet with:

> Task 3 split, a departure from rule §8 step (1): the seed-42 runner first; the test-seed runners and the §10 tests
> that guard them (GO-pass assertions (a) to (d), M27, T3-1, T3-10, the wiring smoke test) only if a candidate were
> carried. The controller ruled this at 01:48, after phase 1 had shown a kill. Nothing on seed 42 depends on these
> tests, and after the kill they guard nothing, but the rule put them before the regression checks; the user is told
> here. They must be written before any of this code runs on a test seed.

**S4. Overreach in the user-decision section** (§9 item 4). The draft says: "A veto on A0 can at most remove AFF's loss
against B on that pair". §6.4 says the opposite, that "an abstention that shut only the harmful episodes could in
principle do more", and my arrays agree: AFF minus B on style × genre is an average over episodes where steering helped
and episodes where it hurt. Replace the sentence with:

> A veto on A0 that shuts every gate on style × genre would score as B there: on seed 42, +0.132 pooled (+0.397 on that
> pair). A selective veto could in principle do more, but none of the label-free signals we have separates the
> harmful episodes (§6.4).

**S5. Test gap: no test pins the content of the D10 clauses** (`r4_stats.dev_record`, `test_r4_fusion.py`). Four
mutations survive every test:

- clause 1 strict (`>` 0.5);
- clause 2 non-strict (`>=` 0);
- clause 3 reading the bar margin's lower bound instead of the gain statistic's;
- the bar threshold set to 0.4.

`test_dev_record_matches_hand_numbers` recomputes the clauses with the same formulas on random data, where the variants
coincide. At run time the boundary stop would catch the first two (an exact-threshold value is within 1e-12), but not
the last two. The real run is confirmed by the third derivation, so there is no rerun. **Fix before `r4_stats` is
reused:** add a test that patches `C.point_ci` to return chosen values and asserts:

```python
cases = [((0.5, 0.1, 0.9), (1.0, 0.2, 2.0), {"c1": True, "c2": True, "c3": True}),   # point exactly 0.5 passes
         ((0.49, 0.1, 0.9), (1.0, 0.2, 2.0), {"c1": False, "c2": True, "c3": True}),  # 0.4 < point < 0.5 fails
         ((0.6, 0.0, 0.9), (1.0, 0.2, 2.0), {"c1": True, "c2": False, "c3": True}),   # lower bound exactly 0 fails
         ((0.6, 0.1, 0.9), (1.0, 0.0, 2.0), {"c1": True, "c2": True, "c3": False}),   # gain lower bound 0 fails, bar's > 0
         ((0.6, -0.1, 0.9), (1.0, 0.2, 2.0), {"c1": True, "c2": False, "c3": True})]  # c3 reads the gain, not the bar
```

Feed the first tuple as the bar margin and the second as the gain statistic. Points exactly at a threshold also raise
the boundary flag, which the test should assert.

**S6. The gates that produced the development numbers are never checked against item 5's verified gates, and the npz
stores the wrong copy** (`run_r4_seed42.py` `develop()` :508-520 and `arrays()` :523-530; deferred minor 3a-3,
confirmed). `develop()` rebuilds each candidate's gates inside `run_candidate` from `st["keep"]` and
`st["ra1"]["pick"]`. `seed42_arrays.npz` stores item 5's `st["G"]`. A mutation that builds a_v with another threshold in
`develop()` (A10) passes all 74 synthetic tests, and the npz would still show the verified gates. For round 4 the third
derivation shows that the stored per-anchor arrays, cells and gates all equal an independent computation from the
verified gates, so no rerun is needed. **Fix before reuse:** have `run_candidate` return the gates it used, assert in
`develop()` that

```python
all(np.array_equal(used[t][c], st["G"][k][t][c]) for t in range(4) for c in CONDITIONS)
```

holds, store those gates, and add a test with A10's mutation.

### Nits

- **N1** (§8.1): "It verified all 17 SHA-256s of D11". D11 had 17 rows when the checker ran. The committed rule has
  22, since the fixes added 5. Replace with: "all 17 SHA-256s D11 had at the time (the 5 rows added by the fixes are
  asserted by `r4_common`; the final review verified all 22)". I verified all 22.
- **N2** (§6.3): "The round-3 report and the brainstorm found the same on R1: … +0.01 and −0.26". Round 3's report has
  no such numbers. The brainstorm (§2.1, line 79) prints +0.01 and −0.25. My exact values at R1's cells are +0.0122 and
  −0.2563. Replace with: "The brainstorm found the same on R1 (its §2.1): R1's steering on the emotion pairs' condition
  b was level with its counterpart on emotion × style (+0.01) and lost on emotion × genre (−0.26; printed there as
  −0.25)."
- **N3** (§6.1): "its R@1 there equals B's in every case (1,490 episodes for V4, 981 for V2, 1,968 for V24)". Those
  counts are the changed episodes now shut in both conditions. The equality holds on every fully shut episode: 4,193,
  3,684 and 4,671, and AFF's own 2,703. Also, "(1 + λ_u)·z(B) ranks exactly as B" is a float32 claim. Replace with:
  "its fused score is (1 + λ_u)·z(B), and its R@1 equals B's on every such episode (4,193 for V4, 3,684 for V2, 4,671
  for V24, among them the 1,490, 981 and 1,968 that a veto changed)".
- **N4** (§6.1, "Second, within the 224 cells the thinner gates did not move the cross-fit's optimum"): add the margins
  from §2(b) of this review: "(by 1 to 7 integer units over the runner-up; V4's half-0 choice by 1 unit, and with the
  runner-up V4's Δ would have been −25)".
- **N5** (Summary, "V4's signal barely separated style × genre from the emotion pairs. It closed 17% of AFF's open
  values off the emotion side and 12% on it"). The second sentence compares sides, not pairs. Replace it with: "As
  scored it closed 19.6% of AFF's open values on style × genre and 12.2% on the two emotion pairs."
- **N6** (§6.8, Table 7): the comparison is at τ_2 only, but AFF and V4 are scored at τ_0 on the parity-1 half. Add a
  row or sentence: "As scored, AFF's gate is open on 8,623 and 2,448 values and V4 shuts 1,268 and 326 of them, so
  condition a is again most of it (80%)".
- **N7** (§6.4): "so v did pick the weaker emotion episodes, but steering still paid on them". On emotion × style the
  interval for the closed episodes includes 0 (+0.490 [−0.594, +1.538]). Replace with: "but on average steering still
  paid on them (the interval excludes 0 on emotion × genre only)".
- **N8** (§4, §8.3): of the 932 agreed quantities, the re-derivation formed 137 scalars and 13 arrays from the
  implementation's own arrays (`rd4_phase1_report.md` §11). Add: "(150 of them derived, as its report labels)".
- **N9** (§5 sources line and the `build_figures.py` docstring): "re-derives every decision number of Table 3". The
  chosen cells and σ* are read from `seed42_arrays.npz` and checked against `dev_seed42.json`; they cannot be derived
  from per-anchor arrays. Replace with: "re-derives every decision number of Table 3 except the chosen cells and σ*,
  which it checks between `seed42_arrays.npz` and `dev_seed42.json`".
- **N10** (terms):
  - "R-c" (§1, the R1 row), "told grouping" (§6.3) and "CSD" (never expanded) are undefined.
  - B is called "the best condition-free score of the project" (Summary, §1). B′(A0) and B′(A1) both beat it on seed
    42, and §3.2 calls B′(A1) the strongest. Suggested wording for B: "the project's standard condition-free score".
  - For "told grouping" (§6.3), drop the term: "the supports share an emotion, the condition the affect grouping is
    meant to detect".
  - Summary: "B′(A0), the strongest condition-free comparator" → "the strongest of round 3's condition-free
    comparators".
- **N11** (§8.4 or §7): the real run was launched without `PYTHONDONTWRITEBYTECODE=1` (rule §10). `.pyc` files dated
  02:25:26 to 02:25:27 sit in `20261117_reader_fix_csd/__pycache__/` (common, rb_build, rb_eval, rb_features) and
  `20261121_round3_affect_gate/__pycache__/` (r3_*), read-only folders. They are gitignored and changed no number. Note
  it, and use the full environment line on the next launch.
- **N12** (§8.4): the `--dry` ruling ran a seed-42 dry run that computed every candidate result, unseen and deleted,
  before the real run. The rule defines smoke runs only on smoke seeds. Mention it in one line.
- **N13** (tests, before reuse): three more mutations survive, each failing safe at run time:
  - G2: `Guard.item_passed` accepts any order. The test passes for another reason, and `release` still needs [1..5].
  - G5: items 4 and 5 swapped in `ITEMS`. The order tests stub the items, and at run time `require_gates` stops it.
  - A5: no "closed where AFF is closed" assertion. It is redundant with construction, and item 5 checks it on seed 42.

## 5. Deferred-minor triage

None needs fixing for round 4's result: the third derivation independently confirms every decision quantity. "Before
reuse" means before any of this code runs again, in round 5 or on a test seed.

| Minor (ledger) | Triage |
|---|---|
| T0: r4_common checks only the r3_* module paths, not round 3's own imports | Low. Round 3's `assert_inputs` hashes common.py and rc_core.py; leave |
| T0: (52, 53, 54) written twice in r4_common | Leave; `test_seed_guard` catches drift (mutation Q1 killed) |
| T2: the AFF-check guard compares by identity only | Before reuse on test seeds, with the §6.4(b) recomputation (S3's tests) |
| T2: two near-vacuous `or` assertions | Before reuse |
| T1: D11 hashes of r3_bundle/r3_common asserted only inside extend_a1 | Harmless here: `run_r4_seed42.check_inputs()` asserted them before item 1. Move before reuse |
| T1: new fields set before validate_a1 | Leave (a failure raises) |
| T1: nested-object fingerprint by identity | Leave (docstring) |
| T1: decimal regex in test_r4_bundle misses ".5", "5e-03" | Leave; the runners' LEAK regex covers both |
| T1: dry_check_42 prints no FAIL marker on a raise | Leave |
| T3a-1: `--boundary-reported` resume skips `check_inputs()` | Path unused (no boundary). Before reuse |
| T3a-2: guard is opt-in; order test stubs items 1 to 4 | Confirmed by survivor G5, which fails safe at run time. Before reuse |
| T3a-3: develop()'s gates not checked against item 5's | **Confirmed by survivor A10. S6, fix before reuse** |
| T3a-4: boundary resume with a carried candidate untested | Before reuse |
| T3a-5: KILL line should say "after the phase-1 agreement" | Cosmetic; the log handled it. Fix before reuse |

## 6. Mutation table

Each mutation ran on a scratch copy of the round-4 modules, the three synthetic test files and the rule, in
`final_review/mut/<id>/` (deleted afterwards). Only r4_common's `ROOT` was patched. The tests were `test_r4_common.py`,
`test_r4_fusion.py` and `test_r4_runners.py` (74 tests; the unmutated copy passes in 13 s). Logs are in
`out/mutations/`, the summary in `out/mutations.json`. **31 of 39 mutations killed; 8 survive.**

| Guard | Mutation | Result (first failing test) |
|---|---|---|
| §5 order guard | G1 `Guard.require` never refuses | killed (`test_guard_refuses_every_candidate_output_before_release`) |
| | G2 `item_passed` accepts any order | **survives** (fails safe: `release` needs [1..5]) |
| | G3 `release` without the written record | killed (`test_guard_items_must_pass_in_order_…`) |
| | G4 item-5 gates before items 1 to 4 | killed (`test_guard_refuses_every_candidate_output_…`) |
| | G5 items 4 and 5 swapped in `ITEMS` | **survives** (fails safe at run time) |
| Gate algebra | A1 V2 factor `!= affect` | killed (`test_candidate_gate_algebra`) |
| | A2 V24 without a_v; A3 V24 without the A1 pick | killed (`test_candidate_gate_algebra`) |
| | A4 a_v = 1[v ≤ v₇₅] | killed (`test_abstain_is_strict_less_than_float32`) |
| | A5 no "closed where AFF closed" assertion | **survives** (redundant by construction) |
| | A6 item 5 checks V2 against its own picks; A7 V4 against its own a_v | killed (`test_item5_fails_when_…`) |
| | A8 `develop` on R1's gates; A9 a_v = 1; A11 A0 picks as A1 picks | killed (`test_a_passing_run_writes_the_regression_record_first`) |
| | A10 `develop` builds a_v with 0.9·v₇₅ | **survives** (S6) |
| Candidate counterpart | C1 `run_candidate` runs AFF's gates | killed (`test_candidate_counterpart_is_built_from_the_candidates_own_gates`) |
| | C2 each candidate's record from AFF's family | killed (`test_a_passing_run_…`) |
| Integer carry, tie band | K1 band exclusive; K2 Δ ≥ 0 admitted; K3 band 25; K4 last tied carried; K5 E ignores D10 | killed (`test_carry_rules`) |
| | K6 Δ from float means; K7 Δ against the counterpart | killed (`test_delta_int_non_multiple_…`, `test_dev_record_matches_hand_numbers`) |
| Bar comparator | B1 V2 order B′(A0) first; B2 ties to the last; B3 V2 without B′(A1) | killed (`test_comparators_order_per_candidate`, `test_bar_comparator_four_way_…`) |
| D10 clauses | D1 c1 strict; D2 c2 `>=`; D3 c3 reads the bar; D4 bar 0.4 | **all survive** (S5) |
| Boundary stop | S1 ε = 0; S2 Δ = 0 not flagged; S3 gap 24 not flagged | killed (`test_boundaries_are_flagged`) |
| | S4 no stop; S5 resume with any SHA; S6 resume with changed files | killed (`test_a_boundary_stops_…`, `test_the_continuation_refuses_…`) |
| Seed guard | Q1 `R3.TEST_SEEDS` not redirected | killed (`test_seed_guard`) |

## 7. Files written (all under `src/test/20261122_round4_aff_vetoes/final_review/`)

| File | Size | What |
|---|---|---|
| `final_review.md` | 29 KB | the review |
| `fr_derive.py` | 18 KB | third derivation (own code) |
| `fr_compare.py` | 8 KB | agreement with the implementation's files and the rule's targets |
| `fr_descriptive.py` | 12 KB | §6 and Summary numbers from my arrays, compared with `figure_data.json` |
| `fr_extra.py` | 7 KB | R1's per pair-condition margins; cross-fit runner-ups and Δ under them |
| `fr_mutate.py` | 12 KB | mutation harness (scratch copies, deleted after each run) |
| `out/fr_results.json`, `out/fr_arrays.npz` | 14 KB, 12.0 MB | derivation results (SHA-256 8c2d7cd7…, a9cf66bd…) |
| `out/fr_agreement.json`, `out/fr_descriptive.json`, `out/fr_extra.json` | 33 KB, 14 KB, 1 KB | comparisons |
| `out/fr_derive.log`, `out/fr_extra.log` | 3 KB, 2 KB | run logs |
| `out/mutations.json`, `out/mutations_run.log`, `out/mutations/*.log` | 11 KB, 9 KB, 40 logs | mutation results |
| `bf/build_figures.py`, `bf/figure_data.json`, `bf/bf.log` | 39 KB, 34 KB, 1 KB | `build_figures.py` scratch run (ROOT patched; regenerated PNGs deleted) |

The total is about 12 MB, and nothing is over 1 GB. `*.json`, `*.npz` and `*.log` are gitignored by this folder's
`.gitignore`; the `.py` and `.md` files are not.
