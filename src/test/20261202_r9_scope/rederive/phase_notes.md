# C13 readings and validation

Single agent, ticket 25 only. No prohibited source was opened or imported. The three decision rules, the ticket,
index and code-map sections 1.6–1.10 and 3 were read. R6's independent `rd_common.py`, `rd_seed42.py`, `rd_heads.py`,
`rd_held_scores.py`, `rd_stats.py`, `rd_held_stats.py` and phase notes supplied permitted independent formulas.
`src/data` loaders were inspected. No earlier-round file was changed.

## Readings where the ticket leaves an interface open

- The specific `rd9_` naming instruction takes precedence over the general round-module/test naming and the
  ticket's `rd_compare.py` spelling: the comparator is `rd9_compare.py`, tests are `rd9_test_arithmetic.py`.
  The requested `inputs.md` and `phase_notes.md` retain their explicit names.
- The rules explicitly allow frozen library primitives outside the prohibited chain. Imports are limited to data
  loaders; episode builder/index/validator/hash; checkpoint load/encoding; `centered_term`, `uniform_probe_scores`,
  RCA/PCA; `zscore_rows`, `fused_scores`, and the reference `cluster_bootstrap` interval identity assertion. The
  writer did not inspect these library definitions. Features, reader probabilities, margins, gates, baseline
  cross-fits, 224-cell statistics, sigma, cell picks, assembly, metrics, bootstrap draws/counts, Holm, two levels,
  sensitivity and comparison are independent code. Some formulas are copied from the permitted R6 independent code.
- Arm-L `draw_rows_sha256` hashes sorted drawn IDs; RCA `fit_rows_sha256` hashes draw order. Fitting retains
  the generator's draw order; its coefficients match R6's independent coefficient digests. This metadata reading
  was checked against pinned data, with a failing test before adding the sorting helper.
- A3 codes are the CPU checkpoint encoder's raw output, passed to the rule-allowed `centered_term(uniform=True)`.
  This reading is confirmed by exact CLIP baseline picks and per-anchor/point agreement.
- New pins' schema is not defined in the index. `inputs.md` documents the explicit accepted file-pin shapes and
  exact-name role mapping. Earlier rounds use MAIN; round-9 paths use the running checkout. CLIP's absent pins file
  is bridged by R6's independent digest constants. Source digests stay metadata because opening those files would
  break the ticket's independence restriction.
- Values JSON: each aspect is a list of `{code, name}` records (ticket 14's file) or a plain code list (smoke, tests,
  H8 standard mode). No label
  name is re-encoded. Canonical R9 language indices survive set-asides; Hindi stays index 5, for example.
- A28 part hashes are computed **after mapping back to the full table's row positions**, before concatenation.
  Largest-remainder arithmetic uses integers throughout; ties retain canonical language order. Zero-share languages
  require no builder call (the actual 4,096-episode shares are all positive). The labels-only smoke may allocate fewer.
- Only our specific values-subset assertion and the primitive's exact failed-draw RuntimeError (>10× allocated
  episodes) trigger set-asides. Validation and unrelated assertions/errors propagate. After any failures, every
  remaining language is rebuilt with its unchanged canonical seed and the newly allocated share.
- H8–H10 share the same episodes. H9/H10 assert equality against the independent H8 record as well as being compared
  to the controller's per-read hashes. Clusters use global painting identity across languages and seeds.
- Level 2 has no Holm order, Holm interval or pass flag after a level-1 failure. Its points/95% intervals and raw
  bootstrap counts remain in the independent audit record; those counts are not used in a test or claim.
- Outputs are independent audit records, not official verdicts or ledger changes. Preread authorization/order belongs
  to the controller's official runner. The scripts refuse switched-off reads and refuse existing output names.
  Raw/cache outputs live in ignored `rd9_out/`; the two committed comparison records preserve this job's evidence.

## Validation completed in this worktree

Arithmetic tests were written first, observed failing, then implemented: shares; all seven/five/two-check integer
boundaries; Holm stopping/ties; serial claim gatekeeping; shared-cluster integer counts including zero; quarter-hit
metrics and strict ties; min-margin/max-R@1 cell ties; agreement tolerances and discrete identities. Additional
boundary tests cover the set-aside error classification and main-checkout/worktree pin paths. A final file-boundary smoke found H5 stores concatenated episode arrays,
not pair-prefixed fields; the adapter was corrected after a failing schema test, and all nine stored hashes pass.

`pytest -q rederive/rd9_test_arithmetic.py`: **17 passed** (use the exact full command in `rd9_commands.md`).

The small synthetic A28 table exercised all three smoke seeds, three pairs, kept/drop masks, deterministic hashes
and language membership. A separate synthetic label-only check exercised both permitted set-aside causes and the
second-pass shares (12 episodes reallocated 6/6), preserving Hindi's seed 9001005. No real A28 table was opened.

CLIP phase-1 prepare and score and phase-2 **selection** replay were each run with a 235-second timeout, CPU only.
All completed. Phase-1 agreement: **143 comparisons, 0 differences**. Phase-2 frozen selection replay:
**129 comparisons, 0 differences**. The latter's 30 scorer/metric arrays match phase 1; all P1–P7 points, counts,
95% and Holm bounds match R6 `seed42_pass_counts.json`. These are 272 comparison items, not an official read.

Derived picks: AFF 39/119, CF 149/10; B (16,8)/(8,2); B0 (16,16)/(16,8); sigma 0/0; RCA lambda 0.5/0.5;
not switched off. Taus equal the four R3 D6 literals exactly. P1–P7 counts are all zero. No boundary case.

Raw outputs remain in this worktree's ignored `rd9_out/`. The pins adapter's read-key union, record identity checks
and worktree path resolution were tightened after these runs; scoring arithmetic did not change. The arithmetic
suite covers those changes. No H6–H10 real read was performed. The real A28 table/values and new-backbone pins were
unavailable here; their commands are handed to the controller. The accepted pins schema needs matching at integration
if tickets 13/16 choose a different serialization.


## Ticket 25 review fix round

1. Added `--phase1 --backbone B` for every backbone, reading file-pinned cells, B/B0 picks, RCA lambda and tau after
   independent results exist. It compares switched-off state instead of relying on the phase-2 refusal. It also checks
   all five means, P1–P7 points/counts, and every pinned array digest/dtype/shape, with mandatory primary arrays.
   The accepted pins seed-42 schema is documented in `inputs.md`. B1 is now scored in phase 1 from pinned CSD
   posteriors; no B1 score was added to phase 2. A tau disagreement is recorded by the comparer instead of causing
   phase-1 scoring to stop before an agreement record can be produced.
2. Committed the actual synthetic set-aside/re-split case in `rd9_test_a28.py`. Tests run real frozen episode
   primitives on a 750-row table, with explicit expected language seeds and post-drop shares. Added both exclusion
   causes, the third exclusion stop, pre-drop indices/lacks, subset/minimum guards, language-order assembly and hashes.
   Arithmetic tests use nonzero integer bootstrap counts (P5=133, P7=3575); zero-sum draws matter, and the Holm
   interval differs from the 95% interval. Natural ranking fixtures exercise RCA 32/64 extension, B/sigma ties,
   inclusive affect gates, opposite parity, both-halves switching, and percentile thresholds.
3. Real A28 smoke CLI defaults to 4,096 per pair, matching ticket 17; 64 remains available explicitly.
4. Recipe-role resolution prefers exactly one `results/recipe/<backbone>/` entry; ambiguity otherwise asserts.
   Its test includes ambiguous names and the backbone-specific preference.
5. Phase-1 prepare, phase-1 score and phase-2 output records include repository-relative SHA-256s of the loaded
   `src` library modules (`src/test` excluded). The path test uses independent temporary main/checkout roots.

Validation: **36 passed**, using the two test files and `pytest -q` command in `rd9_commands.md`. Every new test
has a discriminating mutation listed below; parameterized cases run together against that mutation. This is our
explicit 42-mutation inventory, rather than a claim to have rerun the reviewer's unavailable 35-mutation script.
**42 killed, 0 survived, 0 infrastructure errors.** Each mutation was applied alone, its selected pytest case run,
and the source restored in `finally`. Invalid syntax/infrastructure exit statuses do not count as kills. The
executable definitions and test selectors are committed in `rd9_mutations.py`; the JSON execution evidence is
`rd9_review_mutations.json`. This mutation run completed within the four-minute command limit.

CLIP scoring reran on the existing cache with `--output-suffix _review`, preserving the original results.
The 143 legacy comparison items still agree; B1's mean is exactly 18.804931640625; the new score output records
11 imported library source hashes. A separate use of the generic comparer against earlier pinned data checks
146 items, zero differences, including all five means and 35 metric array hashes. Expected B/B0/B1 arrays come
from the pinned R3 AFF array minus R6's pinned per-episode comparator differences; expected condition-free gain,
swap and strict are zero. RCA's five arrays come from pinned `per_anchor_seed42.npz`.

That generic-validation reference initially used float32 `mean()` on the pinned gate arrays. It disagreed by
9.93e-9 with the independent float64 gate share. The reference was corrected to integer open counts divided by
integer total; scoring code was unchanged. The first JSON is retained beside `_fix1`, which records the count,
denominator, source hash and cause. Neither reference-validation JSON replaces the controller's real file-pins
agreement; the three-backbone synthetic tests cover the new interface until those pins exist here.

| Mutation | Module | Result |
|---|---|---|
| P5 uses CF | `rd9_stats.py` | killed |
| P7 uses RCA gain | `rd9_stats.py` | killed |
| zero integer sums count | `rd9_stats.py` | killed |
| Holm interval quantiles | `rd9_stats.py` | killed |
| bootstrap seed 42 | `rd9_stats.py` | killed |
| switched off needs both | `rd9_scores.py` | killed |
| sigma ascending tie order | `rd9_scores.py` | killed |
| sigma first argmax | `rd9_scores.py` | killed |
| B first tied cell | `rd9_scores.py` | killed |
| gate inclusive margin | `rd9_scores.py` | killed |
| gate affect restriction | `rd9_scores.py` | killed |
| opposite parity | `rd9_scores.py` | killed |
| tau 75th percentile | `rd9_scores.py` | killed |
| RCA extension trigger | `rd9_scores.py` | killed |
| RCA extends to 32 | `rd9_scores.py` | killed |
| RCA extends to 64 | `rd9_scores.py` | killed |
| RCA extension first tie | `rd9_scores.py` | killed |
| language seed multiplier | `rd9_episodes.py` | killed |
| canonical language index | `rd9_episodes.py` | killed |
| post-drop shares | `rd9_episodes.py` | killed |
| kept rows only | `rd9_episodes.py` | killed |
| pre-drop value index | `rd9_episodes.py` | killed |
| global row mapping | `rd9_episodes.py` | killed |
| language-order assembly | `rd9_episodes.py` | killed |
| values subset guard | `rd9_episodes.py` | killed |
| minimum passed to primitive | `rd9_episodes.py` | killed |
| A28 minimum is one | `rd9_episodes.py` | killed |
| subset set-aside catch | `rd9_episodes.py` | killed |
| failed-draw catch | `rd9_episodes.py` | killed |
| re-split remaining languages | `rd9_episodes.py` | killed |
| record set-asides | `rd9_episodes.py` | killed |
| third set-aside stops | `rd9_episodes.py` | killed |
| hash within-read uniqueness | `rd9_phase2.py` | killed |
| hash prior collision | `rd9_phase2.py` | killed |
| phase1 switched-off comparison | `rd9_compare.py` | killed |
| phase1 n_j comparison | `rd9_compare.py` | killed |
| phase1 array hash comparison | `rd9_compare.py` | killed |
| phase1 B1 mean comparison | `rd9_compare.py` | killed |
| pin-name ambiguity | `rd9_inputs.py` | killed |
| backbone recipe preference | `rd9_inputs.py` | killed |
| round9 paths use checkout | `rd9_inputs.py` | killed |
| library source hash | `rd9_inputs.py` | killed |
