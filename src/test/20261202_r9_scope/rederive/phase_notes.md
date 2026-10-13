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
- Values JSON accepts plain code arrays or per-aspect `{codes, names}`, optionally wrapped in `values`. No label
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
