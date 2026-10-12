# Round 7 run log: design L from the style source

The run chat's record (rule §9 order). The build's record is `.scratch/r7-design-l/notes/build_log.md` (outside git);
stage 0 passed at 2026-10-12 04:42 in the build. Run logs: `results/logs/` (gitignored, with the results).

## 2026-10-12 05:50 `prep`, first try: stopped on L3's assertion

- `run_r7.py prep` stopped after about a minute: `AssertionError: L3 Orth projection assertion` in `orth` (OrthCSD, for
  V1 and V4). No grouping file was written.
- Diagnosis (scratch script on the same inputs): M (768 × 64, content means of the centred CSD painting vectors) has
  rank 63 in exact arithmetic, since the size-weighted means sum to zero. Its 64th singular value is 7.97e-15 of the
  first, just above pinv's default cutoff (1e-15), so pinv keeps it and the 1/σ round trip amplifies rounding.

  | Vectors | σ64 / σ1 | pinv rank | Literal form: after / before | SVD basis U_k U_kᵀ: after / before |
  |---|---|---|---|---|
  | CSD (V1, V4) | 7.97e-15 | 64 | 1.72e-3 (fails; bar 1e-6) | 1.39e-15 |
  | U (V6) | 5.72e-17 | 63 | 4.85e-15 (passes) | 1.98e-15 |

- Rule status: the code matches L3 word for word; L3's recipe fails its own assertion on CSD. The rule does not
  cover this (§8 last row: the user decides; phone note sent 05:55).
- Agent default, to keep the night's run going: `orth` keeps the literal pinv form wherever it passes (V6 unchanged),
  and only when it fails uses the same projector M·pinv(M) = U_k U_kᵀ at pinv's own cutoff (k = 64 here), from the
  SVD's orthonormal basis. The grouping record stores `projector`, `basis_rank` and `max_abs_after_pinv`. Commit
  e3776ca. The alternatives the user may prefer: drop V1 and V4 (with Z1, Z4), or a rank-63 projector.
- Where the choice matters: only if V1 or V4 becomes a pick at the gate. Seed-42 numbers can be redone with `_fix`
  files; the fresh seeds cannot. So if V1 or V4 is a pick and the user has not answered, the run holds before
  `build` (§7.2) and says so.

## 2026-10-12 05:55 `prep`, second try

- Started in the background (`results/logs/prep_2.log`).
- Done 06:50, exit 0: 15 grouping records, nothing dropped (`results/prep.json`). OrthCSD used the fallback
  (`projector` svd_basis_fallback, basis rank 64, after 7.65e-16 against before 0.552; the literal form gave 9.50e-4).
  OrthU used the literal pinv form (rank 63). Group counts: V1 18, V2 12, V3 13, V4 18, V5 13, V6 16, R0 17, T 27;
  each Zk has Vk's sizes. NrKmeans (N) ran within its 30-minute limit and was kept. All 13 new readers written.
  Prep took under an hour (the spec budgeted about 4).

## 2026-10-12 06:52 `ceiling` (K1)

- Started in the background (`results/logs/ceiling.log`).
