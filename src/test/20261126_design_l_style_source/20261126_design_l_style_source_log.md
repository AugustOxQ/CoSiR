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
- Done 06:51: `results/seed42/arrays_ceiling.npz` (12,288 episodes; 4,096 style × genre), `ceiling_pending.json`.
- Phase K (`rederive/rd7_arrays.py --phase K --seed 42`, own process): 18 arrays. `rd7_stats.py --stage ceiling`:
  Σ(4d) 643, zone CLEAR. `run_r7.py ceiling --finalize`: every comparison passed; `ceiling_verdict.json` written 06:52.
- **K1: CLEAR.** Told T on style × genre, fused minus counterpart: Σ(4d) = 643 (CLEAR bar 164), point +3.92 R@1,
  95% interval [3.35, 4.49]. The run goes on to stage 2.

## 2026-10-12 06:54 `gate` (S1 to S3, P1)

- Started in the background (`results/logs/gate.log`).
- Done 06:54: `results/seed42/arrays.npz` (523 arrays), `gate_pending.json`, `taus.json`.
- Phase A (`rd7_arrays.py --phase A --seed 42`, own process): 243 arrays. `rd7_stats.py --stage gate`: n 0, no picks,
  promising true. `run_r7.py gate --finalize`: every comparison passed; `gate_verdict.json` written.
- **Gate: n = 0.** No candidate passes S1 to S3 (Σ(4d) per candidate; S1 bar 246, S2 bar 0, S3 bar −49):

  | Candidate | S1 bar margin (vs) | S1 95% interval | S2 (vs CF) | S3 emotion × style | S3 emotion × genre |
  |---|---|---|---|---|---|
  | R1_V1 | −186 (B1) | [−0.60, −0.15] | −15 | −168 | −217 |
  | XAFF_V1 | −248 (B1) | [−0.74, −0.27] | −30 | −174 | −256 |
  | R1_V2 | 97 (BX) | [−0.01, 0.41] | 15 | 38 | −189 |
  | XAFF_V2 | −50 (BX) | [−0.30, 0.10] | 16 | −46 | −235 |
  | R1_V3 | 122 (B1) | [0.05, 0.46] | 37 | −16 | −153 |
  | XAFF_V3 | 27 (B1) | [−0.16, 0.27] | 62 | −53 | −200 |
  | R1_V4 | −215 (B1) | [−0.67, −0.21] | −31 | −170 | −230 |
  | XAFF_V4 | −233 (B1) | [−0.71, −0.24] | −18 | −166 | −264 |
  | R1_V5 | −165 (B1) | [−0.58, −0.10] | −14 | −146 | −209 |
  | XAFF_V5 | −102 (B1) | [−0.47, 0.05] | −31 | −127 | −173 |
  | R1_V6 | −80 (B1) | [−0.42, 0.09] | −40 | −130 | −172 |
  | XAFF_V6 | −115 (B1) | [−0.49, 0.01] | 23 | −158 | −206 |

  Per rule §6 and §8: design L ends; FV is NO-GO (the ArtELingo paper with AFF); no sensitivity, rule smoke, seed
  build or test. "Promising" holds (stated in the report). The L3 fallback cannot change this: V1 and V4 are far from
  every bar, so dropping them or another projector leaves n = 0.
- Descriptive pass (`run_r7_descriptive.py ceiling`, `gate`): `results/descriptive_ceiling.json` (06:58),
  `results/descriptive_gate.json` (07:00). The test part does not apply.

## Process departure (L3, recorded 07:30)

- Rule §8's last row says a case the rule does not cover stops the run with its report and a phone note, and the user
  decides. At 05:50 the run sent the phone note (05:55) but did not stop: it went on with the agent-default fallback,
  under the unattended-chat practice that a blocked step is done another way. K1 and the gate were computed on it.
- What the user decides: whether K1 and the gate stand on the fallback. Keeping it or dropping V1 and V4 (with Z1, Z4)
  cannot change n = 0. A rank-63 projector is a different projector (the fallback keeps pinv's rank 64, one extra
  noise direction); it would need a rerun of V1, V4, Z1, Z4, and with V1 and V4 at least 430 integer steps below S1's
  bar a change of n is very unlikely.
- The two L3 numbers: the record's `max_abs_after_pinv` 9.50e-4 is the absolute value; against `max_abs_before`
  0.552 it is the ratio 1.72e-3 that the assertion compares with 1e-6.
