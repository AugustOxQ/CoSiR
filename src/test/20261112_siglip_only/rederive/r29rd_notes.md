# Independent SC1 derivation handoff

The full SC1 result is pending the controller's CPU queue and K2 checkpoints. No final `sc1_rederived.json` has been written: sample estimates must not stand in for the four full groups or either pick.

Run after both arms' six seed-0 weight files and sidecars exist:

```bash
bash /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_run_sc1.sh
```

The script writes `sc1_rederived.json` with precisely the requested fields, `sc1_rederived.run.json` with CPU runtime and input provenance, and `r29rd_full.log`. All outputs stay here. No project modules are imported. It fails before scoring if a required weight/sidecar is absent. Default arguments score all queries against complete galleries, with no truncation, centering or expansion.

## Verified

- Eight scalar/synthetic tests pass, including exact score ties, own-row removal, unlabelled gallery masks, no positives, exact decision ties, joins and an analytic GELU forward pass. Test-body runtime: 0.022 seconds.
- Validation reconstructed from original CLIP extraction sample IDs and painting groups; sorted painting-name SHA256 matches the frozen split. Counts: 6,152 paintings, 30,872 captions. Emotion plurality ties: 1,577.
- Real K1 checkpoints and both P1 forms tested on 256 deterministic random queries per direction, full validation galleries. Total process runtime: 12.461603269912302 seconds. K1: 5.8473369516432285 seconds; P1 plain: 1.7263868539594114 seconds; P1 prefixed: 1.7618121481500566 seconds. Setup accounts for the remaining time. K2 unavailable.
- This sample suggests a full computation above four minutes (roughly 10–20 minutes by simple query scaling; unmeasured). The full computation was not launched.

## Groups from the subset test only

| Arm | 256-query subset group |
|---|---:|
| siglip1 | 0.25457869402515904 |
| siglip1_prefix | unavailable: checkpoints missing |
| p1_siglip1 | 0.1786116576477776 |
| p1_siglip1_prefix | 0.17768834527678193 |

K*, K* form and P1 form remain undecided. All decision fields are null in `r29rd_smoke_sc1.json`, which is explicitly a partial validation artifact.

## Query counts, identical across arms

| Cell | Actually scored in subset | Eligible/scored in full run |
|---|---:|---:|
| emotion/t2i | 256 | 30872 |
| emotion/i2t | 256 | 6152 |
| emotion/t2t | 256 | 30872 |
| style/t2i | 256 | 30872 |
| style/i2t | 256 | 6152 |
| style/t2t | 256 | 30872 |
| genre/t2i | 215 | 24886 |
| genre/i2t | 202 | 4960 |
| genre/t2t | 215 | 24886 |

Full-run counts are derived exactly from label eligibility, not estimated from this subset. Every query with a label has positives here; full scorer asserts that actual finite AP counts equal these counts.

## Implementation choices and protocol clarifications

1. Image gallery order is ascending painting name; caption gallery order is ascending extraction sample ID. Exact cosine ties keep gallery order, following r22/r23. Subset queries use sorted `default_rng(0).choice(..., replace=False)` positions; galleries remain complete. Subset selection never affects the full command.
2. Painting emotion plurality ties choose the alphabetically earliest emotion (frozen r22 convention). All nine classes, including `something else`, remain labels. Caption emotion is its own annotation's emotion. Style must be constant per painting. Genre CSV conflicts/missing values are unlabelled (-1).
3. Float16 stored inputs become float32 and receive one L2 normalisation. Towers use float32 exact GELU with dropout inactive, followed by output L2; cosines are float32 CPU products. AP arithmetic, cell means and the equally weighted nine-cell mean are float64 without an intermediate float32 AP cast. The permitted r23 independent module stores AP as float32; this derivation retains double precision, following the mathematical AP rule and requested float64 means.
4. Unlabelled gallery rows participate in ranking and are masked from AP denominators only. Unlabelled query APs and APs with zero relevant rows are nonfinite and excluded. Own-row exclusion is the identical caption ID in t2t. Cross-modal paired paintings/captions remain eligible.
5. The specified prefixed store's manifest records lowercase `represent the caption: ` while the spec writes `Represent the caption: `. Use those stored vectors as requested, without re-encoding or altering inputs. Checkpoint caption-store hashes are checked against the actual store indexes.
6. `p1_form` will record the higher of the two full P1 groups (ties plain), regardless of K*. Its downstream use is conditional on K2 winning, per D20. K* ties go K1; K* form follows K*.
7. Batch size 128 for ranking and 1,024 for CPU forward passes limits intermediate memory; eight OMP/MKL/OpenBLAS/Torch threads, one Torch interop thread. No GPU initialization or usage.

The source reads permitted specs, data, architectural/input modules and the permitted r23 independent module only; none of the forbidden scoring code, numerical records or round logs was read/imported. No commits, deletions or writes outside this directory.

Cleanup candidates after the controller finishes: `task_plan.md`, `findings.md`, `progress.md`, `r29rd_tests.log`, `r29rd_inspection.log`, `r29rd_inspection.json`, `r29rd_smoke.log`, `r29rd_smoke_sc1.json`, `r29rd_smoke_sc1.run.json`. Nothing was deleted. Keep the source, tests, runner, notes and eventual final outputs for audit.
