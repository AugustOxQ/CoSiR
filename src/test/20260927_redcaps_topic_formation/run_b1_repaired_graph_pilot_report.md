# B1 re-run with graph repair before Leiden

Generated 2026-09-27 14:30:50; seed 42; reuses B1's saved split (`b1_redcaps_single_teacher_pilot_split.npz`), models, and training loop unmodified. Companion to [`docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md`](../../../docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md), [`b1_redcaps_single_teacher_pilot_report.md`](b1_redcaps_single_teacher_pilot_report.md), and [`run_leiden_resolution_sweep_pilot_report.md`](run_leiden_resolution_sweep_pilot_report.md).

The only change from B1: `ensure_min_degree` is applied to each graph before its Leiden partition (previously only used for B1's sanity-check lift table, never before Leiden). For the raw teacher graph this uses the real image/text modality arrays; for each student's own embedding-space graph, the student's single embedding array is passed as both modality arguments (a minimal generalization: with one view twice, ensure_min_degree degenerates to a top-1 nearest-neighbor repair in that one space).

## Results

| Method | Isolated nodes repaired | Communities (before -> after) | Below-1% (before -> after) | Validation graph lift | Community-level lift (before -> after) |
|---|---:|---:|---:|---:|---:|
| Graph-only baseline | 1354 | 1423 -> 71 | 1404 -> 50 | 27.114× | 4.773× -> 4.775× |
| Attention student | 279 | 326 -> 46 | 304 -> 23 | 18.391× | 6.720× -> 6.577× |
| Mean-pool control | 497 | 555 -> 56 | 529 -> 31 | 18.973× | 6.865× -> 6.195× |

Occupancy min/median/max per method (post-repair): Graph-only baseline: 2/2.0/21672; Attention student: 2/1231.0/11252; Mean-pool control: 2/815.5/12306.

## Verdict

Repair reduces occupancy collapse substantially across every method (see table). Attention student below-1% share: 304/326 (B1) -> 23/46 (repaired). Occupancy improves but the trained students' embedding-space graphs remain majority below-1% even after repair -- the fix is necessary but not sufficient for the trained representations specifically. Attention still does not clearly beat the graph-only baseline or mean-pooling on validation embedding-graph lift under repair. This B1 follow-up's original conclusion (do not proceed to B2/scale-up) stands: repair fixes occupancy, not the core finding that the trained students do not outperform raw CLIP features on lift. Recommended framing for any future writeup: on RedCaps as tested, the isolated-node repair is a necessary graph-hygiene fix (already implemented as `ensure_min_degree` in this codebase) that should be applied before Leiden by default, but it does not by itself resolve the negative training-value finding from B1.
