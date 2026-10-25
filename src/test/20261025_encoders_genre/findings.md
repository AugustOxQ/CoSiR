# Findings
Reviews identify missing-vs-failed gates, stage 3 outcome-vs-weights, A28-only V5 handling, unrecorded env deferral, lost V5 causes, reporter staleness, and VL4 identity/logging gaps. Statistical kernels are already independently validated; do not change them.
The reporter's no-outcome fix also needs cpu_i to permit reporting when V2 has nothing to re-derive; added a dry-run reproduction before the queue change. All eight workers of cpu_i's rb_score pool, eg_lift's pool and eg_vl2's bootstrap chunks report one BLAS/OMP thread. CPU implementation therefore needs no change.
