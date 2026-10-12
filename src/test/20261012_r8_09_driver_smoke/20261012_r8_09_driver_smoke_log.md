# r8 09 driver smoke (2026-10-12)

- make_fixture.py: R7's R0 labels, post_R0 and xheads_R0 stand in for V100 and ZV100; V67 listed as dropped. Outputs in results/ (R8_RESULTS), untracked.
- stage0 smoke (run_r8_stage0.py --smoke): wired, 54 s.
- run_r8.py readers --smoke: 2 readers, 1 skipped (dropped), 14 s.
- run_r8.py score --smoke: arrays_main.npz (103 arrays), index.json, 181 s; ticket 10's load_batches reads it.
- run_r8.py taus --smoke: taus match the frozen taus_main.json, reader npz equal, about 105 s.
