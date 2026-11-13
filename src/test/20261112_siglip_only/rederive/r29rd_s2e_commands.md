# Controller commands
Run from `/project/CoSiR/src/test/20261112_siglip_only/rederive/`. The runner sets OMP/MKL/OpenBLAS to 8 threads and disables bytecode. It uses `/root/miniconda3/envs/CoSiR/bin/python`. CPU is default; CUDA is controller-only. No deletion or commit commands are included.

```bash
bash r29rd_run_s2e.sh test > r29rd_s2e_tests.log 2>&1
bash r29rd_run_s2e.sh val --check-inputs --output r29rd_s2e_status.json > r29rd_s2e_status.log 2>&1
```

Full commands, for the controller's resource queue (replace `cpu` with `cuda` for GPU). Long output goes to logs; inspect the tail after each command.

```bash
bash r29rd_run_s2e.sh replay --device cpu --output r29rd_s2e_replay.json > r29rd_s2e_replay.log 2>&1
bash r29rd_run_s2e.sh val --device cpu --replay-reports r29rd_s2e_replay.json --output s2e_val.json > r29rd_s2e_val.log 2>&1
```

After the controller has committed the validation picks and training part b has exported held rows:

```bash
bash r29rd_run_s2e.sh held --device cpu --picks s2e_val.json --picks-committed --replay-reports r29rd_s2e_replay.json --output s2e_held.json > r29rd_s2e_held.log 2>&1
```

`--picks-committed` asserts that external precondition. The script makes no commits. `val` and `replay` cannot open stage-2 held exports. Full reports pin code and every input file by SHA256. Full held scoring rejects stale/incomplete validation picks. Missing/failed full replay certification makes affected outcomes INCONCLUSIVE, even if numerical diagnostics exist.

Estimated cold-run wall time, eight CPU threads; GPU estimates are unmeasured (this job never used GPU):

| Command | CPU | RTX 3090 controller GPU |
| --- | --- | --- |
| test | 2–5 s including startup | 2–5 s (CPU fixtures) |
| val --check-inputs | 2–5 s | 2–5 s |
| replay (all) | 45–90 min | 30–60 min |
| val (all) | 25–60 min | 15–45 min |
| held (all) | 20–40 min | 8–20 min |

The estimates are deliberately broad: full crossfit includes 6,144-dimensional covariance/eigensolves on CPU; GPU rankings use float64 as r11 specifies, and AMI/bootstraps stay on CPU. Source/input-bound caches shorten repeated commands. An observed full held told mAP seed/aspect took 28–30 s CPU; joined untold took 43 s. These are frozen-model timings, not stage-2 numbers.

Bounded replays can use `--family groups|val|ami|map`, `--encoders`, `--aspects`, `--seeds`, `--sides`, and `--query-limit`. `replay --family val` needs all three aspects for its one arm-level pick. `--quick` drops whitening and cannot certify picks. `--family ami|map` always uses all five training seeds for its paired intervals; `--seeds` subsets validation only. Sampled query reports do not certify full scoring. `--frozen-units` (only replay map) tests a complete bootstrap from old independent C13 per-query AP caches located by input identity, and separately compares our own sampled APs; it never certifies full scoring when queries were sampled.

No files should be removed to run these commands. Interrupted-run caches/logs can be retained. Nothing was deleted.

Additional completed bounded checks, CPU (each command below has been exercised; use a separate log as shown):

```bash
bash r29rd_run_s2e.sh replay --family groups --device cpu --output r29rd_s2e_groups.json > r29rd_s2e_groups.log 2>&1
bash r29rd_run_s2e.sh replay --family ami --aspects emotion --device cpu --output r29rd_s2e_ami_emotion.json > r29rd_s2e_ami_emotion.log 2>&1
bash r29rd_run_s2e.sh replay --family map --frozen-units --query-limit 64 --device cpu --output r29rd_s2e_map_units.json > r29rd_s2e_map_units.log 2>&1
bash r29rd_run_s2e.sh replay --family val --quick --query-limit 16 --seeds 0 --device cpu --output r29rd_s2e_val_smoke.json > r29rd_s2e_val_smoke.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python r29rd_s2e_forward_replay.py --device cpu > r29rd_s2e_forward_replay.log 2>&1
```

Observed cold CPU: groups 220.8 s, emotion AMI 142.4 s (some partitions cached), map sampled-scoring/full frozen-unit bootstrap 221.8 s, quick crossfit smoke 7.4 s. Estimated GPU: groups 2–4 min, emotion AMI 2–3 min (CPU AMI unchanged), sampled map/frozen-unit bootstrap 2–4 min, quick smoke 5–15 s; frozen forward CPU 10–30 s / GPU 5–15 s. GPU estimates remain unmeasured. Frozen forward equality shares: SigLIP2 0.9983391135504714, CLIP L14 0.998222519734436; max absolute quantized difference 0.00006103515625, all remaining elements within one float16 step.

The full whitening-menu pick replay, full independent mAP ranking replay and style/genre AMI intervals are pending the full controller replay. No final stage-2 val/held result is claimed by the smoke reports. See r29rd_s2e_notes.md for exact numerical replay differences and the optional cleanup list.
