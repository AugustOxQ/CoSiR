# Controller commands and scope

All commands run in `/project/CoSiR/src/test/20261112_siglip_only/rederive`.
The runner fixes OMP/MKL/OpenBLAS threads at eight and disables bytecode writes.
GPU commands are for the controller's queue only. AMI always stays on CPU,
including when `--device cuda` accelerates retrieval cosine products.
CUDA uses float64 products, TF32 off, and CPU stable sorting/AP, preserving the
CPU protocol. GPU timings below are estimates, not measurements.

## Development fixtures (CPU, under one minute)

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m unittest -q test_r29rd_s2mp
```

## Full frozen replays (safe before stage-2 picks)

```bash
./r29rd_run_s2mp.sh replay --piece mmart --device cpu --out r29rd_s2mp_replay_mmart.json > r29rd_s2mp_replay_mmart.log 2>&1
./r29rd_run_s2mp.sh replay --piece photo --device cpu --out r29rd_s2mp_replay_photo.json > r29rd_s2mp_replay_photo.log 2>&1
```

For the GPU queue, replace `--device cpu` with `--device cuda`; use a separate
output/log filename. Full frozen replay is deliberately larger than four minutes;
run it through the controller's resource queue. Expected uncached CPU: MMArt
60–150 min, photo 25–60 min; GPU: MMArt 30–90 min, photo 15–40 min. The joined
6144-dimensional MMArt validation PCA and CPU stable sorting limit GPU speedup.

Fragments can be scheduled separately with `--dataset affection|emoset`,
`--replay-group validation|ami|map|rebuild`, `--metric ami|map`, `--aspects ...`,
and `--comparators p1|q1`. `--seeds 0` and `--query-limit 32 --quick` are diagnostics
and cannot certify the held gates. All fragments retain complete query galleries.
Use all five seeds and protocol draws for complete evidence; supply the resulting
JSON files together with `--replay-json`. A single invocation's result is never
silently presented as coverage of unrun pieces.

## Stage-2 validation, once exports exist

```bash
./r29rd_run_s2mp.sh val --piece mmart --device cpu --forward-check > r29rd_s2mp_val_mmart.log 2>&1
./r29rd_run_s2mp.sh val --piece photo --device cpu > r29rd_s2mp_val_photo.log 2>&1
```

These merge their own piece into `s2mp_val.json`. Missing exports produce explicit
pending pieces, never fabricated picks. CPU estimates: MMArt 25–70 min, photo
8–20 min; GPU estimates: MMArt 15–50 min, photo 5–15 min. The mandatory MMArt forward
check applies all three seed-0 image heads on the first 500 validation rows on CPU
and compares float16-rounded exports (99.5% equal, remaining entries at most one
float16 step). Add about 5–20 sec CPU/GPU. Photo forward check is optional and is
reported as not implemented.

## Held, only after each piece's picks/ranges are committed

After the controller commits the independent validation record, compute its exact
digest. `--picks-committed` is the controller's attestation of that ordering; this
script does not inspect forbidden r29 records or create a commit.

```bash
S2MP_VAL_SHA=$(sha256sum s2mp_val.json | cut -d' ' -f1)
./r29rd_run_s2mp.sh held --piece mmart --device cpu --picks-committed --val-sha256 "$S2MP_VAL_SHA" --replay-json r29rd_s2mp_replay_mmart.json > r29rd_s2mp_held_mmart.log 2>&1
./r29rd_run_s2mp.sh held --piece photo --device cpu --picks-committed --val-sha256 "$S2MP_VAL_SHA" --replay-json r29rd_s2mp_replay_photo.json > r29rd_s2mp_held_photo.log 2>&1
```

GPU queue: replace `--device cpu` with `--device cuda`. Expected uncached CPU:
MMArt 15–40 min, photo 10–25 min; GPU: MMArt 8–25 min, photo 5–15 min. Frozen
comparator AP caches can reduce these times. Each piece merges into
`s2mp_held.json`. Missing/failed replay evidence, picks or model inputs yields
INCONCLUSIVE cells and D22 counts. A rebuilt comparator error >1e-9 parks only
its dependent cells. All final intervals use 2000 AMI/10000 retrieval draws.

No stage-2 held command was executed during development. No deletion or commit
is requested. Own cache and probe files may be removed by the controller later;
this job leaves them intact.

## Executed diagnostics

All development executions used CPU. Representative full-query runtimes:
Affection five-seed mAP points/intervals 160 s; Affection AMI 56 s; EmoSet AMI
118 s; EmoSet photo seed-0 mAP + plain rebuild 126 s; MMArt emotion comparator
mAP rebuild pair 88 s. A whole replay includes many such operations plus all
validation read-out menus. The estimates above are deliberately for uncached
whole commands; they are not GPU measurements.

Cleanup candidates, only after controller review: `r29rd_s2mp_cache_*.npz`,
`r29rd_s2mp_numeric_probe.{py,json,log}`, `r29rd_s2mp_replay_mm_ami_probe.{json,log}`,
and the initial failed `r29rd_s2mp_replay_mm_rebuild_emotion.{json,log}`. Nothing
has been deleted. The final corrected emotion rebuild is saved with `_final`.

## Current verification scope and unresolved wording

`r29rd_s2mp_checks.json` contains the selected successful frozen replay evidence,
including the exact target record/key, both values, absolute difference, input
hashes and source-snapshot identity. Historical probes remain separate. Some
successful reports predate the none-map normalization correction; their unchanged
AMI/center-map paths were tested there, and the snapshot difference is explicit.
The corrected current source must be used for the controller's complete replay.
Full validation tables/ranges, all MMArt AMI aspects and complete candidate MMArt
mAP/SC7 and five-seed EmoSet SC10 mAP have NOT been certified by the subset run.

Two wording choices are recorded in every generated JSON's notes:
1. MMArt plain-side mAP inherits r17 SC7 untold joined heads; Q1-side inherits r15
   told aspect heads. The r29 table does not itself repeat the word untold.
2. EmoSet uses full committed r14 r_E=0.0069460867449697705. The rule's 0.00695 is
   rounded prose, under D18/D19 and r28's actual source.

Other numerical defaults follow the permitted older C13/read-out convention:
full float64 symmetric NumPy eigensolve; covariance ddof=1; nonnegative eigenvalue
clipping and a 1e-12 floor on the largest eigenvalue in epsilon's multiplier for
an otherwise degenerate fit. The floor is inactive for these real stores.
The photo forward check is optional and is explicitly reported as unimplemented;
the required MMArt 500-row CPU check is implemented and pending exports.

At handoff, `s2mp_val.json` is complete=false with both missing stage-2 export
paths and pending forward checks; `s2mp_held.json` is controller_pending with no
scores. No new stage-2 held model number, GPU execution, commit, or deletion was
made. The final 16 fixtures passed in 1.147 s. Frozen photo validation smoke (five
seeds, 16 queries, reduced menu on both sets) completed in 7.35 s.
