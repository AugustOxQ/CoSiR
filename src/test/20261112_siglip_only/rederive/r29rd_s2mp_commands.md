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

## Fix 2026-11-13 (Claude)

Symptom: the full frozen replay (`r29rd_s2mp_replay.json`, 73 min) failed one MMArt mAP entry,
`SC7/mmart/siglip1/genre/map/-/plain.hi` (difference 1.026e-9 against 1e-9); the other entries of the six MMArt
cells differed by 1e-10 to 7e-10. Per-query AP against the frozen r17 independent re-derivation's saved per-query APs
(`/data/SSD2/cosir_r17/20261103_new_scores/c13/cache/picked_ap__*.npz`, a frozen output) differed on 86 to 110 of
13,420 queries per candidate seed and 675 for the plain comparator (up to 2e-6 per query): rank swaps, not
accumulation noise.

Cause: query expansion. The code formed the expanded query in float64, normalized it, then cast it to float32
before the similarity product. The r11 written numeric convention (`20261029_retrieval/20261029_retrieval_log.md`,
paragraph "Numeric convention"; r17/r15 use r11's frozen retrieval) says: "QE uses the query plus the float64 sum of
its top-k gallery neighbours (the mean's common factor cancels on normalization), then float64 L2 normalization, with
no downcast; both initial and expanded similarities are float64 matrix products." The float32 cast moved near-tied
scores across each other.

Change (`aps`, `r29rd_s2mp.py`): `expanded = unit(q64 + sum64(top-k neighbours), float64)`, no `.astype(np.float32)`,
no division by k+1 (it cancels on normalization). The NOTES line on the retrieval numeric convention is updated. New
fixture `test_query_expansion_stays_float64_without_downcast`. No tolerance changed. Read-out maps and the
store-row normalization are unchanged (they already follow the same log: float64 then one float32 cast).

Check (genre cell, all 10,000 draws, both comparators, `r29rd_s2mp_replay_mm_map_genre_fix.json`, log `.log`,
467 s CPU): every `mmart/map` entry (SC7 plain: cand, comp, d, lo, hi, outcome; SC4 q1: comparator, candidate, d,
lo, hi; siglip_main cstar_map) within 2.2e-16; both comparator rebuilds 0.0. Before: genre plain hi 1.026e-9,
cand 1.1e-10, comp 2.2e-10, d/lo 3.3e-10. Fixtures: 17 passed.

Consequence: `replay_pass` binds each replay JSON to the sha256 of this source, so every replay JSON made before
this edit (all datasets, all groups) is stale for `--replay-json`. Rerun them all (the old caches are also
invalidated, they are keyed by the source sha). Also aspect-limited fragments (such as the genre check above) are
ignored by `replay_pass` (it requires all three aspects). Replay commands for the MMArt mAP part:

```bash
./r29rd_run_s2mp.sh replay --dataset mmart --replay-group map --out r29rd_s2mp_replay_mmart_map.json > r29rd_s2mp_replay_mmart_map.log 2>&1
./r29rd_run_s2mp.sh replay --dataset mmart --replay-group rebuild --metric map --out r29rd_s2mp_replay_mmart_rebuild_map.json > r29rd_s2mp_replay_mmart_rebuild_map.log 2>&1
```

Expected CPU: first command about 18 min (genre is cached from the check above: about 1 min; emotion and style
about 8 min each), second command about 2 min after the first (cache hits). The other groups (mmart ami, mmart
validation, affection, emoset) also need a rerun on the new source; the old uncached full `replay` took 73 min.
