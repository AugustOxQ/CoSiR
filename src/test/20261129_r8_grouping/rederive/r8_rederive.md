# Ticket 11 stored-file interface

`r8_rederive.py` is separate code for decision-rule §7(a)--(d), including §6.1--6.5.
It imports only `r8_common` and the frozen library pieces permitted by R7 §9 phase A.
It loads the frozen A3 object on CPU to recover N1u inputs; it fits no grouping,
head, reader or builder. All real joins come from `r8_common`. Own-map image
posteriors recovered from φ drive its scoring, after comparison with stored ones.

Run from the round folder with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
MKL_NUM_THREADS=8`, using `/root/miniconda3/envs/CoSiR/bin/python`.

```bash
python rederive/r8_rederive.py --step trigger --pending results/trigger_pending.json --stage-evidence results/rederive_inputs.json
python rederive/r8_rederive.py --step gate --pending results/gate_pending.json --stage-evidence results/rederive_inputs.json
```

`--root` defaults to `r8_common.RESULTS`. `--index` and `--groupings` can each be
repeated for the NBC extension. Defaults are `ROOT/seed42/index.json` and
`ROOT/groupings/r8_groupings.json`; each grouping JSON has a same-stem NPZ.
Every batch carries the index's seven required common arrays, and duplicate
keys must agree exactly. Every grouping must have both scorers, their five
metrics, both cross-fits, gate arrays, cells, sigma, and BX's five metrics.
AFF/AFF_cf and B/B0/B1 are independently recovered too. Optional 224-cell
statistics use `NAME__rho_f`, `gam_f`, `rho_c`, `crit_f`, `ctrl_rho`, `f_cnt`,
`c_cnt`. All supplied arrays must be independently recovered; none is ignored.

Real arrays/heads/readers must already exist. A0's stored probabilities default
to `ROOT/seed42/reader_A0.npz`; `--aff-reader PATH` supplies another path. τ^X
is read from `ROOT/taus.json` and independently checked against percentiles of
the stored reader probabilities. AFF keeps the rule's four frozen thresholds.
`--tp-map PATH` is required when TP is evaluated: its NPZ contains float32
`text_vectors` (prompt order) and integer `merge_map` (one final class per prompt).
TP's selection and bank maps are recomputed on the common unit CLIP vectors.

## Pending records (explicit rule-gap convention)

The index specifies hash coverage but no pending verdict field schema. This
module's canonical pending record is:

```json
{"smoke": false, "inputs": {"/absolute/input/path": "SHA256"}, "expected": {}}
```

`expected` must be produced by the pipeline, not by this re-derivation. For
`trigger`, it has `triggered`, `strength`, `reason`, and `NB` (evaluated NB
scorer records). Reasons are `no NB scorer was evaluated`, `no NB scorer
passed`, or `an NB scorer passed`; strength is null unless triggered.

For `gate`, `expected` has `main`, `screen`, `references`, `controls`, `trigger`,
`groupings`. Each track has `candidates` keyed by array scorer ID, `pick` (array
scorer ID or null), `random_control_flag` (bool, `not evaluable`, or null without
a pick), and `dropped` (grouping -> reason). Main also has `promising` and
`H_target`. `groupings` preserves the supplied grouping records. Dropped
scorers never pass; their reasons are in `dropped`, rather than fabricated
numeric scorer records. References/controls pick nothing.

Each evaluated scorer record has:

- `SC3`: `sum4d`, `n`, `point`, `ci95`, `pass`.
- `SC4`: `emotion__style` and `emotion__genre` each with `sum4d`, `n`, `point`,
  `pass`; plus the combined `pass`.
- `S1`: `sum4d`, `point`, `ci95`, `gain_sum4d`, `gain_point`, `gain_ci95`, `pass`.
- `S2`: `sum4d`, `pass`; `S3`: the same record as SC4.
- `bar_comparator`, `pass` (SC3 and SC4), `dropped: false`.

All points/intervals are percentage points. Only intervals have a 1e-9
tolerance. All discrete quantities and arrays compare exactly. S1/SC4 use
the rule's fixed integer bars, including in smoke mode.

The same canonical fields may instead be at the pending record's top level.
Metadata `inputs`, `input_sha256`, `smoke`, `rule_sha256`, `written`, `step` is
excluded from that comparison. Input manifests can also be lists of paths or
`{path, sha256}` records, or dictionaries whose values are such records.
Relative input paths are relative to the pending file. Every declared input
is hashed, including inputs the numerical calculation does not otherwise use.

## Stage evidence (explicit rule-gap convention)

The rule provides no raw stage-evidence filenames or stage-record subkeys.
`--stage-evidence PATH` provides this JSON (relative paths use its directory):

```json
{
  "encoders": {
    "AL": {
      "record": "stageE_AL.json",
      "embeddings": "/path/to/raw_embeddings.npy",
      "image_index": "/path/to/image_index.npy",
      "index_sha256": "SHA256_OF_CONTIGUOUS_INT64_INDEX_BYTES",
      "keys": {"shape": "checks.shape", "finite": "checks.finite", "index": "checks.index"}
    }
  },
  "fcmi": {
    "record": "stage1_fcmi.json",
    "labels": "/path/to/premerge_labels.npy",
    "losses": "/path/to/all_step_total_losses.npy",
    "keys": {"label_count": "checks.label_count", "finite_loss": "checks.finite_loss"}
  }
}
```

Include AL, DI, SG in real runs. Encoder dimensions are 256, 1024, 1152;
49,121 rows and the common non-held index are independently checked. FCMI's
labels have 36,518 entries before merging. `keys` maps each check to the
boolean field of the actual pipeline stage record (dotted paths supported).
The finite-loss input contains every step, not just the last epoch.
A failed stage that produced no arrays instead has `record` and a nonempty
`not_produced` reason; its stage record must have `pass: false`. The module
does not fabricate missing evidence. Smoke may omit stage evidence; a smoke
encoder entry can name a `fixture` NPZ containing its tiny `image_index`.

## Output and smoke

CLI writes `ROOT/rederive/<step>_rederive.json` via the common write-once writer,
with `status`, `step`, `smoke`, `disagreements`, `input_sha256`, recovered
verdict fields, and stage checks. Successful exit is 0, disagreement is 1;
malformed/missing files assert. Immutable reports are never replaced.
Hashes include audited files read by frozen libraries; rule §10 inputs are
checked against the governing table before use. Inputs are hashed again at
the end to catch changes during the run.

CPU smoke with small stored files, no external writes/cleanup:

```bash
python -m pytest -q -p no:cacheprovider --rootdir=. test_r8_rederive.py
```

The tests retain their tiny files under the worktree round folder's
`results/smoke/rederive_tests/`. The full real re-derivation, including SHA
passes over image arrays and all scorers, may exceed four minutes and is for
the controller's run chat. `--fixture` is a synthetic-only, `--smoke` input
for CPU integration tests; it supplies common joins, frozen cosine/N1u/N6
inputs and episodes, never a precomputed candidate or comparator score.
