# Independent phase-1 inputs

Paths in this table are relative to `/project/CoSiR/src/test/`. Old-round inputs resolve in the main checkout.
Round-9 paths in pins resolve in the checkout running the scripts. Nothing imports or opens any prohibited source
file. The frozen library functions allowed by R6 §8.4/R3 §8 are imported without inspecting their source.

These are **all pinned data files/keys actually consumed by phase 1**. A recipe on another backbone must emit the
same file names and key layouts for the fitted roles. Frozen roles retain CLIP's files.

| Role | CLIP pinned file | Keys read | Other backbones |
|---|---|---|---|
| `prepare` | `20261013_stage_d_selection/cache/prepare.npz` | `groups`, `split_train`, `scorer_train`, `selection` | frozen |
| `partition_L` | `20261111_community_told_oracle/results/per_anchor_told_oracle.npz` | `partition_L` | frozen |
| `told_oracle` | `20261111_community_told_oracle/results/told_oracle.json` | `arms.L.head.n_classes`, `arms.L.head.draw_rows_sha256` | frozen; draw identity |
| `partitions` | `20261031_pseudo_partitions/results/partitions.npz` | `affect`, `image`, `caption`, `local_groups` | affect and local groups copied; image and caption fitted |
| `n6_posteriors` | `20261108_new_method_quick_checks/results/n6_posteriors.npz` | `selection`, `{affect,image,caption}__{img,txt}` | fitted; selection copied |
| `A3_seed42` | `20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt` | `state_dict`, `config`, `feature_dim` through the allowed checkpoint loader | fitted |
| `picked` | `20261101_aspect_factor_gonogo/results/picked.json` | `run`, `checkpoint_sha256` | backbone's A3 identity |
| `rb_reader_A0` | `20261117_reader_fix_csd/results/rb_reader_A0.pkl` | `config`, `groupings`, `feature_names`, `sklearn_version`; both `halves`' `scaler`, `model` (classes and probability prediction) | fitted |
| `rb_reader_A0_record` | `20261117_reader_fix_csd/results/rb_reader_A0.json` | `config`, `groupings`, `feature_names`, `sklearn_version` | fitted |
| `rc_tau` | `20261117_reader_fix_csd/results/rc_tau.json` | `taus` (comparison after independent percentile calculation) | fitted |
| `baselines_seed42` | `20261030_aspect_baselines/results/baselines_seed42.json` | `fit_rows_sha256`, `episodes_sha256` | RCA/PM draw identity; episodes frozen |

Phase 1 rebuilds episodes from labels and development value sets. It does **not** read `episodes_seed42.npz` to
compute scores. The manifest adapter requires that frozen episode file to be present in a pins file because it is
a shared frozen role, but it is not used as an arithmetic input. The reader pickle already contains both fitted
half-readers; `rb_heads_*`, `rb_bank_A0_half0/1`, `rb_halves`, `rb_reader_A0.npz`, graph and bank AIC are recipe
inputs rather than inputs to this re-derivation. R1 and PM are outside the ticket's P1–P7 computation. The review fix adds B1
for the full phase-1 mean/hash audit, as described below.

Feature/label loader inputs: CLIP's `src.data.artelingo.load_artelingo()` uses the default shard store
`/data/SSD2/pre_extract/artelingo/features` (`img_features`, `txt_features`, extraction `sample_ids`) and the annotation
file `/data/PDD/artelingo/artelingo_train.json`. `artelingo_aspect_labels` supplies the three existing label codes.
Other backbones use `/data/SSD2/pre_extract/artelingo_full/<backbone>/{img.npy,txt.npy,index.json}`;
`index.json.texts[*]` must contain `sample_id`, `painting`, `img`. The join is explicitly to CLIP's extraction IDs.
The leakage split always comes from the pinned CLIP split, never a new encoder's image bytes.

## Pins contract

`--pins PATH` accepts nested objects/lists with `{ "path": "...", "sha256": "..." }` entries, or
file-path-to-SHA maps (also file-path-to-`{sha256}` maps). Roles resolve by exact file name. A missing required role
asserts. All listed data files are hashed before use. Source-file digests are treated as metadata and never opened:
the controller's provenance gates validate those. The pins schema is not specified by the ticket/index, so this
adapter contract is explicit rather than inferred from unread implementation code.

Without `pins_clip.json`, CLIP uses the digest constants in R6's allowed independent `rd_common.py`, parsed as
literal data without importing it. The R3/R6 rule digests and the current R9 rule digest are always checked. The
checkpoint must also match `picked.json`. With pins, every required role has to be pinned. Every actual read is
recorded with its hash and the union of keys read. Cached phase-1 inputs are checked again before scoring/phase 2.

## Phase 2 additions

For ArtELingo held: `20261014_stage_d_final/cache/held_codes.npz` reads only `held_rows`; H5's
`20261125_artelingo_held_test/results/held_episodes_seed{52,53,54}.npz` supplies concatenated `anchor`, `candidates`, `pairs_{a,b}_{img,txt}`, with `pair_index`, `pair_names` and
`sha__<pair>` (split into three 4,096-row parts) for
identity checks **after independently rebuilding them**. H5 hashes also enter the A28 distinctness guard, along
with `episodes_sha256` in every non-smoke `20261030_aspect_baselines/results/baselines_seed*.json`.

For ArtELingo-28: `read_set_a28.parquet` has exactly `row_id`, `painting`, `language`, `emotion`, `style`, `genre`,
`held`, `dropped`, in increasing CSV row-ID order. Row positions, rather than sparse CSV IDs, index episode arrays.
`value_sets_a28.json` supplies per-aspect code arrays or `{codes, names}` records. The complete `pins_a28.json` is
hashed against its data files; the stores are `/data/SSD2/pre_extract/artelingo28/<backbone>/`, with
`index.json.texts[*].{row_id,painting,img}` in kept-row order. The full table stays the 74,569-position row space;
features, codes and posteriors are finite on kept rows and NaN elsewhere.

Comparison-only inputs (opened once our quantities exist): R6 `picks_seed42.json`, `seed42_pass_counts.json`, R3
`seed42_arrays.npz`, and a controller-specified per-read pass JSON/array archive. These never supply computed picks,
metrics or bootstrap differences.

## Review fix: all-backbone phase-1 agreement

`rd9_compare.py --phase1 --backbone B --pins FILE` reads the file-pinned `cells_seed42.json`, `picks_seed42.json`,
`baselines_seed42.json`, `rc_tau.json` **after** the independent record and array archive exist. It compares cells,
sigma, switched-off state and gate share; B/B0 picks; `lambda_picks.rca` (the original CLIP `scorers.rca.lambda_picks`
layout is also accepted); thresholds; and all five seed-42 means, seven check points/counts and every recorded array
hash with its dtype and shape. A disagreement gets an output JSON and a failing exit status, including switched-off
backbones; the phase-2 refusal no longer serves as the only check of that flag.

The pins' seed-42 record is `seed42` (or `seed42_record`) with:

```json
{
  "mean_r1": {"B": 0.0, "B0": 0.0, "B1": 0.0, "AFF": 0.0, "CF": 0.0},
  "checks": {"P1": {"point": 0.0, "n_j": 0}},
  "array_hashes": {"B__r1": {"sha256": "64 hex characters", "dtype": "float64", "shape": [12288]}}
}
```

All P1–P7 entries and all 30 primary per-anchor metric arrays are mandatory; any additional recorded arrays are
also compared. Numbers shown above illustrate the schema only. The five means are percentages. B1 is independently
cross-fitted in **phase 1 only**, from the pinned `step1_heads_style.npz` keys `selection`, `style_csd__img`,
`style_csd__txt` and the already-derived uniform A3 term. B1 remains absent from the pre-verdict read computation.
These three keys are additional phase-1 input keys. CSD labels need not be fitted or re-encoded for this calculation.

A recipe role with several entries sharing its file name prefers exactly one entry under
`results/recipe/<backbone>/`; otherwise ambiguity asserts. Frozen-role resolution is unchanged. Phase-1 prepare,
phase-1 scoring and phase-2 records now include `library_sha256`: SHA-256s of loaded `src` Python library modules,
keyed by repository-relative path. `src/test` is excluded, preserving implementation-chain independence.
