# Independent recompute: lightweight CLIP fine-tuning comparator

> 2026-10-07 09:46 (Amsterdam). Spec `docs/superpowers/specs/2026-10-07-clip-lightweight-ft-design.md` §4, §5.
> Code: `rd_ft.py` (recompute), `rd_compare.py` (comparison). Outputs (gitignored): `rd_ft.json`, `rd_compare.json`.

## Process

- We wrote `rd_ft.py` without reading `ft_eval.py`, its tests, or `results/eval.json` / `eval.log`.
- Our own code covers the selection from `metrics.json`, the val-retrieval check of each features file, row
  placement (NaN outside selection), cosine scoring, and per-anchor R@1, other-aspect rate and either rate (ties miss).
- Shared code: `src.eval.aspect_metrics.cluster_bootstrap` (5,000 resamples, seed 42, chunk 250) for the intervals,
  and `src.eval.aspect_episodes` for the episode container and its SHA-256.
- We wrote `rd_ft.json` and recorded its SHA-256 (`9f681d74…f7f20f`, file `rd_ft.json.sha256`) in commit e98f21d,
  before we opened `eval.json`.

## Selection (spec §4)

| Variant | lr | Epoch | Val selection | Epoch-0 selection | Notes |
|---|---|---|---|---|---|
| LP | 3e-4 | 9 | 0.116827 | 0.066829 (frozen cached features = plain CLIP) | interior lr and epoch |
| LB | 3e-5 | 8 | 0.133356 | 0.066927 (image-cache path) | best lr is the grid's largest |
| LoRA | 1e-4 | 10 | 0.151524 | 0.066927 (image-cache path) | best epoch is the last; the 3e-4 run also peaks at 10 |

- Each maximum is unique (no ties). The exact rational selection and the stored floats give the same choice.
- In every run, `run_record.json`'s `best_epoch` and `best_selection` equal the epoch ≥ 1 maximum in `metrics.json`.
  No run's maximum is at epoch 0.
- Each selected run's `features.npz` reproduces the claimed epoch's val counts exactly when we recompute val
  retrieval from it. LP epoch 9 gives 955 / 2,421, LB epoch 8 gives 1,094 / 2,744, and LoRA epoch 10 gives
  1,246 / 3,103. Every other epoch differs. Each `features_epoch0.npz` reproduces epoch 0 exactly.
- LP's `features_epoch0.npz` equals the frozen cached selection features exactly (max abs diff 0).

## Episode evaluation, pooled over seeds 49 to 51 (pp; 36,864 episodes, 5,195 painting clusters)

R@1 per scorer: plain 13.040, cache-path reference 13.032, LP 14.998, LB 14.969, LoRA 15.135, B 18.073,
B′(A0) 18.288, AFF 18.880.

| Variant | AFF − ft, R@1 | ft − plain, R@1 | ft − B′(A0), R@1 | AFF − ft, either | ft − plain, either | ft − B′(A0), either |
|---|---|---|---|---|---|---|
| LP | +3.881 [3.682, 4.089] | +1.959 [1.787, 2.129] | −3.290 [−3.472, −3.112] | +4.443 [4.105, 4.798] | +3.917 [3.574, 4.258] | −6.580 [−6.944, −6.224] |
| LB | +3.911 [3.707, 4.121] | +1.929 [1.750, 2.104] | −3.320 [−3.504, −3.137] | +4.503 [4.159, 4.864] | +3.857 [3.501, 4.209] | −6.639 [−7.009, −6.273] |
| LoRA | +3.744 [3.537, 3.961] | +2.096 [1.909, 2.283] | −3.153 [−3.344, −2.966] | +4.169 [3.811, 4.545] | +4.191 [3.819, 4.566] | −6.306 [−6.689, −5.932] |
| cache ref | +5.848 [5.619, 6.077] | −0.008 [−0.020, 0.005] | −5.256 [−5.467, −5.038] | +8.377 [7.986, 8.775] | −0.016 [−0.041, 0.009] | −10.513 [−10.934, −10.075] |

Per aspect pair (pooled 49 to 51, R@1 points):

| Pair | plain | LP | LB | LoRA | cache ref | B | B′(A0) | AFF |
|---|---|---|---|---|---|---|---|---|
| emotion × style | 10.150 | 10.984 | 11.051 | 10.933 | 10.134 | 12.427 | 12.642 | 13.452 |
| emotion × genre | 14.667 | 17.076 | 16.980 | 17.346 | 14.657 | 21.149 | 21.334 | 22.878 |
| style × genre | 14.303 | 16.935 | 16.874 | 17.126 | 14.305 | 20.644 | 20.888 | 20.308 |

Per-seed values (42, 49, 50, 51) are in `rd_ft.json`. Fine-tuning lifts R@1 least on emotion × style (+0.8 to +0.9
over plain) and most on the two genre pairs (+2.3 to +2.8).

## Checks

- Our plain-CLIP cosine equals `cosine__r1` and `cosine__other` in `per_anchor_seed{s}.npz` exactly on all four seeds.
  The anchor groups match too.
- The stored AFF, B and B′ arrays align with our episodes. Their `cl`, `pair_index` and `cosine__r1` are identical to
  ours on all four seeds.
- Round 3's and round 4's seed-42 `aff_fused` arrays are identical.
- We used AFF = `aff_fused` and B′(A0) = `Bp` (round 3) or `Bp0` (round 4). This choice reproduces round 3's pooled
  AFF − B′ = +0.5914 and AFF − B = +0.8063.
- Scoring in float64 instead of float32 changes no per-anchor value (zero flips on any seed or scorer).
- Leakage groups and painting names give the same 5,195 pooled clusters.

## Comparison with `results/eval.json` (sha256 `4204be40…6a18b`)

- **Agreement: yes.** We compared 531 quantities: means (R@1, either, other) per seed and pooled; the three
  differences per variant and the cache reference (point, both bounds and cluster count, per seed and pooled); and the
  per-pair pooled points. The maximum absolute difference is 0.0 pp.
- **Selection: identical.** `eval.json` does not name its input files. Of the nine runs' `features.npz` files, only
  the selected run's reproduces `ft:LP`, `ft:LB` and `ft:LoRA` on all four seeds. `ft:CLIPcache` is reproduced by the
  selected LB run's `features_epoch0.npz`, and equally by all six LB and LoRA epoch-0 files, which give identical
  episode results. The main session's record (`selected.json`, log line 09:35) names the same three runs.
- **Not recomputed:** the single-scorer intervals, the condition gain, the per-pair interval comparisons, and the
  seed-42 per-pair block (`pairs_seed42`).
