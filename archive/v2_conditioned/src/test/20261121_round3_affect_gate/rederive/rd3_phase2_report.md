# Reader fix round 3: independent re-derivation, phase 2 (test seeds 49, 50, 51)

Written 2026-10-06 21:15 (Amsterdam) by the independent re-derivation agent, under `../DECISION_RULE.md` (SHA-256
2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925, asserted). Phase 2 was authorised at 21:08
(`out/PHASE2_AUTHORISED`). The round-3 implementation's code was never opened or imported. This phase used the same
own code as phase 1 (`rd3_core.py`, `rd3_bundle.py`, `rd3_family.py`), plus `rd3_phase2.py` (`hash`, `go`,
`compare`).

**Overall: agreement.** All 229 compared quantities agree under rule §8. Every discrete quantity and every per-anchor
array is identical, and every pooled point and bound equals the implementation's bit for bit (difference 0.0). No lower
bound lies within 1e-12 of 0. All seven GO checks and the secondary check have lower bounds above 0 in both
computations. The verdict itself is for `r3_apply_rule.py` to write.

## 1. Order of work

| Time (Amsterdam) | Step |
|---|---|
| 21:11 | §6.2 hash check (`rd3_phase2.py hash`): passed |
| 21:11 to 21:13 | own bundles for seeds 49, 50, 51 (`rd3_bundle.py --seed s`, the rule's §4 item 1 call sequence, 34 s each) |
| 21:13 | own per-seed families and pooled checks (`rd3_phase2.py go`); `out/phase2.json` (SHA-256 02000158…) and `out/phase2_arrays.npz` (b23cf5c0…) written |
| 21:13:57 | own files hashed; only then the implementation's `results/go_seed{49,50,51}.npz`, `go_pooled.json` and the caches they point to (`cache_seed{s}.npz`, `cache_reader_seed{s}.npz`) were opened |
| 21:14 | `rd3_phase2.py compare` → `out/phase2_agreement.json`; own files unchanged since (hashes re-checked there) |

`results/build_seed{s}.log` was never opened. From `baselines_seed{s}.json` only the fields `episodes_sha256`,
`n_per_pair`, `pair_order` and `episodes_seed` were read. Nothing that §6.4 reserves for after the verdict was computed:
no per-seed or per-pair summary, no bar margin or bar comparator, nothing of R1's counterpart (its cells, G_cf and
cross-fit are skipped by `run_family(..., with_counterpart=False)`), no gate shares and no pick accuracy. The integer
cell statistics were used only to choose cells and were not written. The fused-only path reproduces phase 1's R1 fused
arrays and cells (116, 119) on seed 42 exactly. After the edit that added it, phase 1 was rerun: `phase1_arrays.npz` is
bit-identical, and all four items still pass.

## 2. §6.2 hash check (passed)

| Check | Result |
|---|---|
| SHA-256 of `episodes_seed{s}.npz`, `per_anchor_seed{s}.npz`, `baselines_seed{s}.json` vs `results/build_seed{s}.json` | equal for all 9 files |
| per-pair episode SHA-256 (own `episodes_sha256` on each pair) vs `baselines_seed{s}.json` and the build record | equal |
| the 9 new per-pair SHA-256s | all distinct |
| against the 15 per-pair SHA-256s of seeds 42, 43, 45, 47, 48 (their `baselines_seed*.json` SHA-256s asserted against D15) | no match |
| 4,096 episodes per pair, pair order emotion×style, emotion×genre, style×genre | yes, every seed |
| `codes_provenance.json` | 8e6a517b…, equal to D15 now and in each build record before and after |

## 3. Per test seed (identical to the implementation's)

Bundles: the affect heads matched told_oracle.json arm L's record on each seed. The §4 item 2 assertions passed (the
`per_anchor_seed{s}.npz` anchor groups and pair index equal the bundle's, and its `cosine__*` equal our own per-anchor
metrics of the bundle's cosine). B and B′(A0) are condition-free. The cross-fit picks (λ_u, λ_a) by tune half 0, 1:

| Seed | B | B′(A0) |
|---|---|---|
| 49 | (8, 4), (16, 8) | (8, 4), (8, 4) |
| 50 | (16, 4), (8, 8) | (16, 8), (16, 16) |
| 51 | (4, 4), (8, 4) | (16, 16), (16, 8) |

Chosen cells (cell: τ index, λ_u, λ_a; "ties" = cells sharing the maximum, the lowest number wins). The cell chosen on
tune half h scores the episodes of parity 1 − h. σ* = 0 on both tune halves of every seed.

| Seed | AFF fused, half 0 / half 1 | AFF counterpart, half 0 / half 1 | R1 fused, half 0 / half 1 |
|---|---|---|---|
| 49 | 119 (2, 0, 16) / 63 (1, 0, 16) | 158 (2, 8, 8) / 156 (2, 8, 2) | 127 (2, 0.5, 16) / 93 (1, 4, 4), 3 ties |
| 50 | 62 (1, 0, 8), 2 ties / 127 (2, 0.5, 16) | 114 (2, 0, 0.5), 2 ties / 137 (2, 2, 0.25) | 14 (0, 0.5, 8), 2 ties / 151 (2, 4, 16) |
| 51 | 118 (2, 0, 8), 4 ties / 119 (2, 0, 16) | 115 (2, 0, 1), 2 ties / 170 (3, 0, 0.5), 2 ties | 117 (2, 0, 4), 2 ties / 68 (1, 0.5, 2), 2 ties |

Per seed, these agree exactly with the implementation (it stores margins and picks, not gates, so the gate check
applies D6 to its arrays):

- picks and float64 margins of both conditions;
- R1's and AFF's gates at τ_0..τ_3;
- the anchor groups, pair index and parity;
- the per-anchor arrays (r1, gain, other, swap, strict) of AFF fused, AFF counterpart, R1 fused, B, B′(A0), cosine and RCA (dtype float64);
- the bundle pieces: anchors; cosine, B and B′ scores (float32); the 18 features; the grouping scores.

## 4. Pooled over 49, 50, 51 (36,864 episodes, 5,195 painting clusters)

Percentage points, 95% painting-bootstrap intervals (5,000 resamples, seed 42, chunk 250). A check passes if its
lower bound is strictly above 0. Ours and the implementation's are equal to the last bit.

| Check (AFF fused minus …) | Point | Lower | Upper | Pass |
|---|---|---|---|---|
| R@1, cosine | 5.839708116319445 | 5.6113334691723 | 6.068553623829037 | yes |
| R@1, RCA | 5.721706814236112 | 5.495126735024904 | 5.9456186249255225 | yes |
| R@1, B | 0.8063422309027778 | 0.6862297310541294 | 0.9302913961004797 | yes |
| R@1, B′(A0) | 0.5913628472222222 | 0.46150280307786573 | 0.7290529246365278 | yes |
| R@1, matched counterpart | 0.7961697048611112 | 0.669718134664606 | 0.9201667199227945 | yes |
| gain statistic (counterpart gain 0 on every episode, asserted) | 3.3189561631944446 | 3.1301212087937755 | 3.5178707882265416 | yes |
| condition gain, RCA | 3.2857259114583335 | 3.0764180203333327 | 3.494111586858046 | yes |
| secondary: R@1, R1 fused | 0.2020941840277778 | 0.09320136111224805 | 0.3087028210199184 | yes |

All seven GO checks pass in our computation, which matches the implementation's `go: true`. The secondary check
passes. The smallest lower bound is 0.0932 (the secondary check), far from 0, so the §8 boundary flag is not raised.

## 5. Agreement (`out/phase2_agreement.json`)

229 quantities, all agree:

| Quantity | Count |
|---|---|
| per-anchor arrays: 7 scorers × 5 metrics × 3 seeds | 105 |
| bundle pieces: cosine, B, B′ scores; features; grouping scores; anchors | 51 |
| picks, margins, R1 gates, AFF gates (× 2 conditions × 3 seeds) | 24 |
| chosen cells: AFF fused, AFF counterpart, R1 fused; σ* | 12 |
| alignment arrays: anchor group, pair index, parity | 9 |
| pooled checks: point and bounds; pass or fail | 16 |
| metadata, hashes, τ, pooled counts, GO flag | 12 |

## 6. Files

- `out/phase2_hash.json`: the hash check.
- `out/phase2.json`: per-seed cells, σ*, bundle picks, assertions, and the pooled checks.
- `out/phase2_arrays.npz`: per-seed per-anchor arrays, gates, picks, margins, and the pooled per-episode differences.
- `out/phase2_agreement.json`: `all_agree`, the rule SHA-256, the time, and the compared list with agree flags.
- `out/rd3_bundle_seed{49,50,51}.{npz,json}`: bundle caches, 61 MB each.

The folder's `.gitignore` covers everything in `out/`.
