# Graph rows: exact definitions (agent defaults within spec §3.2; recorded in the run log)

Spec: `docs/superpowers/specs/2026-10-18-aspect-graph-sweep-design.md` (§3.2 rows, §3.4 scores, §3.5 stage 0).

## Spaces (painting level, all 19,889 paintings in `load_frozen()` order, L2-normalised)

- `cap[key]` = `itp_data.targets(s, key)` (painting mean of the key's caption vectors), key in
  {default, emotion_focus, style_word, genre_word}.
- `img[key]` = `ags_trainimg.aspect_images(s, key)` for the three aspect keys; `img['default']` = `itp_data.default_images(s)[0]`.
- CLIP content (`Dc`, `Ic`, `Cc = Dc | Ic`): exactly as `pbp_clip_content.py` builds them.

## Aspects

| Aspect | key | primary view `a` | other view `o` | content vector for F-orth |
|---|---|---|---|---|
| emotion | emotion_focus | `cap[key]` | `img[key]` | `cap['default']` |
| style | style_word | `img[key]` | `cap[key]` | `img['default']` |
| genre | genre_word | `cap[key]` | `img[key]` | `cap['default']` |

## Shared pieces (each built separately on the fit partition, 13,400 nodes, and the dev partition, 1,489 nodes; `itp_data.split(s)` order; local indices)

- `mutual(x)` = `pav_score.mutual_pairs(x)` unchanged (reciprocal cosine top-20).
- `D = mutual(cap['default'])`, `I = mutual(img['default'])`, `C = D | I` (the pilot's content graph).
- Standardised similarity: `z_X(i,j) = cos_X(i,j) / sd_X`, where `sd_X` is the population sd of `cos_X` over 200,000
  random off-diagonal pairs of that partition (rng `np.random.default_rng(0)`, drawn once per partition and reused for
  every space).
- Content score: `s_c(i,j) = (z_D(i,j) + z_I(i,j)) / 2` (z of `cap['default']` and `img['default']`).
- `mutual_top20(score)`: for each row i, the 20 columns with the highest score, self excluded (ties as `argpartition`
  gives them); keep (i,j), i<j, when each is in the other's 20. Computed in row blocks of 256 so no n×n matrix is held.

## The 21 rows per aspect

| Row | Family | Pairs |
|---|---|---|
| R0 | reference | `mutual(a)` |
| R0o | reference | `mutual(o)` (the other side alone) |
| R1 | reference | `R0 - C` |
| R1c | reference | `R0 - Cc` |
| Prank5, Prank10 | prune | `R0 - (mtop_r(cap['default']) | mtop_r(img['default']))`, `mtop_r` = reciprocal top-r (r=20 gives R1) |
| Psim50, Psim75, Psim90 | prune | `R0` minus pairs with `s_c > t_q`, `t_q` = q-quantile of `s_c` over the partition's R0 pairs |
| Fres025, Fres05, Fres1 | fuse | `mutual_top20(z_a - λ·s_c)`, λ = .25, .5, 1 |
| Forth | fuse | `mutual(l2(a_i - (a_i·c_i) c_i))`, `c_i` the content vector of the table above |
| Fside025, Fside05, Fside075 | fuse | `mutual_top20(w·z_cap[key] + (1-w)·z_img[key])`, w = .25, .5, .75 (w = caption weight) |
| Frank | fuse | `mutual_top20(1/(60+r_cap) + 1/(60+r_img) - 1/(60+r_c))`; r = 1-based rank of j in row i under `cap[key]`, `img[key]`, `s_c`, self excluded |
| Gdown025, Gdown05 | guided | pairs of R0; weight α for pairs in `R0 & C`, 1 for the rest; α = .25, .5 |
| Gagree | guided | `(R0 - C) | (R0 & C & R0o)` |
| Gres | guided | `mutual_top20(z_a - λ_i·s_c)` with row-wise `λ_i` = share of i's R0 pairs that are in C; nodes with R0 degree 0 take the mean `λ_i` of the others |

Unweighted rows have weight 1 on every pair.

## Scores per row (fit partition; labels from `pbp_graphs.label_vectors(s, fit_nodes)`, read only after every graph is built)

- fit pairs, dev pairs, weight sum, degree-0 count, mean degree.
- Lift per label (emotion, style, genre), as `pav_score.graph_stats`, weighted for weighted rows:
  same-label rate = Σ w·[same] / Σ w over pairs where both labels are ≥ 0; random rate unchanged.
- Target lift (the aspect's own label); lift ratio = target lift / mean(lifts of the other two labels).
- Overlaps: |G ∩ R0| / |G|, |G ∩ C| / |G|, Jaccard(G, R0).
- Diagnostics (spec §3.6): lifts of the dropped pairs `R0 - G` (when not empty); count of nodes with R0 degree > 0
  and G degree 0.

## Downstream selection (spec §3.4, D9)

Eligible: dev pairs ≥ 200 and fit pairs ≥ 0.2·|R0 fit|. Per aspect, in this order, without repeats: R0, R1, the best
eligible prune row by lift ratio, the best eligible fuse row by lift ratio, the best eligible guided row by lift ratio,
the best eligible row of all 21 by target lift. At most six rows.

## Stage 0 gate (spec §3.5)

For emotion: `R0`, `R1`, `R0 & C` and `C` are pair-identical, on fit and on dev, to the pilot's `E`, `P`, `EC`, `C`
(`/project/CoSiR/src/test/20261018_purified_buddy_pilot/graphs/{fit,dev}.npz`), and their fit emotion lifts equal the
pilot's `graphs/stats.json` values (2.4204897561, 2.3849313416, 2.5528809893) within 1e-9. Any mismatch: write
`STAGE0 FAIL` with the differences, build nothing else, exit 1.
