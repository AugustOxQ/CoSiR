# ags 01 graph builder build log

Worktree: `/project/CoSiR-wt-ags01`; branch: `r10-ags-01-graphs`.
Scope: `ags_graphs.py` and this log; generated graphs and command logs stay in this round folder.
No files deleted. The pre-existing untracked sibling `20261017_prompted_aspect_views/sample.npz` was left alone.

## Commands

Every Python invocation uses this prefix (CPU only, eight threads):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/CoSiR/bin/python
```

Run from the worktree root:

```bash
# With the prefix above:
src/test/20261018_aspect_graph_sheet/ags_graphs.py --stage0-only > src/test/20261018_aspect_graph_sheet/ags_graphs_stage0.log 2>&1
src/test/20261018_aspect_graph_sheet/ags_graphs.py --help > src/test/20261018_aspect_graph_sheet/ags_graphs_help.log 2>&1
# Put timeout 235 before the Python executable, retaining the same environment:
src/test/20261018_aspect_graph_sheet/ags_graphs.py --aspects emotion > src/test/20261018_aspect_graph_sheet/ags_graphs_emotion.log 2>&1
```

## Implementation conventions

- All exact row definitions follow `notes/rows.md`. `--stage0-only` permits a bounded independent gate run; normal invocations always rebuild the gate first.
- The stage 0 regression necessarily uses fit labels after both gate partitions are constructed. These vectors are local to the gate; sweep scoring reads label vectors only after all requested aspect archives are sealed.
- Off-diagonal sampling draws 200,000 ordered pairs with replacement: draw `i` uniformly in `[0,n)`, draw `j` uniformly in `[0,n-1)`, then increment `j` when `j >= i`. This has no self pairs or rejection-dependent RNG consumption. One draw per partition is reused across spaces. Population sd uses float64 accumulation over float32 cosine values.
- Rank fusion uses full 1-based ranks over all non-self columns; equal scores use NumPy quicksort order. Mutual top-20 uses the required argpartition ties. The P-sim quantile uses NumPy's default linear interpolation and keeps scores at the threshold (`<=`).
- Selection ties retain the first row in the published row order. A repeated winner is omitted; another row is not substituted. The best overall target-lift row is also restricted to eligible rows.
- Coverage and overlaps count unique unweighted edges; weighted rows affect agreement rates and weight sums. Mean degree is the unweighted symmetric degree. Dropped-edge lifts use the original R0 edge set, so downweighting drops no pairs.
- Content cosine blocks, content ranks, content top-5/top-10 graphs, CLIP content graphs and sd estimates are shared across all new aspects in one invocation. No full square similarity or rank array is held. Each aspect's two source graphs use unchanged imported `mutual_pairs` and independently check unchanged `top20_neighbours` reciprocity.
- Sealed archives require the sample/method identity, SHA256, exact node order, array schema, pair/weight invariants and pair counts. Sheets are rebuilt from every sealed aspect, including aspects from earlier invocations. Source-vector SHA256 values and per-partition sd/quantile metadata are recorded in the manifest.

## Results

- Full emotion run exited 0 in **103.2 s**, well below the 235 s command bound. Aspect construction: **35.137 s**; shared preparation/construction: **26.272 s**. The remaining wall time includes frozen feature loading, stage 0, sealing, archive validation and scoring.
- `STAGE0 PASS 2.4204897561 / 2.3849313416 / 2.5528809893`. Latest gate construction/statistics time: **12.934 s**. Fit D/I/C/E/P/EC: **47,516 / 69,853 / 112,010 / 57,727 / 45,505 / 12,222**. Dev: **6,602 / 8,986 / 13,988 / 7,413 / 4,909 / 2,504**. All six pair arrays match the pilot on both partitions; all three lift differences are within 1e-9.
- All 21 emotion rows built on 13,400 fit and 1,489 dev nodes. Every requested cheap invariant passed, including zero-residual fit mutual-top-20 equality with R0 (57,727 pairs), pruning nesting, canonical arrays, weights and node alignment.
- `--help` exited 0. No unit tests were added or run.
- Repeated the emotion command with the same environment and `timeout 235`, redirecting to `ags_graphs_resume.log`: exited 0 in **43.4 s**, printed `emotion: sealed identity/checksum verified; skip`, and rebuilt the sheet and identical selection from the sealed archive.
- Emotion archive SHA256: `6491cda561ab5cef0e8c6119475ff6df8d5863ed100808ef49eab173e1da51db`. NPZ has the two node-ID arrays plus 84 pair/weight arrays (21 rows times two partitions times two arrays).
- Selection: **R0, R1, Psim50, Fres1, Gres**. The best target-lift row is R0, already included, hence five rows. Fres1 has the highest eligible lift ratio (3.300965), with 13,810 fit pairs, 3,162 dev pairs and 7,578 isolated fit nodes; its emotion lift is 2.192416 versus R0's 2.420490. This is a descriptive graph sheet, not a decision.
- Final availability check at about **2026-10-18 16:36 Europe/Amsterdam**: neither `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/style_word_trainimg/done.json` nor the corresponding `genre_word_trainimg/done.json` exists. Per ticket, stop here; the controller runs `--aspects style` and `--aspects genre` when their extractions seal. The implementation supports both and regenerates the cumulative sheet. No other open implementation issues observed.

### Emotion sheet

| Row | Family | Fit pairs | Dev pairs | Degree 0 | Emotion lift | Style lift | Genre lift | Lift ratio | Overlap R0 | Overlap C | Jaccard R0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | reference | 57727 | 7413 | 440 | 2.420490 | 2.282971 | 3.949175 | 0.776776 | 1.000000 | 0.211721 | 1.000000 |
| R0o | reference | 65449 | 8323 | 179 | 1.744995 | 4.302400 | 5.103562 | 0.371040 | 0.045302 | 0.417333 | 0.024665 |
| R1 | reference | 45505 | 4909 | 705 | 2.384931 | 2.042427 | 3.752610 | 0.823094 | 1.000000 | 0.000000 | 0.788279 |
| R1c | reference | 52596 | 6040 | 529 | 2.401780 | 2.108762 | 3.836025 | 0.808029 | 1.000000 | 0.157103 | 0.911116 |
| Prank5 | prune | 53454 | 6510 | 546 | 2.409480 | 2.175092 | 3.871998 | 0.796906 | 1.000000 | 0.148707 | 0.925979 |
| Prank10 | prune | 50305 | 5825 | 606 | 2.402122 | 2.120444 | 3.823293 | 0.808287 | 1.000000 | 0.095418 | 0.871429 |
| Psim50 | prune | 28864 | 3707 | 1418 | 2.302440 | 1.473240 | 2.888124 | 1.055835 | 1.000000 | 0.058204 | 0.500009 |
| Psim75 | prune | 43295 | 5560 | 681 | 2.344441 | 1.774755 | 3.399502 | 0.906194 | 1.000000 | 0.110197 | 0.749996 |
| Psim90 | prune | 51954 | 6671 | 512 | 2.377912 | 2.000634 | 3.717838 | 0.831660 | 1.000000 | 0.156581 | 0.899995 |
| Fres025 | fuse | 45081 | 6339 | 986 | 2.354374 | 1.100156 | 2.136772 | 1.454697 | 0.347131 | 0.036623 | 0.179545 |
| Fres05 | fuse | 29093 | 4983 | 3243 | 2.282627 | 0.741892 | 1.181407 | 2.373659 | 0.080879 | 0.002475 | 0.027857 |
| Fres1 | fuse | 13810 | 3162 | 7578 | 2.192416 | 0.637420 | 0.690928 | 3.300965 | 0.005938 | 0.000000 | 0.001148 |
| Forth | fuse | 49852 | 6970 | 1027 | 2.384709 | 2.605858 | 4.232557 | 0.697445 | 0.342434 | 0.398219 | 0.188613 |
| Fside025 | fuse | 62746 | 8176 | 211 | 2.063316 | 4.229499 | 5.091103 | 0.442743 | 0.109808 | 0.409062 | 0.060660 |
| Fside05 | fuse | 60270 | 7913 | 286 | 2.247655 | 3.881477 | 5.018591 | 0.505087 | 0.229036 | 0.359482 | 0.132485 |
| Fside075 | fuse | 58620 | 7771 | 423 | 2.358714 | 3.307547 | 4.790182 | 0.582562 | 0.463016 | 0.302559 | 0.304265 |
| Frank | fuse | 47837 | 6135 | 425 | 2.324152 | 2.156710 | 3.929613 | 0.763729 | 0.615632 | 0.096745 | 0.386920 |
| Gdown025 | guided | 57727 | 7413 | 440 | 2.395499 | 2.113915 | 3.811238 | 0.808586 | 1.000000 | 0.211721 | 1.000000 |
| Gdown05 | guided | 57727 | 7413 | 440 | 2.404815 | 2.176938 | 3.862777 | 0.796334 | 1.000000 | 0.211721 | 1.000000 |
| Gagree | guided | 47644 | 5701 | 602 | 2.397529 | 2.181087 | 3.831790 | 0.797465 | 1.000000 | 0.044895 | 0.825333 |
| Gres | guided | 35754 | 4926 | 1628 | 2.351168 | 1.378318 | 2.711316 | 1.149818 | 0.629608 | 0.023018 | 0.317190 |

| Row | Weight sum | Mean degree | Dropped pairs | Dropped emotion lift | Dropped style lift | Dropped genre lift | Lost all pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 57727.00 | 8.615970 | 0 | — | — | — | 0 |
| R0o | 65449.00 | 9.768507 | 54762 | 2.408833 | 2.158425 | 3.872986 | 168 |
| R1 | 45505.00 | 6.791791 | 12222 | 2.552881 | 3.178564 | 4.663660 | 265 |
| R1c | 52596.00 | 7.850149 | 5131 | 2.612276 | 4.068719 | 5.096700 | 89 |
| Prank5 | 53454.00 | 7.978209 | 4273 | 2.558220 | 3.632502 | 4.881226 | 106 |
| Prank10 | 50305.00 | 7.508209 | 7422 | 2.544986 | 3.384548 | 4.775966 | 166 |
| Psim50 | 28864.00 | 4.308060 | 28863 | 2.538544 | 3.092730 | 4.863713 | 978 |
| Psim75 | 43295.00 | 6.461940 | 14432 | 2.648631 | 3.807584 | 5.389539 | 241 |
| Psim90 | 51954.00 | 7.754328 | 5773 | 2.803665 | 4.823851 | 5.821118 | 72 |
| Fres025 | 45081.00 | 6.728507 | 42078 | 2.408744 | 2.622971 | 4.360122 | 783 |
| Fres05 | 29093.00 | 4.342239 | 55374 | 2.413544 | 2.340067 | 4.026430 | 3024 |
| Fres1 | 13810.00 | 2.061194 | 57645 | 2.419925 | 2.284246 | 3.951352 | 7300 |
| Forth | 49852.00 | 7.440597 | 40656 | 2.354446 | 2.092494 | 3.735200 | 850 |
| Fside025 | 62746.00 | 9.365075 | 50837 | 2.385879 | 2.035898 | 3.756992 | 180 |
| Fside05 | 60270.00 | 8.995522 | 43923 | 2.364410 | 1.880497 | 3.569383 | 221 |
| Fside075 | 58620.00 | 8.749254 | 30585 | 2.328766 | 1.662877 | 3.219612 | 255 |
| Frank | 47837.00 | 7.139851 | 28277 | 2.444447 | 2.543591 | 4.167749 | 257 |
| Gdown025 | 48560.50 | 8.615970 | 0 | — | — | — | 0 |
| Gdown05 | 51616.00 | 8.615970 | 0 | — | — | — | 0 |
| Gagree | 47644.00 | 7.111045 | 10083 | 2.528984 | 2.764391 | 4.489630 | 162 |
| Gres | 35754.00 | 5.336418 | 35216 | 2.487159 | 2.727024 | 4.457932 | 1360 |

Selected: R0, R1, Psim50, Fres1, Gres.


## Retained generated files

No removals performed. Generated `graphs/*.npz`, `graphs/*.json`, `graphs/sheet.md` and `ags_graphs_*.log` remain local and are not part of the commit. These can be removed later when the controller no longer needs the graph handoff.
