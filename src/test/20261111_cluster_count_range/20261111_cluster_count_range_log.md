# 20261111 cluster-count range: four Table 2 rows on a grid of cluster counts

Written 2026-11-11 (Amsterdam). Plumbing round; module prefix `kr_`. Nothing outside this folder was touched.

## Problem
Table 2 reports clustering AMI on ArtELingo held (12,281 paintings; genre labelled on 9,949) at each aspect's class count
and at 20 clusters. The supplementary shows a grid of cluster counts K: emotion 5, 8, 9, 16, 20; style 10, 20, 27, 40;
genre 5, 10, 16, 20. Four rows lacked the counts outside {class count, 20}: TURTLE on plain Qwen3, TPIPS, CLAY, CRL.
This round scores them with the project's own loaders and scorers, unchanged, after reproducing the 24 recorded cells.

## Code
`kr_grid.py` imports `om_data`, `om_cells` (r18, `20261103_outside_methods`) and `ns_data`, `ns_metrics` (r17,
`20261103_new_scores`). Protocol: `ns_metrics.partitions` = MiniBatchKMeans, n_init 3, batch 4096, random states 0, 1, 2,
float32, thread limit `KMEANS_THREADS['artelingo']`; all 12,281 held rows clustered, AMI on rows with y >= 0
(`ns_metrics.score_partitions`); mean over the three states. Labels: `ns_data.labels('artelingo','held',aspect)`.
One guard: `om_data._atomic` is replaced by a function that raises, because CLAY's loader would write a cache file if its
sidecar did not match (it matched; nothing was written). No vectors were saved.
CLI: `--rows turtle,tpips,clay,crl`, `--only-existing` (the 24 gate cells), `--out`.
Code commit read: files unchanged since 26e0952b; HEAD at run time was 5f6e0c16 (another session committed meanwhile).

## Inputs
- TURTLE: `/data/SSD2/cosir_r18/20261103_outside_methods/outputs/parts/turtle/artelingo/qwen/plain/k{K}__start{0,1,2}.npz` (saved fits, no refit; one fit per K, scored against each aspect), via `om_data.turtle_parts` (checks sidecar, sha256, keys).
- TPIPS: `/data/SSD2/pre_extract/artelingo_full/tpips_8b/{emotion,style,genre}/emb.npy` + keys.json, held rows, via `om_data.tpips`.
- CLAY: `/data/SSD2/cosir_r18/20261103_outside_methods/outputs/clay/artelingo_held/A/{aspect}.npz` (text set A; pick `clay_textset/clu/artelingo` in `.../stage1/om_picks.json`, text pin checked), via `om_data.clay`.
- CRL: CLIP L/14 image store held rows + `/data/SSD2/cosir_r11/20261029_retrieval/outputs/crl/{aspect}_text.npy`, rebuilt in memory by `ns_data.crl`.
- References: `20261103_outside_methods/records/labels.json` (`comp`) for TURTLE, TPIPS, CLAY; `20261103_new_scores/records/verdicts.json` (`comp`) for CRL (K = 20 uses the id form `ami@20/20/crl`).

## Gate
24 cells reproduced. Max |diff| = 0 (bit-identical to the recorded value in every cell). Run twice (gate run and the full-grid run), same values.

| id | reference | ours | diff |
|---|---|---|---|
| SC5/artelingo/qwen/emotion/ami/9/plain_turtle | 0.06685289656462441 | 0.06685289656462441 | 0.0e+00 |
| SC5/artelingo/qwen/emotion/ami/20/plain_turtle | 0.06958172727290952 | 0.06958172727290952 | 0.0e+00 |
| SC5/artelingo/qwen/style/ami/20/plain_turtle | 0.21753700869069328 | 0.21753700869069328 | 0.0e+00 |
| SC5/artelingo/qwen/style/ami/27/plain_turtle | 0.21527952369467815 | 0.21527952369467815 | 0.0e+00 |
| SC5/artelingo/qwen/genre/ami/10/plain_turtle | 0.43603166292491774 | 0.43603166292491774 | 0.0e+00 |
| SC5/artelingo/qwen/genre/ami/20/plain_turtle | 0.42856354194142643 | 0.42856354194142643 | 0.0e+00 |
| SC1/artelingo/qwen/emotion/ami/9/tpips | 0.07182557630263418 | 0.07182557630263418 | 0.0e+00 |
| SC1/artelingo/qwen/emotion/ami/20/tpips | 0.07723914743576017 | 0.07723914743576017 | 0.0e+00 |
| SC1/artelingo/qwen/style/ami/20/tpips | 0.3243210078059277 | 0.3243210078059277 | 0.0e+00 |
| SC1/artelingo/qwen/style/ami/27/tpips | 0.3316347713760315 | 0.3316347713760315 | 0.0e+00 |
| SC1/artelingo/qwen/genre/ami/10/tpips | 0.4961005226813753 | 0.4961005226813753 | 0.0e+00 |
| SC1/artelingo/qwen/genre/ami/20/tpips | 0.4769237419200227 | 0.4769237419200227 | 0.0e+00 |
| SC3/artelingo/qwen/emotion/ami/9/clay | 0.06764750458422945 | 0.06764750458422945 | 0.0e+00 |
| SC3/artelingo/qwen/emotion/ami/20/clay | 0.07286497979852812 | 0.07286497979852812 | 0.0e+00 |
| SC3/artelingo/qwen/style/ami/20/clay | 0.3994693490840693 | 0.3994693490840693 | 0.0e+00 |
| SC3/artelingo/qwen/style/ami/27/clay | 0.40606567571999497 | 0.40606567571999497 | 0.0e+00 |
| SC3/artelingo/qwen/genre/ami/10/clay | 0.4854320772940263 | 0.4854320772940263 | 0.0e+00 |
| SC3/artelingo/qwen/genre/ami/20/clay | 0.46291895869294386 | 0.46291895869294386 | 0.0e+00 |
| SC5/artelingo/clip_l14/emotion/ami/9/crl | 0.07703790450839205 | 0.07703790450839205 | 0.0e+00 |
| SC5/artelingo/clip_l14/emotion/ami@20/20/crl | 0.07857334633972195 | 0.07857334633972195 | 0.0e+00 |
| SC5/artelingo/clip_l14/style/ami@20/20/crl | 0.3873050627860066 | 0.3873050627860066 | 0.0e+00 |
| SC5/artelingo/clip_l14/style/ami/27/crl | 0.39039690415578976 | 0.39039690415578976 | 0.0e+00 |
| SC5/artelingo/clip_l14/genre/ami/10/crl | 0.4282651634165209 | 0.4282651634165209 | 0.0e+00 |
| SC5/artelingo/clip_l14/genre/ami@20/20/crl | 0.432930161896796 | 0.432930161896796 | 0.0e+00 |

## Run times
Gate run (24 cells): 125 s wall, foreground, eight threads (it ran slightly over 2 minutes outside the queue; TPIPS cells 10 to 22 s each). Full grid (52 cells): 180 s inside the script, through `res-queue cpu` (label `paper:kgrid`); TPIPS 148 s, CLAY 27 s, CRL and TURTLE under 15 s each. Log: `logs/full_grid.log`.

## Full grid (mean AMI over the three states; `G` in the last column marks a gate cell: TURTLE, TPIPS, CLAY, CRL)
| aspect | K | TURTLE | TPIPS | CLAY | CRL | gate |
|---|---|---|---|---|---|---|
| emotion | 5 | 0.069 | 0.067 | 0.068 | 0.072 | ---- |
| emotion | 8 | 0.067 | 0.069 | 0.070 | 0.076 | ---- |
| emotion | 9 | 0.067 | 0.072 | 0.068 | 0.077 | GGGG |
| emotion | 16 | 0.068 | 0.080 | 0.075 | 0.077 | ---- |
| emotion | 20 | 0.070 | 0.077 | 0.073 | 0.079 | GGGG |
| style | 10 | 0.206 | 0.274 | 0.387 | 0.380 | ---- |
| style | 20 | 0.218 | 0.324 | 0.399 | 0.387 | GGGG |
| style | 27 | 0.215 | 0.332 | 0.406 | 0.390 | GGGG |
| style | 40 | 0.215 | 0.345 | 0.403 | 0.381 | ---- |
| genre | 5 | 0.401 | 0.446 | 0.447 | 0.397 | ---- |
| genre | 10 | 0.436 | 0.496 | 0.485 | 0.428 | GGGG |
| genre | 16 | 0.433 | 0.482 | 0.471 | 0.425 | ---- |
| genre | 20 | 0.429 | 0.477 | 0.463 | 0.433 | GGGG |

No k-means run left an empty cluster: every state at every K used exactly K clusters (`n_clusters_used` in the JSON).

## Exact values (also in `records/cells.json`)
| row | aspect | K | mean | per state (0,1,2) | gate ref |
|---|---|---|---|---|---|
| turtle | emotion | 5 | 0.06893484519772154 | 0.06862429517806816, 0.06898775481471203, 0.06919248560038443 | '-' |
| turtle | emotion | 8 | 0.06746174154974897 | 0.06747678564967682, 0.06760730876769486, 0.06730113023187523 | '-' |
| turtle | emotion | 9 | 0.06685289656462441 | 0.06873547492270032, 0.06497035940185755, 0.06685285536931536 | 0.06685289656462441 |
| turtle | emotion | 16 | 0.06835191962176657 | 0.06696718089086859, 0.07000930753113428, 0.06807927044329683 | '-' |
| turtle | emotion | 20 | 0.06958172727290952 | 0.06995643704412323, 0.07056990142603989, 0.0682188433485655 | 0.06958172727290952 |
| turtle | style | 10 | 0.2056795942215747 | 0.2025205770997147, 0.2100355440346002, 0.20448266153040928 | '-' |
| turtle | style | 20 | 0.21753700869069328 | 0.21297714841813487, 0.21717413257688312, 0.22245974507706193 | 0.21753700869069328 |
| turtle | style | 27 | 0.21527952369467815 | 0.21373607255875232, 0.212562167784862, 0.2195403307404201 | 0.21527952369467815 |
| turtle | style | 40 | 0.21463960421303382 | 0.2152118170171663, 0.21349262701995272, 0.21521436860198237 | '-' |
| turtle | genre | 5 | 0.40149589768937627 | 0.40179293597943055, 0.4013136981958572, 0.40138105889284104 | '-' |
| turtle | genre | 10 | 0.43603166292491774 | 0.4291976464560248, 0.4276576910585161, 0.4512396512602125 | 0.43603166292491774 |
| turtle | genre | 16 | 0.43286317815472586 | 0.4398357078037887, 0.431637856229448, 0.42711597043094096 | '-' |
| turtle | genre | 20 | 0.42856354194142643 | 0.42869449298587753, 0.433954253711739, 0.4230418791266627 | 0.42856354194142643 |
| tpips | emotion | 5 | 0.0665802147480704 | 0.06012428108938049, 0.0681815910565287, 0.071434772098302 | '-' |
| tpips | emotion | 8 | 0.06878415739890849 | 0.06933505608105467, 0.06974212413729508, 0.06727529197837574 | '-' |
| tpips | emotion | 9 | 0.07182557630263418 | 0.07498916266291636, 0.06658743889686007, 0.07390012734812614 | 0.07182557630263418 |
| tpips | emotion | 16 | 0.08046776086323855 | 0.0803197317486707, 0.08138452483535397, 0.079699026005691 | '-' |
| tpips | emotion | 20 | 0.07723914743576017 | 0.07884813725179103, 0.07706455364285102, 0.07580475141263845 | 0.07723914743576017 |
| tpips | style | 10 | 0.2738503443117061 | 0.2676968877643264, 0.27952457035500233, 0.27432957481578957 | '-' |
| tpips | style | 20 | 0.3243210078059277 | 0.31986689778685573, 0.3302019005235859, 0.32289422510734156 | 0.3243210078059277 |
| tpips | style | 27 | 0.3316347713760315 | 0.3275589634508302, 0.3363342560072231, 0.33101109467004114 | 0.3316347713760315 |
| tpips | style | 40 | 0.34462257429887605 | 0.3483355026758162, 0.3407440436277232, 0.3447881765930889 | '-' |
| tpips | genre | 5 | 0.44591967859386905 | 0.44276952182834045, 0.4458630384322243, 0.4491264755210424 | '-' |
| tpips | genre | 10 | 0.4961005226813753 | 0.48333357801144083, 0.48662808879060887, 0.5183399012420761 | 0.4961005226813753 |
| tpips | genre | 16 | 0.48190010062667693 | 0.49610457612081976, 0.46541309144258514, 0.4841826343166258 | '-' |
| tpips | genre | 20 | 0.4769237419200227 | 0.47833794369288496, 0.4712383187314201, 0.481194963335763 | 0.4769237419200227 |
| clay | emotion | 5 | 0.06759878657910141 | 0.06629615355240306, 0.06965949221108007, 0.06684071397382112 | '-' |
| clay | emotion | 8 | 0.07018342337511706 | 0.07134269727726847, 0.07125352550267923, 0.06795404734540345 | '-' |
| clay | emotion | 9 | 0.06764750458422945 | 0.06673193707674117, 0.0717107230468459, 0.06449985362910128 | 0.06764750458422945 |
| clay | emotion | 16 | 0.07490555707974471 | 0.07549512296053296, 0.07654747850312037, 0.07267406977558077 | '-' |
| clay | emotion | 20 | 0.07286497979852812 | 0.07282092533530202, 0.07157344297416594, 0.0742005710861164 | 0.07286497979852812 |
| clay | style | 10 | 0.3873133958374966 | 0.38902658803165824, 0.39500434168124815, 0.37790925779958334 | '-' |
| clay | style | 20 | 0.3994693490840693 | 0.40208607280862235, 0.39621880468873294, 0.40010316975485255 | 0.3994693490840693 |
| clay | style | 27 | 0.40606567571999497 | 0.39972996045516695, 0.40834343818810553, 0.4101236285167126 | 0.40606567571999497 |
| clay | style | 40 | 0.402954773363933 | 0.4030413799038572, 0.4046691175899812, 0.40115382259796056 | '-' |
| clay | genre | 5 | 0.44726620249474686 | 0.4581842707570553, 0.4336645752457759, 0.4499497614814094 | '-' |
| clay | genre | 10 | 0.4854320772940263 | 0.4874396813157345, 0.4895949921512677, 0.4792615584150768 | 0.4854320772940263 |
| clay | genre | 16 | 0.47109703238641165 | 0.4669393768133013, 0.48241794336261495, 0.4639337769833187 | '-' |
| clay | genre | 20 | 0.46291895869294386 | 0.45742302841713056, 0.47046171767647416, 0.460872129985227 | 0.46291895869294386 |
| crl | emotion | 5 | 0.07205947247582267 | 0.07452631658659653, 0.06848470919582612, 0.07316739164504536 | '-' |
| crl | emotion | 8 | 0.07647203909450978 | 0.07588456114021444, 0.07658695394331487, 0.07694460220000006 | '-' |
| crl | emotion | 9 | 0.07703790450839205 | 0.0739914027873991, 0.07737464264423206, 0.07974766809354498 | 0.07703790450839205 |
| crl | emotion | 16 | 0.0770491903071626 | 0.07800531355161971, 0.07801749442581911, 0.07512476294404895 | '-' |
| crl | emotion | 20 | 0.07857334633972195 | 0.07863076434605779, 0.08009414652878456, 0.07699512814432347 | 0.07857334633972195 |
| crl | style | 10 | 0.38030675852787327 | 0.37711421651578164, 0.3861797226147782, 0.37762633645306015 | '-' |
| crl | style | 20 | 0.3873050627860066 | 0.38644600610825725, 0.38140810004467274, 0.39406108220508984 | 0.3873050627860066 |
| crl | style | 27 | 0.39039690415578976 | 0.39333706551992265, 0.387702244288554, 0.3901514026588925 | 0.39039690415578976 |
| crl | style | 40 | 0.3807521362698026 | 0.37580898030416104, 0.37946947680709514, 0.38697795169815163 | '-' |
| crl | genre | 5 | 0.39740673357364 | 0.39407819878158606, 0.4215259036410239, 0.37661609829831 | '-' |
| crl | genre | 10 | 0.4282651634165209 | 0.4357880196528127, 0.40813825866084036, 0.4408692119359096 | 0.4282651634165209 |
| crl | genre | 16 | 0.42513888700636854 | 0.440545396707094, 0.4126556420221903, 0.4222156222898213 | '-' |
| crl | genre | 20 | 0.432930161896796 | 0.43230713553724304, 0.4336604727798103, 0.43282287737333464 | 0.432930161896796 |

## Addendum 2026-11-11: VLM2Vec-V2 and BERTopic on the same grid
Plumbing only. Two more rows of Table 2 (r25's comparators) on the same K grid. Nothing in `20261111_paper_baselines/`,
`/data/SSD2/cosir_r25/` or the 52 existing cells was written; `records/cells.json` got 26 new cells appended
(the script asserts the old 52 are unchanged; no git baseline, the file is not tracked) and a `header_r25_rows` key.

### Code
- `kr_grid_r25.py` (CoSiR env): imports r25's `pb_score.load_vlm2vec` / `load_bertopic` (embedding sha, keys, meta, partition sidecar checks) and r17's `ns_metrics.partitions` / `score_partitions` unchanged. `--gate` writes `records/gate_r25.json`; `--grid --append records/cells.json` scores both rows.
- `kr_bertopic_fit.py` (`/root/venvs/bertopic/bin/python`): calls r25's `pb_bertopic.run_seed` unchanged, one fresh default BERTopic fit per (UMAP seed 0 to 2, K), `nr_topics` = K + 1 (outliers present), outliers reassigned (distributions, then embeddings). Reads r25's MiniLM cache and `held_docs.json` read only, writes only to the folder given by `--out`.
- VLM2Vec: `/data/SSD2/pre_extract/artelingo/vlm2vec_v2/{emotion,style,genre}/` (emb, keys, meta), k-means = `ns_metrics.partitions` (mini-batch, n_init 3, batch 4096, states 0 to 2), mean AMI over the states.
- BERTopic: MiniLM embeddings `/data/SSD2/cosir_r25/20261111_paper_baselines/outputs/bertopic/minilm.npy`. Partitions at K = 9, 10, 20, 27 are r25's (`topics_k{K}_seed{s}.npz`, legacy `topics_seed{s}.npz` for 20); K = 5, 8, 16, 40 are new, in `/data/SSD2/cosir_paper/20261111_cluster_count_range/outputs/bertopic/topics_k{K}_seed{s}.npz` with a JSON sidecar each (names hash, n HDBSCAN topics, outliers before and after, seconds, versions, npz sha256). A partition at K is scored against every aspect whose grid contains K (K = 10: style and genre; 20: all three; 5 and 16: emotion and genre; 8: emotion; 40: style; 9: emotion; 27: style).

### Gate (all 12 recorded r25 cells, `comp` in `20261111_paper_baselines/records/cells.json`)
Max |diff| = 0.0 (bit-identical), tolerance 1e-6. PASS.

| id | reference | ours | diff |
|---|---|---|---|
| bertopic/emotion/ami/9 | 0.037521540764450294 | 0.037521540764450294 | 0.0e+00 |
| bertopic/emotion/ami/20 | 0.06620297846034479 | 0.06620297846034479 | 0.0e+00 |
| bertopic/genre/ami/10 | 0.26047158886368543 | 0.26047158886368543 | 0.0e+00 |
| bertopic/genre/ami/20 | 0.3092043551789912 | 0.3092043551789912 | 0.0e+00 |
| bertopic/style/ami/20 | 0.09359249418927225 | 0.09359249418927225 | 0.0e+00 |
| bertopic/style/ami/27 | 0.10500601951531363 | 0.10500601951531363 | 0.0e+00 |
| vlm2vec/emotion/ami/9 | 0.06377259570732573 | 0.06377259570732573 | 0.0e+00 |
| vlm2vec/emotion/ami/20 | 0.07100613895961792 | 0.07100613895961792 | 0.0e+00 |
| vlm2vec/genre/ami/10 | 0.4593010201144143 | 0.4593010201144143 | 0.0e+00 |
| vlm2vec/genre/ami/20 | 0.4594993944514876 | 0.4594993944514876 | 0.0e+00 |
| vlm2vec/style/ami/20 | 0.3699943225623416 | 0.3699943225623416 | 0.0e+00 |
| vlm2vec/style/ami/27 | 0.3729743958098484 | 0.3729743958098484 | 0.0e+00 |

### Refit check (K = 10, all three seeds, new fit path against r25's saved partitions)
Refit by `kr_bertopic_fit.py` into `/data/SSD2/cosir_paper/20261111_cluster_count_range/outputs/bertopic_refit_gate/` (scratch, kept).
Labels identical in all 12,281 rows for every seed (0 rows differ, AMI between = 1.0, npz files byte-identical: same sha256). HDBSCAN topics 92 / 87 / 91 and outliers before reassignment 5384 / 5102 / 5134 (seeds 0, 1, 2) equal r25's. BERTopic, UMAP, HDBSCAN arguments and package versions equal r25's sidecars.

| seed | labels identical | rows differing | AMI between | HDBSCAN topics | outliers before |
|---|---|---|---|---|---|
| 0 | True | 0 | 1.0 | 92 | 5384 |
| 1 | True | 0 | 1.0 | 87 | 5102 |
| 2 | True | 0 | 1.0 | 91 | 5134 |

### New BERTopic fits (K = 5, 8, 16, 40)
HDBSCAN found 92 / 87 / 91 real topics and 5384 / 5102 / 5134 outliers (seeds 0 / 1 / 2), the same as in r25's fits; reduction to K always succeeded, the distributions pass reassigned all outliers (0 left, the embeddings pass was not needed), every partition has exactly K non-empty clusters. 12 fits, 8 to 10 s each.

### Run times
Gate: 137 s foreground (CPU queue was busy, exit 75; run under the 8-thread setting without a slot). Refit K = 10: 97 s foreground, same reason. New fits: 173 s inside the script through `res-queue cpu` (label `paper:kgrid2_fit`, log `logs/fit_new_k.log`). Grid scoring: 68 s inside the script through the queue (`paper:kgrid2_score`, log `logs/grid_r25.log`); the 12 gate cells were reproduced again inside it, diff 0.

### Grid (mean AMI over the three states; G marks a gate cell: r25 recorded value reproduced exactly)
| aspect | K | VLM2Vec-V2 | BERTopic | gate |
|---|---|---|---|---|
| emotion | 5 | 0.063 | 0.031 | -- |
| emotion | 8 | 0.066 | 0.036 | -- |
| emotion | 9 | 0.064 | 0.038 | GG |
| emotion | 16 | 0.072 | 0.054 | -- |
| emotion | 20 | 0.071 | 0.066 | GG |
| style | 10 | 0.341 | 0.059 | -- |
| style | 20 | 0.370 | 0.094 | GG |
| style | 27 | 0.373 | 0.105 | GG |
| style | 40 | 0.378 | 0.108 | -- |
| genre | 5 | 0.435 | 0.245 | -- |
| genre | 10 | 0.459 | 0.260 | GG |
| genre | 16 | 0.466 | 0.293 | -- |
| genre | 20 | 0.459 | 0.309 | GG |

Every k-means state at every K used exactly K clusters (`n_clusters_used`).

### Exact values (also in `records/cells.json`)
| row | aspect | K | mean | per state / seed (0,1,2) | gate ref |
|---|---|---|---|---|---|
| vlm2vec | emotion | 5 | 0.06270590058751178 | 0.06321754243579471, 0.062023158241039336, 0.0628770010857013 | '-' |
| vlm2vec | emotion | 8 | 0.06593060283968576 | 0.06497252431626202, 0.0663097245624985, 0.06650955964029671 | '-' |
| vlm2vec | emotion | 9 | 0.06377259570732573 | 0.061442908734598796, 0.06461672203526066, 0.0652581563521177 | 0.06377259570732573 |
| vlm2vec | emotion | 16 | 0.07200360021970541 | 0.06931077841084439, 0.07367859221794491, 0.07302143003032689 | '-' |
| vlm2vec | emotion | 20 | 0.07100613895961792 | 0.0699916185581456, 0.07047770420523088, 0.07254909411547729 | 0.07100613895961792 |
| vlm2vec | style | 10 | 0.3409932893757075 | 0.3309736011714167, 0.34611124623639716, 0.34589502071930855 | '-' |
| vlm2vec | style | 20 | 0.3699943225623416 | 0.36852987724692154, 0.3692524739806496, 0.3722006164594535 | 0.3699943225623416 |
| vlm2vec | style | 27 | 0.3729743958098484 | 0.36864147121951, 0.37626554596577416, 0.37401617024426115 | 0.3729743958098484 |
| vlm2vec | style | 40 | 0.37774930511529486 | 0.3741718286922552, 0.3786016621790315, 0.3804744244745979 | '-' |
| vlm2vec | genre | 5 | 0.43457561401919814 | 0.43938006572585253, 0.4252135731363379, 0.43913320319540394 | '-' |
| vlm2vec | genre | 10 | 0.4593010201144143 | 0.47027126526333674, 0.45289829766715933, 0.4547334974127468 | 0.4593010201144143 |
| vlm2vec | genre | 16 | 0.4664662352024587 | 0.4462365395757638, 0.4806224383114992, 0.47253972772011327 | '-' |
| vlm2vec | genre | 20 | 0.4594993944514876 | 0.47385211452174697, 0.4434016505015452, 0.46124441833117064 | 0.4594993944514876 |
| bertopic | emotion | 5 | 0.03136393563683861 | 0.03568673084897716, 0.02946418641728922, 0.028940889644249474 | '-' |
| bertopic | emotion | 8 | 0.036411964564117584 | 0.04587399551709729, 0.03388127429770504, 0.029480623877550433 | '-' |
| bertopic | emotion | 9 | 0.037521540764450294 | 0.04614398261063758, 0.03407926461502229, 0.032341375067691004 | 0.037521540764450294 |
| bertopic | emotion | 16 | 0.05417006463156083 | 0.06342718829544935, 0.054390297930680356, 0.04469270766855278 | '-' |
| bertopic | emotion | 20 | 0.06620297846034479 | 0.06975928445775437, 0.07261828399086959, 0.05623136693241039 | 0.06620297846034479 |
| bertopic | style | 10 | 0.05873459661495645 | 0.05706532335308226, 0.05930724688381053, 0.059831219607976556 | '-' |
| bertopic | style | 20 | 0.09359249418927225 | 0.0878904844120766, 0.09579861325399135, 0.09708838490174883 | 0.09359249418927225 |
| bertopic | style | 27 | 0.10500601951531363 | 0.08911879970442722, 0.11479020155143604, 0.11110905729007767 | 0.10500601951531363 |
| bertopic | style | 40 | 0.10822640691889363 | 0.09704389021354803, 0.11512217610987034, 0.11251315443326247 | '-' |
| bertopic | genre | 5 | 0.24517615664768322 | 0.23544772518436902, 0.25012982074203444, 0.2499509240166462 | '-' |
| bertopic | genre | 10 | 0.26047158886368543 | 0.267611577413806, 0.2644303254054983, 0.24937286377175194 | 0.26047158886368543 |
| bertopic | genre | 16 | 0.2932866272899837 | 0.2946075695718726, 0.3228404457933832, 0.2624118665046954 | '-' |
| bertopic | genre | 20 | 0.3092043551789912 | 0.30009600467260156, 0.32400608774771505, 0.3035109731166569 | 0.3092043551789912 |

## PercepT on CLIP L/14 (r26), 2026-11-11 20:12

- Problem: the eleventh Table 2 row (PercepT at paper settings on CLIP L/14 with ModernBERT-GoEmotions, r26) had no
  grid cells, so the figure showed 10 of 11 rows.
- Inputs: r26's test probabilities `/data/SSD2/cosir_r26/20261111_percept_original/outputs/held/L1__k40__s{0..4}.npz`
  (key `prob`, [12281, 40], float32). Checked equal to r26's exports `outputs/test/k40/L1__s{seed}.npz`, whose `names`
  equal r17's `ns_data.keys('artelingo', 'held')`. Seeds 0 to 4 (r26 `po_scores.json`, cut k40: 5 in the mean, none
  collapsed).
- Code: `kr_grid_r26.py`, the same scorer as r26's `po_score.score_probs` (`ns_metrics.percept_partitions`, k-means on
  the L2-normalised probabilities, states 0 to 2; masked `ns_metrics.ami`). Cell = mean over seeds of the mean over
  states. Run through `res-queue cpu --short` (18 s). Log: `logs/kr_grid_r26.log`.
- Gate: r26's six recorded cells (class count and k = 20, `po_scores.json` cuts.k40.table) reproduced with
  max |diff| = 0.0. 13 cells appended to `records/cells.json` (row `percept_clip`, header `header_r26_row`), and
  `records/percept_clip.json` (the {aspect, k, mean} list).
- `kr_figure.py`: the row now reads from `records/cells.json` (`KR_ROWS["percept_clip"]`); `--percept-clip` stays as
  an optional extra file. Figure and `figures/values.md` redrawn with all 11 rows; values checked against the grid log.
- Result: ours is still the highest line at every grid count except genre at 20 clusters (teacher 0.5062, ours
  0.5001). PercepT (CLIP L/14) is above PercepT (Qwen3) on style and genre at every count, and on emotion it is below
  it from 8 clusters on (above only at 5).

| Aspect | K | Mean AMI |
|---|---:|---:|
| emotion | 5 / 8 / 9 / 16 / 20 | 0.06011 / 0.06433 / 0.06358 / 0.06674 / 0.06779 |
| style | 10 / 20 / 27 / 40 | 0.23288 / 0.25639 / 0.26218 / 0.26280 |
| genre | 5 / 10 / 16 / 20 | 0.45044 / 0.47444 / 0.46522 / 0.46005 |

## SigLIP 1 lines (the main result moved to SigLIP 1), started 2026-11-12 19:47

Handoff: `docs/superpowers/handoffs/2026-11-12-paper-cluster-counts-handoff.md` (paper chat "paper cluster counts").

### Which saved inputs exist (checked 2026-11-12 19:47)
| Row (SigLIP 1 unless named) | Grid source | Scoring needed |
|---|---|---|
| CoSiR (ours), plain, PercepT (matched) | r17 sweep `/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__siglip1.json` (arms cstar, plain, matched; every grid count; complete, not smoke, held split) | none |
| + TURTLE | r18 saved fits `/data/SSD2/cosir_r18/20261103_outside_methods/outputs/parts/turtle/artelingo/siglip1/plain/k{K}__start{0,1,2}.npz`, all 8 K | AMI of saved partitions, no refit, no GPU |
| CLAY, CRL | r28 job E (`20261112_siglip_main`, tier 1): SigLIP 1 text embeddings in `/data/SSD2/cosir_r28/20261112_siglip_main/clay_crl/text/` | k-means on the held rows, after r28's cells exist |
| Instructed Qwen3, plain Qwen3 | r17 sweep `artelingo__qwen.json` (teacher, plain) | none |
| TPIPS, VLM2Vec, BERTopic, PercepT (CLIP L/14) | `records/cells.json` (their own encoders) | none |

Gate of the r17 SigLIP 1 file: its plain and matched class-count cells equal r17's `verdicts.json` (`SC1/.../plain`,
`SC2/.../matched`) exactly; its cstar class-count cells round to Table 6's 0.099 / 0.441 / 0.528. The final gate
against the new Table 2 cells waits for r28's numbers sheet.

### + TURTLE on SigLIP 1 (`kr_grid_siglip1.py`)
`kr_grid.Grid` with the encoder switched to `siglip1` (module global read at call time); no other change. Gate: r18's six
recorded cells `SC5/artelingo/siglip1/{aspect}/ami/{k}/plain_turtle` (class count and 20), max |diff| = 0.0. Foreground,
under 15 s. Log `logs/kr_grid_siglip1_turtle.log`, cells `records/cells_siglip1.json` (rows `turtle_siglip1`).
TURTLE left one cluster empty in two of three starts at style K = 40 (39, 39, 40 clusters used); at every other K all
starts used K clusters. The other rows' scorers force K clusters; TURTLE's own fits are kept as they are.

| aspect | K | + TURTLE (SigLIP 1) |
|---|---|---|
| emotion | 5 / 8 / 9 / 16 / 20 | 0.0582 / 0.0575 / 0.0567 / 0.0552 / 0.0550 |
| style | 10 / 20 / 27 / 40 | 0.3209 / 0.3257 / 0.3372 / 0.3292 |
| genre | 5 / 10 / 16 / 20 | 0.3306 / 0.3651 / 0.3543 / 0.3468 |

### `kr_figure.py --profile siglip1`
New profile; the default `qwen3` profile is unchanged (replayed into the scratchpad: `values.md` equal, PNG max pixel
diff 0.0). Rows: the expected new Table 2 (r28 handoff: plain, + TURTLE, PercepT matched, CLAY, CRL on SigLIP 1; plain
Qwen3 stays; TPIPS, VLM2Vec, BERTopic, PercepT on CLIP L/14 keep their encoders). Not yet checked against the filled
`Tables/main_clustering.tex`. Legend names follow README 2026-11-12 ("Instructed Qwen3"). Legend 5 columns in 3 rows,
figure 6.875 x 2.7 in (7 columns ran off both edges). Plain Qwen3: light grey dashed, open squares; the other rows
keep their 2026-11-11 styles. Output `figures/siglip1/`. Preview drawn without CLAY and CRL.

### CLAY and CRL on SigLIP 1 (2026-11-12 20:07)
`kr_grid_siglip1.py` now also scores CLAY and CRL: r28's `r28_clay_crl.clay_vectors` / `crl_vectors` (SigLIP 1 image and
text rows; CLAY text set A, the set r28 used for its clustering cell) and `r28_clay_crl.clu_cell` at each K, imported
read only from r28 commit `f94a90d0`. Run through `res-queue cpu --short` (label `paper:kgrid_s1`, 42 s). Gate: 12 of 12
reference cells exact (r18's six + TURTLE cells, r28's six class-count cells in `20261112_siglip_main/records/clay_crl.json`),
max |diff| 0.0. The 13 + TURTLE cells equal the first run. Every CLAY and CRL state used exactly K clusters. Log
`logs/kr_grid_siglip1.log`; `records/cells_siglip1.json` now holds 39 cells (rows `turtle_siglip1`, `clay_siglip1`,
`crl_siglip1`).

| aspect | K | CLAY (SigLIP 1) | CRL (SigLIP 1) |
|---|---|---|---|
| emotion | 5 / 8 / 9 / 16 / 20 | 0.0651 / 0.0703 / 0.0691 / 0.0761 / 0.0773 | 0.0670 / 0.0793 / 0.0747 / 0.0809 / 0.0803 |
| style | 10 / 20 / 27 / 40 | 0.4095 / 0.4327 / 0.4371 / 0.4324 | 0.3293 / 0.3650 / 0.3750 / 0.3755 |
| genre | 5 / 10 / 16 / 20 | 0.4589 / 0.5042 / 0.4934 / 0.4773 | 0.3093 / 0.3935 / 0.4041 / 0.4164 |

Class-count cells against r28's own records: ours 0.0985 / 0.4408 / 0.5279 and plain 0.0612 / 0.3820 / 0.4016 equal
r28's held replays (r28 log 19:58 and 20:02); PercepT (SigLIP 1) equals r17's `verdicts.json`; CLAY, CRL, + TURTLE as
above. r28's numbers sheet (`paper_cells.md`) did not exist yet at this time.

### The figure with all 12 rows (`figures/siglip1/`, log `logs/kr_figure_siglip1.log`)
Ours against the best other line at each K (exact means):

| aspect | K | ours | best other | margin |
|---|---|---|---|---|
| emotion | 5 / 8 / 9 / 16 / 20 | 0.0883 / 0.0979 / 0.0985 / 0.1064 / 0.1071 | Instructed Qwen3 at every K | +0.0047 / +0.0080 / +0.0097 / +0.0130 / +0.0150 |
| style | 10 | 0.4094 | CLAY 0.4095 | -0.0001 |
| style | 20 / 27 / 40 | 0.4356 / 0.4408 / 0.4478 | CLAY 0.4327 / 0.4371 / 0.4324 | +0.0029 / +0.0037 / +0.0154 |
| genre | 5 | 0.4762 | Instructed Qwen3 0.4629 | +0.0133 |
| genre | 10 / 16 | 0.5279 / 0.5182 | CLAY 0.5042 / 0.4934 | +0.0237 / +0.0247 |
| genre | 20 | 0.5065 | Instructed Qwen3 0.5062 | +0.0003 |

Overlaps in the drawing: CLAY runs under ours in the style panel from 10 to 27 clusters (within 0.004); in the genre
panel CLAY meets the instructed Qwen3 at 10 and 16 and TPIPS at 20. Open with the user: caption note or inset.

### Gate against r28's numbers sheet (2026-11-12 20:18)
`20261112_siglip_main/paper_cells.md` (written 20:16), section Table 2: all 11 rows equal the figure's class-count values
at three decimals (plain, + TURTLE, CLAY, CRL, PercepT matched and ours on SigLIP 1; Instructed Qwen3, TPIPS, BERTopic,
VLM2Vec, PercepT on CLIP L/14 unchanged). Ours exactly: `records/per_seed.json` `ours_siglip1/ami/{aspect}/mean` minus the
r17 SigLIP 1 sweep's cstar mean = 0.0 for all three aspects. The sheet maps the plain row "Qwen3 -> SigLIP 1"; the figure
also keeps a plain Qwen3 line (0.070 / 0.235 / 0.425, the current .tex L9), as the r28 handoff says plain Qwen3 stays in
Table 2. Remove that line if the filled Table 2 drops the row.

### CLAY drawn over ours; checked against the filled Table 2 (2026-11-13)
Check: at each aspect's class count, the 12 lines equal the filled `Tables/main_clustering.tex` (REPO `de7731c`) in
36 of 36 cells (script compare at three decimals; legend names map to the table rows). `kr_figure.py`: in profile
siglip1 only, CLAY gets z 21 (over ours, z 20), so its markers show where it meets ours in the style panel (user,
option b). `values.md` unchanged by the redraw. Figure copied to REPO `Figures/fig_supp_cluster_counts.pdf` (`a7d2ef9`).
