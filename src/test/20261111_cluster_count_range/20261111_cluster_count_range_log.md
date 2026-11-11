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
