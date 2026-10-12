# Decision rule: round 8, stages 0, 1, E and the seed-42 gate of both tracks (main line and encoder screen), before any code

> Specs: A (main line) `docs/superpowers/specs/2026-10-12-r8-learned-grouping-design.md` @ 31f41dc (approved
> 2026-10-12 19:47), SHA-256 83ff81bb320570b0b7d8db088ea88a289c73be39cf00f42cf15083b4ccd1d2b4; B (encoder screen)
> `docs/superpowers/specs/2026-10-12-r8-encoder-screen-design.md` @ fc4ef0e (approved 2026-10-12 19:52), SHA-256
> e85981161d5ae2a80fad0265562ced341ae173141176a13f88edef7ed2f912b1 · Written 2026-10-12 from 20:05 (Amsterdam) before
> any implementation script of this round exists; review fixes applied from 2026-10-12 20:26 · Reviewer: fresh Opus
> reviewer, `rule_check/opus_rule_check.md` (fixes: `rule_check/fixes.md`)

**Precedence.** Where this file differs from spec A, spec B or round 7's rule, this file governs (C7; A:D33, B header).
Round 7's rule (`src/test/20261126_design_l_style_source/DECISION_RULE.md`, SHA-256
204a0ac5d0f35afe6cd490662036c35710a8180e02802b3a80f7c45a2c38647d; "R7") supplies L1 to L16, its stage-0 meaning of
"exactly" (§4 G3), its §9 read-only and re-derivation practice and its §10 reused-code table, unchanged except where
this file says. Through R7 come R3 (SHA-256 2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925), R6
(7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724) and R1rule
(613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c) as R7's precedence line cites them. Constitution
version 2 (SHA-256 bdd66f44484ec89093be3434b06f76c37b32a3bfcecb6c4055b4a8644dff516b): C9, C11, C12, C13 apply except
§11. IDs: R7's L-items keep their meaning; this file's new definitions are M1 to M16; A's D/SC and B's ED/ESC are cited
as such. Stage 3 (A) and B's stage 2 (both held reads) are governed by `HELD_RULE.md`, written after
`gate_verdict.json` and before any held-read code (A:D33); this file covers no held number.

**Authorisation.** The user's decisions in the `r8 decide` and `r8 decide 2` grill of 2026-10-12 (direction log entry
2026-10-12; A:D1 to D13, D36; B:ED1, ED3, ED4, ED6, ED12, ED14 to ED16, ED18, ED19) and both specs' approvals. After
its commit this file changes only with the user's approval, as an amendment row in spec A, marked before or after
results. The controller commits; agent defaults set by this file are marked "(agent default)".

**Status.** Seed 42 is development data (C4). No held row and no held image is read under this file (C5; B:ED7).

## 1. Prior (C8) and disclosure

Written before any number of this round; copied from A §3.1 and B §3.1.
- **Main line, groupings:** AMI with the content grouping below raw CSD's 0.40, lowest at 100% own-group negatives
  and at the largest FCMI weights; at those settings AMI with style falls too (round 7's projection: style 0.10). No
  source says whether views or neighbours keep more style.
- **Main line, gate:** about 15% that one of the 24 passes (AFF is only +0.33 over B′(A1); every round 7 scorer with
  a fourth grouping, random controls included, failed S3 on emotion × genre). Extended AFF more likely than R1 to pass.
  Promising (D30) about 30%. Given a pick: Tier 1 about 35%, Tier 3 about 20%, Tier 2 about 10%; Tier 1 overall about
  5%. Surprises: a random control passing; R0 with own assignment passing; FCMI far above (ii).
- **Screen:** gate pass of any of the six about 8 to 10% (raw CSD with a CLIP-trained reader lost 1.11 and 1.28 to
  AFF on emotion × genre against the bar −0.3; DINOv2-L 40.0 against CSD's 63.1 mAP@1 on WikiArt style retrieval).
  Tier 1 given a pick about 30%, about 3% overall. Surprises: a random control passing; DI or SG passing where AL fails.
- **Disclosure (A §8, B §8):** on seed 42, about 50 variants before AFF, about 10 in rounds 4 and 5, round 7's 12
  candidates; now 24 (26 with NBC) main-line candidates and 6 screen candidates, uncorrected (§11). References and
  controls pick nothing. A pick's seed-42 margin is inflated; the held read protects the verdict. Held rows were read
  before (H1 to H3, H5); this design follows H5's result.

## 2. Candidates, references, controls, comparators

| ID (file ID) | What | Track | Role |
|---|---|---|---|
| V100, V67, V33 | (ii) head, view positives, 100 / 67 / 33% own-content-group negatives (M4, M6) | main | candidates |
| NB100, NB67, NB33 | (ii) head, positive = nearest CLIP neighbour in another content group (M5, M6) | main | candidates |
| FC1 to FC6 | FCMI, fairness weight 0.05, 0.10, 0.20, 0.40, 0.80, 1.60 (M7) | main | candidates |
| NBC | (ii) head with the nearest CSD neighbour, only if triggered (§6.3) | main | extra candidates |
| AL, DI, SG | ALADIN, DINOv2-L, SigLIP 2 embeddings → 17 groups (M10) | screen | candidates |
| R0own | R0 (R7 §3 L7's R0 file) with own assignment in CSD space; shared by both tracks, run once | both | reference |
| NRK | NrKmeans on CLIP painting vectors (M8) | main | reference |
| TP | told prompts, CLIP zero-shot with the 27 style names (M9) | main | privileged reference |
| Z\<X\> (e.g. ZV100, ZAL) | random grouping with X's group sizes, one per candidate grouping (M13) | both | control |
| H\<X\> | the CLIP-head arm (§6.6) | main | descriptive |

Every grouping X yields two scorers R1[X] and X-AFF[X] (file names `R1_<X>`, `XAFF_<X>`, as R7). Candidates: 12
groupings × 2 = 24 main-line scorers (+2 with NBC); 3 × 2 = 6 screen scorers. Comparators of every scorer s, on seed
42 with the 224-cell family and per-half cross-fit: AFF (gate, C3), B, B0 = B′(A0), B1 = B′(A1), BX = B′(A0+X) and
CF[s] (C2), as R7 L13 to L15. R7's R0 numbers with CLIP heads stand beside R0own.

## 3. Definitions (M1 to M16)

**M1. Paintings, joins, halves.** R7 L1 unchanged: 36,518 scorer-train paintings; painting index p = position in round
1's `rb_halves.npz` `paintings_sorted` (= R7's `g_st` code, ascending `artelingo_splits().groups` code; not
painting-id string order); every painting-ordered array (views, φ, n(p), M6's permutation, M13's scorer-train draw)
uses it ("M1 order"). A painting's vector in any source is that of its first scorer-train row; image-level sources
join only through `style_csd_vitl/row_to_image.npy` with R7's hash checks (C6). The two painting halves are round 1's,
as in R7's `xheads_*.npz` (R1rule §4.2 item 2). "Unit" means L2-normalised in float32. The CLIP painting vector x_p is
the unit cached `img_features` (openai/clip-vit-base-patch32, §10) of p's first scorer-train row.
**GPU passes** (M4, M7, M10; agent default): `torch.backends.cuda.matmul.allow_tf32 = False`,
`torch.backends.cudnn.allow_tf32 = False`; the batch size is recorded.

**M2. Content grouping.** R7 L2: `partitions.npz` key `image`, 64 groups, c(p) the group of painting p (smallest group
322 paintings, largest 992; asserted ≥ 129 so own-group negatives draw without replacement).

**M3. 17-group recipe and resolution search (A:D15, B:ED8).** Painting vectors φ (unit) → `run_sweep.knn_union_graph`
with k = 20, built once per grouping → `leidenalg.RBConfigurationVertexPartition` at resolution r, seed 42 →
`run_told_oracle.merge_small` at 200 rows in φ's space (rows inherit their painting's vector), as R7 L6. K(r) = the
number of groups after merging. Search (agent default): (1) r = 1.0; stop if K = 17. (2) The grid r = j/10, j = 1 to
30; among grid points with K = 17, the one nearest 1.0, ties to the smaller r. (3) If none, the adjacent grid pair
whose K values bracket 17 with midpoint nearest 1.0 (ties to the smaller) is bisected at most 10 times, keeping the
half that still brackets 17; stop at the first K = 17. (4) If none, the nearest count: among all evaluated r, the
smallest |K(r) − 17|, ties to r nearest 1.0, then the smaller r; K is reported. If `merge_small` raises `SystemExit`
(no group reaches 200 rows), K(r) is undefined: that r is skipped and recorded (agent default). Every evaluated
(r, K) pair is recorded; seeds 43 and 44 at the chosen r are descriptive (R7 L6). Applies to V*, NB*, NBC, AL, DI,
SG. R0own uses R0's frozen 17 groups (no search). A grouping with fewer than 2 groups is dropped (R7 L7).
**Group counts.** A §3.2's "17 groups" is met where the recipe allows: TP keeps its own count (≤ 27; A:D24; a
privileged reference), and FC* or NRK may end below 17 after M7's or M8's merge. Every final K is reported (agent
default; listed for a spec note).

**M4. Views (A:D7, D19) and the view check (A §5).** For each scorer-train painting, its image in
`artelingo_clip224/images_uint8.npy` (joined by `paintings.json` and painting id, hash-checked, C6). Five views in
this order: horizontal flip of the full 224 image; the 160 × 160 top-left, top-right, bottom-left, bottom-right crops,
each resized by
`torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)` on float32
pixels /255, then clamped to [0, 1] (agent default). CLIP mean and std, `get_image_features` of
openai/clip-vit-base-patch32 (the cached features' weights, §10), float32, local GPU under the lock. Stored as
`/data/SSD2/pre_extract/artelingo/r8_clip_views/views.npy`, [36,518, 5, 512] in M1 order, with a `meta.json` (joins,
hashes, view order, the painting-id string per row, asserted equal to `data.paintings` of p's first scorer-train row).
View check (agent default): before the views, the unchanged cached image of the first 256 scorer-train paintings must
give cosine ≥ 0.9999 with its cached feature each; a failure drops V100 to V33 (reported as a fair mechanic), and NB,
FC go on.

**M5. Neighbours.** n(p) = argmax of x_p · x_q over scorer-train paintings q with c(q) ≠ c(p) (float64; ties to the
lowest painting index); fixed before training. NBC: the same with unit CSD painting vectors in place of x.

**M6. Head (ii) (A:D19).** f = Linear(512, 512) → ReLU → Linear(512, 128) → unit; PyTorch default initialisation after
`torch.manual_seed(0)`; CPU, 8 threads, `torch.use_deterministic_algorithms(True)` (agent default). Input x_p. One
`rng = numpy.random.default_rng(0)` drives, per epoch, `rng.permutation(36518)` (M1 indices), then per batch: views
`rng.integers(0, 5, size=len(batch))`; then per query in batch order `rng.choice(own_pool, n_own, replace=False)`,
then `rng.choice(other_pool, 128 − n_own, replace=False)`; pools are ascending M1 indices without p and n(p) (M5's
CLIP neighbour for V and NB, the CSD neighbour for NBC; agent default). Epoch = the permutation in batches of 512,
last partial batch kept (72 batches). Positive: V forms f(unit view v), v uniform on the five views per query per
step; NB forms f(x_n(p)). Negatives: 128 per query, n_own = round(128 · s) = 128, 86, 42 for s = 1.00, 0.67, 0.33 from
{q : c(q) = c(p)} and 128 − n_own from {q : c(q) ≠ c(p)} (the draws above); embedded as f(x_q) with gradient. Loss:
InfoNCE, mean over the batch of −log softmax([f(x_p)·pos, f(x_p)·f(x_q1), …]/0.1)[0]. Adam, lr 1e-3, torch defaults
otherwise, 30 epochs. φ = f(x) of every painting after training; the last epoch's mean loss is recorded.

**M7. FCMI (A:D20, D21).** Code: `/data/SSD2/third_party/FCMI` @ 9750ccfc91a64c10364977302fc53875ed39a23f; `NetFCN`
(`networkExplicitDecoupleShuffle.py` l.839) and `Net.run`'s training loop copied into an `r8_` module, with the
display, scoring and data imports dropped and nothing else changed except as listed here. Settings: `main.py` argparse
defaults (l.20 to 82), the decoder branch of the code's default feature-vector dataset `Office` (`ActivationType`
default Tanh; agent default), input_dim 512, group_num 64, class_num 17, `LambdaFair` = w with FC1 to FC6 at w = 0.05,
0.10, 0.20, 0.40, 0.80, 1.60 (0.25 to 8 × the default 0.20, l.82; `LambdaClu` stays 0.04), seed 0, 300 epochs, batch
512, `GroupWiseDecoder` 1, shared encoder, deterministic algorithms as `main.py` l.114, run with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (as the authors' command line, `main.py` l.155). `MainLauncher` is not imported
(probe §1); the `r8_` copy sets its values `LimitKmeans = True`, `CodeTest = False`, `UnDeterministic = False`,
`DrawMax = 10000` (`MainLauncher.py` l.13 to 21); faiss `Clustering` keeps its default seed. Data: x_p, protected
group c(p), a dummy class 0; training loader shuffled by a torch generator with seed 0; test loader = the 36,518
scorer-train paintings in M1 order (agent default; it feeds the per-epoch k-means, so no selection or held image
enters the fit, A §3.3, C5). The per-epoch `faiss` k-means (l.81 to 96) is kept as written (`faiss.Clustering`, niter
300, nredo 10) with `faiss.IndexFlatL2(d)` in place of `GpuIndexFlatL2` (faiss-cpu 1.15.1 in the CoSiR env; the GPU
build is not installable; agent default). Checkpoint and `NpPoints` writes go to `/data/SSD2/cosir_r8/fcmi/<run>/`
(about 1.7 GB per checkpoint; agent default). Device: the local GPU under the lock, one run at a time (agent default;
it makes D21's 60 minutes realistic; it departs from A §9's CPU budget by up to 6 GPU hours, a mechanic). Output: φ of
any image i (scorer-train painting or selection-row image) = unit `encoder_linear[0](encoder[0](x_i))` in `net.eval()`
under `torch.no_grad()` after training (the shared encoder needs no group), the 17-value representation where FCMI
clusters (agent default for A §3.2's "hidden features"); labels = argmax of `encode_class` after a final inference and
centre update; empty clusters removed and labels compacted; then `merge_small` at 200 rows in φ's space (agent
default). **Wiring smoke** (agent default): one run before stage 1, 2 epochs on the first 2,048 paintings in M1 order,
outputs under `results/smoke/`; it decides nothing (no SC2 decision) and no number from it is reported.
**SC2 / D21 test** (on any FCMI run except the wiring smoke): fails if the process raises or exits non-zero, any step's total loss is not
finite, the final labels hold fewer than 2 distinct clusters before merging, or the wall time from model construction
to final labels exceeds 60 min (lock wait excluded). Stage 1 is the FC3 run (w = 0.20), reused as FC3 if it passes.

**M8. NRK (A:D23).** R7 L7's N recipe (ClustPy 0.0.3 `NrKmeans(n_clusters=[17, 17], random_state=0)`, disposable
child, 30-minute limit, dropped past it) on the 36,518 x_p; kept clustering = lower AMI with the content grouping over
the scorer-train rows (label-free; ties to the first); merge at 200 rows in its subspace. Its space: unit
`transform_subspace(x, s)` = x @ `V_[:, P_[s]]` (`nrkmeans.py` l.944; attributes `V_`, `P_`, l.860 to 862) for the
kept subspace s (agent default). The `r8_` worker also saves `V_` and `P_` and asserts `labels_.shape == (n, 2)` (R7's
worker saved labels only). No random control.

**M9. TP (A:D24).** Prompts "a {style} painting" for the 27 ArtELingo style names = the sorted top-level folders of
the `paintings.json` image paths (not `encode_labels(data.art_styles)`, which reads label columns of all rows, held
included); underscores as spaces; the exact strings recorded (agent default). CLIP B/32 text features of the cached
features' weights (§10), unit; text check (agent default): the captions of the first 256 scorer-train rows,
re-encoded, give cosine ≥ 0.9999 with their cached `txt_features` each, else TP is dropped (M16). Map: argmax of x_i ·
t_s (ties to the lowest index); groups under 200 scorer-train rows merged by `merge_small` in x's space; the gallery
map goes through the same merge map. TP keeps its count (M3). TP's `post_TP` img and its bank inputs = the TP map
(argmax, then merge map) of the row's image, the same in both halves (M12's format). No item label is read.

**M10. Encoders and ESC1 (B:ED5, §3.4, §6).** Images: the CSD image indices referenced by non-held rows through
`row_to_image.npy`, asserted to number 49,121 and to equal the 224 cache's image set; ascending CSD order; files from
CSD's `meta.json` under `/data/PDD/wikiart_proj/wikiart`. AL: `/data/SSD2/third_party/ALADIN` @ 3ddd112, weights
`aladinVIT_bamfg_64.47.pt`, the README's only released weights: ALADIN-ViT from StyleBabel (BAM-FG), not the ICCV 2021
VGG model; used as ED5's "official weights from its README" (agent default; the report names the model so).
`main.AViT.encode`, 256 values, images loaded by `main.load_img` and its `main.py` preprocessing (alpha on white,
Resize((224, 224)), mean and std 0.5). DI: `facebook/dinov2-large` @ 47b73ee, `PIL.Image.open(p).convert("RGB")`, its
processor, `pooler_output` (CLS after the final norm), 1024 values. SG: `google/siglip2-so400m-patch14-384` @ e8e4872,
the same loading, its processor, `get_image_features`, 1152 values (loading: agent default). Float32, eval mode,
local GPU under the lock; stored raw in `/data/SSD2/pre_extract/artelingo/r8_enc_{aladin,dinov2l,siglip2}/` with a `meta.json`
(indices and their SHA-256, weight SHA-256, revision, preprocessing). **ESC1** per encoder, all must hold: (a) strict
weight load and the weight SHA-256 of §10; (b) shape (49,121, d); (c) all values finite; (d) the SHA-256 of the stored
index array (contiguous int64) equals the one computed the same way from `row_to_image.npy` at the start; (e) the
first 200 images embedded again in a separate pass after the full pass give cosine ≥ 0.9999 each with the stored rows.
φ = unit embedding of the painting's image; grouping by M3.

**M11. Own map and centres (A:D16).** Spaces φ: M6 head output (V*, NB*, NBC), M7 representation (FC*), unit CSD
(R0own), M8 subspace (NRK), M10 embedding (AL, DI, SG). Centre μ_g = unit(mean of φ_p over the scorer-train paintings
of group g of the final grouping). Map m(i) = argmax_g φ_i · μ_g (float64; ties to the lowest group). Posterior =
one-hot, float32. `post_<X>.npz` keeps R7's format (img, txt, rows = selection; NaN outside): img = the own map of
each selection row's image with centres from all scorer-train paintings; txt = X's CLIP text head (M14). R0own's txt
is R7's `post_R0.npz` txt (same grouping and recipe; agent default).

**M12. Cross-fitted bank inputs (A:D17).** `xheads_<X>.npz` in R7's format (rows = scorer-train): img of a row whose
painting is in half h = one-hot argmax_g φ_p · μ_g^(1−h), with μ^(1−h) from the half-(1 − h) members of each group
(labels of the full grouping; φ not refit); txt = R7 L10's cross-fitted text head (R0own: R7's `xheads_R0.npz` txt).
For every X (TP and Z\<X\> included): `classes_half{0,1}` = arange(K), `filled_by_half` = 1 − half. A group with no
member in a half drops X, as R7 L11 (agent default).

**M13. Random controls (A:D18, B §3.2).** Index i of X: V100 1, V67 2, V33 3, NB100 4, NB67 5, NB33 6, FC1 to FC6 7 to
12, NBC 13, AL 14, DI 15, SG 16. Labels: R7 L8 with `default_rng(7100 + i)` and X's group sizes in paintings. Text
heads by M14 on these labels. Image side (agent default for the banks): shares =
`bincount(painting labels of X, minlength=K) / 36518` (float64); scorer-train paintings (bank inputs, the same draw
in both halves): `default_rng(7200 + i).choice(K, size=36518, p=shares)` in M1 order; selection-row paintings:
`default_rng(7300 + i).choice(K, size=n_sel, p=shares)`, the n_sel paintings in ascending
`artelingo_splits().groups` code over the selection rows; a row takes its painting's draw; posterior one-hot. A
dropped X has no control.

**M14. Heads, readers, thresholds, scorers.** R7 L10 to L15 unchanged on configuration A0+X: L10's image and caption
heads fit on X's labels (the image head serves only H\<X\> and diagnostics); L11's reader on M12's banks; τ^X = L12
(A:D25); L13's scorers; L14's B, B0, B1 and BX (BX from `post_<X>`); L15's bar comparator and gain statistic (for
promising). The only code change to R7's reader and bundle paths is the grouping-ID whitelist, in `r8_` copies.

**M15. Integer form.** R7 L16: N = 12,288 pooled, 4,096 per pair; intervals are C9's (`cluster_bootstrap`,
anchor-painting clusters `cl`, 5,000 resamples, seed 42, percentile 95%).

**M16. Drops.** A dropped grouping (fewer than 2 groups, D21, ESC1, M4's check, M8's limit, M9's text check, a missing
head or bank block per R7 L11, M12) has scorers that count as not passing; the reason is recorded.

## 4. Stage 0: regression check (SC1, C12; shared by both tracks)

Before any seed-42 number of a round-8 grouping. Views, extraction (M4, M10) and M7's wiring smoke may run alongside;
nothing else. Through the round-8 code path on seed 42, each must hold exactly (R7 §4 G3's meaning:
`numpy.array_equal` for integer and per-anchor R@1 arrays; float arrays within 1e-12 absolute; 8 threads, scikit-learn
1.6.1, numpy 2.2.6):
- **SC1a, AFF:** fused cells 39 and 119, counterpart cells 149 and 10, σ* = 0 on both halves; R3's `seed42_arrays.npz`
  arrays `aff_fused_cells`, `aff_cf_cells`, `aff_sigma`, `aff_gate__a`, `aff_gate__b`,
  `aff_fused__{r1,gain,other,strict,swap}`, `aff_cf__{r1,gain,other,strict,swap}` (as R7 `run_r7_stage0.py` l.166 to
  173; `aff_bar_v` is not compared); every `AFF__*`, `AFF_cf__*` array of R7's `seed42/arrays.npz`.
- **SC1b, R1[R0] and X-AFF[R0]** from R7's `heads/{post,xheads}_{affect,image,caption,R0}.npz`: the reader rebuilt by
  the round-8 reader code gives R7's `readers/bank_R0_half{0,1}.npz`, the chosen C of `reader_R0.json`, and the
  `P__a`, `P__b`, `m__a`, `m__b`, `pick__a`, `pick__b` of R7's `seed42/reader_R0.npz`; τ^R0 equals `taus.json` key R0;
  every `R1_R0__*`, `R1_R0_cf__*`, `XAFF_R0__*`, `XAFF_R0_cf__*` array equals R7's `seed42/arrays.npz`.
- **SC1c:** `B__*`, `B0__*`, `B1__*`, `BX_R0__*`, `cl`, `pair_index`, `parity` equal R7's `seed42/arrays.npz`.
- **SC1d, head path (C12):** the round-8 head code refits R0's L10 caption head and cross-fitted text heads from R7's
  R0 labels (`r7_groupings.npz` key R0); `txt` equals R7's `post_R0.npz` and `xheads_R0.npz` `txt` within 1e-12.
- **SC1e, recipe (C12):** M3's code at r = 1.0 on unit CSD painting vectors gives `r7_groupings.npz` key R0 exactly
  (17 groups).
- Written to `results/stage0_regression.json` (each comparison's max difference, pass, the §10 SHA-256s at start and
  end, this file's SHA-256, the time). A failure stops both tracks (§8).

## 5. Stage 1 (SC2, main line) and stage E (ESC1, screen)

- **Stage 1:** the FC3 run of M7. SC2 passes iff M7's D21 test passes. Fail: FC1 to FC6 drop out (12 scorers not
  passing; no ZFC*); (ii) goes on. Written to `results/stage1_fcmi.json` (the four tests, wall time, device), with the
  per-step total-loss log and the final labels before merging stored beside it for §7(d).
- **Stage E:** each encoder by M10; ESC1 decides per encoder; a failed load or a failed item drops it (B:ED4); all
  three dropped ends the screen (Kill). Written to `results/stageE_<AL|DI|SG>.json`.

## 6. Stage 2 (main) and the screen's stage 1: the seed-42 gate

### 6.1 Per scorer (A:SC3, SC4; B:ESC2, the same gate)
With d = s − AFF per anchor (fractions; M15): **SC3** pooled over 12,288 episodes, 95% lower bound > 0 (strictly);
**SC4** Σ(4d) ≥ −49 on the 4,096 emotion × style episodes and on the 4,096 emotion × genre episodes.
**Pass = SC3 and SC4.** Recorded also, for promising and ties: R7's S1 (bar margin vs L15's comparator, Σ(4d) ≥ 246,
bar-margin and gain lower bounds > 0), S2 (style × genre, s − CF[s], Σ(4d) ≥ 0), S3 (= SC4), and the gain sum
Σ4(s.gain − CF.gain) pooled.

### 6.2 Who is evaluated
Candidates of both tracks; references (R0own, NRK, TP) and controls (Z\<X\>) get the same quantities in separate keys
and pick nothing. Labels and AMIs never enter §6.

### 6.3 NBC trigger (A:D8, D22)
After the 24 main-line scorers are evaluated: triggered iff none of the six NB scorers passes. Strength = that of the
NB scorer, among evaluated NB scorers, with the largest pooled Σ(4d) of s − AFF (ties: larger gain sum, then NB100,
NB67, NB33, R1 before X-AFF); if no NB scorer was evaluated, NBC does not run (recorded). NBC = M6 with M5's CSD
neighbour at that strength, grouping by M3, its control ZNBC, both scorers; it joins the candidates. Written to
`results/probe_trigger.json` (the six NB records, triggered, strength, this file's SHA-256, the time) after its C13
check (§7) and before any NBC code runs.

### 6.4 Pick per track (A §6 selection rule, D26; B:ED13)
Among passing candidates of the track: the largest pooled Σ(4d) of s − AFF; ties to the larger gain sum; then the
order V100, V67, V33, NB100, NB67, NB33, FC1 to FC6, NBC (main) or AL, DI, SG (screen), grouping first, R1 before
X-AFF within a grouping. At most one pick per track.

### 6.5 Flags and promising
- **Random-control flag (A:D27):** if Z\<X\> of the pick's grouping passes SC3 and SC4 with the pick's scorer kind,
  the report's first lines say so; the pick stands. "Not evaluable" if Z\<X\> is dropped.
- **Promising (A:D30; main line only):** no main-line candidate passes, and at least one passes R7's S1, S2 and S3
  (§6.1). No dig before the cut-off. The screen has no promising band (B §2).

### 6.6 CLIP-head arm (A:D28; descriptive; main line only)
After `gate_verdict.json`: for the main pick, or with no pick the main-line candidate with the largest pooled Σ(4d) of
s − AFF (ties as §6.4): X's L10 image and caption heads (CLIP features), R7's cross-fitted heads, its own reader and
τ, the pick's scorer kind; ID H\<X\>. Reported beside the pick with §6.1's quantities; it decides nothing.

## 7. `gate_verdict.json` and re-derivation (C13)

- **Content** (`results/gate_verdict.json`, one file, keys `main` and `screen`): per scorer the §6.1 quantities
  (integer sums, points, intervals, passes), dropped groupings with reasons, M3's (r, K) records, the trigger record,
  the pick, the random-control flag, promising (main), the H\<X\> target, the SHA-256s of `stage0_regression.json`,
  `stage1_fcmi.json`, `stageE_*.json` and `probe_trigger.json`, this file's SHA-256, `smoke: false`, the time.
- **Re-derivation before `probe_trigger.json` and before `gate_verdict.json`:** code written separately from the
  pipeline (a Codex job or a fresh subagent) recomputes from the stored arrays (a) the own-map one-hots of the
  selection rows and the M12 bank inputs from the stored φ, labels and halves, and M13's control draws from their
  seeds; (b) R7 §9 phase A for every candidate, reference and control (gates from stored reader probabilities and τ^X,
  224-cell statistics, σ*, both cross-fits, per-anchor arrays of s, CF[s] and BX); (c) §6.1 to §6.5 with its own
  bootstrap call; (d) from the stored files, ESC1 (b) to (d) per encoder and SC2's label count and finite-loss log.
  Integer arrays, sums, passes, the trigger, picks and flags must agree exactly; intervals within 1e-9; a disagreement
  stops that step until traced. Groupings, heads and builder fits are not re-derived (their checks: M3's records, M7's
  other tests, ESC1 (a) and (e), SC1d, SC1e, C6 hashes).
- **Report flags (agent defaults the report names):** ALADIN-ViT from StyleBabel as ED5's ALADIN (M10); final K per
  grouping, TP ≤ 27 and FC* or NRK below 17 after merging (M3); a V100-to-V33 drop on M4's check; φ of FCMI as its
  representation and FCMI on the GPU (M7). Each is a fair mechanic, not a design change.
- **After it, not before:** label-based diagnostics (AMIs with style and genre, own-map accuracy against labels),
  H\<X\>, and A §3.6 / B §3.6's descriptive numbers. Label-free AMIs with the content grouping may run earlier.

## 8. Outcome-to-action table

| Outcome | Action |
|---|---|
| SC1 fails | Stop both tracks before any candidate number; trace the cause; report and phone note |
| SC2 fails | FC1 to FC6 out; (ii) goes on |
| ESC1 fails for an encoder / for all three | That encoder out / the screen ends (Kill) |
| Main pick exists | Phone note with the gate verdict (no stop, D36); `HELD_RULE.md` written, reviewed (fresh Opus) and committed before any held-read code; then stage 3 on the pick (H6) under it and D34 |
| No main pick, promising | "Promising" reported; no dig before the cut-off (D30); the main line fails (D31) |
| No main pick, not promising | Main line killed; the main line fails (D31) |
| Main line fails (no pick now, or later its held read without Tier 1) and the screen has a pick | The screen's pick is read (H7) under `HELD_RULE.md` (B:ED14), held images embedded only then (B:ED7) |
| Screen pick, main line does not fail | No screen read; the screen pick is a development finding only |
| No pick in either track | Round 8 ends at the gate; AFF stays current best; report |
| Random-control flag | Flag in the report's first lines; the pick stands |
| A held read cannot start by Fri 2026-10-23 12:00 | It does not start; it waits for the next decision A (D32) |
| A bug found after a verdict file exists | R7 §8's rule: the file is never overwritten; `<name>_fix<n>.json` beside it; both reported; the user decides |
| Anything this file does not cover | The run stops with its report and a phone note; the user decides |

## 9. Order of work and process

- **Order:** this file committed → wiring smokes (results/smoke/, no number reported) → SC1 → stage 1 and stage E, views, builders, groupings (scorer-train only) → heads,
  readers, τ^X → seed-42 scoring → C13 → `probe_trigger.json` → NBC if triggered → C13 → `gate_verdict.json` → phone
  note → diagnostics and H\<X\> → `HELD_RULE.md` → held read(s).
- **Read only:** R7's folder and every folder of R7 §10 are read only; R7 §9's writer ban applies. Third-party clones
  are read only; FCMI runs from the `r8_` copy.
- **Modules** start with `r8_` (`run_r8_`, `test_r8_`); screen files are marked `r8_enc_`.
- **Resources:** GPU work (views, FCMI, extraction) under `flock -n -o -E 75 /tmp/gpu0.lock`; exit 75 waits and
  retries later, it never runs unlocked; CPU work at 8 threads, at most 3 reader jobs in parallel; every Python call
  with `PYTHONDONTWRITEBYTECODE=1`; ALADIN with `PYTHONPATH=/data/SSD2/third_party/pydeps`.
- **Unattended:** the run chat does not wait for the user. Correcting code to match this file is not a change; it is
  logged with numbers before and after.

## 10. Inputs (read only; SHA-256 asserted before use and at the end of stage 0)

| File (under `src/test/` unless absolute) | SHA-256 | Used for |
|---|---|---|
| `20261121_round3_affect_gate/results/seed42_arrays.npz` | 5ea4b09a4161a5eac6ca78942cf0f4b99b9c634edca651fba4c2689c7c24ab8a | SC1a; AFF |
| `20261126_design_l_style_source/results/seed42/arrays.npz` | b8678e21ce2f1b005b6fd0c2d077c05f6b869847dab02082ad8871f3e6a27fa1 | SC1 |
| `…/results/seed42/reader_R0.npz` | 69d688ad72afef5b4c920337c7f6462d2ba23c0faab27b1f2d29ef683a63093c | SC1b |
| `…/results/taus.json` | 09d1196274b162e2aff8d6b021c570901ff6fe9afbee66e3875b6cf1b83be1da | SC1b |
| `…/results/readers/bank_R0_half0.npz` | a499041d94b2396d28c586d1027bfe9fbd39d733334319ca8e2fe1ebec83bac7 | SC1b |
| `…/results/readers/bank_R0_half1.npz` | 887c80658ea295c72e21f2724ac8b289cbf9aee6789ffa5948f892f94ade87fd | SC1b |
| `…/results/readers/reader_R0.json` | 742311d4013825a6614109130027a7a5ddb5fb95703b8b87b31d987da78384ae | SC1b |
| `…/results/heads/post_R0.npz` | 0e547fb9a97815da5f7b512652c5c2508d509430341abaf55ac0ed6e9b8bae19 | SC1b; R0own txt |
| `…/results/heads/xheads_R0.npz` | 31246c1e83b2e51a1e21e8192230103856f5f91ee5194440675a6d8af984010a | SC1b; R0own txt |
| `…/results/heads/post_affect.npz` | 936a28adb461ad2713ff13af89f31310fadc33d3dd1b9b2d9cdadb3e3d70ddfd | all readers |
| `…/results/heads/post_image.npz` | 871ed7c577dd42d2d6988ee186cfef5b1a714545b54e02ba32bd7408ab8cd84d | all readers |
| `…/results/heads/post_caption.npz` | 116b0b150967972ce7dc75298b9e190ab7787816aa55187b27b53af747533e66 | all readers |
| `…/results/heads/xheads_affect.npz` | fb1327205e055040c97a5001505149fd3819d54ae871dd79ade16d35f2e3a838 | all readers |
| `…/results/heads/xheads_image.npz` | 1a1f35d82b371250952422b49d69f578fc8aa114a5f06645fdd4f8d4fc6caff6 | all readers |
| `…/results/heads/xheads_caption.npz` | 0d887a4e386ff47d17bb1b95ae8c089553920de9dd8c76442629b176fea84983 | all readers |
| `20261031_pseudo_partitions/results/partitions.npz` | cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa | M2 |
| `20261116_grouping_step1_style/results/step1_group_style.npz` | b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2 | R0 |
| `/data/SSD2/pre_extract/artelingo/style_csd_vitl/embeddings.npy` | 7eefe522383b753bad32f404651103471f9d5b6c41926ee57bfc7386c84818ad | R0own, NBC |
| `/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json` | 6013b2f801eb1e6ae104e9fb3c71a377890bb9aab58e4cb27991c269febdf13b | joins, M10 files |
| `/data/SSD2/pre_extract/artelingo/style_csd_vitl/row_to_image.npy` | 914f4e621ffecd4f0bcb7bc13465544553f7d48646668f02a0ffe9438050ab10 | joins (C6) |
| `/data/SSD2/pre_extract/artelingo_clip224/images_uint8.npy` | 0e56913f818e4604805a6b3a778c73ad717659b63037b632337b5e3cf4fcdb53 | M4 |
| `/data/SSD2/pre_extract/artelingo_clip224/paintings.json` | 3ea9522c7e86f14e75b219d66318ceae55d8953a0bb3e2698a1da92c6477a7d8 | M4 joins |
| `/data/SSD2/pre_extract/artelingo/features/metadata.json` | 43963ca92f46ee638ace94f8ff72807de29ec87471aa5b68904bdc4afb7e8011 | CLIP features |
| `/data/SSD2/third_party/FCMI/networkExplicitDecoupleShuffle.py` | e74c1b65f8d5cb0cdde6df3dcc44725801ea836d581d68b380359659d3d9e6c4 | M7 source |
| `/data/SSD2/third_party/FCMI/main.py` | c4cb24540e0211b7ecc7fe7ccc445788cfa42a71bbb058c16df720d5ed45591d | M7 defaults |
| `/data/SSD2/third_party/ALADIN/main.py`, `vit_networks.py` | 68b973ef3cba1a426cc25a69fed5466149ba90a4f883871801dbcb50b2f6a44d, ff63666fb13e69710d5f3f2bed5f038d386ddd383eb522ee22688ee6abff7ad1 | M10 AL |
| `/data/SSD2/third_party/ALADIN_weights/aladinVIT_bamfg_64.47.pt` | 310835f8b65349f8900ae7f70c95ee58858537939bdd2c7b86ce36a666b2a91d | M10 AL |
| HF `facebook/dinov2-large` model.safetensors | 399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23 | M10 DI |
| HF `google/siglip2-so400m-patch14-384` model.safetensors | 9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2 | M10 SG |
| ClustPy 0.0.3 `clustpy/alternative/nrkmeans.py` | 1145bd9f79cc12931ecf91ab38a94bfed7d7df70f98f06056b5e7b0ec855e569 | M8 |
| HF `openai/clip-vit-base-patch32` @ 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 (refs/main), the weight file that loads: `pytorch_model.bin` or `model.safetensors` (refs/pr/66, c237dc4); which one is recorded | a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f (bin), 99d28a652e6ec46629ab7047a0ac82c69b1fe11e0ce672c43af65d3a9a3fc05d (safetensors) | M4, M9 |
| `20261117_reader_fix_csd/results/rb_halves.npz` | ed2d359e6517b4237f8aa4de5e78f787699b366a975ac993eaef0e7c4491c7da | M1 order, halves |
| `20261126_design_l_style_source/results/groupings/r7_groupings.npz` | e278fa62093a89fec4c6bea37dc224b15af2fa5464416067611702e5143ff3d2 | SC1d, SC1e |

Reused code: R7 §10's table (SHA-256s unchanged) plus these R7 modules, imported read only or copied into `r8_`
with only M14's whitelist change (under `20261126_design_l_style_source/`, at commit e3776ca):

| File | SHA-256 |
|---|---|
| `r7_bundle.py` | 051e9922a53c9630c24e782f64044e2409b535e6d970d7a0ecfbcd51962792ab |
| `r7_common.py` | 82af92cd71bcb44279d788f2d49db236277449d6d564aa0ba3218ade2575ad12 |
| `r7_fusion.py` | f8ac876f1dbf514ddd6985be06de30cddbd613efbb4bb52ac3b3e3d6f654f506 |
| `r7_gate.py` | 52eb81648d514c5a76c62f94ae61313632da1d59019e3505a61c12c6ab91aa33 |
| `r7_groupings.py` | 888fbe468c225bd842f6302a6114ce8cad68cfc44b266375994b69fe1faa1521 |
| `r7_heads.py` | c221741afb6a37dd63b14b942752992b5b0aba6a761de3b74c7c47e99f299b85 |
| `r7_reader.py` | 19726c5f92c9fa15186f13476d3da815ca877c2be3a60b6316c61d91f1f9ce00 |
| `r7_nrkmeans.py` | 6b8b0aa7e1d60b14389abe488616d98323215ba3a853659f6cf4d2afa1074e36 |
| `run_r7.py` | 639828e74d8f754a36b4ddba017859e42c84bd8392498ea68f20a6ae9d588e66 |
| `run_r7_stage0.py` | b3bd38ab96e6b08a511ce5d2b6f6778b5327b2f46c1a0d2ee4b1e40175abc6b4 |

## 11. Exceptions to the constitution

| Principle | Exception | Why | Simpler option rejected |
|---|---|---|---|
| C4 | The later test episodes are H5's held episodes, not fresh seeds (stage 3, under `HELD_RULE.md`) | P8 must pair with AFF's one held read (A §7) | new held seeds, needing a second AFF read (C5) |
| C9 | The gate looks at 24 (26) + 6 candidates with no multiplicity correction; each uses its own 95% lower bound | the user fixed the gate (A:D10, B:ED12); the held read carries the correction | a corrected gate, which the user did not choose |
| C9 | Held checks under Holm at family-wise one-sided 0.0125 per track (governed by `HELD_RULE.md`) | A:D11, D12; both tracks share one 0.025 | each check at 95% |

## 12. ID glossary

| ID | Meaning |
|---|---|
| R7, R3, R6, R1rule | round 7, 3, 6 and 1 decision rules (precedence line) |
| L1 to L16 | R7's definitions, unchanged |
| M1 to M16 | this file's definitions (§3) |
| A:Dn, A:SCn; B:EDn, ESCn | spec A's and spec B's decisions and criteria |
| V100 … NB33, FC1 … FC6, NBC | main-line candidate groupings |
| AL, DI, SG | screen candidate groupings |
| R0own, NRK, TP | references (raw CSD with own assignment, NrKmeans on CLIP, told prompts) |
| Z\<X\>, H\<X\> | random control of X; CLIP-head arm of X |
| A0+X, R1[X], X-AFF[X], CF[s] | R7 L11, L13 |
| B, B0, B1, BX | B, B′(A0), B′(A1), B′(A0+X) |
| SC1 to SC4, ESC1, ESC2 | stage 0, stage 1 (FCMI), the gate items, extraction, the screen gate |
| φ, μ_g, m(i), n(p), c(p), x_p | builder space, group centre, own map, neighbour, content group, CLIP painting vector |
| K(r) | groups after merging at Leiden resolution r (M3) |
