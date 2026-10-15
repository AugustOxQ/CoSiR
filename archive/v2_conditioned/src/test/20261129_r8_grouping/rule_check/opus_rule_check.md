# Rule check: round 8 gate decision rule (both tracks), fresh Opus reviewer (C7)

> Reviewed 2026-10-12 20:23 (Amsterdam). Rule: `src/test/20261129_r8_grouping/DECISION_RULE.md` (uncommitted, 364
> lines). Specs: A `docs/superpowers/specs/2026-10-12-r8-learned-grouping-design.md`, B
> `docs/superpowers/specs/2026-10-12-r8-encoder-screen-design.md`. Base: R7's rule and code. Only this file was written.

## Verdict

**Commit after fixes.** The rule keeps every bar, comparator, candidate, control and data role of both specs. No user
row of A §4 or B §4 is changed. All 46 SHA-256s it pins match. The integer bound, the negative counts and the
content-group facts are correct. One blocker: SC1a, read literally, requires an array that the round-8 path does not
produce, so stage 0 could stop both tracks. Six major findings are gaps where two implementers would get different
groupings or a different stage-0 or stage-1 result. Four of the six are in FCMI (M7) or in the painting order.

## Blocker

**1. §4 SC1a, "every `aff_*` array of R3's `seed42_arrays.npz`".**
- What is wrong: R3's file holds `aff_bar_v` as well. It is R3's bar-margin vector from `C.bar_info(fused, cf, pBp, pB,
  cl, pair_index)`. R7's G1 did not compare it, and the round-8 path does not compute it: L15's bar comparator is a
  different quantity. A literal implementer either skips it, or computes L15's bar and fails SC1, which stops both
  tracks.
- Evidence: `numpy.load(...).files` on R3's file lists `aff_bar_v`. The R3 writer is
  `20261121_round3_affect_gate/run_r3_seed42.py:166, 258`. R7's G1 compares only cells, σ, gates, `aff_fused__*` and
  `aff_cf__*`: `20261126_design_l_style_source/run_r7_stage0.py:166-173`.
- Fix: replace "every `aff_*` array" with "`aff_fused_cells`, `aff_cf_cells`, `aff_sigma`, `aff_gate__a`,
  `aff_gate__b`, `aff_fused__{r1,gain,other,strict,swap}`, `aff_cf__{r1,gain,other,strict,swap}` (as R7
  `run_r7_stage0.py` l.166 to 173; `aff_bar_v` is not compared)".

## Major

**2. M1, M4, M13: "sorted painting order" has two meanings.**
- What is wrong: R7's painting index is the leakage-group code `g_st`. It is `np.unique(groups[st])` from
  `artelingo_splits().groups`, the same as `rb_halves.npz` `paintings_sorted`. M4 joins views through
  `paintings.json`, which is sorted by painting-id string. The two orders differ. If `views.npy` is written in string
  order but read by `g_st` code, every positive is a view of another painting, and no assertion catches it. M13's draws
  "in sorted painting order" also change with the order.
- Evidence: my check (`run_step1.base_data()`; painting id = `data.paintings[first scorer-train row]`): "g-code order ==
  sorted painting id order: False"; 36,518 codes with one painting id each.
- Fix: in M1, define "painting index p = position in round 1's `rb_halves.npz` `paintings_sorted` (= R7's `g_st`
  code); every painting-ordered array (views, φ, n(p), M6's permutation, M13's scorer-train draw) uses it". In M4,
  `meta.json` stores the painting-id string per row, and the head code asserts it equals `data.paintings` of p's first
  scorer-train row. In M13, selection-row paintings are in ascending `artelingo_splits().groups` code over the
  selection rows.

**3. M7: the FCMI test loader and φ of selection images.**
- What is wrong: FCMI runs `inference` on the test loader at the start of every epoch and trains its k-means centres
  on those features. So the loader's content is part of the fit. "All paintings" can be read as including selection
  images, which would break A §3.3 (groupings fit on scorer-train only), or held images, which would break C5. Also,
  M11's own map needs φ for every selection-row image, and M7 never says how φ is computed for an image that is not a
  scorer-train painting.
- Evidence: `/data/SSD2/third_party/FCMI/networkExplicitDecoupleShuffle.py:60-99` (`inference`, faiss k-means on
  `feature_vec`), called each epoch at l.398, then `update_cluster_center` at l.403.
- Fix: "Test loader: the 36,518 scorer-train paintings in M1 order, x_p with c(p), dummy class 0. After training,
  φ of any image i (scorer-train painting or selection-row image) = the unit `encoder_linear[0](encoder[0](x_i))` in
  `net.eval()` under `torch.no_grad()` (the shared encoder needs no group)."

**4. M7: MainLauncher's constants are not pinned.**
- What is wrong: `inference` reads `LimitKmeans` from `MainLauncher`. With it set (the code's value), faiss trains on
  at most 1,000 points per centroid, that is 17,000 of 36,518. `MainLauncher` cannot be imported here, because it
  imports `PathPresettingOperator`, which raises on this GPU (probe §1). An `r8_` copy has to set these constants, and
  two copies can disagree. The FC1 to FC6 groupings then differ.
- Evidence: `MainLauncher.py:19-21` (`UnDeterministic = False`, `LimitKmeans = True`, `DrawMax = 10000`);
  `networkExplicitDecoupleShuffle.py:22, 86-88`.
- Fix: in M7, "`LimitKmeans = True`, `CodeTest = False`, `UnDeterministic = False` (MainLauncher's values), set in
  the `r8_` copy; faiss `Clustering` keeps its default seed".

**5. M7: deterministic CUDA needs `CUBLAS_WORKSPACE_CONFIG`.**
- What is wrong: M7 runs FCMI on the GPU with `torch.use_deterministic_algorithms(True)`. Without
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the first cuBLAS call raises a RuntimeError. Under the D21 test, "the process
  raises", so SC2 fails and the 12 FCMI scorers drop out for an environment reason. The rule allows no FCMI run before
  the stage-1 run.
- Evidence: FCMI `main.py:17` (comment) and l.155 (the authors' command line sets it); `main.py` l.111 is
  `use_deterministic_algorithms(True)`.
- Fix: add "with `CUBLAS_WORKSPACE_CONFIG=:4096:8`" to M7. Allow one wiring smoke before stage 1 (for example 2
  epochs on 2,048 paintings, outputs under `results/smoke/`, no SC2 decision, no number reported).

**6. §4 SC1 does not check two reused paths that every new grouping runs through (C12).**
- What is wrong: SC1b starts from R7's stored heads. So the head-fitting path (R7 L10 caption head, cross-fitted text
  head) is never regression-checked, yet every new grouping's text side comes from it. The M3 recipe code (R7's
  `leiden` with the resolution now a parameter, `knn_union_graph`, `merge_small`) is also new in its `r8_` form and is
  not checked. R7's G3 did check its head path (cross-fitted posteriors against `rb_heads_*`).
- Fix: add "SC1d: the round-8 head path refits R0's L10 caption head and cross-fitted text heads from R7's R0 labels
  (`r7_groupings.npz` key R0); `txt` equals R7's `post_R0.npz` and `xheads_R0.npz` `txt` within 1e-12" and "SC1e: M3's
  code at r = 1.0 on unit CSD painting vectors gives `r7_groupings.npz` key R0 exactly (17 groups)". Both take minutes
  on CPU.

**7. M13: the random-control image draws are not pinned, and C13 does not re-derive them.**
- What is wrong: "each painting draws group g with probability X's painting share" can be coded as `choice`,
  `multinomial` or `random` plus `searchsorted`. These give different draws from the same seed. The flag (D27) depends
  on them. §7(a) re-derives the bank inputs only "from the stored φ, labels and halves", so the control draws are not
  checked.
- Fix: "shares = `bincount(painting labels of X) / 36518` (float64); scorer-train: `default_rng(7200 + i).choice(K,
  size=36518, p=shares)` in M1 order; selection: `default_rng(7300 + i).choice(K, size=n_sel, p=shares)` in item 2's
  order". In §7(a), add "and the control draws from their seeds".

## Minor

**8. M6, RNG order for views and negatives.** "Per batch the view draws and the negatives" leaves the calls open. Fix:
"views: `rng.integers(0, 5, size=len(batch))`; then per query in batch order: `rng.choice(own_pool, n_own,
replace=False)`, then `rng.choice(other_pool, 128 − n_own, replace=False)`; pools are ascending painting indices with p
and n(p) removed". Also say that n(p) in the exclusion is M5's CLIP neighbour for V and NB, and the CSD neighbour for
NBC.

**9. M4, M7, M10: GPU numerics.** "Bicubic, antialias" names no function, and PIL bicubic differs from torch bicubic.
Fix: `torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)` on
float32 /255, then clamp. For every GPU pass, set `torch.backends.cuda.matmul.allow_tf32 = False` and
`torch.backends.cudnn.allow_tf32 = False` (as CSD's `meta.json` records), and record the batch size. Leiden on a kNN
graph is sensitive to small feature changes.

**10. M9, M11, M12: TP's inputs are not defined, and its name source reads label columns.**
- Fix 1: say "TP's `post_TP` img and its bank inputs = the TP map (argmax, then merge map) of the row's image, the
  same in both halves; `classes_half{0,1}` = arange(K); `filled_by_half` = 1 − half".
- Fix 2: `encode_labels(data.art_styles)` builds its names over all rows, held rows included. Take the 27 names as the
  sorted top-level folders of `paintings.json` image paths. I checked that these are the same 27 names, for example
  `Ukiyo_e` and `Naive_Art_Primitivism`.

**11. M8: ClustPy names.** ClustPy 0.0.3 stores `V_` and `P_` (`nrkmeans.py:860-862`); `transform_subspace(X, s)` =
`X @ V_[:, P_[s]]` (l.944). R7's worker saves only the labels (`r7_nrkmeans.py`, `worker`). Fix: cite
`transform_subspace`; the `r8_` worker also saves `V_` and `P_` and asserts `labels_.shape == (n, 2)`.

**12. §6.3: trigger edge case.** If all three NB groupings are dropped, "strength" is undefined. Fix: "among evaluated
NB scorers; if none was evaluated, NBC does not run (recorded)".

**13. M3, merges and group counts.**
- `merge_small` raises `SystemExit`, not an Exception, when no group reaches 200 rows
  (`run_told_oracle.py:73-74`). Fix: at a search point this means K is undefined; the point is skipped and recorded.
- M7's and M8's merges can take FC* or NRK below the user's 17 (A:D9). This is a fair mechanic: record K per grouping.
- TP keeping up to 27 groups settles a conflict inside spec A (§3.2 "every grouping X has 17 groups" against D24). List
  it for a spec note.

**14. M7, agent defaults to record.**
- φ is the 17-value representation, where FCMI clusters. A §3.2 says "FCMI's hidden features". Mark this as an agent
  default.
- FCMI on the GPU departs from A §9's budget, which had FCMI on CPU. Six runs of up to 60 min each come on top of B's
  3 h. This is a mechanic, not a design change.
- `save_checkpoint` writes epochs 2, 20 and 300 of a network with about 145 million parameters plus Adam states. My
  estimate is about 1.7 GB per checkpoint, about 30 GB over six runs. Fix: put `results/fcmi/` checkpoints on
  `/data/SSD2/`.

**15. M10, image loading.** Fix: DI and SG load with `PIL.Image.open(p).convert("RGB")` before the processor. AL uses
`main.load_img` as written. ESC1(d) hashes the index array as contiguous int64.

**16. §7, C13 exemptions.** ESC1 (b) to (d), and SC2's label count and finite-loss log, can be re-checked by the
separate code from the stored files in seconds. Add them to the re-derivation list.

**17. M10, ALADIN.** The only released weights are ALADIN-ViT from StyleBabel (README l.5-7). They are not the ICCV 2021
VGG model that ED5's wording suggests. The rule's choice is right; the report should name the model correctly.

**18. §10, pins.** Add the CLIP B/32 revision and weight SHA-256 (M4, M9), and `rb_halves.npz`. For TP's text tower,
add a check: re-encode 256 cached captions and require cosine ≥ 0.9999 with `txt_features`.

## Agent defaults

None of the marked defaults moves a bar or changes a user decision. The closest to design choices are the merge after
FCMI and NRK (item 13) and the drop of V100 to V33 on a failed view check. Both are fair mechanics; record them in the
report.

## Checks run

| Check | Result |
|---|---|
| SHA-256 of the 29 §10 inputs (including `images_uint8.npy` and both HF blobs), the 10 R7 code files, R7, R3, R6, R1rule, the constitution, and specs A and B at 31f41dc and fc4ef0e | all 46 match |
| ⌈−0.3 · 4,096 / 25⌉ = ⌈−49.152⌉ | −49, correct |
| round(128 · s) for s = 1.00, 0.67, 0.33 | 128, 86, 42, correct (2/3 would give 85 and 43; the rule's literal 0.67 is pinned) |
| M2 (`partitions.npz`, `image` by `local_groups`) | 64 groups; 322 to 992 paintings; smallest other-group pool 35,526; ≥ 129 holds |
| Batches per epoch | ⌈36,518 / 512⌉ = 72 |
| SC1 keys in R7's `seed42/arrays.npz` | `AFF__*`, `AFF_cf__*`, `R1_R0__*`, `R1_R0_cf__*`, `XAFF_R0__*`, `XAFF_R0_cf__*`, `B__*`, `B0__*`, `B1__*`, `BX_R0__*`, `cl`, `pair_index`, `parity`: all present; `reader_R0.npz` has `P__a/b`, `m__a/b`, `pick__a/b`; `taus.json` has R0 |
| M10 image set | non-held rows give 49,121 CSD images; held rows 12,281; no overlap; non-held painting set equals `paintings.json` |
| R7 formats (`xheads_*`, `post_*`, `load_crossfit`) | own-map one-hots fit (`filled_by_half` = 1 − half; all classes per half) |
| Fidelity (A §3 to §6, B §3 to §6) | bars SC3, SC4 and ESC2, pick order, NBC trigger, flag and promising equal the specs; no user row changed |
| C5 | no held row or image is read, except the risks in items 3 and 10 |
| Environment | numpy 2.2.6, scikit-learn 1.6.1, torch 2.11.0, faiss-cpu 1.15.1, transformers 5.6.2, leidenalg 0.12.0 |
