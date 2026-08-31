# `combine_side` consistency review

**Scope.** Read-only audit of the eight requested Python files, plus `configs/model/clip_base.yaml` to establish the relevant configuration.  The configured runs use `combine_side: "img"` at `configs/model/clip_base.yaml:12`.

## Result

The training path and the evaluation paths used by `train_cosir.py`'s snapshots are consistent with `combine_side`.  With `img`, they condition image embeddings and project/use text as the other side; with `txt`, they do the inverse.  The same is true of the condition-analysis cache, qualitative retrieval snapshots, automatic condition-space evaluator, oracle/predictor metrics, and both post-hoc condition diagnostics.

There is one **real configuration-consistency defect** in a separate, legacy non-oracle test-evaluation route: three helpers in `src/eval/metrics.py` unconditionally combine text and use an unprojected image gallery.  `src/eval/pipeline.py` calls all three whenever `use_oracle=False`, including through its public default.  They therefore do not evaluate an `img`-combine model according to its configured mechanics.  This route is **not** the training snapshot route: `src/hook/train_cosir.py:948-956` passes `use_oracle=True`.

## Invariant used for every check

| `combine_side` | input to `model.combine` / combiner | other-side input to `project_other` | similarity shape (text rows, image columns) |
|---|---|---|---|
| `"img"` | `img_emb` / `img_full` | `txt_emb` | `projected_txt @ combined_img.T` |
| `"txt"` | `txt_emb` / `txt_full` | `img_emb` | `combined_txt @ projected_img.T` |

## Locations checked

### `src/model/cosirmodel.py`

- **OK — `67`, `98-100`:** accepts only `txt`/`img` and stores the selected value.
- **OK — `155-186`:** `combine`, `project_other`, and `predict_condition` deliberately accept caller-selected tensors; their contracts correctly state that the caller must supply the configured side.
- **OK — `188-204`:** `forward` uses `txt_emb, txt_full` for `txt` (`193-194`) and `img_emb, img_full` for `img` (`195-196`).  This is the authoritative model-forward branch.

### `src/hook/train_cosir.py`

- **OK — `81-87`:** passes `cfg.model.combine_side` into the model rather than hardcoding a side.
- **OK — `588-603`, `1274-1305`:** condition-viz and final-model artifacts persist `combine_side` alongside the combiner and `other_proj` state, enabling downstream side-aware reconstruction.
- **OK — condition-analysis cache, `649-688`:** `654-657` derives the projected other side correctly; `675-681` combines text / scores `combined_txt @ projected_img.T` for `txt`, and `682-688` combines image / scores `projected_txt @ combined_img.T` for `img`.  Both yield text-by-image `sims` at `681`/`688`.
- **OK — cache T2I/I2T labels, `692-762`:** T2I operates on text rows of that canonical matrix; I2T is its transpose.  Ground-truth indexing and saved `per_rep_*` versus `per_rep_i2t_*` shapes match those query modalities; no conditioned/other-side swap occurs.
- **OK — retrieval snapshot, `803-831`:** `810-815` projects the correctly inverted side.  `822-825` produces I2T/T2I for `txt`; `826-829` produces them with operands transposed for `img`, so keys `i2t` and `t2i` retain their real query meanings.
- **OK — snapshot labels/dumps, `833-910`:** top-k shapes, GT maps, and persisted `combine_side` are consistent.  The raw CLIP baseline (`874-905`) is intentionally unconditioned and correctly keeps image queries for I2T and text queries for T2I.
- **OK, config-limited by design — Phase-1 evaluation, `1088-1167`:** this block's `_other_n`, `predict_condition`, and `combine` calls (`1106`, `1139-1150`) are hardwired to the configured `img` case, but the current configuration is `img`; its direction calculation is explicitly guarded by `combine_side == "img"` at `1109`.  If this metric were expected to support `txt`, it would be incorrect (it projects text and combines image regardless of side).  It does **not** affect the current `img` runs.
- **OK, config-limited by design — Phase-2 transfer score, `1176-1228`:** explicitly executes only for `combine_side == "img"`, and its `project_other(txt)` / `combine(img)` choices are correct under that guard.  It is skipped, rather than miscomputed, for `txt`.
- **OK — other-side feature key, `1370-1387`:** `_other_key = "txt_features" if ... == "img" else "img_features"` is exactly the required inversion and is used consistently to load/reorder `other_feat_table`.
- **OK — combine-side feature key, `1400-1432`:** `_combine_key = "img_features" if ... == "img" else "txt_features"` is the complementary derivation.
- **OK — core feature training branch, `1519-1619`:** `1539-1546` selects the configured combine embedding/full sequence and the inverted other embedding; `1546` projects only the latter.  The oracle-guided (`1556-1572`), main combiner (`1574-1582`), loss (`1584-1594`), and condition-predictor (`1612-1619`) consumers all use those derived variables.  Variable names `loss_img_target` / `loss_txt_ref` are misleading when `img` is selected, but the positional loss inputs are intentionally side-relative and the values are correct.
- **OK — adjacent loss-derived calculations, `1662-1673`, `1693-1713`:** gap alignment uses the inverted other feature minus the combined feature; buddy contrastive uses the side-inverted `other_feat_table`, `project_other`, and the per-batch `other_emb` consistently.

### `src/eval/pipeline.py`

- **OK — train evaluator, `176-199`:** `183-189` selects combine feature/full and the opposite anchor correctly.  `192-198` uses the anchor consistently for raw, combined, and shuffled rank comparisons.
- **Defect in the non-oracle dispatch — `305-322`:** when `use_oracle=False`, the pipeline invokes all three hardcoded helpers below, without consulting `model.combine_side`.  Thus the default `evaluate()` / `evaluate_test()` parameter (`297`, `411`) can report incompatible metrics for an `img` model.  This is reachable from `inference_test` (`428-430`) and `src/hook/eval_cosir.py:258-265`.  It is not used by the training snapshot (`train_cosir.py:948-956`, `use_oracle=True`).
- **OK / not conditioned — `324-380`:** raw recalls and `encode_data_only` intentionally use raw image/text embeddings, so their fixed I2T/T2I operand order is unrelated to `combine_side`.

### `src/eval/metrics.py`

- **OK — oracle recalls, `188-289`:** `198-205` projects the actual other side; `207-208` choose the same combine source/count; `225-249` builds text-by-image similarities for both configurations; `277-289` derives T2I/I2T by matrix and transpose.  No label swap.
- **Defect — `297-342` (`compute_non_oracle_recall_txt`):** always predicts from and combines `text_embeddings` (`323-324`) against raw, unprojected image embeddings (`313`, `329`).  Correct only for the pre-`other_proj`, `combine_side="txt"` interpretation; it is mechanically wrong for `combine_side="img"` and does not use the configured other projection even for `txt`.
- **Defect — `344-408` (`compute_non_oracle_recall_img`):** predicts a condition from each image (`371-379`), but then unconditionally combines **text** at `381-391`; this violates the `img` invariant and compares against raw, not `project_other`, images.  The name describes the predictor input, not the combined side, so it should not be interpreted as configured image-side evaluation.
- **Defect — `410-462` (`compute_non_oracle_recall_imgtxt`):** documentation itself fixes the combine side to text (`421-425`), and `442-449` predicts from / combines text against raw image embeddings.  It does not follow `combine_side="img"`.
- **OK — side-aware predictor recall, `464-519`:** unlike the three legacy helpers, `477-507` branches on `model.combine_side`, projects the inverse side, predicts from the selected side, combines that same side, and constructs text-by-image `sims`.  `511-519` label T2I/I2T correctly.

### `src/metrics/loss.py`

- **OK — main contrastive loss, `109-153`, as called by `train_cosir.py:1584-1594`:** `image_features` is a historical parameter name; the caller supplies projected *other-side* embeddings there, and supplies combine-side raw features as `text_features`.  Therefore `123` compares combined to other, while `145-150` receives the combine-side reference.  This remains correct for both sides.
- **OK — mixup, `19-54`, `172-183`:** despite `text_emb`/`image_emb` parameter names, it is also side-relative because the caller passes `loss_txt_ref` (combine side) and `other_emb` (projected other side).  It combines the former and compares with the latter.
- **OK — preserve loss, `198-214`:** explicitly selects text for `txt`, image for `img`; the input is the caller's combine-side raw feature.

### `src/utils/condition_space_evaluator.py`

- **OK — setup, `61-92`:** `63-87` selects combine embeddings, query/other embeddings, counts, and GT maps as inverse pairs.  `92` projects exactly the query/other side.
- **OK — all evaluator retrieval computations, `122-403`:** every `model.combine` call uses `self.combine_embs` (`129-132`, `175-177`, `236-240`, `300-305`, `371-377`), while every similarity uses `self.query_embs`.  These are initialized side-aware above.  The names/comments at `102-107`, `146-154`, and `205-206` say “text”, but that is stale wording only: the tensors are side-generic and correct for `img`.

### `scripts/analyze_condition_geometry.py`

- **OK — `249-281`:** reads the persisted snapshot side (`259`), selects `img_t` for `img` and `txt_t` for `txt` (`260`), reorders it to the condition-table IDs, and uses it for both combined embedding and shift (`263-269`).  `embedding_shift` therefore measures the configured combine side.
- **Ambiguous labeling only — `191-219`, `238-245`, `299-304`:** helper argument names and output labels (`text_feat`, `text_sample`, `n_text_sample`) retain text-specific wording even though `analyze_run` supplies the selected side.  The calculation is correct; output labels can be misleading for image-side runs but do not swap tensors.

### `scripts/analyze_condition_retrieval_correlation.py`

- **OK — drift and dump plumbing, `78-83`, `251-273`, `419-431`, `472-482`:** condition drift is modality-independent and aligned by sample ID.  The per-sample dump validates equal lengths and preserves `sample_ids`, rank deltas, drift, and shift from the same query index set; no modality label is fabricated.
- **OK — paired-run side guard, `301-307`:** takes `combine_side` from the frozen snapshot and asserts the trained snapshot agrees, preventing cross-side comparisons.
- **OK — other-side derivation, `371-390`:** `373` selects combine features and `376` selects the exact inverse other features; the latter alone feeds rebuilt `other_proj` at `390`.
- **OK — rank/shift consistency, `393-417`:** combines only `combine_feat[query_idx]`, computes shift against that same uncombined source, and ranks against the projected inverse gallery.  Frozen/trained/counterfactual branches reuse precisely this side-aware pipeline.
- **Terminology caveat, not a tensor bug — module docstring `5-24`, function docstring `285-300`:** it calls the rank diagnostic “i2t” generically.  Under `combine_side="img"`, the actual query is an image and gallery is text (I2T); under `txt` it is text-to-image (T2I).  Computation follows the side correctly, but the generic I2T wording would be mislabeled if this script were run on a `txt` snapshot.

## Assessment for the research explanation

For the `combine_side: "img"` runs in scope, the evidence supports the mechanical premise: conditions enter the image representation in training and in the oracle/retrieval/condition-drift analyses used by the training workflow, while text is the projected other side.  I found no accidental text-side conditioning or opposite-side projection in those paths.

Do not use results from the non-oracle `txt_non_oracle`, `img_non_oracle`, or `both_non_oracle` helper metrics as confirmation of that premise for image-side runs: they are not configuration-aware.  The similarly named generic “i2t” text in the post-hoc correlation script is a documentation-label caveat, not a computational inversion.
