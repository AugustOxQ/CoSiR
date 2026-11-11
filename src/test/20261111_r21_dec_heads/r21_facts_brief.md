> Facts brief for r21, read by Codex (gpt-6.1-terra, medium) on 2026-11-11 05:48 for the r21 decide chat; spot-checked
> by the decide chat (model shape, weights, DEC primitives). Read-only facts, not results. Added in the decide chat:
> c* seed 0 best epochs per head are emotion 4, style 1, genre 18 (run__s0.json, best_epoch).

# c* and DEC fact-finding brief

## 1. Training recipe

- **c* is `k11=40`**: the G configuration with 40 mutual top-k graph neighbours, rather than 20. Its saved run record has no removals. Sources: [rs_sheet.py](src/test/20261027_recipe_sweep/rs_sheet.py:11), [run__s0.json](/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/train/k11=40/run__s0.json), [queue_stage4_status.txt](/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/queue_stage4_status.txt).
- The model has a shared **2048 → 1024 linear → GELU → dropout 0.1 trunk**, followed by one **1024 → 2048 linear head per aspect**. Head outputs are L2-normalised. Image and caption sides use this model structure. Source: [ms_train.py](src/test/20261019_model_sheet/ms_train.py:31).
- The base loss adds the mean of four weighted symmetric buddy cross-entropies, the mean of two own-caption symmetric cross-entropies, and **0.25 × direct teacher loss**. Genre additionally has **8 × relational teacher loss**, with teacher temperature 0.02; the emotion and style relational weights are zero. Buddy temperature is 0.05. Sources: [ms_train.py](src/test/20261019_model_sheet/ms_train.py:119), [rg_loss.py](src/test/20261027_relational_genre/rg_loss.py:83), [rs_sheet.py](src/test/20261027_recipe_sweep/rs_sheet.py:11).
- AdamW uses learning rate **0.001**, weight decay **0.05**, batch size **512 graph pairs**—1,024 endpoint rows—and at most **200 epochs**. Each aspect stops after **15 epochs without improvement** in its own development loss. Sources: [itp_train.py](src/test/20261017_image_transfer_pilot/itp_train.py:15), [rs_sheet.py](src/test/20261027_recipe_sweep/rs_sheet.py:19), [ct2_d2.py](src/test/20261020_claim_test/ct2_d2.py:114).
- On improvement, the trainer deep-copies that aspect’s **view state**, which contains its head and the shared trunk at that time. Each view is restored and saved separately; active heads continue training until their own stopping condition. Thus the exported heads can contain trunks from different epochs. Sources: [ct2_d2.py](src/test/20261020_claim_test/ct2_d2.py:161), [ct2_d2.py](src/test/20261020_claim_test/ct2_d2.py:217), [ct2_d2.py](src/test/20261020_claim_test/ct2_d2.py:340).

## 2. Launch, runtime and seeds

- The full-scale queue invokes `rs_full.py train --config k11=40 --seeds 0 1 2 3 4 --device cuda --outputs …`, followed by its `held` command. The CLI has stage and integrity gates; the queue shows the complete invocation. Sources: [rs_queue_stage4.sh](src/test/20261027_recipe_sweep/rs_queue_stage4.sh:56), [rs_full.py](src/test/20261027_recipe_sweep/rs_full.py:483).
- Full-scale CUDA training records give **317.0, 243.5, 212.4, 296.1 and 241.3 seconds** for seeds 0–4 respectively, approximately **3.5–5.3 minutes per seed**. These are recorded run times, not a prediction for DEC. Source: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/train/k11=40/run__s{0..4}.json`, `seconds`.
- Full-scale **held and validation exports exist for seeds 0–4**. The separate sample-scale development comparison used seeds **1–3**. Sources: [ns_saved_outputs.md](src/test/20261103_new_scores/ns_saved_outputs.md:18), [ns_saved_outputs.md](src/test/20261103_new_scores/ns_saved_outputs.md:85), [rs_sheet.py](src/test/20261027_recipe_sweep/rs_sheet.py:46).

## 3. Checkpoints and exports

Let `R` be `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full`.

- Weights: `R/weights/k11=40/{emotion,style,genre}__s{0..4}.pt`, with JSON sidecars. The `.pt` files contain CPU PyTorch state dictionaries for individual aspect views. Source: [ct2_d2.py](src/test/20261020_claim_test/ct2_d2.py:340).
- Held images: `R/held/k11=40/{aspect}__s{seed}.npz`, key `img`, float16, shape **[12281, 2048]**, with JSON sidecars. Validation image and caption exports: `R/train/k11=40/R6-5/{emotion__Gres,style__Fres05,genre__R0o}__s{seed}.npz`, keys `img` and `cap`, **[6152, 2048]** each. Source: [ns_saved_outputs.md](src/test/20261103_new_scores/ns_saved_outputs.md:18).
- Existing inference code reloads a view with `torch.load(..., weights_only=True)` and `load_state_dict`. A training-resume or fine-tuning CLI that loads these checkpoints was **not found**; the full runner instead verifies and skips completed runs. The saved view alone also does not restore AdamW state. Sources: [rb_student.py](src/test/20261025_recipe_backbones/rb_student.py:308), [rs_full.py](src/test/20261027_recipe_sweep/rs_full.py:249).

## 4. Class-count AMI

- [ns_clu_cells.py](src/test/20261103_new_scores/ns_clu_cells.py:1) and [ns_metrics.py](src/test/20261103_new_scores/ns_metrics.py:174) use L2-normalised rows and **MiniBatchKMeans**, `n_init=3`, batch size **4096**, random states **0, 1, 2**. Each model seed’s score averages those three partitions. K is the labelled class count: emotion 9, style 27, genre 10.
- Clustering fits **all 12,281 held paintings**; labels are masked only when scoring. Emotion and style have 12,281 labelled paintings; genre has **9,949**. Labels are painting-level emotion, style and WikiArt genre; genre `-1` means unlabelled. Sources: [ns_clu_cells.py](src/test/20261103_new_scores/ns_clu_cells.py:6), [rt_data.py](src/test/20261029_retrieval/rt_data.py:61).
- The held result file records these **five-seed c* means** and the **unseeded instructed-image teacher** score:

| Held AMI | Emotion, K=9 | Style, K=27 | Genre, K=10 |
|---|---:|---:|---:|
| c*, mean | 0.102413 | 0.410926 | 0.510070 |
| Teacher | 0.088875 | 0.406315 | 0.501466 |

  Source: `/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage2/cells/clu__artelingo__qwen.json`, teacher comparator cells. The held **per-seed class-count AMIs were not found** in saved results. Partitions are documented as kept in memory and never written; the saved cells contain aggregate scores. [ns_clu_cells.py](src/test/20261103_new_scores/ns_clu_cells.py:28).

- Validation c* AMI exists in [stage1_ranges.json](src/test/20261103_new_scores/records/stage1_ranges.json), in seed order 0–4:

| Validation AMI | Seeds 0–4 | Mean |
|---|---|---:|
| Emotion | 0.100984, 0.104611, 0.103668, 0.102795, 0.103551 | 0.103122 |
| Style | 0.420558, 0.412509, 0.417235, 0.420626, 0.421187 | 0.418423 |
| Genre | 0.529839, 0.516278, 0.527559, 0.521734, 0.518845 | 0.522851 |

  Validation teacher class-count AMI: **not found**.

## 5. Ablation-table retrieval

- [r20_ret.py](src/test/20261109_r20_figures/r20_ret.py) calls [rt_metrics.py](src/test/20261029_retrieval/rt_metrics.py) for held image-to-image label mAP, plain cosine (`R0 / none+k0`), excluding the query itself. Source metadata and results: [ablation_ret.json](src/test/20261109_r20_figures/ret_full/ablation_ret.json).

| c* label mAP | Seeds 0–4 | Mean | Max − min |
|---|---|---:|---:|
| Emotion | 0.263832, 0.264710, 0.264659, 0.262999, 0.264635 | 0.264167 | 0.001711 |
| Style | 0.310774, 0.311448, 0.310932, 0.311860, 0.309903 | 0.310983 | 0.001957 |
| Genre | 0.588836, 0.588064, 0.589209, 0.591626, 0.590512 | 0.589649 | 0.003562 |

## 6. Held AMI seed spread

**Not found** for class-count held AMI: the aggregate result file does not retain the five seed values. It cannot be derived from the means above. The validation class-count seed ranges calculated from the saved values are emotion **0.003628**, style **0.008678**, genre **0.013561**; these are validation ranges, not held ranges. Source: [stage1_ranges.json](src/test/20261103_new_scores/records/stage1_ranges.json).

## 7. Existing DEC code

- The PercepT reimplementation has reusable PyTorch primitives in [pc_sept.py](src/test/20261018_percept_core/pc_sept.py:68): Student-t assignments with α=1, row normalisation, and detached `q² / frequency` targets. Its training loop applies `F.kl_div(q.log(), p, reduction="batchmean")`.
- That loop also trains an autoencoder with reconstruction and balance losses, uses full-batch updates, and initialises trainable centroids with KMeans. The assignment and target functions are reusable building blocks; the complete training loop would require adaptation to c*’s three heads, graph batches and stopping scheme. Source: [pc_sept.py](src/test/20261018_percept_core/pc_sept.py:87).
- Other occurrences and a PercepT caller exist in [ct_percept.py](src/test/20261020_claim_test/ct_percept.py:229). An existing **c* DEC arm or start-cluster pull** was not found.

## 8. Validation and development splits

- The full-scale split groups by painting, seed 42: **42,969 train**, **6,152 validation**, **12,281 held paintings**. Their annotation-row counts are 216,107, 30,872 and 61,744 respectively. Source: [ct2_data.py](src/test/20261020_claim_test/ct2_data.py:34).
- In the full trainer, `fit` is train paintings and `dev` is the validation paintings used for stopping; held paintings are separate. Source: [ct2_data.py](src/test/20261020_claim_test/ct2_data.py:45).
- “Development” in the recipe sweep can also mean the earlier **sample-scale P** evaluation, so it should not automatically be equated with the full 6,152-painting validation export. Source: [rs_train.py](src/test/20261027_recipe_sweep/rs_train.py:1).

## 9. Eight-removal ablation

- [rs_ablation.py](src/test/20261027_recipe_sweep/rs_ablation.py:137) resolves the final configuration and removal arms; [rs_knobs.py](src/test/20261027_recipe_sweep/rs_knobs.py:29) applies their patches; [rs_full.py](src/test/20261027_recipe_sweep/rs_full.py:263) trains full-scale arms.
- The eight configuration suffixes are `-no_relational`, `-no_sharp_row`, `-no_direct`, `-no_buddy`, `-no_caption_endpoints`, `-no_content_correction`, `-no_instructed_graphs`, and `-no_shared_trunk`. They respectively remove the relational term, sharpened relational teacher row, direct teacher, buddy terms, caption buddy endpoints, content correction, aspect-instructed graphs, and shared trunk. Source: [rs_sheet.py](src/test/20261027_recipe_sweep/rs_sheet.py:31).
- These are **named configuration removals**, not generic CLI boolean flags: e.g. `k11=40-no_buddy` is passed through `--config`. A `k11=40` plus DEC arm would need a new supported configuration and training-loss implementation; such an arm is **not found**. Source: [rs_sheet.py](src/test/20261027_recipe_sweep/rs_sheet.py:62).
