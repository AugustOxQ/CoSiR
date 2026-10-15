# Can CoSiR v2 use the GeneCIS benchmark?

**Question.** GeneCIS ([Vaze, Carion and Misra, CVPR 2023](https://arxiv.org/abs/2306.07969); code at [facebookresearch/genecis](https://github.com/facebookresearch/genecis)) is the paper CoSiR builds on. We checked whether CoSiR v2 can use its benchmark: the license, whether the data is on this machine, and whether the task fits what the v2 model scores.

**Answer.** Yes as an evaluation benchmark, with three caveats. (1) The license allows it for non-commercial research. (2) The object half of the benchmark (3,920 templates) can be evaluated today from images already on disk; the attribute half (4,112 templates) needs a Visual Genome 1.2 download that we have not made. (3) GeneCIS scores reference image plus text condition against target images, while CoSiR v2 scores `s(I,T|c)` with a condition given as support and contrast pairs. Using GeneCIS therefore needs an image to image scoring mode and a text-phrase condition adapter, and it measures zero-shot transfer of the factor model rather than the v2 primary claim. The mined CC3M training triplets already on disk cannot be used on their own and would break the no-condition-labels claim if used for training.

## 1. What GeneCIS measures

GeneCIS defines conditional similarity as ranking target images for a reference image under a text condition. Each template has one reference, one condition phrase, one correct target and a small gallery of distractors chosen to match only the reference or only the condition ([paper §4](https://arxiv.org/html/2306.07969)). The metric is Recall@1/2/3 within the template's gallery. The benchmark is zero-shot: the authors train on CC3M and select checkpoints on CIRR, never on GeneCIS ([repo README](https://github.com/facebookresearch/genecis/blob/main/README.md)).

We re-derived the counts below from the four annotation files in our clone (`/project/genecis/genecis/*.json`). Gallery size includes the target; the JSON lists the target separately from 9 or 14 distractors.

| Task | Condition is | Templates | Gallery | Distinct conditions | Image source | Unique items |
|---|---|---|---|---|---|---|
| Focus attribute | an attribute type ("color", "material") | 2,000 | 10 | 40 types × 50 templates | Visual Genome 1.2, bbox crop | 15,773 crops |
| Change attribute | the target attribute value ("murky") | 2,112 | 15 | 400 | Visual Genome 1.2, bbox crop | 18,212 crops |
| Focus object | an object in the reference to keep | 1,960 | 15 | 97 | COCO 2017 val, full image | 2,198 images |
| Change object | an object to add to the scene | 1,960 | 15 | 109 | COCO 2017 val, full image | 2,739 images |

Both attribute tasks together reference 23,640 unique Visual Genome images; both object tasks reference 3,033 unique COCO images. The template counts and gallery sizes match the paper.

## 2. License

The GeneCIS code and annotations are under CC BY-NC 4.0 ([LICENSE](https://github.com/facebookresearch/genecis/blob/main/LICENSE)). Non-commercial academic evaluation and publication of results are allowed with attribution, which means citing the paper. The images stay under their own datasets' terms, Visual Genome for the attribute tasks and COCO (Flickr-sourced images) for the object tasks. We did not re-read those two datasets' license pages in this check; that is listed under open items.

## 3. What is on disk

| Component | Needed for | Status on this machine | How we checked |
|---|---|---|---|
| GeneCIS annotations (4 JSON files) | all tasks | present, `/project/genecis/genecis/` | loaded and counted (§1) |
| GeneCIS eval code | all tasks | present, `/project/genecis/eval/`, `datasets/` | read `vaw_dataset.py`, `coco_dataset.py` |
| COCO 2017 val images | object tasks | present as COCO 2014 files: all 3,033 needed IDs are in `/data/SSD/coco/images/val2014` | matched every `val_image_id` against local file names |
| Visual Genome 1.2 images | attribute tasks | **missing**, no `VG_100K*` anywhere under `/data`, `/project`, `/root` | `find` over those trees |
| CC3M mined triplets (1.6M) | GeneCIS training only | present, `/data/PDD/cc3m_training_triplets_1.6M.pt` (294 MB) | loaded: 1.6M dicts of `ref_img_idx`, `target_img_idx`, `text_distractor_img_idx` |
| CC3M parsed scene graphs | needed to read the triplets | **missing** | not found |
| GeneCIS pretrained Combiner weights | published reference model | **missing** | not found |

COCO 2017 re-partitions the same 123,287 images as COCO 2014 with unchanged IDs, so the 2014 files serve the object tasks once the loader maps `val_image_id` to `COCO_val2014_%012d.jpg`. Disk space is not a constraint (`/data/HDD` has 3.4 TB free).

The triplet file is not usable by itself. Its indices point into rows of GeneCIS's parsed scene-graph dataframe (`train/` and `datasets/cc_3m_dataset.py`, `annotated_samples_df.iloc[...]`), which we do not have, and the condition text is read from those scene graphs rather than stored in the triplets.

## 4. Fit with the v2 model

The v2 scorer is `s(q, k | c) = β·cos(q, k) + Σ_l w_l(c)·a_l(q)·a_l(k)` (`src/model/conditioning.py`), with factor codes from `SharedFactorEncoder.encode_image` / `encode_text` (`src/model/factors.py`) and condition weights `w(c)` built from support and contrast pair sets (`naive_condition_weights`, `ConditionEncoder` in `src/model/condition.py`). Five gaps separate it from GeneCIS.

1. **Modality.** GeneCIS ranks images for an image query. The scorer code is modality-agnostic, so image to image is computable by encoding both sides with `encode_image`, but it tests a different relation from the v2 primary target `s(I,T|c)`. The [Stage 1 to GeneCIS synthesis](2026-09-28_stage1_genecis_synthesis_brainstorm.md) already classified existing CIR benchmarks as adapted diagnostics for this reason.
2. **Condition interface.** GeneCIS supplies a short phrase; v2 builds `w(c)` from support and contrast pairs. An adapter is needed: either the prompt route `w(c) = a_T(CLIP_text(phrase))` after normalization, or retrieving support pairs from the training pool by CLIP text similarity to the phrase. The synthesis brainstorm proposed both and flagged neither as validated; an abstract phrase such as "color" may not activate the factors that captions about red objects do.
3. **Change tasks.** "Change attribute" and "change object" ask for a directed edit (add a wall, make it murky). The v2 factor term is a symmetric agreement score with no composition operator, so we expect these two tasks to reduce to roughly the image-only baseline. The focus tasks match the v2 notion of selecting an aspect.
4. **Domain.** Every current factor model was trained on ArtELingo paintings (`load_artelingo` in `src/test/20261018_affect_factor_learning/run_affect.py`), and its learned axes are emotion and art style. GeneCIS conditions are color, material, object identity and scene content on natural photographs. Transfer from ArtELingo factors is unlikely to be informative; a fair test would retrain factors on natural image-caption pairs (COCO captions and CC3M are on disk) while keeping GeneCIS out of training.
5. **Inputs.** Attribute tasks crop each Visual Genome box, dilate it and pad it to a square before encoding (`datasets/vaw_dataset.py`). This needs a new frozen CLIP ViT-B/32 cache of about 34,000 crops plus 3,033 COCO images, which is small.

## 5. Baselines we would compare against

The paper reports no CLIP ViT-B/32 numbers (main results use RN50x4, the appendix ViT-B/16), so its numbers below are a reference, not a matched baseline. A matched baseline means rerunning the paper's three CLIP-only rows with our frozen ViT-B/32 on the same templates.

| Method (paper, RN50x4) | Focus attr R@1 | Change attr R@1 | Focus obj R@1 | Change obj R@1 | Avg R@1 |
|---|---|---|---|---|---|
| Image only | 17.7 | 11.9 | 9.3 | 7.2 | 11.5 |
| Text only | 10.2 | 9.5 | 6.5 | 6.2 | 8.1 |
| Image + Text | 15.6 | 12.6 | 10.8 | 11.3 | 12.6 |
| Combiner trained on CIRR | 15.1 | 12.1 | 13.5 | 15.4 | 14.0 |
| Combiner trained on CC3M (GeneCIS) | 19.0 | 16.6 | 14.7 | 16.8 | 16.8 |

Source: [paper Table 2](https://arxiv.org/html/2306.07969). Chance R@1 is 10% on focus attribute and 6.7% on the other three. Freezing the whole backbone lowers the CC3M Combiner to 15.1 average R@1 (paper Table 5); CoSiR keeps CLIP frozen, so 15.1 is the closer published comparison. The README reports a 0.2 point standard deviation in average R@1 across seeds and calls the benchmark "GeneCIS v0" with known annotation noise, so differences under about 0.2 points are not meaningful.

Two observations shape what a CoSiR result could show. Image-only CLIP already beats text-only and image + text on focus attribute (17.7 vs 15.6), so a conditioned model has to beat the plain image query there, not just a fusion. On the object tasks the trained Combiner's gain over image + text is 3.9 and 5.5 points, which is the size of effect a conditioning mechanism has earned on this benchmark.

## 6. Recommendation

1. Use GeneCIS as a secondary zero-shot diagnostic, as the synthesis brainstorm planned, and keep the human-judged `s(I,T|c)` swap test as the primary evaluation.
2. Start with the object half, which needs no download: map COCO IDs to the 2014 files, cache ViT-B/32 features, and reproduce the three CLIP-only rows with ViT-B/32. That gives the matched baseline before any CoSiR number.
3. Download Visual Genome 1.2 for the attribute half, which carries the focus-attribute task, the closest match to CoSiR's aspect selection.
4. Do not train on the GeneCIS CC3M triplets for the main claim, because they are mined condition labels. If a supervised reference is wanted, the published Combiner weights serve that role without retraining.
5. Retrain factors on natural image-caption data before reading anything into a GeneCIS score; ArtELingo factors encode emotion and art style, which no GeneCIS condition asks about.

## 7. The object tasks as CoSiR episodes

GeneCIS builds both object tasks from COCO Panoptic, counting an object as present when it covers more than 1% of the image, for "thing" and "stuff" classes alike ([paper §4](https://arxiv.org/html/2306.07969)). In focus object the reference is a busy scene, the condition names one object in it, and the target shares at least six object categories with the reference and contains the condition. In change object the condition names an object absent from the reference, and the target is the closest scene that contains it. One template pair shows both: reference 488673 ("A couple of people at a table in a kitchen.") and target 246968 ("A young woman standing in the kitchen pours from a large measuring cup.") appear as focus object with condition "counter" and as change object with condition "light". 663 of the 1,960 index-aligned template pairs share reference and target this way. About 60% of templates use a stuff condition (focus object: 1,130 of 1,960; change object: 1,186 of 1,960), such as "light", "floor", "wall" or "window".

The gallery order carries the distractor type. On the templates whose condition is a COCO thing class (830 focus, 774 change), we looked the condition up in the COCO 2014 instance labels: slots 9 to 13 contained it in 100% of templates, and slots 0 to 8 in 10 to 20%. The 10 to 20% are mostly small instances below GeneCIS's 1% pixel rule. Slots 0 to 8 are therefore similar scenes without the condition, and slots 9 to 13 are other scenes with it. The GeneCIS loader also puts the target first ("by construction, target_rank = 0", `datasets/coco_dataset.py`), the same convention as `tie_aware_rank`.

These facts make the conversion to `LabelEpisodes` (`src/eval/label_episodes.py`) mechanical:

| `LabelEpisodes` field | From GeneCIS |
|---|---|
| `anchor` | reference image row |
| `positive` | target image row |
| `distractors` | the 14 gallery rows, in order |
| `labels` | condition phrase |
| `supports` | slot 9 to 13 images of other templates with the same condition |
| `contrasts` | slot 0 to 8 images of those templates |

Supports and contrasts must exclude every image of the scored template, because images recur heavily (2,198 unique images fill 1,960 × 16 slots in focus object). All 3,033 images have five COCO captions in the local `captions_val2014.json`, so the existing i2t and t2i directions work once we fix which caption forms each row; the GeneCIS-native image to image direction is a small addition to `_scores` in `src/eval/condition_eval.py`. Building supports from GeneCIS's own slots makes the condition few-shot, which is CoSiR's interface but not GeneCIS's zero-shot text protocol, so the two results must be reported apart.

Two cautions apply. First, target captions name the condition word in 23% of focus templates (457) and 25% of change templates (493), so in i2t a candidate caption can match the condition by string; a text-only baseline will show how much this pays. Second, a factor model trained on COCO `train2014` (82,783 images, none of them GeneCIS images) needs only features, a content graph and `group_ids` set to the COCO image id, since `train_factors` (`src/train/train_factors.py`) takes nothing ArtELingo-specific. The old COCO cache in `/data/SSD/coco/preprocess` (113,286 rows, the Karpathy train split) should not be used: that split includes about 30,000 val2014 images, so it may contain GeneCIS images, and it carries no image-id map.

**Preprocessed.** `scripts/preprocess_genecis.py` built the object half in the standard paired format: 15,178 image–caption rows over the 3,033 images (`/data/PDD/genecis/genecis_coco.json`, positional join), byte-identical template copies and a manifest in `/data/PDD/genecis/`, and a CLIP ViT-B/32 feature store at `/data/SSD2/pre_extract/genecis_coco/features`; the dataset config is `configs/dataset/genecis.yaml`. As an alignment check, zero-shot CLIP over the 3,033 images gave t2i R@1 36.9% and i2t R@1 56.6%, against 0.01% with images deliberately misaligned.

## 8. Open items not verified in this check

- Whether upstream has commits after our clone's `0c5c968` and whether the `dl.fbaipublicfiles.com` weight, triplet and scene-graph links still resolve. Network access from the shell needs approval here, so we did not test them. The GitHub issue list showed two open issues, neither about broken links or evaluation bugs.
- The current download URLs, sizes and license terms of Visual Genome 1.2 and COCO images, from their official pages.
- Whether the prompt-route condition adapter produces sensible weights for abstract condition phrases. This needs a short pilot once factors exist on natural images.
