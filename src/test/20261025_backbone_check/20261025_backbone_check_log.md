# Backbone check (throwaway, diagnostic), 2026-10-25 run

Question: which frozen backbone sits beside CLIP ViT-B/32 in the final tables, judged by aspect information in the weaker modality (label-probe ceiling).
Data: ArtELingo = 60,000 scorer-train rows (rng seed 0, as in aspect_ceiling.py) + all selection rows (92,413 rows, 37,738 unique paintings); val/held never read. CUB: all 11,788 images and 117,880 captions.
Features: /data/SSD2/pre_extract/backbone_check/<dataset>/<model>/{img.npy,txt.npy,...} (fp16; ArtELingo rows.npy + row_to_painting.npy, CUB index.json). CLIP ArtELingo comes from the existing cache.

## CLIP reproduction check
All four aspect R@1 values match exactly (9.42 / 10.86 / 10.77 / 13.45). Probes and ceilings differ slightly from the quoted numbers
(probe style img 60.92 vs 60.8; emotion cross 23.78/24.54 vs 23.58/24.29; style same img-img 49.66 vs 49.80; all other deviations <= 0.25).
Running the unmodified `20261023_aspect_episode_spike/aspect_ceiling.py` today gives exactly my numbers, so the difference is not in this pipeline.
Most likely cause: lbfgs with max_iter=300 is not converged and is not bit-reproducible across runs/thread settings. Not verified further.
Tolerance used: 0.006 on R@1, 0.3 on probes/ceilings.

## ArtELingo (selection rows; episodes aspect_episodes.npz)
| model | probe emo img | emo txt | style img | style txt | ceil emo cross (i2t/t2i, mean) | ceil style cross (i2t/t2i, mean) | same img-img emo/style | same txt-txt emo/style | aspect R@1 pooled |
|---|---|---|---|---|---|---|---|---|---|
| CLIP B/32 | 35.1 | 56.9 | 60.9 | 25.4 | 23.78/24.54, 24.16 | 21.51/23.02, 22.27 | 18.04/49.66 | 36.67/13.06 | 11.13 |
| SigLIP 2 So400m/14-384 | 36.3 | 58.7 | 70.7 | 25.9 | 25.54/26.81, 26.17 | 22.63/26.00, 24.32 | 19.14/62.30 | 37.96/12.70 | 10.45 |
| PE-Core L/14-336 | 36.6 | 58.9 | 71.9 | 26.3 | 25.00/26.44, 25.72 | 23.22/26.32, 24.77 | 18.90/64.16 | 38.53/13.09 | 10.94 |
| Qwen3-VL-Embedding-2B | 36.4 | 62.5 | 64.2 | 26.1 | 26.54/28.20, 27.37 | 22.29/24.95, 23.62 | 19.36/53.81 | 43.46/14.40 | 11.93 |

Backbone-only aspect R@1 (emo i2t / emo t2i / style i2t / style t2i): CLIP 9.42/10.86/10.77/13.45; SigLIP2 9.33/10.33/10.01/12.13; PE 9.91/11.28/10.33/12.23; Qwen 10.86/12.77/11.40/12.67. Pooled = mean of the four.

## CUB (train split -> test split, LogisticRegression C=1.0 max_iter=300 on L2-normalised features; caption feature = renormalised mean of the 10 L2-normalised caption features)
Retrieval: 5,794 test images, first caption each.
| model | R@1 i2t | R@1 t2i |
|---|---|---|
| CLIP B/32 | 0.88 | 0.64 |
| SigLIP 2 | 2.26 | 1.38 |
| PE-Core | 1.90 | 1.48 |
| Qwen3-VL-Emb | 1.12 | 1.02 |

Attribute probes (label = the single value with is_present=1 and certainty>=3; images with zero or several such values dropped). Test n: colour 2431, bill shape 5266, size 5018.
| group (n labelled, classes, majority) | CLIP img/txt | SigLIP2 img/txt | PE img/txt | Qwen img/txt |
|---|---|---|---|---|
| has_primary_color (4987, 15, 21.0) | 61.3/62.6 | 61.9/64.5 | 62.5/64.2 | 64.3/64.3 |
| has_bill_shape (10660, 9, 41.4) | 60.8/50.2 | 67.3/55.3 | 67.6/52.9 | 66.3/55.5 |
| has_size (10210, 5, 50.7) | 62.2/57.8 | 62.5/60.2 | 63.2/59.3 | 62.4/60.0 |

## Timings (seconds; GPU RTX 3090 under /tmp/gpu0.lock, 8 loader workers)
| model | load | ArtELingo img (37,738) | ArtELingo txt (92,413) | CUB img (11,788) | CUB txt (117,880) |
|---|---|---|---|---|---|
| CLIP | 6 | cached | cached | 7 | 15 |
| SigLIP 2 (bf16) | 7 | 558 | 99 | 184 | 131 |
| PE-Core (fp16) | 9 | 361 | 44 | 113 | 56 |
| Qwen3-VL-Emb (bf16) | 7 | 2691 | 268 | 390 | 340 |
Total GPU extraction about 62 min (17:26-18:54 container clock); measures are CPU only (about 1-4 min per model for ArtELingo).

## Loading notes
- open_clip_torch 3.3.0, timm 1.0.30, ftfy, regex installed with `pip --target /data/SSD2/pyenvs/backbone_check` (no-deps); CoSiR env untouched. PE-Core loads via open_clip `hf-hub:timm/PE-Core-L-14-336` (text context length 32).
- SigLIP 2: transformers AutoModel get_*_features, text lower-cased, padding=max_length 64 (as the model card requires), bf16.
- Qwen3-VL-Embedding-2B: the model-card script needs qwen_vl_utils and transformers 4.57; instead I reimplemented its recipe on transformers 5.6.2: system prompt "Represent the user's input." + user content, add_generation_prompt, last-token pooling, L2 norm, bf16. transformers 5 needs `mm_token_type_ids` from the processor (passed). Deviation: image max_pixels capped at 512*32*32 (default is 1.31M) for speed, same for all images. Text max_length 512.
- Qwen needed `LD_LIBRARY_PATH` pointing at the CoSiR env's own `nvidia/cu13/lib` (nvrtc-builtins 13.0 missing from the default path); no install.
- CLIP CUB features: transformers get_image_features/get_text_features, fp16, text truncated to 77.
- CUB image_attribute_labels.txt lives under CUB_200_2011/attributes/.
