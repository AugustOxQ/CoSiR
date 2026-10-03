# E4 feature extraction (Task 16) log

## Problem
Later experiments need frozen-backbone features that do not exist yet: ArtELingo full (Qwen), SemArt (both),
COCO train2014 (both), GeneCIS Visual Genome crops (both), GeneCIS COCO (Qwen). CUB features for both backbones come
from the backbone check and are reused.

## Setup
- `scripts/extract_features.py --dataset NAME --backbone {clip_b32,qwen3vl_emb_2b} [--limit N]`; adapters yield one
  image list, one text list and per-row metadata. Output `/data/SSD2/pre_extract/<dataset>/<backbone>/{img.npy,
  txt.npy, index.json, progress.json}`, float16, L2-normalised; `--limit` writes to `_smoke/` and may overwrite there.
  A finished output (index.json) is refused without `--overwrite`; an unfinished one resumes from progress.json.
- Alignment: `artelingo_full` text rows follow `load_artelingo()` order (sample ids from `FeatureManager`), one image
  per painting (61,402; 308,723 captions, both asserted). `genecis_coco` text row j is `genecis_coco.json[j]`
  (asserted equal to the B/32 store's sample ids 0..15177). `coco_train2014` sorted by image id, captions by
  annotation id. `semart`: train, val, test CSVs in that order, description scrubbed by `scrub_semart`.
- Truncation: every text hitting the tokenizer limit (CLIP 77 tokens; Qwen 512 tokens of the full chat prompt, which
  loses the generation suffix) is counted and stored as `n_texts_truncated` in index.json.
- Batches: Qwen images 32, texts 128; CLIP images 256, texts 512. Images load in 8 threads, one chunk ahead.
- `scrub_semart` removes the title string, the author's name words (3+ letters, particles such as the/of/de/van kept)
  with a trailing possessive, and every 3 or 4 digit number. Centuries written in words ("16th century") stay.

## GeneCIS access disclosure
`genecis_vg_crops` opens `/project/genecis/genecis/focus_attribute.json` only to build the union of (image_id, bbox)
crops over every role (reference, target, gallery). It does not store or print any role, condition text, target
identity or gallery membership. `index.json` holds only `image_id` and `bbox` per row, sorted deterministically
(numeric image id, then bbox), so the crop list cannot reveal targets. Nothing is scored; this is a disclosed access,
not a held read. The union has 15,773 crops over 13,348 Visual Genome images, all present in `VG_100K_all`.
`genecis_coco` reads `/data/PDD/genecis/genecis_coco.json` (an item list, no templates).

## Smoke results (`--limit 32`, GPU shared with other jobs of this session, throughput includes warm-up)
| dataset / backbone | images | texts | img/s | txt/s | truncated |
|---|---|---|---|---|---|
| artelingo_full / qwen | 32 | 158 | 10.4 | 333 | 0 |
| semart / clip | 32 | 32 | 78.7 | 523 | 17 (77 tokens) |
| semart / qwen | 32 | 32 | 9.6 | 31.7 | 1 (512 tokens) |
| genecis_vg_crops / clip | 32 | 0 | 84.8 | | |
| genecis_vg_crops / qwen | 32 | 0 | 14.7 | | |
| genecis_coco / qwen | 32 | 161 | 15.6 | 397 | 0 |
| coco_train2014 / clip | 32 | 160 | 93.4 | 2609 | 0 |
| coco_train2014 / qwen | 32 | 160 | 15.0 | 371 | 0 |

All smoke arrays finite, row norms 1.000. Caption-to-image top-1 inside the 32-image smoke set: coco clip 0.81,
coco qwen 0.94, genecis_coco qwen 0.93, artelingo qwen 0.69, semart clip 0.53, qwen 0.59.

Projected full runs (smoke throughput, so conservative): artelingo qwen 98 min images + 15.5 min texts; semart clip
5 + 1 min, qwen 37 + 11 min; vg crops clip 3 min, qwen 18 min; genecis_coco qwen 3 + 1 min; coco clip 15 + 3 min,
qwen 92 + 19 min. Total about 5.3 h. Final sizes: about 4.4 GB (artelingo qwen 1.5 GB, coco qwen 2.0 GB, rest under
0.6 GB); `/data/SSD2` has 2.0 TB free.

## Full-run results
(counts, timings and SHA-256 of each index.json are appended here after the full run)
