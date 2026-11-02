# MMArt (PolyArt) preparation, 2026-11-02

A side chat outside the r14 loop, opened by the user at about 14:30: "R14 is doing the sweep and on other datasets,
and it did not include polyart-in cause I don't have access to the dataset around that time, but now I got it
https://huggingface.co/datasets/Shuaimyself/MMArt. So this chat we pre-process the dataset a bit, then we handover to
the main chat later." The user is a co-author of MMArt (ACM MM 2026 dataset track; paper
https://shuaiwang97.github.io/MMArt/ACMMM_MMArt_2026.pdf). "PolyArt" in earlier notes is this dataset.

## What MMArt is

74,234 WikiArt paintings, each with five generated texts (narrative, formal, emotional, historical, unified; about 80
words each, unified about 150), a 20-class WikiArt style, a 9-class `dominant_emotion` (ArtEmis classes incl.
"something else"; 720 empty), artist, title, date. No split. No genre field.

## Steps and findings

1. **Download** (14:33 to 14:45): `hf download Shuaimyself/MMArt --repo-type dataset` to `/data/SSD/mmart/raw`
   (24 GB: 22 image shards and `data/data.parquet`, 110 MB, the texts and labels).
2. **The images are our WikiArt files.** All 74,234 `image_id`s exist under `/data/PDD/wikiart_proj/wikiart`
   (81,446 files), and the shard bytes equal the local files' bytes for all 74,234 (md5). The user confirmed MMArt uses
   the same WikiArt images. So nothing is unpacked; `image_path` points at the local file.
3. **Index table** (`mp_index.py build`, about 2 min CPU): `/data/SSD/mmart/mmart_index.parquet`, one row per
   painting in `data.parquet` order; summary in `mp_index_summary.json` (committed, e5a9e19).
4. **Overlap with ArtELingo** (by painting name, the file stem):

   | Group | Paintings | Note |
   |---|---|---|
   | In ArtELingo held (12,281 held paintings) | 11,528 | features exist in the ArtELingo stores |
   | In ArtELingo train or val | 45,924 | c* trained on them or picked on them: not a test |
   | Not in ArtELingo | 16,782 | the fresh set; 34 are exact byte copies of ArtELingo train/val images |

   547 shared paintings sit in another style folder in MMArt than in ArtELingo. 97 MMArt rows share their md5 with
   another MMArt row (duplicates inside MMArt).
5. **Labels.**
   - Style: MMArt's 20 classes agree with ArtELingo's 27-class style on 55,681 of 57,452 shared paintings (96.9%);
     the rest are merges (Pointillism to post_impressionism 390, Contemporary and New Realism to realism 575,
     Synthetic and Analytical Cubism to cubism 235, Action painting to abstract_expressionism 74, ...).
   - Genre: the WikiArt genre CSVs (`src.data.wikiart_genre`) label 13,739 of the 16,782 new paintings (9,401 of the
     held ones), so all three aspects can be scored.
   - Emotion: MMArt's `dominant_emotion` equals our ArtELingo painting label (majority over captions, ties by name,
     as `ct_held.painting_labels`) on only 43.4% of held paintings (11,398 with a label). MMArt leans to sadness and
     fear (for example, 5,234 ArtELingo "contentment" paintings are "sadness" in MMArt). The paper says the emotional
     view is grounded in ArtEmis utterances, about 5.7 per painting; it does not define `dominant_emotion` further.
     **User's decision:** treat MMArt emotion as its own label set.
6. **Storage** (user): the 22 image shards (24 GB) deleted at 14:48; `data.parquet` kept. Disk ledger row added.

## Feature stores (planned for 19:00, user)

Seven image views for the 16,782 new paintings (Qwen default; the three teacher instructions emotion_focus,
style_word, genre_word; SigLIP 2; SigLIP 1; CLIP L/14) under `/data/SSD2/pre_extract/mmart/`, a 200-painting parity
check against the ArtELingo stores, and a near-copy check of the new set against ArtELingo train/val (cosine >= 0.85,
r14's cut). About 2 h on the local GPU at r14's rates (Qwen about 95 ms/image). The user: "no extraction now, we do
that after 19:00".

## Decisions after the first findings (user, 2026-11-02, about 14:50 to 15:15)

- **Split** (user): "the same split" as ArtELingo. Shared paintings keep their ArtELingo split; the index column
  `split` holds it: train 40,155 (not used: c* trained on them), val 5,769 (validation, scored with MMArt labels),
  held 11,528 (an extra test: ArtELingo's held paintings with MMArt's own labels), new 16,782 (the main test, minus
  the 34 exact copies and the near-copies that tonight's check finds).
- **Emotion labels** (user): MMArt's `dominant_emotion` is its own label set.
- **Aspects and instructions** (agent, user agreed): the three aspects stay emotion, style, genre with the ArtELingo
  teacher instructions; MMArt's four perspectives are captions, not aspects. Two uses of them are parked in the
  direction log (user): perspective retrieval (captions as queries in c*'s aspect spaces) and perspective names as
  new Qwen instructions.
- **Tonight's queue** (user, after seeing the recommendation): the seven views for the 16,782 new paintings only. Val
  teacher views and caption embeddings wait for the decide chat that designs the MMArt test (about 25 min GPU for the
  val teacher views).
- **Storage** (user): the 22 image shards deleted, `data.parquet` kept.

## The extractor

Codex (`gpt-6.1-sol`, medium) wrote `mp_extract.py`, `mp_stores.py`, `mp_extract_queue.sh` and 16 CPU tests in a
worktree (commit 4b8dbda); the controller read the diff, fixed the smoke status path and added the split column
(2b18f42), merged into `r14-backbones-sweep` (1da403f). The Qwen views assert revision
`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` and take the teacher instruction from the ArtELingo `*_held` store's
`meta.json` (checked against `pav_prompts.py`).

**GPU smoke** (14:57, 76 s, 4 images per store, scratchpad): all 14 stores sealed; parity against the ArtELingo
stores, min cosine per view: Qwen default 0.99974, emotion 0.99980, style 0.99971, genre 0.99976, SigLIP 2 0.99991,
SigLIP 1 0.99987, CLIP L/14 1.00000 (bar 0.999); store check 14 of 14; near-copy step runs. Rates: Qwen about 88
ms/image, SigLIP about 26, CLIP L/14 about 13: the full run is about 2 h (new set) plus the 200-painting parity set.

## Run (2026-11-02 18:01 to 19:55, local GPU, this decide chat)

`mp_extract_queue.sh` under the GPU lock; it ended "MP QUEUE DONE ok in 6816s". Log:
`/data/SSD2/pre_extract/mmart/logs/queue_20261102_180138.log`.

- **Stores** (16,782 new paintings each): Qwen default 1,449 s (86.3 ms/image), emotion_focus 1,489 s, style_word
  1,472 s, genre_word 1,466 s, SigLIP 2 324 s (19.3 ms), SigLIP 1 323 s (19.2 ms), CLIP L/14 134 s (8.0 ms). The
  200-painting parity stores took 2 to 18 s each.
- **Parity** against the ArtELingo stores, min cosine over 200 paintings (bar 0.999): Qwen default 0.99935, emotion
  0.99976, style 0.99970, genre 0.99969, SigLIP 2 0.99978, SigLIP 1 0.99996, CLIP L/14 1.00000. All pass
  (`mp_parity.json`).
- **Store check**: 14 of 14 stores pass (`mp_stores_check.json`).
- **Near-copy check** (default Qwen image of each new painting against the 49,121 ArtELingo train and val paintings;
  `/data/SSD/mmart/mmart_nearcopy.parquet`, summary `mp_nearcopy.json`): at cosine >= 0.85, 6,375 paintings; >= 0.90,
  2,249; >= 0.95, 390; >= 0.99, 49. The median nearest cosine is 0.829, so 0.85 cuts 38% of the set.
- **What the bands hold** (nearest ArtELingo painting has the same artist): 0.85 to 0.90, 45% of 4,126; 0.90 to 0.95,
  70% of 1,859; 0.95 to 0.99, 96% of 341; >= 0.99, 76% of 49. Samples at 0.85 to 0.99 are different paintings,
  mostly by the same artist (a Serov portrait next to a Serov self-portrait; two Schongauer "wise virgin" prints at
  0.959). Samples at >= 0.99 are the same painting under another name (`...-avenue.jpg` against `...-avenue-1`,
  cosine 1.000). On paintings, 0.85 removes neighbours by artist and subject, not only copies; on EmoSet photos it
  removed 2% (r14).
- **Test cut** (exact copies, then near-copies, then the md5 groups; D5, D10): at 0.85, 10,403 paintings. The user
  moved the threshold to 0.95 (2026-11-02 21:00, after seeing the recommendation; spec D19): **16,384 test
  paintings**. The 0.85 set (10,403) becomes the far-subset diagnostic. All 34 exact copies have cosine >= 0.9995, so
  they fall inside every threshold.
