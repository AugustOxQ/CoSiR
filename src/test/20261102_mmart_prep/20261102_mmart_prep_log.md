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
