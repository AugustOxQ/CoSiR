# Second photo emotion set and photo style sets: EmoSet-118K is the best second emotion test, Flickr Style the best other aspect; FI leaks (it is an Affection source)

> Dataset check for r14 decide (step 3, side job), 2026-11-02; Sonnet, standard depth (the 2026-10-13 note
> `2026-12-01_affect_caption_datasets.md` already covers the area and names EmoSet; this question is narrow). Builds on
> that note and `2026-11-28_design_l_novelty.md` §6. About 14 searches and 11 page reads; nothing downloaded. Facts seen
> only in a search snippet are marked "snippet".
>
> **Checked by the controller (2026-11-02 02:40):** the agent's inference that FI is one of Affection's sources is
> confirmed by our own records: `src/test/20261029_r11_data/20261029_r11_data_log.md` lists "You et al. (FI) 21,808"
> among Affection's sources, and `/data/SSD/affection/fi_t4sa/` exists locally. So FI is out as a second test set.
> Affection's 9 classes are the 8 Mikels classes plus "something else", so EmoSet's map is one-to-one on 8 classes.
> The EmoSet download route, licence and overlap stay unverified (see Unverified).

## Question

For r14: (A) a second photo emotion dataset for a test of the photo model on data that did not shape the method
(images and emotion class labels only); (B) photo datasets with class labels for another perceptual aspect (style,
mood), and whether they have text per image.

## Answer

**A. Second photo emotion test set**

1. **EmoSet-118K** (ICCV 2023, peer-reviewed). 118,102 human-labelled photos, large enough for a stable AMI and mAP.
   Its 8 classes are Mikels' (amusement, awe, contentment, excitement, anger, disgust, fear, sadness), the same 8 as
   Affection's; only Affection's "something else" has no counterpart, so it is dropped when comparing. Image sources
   (Openverse, Pexels, Pixabay, Rawpixel) are none of Affection's; near-duplicates look unlikely, not checked. Main
   risk: the data licence is unclear (the repo shows MIT, no separate data terms); the Dropbox link may rot. Confirm
   the link before the spec.
2. **Emotion6** (Peng et al., CVPR 2015; 1,980 Flickr images). Ekman 6 plus neutral, as distributions; the map to
   Affection's classes is lossy (joy splits three ways, surprise to awe at best). Too small for strong claims. Access
   by request (snippet). Flickr source, so the Flickr30k overlap is not checked.
3. **FindingEmo** (NeurIPS 2024 Datasets track; 25,869 images). Plutchik's 24 emotions with valence and arousal, on
   multi-person social scenes; the map to the 8 classes is coarse. Only a URL list is released, so images may have
   rotted. Licence not found.

- **Excluded: FI** (You et al. 2016, 23,308 images, Mikels 8): an Affection source (controller's check above), so
  its images sit in Affection's splits; a leak.
- **Excluded too:** EmoVerse (contains EmoSet and Flickr30k subsets plus 14.79% AI images; web images as URLs) and
  ArtPhoto (806 images).

**B. Another perceptual aspect on photos**

1. **Flickr Style** (Karayev et al., BMVC 2014). 80,000 Flickr photos, 20 style labels (techniques, moods such as
   Serene and Melancholy, atmosphere such as Hazy and Sunny). Labels come from curated Flickr groups and are not
   exclusive. No text. The only route is the author's page, not confirmed today (one URL gave 404). Risk: access and
   link rot; Flickr images could overlap Flickr30k (not checked).
2. **AVA style subset** (Murray et al., CVPR 2012). 14 photographic style labels on part of AVA (about 30,957 items,
   snippet). Images from DPChallenge; the original download is partly dead, Hugging Face mirrors exist (about 255k
   images, 33 GB, third-party, licence uncertain). No text in AVA, but DPC-Captions and DPC-CaptionsV2 give per-image
   comments for DPChallenge images (the ID join is not checked).
3. **Photozilla** (arXiv 2106.11359, preprint). Over 990k images, 10 photographic styles; labels seem classifier-made
   (snippet). No text found. Weakest label quality of the three.

- Not usable now: **DataSeeds** (10,610 photos with mood and captions, 2025 preprint; release not confirmed).
- **Local Unsplash** (`/data/SSD/unsplash`): its README names photos, keywords, collections, conversions and colours
  tables plus a captions file; no style or mood class label seen (keywords are free text). The tsv files were not
  opened.

## Table

| Dataset | Size | Labels (how made) | Text per image | Licence | Route today | Affection overlap |
|---|---|---|---|---|---|---|
| EmoSet-118K | 118,102 | emotion 8 (human), plus scene, object, action, face, brightness, colour (machine, checked) | none confirmed; JSON per image | repo MIT; data terms unclear | Dropbox zip via GitHub (not tested) | no shared source; not checked |
| FI | 23,308 | Mikels 8 (AMT, 5 workers) | none | not found | no link found | yes: an Affection source |
| Emotion6 | 1,980 | Ekman 6 + neutral, distributions | none | not found | request to Penn State | Flickr; not checked |
| FindingEmo | 25,869 | Plutchik 24, valence, arousal (Prolific, 655 raters) | none | not found | URL list plus GitLab code | not checked |
| WEBEmo | about 268k | 25 emotion classes, weak (keywords) | tags only (unverified) | not found | no link found | not checked |
| ArtPhoto | 806 | Mikels 8 | none | not found | not found | not checked |
| Twitter I / II | 1,269 / 603 | positive/negative only | tweet text for II | research use only | not found | none useful |
| EmoVerse | 234,189 | Mikels 8, confidence, boxes | MLLM description | CC BY-NC 4.0 | Hugging Face; web images as URLs | includes EmoSet and Flickr30k parts |
| Flickr Style | 80,000 | 20 styles (group membership) | none | not found | author page only | Flickr; not checked |
| AVA style | about 31k labelled (of about 255k) | 14 styles (from challenge titles) | none; DPC comments join | DPChallenge terms; mirrors vary | HF mirrors; original partly dead | none expected |
| Photozilla | over 990k | 10 styles (classifier-made, snippet) | none found | not found | project page | not checked |
| DataSeeds | 10,610 | mood, captions | yes | not found | release unconfirmed | none expected |

## Per-candidate notes

- **EmoSet.** The 2026-10-13 note's open questions: the download is a Dropbox zip linked from the GitHub README (not
  form-gated); the licence is MIT for the repo with no data-specific terms; annotation JSON files sit in per-emotion
  folders, and the README names no caption. So "no text" stands, unverified. Images are third-party stock, so their
  own rights remain.
- **FI.** AICA-Bench (2026) still uses it (1,000 of 23,308), so copies exist; no hosting page found. Counts conflict
  (23,308 against 23,164 in one table).
- **WEBEmo.** About 268k weakly labelled stock images, 25 classes (arXiv 1808.02212); labels are search keywords, so
  the noise is high. No download link found.
- **EmoVerse.** CC BY-NC 4.0; web images are URL and script only. It contains EmoSet, so it cannot be a second set
  beside EmoSet.
- **Twitter I/II.** Binary sentiment; no use for 8-class AMI.
- **EmoScene** (arXiv 2604.00933, 2026, preprint). 1.2M images, labels from models with human quality control, so not
  a human test set.
- **EL-CUES** (2025). 210 images, too small.
- **FlickrLDL / TwitterLDL** (10,700 / 10,045) appear in AICA-Bench with Mikels 8 label distributions; not checked; a
  possible extra A candidate.

## Unverified

- EmoSet: the Dropbox link works today, the licence covers the data, no text in the JSON, no overlap with COCO,
  Visual Genome or Flickr30k. Settle with one download of the smallest file, or a perceptual-hash check against the
  local sets.
- Flickr Style: the author's page returned 404 at `sergeykarayev.com/recognizing-image-style/`; the paper cites the
  http form. The HF set `disl/image-style` has 3 classes and no card, so it is not this one.
- AVA style labels per image, the DPC-Captions to AVA ID join, and the licence of the mirrors.
- Emotion6, ArtPhoto, WEBEmo and FindingEmo licences and download routes.
- Photozilla label quality and text; the UNIAA "30,957 style items" number (snippet).
- Overlap of any candidate with COCO, Visual Genome and Flickr30k: not checked for any.

## Sources

- Affection, Achlioptas et al., CVPR 2023: https://arxiv.org/abs/2210.01946 ; PDF
  https://www.lix.polytechnique.fr/~maks/papers/Affection_CVPR2023.pdf (not read); our source counts:
  `src/test/20261029_r11_data/20261029_r11_data_log.md`
- EmoSet, Yang et al., ICCV 2023: https://arxiv.org/abs/2307.07961 ; repo https://github.com/JingyuanYY/EmoSet
- EmoVerse card: https://huggingface.co/datasets/PiLabUSTC/EmoVerse-Dataset/blob/main/README.md
- FI, You et al. 2016: https://arxiv.org/abs/1605.02677 (snippet only)
- WEBEmo, Panda et al. 2018: https://arxiv.org/pdf/1808.02212
- FindingEmo, Mertens et al., NeurIPS 2024:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/08a7229eaba3b35cd8f933ac678f2096-Abstract.html
- AICA-Bench 2026 (preprint), dataset sizes: https://arxiv.org/html/2604.05900
- Emotion6, ArtPhoto and Penn State access: https://wang.ist.psu.edu/emotion (snippet)
- Twitter datasets: https://www.ee.columbia.edu/dvmm/vso/download/twitter_dataset.html ; https://arxiv.org/pdf/2511.23115
- Flickr Style, Karayev et al.: https://arxiv.org/abs/1311.3715
- AVA: https://deepsense.ai/wp-content/uploads/2023/04/MMP2012a.pdf ; mirrors https://github.com/imfing/ava_downloader ,
  https://huggingface.co/datasets/Iceclear/AVA/tree/main
- UNIAA 2024: https://arxiv.org/pdf/2404.09619 ; DPC-Captions: https://arxiv.org/pdf/1907.04983 ; DPC-CaptionsV2:
  https://arxiv.org/abs/2208.04522
- Photozilla (preprint): https://arxiv.org/abs/2106.11359
- DataSeeds (preprint): https://arxiv.org/abs/2506.05673
- EmoScene (preprint): https://arxiv.org/pdf/2604.00933
- Local Unsplash README: `/data/SSD/unsplash/README.md`
