# Second datasets with affective captions and a second labelled aspect: Affection is the only cross-domain fit, without a second label in its release

> Literature check for r9 decide (step 3), 2026-10-13; Sonnet, standard depth (the 2026-11-28 note already surveys
> datasets for a generality test; this question is narrow). Builds on `2026-11-28_design_l_novelty.md` §6 and
> `2026-10-21_cvpr_literature_review.md`. Two PDFs (Affection, ArtELingo-28) came back as binary, so the agent used the
> ar5iv and arXiv HTML versions. Checked by the controller (2026-10-13 12:39): Affection's arXiv abstract confirms
> 85,007 images and 526,749 responses and names no label besides emotion and explanation. ArtELingo-28's abstract gives
> about 200,000 annotations at 140 per image (about 1,430 images by division) and does not state the image count; the
> agent read 2,000 images from the body; unresolved. Corrected by the controller: AFF does not run unchanged on
> non-English captions (its affect grouping uses GoEmotions, an English classifier).

## Answer

**Best candidate: Affection (Achlioptas et al., CVPR 2023).** It is the only set found with human captions that are
affective explanations, written in the ArtEmis protocol, on real photos. That makes it a cross-domain test (not
paintings). The user's quick view is right on emotion and explanations. The second aspect is **not** in the release.
It would have to be joined from the source datasets, and that join is unverified. Affection's images come from
MS-COCO, Emotional-Machines, Flickr30k Entities, Visual Genome and the Quanzeng et al. image set, with no per-source
counts. COCO images would carry COCO object categories (our inference). Other sources carry boxes or captions, not a
clean class label. PercepT (2606.03345, preprint) says Affection "lacks genre labels" and does not state which labels
it scored against on Affection. So PercepT gives no evidence of a second Affection aspect.

**Runner-up: EmoSet-118K.** It has emotion plus scene, object, action, facial expression, brightness and colour, but
**no human captions**. The attributes are machine-predicted, then human-verified only as yes/no checks (as the paper
implies). Captions would have to be generated, which breaks the "affect from captions" mechanism.

**ArtELingo-28 is the same domain** (WikiArt paintings). It tests other languages and annotator pools, not a new
domain.

No 2024 to 2026 photo dataset with human affective explanations plus a second human label turned up in this budget.

## Table (columns as in the 2026-11-28 note)

| Dataset | Size | Aspects (label source) | Captions, affective? | Licence | Availability | Fit for aspect episodes |
|---|---|---|---|---|---|---|
| **Affection** | 85,007 images; 526,749 explanations; 6,283 annotators; at least 6 annotators per image | emotion (8 + "something-else", human, per annotation); second aspect only by join from sources (COCO categories likely; unverified) | human explanations, yes; mean 18.8 words (ArtEmis 15.9) | own terms of use (PDF, not summarised on page); paper is CC BY-NC-SA 4.0; source image rights stay with sources | request form; "in the process of releasing" | Best. Cross-domain. Join and leakage need a check |
| EmoSet-118K | 118,102 human-labelled (of 3.3M) | emotion (8); scene (365), object (600), facial expr. (6), action (400), brightness, colourfulness (machine-predicted, human-verified) | none released (unverified) | not stated in paper | project page vcc.tech/EmoSet; current status unverified | Many aspects, but no captions |
| EmoVerse (Guo et al., TAffC 2026) | about 234K images | emotion (8, with 1-10 confidence); background-attribute-subject triplets; boxes | MLLM-generated descriptions (not human) | CC BY-NC 4.0 | Hugging Face `PiLabUSTC/EmoVerse-Dataset`; web images as URLs | Machine text; aspect labels are not human classes |
| ArtELingo-28 (Mohamed et al., EMNLP 2024) | 2,000 images per the body (about 1,430 by the abstract's numbers; unresolved); about 200K new annotations (140 per image) in 25 added languages | emotion per annotation (8 + other, human); style (27) and genre (9) per image | human, affective explanations, 28 languages | CC BY-NC-SA 4.0 (stated for the paper; dataset licence not stated) | artelingo.org; "will be made public" | Same domain. Tests language and annotators only |
| ArtEmis v2.0 (Mohamed et al., CVPR 2022) | +260,533 instances | emotion; WikiArt style/genre | human, affective | not checked | artemisdataset-v2.org | Same domain (paintings) |
| SentiCap | about 2,360 COCO val images; 5,013 positive + 4,500 negative captions (secondary source) | sentiment (pos/neg); COCO categories via image ID | human, sentiment-styled, but only polarity | not checked | not checked | Weak: two polarities, not emotions; leaks (caption states the polarity) |
| FlickrStyle10K | 10,000 images built; 7,000 public (secondary source) | romantic / humorous style; Flickr30k captions | human, style not emotion | not checked | partly public | Poor affect fit |
| COCO-4c (earlier note) | 5,000 | activity, location, mood, time of day | COCO captions, rarely state mood | COCO terms | per earlier note | Same-domain issue for AFF (captions lack affect) |
| SemArt / CUB (earlier note) | 21,384 / n/a | art / bird attributes | no affect | n/a | n/a | Expected no gain |

IC|TC (Kwon et al., ICLR 2024) is a method, not a dataset. It clusters existing sets (Stanford 40 Actions) by Action,
Location or Mood from VLM-generated descriptions. It has no human affective captions. WEBEmo and SemStyle were not
checked (SemStyle is a model, not a dataset).

## Per-candidate notes

**Affection** (CVPR 2023, peer-reviewed).
- Emotions: amusement, awe, contentment, excitement, anger, disgust, fear, sadness, and "something-else" (7.6% of
  choices). 71.3% of choices are positive and 21.1% negative; 50.0% of images got both a positive and a negative label.
- Agreement: 67.5% of images (57,381) have a strong majority emotion, against 45.6% for ArtEmis.
- Text: annotators had to cite "at least one specific reference to visual elements". Concreteness 2.82 (ArtEmis 2.81);
  neutral sentiment 10.5% (16.5%); subjectivity 0.53 (0.47). So the explanations read like ArtEmis, with slightly more
  subjectivity. This is the strongest match to the ArtELingo caption style.
- Split: 85/5/10 train/val/test, no image overlap (paper Section 6). The test split has 52,188 explanations.
- Leakage: emotion words appear in the text by design (as in ArtEmis); the paper does not report their rate
  (unverified). A COCO-category join may leak less than style/genre do in ArtELingo, but objects are named in
  explanations by design. That is the "second aspect leaks into captions" risk and needs a count.
- Risk: access latency (request form), and the image-ID file format is unknown. The image-set mix is unequal, and
  non-COCO sources may have no class label.

**EmoSet.** Images come from Openverse, Pexels, Pixabay and Rawpixel (4.3M collected, 3.3M kept). Emotion came from the
query keyword for the 3.3M and from 10 annotators per image (more than 7 agreeing votes) for the 118K. Retrieval used
text tags, but the paper does not say text is released.

**EmoVerse.** It contains EmoSet, EmoArt and Flickr30k subsets plus 14.79% AI-generated images. Its text is
machine-written. It is useful only if machine captions are acceptable.

**ArtELingo-28.**
- Images: 2,000 WikiArt images per the body, a representative subset of the 80K ArtELingo images; 27 styles and 9
  genres. The abstract's numbers (about 200,000 annotations, 140 per image) give about 1,430; unresolved.
- Annotations: 200K new, 10,493 (Urdu) down to 2,032 (Kazakh). 220 annotators from 23 countries.
- Languages: Arabic, Chinese and English carried over, plus 25 new ones: Kinyarwanda, Swahili, IsiZulu, Setswana,
  Yoruba, Hausa, Igbo, IsiNdebele, IsiXhosa, Emakhuwa, Vietnamese, Indonesian, Thai, Burmese, Malay, Tagalog, Tamil,
  Hindi, Urdu, Korean, Turkish, Darija, Uzbek, Kazakh, Kyrgyz.
- Overlap: subset of ArtELingo, so it likely overlaps our held paintings (to be counted). Overlap with ArtEmis images
  is not stated. Splits: not reported (only zero-shot, few-shot, one-vs-all setups).
- Fit: the same labels as our current data. AFF's affect grouping uses GoEmotions, an English classifier, so
  non-English captions need machine translation to English or a multilingual emotion classifier (PercepT translated
  with GemmaX2-28-9B). At 1,430 to 2,000 images it is small for four-example episodes.
- On disk: only a release with genre and emotion for 1,303 paintings (6,515 rows), used as the genre source in
  September (`docs/reports/stage/2026-09-22_artelingo_affect_investigation.md`).

**PercepT** (Mohamed, Church, Elhoseiny, arXiv 2606.03345, preprint). The abstract names only ArtELingo. In the body,
ArtELingo plus ArtELingo-28 give about 1.2M pairs in 28 languages, with emotion and genre labels and machine
translation to English (GemmaX2-28-9B). Affection is about 500K English captions "lacking genre labels". Table 3 uses
ARI, AMI, homogeneity, completeness and V-measure, but the text does not name the reference labels for Affection. The
intro calls ArtELingo-28 "about 1.2M art images", which conflicts with the image count in the ArtELingo-28 paper;
unresolved.

## Addendum: three more datasets from a list the user received (controller, 2026-10-13)

None fits: none has image captions that express affect.

- **WikiArt Emotions** (Mohammad and Kiritchenko, LREC 2018): 4,105 art pieces, mostly paintings, annotated for the
  emotions evoked by the image, the title and both, a liking score (−3 to +3), and whether a face or body is shown. The
  only text is the title (checked on https://saifmohammad.com/WebPages/wikiartemotions.html). Paintings again, small,
  no captions.
- **EMOTIC** (Kosti et al., CVPR 2017): people in context, 26 emotion categories plus valence, arousal and dominance,
  per person box. No captions, and the emotion is the depicted person's, not the viewer's (not re-checked here).
- **MELD** (Poria et al., ACL 2019): emotion and sentiment of utterances in TV dialogue (video, audio, text). The text
  is speech, not an image caption (not re-checked here).

## Ranked recommendation

1. **Affection**, with a COCO-category join as the second aspect. Before committing: request the data and check that
   the image IDs map to COCO, and how many of the 85K are COCO images. Count how often explanations name the COCO
   category (leakage).
2. **Affection with emotion as one aspect and a source-dataset or scene label as the other**, if the COCO subset is too
   small (needs the per-source counts).
3. **ArtELingo-28** as a language-transfer check (same labels; needs translation or a multilingual emotion
   classifier), not as the second domain.
4. **EmoSet** only with generated captions; label it a different mechanism.
5. Skip SentiCap, FlickrStyle10K, IC|TC.

## Unverified

- Per-source image counts of Affection, the image-ID file format, and the exact terms of use (the PDF was not read).
- Whether Affection ships any label beyond emotion and the explanation. Search: the supplementary PDF and the
  `affectivetools/eeai` GitHub repo.
- Emotion-word and object-word rates in Affection explanations (the paper does not report them).
- EmoSet's current download and licence, and whether it has any text.
- ArtELingo-28: image count (1,430 or 2,000), dataset licence, splits, real release status; overlap with our held
  paintings and with ArtEmis images.
- SentiCap and FlickrStyle10K sizes (secondary sources only); licences of SentiCap, ArtEmis v2.
- WEBEmo and any 2024 to 2026 human-written affective photo captions. Search: "affective image captioning dataset
  2025", the AICA paper (2026) and EmoCaliber (Wu et al., 2025) mentioned in a search result.
- The PercepT size conflict (1.2M images against 1,430 to 2,000).

## Sources

- Achlioptas, Ovsjanikov, Guibas, Tulyakov. Affection. CVPR 2023 (peer-reviewed). https://arxiv.org/abs/2210.01946 ;
  summary of Section 3 via https://ar5iv.labs.arxiv.org/html/2210.01946 ; project and terms
  https://affective-explanations.org ; CVF page
  https://openaccess.thecvf.com/content/CVPR2023/html/Achlioptas_Affection_Learning_Affective_Explanations_for_Real-World_Visual_Data_CVPR_2023_paper.html .
  Shows: size, emotions, sources, splits, access, licence, ArtEmis comparison.
- Yang et al. EmoSet. ICCV 2023 (peer-reviewed). https://arxiv.org/abs/2307.07961 ;
  https://ar5iv.labs.arxiv.org/html/2307.07961 . Shows: attributes, machine labelling, sources.
- EmoVerse, Guo et al., IEEE TAffC 2026 (as listed on the dataset page).
  https://huggingface.co/datasets/PiLabUSTC/EmoVerse-Dataset
- Mohamed et al. ArtELingo-28. EMNLP 2024. https://arxiv.org/abs/2411.03769 ; https://arxiv.org/html/2411.03769 .
  Shows: images, languages, labels, licence line.
- Mohamed et al. ArtEmis v2.0 (CVPR 2022). https://arxiv.org/abs/2101.07396 ; data at artemisdataset-v2.org (via
  search result only).
- Mohamed, Church, Elhoseiny. PercepT. arXiv 2606.03345 (preprint). https://arxiv.org/html/2606.03345 . Shows:
  Affection has no genre labels; evaluation labels not stated.
- Kwon et al. Image Clustering Conditioned on Text Criteria. ICLR 2024. https://arxiv.org/abs/2310.18297
- SentiCap / FlickrStyle10K / SemStyle sizes: secondary sources only, e.g. https://arxiv.org/pdf/2308.01143 and
  https://arxiv.org/pdf/2108.11912 (not verified against the originals).
- Earlier notes: `docs/reports/literature/2026-11-28_design_l_novelty.md`,
  `docs/reports/literature/2026-10-21_cvpr_literature_review.md`.
