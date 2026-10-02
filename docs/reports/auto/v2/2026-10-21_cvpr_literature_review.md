# CVPR literature review for CoSiR v2: prior art, baselines and benchmarks

**Report date:** 2026-10-21 (sequence date in this folder; the research was done on 2026-10-02).
**Scope:** read-only literature review for the CVPR submission (abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23). We trained nothing and ran no evaluation. We fetched every cited paper's arXiv, venue or repository page and checked title and authors. Numbers are copied from each paper's own tables, with the table number (written T2 for Table 2). Anything we could not open is marked **UNVERIFIED**; where we draw a conclusion a paper does not state, we write "we infer". Three subagents swept threads in parallel (GeneCIS numbers, benchmarks, affect and instruction-conditioned embeddings); we spot-checked their key rows (CIReVL T3, OSrCIR T2, CLAY T2(a), CRL T3, GeneCIS T6, FocalLens T1, COCO-Facet T1, TEVI T2, the PercepT encoder) against the arXiv HTML and found them correct. The session's web search budget (200 searches) ran out late in the sweep; after that we searched arXiv titles and abstracts through the arXiv API and fetched known pages directly.
**Builds on:** the [Stage 1 to GeneCIS synthesis](2026-09-28_stage1_genecis_synthesis_brainstorm.md) (an earlier, shorter survey) and the [GeneCIS feasibility check](2026-10-20_genecis_feasibility.md) (GeneCIS counts, licence and on-disk data). We extend those and do not repeat them.

**CoSiR v2 in the terms used below.** CoSiR v2 scores an image I and a caption T under a condition c, s(I,T|c), in both retrieval directions. The condition is given **by examples**: 4 support image–caption pairs that have the wanted aspect and 4 contrast pairs that do not. Two small encoders map frozen CLIP ViT-B/32 image and text features into a shared, sparse, non-negative 32-d factor space; the parameter-free naive rule sets w = ReLU(mean support code − mean contrast code), L1-normalised; the score is β·cos(CLIP_I, CLIP_T) + Σ_l w_l a_I,l(I) a_T,l(T). The factors are trained without ArtELingo labels, from condition episodes built on k-means of GoEmotions RoBERTa probabilities of the captions and k-means of CLIP image features (distant supervision). Held ArtELingo label episodes (anchor, 4 supports, 4 contrasts, 13 candidates with 1 positive, R@1): CLIP only 13.2, SE 21.2 pooled (emotion 17.0, style 25.5). After this review started, the user widened the backbone: any frozen encoder is eligible, including encoder pairs that do not share an embedding space (§2.8).

## 1. Verdict

The closest prior art falls into three lines that no paper we found combines: text-conditioned similarity (GeneCIS, and on frozen vision–language encoders CLAY, CVPR 2026, and CRL, NeurIPS 2025), image-only similarity whose condition is inferred from examples or learned without condition labels (SCE-Net, DiscoverNet, few-shot attribute learning, classic relevance feedback), and cross-modal retrieval of a personal instance from a few example images (PALAVRA, POLAR). (a) We found no paper that scores an image against a caption of another item under a condition given by support and contrast image–caption pairs, so the novelty risk for example-conditioned cross-modal similarity is moderate; but the naive rule is a Rocchio relevance-feedback update applied to factor codes and the weighted factor score is a Conditional Similarity Network mask, so the novelty must rest on the combination and on the shared image–text factors, not on either mechanism. (b) The risk for condition learning without condition labels is high: SCE-Net (ICCV 2019) and DiscoverNet (CVPR 2022) already learn similarity conditions without condition labels, GeneCIS and MagicLens mine conditions from captions, and EmotionCLIP (CVPR 2023) already used a frozen text sentiment classifier as distant supervision for vision–text contrastive learning, so the claim has to shrink to "no labels from the evaluation taxonomy". A second risk sits in our own protocol: in the ArtELingo label episodes the positive is the only candidate that carries the support label, so an anchor-free 4-shot prototype on raw CLIP may solve them, and a reviewer will ask for that baseline first. The must-have baselines are a support prototype and a Rocchio query on raw features (with and without the anchor), a Tip-Adapter style cache and a linear probe on the eight support and contrast pairs, a text-named condition on the same frozen encoder (label prompt, CRL projection), the naive rule on unsupervised codes and on the GoEmotions teacher, PercepT, and on GeneCIS the published frozen ViT-B/32 rows (SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1). Because any frozen encoder is now eligible, an instruction-following embedder with the condition written into its instruction (Qwen3-VL-Embedding, GME or VLM2Vec-V2, all with open weights) is both a must-have text-named baseline and a candidate backbone; a pair of unimodal encoders would remove the β·cos term and need a LiT, SAIL or ASIF style alignment. The most promising extra benchmark is CUB-200-2011 with Reed et al.'s human captions (species crossed with attribute groups, captions written without species names), with GeneCIS kept as the reviewer-facing standard and SemArt as a second art benchmark.

## 2. Literature by thread

### 2.1 Conditional similarity learning

Conditional similarity began as image-only metric learning with a known condition id (CSN), moved to conditions inferred without labels (SCE-Net, DiscoverNet), and since 2023 has moved to open text conditions on large pretrained encoders (GeneCIS, then the training-free methods of §2.6). GeneCIS numbers of later methods are in Table 2.

**Table 1. Conditional similarity methods.**

| Paper | Condition form | Modalities | Backbone, frozen? | Benchmarks, metric | Headline numbers (source) | Code | Closeness to CoSiR v2 |
|---|---|---|---|---|---|---|---|
| CSN, Veit, Belongie, Karaletsos, CVPR 2017 | attribute id selects a learned mask over embedding dimensions | I→I triplets | CNN trained end to end | UT-Zappos50k (4 notions), fonts; triplet error | 10.73% error, learned masks (quoted in SCE-Net T5) | yes | our factor term is a CSN-style masked similarity whose mask comes from examples instead of an id |
| SCE-Net, Tan, Vasileva, Saenko, Plummer, ICCV 2019 | none at test time; a weight branch mixes K learned masks computed from the inputs (on Zappos, the concatenated triplet images) | I→I; outfit items | ResNet-18 trained | Polyvore Outfits, Maryland Polyvore (compatibility AUC, FITB), UT-Zappos50k (triplet error) | Zappos error 7.53% with 4 masks vs CSN 10.73% (T5); Polyvore Outfits AUC 0.91, FITB 61.6 vs 0.86, 55.3 for the type-aware network (T1) | yes | the closest image-only analogue of inferring the condition from examples; training triplets are still sampled per attribute |
| DiscoverNet, Ye, Shi, Zhan, CVPR 2022 | weakly supervised: triplets without condition labels; a set module matches triplets to embeddings | I→I | trained CNN | UT-Zappos-50K, Celeb-A | not verified (CVF PDF returned 403) | not found | label-free condition discovery from triplets |
| Generalized CSL, Shi, Li, Gan, Zhan, Ye, TPAMI 2025 | supervised, weakly supervised or semi-supervised conditions | I→I | trained | not verified | not verified | not checked | the latest paper of the label-free line |
| Few-shot attribute learning, Ren et al., arXiv 2012.05895 (technical report; v1 titled "Flexible Few-Shot Learning with Contextual Similarity") | a support set of positives and negatives that share one or two attributes | image classification episodes | trained, self-supervised vs supervised pretraining | Celeb-A, Zappos-50K, ImageNet-with-attributes | claims supervised pretraining generalises poorly to unseen attributes; no numbers copied | not checked | the same episode design as our label episodes, image only |
| ASEN, Ma et al., AAAI 2020; ASEN++, arXiv 2104.02429 | attribute id drives spatial and channel attention | I→I | trained CNN | FashionAI, DARN, DeepFashion, Zappos50k; MAP | DeepFashion mean MAP 8.74 and 9.64 (quoted in CRL T4) | yes | supervised attribute-specific similarity |
| Conditional Image-Text Embedding Networks, Plummer et al., ECCV 2018 | the phrase is softly assigned to K conditional embeddings | phrase↔region grounding | trained | Flickr30K Entities, ReferIt, Visual Genome | +3 to 4% grounding (abstract only) | not checked | an early cross-modal model with several conditional subspaces; the condition is inferred from the phrase itself |
| GeneCIS, Vaze, Carion, Misra, CVPR 2023 | free text phrase | I+T→I | CLIP RN50x4 fine-tuned on 1.6M triplets mined from CC3M | GeneCIS, R@1/2/3 | average R@1 16.8 (T2), 15.1 with the backbone frozen (T5), 17.6 with ViT-B/16 (T6) | yes; RN50x4 and ViT-B/16 weights, CC3M only | ancestor; conditions mined from captions |
| InDiReCT, Kobs, Steininger, Hotho, WACV 2023 | a few text prompts define the similarity notion | I→I | frozen CLIP; a projection trained on text embeddings | LanZ-DML (5 datasets, 13 notions), MAP@R | not copied | yes | the earliest frozen-CLIP, text-defined notion of similarity |

**Learning conditions without condition labels.** Three groups of papers already claim it, and each narrows our claim. SCE-Net and DiscoverNet learn condition subspaces from triplets without condition labels and infer the condition from the compared items at test time; their supervision is the triplet itself, which in UT-Zappos50k is generated per attribute. GeneCIS and MagicLens mine conditions automatically from caption structure (scene graphs; LLM-written instructions): no labelled taxonomy, yet still supervision. Multi-clustering methods that find several partitions of the same images under a user criterion (IC|TC, ICLR 2024; Multi-MaP, CVPR 2024) resemble our use of two k-means pseudo-partitions to make condition episodes, although they cluster and do not retrieve. Distant supervision from an external affect classifier also has a precedent: EmotionCLIP (Zhang, Pan, Wang, CVPR 2023) passes captions through a frozen DistilBERT sentiment model (7-d pseudo sentiment scores) and reweights the negatives of its contrastive loss by the KL divergence between those scores (§3.4, Eq. 5 and 6). We infer that a reviewer will accept "trained without labels from the evaluation taxonomy, with distant affect supervision from GoEmotions" and will reject "unsupervised condition discovery".

### 2.2 Composed image retrieval and zero-shot CIR

Composed image retrieval (CIR) ranks target images for a reference image plus a modification text; zero-shot CIR (ZS-CIR) trains without CIR triplets. The standard benchmarks are FashionIQ, CIRR, CIRCO and GeneCIS. The GeneCIS subagent found 26 papers from 2023 to 2026 that report GeneCIS numbers in some form. Table 2 lists the rows that matter for us, as **focus attribute / change attribute / focus object / change object / average R@1**. All conditions are text phrases; none is example-conditioned.

**Table 2. Methods that report GeneCIS numbers (selected).**

| Method, venue | Condition and modality | Backbone, frozen? | GeneCIS R@1 (source table) | Code | Closeness |
|---|---|---|---|---|---|
| GeneCIS Combiner, CVPR 2023 | text; I+T→I | RN50x4 fine-tuned | 19.0 / 16.6 / 14.7 / 16.8 / 16.8 (T2); 15.1 frozen (T5) | yes | ancestor |
| CIReVL, Karthik et al., ICLR 2024 | text via BLIP-2 caption and LLM rewrite, then CLIP text→image | frozen CLIP, training-free | **ViT-B/32 17.9 / 14.8 / 14.6 / 16.1 / 15.9**; SEARLE re-run at B/32 18.9 / 13.0 / 12.2 / 13.6 / 14.4 (T3) | yes | training-free on frozen CLIP; needs an LLM |
| LinCIR, Gu et al., CVPR 2024 | text; projection trained on text only | frozen CLIP L, H, G | ViT-L 16.90 / 16.19 / 8.27 / 7.40 / 12.19; re-runs Pic2Word (L avg 11.16) and SEARLE (L avg 12.26) (T B.3); no B/32 row | yes | light projection on frozen CLIP |
| CompoDiff, Gu et al., TMLR 2024 | text (optional negative text); diffusion over CLIP embeddings | frozen CLIP RN50 to ViT-G | average only: RN50 14.65 to ViT-G 15.48 (T3) | yes | far |
| MagicLens, Zhang et al., ICML 2024 | open text instruction | CLIP or CoCa, fine-tuned | CLIP-B (B/16) 15.5 / 12.3 / 14.4 / 17.7 / 15.0 (T16) | yes | far |
| OSrCIR, Tang et al., CVPR 2025 | text; GPT-4o reasons over image and text, then CLIP retrieval | frozen CLIP, training-free | **ViT-B/32 19.4 / 16.4 / 15.7 / 18.2 / 17.4** (T2); Paracosm v1 (ECCV 2026) reproduced 14.0 with GPT-4o | yes | training-free; reproduction disputed |
| PrediCIR, Tang et al., CVPR 2025 | text; trained mapper | frozen CLIP L/14 | 18.2 / 18.7 / 12.7 / 16.9 / 16.6 (T9) | yes | medium |
| RTD, Byun et al., ICCV 2025 | text; post-hoc text-encoder tuning | image encoder frozen | B/32 average: Pic2Word 11.13, SEARLE 12.19, LinCIR 12.23 before RTD (T10) | announced | B/32 runs of three ZS-CIR methods |
| MegaPairs (MMRet), Zhou et al., ACL 2025 | text plus image | CLIP-B, CLIP-L unfrozen; MLLM | MMRet-Base 18.3 / 15.2 / 16.6 / 21.7 / 18.0 (T9) | yes | far |
| LamRA, Liu et al., CVPR 2025 | instruction, image and text | Qwen2-VL-7B with LoRA | one R@1: 18.9, 24.8 with reranking; its own runs of CLIP-L 13.3, E5-V 18.5, UniIR-CLIP 16.8, MagicLens-L 16.3, EVA-CLIP-8B 13.1 (T4) | yes | far |
| STiTch, Li et al., arXiv 2605.21261 | text; transition vector and optimal transport | frozen CLIP | B/32 21.1 / 17.9 / 16.4 / 18.3 / 18.4 (T8) | not found | medium |
| SQUARE, Wu et al., arXiv 2509.26330 | text; GPT-4o captions and MLLM reranking | frozen CLIP | B/32 25.6 / 19.0 / 17.3 / 16.8 / 19.7; 16.4 without the reranker (T3) | not found | far |
| DIOR, Kawarada et al., arXiv 2512.21860 | condition word in an LVLM prompt; I→I | frozen LVLM (Llama-3.2-Vision 11B is the main model) | focus tasks only: focus attribute 24.0, focus object 21.1 (T C-1) | yes | training-free conditional similarity |
| CRL, Liu et al., NeurIPS 2025 | criterion text → LLM-written basis | frozen CLIP ViT-B/16 | object half, training-free: focus object 15.4, change object 17.0 vs CLIP image + text 11.5 / 9.8 and the CC3M Combiner 16.6 / 18.0 (T3); we infer the object half from the CLIP rows, which equal GeneCIS T6 | yes | training-free, frozen CLIP |
| SteerViT, Ruthardt et al., arXiv 2604.02327 | text cross-attention inside a frozen ViT | DINOv2 ViT-B/14 frozen plus about 21M trained parameters | focus object only: 25.4 vs DINOv2 9.6 and an unnamed "specialized" baseline 18.7 (T6) | project page | conditional representation, text condition |
| FocalLens, Hsieh et al., arXiv 2504.08368 | text instruction | CLIP ViT-L/14-336 with the vision tower trained | focus attribute and focus object only, at **R@3**: 43.30 / 43.72 (T1); not comparable with R@1 | not found | close in problem |

**Frozen CLIP ViT-B/32 rows on GeneCIS** (average R@1): Pic2Word 11.13, SEARLE 12.19 and LinCIR 12.23 (RTD T10 runs); SEARLE 14.4 (CIReVL T3 run); CIReVL 15.9; Paracosm 16.1; DeCIR 16.5 (LoRA on CLIP); PACT 17.23 (LoRA on the text encoder, validation split); OSrCIR 17.4 (disputed); STiTch 18.4; SQUARE 19.7 with an MLLM reranker. Paracosm v1 gives trivial B/32 baselines of 11.5 (image only), 8.4 (text only) and 12.7 (image + text). Pic2Word, SEARLE, Context-I2W, KEDs, ISA, LDRE, SEIZE, Slerp, Denoise-I2W, TransAgg, Visual Delta Generator, CoVR, E5-V, VLM2Vec, UniIR, GENIUS, MM-Embed and CoLLM do not report GeneCIS in their own papers; their GeneCIS rows, where they exist, are re-runs by others.

Three pitfalls matter for a GeneCIS table. Papers use five formats (the four tasks; average only; an undefined single R@1; overlapping pairwise averages, as in FoCo, DiffComp and PACT; focus-only rows, one at R@3). SEARLE at ViT-L/14 circulates as 14.4 (CIReVL's run) and 12.26 (LinCIR's run). Two printed averages are arithmetically wrong (Context-I2W L/14 in OSrCIR T2; PrediCIR G/14 in T9). The GeneCIS README calls the release "v0" and reports a 0.2 point seed standard deviation in average R@1; the repository was archived in August 2024 and no "GeneCIS v1" exists.

**Successor benchmarks.** CLAY-EVAL (synthetic FLUX images with object and human attributes, mAP), CORE (SteerViT; SUN397 scenes with inpainted objects), COCO-Facet (Li, Gao, Du, NeurIPS 2025; 9,112 attribute-focused text→image queries, 1 positive among 100), ZeroSight (video-sourced ZS-CIR that criticises overlap with CLIP pretraining) and MCMR (CVPR 2026; multi-condition product retrieval). None uses example conditions or scores cross-item image↔text pairs.

### 2.3 Example-conditioned and few-shot retrieval

This thread answers the key novelty question.

**Table 3. Retrieval or similarity conditioned on examples.**

| Paper | What the examples define | Modalities | Backbone, frozen? | Benchmark | Code | Closeness to CoSiR v2 |
|---|---|---|---|---|---|---|
| Rocchio, "Relevance feedback in information retrieval", in The SMART Retrieval System, 1971 | the query moves towards the mean relevant and away from the mean non-relevant document | text documents | vector space | n/a | n/a | our naive rule w = ReLU(μ⁺ − μ⁻) is a Rocchio update applied to factor codes |
| Rui, Huang, Ortega, Mehrotra, IEEE TCSVT 1998 | feature weights updated from images the user marks relevant | I→I (content-based retrieval) | hand-crafted features | not checked | n/a | per-feature reweighting from examples, as in our rule |
| MindReader, Ishikawa, Subramanya, Faloutsos, VLDB 1998 | several examples, optionally scored, define the hidden distance function | database records | n/a | synthetic and real data | n/a | infers which attributes matter from examples |
| VISALOGY, Sadeghi, Zitnick, Farhadi, NIPS 2015 | an image pair A:B defines a transformation used to retrieve D for C | I→I | quadruple Siamese CNN, trained | VAQA | not checked | a pair as the condition, image only |
| Few-shot attribute learning, Ren et al., arXiv 2012.05895 | positive and negative supports define an attribute or a conjunction of two | image classification | trained | Celeb-A, Zappos-50K, ImageNet-with-attributes | not checked | the closest episode design; unimodal classification |
| PALAVRA (PerVL), Cohen et al., ECCV 2022 | a few images of a personal concept learn a new word embedding | T→I retrieval with the new word; segmentation | frozen CLIP | two new PerVL benchmarks | yes | cross-modal and example-conditioned, but the examples define an instance, not an aspect shared by different items |
| POLAR, Ryan et al., CVPR 2025 | a few images of a personal concept; low-rank update of the text encoder's last layer | T→I | dual encoder, partly tuned | DeepFashion2, ConCon-Chi | yes (CC BY-NC-SA 4.0) | as PALAVRA |
| Relevance feedback for CLIP, Nara et al., ECCV Workshops 2024 | binary feedback on retrieved images | I→I, T→I | frozen CLIP, training-free | category-based retrieval | not checked | Rocchio-style feedback on CLIP |
| CLIP-Branches, Lülf et al., SIGIR 2024 (arXiv 2406.13322) | positive and negative images marked after a text query train a classifier | T→I | frozen CLIP features | interactive search | not checked | example-refined text-to-image search |
| FSIR, Idan et al., arXiv 2603.25891 | a text query plus exemplar positives and hard negatives | T→I | encoder-agnostic | FSIR-BD: 38,353 images, 303 queries | not stated | the nearest recent cross-modal setting; examples refine a text query, they do not define an aspect for I↔T matching |
| Tip-Adapter, Zhang et al., ECCV 2022 | few-shot labelled images form a key–value cache | image classification | frozen CLIP | 11 datasets | yes | support-set baseline we can run |
| LP++, Huang et al., CVPR 2024; CLAP, Silva-Rodríguez et al., CVPR 2024 | few-shot labelled images fit a linear probe | image classification | frozen CLIP | 11 datasets | yes | linear-probe baselines we can run |
| ProtoNets (NeurIPS 2017), TADAM (NeurIPS 2018), FEAT (CVPR 2020) | the support set defines prototypes or adapts the metric | image classification | trained | miniImageNet and others | yes | support-conditioned metric, unimodal |
| Rankability of visual embeddings, Sonthalia, Uselis, Oh, arXiv 2507.03683 | two extreme examples recover an ordinal axis (age, aesthetics and others) | image embeddings | 7 frozen encoders | 9 datasets | yes | evidence that raw-feature prototypes from very few examples are already strong |
| Affection CLIP listener, Achlioptas et al., CVPR 2023 | none (affective explanation → target image among distractors) | T→I | CLIP | Affection; accuracy reported in a figure (Fig. 7), not copied | yes | affective caption→image discrimination, unconditioned |

**Is there prior work on cross-modal image↔text similarity conditioned by example pairs?** We found none. Every cross-modal method that takes examples uses them either to name an instance (PALAVRA, POLAR) or to refine a text query towards a set of images (CLIP-Branches, FSIR, relevance feedback). None scores a caption against an image of a different item under an aspect defined by support and contrast image–caption pairs, and none evaluates both retrieval directions under the same condition. The only cross-modal conditional retrieval with a learned non-text condition we found is user-conditioned hashtag retrieval (Veit, Nickel, Belongie, van der Maaten, arXiv 1711.09825), where the condition is a per-user embedding. This is a negative search result from an incomplete search (the search budget ran out; arXiv API queries such as "support set" with "image-text retrieval" returned nothing), so the paper should say "to our knowledge". "Few-shot cross-modal retrieval" already names a different setting (novel categories with few training pairs, for example GCRDP, arXiv 2505.13306); the paper should define its own term to avoid the confusion.

### 2.4 Affective and subjective cross-modal retrieval

The affect literature on our data is about classification, captioning and topic discovery; we found no conditional or even plain image↔text retrieval benchmark on ArtEmis or ArtELingo.

**Table 4. Affective, subjective and art-style work.**

| Paper | Task, condition | Backbone, frozen? | Benchmarks, metric | Headline numbers (source) | Code | Closeness |
|---|---|---|---|---|---|---|
| PercepT, Mohamed, Church, Elhoseiny, arXiv 2606.03345 | no condition; discovers perception topics from image–caption pairs, then maps images to topics | frozen CLIP **ViT-L/14**; affect encoder **ModernBERT-base fine-tuned on GoEmotions** (fused with CLIP) | ArtELingo, ArtELingo-28, Affection; clustering (SI, AMI) and multi-label AUC, F1 | SI 0.97 vs BERTopic 0.37, AMI 0.18 vs 0.07 (T1); AUC 0.94 vs 0.77 (T2); no retrieval | "will be made public" | same data and the same kind of affect signal; topic discovery, not conditional scoring |
| ArtEmis, Achlioptas et al., CVPR 2021 | emotion prediction and affective captioning | ResNet, LSTM, BERT | 439,121 explanations, 81K WikiArt paintings, 9 classes | text→emotion BERT 65.7% (§6); best speaker emotional alignment 0.522 (T4) | yes | our data; no retrieval |
| ArtEmis 2.0, Mohamed et al., CVPR 2022 | captioning with contrastive data | SAT speaker | 260,533 new explanations collected on visually similar paintings with opposite valence | combined METEOR 0.144, CIDEr 0.111 vs 0.135, 0.091 (T3) | yes | a ready source of similar-image, opposite-emotion contrast pairs |
| ArtELingo, EMNLP 2022; ArtELingo-28, EMNLP 2024 | multilingual affective captions | captioning models | about 0.8M added annotations; 28 languages | captioning metrics | yes | our data |
| Affection, Achlioptas et al., CVPR 2023 | affective explanations of real photos; CLIP listener | CLIP | 85,007 images, 526,749 explanations | listener results in a figure (§2.3); pragmatic speaker CLIPScore 69.2 (T3) | yes | the photo counterpart of ArtEmis |
| EmoSet, Yang et al., ICCV 2023 | emotion classification with 6 attributes | CNNs | 118,102 human-labelled images | best top-1 78.40% (T3) | yes | alternative affect benchmark, no captions |
| EmotionCLIP, Zhang, Pan, Wang, CVPR 2023 | sentiment-guided contrastive pretraining | trained | BoLD, Emotic and others; linear probe mAP | BoLD 22.51 vs X-CLIP 13.26 (T4) | yes | distant supervision from a text sentiment model (§2.1) |
| GOYA, Wu, Nakashima, Garcia, ICMR 2023 | content and style subspaces, I→I | **frozen CLIP ViT-B/32** plus two projections trained on Stable Diffusion images | WikiArt (10 genres, 27 styles): distance correlation, classification | style 50.90 vs pre-trained CLIP 51.23 (T2) | yes | the closest style-side work: same backbone, one subspace per aspect |
| CSD, Somepalli et al., arXiv 2404.01292 | style descriptor, I→I | CLIP ViT-L fine-tuned | WikiArt with artist as style, mAP@1 | 64.56 vs CLIP ViT-L 59.4, ViT-B/16 52.2 (T1) | yes | strong style baseline |
| DIOR, arXiv 2512.21860 | condition word in an LVLM prompt | frozen LVLM | WikiArt style mAP@1 | CLIP ViT-L 59.3, CSD 58.2, DIOR 45.6 (T3) | yes | a conditional embedding that loses to plain CLIP on art style |
| SemArt, Garcia, Vogiatzis, arXiv 1810.09617 | text↔painting retrieval, no condition | ResNet-50 and bag of words | 1,069 test paintings; R@K, median rank | T→I R@1 0.144, I→T R@1 0.138 (T4) | yes | art I↔T retrieval, unconditioned |
| HPIR, Zhang et al., arXiv 2406.09397 | aesthetic preference in T→I retrieval | CLIP fine-tuned with preference RL | 150 queries, human group choice | CLIP 62.1% → 71.7% (T1) | not found | the only subjective-retrieval benchmark we found; no condition |

**Correction for the project.** PercepT's paper uses ModernBERT-base fine-tuned on GoEmotions and CLIP ViT-L/14, while our handoff calls SamLowe/roberta-base-go_emotions "PercepT's affect teacher" and our factors use ViT-B/32. The teacher shares PercepT's label set but not its model; the paper should describe the teacher by name and not as "PercepT's teacher", and any PercepT comparison should state which encoder our port uses.

### 2.5 Sparse or interpretable shared concept spaces on CLIP

**Table 5. Sparse and concept codes on CLIP-like encoders.**

| Paper | Code type | Shared image and text? | Backbone, frozen? | Used for retrieval? | Condition-weighted retrieval? | Closeness |
|---|---|---|---|---|---|---|
| SpLiCE, Bhalla et al., NeurIPS 2024 | sparse non-negative combination of concept-word embeddings (LAION vocabulary, top 10k recommended) | yes, both modalities decompose over one vocabulary | frozen CLIP ViT-B/32, ViT-B/16, RN50, OpenCLIP ViT-B-32 (repository, Apache-2.0) | spurious-correlation and editing uses | not shown | a training-free shared sparse code for a dimension-matched baseline |
| Discover-then-Name, Rao, Mahajan, Böhle, Schiele, ECCV 2024 | SAE on CLIP image features, concepts named with CLIP text | named through text | frozen CLIP | classification | no | concept bottleneck, image side |
| Matryoshka SAE, Zaigrajew, Baniecki, Biecek, ICML 2025 | hierarchical SAE | image side | frozen CLIP ViT-L/14 | §5.2: boosting one concept in SAE space and mapping back changes ImageNet nearest neighbours | qualitative only | the only image retrieval steered by sparse concepts we found |
| Papadimitriou, Su, Fel, Gil, Kakade, COLM 2025 | SAEs on CLIP, SigLIP, SigLIP2, AIMv2 | finds mostly single-modality concepts that "bridge" across modalities | frozen | analysis | no | warns that shared dictionaries split by modality |
| MGSAE, Kaushik, Barch, Fanelli, arXiv 2601.20028 | group-sparse SAE with cross-modal masking | yes, built against split dictionaries | frozen CLIP and CLAP | cross-modal control | no | a shared-dictionary method to cite and possibly compare |
| LUCID-SAE, Gu et al., arXiv 2602.07311 | shared patch and token dictionary, optimal-transport matching | yes, plus private capacity | frozen | grounding, interpretation | no | token-level analogue |
| SCoCCA, Gordon, Levi, Gilboa, arXiv 2603.13884 | sparse concept decomposition via CCA | yes | frozen CLIP | concept ablation | no | sparse CCA alternative |
| SPARC, Nasiri-Sarvi, Rivaz, Hosseini, TMLR 2026 | concept-aligned SAE, global TopK across models and modalities | yes | frozen | cross-modal concept retrieval | no | shared sparse space |
| TEVI, Mahajan, Rao, Xie, Koller, Schiele, EMNLP 2026 | caption → MLP → sigmoid mask over TopK-SAE latents of the CLIP image embedding | mask conditioned on text | frozen CLIP vision tower; text tower fine-tuned | image↔text retrieval on COCO, Flickr, DOCCI, IIW | yes, but the condition is the caption being matched: DOCCI B/16 I→T R@1 20.38 → 24.52, T→I 7.16 → 8.30 (T2) | yes | the closest mechanism to our weighted sparse factors |
| STAIR (EMNLP 2023), LexLIP (ICCV 2023), VDR (ICLR 2024) | sparse lexical image and text codes | yes, trained jointly | trained | standard I↔T retrieval | not shown | shared sparse I↔T spaces without conditions |
| Kang, Wang, Xiong, arXiv 2411.00786 | SAE with a retrieval contrastive loss on a dense text retriever | text only | frozen retriever | yes | yes: editing latents prioritises documents "from specific perspectives" | the closest precedent for user-steered retrieval through sparse latents, in text IR |

**Has anyone used sparse codes for condition-weighted retrieval?** Partly. TEVI masks sparse latents of frozen CLIP for image↔text retrieval, but its mask comes from the caption being matched, not from a user condition; Matryoshka SAE shows one qualitative concept-boosted search; Kang et al. steer text retrieval. Per-dimension weights that select a notion of similarity are CSN's masks. We infer that a shared sparse image–text factor space whose weights come from example pairs, evaluated on condition-defined cross-modal episodes, is not in the literature, while each of its parts is.

### 2.6 Text-conditioned and instruction-conditioned embeddings

**Table 6. Conditional embeddings with a text or instruction condition.** Universal multimodal embedders are in §2.8.

| Paper | Condition | Modalities | Backbone, frozen? | Benchmarks, metric | Headline (source) | Code | Closeness |
|---|---|---|---|---|---|---|---|
| CLAY, Lim, Lee, Park, Oh, CVPR 2026 | text keyword → LLM descriptions → SVD subspace and log map; training-free | I→I | **frozen CLIP ViT-B/32**, SigLIP-B (L variants in the supplement) | Stanford40 action, location and **mood** (mood labels from IC\|TC), 7 fine-grained sets, CLEVR4, CLAY-EVAL; mAP | Stanford40 action / location / mood: CLIP-B 43.0 / 47.0 / 53.0, GeneCIS Combiner 50.0 / 50.9 / 51.8, CLAY 66.0 / 55.4 / 57.9 (T2(a)) | project page only | the closest frozen ViT-B/32 conditional similarity; text instead of examples, image only; includes a subjective condition |
| CRL, Liu, Sun, Hu, Li, Peng, NeurIPS 2025 | criterion → LLM-written descriptive texts → basis T; R = I·Tᵀ; training-free | I→I | frozen CLIP ViT-B/32 or B/16 | Clevr4-10k, Cards, GeneCIS object half, DeepFashion (MAP) | DeepFashion mean MAP 7.93 training-free vs CLIP 6.08 (T4); GeneCIS in Table 2 | yes | trivially runnable on our features |
| SP-CRL, Wang, Lyu, Li, Jia, arXiv 2602.05464 | CRL bases purified by truncation and null-space projection | I→I | frozen VLM | clustering, few-shot, retrieval | not copied | not found | CRL successor |
| Fioresi, Caba Heilbron, Nathani, Shah, Kafle, ECCV 2026 (arXiv 2607.22919) | attribute text → hypernetwork → affine map of frozen embeddings | I→I | frozen CLIP ViT-L/14; hypernetwork trained with attribute labels (19 attributes) | Clevr-4, Stanford40 (with mood), ShotBench; mAP and clustering | mAP base 33.8 / 43.7 / 25.2 vs 77.0 / 86.3 / 41.0 (T1, as read by the subagent) | project page | supervised light transform on frozen features |
| FocalLens, Hsieh et al., arXiv 2504.08368 (ICLR 2025 workshop per mlanthology) | free text instruction | I→I | CLIP ViT-L/14-336 trained with condition tokens; MLLM variant | CelebA-Attribute (29 attributes, scaled mAP), GeneCIS (R@3), SugarCrepe, MMVP-VLM | CelebA average: CLIP 13.59, MagicLens 13.42, InstructBLIP 16.19, FocalLens-CLIP 21.32, FocalLens-MLLM 22.67 (T1) | not found | reports both GeneCIS and a multi-attribute face benchmark |
| DIOR, arXiv 2512.21860 | condition word in an LVLM prompt | I→I | frozen Llama-3.2-Vision 11B | LanZ-DML, WikiArt and DomainNet style, GeneCIS focus | LanZ-DML average CLIP 31.0 vs 44.5 (T1); WikiArt in Table 4 | yes | training-free LVLM conditioning |
| SteerViT, arXiv 2604.02327 | text injected by gated cross-attention | I→I | frozen ViT plus about 21M parameters trained on referring data | CORE, GeneCIS focus object, MOSAIC, PODS | Table 2 | project page | text-steered encoder |
| TPIPS, Wang, Nitzan, Hertzmann, Zhu, Shechtman, Efros, Zhang, arXiv 2607.18237 | free-form text aspect | I→I perceptual similarity | Qwen3-VL-8B-Embedding fine-tuned | new odd-one-out data: 24,342 triplets, 257,391 triplet–aspect pairs, about 1M votes | main results are in a figure; not copied | yes | concurrent human-judged aspect-conditioned similarity |
| COCO-Facet promptable embeddings, Li, Gao, Du, NeurIPS 2025 | a GPT-4o question about the attribute used as the gallery-side prompt | T→I | MLLM embedders, not fine-tuned | COCO-Facet: 9,112 queries, 8 attribute types; R@1, R@5 | R@5: CLIP ViT-L 47.0, SigLIP2 52.6, VLM2Vec 58.9, 75.5 with prompts (T1) | yes | attribute-conditioned cross-modal retrieval with a text condition |
| FLAIR, Xiao et al., arXiv 2412.03561 | the caption embedding is the pooling query over patch tokens | I↔T | ViT-B/16 trained from scratch | COCO, Flickr, fine-grained retrieval | COCO T→I R@1 53.3 vs SigLIP 47.2 (T1) | yes | text-dependent image embedding, with the caption as the condition |
| Alpha-CLIP, Sun et al., CVPR 2024 | region mask (alpha channel) | I↔T | image encoder fine-tuned | ImageNet-S, referring tasks | L/14 zero-shot 73.48 → 77.41 (T2) | yes | spatial focus only |

### 2.7 Candidate benchmarks

The benchmark subagent verified 37 datasets; the ranked shortlist with all requested fields is in §4. Two facts shape the choice. Many multi-attribute face and fashion sets generate their captions from the labels (MM-CelebA-HQ, CelebA-Dialog, and, we infer from released examples, DeepFashion-MultiModal), so the text side leaks the label and cannot test cross-modal conditional matching. The datasets with human captions and independent labels are few: CUB-200-2011 with Reed et al. captions, SemArt, Affection, CelebAText-HQ and our ArtEmis/ArtELingo. Prior conditional-similarity papers used mostly caption-free sets (UT-Zappos50k, Polyvore, DeepFashion, FashionAI, DARN, CelebA attributes, Stanford40, CLEVR4).

### 2.8 Backbones and universal or instruction-following multimodal embedders

Universal multimodal embedders turn an MLLM into a retriever; most take a task instruction at encode time, so a condition can be written into the instruction ("Represent this painting with respect to the emotion it evokes"). That makes them both candidate frozen backbones and text-named-condition baselines. Their standard benchmark is MMEB (36 datasets: classification, VQA, retrieval, grounding; precision@1) or MMEB-V2 (78 datasets, adding video and visual documents); neither is a conditional-similarity benchmark, so the rows closest to our problem are GeneCIS (Table 2), COCO-Facet (Table 6) and CelebA-Attribute (FocalLens). Memory figures below are our estimate of bf16 weight memory (2 bytes per parameter); inference needs more.

**Table 7. Frozen backbone candidates.**

| Model (paper) | Type, size | Embedding dim | Weights memory (bf16, our estimate) | Licence, weights | Instruction or text condition at encode time | MMEB or related (source) | GeneCIS (source) |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (current) | dual encoder, about 151M | 512 | under 1 GB | MIT, open | no | CLIP 37.8, variant not stated (VLM2Vec T2) | B/32 image + text 12.7 average (Paracosm v1 T3) |
| OpenCLIP ViT-L/14, ViT-H/14 (Cherti et al., CVPR 2023) | dual encoder, LAION-2B | 768, 1024 | about 1 to 2 GB | open (licence not re-checked) | no | OpenCLIP 39.7 (VLM2Vec T2) | LinCIR uses H as a backbone (Table 2) |
| EVA-CLIP, EVA-CLIP-18B (Sun et al. 2023, 2024) | dual encoder up to 18B | varies | up to about 36 GB | open (licence not re-checked) | no | EVA-CLIP 8B 43.7 (UniME T1) | EVA-CLIP-8B 13.1, 18B 13.6 single R@1 (LamRA T4) |
| SigLIP 2 (Tschannen et al., arXiv 2502.14786) | sigmoid dual encoder: B, L, So400m (about 1B total per the card), g | not stated on the card | about 2 GB for So400m | Apache-2.0, open | no | SigLIP (v1) 34.8 (VLM2Vec T2); SigLIP2 R@5 52.6 on COCO-Facet (Li et al. T1) | none found |
| Perception Encoder, PE-Core (Bolya et al., arXiv 2504.13181) | CLIP-style, B/16, L/14, G/14 | not checked | up to a few GB | licence not verified | no | ImageNet zero-shot 83.5 for L14-336 (repository) | none found |
| jina-clip-v2 (Koukounas et al., arXiv 2412.08802) | dual encoder: 561M text (XLM-RoBERTa) plus 304M EVA02-L14 vision | 1024, Matryoshka down to 64 | about 2 GB | CC BY-NC 4.0, open | task prefix only (for example retrieval.query), no free instruction | not found | none found |
| BLIP-2 (Li et al., ICML 2023) | Q-Former on a frozen ViT-g; LLM for generation | 256 (projected ITC features) | about 2 to 8 GB for the feature extractor (our estimate) | BSD-3-Clause (LAVIS, archived September 2026), open | no | 25.2 (VLM2Vec T2) | none found |
| UniIR CLIP_SF, BLIP_FF (Wei et al., ECCV 2024) | fine-tuned CLIP-L or BLIP fusion | 768 for CLIP_SF (we infer from CLIP-L) | 5.1 GB and 7.5 GB checkpoints | MIT, open | instruction prefix | 44.7 (VLM2Vec T2); M-BEIR 48.9 CLIP_SF (UniIR T2) | 16.8 single R@1 (LamRA T4) |
| E5-V (Jiang et al., arXiv 2407.12580) | LLaVA-NeXT-8B trained on text pairs only | 4096 | about 16 GB | not stated on the card | fixed "in one word" prompt; a condition can be written into the prompt string (we infer; not evaluated by its authors) | 13.3 (VLM2Vec T2) but 37.5 (UniME T1); CIRR R@1 33.90 (E5-V T2) | 18.5 single R@1 (LamRA T4) |
| VLM2Vec (Jiang et al., ICLR 2025) | Phi-3.5-V 4.2B or LLaVA-1.6 with LoRA | backbone hidden size | about 8 to 16 GB | open (licence not re-checked) | yes: "Instruct: {task} Query: {input}" | 62.9 best variant (T2) | none in own paper |
| VLM2Vec-V2 (Meng et al., arXiv 2507.04590) | Qwen2-VL-2B with LoRA | backbone hidden size | about 4 GB | model card returned 401 | yes | MMEB-V2 58.0 overall, 64.9 image (T2); GME-7B 57.8, LamRA 40.4 in the same table | none |
| GME (Zhang et al., CVPR 2025) | Qwen2-VL 2B (2.21B) or 7B | 1536 or 3584 | about 4.5 or 16 GB | Apache-2.0, open | yes, through a `prompt` or `instruction` argument | UMRB 64.45 (2B), 67.44 (7B) (T3); CIRR 51.79 for 7B (T7) | none |
| MM-Embed (Lin et al., ICLR 2025) | LLaVA-1.6-Mistral-7B plus NV-Embed-v1, 8B | not stated on the card | about 16 GB | CC BY-NC 4.0, open | yes, required for queries | M-BEIR 52.7 vs CLIP-SF 48.3 (T1) | none in own paper |
| LamRA (Liu et al., CVPR 2025) | Qwen2-VL-7B (Qwen2.5-VL branch) | backbone hidden size | about 16 GB | MIT, open | instruction-tuned; condition through the query text | MMEB-V2 40.4 (VLM2Vec-V2 T2) | 18.9, 24.8 with reranking (T4) |
| mmE5 (Chen et al., arXiv 2502.08468) | Llama-3.2-11B-Vision | backbone hidden size | about 22 GB | not stated | yes | 69.8 supervised, 58.6 zero-shot (T2) | none |
| UniME (Gu et al., ACM MM 2025) | Phi-3.5-V 4.2B, LLaVA-1.6 7B, LLaVA-OneVision 7B | backbone hidden size | about 8 to 16 GB | open (licence not re-checked) | yes, VLM2Vec task prompts | 66.6 (LLaVA-1.6), 70.7 (OneVision) (T1) | none |
| LLaVE (Lan et al., EMNLP 2025 Findings) | 0.5B, 2B, 7B | backbone hidden size | about 1 to 16 GB | not checked | yes | 59.1, 65.2, 70.3 (T2) | none |
| B3 (Thirukovalluru et al., arXiv 2505.11293) | Qwen2-VL 2B or 7B, InternVL3 | backbone hidden size | about 4 to 16 GB | not checked | yes | 68.1 (2B), 72.0 (7B) (T1) | none |
| MetaEmbed (Xiao et al., ICLR 2026) | multi-vector late interaction on Qwen2.5-VL and others | multi-vector | up to 32B models | not checked | not central | 76.6 (7B), 78.7 (32B) (T1) | none |
| Qwen3-VL-Embedding (Li et al., arXiv 2601.04720) | 2B or 8B | 64 to 2048 (2B, Matryoshka), 4096 (8B) | about 4 or 16 GB | Apache-2.0, open | yes, a custom instruction replaces the default "Represent the user's input." | MMEB-V2 73.2 (2B), 77.8 (8B) (model card table) | none |
| jina-embeddings-v4 (Günther et al., arXiv 2506.18902) | Qwen2.5-VL-3B, about 4B | 2048 dense (Matryoshka to 128) or 128 multi-vector | about 8 GB | Qwen Research License | three task adapters (retrieval, text matching, code); no free instruction | not copied | none |

Three observations follow. First, the best open, permissively licensed instruction-following embedders that fit one GPU are Qwen3-VL-Embedding-2B and GME-Qwen2-VL-2B (Apache-2.0, about 2B parameters); MM-Embed and jina-clip-v2 are non-commercial, and jina-embeddings-v4 uses the Qwen Research License. Second, MMEB numbers for the same model disagree across papers (E5-V 13.3 in VLM2Vec T2 vs 37.5 in UniME T1), so we should cite each number with its source and re-run what we compare. Third, no universal embedder reports GeneCIS in its own paper; LamRA's single-number GeneCIS evaluation (T4) is the only source, and it puts E5-V (18.5) and LamRA (18.9) above the frozen ViT-B/32 ZS-CIR methods but below training-free MLLM pipelines such as SQUARE (19.7 with reranking).

**When the image and text encoders do not share a space.** With a pair of unimodal encoders (for example an e5 text encoder and a DINOv2 image encoder), cos(f_I(I), f_T(T)) is meaningless, so our β·cos term is unavailable and the factor term must carry all cross-modal matching. Prior work offers three ways to restore a shared term. Trained alignment on locked towers: LiT (Zhai et al., CVPR 2022) locks a pretrained image tower and trains the text tower; Three Towers (Kossen et al., NeurIPS 2023) adds a frozen pretrained tower as a teacher; Maniparambil et al. (CVPR 2025, arXiv 2409.19425) train only MLP projectors on frozen DINOv2 and All-RoBERTa-Large and reach 76% ImageNet accuracy with 20 times less data and 65 times less compute than training from scratch; SAIL (Zhang, Yang, Agrawal, CVPR 2025) trains an alignment layer on frozen unimodal encoders with 6% of CLIP's paired data and reports 73.4% ImageNet zero-shot against CLIP's 72.7%. Training-free alignment: ASIF (Norelli et al., arXiv 2210.01738) represents each input by its similarities to a set of anchor image–text pairs, so every dimension is "the similarity of the input to a unique image-text pair", building on relative representations (Moschella et al., ICLR 2023). Evidence that it can work: Maniparambil et al. (CVPR 2024, arXiv 2401.05224) find that unimodal vision and language encoders have similar similarity structure, Merullo et al. (ICLR 2023) map image features into a language model's input space with one linear layer, the Platonic representation hypothesis (Huh et al., arXiv 2405.07987) argues for convergence, and Schnaus, Araslanov, Cremers (CVPR 2025) match vision and language without parallel data. For CoSiR this means the factor encoders already map each modality into one shared 32-d space and survive the switch, while the β term needs a LiT or SAIL style projector or an ASIF anchor representation. ASIF is worth noting: its anchors are image–text pairs, so our support and contrast pairs could serve as anchors, which makes it a natural example-conditioned baseline for the non-shared case (we infer; untested). Any non-shared-backbone result must be read against a shared CLIP control, since alignment quality and conditioning would otherwise be confounded.

## 3. Baselines we can run

All baselines run on the same frozen features as CoSiR (CLIP ViT-B/32 caches exist for ArtELingo; if we change backbone, every baseline moves with it) and on the same episodes, within the held-row budget of the handoff. We rank by how directly each answers a question a CVPR reviewer will ask, then by cost. "Needs" lists what goes beyond our own episode inputs (anchor, 4 supports, 4 contrasts, 13 candidates).

**Table 8. Ranked baselines.**

| Rank | Baseline | What it computes | Needs | Reviewer question it answers |
|---|---|---|---|---|
| 1 | **Support prototype and Rocchio query on raw features** | anchor-free: score each candidate by cos(f(c), μ⁺ − μ⁻), μ the mean feature of the support or contrast items in the candidate's modality (or the pair mean); anchor-plus: q = f(anchor) + α(μ⁺ − μ⁻), scored cross-modally | nothing new; α tuned on selection rows | "Is this just 4-shot classification of the candidates?" In our label episodes the positive is the only candidate with the support label, so the anchor-free prototype can in principle solve the task; it also tests learned factors against textbook relevance feedback |
| 2 | **Tip-Adapter style cache (training-free)** | candidate affinity Σ exp(−γ(1 − cos)) to the supports minus the same to the contrasts, plus β·cos(anchor, candidate) | nothing new; γ, β on selection rows | "Did you compare with the standard training-free few-shot CLIP adapter?" |
| 3 | **Linear probe on the eight pairs** | logistic regression on 4 support vs 4 contrast pairs (image and text features), candidate score = probe logit + β·cos(anchor, candidate) | scikit-learn | "Does a fitted classifier on raw features beat a parameter-free rule on learned factors?" |
| 4 | **Text-named condition on the same frozen encoder** | (a) label prompt: cos(anchor, c) + λ·cos(c, text("a painting that evokes sadness")); (b) CRL projection: an LLM lists the values of the criterion ("emotion", "art style"), anchor and candidates are projected onto that text basis and compared; (c) GeneCIS-style image + text sum | label names at test time; one LLM call per criterion for (b); CRL code is public | "Why examples instead of naming the condition?" Examples must win where the aspect is hard to name, or the paper needs another argument |
| 5 | **Instruction-following embedder with the condition in the instruction** | Qwen3-VL-Embedding-2B or GME-Qwen2-VL-2B (Apache-2.0), or VLM2Vec-V2, encoding anchor and candidates under an instruction such as "Represent this painting by the emotion it evokes"; optionally E5-V with an edited prompt or a DIOR-style LVLM prompt | about 4 to 5 GB of weights, a few GPU hours to encode ArtELingo (our estimate) | "Would a 2026 instruction-following embedder make the method unnecessary?" |
| 6 | **Naive rule on unsupervised codes** | the same w = ReLU(μ⁺ − μ⁻) on PCA-32, ICA-32 or NMF-32 of the training features, on SpLiCE codes (supports ViT-B/32, Apache-2.0), and on the positive part of centred raw features | SpLiCE vocabulary | "Is the gain from the learned factors or from the rule plus any sparse basis?" |
| 7 | **Teacher-only** | naive rule on the 28-d GoEmotions probabilities of the captions | the GoEmotions model already used | "Is SE more than its distant teacher?" (planned in the handoff) |
| 8 | **PercepT topic codes** | naive rule on PercepT Stage 1 topic memberships | the project's PercepT port as a clean v2 module (state its encoder, §2.4) | "How does the closest image–caption topic method do on the same episodes?" |
| 9 | **Published text-conditioned models** | SEARLE ViT-B/32 (torch.hub weights, CC BY-NC 4.0) with the label name as the modification text; CLAY reimplemented from its formula; GeneCIS Combiner (RN50x4 or ViT-B/16 CC3M weights, fine-tuned backbone) | released weights; label names | "How do established CIR and conditional-similarity models do when told the condition?" |
| 10 | **Supervised upper reference** | a CSN-style mask or linear head trained with ArtELingo labels on training rows | labels on training rows | "How far is training without evaluation labels from training with them?" (the headroom probe's label-aligned code at 49.8% R@1 is a first answer) |
| 11 | **Style subspace** (optional) | GOYA's style projection on frozen ViT-B/32 for style episodes | GOYA code | "Does a fixed style subspace match the learned factors on style?" |

On GeneCIS itself, a frozen ViT-B/32 model should be compared with image only, text only and image + text (re-run on our cache), with SEARLE 14.4, CIReVL 15.9 and OSrCIR 17.4 average R@1 (published, with OSrCIR's reproduction at 14.0 noted), and with the GeneCIS Combiner at 15.1 with a frozen RN50x4 (T5) as the closest frozen trained reference. Our model enters through an image→image mode and either a text adapter or few-shot supports drawn from other templates, reported separately, as the feasibility check recommends.

## 4. Benchmarks

Ranked for example-conditioned image↔text episodes with at least two independent aspects, small compute and the deadline. Facts come from the benchmark subagent's fetched pages; "(we infer)" marks inference.

**Table 9. Ranked candidate benchmarks.**

| Rank | Benchmark | Size | Aspects | Caption source | Licence | Download | Prior conditional-similarity use | Main risk |
|---|---|---|---|---|---|---|---|---|
| 1 | **CUB-200-2011 + Reed et al. captions** (CVPR 2016) | 11,788 images, 200 species, 312 binary attributes; 10 captions per image (about 118k, we infer) | species; attribute groups (colour, bill shape, size and others) | human (AMT); annotators were told not to name the species | Caltech page: non-commercial research and education; the CaltechDATA record says "cc-by" (conflict unresolved); caption licence not stated | live (1.2 GB); caption archive on Google Drive | attributes used in few-shot attribute work (PAN, ICCV 2021); CUB captions used for cross-modal retrieval by PCME (CVPR 2021) | noisy per-image attributes; colour correlates with species; captions name colours, so colour conditions are easy on the text side |
| 2 | **GeneCIS** (object half now; attribute half after a Visual Genome 1.2 download) | 4 tasks, 1,960 to 2,112 templates each, galleries of 10 or 15 | text keyword (object or attribute) | none native; COCO captions exist for the object half | CC BY-NC 4.0 | live; object half preprocessed in our repository | the standard benchmark (Table 2) | image→image with text conditions; measures transfer, not the primary claim |
| 3 | **SemArt** (arXiv 1810.09617) | 21,384 paintings | type (10), school (26), timeframe (22), author, technique | human catalogue comments (Web Gallery of Art) | CC BY-NC 4.0 | live (3 GB, DOI) | Text2Art retrieval (R@K) | comments often name artist and date; school and timeframe correlate |
| 4 | **ArtEmis/ArtELingo plus WikiArt genre** | current data | emotion, style, and genre as a third aspect | human | ArtEmis terms of use | on disk; genre through the ArtGAN WikiArt class lists (we infer) | our current benchmark | same domain; enables same-anchor swaps across three aspects, and ArtEmis 2.0 adds similar-image, opposite-emotion pairs for hard contrasts |
| 5 | **Affection** (CVPR 2023) | 85,007 images, 526,749 explanations | emotion (ArtEmis taxonomy); objects from its COCO, VG and Flickr30k sources (we infer) | human explanations | Affection terms of use | request form | none found | access latency; 71.3% positive vs 21.1% negative |
| 6 | CelebAText-HQ + CelebA attributes | 15,010 images, 10 human captions each | 40 binary attributes, identity | human | not stated | Drive folder (not tested) | CelebA attributes used by FocalLens | faces and ethics; binary aspects |
| 7 | DeepFashion-MultiModal (SIGGRAPH 2022) | 44,096 images | 12 shape attributes, fabric (8), colour or pattern (8) | one description per image, apparently templated from the labels (we infer) | non-commercial research | live | DeepFashion used by ASEN and CRL | text leaks the label |
| runner-up | EmoSet-118K (ICCV 2023) | 118,102 human-labelled | emotion (8), scene, object, facial expression, action, brightness, colourfulness | none | non-commercial | live | none | captions would have to be generated; attributes machine-predicted |
| not recommended | MM-CelebA-HQ, CelebA-Dialog | 30k, 202k | face attributes | generated from the labels | non-commercial | MM-CelebA-HQ links removed | none | total label leakage |
| not recommended | UT-Zappos50k | 50,025 | category, gender, heel height, closure | none | academic | live | CSN, SCE-Net, ASEN | no captions |

## 5. Novelty risks and how a reviewer would phrase them

1. **"The condition rule is relevance feedback."** "ReLU(mean support − mean contrast) is the Rocchio update (1971), and per-feature reweighting from examples goes back to MindReader and Rui et al. (1998)." Answer: present the rule as a deliberately simple interface, claim the learned shared factors, and report Rocchio on raw features (baseline 1).
2. **"This is few-shot classification, not conditional similarity."** "The positive is the only candidate with the support label, so a 4-shot prototype that ignores the anchor solves the task; Ren et al. already studied episodes where positive and negative supports define an attribute." Answer: report the anchor-free prototype, and add episodes where the anchor matters, such as GeneCIS-style condition-only distractors (same label as the supports, different second aspect) and same-anchor swaps across emotion, style and genre.
3. **"Weighted per-dimension similarity is a Conditional Similarity Network."** "Σ w_l a_I,l a_T,l is CSN's masked distance, and SCE-Net infers the mask from the compared items without condition labels." Answer: show what is new with ablations: factors shared by image and text (against a split dictionary, as MGSAE warns), masks set at test time from example pairs, both retrieval directions.
4. **"Frozen-encoder conditional similarity already exists without training."** "CLAY (CVPR 2026) and CRL (NeurIPS 2025) reshape frozen CLIP similarity from a text condition; InDiReCT did so in 2023; Qwen3-VL-Embedding takes the condition as an instruction." Answer: baselines 4 and 5, and evidence that examples beat names where the aspect is subjective (CLAY's own mood condition gains only 4.9 mAP over CLIP-B, T2(a)).
5. **"The no-label claim is overstated."** "The factors are trained on outputs of a supervised GoEmotions classifier that names 6 of the 8 emotions; EmotionCLIP used this kind of distant supervision in 2023." Answer: claim "no labels from the evaluation taxonomy", name the teacher model exactly (it is not the model PercepT used, §2.4), and report the teacher-only baseline and the caption-word shortcut analysis from the held report.
6. **"Small numbers on an in-house benchmark."** "17 to 25% R@1 among 13 candidates on home-made episodes." Answer: add GeneCIS with its published frozen B/32 rows and CUB with Reed captions.
7. **"Missing state-of-the-art comparisons."** "No CIR, conditional-similarity or universal-embedder baseline." Answer: SEARLE, CIReVL and OSrCIR on GeneCIS; CRL or CLAY and an instruction-following embedder on our episodes.
8. **"ViT-B/32 is outdated."** Now that any frozen encoder is allowed, the answer is to report the main result on one strong open backbone (for example SigLIP 2 or Qwen3-VL-Embedding) beside B/32, keeping the backbone frozen.
9. **"Concurrent work crowds the problem."** TPIPS (July 2026), SteerViT, Fioresi et al. (ECCV 2026), COCO-Facet (NeurIPS 2025), TEVI (EMNLP 2026) and CLAY all make similarity or retrieval conditional on text. Answer: cite them and position the paper on the example interface and the cross-modal score, where none of them works.

## 6. References

Links were fetched during this review unless marked UNVERIFIED. Venues come from the arXiv comment, the venue page or the repository; where only a secondary source gave the venue we say so.

**Conditional similarity (§2.1)**
- Veit, Belongie, Karaletsos. Conditional Similarity Networks. CVPR 2017. https://arxiv.org/abs/1603.07810
- Tan, Vasileva, Saenko, Plummer. Learning Similarity Conditions Without Explicit Supervision. ICCV 2019. https://arxiv.org/abs/1908.08589
- Ye, Shi, Zhan. Identifying Ambiguous Similarity Conditions via Semantic Matching. CVPR 2022. https://arxiv.org/abs/2204.04053
- Shi, Li, Gan, Zhan, Ye. Generalized Conditional Similarity Learning via Semantic Matching. IEEE TPAMI 2025. https://doi.org/10.1109/TPAMI.2025.3535730 (abstract page not fetched; metadata from search)
- Ren, Triantafillou, Wang, Lucas, Snell, Pitkow, Tolias, Zemel. Probing Few-Shot Generalization with Attributes. arXiv 2012.05895. https://arxiv.org/abs/2012.05895
- Ma, Dong, Long, Zhang, He, Xue, Ji. Fine-Grained Fashion Similarity Learning by Attribute-Specific Embedding Network. AAAI 2020. https://arxiv.org/abs/2002.02814
- Dong et al. Fine-Grained Fashion Similarity Prediction by Attribute-Specific Embedding Learning (ASEN++). https://arxiv.org/abs/2104.02429 (venue UNVERIFIED)
- Plummer, Kordas, Kiapour, Zheng, Piramuthu, Lazebnik. Conditional Image-Text Embedding Networks. ECCV 2018. https://arxiv.org/abs/1711.08389
- Vaze, Carion, Misra. GeneCIS: A Benchmark for General Conditional Image Similarity. CVPR 2023. https://arxiv.org/abs/2306.07969 ; weights https://github.com/facebookresearch/genecis/blob/main/DOWNLOAD.md
- Kobs, Steininger, Hotho. InDiReCT: Language-Guided Zero-Shot Deep Metric Learning for Images. WACV 2023. https://arxiv.org/abs/2211.12760
- Kwon et al. Image Clustering Conditioned on Text Criteria. ICLR 2024. https://arxiv.org/abs/2310.18297
- Yao, Qian, Hu. Multi-Modal Proxy Learning Towards Personalized Visual Multiple Clustering. CVPR 2024. https://arxiv.org/abs/2404.15655
- Zhang, Pan, Wang. Learning Emotion Representations from Verbal and Nonverbal Communication (EmotionCLIP). CVPR 2023. https://arxiv.org/abs/2305.13500

**Composed image retrieval (§2.2)**
- Saito et al. Pic2Word. CVPR 2023. https://arxiv.org/abs/2302.03084
- Baldrati et al. SEARLE (and CIRCO). ICCV 2023. https://arxiv.org/abs/2303.15247 ; code https://github.com/miccunifi/SEARLE
- Agnolucci et al. iSEARLE. https://arxiv.org/abs/2405.02951
- Gu et al. Language-only Efficient Training of Zero-shot Composed Image Retrieval (LinCIR). CVPR 2024. https://arxiv.org/abs/2312.01998
- Gu et al. CompoDiff. TMLR 2024. https://arxiv.org/abs/2303.11916
- Zhang et al. MagicLens. ICML 2024. https://arxiv.org/abs/2403.19651
- Karthik et al. Vision-by-Language for Training-Free Compositional Image Retrieval (CIReVL). ICLR 2024. https://arxiv.org/abs/2310.09291
- Tang et al. Reason-before-Retrieve (OSrCIR). CVPR 2025. https://arxiv.org/abs/2412.11077
- Tang et al. PrediCIR. CVPR 2025. https://arxiv.org/abs/2503.17109
- Byun et al. RTD. ICCV 2025. https://arxiv.org/abs/2406.09188
- Zhou et al. MegaPairs. ACL 2025. https://arxiv.org/abs/2412.14475
- Liu et al. LamRA. CVPR 2025. https://arxiv.org/abs/2412.01720
- Li et al. STiTch. https://arxiv.org/abs/2605.21261
- Wu, Lin, Yang. SQUARE. https://arxiv.org/abs/2509.26330
- Wang, Zhao, Kong. Generating a Paracosm for Training-Free ZS-CIR. ECCV 2026. https://arxiv.org/abs/2602.00813
- Liu et al. DeCIR. https://arxiv.org/abs/2605.08389
- Kwon. PACT. https://arxiv.org/abs/2609.31202
- Zhang et al. FoCo. ECCV 2026. https://arxiv.org/abs/2607.00374
- Yang, Du, Qian, Xu. ZeroSight. https://arxiv.org/abs/2606.07032
- Li et al. COMBINER. IEEE TIP 2026. https://arxiv.org/abs/2606.04604
- Lu et al. MCMR. CVPR 2026. https://arxiv.org/abs/2603.01082
- Wu et al. Fashion IQ. CVPR 2021. https://arxiv.org/abs/1905.12794
- Liu et al. CIRR. ICCV 2021. https://arxiv.org/abs/2108.04024

**Example-conditioned retrieval (§2.3)**
- Rocchio. Relevance feedback in information retrieval. In Salton (ed.), The SMART Retrieval System, Prentice-Hall, 1971, pp. 313 to 323 (book chapter; metadata from search)
- Rui, Huang, Ortega, Mehrotra. Relevance feedback: a power tool for interactive content-based image retrieval. IEEE TCSVT 8(5), 1998. https://doi.org/10.1109/76.718510
- Ishikawa, Subramanya, Faloutsos. MindReader: Querying Databases Through Multiple Examples. VLDB 1998. https://www.semanticscholar.org/paper/04938be9fd727ea6363cc950efd263ff82d02b77
- Sadeghi, Zitnick, Farhadi. VISALOGY: Answering Visual Analogy Questions. NIPS 2015. https://arxiv.org/abs/1510.08973
- Cohen, Gal, Meirom, Chechik, Atzmon. "This is my unicorn, Fluffy": Personalizing frozen vision-language representations (PALAVRA). ECCV 2022. https://arxiv.org/abs/2204.01694
- Ryan, Sivic, Caba Heilbron, Hoffman, Rehg, Russell. Improving Personalized Search with Regularized Low-Rank Parameter Updates. CVPR 2025. https://arxiv.org/abs/2506.10182
- Nara et al. Revisiting Relevance Feedback for CLIP-based Interactive Image Retrieval. ECCV Workshops 2024. https://arxiv.org/abs/2404.16398
- Lülf, Martins, Salles, Zhou, Gieseke. CLIP-Branches: Interactive Fine-Tuning for Text-Image Retrieval. SIGIR 2024. https://arxiv.org/abs/2406.13322
- Idan et al. Few Shots Text to Image Retrieval. https://arxiv.org/abs/2603.25891
- Zhang et al. Tip-Adapter. ECCV 2022. https://arxiv.org/abs/2207.09519
- Huang et al. LP++. CVPR 2024. https://arxiv.org/abs/2404.02285
- Silva-Rodríguez, Hajimiri, Ben Ayed, Dolz. A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models (CLAP). CVPR 2024. https://arxiv.org/abs/2312.12730
- Snell, Swersky, Zemel. Prototypical Networks for Few-shot Learning. NeurIPS 2017. https://arxiv.org/abs/1703.05175
- Oreshkin, Rodriguez, Lacoste. TADAM. NeurIPS 2018. https://arxiv.org/abs/1805.10123
- Ye, Hu, Zhan, Sha. FEAT. CVPR 2020. https://arxiv.org/abs/1812.03664
- Sonthalia, Uselis, Oh. On the rankability of visual embeddings. https://arxiv.org/abs/2507.03683
- Veit, Nickel, Belongie, van der Maaten. Separating Self-Expression and Visual Content in Hashtag Supervision. https://arxiv.org/abs/1711.09825
- Sun et al. GCRDP, few-shot cross-modal retrieval. https://arxiv.org/abs/2505.13306
- Nguyen et al. Visual Instruction Inversion. NeurIPS 2023. https://arxiv.org/abs/2307.14331

**Affect and art (§2.4)**
- Mohamed, Church, Elhoseiny. Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from Vision-Language Data (PercepT). https://arxiv.org/abs/2606.03345
- Achlioptas, Ovsjanikov, Haydarov, Elhoseiny, Guibas. ArtEmis. CVPR 2021. https://arxiv.org/abs/2101.07396
- Mohamed, Khan, Haydarov, Elhoseiny. It is Okay to Not Be Okay (ArtEmis 2.0). CVPR 2022. https://arxiv.org/abs/2204.07660
- Mohamed et al. ArtELingo. EMNLP 2022. https://arxiv.org/abs/2211.10780 ; ArtELingo-28. EMNLP 2024. https://arxiv.org/abs/2411.03769
- Achlioptas, Ovsjanikov, Guibas, Tulyakov. Affection. CVPR 2023. https://arxiv.org/abs/2210.01946
- Yang et al. EmoSet. ICCV 2023. https://arxiv.org/abs/2307.07961
- Wu, Nakashima, Garcia. Not Only Generative Art (GOYA). ICMR 2023 (DOI 10.1145/3591106.3592262). https://arxiv.org/abs/2304.10278
- Somepalli et al. Measuring Style Similarity in Diffusion Models (CSD). https://arxiv.org/abs/2404.01292
- Garcia, Vogiatzis. How to Read Paintings (SemArt). https://arxiv.org/abs/1810.09617 (ECCV Workshops 2018 per secondary sources)
- Zhang et al. Aligning Vision Models with Human Aesthetics in Retrieval (HPIR). https://arxiv.org/abs/2406.09397

**Sparse and concept codes (§2.5)**
- Bhalla et al. Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE). NeurIPS 2024. https://arxiv.org/abs/2402.10376 ; code https://github.com/AI4LIFE-GROUP/SpLiCE
- Rao, Mahajan, Böhle, Schiele. Discover-then-Name. ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09973.pdf
- Zaigrajew, Baniecki, Biecek. Interpreting CLIP with Hierarchical Sparse Autoencoders. ICML 2025. https://arxiv.org/abs/2502.20578
- Papadimitriou, Su, Fel, Gil, Kakade. Interpreting the linear structure of vision-language model embedding spaces. COLM 2025. https://arxiv.org/abs/2504.11695
- Kaushik, Barch, Fanelli. Decomposing multimodal embedding spaces with group-sparse autoencoders. https://arxiv.org/abs/2601.20028
- Gu et al. LUCID-SAE. https://arxiv.org/abs/2602.07311
- Gordon, Levi, Gilboa. SCoCCA. https://arxiv.org/abs/2603.13884
- Nasiri-Sarvi, Rivaz, Hosseini. SPARC. TMLR 2026. https://arxiv.org/abs/2507.06265
- Kubaty et al. Conceptualizing Embeddings (CEDAR). https://arxiv.org/abs/2605.22679
- Mahajan, Rao, Xie, Koller, Schiele. TEVI. EMNLP 2026. https://arxiv.org/abs/2606.07451
- Chen et al. STAIR. EMNLP 2023. https://arxiv.org/abs/2301.13081
- Luo et al. LexLIP. ICCV 2023. https://arxiv.org/abs/2302.02908
- Zhou et al. Retrieval-based Disentangled Representation Learning with Natural Language Supervision (VDR). ICLR 2024. https://arxiv.org/abs/2212.07699
- Kang, Wang, Xiong. Interpret and Control Dense Retrieval with Sparse Latent Features. https://arxiv.org/abs/2411.00786

**Text-conditioned embeddings (§2.6)**
- Lim, Lee, Park, Oh. CLAY. CVPR 2026. https://arxiv.org/abs/2604.11539
- Liu, Sun, Hu, Li, Peng. Conditional Representation Learning for Customized Tasks (CRL). NeurIPS 2025. https://arxiv.org/abs/2510.04564 ; code https://github.com/XLearning-SCU/2025-NeurIPS-CRL
- Wang, Lyu, Li, Jia. Semantic Purification for Conditional Representation Learning (SP-CRL). https://arxiv.org/abs/2602.05464
- Fioresi, Caba Heilbron, Nathani, Shah, Kafle. Controlling Embedding Spaces with Text-Conditioned Transformations. ECCV 2026. https://arxiv.org/abs/2607.22919
- Hsieh et al. FocalLens. https://arxiv.org/abs/2504.08368
- Kawarada, Yamada, Tejero-de-Pablos, Inoue. Training-free Conditional Image Embedding Framework Leveraging Large Vision Language Models (DIOR). https://arxiv.org/abs/2512.21860
- Ruthardt, Gaur, Ramanan, Tapaswi, Asano. Steerable Visual Representations (SteerViT). https://arxiv.org/abs/2604.02327 (ECCV 2026 per secondary sources)
- Wang, Nitzan, Hertzmann, Zhu, Shechtman, Efros, Zhang. The Many Senses of Visual Similarity (TPIPS). https://arxiv.org/abs/2607.18237
- Li, Gao, Du. Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval (COCO-Facet). NeurIPS 2025. https://arxiv.org/abs/2505.15877
- Xiao et al. FLAIR. https://arxiv.org/abs/2412.03561
- Sun et al. Alpha-CLIP. CVPR 2024. https://arxiv.org/abs/2312.03818

**Benchmarks (§2.7, §4)**
- Reed, Akata, Schiele, Lee. Learning Deep Representations of Fine-grained Visual Descriptions. CVPR 2016. https://arxiv.org/abs/1605.05395 ; CUB https://www.vision.caltech.edu/datasets/cub_200_2011/
- Chun et al. Probabilistic Embeddings for Cross-Modal Retrieval (PCME). CVPR 2021. https://arxiv.org/abs/2101.05068
- Jiang et al. Text2Human (DeepFashion-MultiModal). SIGGRAPH 2022. https://arxiv.org/abs/2205.15996
- Xia et al. TediGAN (MM-CelebA-HQ). CVPR 2021. https://arxiv.org/abs/2012.03308
- Yu, Grauman. UT-Zappos50K. https://vision.cs.utexas.edu/projects/finegrained/utzap50k/
- SemArt data: https://researchdata.aston.ac.uk/id/eprint/380

**Backbones and universal embedders (§2.8)**
- Radford et al. CLIP. https://arxiv.org/abs/2103.00020 (not re-fetched)
- Cherti et al. Reproducible scaling laws for contrastive language-image learning (OpenCLIP). CVPR 2023. https://arxiv.org/abs/2212.07143
- Sun et al. EVA-CLIP. https://arxiv.org/abs/2303.15389 ; EVA-CLIP-18B https://arxiv.org/abs/2402.04252
- Tschannen et al. SigLIP 2. https://arxiv.org/abs/2502.14786 ; card https://huggingface.co/google/siglip2-so400m-patch14-384
- Bolya et al. Perception Encoder. https://arxiv.org/abs/2504.13181
- Koukounas et al. jina-clip-v2. https://arxiv.org/abs/2412.08802 ; card https://huggingface.co/jinaai/jina-clip-v2
- Li, Li, Savarese, Hoi. BLIP-2. ICML 2023. https://arxiv.org/abs/2301.12597 ; LAVIS https://github.com/salesforce/LAVIS
- Wei et al. UniIR. ECCV 2024. https://arxiv.org/abs/2311.17136 ; https://github.com/TIGER-AI-Lab/UniIR
- Jiang et al. E5-V. https://arxiv.org/abs/2407.12580 ; card https://huggingface.co/royokong/e5-v
- Jiang et al. VLM2Vec. ICLR 2025. https://arxiv.org/abs/2410.05160
- Meng et al. VLM2Vec-V2. https://arxiv.org/abs/2507.04590
- Zhang et al. GME. CVPR 2025. https://arxiv.org/abs/2412.16855 ; card https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
- Lin et al. MM-Embed. ICLR 2025. https://arxiv.org/abs/2411.02571 ; card https://huggingface.co/nvidia/MM-Embed
- Chen et al. mmE5. https://arxiv.org/abs/2502.08468
- Gu et al. UniME. ACM MM 2025. https://arxiv.org/abs/2504.17432
- Lan et al. LLaVE. EMNLP 2025 Findings. https://arxiv.org/abs/2503.04812
- Thirukovalluru et al. B3. https://arxiv.org/abs/2505.11293
- Xiao et al. MetaEmbed. ICLR 2026. https://arxiv.org/abs/2509.18095
- Li et al. Qwen3-VL-Embedding and Qwen3-VL-Reranker. https://arxiv.org/abs/2601.04720 ; card https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
- Günther et al. jina-embeddings-v4. https://arxiv.org/abs/2506.18902 ; card https://huggingface.co/jinaai/jina-embeddings-v4
- Zhai et al. LiT. CVPR 2022. https://arxiv.org/abs/2111.07991
- Kossen et al. Three Towers. NeurIPS 2023. https://arxiv.org/abs/2305.16999
- Maniparambil et al. Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment. CVPR 2025. https://arxiv.org/abs/2409.19425
- Maniparambil et al. Do Vision and Language Encoders Represent the World Similarly? CVPR 2024. https://arxiv.org/abs/2401.05224
- Zhang, Yang, Agrawal. Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL). CVPR 2025. https://arxiv.org/abs/2412.04616
- Norelli et al. ASIF: Coupled Data Turns Unimodal Models to Multimodal Without Training. https://arxiv.org/abs/2210.01738
- Moschella et al. Relative representations enable zero-shot latent space communication. ICLR 2023. https://arxiv.org/abs/2209.15430
- Merullo, Castricato, Eickhoff, Pavlick. Linearly Mapping from Image to Text Space. ICLR 2023. https://arxiv.org/abs/2209.15162
- Huh, Cheung, Wang, Isola. The Platonic Representation Hypothesis. https://arxiv.org/abs/2405.07987
- Schnaus, Araslanov, Cremers. It's a (Blind) Match! CVPR 2025. https://arxiv.org/abs/2503.24129

**UNVERIFIED or second-hand items.** DistillCIR, DiffComp and CIG GeneCIS tables (CVF returned 403; numbers only as quoted by others); SEIZE's GeneCIS numbers; whether MCL (ICML 2024) reports GeneCIS; DiscoverNet and Generalized CSL numbers; the identity of SteerViT's "specialized" baseline; MMRet's CLIP-B patch size; licences of VLM2Vec-V2, E5-V, mmE5, PE-Core, the Reed caption files and CelebAText-HQ; the CUB licence conflict; how DeepFashion-MultiModal captions were made; venues of FocalLens, SemArt, SteerViT and DIOR from secondary sources only.
