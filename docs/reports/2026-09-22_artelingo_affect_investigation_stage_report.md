# ArtELingo stage report: affect investigation

## I. Why this investigation started

CoSiR conditions a frozen CLIP embedding using a **buddy graph**: a graph connecting samples whose CLIP image/text embeddings are mutual nearest neighbors. CLIP is a pretrained model that places images and text in shared numeric representations called embeddings; “frozen” means its pretrained parameters are not updated during this work. In other words, two samples are connected when each considers the other among its top-K most similar samples. The graph gives each sample's trainable condition vector, a small numeric vector learned during training to adjust that sample's embedding, a content-aware starting point. Experiment 18 tried replacing CoSiR's old per-sample condition table with a small shared prototype bank, a set of learned representative vectors, seeded from buddy-graph communities.

Separately, the team read *Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from Vision-Language Data* (PercepT). That paper discovers “P-Topics”: visual-textual clusters intended to capture both factual and affective, meaning emotional, structure. It uses a fundamentally different mechanism: a learned latent space called `Z`, implemented as a small autoencoder, together with an ongoing Deep Embedded Clustering loss. An autoencoder is a small neural network that learns a compact representation while learning to reconstruct its input. Deep Embedded Clustering (DEC) repeatedly sharpens fuzzy cluster assignments during training and is separate from any downstream task.

The question that started this investigation was whether CoSiR's existing buddy-graph initialization could play the same role as PercepT's `Z` plus DEC mechanism. A joint brainstorm by the author and an independently-dispatched review concluded that it cannot: buddy-graph initialization is only a one-shot seed, like PercepT's K-means initialization step, rather than DEC's ongoing clustering-shaping loss anchored by reconstruction. K-means is a standard method that initially groups items around cluster centers. This distinction was flagged as a likely root cause of an unresolved Experiment 18 trade-off: geometric cluster separation improved with training, but retrieval became worse in a pattern replicated across random starting seeds.

A related hypothesis was that CoSiR's RedCaps-based `warmth` and `register` proxy probes found little affect signal because CoSiR has never supplied an affect-aware signal to its pipeline. It uses plain CLIP features, which are known to carry semantic or factual content much better than emotion.

ArtELingo was selected because RedCaps has no ground-truth emotion or content-category labels. Any RedCaps probe of whether buddy-graph structure carries affect therefore had to rely on indirect subreddit-content proxies, which are weak and difficult to interpret cleanly. ArtELingo contains WikiArt paintings with ArtEmis-style human emotion annotations in a 9-way taxonomy: amusement, awe, contentment, excitement, anger, disgust, fear, sadness, and “something else.” A smaller subset also includes a genre label, such as landscape, portrait, or religious_painting. ArtELingo therefore makes it possible to test these questions directly with real labels in a domain unrelated to RedCaps.

## II. Glossary

**Buddy graph.** A mutual-nearest-neighbor graph over CLIP features. It connects samples that each judge the other similar in CLIP's image, text, or combined feature space.

**Mutual-kNN.** “kNN” means K nearest neighbors. An edge between two samples exists only when each sample considers the other one of its top-K nearest neighbors. This is reciprocal rather than a one-directional nearest-neighbor relationship.

**Leiden community detection.** An algorithm that partitions a graph into densely connected sub-groups, called communities, without being told in advance how many groups to find.

**DEC (Deep Embedded Clustering).** A clustering method that jointly trains a small neural-network encoder and a set of cluster-center points. Its loss repeatedly sharpens soft, or fuzzy, cluster assignments into more confident assignments over many training steps. This is the clustering mechanism used by PercepT.

**Genre / emotion ground truth.** Real human-provided labels from ArtELingo. Genre describes what is depicted, such as a landscape or portrait. Emotion is the felt reaction reported by a human annotator viewing a painting and is one of ArtELingo's 9 categories.

**Majority-vote label.** Each ArtELingo painting has ~5 independent human emotion annotations, from different captions or annotators. The single most common annotation is used as that painting's emotion ground truth for these tests.

**AMI (Adjusted Mutual Information).** The primary agreement metric used here. It measures how much two groupings of the same items agree, for example graph communities against real emotion or genre labels. It ranges roughly from 0, no better than chance agreement, to 1, perfect agreement. “Adjusted” means it corrects for agreement that random groupings can show by luck. Unlike raw accuracy, AMI does not need a computer-found cluster to be matched to a particular real label; it assesses whether the grouping structure matches.

**V-measure.** A second, closely related 0–1 agreement score. It is the harmonic mean of homogeneity, meaning that each found cluster contains mostly one true category, and completeness, meaning that each true category falls mostly into one found cluster. It is reported alongside AMI, and the two scores are usually similar in magnitude.

**ARI (Adjusted Rand Index).** Another chance-corrected 0–1 agreement score. It uses pairs of items, checking for every pair whether the two groupings agree that the items belong together or apart. It is reported only for the first pilot.

**Silhouette score.** A geometric cluster-quality measure that does not compare clusters with ground-truth labels. It asks whether points are much closer to points in their own cluster than to points in other clusters. Higher means better separation; near-zero or negative values mean little real structure.

**Off-the-shelf vs. in-domain/fine-tuned encoder.** An off-the-shelf encoder is a general-purpose model not specifically trained on this dataset's own labels, such as a GoEmotions classifier trained on Reddit comments. An in-domain or fine-tuned encoder is trained directly on this dataset's human-provided labels. It usually scores better on that exact data, but can do so partly by memorizing it. A supervised model is trained using human-provided ground-truth labels. Unsupervised methods, including buddy graph, Leiden, and DEC, seek structure without being told the correct labels.

**Train / held-out (val+test) split; memorization vs. generalization.** Standard machine-learning practice evaluates a model on data it was not trained on, called held-out data. Testing on training data can partly reflect memorization, meaning that the model remembers specific examples, rather than generalization, meaning that it understands patterns that also apply to unseen examples. This matters directly in Pilot 6.

**Painting-node deduplication.** Each ArtELingo painting has ~5 caption rows, one per human annotation. Graphs in this investigation use one node per unique painting rather than one per caption row, avoiding meaningless zero-distance edges between rows for the exact same image.

## III. Data setup

The source was the ArtELingo dataset, containing WikiArt paintings and ArtEmis-style emotion captions, downloaded from the authors' distribution. English-only rows were used throughout because CLIP's text tower is English-tuned. Images were reused directly from an already-present local WikiArt image archive, with no re-download needed. CLIP ViT-B/32 features were extracted for both the train and held-out painting sets.

The main annotation set contained 308,723 train caption rows representing 61,402 unique paintings. The combined held-out validation and test set contained 46,813 caption rows representing 9,365 unique paintings. Painting-level overlap between train and held-out was verified to be zero.

A smaller subset from a separate ArtELingo-28 release contained both genre and emotion ground truth: 6,515 rows representing 1,303 unique paintings. Of those 1,303 paintings, 1,144 are inside the train set and were used as the genre-ground-truth source for pilots run on train-set paintings.

## IV. The six pilots, in order run

### Pilot 1 — CLIP-only buddy graph, ground-truth correlation (`run_pipeline.py`)

The buddy graph used plain CLIP image and text features only. It used K=20, a smaller K than the K=30 validated elsewhere for RedCaps because this 61,402-node corpus is smaller; this was a judgment call rather than an independently validated setting. Leiden found 28 communities.

| comparison | AMI | V-measure | ARI |
|---|---:|---:|---:|
| community vs. genre (n=1,144) | 0.4384 | 0.4572 | 0.3257 |
| community vs. emotion (n=61,402) | 0.0593 | 0.0600 | 0.0271 |
| community vs. coarse valence | 0.0270 | — | — |
| ground-truth genre vs. emotion | 0.0723 | — | — |

Coarse valence is a positive/negative/“something else” 3-way collapse of the 9 emotions. The ground-truth genre-versus-emotion baseline is how entangled genre and emotion already are in the real labels before graph clustering; its weak AMI of 0.0723 means they are largely independent properties in the data.

In a content-stratified check, which asks whether the graph separates emotion within a single genre, AMI(community, emotion | genre) ranged from −0.0114 for abstract_painting to 0.1369 for still_life, where the sample size was n=40. These values are comparable to the unconditional 0.0593: controlling for genre did not reveal a much larger hidden emotion signal, but also did not fully explain away the weak signal that exists. In the rare-class check, 422 anger-labeled paintings, the smallest class, were concentrated about 3.2x above base rate in one community rather than diluted among random neighbors.

A cross-lingual check used the same English-built graph against Arabic and Chinese emotion labels for the same images. AMI was approximately 0.057–0.060 in all three languages on matched subsets, so the weak signal is language-independent rather than an English-caption-vocabulary artifact. For robustness, 25.9% of majority-vote emotion labels were statistical ties, meaning no single emotion was most common. Restricting to the 74% non-tied paintings gave AMI=0.0758, slightly higher rather than lower, so the weak emotion signal is not a tie-handling artifact.

The plain-language result is that buddy-graph structure from plain CLIP features strongly tracks content and genre, with AMI 0.44, but only weakly tracks emotion, with AMI 0.06: about a 7x gap. This is confirmed with real human labels rather than proxies and holds across several checks.

### Pilot 2 — Adding an off-the-shelf affect encoder, fusion sweep (`run_affect_pilot.py`)

GoEmotions (`SamLowe/roberta-base-go_emotions`) is a Reddit-comment-trained 28-category emotion classifier that was not trained on ArtELingo. It was run over each painting's captions and concatenated with the buddy graph's text features at 5 weights. A weight of 0 is the pure-CLIP control.

| affect weight | emotion AMI | genre AMI |
|---:|---:|---:|
| 0.0 | 0.0593 | 0.4384 |
| 0.5 | 0.0614 | 0.4045 |
| 1.0 | 0.0759 | 0.4025 |
| 2.0 | 0.1119 | 0.2042 |
| 4.0 | 0.1160 | 0.0867 |

The weight=0.0 control reproduced Pilot 1 exactly, confirming the comparison is correct. This is a real monotonic trade-off rather than a free win: emotion signal roughly doubles as affect weight rises, while genre signal collapses by 80% at the same time. No weight both produced a large emotion gain, defined in advance as AMI > 0.177 for a “50% relative improvement,” and kept genre AMI above 80% of its original value.

### Pilot 3 — Single-modality ceiling check (`run_single_modality_pilot.py`)

Three separate graphs, each built from one feature type rather than a combination, tested whether Pilot 2's fusion was diluting a stronger affect signal.

| graph built from | emotion AMI | genre AMI |
|---|---:|---:|
| CLIP image only | 0.0540 | 0.4290 |
| CLIP text only | 0.0638 | 0.2931 |
| GoEmotions affect only | 0.1180 | 0.0396 |

The affect-only graph's emotion ceiling of 0.1180 is barely above Pilot 2's best fused result of 0.1160. Fusion was therefore not hiding a much stronger affect signal. Genre AMI in the affect-only graph falls to near-zero, 0.0396, confirming that content and affect specialize into largely separate structure. The 0.1180 result is the reference off-the-shelf affect ceiling for later pilots.

### Pilot 4 — DEC instead of Leiden, first attempt (`run_dec_pilot.py`)

This pilot tested PercepT's DEC method rather than Leiden on the same GoEmotions-only input. It used a small 28→16-dimensional autoencoder, K=28 clusters, and a fixed 100 epochs of DEC training.

| result | emotion AMI | genre AMI |
|---|---:|---:|
| DEC, fixed training | 0.1258 | 0.0365 |

There was no cluster collapse, meaning cluster sizes were healthy. However, both loss components rose throughout the 100 epochs without leveling off, a sign that the run had not reached a stable equilibrium. The modest gain over Leiden's 0.1180 is therefore inconclusive and cannot be treated as DEC's real ceiling.

### Pilot 5 — DEC with proper convergence control (`run_dec_pilot_v2.py`)

This used the same setup but stopped on the original DEC method's standard assignment-stability rule rather than at a fixed epoch: training stops when fewer than 0.1% of paintings change cluster assignment between one epoch and the next. Training stabilized at epoch 185, below a 500-epoch safety cap.

| result | emotion AMI | genre AMI |
|---|---:|---:|
| DEC, converged | 0.1492 | 0.0554 |

Both values improved over Leiden's 0.1180 / 0.0396 reference, and cluster sizes were healthy, with zero collapsed clusters, even better than Pilot 4. This is a real, reproducible, properly converged +26.4% relative improvement in emotion AMI over Leiden: ((0.1492−0.1180)/0.1180 = +26.4%). Clustering method matters somewhat, but the ceiling moved only modestly and still missed the predeclared meaningful-win bar of AMI > 0.177.

### Pilot 6a — ArtELingo's fine-tuned emotion classifier, train split (`run_bert_ceiling_pilot.py`)

The ArtELingo authors published a fine-tuned 9-way emotion classifier based on BERT-base, a language model. It was trained directly on ArtELingo captions and labels and was used here instead of GoEmotions. A direct sanity check compared each caption prediction with that caption's human-provided emotion label on the train split.

| result | value |
|---|---:|
| train per-caption accuracy | 93.67% (289,196/308,723 caption rows) |
| graph emotion AMI | 0.2897 |
| graph genre AMI | 0.0582 |

This train-split accuracy comes from the split on which the checkpoint was almost certainly trained, so it is expected to include memorization rather than only genuine understanding. The graph emotion result is about 2.45x GoEmotions' 0.1180, but it is explicitly likely to be inflated by training on the same data and is not a fair result on its own.

### Pilot 6b — Same classifier, genuinely held-out val+test split (`run_bert_heldout_pilot.py`)

The same classifier was run on 46,813 caption rows and 9,365 paintings with verified zero overlap with the train split used for fine-tuning. This is the standard fair generalization test.

| result | value |
|---|---:|
| held-out per-caption accuracy | 64.38% (30,140/46,813) |
| drop from train accuracy | 29.29 percentage points |
| held-out emotion AMI | 0.1693 |
| held-out genre AMI | 0.0138 (n=159) |

The 29.29 percentage-point drop from the train split's 93.67% confirms substantial memorization inflation in Pilot 6a. Held-out emotion AMI of 0.1693 remains above the off-the-shelf 0.1180 ceiling by +43.5% relative: ((0.1693−0.1180)/0.1180 = +43.5%). This is the largest non-leaky, fairly measured improvement found in the investigation.

The held-out genre result has low confidence and low statistical power because only 159 genre-labeled paintings overlap the held-out set, compared with 1,144 for the train-split pilots. As an independent stability check, anonymized classifier output labels were re-mapped to real emotion names separately on held-out data, without reusing the train mapping. The recovered mapping matched the train mapping exactly, label for label, across all 9 categories. The internal representation is therefore stable rather than a train-run fluke.

The result is a real and substantial generalization gap, not a total washout. A fairly measured, domain-matched, human-label-supervised emotion signal still meaningfully outperforms every unsupervised and off-the-shelf approach tried here.

## V. Cross-pilot summary

| pilot | signal | emotion AMI | genre AMI | note |
|---|---|---:|---:|---|
| 1 | CLIP only | 0.0593 | 0.4384 | ground-truth-confirmed baseline |
| 2 (best) | CLIP + GoEmotions, weight=4 | 0.1160 | 0.0867 | real trade-off, no free win |
| 3 | GoEmotions only | 0.1180 | 0.0396 | off-the-shelf affect ceiling |
| 4 | DEC, non-converged | 0.1258 | 0.0365 | inconclusive |
| 5 | DEC, converged | 0.1492 | 0.0554 | +26.4% relative, real but modest |
| 6a | ArtELingo BERT, train (leaky) | 0.2897 | 0.0582 | inflated by memorization |
| 6b | ArtELingo BERT, held-out | 0.1693 | 0.0138 (n=159) | +43.5% relative, best fair result |

## VI. What this means, and what remains open

Buddy-graph structure built from plain frozen CLIP features is confirmed, with real human ground truth rather than indirect proxies and in fine art rather than CoSiR's usual RedCaps/Impressions data, to be fundamentally a content- and genre-detecting mechanism rather than an emotion-detecting mechanism. This independently validates the concern raised in the original PercepT brainstorm.

Replacing the buddy graph's Leiden step with PercepT's own DEC method, and verifying actual convergence rather than simply running a fixed number of epochs, closes only part of the gap: +26.4% relative. The clustering method is therefore not the main bottleneck.

A domain-matched, human-label-supervised affect signal closes substantially more of the gap, with +43.5% relative improvement fairly measured on held-out data. However, it requires labeled emotion data in the first place. That is a fundamentally different and much less general approach than discovering affect unsupervised from buddy-graph structure. It does not transfer to RedCaps, CoSiR's other main dataset, because RedCaps has no emotion labels from which to build a comparable classifier.

The net conclusion is that buddy graph alone will not deliver PercepT-style affect interpretability. Under realistic, unsupervised, RedCaps-compatible conditions — no human emotion labels available to supervise an encoder — the evidence supports accepting a much more modest approximately 12–15% AMI ceiling (the range spanned by the off-the-shelf GoEmotions signal in Pilot 3 and properly-converged DEC in Pilot 5). Pilot 6b's higher 16.93% figure required human-labeled supervision that RedCaps does not have, so it does not count as a RedCaps-compatible ceiling; reaching it would require a genuine architectural or data change, such as targeting a labeled dataset or building a supervised affect branch. Both options go beyond what buddy-graph initialization was designed to do.

The following remain explicitly open and untested:

- Whether a different off-the-shelf affect encoder, not fine-tuned on ArtELingo labels and better matched to art-caption language than a Reddit-trained model, could close some of the gap between 0.1180 and 0.1693 without in-domain supervision.

- Whether a genuinely two-part architecture, with separate content-oriented and affect-oriented graph or bank components rather than one shared graph, would outperform every single-graph approach tried here. Pilot 1 and Pilot 3 argue for this possibility through their content/affect-specialization finding, but it has not been built or tested.

- How, or whether, these findings feed back into Experiment 18's separate and still-unresolved retrieval-versus-interpretability trade-off. That trade-off was found on RedCaps rather than ArtELingo; this investigation was an exploratory side branch prompted by the PercepT paper, not a direct fix attempt.

## VII. Process notes

Every pilot was designed by the report author before results were seen: architecture, hyperparameters, and success criteria were decided and written down in advance. Codex implemented each pilot from a detailed written brief, and each was independently code-reviewed line by line before it ran. The author ran the GPU work directly rather than delegating it, following established project policy because long GPU-holding runs delegated to Codex have shown session instability in the past. Two issues were caught and fixed during the process: a feature-store completeness check that could have raced a still-writing extraction job, caught before it caused a problem; and a majority-vote tie-handling issue in the labeling logic, caught by an independent Codex review and confirmed not to affect results through a dedicated sensitivity check. All pilot scripts, briefs, and reports are committed on the `experiment/buddy_prototype_conditioning` branch under `src/test/20260923_artelingo_buddy_analysis/`.
