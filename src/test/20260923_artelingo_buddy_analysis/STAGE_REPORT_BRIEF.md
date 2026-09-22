# Brief: stage report wrapping up the ArtELingo affect-vs-buddy-graph investigation

Write `docs/reports/2026-09-22_artelingo_affect_investigation_stage_report.md`.

**Audience:** someone who has NOT been following this investigation closely.
The file must be fully self-contained — every metric/term used (AMI,
V-measure, ARI, silhouette, buddy graph, Leiden, DEC, etc.) must be defined
in plain language, in the file itself, before or at first use. Do not assume
the reader knows any of this vocabulary already. Semi-formal tone, matching
the style of `docs/reports/2026-09-16_stage_report_prototype_conditioning.md`
(read that file first for tone/structure reference, but do not copy its
content — this is a different, unrelated investigation).

**Every number below is final and verified — use them exactly as given, to
the precision shown. Do not round differently, do not invent any number not
listed here, do not compute new derived numbers beyond simple, clearly
-labeled arithmetic (e.g. relative-percent-change) using only these given
values.**

## Section 1 — Why this investigation started

CoSiR conditions a frozen CLIP embedding using a "buddy graph": a graph
connecting samples whose CLIP image/text embeddings are mutual nearest
neighbors (each considers the other a top-K neighbor), used to give each
sample's trainable condition vector a content-aware starting point.
Experiment 18 tried replacing CoSiR's old per-sample condition table with a
small shared "prototype bank," seeded from buddy-graph communities.

Separately, the team found and read a paper, "Beyond Semantics: Modeling
Factual and Affective Perceptual Experiences from Vision-Language Data"
(PercepT), which discovers "P-Topics" — visual-textual clusters capturing
both factual and *affective* (emotional) structure — using a fundamentally
different mechanism: a learned latent space `Z` (a small autoencoder)
shaped by an ongoing "Deep Embedded Clustering" (DEC) loss that iteratively
sharpens soft cluster assignments, decoupled from any downstream task.

**The question that kicked off this investigation:** can CoSiR's existing
buddy-graph initialization play the same role as PercepT's `Z` + DEC
mechanism? A joint brainstorm (author + an independently-dispatched review)
concluded: no — buddy-graph init is only a *one-shot* seed (like PercepT's
K-means initialization step), not a substitute for DEC's *ongoing*,
reconstruction-anchored clustering-shaping loss. This was flagged as the
likely root cause of an unresolved Experiment 18 trade-off (silhouette
improved with training, but retrieval got seed-replicated-worse). A second,
related hypothesis: CoSiR's RedCaps-based `warmth`/`register` proxy probes
found little affect signal because CoSiR has never fed an affect-aware
signal into its pipeline at all — it only ever uses plain CLIP features,
which are known to carry semantic/factual content far better than emotion.

**Why ArtELingo:** RedCaps has no ground-truth emotion or content-category
labels, so any probe of "does buddy-graph structure carry affect" had to use
indirect subreddit-content proxies (weak, hard to interpret cleanly). The
ArtELingo dataset (WikiArt paintings with ArtEmis-style human emotion
annotations, 9-way taxonomy: amusement, awe, contentment, excitement, anger,
disgust, fear, sadness, "something else") gives REAL ground-truth emotion
labels, plus a genre label (landscape, portrait, religious_painting, etc.)
on a smaller subset — letting these questions be tested directly, with real
labels, on a domain unrelated to RedCaps.

## Section 2 — Glossary (define ALL of these, plainly, before or at first use)

- **Buddy graph**: see above — mutual-nearest-neighbor graph over CLIP
  features.
- **Mutual-kNN**: an edge between two samples exists only if EACH considers
  the OTHER one of its top-K nearest neighbors (a reciprocal/mutual
  relationship, not a one-directional "nearest neighbor").
- **Leiden community detection**: an algorithm that partitions a graph into
  densely-connected sub-groups ("communities") automatically, without being
  told in advance how many groups to find.
- **DEC (Deep Embedded Clustering)**: a different clustering approach that
  jointly trains a small neural network encoder and a set of "cluster
  center" points, using a loss that repeatedly sharpens soft (fuzzy)
  cluster assignments into more confident ones over many training steps —
  this is the mechanism the PercepT paper uses.
- **Genre / emotion ground truth**: real human-provided labels from the
  ArtELingo dataset. Genre = what is depicted (a landscape? a portrait?).
  Emotion = the felt reaction reported by a human annotator viewing the
  painting, one of the 9 categories listed above.
- **Majority-vote label**: each painting in ArtELingo has ~5 independent
  human emotion annotations (from different captions/annotators); the
  single most common one among them is used as that painting's one "ground
  truth" emotion for these tests.
- **AMI (Adjusted Mutual Information)**: the primary metric used throughout.
  It measures how much two groupings of the same items agree with each
  other — e.g., "the graph's computer-found communities" vs. "the real
  emotion/genre labels." It ranges roughly 0 (no better than random/chance
  agreement — this is the "adjusted" part, it corrects for the fact that
  random groupings can look like they agree somewhat purely by luck) to 1
  (perfect agreement). Unlike raw accuracy, AMI does not require knowing
  which computer-found cluster corresponds to which real label — it only
  measures whether the *grouping structure* matches.
- **V-measure**: a second, closely related 0–1 agreement score, computed as
  the harmonic mean of "homogeneity" (each found cluster contains mostly one
  true category) and "completeness" (each true category falls mostly into
  one found cluster). Reported alongside AMI throughout; the two numbers are
  usually close in magnitude.
- **ARI (Adjusted Rand Index)**: another chance-corrected 0–1 agreement
  score, computed differently (by looking at pairs of items and checking
  whether the two groupings agree on "these two items are together" or
  "these two items are apart," for every pair). Reported for a subset of
  results in Section 1's pilot only.
- **Silhouette score**: unlike AMI/V-measure/ARI, this does NOT compare
  against any ground-truth label — it measures purely geometric cluster
  quality (are points much closer to others in their own cluster than to
  points in other clusters?). Higher is more separated; near zero or
  negative means little real structure.
- **Off-the-shelf vs. in-domain/fine-tuned encoder**: "off-the-shelf" means
  a general-purpose model not specifically trained on this dataset's own
  labels (e.g. a GoEmotions classifier trained on Reddit comments).
  "In-domain" or "fine-tuned" means a model directly trained on this exact
  dataset's own human-provided labels — usually scores higher on that exact
  data, but for a less generalizable reason (see memorization, below).
  "Supervised" describes any model trained using human-provided ground-truth
  labels (as opposed to "unsupervised" methods like buddy-graph/Leiden/DEC,
  which find structure without ever being told the correct answer).
- **Train / held-out (val+test) split, memorization vs. generalization**:
  standard machine-learning practice — a model's usefulness should be judged
  on data it was NOT trained on ("held-out" data), because testing it on the
  same data it was trained on can just reflect memorization (the model
  "remembering" specific training examples) rather than real, generalizable
  understanding. This distinction becomes directly important in Section 3,
  pilot 6.
- **Painting-node deduplication**: each ArtELingo painting has ~5 separate
  caption rows (one per human annotation). Graphs in this investigation are
  built with one node per unique painting (not per caption row), to avoid
  creating meaningless zero-distance edges between multiple rows of the
  exact same image.

## Section 3 — Data setup (state plainly, no invented numbers)

- Source: ArtELingo dataset (WikiArt paintings + ArtEmis-style emotion
  captions), downloaded from the authors' distribution.
- English-only rows used throughout (CLIP's text tower is English-tuned).
- Main annotation set: train 308,723 caption rows → 61,402 unique paintings;
  held-out val+test combined: 46,813 caption rows → 9,365 unique paintings.
  **Verified zero painting-level overlap between train and held-out.**
- A smaller subset (6,515 rows / 1,303 unique paintings) carries BOTH genre
  and emotion ground truth together (from a separate ArtELingo-28 release);
  1,144 of those 1,303 paintings are inside the train set, used as the
  genre-ground-truth source for pilots run on train-set paintings.
- Images reused directly from an already-present local WikiArt image
  archive — no re-download needed.
- CLIP ViT-B/32 features extracted for both the train and held-out painting
  sets.

## Section 4 — The six pilots, in order run

For each pilot: state the setup in 1–2 sentences, then the exact result
numbers in a small table, then 1–2 sentences of plain-language
interpretation. Use these EXACT numbers:

**Pilot 1 — CLIP-only buddy graph, ground-truth correlation (`run_pipeline.py`)**
Buddy graph built from plain CLIP image+text features only, K=20 (a smaller
K than the K=30 validated elsewhere for RedCaps, since this corpus — 61,402
nodes — is smaller scale; noted as a judgment call, not independently
validated), Leiden found 28 communities.
- Community vs. genre (n=1,144 genre-labeled paintings): AMI=0.4384,
  V-measure=0.4572, ARI=0.3257.
- Community vs. emotion (full graph, n=61,402): AMI=0.0593,
  V-measure=0.0600, ARI=0.0271.
- Community vs. coarse valence (positive/negative/"something else" 3-way
  collapse of the 9 emotions): AMI=0.0270.
- Ground-truth genre-vs-emotion baseline correlation (i.e. how entangled
  are genre and emotion already, in the real labels, before any buddy-graph
  clustering): AMI=0.0723 — weak, meaning genre and emotion are largely
  independent properties in the data itself.
- Content-stratified check (does the graph still separate emotion WITHIN a
  single genre, controlling for content?): AMI(community, emotion | genre)
  ranged from −0.0114 (abstract_painting) to 0.1369 (still_life, small
  n=40) across genres — comparable in magnitude to the unconditional 0.0593,
  meaning controlling for genre did not reveal a much bigger hidden emotion
  signal, but also did not fully explain away the weak one that exists.
- Rare-class check (the 422 anger-labeled paintings, the smallest class):
  concentrated about 3.2x above base rate in one community — not diluted
  into random neighbors.
- Cross-lingual check (same English-built graph, correlated against Arabic
  and Chinese emotion labels for the same images): AMI≈0.057–0.060 in all
  three languages on matched subsets — the weak signal is language-
  independent, not an English-caption-vocabulary artifact.
- Robustness check: 25.9% of majority-vote emotion labels are statistical
  ties (no single most-common emotion); restricting to the 74% non-tied
  paintings gives AMI=0.0758 (slightly higher, not lower) — the weak
  emotion signal is not a tie-handling artifact.
- **Plain-language takeaway:** buddy-graph structure built from plain CLIP
  features strongly tracks content/genre (AMI 0.44) but only weakly tracks
  emotion (AMI 0.06) — about a 7x gap — confirmed with real human labels,
  not proxies, and robust to several different checks.

**Pilot 2 — Adding an off-the-shelf affect encoder, fusion sweep (`run_affect_pilot.py`)**
GoEmotions (`SamLowe/roberta-base-go_emotions`, a Reddit-comment-trained,
28-category emotion classifier — NOT trained on ArtELingo) run over each
painting's captions, concatenated into the buddy graph's text features at 5
weights: 0 (pure CLIP control), 0.5, 1.0, 2.0, 4.0.
| affect weight | emotion AMI | genre AMI |
|---:|---:|---:|
| 0.0 | 0.0593 | 0.4384 |
| 0.5 | 0.0614 | 0.4045 |
| 1.0 | 0.0759 | 0.4025 |
| 2.0 | 0.1119 | 0.2042 |
| 4.0 | 0.1160 | 0.0867 |
The weight=0.0 control reproduced Pilot 1's numbers exactly, confirming
correctness. **Plain-language takeaway:** a real, monotonic trade-off, not a
free win — emotion signal roughly doubles as affect weight increases, but
genre signal collapses by 80% at the same time. No single weight achieved
both a large emotion gain (AMI > 0.177, a predeclared "50% relative
improvement" bar) AND kept genre AMI above 80% of its original value.

**Pilot 3 — Single-modality ceiling check (`run_single_modality_pilot.py`)**
Three SEPARATE graphs, each built from only one feature type (not combined),
to check whether Pilot 2's trade-off was hiding a stronger affect signal
that concatenation was diluting.
| graph built from | emotion AMI | genre AMI |
|---|---:|---:|
| CLIP image only | 0.0540 | 0.4290 |
| CLIP text only | 0.0638 | 0.2931 |
| GoEmotions affect only | 0.1180 | 0.0396 |
**Plain-language takeaway:** the affect-only graph's emotion ceiling
(0.1180) is barely above the best fused result from Pilot 2 (0.1160) —
concatenation was NOT hiding a much stronger signal; genre AMI in the
affect-only graph collapses to near-zero (0.0396), confirming content and
affect specialize into largely separate structure. This 0.1180 number
becomes the reference "off-the-shelf affect ceiling" for later pilots.

**Pilot 4 — DEC instead of Leiden, first attempt (`run_dec_pilot.py`)**
Tests whether PercepT's actual clustering method (DEC, not Leiden) does
better on the exact same GoEmotions-only input. Small autoencoder (28→16
dimensions), K=28 clusters, DEC training for a FIXED 100 epochs.
Result: emotion AMI=0.1258, genre AMI=0.0365. No cluster collapse (healthy
cluster sizes). BUT: both loss components kept rising for the entire 100
epochs, never leveling off — a sign the run had not reached a stable
equilibrium. **Plain-language takeaway:** inconclusive — a modest gain over
Leiden's 0.1180, but not trustworthy as DEC's real ceiling since the
training run had not properly finished converging.

**Pilot 5 — DEC with proper convergence control (`run_dec_pilot_v2.py`)**
Same setup, but training now stops based on an "assignment-stability"
rule (the standard one from the original DEC method): stop once fewer than
0.1% of paintings change which cluster they're assigned to between one
epoch and the next, rather than after a fixed number of epochs.
Training genuinely stabilized at epoch 185 (well under a 500-epoch safety
cap). Result: emotion AMI=0.1492, genre AMI=0.0554 — both numbers improved
over Leiden's reference (0.1180 / 0.0396), and cluster sizes were healthy
(zero collapsed clusters, even better than Pilot 4). **Plain-language
takeaway:** a real, reproducible, properly-converged +26.4% relative
improvement in emotion AMI over Leiden ((0.1492−0.1180)/0.1180 = +26.4%) —
clustering method does matter somewhat, but the ceiling only moved
modestly, not dramatically, and still fell short of the predeclared
"meaningful win" bar (AMI > 0.177).

**Pilot 6a — Using ArtELingo's own fine-tuned emotion classifier, train split (`run_bert_ceiling_pilot.py`)**
The ArtELingo authors published their own fine-tuned 9-way emotion
classifier (BERT-base, trained directly on ArtELingo's own captions and
labels) — used here instead of GoEmotions. First, a direct sanity check:
does the classifier's prediction on each caption match that caption's own
human-provided emotion label? Result: 93.67% accuracy (289,196/308,723
caption rows) on the TRAIN split — the same split this checkpoint was
almost certainly trained on, so this number is expected to include
memorization, not just genuine understanding.
Graph ceiling on this train-split signal: emotion AMI=0.2897 (about 2.45x
GoEmotions' 0.1180), genre AMI=0.0582. **Plain-language takeaway:** a huge
jump, but explicitly flagged as likely inflated by the model having been
trained on this exact data — not yet a fair or trustworthy number on its
own.

**Pilot 6b — Same classifier, genuinely held-out val+test split (`run_bert_heldout_pilot.py`)**
The decisive check: the same classifier run on 46,813 caption rows / 9,365
paintings verified to have ZERO overlap with the train split the model was
fine-tuned on — a fair, standard generalization test.
- Per-caption accuracy on held-out data: 64.38% (30,140/46,813) — a real
  drop of 29.29 percentage points from the train split's 93.67%, confirming
  substantial memorization inflation in Pilot 6a's train-split number.
- Emotion AMI on held-out data: 0.1693 — down from the train split's
  0.2897, but STILL above GoEmotions' off-the-shelf ceiling of 0.1180, by
  +43.5% relative ((0.1693−0.1180)/0.1180 = +43.5%) — the single largest
  non-leaky (fairly measured) improvement found across this entire
  investigation.
- Genre AMI on held-out data: 0.0138, but only 159 genre-labeled paintings
  overlapped the held-out set (versus 1,144 for the train-split pilots), so
  this specific number is low-confidence/low-statistical-power and should
  not be read as strongly as the emotion result.
- As an independent stability check, the classifier's anonymized output
  labels were re-mapped to real emotion names separately on this held-out
  data (without reusing the train run's mapping) and the recovered mapping
  matched the train run's mapping exactly, label for label, across all 9
  categories — the model's internal representation is stable, not a
  train-run fluke.
**Plain-language takeaway:** a real, substantial generalization gap exists
(the train-split number was significantly inflated by memorization), but
it is not a total washout — a genuinely fairly-measured, domain-matched,
human-label-supervised emotion signal still meaningfully outperforms every
other (unsupervised, off-the-shelf) approach tried in this investigation.

## Section 5 — Cross-pilot summary table

Reproduce this table exactly (all values given above):

| pilot | signal | emotion AMI | genre AMI | note |
|---|---|---:|---:|---|
| 1 | CLIP only | 0.0593 | 0.4384 | ground-truth-confirmed baseline |
| 2 (best) | CLIP + GoEmotions, weight=4 | 0.1160 | 0.0867 | real trade-off, no free win |
| 3 | GoEmotions only | 0.1180 | 0.0396 | off-the-shelf affect ceiling |
| 4 | DEC, non-converged | 0.1258 | 0.0365 | inconclusive |
| 5 | DEC, converged | 0.1492 | 0.0554 | +26.4% relative, real but modest |
| 6a | ArtELingo BERT, train (leaky) | 0.2897 | 0.0582 | inflated by memorization |
| 6b | ArtELingo BERT, held-out | 0.1693 | 0.0138 (n=159) | +43.5% relative, best fair result |

## Section 6 — What this means, and what's still open

State these points plainly, as the closing synthesis:

1. Buddy-graph structure built from plain, frozen CLIP features is
   confirmed — with real human ground truth, not indirect proxies, and on a
   domain (fine art) unrelated to CoSiR's usual RedCaps/Impressions data —
   to be fundamentally a content/genre-detecting mechanism, not an
   emotion-detecting one. This directly validates, with independent
   evidence, the concern raised in the original PercepT brainstorm.
2. Switching to PercepT's own clustering mechanism (DEC) instead of
   buddy-graph's Leiden step, done properly (with real convergence
   verification, not just running a fixed number of epochs), only closes
   part of the gap (+26.4% relative) — clustering method is not the main
   bottleneck.
3. A domain-matched, human-label-supervised affect signal closes much more
   of the gap (+43.5% relative, fairly measured on held-out data) — but
   requires labeled emotion data to build in the first place. This is a
   fundamentally different, far less general approach than the original
   "discover affect unsupervised from buddy-graph structure" ambition, and
   it does NOT transfer to RedCaps, CoSiR's other main dataset, which has no
   emotion labels to build a comparable classifier from.
4. Net conclusion: buddy-graph alone will not deliver PercepT-style affect
   interpretability. Getting meaningfully closer requires either accepting
   a much more modest (~15–17% AMI) ceiling under realistic, unsupervised,
   RedCaps-compatible conditions, or a genuine architectural/data change
   (e.g. targeting a labeled dataset, or building a supervised affect
   branch) that goes beyond what buddy-graph initialization was ever
   designed to do.

State as explicitly open / not yet tested, without editorializing further:
- Whether a different OFF-THE-SHELF (not fine-tuned-on-ArtELingo-labels)
  affect encoder, better matched to art-caption language than a
  Reddit-trained one, could close some of the gap between 0.1180 and 0.1693
  without needing in-domain supervision.
- Whether a genuinely two-part architecture (a separate content-oriented
  graph/bank and a separate affect-oriented graph/bank, rather than one
  shared graph) would do better than any single-graph approach tried here —
  argued for based on Pilot 1/3's content/affect-specialization finding,
  but not yet built or tested.
- How, or whether, any of this feeds back into Experiment 18's own,
  separate, still-unresolved retrieval-vs-interpretability trade-off
  (found on RedCaps, not ArtELingo) — this investigation was an exploratory
  side-branch prompted by reading the PercepT paper, not a direct fix
  attempt for that trade-off.

## Section 7 — Process notes (short, factual, not self-congratulatory)

One short paragraph, factual tone: every pilot was designed (architecture,
hyperparameters, and success criteria decided and written down BEFORE
seeing results) by the report author, implemented by Codex per a detailed
written brief, independently code-reviewed line-by-line before running, and
executed directly by the author on GPU rather than delegated (an established
project policy, since long GPU-holding runs delegated to Codex have shown
session instability in the past). Two real issues were caught and fixed
during this process, not just at the end: (1) a feature-store completeness
check that could have raced a still-writing extraction job, caught before
it caused a problem; (2) a majority-vote tie-handling issue in the labeling
logic, caught by an independent Codex review and confirmed not to affect
results via a dedicated sensitivity check. All pilot scripts, briefs, and
reports are committed to the `experiment/buddy_prototype_conditioning`
branch under `src/test/20260923_artelingo_buddy_analysis/`.

---

Do not add a "Next steps" section beyond what Section 6 already states, and
do not add speculative content not grounded in the numbers given above.
