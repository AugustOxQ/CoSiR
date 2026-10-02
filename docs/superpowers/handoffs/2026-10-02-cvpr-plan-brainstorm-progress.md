# CVPR plan brainstorm: progress note (2026-10-02)

This note continues [the CVPR handoff](2026-10-02-cosir-v2-cvpr-publication-handoff.md). A session working through the
brainstorm (superpowers:brainstorming, architectural path) wrote it before a container restart. Resume from
"Where to resume".

## Fixed facts

- **CVPR deadlines** (from the user): abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23.
- **Compute:** DAS6 node404 is available (the user said so on 2026-10-02). The user normally gets up to 3 GPUs
  (one node), and 3 to 6 nodes when the cluster is idle. The local RTX 3090 is for smoke tests; it lost GPU access
  on 2026-10-02 (NVML "Unknown Error"), which is why the user is restarting the container.
- **Datasets on disk:** under `/data/SSD/`, by `scripts/download_benchmarks.sh`.
  - CUB: 11,788 images plus Reed captions in `captions/extracted/text_c10`.
  - SemArt: 21,384 images, with TYPE, SCHOOL and TIMEFRAME in the CSVs.
  - Visual Genome 1.2: 108,249 images in `VG_100K_all`, covering all 23,640 GeneCIS attribute images.
  - Point `/project/genecis/config.py` `visual_genome_images` at `/data/SSD/visual_genome/VG_100K_all`.

## Evidence produced in this session (all committed, all on selection rows, held rows untouched)

| Report | Finding |
|---|---|
| [CVPR literature review](../../reports/auto/v2/2026-10-21_cvpr_literature_review.md) | No prior example-conditioned cross-item image–text similarity (to our knowledge). The naive rule is Rocchio and the factor term a CSN mask. "Unsupervised condition discovery" is taken (SCE-Net, DiscoverNet, EmotionCLIP). PercepT itself uses ModernBERT-GoEmotions and CLIP ViT-L/14, so do not call our RoBERTa "PercepT's teacher". |
| [Support-baseline spike](../../reports/auto/v2/2026-10-22_support_baseline_spike.md) | On value episodes, raw-CLIP support baselines beat SE (probe 24.10 vs 21.22 pooled R@1). A prototype that ignores the query reaches 22.83, so value episodes are few-shot classification. SE wins only cross-modally (style i2t, emotion t2i). |
| [Aspect-episode spike](../../reports/auto/v2/2026-10-23_aspect_episode_spike.md) | On aspect episodes no factor model beats CLIP (SE 11.32 vs CLIP 11.13). The label-probe ceiling is about 23, so the task is learnable. Emotion lives in captions and style in images, which caps cross-modal matching on ArtELingo. |
| [Aspect novelty check](../../reports/auto/v2/2026-10-24_aspect_task_novelty_check.md) | No paper defines the task, though every property exists separately. Closest: Contextual Visual Similarity, metric learning from pairs (Xing, RCA, KISSME), MARS, in-context text embedders. Expect the reviewer line "a few-shot cross-modal diagonal KISSME". |

## Decisions (the user's)

1. **Problem: aspect, not value.** The condition is shown by example pairs that share an aspect, with values
   different from the query's; contrast pairs share another aspect; swapping the condition flips the positive.
2. **Benchmarks:** ArtELingo (primary), CUB + Reed captions, GeneCIS focus tasks, SemArt. Affection is a potential
   extra the user will check.
3. **Method A: aspect-trained factors.** Shared sparse I–T factors, trained on pseudo-aspect episodes built from 2+
   pseudo-partitions, with the training-free agreement rule. The user chose A alone, not "A with an
   example-estimated-subspace fallback".
4. **Backbones:** develop on CLIP ViT-B/32. Report final tables also on ONE strong frozen backbone, picked by the
   label-probe ceiling on ArtELingo and CUB among SigLIP 2 So400m, PE-Core L/14 and Qwen3-VL-Embedding-2B.

## Design section 1 (APPROVED by the user, revised after the novelty check)

**Problem definition: example-conditioned aspect similarity across modalities.**
- The input is a query x (an image or a caption) and a condition c = (S, C).
  - S: a few support pairs (an image of one item, a caption of another) that agree on the wanted aspect, with any
    values.
  - C: a few contrast pairs that agree on a different aspect.
- The output ranks candidates y in the other modality by s(x, y | c), so that the candidate sharing the query's value
  on the aspect S demonstrates comes first.
- The condition never shows the query's value and never names the aspect. Both directions are scored.
- Swap test: the same query and gallery under the swapped condition must re-rank.
- Against GeneCIS: there, focus conditions name the aspect in text and compare image with image.

**Contributions.**
- **C1, task and protocol.** Novelty statement: "to our knowledge, the first cross-modal similarity in which the
  aspect is fixed at test time only by value-disjoint, cross-item image–caption examples with a contrast aspect,
  evaluated in both directions with a paired swap test". Neighbours to cite: metric learning from pairs, Contextual
  Visual Similarity, MARS, GeneCIS, CLAY, CRL. Motivation: value episodes are solved by the supports alone.
- **C2, method.** A shared image–text factor basis, trained on pseudo-aspect episodes with no labels from the
  evaluation taxonomy, is the prior that makes test-time metric estimation from four pairs work. The agreement rule is
  presented as a Rocchio/KISSME-style estimator, not as the novelty.
- **C3, analysis.** Aspects live asymmetrically across modalities, and the weaker modality caps cross-modal matching.

**Claims.**

| # | Claim | Status |
|---|---|---|
| K1 | Value episodes are solved by the supports alone | done (support spike) |
| K2 | Beats CLIP, per-episode metric-from-pairs baselines (diagonal and low-rank KISSME/RCA on raw features) and the same rule on unsupervised bases, on aspect episodes, both directions, all datasets | open; factors are at CLIP level today |
| K3 | Examples beat names on subjective aspects and match them on objective ones | open |
| K4 | Gains hold on a strong backbone | open |
| K5 | GeneCIS focus is competitive with the published frozen-B/32 rows (example and text protocols reported apart) | open |
| K6 | The rule selects aspect factors; the factors are shared across modalities | open |
| K7 | The gain comes from the learned basis (same rule on raw, PCA/NMF and learned factors) | open |

Fallback if K2 fails: K1, K3 and C3 could make an analysis or benchmark paper, not the method paper.

## Design section 2 (PRESENTED, awaiting the user's approval)

- **Common protocol.**
  - Aspect episodes as in the aspect spike: 13 candidates [p_A, p_B, 11 sharing neither]; P_A and P_B are 4
    cross-item pairs each, with values distinct from each other and from the anchor's; condition A uses (S=P_A,
    C=P_B) and condition B swaps them; all items in an episode are distinct.
  - Primary metric: R@1, the mean over directions and conditions. Secondary: swap success (pairwise, read with R@1),
    the other-aspect rate, and a 101-candidate gallery variant on ArtELingo and CUB.
  - Factor training uses no labels. A labelled development split serves model selection. The test split is read
    within the held budget. Paired bootstrap over anchors, 3 seeds.
- **ArtELingo:** emotion (8) × style (23); the existing held rows (read 3 times before, on value episodes). ArtELingo
  has NO genre field (genre exists for only 6,515 rows in `artelingo_genre_emotion_eng.json`), so the proposal is two
  aspects.
- **CUB + Reed:** aspects primary colour (15), bill shape (9) and size (5), per-image labels at certainty ≥
  "probably" with exactly one value; proposed test on the zero-shot split's 50 unseen species.
- **SemArt:** type, school and timeframe, captions scrubbed of artist names and dates; official test (1,069
  paintings); main paper if on time, else the supplement.
- **GeneCIS focus attribute** (condition = attribute type, an aspect; image to image): the example protocol is
  ours; the text protocol (phrase-to-weights adapter, needed for comparison with SEARLE 14.4, CIReVL 15.9 and
  OSrCIR 17.4) is a stretch goal. Focus object is value-type and goes to the supplement only.
- **Priority:** ArtELingo and CUB, then GeneCIS focus attribute, then SemArt.
- **Open calls for the user:** CUB on unseen species? ArtELingo without genre?

## Where to resume

1. Get the user's answer on section 2.
2. Re-dispatch the **backbone check** once the GPU works. The brief: label-probe ceilings and backbone-only aspect
   R@1 on ArtELingo (reproducing `src/test/20261023_aspect_episode_spike/aspect_ceiling.py` for CLIP B/32 exactly),
   plus CUB retrieval sanity and attribute probes per modality, for SigLIP 2 So400m, PE-Core L/14 and
   Qwen3-VL-Embedding-2B.
   - Install newer packages only with `pip install --target /data/SSD2/pyenvs/backbone_check/`, never into the
     CoSiR env.
   - Folder `src/test/20261025_backbone_check/`.
3. Present the remaining sections, one at a time:
   - 3, method A and its go/no-go: pseudo-aspect episodes, measured against CLIP (about 11) and the ceiling
     (about 23) on selection rows;
   - 4, baselines: raw-feature metric-from-pairs (diagonal and low-rank KISSME/RCA, Xing per episode), the same rule
     on PCA, NMF and SpLiCE, prototype or Rocchio, names or CRL, CLAY, instruction embedders (Qwen3-VL-Embedding, GME)
     given the aspect, teacher-only, PercepT (rebuilt cleanly), the label-supervised ceiling, and published GeneCIS
     rows;
   - 5, held budget and statistics;
   - 6, schedule backwards from Nov 10, 16 and 23, and risks.
4. Then write the spec at `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`, self-review
   it, get the user's review, and move on to writing-plans.
