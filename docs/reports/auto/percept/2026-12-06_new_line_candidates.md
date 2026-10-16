# New line: step 3 findings and three candidate ideas (GeneCIS + PercepT + buddy)

> Step 3 note of the r10 decide chat, 2026-10-17 01:03 (Amsterdam). Brainstorm write-up; nothing was trained. The file
> name uses CoSiR's sequence date. Facts come from four checks run in that chat, linked below. The user asked for ideas
> "based on GeneCIS (conditional similarity as the core idea), PercepT (the main model framework) and our conditional
> buddy". The grill on these candidates runs in the next decide chat.

## Brief

- **PercepT is beatable on its own terms.** Its Stage 1 is IDEC; its own tables show DEC adds little to label agreement
  (ArtELingo AMI 0.16 without DEC vs 0.18 with it; Affection 0.21 and 0.25 without vs 0.18 with). Its Stage 2 is scored
  against its own clusters, its affect signal is a supervised emotion classifier, and it has no retrieval.
- **Our PercepT code is not PercepT.** Every generation averages captions per painting (the paper clusters
  image-caption pairs) and differs in encoders, fusion, topic count and Stage 2 targets. The 09-30 tie was against a
  PercepT-like baseline.
- **The GeneCIS-side gap:** no work found uses aspects discovered from image-text data as retrieval conditions. Each
  half is crowded; the expected baseline is CRL (NeurIPS 2025) given aspect names from an LLM.
- **Our data:** different views give different partitions (each view follows a different label), but affect is weak
  without a supervised encoder (caption-emotion AMI 0.05 from CLIP text vs 0.20 from a GoEmotions classifier).
- **Candidates:** (a) one partition per aspect, heads as retrieval conditions; (b) learned aspects; (c) PercepT under
  a fair protocol. Two ideas parked with evidence.

## 1. Sources

| Check | File | Who |
|---|---|---|
| PercepT literature (deep) | [2026-12-05_percept_neighbourhood](../../literature/2026-12-05_percept_neighbourhood.md) | Opus deep-research; no-DEC rows checked by the controller |
| Conditional similarity update (standard) | [2026-12-04_conditional_similarity_update](../../literature/2026-12-04_conditional_similarity_update.md) | Sonnet deep-research |
| Our PercepT code against the paper | [2026-12-04_percept_code_audit](2026-12-04_percept_code_audit.md) | Codex; 5 claims checked by the controller |
| Premise checks on cached features | `src/test/20261017_aspect_premise_checks/` (log, `results.json`) | Sonnet worker; probe and GoEmotions numbers checked by the controller |

## 2. Premise checks (ArtELingo train side, about 100k captions, 19.9k paintings; k-means mean of 3 seeds, sd ≤ 0.01)

**Within-painting caption variation does not isolate affect.** Removing the painting's mean caption removes content
(genre AMI 0.224 → 0.023 to 0.057) but also removes emotion:

| Feature (CLIP B/32) | Caption-emotion AMI, k = 9 / 20 / 40 | Emotion probe accuracy (chance 28.4%) |
|---|---|---|
| raw caption text | 0.037 / 0.047 / 0.052 | 57.3% |
| caption minus painting mean | 0.017 / 0.024 / 0.031 | 46.6% |
| caption minus projection on the image | 0.037 / 0.040 / 0.048 | 56.6% |
| GoEmotions classifier output (28-d, supervised elsewhere) | 0.197 / 0.209 / 0.199 | not run |

**Views give different partitions.** Painting level, k = 20; the same view re-run with another seed gives 0.54 to 0.63.

| View | Emotion | Genre | Style |
|---|---|---|---|
| CLIP image | 0.067 | 0.421 | 0.302 |
| mean CLIP caption text | 0.074 | 0.330 | 0.115 |
| CSD style | 0.047 | 0.249 | 0.315 |
| VGG Gram | 0.034 | 0.149 | 0.123 |

Pairwise AMI between the views' partitions: 0.11 to 0.36 (image/CSD 0.36, image/text 0.31, text/VGG 0.11).

**ArtELingo-28 is small.** The read set has 746 paintings, 27 languages (no English), about 4.3 captions per painting
per language. Language groups disagree on emotion slightly more than random groups of the same size (Jensen-Shannon
+0.052, se 0.001; higher on 93% of paintings).

## 3. Candidate ideas

### (a) One partition per aspect, heads as retrieval conditions

- **Idea.** PercepT fuses content and affect into one partition, so it cannot answer "similar in feeling" apart from
  "similar in subject". We build one buddy teacher graph per aspect view (content: CLIP image and caption; affect: an
  off-the-shelf caption affect encoder, as PercepT uses; style: CSD), and train one student with a shared trunk and
  one head per aspect. Each head gives that aspect's P-topics, an image-only mapping (PercepT's Stage 2, per aspect),
  and a similarity. Choosing a head is the condition: GeneCIS's "focus on X", with X an aspect.
- **Claim.** Per-aspect topics agree with each aspect's labels better than PercepT's single fused partition at matched
  topic count, and they support aspect-conditioned retrieval, which PercepT cannot do.
- **Support.** Views give different partitions, each following a different label (section 2).
- **Cost.** About 5 to 7 days to a first result, the faithful PercepT baseline included.
- **Risk.** The obvious review: "you cluster each encoder separately". The aspects are chosen by us, not discovered.

### (b) Learned aspects: discover the aspects too

- **Idea.** From the image-caption buddy graph (or several raw views), train K student heads so that each head keeps
  part of the neighbour structure while the heads are pushed to have different neighbours. Name each head afterwards
  from the caption words of its topics. Heads act as retrieval conditions, as in (a).
- **Claim.** Label-free aspect discovery plus conditional retrieval: the intersection the literature check found empty.
- **Comparators.** X-Cluster and Multi-MaP (several clusterings); CRL and an instruction embedder (Qwen3-VL-Embedding)
  given aspect names from an LLM (retrieval).
- **Cost.** About 8 to 12 days; may not work by 2026-10-29.
- **Risk.** High. Unsupervised multiple clustering tends to find trivial axes, and affect is weak without a supervised
  encoder (0.05 vs 0.20, section 2).

### (c) PercepT under a fair protocol

- **Idea.** A faithful PercepT reimplementation against buddy-based P-topics (one partition), judged on what a
  reviewer would ask: per-label held-out AMI at matched topic count, Stage 2 scored against human labels, retrieval.
- **Claim.** Buddy P-topics beat PercepT under a fair protocol; PercepT's own tables show DEC adds little.
- **Cost.** About 3 to 5 days.
- **Risk.** Our 09-30 head-to-head tied (against a PercepT-like baseline); an incremental result; no conditional
  similarity, so GeneCIS drops out.

(a) and (b) can stack: (b) replaces (a)'s chosen views with learned heads, so (a) is (b)'s upper reference.

### Parked (go to the direction log's "Parked ideas" after the choice)

- **Culture as the condition** (ArtELingo-28): a real but small language effect (+0.052 Jensen-Shannon) on 746 paintings.
- **Within-painting caption variation as a label-free affect signal:** the premise failed (the residual loses emotion:
  probe 57.3% → 46.6%).

## 4. Questions for the grill (round 1, key, asked answer-first)

1. **Which idea (decision B):** (a), (b), (c), or (a) then (b).
2. **What "label-free" means for affect:** (i) nothing trained with emotion labels anywhere (CLIP only: caption-emotion
   AMI 0.05); (ii) no labels of the target dataset, encoders trained on other data allowed, as PercepT does (0.20);
   (iii) a loose use of target labels (for example a few labelled examples per aspect).
3. **The PercepT baseline:** (i) faithful, all eight audit changes (CLIP L/14 and ModernBERT features, about 1.5
   days); (ii) only the result-changing core (pairs, Eq. 2 fusion, Algorithm 1, multi-hot targets) on our B/32
   features (about 0.5 to 1 day); (iii) no reimplementation, cite the paper's numbers (not comparable: other split
   and encoders).

<details>
<summary>Our recommendation (open after the user's answers)</summary>

1. (a) then (b): (a) can show a result by 2026-10-29 and gives (b) its reference; (b) carries the novelty if it works.
2. (ii): it matches PercepT, so the comparison is fair, and it is "label-free" for the target data; (i) loses affect.
3. (i): PercepT has no public code, so a reviewer will check our reimplementation; the features cost hours on the GPU.

</details>

Minor questions for the batch after round 1 (with defaults): the second domain (Affection; it has no genre labels);
the backbone (CLIP L/14 to match PercepT, B/32 kept for continuity); the retrieval evaluation (ArtELingo
aspect-conditioned retrieval with emotion, genre and style labels as ground truth; COCO-4c mood as a second set;
GeneCIS itself optional); what counts as "something works" by 2026-10-29.
