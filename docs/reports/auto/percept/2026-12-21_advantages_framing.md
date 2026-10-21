# Our advantages and how to show them: framing note

> Date: 2026-10-21 (sequence name 12-21) · Kind: framing note (design judgment, no experiment) · Loop: r10, item 4 · Written by: Opus xhigh worker in the r10 run 12 chat · Feeds: the r10 decide 17 chat

AMI (adjusted mutual information between clusters and labels) has 4 decimals; a few development numbers keep their
report's 3. "Held" means the 12,281 held ArtELingo paintings, "dev" the 5,000 development paintings. Sources by
sequence name: [12-08](2026-12-08_prompted_aspect_views_check.md), [12-14](2026-12-14_percept_core_arms.md),
[12-15](2026-12-15_percept_core_k40.md), [12-16](2026-12-16_model_sheet.md), [12-19](2026-12-19_claim_test_stage2.md),
literature [10-21](../../literature/2026-10-21_cvpr_literature_review.md), [12-05](../../literature/2026-12-05_percept_neighbourhood.md), [12-07](../../literature/2026-12-07_prompted_aspect_views.md).

## Summary

1. We can claim one thing now: held emotion AMI 0.1008 against 0.0874 for the strongest uncollapsed PercepT, +0.0133 [+0.0081, +0.0146] (12-19).
2. Measured, not licensed: one read of the default image gives three aspect spaces, all above PercepT core. Emotion is also above the instructed Qwen image (+0.0087, no interval).
3. "Label-free" is not PercepT's edge: its published emotion encoder is tuned on GoEmotions labels. Its real edge: it needs no aspect list.
4. "Cluster quality" is ours under k-means only. Under Leiden the held emotion lead is −0.0005 [−0.0048, +0.0023].
5. "Lightweight" is mixed: about 6 times less training time per seed than PercepT core, but 2.3 times its trainable parameters and the same Qwen pass at test time.
6. "Any dataset" and "any backbone" have no evidence yet. Tonight's no-instruction row and the item 5 code decide whether 11-10 can show them.
7. Our reading: lead with "one read, all aspects, no labels". Add "any backbone" if a SigLIP 2 or CLIP student stays above PercepT core; "lightweight" then follows.

## 1. Candidate advantages

Two advantages are measured, PercepT's two need rewording, and the rest need new evidence. "Our reading" is our
judgment; "What we have now" holds measured facts only.

| Advantage | Our reading | What we have now (measured) | What is missing | Experiment or analysis | Cost (local 3090) | Risk: what could make it false |
|---|---|---|---|---|---|---|
| Any dataset through Qwen prompting (user) | needs evidence | Instructions lift Qwen views on ArtELingo: caption emotion 0.278 vs 0.122 default; image style 0.395 vs 0.222 (12-08) | a second labelled dataset | same recipe, new instructions only: ArtELingo-28 (1,658 paintings; its current store holds the 746 outside our train and validation paintings), then another domain (SemArt stores exist) | views: about 4 GPU min for the 746 ArtELingo-28 paintings, 1.7 GPU h for SemArt (3 views × 0.094 s per image, derived) | PercepT core takes prompts too (its Qwen arm uses the emotion instruction); ArtELingo-28 shares paintings with ArtELingo: 790 of its 1,658 are in our train paintings, 248 in our held paintings (overlap facts note, below); the recipe was tuned on ArtELingo dev; instructions move embedders unevenly (12-07) |
| Any backbone (user, "to prove") | needs evidence | none for our model; SigLIP 2 and CLIP B/32 stores exist; CLIP images clustered directly beat default Qwen on dev style, 0.300 vs 0.222 (12-15) | a student on another backbone | D2 recipe with SigLIP 2 or CLIP input, Qwen teacher and graphs kept; dev, 3 seeds; then held under its own rule | item 5 code, then under 10 GPU min (D2 seed 53 to 55 s on dev, claim test run log) | the no-instruction row near PercepT core: only "any student backbone, Qwen teacher at training"; a weaker input may fall below PercepT core |
| Lightweight (user, "not sure") | mixed (§2) | §2 | published PercepT costs; one timing session | one-session cost table | under 1 GPU h (guess) | the Qwen pass dominates test cost; more parameters than PercepT core |
| Label-free training (PercepT's) | level | ours: labels only score, label-free early stopping (12-19); published PercepT: ModernBERT tuned on GoEmotions (12-05); PercepT core's Qwen arm uses the Qwen emotion instruction instead | nothing to run | state it in the method and related work | 0 | reviewers count aspect names and Qwen's own training as supervision |
| Cluster quality (PercepT's) | ours under k-means; tie under Leiden | held margins over PercepT core: emotion +0.0133 [+0.0081, +0.0146], style +0.1388 [+0.1283, +0.1427], genre +0.0208 [+0.0103, +0.0310]; Leiden emotion −0.0005 [−0.0048, +0.0023]; k-means at Leiden's count +0.0190 (12-19) | why Leiden lowers our model (0.1008 to 0.0757) more than core (0.0799 to 0.0762) | Leiden diagnostics on dev graphs | CPU, a few hours (guess) | reviewers prefer count-free clustering; PercepT's silhouette 0.97 is in its own space (12-05) |
| One read gives all aspects | measured | three aspect spaces from one default pass, all above PercepT core; the instructed image needs a pass per aspect | intervals against the instructed image | the instructed bar test | next row | the instructed image wins on style and genre |
| Above the instructed Qwen image on emotion | measured, no interval | held 0.1008 vs 0.0921 (+0.0087); dev 0.0962 vs 0.0929 (12-19, 12-16) | a rule and a paired interval (11-10 gate) | pre-registered held comparison on stored arrays | under 1 CPU h (12-19: verdict 126 s, diagnostics 1,870 s) | style −0.0340 and genre −0.0623 below it (12-19): "just prompt Qwen" |
| Stable across seeds, no topic-count collapse | measured, on our reimplementation | held seed sd 0.0011 vs 0.0018 for core (12-19); strict paper settings: all 10 runs collapsed in each stage; Algorithm 1 kept 1, 2, 3 or 99 topics by seed (12-14) | PercepT with its own encoders | faithful reimplementation, checked against the paper's numbers | several GPU h (guess) | a faithful run may not collapse |
| Retrieval by aspect | not measured | caption heads exist (dev caption emotion AMI 0.1828, 12-16); no affective image–text retrieval benchmark found (10-21) | protocol, comparator, numbers | 11-10 gate: spec, then run | 1 to 2 days of agent work, under 1 GPU h (guess) | the instructed Qwen image may retrieve better, as it clusters better on style and genre |
| No LLM or caption at test time | shared with PercepT | both read only the image | — | state it against caption-based pipelines (12-07) | 0 | no edge over PercepT |

## 2. Lightweight: the measured facts

Our model is light to train and to read out, but it has more parameters than PercepT core. At test time the backbone
pass dominates: our network takes 32 µs per painting on CPU, one Qwen pass 0.0871 s on the GPU.

| Cost item | Our model (D2) | PercepT core, Qwen arm (P3, the bar) | Published PercepT | Instructed Qwen image (target) |
|---|---|---|---|---|
| Trainable parameters | 16,791,552 (both sides); 8,395,776 image side | 7,220,152 (autoencoder and 100 centres 5,080,976; mapper 2,139,176) | not checked | 0 |
| Network pass per painting | 8,388,608 multiply-adds; 32 µs on CPU, 8 threads | 2,138,112 multiply-adds | not checked | none |
| Backbone at test time | 1 default Qwen pass, 0.0871 s per image | 1 default Qwen pass, 0.0871 s | 1 CLIP ViT-L/14 pass, not checked | 3 instructed Qwen passes, 0.0936 s each |
| Training per model seed, 42,969 paintings | about 120 s (598 s for 5 seeds, 3 aspects) | about 712 s for its four arms together (3,558 s for 5 seeds) | paper settings: about 750 s per Stage 1 run | none |
| Extra training inputs | 3 instructed image views (about 0.094 s per image each), instructed caption views, graphs 197 s CPU | the instructed emotion caption view | CLIP ViT-L/14 image and text; the GoEmotions-tuned encoder | — |

Sources: counts from `build_model(ROWS['D2'])` in `src/test/20261019_model_sheet/ms_train.py` and the layer sizes in
`src/test/20261018_percept_core/` (`pc_sept.py`, `pc_stage2.py`); 32 µs is the median of 5 CPU timings on 12,281
inputs. Pass times: the stores' progress files and the claim test run log, which also gives training times. SigLIP 2
takes 0.0134 s and CLIP B/32 0.0044 s per image. Rates come from different runs and include image loading.

Our reading: "lightweight" holds against the instructed Qwen image (3.2 times fewer backbone seconds) and for
training time against PercepT core. It fails for parameters against PercepT core. A SigLIP 2 student would read an
image 6.5 times faster than one Qwen pass, 21 times faster than three instructed passes. So "lightweight" follows
from "any backbone"; alone it is weak.

## 3. Which ablation rows bear on which advantage

The no-instruction row and the two instruction controls decide the "any backbone" story. The ablation runs on dev
data with 3 seeds, so it can steer the framing but licenses no held claim (spec §6). Its results come later tonight.

| Row | Change from D2 | Bears on | Our reading of the possible results |
|---|---|---|---|
| R11 | no instruction anywhere | any backbone | near PercepT core: the instructed teacher is needed at training; well above it: a teacher-free story is possible |
| R3 | teacher = default Qwen image | any backbone; emotion lead | close to D2: the teacher need not take instructions |
| R9 | one default-view graph | any dataset; any backbone | close to D2: the graphs need no instruction |
| R2 | no teacher | emotion lead over the instructed image | a lead without distillation: the graphs add what the image teacher lacks |
| R4 | distillation only | "it is just distillation" | far below D2 (model sheet, weight 1.0: 0.0825 vs 0.0962, 1 seed): the objection fails |
| R7 | image endpoints only | any dataset (image-only collections); lightweight | close to D2: captions are not needed at training |
| R5, R6, R8, R10 | buddy, own-caption, content-correction or graph-structure part removed | any dataset | which parts a new dataset needs; R8 close to D2: a simpler recipe |
| R1 | separate heads | one read; lightweight | separate heads have 25,184,256 parameters against 16,791,552 (counted from code) |
| R12 | instructed inputs, three passes | one read | far above D2: the input limits style and genre, and one read costs accuracy |

## 4. Candidate storylines

**A. One read, all aspects, no labels (recommended main line, our reading).**
- Claim: a small network on a frozen embedder, trained without labels, turns one image read into emotion, style and
  genre spaces that beat PercepT on all three.
- Carried by: the claim test (have), the style and genre margins (have), the instructed bar with intervals and
  retrieval by aspect (11-10), the ablation.
- Strongest objection: "prompt Qwen once per aspect; it beats you on style and genre." Our answer: one pass instead of
  three, and a lead on emotion. The gap: on style and genre our only answer today is cost.

**B. Instructions as the only supervision, distilled into any backbone.**
- Claim: aspect instructions to one embedder, used only at training, teach aspect heads on any image backbone.
- Carried by: SigLIP 2 and CLIP students (item 5), the ablation's three instruction rows, a second dataset, the cost
  table.
- Strongest objection: "this is distillation; the student is bounded by its teacher." Our answer: our model passes its
  teacher on emotion (+0.0087 held), and the graph and caption terms carry most of the result (12-16). The gap: one
  dataset and one backbone so far, and the instruction needs the Qwen teacher.

**C. A fair protocol for perceptual topics.**
- Claim: under held, per-label scoring, PercepT's topics add little over its encoder and its paper settings collapse;
  aspect heads fix both.
- Carried by: the held protocol, the collapse counts, PercepT core against its encoder clustered directly (12-15).
- Strongest objection: "you attack your own reimplementation of an uncited preprint." Our answer needs the faithful
  reimplementation with PercepT's own encoders, not built yet. If it reproduces the paper, this story fails.

Our reading: A is the paper. B becomes its second contribution if a SigLIP 2 or CLIP student stays above PercepT core
on held emotion. C belongs in the evaluation section. Under Leiden, A's emotion lead is gone, so the claim must name
its clustering method.

## 5. What the paper can claim today, and what comes next

The licensed sentence (12-19, stage 2 GO): "On 12,281 held ArtELingo paintings, a model trained on the 42,969 train
paintings and read from the default Qwen image gives emotion clusters (k-means, 20 clusters) with AMI at least 0.01
above the strongest uncollapsed PercepT reimplementation trained on the same paintings, each with its own recipe on
our feature stores (point estimate over 5 seeds; paired 95% interval of the difference, over paintings and seeds,
above 0)." It covers no style or genre, no other dataset or backbone, no instructed bar and no retrieval.

Next experiments, in our order of value per hour; the abstract is due 2026-11-10, in 20 days.

| Order | Experiment | What it adds | Cost (local 3090) | Before 11-10? |
|---|---|---|---|---|
| 1 | Read tonight's ablation | does "any backbone" need the Qwen teacher; are captions needed | 0 extra | yes |
| 2 | SigLIP 2 and CLIP students, dev, 3 seeds | first evidence for "any backbone" and "lightweight" | item 5 code, then under 10 GPU min | yes |
| 3 | Instructed Qwen bar on held, with a rule (gate item 1) | an interval for the emotion lead | under 1 CPU h | yes |
| 4 | One-session cost table | §2's pass times from one session | under 1 GPU h (guess) | yes |
| 5 | Aspect-named retrieval (gate item 2) | a second task that PercepT lacks | 1 to 2 days of agent work (guess) | yes, tight |
| 6 | Held claim for the best other backbone | "any backbone" on held data | about 12 GPU min and 33 CPU min (12-19 timings) | if order 2 works |
| 7 | ArtELingo-28 | "any dataset", weak form (same paintings, translated captions) | about 4 GPU min of views, then training | yes |
| 8 | A second domain (SemArt or another) | "any dataset", strong form | about 1.7 GPU h of views; label audit first | maybe |
| 9 | Faithful PercepT with its own encoders | answers storyline C's objection | several GPU h (guess) | before submission |

## 6. Not checked

- Published PercepT's parameter count and test-time cost (its CLIP ViT-L/14 pass).
- Qwen3-VL-Embedding-2B's parameter count (only its name says 2B).
- Our network's GPU time; D2's GPU memory; PercepT core's time per arm; whether §2's rates share one GPU and load.
- How SigLIP 2 or CLIP images cluster on emotion and genre under the new line's protocol.
- Whether aspect instructions lift Qwen views on any dataset other than ArtELingo.
- SemArt's and Impressions' label fields; no Affection store was found under `/data/SSD2/pre_extract/`.
- Checked after this note was drafted (overlap facts note, `.scratch/r10-run-12/a28/a28_overlap_facts.md`, controller
  recount agrees): ArtELingo-28 has 1,658 paintings; 790 are in our train paintings, 122 in validation, 248 in held,
  498 outside our ArtELingo store. The 746-painting store holds the 248 held and the 498 outside ones.
- Whether artists are shared between train and held paintings (12-19).
- Any interval for our model against the instructed Qwen image; any retrieval number.
- How interpretable our clusters are (PercepT has a human study).
