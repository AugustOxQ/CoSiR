# Research Proposal: Conditional Buddies — When Does Graph-Based Neighbor Structure Help Contrastive Image–Text Retrieval?

**Date:** 2026-08-04
**Project:** CoSiR — conditional buddy initialization and training
**Full technical plan:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
**Prepared for:** publication decision / advisor–collaborator review

---

## Abstract

CoSiR fuses frozen CLIP image/text features with a per-sample trainable "condition" vector that must start somewhere. We built and validated a graph-based initialization — a "conditional buddy" is a pair of samples that are mutual nearest neighbors in *both* CLIP image space and CLIP text space — and asked three questions over two months of work: (1) is the buddy signal real, or an artifact of CLIP / near-duplicate data? (2) does using it during training, beyond initialization, help retrieval? (3) does it survive changing the vision-language model entirely? The answers so far: the signal is real and remarkably robust — it survives three independent validation probes, six held-out encoders never used to build the graph, and a full graph-rebuild across sixteen vision×text encoder combinations. But of three natural ways to *use* the signal during training, two are statistically null and the third's apparent win is substantially explained by a measured near-duplicate confound in one of our two datasets, not by transferable cross-modal signal. What remains unanswered — and is the critical gap before this is submittable — is whether the validated signal actually improves retrieval at the one place it was designed for: initialization itself, and how this positions against existing neighbor/graph-based contrastive learning methods, which no report currently addresses. We propose a focused plan (a fast prior-art check plus four experiments) to close those gaps, targeting **TMLR** as the primary venue (with a workshop submission as a scheduling hedge, and a main-track conference as a contingent, lower-probability stretch), submittable within 1–2 months.

---

## 1. Motivation

Per-sample trainable embeddings need an initial value, and the standard choice — a generic linear/PCA transform of the frozen backbone features — encodes no notion of which samples are related. The hypothesis behind this project: samples that share cross-modal content should start close together in the space the model will optimize, and a cross-modal mutual-nearest-neighbor graph is a natural, cheap way to define "related" without any extra supervision.

This is a useful idea independent of any one architecture — condition/label embeddings, adapters, and per-sample learnable tokens are increasingly common in retrieval and personalization systems, and all of them face the same "where do I start" problem. A rigorous answer to *when a data-driven graph signal is worth using, and when it silently rides on dataset artifacts instead*, is broader than CoSiR specifically.

## 2. What we already know (validated, not proposed work)

Over roughly two months (`docs/reports/2026-06-{09,22,23,24}_*.md`, `docs/reports/2026-07-{08,16}_*.md`) we established, with independent evidence on two datasets:

- **The signal is real and content-specific**, not a near-duplicate or type-matching artifact — confirmed by three independent probes (category/subreddit lift, a held-out DINOv2 encoder, and a VLM-judge caption-match test).
- **The signal is not a CLIP, single-encoder, or single-modality artifact** — it survives being *scored* by six held-out encoders (three vision paradigms, three text embedders) it never built, on both graphs, on both datasets: 24/24 cells confirm buddies sit closer together than random pairs.
- **The signal survives having the graph itself rebuilt** by sixteen different vision×text encoder combinations — about one buddy pair in five recurs exactly between arbitrary encoder choices, at 10³–10⁵× the chance rate, with a stable, semantically coherent consensus core.
- **Using the signal during training is mostly a null result, and the one exception has a diagnosed cause.** Of three mechanisms — a smoothness regularizer, contrastive supervision, and a self-refreshing graph — two show no reliable effect after seed replication. The third (contrastive supervision) shows a real, dose-dependent win on one dataset (Impressions) but *not* on a second, cleaner dataset (RedCaps) — and we traced this directly to a confound: 40.6% of the edges the training term optimizes on Impressions connect two records of the literal same source photo, a 279× enrichment over chance.

This combination — a signal this thoroughly stress-tested, paired with an honestly diagnosed negative result on the training-time question — is unusual and is the project's core strength. Most papers report one or the other; having both, with the causal thread connecting them, is a stronger empirical story than a single positive number.

## 3. The gap

No experiment currently compares buddy-graph initialization against the prior generic initialization on final retrieval quality, with training-time terms held off. Every existing ablation measures a delta *from* buddy-initialized training, not a delta *to* the initialization choice itself — even though the codebase's default configuration already assumes buddy initialization is the better choice. This is the paper's foundational claim, and it is currently unmeasured. We also have no baseline comparison against standard CLIP fine-tuning, which any reviewer will ask for first.

## 4. Proposed work

Five items close these gaps, in priority order (full detail, exact tooling, and success criteria in the linked spec):

| # | Item | Answers | Priority |
|---|---|---|---|
| 0 | Related-work / prior-art grounding (NNCLR, mean-shift/prototype SSL, graph-Laplacian init precedent) — reading and writing only, no compute | Does this differentiate cleanly from existing neighbor/graph-based SSL methods, or is there a fatal overlap? | **Critical — do first, in parallel with Exp. 1** |
| 1 | Buddy-init vs. generic-init, no training-time terms, 3 seeds × 2–3 datasets | Does the core idea work at all? | **Critical — do first** |
| 2 | Retuned, gentler contrastive-supervision dose on clean (RedCaps) data | Does any dose of the one promising training mechanism survive off near-duplicates? | High |
| 3 | The "B-lean" initialization variant, validated on clean data | Is the "cleaner buddy signal" finding itself a near-duplicate artifact, like the training-term result was? | High |
| 4 | Standard CLIP fine-tuning baselines, run and reported | Where does this sit relative to the obvious alternative? | Required for any venue |

Item 0 is new relative to the original plan: no report or spec in this project currently cites or differentiates against the existing neighbor/graph-based contrastive learning literature (NNCLR's nearest-neighbor positives, mean-shift/prototype SSL, graph-Laplacian embedding init in recsys/node2vec). Any reviewer will map this work onto that lineage immediately. It's answerable — the confound-diagnosis finding on Family #2 is a legitimate point of differentiation from NNCLR-class claims — but the positioning needs to happen in week 1, not during writing, so a fatal overlap (if one exists) is found before the experiment budget is spent.

Two stretch items — extending validation to MS-COCO, and testing whether cross-encoder signal *survival* predicts downstream *usefulness* — are proposed as contingent, schedule-permitting additions, not commitments.

Design principle: **the paper's framing does not depend on Experiment 1's outcome.** If it's positive, the paper gains a genuine headline result and a stronger venue becomes realistic. If it's null, the paper still stands as a rigorous, causally-grounded account of when a validated data signal does and doesn't translate into model improvement — itself a useful and citable finding, not a failure to publish around.

## 5. Target venue

| Choice | Rationale |
|---|---|
| **Primary: TMLR** | Reviewed on soundness and community value rather than novelty-or-SOTA pressure — the best match for a confound-controlled analysis paper with a mix of positive, null, and negative findings. Rolling submission removes deadline risk from a timeline that depends on experiment outcomes. **Not unconditional:** clean if Experiment 1 is positive; if it's null, TMLR is still reachable but requires explicitly reframing as a methodology/cautionary-tale paper, and is also contingent on Experiment 0 not surfacing a fatal prior-art overlap. |
| **Hedge: a CVPR/ICCV/NeurIPS workshop** (multimodal representation learning / data-centric AI) | Submitted in parallel at reduced length. Fast turnaround gives a guaranteed in-window publication even if TMLR's review process runs long. |
| **Stretch: ICLR/NeurIPS main track** | Only pursued if Experiment 1 lands a clean, seed-replicated positive result — a genuine "this initialization measurably helps" headline is what main-track reviewers need to see past the mostly-null training-time story. Decided at the Week-3 checkpoint. |
| Not pursued: CVPR/ICCV/ECCV main track, ACL/EMNLP | Wrong audience fit / wrong evaluation criteria for this paper's actual strengths (see spec §3.2 for the full reasoning). |

## 6. Timeline

Eight weeks, front-loaded on the highest-information items:

- **Week 1:** Experiment 0 (related-work/prior-art grounding, reading and writing only) runs in parallel with the start of Experiment 1 — cheap, and resolved before deeper compute commitment.
- **Weeks 1–2:** Experiments 1–4 run in parallel (independent, share existing infrastructure).
- **Week 3:** Analyze results, including Experiment 0's note; decide stretch-tier ambition, whether the framing needs adjustment, and whether COCO extension is worth pursuing.
- **Weeks 4–5:** COCO (if warranted) and the cheap causal-tightening follow-up on the near-duplicate confound.
- **Week 6:** Optional cross-encoder-survival-vs-usefulness stretch study, only if ahead of schedule.
- **Weeks 6–8:** Writing, related-work section (drafted from Experiment 0's note), baseline tables, figures, internal review, and preparing both the TMLR and workshop submission packages.

## 7. Risks

The main risk is Experiment 1 returning null or negative — mitigated by design, since the paper's framing was chosen specifically to remain publishable either way (§4), though a null result does narrow TMLR to the methodology-paper framing rather than leaving it a clean submit. A second risk, not previously flagged, is prior-art overlap with existing neighbor/graph-based SSL methods (NNCLR, mean-shift/prototype SSL) — mitigated by running Experiment 0 in week 1, before further compute or writing investment, rather than discovering it during drafting in weeks 6–8. The secondary risk is TMLR's review timeline extending past the target window — mitigated by the parallel workshop submission. Compute risk is managed by strict prioritization: the cheapest, most load-bearing experiment runs first, and every stretch item is explicitly gated on remaining schedule rather than committed up front.

## 8. Expected outcome

At minimum: a methodologically rigorous paper establishing that a specific, easy-to-compute graph signal in multimodal contrastive data is robust across encoders and datasets, together with a causally-grounded account of why one plausible way of exploiting it during training doesn't generalize — a result with value beyond this specific architecture, as a cautionary methodology for anyone using auxiliary graph-based supervision on data with near-duplicate structure. At best (Experiment 1 positive, COCO extension completed): the same paper plus a genuine, validated initialization-quality improvement, opening the door to a stronger venue.

---

*See `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` for exact experiment configurations, tooling references, statistical methodology standards, and the full risk/decision framework.*
