# CoSiR v2: ground-up redesign — foundation decisions + open question

**Date:** 2026-09-28
**Status:** foundation decisions settled by user; architecture (§3) explicitly open, pending literature review + brainstorm
**Motivates from:**
- `docs/reports/literature/2026-09-28_architecture_rethink_literature_brainstorm.md` (tonight's earlier clean-slate candidate survey, Candidates A-E)
- `docs/reports/auto/percept/2026-09-26_artelingo_buddy_vs_percept_stage1.md` (the buddy-vs-PercepT Stage1/Stage2 investigation)
- Experiment 18's real result (`docs/reports/stage/2026-09-16_prototype_conditioning.md`, on `experiment/buddy_prototype_conditioning`): patching the existing free-vector + combiner architecture found a genuine, seed-replicated interpretability/retrieval trade-off, never resolved
- Tonight's live baseline runs on ArtELingo (`/project/CoSiR-exp18_artelingo`): confirmed the current architecture's conditioning is weak, asymmetric (i2t >> t2i under `combine_side="img"`), and that the condition-predictor's training-order coupling, while real, isn't the dominant bottleneck (verified via a controlled A/B — see session transcript, 2026-09-28)
- GeneCIS (Vaze, Carion, Misra, arXiv 2306.07969) — direct intellectual ancestor of this project's own `combiner.py` (its file header already credits GeneCIS's `combiner_model.py`)

## 1. Why a rewrite, not another patch

This project has run ~18 numbered experiments incrementally patching one core skeleton (frozen CLIP → free per-sample condition vector → combiner). Each patch (new combiner variant, `other_proj`, `ConditionPredictor`, `PrototypeBank`, various loss terms) added to `src/model/`, `src/hook/train_cosir.py`, and `src/eval/` without ever being able to remove what came before, since every new idea had to stay backward-compatible with the free-vector default. The result: `train_cosir.py` alone carries multiple conditioning modes, multiple combiner types, an EM-interval mechanism, oracle-guided advantage weighting, several regularizer terms, and a training loop over 1000+ lines mixing all of it. Tonight's real finding — that the existing architecture's own condition mechanism is weak, asymmetric, and doesn't obviously get better from patch-level fixes (see Experiment 18's regression, and tonight's predictor-timing A/B null result) — is the concrete evidence that further patching has diminishing returns.

**Decision (settled):** stop patching. Build CoSiR v2 as a fresh codebase, block by block, each block validated with real numbers before the next block is built on top of it — not written all at once and debugged as one tangled system.

## 2. What gets reused vs. rewritten

**Reused (as infrastructure, not modeling logic):**
- `FeatureManager` / the chunked feature-cache system (`src/utils/feature_manager.py` and friends) — pre-extraction and caching of frozen CLIP features is a solved, validated problem, not part of what needs rethinking.
- The dataloading/dataset-config pattern (Hydra `configs/dataset/*.yaml`, the `dataset_type` dispatch) — reused for its plumbing (paths, splits, feature-cache wiring), not its modeling assumptions.
- Raw feature extraction itself (frozen CLIP forward passes) — unchanged; the backbone stays frozen per every prior experiment's own constraint.

**Rewritten from scratch, block by block (not copied, not incrementally patched):**
- The model architecture itself (whatever replaces `CoSiRModel`/`combiner.py`/`condition_predictor.py`/`prototype_bank.py`).
- The loss functions (whatever replaces `LabelContrastiveLoss` and its many regularizer terms).
- The training loop (whatever replaces `train_cosir.py`).
- Evaluation code gets rebuilt too, but keeps the same *conceptual* metrics this project already validated matter (t2i/i2t recall against a raw/no-conditioning floor, interpretability/coherence probes) — the metrics are right, tonight's session just found real problems with how "oracle" and "predictor" specifically were computed and used; v2 should design its own eval from these lessons, not inherit the old implementation.

**Not yet decided:** exactly which of the old model/loss/training code, if any, is worth reading for lessons learned (e.g. `buddy_contrastive_loss` in `src/metrics/regularizer.py`, which multiple past experiments validated) vs. rewriting outright. Default assumption per the user's direction is "rewrite everything in this category" — any reuse is a deliberate exception requiring its own justification, not a default.

## 3. Block 1 (settled): the validated attention Stage 1, rebuilt fresh

**Decision (settled):** the first block of CoSiR v2 is a rebuilt version of the already-validated buddy-graph topic-formation mechanism — "Attention-h1": a two-teacher-graph (content + affect) symmetric InfoNCE-trained student with one-head self-attention fusion, LayerNorm + L2-normalized output, producing a topic/community embedding space over frozen CLIP features via the buddy graph (cross-modal mutual-kNN). This architecture is what "clears the held-out AMI Pareto bar" in the ArtELingo investigation (`docs/reports/auto/percept/2026-09-26_artelingo_buddy_vs_percept_stage1.md`, and the architecture-sweep commit `4b0e48d` on `experiment/percept_topic_pipeline`).

**Reference implementation** (for understanding the validated mechanism, not for copying): `src/test/20260923_artelingo_buddy_analysis/run_learned_student_arch_sweep_pilot.py` (`LearnedStudent("attn1")`), on branch `experiment/percept_topic_pipeline`. The currently-running buddy-percept sweep (`polysemic/CoSiR-buddy-percept-sweep/40i43gt5`, 55+ gate-passing trials as of tonight, Task 11 top-10 stress test still pending) is independently validating and tuning this same Stage 1 mechanism jointly with a Stage 2 classifier — its eventual winning hyperparameters are a direct input to this block once available, but Block 1's own construction does not need to wait for Task 11 to begin.

**Why start here:** it's the one piece of this whole investigation that's already empirically validated (competitive AMI against PercepT's own replication, well-populated/balanced held-out communities, no occupancy collapse) and it's the natural foundation the "open question" in §4 needs to build on regardless of how that question resolves — every candidate direction discussed tonight (Candidates A-E, the PercepT-style Stage2-assigner idea) assumes *some* topic-formation mechanism exists first.

## 4. Open question (NOT settled): how does Stage1(+Stage2) relate to CoSiR's actual target?

This is the question a literature review + brainstorm needs to resolve before Block 2 gets designed — explicitly not answered by this document.

**The tension to resolve:** the buddy-vs-PercepT investigation validated a Stage1 (topic formation) + Stage2 (attention-pooling classifier assigning a query to that topic space) pipeline, and tonight's finding is that buddy's Stage2, copied straight from PercepT's design, is currently a little better than PercepT's own (pending the sweep's full hyperparameter confirmation via Task 11). But **Stage1+Stage2 topic classification is not the same problem CoSiR was ever meant to solve.** From the start, this project's actual target has been closer to GeneCIS (arXiv 2306.07969): not "assign this sample to one of K discovered topics," but "adapt what counts as similar, conditionally" — a genuinely open-ended notion of conditional similarity, not classification into a fixed discovered vocabulary.

Two new points the user wants this redesign to establish as CoSiR's actual differentiators, relative to GeneCIS specifically:

1. **Self/unsupervised condition discovery.** GeneCIS's method still mines an explicit, structured taxonomy of condition types from parsed image-caption data (a designed category system: attribute, spatial relation, etc.), and its benchmark's conditions are drawn from that designed set. CoSiR should discover the condition space itself from unsupervised structure — no hand-designed category list, no structured/labeled dataset construction step.
2. **Genuine cross-modal relations.** GeneCIS's actual similarity computation is image→image (a caption-derived condition modifies an image query, but the retrieval target is still images — the CIR/composed-retrieval lineage `combiner.py` already comes from). CoSiR's target is real image↔text conditional similarity: the condition should relate two different modalities to each other, not modify a same-modality retrieval task.

**What needs research, not assumption:** whether/how Stage1(+Stage2)'s validated topic-formation-and-classification mechanism can be extended or recombined to serve these two goals — e.g., does a discovered topic space (Stage 1) function as a *self-supervised proxy* for GeneCIS-style condition discovery, with Stage 2 generalized from "classify into K topics" into something that expresses open-ended, cross-modal conditional similarity rather than fixed-K classification? Or does a fundamentally different mechanism (one of tonight's Candidates A-E, or something not yet surfaced) fit better? This is genuinely undecided and is the actual subject of the commissioned literature review.

## §4 resolved: Block 2 is Candidate A (shared factors + support-conditioned score), Candidate B held in reserve

The commissioned review (`docs/reports/auto/v2/2026-09-28_stage1_genecis_synthesis_brainstorm.md`) resolved §4 above. **Settled, after discussion with the user:** build **Candidate A** first — build it as the primary Block 2+ line; **Candidate B** (nonparametric graph-local metric) is not built now, held in reserve as a smaller ablation once Candidate A has first results, specifically to answer "is the factor layer doing real work, or is Stage 1's graph carrying everything." This document now authorizes Block 1 and Block 2 (Candidate A) implementation, task by task, per the plan this spec motivates.

**Two open sub-decisions the review surfaced, both resolved:**
- *Affect-teacher supervision*: Attention-h1's affect graph uses a GoEmotions-**supervised**-pretrained encoder, not a self-supervised one — a real asterisk on the "self/unsupervised" differentiator. Resolution: Block 1 tests **content-only buddy guidance first** (the strictest, genuinely unsupervised variant), and a content+self-supervised-pretrained-affect variant as a second arm — not a straight carry-over of the GoEmotions-supervised teacher. Whether either affect arm captures anything useful without supervision is an empirical question, not an assumption.
- *Episode-mining vs. the "no structured dataset" constraint*: training `s(I,T|c)` needs some source of condition variation; plain paired InfoNCE cannot teach ranking reversal under a changed condition. Resolution: self-mined temporary episodes (support/contrast/query tuples sampled fresh from graph + factor structure, via a frozen/lagging snapshot to avoid circularity) are **within** the constraint — they are automatic training-task construction from the data CoSiR already has, not a stored, hand-labeled, or externally-sourced condition corpus. This boundary is deliberately narrow: importing any external structured condition taxonomy (GeneCIS's scene-graph-parsed triplets, a hand-built attribute list, etc.) would violate it; mining episodes from CoSiR's own graph/factor structure does not.

### Candidate A architecture (settled design, implementation detail deferred to the plan)

**Target task:** `s(I,T|c)` — score an image `I` and text `T` under an independently-supplied condition `c`. For i2t, rank candidate `T`s for fixed `(I,c)`; for t2i, rank candidate `I`s for fixed `(T,c)`. This is **cross-item aspect matching** (e.g., under "shared color," a red-car image should rank a red-bicycle caption above a blue-car caption) — not "retrieve this image's own paired caption," and not K-way topic classification.

1. **Factor discovery** (block-testable alone, before any condition mechanism exists): two small encoders `a_I`, `a_T` map frozen CLIP image/text features into a shared sparse factor space `R_+^L`. Trained via sparse reconstruction of the frozen CLIP features, paired-activation agreement (genuine pairs should share factor activations), and Stage 1's graph neighborhoods as additional structure — plus a group-sparse/CCA control against the known failure mode where a nominally-shared dictionary splits into disjoint image-only and text-only halves.
2. **Condition interface**: a handful of support pairs (optionally with contrast pairs) get encoded; a small set-encoder reads off which factors are consistently active in positives but not contrasts, producing `w(c) ∈ R_+^L`. A free-text condition path (`w(c) = a_T(condition phrase)`, GeneCIS-style) is a separate, later, explicitly-unvalidated extension — not part of the first build.
3. **Scoring**: `s(I,T|c) = β·cos(CLIP_I(I), CLIP_T(T)) + Σ_l w_l(c)·a_I,l(I)·a_T,l(T)` — symmetric, usable for both retrieval directions.
4. **Training**: self-mined condition-swap episodes (§ above) — support set + query + candidate pool (real positive, hard negatives matched on other axes but differing on the targeted factor, condition-only distractors, anchor-only distractors) — trained with a bidirectional conditional ranking/InfoNCE-style loss, plus an explicit swap term requiring the ranking to correctly reverse when `c` is replaced by a different valid `c′` for the same anchor and pool. Mining runs off a frozen/lagging snapshot, refreshed on a slower schedule than the scorer trains, to bound (not eliminate) circularity.

**Build order within Candidate A** (each stage validated before the next, per the user's stated working style): (a) Block 1 — rebuild Attention-h1 Stage 1 fresh, content-only first; (b) factor discovery on top of Block 1's frozen output, validated on its own (do factors reconstruct, do paired items agree, does the group-sparse control actually prevent a split dictionary) before any condition mechanism is built; (c) the condition interface and episode-mining pipeline; (d) the scoring function and swap-loss training; (e) the new cross-item, human-judged evaluation set (does not exist yet — a real deliverable, not reused from any prior CoSiR eval).

**Global engineering constraints for the rebuild** (per the user, 2026-09-28): everything is written function/class-formal — proper typed modules, not ad hoc scripts. Configs start minimal and are added to per block, not carried over from the old bloated loss-weight-laden config files (removed in the foundation-stripping commit). Implementation is dispatched to Codex per task (controller = Claude, reviewing each diff; implementer = Codex, direct `codex e --dangerously-bypass-approvals-and-sandbox` invocation, not the broken `codeagent-wrapper`); execution proceeds automatically task to task without pausing for check-ins, except when Codex hits a usage/context limit or gets stuck — that stops for the user's decision.

## Amendment 2026-09-30: external affect signal allowed (user decision)

Following PercepT, the GoEmotions RoBERTa (`SamLowe/roberta-base-go_emotions`, fine-tuned on Reddit comments with 28
emotion categories, never trained on ArtELingo) may be used as a **training signal** for Candidate A, in the form of
its 28 sigmoid probabilities per caption. This relaxes the "no external structured condition taxonomy" boundary above
for this one model only. ArtELingo's emotion and style labels remain evaluation-only, no other external model or
taxonomy is added, and the retrieval model at test time still reads only frozen CLIP features. Motivation: the
factor-learning 2×2 (`docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md`) found no label-free
route to emotion. Design: `docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-affect-factor-learning-design.md`.

## Next step

Task-by-task implementation plan for Block 1 + Candidate A, per `superpowers:writing-plans` conventions, execution via Codex-as-implementer with Claude as controller/reviewer (`superpowers:subagent-driven-development`'s discipline — fresh dispatch per task, review, fix loop, ledger — adapted for a Codex implementer instead of a Claude subagent).
