# CoSiR v2: ground-up redesign — foundation decisions + open question

**Date:** 2026-09-28
**Status:** foundation decisions settled by user; architecture (§3) explicitly open, pending literature review + brainstorm
**Motivates from:**
- `docs/reports/2026-09-28_cosir_architecture_rethink_literature_brainstorm.md` (tonight's earlier clean-slate candidate survey, Candidates A-E)
- `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md` (the buddy-vs-PercepT Stage1/Stage2 investigation)
- Experiment 18's real result (`docs/reports/2026-09-16_stage_report_prototype_conditioning.md`, on `experiment/buddy_prototype_conditioning`): patching the existing free-vector + combiner architecture found a genuine, seed-replicated interpretability/retrieval trade-off, never resolved
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

**Decision (settled):** the first block of CoSiR v2 is a rebuilt version of the already-validated buddy-graph topic-formation mechanism — "Attention-h1": a two-teacher-graph (content + affect) symmetric InfoNCE-trained student with one-head self-attention fusion, LayerNorm + L2-normalized output, producing a topic/community embedding space over frozen CLIP features via the buddy graph (cross-modal mutual-kNN). This architecture is what "clears the held-out AMI Pareto bar" in the ArtELingo investigation (`docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`, and the architecture-sweep commit `4b0e48d` on `experiment/percept_topic_pipeline`).

**Reference implementation** (for understanding the validated mechanism, not for copying): `src/test/20260923_artelingo_buddy_analysis/run_learned_student_arch_sweep_pilot.py` (`LearnedStudent("attn1")`), on branch `experiment/percept_topic_pipeline`. The currently-running buddy-percept sweep (`polysemic/CoSiR-buddy-percept-sweep/40i43gt5`, 55+ gate-passing trials as of tonight, Task 11 top-10 stress test still pending) is independently validating and tuning this same Stage 1 mechanism jointly with a Stage 2 classifier — its eventual winning hyperparameters are a direct input to this block once available, but Block 1's own construction does not need to wait for Task 11 to begin.

**Why start here:** it's the one piece of this whole investigation that's already empirically validated (competitive AMI against PercepT's own replication, well-populated/balanced held-out communities, no occupancy collapse) and it's the natural foundation the "open question" in §4 needs to build on regardless of how that question resolves — every candidate direction discussed tonight (Candidates A-E, the PercepT-style Stage2-assigner idea) assumes *some* topic-formation mechanism exists first.

## 4. Open question (NOT settled): how does Stage1(+Stage2) relate to CoSiR's actual target?

This is the question a literature review + brainstorm needs to resolve before Block 2 gets designed — explicitly not answered by this document.

**The tension to resolve:** the buddy-vs-PercepT investigation validated a Stage1 (topic formation) + Stage2 (attention-pooling classifier assigning a query to that topic space) pipeline, and tonight's finding is that buddy's Stage2, copied straight from PercepT's design, is currently a little better than PercepT's own (pending the sweep's full hyperparameter confirmation via Task 11). But **Stage1+Stage2 topic classification is not the same problem CoSiR was ever meant to solve.** From the start, this project's actual target has been closer to GeneCIS (arXiv 2306.07969): not "assign this sample to one of K discovered topics," but "adapt what counts as similar, conditionally" — a genuinely open-ended notion of conditional similarity, not classification into a fixed discovered vocabulary.

Two new points the user wants this redesign to establish as CoSiR's actual differentiators, relative to GeneCIS specifically:

1. **Self/unsupervised condition discovery.** GeneCIS's method still mines an explicit, structured taxonomy of condition types from parsed image-caption data (a designed category system: attribute, spatial relation, etc.), and its benchmark's conditions are drawn from that designed set. CoSiR should discover the condition space itself from unsupervised structure — no hand-designed category list, no structured/labeled dataset construction step.
2. **Genuine cross-modal relations.** GeneCIS's actual similarity computation is image→image (a caption-derived condition modifies an image query, but the retrieval target is still images — the CIR/composed-retrieval lineage `combiner.py` already comes from). CoSiR's target is real image↔text conditional similarity: the condition should relate two different modalities to each other, not modify a same-modality retrieval task.

**What needs research, not assumption:** whether/how Stage1(+Stage2)'s validated topic-formation-and-classification mechanism can be extended or recombined to serve these two goals — e.g., does a discovered topic space (Stage 1) function as a *self-supervised proxy* for GeneCIS-style condition discovery, with Stage 2 generalized from "classify into K topics" into something that expresses open-ended, cross-modal conditional similarity rather than fixed-K classification? Or does a fundamentally different mechanism (one of tonight's Candidates A-E, or something not yet surfaced) fit better? This is genuinely undecided and is the actual subject of the commissioned literature review.

## Next step

Dispatch a comprehensive literature review + brainstorm (via Codex, per this project's established practice for research-heavy exploration) to propose how to combine or extend Stage1(+Stage2) with GeneCIS-style conditional similarity, informed by both new points above, and survey what else exists in the literature that might fit better than either reference point alone. Report back as a design document; §4 of this spec gets rewritten (or superseded by a new spec) once that discussion happens with the user — this document does not authorize any Block 2+ implementation yet.
