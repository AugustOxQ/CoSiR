# Experiment 13 Symmetric Shared Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add Option A as an opt-in `symmetric_shared` conditioning architecture while retaining the existing asymmetric behavior as the default.

**Architecture:** The new mode keeps one `TrainableEmbeddingManager` table, evaluates the existing single `Combiner_new` (and predictor) on both image and text inputs, and computes a single bidirectional contrastive matrix from both conditioned representations. Auxiliary per-side terms are averaged; shared-table regularizers remain once-per-table. Evaluation exposes coupled-table oracle, two-sided predictor retrieval, and independent two-sided diagnostic oracle without changing legacy metric semantics.

**Tech Stack:** Python, PyTorch, Hydra/OmegaConf, pytest.

**Spec:** `docs/archive/buddy_publication_plan/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 13; `docs/superpowers/scratch/2026-08-27_codex_symmetric_combiner_brainstorm.md`.

## Global Constraints

- `model.conditioning_mode` accepts `asymmetric` (default) and `symmetric_shared`; all existing asymmetric behavior must remain byte-for-byte reachable through the default.
- Implement Option A only: one shared condition table, one tied combiner, and one shared predictor called for both modalities.
- Do not delete or restructure legacy `combine_side`/`other_proj` behavior.
- In symmetric mode, average paired preservation, delta/gate/logit, laplacian, mixup, and predictor-distillation terms; apply shared-table regularizers once.
- Evaluation must provide coupled-table oracle (primary), two-sided predictor retrieval (deployable), and independent two-sided oracle (diagnostic only); never equate the new oracle with the legacy one-sided oracle.
- No training epochs, sweeps, wandb runs, pushes, or writes outside this worktree.
- Tests precede production changes; commit only verified task boundaries using ordinary `git commit`.

---

### Task 0: Config and symmetric model API

**Files:**
- Modify: `src/model/cosirmodel.py`, relevant model configuration files.
- Test: new symmetric model test under `src/test/`.

**Interfaces:**
- Produces a mode-aware model API that returns both conditioned embeddings and both combiner diagnostics in `symmetric_shared`, while leaving the asymmetric forward and helper APIs unchanged.

- [x] Write failing shape, identity-sharing, and default-asymmetric regression tests.
- [x] Implement the smallest config validation and symmetric forward/combine API to satisfy them.
- [x] Run targeted tests, inspect the diff, and commit.

### Task 1: Symmetric contrastive loss API

**Files:**
- Modify: `src/metrics/loss.py`.
- Test: new loss-focused tests under `src/test/`.

**Interfaces:**
- Consumes both conditioned outputs, raw embeddings, full-sequence inputs, shared conditions, and paired combiner diagnostics.
- Produces mode-aware loss/diagnostics that preserve the legacy asymmetric call path exactly.

- [x] Write failing tests for averaged two-sided preservation/diagnostics and single-application shared-table regularizers.
- [x] Implement the symmetric loss interface and retain the legacy branch unchanged.
- [x] Run targeted tests, inspect the diff, and commit.

### Task 2: Symmetric training integration and persistence

**Files:**
- Modify: `src/hook/train_cosir.py`, `src/eval/pipeline.py` as required for train evaluation.
- Test: focused CPU synthetic frozen-vs-trained sanity test under `src/test/`.

**Interfaces:**
- Uses Task 0 symmetric outputs and Task 1 symmetric criterion.
- Produces symmetric optimizer grouping, predictor distillation, snapshot/checkpoint metadata, and train-evaluation routing without modifying the legacy branch.

- [x] Write the synthetic frozen-vs-trained sanity test.
- [x] Implement the guarded training, optimizer, snapshot/checkpoint, and train-evaluation branch.
- [x] Run focused CPU tests, inspect the diff, and commit.

### Task 3: Three-tier symmetric evaluation

**Files:**
- Modify: `src/eval/metrics.py`, `src/eval/pipeline.py`.
- Test: new symmetric-evaluation consistency test under `src/test/`.

**Interfaces:**
- Produces coupled-table oracle, two-sided predictor retrieval, and independent two-sided diagnostic oracle only when `conditioning_mode=symmetric_shared`.

- [x] Write a falsifiable coupled-oracle consistency test with one side held at its initialization.
- [x] Implement symmetric evaluation primitives and mode dispatch while preserving legacy metrics unchanged.
- [x] Run targeted tests, inspect the diff, and commit.

### Task 4: Integrated verification and records

**Files:**
- Modify: tests from Tasks 0–3 as required; `.claude/20260827_log.md`; CCG task record.

- [x] Run the specified legacy and new test suite using the `CoSiR` conda environment; document pre-existing failures separately from introduced failures.
- [x] Run final review and correct actionable findings.
- [x] Write the per-file code-change log, archive the CCG task, and commit records.
