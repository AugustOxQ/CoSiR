# Buddy Type-Weighted Contrastive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement approved Experiment 15.3 by optionally excluding repair edges and type-weighting positive buddy sampling, with Family #2 as the primary target and Family #1 included symmetrically.

**Architecture:** Keep the persisted graph as the connectivity graph. At training setup, derive an optional repair-free semantic graph and a directed CSR-aligned weight tensor from the persisted edge provenance. The two existing losses consume that precomputed tensor only when configured; absent weights preserve the existing uniform random offset path exactly.

**Tech Stack:** Python, PyTorch, NumPy, Hydra, existing runnable CPU test scripts.

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` (Experiment 15.3; 15.2 Result immediately above it)

## Global Constraints

- Do not alter `lambda_buddy`, `lambda_buddy_con`, or `buddy_refresh` defaults or gates.
- Do not change `refresh_buddy_graph`.
- Build/filter CSRs and edge weights once at setup, never per training step.
- `loss.buddy_exclude_repair` defaults to `False`; `loss.buddy_type_weights` defaults to `None`.
- Type-aware flags with absent provenance must fail loudly.
- Run CPU-only tests; do not launch GPU training.

---

### Task 1: CSR provenance helpers and sampling behavior

**Files:**

- Modify: `src/metrics/regularizer.py`
- Test: `src/test/20260623_buddy_train_reg/test_buddy_reg.py`

**Interfaces:**

- Consumes: existing CSR `indptr`/`indices`, optional CSR-aligned edge-weight tensor.
- Produces: optional weighted positive selection in `buddy_graph_smoothness_loss` and `buddy_contrastive_loss`, while `None` follows the pre-existing uniform path byte-for-byte.

- [ ] **Step 1: Write failing CPU tests** for an all-repair semantic degree-zero anchor, uniform-weight sampling frequency, and skewed type-weighted positive sampling.

- [ ] **Step 2: Run the targeted test script** and confirm the new tests fail because the helpers/arguments do not yet exist.

- [ ] **Step 3: Implement minimal CSR provenance and weighted-sampling helpers** in `regularizer.py`, preserving the current `rand * degree` selection when no weights are supplied and using row-local cumulative weights plus `searchsorted` when they are.

- [ ] **Step 4: Extend both loss signatures** with an optional CSR-aligned `edge_weights` argument and route their positive-neighbour selection through the helper.

- [ ] **Step 5: Re-run the targeted test script** and confirm all original and new CPU tests pass.

### Task 2: Training setup, configuration gates, and fail-fast validation

**Files:**

- Modify: `src/hook/train_cosir.py`
- Test: `src/test/20260623_buddy_train_reg/test_buddy_reg.py`

**Interfaces:**

- Consumes: `embedding_manager.get_buddy_edges()` and `get_buddy_edge_types()`; `cfg.loss.buddy_exclude_repair`; `cfg.loss.buddy_type_weights`.
- Produces: static and Family #2 CSR/weight values selected at setup and passed to their existing loss call sites.

- [ ] **Step 1: Write a failing test** covering the explicit error raised when type-aware behavior is requested without edge provenance.

- [ ] **Step 2: Run the targeted test script** and confirm the failure is specifically the missing setup validation/helper.

- [ ] **Step 3: Add setup-time config parsing and provenance validation.** Validate the mapping’s known string keys and positive finite weights; derive an optional repair-filtered edge list and a matching per-undirected-edge weight vector, then construct CSR-aligned directed weights once.

- [ ] **Step 4: Wire semantic CSR/weights into Family #1 and Family #2 existing call sites, without changing Family #3 refresh behavior.** Emit a one-line `[buddy-con] type-aware sampling` confirmation when either option is active.

- [ ] **Step 5: Re-run the targeted tests and a Hydra `--cfg job` dry run** using the requested dict override syntax.

### Task 3: Verification, review, and handoff

**Files:**

- Modify: `.ccg/tasks/buddy-type-weighted-contrastive/task.json`
- Create: `.ccg/tasks/buddy-type-weighted-contrastive/review.md`

- [ ] **Step 1: Run the relevant CPU test scripts and inspect `git diff`.**

- [ ] **Step 2: Run Codex and Claude reviews in parallel against the final diff; address any critical finding and re-review if needed.**

- [ ] **Step 3: Commit the scoped implementation and archive the completed CCG task.**
