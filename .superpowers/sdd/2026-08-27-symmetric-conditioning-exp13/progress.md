# SDD ledger — plan: docs/superpowers/plans/2026-08-27-symmetric-conditioning-exp13.md

## Setup

- Worktree verified: `/project/CoSiR-exp13-symmetric-conditioning` on `experiment/symmetric_conditioning_exp13`.
- Spec authority read in full: publication plan §4 Experiment 13 and the Option A architecture memo.
- Ruling: implement only `symmetric_shared` Option A behind an explicit mode switch; retain existing asymmetric paths, including `combine_side` and `other_proj`, unchanged. Cost if wrong: the experiment may require a follow-up architecture rather than a rewrite.
- Baseline command: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && pytest -q src/test/test_model.py src/test/test_manager.py src/test/test_chunk_features.py`.
- Baseline result: 1 passed, 2 failed before source changes: `test_model.py` supplies a Tensor where current `encode_img` expects a mapping; `test_chunk_features.py` references missing `/project/CoSiR/data/test_features_new`.

## Preflight interface scan

| Tasks | Shared file/interface | Finding |
| --- | --- | --- |
| 0 → 1 | `CoSiRModel` symmetric output/diagnostics | Task 0 must define a stable paired diagnostic container before Task 1 consumes it. |
| 1 → 2 | symmetric criterion call and diagnostic names | Task 2 must use Task 1’s new mode-specific interface only. |
| 2 → 3 | mode routing in `TrainEvaluator` | Task 3 owns final metric semantics; Task 2 only adds train-evaluation transport. |
| 0, 2, 3 | config mode selection | One defaulted config lookup must guard all new code; legacy paths remain otherwise intact. |

| Task | Internal consistency check | Finding |
| --- | --- | --- |
| 0 | config validation, symmetric paired output, and legacy regression tests | Consistent: the explicit default preserves existing call shapes, while the new mode has a distinct paired API. |
| 1 | paired inputs/diagnostics and once-only shared regularizers | Consistent: depends only on Task 0's paired container and leaves the legacy criterion branch intact. |
| 2 | consumes Tasks 0–1 and adds guarded persistence/train evaluation | Consistent: its CPU sanity test validates integration without requiring training epochs. |
| 3 | symmetric-only metrics and a coupled-oracle consistency test | Consistent: consumes Task 2 routing and explicitly preserves legacy metric semantics. |
| 4 | verification, review, and records only | Consistent: consumes prior changes and introduces no product interface. |

Task 0: fix round 1/5 (1 addressed, 0 open — base-worktree RED evidence; commits c9cadd4..c9cadd4)
Task 0: complete (commits de4edd1..c9cadd4, review clean)
Task 1: fix round 1/5 (1 addressed, 0 open — paired laplacian/mixup test coverage; commits d9d3472..bfab892)
Task 1: complete (commits c9cadd4..bfab892, review clean)
Task 2: minor (deferred): ancillary symmetric snapshot metadata directly accesses `conditioning_mode`; use an asymmetric fallback for duck-typed legacy callers if final review judges it material.
Task 2: complete (commits bfab892..150f095, review clean)
Task 3: review/verification complete (commit e57daea plus uncached diagnostic-gallery optimization)

## Task 3 verification

- Verified `e57daea` against the Task 3 interface: symmetric mode dispatches only to a coupled-table oracle, independent two-sided diagnostic oracle, and two-sided predictor retrieval; legacy metric groups remain on the asymmetric branch.
- Focused command: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q src/test/test_symmetric_evaluation.py` — **9 passed**. The bytecode setting was necessary because timestamp-valid `.pyc` files initially made pytest import stale pre-Task-3 modules; direct Python import and the uncached run confirmed the worktree sources.
- Codex review: no Critical findings. One Warning found redundant K² text-gallery conditioning in the independent diagnostic; addressed by caching one conditioned text gallery per representative while keeping all K² comparisons.
- Claude review could not run: its wrapper unconditionally passes `--dangerously-skip-permissions`, which the installed Claude CLI rejects under root. This is an environment limitation, not a code failure.
- Post-review focused command: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q src/test/test_symmetric_evaluation.py` — **9 passed**.

## Task 4 integrated verification

- Command: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q src/test/test_symmetric_model.py src/test/test_symmetric_loss.py src/test/test_symmetric_training.py src/test/test_symmetric_evaluation.py src/test/test_model.py src/test/test_manager.py src/test/test_chunk_features.py`.
- Result: **24 passed, 2 failed**. The two failures are pre-existing and unrelated to Experiment 13: `test_model.py::test_cosirmodel` passes a Tensor to an existing mapping-only `encode_img`; `test_chunk_features.py::test_get_features_by_chunk` expects missing `/project/CoSiR/data/test_features_new` fixture data. These exactly match the recorded baseline failures.
- Scope/diff check: `git diff --check` passed. No production test regression was introduced.
