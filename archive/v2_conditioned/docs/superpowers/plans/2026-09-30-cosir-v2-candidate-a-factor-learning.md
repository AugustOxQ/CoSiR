# CoSiR v2 Candidate A: condition-aware factor learning (2×2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train four factor models (agreement per pair / per painting × with / without CLIP-image-cluster
condition episodes) on scorer-train rows. Pick one on the selection rows by a pre-registered rule, then test it
once against the matched control on fresh held episodes.

**Architecture:** Two new pure losses (`painting_infonce_loss`, `naive_episode_loss`) and a default-off extension
of `train_factors`: painting-expanded batches, painting-level agreement, and a naive-rule condition-episode term
with a learnable temperature. Real runs live in two dated script folders that reuse stage (d)'s cache and the
headroom probe's evaluation helpers. A timing smoke test decides local GPU vs a DAS6 node before any full run.

**Tech Stack:** Python 3.10 (conda env `CoSiR`), PyTorch, NumPy, SciPy (sparse), scikit-learn, pytest; RTX 3090
locally, DAS6 via the `cluster-run` skill if needed.

**Spec:** `docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-factor-learning-design.md` (read §0 first for
every term: R3, R0, β, naive rule, label episodes, gates). Executors read the spec and this plan.

## Global Constraints

- Python: `/root/miniconda3/envs/CoSiR/bin/python`, run from `/project/CoSiR` (branch `main`). Tests:
  `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q <path>`.
- Seed 42 unless stated. No `cuml` / `cugraph` imports.
- **Human labels (emotion, art style) are evaluation-only.** No external condition taxonomy or pretrained affect
  signal.
- **Rows:** training, graphs and partitions use scorer-train rows; the selection rows are for choosing; held rows
  are read only in Task 6 (`--run`, once). Val is never read.
- **No tuning:** R3_CONFIG weights, 2,000 steps, λ_condition 1.0, β 0.3, 64 episodes per step, 12 random
  negatives, and every threshold below are fixed.
- New `FactorTrainingConfig` fields default off: `FactorTrainingConfig()` must still reproduce R0's golden codes
  and `R3_CONFIG` must still equal the stored R3 config.
- Every modified `src/` file gets an entry in `.claude/20261016_log.md`: file path as header, before/after
  snippets, explanation. The file is gitignored: write it, never `git add` it.
- Real runs go in `src/test/20261016_factor_learning_grid/` and `src/test/20261017_factor_learning_held/`, each with
  a `<folder>_log.md` and a `.gitignore` containing `*.npy *.npz *.json *.pt *.log cache/ checkpoints/ results/`.
- Reports go in `docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md` and
  `docs/reports/auto/v2/2026-10-17_candidate_a_factor_learning_held.md`. Each has its verdict first and every
  number next to its baseline: C0 for criteria, naive on original R3 at β 0.3 as the current system. Figures go in
  `docs/reports/assets/2026-10-16_factor_learning/`, built by a script in `docs/reports/assets/`. Use no em dashes.
  Each report adds one row to `docs/reports/reports_sum.md` and updates its "Current work (v2)" line;
  `/root/miniconda3/envs/CoSiR/bin/python scripts/check_reports_sum.py` must print OK.
- **Another Claude session may have uncommitted edits in this tree** (e.g. `docs/reports/reports_sum.md`,
  `docs/reports/weekly/*`, `bin/`). Stage only this task's files by explicit path. For `reports_sum.md`, stage only
  your own lines. Take `git show HEAD:docs/reports/reports_sum.md`, apply your edit to that copy, then
  `git hash-object -w <copy>` and `git update-index --cacheinfo 100644,<hash>,docs/reports/reports_sum.md`. On an
  `index.lock` error, wait a few seconds and retry.
- Commit locally, never push. Every commit message ends with a blank line and then:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP`
- **Stop points** (report, then wait for the user):
  - Task 3: the timing smoke projects heavy runs, so ask the user to reserve a DAS6 node.
  - Task 4: C0 fails a gate, or no cell qualifies.
  - Any destructive action.

## Review Focus

- A painting with a single row in the batch under painting agreement: its mean is that row and the loss must stay
  finite. Pinned in Task 1 (`test_painting_infonce_handles_single_row_paintings`).
- An episode whose supports and contrasts give an all-zero naive weight row. The condition loss must fall back to
  the CLIP term without NaN. Pinned in Task 1 (`test_all_zero_weights_fall_back_to_clip_only`).
- A condition source whose rows are global row ids while `train_factors` sees only local scorer-train rows. This
  must raise a clear error, not index the wrong rows. Pinned in Task 2
  (`test_condition_source_rows_must_index_training_rows`).
- Loading R0/R3 checkpoints saved before the new config fields existed must still work, with defaults filled in.
  Pinned in Task 2 (`test_checkpoint_without_new_fields_loads_with_defaults`).
- `mine_condition_episodes(..., units=None)` with `num_hard > 0` must raise a clear error up front, not loop into
  "Too many conditions". Pinned in Task 1 (`test_hard_negatives_without_units_raise_up_front`).

---

### Task 1: The two losses and zero-hard-negative mining

**Files:**
- Modify: `src/train/factors.py` (add `painting_infonce_loss` after `cross_modal_infonce_loss`)
- Create: `src/train/factor_condition_loss.py`
- Modify: `src/train/condition_episodes.py` (`_hard_negatives`, `mine_condition_episodes` guard)
- Test: `src/test/test_factor_losses.py` (append), `src/test/test_factor_condition_loss.py` (new),
  `src/test/test_condition_episodes.py` (append)

**Interfaces:**
- Consumes: `cross_modal_infonce_loss(img_codes, txt_codes, temperature, group_ids)` (`src/train/factors.py`);
  `naive_condition_weights`, `pair_codes`, `conditional_score` (`src/model/conditioning.py`);
  `multi_positive_nce(logits, positive_mask)` (`src/train/train_scorer.py`);
  `mine_condition_episodes(source, units, keys, n_episodes, rng, episodes_per_condition=4, num_support=4,
  num_contrast=4, num_positive=4, num_hard=6, num_random=6, hard_pool=2048, max_failures=1000)`.
- Produces:
  - `painting_infonce_loss(img_codes: Tensor, txt_codes: Tensor, painting_ids: Tensor | np.ndarray,
    temperature: float = 0.1) -> Tensor`
  - `naive_episode_scores(img_feat, txt_feat, img_codes, txt_codes, anchor, supports, contrasts, candidates,
    beta: float) -> dict[str, Tensor]` with keys `"i2t"`, `"t2i"`, each `(E, K)`
  - `naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, anchor, supports, contrasts, candidates,
    positive_mask: Tensor, beta: float, log_tau: Tensor) -> Tensor`
  - `mine_condition_episodes(..., units=None, num_hard=0, ...)` works and returns only random negatives.

- [ ] **Step 1: Write the failing painting-InfoNCE tests** (append to `src/test/test_factor_losses.py`)

```python
from src.train.factors import cross_modal_infonce_loss, painting_infonce_loss


def _painting_codes(n, f=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, f, generator=g), torch.rand(n, f, generator=g)


def test_painting_infonce_equals_row_infonce_when_each_painting_has_one_row():
    img, txt = _painting_codes(10)
    ids = np.array([7, 3, 9, 1, 0, 5, 2, 8, 6, 4])
    expected = cross_modal_infonce_loss(img, txt, 0.1, torch.as_tensor(ids))
    assert torch.allclose(painting_infonce_loss(img, txt, ids, 0.1), expected, atol=1e-6)


def test_painting_infonce_ignores_caption_spread_when_painting_means_are_fixed():
    img, txt = _painting_codes(8)
    ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    img = img[[0, 0, 2, 2, 4, 4, 6, 6]]                  # rows of a painting share one image
    shift = torch.zeros_like(txt)
    shift[0], shift[1] = 0.05, -0.05                     # painting 0's mean caption code is unchanged
    assert torch.allclose(painting_infonce_loss(img, txt, ids), painting_infonce_loss(img, txt + shift, ids),
                          atol=1e-6)


def test_painting_infonce_is_invariant_to_row_order():
    img, txt = _painting_codes(9, seed=1)
    ids = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3])
    perm = np.random.default_rng(0).permutation(9)
    assert torch.allclose(painting_infonce_loss(img, txt, ids),
                          painting_infonce_loss(img[perm], txt[perm], ids[perm]), atol=1e-6)


def test_painting_infonce_handles_single_row_paintings():
    img, txt = _painting_codes(5, seed=2)
    loss = painting_infonce_loss(img, txt, np.array([0, 0, 1, 2, 2]))
    assert torch.isfinite(loss)
    assert painting_infonce_loss(img[:1], txt[:1], np.array([4])).item() == 0.0   # one painting: nothing to contrast


def test_painting_infonce_gradient_reaches_every_caption_row():
    img, txt = _painting_codes(6, seed=3)
    txt.requires_grad_(True)
    painting_infonce_loss(img, txt, np.array([0, 0, 1, 1, 2, 2])).backward()
    assert (txt.grad.abs().sum(dim=1) > 0).all()


def test_painting_infonce_validates_inputs():
    img, txt = _painting_codes(4)
    with pytest.raises(ValueError, match="temperature"):
        painting_infonce_loss(img, txt, np.arange(4), 0.0)
    with pytest.raises(ValueError, match="one entry per row"):
        painting_infonce_loss(img, txt, np.arange(3))
```

If `numpy`, `pytest` or `torch` are not yet imported at the top of `test_factor_losses.py`, add the imports.

- [ ] **Step 2: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_factor_losses.py -k painting`
Expected: FAIL with `ImportError: cannot import name 'painting_infonce_loss'`.

- [ ] **Step 3: Implement `painting_infonce_loss`** (in `src/train/factors.py`, right after `cross_modal_infonce_loss`)

```python
def painting_infonce_loss(
    img_codes: Tensor,
    txt_codes: Tensor,
    painting_ids: Tensor | np.ndarray,
    temperature: float = 0.1,
) -> Tensor:
    """Symmetric InfoNCE between each painting's mean image code and its mean caption code.

    Rows with the same painting id are averaged per modality. A painting's rows share one image, so its
    image mean is that image's code; its caption mean is the average code of its captions. The means are
    matched against the batch's other paintings with ``cross_modal_infonce_loss``, so one caption's code may
    differ from the image code as long as the painting's mean caption code matches it (factor-learning spec §5).
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = painting_ids if isinstance(painting_ids, Tensor) else torch.as_tensor(np.asarray(painting_ids))
    ids = ids.to(img_codes.device)
    if ids.shape != (len(img_codes),) or txt_codes.shape != img_codes.shape:
        raise ValueError("painting_ids must have one entry per row, and both code arrays the same shape")
    _, inverse = torch.unique(ids, return_inverse=True)
    n = int(inverse.max()) + 1
    counts = torch.zeros(n, device=img_codes.device, dtype=img_codes.dtype).index_add_(
        0, inverse, torch.ones(len(inverse), device=img_codes.device, dtype=img_codes.dtype))[:, None]
    img_mean = torch.zeros(n, img_codes.shape[1], device=img_codes.device, dtype=img_codes.dtype).index_add_(
        0, inverse, img_codes) / counts
    txt_mean = torch.zeros(n, txt_codes.shape[1], device=txt_codes.device, dtype=txt_codes.dtype).index_add_(
        0, inverse, txt_codes) / counts
    return cross_modal_infonce_loss(img_mean, txt_mean, temperature)
```

- [ ] **Step 4: Run the painting tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_factor_losses.py`
Expected: all PASS (the older tests in the file included).

- [ ] **Step 5: Write the failing condition-loss tests** (`src/test/test_factor_condition_loss.py`)

```python
"""Tests for the naive-rule condition-episode loss (factor-learning spec §5)."""

import math

import pytest
import torch

from src.train.factor_condition_loss import naive_episode_loss, naive_episode_scores


def _episode(zero_gap: bool = False):
    """Rows: 0 anchor, 1 support, 2 contrast, 3 positive, 4 negative. D = 2 features, F = 2 factors.

    Support pair code [2, 1], contrast pair code [0, 0.5] -> gap [2, 0.5] -> naive weights [0.8, 0.2].
    With zero_gap the contrast equals the support, so the weights are all zero.
    """
    img_feat = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    txt_feat = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, 0.6]])
    contrast = [2.0, 1.0] if zero_gap else [0.0, 0.5]
    img_codes = torch.tensor([[1.0, 1.0], [2.0, 1.0], contrast, [1.0, 0.0], [0.0, 1.0]])
    txt_codes = torch.tensor([[1.0, 1.0], [2.0, 1.0], contrast, [2.0, 0.0], [0.0, 2.0]])
    index = {"anchor": torch.tensor([0]), "supports": torch.tensor([[1]]), "contrasts": torch.tensor([[2]]),
             "candidates": torch.tensor([[3, 4]])}
    return img_feat, txt_feat, img_codes, txt_codes, index


MASK = torch.tensor([[True, False]])


def test_scores_match_hand_computation():
    # i2t: cos [0.6, 0.8] * 0.3 + factors [0.8*1*2, 0.2*1*2] = [1.78, 0.64]
    # t2i: cos [1, 0] * 0.3 + factors [0.8*1*1, 0.2*1*1] = [1.1, 0.2]
    img_feat, txt_feat, img_codes, txt_codes, index = _episode()
    scores = naive_episode_scores(img_feat, txt_feat, img_codes, txt_codes, **index, beta=0.3)
    assert torch.allclose(scores["i2t"], torch.tensor([[1.78, 0.64]]), atol=1e-6)
    assert torch.allclose(scores["t2i"], torch.tensor([[1.1, 0.2]]), atol=1e-6)


def test_loss_matches_hand_computation_and_uses_tau():
    img_feat, txt_feat, img_codes, txt_codes, index = _episode()
    expected = 0.5 * (math.log1p(math.exp(0.64 - 1.78)) + math.log1p(math.exp(0.2 - 1.1)))
    loss = naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index, positive_mask=MASK, beta=0.3,
                              log_tau=torch.tensor(0.0))
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    halved = 0.5 * (math.log1p(math.exp((0.64 - 1.78) / 2)) + math.log1p(math.exp((0.2 - 1.1) / 2)))
    loss_tau2 = naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index, positive_mask=MASK, beta=0.3,
                                   log_tau=torch.tensor(math.log(2.0)))
    assert loss_tau2.item() == pytest.approx(halved, abs=1e-6)


def test_gradients_reach_codes_through_weights_and_scores():
    img_feat, txt_feat, img_codes, txt_codes, index = _episode()
    img_codes.requires_grad_(True)
    txt_codes.requires_grad_(True)
    naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index, positive_mask=MASK, beta=0.3,
                       log_tau=torch.tensor(0.0)).backward()
    assert img_codes.grad[1].abs().sum() > 0          # support row: through the naive weights
    assert img_codes.grad[2].abs().sum() > 0          # contrast row: through the naive weights
    assert txt_codes.grad[3].abs().sum() > 0          # positive candidate caption (i2t)
    assert img_codes.grad[3].abs().sum() > 0          # positive candidate image (t2i)


def test_all_zero_weights_fall_back_to_clip_only():
    img_feat, txt_feat, img_codes, txt_codes, index = _episode(zero_gap=True)
    scores = naive_episode_scores(img_feat, txt_feat, img_codes, txt_codes, **index, beta=0.3)
    assert torch.allclose(scores["i2t"], torch.tensor([[0.18, 0.24]]), atol=1e-6)
    assert torch.allclose(scores["t2i"], torch.tensor([[0.3, 0.0]]), atol=1e-6)
    loss = naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index, positive_mask=MASK, beta=0.3,
                              log_tau=torch.tensor(0.0))
    assert torch.isfinite(loss)


def test_episode_without_positive_raises():
    img_feat, txt_feat, img_codes, txt_codes, index = _episode()
    with pytest.raises(ValueError, match="positive"):
        naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index,
                           positive_mask=torch.tensor([[False, False]]), beta=0.3, log_tau=torch.tensor(0.0))
```

- [ ] **Step 6: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_factor_condition_loss.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train.factor_condition_loss'`.

- [ ] **Step 7: Implement `src/train/factor_condition_loss.py`**

```python
"""Naive-rule condition-episode loss for training factor encoders (factor-learning spec §5).

The zero-parameter naive rule turns an episode's supports and contrasts into factor weights. The episode's
candidates are scored with ``conditional_score`` at a fixed beta in both retrieval directions, and a
multi-positive softmax rewards ranking the positives first. Gradients reach the factor codes through both the
weights and the candidate scores. The only free parameter is the temperature, which does not change rankings.
"""

import torch
from torch import Tensor

from src.model.conditioning import conditional_score, naive_condition_weights, pair_codes
from src.train.train_scorer import multi_positive_nce

DIRECTIONS = ("i2t", "t2i")


def naive_episode_scores(img_feat: Tensor, txt_feat: Tensor, img_codes: Tensor, txt_codes: Tensor,
                         anchor: Tensor, supports: Tensor, contrasts: Tensor, candidates: Tensor,
                         beta: float) -> dict[str, Tensor]:
    """Scores ``(E, K)`` per direction for E episodes over R encoded rows.

    ``img_feat`` / ``txt_feat`` are the rows' CLIP features ``(R, D)``; ``img_codes`` / ``txt_codes`` their factor
    codes ``(R, F)``. ``anchor (E,)``, ``supports (E, S)``, ``contrasts (E, C)`` and ``candidates (E, K)`` index
    those rows. i2t scores the anchor's image against the candidates' captions; t2i the anchor's caption against
    the candidates' images.
    """
    weights = naive_condition_weights(pair_codes(img_codes[supports], txt_codes[supports]),
                                      pair_codes(img_codes[contrasts], txt_codes[contrasts]))
    return {
        "i2t": conditional_score(img_feat[anchor], txt_feat[candidates], img_codes[anchor], txt_codes[candidates],
                                 weights, beta),
        "t2i": conditional_score(txt_feat[anchor], img_feat[candidates], txt_codes[anchor], img_codes[candidates],
                                 weights, beta),
    }


def naive_episode_loss(img_feat: Tensor, txt_feat: Tensor, img_codes: Tensor, txt_codes: Tensor,
                       anchor: Tensor, supports: Tensor, contrasts: Tensor, candidates: Tensor,
                       positive_mask: Tensor, beta: float, log_tau: Tensor) -> Tensor:
    """Mean over both directions of the multi-positive softmax loss of ``naive_episode_scores / exp(log_tau)``."""
    scores = naive_episode_scores(img_feat, txt_feat, img_codes, txt_codes, anchor, supports, contrasts,
                                  candidates, beta)
    tau = log_tau.exp()
    return 0.5 * sum(multi_positive_nce(scores[d] / tau, positive_mask) for d in DIRECTIONS)
```

- [ ] **Step 8: Run the condition-loss tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_factor_condition_loss.py`
Expected: 5 PASS.

- [ ] **Step 9: Write the failing zero-hard-negative mining tests** (append to `src/test/test_condition_episodes.py`)

```python
def test_zero_hard_negatives_need_no_units_and_give_random_outside_negatives():
    _, _, _, keys = _world()
    labels = np.arange(4000) % 5
    src = CommunitySource(labels, np.arange(4000), min_group_rows=100)
    ep = mine_condition_episodes(src, None, keys, 30, np.random.default_rng(4), num_hard=0, num_random=12)
    assert ep.candidates.shape == (30, 16)
    assert ep.positive_mask[:, :4].all() and not ep.positive_mask[:, 4:].any()
    for i in range(30):
        group = labels[ep.anchor[i]]
        assert (labels[ep.supports[i]] == group).all() and (labels[ep.candidates[i, :4]] == group).all()
        assert (labels[ep.contrasts[i]] != group).all() and (labels[ep.candidates[i, 4:]] != group).all()
        rows = [ep.anchor[i], *ep.supports[i], *ep.contrasts[i], *ep.candidates[i]]
        assert len(set(keys[rows].tolist())) == len(rows)            # no painting repeats


def test_hard_negatives_without_units_raise_up_front():
    _, _, _, keys = _world()
    src = CommunitySource(np.arange(4000) % 5, np.arange(4000), min_group_rows=100)
    with pytest.raises(ValueError, match="units"):
        mine_condition_episodes(src, None, keys, 5, np.random.default_rng(5), num_hard=2)
```

- [ ] **Step 10: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_episodes.py -k "zero_hard or without_units"`
Expected: FAIL. The first test hits `TypeError`/`RuntimeError` from `_hard_negatives` using `units=None`; the second
raises `RuntimeError: Too many conditions ...` instead of `ValueError`.

- [ ] **Step 11: Implement the mining change** (`src/train/condition_episodes.py`)

At the top of `_hard_negatives`, before the sampling line:

```python
    if count == 0:
        return []
```

At the top of `mine_condition_episodes`, before `fields = ...`:

```python
    if num_hard > 0 and units is None:
        raise ValueError("hard negatives need CLIP units (pair_feature_units); pass num_hard=0 to mine without them")
```

- [ ] **Step 12: Run the whole suite**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test`
Expected: all PASS (202 before this task plus 13 new).

- [ ] **Step 13: Change log and commit**

Append entries for `src/train/factors.py`, `src/train/factor_condition_loss.py` (new) and
`src/train/condition_episodes.py` to `.claude/20261016_log.md` (not staged). Then:

```bash
git add src/train/factors.py src/train/factor_condition_loss.py src/train/condition_episodes.py \
        src/test/test_factor_losses.py src/test/test_factor_condition_loss.py src/test/test_condition_episodes.py
git commit -m "feat(v2): painting-level InfoNCE, naive-rule condition-episode loss, zero-hard-negative mining

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP"
```

---

### Task 2: `train_factors` integration (painting batches, painting agreement, condition loss)

**Files:**
- Modify: `src/train/train_factors.py`
- Test: `src/test/test_train_factors.py` (append)

**Interfaces:**
- Consumes (Task 1): `painting_infonce_loss`, `naive_episode_scores`, `naive_episode_loss`,
  `mine_condition_episodes(..., units=None, num_hard=0, num_random=12)`; `CommunitySource(labels, rows,
  min_group_rows=200)` from `src/train/condition_sources.py` (its `.rows` attribute is the sorted row array).
- Produces:
  - `FactorTrainingConfig` new fields: `agreement_level: str = "pair"`, `painting_batches: bool = False`,
    `lambda_condition: float = 0.0`, `condition_episodes_per_step: int = 64`, `condition_beta: float = 0.3`.
  - `class GroupRows(group_ids)` with `.expand(rows) -> np.ndarray` (sorted rows of every group touched).
  - `train_factors(img_features, txt_features, graph, config, device=None, group_ids=None,
    condition_source=None, history: dict | None = None, log_every: int = 50)
    -> tuple[SharedFactorEncoder, np.ndarray, np.ndarray]` (unchanged return). When `history` is a dict it
    receives lists `step`, `loss`, `agreement`, `batch_rows` and, with the condition loss, `condition_loss`,
    `tau`.
  - `condition_source` rows index the rows passed in (0 .. n−1), not global ArtELingo rows.

- [ ] **Step 1: Write the failing tests** (append to `src/test/test_train_factors.py`)

```python
import dataclasses

from src.train.condition_episodes import mine_condition_episodes
from src.train.condition_sources import CommunitySource
from src.train.factor_condition_loss import naive_episode_scores
from src.train.train_factors import GroupRows


def test_group_rows_expand_returns_every_row_of_each_touched_group():
    groups = np.array([3, 1, 3, 2, 1, 3, 0])
    index = GroupRows(groups)
    assert index.expand(np.array([0])).tolist() == [0, 2, 5]
    assert index.expand(np.array([4, 6])).tolist() == [1, 4, 6]
    assert index.expand(np.array([2, 5, 0])).tolist() == [0, 2, 5]


def test_painting_batches_contain_complete_paintings(paired_features, monkeypatch):
    import src.train.train_factors as tf

    img, txt, graph = paired_features                       # 48 rows, edges (2k, 2k+1)
    groups = np.arange(48) // 4                              # paintings of 4 rows
    seen = []
    real = tf.cross_modal_infonce_loss

    def spy(img_codes, txt_codes, temperature, group_ids):
        seen.append(group_ids.cpu().numpy())
        return real(img_codes, txt_codes, temperature, group_ids)

    monkeypatch.setattr(tf, "cross_modal_infonce_loss", spy)
    config = FactorTrainingConfig(num_factors=4, epochs=3, batch_size=4, agreement="infonce",
                                  painting_batches=True)
    train_factors(img, txt, graph, config, device="cpu", group_ids=groups)
    assert len(seen) == 3
    for batch_groups in seen:
        assert (np.bincount(batch_groups)[np.unique(batch_groups)] == 4).all()


def test_painting_agreement_replaces_the_row_infonce(paired_features, monkeypatch):
    import src.train.train_factors as tf

    img, txt, graph = paired_features
    calls = {"painting": 0, "row": 0}
    real_painting = tf.painting_infonce_loss

    def painting_spy(*args, **kwargs):
        calls["painting"] += 1
        return real_painting(*args, **kwargs)

    def row_spy(*args, **kwargs):
        calls["row"] += 1
        raise AssertionError("row InfoNCE must not run under painting agreement")

    monkeypatch.setattr(tf, "painting_infonce_loss", painting_spy)
    monkeypatch.setattr(tf, "cross_modal_infonce_loss", row_spy)
    config = FactorTrainingConfig(num_factors=4, epochs=2, batch_size=4, agreement="infonce",
                                  agreement_level="painting", painting_batches=True)
    train_factors(img, txt, graph, config, device="cpu", group_ids=np.arange(48) // 4)
    assert calls == {"painting": 2, "row": 0}


def _condition_world():
    img, txt, graph = _ring_fixture()                        # 64 rows, D = 16
    keys = np.arange(64)                                     # each row its own painting
    source = CommunitySource(np.arange(64) % 4, np.arange(64), min_group_rows=4)
    return img, txt, graph, keys, source


def test_condition_loss_trains_and_records_history():
    img, txt, graph, keys, source = _condition_world()
    config = FactorTrainingConfig(num_factors=8, epochs=5, batch_size=16, agreement="infonce",
                                  painting_batches=True, lambda_condition=1.0, condition_episodes_per_step=8)
    history = {}
    _, img_codes, txt_codes = train_factors(img, txt, graph, config, device="cpu", group_ids=keys,
                                            condition_source=source, history=history, log_every=1)
    assert np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()
    assert history["step"] == [1, 2, 3, 4, 5]
    assert all(np.isfinite(history["condition_loss"])) and all(t > 0 for t in history["tau"])


def test_condition_source_rows_must_index_training_rows():
    img, txt, graph, keys, _ = _condition_world()
    global_source = CommunitySource(np.arange(200) % 4, np.arange(100, 200), min_group_rows=4)
    config = FactorTrainingConfig(num_factors=8, epochs=1, batch_size=16, agreement="infonce",
                                  painting_batches=True, lambda_condition=1.0)
    with pytest.raises(ValueError, match="training rows"):
        train_factors(img, txt, graph, config, device="cpu", group_ids=keys, condition_source=global_source)


@pytest.mark.parametrize("overrides, kwargs, message", [
    (dict(lambda_condition=1.0), dict(), "condition_source"),
    (dict(), dict(condition_source="SOURCE"), "condition_source"),
    (dict(agreement_level="painting"), dict(), "painting_batches"),
    (dict(agreement_level="rows"), dict(), "agreement_level"),
    (dict(painting_batches=True), dict(group_ids=None), "group_ids"),
])
def test_new_options_are_validated(overrides, kwargs, message):
    img, txt, graph, keys, source = _condition_world()
    config = dataclasses.replace(FactorTrainingConfig(num_factors=8, epochs=1, batch_size=16, agreement="infonce"),
                                 **overrides)
    call = {"group_ids": keys, **kwargs}
    if call.get("condition_source") == "SOURCE":
        call["condition_source"] = source
    if call["group_ids"] is None and config.agreement == "infonce":
        config = dataclasses.replace(config, agreement="cosine")
    with pytest.raises(ValueError, match=message):
        train_factors(img, txt, graph, config, device="cpu", **call)


def test_checkpoint_without_new_fields_loads_with_defaults(tmp_path):
    from src.train.train_factors import load_factor_checkpoint, save_factor_checkpoint

    img, txt, graph = _ring_fixture()
    config = FactorTrainingConfig(num_factors=8, epochs=2, batch_size=16)
    model, _, _ = train_factors(img, txt, graph, config, device="cpu")
    save_factor_checkpoint(model, config, tmp_path / "old.pt")
    payload = torch.load(tmp_path / "old.pt", weights_only=True)
    for field in ("agreement_level", "painting_batches", "lambda_condition", "condition_episodes_per_step",
                  "condition_beta"):
        payload["config"].pop(field)
    torch.save(payload, tmp_path / "old.pt")
    _, loaded = load_factor_checkpoint(tmp_path / "old.pt")
    assert loaded == config


def _style_world(seed=7):
    """480 rows / 240 paintings; a 4-way 'style' carried by a low-variance direction, content high-variance."""
    rng = np.random.default_rng(seed)
    paint = np.arange(480) // 2
    style = rng.integers(0, 4, 240)[paint]
    content = rng.standard_normal((240, 12))[paint]
    onehot = np.eye(4)[style] * 0.25
    img = np.hstack([content, onehot]) + 0.05 * rng.standard_normal((480, 16))
    txt = np.hstack([content + 0.3 * rng.standard_normal((480, 12)), onehot]) + 0.05 * rng.standard_normal((480, 16))
    left = np.arange(0, 480, 2)
    graph = csr_matrix((np.ones(480, dtype=np.float32),
                        (np.concatenate((left, left + 1)), np.concatenate((left + 1, left)))), shape=(480, 480))
    return img.astype(np.float32), txt.astype(np.float32), graph, paint, style


def _naive_r1(model, img, txt, source, keys, n=200, seed=99):
    from src.train.train_factors import encode_rows

    ic, tc = encode_rows(model, img, txt, device="cpu")
    ep = mine_condition_episodes(source, None, keys, n, np.random.default_rng(seed), num_hard=0, num_random=12)
    t = lambda a: torch.as_tensor(a)                                   # noqa: E731
    scores = naive_episode_scores(t(img), t(txt), t(ic), t(tc), t(ep.anchor), t(ep.supports), t(ep.contrasts),
                                  t(ep.candidates), beta=0.3)
    mask = torch.as_tensor(ep.positive_mask).float()
    hit = [mask.gather(1, s.argmax(dim=1, keepdim=True)).mean().item() for s in scores.values()]
    return sum(hit) / 2


def test_condition_loss_teaches_a_low_variance_condition():
    from src.train.train_factors import R3_CONFIG

    img, txt, graph, paint, style = _style_world()
    source = CommunitySource(style, np.arange(480), min_group_rows=20)
    base = dataclasses.replace(R3_CONFIG, num_factors=8, epochs=300, batch_size=64, painting_batches=True,
                               condition_episodes_per_step=16)
    runs = {}
    for name, lam in (("none", 0.0), ("condition", 1.0)):
        config = dataclasses.replace(base, lambda_condition=lam)
        model, _, _ = train_factors(img, txt, graph, config, device="cpu", group_ids=paint,
                                    condition_source=source if lam > 0 else None)
        runs[name] = _naive_r1(model, img, txt, source, paint)
    assert runs["condition"] >= runs["none"] + 0.10, runs
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_train_factors.py`
Expected: the new tests FAIL (`ImportError: cannot import name 'GroupRows'`); the existing tests still pass.

- [ ] **Step 3: Implement the config fields** (`src/train/train_factors.py`)

Add to `FactorTrainingConfig`, after `center_inputs`:

```python
    agreement_level: str = "pair"
    painting_batches: bool = False
    lambda_condition: float = 0.0
    condition_episodes_per_step: int = 64
    condition_beta: float = 0.3
```

Extend the class docstring with one paragraph:

```
    The last five fields (factor-learning spec 2026-09-30) are also OFF by default.
    ``painting_batches`` expands every edge-sampled batch to all rows of each sampled painting (group);
    ``agreement_level="painting"`` replaces the row InfoNCE with ``painting_infonce_loss`` (needs
    ``painting_batches`` and ``agreement="infonce"``); ``lambda_condition > 0`` adds the naive-rule
    condition-episode loss (``naive_episode_loss``) on ``condition_episodes_per_step`` episodes mined per
    step from a ``condition_source``, scored at the fixed ``condition_beta`` with 12 random negatives.
```

Imports to add at the top of the module:

```python
import math

from src.train.condition_episodes import mine_condition_episodes
from src.train.factor_condition_loss import naive_episode_loss, naive_episode_scores
```

and add `painting_infonce_loss` to the existing `from src.train.factors import (...)` list.

- [ ] **Step 4: Implement `GroupRows` and the episode helpers** (module level, before `train_factors`)

```python
class GroupRows:
    """Row lookup by group id: ``expand(rows)`` returns every row of every group that ``rows`` touches, sorted."""

    def __init__(self, group_ids: np.ndarray) -> None:
        self._group_ids = np.asarray(group_ids)
        self._order = np.argsort(self._group_ids, kind="stable")
        self._sorted = self._group_ids[self._order]

    def expand(self, rows: np.ndarray) -> np.ndarray:
        groups = np.unique(self._group_ids[np.asarray(rows)])
        starts = np.searchsorted(self._sorted, groups, side="left")
        stops = np.searchsorted(self._sorted, groups, side="right")
        return np.sort(np.concatenate([self._order[a:b] for a, b in zip(starts, stops)]))


CONDITION_RANDOM_NEGATIVES = 12


def _mine_condition(source, group_ids: np.ndarray, config: FactorTrainingConfig, rng):
    return mine_condition_episodes(source, None, group_ids, config.condition_episodes_per_step, rng,
                                   num_hard=0, num_random=CONDITION_RANDOM_NEGATIVES)


def _episode_tensors(model: SharedFactorEncoder, img: torch.Tensor, txt: torch.Tensor, episodes, device):
    """Encode the unique rows of a batch of episodes; return their features, codes and row-local indices."""
    table = np.concatenate([episodes.anchor[:, None], episodes.supports, episodes.contrasts, episodes.candidates],
                           axis=1)
    rows, inverse = np.unique(table, return_inverse=True)
    inverse = torch.as_tensor(inverse.reshape(table.shape), device=device)
    s, c = episodes.supports.shape[1], episodes.contrasts.shape[1]
    index = {"anchor": inverse[:, 0], "supports": inverse[:, 1:1 + s], "contrasts": inverse[:, 1 + s:1 + s + c],
             "candidates": inverse[:, 1 + s + c:]}
    img_rows, txt_rows = img[rows].to(device), txt[rows].to(device)
    return img_rows, txt_rows, model.encode_image(img_rows), model.encode_text(txt_rows), index
```

- [ ] **Step 5: Extend `train_factors`**

New signature:

```python
def train_factors(
    img_features: np.ndarray,
    txt_features: np.ndarray,
    graph: csr_matrix,
    config: FactorTrainingConfig,
    device: str | None = None,
    group_ids: np.ndarray | None = None,
    condition_source=None,
    history: dict | None = None,
    log_every: int = 50,
) -> tuple[SharedFactorEncoder, np.ndarray, np.ndarray]:
```

Add to the docstring: "``condition_source`` (rows 0 .. n−1 of the arrays passed in) is required exactly when
``config.lambda_condition > 0``. ``history``, if a dict, receives per-step logs every ``log_every`` steps (plus
the first and last)."

Validation: right after the existing `group_ids` shape/dtype checks (before the model is built), add:

```python
    if config.agreement_level not in {"pair", "painting"}:
        raise ValueError(f"agreement_level must be 'pair' or 'painting', got {config.agreement_level!r}")
    if config.agreement_level == "painting" and not (config.painting_batches and config.agreement == "infonce"):
        raise ValueError("agreement_level='painting' needs painting_batches=True and agreement='infonce'")
    if (config.painting_batches or config.lambda_condition > 0) and group_ids is None:
        raise ValueError("painting_batches and the condition loss need group_ids (painting / leakage-group ids)")
    if config.lambda_condition < 0:
        raise ValueError("lambda_condition must be >= 0")
    if (config.lambda_condition > 0) != (condition_source is not None):
        raise ValueError("pass a condition_source exactly when lambda_condition > 0")
    if condition_source is not None and (condition_source.rows.min() < 0
                                         or condition_source.rows.max() >= len(img_features)):
        raise ValueError("condition_source rows must index the training rows (0 .. n-1), not global rows")
```

The existing InfoNCE-needs-`group_ids` check runs earlier; the parametrized validation test switches that case
to `agreement="cosine"` so the new `painting_batches` message is the one raised.

After the optimizer is created, add:

```python
    group_rows = GroupRows(group_ids) if config.painting_batches else None
    log_tau = None
    if config.lambda_condition > 0:
        condition_rng = np.random.default_rng([config.seed, 1])
        first_episodes = _mine_condition(condition_source, group_ids, config, condition_rng)
        with torch.no_grad():                                    # tau := std of step-0 scores (unit-scale logits)
            img_rows, txt_rows, ic, tc, index = _episode_tensors(model, img, txt, first_episodes, selected_device)
            scores = naive_episode_scores(img_rows, txt_rows, ic, tc, index["anchor"], index["supports"],
                                          index["contrasts"], index["candidates"], config.condition_beta)
            std = float(torch.cat([scores["i2t"].ravel(), scores["t2i"].ravel()]).std())
        log_tau = torch.nn.Parameter(torch.tensor(math.log(max(std, 1e-6)), device=selected_device))
        optimizer.add_param_group({"params": [log_tau]})
```

Inside the loop, right after `node_ids = np.unique(sampled.reshape(-1))`:

```python
        if group_rows is not None:
            node_ids = group_rows.expand(node_ids)
```

Replace the agreement block with:

```python
        if config.agreement_level == "painting":
            agreement = painting_infonce_loss(
                img_codes, txt_codes, torch.as_tensor(group_ids[node_ids], device=selected_device),
                config.infonce_temperature)
        elif config.agreement == "cosine":
            agreement = paired_agreement_loss(img_codes, txt_codes)
        else:
            agreement = cross_modal_infonce_loss(
                img_codes,
                txt_codes,
                config.infonce_temperature,
                None
                if group_ids is None
                else torch.as_tensor(group_ids[node_ids], device=selected_device),
            )
```

After the decorrelation block, before `optimizer.zero_grad`:

```python
        condition_loss = None
        if config.lambda_condition > 0:
            episodes = first_episodes if epoch == 1 else _mine_condition(condition_source, group_ids, config,
                                                                         condition_rng)
            img_rows, txt_rows, ic, tc, index = _episode_tensors(model, img, txt, episodes, selected_device)
            condition_loss = naive_episode_loss(
                img_rows, txt_rows, ic, tc, **index,
                positive_mask=torch.as_tensor(episodes.positive_mask, device=selected_device),
                beta=config.condition_beta, log_tau=log_tau)
            loss = loss + config.lambda_condition * condition_loss
```

After `optimizer.step()` (keep the existing per-epoch print):

```python
        if history is not None and (epoch % log_every == 0 or epoch in (1, config.epochs)):
            history.setdefault("step", []).append(epoch)
            history.setdefault("loss", []).append(loss.item())
            history.setdefault("agreement", []).append(agreement.item())
            history.setdefault("batch_rows", []).append(int(len(node_ids)))
            if condition_loss is not None:
                history.setdefault("condition_loss", []).append(condition_loss.item())
                history.setdefault("tau", []).append(float(log_tau.detach().exp()))
```

- [ ] **Step 6: Run the train-factors tests**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_train_factors.py`
Expected: all PASS, including `test_default_config_codes_match_pre_change_golden_values`,
`test_r3_preset_*` and the new tests.

If `test_condition_loss_teaches_a_low_variance_condition` fails: first check the implementation (the condition
loss must change the codes). Then report DONE_WITH_CONCERNS with the two R@1 values. You may adjust the synthetic
world's noise levels (not the +0.10 threshold, not the training config), but only if you explain in the report
why the world was mis-calibrated.

- [ ] **Step 7: Run the whole suite**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test`
Expected: all PASS.

- [ ] **Step 8: Change log and commit**

Append the `src/train/train_factors.py` entry to `.claude/20261016_log.md` (not staged).

```bash
git add src/train/train_factors.py src/test/test_train_factors.py
git commit -m "feat(v2): train_factors painting batches, painting-level agreement and naive-rule condition loss

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP"
```

---

### Task 3: Grid script, prepare, and the timing smoke (STOP POINT: local vs DAS6)

**Files:**
- Create: `src/test/20261016_factor_learning_grid/run_grid.py`,
  `src/test/20261016_factor_learning_grid/20261016_factor_learning_grid_log.md`,
  `src/test/20261016_factor_learning_grid/.gitignore`

**Interfaces:**
- Consumes (Task 2): `train_factors(..., group_ids, condition_source, history)`, `R3_CONFIG`,
  `save_factor_checkpoint`, `load_factor_checkpoint`, `encode_rows`; `CommunitySource`;
  `evaluate_factor_gates`, `AMENDED_2026_09_29_THRESHOLDS`; `build_content_graph`, `GraphConfig`; stage (d)'s
  `run_selection.py` via `run_final.py` exactly as `src/test/20261015_factor_headroom_probe/run_probe.py` imports it
  (`sel.load_prepared()`, `sel.masked`, `sel.log`, `fin.SELECTION_SHA256`).
- Produces for Tasks 4-6:
  - `cache/grid_prepare.npz`: `local_groups`, `clip_image_local`, `community_local`.
  - `cache/graph.npz`: `scipy.sparse.save_npz`, local scorer-train indices.
  - `cache/grid_prepare.json`: `r0_readout_reference`, sizes, SHA-256s, timings.
  - `CELLS`, `cell_config(cell, seed, steps)`.
  - `run_cell(cell, seed, steps, tag) -> dict`, which writes `checkpoints/{cell}_seed{seed}{tag}.pt` and
    `results/history_{cell}_seed{seed}{tag}.json`.
  - `results/smoke_timing.json` with the decision.

- [ ] **Step 1: Write the script's constants, cells and prepare phase**

```python
"""CoSiR v2 Candidate A factor-learning 2x2 grid (spec docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-factor-learning-design.md).

Four factor models on scorer-train rows (spec §4): C0 (row agreement, no condition; the matched control),
A (painting agreement), S (CLIP image k-means condition episodes), AS (both). Run from the repository root:

    python src/test/20261016_factor_learning_grid/run_grid.py --prepare
    python src/test/20261016_factor_learning_grid/run_grid.py --smoke            # timing: local GPU or DAS6?
    python src/test/20261016_factor_learning_grid/run_grid.py --run C0 --seed 42
    python src/test/20261016_factor_learning_grid/run_grid.py --evaluate         # Task 4
    python src/test/20261016_factor_learning_grid/run_grid.py --tables

Row scope: training, the graph and the partitions use scorer-train rows only (local indices 0..n-1); evaluation
reads selection rows only; val and held rows are never read.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.sparse import load_npz, save_npz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_FINAL_PATH = ROOT / "src/test/20261014_stage_d_final/run_final.py"
_spec = importlib.util.spec_from_file_location("run_final", _FINAL_PATH)
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)
sel = fin.sel

from src.data.artelingo import load_artelingo  # noqa: E402
from src.eval.factor_gates import AMENDED_2026_09_29_THRESHOLDS, evaluate_factor_gates  # noqa: E402
from src.model.graph import GraphConfig, build_content_graph  # noqa: E402
from src.train.condition_sources import CommunitySource  # noqa: E402
from src.train.train_factors import (R3_CONFIG, encode_rows, load_factor_checkpoint,  # noqa: E402
                                     save_factor_checkpoint, train_factors)

SEED = 42
FULL_STEPS = 2000
CELLS = {"C0": {"agreement_level": "pair", "lambda_condition": 0.0},
         "A": {"agreement_level": "painting", "lambda_condition": 0.0},
         "S": {"agreement_level": "pair", "lambda_condition": 1.0},
         "AS": {"agreement_level": "painting", "lambda_condition": 1.0}}
R0_PATH = ROOT / "src/test/20261011_factor_repair_grid/checkpoints/R0_seed42.pt"
R0_SHA256 = "4229dfe55f735bc7e9849c8d7af623b5872a9de940f616969ef477fb00a253a7"
CACHE, CKPT, RESULTS = HERE / "cache", HERE / "checkpoints", HERE / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE_STEPS = (10, 60)                      # two lengths; per-step time = slope (removes warm-up cost)
HEAVY_RUN_SECONDS = 45 * 60                 # stop-point thresholds (plan Task 3 Step 5)
HEAVY_TOTAL_SECONDS = 3 * 3600
HEAVY_PEAK_GIB = 20.0
log = sel.log


def cell_config(cell: str, seed: int, steps: int = FULL_STEPS):
    return dataclasses.replace(R3_CONFIG, painting_batches=True, seed=seed, epochs=steps, **CELLS[cell])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_report(img_fit, txt_fit, img_eval, txt_eval, data, cache, community_local, reference):
    st, sl = cache["scorer_train"], cache["selection"]
    return evaluate_factor_gates(
        fit_img_codes=img_fit, fit_txt_codes=txt_fit,
        fit_img_features=data.img_features[st], fit_txt_features=data.txt_features[st],
        eval_img_codes=img_eval, eval_txt_codes=txt_eval,
        eval_img_features=data.img_features[sl], eval_txt_features=data.txt_features[sl],
        community_img_codes=img_fit, community_txt_codes=txt_fit, community_labels=community_local,
        thresholds=AMENDED_2026_09_29_THRESHOLDS, readout_reference=reference)


def prepare() -> None:
    started = perf_counter()
    for folder in (CACHE, CKPT, RESULTS):
        folder.mkdir(exist_ok=True)
    cache, _ = sel.load_prepared()
    data = load_artelingo()
    st, sl = cache["scorer_train"], cache["selection"]
    if np.intersect1d(cache["groups"][st], cache["groups"][sl]).size:
        raise AssertionError("a painting spans scorer-train and selection")
    _, local_groups = np.unique(cache["groups"][st], return_inverse=True)
    clip_image_local, community_local = cache["clip_image"][st], cache["community"][st]
    if (clip_image_local < 0).any() or (community_local < 0).any():
        raise AssertionError("cached partitions must label every scorer-train row")
    t0 = perf_counter()
    graph = build_content_graph(data.img_features[st], data.txt_features[st], GraphConfig())
    save_npz(CACHE / "graph.npz", graph)
    graph_seconds = perf_counter() - t0
    if sha256_file(R0_PATH) != R0_SHA256:
        raise AssertionError("R0 checkpoint SHA-256 mismatch")
    r0, _ = load_factor_checkpoint(R0_PATH, device=DEVICE)
    fit = encode_rows(r0, data.img_features, data.txt_features, rows=st)
    ev = encode_rows(r0, data.img_features, data.txt_features, rows=sl)
    r0_gates = gate_report(*fit, *ev, data, cache, community_local, reference=None)
    reference = [float(r0_gates.values["readout_img"]), float(r0_gates.values["readout_txt"])]
    np.savez(CACHE / "grid_prepare.npz", local_groups=local_groups.astype(np.int64),
             clip_image_local=clip_image_local.astype(np.int64), community_local=community_local.astype(np.int64))
    meta = {"scorer_train_rows": int(len(st)), "selection_rows": int(len(sl)),
            "paintings": int(local_groups.max() + 1), "graph_edges": int(graph.nnz // 2),
            "graph_seconds": graph_seconds, "r0_sha256": R0_SHA256, "r0_readout_reference": reference,
            "clip_image_groups": int(len(np.unique(clip_image_local))), "seconds": perf_counter() - started}
    (CACHE / "grid_prepare.json").write_text(json.dumps(meta, indent=2))
    log(f"Prepared: {meta}")


def load_grid():
    cache, _ = sel.load_prepared()
    prep = dict(np.load(CACHE / "grid_prepare.npz"))
    meta = json.loads((CACHE / "grid_prepare.json").read_text())
    return cache, prep, meta, load_npz(CACHE / "graph.npz").tocsr()
```

- [ ] **Step 2: Write `run_cell` and the smoke phase**

```python
def run_cell(cell: str, seed: int, steps: int = FULL_STEPS, tag: str = "") -> dict:
    cache, prep, _, graph = load_grid()
    data = load_artelingo()
    st = cache["scorer_train"]
    config = cell_config(cell, seed, steps)
    source = (CommunitySource(prep["clip_image_local"], np.arange(len(st)))
              if config.lambda_condition > 0 else None)
    history: dict = {}
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = perf_counter()
    out = io.StringIO()
    with redirect_stdout(out):                                  # train_factors prints one line per step
        model, img_codes, txt_codes = train_factors(
            data.img_features[st], data.txt_features[st], graph, config, device=DEVICE,
            group_ids=prep["local_groups"], condition_source=source, history=history)
    seconds = perf_counter() - t0
    if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
        raise AssertionError(f"{cell} seed {seed}: non-finite codes")
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0
    name = f"{cell}_seed{seed}{tag}"
    save_factor_checkpoint(model, config, CKPT / f"{name}.pt")
    record = {"cell": cell, "seed": seed, "steps": steps, "seconds": seconds, "peak_gpu_gib": peak,
              "history": history, "last_print": out.getvalue().strip().splitlines()[-1],
              "config": dataclasses.asdict(config)}
    (RESULTS / f"history_{name}.json").write_text(json.dumps(record, indent=2))
    log(f"{name}: {steps} steps in {seconds:.1f} s, peak GPU {peak:.2f} GiB")
    return record


def smoke() -> dict:
    per_cell = {}
    for cell in CELLS:
        short, long = (run_cell(cell, SEED, steps, tag=f"_smoke{steps}") for steps in SMOKE_STEPS)
        per_step = (long["seconds"] - short["seconds"]) / (SMOKE_STEPS[1] - SMOKE_STEPS[0])
        fixed = max(short["seconds"] - per_step * SMOKE_STEPS[0], 0.0)
        per_cell[cell] = {"seconds_per_step": per_step, "fixed_seconds": fixed,
                          "projected_full_seconds": fixed + per_step * FULL_STEPS,
                          "peak_gpu_gib": max(short["peak_gpu_gib"], long["peak_gpu_gib"])}
    slowest = max(v["projected_full_seconds"] for v in per_cell.values())
    grid = sum(v["projected_full_seconds"] for v in per_cell.values())
    replication = 2 * (slowest + per_cell["C0"]["projected_full_seconds"])   # picked cell unknown: slowest
    total = grid + replication
    peak = max(v["peak_gpu_gib"] for v in per_cell.values())
    heavy = slowest > HEAVY_RUN_SECONDS or total > HEAVY_TOTAL_SECONDS or peak > HEAVY_PEAK_GIB
    result = {"per_cell": per_cell, "projected_grid_seconds": grid, "projected_replication_seconds": replication,
              "projected_total_training_seconds_sequential": total, "peak_gpu_gib": peak,
              "thresholds": {"run_seconds": HEAVY_RUN_SECONDS, "total_seconds": HEAVY_TOTAL_SECONDS,
                             "peak_gib": HEAVY_PEAK_GIB},
              "decision": "ask_user_for_das6_node" if heavy else "run_locally", "device": DEVICE}
    (RESULTS / "smoke_timing.json").write_text(json.dumps(result, indent=2))
    log(f"Smoke timing: {json.dumps(result, indent=2)}")
    return result
```

Add `main()` with the flags `--prepare`, `--smoke`, `--run CELL --seed S`, `--evaluate` (Task 4; for now raise
`NotImplementedError("Task 4")`) and `--tables`.

- [ ] **Step 3: Run prepare**

Run: `/root/miniconda3/envs/CoSiR/bin/python src/test/20261016_factor_learning_grid/run_grid.py --prepare 2>&1 | tee src/test/20261016_factor_learning_grid/run_prepare.log`
Expected: completes; 183,694 rows, 36,518 paintings, 64 CLIP image groups, a finite R0 readout reference.

- [ ] **Step 4: Run the timing smoke**

Run: `/root/miniconda3/envs/CoSiR/bin/python src/test/20261016_factor_learning_grid/run_grid.py --smoke 2>&1 | tee src/test/20261016_factor_learning_grid/run_smoke.log`
Expected: `results/smoke_timing.json` with per-cell seconds per step, projected full-run seconds, peak GPU memory
and `decision`.

- [ ] **Step 5: STOP POINT: report the projection to the controller**

Write the smoke numbers into the log (a table: cell, s/step, projected minutes, peak GiB; projected total). Then:
- `decision == "run_locally"`: no single run over 45 min, projected total training under 3 h, and peak under
  20 GiB. Task 4 runs on the local RTX 3090.
- `decision == "ask_user_for_das6_node"`: **the controller stops and asks the user to reserve a DAS6 node.** Once
  the user confirms, the full runs go through the `cluster-run` skill (invoke it; follow it for syncing code and
  data, launching `run_grid.py --prepare` then `--run CELL --seed S`, and pulling `checkpoints/` and `results/`
  back into this folder). `--evaluate` can run locally on the pulled checkpoints.

Smoke checkpoints (`*_smoke*.pt`) are discarded; they are never evaluated.

- [ ] **Step 6: Commit**

```bash
git add src/test/20261016_factor_learning_grid/run_grid.py src/test/20261016_factor_learning_grid/.gitignore \
        src/test/20261016_factor_learning_grid/20261016_factor_learning_grid_log.md
git commit -m "feat(v2): factor-learning grid script (prepare, cells, timing smoke)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP"
```

---

### Task 4: Full runs, selection evaluation, and the selection report (STOP POINT: gates / no qualifying cell)

**Files:**
- Modify: `src/test/20261016_factor_learning_grid/run_grid.py` (add `evaluate`, `tables`)
- Modify: `src/test/20261016_factor_learning_grid/20261016_factor_learning_grid_log.md`
- Create: `docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md`,
  `docs/reports/assets/build_2026-10-16_factor_learning_figures.py`, `docs/reports/assets/2026-10-16_factor_learning/*.png`
- Modify: `docs/reports/reports_sum.md` (own lines only; see Global Constraints)

**Interfaces:**
- Consumes (Task 3): `load_grid`, `run_cell`, `gate_report`, `CELLS`; the headroom probe's helpers, imported
  via importlib from `src/test/20261015_factor_headroom_probe/run_probe.py`: `hits`, `r1_points`,
  `r1_with_ci`, `r1_diff`, `fixed_weight_ranks`, `oracle_ranks`, `key`, `BETAS`, `CHANCE_R1`. Also
  `label_episode_weights` and `standard_label_episodes`.
- Produces for Tasks 5-6:
  - `results/selection_results.json`: per model, the gates (values, passed), R@1 blocks, `D`, `D_emotion`,
    the rule outcome with `picked`, and the β grid.
  - `results/selection_ranks.npz`.
  - `apply_rule(gates_ok, d, d_emotion) -> dict` with keys `stop`, `picked`, `qualifying`, `tie_band`.

- [ ] **Step 1: Train the four cells (seed 42)**

Locally, run the four `--run` commands as parallel OS processes if `4 × peak_gpu_gib ≤ 20`, else two at a time:

```bash
cd /project/CoSiR
for c in C0 A S AS; do
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261016_factor_learning_grid/run_grid.py --run $c --seed 42 \
    > src/test/20261016_factor_learning_grid/run_${c}_seed42.log 2>&1 &
done
wait
```

On DAS6, launch the same four commands through `cluster-run`. Expected: four checkpoints, finite histories.

- [ ] **Step 2: Implement `apply_rule` with its hand checks**

```python
CHANGES = {"A": 1, "S": 1, "AS": 2}
TIE_POINTS, GUARD_POINTS = 0.5, -1.0


def apply_rule(gates_ok: dict, d: dict, d_emotion: dict) -> dict:
    """Spec §6. Eligible = all gates pass; qualifies = eligible and D lower bound > 0 and D_emotion lower bound
    > -1.0; pick = highest D, cells within 0.5 points tie, a tie goes to fewer changes (A or S before AS),
    then to higher D. ``d`` / ``d_emotion`` map cell -> {"point": R@1 points, "ci95": [lo, hi]}."""
    if not gates_ok["C0"]:
        return {"stop": "C0 fails the gates: the setup is broken", "picked": None, "qualifying": [], "tie_band": []}
    qualifying = [c for c in ("A", "S", "AS")
                  if gates_ok[c] and d[c]["ci95"][0] > 0 and d_emotion[c]["ci95"][0] > GUARD_POINTS]
    if not qualifying:
        return {"stop": "no cell qualifies", "picked": None, "qualifying": [], "tie_band": []}
    best = max(d[c]["point"] for c in qualifying)
    tied = [c for c in qualifying if d[c]["point"] >= best - TIE_POINTS]
    picked = min(tied, key=lambda c: (CHANGES[c], -d[c]["point"]))
    return {"stop": None, "picked": picked, "qualifying": qualifying, "tie_band": tied}


def _check_rule() -> None:
    ok = {"C0": True, "A": True, "S": True, "AS": True}
    blk = lambda p, lo: {"point": p, "ci95": [lo, p + 1]}          # noqa: E731
    fine = {c: blk(0.0, -0.5) for c in CHANGES}
    assert apply_rule({**ok, "C0": False}, fine, fine)["stop"].startswith("C0")
    assert apply_rule(ok, {c: blk(1.0, -0.1) for c in CHANGES}, fine)["stop"] == "no cell qualifies"
    d = {"A": blk(2.0, 0.5), "S": blk(1.2, 0.1), "AS": blk(2.4, 0.9)}
    assert apply_rule(ok, d, fine)["picked"] == "A"                  # AS within 0.5 of best -> fewer changes
    d = {"A": blk(1.0, 0.2), "S": blk(1.3, 0.3), "AS": blk(2.4, 0.9)}
    assert apply_rule(ok, d, fine)["picked"] == "AS"                 # A and S fall outside the tie band
    assert apply_rule(ok, d, {**fine, "AS": blk(-2.0, -3.0)})["picked"] == "S"   # emotion guard drops AS
```

Call `_check_rule()` at the start of `evaluate()`.

- [ ] **Step 3: Implement `evaluate()`**

Structure (every helper named here exists in the probe script or in `src/`):

```python
MODELS = ("R3", "C0", "A", "S", "AS")                        # R3 = original R3 codes from stage (d)'s cache


def model_codes(name: str, data, cache) -> tuple[np.ndarray, np.ndarray]:
    """Full-length (n_rows, 32) codes: finite on scorer-train and selection rows, NaN elsewhere."""
    rows = np.concatenate([cache["scorer_train"], cache["selection"]])
    if name == "R3":
        return (sel.masked(cache["img_codes"], rows), sel.masked(cache["txt_codes"], rows))
    model, _ = load_factor_checkpoint(CKPT / f"{name}_seed{SEED}.pt", device=DEVICE)
    ic, tc = encode_rows(model, data.img_features, data.txt_features, rows=rows)
    out = []
    for codes in (ic, tc):
        full = np.full((len(cache["groups"]), codes.shape[1]), np.nan, dtype=np.float32)
        full[rows] = codes
        out.append(full)
    return tuple(out)
```

`evaluate()` does the following, in order:
1. `_check_rule()`.
2. Load `data`, `load_grid()`.
3. Build the selection episodes exactly as the probe does (`standard_label_episodes(data, cache["groups"],
   cache["selection"], label, 2048, seed=42)` per label). Assert the SHA-256s against `fin.SELECTION_SHA256` and
   that every episode row is a selection row. Build the null targets as the probe does
   (`np.random.default_rng(42)`, `integers(1, 13, n)` per label in `("emotion", "art_style")` order).
4. CLIP features masked to selection rows: `img, txt = (sel.masked(x, cache["selection"]) for x in
   (data.img_features, data.txt_features))`.
5. For every model in `MODELS`:
   - codes via `model_codes`;
   - gates via `gate_report(fit codes on scorer-train rows, eval codes on selection rows, ...,
     reference=meta["r0_readout_reference"])` (for R3 too, as a reference row);
   - naive ranks at every β in `BETAS` via `fixed_weight_ranks(img, txt, ic_sel, tc_sel, episodes, weights, β)`,
     where `ic_sel, tc_sel` are the codes masked to selection rows and `weights[label] =
     label_episode_weights(ic_sel, tc_sel, episodes[label])`;
   - label-oracle ranks at β 0.3 and β 0 via `oracle_ranks(..., steps=200, device=DEVICE)`, and the oracle null
     at β 0.
6. CLIP-only ranks (zero weights, β 0.3).
7. `d[X] = r1_diff(naive(X, 0.3), naive(C0, 0.3))["pooled"]["mean"]` and
   `d_emotion[X] = r1_diff(...)["emotion"]["mean"]` for X in A, S, AS. These blocks are in R@1 points with
   `"point"` and `"ci95"`. Also `r1_diff` of every model vs original R3 naive at 0.3, as context.
8. `rule = apply_rule({m: gates[m].all_passed for m in ("C0", "A", "S", "AS")}, d, d_emotion)`.
9. Save `results/selection_results.json`: episodes meta, gates (values and passed), R@1 for every
   (scorer, model, β), headline R@1 with CIs at β 0.3, `d`, `d_emotion`, the vs-R3 context, the rule, the
   histories' final condition loss and τ, and timings. Save `results/selection_ranks.npz`.
10. Log the verdict.

`--tables` reprints: the gates table (9 gates × 5 models), naive R@1 at β 0.3 per scope and direction, the
`D` / `D_emotion` table with CIs, the β-grid table, the label-oracle table with nulls, and the rule outcome.

- [ ] **Step 4: Run the evaluation**

Run: `/root/miniconda3/envs/CoSiR/bin/python src/test/20261016_factor_learning_grid/run_grid.py --evaluate 2>&1 | tee src/test/20261016_factor_learning_grid/run_evaluate.log`
Expected: SHA asserts pass; C0's gates are printed; the rule outcome is printed.

- [ ] **Step 5: STOP POINT check**

If `rule["stop"]` is not `None` (C0 fails a gate, or no cell qualifies), finish Steps 6-7 (report and commit)
with the stop verdict, then **stop and report to the controller. Task 5 and Task 6 are not run.**

- [ ] **Step 6: Write the selection report and figures**

Report `docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md`, verdict first. Required
content:
- the rule outcome, plus `D` and `D_emotion` for A, S and AS with CIs (baseline C0);
- each cell's plain naive R@1 at β 0.3 next to C0 and original R3;
- the gates table;
- the label oracle vs naive per model (does any cell raise the oracle above R3's 20.5%?);
- per label and per direction;
- the β grid (does a gain come from the code scale acting like β?);
- the condition loss and τ over training;
- caveats from spec §11;
- the smoke timing and where the runs ran.

Figures, built by `docs/reports/assets/build_2026-10-16_factor_learning_figures.py` from
`results/selection_results.json` into `docs/reports/assets/2026-10-16_factor_learning/`:
1. naive R@1 per model and label type with CIs and the C0 / R3 / CLIP-only / chance lines;
2. `D` and `D_emotion` per cell with CIs and the 0 and −1.0 reference lines.

Use categorical slots `#2a78d6` and `#eb6834` (and `#1baf7a` if a third series is needed) on a white surface. Check
the rendered PNGs for label collisions. Add the `reports_sum.md` row (date `10-16`) and update the "Current work
(v2)" line, staging only your own lines. Then run `scripts/check_reports_sum.py` and confirm it prints OK.

- [ ] **Step 7: Commit**

```bash
git add src/test/20261016_factor_learning_grid/run_grid.py \
        src/test/20261016_factor_learning_grid/20261016_factor_learning_grid_log.md \
        docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md \
        docs/reports/assets/build_2026-10-16_factor_learning_figures.py docs/reports/assets/2026-10-16_factor_learning/
# plus reports_sum.md own lines via git update-index (Global Constraints)
git commit -m "docs(v2): factor-learning 2x2 selection (C0 / A / S / AS on selection rows)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP"
```

---

### Task 5: Replication seeds 43 and 44 (picked cell and C0)

**Files:**
- Modify: `src/test/20261016_factor_learning_grid/run_grid.py` (add `--replicate`)
- Modify: the grid log, and the selection report (a "Replication" section)

**Interfaces:**
- Consumes (Task 4): `results/selection_results.json` (`rule.picked`), `run_cell`, the evaluation helpers.
- Produces for Task 6: checkpoints `{picked}_seed{43,44}.pt` and `C0_seed{43,44}.pt`; `results/replication.json`
  with, per seed, `D`, `D_emotion`, gates for both models, and naive R@1 at β 0.3.

- [ ] **Step 1: Train the four replication runs**

`run_cell(picked, s)` and `run_cell("C0", s)` for s in (43, 44), in parallel if memory allows. On DAS6, use
`cluster-run` as in Task 4.

- [ ] **Step 2: Implement and run `--replicate`**

It evaluates the four replication checkpoints on the same selection episodes with the same code path as
`evaluate()` (gates, naive at β 0.3, `D` and `D_emotion` of picked seed s vs C0 seed s). It writes
`results/replication.json`.

Run: `/root/miniconda3/envs/CoSiR/bin/python src/test/20261016_factor_learning_grid/run_grid.py --replicate`
Expected: finite results for both seeds. These are reported only; the verdict rests on seed 42.

- [ ] **Step 3: Update the report and commit**

Add the Replication section (a table per seed: `D`, `D_emotion`, gates passed, R@1). Then:

```bash
git add src/test/20261016_factor_learning_grid/run_grid.py \
        src/test/20261016_factor_learning_grid/20261016_factor_learning_grid_log.md \
        docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md
git commit -m "docs(v2): factor-learning replication seeds 43/44 (picked cell and C0)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP"
```

---

### Task 6: Held test (power, smoke, one run) and the held report

**Files:**
- Create: `src/test/20261017_factor_learning_held/run_held.py`,
  `src/test/20261017_factor_learning_held/20261017_factor_learning_held_log.md`,
  `src/test/20261017_factor_learning_held/.gitignore`,
  `docs/reports/auto/v2/2026-10-17_candidate_a_factor_learning_held.md`
- Modify: `docs/reports/reports_sum.md` (own lines only), `docs/reports/assets/build_2026-10-16_factor_learning_figures.py`
  (one held figure)

**Interfaces:**
- Consumes: the grid's `selection_results.json` (picked cell, its `D` block), `replication.json`, the checkpoints
  `{picked,C0}_seed{42,43,44}.pt`, and the original R3 checkpoint
  `src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt` (SHA-256
  `1c299fc008ca006ffd2de37315ac3557523bf06e0b755d4dcbf70b999b4e453f`). Also the probe's helpers.
- Produces: `results/power.json`, `results/smoke_held.json` (discarded numbers), and `results/held_results.json`
  (written once).

- [ ] **Step 1: Power (`--power`)**

```python
from scipy.stats import norm

N_CHOICES, SHRINK, TARGET_POWER, N_SELECTION = (2048, 4096, 8192), 0.75, 0.8, 2048


def held_episode_count(d_point: float, d_ci95: list[float]) -> tuple[int, dict]:
    """Spec §7: SE from the selection CI, effect shrunk by 0.75, smallest n per label with power >= 0.8."""
    se = (d_ci95[1] - d_ci95[0]) / (2 * 1.959964)
    effect = SHRINK * d_point
    table = {n: float(norm.cdf(effect / (se * (N_SELECTION / n) ** 0.5) - 1.959964)) for n in N_CHOICES}
    chosen = next((n for n in N_CHOICES if table[n] >= TARGET_POWER), N_CHOICES[-1])
    return chosen, table
```

Hand check, asserted in the script: for `d_point=2.0` and `d_ci95=[1.0, 3.0]`, SE = 0.5102 and effect = 1.5, so
power at 2048 = Φ(2.940 − 1.960) = 0.836 and `chosen == 2048`.

Run `--power`, then write `results/power.json` (n, table, inputs) and log it **before** any held read.

- [ ] **Step 2: Smoke (`--smoke`) on selection rows**

This is the same code path as `--run`, with the selection rows in place of the held rows (seed 43, n = the chosen
n). It writes `results/smoke_held.json`, whose numbers are discarded. Confirm it runs end to end and that its
outputs are finite.

- [ ] **Step 3: The held run (`--run`, once)**

`--run` refuses to start if `results/held_results.json` exists. It does the following:
1. Loads the data and recomputes `grouped_split(leakage_groups(...), seed=42)`. It asserts the split sizes and
   that `split.train` equals stage (d)'s cache `split_train`.
2. Builds `held = split.held`, encodes held rows only, with the picked cell and C0 (seeds 42, 43, 44) and the
   original R3 checkpoint (SHA asserted), and masks the CLIP features to held rows.
3. Builds `standard_label_episodes(data, groups, held, label, n, seed=43)` per label, records their SHA-256s,
   and asserts every row is a held row.
4. Computes naive ranks at β 0.3 for all models, CLIP-only, and the label oracle and its null (β 0.3) for the
   picked cell, C0 and R3.
5. Computes `D_held = r1_diff(picked, C0)["pooled"]["mean"]` and `D_emotion,held`. The verdict is "confirmed" iff
   `D_held` lower bound > 0 and `D_emotion,held` lower bound > −1.0 (seed 42). It also computes the seeds 43/44
   `D_held` and the context rows vs original R3 naive.
6. Writes `results/held_results.json` and `results/held_ranks.npz`.

Run: `/root/miniconda3/envs/CoSiR/bin/python src/test/20261017_factor_learning_held/run_held.py --run 2>&1 | tee src/test/20261017_factor_learning_held/run_held.log`

- [ ] **Step 4: Held report, figure, index, commit**

Report `docs/reports/auto/v2/2026-10-17_candidate_a_factor_learning_held.md`, verdict first. It covers:
- confirmed or not, with `D_held` and `D_emotion,held` against C0;
- the plain R@1 of the picked cell, C0 and original R3 (the current system), plus CLIP-only;
- per label and per direction;
- the label oracle;
- the seeds;
- the power table;
- the disclosure of earlier held-row reads (spec §7);
- the caveats (spec §11).

Add one held figure (R@1 per model and label type with CIs) to the figure script. Then add the `reports_sum.md`
row (date `10-17`) and update "Current work (v2)", staging only your own lines, and confirm
`scripts/check_reports_sum.py` prints OK.

```bash
git add src/test/20261017_factor_learning_held/run_held.py src/test/20261017_factor_learning_held/.gitignore \
        src/test/20261017_factor_learning_held/20261017_factor_learning_held_log.md \
        docs/reports/auto/v2/2026-10-17_candidate_a_factor_learning_held.md \
        docs/reports/assets/build_2026-10-16_factor_learning_figures.py docs/reports/assets/2026-10-16_factor_learning/
# plus reports_sum.md own lines via git update-index
git commit -m "docs(v2): factor-learning held test (picked cell vs C0 on fresh seed-43 held episodes)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP"
```

---

## After the last task (controller)

- Run an Opus whole-branch final review over this plan's commits, then ONE fix wave. Held rows are never re-read
  in the fix wave; any held-side control is offered to the user as an option.
- Update the memory file `project_next-step-factor-learning.md` with the outcome and the next decision for the
  user (for example, an emotion signal, or stage (e)).
