# Buddy-Graph Prototype Conditioning (Experiment 18) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the free per-sample trainable condition vector with a small, differentiable, buddy-graph-community-seeded prototype bank + attention pooling, validated on RedCaps-150k against the existing architecture-fix-only baseline, on DAS6.

**Architecture:** A learnable `PrototypeBank` (P key/value prototypes) is seeded once from Leiden communities detected over the existing buddy `union_graph`. Each sample's condition vector becomes a differentiable attention-pooled combination of prototype values, computed inside the same forward pass that feeds the (unchanged) `CombinerLowRankAdapter` and is regularized by the (unchanged) `buddy_contrastive_loss`. Gated behind an additive `model.conditioning_mode` flag whose default (`"free_vector"`) preserves every existing code path byte-for-byte.

**Tech Stack:** PyTorch, Hydra configs, scipy.sparse (buddy graph, unchanged), `leidenalg`/`python-igraph` (new, CPU community detection — not `cuml`), sklearn (KMeans coarsening, silhouette reuse), DAS6 (`node411`/`node412`) via the project's `cluster-run` skill.

**Spec:** `docs/superpowers/specs/2026-09-15-buddy-prototype-conditioning-design.md` (read this too — it has the full architecture rationale and the failure-mode analysis of the project's earlier, non-differentiable cluster-conditioning attempt).

## Global Constraints

- Isolate in its own `git worktree` off `experiment/condition_drift_retrieval_correlation`, branch `experiment/buddy_prototype_conditioning` — never work directly on the source branch (project convention, validated for Experiment 13).
- New behavior is opt-in via `model.conditioning_mode` (default `"free_vector"` = current behavior, byte-for-byte). Never change the default.
- `src/conditional_buddy/buddy_graph.py` is **unchanged** — only buddy-graph *construction* is fixed per the user's 2026-09-15 decision.
- No `cuml`/`cugraph` imports in any new file — this project's `libllvmlite.so` import chain is documented broken locally (CLAUDE.md); use `leidenalg`/`python-igraph`/sklearn instead.
- `seed=42` for every stochastic step (community detection, KMeans coarsening, torch inits) — project convention.
- Never `import src.model`/`CoSiRModel` from a standalone script outside the main training entrypoint — use raw HF `transformers` directly if a script needs CLIP outside `main_cosir.py`'s own path (same convention as Experiment 17.1).
- New standalone diagnostic/smoke scripts go under `src/test/YYYYMMDD_<name>/` (dated-debugging-folder convention).
- Implementation tasks route through Codex via the `ccg` `codeagent-wrapper --backend codex` bridge; Codex never launches or holds a GPU training run — only implementation and read-only monitoring.
- `cluster_launch.sh` (DAS6 GPU launch) always requires the controller to stop and get the user's explicit confirmation immediately before the call — never assume prior authorization covers a specific launch call.
- Every `cluster_sync_up.sh` / `cluster_launch.sh` / `cluster_sync_down.sh` invocation passes `--node node411` or `--node node412` explicitly (two nodes are reserved concurrently — `--detect-only` without `--node` will correctly error on the ambiguity).
- `cluster_sync_up.sh` requires a clean working tree and a "cluster run" mention (case-insensitive) in the branch name or the tip commit's subject line, or it refuses — since the branch name has no such marker, use a commit subject containing "cluster run" at the point code is pushed (see Task 7/8), not `--allow-any-branch`.

---

### Task 1: Isolate the experiment (worktree, branch, opt-in config flag skeleton)

**Files:**
- Create (via shell, not code): git worktree at `../CoSiR-buddy_prototype_conditioning`, branch `experiment/buddy_prototype_conditioning`
- Modify: `configs/model/clip_base.yaml`

**Interfaces:**
- Produces: `model.conditioning_mode` config key, default `"free_vector"`, consumed by Task 4.

- [ ] **Step 1: Create the isolated worktree**

```bash
cd /project/CoSiR
git worktree add ../CoSiR-buddy_prototype_conditioning -b experiment/buddy_prototype_conditioning experiment/condition_drift_retrieval_correlation
cd ../CoSiR-buddy_prototype_conditioning
ln -s /project/CoSiR/res res
ln -s /project/CoSiR/data data
```

Confirm both symlinks resolve (`ls -la res data`) and that `git status` in the new worktree is clean.

- [ ] **Step 2: Add the opt-in `conditioning_mode` flag**

Edit `configs/model/clip_base.yaml`, adding one line after `combiner_type: "legacy"`:

```yaml
  conditioning_mode: "free_vector"  # "free_vector" (default, unchanged) | "prototype_pooled" (Experiment 18)
```

- [ ] **Step 3: Run the existing test suite to confirm the default path is untouched**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/test_model.py
python src/test/test_manager.py
```

Expected: PASS, identical to pre-change behavior (the new config key is unread by any code yet).

- [ ] **Step 4: Commit**

```bash
git add configs/model/clip_base.yaml
git commit -m "feat(exp18): add conditioning_mode config flag (default unchanged)"
```

---

### Task 2: `PrototypeBank` module

**Files:**
- Create: `src/model/prototype_bank.py`
- Test: `src/test/test_prototype_bank.py`

**Interfaces:**
- Produces: `PrototypeBank(nn.Module)` — `__init__(num_prototypes: int, condition_dim: int, query_dim: int, temperature_init: float = 1.0)`, `forward(query_features: Tensor[B, query_dim]) -> Tensor[B, condition_dim]`, `seed_from_communities(community_means: Tensor[C, condition_dim]) -> None`, `usage_entropy() -> Tensor[scalar]`.

- [ ] **Step 1: Write the failing tests**

```python
# src/test/test_prototype_bank.py
import math
import torch
import pytest

from src.model.prototype_bank import PrototypeBank


def test_prototype_bank_forward_shape():
    bank = PrototypeBank(num_prototypes=8, condition_dim=16, query_dim=512)
    q = torch.randn(4, 512)
    out = bank(q)
    assert out.shape == (4, 16)


def test_prototype_bank_gradient_flows_to_keys_values_and_query_proj():
    bank = PrototypeBank(num_prototypes=8, condition_dim=16, query_dim=512)
    q = torch.randn(4, 512, requires_grad=True)
    out = bank(q)
    out.sum().backward()
    assert bank.keys.grad is not None and bank.keys.grad.abs().sum() > 0
    assert bank.values.grad is not None and bank.values.grad.abs().sum() > 0
    assert bank.query_proj.weight.grad is not None
    assert q.grad is not None


def test_seed_from_communities_overwrites_keys_and_values():
    bank = PrototypeBank(num_prototypes=4, condition_dim=3, query_dim=5)
    means = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bank.seed_from_communities(means)
    assert torch.allclose(bank.keys.data[:2], means)
    assert torch.allclose(bank.values.data[:2], means)


def test_seed_from_communities_raises_when_too_many_communities():
    bank = PrototypeBank(num_prototypes=2, condition_dim=3, query_dim=5)
    means = torch.randn(3, 3)
    with pytest.raises(ValueError):
        bank.seed_from_communities(means)


def test_usage_entropy_uniform_attention_near_max_entropy():
    bank = PrototypeBank(num_prototypes=8, condition_dim=16, query_dim=512)
    with torch.no_grad():
        bank.query_proj.weight.zero_()
        bank.query_proj.bias.zero_()
    q = torch.randn(4, 512)
    bank(q)
    assert bank.usage_entropy().item() == pytest.approx(math.log(8), abs=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/test/test_prototype_bank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.prototype_bank'`

- [ ] **Step 3: Implement**

```python
# src/model/prototype_bank.py
"""Differentiable prototype bank + attention pooling for buddy-graph conditioning.

Replaces a free per-sample condition vector: a sample's condition vector is a
softmax-weighted sum over a small set of learnable prototype values, with
attention computed from a query projected from the sample's frozen CLIP
feature. This is one differentiable forward pass — no separate non-
differentiable update step — see
docs/superpowers/specs/2026-09-15-buddy-prototype-conditioning-design.md §3
for why that distinction is the point of this design.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        condition_dim: int,
        query_dim: int,
        temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.condition_dim = condition_dim
        self.keys = nn.Parameter(torch.empty(num_prototypes, condition_dim))
        self.values = nn.Parameter(torch.empty(num_prototypes, condition_dim))
        nn.init.normal_(self.keys, std=0.02)
        nn.init.normal_(self.values, std=0.02)
        self.query_proj = nn.Linear(query_dim, condition_dim)
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(temperature_init)))
        )
        self._last_attn: Optional[torch.Tensor] = None

    @torch.no_grad()
    def seed_from_communities(self, community_means: torch.Tensor) -> None:
        """Overwrite keys/values from precomputed per-community mean features.

        community_means: [C, condition_dim]. If C < num_prototypes, the
        remaining prototype rows keep their random init (documented fallback,
        spec §6). If C > num_prototypes, raises — the caller must coarsen
        upstream (spec §6, src/conditional_buddy/prototype_seed.py's
        coarsen_to_prototype_count).
        """
        c = community_means.shape[0]
        if c > self.num_prototypes:
            raise ValueError(
                f"{c} community means but only {self.num_prototypes} prototype "
                "slots; coarsen community_means to num_prototypes rows first."
            )
        self.keys.data[:c] = community_means.to(self.keys.dtype)
        self.values.data[:c] = community_means.to(self.values.dtype)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """query_features: [B, query_dim] frozen CLIP features. Returns [B, condition_dim]."""
        q = self.query_proj(query_features)  # [B, D]
        temperature = self.log_temperature.exp().clamp(min=1e-3)
        logits = (q @ self.keys.t()) / temperature  # [B, P]
        attn = F.softmax(logits, dim=-1)  # [B, P]
        self._last_attn = attn.detach()
        return attn @ self.values  # [B, D]

    def usage_entropy(self) -> torch.Tensor:
        """Mean per-batch attention entropy (nats) — collapse monitor, spec §6.

        Low values relative to log(num_prototypes) mean attention is
        concentrating on a small subset of prototypes.
        """
        if self._last_attn is None:
            raise RuntimeError("call forward() before usage_entropy()")
        p = self._last_attn.clamp_min(1e-12)
        return -(p * p.log()).sum(dim=-1).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest src/test/test_prototype_bank.py -v`
Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
git add src/model/prototype_bank.py src/test/test_prototype_bank.py
git commit -m "feat(exp18): add PrototypeBank (differentiable attention-pooled conditioning)"
```

---

### Task 3: Buddy-graph community seeding

**Files:**
- Create: `src/conditional_buddy/prototype_seed.py`
- Test: `src/test/test_prototype_seed.py`
- Modify: `requirements.txt` (append `python-igraph`, `leidenalg`)

**Interfaces:**
- Consumes: `scipy.sparse.csr_matrix` from `buddy_graph.union_graph` (unchanged, Task-external — this task only consumes its output type, never imports/calls `buddy_graph.py` internals it doesn't need).
- Produces: `detect_communities(E: csr_matrix, seed: int = 42) -> np.ndarray`, `community_mean_features(labels, img_feats, txt_feats) -> np.ndarray`, `coarsen_to_prototype_count(community_means, num_prototypes, seed=42) -> np.ndarray`. Task 4 calls these three in sequence.

- [ ] **Step 1: Install and pin the new dependencies**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
pip install python-igraph leidenalg
```

Append to `requirements.txt` (check current diff first — it may already have unrelated pending changes; add these as new lines, don't touch anything else):

```
python-igraph
leidenalg
```

- [ ] **Step 2: Write the failing tests**

```python
# src/test/test_prototype_seed.py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.conditional_buddy.prototype_seed import (
    coarsen_to_prototype_count,
    community_mean_features,
    detect_communities,
)


def test_detect_communities_two_disconnected_cliques():
    # nodes 0,1,2 fully connected; nodes 3,4,5 fully connected; no cross edges.
    rows = [0, 0, 1, 3, 3, 4]
    cols = [1, 2, 2, 4, 5, 5]
    all_rows = rows + cols
    all_cols = cols + rows
    E = csr_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(6, 6))
    labels = detect_communities(E, seed=42)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_community_mean_features_matches_manual_average():
    labels = np.array([0, 0, 1])
    img = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    txt = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    means = community_mean_features(labels, img, txt)
    assert means.shape == (2, 2)
    np.testing.assert_allclose(means[0], [1.5, 0.0])
    np.testing.assert_allclose(means[1], [0.0, 1.0])


def test_community_mean_features_raises_on_missing_label():
    labels = np.array([0, 0, 2])  # label 1 never appears -> gap
    img = txt = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        community_mean_features(labels, img, txt)


def test_coarsen_to_prototype_count_noop_when_already_small():
    means = np.random.RandomState(42).randn(3, 4).astype(np.float32)
    out = coarsen_to_prototype_count(means, num_prototypes=8)
    np.testing.assert_array_equal(out, means)


def test_coarsen_to_prototype_count_reduces_row_count():
    means = np.random.RandomState(42).randn(20, 4).astype(np.float32)
    out = coarsen_to_prototype_count(means, num_prototypes=5, seed=42)
    assert out.shape == (5, 4)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest src/test/test_prototype_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.conditional_buddy.prototype_seed'`

- [ ] **Step 4: Implement**

```python
# src/conditional_buddy/prototype_seed.py
"""CPU Leiden community detection over the buddy union graph, for PrototypeBank
seeding (Experiment 18). Deliberately avoids cuml/cugraph — this project's
libllvmlite.so import chain is documented broken locally (CLAUDE.md) — using
leidenalg + python-igraph instead, both pure CPU, no RAPIDS dependency.

See docs/superpowers/specs/2026-09-15-buddy-prototype-conditioning-design.md §3-4.
"""
from typing import Optional

import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse import csr_matrix


def detect_communities(E: csr_matrix, seed: int = 42) -> np.ndarray:
    """Leiden community detection (modularity objective) over a binary union graph.

    E: (N, N) symmetric binary sparse adjacency (buddy_graph.union_graph's output).
    Returns: (N,) int64 array of community labels, 0-indexed, no gaps.
    """
    E_coo = E.tocoo()
    mask = E_coo.row < E_coo.col  # undirected: one edge per pair
    edges = list(zip(E_coo.row[mask].tolist(), E_coo.col[mask].tolist()))
    g = ig.Graph(n=E.shape[0], edges=edges)
    partition = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, seed=seed,
    )
    return np.array(partition.membership, dtype=np.int64)


def community_mean_features(
    labels: np.ndarray, img_feats: np.ndarray, txt_feats: np.ndarray,
) -> np.ndarray:
    """Per-community mean of the (img, txt) feature average.

    labels: (N,) from detect_communities, 0-indexed, no gaps.
    img_feats/txt_feats: (N, D), L2-normalized, same D as PrototypeBank's
    condition_dim (see Task 4 for the projection that ensures this).
    Returns: (C, D) float32, C = labels.max() + 1.
    """
    n_communities = int(labels.max()) + 1
    mean_feat = 0.5 * (img_feats + txt_feats)
    sums = np.zeros((n_communities, mean_feat.shape[1]), dtype=np.float32)
    counts = np.zeros(n_communities, dtype=np.int64)
    np.add.at(sums, labels, mean_feat)
    np.add.at(counts, labels, 1)
    if (counts == 0).any():
        raise ValueError("every community label 0..C-1 must have at least one member")
    return sums / counts[:, None]


def coarsen_to_prototype_count(
    community_means: np.ndarray, num_prototypes: int, seed: int = 42,
) -> np.ndarray:
    """KMeans-coarsen community means down to num_prototypes rows (spec §6).

    No-op (returns community_means unchanged) if it already has
    <= num_prototypes rows.
    """
    if community_means.shape[0] <= num_prototypes:
        return community_means
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=num_prototypes, random_state=seed, n_init=10)
    km.fit(community_means)
    return km.cluster_centers_.astype(np.float32)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest src/test/test_prototype_seed.py -v`
Expected: PASS (5/5)

- [ ] **Step 6: Commit**

```bash
git add src/conditional_buddy/prototype_seed.py src/test/test_prototype_seed.py requirements.txt
git commit -m "feat(exp18): add buddy-graph community seeding (CPU Leiden)"
```

---

### Task 4: Wire `conditioning_mode` into `CoSiRModel` and initialization

**Files:**
- Modify: `src/model/cosirmodel.py` (constructor, around the `combiner_classes` dict at lines 95-105)
- Modify: `src/hook/train_cosir.py` (initialization region around the `embedding_manager.initialize_embeddings_buddies(...)` call, ~line 325)
- Test: `src/test/test_cosirmodel_prototype_mode.py`

**Interfaces:**
- Consumes: `PrototypeBank` (Task 2), `detect_communities`/`community_mean_features`/`coarsen_to_prototype_count` (Task 3).
- Produces: `CoSiRModel.prototype_bank: Optional[PrototypeBank]` (`None` when `conditioning_mode="free_vector"`, else a constructed `PrototypeBank`), `CoSiRModel.conditioning_mode: str`.

- [ ] **Step 1: Write the failing test**

```python
# src/test/test_cosirmodel_prototype_mode.py
import torch

from src.model.cosirmodel import CoSiRModel


def test_free_vector_mode_has_no_prototype_bank():
    model = CoSiRModel(conditioning_mode="free_vector", label_dim=16)
    assert model.prototype_bank is None


def test_prototype_pooled_mode_constructs_prototype_bank_with_right_dims():
    model = CoSiRModel(
        conditioning_mode="prototype_pooled",
        label_dim=16,
        num_prototypes=8,
    )
    assert model.prototype_bank is not None
    assert model.prototype_bank.num_prototypes == 8
    assert model.prototype_bank.condition_dim == 16
    assert model.prototype_bank.query_proj.in_features == model.feature_dim


def test_invalid_conditioning_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        CoSiRModel(conditioning_mode="not_a_real_mode", label_dim=16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/test/test_cosirmodel_prototype_mode.py -v`
Expected: FAIL — `CoSiRModel.__init__() got an unexpected keyword argument 'conditioning_mode'`

- [ ] **Step 3: Implement — `cosirmodel.py`**

In `src/model/cosirmodel.py`, add the import at the top (alongside the existing `.combiner` import block):

```python
from .prototype_bank import PrototypeBank
```

In `CoSiRModel.__init__`, add two new parameters after `combiner_type: str = "legacy",`:

```python
        conditioning_mode: str = "free_vector",
        num_prototypes: int = 16,
```

Immediately after the existing `self.combine_side = combine_side` assignment (right after the `combiner_type` dispatch block, before `self.other_proj`), add:

```python
        if conditioning_mode not in ("free_vector", "prototype_pooled"):
            raise ValueError(
                f"conditioning_mode must be 'free_vector' or 'prototype_pooled', "
                f"got '{conditioning_mode}'"
            )
        self.conditioning_mode = conditioning_mode
        if conditioning_mode == "prototype_pooled":
            self.prototype_bank = PrototypeBank(
                num_prototypes=num_prototypes,
                condition_dim=label_dim,
                query_dim=self.feature_dim,
            )
        else:
            self.prototype_bank = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/test/test_cosirmodel_prototype_mode.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Wire config + seeding call site in `train_cosir.py`**

In `configs/model/clip_base.yaml`, add (sibling to `conditioning_mode` added in Task 1):

```yaml
  num_prototypes: 16
```

In `src/hook/train_cosir.py`, `model = CoSiRModel(...)` construction call (find it — it passes `combiner_type=cfg.model.combiner_type` already) — add two more kwargs:

```python
        conditioning_mode=cfg.model.conditioning_mode,
        num_prototypes=cfg.model.num_prototypes,
```

Immediately after the existing `if _need_initialize:` / `strategy == "buddies"` block (~line 325, the `embedding_manager.initialize_embeddings_buddies(...)` call already there and unchanged), add a new conditional seeding step. This must reuse the same buddy graph `E` (union graph) that `initialize_embeddings_buddies` itself builds internally — read `TrainableEmbeddingManager.initialize_embeddings_buddies`'s implementation first (`src/utils/embedding_manager.py`) to find its already-computed `E`/img_n/txt_n rather than rebuilding them a second time; if it does not already expose these as return values or instance attributes, add a minimal `return_graph: bool = False` opt-in parameter to that method (default `False`, unchanged for every other caller) that additionally returns `(E, img_n, txt_n)` when `True`, and pass `return_graph=True` only from this new call site:

```python
        if cfg.model.conditioning_mode == "prototype_pooled":
            from src.conditional_buddy.prototype_seed import (
                coarsen_to_prototype_count,
                community_mean_features,
                detect_communities,
            )

            labels = detect_communities(E, seed=cfg.seed)
            community_means = community_mean_features(labels, img_n, txt_n)
            community_means = coarsen_to_prototype_count(
                community_means, num_prototypes=cfg.model.num_prototypes, seed=cfg.seed,
            )
            model.prototype_bank.seed_from_communities(
                torch.from_numpy(community_means)
            )
            print(
                f"[prototype-seed] {len(set(labels.tolist()))} communities -> "
                f"{cfg.model.num_prototypes} prototypes"
            )
```

- [ ] **Step 6: Run full existing test suite (regression check on default path)**

```bash
python src/test/test_model.py
python src/test/test_manager.py
python -m pytest src/test/test_cosirmodel_prototype_mode.py -v
```

Expected: all PASS. `conditioning_mode="free_vector"` (the untouched default) must produce identical behavior to before this task.

- [ ] **Step 7: Commit**

```bash
git add src/model/cosirmodel.py src/hook/train_cosir.py configs/model/clip_base.yaml src/test/test_cosirmodel_prototype_mode.py src/utils/embedding_manager.py
git commit -m "feat(exp18): wire conditioning_mode into CoSiRModel + buddy-graph prototype seeding"
```

---

### Task 5: Training-loop integration + collapse monitoring

**Files:**
- Modify: `src/hook/train_cosir.py` (batch loop, ~line 1587-1590)

**Interfaces:**
- Consumes: `model.prototype_bank`, `model.conditioning_mode` (Task 4).

- [ ] **Step 1: Locate and confirm the current anchor**

Before editing, re-read `src/hook/train_cosir.py` around the comment `# Differentiable slice — gradients flow back to embedding_manager.embeddings` (found at implementation time around line 1587) to confirm the surrounding code still matches this plan's description — the file may have shifted since this plan was written. If the anchor has moved or changed shape, treat this as a BLOCKED report (not a silent improvisation) and let the controller re-ground the task.

- [ ] **Step 2: Replace the label_embeddings source, gated by conditioning_mode**

Original code:

```python
            # Differentiable slice — gradients flow back to embedding_manager.embeddings
            batch_indices = [embedding_manager.id_to_index[sid] for sid in batch_sample_ids]
            label_embeddings_before = embedding_manager.embeddings.data[batch_indices].clone()
            label_embeddings = embedding_manager.embeddings[batch_indices]
```

Replace with:

```python
            if model.conditioning_mode == "prototype_pooled":
                # Differentiable attention-pooling — gradients flow into
                # model.prototype_bank, not into a per-sample parameter table.
                # Query = mean of img/txt frozen features, matching the same
                # convention prototype_seed.community_mean_features used to
                # seed the bank (Task 3/4) — keeps seeding and querying
                # consistent.
                query_features = 0.5 * (img_features + txt_features)
                label_embeddings = model.prototype_bank(query_features)
                label_embeddings_before = label_embeddings.detach().clone()
                batch_indices = None  # unused downstream in this mode
            else:
                # Differentiable slice — gradients flow back to embedding_manager.embeddings
                batch_indices = [embedding_manager.id_to_index[sid] for sid in batch_sample_ids]
                label_embeddings_before = embedding_manager.embeddings.data[batch_indices].clone()
                label_embeddings = embedding_manager.embeddings[batch_indices]
```

Every later use of `batch_indices` in this loop (e.g. the oracle-weighting/gradient-scaling block around lines 1848-1863, and the `embedding_manager.embeddings.data[batch_indices] = ...` renormalization) must be inside an `if model.conditioning_mode == "free_vector":` guard, or reached only when `batch_indices is not None` — read each such use-site before editing and confirm it's guarded; these sites write directly into `embedding_manager.embeddings`, which does not exist as a per-sample table in `prototype_pooled` mode, so leaving them unguarded would crash.

- [ ] **Step 3: Add collapse-monitoring logging, gated the same way**

Find the per-batch or per-epoch `logger.log_train(...)` call already used for other training metrics (e.g. the `_refresh_stats` logging seen near the epoch-start buddy-refresh block). Add, inside the batch loop, only when `model.conditioning_mode == "prototype_pooled"`:

```python
            if model.conditioning_mode == "prototype_pooled" and batch_idx % 50 == 0:
                logger.log_train(
                    {"prototype_usage_entropy": model.prototype_bank.usage_entropy().item()},
                    epoch=epoch,
                    section="prototype_conditioning",
                )
```

- [ ] **Step 4: Regression + smoke test**

```bash
python src/test/test_model.py
python src/test/test_manager.py
```

Expected: PASS — `free_vector` mode's branch is untouched logic, just moved under an `if`.

A `prototype_pooled`-mode functional smoke test happens in Task 6 (requires real cached features, not a unit-test fixture).

- [ ] **Step 5: Commit**

```bash
git add src/hook/train_cosir.py
git commit -m "feat(exp18): integrate PrototypeBank into the training batch loop"
```

---

### Task 6: DAS6 dataset config + local-GPU smoke test

**Files:**
- Create: `configs/dataset/redcaps_150k_cluster.yaml`
- Create: `src/test/20260915_prototype_conditioning_smoke/run_smoke.sh`, `src/test/20260915_prototype_conditioning_smoke/README.md`

**Interfaces:**
- Produces: a validated `redcaps_150k_cluster` dataset name and a chosen `num_prototypes` value, both consumed by Task 7/8's launch commands.

- [ ] **Step 1: Create the DAS6-ready RedCaps-150k config**

```yaml
# configs/dataset/redcaps_150k_cluster.yaml
# DAS6 node variant of redcaps_150k.yaml — real node paths, not local /data/PDD paths.
# NOTE: train_annotation_path below is a prediction from redcaps_500k_diverse_cluster.yaml's
# pattern (same redcaps_plus family, same test_annotation_path/image paths already used
# there) — it has NOT been confirmed to exist on the node from this container. Task 7's
# first sync is the actual verification; if it 404s, this file needs a path correction,
# not a workaround.
dataset:
  name: redcaps_150k_cluster
  train_annotation_path: "/var/scratch/wding/Dataset/redcaps_plus/redcaps_150k.json"
  test_annotation_path: "/var/scratch/wding/Dataset/redcaps_plus/redcaps_test.json"
  train_image_path: "/var/scratch/wding/Dataset/redcaps_plus/images"
  test_image_path: "/var/scratch/wding/Dataset/redcaps_plus/images"

experiment:
  results_dir: "/local/wding/res/CoSiR_Experiment/redcaps_150k_cluster"
```

Read `configs/dataset/redcaps_500k_diverse_cluster.yaml` in full first and match every other key it sets (featuremanager paths, any dataset-specific overrides beyond the four paths + results_dir shown above) — the snippet above is the minimum; copy the full structure, only substituting the 150k-specific values.

- [ ] **Step 2: Local-GPU smoke test — verify gradients flow and pick `num_prototypes`**

```bash
# src/test/20260915_prototype_conditioning_smoke/run_smoke.sh
#!/bin/bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
cd /project/CoSiR-buddy_prototype_conditioning

for P in 8 16 32; do
  echo "=== num_prototypes=$P ==="
  python main_cosir.py \
    dataset=redcaps_150k \
    model=clip_base \
    model.conditioning_mode=prototype_pooled \
    model.num_prototypes=$P \
    train.initialization_strategy=buddies \
    train.epochs=1 \
    train.max_train_samples=1500 \
    eval.evaluation_interval=1 \
    seed=42 \
    experiment.results_dir=/tmp/exp18_smoke_p${P} \
    wandb.mode=disabled
done
```

Check `train.max_train_samples` (or the nearest equivalent config key for capping dataset size — grep `configs/train/` for the actual key name, it may differ from this guess) actually limits the run to ~1-2k samples per the spec's smoke-test scope; if no such key exists yet, this is a small addition to the training config/dataloader, not a blocker — a `Subset` wrap is the minimal fix.

For each `P`, confirm from the run's stdout/wandb-disabled log: (a) no NaN losses, (b) `prototype_usage_entropy` logged and not collapsing toward 0 over the 1-epoch run (compare against `log(P)` — the max-entropy reference from Task 2's test). Record the three entropy trajectories in `src/test/20260915_prototype_conditioning_smoke/README.md` and pick the `num_prototypes` value carried into Task 7/8 (default to 16 if all three look healthy — matches `embedding_dim`, no other reason to prefer a different P without evidence).

- [ ] **Step 3: Commit**

```bash
git add configs/dataset/redcaps_150k_cluster.yaml src/test/20260915_prototype_conditioning_smoke/
git commit -m "feat(exp18): add redcaps_150k_cluster config + local-GPU smoke test"
```

---

### Task 7: DAS6 first-launch validation (single short run, one node)

**Files:** none (operational task — verifies the cluster-run pathway itself, which per `.claude/skills/cluster-run/SKILL.md`'s own "State of this setup" section has never carried a real `main_cosir.py` training launch end-to-end before this).

- [ ] **Step 1: Commit with a "cluster run" marker**

```bash
git commit --allow-empty -m "chore(exp18): cluster run — ready for first DAS6 validation launch"
```

- [ ] **Step 2: Detect the reserved nodes**

```bash
./cluster_launch.sh --detect-only --node node411
./cluster_launch.sh --detect-only --node node412
```

- [ ] **Step 3: Sync code (and the redcaps_150k feature cache) to node411**

```bash
./cluster_sync_up.sh ./cluster_sync.conf redcaps_150k --node node411
```

Take the exit code seriously (HEAD-SHA and byte-count verification, per the cluster-run skill) — do not proceed on a nonzero exit.

- [ ] **Step 4: STOP — get explicit user confirmation before the launch call**

Tell the user exactly what will run and on which node, and wait for their explicit go-ahead before Step 5. This is the plan's first real GPU launch through a pathway that has never carried real training before — say that plainly, not just "launching now."

- [ ] **Step 5: Launch a short smoke run on node411**

```bash
./cluster_launch.sh "python main_cosir.py dataset=redcaps_150k_cluster model=clip_base model.conditioning_mode=prototype_pooled model.num_prototypes=16 train.initialization_strategy=buddies train.epochs=2 seed=42 experiment.results_dir=/local/wding/res/CoSiR_Experiment/redcaps_150k_cluster/exp18_smoke wandb.group='exp18 das6 smoke'" --node node411
```

- [ ] **Step 6: Wait for the user to say it's done — do not poll**

Per the cluster-run skill: there is no "finished" signal observable on your own; stop and wait.

- [ ] **Step 7: Pull results and verify the round-trip**

```bash
./cluster_sync_down.sh --node node411
ls res/CoSiR_Experiment/redcaps_150k_cluster/exp18_smoke/
```

Confirm the expected checkpoint/metric files actually landed in the normal `res/CoSiR_Experiment/<dataset>/` layout before treating the pathway as validated. If anything is missing, this is a real finding to report, not something to route around silently — check `cluster_sync.log` (per the skill's troubleshooting table) before assuming user error.

---

### Task 8: DAS6 full sweep — 2 arms × 3 seeds, RedCaps-150k

**Files:**
- Create: `scripts/run_exp18_150k_sweep.sh` (mirrors `scripts/run_combiner_architecture_fullscale.sh`'s structure)

- [ ] **Step 1: Write the sweep script**

```bash
#!/bin/bash
# scripts/run_exp18_150k_sweep.sh
# Experiment 18: architecture-fix-only baseline vs. prototype-conditioning arm.
# One arm per DAS6 node (run concurrently): node411 = baseline, node412 = prototype_pooled.
#
#   ARM=baseline bash scripts/run_exp18_150k_sweep.sh         # on node411
#   ARM=prototype_pooled bash scripts/run_exp18_150k_sweep.sh # on node412
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM="${ARM:?Set ARM=baseline or ARM=prototype_pooled}"
DATASET="${DATASET:-redcaps_150k_cluster}"
SEED_SWEEP="${SEED_SWEEP:-1 2 3}"
EPOCHS="${EPOCHS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-16}"  # Task 6's chosen value — override if smoke test picked differently
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-/local/wding/res/CoSiR_Experiment/exp18_${ARM}}"
WANDB_GROUP="${WANDB_GROUP:-exp18 buddy prototype conditioning}"

if [ "$ARM" = "baseline" ]; then
  CONDITIONING_MODE="free_vector"
  COMBINER_TYPE="lowrank"
elif [ "$ARM" = "prototype_pooled" ]; then
  CONDITIONING_MODE="prototype_pooled"
  COMBINER_TYPE="lowrank"
else
  echo "ARM must be 'baseline' or 'prototype_pooled', got '$ARM'" >&2
  exit 1
fi

for SEED in $SEED_SWEEP; do
  RD="${BASE_RESULTS_DIR}/seed${SEED}"
  TAG="exp18-150k-${ARM}-seed${SEED}"
  echo ">>> arm=${ARM} seed=${SEED} -> results_dir=${RD}"
  python main_cosir.py \
    dataset="$DATASET" \
    model=clip_base \
    model.combiner_type="$COMBINER_TYPE" \
    model.conditioning_mode="$CONDITIONING_MODE" \
    model.num_prototypes="$NUM_PROTOTYPES" \
    train.initialization_strategy=buddies \
    train.epochs="$EPOCHS" \
    eval.evaluation_interval="$EVAL_INTERVAL" \
    seed="$SEED" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    ++wandb.tags=[$TAG]
done
```

- [ ] **Step 2: Commit with a "cluster run" marker**

```bash
git add scripts/run_exp18_150k_sweep.sh
git commit -m "feat(exp18): cluster run — 2-arm x 3-seed 150k sweep script"
```

- [ ] **Step 3: Sync to both nodes**

```bash
./cluster_sync_up.sh ./cluster_sync.conf --node node411
./cluster_sync_up.sh ./cluster_sync.conf --node node412
```

(The `redcaps_150k` feature cache is already on both nodes from Task 7's sync — no `--force-data` needed unless Task 7 used a different node.)

- [ ] **Step 4: STOP — get explicit user confirmation before EACH launch call**

Two separate launches, two separate confirmations — do not batch them into one approval.

- [ ] **Step 5: Launch the baseline arm on node411**

```bash
./cluster_launch.sh "ARM=baseline bash scripts/run_exp18_150k_sweep.sh" --node node411
```

- [ ] **Step 6: Launch the prototype-conditioning arm on node412**

```bash
./cluster_launch.sh "ARM=prototype_pooled bash scripts/run_exp18_150k_sweep.sh" --node node412
```

- [ ] **Step 7: Wait for the user to say both runs are done — do not poll**

---

### Task 9: Pull results, evaluate, and write the report

**Files:**
- Create: `docs/reports/2026-09-15_buddy_prototype_conditioning.md` (date reflects actual completion, adjust if this lands later)

- [ ] **Step 1: Pull results from both nodes**

```bash
./cluster_sync_down.sh --node node411
./cluster_sync_down.sh --node node412
```

- [ ] **Step 2: Retrieval comparison**

For each seed, compare `test_oracle`/`test_pre_diff` t2i/i2t R1 between the two arms' `res/CoSiR_Experiment/exp18_baseline/seed{1,2,3}` and `res/CoSiR_Experiment/exp18_prototype_pooled/seed{1,2,3}` result directories, against this project's standard noise floor (§5, `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`) — mean Δ ± std across seeds, `mean/SEM` significance read, matching this project's established reporting convention.

- [ ] **Step 3: Interpretability readout — reuse 17.1's probe harness**

Run the control-task-gated probe harness from `src/test/20260915_condition_space_audit/checkpoint_probe.py` against the prototype-conditioning arm's trained condition vectors (obtained from `model.prototype_bank(query_features)` evaluated on the full RedCaps-150k `warmth`/`register` proxy labels — the harness's `load_checkpoint`/`probe_selectivity` functions take a `[N, D]` array, so this only needs a small script computing that array from the saved checkpoint's `prototype_bank` state_dict + the same cached CLIP features, not a modification to the harness itself). Report selectivity, same metric as 17.1, directly comparable to its raw-CLIP and free-vector-checkpoint baselines already on record.

- [ ] **Step 4: Prototype coherence — reuse `condition_space_evaluator.py`'s silhouette machinery**

Call `condition_space_evaluator.py`'s existing `sklearn.metrics.silhouette_score`-based machinery (`compute_condition_space_quality`, `src/utils/condition_space_evaluator.py`) on the prototype-conditioning arm's trained condition vectors, using each sample's argmax-attention prototype id as the cluster label (in place of the old HDBSCAN labels that function was originally built around) — the same metric PercepT itself reports (0.97 vs. 0.37 baseline), giving a literature-aligned readout.

- [ ] **Step 5: Write the report**

Follow this project's report convention (see `docs/reports/2026-09-15_condition_space_audit.md` for the template: verdict up front, then evidence, then caveats). State explicitly: (a) the retrieval R1 result and whether it beats the noise floor, (b) the probe-selectivity comparison against 17.1's raw-CLIP and free-vector baselines, (c) the silhouette/prototype-coherence number, (d) whether prototype collapse was observed during the real 150k/100-epoch runs (not just the 1-2k smoke test), (e) a plain verdict: did this redesign beat the architecture-fix-only baseline on retrieval, on interpretability, on both, or on neither.

- [ ] **Step 6: Commit**

```bash
git add docs/reports/2026-09-15_buddy_prototype_conditioning.md
git commit -m "docs(exp18): record buddy-graph prototype conditioning result"
```

---

## Self-review

**Spec coverage:** §3 architecture → Tasks 2-5. §4 components → Tasks 2, 3, 4 (files table matches exactly). §5 data flow → Task 5. §6 failure modes (collapse, community/prototype-count mismatch) → Task 2's `usage_entropy`, Task 3's `coarsen_to_prototype_count`, Task 5's monitoring, Task 6's smoke-test entropy check. §7 evaluation/success criteria → Task 6 (P sweep), Task 9 (retrieval, selectivity, silhouette). §8 compute plan → Tasks 6 (local), 7-8 (DAS6). §9 execution mode → Global Constraints + every task's commit-and-Codex framing. §10a isolation → Task 1. §10 supersession → already committed to the spec docs directly (no plan task needed, it's a documentation-only change already made during brainstorming). Out of scope (300k/500k, backbone unfreeze) → correctly absent from every task.

**Placeholder scan:** no TBD/TODO. Two explicitly-flagged uncertainties are real, not placeholders-in-disguise: Task 4 Step 5's "read the implementation first, add `return_graph` only if needed" (genuine unknown at plan-writing time, resolved by the implementer reading real code, not deferred indefinitely), and Task 6's `redcaps_150k.json` path prediction (explicitly named as unverified, with Task 7 as its real verification step, not silently assumed).

**Type consistency:** `PrototypeBank.forward(query_features: Tensor[B, query_dim]) -> Tensor[B, condition_dim]` (Task 2) matches every call site in Task 4 (`model.prototype_bank(query_features)`, `query_dim=self.feature_dim`) and Task 5 (`query_features = 0.5 * (img_features + txt_features)`, both `[B, feature_dim]`). `detect_communities`/`community_mean_features`/`coarsen_to_prototype_count` (Task 3) signatures match their Task 4 call sequence exactly (labels → means → coarsened means → `seed_from_communities`).
