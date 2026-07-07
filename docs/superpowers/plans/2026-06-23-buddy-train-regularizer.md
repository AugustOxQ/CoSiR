# Buddy-Graph Smoothness Regularizer (Family #1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Laplacian smoothness term on the trainable label embeddings `z` along the buddy union graph `E`, so the validated buddy geometry that seeds the init is kept alive during training instead of drifting.

**Architecture:** A pure, vectorized loss function (`buddy_graph_smoothness_loss`) computes mean squared distance between batch anchors and a small random sample of their `E`-neighbors, gathered from the full `embedding_manager.embeddings` table (a differentiable leaf). `compute_buddy_init` is extended to also return `E`'s edge list (remapped to sample-id order); the embedding manager persists it as `buddy_edges.npy` alongside the init template; the training loop loads it once, builds a CSR neighbor index, and adds `lambda_buddy · L_buddy` to the loss each step.

**Tech Stack:** Python 3.10, PyTorch, NumPy, SciPy sparse (existing `src/conditional_buddy/`), the existing `LabelContrastiveLoss_enhance` train loop in `src/hook/train_cosir.py`.

## Global Constraints

- **Do not run `git commit` — the user commits.** Each task's final step stages changes (`git add ...`) and stops; never invoke `git commit`.
- **Backward compatibility is mandatory.** `cfg.loss.lambda_buddy` defaults to `0.0` via `getattr` (matching the existing `lambda_var` / `lambda_cov` / `lambda_gap_align` pattern). With the default, no edges are loaded and no term is added — the pipeline is byte-for-byte unchanged. No config-schema file edits are required.
- **Graph = `E` (union), target = raw `z`, unit edge weights, plain L2** — per the approved spec. No normalization of `z`, no distance-weighting, no `B` graph, no schedules. Those are deferred ablations.
- **Tests are plain scripts** (the repo convention): `def test_*()` functions invoked from a `if __name__ == "__main__":` block, run with `python <path>`, using `assert`. No pytest.
- **Run everything in the CoSiR conda env:** prefix run commands with
  `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR &&`.
- Embeddings are `[N, D]` with `D=16`, indexed by table position; `embedding_manager.id_to_index[sample_id] -> position`; `embedding_manager.sample_ids[position] -> sample_id`.

---

## File Structure

- **Create** `src/metrics/regularizer.py` additions — `build_neighbor_csr`, `buddy_graph_smoothness_loss` (core term; pure, GPU/CPU agnostic, no model forward).
- **Modify** `src/conditional_buddy/compute_buddies.py` — `compute_buddy_init(..., return_edges=False)` returns `E`'s edge list, remapped to output sample-id order.
- **Modify** `src/utils/embedding_manager_nocache.py` — `_buddy_init` saves `buddy_edges.npy`; `_copy_to`/`_copy_from` include it; new `get_buddy_edges()`.
- **Modify** `src/hook/train_cosir.py` — load edges + build CSR before the loop; add `lambda_buddy · L_buddy` in the batch loop with wandb logging.
- **Create** `src/test/20260623_buddy_train_reg/test_buddy_reg.py` — unit tests for the loss, CSR, edge remap, and manager round-trip.
- **Create** `.claude/20260623_buddy_train_log.md` — change log for the modified source files (repo convention).

---

### Task 1: Core loss — `buddy_graph_smoothness_loss` + `build_neighbor_csr`

**Files:**
- Modify: `src/metrics/regularizer.py` (append two functions; add `from typing import Optional` and ensure `import torch` at top)
- Test: `src/test/20260623_buddy_train_reg/test_buddy_reg.py`

**Interfaces:**
- Produces:
  - `build_neighbor_csr(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]`
    returning `(indptr [num_nodes+1] long, indices [2M] long)` on `edge_index.device`.
  - `buddy_graph_smoothness_loss(embeddings: torch.Tensor, indptr: torch.Tensor, indices: torch.Tensor, anchor_positions: torch.Tensor, num_samples: int = 4, generator: Optional[torch.Generator] = None) -> torch.Tensor`
    returning a scalar tensor (mean squared anchor–buddy distance; `embeddings.sum()*0.0` when no anchor has a neighbor).

- [ ] **Step 1: Write the failing test**

Create `src/test/20260623_buddy_train_reg/test_buddy_reg.py`:

```python
"""
Unit tests for the Family #1 buddy-graph smoothness regularizer.

Run:
    python src/test/20260623_buddy_train_reg/test_buddy_reg.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch

from src.metrics.regularizer import build_neighbor_csr, buddy_graph_smoothness_loss


def test_csr_symmetric():
    # one undirected edge 0-1, node 2 isolated
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    indptr, indices = build_neighbor_csr(edge_index, num_nodes=3)
    assert indptr.tolist() == [0, 1, 2, 2], indptr.tolist()
    # node 0's neighbor is 1, node 1's neighbor is 0
    assert indices[indptr[0]:indptr[1]].tolist() == [1]
    assert indices[indptr[1]:indptr[2]].tolist() == [0]
    assert indices[indptr[2]:indptr[3]].tolist() == []
    print("  test_csr_symmetric OK")


def test_loss_value_single_neighbor():
    # nodes 0,1 connected; 2,3 isolated. Each anchor has exactly one neighbor,
    # so sampling is deterministic regardless of num_samples.
    emb = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    indptr, indices = build_neighbor_csr(torch.tensor([[0], [1]]), num_nodes=4)
    anchors = torch.tensor([0, 1], dtype=torch.long)
    loss = buddy_graph_smoothness_loss(emb, indptr, indices, anchors, num_samples=4)
    # ||z0 - z1||^2 = 9 + 16 = 25 for both anchors
    assert abs(loss.item() - 25.0) < 1e-5, loss.item()
    print("  test_loss_value_single_neighbor OK")


def test_isolated_contributes_zero():
    emb = torch.zeros(4, 2, requires_grad=True)
    indptr, indices = build_neighbor_csr(torch.tensor([[0], [1]]), num_nodes=4)
    anchors = torch.tensor([2, 3], dtype=torch.long)  # both isolated
    loss = buddy_graph_smoothness_loss(emb, indptr, indices, anchors, num_samples=4)
    assert loss.item() == 0.0, loss.item()
    loss.backward()
    assert torch.count_nonzero(emb.grad) == 0
    print("  test_isolated_contributes_zero OK")


def test_gradient_shrinks_pair():
    emb = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
    indptr, indices = build_neighbor_csr(torch.tensor([[0], [1]]), num_nodes=2)
    anchors = torch.tensor([0, 1], dtype=torch.long)
    before = (emb[0] - emb[1]).norm().item()
    opt = torch.optim.SGD([emb], lr=0.01)
    opt.zero_grad()
    loss = buddy_graph_smoothness_loss(emb, indptr, indices, anchors, num_samples=4)
    loss.backward()
    opt.step()
    after = (emb[0] - emb[1]).norm().item()
    assert after < before, (before, after)
    print("  test_gradient_shrinks_pair OK")


if __name__ == "__main__":
    test_csr_symmetric()
    test_loss_value_single_neighbor()
    test_isolated_contributes_zero()
    test_gradient_shrinks_pair()
    print("ALL TASK 1 TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260623_buddy_train_reg/test_buddy_reg.py`
Expected: FAIL with `ImportError: cannot import name 'build_neighbor_csr' from 'src.metrics.regularizer'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/metrics/regularizer.py` (add `from typing import Optional` near the top imports if absent; `import torch` is already present):

```python
def build_neighbor_csr(edge_index: torch.Tensor, num_nodes: int):
    """Build a symmetric CSR neighbour structure from an undirected edge list.

    edge_index: LongTensor [2, M], one direction per edge (endpoints in any order).
    Returns (indptr [num_nodes+1], indices [2M]) LongTensors on edge_index.device.
    Self-loops are dropped; the graph is symmetrised (each edge stored both ways).
    """
    device = edge_index.device
    if edge_index.numel() == 0:
        indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        indices = torch.empty(0, dtype=torch.long, device=device)
        return indptr, indices
    src = torch.cat([edge_index[0], edge_index[1]])
    dst = torch.cat([edge_index[1], edge_index[0]])
    keep = src != dst
    src, dst = src[keep], dst[keep]
    order = torch.argsort(src)
    src, dst = src[order], dst[order]
    counts = torch.bincount(src, minlength=num_nodes)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    indptr[1:] = torch.cumsum(counts, dim=0)
    return indptr, dst.contiguous()


def buddy_graph_smoothness_loss(
    embeddings: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    anchor_positions: torch.Tensor,
    num_samples: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Mean squared distance between batch anchors and sampled buddies along E.

    This is the Laplacian-Eigenmaps energy that produced the buddy init, kept alive
    during training. Anchors with no neighbour are skipped; if none have a neighbour
    the term is exactly 0 (with no gradient).

    embeddings:       [N, D] differentiable table (embedding_manager.embeddings).
    indptr, indices:  CSR neighbour structure from build_neighbor_csr.
    anchor_positions: [A] long, table positions of the batch's samples.
    num_samples:      buddies sampled per anchor (with replacement).
    """
    device = embeddings.device
    deg = indptr[anchor_positions + 1] - indptr[anchor_positions]   # [A]
    mask = deg > 0
    if not torch.any(mask):
        return embeddings.sum() * 0.0
    anchors = anchor_positions[mask]
    deg = deg[mask].unsqueeze(1)                                    # [A', 1]
    starts = indptr[anchors].unsqueeze(1)                          # [A', 1]
    A = anchors.shape[0]
    rand = torch.rand(A, num_samples, device=device, generator=generator)
    offsets = torch.clamp((rand * deg).long(), max=deg - 1)        # [A', num_samples] in [0, deg)
    nbr_pos = indices[starts + offsets]                            # [A', num_samples] table positions
    z_a = embeddings[anchors].unsqueeze(1)                         # [A', 1, D]
    z_n = embeddings[nbr_pos]                                      # [A', num_samples, D]
    return (z_a - z_n).pow(2).sum(-1).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260623_buddy_train_reg/test_buddy_reg.py`
Expected: PASS — prints `ALL TASK 1 TESTS PASSED`.

- [ ] **Step 5: Stage changes (user commits)**

```bash
git add src/metrics/regularizer.py src/test/20260623_buddy_train_reg/test_buddy_reg.py
```

---

### Task 2: Persist `E` — `compute_buddy_init(return_edges=True)`

**Files:**
- Modify: `src/conditional_buddy/compute_buddies.py:75-149` (`compute_buddy_init`)
- Test: `src/test/20260623_buddy_train_reg/test_buddy_reg.py` (append a test)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `compute_buddy_init(..., return_edges: bool = False)`. When `False` (default) returns `emb` (unchanged). When `True` returns `(emb, edges)` where `edges` is `np.int64 [2, M]`, undirected (`i < j`), endpoints expressed as **table positions in `output_sample_ids` order** when reordering is requested, else input-row order.

- [ ] **Step 1: Write the failing test**

Append to `src/test/20260623_buddy_train_reg/test_buddy_reg.py` (above the `__main__` block):

```python
def test_return_edges_and_remap():
    from src.conditional_buddy.compute_buddies import compute_buddy_init

    rng = np.random.default_rng(0)
    dim = 32
    c0 = rng.normal(0, 1, dim); c1 = rng.normal(6, 1, dim)
    labels = np.array([0] * 40 + [1] * 40)
    centers = np.stack([c0, c1])
    img = (centers[labels] + rng.normal(0, 0.4, (80, dim))).astype(np.float32)
    txt = (centers[labels] + rng.normal(0, 0.4, (80, dim))).astype(np.float32)
    N = 80
    ids = list(range(N))

    # input-order edges
    _, edges0 = compute_buddy_init(
        img, txt, n_dim=16, K=10, device="cpu", use_half=False, return_edges=True,
    )
    assert edges0.shape[0] == 2 and edges0.dtype == np.int64
    assert (edges0[0] < edges0[1]).all(), "edges must be stored with i < j"

    # reordered output: output row k holds input id perm[k]
    perm = list(rng.permutation(N))
    _, edges_perm = compute_buddy_init(
        img, txt, n_dim=16, K=10, device="cpu", use_half=False, return_edges=True,
        input_sample_ids=ids, output_sample_ids=perm,
    )
    # map output positions back to input positions via reorder == perm
    reorder = np.array(perm)
    recovered = reorder[edges_perm]  # [2, M] input positions
    set0 = {frozenset((int(a), int(b))) for a, b in edges0.T}
    setr = {frozenset((int(a), int(b))) for a, b in recovered.T}
    assert set0 == setr, "remapped edges do not connect the same samples"
    print("  test_return_edges_and_remap OK")
```

Add `test_return_edges_and_remap()` to the `__main__` block (before the final print).

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260623_buddy_train_reg/test_buddy_reg.py`
Expected: FAIL — `compute_buddy_init() got an unexpected keyword argument 'return_edges'`.

- [ ] **Step 3: Write minimal implementation**

In `src/conditional_buddy/compute_buddies.py`, add the parameter to the signature (after `connect_components: bool = True,`):

```python
    return_edges: bool = False,
```

Capture `E` (it is already bound as the third element of `build_buddy_graphs(...)` at the top of the function). Replace the final `return emb` block (currently lines ~142-149) with:

```python
    if output_sample_ids is not None:
        if input_sample_ids is None:
            raise ValueError("output_sample_ids given but input_sample_ids is None.")
        pos = {sid: i for i, sid in enumerate(input_sample_ids)}
        reorder = [pos[sid] for sid in output_sample_ids]
        emb = emb[reorder]

    if not return_edges:
        return emb

    coo = E.tocoo()
    upper = coo.row < coo.col
    edges = np.stack([coo.row[upper], coo.col[upper]]).astype(np.int64)  # input-row positions
    if output_sample_ids is not None:
        inv = np.empty(len(reorder), dtype=np.int64)
        inv[np.asarray(reorder, dtype=np.int64)] = np.arange(len(reorder), dtype=np.int64)
        edges = inv[edges]  # remap input positions -> output positions
    return emb, edges
```

(Also update the docstring `Returns:` line to note the optional `(emb, edges)` tuple.)

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260623_buddy_train_reg/test_buddy_reg.py`
Expected: PASS — `ALL TASK 1 TESTS PASSED` (now includes the remap test).

- [ ] **Step 5: Run the pre-existing buddy tests to confirm no regression**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260609_conditional_buddy/test_compute_buddies.py`
Expected: PASS (the default `return_edges=False` keeps the old return type).

- [ ] **Step 6: Stage changes (user commits)**

```bash
git add src/conditional_buddy/compute_buddies.py src/test/20260623_buddy_train_reg/test_buddy_reg.py
```

---

### Task 3: Persist edges through the embedding manager

**Files:**
- Modify: `src/utils/embedding_manager_nocache.py` — `_buddy_init` (~375-393), `_copy_to` (~397-403), `_copy_from` (~405-411); add `get_buddy_edges` after `_copy_from`.
- Test: `src/test/20260623_buddy_train_reg/test_buddy_reg.py` (append a test)

**Interfaces:**
- Consumes: `compute_buddy_init(..., return_edges=True) -> (emb, edges)` from Task 2.
- Produces:
  - `_buddy_init` still returns `np.ndarray` (the init) but, as a side effect, writes `buddy_edges.npy` into `self.embeddings_dir`.
  - `buddy_edges.npy` round-trips via `_copy_to` / `_copy_from` (so it travels with the template).
  - `get_buddy_edges(self) -> Optional[np.ndarray]` — loads `self.embeddings_dir / "buddy_edges.npy"`, or `None` if absent.

- [ ] **Step 1: Write the failing test**

Append to `src/test/20260623_buddy_train_reg/test_buddy_reg.py` (above `__main__`):

```python
def test_manager_edges_roundtrip(tmp_root=None):
    import tempfile, shutil
    from pathlib import Path
    from src.utils.embedding_manager_nocache import TrainableEmbeddingManager

    root = Path(tempfile.mkdtemp())
    try:
        exp = root / "exp" / "run0"
        emb_dir = exp / "training_embeddings"
        mgr = TrainableEmbeddingManager(
            sample_ids=list(range(6)), embedding_dim=16,
            embeddings_dir=str(emb_dir), mode="ram", initialization_strategy="zeros",
        )
        edges = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int64)
        np.save(emb_dir / "buddy_edges.npy", edges)

        # get_buddy_edges reads it back
        got = mgr.get_buddy_edges()
        assert got is not None and np.array_equal(got, edges)

        # round-trips through _copy_to / _copy_from (template persistence)
        tmpl = exp.parent / "template_embeddings"
        mgr._copy_to(tmpl)
        assert (tmpl / "buddy_edges.npy").exists(), "edges not copied into template"
        (emb_dir / "buddy_edges.npy").unlink()
        mgr._copy_from(tmpl)
        assert np.array_equal(mgr.get_buddy_edges(), edges), "edges not restored from template"
        print("  test_manager_edges_roundtrip OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)
```

Add `test_manager_edges_roundtrip()` to the `__main__` block.

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260623_buddy_train_reg/test_buddy_reg.py`
Expected: FAIL — `AttributeError: 'TrainableEmbeddingManager' object has no attribute 'get_buddy_edges'`.

- [ ] **Step 3: Write minimal implementation**

In `src/utils/embedding_manager_nocache.py`:

(a) In `_buddy_init`, change the `compute_buddy_init(...)` call to request edges and save them. Replace `emb = compute_buddy_init(...)` (the assignment at ~375-388) with:

```python
        emb, edges = compute_buddy_init(
            img,
            txt,
            n_dim=self.embedding_dim,
            method=method,
            K=k,
            alpha=alpha,
            device=device,
            knn_batch_size=knn_batch_size,
            normalize_method=normalize_method,
            seed=seed,
            input_sample_ids=fm_sample_ids,
            output_sample_ids=self.sample_ids,
            return_edges=True,
        )
        np.save(self.embeddings_dir / "buddy_edges.npy", edges.astype(np.int64))
```

(b) Add `"buddy_edges.npy"` to the file tuples in **both** `_copy_to` and `_copy_from`:

```python
        for fname in ("embeddings.npy", "sample_ids.npy", "metadata.json", "buddy_edges.npy"):
```

(c) Add `get_buddy_edges` immediately after `_copy_from`:

```python
    def get_buddy_edges(self):
        """Return the persisted buddy edge list [2, M] (int64) or None if absent."""
        path = self.embeddings_dir / "buddy_edges.npy"
        if not path.exists():
            return None
        return np.load(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260623_buddy_train_reg/test_buddy_reg.py`
Expected: PASS — `ALL TASK 1 TESTS PASSED`.

- [ ] **Step 5: Stage changes (user commits)**

```bash
git add src/utils/embedding_manager_nocache.py src/test/20260623_buddy_train_reg/test_buddy_reg.py
```

---

### Task 4: Wire the term into the training loop

**Files:**
- Modify: `src/hook/train_cosir.py` — import the term; load edges + build CSR once after `_init_embedding_manager` (within `train_cosir`, near where `embedding_manager` is finalized ~1293-1298); add the term inside the batch loop after the other extra terms (~1495, before `epoch_loss += loss.item()`).

**Interfaces:**
- Consumes: `buddy_graph_smoothness_loss`, `build_neighbor_csr` (Task 1); `embedding_manager.get_buddy_edges()` (Task 3); `cfg.loss.lambda_buddy`, `cfg.loss.buddy_reg_samples` via `getattr`.
- Produces: `loss_dict["loss_buddy"]` logged in the `loss` wandb section; `lambda_buddy · L_buddy` added to `loss`.

- [ ] **Step 1: Add the import**

At the top of `src/hook/train_cosir.py`, alongside the existing `from src.metrics import LabelContrastiveLoss_enhance` (line 48), add:

```python
from src.metrics.regularizer import build_neighbor_csr, buddy_graph_smoothness_loss
```

- [ ] **Step 2: Load edges + build the CSR once, before the training loop**

In `train_cosir`, right after `optimizer, scheduler = _build_optimizer_and_scheduler(cfg, model, embedding_manager)` (~line 1298), insert:

```python
    # Family #1 buddy regularizer: load the persisted E edge list and build a CSR
    # neighbour index once. Disabled (no-op) when lambda_buddy == 0 or edges absent.
    _lambda_buddy = getattr(cfg.loss, "lambda_buddy", 0.0)
    _buddy_reg_samples = int(getattr(cfg.loss, "buddy_reg_samples", 4))
    buddy_indptr = buddy_indices = None
    if _lambda_buddy > 0:
        _edges = embedding_manager.get_buddy_edges()
        if _edges is None:
            print("[buddy-reg] lambda_buddy>0 but no buddy_edges.npy found — "
                  "disabling buddy regularizer for this run.")
            _lambda_buddy = 0.0
        else:
            _edge_index = torch.from_numpy(_edges.astype(np.int64)).to(device)
            buddy_indptr, buddy_indices = build_neighbor_csr(
                _edge_index, num_nodes=len(embedding_manager.sample_ids)
            )
            print(f"[buddy-reg] enabled: lambda_buddy={_lambda_buddy}, "
                  f"samples/anchor={_buddy_reg_samples}, edges={_edge_index.shape[1]:,}")
```

- [ ] **Step 3: Add the term inside the batch loop**

In the batch loop, after the gap-alignment block and before `epoch_loss += loss.item()` (~line 1496), insert:

```python
            # Family #1: buddy-graph smoothness on z along E (only when conditions train)
            if (
                _lambda_buddy > 0
                and buddy_indptr is not None
                and embedding_manager.embeddings.requires_grad
            ):
                _anchor_pos = torch.tensor(batch_indices, device=device, dtype=torch.long)
                buddy_loss = buddy_graph_smoothness_loss(
                    embedding_manager.embeddings,
                    buddy_indptr,
                    buddy_indices,
                    _anchor_pos,
                    num_samples=_buddy_reg_samples,
                )
                loss = loss + _lambda_buddy * buddy_loss
                loss_dict["loss_buddy"] = buddy_loss.detach()
```

(`loss_buddy` is not in `_monitor_keys` / `_phase_keys`, so the existing `loss_metrics` comprehension routes it to the `loss` wandb section automatically — no logging change needed.)

- [ ] **Step 4: Syntax + import smoke check**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -c "import ast; ast.parse(open('src/hook/train_cosir.py').read()); print('parse OK')"`
Expected: prints `parse OK`.

- [ ] **Step 5: Backward-compat smoke check (term off by default)**

Confirm the default path is inert: with no `lambda_buddy` in config, `getattr(cfg.loss, "lambda_buddy", 0.0)` is `0.0`, so no edges load and the loop block is skipped.

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -c "
import torch, numpy as np
from src.metrics.regularizer import build_neighbor_csr, buddy_graph_smoothness_loss
ei = torch.tensor([[0,1,2],[1,2,3]])
ip, ix = build_neighbor_csr(ei, 4)
emb = torch.nn.Parameter(torch.randn(4,16))
l = buddy_graph_smoothness_loss(emb, ip, ix, torch.tensor([0,1,2,3]), num_samples=4)
print('loss finite:', bool(torch.isfinite(l)))
"`
Expected: prints `loss finite: True`.

- [ ] **Step 6: Real-data smoke run (if a training config + features are available)**

Run a few steps with the term on to confirm end-to-end wiring, the artifact reload, and a finite `loss_buddy`. Use the project's standard training entrypoint with `lambda_buddy` set (>0) and `initialization_strategy=buddies`:

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/hook/train_cosir.py` (with a config whose `train.initialization_strategy=buddies` and `loss.lambda_buddy>0`, `train.epochs` small).
Expected: console shows `[buddy-reg] enabled: ...`, training runs, and `loss_buddy` appears in the wandb `loss` section with finite values. If no runnable config/features are present in this environment, skip and note it for the user to run on the cluster.

- [ ] **Step 7: Stage changes (user commits)**

```bash
git add src/hook/train_cosir.py
```

---

### Task 5: Change log

**Files:**
- Create: `.claude/20260623_buddy_train_log.md`

- [ ] **Step 1: Write the change log**

Create `.claude/20260623_buddy_train_log.md` documenting each modified source file with before/after snippets and rationale, following the repo convention (file path as header). Cover:
- `# /src/metrics/regularizer.py` — added `build_neighbor_csr`, `buddy_graph_smoothness_loss`.
- `# /src/conditional_buddy/compute_buddies.py` — added `return_edges` (default `False`, backward compatible).
- `# /src/utils/embedding_manager_nocache.py` — `_buddy_init` saves `buddy_edges.npy`; copy lists include it; new `get_buddy_edges`.
- `# /src/hook/train_cosir.py` — load edges + CSR; add `lambda_buddy · L_buddy` term (off by default).

- [ ] **Step 2: Stage changes (user commits)**

```bash
git add .claude/20260623_buddy_train_log.md
```

---

## Self-Review

**Spec coverage:**
- Core term `L_buddy = mean ‖z_i − z_j‖²` along `E`, raw `z`, unit weights → Task 1. ✓
- Global table gather, sampled buddies per anchor, isolated anchors contribute 0 → Task 1 (`buddy_graph_smoothness_loss`) + tests. ✓
- Persist `E` (remapped to sample-id order, `i<j`, self-loops dropped) → Task 2. ✓
- Save `buddy_edges.npy` next to template; round-trip via copy; missing-file fallback → Task 3 (`_buddy_init`, copy lists, `get_buddy_edges`) + Task 4 Step 2 (warn + disable). ✓
- Wire into train loop, add `lambda_buddy · L_buddy`, log `loss_buddy`, guard on `requires_grad` (EM phase) → Task 4. ✓
- Config via `getattr` defaults (`lambda_buddy=0.0`, `buddy_reg_samples=4`); backward compatible → Task 4 Steps 2/5. ✓
- Tests (value, gradient, isolated, edge remap, manager round-trip) → Tasks 1–3. ✓
- Out-of-scope items (B graph, delta target, normalized z, schedules) → not implemented. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The only deferred runtime detail is Task 4 Step 6 (real-data smoke), which depends on a cluster config/features and is explicitly marked skippable.

**Type consistency:** `build_neighbor_csr` returns `(indptr, indices)` used identically in Task 1 tests and Task 4. `buddy_graph_smoothness_loss` signature matches all call sites. `compute_buddy_init(return_edges=True)` returns `(emb, edges)` consumed in Task 3's `_buddy_init`. `get_buddy_edges()` returns `np.ndarray | None`, consumed in Task 4 Step 2. `buddy_edges.npy` filename is identical across Tasks 3 and 4. ✓
