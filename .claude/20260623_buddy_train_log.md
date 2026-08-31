# Buddy-Graph Regularizer Training Integration — Change Log

**Reference Documents:**
- Design: `docs/superpowers/specs/2026-06-23-buddy-train-regularizer-design.md`
- Plan: `docs/superpowers/plans/2026-06-23-buddy-train-regularizer.md`

**Config Knobs:**
- `cfg.loss.lambda_buddy` (default `0.0` → off) — weight of the buddy-graph smoothness loss.
- `cfg.loss.buddy_reg_samples` (default `4`) — buddies sampled per anchor per batch.
- `cfg.loss.buddy_reg_graph` (default `"E"`) — graph selection (currently only E supported).

---

## `/src/metrics/regularizer.py`

**Added:** Two new functions for buddy-graph regularization.

### `build_neighbor_csr(edge_index, num_nodes) → (indptr, indices)`

**Before:** Function did not exist.

**After:**
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
```

**Rationale:** Converts the undirected edge list from `compute_buddy_init` into a CSR format for efficient neighbor sampling during training. Self-loops dropped; graph is symmetrised so each edge is stored bidirectionally.

### `buddy_graph_smoothness_loss(embeddings, indptr, indices, anchor_positions, num_samples, generator) → loss`

**Before:** Function did not exist.

**After:**
```python
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

**Rationale:** Implements the Laplacian-Eigenmaps energy L_buddy = mean ‖z_i − z_j‖² along edges E. For each batch anchor, samples `num_samples` buddies uniformly from its neighbors and computes MSE. Isolated anchors (no neighbors) contribute 0 to the loss; if the batch has no anchored neighbors, the term is a no-op (0.0 with no gradient). This keeps the buddy initialization energy alive during training.

---

## `/src/conditional_buddy/compute_buddies.py`

**Added:** `return_edges` parameter (default `False`, backward compatible) to optionally return the edge list.

**Before:**
```python
def compute_buddy_init(
    ...
    connect_components: bool = True,
    input_sample_ids: Optional[List[int]] = None,
    output_sample_ids: Optional[List[int]] = None,
) -> np.ndarray:
    ...
    Returns: (N, n_dim) float32 in ~[-1, 1].
    ...
    return emb
```

**After:**
```python
def compute_buddy_init(
    ...
    connect_components: bool = True,
    return_edges: bool = False,
    input_sample_ids: Optional[List[int]] = None,
    output_sample_ids: Optional[List[int]] = None,
) -> np.ndarray:
    ...
    return_edges:      if True, return (emb, edges); if False (default) return emb only.
    Returns: (N, n_dim) float32 in ~[-1, 1], or if return_edges=True, a tuple (emb, edges)
             where edges is np.int64 [2, M] undirected edge list (i < j), with endpoints
             expressed as table positions in output_sample_ids order if reordering is
             requested, else input-row order.
    ...
    if not return_edges:
        return emb

    coo = E.tocoo()
    upper = coo.row < coo.col
    edges = np.stack([coo.row[upper], coo.col[upper]]).astype(np.int64)  # input-row positions
    if output_sample_ids is not None:
        inv = np.empty(len(reorder), dtype=np.int64)
        inv[np.asarray(reorder, dtype=np.int64)] = np.arange(len(reorder), dtype=np.int64)
        edges = inv[edges]  # remap input positions -> output positions
        edges = np.sort(edges, axis=0)  # re-enforce i < j after remap
    return emb, edges
```

**Rationale:** When `return_edges=True`, extracts the undirected edge list E in `[2, M]` int64 format (with `i < j` and no self-loops). If reordering from input to output sample-id order is requested, remaps edge endpoints and re-enforces the canonical ordering. Default `False` maintains backward compatibility; existing callers see no change. Task 3 uses `return_edges=True` to persist the edge list.

---

## `/src/utils/embedding_manager_nocache.py`

**Modified:** `_buddy_init`, `_copy_to`, `_copy_from` methods; added `get_buddy_edges()`.

### `_buddy_init` — Save buddy edge list

**Before:**
```python
emb = compute_buddy_init(
    img,
    txt,
    n_dim=self.embedding_dim,
    ...
    input_sample_ids=fm_sample_ids,
    output_sample_ids=self.sample_ids,
)
```

**After:**
```python
emb, edges = compute_buddy_init(
    img,
    txt,
    n_dim=self.embedding_dim,
    ...
    input_sample_ids=fm_sample_ids,
    output_sample_ids=self.sample_ids,
    return_edges=True,
)
np.save(self.embeddings_dir / "buddy_edges.npy", edges.astype(np.int64))
```

**Rationale:** Captures the returned edge list and persists it as `buddy_edges.npy` next to the embeddings and metadata. This edge list is then loaded by the train loop to enable buddy-graph regularization.

### `_copy_to` / `_copy_from` — Include buddy edges

**Before:**
```python
for fname in ("embeddings.npy", "sample_ids.npy", "metadata.json"):
    src = self.embeddings_dir / fname
    if src.exists():
        shutil.copy2(src, dest_dir / fname)
```

**After:**
```python
for fname in ("embeddings.npy", "sample_ids.npy", "metadata.json", "buddy_edges.npy"):
    src = self.embeddings_dir / fname
    if src.exists():
        shutil.copy2(src, dest_dir / fname)
```

**Rationale:** Ensures `buddy_edges.npy` travels with the template when copying embeddings to/from destination directories. The file is created by `_buddy_init`, so it may not exist for legacy templates; the `exists()` check makes the addition graceful.

### `get_buddy_edges()` — New accessor

**Before:** Method did not exist.

**After:**
```python
def get_buddy_edges(self):
    """Return the persisted buddy edge list [2, M] (int64) or None if absent."""
    path = self.embeddings_dir / "buddy_edges.npy"
    if not path.exists():
        return None
    return np.load(path)
```

**Rationale:** Simple accessor used by `train_cosir.py` to load the edge list before the training loop. Returns `None` if the file is missing (e.g., legacy templates or buddy-init was skipped), allowing the train loop to gracefully disable buddy regularization.

---

## `/src/hook/train_cosir.py`

**Added:** Buddy regularizer initialization and training-loop integration.

### Phase 5 Setup — Load edges and build CSR

**Before:** No buddy regularizer setup.

**After:**
```python
from src.metrics.regularizer import build_neighbor_csr, buddy_graph_smoothness_loss

# ... later, after optimizer/scheduler setup:

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

**Rationale:** Once before training, load the persisted edge list and build a CSR neighbor structure. If `lambda_buddy=0.0` (default, off) or the edge file is missing, the regularizer is disabled with a user-facing message. Otherwise, print confirmation of the enabled regularizer and parameters.

### Training Loop — Add buddy-graph smoothness term

**Before:** No buddy-graph loss in the batch loop.

**After:**
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

**Rationale:** In each batch iteration (only when embeddings have `requires_grad`, i.e., the conditions phase of training), compute the buddy-graph smoothness loss on the batch's anchors and add it to the total loss with weight `lambda_buddy`. The loss is logged as `loss_buddy` for monitoring. The guard ensures the term is off by default (lambda_buddy=0.0 in config), backward compatible, and only active during the EM phase when embeddings are learnable.

---

# /src/conditional_buddy/init_conditions.py

**Change:** The standalone Hydra template-builder (`main`) now requests `return_edges=True` and saves `buddy_edges.npy` alongside the template it writes.

**Before:**
```python
emb = compute_buddy_init(img, txt, n_dim=n_dim, ..., seed=seed)
...
np.lib.format.open_memmap(template_dir / "embeddings.npy", ...)[:] = emb
```

**After:**
```python
emb, edges = compute_buddy_init(img, txt, n_dim=n_dim, ..., seed=seed, return_edges=True)
...
np.lib.format.open_memmap(template_dir / "embeddings.npy", ...)[:] = emb
np.save(template_dir / "buddy_edges.npy", edges.astype(np.int64))
```

**Rationale:** Templates built via this path previously lacked `buddy_edges.npy`, so a later `lambda_buddy>0` run reusing such a template would warn and silently train without the regularizer. Edges here are in the same row order as `emb` (no output reorder), matching `embeddings.npy`/`sample_ids.npy`. Closes the one gap found in the final whole-branch review.

---

## Summary

This change log documents the integration of buddy-graph regularization into the training pipeline, completing Tasks 1–4 (plus the init_conditions.py template-path fix). The implementation:

1. **Adds loss functions** (`build_neighbor_csr`, `buddy_graph_smoothness_loss`) for efficient CSR-based neighbor sampling and Laplacian-Eigenmaps energy computation.
2. **Persists the edge list** from `compute_buddy_init` as `buddy_edges.npy` in the embedding manager's template.
3. **Rounds edges through copy/restore** so templates retain buddy information across distributed runs.
4. **Wires the term into training** with default-off config, graceful fallback for missing edges, and logging for monitoring.

The feature is backward compatible (default config disables it) and includes robust error handling for legacy templates and missing files.

---

# 2026-06-24 — drift-from-init diagnostic (`/src/hook/train_cosir.py`)

**Why:** the v4 sweep (lambda_buddy × lr × lr_label) needs to distinguish a flat
R1 that means "regularizer inert" from one that means "active but redundant". The
separator is whether the term actually grips z, measured as mean ‖z − z_init‖.

**After init / buddy-reg setup (before Phase 6 dataloaders):**
```python
# Snapshot the initial label embeddings so we can log mean drift ‖z − z_init‖
# on the eval cadence. Logged for every run (incl. lambda_buddy=0) ...
_z_init = embedding_manager.embeddings.detach().clone()
```

**In the eval-cadence block (guarded by `eval_due`, runs for every config):**
```python
if eval_due:
    with torch.no_grad():
        _drift = (
            (embedding_manager.embeddings.detach() - _z_init).norm(dim=1).mean().item()
        )
    logger.log_train({"drift_from_init": _drift}, epoch=epoch, section="buddy_diag")
```
Lands under wandb key `train_buddy_diag/drift_from_init`. lambda_buddy=0 logs it
too, so the baseline drift is the control.

**Companion analysis:** `scripts/analyze_buddyreg_sweep.py` pulls the sweep from
W&B and prints the paired per-cell ΔR1 table, the λ × lr_label interaction, and
mean final drift by λ.
