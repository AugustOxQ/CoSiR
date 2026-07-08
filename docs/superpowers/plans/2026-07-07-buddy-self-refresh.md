# Buddy Self-Refresh / Co-Training (Family #3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the buddy graph feeding Family #2's contrastive term evolve with the model — on a warm-up-then-periodic schedule, recompute a mutual-KNN graph from the current `comb_emb`, union it with the frozen CLIP graph, and rebuild the CSR that `buddy_contrastive_loss` reads.

**Architecture:** One new pure function `refresh_buddy_graph` (in `src/metrics/regularizer.py`) that, under `no_grad`, runs a chunked full-N `model.combine` pass, builds a comb-space mutual-KNN, blends it with the frozen CLIP edge list (CLIP edges always kept), and returns a rebuilt CSR + diagnostics. The train loop builds a `combine_feat_table` once at setup, then calls the function at scheduled epoch boundaries and rebinds `buddy_indptr`/`buddy_indices`. The per-batch #2 term is untouched. Default-off; `blend=0` reproduces Family #2 byte-for-byte.

**Tech Stack:** PyTorch, NumPy, SciPy sparse, Hydra config, existing `mutual_knn` (GPU brute-force topk) and `build_neighbor_csr`.

## Global Constraints

- **The user owns ALL git commits.** NEVER run `git commit`. Each task's final step **stages** changes with `git add` (use `git add -f` for anything under `.claude/`) and then STOPS. Do not commit.
- **Default-off, byte-for-byte backward compatible.** With `buddy_refresh` unset/false the pipeline must be numerically identical to today. New config keys are read via `getattr(cfg.loss, …, default)` and are NOT added to any YAML; they are passed on the CLI with a leading `+`.
- **`blend=0` ⇒ exactly Family #2.** At `buddy_refresh_blend=0` the refreshed CSR must equal the frozen CLIP CSR.
- **Index space is z-table position order everywhere.** Edges, CSR, `combine_feat_table`, and `anchor_positions` are all in `embedding_manager.sample_ids` order. Never mix in feature-store order.
- **Refresh feeds Family #2 only.** Recommended/tested config holds Family #1 off (`+loss.lambda_buddy=0`). Combining refresh with #1 is out of scope.
- **Grad-safe.** Refresh runs under `torch.no_grad()`; the z table is detached inside; the refresh must never perturb `z` or create autograd tape.
- Tests are runnable scripts (`if __name__ == "__main__":` prints `PASS`/`FAIL`), matching `src/test/20260624_buddy_contrastive/`. Run with `PYTHONPATH=/project/CoSiR` under the `CoSiR` conda env.
- Design reference: `docs/superpowers/specs/2026-07-07-buddy-self-refresh-design.md`.

---

### Task 1: Core refresh function + graph-diff helper (`refresh_buddy_graph`, `edge_jaccard`)

**Files:**
- Modify: `src/metrics/regularizer.py` (add two public functions + a private helper; add `numpy`/`scipy`/`mutual_knn` imports)
- Test: `src/test/20260707_buddy_refresh/test_buddy_refresh.py` (create)

**Interfaces:**
- Consumes: `build_neighbor_csr(edge_index[2,M], num_nodes) -> (indptr, indices)` and `mutual_knn(features_np, K, device=…) -> scipy.csr` (both already in the repo).
- Produces:
  - `refresh_buddy_graph(model, combine_feat_table, z_table, clip_edge_index, num_nodes, k=30, blend=1.0, chunk=4096, knn_device="cuda", generator=None) -> (indptr[LongTensor], indices[LongTensor], comb_edges[LongTensor 2×Mc], stats[dict])`
  - `edge_jaccard(ei_a[2,·], ei_b[2,·]) -> float` (undirected, self-loops ignored; both empty ⇒ 1.0)

- [ ] **Step 1: Write the failing tests**

Create `src/test/20260707_buddy_refresh/test_buddy_refresh.py`:

```python
"""Family #3 buddy self-refresh — unit tests (runnable script)."""
import numpy as np
import torch
from scipy.sparse import csr_matrix

import src.metrics.regularizer as reg
from src.metrics.regularizer import (
    refresh_buddy_graph,
    edge_jaccard,
    build_neighbor_csr,
)


class _IdentityCombine:
    """Stub model whose combine() returns the combine-side feature unchanged."""
    def combine(self, emb, emb_full, labels, **kw):
        return emb


def _sorted_neighbors(indptr, indices, n):
    return [sorted(indices[indptr[i]:indptr[i + 1]].tolist()) for i in range(n)]


def _sym_csr(n, undirected_edges):
    """Build a symmetric binary csr from a list of (i, j) undirected edges."""
    rows, cols = [], []
    for i, j in undirected_edges:
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def test_equivalence_at_blend_zero():
    # blend=0 => mutual_knn never called => CSR must equal the CLIP-only CSR.
    n = 6
    clip = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)  # edges 0-1,2-3,4-5
    z = torch.randn(n, 4)
    feat = torch.randn(n, 4)
    indptr, indices, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), feat, z, clip, num_nodes=n, blend=0.0
    )
    e_ip, e_ix = build_neighbor_csr(clip, n)
    assert torch.equal(indptr, e_ip), "blend=0 indptr differs from CLIP CSR"
    assert _sorted_neighbors(indptr, indices, n) == _sorted_neighbors(e_ip, e_ix, n)
    assert comb_edges.shape[1] == 0
    assert stats["graph_n_comb_edges"] == 0.0
    print("PASS test_equivalence_at_blend_zero")


def test_union_keeps_all_clip_edges(monkeypatch=None):
    # blend=1 => every CLIP edge present, plus the comb edges from mutual_knn.
    n = 6
    clip = torch.tensor([[0], [1]], dtype=torch.long)  # single CLIP edge 0-1
    fake = _sym_csr(n, [(2, 3), (4, 5)])  # comb graph disjoint from CLIP
    reg.mutual_knn = lambda features, K, **kw: fake
    z = torch.randn(n, 4)
    feat = torch.randn(n, 4)
    indptr, indices, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), feat, z, clip, num_nodes=n, blend=1.0
    )
    nbrs = _sorted_neighbors(indptr, indices, n)
    assert 1 in nbrs[0] and 0 in nbrs[1], "CLIP edge 0-1 missing after union"
    assert 3 in nbrs[2] and 5 in nbrs[4], "comb edges missing after union"
    assert stats["graph_n_comb_edges"] == 2.0
    assert abs(stats["graph_new_edge_frac"] - 1.0) < 1e-9  # both comb edges are new
    print("PASS test_union_keeps_all_clip_edges")


def test_blend_fraction_is_respected():
    n = 8
    clip = torch.tensor([[0], [1]], dtype=torch.long)
    fake = _sym_csr(n, [(2, 3), (4, 5), (6, 7), (2, 5)])  # 4 undirected comb edges
    reg.mutual_knn = lambda features, K, **kw: fake
    g = torch.Generator().manual_seed(0)
    idp, idx, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), torch.randn(n, 4), torch.randn(n, 4),
        clip, num_nodes=n, blend=0.5, generator=g,
    )
    assert stats["graph_n_comb_edges"] == 2.0, stats  # round(0.5*4)=2
    assert comb_edges.shape[1] == 2
    print("PASS test_blend_fraction_is_respected")


def test_index_alignment_comb_edges_are_z_positions():
    # A comb edge between positions (0,3) must land at z-positions 0 and 3.
    n = 5
    clip = torch.empty(2, 0, dtype=torch.long)
    fake = _sym_csr(n, [(0, 3)])
    reg.mutual_knn = lambda features, K, **kw: fake
    idp, idx, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), torch.randn(n, 4), torch.randn(n, 4),
        clip, num_nodes=n, blend=1.0,
    )
    nbrs = _sorted_neighbors(idp, idx, n)
    assert nbrs[0] == [3] and nbrs[3] == [0], nbrs
    print("PASS test_index_alignment_comb_edges_are_z_positions")


def test_no_grad_safety():
    n = 4
    clip = torch.tensor([[0], [1]], dtype=torch.long)
    z = torch.randn(n, 4, requires_grad=True)
    z_before = z.detach().clone()
    idp, idx, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), torch.randn(n, 4), z, clip, num_nodes=n, blend=0.0
    )
    assert torch.equal(z.detach(), z_before), "z was modified by refresh"
    assert idx.grad_fn is None and not idx.requires_grad
    print("PASS test_no_grad_safety")


def test_edge_jaccard():
    a = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)  # {0-1, 2-3}
    b = torch.tensor([[0, 4], [1, 5]], dtype=torch.long)  # {0-1, 4-5}
    assert abs(edge_jaccard(a, b) - 1.0 / 3.0) < 1e-9     # 1 shared / 3 union
    assert edge_jaccard(torch.empty(2, 0, dtype=torch.long),
                        torch.empty(2, 0, dtype=torch.long)) == 1.0
    print("PASS test_edge_jaccard")


if __name__ == "__main__":
    test_equivalence_at_blend_zero()
    test_union_keeps_all_clip_edges()
    test_blend_fraction_is_respected()
    test_index_alignment_comb_edges_are_z_positions()
    test_no_grad_safety()
    test_edge_jaccard()
    print("ALL PASS")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && PYTHONPATH=/project/CoSiR python src/test/20260707_buddy_refresh/test_buddy_refresh.py`
Expected: FAIL — `ImportError: cannot import name 'refresh_buddy_graph'`.

- [ ] **Step 3: Add imports + the two functions to `src/metrics/regularizer.py`**

At the top of the file, extend the imports (currently `import torch`, `import torch.nn.functional as F`, `from typing import Optional`):

```python
import numpy as np
from src.conditional_buddy.buddy_graph import mutual_knn
```

At the end of `src/metrics/regularizer.py`, append:

```python
def _undirected_edge_set(edge_index: torch.Tensor):
    """Set of frozenset{u, v} for each column; self-loops dropped."""
    s = set()
    ei = edge_index.detach().cpu()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u != v:
            s.add(frozenset((u, v)))
    return s


def edge_jaccard(ei_a: torch.Tensor, ei_b: torch.Tensor) -> float:
    """Undirected Jaccard between two edge_index tensors (both empty -> 1.0)."""
    sa, sb = _undirected_edge_set(ei_a), _undirected_edge_set(ei_b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def refresh_buddy_graph(
    model,
    combine_feat_table: torch.Tensor,
    z_table: torch.Tensor,
    clip_edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 30,
    blend: float = 1.0,
    chunk: int = 4096,
    knn_device: str = "cuda",
    generator: Optional[torch.Generator] = None,
):
    """Rebuild the buddy CSR from the model's current combined features.

    Under no_grad: run a chunked full-N ``model.combine`` pass to get
    ``comb_all`` [N, Dp], build a comb-space mutual-KNN, keep a ``blend``
    fraction of its edges, union them with the always-kept frozen CLIP edges,
    and rebuild the CSR. Returns (indptr, indices, comb_edges[2, Mc], stats).
    ``blend == 0`` never calls mutual_knn and reproduces the CLIP-only CSR.
    """
    device = clip_edge_index.device
    with torch.no_grad():
        if blend > 0:
            combs = []
            for s in range(0, num_nodes, chunk):
                e = min(s + chunk, num_nodes)
                c = model.combine(combine_feat_table[s:e], None, z_table[s:e].detach())
                combs.append(c.detach())
            comb_all = torch.cat(combs, dim=0)  # [N, Dp]

            A = mutual_knn(comb_all.float().cpu().numpy(), k, device=knn_device)  # scipy csr
            coo = A.tocoo()
            keep_dir = coo.row < coo.col  # one direction per edge, drop self-loops
            src = torch.from_numpy(coo.row[keep_dir].astype(np.int64))
            dst = torch.from_numpy(coo.col[keep_dir].astype(np.int64))
            comb_edges = torch.stack([src, dst], dim=0).to(device)  # [2, Mc]

            m = comb_edges.shape[1]
            if blend < 1.0 and m > 0:
                keep = int(round(blend * m))
                perm = torch.randperm(m, generator=generator)[:keep].to(device)
                comb_edges = comb_edges[:, perm]
        else:
            comb_edges = torch.empty(2, 0, dtype=torch.long, device=device)

        edge_index = torch.cat([clip_edge_index, comb_edges], dim=1)
        indptr, indices = build_neighbor_csr(edge_index, num_nodes)

        clip_set = _undirected_edge_set(clip_edge_index)
        comb_set = _undirected_edge_set(comb_edges)
        new_frac = len(comb_set - clip_set) / max(len(comb_set), 1)
        stats = {
            "graph_n_comb_edges": float(comb_edges.shape[1]),
            "graph_new_edge_frac": float(new_frac),
            "graph_avg_degree": float(indptr[-1].item()) / max(num_nodes, 1),
        }
    return indptr, indices, comb_edges, stats
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && PYTHONPATH=/project/CoSiR python src/test/20260707_buddy_refresh/test_buddy_refresh.py`
Expected: `ALL PASS` (6 `PASS` lines). The equivalence/blend/jaccard tests run CPU-only (they monkeypatch or skip `mutual_knn`).

- [ ] **Step 5: Stage (do NOT commit)**

```bash
git add src/metrics/regularizer.py src/test/20260707_buddy_refresh/test_buddy_refresh.py
```
Then STOP — the user commits.

---

### Task 2: Wire refresh into the train loop (`train_cosir.py`) + smoke script

**Files:**
- Modify: `src/hook/train_cosir.py` (import; setup block ~1306–1347; epoch-boundary hook ~1389)
- Create: `scripts/run_buddyrefresh_smoke.sh`

**Interfaces:**
- Consumes: `refresh_buddy_graph`, `edge_jaccard` (Task 1); existing `buddy_indptr/buddy_indices`, `other_feat_table`, `reorder_features_to_z`, `_lambda_buddy_con`, `_edge_index` (the loaded CLIP edges), `embedding_manager.sample_ids`, `logger.log_train`.
- Produces: five new `getattr` config knobs, a `combine_feat_table`, and a scheduled rebind of `buddy_indptr/buddy_indices`. No new public API.

- [ ] **Step 1: Extend the regularizer import block**

In `src/hook/train_cosir.py`, the `from src.metrics.regularizer import (…)` block currently ends with `buddy_contrastive_loss,` and `reorder_features_to_z,`. Add two names:

```python
    buddy_contrastive_loss,
    reorder_features_to_z,
    refresh_buddy_graph,
    edge_jaccard,
```

- [ ] **Step 2: Retain the CLIP edge tensor + read refresh config (setup block)**

In the setup block (~line 1313), change `buddy_indptr = buddy_indices = None` to also declare the CLIP-edge holder, and inside the `else:` branch (where `_edge_index` is built, ~1323) keep a reference:

```python
    buddy_indptr = buddy_indices = None
    other_feat_table = None
    _clip_edge_index = None
    if _lambda_buddy > 0 or _lambda_buddy_con > 0:
        _edges = embedding_manager.get_buddy_edges()
        if _edges is None:
            print("[buddy] lambda_buddy/lambda_buddy_con>0 but no buddy_edges.npy "
                  "found — disabling buddy terms for this run.")
            _lambda_buddy = 0.0
            _lambda_buddy_con = 0.0
        else:
            _edge_index = torch.from_numpy(_edges.astype(np.int64)).to(device)
            _clip_edge_index = _edge_index
            buddy_indptr, buddy_indices = build_neighbor_csr(
                _edge_index, num_nodes=len(embedding_manager.sample_ids)
            )
            print(f"[buddy] edges loaded: {_edge_index.shape[1]:,}; "
                  f"lambda_buddy={_lambda_buddy}, lambda_buddy_con={_lambda_buddy_con}")
```

Then, immediately AFTER the Family #2 `other_feat_table` block (after the `[buddy-con] enabled:` print, ~line 1347), add the Family #3 setup:

```python
    # Family #3: self-refreshing buddy graph. Recompute the CSR fed to the #2
    # contrastive term from the evolving combined features on a schedule.
    _buddy_refresh = bool(getattr(cfg.loss, "buddy_refresh", False))
    _buddy_refresh_warmup = int(getattr(cfg.loss, "buddy_refresh_warmup", 50))
    _buddy_refresh_period = int(getattr(cfg.loss, "buddy_refresh_period", 50))
    _buddy_refresh_blend = float(getattr(cfg.loss, "buddy_refresh_blend", 1.0))
    _buddy_refresh_k = int(getattr(cfg.loss, "buddy_refresh_k", 30))
    combine_feat_table = None
    _refresh_gen = torch.Generator().manual_seed(0)
    _prev_comb_edges = None
    if _buddy_refresh:
        if _lambda_buddy_con <= 0 or other_feat_table is None or _clip_edge_index is None:
            print("[buddy-refresh] requires lambda_buddy_con>0 with buddy edges — "
                  "disabling refresh for this run.")
            _buddy_refresh = False
        else:
            _combine_key = "img_features" if cfg.model.combine_side == "img" else "txt_features"
            _cfeat = feature_manager.load_all_to_ram([_combine_key])
            combine_feat_table = reorder_features_to_z(
                _cfeat[_combine_key],
                [int(s) for s in _cfeat["sample_ids"].tolist()],
                embedding_manager.sample_ids,
            ).to(device)
            print(f"[buddy-refresh] enabled: warmup={_buddy_refresh_warmup}, "
                  f"period={_buddy_refresh_period}, blend={_buddy_refresh_blend}, "
                  f"k={_buddy_refresh_k}, combine_feat={_combine_key} "
                  f"{tuple(combine_feat_table.shape)}")
```

- [ ] **Step 3: Add the scheduled refresh at the epoch boundary**

In the epoch loop, after the EM-phase block (right after the `else: em_phase = "both"` at ~line 1389, before `epoch_start_time = time.time()`), insert:

```python
        # Family #3: refresh the buddy graph feeding the #2 term, on schedule.
        if (
            _buddy_refresh
            and _lambda_buddy_con > 0
            and combine_feat_table is not None
            and embedding_manager.embeddings.requires_grad
            and epoch >= _buddy_refresh_warmup
            and (epoch - _buddy_refresh_warmup) % _buddy_refresh_period == 0
        ):
            buddy_indptr, buddy_indices, _comb_edges, _refresh_stats = refresh_buddy_graph(
                model,
                combine_feat_table,
                embedding_manager.embeddings,
                _clip_edge_index,
                num_nodes=len(embedding_manager.sample_ids),
                k=_buddy_refresh_k,
                blend=_buddy_refresh_blend,
                generator=_refresh_gen,
            )
            if _prev_comb_edges is not None:
                _refresh_stats["graph_churn"] = edge_jaccard(_comb_edges, _prev_comb_edges)
            _prev_comb_edges = _comb_edges
            logger.log_train(_refresh_stats, epoch=epoch, section="buddy_refresh")
            print(f"[buddy-refresh] epoch {epoch}: {_refresh_stats}")
```

- [ ] **Step 4: Create the smoke script**

Create `scripts/run_buddyrefresh_smoke.sh`:

```bash
#!/bin/bash
set -euo pipefail
# Family #3 smoke: exercises the refresh code path end-to-end on a tiny run.
# warmup=1, period=1 so refresh fires at epochs 1 and 2; blend=1 = full refresh.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_buddyrefresh_smoke/impressions}"
python main_cosir.py \
  dataset=impressions \
  eval.evaluation_interval=100 \
  model=clip_base \
  model.num_layers=6 \
  model.embedding_dim=16 \
  optimizer.lr=1e-4 \
  optimizer.lr_label=1e-4 \
  train.buddies.alpha=0.5 \
  train.epochs=3 \
  +loss.lambda_buddy=0 \
  +loss.lambda_buddy_con=0.3 \
  +loss.buddy_con_samples=4 \
  +loss.buddy_con_temperature=0.07 \
  +loss.buddy_refresh=true \
  +loss.buddy_refresh_warmup=1 \
  +loss.buddy_refresh_period=1 \
  +loss.buddy_refresh_blend=1.0 \
  +loss.buddy_refresh_k=30 \
  experiment.results_dir="${RESULTS_DIR}"
```

- [ ] **Step 5: Verify default-off is unchanged (no refresh path)**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && cd /project/CoSiR && python -c "import ast; ast.parse(open('src/hook/train_cosir.py').read()); print('parse-ok')"`
Expected: `parse-ok` (syntax valid). Confirm by inspection that with `buddy_refresh` unset, `_buddy_refresh` is `False`, so the setup builds no table and the epoch hook is skipped — pipeline unchanged.

- [ ] **Step 6: Run the smoke to verify the refresh path fires**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && cd /project/CoSiR && bash scripts/run_buddyrefresh_smoke.sh 2>&1 | tail -40`
Expected: `[buddy-refresh] enabled:` at setup, then `[buddy-refresh] epoch 1: {...}` and `[buddy-refresh] epoch 2: {...}` lines showing `graph_n_comb_edges`, `graph_new_edge_frac`, `graph_avg_degree` (and `graph_churn` at epoch 2). Training completes 3 epochs without error.

- [ ] **Step 7: Stage (do NOT commit)**

```bash
git add src/hook/train_cosir.py scripts/run_buddyrefresh_smoke.sh
```
Then STOP — the user commits.

---

### Task 3: Full ablation runner + change log

**Files:**
- Create: `scripts/run_buddyrefresh_full.sh`
- Create: `.claude/20260707_log.md`

**Interfaces:**
- Consumes: the config knobs wired in Task 2.
- Produces: a Hydra multirun ablation script and a change log. No code.

- [ ] **Step 1: Create the full ablation runner**

Create `scripts/run_buddyrefresh_full.sh` (mirrors `scripts/run_buddycon_full.sh`; sweeps only `buddy_refresh_blend`, isolating static vs refreshed graph):

```bash
#!/bin/bash
set -euo pipefail
# Full-scale FOCUSED ABLATION for Family #3 (self-refreshing buddy graph).
#
# Holds Family #1 OFF and the #2 contrastive term ON (lambda_buddy_con fixed),
# and sweeps ONLY +loss.buddy_refresh_blend. blend=0 = static Family #2 graph
# (baseline); blend=1.0 = full CLIP-anchored refresh. Because none of the
# buddy_refresh* keys are part of the buddy template key, every arm reuses the
# SAME buddy init template — so the arms differ only by the training-time graph.
#
# RESULTS_DIR is intentionally FRESH so each per-(dim,alpha) template is rebuilt
# with buddy_edges.npy present (needed by both #2 and the refresh union).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
    IFS=',' read -r -a __cvd_arr <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_PROCS=${#__cvd_arr[@]}
  else
    NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
  fi
else
  NUM_PROCS=1
fi
[ -z "${NUM_PROCS}" ] && NUM_PROCS=1
[ "${NUM_PROCS}" -lt 1 ] && NUM_PROCS=1
echo "Using ${NUM_PROCS} processes"

# ── Swept axis ───────────────────────────────────────────────────────────────
BUDDY_REFRESH_BLEND_SWEEP="${BUDDY_REFRESH_BLEND_SWEEP:-0, 1.0}"  # 0 = static #2 baseline

# ── Held constant ────────────────────────────────────────────────────────────
LAMBDA_BUDDYCON="${LAMBDA_BUDDYCON:-0.3}"
BUDDY_CON_SAMPLES="${BUDDY_CON_SAMPLES:-4}"
BUDDY_CON_TEMP="${BUDDY_CON_TEMP:-0.07}"
BUDDY_REFRESH_WARMUP="${BUDDY_REFRESH_WARMUP:-50}"
BUDDY_REFRESH_PERIOD="${BUDDY_REFRESH_PERIOD:-50}"
BUDDY_REFRESH_K="${BUDDY_REFRESH_K:-30}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
LR="${LR:-1e-4}"
LR_LABEL="${LR_LABEL:-1e-4}"
ALPHA="${ALPHA:-0.5}"
EPOCHS="${EPOCHS:-500}"
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_buddyrefresh_ablation/impressions}"

python main_cosir.py -m \
  dataset=impressions \
  eval.evaluation_interval=100 \
  eval.oracle_aggregation=max \
  model=clip_base \
  model.num_layers=6 \
  model.embedding_dim="${EMBEDDING_DIM}" \
  optimizer.lr="${LR}" \
  optimizer.lr_label="${LR_LABEL}" \
  train.buddies.alpha="${ALPHA}" \
  train.epochs="${EPOCHS}" \
  +loss.lambda_buddy=0 \
  +loss.lambda_buddy_con="${LAMBDA_BUDDYCON}" \
  +loss.buddy_con_samples="${BUDDY_CON_SAMPLES}" \
  +loss.buddy_con_temperature="${BUDDY_CON_TEMP}" \
  +loss.buddy_refresh=true \
  +loss.buddy_refresh_warmup="${BUDDY_REFRESH_WARMUP}" \
  +loss.buddy_refresh_period="${BUDDY_REFRESH_PERIOD}" \
  +loss.buddy_refresh_blend="${BUDDY_REFRESH_BLEND_SWEEP}" \
  +loss.buddy_refresh_k="${BUDDY_REFRESH_K}" \
  experiment.results_dir="${RESULTS_DIR}" \
  wandb.group="buddy-refresh ablation"
```

- [ ] **Step 2: Write the change log**

Create `.claude/20260707_log.md` documenting the modified source files. Structure with each file path as a header per the repo convention:

```markdown
# Family #3 — Self-Refreshing Buddies (change log, 2026-07-07)

Design: docs/superpowers/specs/2026-07-07-buddy-self-refresh-design.md
Plan:   docs/superpowers/plans/2026-07-07-buddy-self-refresh.md

## /src/metrics/regularizer.py
Added `refresh_buddy_graph(...)` — rebuilds the buddy CSR from the model's current
combined features (chunked no_grad `model.combine` pass → comb-space `mutual_knn` →
blend a fraction of comb edges with the always-kept frozen CLIP edges →
`build_neighbor_csr`). Returns `(indptr, indices, comb_edges, stats)`; `blend=0`
skips `mutual_knn` and reproduces the CLIP-only CSR. Added `edge_jaccard(...)` (used
for the `graph_churn` diagnostic) and private `_undirected_edge_set`. New imports:
`numpy`, and `mutual_knn` from `src.conditional_buddy.buddy_graph`.

## /src/hook/train_cosir.py
- Imported `refresh_buddy_graph, edge_jaccard`.
- Setup: retained the loaded CLIP edge tensor as `_clip_edge_index`; read five
  `getattr` knobs (`buddy_refresh`, `buddy_refresh_warmup/period/blend/k`); built a
  frozen `combine_feat_table` (combine-side pooled feature in z-order) when refresh
  is enabled (requires `lambda_buddy_con>0` + buddy edges + RAM feature store).
- Epoch boundary: when enabled and `epoch>=warmup` and on period, call
  `refresh_buddy_graph`, rebind `buddy_indptr/buddy_indices`, log
  `graph_*`/`graph_churn` under section `buddy_refresh`.
- Default-off (`buddy_refresh` unset) ⇒ byte-for-byte unchanged.

## scripts/
- `run_buddyrefresh_smoke.sh` — 3-epoch smoke; refresh fires at epochs 1–2.
- `run_buddyrefresh_full.sh` — Hydra multirun sweeping `buddy_refresh_blend ∈ {0,1.0}`
  with #1 off and #2 on; 0 = static #2 baseline, 1.0 = full refresh.
```

- [ ] **Step 3: Stage (do NOT commit)**

```bash
git add -f .claude/20260707_log.md
git add scripts/run_buddyrefresh_full.sh
```
Then STOP — the user commits.

---

## Self-Review

**Spec coverage:**
- Refresh source = combined space → Task 2 `combine_feat_table` + Task 1 `model.combine` pass. ✓
- Consumer = #2 term via CSR swap → Task 2 rebind of `buddy_indptr/buddy_indices`. ✓
- Guard = blend with always-kept CLIP graph, `blend=0 ⇒ #2` → Task 1 union logic + `test_equivalence_at_blend_zero`. ✓
- Schedule = warmup then every R → Task 2 epoch-boundary condition. ✓
- Config (5 getattr knobs) → Task 2 Step 2. ✓
- Diagnostics (`graph_churn`, `graph_new_edge_frac`, `graph_avg_degree`, keep `buddy_con_alignment`) → Task 1 stats + Task 2 churn/log; `buddy_con_alignment` already logged by #2. ✓
- Tests (equivalence, index alignment, union/blend, no-grad, churn) → Task 1 test file (6 tests). ✓
- Runner → Task 3 `run_buddyrefresh_full.sh`. ✓
- Guards (refresh needs lambda_buddy_con>0; streaming/no-RAM disabled; no CLIP edges disabled) → Task 2 Step 2 disable branch (reuses #2's RAM/edge guards). ✓

**Placeholder scan:** none — every code/test/command step is concrete.

**Type consistency:** `refresh_buddy_graph` returns `(indptr, indices, comb_edges, stats)` in Task 1 and is unpacked identically in Task 2 Step 3. `edge_jaccard(comb_edges, prev)` matches its Task 1 signature. `stats` keys (`graph_n_comb_edges`, `graph_new_edge_frac`, `graph_avg_degree`) are produced in Task 1 and logged in Task 2. `combine_feat_table` built via `reorder_features_to_z` matches #2's `other_feat_table` pattern. ✓
