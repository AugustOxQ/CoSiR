# Buddy Contrastive Supervision (Family #2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently-gated InfoNCE term that pulls each anchor's fused feature toward its buddies' projected other-side features, turning the buddy graph into extra positives for the retrieval objective.

**Architecture:** A new loss function `buddy_contrastive_loss` in `src/metrics/regularizer.py` (beside Family #1's `buddy_graph_smoothness_loss`), plus a small `reorder_features_to_z` index helper. Wired into `src/hook/train_cosir.py` by (a) extending Family #1's edge/CSR setup block to also trigger when `lambda_buddy_con > 0`, (b) building a frozen `other_feat_table` once, and (c) adding the term in the training loop after Family #1's term. Reuses the persisted `buddy_edges.npy` (graph E) and `build_neighbor_csr` from #1 — no new persistence.

**Tech Stack:** PyTorch, Hydra/OmegaConf config, the existing CoSiR `train_cosir` loop, `model.project_other` (trainable identity-init linear), `model.combine` / `CombinerGated`.

## Global Constraints

- **Default-off:** `lambda_buddy_con = 0.0` (default) ⇒ no `other_feat_table` build, no loop term, byte-for-byte unchanged pipeline.
- **Config via `getattr`:** `lambda_buddy_con`, `buddy_con_samples`, `buddy_con_temperature` are NOT in any YAML; read with `getattr(cfg.loss, ...)` defaults and ADDED on the CLI with a leading `+` (e.g. `+loss.lambda_buddy_con=0.3`). Existing keys use plain override.
- **Index space:** edges/CSR are in z-table position order (`embedding_manager.sample_ids`); `anchor_positions == batch_indices` (already z-order via `embedding_manager.id_to_index`); `other_feat_table` MUST be built aligned to `embedding_manager.sample_ids` by mapping through the feature store's own `sample_ids` (never assume the two orders match).
- **Reuse #1, add nothing persisted:** use `embedding_manager.get_buddy_edges()` and `build_neighbor_csr`; do not add new `.npy` files or touch `compute_buddies.py` / `init_conditions.py`.
- **Graph = E (union):** the same `buddy_edges.npy` Family #1 uses.
- **Positive target:** the buddy's non-combine-side pooled feature passed through `model.project_other`. Underlying feature frozen; gradient flows into `project_other`, `z_i` (anchor), and the combiner — never into buddy `z_j`.
- **Negatives:** the batch's own `other_emb`, with the anchor's own row masked (self-mask only; rare in-batch buddy false-negatives are accepted).
- **Gating guard:** apply the term only when `lambda_buddy_con > 0` AND the CSR is present AND `embedding_manager.embeddings.requires_grad` (skips the EM "network" phase, parallel to #1).
- **Grad-safe zero:** when no batch anchor has a buddy, return `comb_emb.sum() * 0.0` (a real zero with a valid graph), never a bare `0.0`.
- **Streaming guard:** if `lambda_buddy_con > 0` but `feature_manager.fits_in_ram()` is False, print a `[buddy-con]` warning and disable the term (out of scope).
- **Git: the user owns commits.** Each task's final step STAGES with `git add` (use `git add -f` for `.claude/`) and STOPS. Do NOT run `git commit`.
- **Test style:** self-contained scripts with a `__main__` runner and plain `assert`s, matching `src/test/20260623_buddy_train_reg/test_buddy_reg.py`. Run with `python <path>`, not pytest.
- **Activate env first:** `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR` before running anything.

---

### Task 1: Core loss `buddy_contrastive_loss`

**Files:**
- Modify: `src/metrics/regularizer.py` (append one function; reuse existing `build_neighbor_csr`)
- Test: `src/test/20260624_buddy_contrastive/test_buddy_contrastive.py` (create)

**Interfaces:**
- Consumes: `build_neighbor_csr(edge_index: LongTensor[2,M], num_nodes: int) -> (indptr LongTensor[N+1], indices LongTensor[2M])` — already in `regularizer.py`.
- Produces:
  `buddy_contrastive_loss(comb_emb: Tensor[B,Dp], anchor_positions: LongTensor[B], other_feat_table: Tensor[N,Dfeat], project_other: Callable[[Tensor],Tensor], other_emb_neg: Tensor[B,Dp], indptr: LongTensor[N+1], indices: LongTensor, num_pos: int = 4, temperature: float = 0.07, generator: Optional[torch.Generator] = None) -> tuple[Tensor scalar, Tensor scalar]`
  Returns `(loss, alignment)`; `alignment` is the detached mean cosine between active anchors and their sampled buddy positives.

- [ ] **Step 1: Write the failing tests**

Create `src/test/20260624_buddy_contrastive/test_buddy_contrastive.py`:

```python
import torch
import torch.nn.functional as F

from src.metrics.regularizer import buddy_contrastive_loss, build_neighbor_csr


def _toy_csr(edges, n):
    if len(edges) == 0:
        ei = torch.empty(2, 0, dtype=torch.long)
    else:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return build_neighbor_csr(ei, n)


def test_isolated_anchor_zero():
    # node 0—1 are buddies; node 2 is isolated. Batch = [2] only -> exactly 0, no grad.
    indptr, indices = _toy_csr([[0, 1]], 3)
    N, D = 3, 5
    table = torch.randn(N, D)
    comb = torch.randn(1, D, requires_grad=True)
    neg = torch.randn(1, D)
    loss, align = buddy_contrastive_loss(
        comb, torch.tensor([2]), table, lambda x: x, neg, indptr, indices, num_pos=2
    )
    assert loss.item() == 0.0
    assert align.item() == 0.0
    loss.backward()  # must not raise; grad-safe zero
    print("test_isolated_anchor_zero PASS")


def test_positive_gathered_from_right_row():
    # anchor at z-pos 0, single buddy z-pos 3. alignment must equal cos(comb, table[3]).
    indptr, indices = _toy_csr([[0, 3]], 4)
    N, D = 4, 6
    table = F.normalize(torch.randn(N, D), dim=-1)
    comb = F.normalize(torch.randn(1, D), dim=-1)
    neg = F.normalize(torch.randn(1, D), dim=-1)
    _, align = buddy_contrastive_loss(
        comb, torch.tensor([0]), table, lambda x: x, neg, indptr, indices, num_pos=1
    )
    expected = F.cosine_similarity(comb, table[3:4], dim=-1).item()
    assert abs(align.item() - expected) < 1e-5
    print("test_positive_gathered_from_right_row PASS")


def test_self_masking_excludes_own_row():
    # B=1: the only candidate negative is the anchor's own row. Masking it leaves
    # only positives -> loss is exactly 0. If self were NOT masked, a real negative
    # would remain and the loss would be > 0. (Scaling the negative would NOT test
    # this, since the loss L2-normalizes negatives and so ignores their magnitude.)
    indptr, indices = _toy_csr([[0, 1]], 2)
    N, D = 2, 4
    table = F.normalize(torch.randn(N, D), dim=-1)
    comb = F.normalize(torch.randn(1, D), dim=-1)
    neg = F.normalize(torch.randn(1, D), dim=-1)
    loss, _ = buddy_contrastive_loss(comb, torch.tensor([0]), table, lambda x: x, neg, indptr, indices, num_pos=1)
    assert abs(loss.item()) < 1e-6, loss.item()
    print("test_self_masking_excludes_own_row PASS")


def test_gradient_raises_alignment():
    # Optimising comb should pull anchors toward their (frozen) buddy targets.
    torch.manual_seed(0)
    indptr, indices = _toy_csr([[0, 1], [2, 3]], 4)
    N, D = 4, 8
    table = F.normalize(torch.randn(N, D), dim=-1)
    comb = torch.randn(2, D, requires_grad=True)  # anchors at z-pos 0 and 2
    neg = F.normalize(torch.randn(2, D), dim=-1)
    anchor_pos = torch.tensor([0, 2])

    def alignment(c):
        with torch.no_grad():
            _, a = buddy_contrastive_loss(c, anchor_pos, table, lambda x: x, neg, indptr, indices, num_pos=1)
        return a.item()

    before = alignment(comb)
    opt = torch.optim.SGD([comb], lr=1.0)
    for _ in range(100):
        opt.zero_grad()
        loss, _ = buddy_contrastive_loss(comb, anchor_pos, table, lambda x: x, neg, indptr, indices, num_pos=1)
        loss.backward()
        opt.step()
    after = alignment(comb)
    assert after > before + 0.1, (before, after)
    print("test_gradient_raises_alignment PASS")


def test_scalar_and_finite():
    indptr, indices = _toy_csr([[0, 1], [1, 2]], 3)
    N, D = 3, 4
    table = F.normalize(torch.randn(N, D), dim=-1)
    comb = F.normalize(torch.randn(3, D), dim=-1)
    neg = F.normalize(torch.randn(3, D), dim=-1)
    pos = torch.tensor([0, 1, 2])
    for temp in (0.07, 1.0):
        loss, align = buddy_contrastive_loss(comb, pos, table, lambda x: x, neg, indptr, indices, num_pos=2, temperature=temp)
        assert loss.dim() == 0 and torch.isfinite(loss)
        assert torch.isfinite(align)
    print("test_scalar_and_finite PASS")


if __name__ == "__main__":
    test_isolated_anchor_zero()
    test_positive_gathered_from_right_row()
    test_self_masking_excludes_own_row()
    test_gradient_raises_alignment()
    test_scalar_and_finite()
    print("ALL PASS")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260624_buddy_contrastive/test_buddy_contrastive.py`
Expected: FAIL with `ImportError: cannot import name 'buddy_contrastive_loss'`.

- [ ] **Step 3: Implement `buddy_contrastive_loss`**

Append to `src/metrics/regularizer.py` (the file already imports `torch`, `torch.nn.functional as F`, and `Optional`):

```python
def buddy_contrastive_loss(
    comb_emb: torch.Tensor,
    anchor_positions: torch.Tensor,
    other_feat_table: torch.Tensor,
    project_other,
    other_emb_neg: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    num_pos: int = 4,
    temperature: float = 0.07,
    generator: Optional[torch.Generator] = None,
):
    """Multi-positive InfoNCE pulling anchors toward their buddies along E.

    For each batch anchor, sample up to ``num_pos`` buddies from the CSR graph,
    look up their (frozen) other-side features, project them into retrieval space
    with ``project_other``, and treat them as positives. Negatives are the batch's
    own ``other_emb_neg`` rows, with each anchor's own row masked out. Anchors with
    no buddy are skipped; if none have a buddy the loss is exactly 0 (grad-safe).

    comb_emb:         [B, Dp]   anchor combined features (differentiable).
    anchor_positions: [B]       z-table positions of the batch (== batch_indices).
    other_feat_table: [N, Dfeat] frozen other-side features in z-table order.
    project_other:    callable [*, Dfeat] -> [*, Dp].
    other_emb_neg:    [B, Dp]   in-batch negatives (the batch's other_emb).
    indptr, indices:  CSR neighbour structure from build_neighbor_csr (graph E).
    num_pos:          buddies sampled per anchor (with replacement).
    temperature:      softmax temperature.
    Returns (loss scalar, alignment scalar detached = mean cos(anchor, buddy)).
    """
    device = comb_emb.device
    deg = indptr[anchor_positions + 1] - indptr[anchor_positions]   # [B]
    mask = deg > 0
    if not torch.any(mask):
        return comb_emb.sum() * 0.0, torch.zeros((), device=device)

    active = torch.nonzero(mask, as_tuple=False).squeeze(1)        # [A] batch rows
    anchors = anchor_positions[active]                            # [A] z-positions
    degm = deg[mask].unsqueeze(1)                                # [A, 1]
    starts = indptr[anchors].unsqueeze(1)                        # [A, 1]
    A = anchors.shape[0]
    rand = torch.rand(A, num_pos, device=device, generator=generator)
    offsets = torch.clamp((rand * degm).long(), max=degm - 1)     # [A, num_pos] in [0, deg)
    buddy_pos = indices[starts + offsets]                        # [A, num_pos] z-positions

    q = F.normalize(comb_emb[active], dim=-1)                    # [A, Dp]
    pos = F.normalize(project_other(other_feat_table[buddy_pos]), dim=-1)  # [A, num_pos, Dp]
    neg = F.normalize(other_emb_neg, dim=-1)                     # [B, Dp]

    logits_pos = torch.einsum("ad,akd->ak", q, pos) / temperature   # [A, num_pos]
    logits_neg = (q @ neg.t()) / temperature                        # [A, B]
    self_mask = torch.zeros(A, neg.shape[0], dtype=torch.bool, device=device)
    self_mask[torch.arange(A, device=device), active] = True
    logits_neg = logits_neg.masked_fill(self_mask, float("-inf"))

    logits_all = torch.cat([logits_pos, logits_neg], dim=1)         # [A, num_pos + B]
    log_z = torch.logsumexp(logits_all, dim=1, keepdim=True)        # [A, 1]
    log_prob_pos = logits_pos - log_z                              # [A, num_pos]
    loss = -log_prob_pos.mean(dim=1).mean()
    alignment = (logits_pos * temperature).mean().detach()
    return loss, alignment
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260624_buddy_contrastive/test_buddy_contrastive.py`
Expected: prints each `... PASS` line then `ALL PASS`.

- [ ] **Step 5: Stage (do NOT commit)**

```bash
git add src/metrics/regularizer.py src/test/20260624_buddy_contrastive/test_buddy_contrastive.py
```
Stop here — the user runs `git commit`.

---

### Task 2: Index helper `reorder_features_to_z`

**Files:**
- Modify: `src/metrics/regularizer.py` (append one pure function)
- Test: `src/test/20260624_buddy_contrastive/test_reorder.py` (create)

**Interfaces:**
- Produces: `reorder_features_to_z(feat: Tensor[M,Dfeat], feat_ids: list[int], z_ids: list[int]) -> Tensor[len(z_ids),Dfeat]` — returns `feat` reindexed so row `p` holds the feature for sample `z_ids[p]`.

- [ ] **Step 1: Write the failing test**

Create `src/test/20260624_buddy_contrastive/test_reorder.py`:

```python
import torch

from src.metrics.regularizer import reorder_features_to_z


def test_reorder_aligns_to_z_order():
    feat = torch.tensor([[10.0], [20.0], [30.0], [40.0]])  # in feature-store order
    feat_ids = [103, 101, 104, 102]                        # store order of sample ids
    z_ids = [101, 102, 103, 104]                           # embedding-manager order
    out = reorder_features_to_z(feat, feat_ids, z_ids)
    assert out[0].item() == 20.0  # z-pos 0 = sample 101 -> store idx 1
    assert out[1].item() == 40.0  # 102 -> store idx 3
    assert out[2].item() == 10.0  # 103 -> store idx 0
    assert out[3].item() == 30.0  # 104 -> store idx 2
    print("test_reorder_aligns_to_z_order PASS")


def test_reorder_identity_when_orders_match():
    feat = torch.randn(5, 3)
    ids = [7, 8, 9, 10, 11]
    out = reorder_features_to_z(feat, ids, ids)
    assert torch.equal(out, feat)
    print("test_reorder_identity_when_orders_match PASS")


if __name__ == "__main__":
    test_reorder_aligns_to_z_order()
    test_reorder_identity_when_orders_match()
    print("ALL PASS")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260624_buddy_contrastive/test_reorder.py`
Expected: FAIL with `ImportError: cannot import name 'reorder_features_to_z'`.

- [ ] **Step 3: Implement `reorder_features_to_z`**

Append to `src/metrics/regularizer.py`:

```python
def reorder_features_to_z(feat: torch.Tensor, feat_ids, z_ids) -> torch.Tensor:
    """Reindex a feature-store tensor into embedding-manager (z-table) order.

    feat:     [M, Dfeat] features in feature-store order.
    feat_ids: list[int] sample ids in feature-store order (len M).
    z_ids:    list[int] sample ids in z-table order (embedding_manager.sample_ids).
    Returns [len(z_ids), Dfeat] where row p is the feature for sample z_ids[p].
    """
    fpos = {int(sid): i for i, sid in enumerate(feat_ids)}
    order = torch.tensor([fpos[int(sid)] for sid in z_ids], dtype=torch.long)
    return feat[order].contiguous()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260624_buddy_contrastive/test_reorder.py`
Expected: prints both `... PASS` lines then `ALL PASS`.

- [ ] **Step 5: Stage (do NOT commit)**

```bash
git add src/metrics/regularizer.py src/test/20260624_buddy_contrastive/test_reorder.py
```
Stop here — the user runs `git commit`.

---

### Task 3: Wire the term into `train_cosir`

**Files:**
- Modify: `src/hook/train_cosir.py:49` (import), the Family-#1 setup block at `src/hook/train_cosir.py:1301-1318`, and the loop term after Family #1's term at `src/hook/train_cosir.py:1523-1538`.

**Interfaces:**
- Consumes: `buddy_contrastive_loss(...)` and `reorder_features_to_z(...)` from Task 1/2; `build_neighbor_csr` (already imported); `embedding_manager.get_buddy_edges()`, `embedding_manager.sample_ids`, `embedding_manager.id_to_index`, `embedding_manager.embeddings`; `feature_manager.fits_in_ram()`, `feature_manager.load_all_to_ram([key])` (returns a dict with the feature key and `sample_ids`); loop locals `comb_emb`, `other_emb`, `batch_indices`; `model.project_other`; `cfg.model.combine_side`.
- Produces: `loss_dict["loss_buddy_con"]` and `loss_dict["buddy_con_alignment"]` per active batch.

**Note on the diagnostic:** the spec described `buddy_con_alignment` "on the eval cadence"; this plan logs it per-batch via `loss_dict` (the existing `loss_metrics` path `.item()`s tensors automatically), which is simpler and consistent with how `loss_buddy` is logged. Same signal, finer granularity.

- [ ] **Step 1: Extend the import**

In `src/hook/train_cosir.py`, replace line 49:

```python
from src.metrics.regularizer import build_neighbor_csr, buddy_graph_smoothness_loss
```
with:
```python
from src.metrics.regularizer import (
    build_neighbor_csr,
    buddy_graph_smoothness_loss,
    buddy_contrastive_loss,
    reorder_features_to_z,
)
```

- [ ] **Step 2: Extend the Family-#1 setup block to also drive Family #2**

Replace the block at `src/hook/train_cosir.py:1301-1318` (from the `# Family #1 buddy regularizer:` comment through the `print(f"[buddy-reg] enabled: ...")` line) with:

```python
    # Families #1 & #2 share the persisted E edge list + CSR neighbour index.
    # Disabled (no-op) when both lambdas are 0 or edges are absent.
    _lambda_buddy = getattr(cfg.loss, "lambda_buddy", 0.0)
    _buddy_reg_samples = int(getattr(cfg.loss, "buddy_reg_samples", 4))
    _lambda_buddy_con = getattr(cfg.loss, "lambda_buddy_con", 0.0)
    _buddy_con_samples = int(getattr(cfg.loss, "buddy_con_samples", 4))
    _buddy_con_temp = float(getattr(cfg.loss, "buddy_con_temperature", 0.07))
    buddy_indptr = buddy_indices = None
    other_feat_table = None
    if _lambda_buddy > 0 or _lambda_buddy_con > 0:
        _edges = embedding_manager.get_buddy_edges()
        if _edges is None:
            print("[buddy] lambda_buddy/lambda_buddy_con>0 but no buddy_edges.npy "
                  "found — disabling buddy terms for this run.")
            _lambda_buddy = 0.0
            _lambda_buddy_con = 0.0
        else:
            _edge_index = torch.from_numpy(_edges.astype(np.int64)).to(device)
            buddy_indptr, buddy_indices = build_neighbor_csr(
                _edge_index, num_nodes=len(embedding_manager.sample_ids)
            )
            print(f"[buddy] edges loaded: {_edge_index.shape[1]:,}; "
                  f"lambda_buddy={_lambda_buddy}, lambda_buddy_con={_lambda_buddy_con}")

    # Family #2: gather the non-combine-side pooled feature per sample, in z-table
    # order, so anchor combined features can be pulled toward buddy targets.
    if _lambda_buddy_con > 0 and buddy_indptr is not None:
        if not feature_manager.fits_in_ram():
            print("[buddy-con] feature store does not fit in RAM (streaming path) — "
                  "buddy contrastive term unsupported here; disabling.")
            _lambda_buddy_con = 0.0
        else:
            _other_key = "txt_features" if cfg.model.combine_side == "img" else "img_features"
            _feat = feature_manager.load_all_to_ram([_other_key])
            other_feat_table = reorder_features_to_z(
                _feat[_other_key],
                [int(s) for s in _feat["sample_ids"].tolist()],
                embedding_manager.sample_ids,
            ).to(device)
            print(f"[buddy-con] enabled: lambda_buddy_con={_lambda_buddy_con}, "
                  f"samples/anchor={_buddy_con_samples}, temp={_buddy_con_temp}, "
                  f"other_feat={_other_key} {tuple(other_feat_table.shape)}")
```

(The `_z_init` snapshot block immediately below — lines 1320-1324 — is unchanged.)

- [ ] **Step 3: Add the term in the training loop**

In `src/hook/train_cosir.py`, immediately after Family #1's term (the block ending with `loss_dict["loss_buddy"] = buddy_loss.detach()` at line 1538), insert:

```python
            # Family #2: buddy contrastive supervision in combined/retrieval space
            if (
                _lambda_buddy_con > 0
                and other_feat_table is not None
                and embedding_manager.embeddings.requires_grad
            ):
                _anchor_pos_con = torch.tensor(batch_indices, device=device, dtype=torch.long)
                buddy_con_loss, buddy_con_align = buddy_contrastive_loss(
                    comb_emb,
                    _anchor_pos_con,
                    other_feat_table,
                    model.project_other,
                    other_emb,
                    buddy_indptr,
                    buddy_indices,
                    num_pos=_buddy_con_samples,
                    temperature=_buddy_con_temp,
                )
                loss = loss + _lambda_buddy_con * buddy_con_loss
                loss_dict["loss_buddy_con"] = buddy_con_loss.detach()
                loss_dict["buddy_con_alignment"] = buddy_con_align
```

- [ ] **Step 4: Verify the file imports and parses**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -c "import ast; ast.parse(open('src/hook/train_cosir.py').read()); print('parse OK')"`
Expected: `parse OK`.

- [ ] **Step 5: Smoke-run the term end-to-end**

Run (writes to a throwaway results dir so the buddy init/template is rebuilt WITH `buddy_edges.npy`):
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main_cosir.py \
  dataset=impressions model=clip_base model.num_layers=6 model.embedding_dim=16 \
  train.epochs=2 \
  +loss.lambda_buddy_con=0.3 +loss.buddy_con_samples=4 \
  experiment.results_dir="res/CoSiR_buddycon_smoke/impressions" \
  wandb.group="buddy-con smoke"
```
Expected: console prints `[buddy-con] enabled: lambda_buddy_con=0.3, ...` once before the loop; training runs without error; a finite `loss_buddy_con` and `buddy_con_alignment` are logged. (If it instead prints `[buddy-con] ... disabling`, the results dir held a pre-feature template — delete it and re-run.)

- [ ] **Step 6: Stage (do NOT commit)**

```bash
git add src/hook/train_cosir.py
```
Stop here — the user runs `git commit`.

---

### Task 4: Change log

**Files:**
- Create: `.claude/20260624_log.md` (gitignored → stage with `git add -f`)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Write the change log**

Create `.claude/20260624_log.md` documenting the two modified source files, following the repo convention (file path as header, what changed, why). Include:

```markdown
# Family #2: Buddy Contrastive Supervision — change log (2026-06-24)

Spec: docs/superpowers/specs/2026-06-24-buddy-contrastive-supervision-design.md
Plan: docs/superpowers/plans/2026-06-24-buddy-contrastive-supervision.md

## /src/metrics/regularizer.py

Added two functions beside Family #1's `buddy_graph_smoothness_loss`:

- `buddy_contrastive_loss(comb_emb, anchor_positions, other_feat_table,
  project_other, other_emb_neg, indptr, indices, num_pos=4, temperature=0.07,
  generator=None) -> (loss, alignment)` — multi-positive InfoNCE pulling each
  anchor's combined feature toward its buddies' projected other-side features;
  negatives are the batch's other_emb with the anchor's own row masked. Anchors
  with no buddy are skipped; grad-safe zero when none have a buddy.
- `reorder_features_to_z(feat, feat_ids, z_ids)` — reindex a feature-store tensor
  into embedding-manager (z-table) order; the index-alignment guard for the
  other-side feature table.

## /src/hook/train_cosir.py

- Extended the import on line 49 to include `buddy_contrastive_loss` and
  `reorder_features_to_z`.
- Generalised the Family-#1 edge/CSR setup block so it loads edges + builds the
  CSR when EITHER `lambda_buddy > 0` OR `lambda_buddy_con > 0`, and (for #2) builds
  a frozen `other_feat_table` of the non-combine-side pooled feature in z-table
  order. Streaming feature stores are warned + disabled.
- Added the Family #2 term in the training loop after the Family #1 term, gated on
  `lambda_buddy_con > 0 and other_feat_table is not None and
  embedding_manager.embeddings.requires_grad`. Logs `loss_buddy_con` and
  `buddy_con_alignment`.

Config keys (read via getattr, not in YAML; add with `+` on the CLI):
`lambda_buddy_con` (0.0), `buddy_con_samples` (4), `buddy_con_temperature` (0.07).

Backward compatible: `lambda_buddy_con=0` ⇒ no table build, no term, unchanged
pipeline. Independent of `lambda_buddy` ⇒ supports a clean #1/#2/both/neither
ablation on one shared buddy init template.
```

- [ ] **Step 2: Stage (do NOT commit)**

```bash
git add -f .claude/20260624_log.md
```
Stop here — the user runs `git commit`.

---

## Post-implementation (not a task)

To sweep Family #2 like the v4 lambda_buddy sweep, add a `lambda_buddy_con` axis to
`scripts/run_sweep_agent.py` (mirror the existing `lambda_buddy` handling — read
`getattr(wandb.config, "lambda_buddy_con", None)`, append `+loss.lambda_buddy_con=...`)
and a `sweep_config_v5.yaml`. The 2×2 #1/#2 ablation reuses one buddy init template
because neither lambda is part of the template key. Run with a FRESH `RESULTS_DIR`
so `buddy_edges.npy` is present.
```
