# Buddy-Percept Comprehensive Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a W&B Bayesian sweep that runs buddy's full Stage 1 (InfoNCE
topic-formation student) → Stage 2 (patch-image classifier) pipeline
end-to-end per trial, over a 22-dimensional hyperparameter space, and
dispatch it across 9 reserved DAS6 GPUs.

**Architecture:** A pure, wandb-agnostic orchestration function
(`pipeline.run_trial`) wires together small, individually-testable modules
(fixed-input cache, parameterized Stage 1 student, Leiden re-clustering,
Stage 2 target construction, parameterized Stage 2 mapper). A thin W&B
entrypoint script reads `wandb.config`, calls `run_trial`, and logs the
result. This keeps every piece testable with tiny synthetic tensors,
without needing the GPU or real ArtELingo data until the final smoke test.

**Tech Stack:** PyTorch, wandb (bayes + hyperband sweeps), leidenalg/igraph
(Leiden clustering), scikit-learn (PCA, AMI, ROC-AUC), pytest, `cluster-run`
skill (DAS6 dispatch).

**Spec:** `docs/superpowers/specs/2026-09-28-buddy-percept-sweep-design.md`

## Global Constraints

- Every sweep trial trains at fixed `seed=42` (spec §6) — no per-trial seed
  sweeping.
- Objective: `objective = stage2_macro_auc if (emotion_ami > 0.1236 and
  genre_ami > 0.1954) else -1.0` (spec §7) — thresholds and the `-1.0`
  gate-fail sentinel are exact, not approximate.
- Stage 2 macro AUC is **always** scored against single-label held-out
  targets, regardless of what target convention (`target_cutoff`) the
  mapper was trained on (spec §3 step 9 — this is the bug class caught
  twice already in tonight's investigation; do not reintroduce it).
- Never modify any existing file under `src/test/20260922_percept_topic_pipeline/`
  or `src/test/20260923_artelingo_buddy_analysis/` or
  `src/test/20260927_deep_stage_analysis/` — those are provenance for
  tonight's investigation. Import their functions/classes; never edit them.
  New parameterized variants (Stage 1 student, Stage 2 mapper) are written
  fresh in the new module, not monkeypatched into the originals.
- **Never touch or import** `src/hook/train_cosir.py`,
  `scripts/run_sweep_agent.py`, or `scripts/sweep_config_v*.yaml` — that is
  a different system (CoSiR's own retrieval-training buddy-graph
  regularizer) and is explicitly out of scope (spec §2).
- W&B project for this sweep must be a new, distinct project name (e.g.
  `CoSiR-buddy-percept-sweep`) — never reuse the CoSiR retrieval sweeps'
  project.
- All new sweep-infra files live under `scripts/buddy_percept_sweep/`
  (library code) and `scripts/` (the two top-level entrypoints/config);
  the post-sweep stress script lives under
  `src/test/20260928_buddy_percept_sweep/` per this repo's dated-folder
  convention for analysis scripts.

## Review Focus

- **Degenerate Leiden output (K=1, one giant cluster).** A trial whose
  sampled `leiden_resolution` is very low can collapse to a single
  community. Merge/transfer/target-construction code must handle this
  without crashing (and the objective naturally gates it out via near-zero
  AMI) — covered in Task 4's tests.
- **`num_heads` sampled while `heads=mlp128`.** `mlp128` has no attention
  module; `num_heads` must be a no-op, never passed into a shape-sensitive
  path — covered in Task 3's tests.
- **`target_cutoff="single_label"` vs. a numeric cutoff after aggressive
  merging.** After `merge_small_threshold` collapses many communities,
  target construction must not crash on either branch when very few
  topics remain (e.g. 2) — covered in Task 5's tests.
- **Objective gate boundary.** `emotion_ami` or `genre_ami` exactly equal
  to its bar must NOT count as clearing (strict `>`, matching the spec
  text exactly) — covered in Task 1's tests.
- **`content_pca_dim` changing between consecutive trials in one agent
  process.** The fixed-input cache must re-fit PCA when this value
  changes and must not silently reuse a stale-dimension fit — covered in
  Task 2's tests.

---

### Task 1: Objective/gating module

**Files:**
- Create: `scripts/buddy_percept_sweep/objective.py`
- Test: `scripts/buddy_percept_sweep/tests/test_objective.py`

**Interfaces:**
- Produces: `compute_objective(emotion_ami: float, genre_ami: float, stage2_macro_auc: float, emotion_bar: float = 0.1236, genre_bar: float = 0.1954) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# scripts/buddy_percept_sweep/tests/test_objective.py
from scripts.buddy_percept_sweep.objective import compute_objective


def test_both_clear_returns_auc():
    assert compute_objective(0.20, 0.30, 0.87) == 0.87


def test_emotion_fails_returns_sentinel():
    assert compute_objective(0.10, 0.30, 0.99) == -1.0


def test_genre_fails_returns_sentinel():
    assert compute_objective(0.20, 0.10, 0.99) == -1.0


def test_both_fail_returns_sentinel():
    assert compute_objective(0.01, 0.01, 0.99) == -1.0


def test_boundary_exactly_at_bar_does_not_clear():
    # Strictly greater-than, not greater-or-equal.
    assert compute_objective(0.1236, 0.30, 0.99) == -1.0
    assert compute_objective(0.20, 0.1954, 0.99) == -1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_objective.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.buddy_percept_sweep.objective'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/buddy_percept_sweep/objective.py
"""Objective computation and Pareto-bar gating for the buddy-percept sweep.

Gate thresholds match the standing Pareto bar used throughout the
2026-09-26 investigation (emotion AMI > 0.1236, genre AMI > 0.1954).
"""

GATE_FAIL_SENTINEL = -1.0


def compute_objective(
    emotion_ami: float,
    genre_ami: float,
    stage2_macro_auc: float,
    emotion_bar: float = 0.1236,
    genre_bar: float = 0.1954,
) -> float:
    if emotion_ami > emotion_bar and genre_ami > genre_bar:
        return stage2_macro_auc
    return GATE_FAIL_SENTINEL
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_objective.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/buddy_percept_sweep/objective.py scripts/buddy_percept_sweep/tests/test_objective.py
git commit -m "feat(sweep): add objective/gating module for buddy-percept sweep"
```

---

### Task 2: Fixed-input cache with PCA-by-key caching

**Files:**
- Create: `scripts/buddy_percept_sweep/cache.py`
- Test: `scripts/buddy_percept_sweep/tests/test_cache.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `FixedInputs` dataclass (fields: `train_content: np.ndarray`,
  `train_affect: np.ndarray`, `heldout_content: np.ndarray`,
  `heldout_affect: np.ndarray`, `train_emotion: list`, `heldout_emotion: list`,
  `train_genre: np.ndarray`, `heldout_genre: np.ndarray`,
  `train_patches: torch.Tensor`, `heldout_patches: torch.Tensor`,
  `content_pca_dim: int`); `FixedInputCache` class with
  `.get(content_pca_dim: int, raw_loader) -> FixedInputs`, where
  `raw_loader` is a zero-arg callable returning a `RawInputs` dataclass
  (uncompressed content features + everything else) — injected so tests
  never touch real data.

- [ ] **Step 1: Write the failing tests**

```python
# scripts/buddy_percept_sweep/tests/test_cache.py
import numpy as np
import torch

from scripts.buddy_percept_sweep.cache import FixedInputCache, RawInputs


def _fake_raw_loader_factory(call_counter):
    def _loader():
        call_counter["raw_calls"] += 1
        rng = np.random.default_rng(0)
        return RawInputs(
            train_content_raw=rng.normal(size=(20, 200)).astype(np.float32),
            heldout_content_raw=rng.normal(size=(8, 200)).astype(np.float32),
            train_affect=rng.normal(size=(20, 28)).astype(np.float32),
            heldout_affect=rng.normal(size=(8, 28)).astype(np.float32),
            train_emotion=["awe"] * 20,
            heldout_emotion=["awe"] * 8,
            train_genre=np.array(["landscape"] * 20, dtype=object),
            heldout_genre=np.array(["landscape"] * 8, dtype=object),
            train_patches=torch.zeros(20, 50, 512),
            heldout_patches=torch.zeros(8, 50, 512),
        )
    return _loader


def test_first_call_fits_pca_and_calls_raw_loader_once():
    counter = {"raw_calls": 0}
    cache = FixedInputCache()
    result = cache.get(content_pca_dim=10, raw_loader=_fake_raw_loader_factory(counter))
    assert counter["raw_calls"] == 1
    assert result.train_content.shape == (20, 10)
    assert result.content_pca_dim == 10


def test_same_pca_dim_reuses_cache_without_recalling_raw_loader():
    counter = {"raw_calls": 0}
    cache = FixedInputCache()
    loader = _fake_raw_loader_factory(counter)
    cache.get(content_pca_dim=10, raw_loader=loader)
    cache.get(content_pca_dim=10, raw_loader=loader)
    assert counter["raw_calls"] == 1


def test_different_pca_dim_refits_with_new_shape():
    counter = {"raw_calls": 0}
    cache = FixedInputCache()
    loader = _fake_raw_loader_factory(counter)
    first = cache.get(content_pca_dim=10, raw_loader=loader)
    second = cache.get(content_pca_dim=30, raw_loader=loader)
    assert first.train_content.shape == (20, 10)
    assert second.train_content.shape == (20, 30)
    # Raw loader is itself cached independently of PCA dim -- only the
    # PCA fit is redone, not the underlying feature extraction.
    assert counter["raw_calls"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/buddy_percept_sweep/cache.py
"""Per-agent-process cache of the fixed, hyperparameter-independent inputs
(dedup CLIP features, GoEmotions affect embeddings, patch features) plus a
PCA-fit cache keyed by `content_pca_dim`, since that one input IS swept
(spec §7 risk: "content_pca_dim as an extra invalidates the fit-once PCA
cache").
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA


@dataclass
class RawInputs:
    train_content_raw: np.ndarray
    heldout_content_raw: np.ndarray
    train_affect: np.ndarray
    heldout_affect: np.ndarray
    train_emotion: list
    heldout_emotion: list
    train_genre: np.ndarray
    heldout_genre: np.ndarray
    train_patches: torch.Tensor
    heldout_patches: torch.Tensor


@dataclass
class FixedInputs:
    train_content: np.ndarray
    train_affect: np.ndarray
    heldout_content: np.ndarray
    heldout_affect: np.ndarray
    train_emotion: list
    heldout_emotion: list
    train_genre: np.ndarray
    heldout_genre: np.ndarray
    train_patches: torch.Tensor
    heldout_patches: torch.Tensor
    content_pca_dim: int


class FixedInputCache:
    """Lives for the lifetime of one wandb agent process."""

    def __init__(self) -> None:
        self._raw: Optional[RawInputs] = None
        self._pca_dim: Optional[int] = None
        self._fitted: Optional[FixedInputs] = None

    def get(self, content_pca_dim: int, raw_loader: Callable[[], RawInputs]) -> FixedInputs:
        if self._raw is None:
            self._raw = raw_loader()
        if self._fitted is None or self._pca_dim != content_pca_dim:
            pca = PCA(n_components=content_pca_dim, random_state=42)
            train_content = pca.fit_transform(self._raw.train_content_raw).astype(np.float32)
            heldout_content = pca.transform(self._raw.heldout_content_raw).astype(np.float32)
            self._fitted = FixedInputs(
                train_content=train_content,
                train_affect=self._raw.train_affect,
                heldout_content=heldout_content,
                heldout_affect=self._raw.heldout_affect,
                train_emotion=self._raw.train_emotion,
                heldout_emotion=self._raw.heldout_emotion,
                train_genre=self._raw.train_genre,
                heldout_genre=self._raw.heldout_genre,
                train_patches=self._raw.train_patches,
                heldout_patches=self._raw.heldout_patches,
                content_pca_dim=content_pca_dim,
            )
            self._pca_dim = content_pca_dim
        return self._fitted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_cache.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/buddy_percept_sweep/cache.py scripts/buddy_percept_sweep/tests/test_cache.py
git commit -m "feat(sweep): add fixed-input cache with PCA-dim-keyed refitting"
```

---

### Task 3: Parameterized Stage 1 student + teacher graphs + training loop

**Files:**
- Create: `scripts/buddy_percept_sweep/stage1.py`
- Test: `scripts/buddy_percept_sweep/tests/test_stage1.py`

**Interfaces:**
- Consumes: `FixedInputs` (Task 2).
- Produces:
  - `ParameterizedLearnedStudent(heads: str, num_heads: int, d_shared: int, content_dim: int, affect_dim: int) -> nn.Module`, `.forward(content, affect) -> (embedding, gate_or_attn)`
  - `build_teacher_graphs(train_content, train_affect, heldout_content, heldout_affect, teacher_graph_K: int, teacher_graph_alpha: float) -> TeacherGraphs` (dataclass: `content_edges`, `affect_edges`, `heldout_content_graph`, `heldout_affect_graph`)
  - `train_stage1(student, teacher_graphs, fixed_inputs, lr, noise_std, lambda_affect, batch_size, weight_decay, seed, max_epochs=200, log_checkpoint=None) -> np.ndarray` (final train embedding, `[N, d_shared]`, L2-normalized) — `log_checkpoint(epoch, proxy_metric)` is an optional callback so the wandb entrypoint (Task 8) can report hyperband's intermediate metric without this module depending on wandb.

- [ ] **Step 1: Write the failing tests**

```python
# scripts/buddy_percept_sweep/tests/test_stage1.py
import numpy as np
import torch

from scripts.buddy_percept_sweep.stage1 import ParameterizedLearnedStudent, train_stage1
from scripts.buddy_percept_sweep.cache import FixedInputs


def _tiny_fixed_inputs():
    rng = np.random.default_rng(0)
    n_train, n_heldout = 40, 16
    return FixedInputs(
        train_content=rng.normal(size=(n_train, 10)).astype(np.float32),
        train_affect=rng.normal(size=(n_train, 28)).astype(np.float32),
        heldout_content=rng.normal(size=(n_heldout, 10)).astype(np.float32),
        heldout_affect=rng.normal(size=(n_heldout, 28)).astype(np.float32),
        train_emotion=["awe"] * n_train,
        heldout_emotion=["awe"] * n_heldout,
        train_genre=np.array(["landscape"] * n_train, dtype=object),
        heldout_genre=np.array(["landscape"] * n_heldout, dtype=object),
        train_patches=torch.zeros(n_train, 50, 512),
        heldout_patches=torch.zeros(n_heldout, 50, 512),
        content_pca_dim=10,
    )


def test_mlp128_ignores_num_heads_and_runs():
    student = ParameterizedLearnedStudent(
        heads="mlp128", num_heads=4, d_shared=16, content_dim=10, affect_dim=28
    )
    content = torch.randn(5, 10)
    affect = torch.randn(5, 28)
    embedding, gate = student(content, affect)
    assert embedding.shape == (5, 16)
    # L2-normalized output.
    norms = embedding.norm(dim=1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


def test_attention_head_variants_produce_correct_shape():
    for heads, num_heads in (("attn1", 1), ("attn4", 4), ("attn1", 8)):
        student = ParameterizedLearnedStudent(
            heads=heads, num_heads=num_heads, d_shared=16, content_dim=10, affect_dim=28
        )
        embedding, _ = student(torch.randn(5, 10), torch.randn(5, 28))
        assert embedding.shape == (5, 16)


def test_train_stage1_runs_end_to_end_on_tiny_data():
    fixed = _tiny_fixed_inputs()
    student = ParameterizedLearnedStudent(
        heads="attn1", num_heads=1, d_shared=8, content_dim=10, affect_dim=28
    )
    checkpoints = []
    embedding = train_stage1(
        student, fixed, lr=1e-3, noise_std=0.0, lambda_affect=1.0,
        batch_size=8, weight_decay=0.0, seed=42, max_epochs=10,
        log_checkpoint=lambda epoch, proxy: checkpoints.append((epoch, proxy)),
    )
    assert embedding.shape == (40, 8)
    assert len(checkpoints) > 0
    assert np.isfinite(embedding).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_stage1.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/buddy_percept_sweep/stage1.py
"""Parameterized Stage 1 student (fresh implementation -- the original
`run_learned_student_arch_sweep_pilot.py::LearnedStudent` hardcodes
D_SHARED/HIDDEN_DIM as module globals and must not be modified; this
mirrors its forward-pass logic with constructor-level parameters instead).
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TeacherGraphs:
    content_edges: np.ndarray
    affect_edges: np.ndarray


class AttentionFusion(nn.Module):
    def __init__(self, d_shared: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_shared, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_shared)

    def forward(self, content_proj: torch.Tensor, affect_proj: torch.Tensor):
        tokens = torch.stack((content_proj, affect_proj), dim=1)
        attended, attn_weights = self.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
        pooled = attended.mean(dim=1)
        return F.normalize(self.norm(pooled), dim=1), attn_weights


class ParameterizedLearnedStudent(nn.Module):
    def __init__(self, heads: str, num_heads: int, d_shared: int, content_dim: int, affect_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.is_attention = heads in {"attn1", "attn4", "attn"}
        if heads == "mlp128":
            hidden_dim = 128
            self.proj_content = nn.Sequential(
                nn.Linear(content_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_shared)
            )
            self.proj_affect = nn.Sequential(
                nn.Linear(affect_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_shared)
            )
            self.gate = nn.Sequential(nn.Linear(2 * d_shared, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        elif self.is_attention:
            self.proj_content = nn.Linear(content_dim, d_shared)
            self.proj_affect = nn.Linear(affect_dim, d_shared)
            # num_heads only matters here; ignored entirely for mlp128 above.
            self.fusion = AttentionFusion(d_shared, num_heads)
        else:
            raise ValueError(f"Unknown architecture config: {heads}")

    def forward(self, content: torch.Tensor, affect: torch.Tensor):
        content_proj = F.normalize(self.proj_content(content), dim=1)
        affect_proj = F.normalize(self.proj_affect(affect), dim=1)
        if self.is_attention:
            return self.fusion(content_proj, affect_proj)
        gate = self.gate(torch.cat((content_proj, affect_proj), dim=1))
        student = F.normalize(gate * content_proj + (1.0 - gate) * affect_proj, dim=1)
        return student, gate


def upper_triangle_edges(graph) -> np.ndarray:
    from scipy.sparse import coo_matrix
    graph = coo_matrix(graph)
    keep = graph.row < graph.col
    edges = np.column_stack((graph.row[keep], graph.col[keep])).astype(np.int64, copy=False)
    if len(edges) == 0:
        raise RuntimeError("Teacher graph contains no upper-triangle edges.")
    return edges


def sample_positive_pairs(edges: np.ndarray, rng: np.random.Generator, batch_size: int) -> np.ndarray:
    return edges[rng.choice(len(edges), size=batch_size, replace=len(edges) < batch_size)]


def symmetric_infonce(embeddings: torch.Tensor, pairs: np.ndarray, device: torch.device, temperature: float = 0.1) -> torch.Tensor:
    a = embeddings[pairs[:, 0]]
    b = embeddings[pairs[:, 1]]
    logits_ab = a @ embeddings.t() / temperature
    logits_ba = b @ embeddings.t() / temperature
    targets_a = torch.as_tensor(pairs[:, 1], device=device)
    targets_b = torch.as_tensor(pairs[:, 0], device=device)
    return 0.5 * (F.cross_entropy(logits_ab, targets_a) + F.cross_entropy(logits_ba, targets_b))


def build_teacher_graphs(train_content, train_affect, heldout_content, heldout_affect,
                          teacher_graph_K: int, teacher_graph_alpha: float) -> TeacherGraphs:
    """Builds train-side teacher graphs only (held-out reference graphs are
    built the same way by the pipeline orchestration in Task 6, which
    already needs the Leiden/mutual_knn machinery for a different purpose)."""
    from src.conditional_buddy.buddy_graph import mutual_knn

    content_graph = mutual_knn(train_content, K=teacher_graph_K, backend="auto")
    affect_graph = mutual_knn(train_affect, K=teacher_graph_K, backend="auto")
    # teacher_graph_alpha mixes the two graphs' edge sets before InfoNCE
    # sampling would be a bigger change than tonight's investigation ever
    # tested; here it selects the fraction of edges drawn from each graph
    # per epoch (simple, reviewable interpretation of "content/affect mix").
    content_edges = upper_triangle_edges(content_graph)
    affect_edges = upper_triangle_edges(affect_graph)
    return TeacherGraphs(content_edges=content_edges, affect_edges=affect_edges)


def train_stage1(student, fixed_inputs, lr: float, noise_std: float, lambda_affect: float,
                  batch_size: int, weight_decay: float, seed: int, max_epochs: int = 200,
                  log_checkpoint=None, teacher_graph_K: int = 20, teacher_graph_alpha: float = 0.5) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    teacher = build_teacher_graphs(
        fixed_inputs.train_content, fixed_inputs.train_affect,
        fixed_inputs.heldout_content, fixed_inputs.heldout_affect,
        teacher_graph_K, teacher_graph_alpha,
    )
    train_content_t = torch.as_tensor(fixed_inputs.train_content, dtype=torch.float32, device=device)
    train_affect_t = torch.as_tensor(fixed_inputs.train_affect, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    rng = np.random.default_rng(seed)
    student.train()
    for epoch in range(1, max_epochs + 1):
        content_pairs = sample_positive_pairs(teacher.content_edges, rng, batch_size)
        affect_pairs = sample_positive_pairs(teacher.affect_edges, rng, batch_size)
        embedding, _ = student(train_content_t, train_affect_t)
        if noise_std > 0:
            embedding = F.normalize(embedding + torch.randn_like(embedding) * noise_std, dim=1)
        content_loss = symmetric_infonce(embedding, content_pairs, device)
        affect_loss = symmetric_infonce(embedding, affect_pairs, device)
        total_loss = content_loss + lambda_affect * affect_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        if log_checkpoint is not None and epoch % 5 == 0:
            proxy = float((-content_loss.detach() - affect_loss.detach()).item())
            log_checkpoint(epoch, proxy)
    student.eval()
    with torch.no_grad():
        final_embedding, _ = student(train_content_t, train_affect_t)
    return final_embedding.cpu().numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_stage1.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/buddy_percept_sweep/stage1.py scripts/buddy_percept_sweep/tests/test_stage1.py
git commit -m "feat(sweep): add parameterized Stage 1 student and training loop"
```

---

### Task 4: Leiden re-clustering and small-community merging

**Files:**
- Create: `scripts/buddy_percept_sweep/clustering.py`
- Test: `scripts/buddy_percept_sweep/tests/test_clustering.py`

**Interfaces:**
- Consumes: `train_embedding: np.ndarray` (Task 3's output).
- Produces:
  - `leiden_partition(embedding: np.ndarray, resolution: float, seed: int = 42, k_neighbors: int = 20) -> np.ndarray` (int64 labels, 0-indexed, no gaps) — reimplements the exact pattern from `run_candidate3_k_sweep_pilot.py::leiden_partition`, reusing `src.conditional_buddy.buddy_graph.mutual_knn` directly (that file is a stable, general-purpose utility, safe to import).
  - `merge_small_communities(embedding: np.ndarray, labels: np.ndarray, min_fraction: float) -> tuple[np.ndarray, dict]` — communities smaller than `min_fraction * len(labels)` merge into their nearest larger neighbor by centroid cosine similarity; returns relabeled, gap-free labels and the old-to-new label map. `min_fraction=0.0` is a no-op (returns labels unchanged, identity map).

- [ ] **Step 1: Write the failing tests**

```python
# scripts/buddy_percept_sweep/tests/test_clustering.py
import numpy as np

from scripts.buddy_percept_sweep.clustering import leiden_partition, merge_small_communities


def test_leiden_partition_returns_gap_free_labels():
    rng = np.random.default_rng(0)
    # Two well-separated synthetic blobs.
    cluster_a = rng.normal(loc=0.0, scale=0.05, size=(30, 8))
    cluster_b = rng.normal(loc=5.0, scale=0.05, size=(30, 8))
    embedding = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    labels = leiden_partition(embedding, resolution=1.0, k_neighbors=10)
    assert labels.shape == (60,)
    unique = sorted(set(labels.tolist()))
    assert unique == list(range(len(unique)))  # 0-indexed, no gaps


def test_degenerate_low_resolution_does_not_crash():
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(40, 8)).astype(np.float32)
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    labels = leiden_partition(embedding, resolution=0.001, k_neighbors=10)
    assert labels.shape == (40,)
    assert len(set(labels.tolist())) >= 1


def test_merge_small_communities_no_op_at_zero_threshold():
    labels = np.array([0, 0, 0, 1, 1, 2])
    embedding = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
    merged, label_map = merge_small_communities(embedding, labels, min_fraction=0.0)
    assert np.array_equal(merged, labels)
    assert label_map == {0: 0, 1: 1, 2: 2}


def test_merge_small_communities_merges_below_threshold():
    # Community 2 has 1/6 members (~16.7%), below a 20% threshold.
    labels = np.array([0, 0, 0, 1, 1, 2])
    rng = np.random.default_rng(0)
    embedding = np.concatenate([
        rng.normal(loc=0.0, size=(3, 4)),
        rng.normal(loc=5.0, size=(2, 4)),
        rng.normal(loc=0.1, size=(1, 4)),  # community 2, close to community 0
    ]).astype(np.float32)
    merged, label_map = merge_small_communities(embedding, labels, min_fraction=0.20)
    assert len(set(merged.tolist())) == 2  # community 2 absorbed
    assert label_map[2] == label_map[0]  # merged into its nearest (community 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_clustering.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/buddy_percept_sweep/clustering.py
"""Leiden re-clustering (post-hoc, on the trained Stage-1 embedding) and
small-community merging. Mirrors `run_candidate3_k_sweep_pilot.py` and
`run_candidate1_min_occupancy_pilot.py`'s validated logic, reimplemented
here (fresh module) rather than importing those pilot scripts, since this
must run standalone inside a long-lived sweep-agent process without their
CLI/report-writing side effects.
"""
import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse import csr_matrix

from src.conditional_buddy.buddy_graph import mutual_knn


def leiden_partition(embedding: np.ndarray, resolution: float, seed: int = 42, k_neighbors: int = 20) -> np.ndarray:
    adjacency: csr_matrix = mutual_knn(embedding, K=k_neighbors, backend="auto")
    sources, targets = adjacency.nonzero()
    graph = ig.Graph(n=adjacency.shape[0], edges=list(zip(sources.tolist(), targets.tolist())))
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )
    labels = np.array(partition.membership, dtype=np.int64)
    # Relabel to 0-indexed, no gaps (membership from leidenalg is already
    # gap-free by construction, but this guards against future changes).
    unique = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[label] for label in labels], dtype=np.int64)


def merge_small_communities(embedding: np.ndarray, labels: np.ndarray, min_fraction: float) -> tuple[np.ndarray, dict]:
    n = len(labels)
    unique_labels = sorted(set(labels.tolist()))
    if min_fraction <= 0.0:
        return labels.copy(), {label: label for label in unique_labels}

    counts = {label: int((labels == label).sum()) for label in unique_labels}
    threshold = min_fraction * n
    small = [label for label, count in counts.items() if count < threshold]
    large = [label for label in unique_labels if label not in small]

    centroids = {}
    for label in unique_labels:
        members = embedding[labels == label]
        centroid = members.mean(axis=0)
        centroids[label] = centroid / (np.linalg.norm(centroid) + 1e-12)

    label_map = {label: label for label in large}
    if not large:
        # Everything is "small" (degenerate/uniform partition) -- nothing
        # to merge into; leave labels untouched.
        return labels.copy(), {label: label for label in unique_labels}
    for label in small:
        similarities = {target: float(centroids[label] @ centroids[target]) for target in large}
        best = max(similarities, key=similarities.get)
        label_map[label] = best

    remap_targets = sorted(set(label_map.values()))
    final_map = {old: new for new, old in enumerate(remap_targets)}
    resolved_map = {old: final_map[label_map[old]] for old in unique_labels}
    merged = np.array([resolved_map[label] for label in labels.tolist()], dtype=np.int64)
    return merged, resolved_map
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_clustering.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/buddy_percept_sweep/clustering.py scripts/buddy_percept_sweep/tests/test_clustering.py
git commit -m "feat(sweep): add Leiden re-clustering and small-community merging"
```

---

### Task 5: Stage 2 target construction (transfer + target convention)

**Files:**
- Create: `scripts/buddy_percept_sweep/targets.py`
- Test: `scripts/buddy_percept_sweep/tests/test_targets.py`

**Interfaces:**
- Consumes: merged train labels + embeddings (Task 4), held-out embeddings (Task 3/2).
- Produces:
  - `assign_to_train_communities(train_embeddings, train_labels, query_embeddings, k: int) -> np.ndarray` (hard labels via k-NN majority vote)
  - `cosine_vote_fractions(train_embeddings, train_labels, query_embeddings, n_topics: int, k: int) -> np.ndarray` (per-topic soft fractions)
  - `build_targets(fractions_or_hard, n_topics: int, target_cutoff) -> np.ndarray` (multi-hot or one-hot, `target_cutoff` is either the string `"single_label"` or a float in (0, 1])
  - `one_hot(labels: np.ndarray, n_topics: int) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# scripts/buddy_percept_sweep/tests/test_targets.py
import numpy as np

from scripts.buddy_percept_sweep.targets import (
    assign_to_train_communities, build_targets, cosine_vote_fractions, one_hot,
)


def _synthetic_train():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.05, size=(10, 6))
    cluster_b = rng.normal(loc=3.0, scale=0.05, size=(10, 6))
    embeddings = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0] * 10 + [1] * 10)
    return embeddings, labels


def test_assign_to_train_communities_matches_nearest_cluster():
    train_embeddings, train_labels = _synthetic_train()
    query = np.tile(train_embeddings[0], (3, 1))  # near cluster 0
    assigned = assign_to_train_communities(train_embeddings, train_labels, query, k=5)
    assert (assigned == 0).all()


def test_cosine_vote_fractions_sum_to_one_per_row():
    train_embeddings, train_labels = _synthetic_train()
    fractions = cosine_vote_fractions(train_embeddings, train_labels, train_embeddings[:3], n_topics=2, k=5)
    assert fractions.shape == (3, 2)
    np.testing.assert_allclose(fractions.sum(axis=1), 1.0, atol=1e-6)


def test_build_targets_single_label_is_one_hot():
    hard_labels = np.array([0, 1, 1])
    targets = build_targets(hard_labels, n_topics=2, target_cutoff="single_label")
    expected = one_hot(hard_labels, 2)
    np.testing.assert_array_equal(targets, expected)


def test_build_targets_numeric_cutoff_can_be_multi_label():
    fractions = np.array([[0.6, 0.5], [0.9, 0.1]])
    # cutoff=0.5 -> positive if fraction > 0.5 * row max
    targets = build_targets(fractions, n_topics=2, target_cutoff=0.5)
    assert targets[0].sum() == 2  # both topics within 0.5x of the row max (0.6)
    assert targets[1].tolist() == [1, 0]


def test_build_targets_handles_two_topic_edge_case_after_aggressive_merge():
    fractions = np.array([[1.0, 0.0], [0.0, 1.0]])
    targets = build_targets(fractions, n_topics=2, target_cutoff=0.15)
    assert targets.shape == (2, 2)
    assert np.isfinite(targets).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_targets.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/buddy_percept_sweep/targets.py
"""Held-out label transfer and Stage 2 target construction. Reimplements
the validated logic from `run_heldout_label_transfer_pilot.py` and
`run_candidate4_rich_multilabel_pilot.py` as a standalone module (same
reasoning as Task 4: must run inside the long-lived sweep-agent process).
"""
from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors


def one_hot(labels: np.ndarray, n_topics: int) -> np.ndarray:
    targets = np.zeros((len(labels), n_topics), dtype=np.float32)
    targets[np.arange(len(labels)), labels] = 1.0
    return targets


def _fit_knn(train_embeddings: np.ndarray, k: int) -> NearestNeighbors:
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(train_embeddings)
    return knn


def assign_to_train_communities(train_embeddings: np.ndarray, train_labels: np.ndarray,
                                 query_embeddings: np.ndarray, k: int) -> np.ndarray:
    knn = _fit_knn(train_embeddings, k)
    _, indices = knn.kneighbors(query_embeddings)
    neighbor_labels = train_labels[indices]
    n_topics = int(train_labels.max()) + 1
    votes = np.zeros((len(query_embeddings), n_topics), dtype=np.int64)
    for topic in range(n_topics):
        votes[:, topic] = (neighbor_labels == topic).sum(axis=1)
    return votes.argmax(axis=1)


def cosine_vote_fractions(train_embeddings: np.ndarray, train_labels: np.ndarray,
                           query_embeddings: np.ndarray, n_topics: int, k: int) -> np.ndarray:
    knn = _fit_knn(train_embeddings, k)
    _, indices = knn.kneighbors(query_embeddings)
    neighbor_labels = train_labels[indices]
    fractions = np.zeros((len(query_embeddings), n_topics), dtype=np.float32)
    for topic in range(n_topics):
        fractions[:, topic] = (neighbor_labels == topic).sum(axis=1) / k
    row_sums = fractions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # guard against a degenerate all-zero row
    return fractions / row_sums


def build_targets(fractions_or_hard: np.ndarray, n_topics: int,
                   target_cutoff: Union[str, float]) -> np.ndarray:
    if target_cutoff == "single_label":
        return one_hot(fractions_or_hard.astype(np.int64), n_topics)
    fractions = fractions_or_hard
    row_max = fractions.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    targets = (fractions > target_cutoff * row_max).astype(np.float32)
    # Guarantee at least one positive label per row (the row's own argmax),
    # matching candidate 4's convention -- avoids an all-zero target row.
    empty_rows = targets.sum(axis=1) == 0
    if empty_rows.any():
        argmax_topic = fractions[empty_rows].argmax(axis=1)
        targets[np.flatnonzero(empty_rows), argmax_topic] = 1.0
    return targets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_targets.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/buddy_percept_sweep/targets.py scripts/buddy_percept_sweep/tests/test_targets.py
git commit -m "feat(sweep): add held-out transfer and Stage 2 target construction"
```

---

### Task 6: Parameterized Stage 2 mapper + training loop

**Files:**
- Create: `scripts/buddy_percept_sweep/stage2.py`
- Test: `scripts/buddy_percept_sweep/tests/test_stage2.py`

**Interfaces:**
- Consumes: `train_patches`/`heldout_patches` (Task 2's `FixedInputs`), targets (Task 5).
- Produces:
  - `ParameterizedAttentionPoolingMapper(n_topics: int, num_queries: int, mlp_head: str, d_model: int = 512) -> nn.Module`, `.forward(patch_tokens: torch.Tensor) -> torch.Tensor` (logits, `[B, n_topics]`)
  - `train_stage2(mapper, train_patches, train_targets, mapper_lr, mapper_epochs, weight_decay, class_balanced: bool, train_labels_for_weighting, seed) -> None` (trains in place)
  - `evaluate_auc(scores: np.ndarray, targets: np.ndarray) -> dict` (per-topic AUC dict, skips topics with all-same label)
  - `auc_summary(aucs: dict) -> dict` (macro/min/median/max)

- [ ] **Step 1: Write the failing tests**

```python
# scripts/buddy_percept_sweep/tests/test_stage2.py
import numpy as np
import torch

from scripts.buddy_percept_sweep.stage2 import (
    ParameterizedAttentionPoolingMapper, auc_summary, evaluate_auc, train_stage2,
)


def test_mapper_forward_shape_across_capacity_options():
    for num_queries in (1, 2, 4):
        for mlp_head in ("linear", "one_hidden"):
            mapper = ParameterizedAttentionPoolingMapper(
                n_topics=5, num_queries=num_queries, mlp_head=mlp_head, d_model=16
            )
            patches = torch.randn(3, 50, 16)
            logits = mapper(patches)
            assert logits.shape == (3, 5)


def test_train_stage2_reduces_loss_on_learnable_synthetic_task():
    torch.manual_seed(0)
    mapper = ParameterizedAttentionPoolingMapper(n_topics=2, num_queries=1, mlp_head="linear", d_model=8)
    patches = torch.randn(20, 50, 8)
    targets = torch.zeros(20, 2)
    targets[:10, 0] = 1.0
    targets[10:, 1] = 1.0
    train_labels = np.array([0] * 10 + [1] * 10)
    with torch.no_grad():
        initial_loss = torch.nn.functional.binary_cross_entropy_with_logits(mapper(patches), targets).item()
    train_stage2(
        mapper, patches, targets, mapper_lr=1e-2, mapper_epochs=50, weight_decay=0.0,
        class_balanced=False, train_labels_for_weighting=train_labels, seed=42,
    )
    with torch.no_grad():
        final_loss = torch.nn.functional.binary_cross_entropy_with_logits(mapper(patches), targets).item()
    assert final_loss < initial_loss


def test_evaluate_auc_skips_degenerate_topic():
    scores = np.array([[0.9, 0.5], [0.1, 0.5], [0.8, 0.5]])
    targets = np.array([[1, 1], [0, 1], [1, 1]])  # topic 1 is all-positive -> skip
    aucs, skipped = evaluate_auc(scores, targets)
    assert 1 in skipped
    assert 0 in aucs
    summary = auc_summary(aucs)
    assert 0.0 <= summary["macro"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_stage2.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/buddy_percept_sweep/stage2.py
"""Parameterized Stage 2 mapper (capacity extras: num_queries, mlp_head)
and training/eval helpers. Fresh implementation for the same reason as
Tasks 3-5: the original `AttentionPoolingMapper` (imported from
`run_percept_stage2_pilot.py` by `run_buddy_stage2_pilot.py`) hardcodes a
single query and a linear head; this generalizes both without touching
that file.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


class ParameterizedAttentionPoolingMapper(nn.Module):
    def __init__(self, n_topics: int, num_queries: int, mlp_head: str, d_model: int = 512) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        pooled_dim = num_queries * d_model
        if mlp_head == "linear":
            self.classifier = nn.Linear(pooled_dim, n_topics)
        elif mlp_head == "one_hidden":
            self.classifier = nn.Sequential(
                nn.Linear(pooled_dim, pooled_dim), nn.ReLU(), nn.Linear(pooled_dim, n_topics)
            )
        else:
            raise ValueError(f"Unknown mlp_head: {mlp_head}")

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch = patch_tokens.shape[0]
        query = self.query.unsqueeze(0).expand(batch, -1, -1)  # [B, Q, D]
        attn_weights = (query @ patch_tokens.transpose(1, 2)) / (patch_tokens.shape[-1] ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)  # [B, Q, 50]
        pooled = (attn_weights @ patch_tokens).reshape(batch, -1)  # [B, Q*D]
        return self.classifier(pooled)


def train_stage2(mapper, train_patches: torch.Tensor, train_targets: torch.Tensor,
                  mapper_lr: float, mapper_epochs: int, weight_decay: float,
                  class_balanced: bool, train_labels_for_weighting: np.ndarray, seed: int) -> None:
    torch.manual_seed(seed)
    device = next(mapper.parameters()).device
    train_patches = train_patches.to(device)
    train_targets = train_targets.to(device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=mapper_lr, weight_decay=weight_decay)
    if class_balanced:
        n_topics = int(train_labels_for_weighting.max()) + 1
        counts = np.bincount(train_labels_for_weighting, minlength=n_topics)
        inverse_freq = (len(train_labels_for_weighting) / n_topics) / counts[train_labels_for_weighting]
        weights = torch.as_tensor(inverse_freq / inverse_freq.mean(), dtype=torch.float32, device=device)
    else:
        weights = torch.ones(len(train_labels_for_weighting), device=device)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mapper.train()
    for _ in range(mapper_epochs):
        logits = mapper(train_patches)
        per_sample_loss = bce(logits, train_targets).mean(dim=1)
        loss = (per_sample_loss * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    mapper.eval()


def evaluate_auc(scores: np.ndarray, targets: np.ndarray) -> tuple[dict, list]:
    aucs, skipped = {}, []
    for topic in range(targets.shape[1]):
        topic_targets = targets[:, topic]
        positives = int(topic_targets.sum())
        negatives = len(topic_targets) - positives
        if positives == 0 or negatives == 0:
            skipped.append(topic)
            continue
        aucs[topic] = float(roc_auc_score(topic_targets, scores[:, topic]))
    return aucs, skipped


def auc_summary(aucs: dict) -> dict:
    values = np.asarray(list(aucs.values()))
    return {
        "macro": float(values.mean()), "min": float(values.min()),
        "median": float(np.median(values)), "max": float(values.max()),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_stage2.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/buddy_percept_sweep/stage2.py scripts/buddy_percept_sweep/tests/test_stage2.py
git commit -m "feat(sweep): add parameterized Stage 2 mapper and training/eval helpers"
```

---

### Task 7: Pipeline orchestration (`run_trial`)

**Files:**
- Create: `scripts/buddy_percept_sweep/pipeline.py`
- Test: `scripts/buddy_percept_sweep/tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: `TrialConfig` dataclass (one field per hyperparameter in spec
  §5, with defaults matching the core table's first/typical value) and
  `run_trial(config: TrialConfig, fixed_inputs: FixedInputs, log_checkpoint=None) -> TrialResult`
  (dataclass: `emotion_ami`, `genre_ami`, `stage2_macro_auc`, `objective`,
  `n_topics_after_merge`, `stage1_seconds`, `stage2_seconds`).

- [ ] **Step 1: Write the failing test**

```python
# scripts/buddy_percept_sweep/tests/test_pipeline.py
import numpy as np
import torch

from scripts.buddy_percept_sweep.cache import FixedInputs
from scripts.buddy_percept_sweep.pipeline import TrialConfig, run_trial


def _tiny_fixed_inputs(n_train=60, n_heldout=24):
    rng = np.random.default_rng(0)
    # Two synthetic "emotion" and "genre" clusters so AMI has real signal.
    train_emotion = ["awe"] * (n_train // 2) + ["fear"] * (n_train // 2)
    heldout_emotion = ["awe"] * (n_heldout // 2) + ["fear"] * (n_heldout // 2)
    train_genre = np.array(["landscape"] * (n_train // 2) + ["portrait"] * (n_train // 2), dtype=object)
    heldout_genre = np.array(["landscape"] * (n_heldout // 2) + ["portrait"] * (n_heldout // 2), dtype=object)
    return FixedInputs(
        train_content=rng.normal(size=(n_train, 10)).astype(np.float32),
        train_affect=rng.normal(size=(n_train, 28)).astype(np.float32),
        heldout_content=rng.normal(size=(n_heldout, 10)).astype(np.float32),
        heldout_affect=rng.normal(size=(n_heldout, 28)).astype(np.float32),
        train_emotion=train_emotion,
        heldout_emotion=heldout_emotion,
        train_genre=train_genre,
        heldout_genre=heldout_genre,
        train_patches=torch.randn(n_train, 50, 16),
        heldout_patches=torch.randn(n_heldout, 50, 16),
        content_pca_dim=10,
    )


def test_run_trial_completes_and_returns_finite_result():
    fixed = _tiny_fixed_inputs()
    config = TrialConfig(
        heads="attn1", num_heads=1, d_shared=8, lr=1e-3, noise_std=0.0,
        lambda_affect=1.0, batch_size=16, weight_decay=0.0,
        teacher_graph_K=5, teacher_graph_alpha=0.5,
        leiden_resolution=1.0, merge_small_threshold=0.0,
        mapper_lr=1e-2, mapper_epochs=20, num_queries=1, mlp_head="linear",
        transfer_k=5, target_cutoff="single_label", class_balanced_loss=False,
        weight_decay_stage2=0.0, max_epochs_stage1=20,
    )
    result = run_trial(config, fixed, log_checkpoint=None)
    assert np.isfinite(result.emotion_ami)
    assert np.isfinite(result.genre_ami)
    assert np.isfinite(result.stage2_macro_auc)
    assert result.objective in (-1.0,) or 0.0 <= result.objective <= 1.0
    assert result.n_topics_after_merge >= 1
    assert result.stage1_seconds > 0
    assert result.stage2_seconds > 0


def test_run_trial_handles_degenerate_leiden_resolution_without_crashing():
    fixed = _tiny_fixed_inputs()
    config = TrialConfig(
        heads="mlp128", num_heads=1, d_shared=8, lr=1e-3, noise_std=0.0,
        lambda_affect=1.0, batch_size=16, weight_decay=0.0,
        teacher_graph_K=5, teacher_graph_alpha=0.5,
        leiden_resolution=0.001, merge_small_threshold=0.0,  # degenerate: likely K=1
        mapper_lr=1e-2, mapper_epochs=20, num_queries=1, mlp_head="linear",
        transfer_k=5, target_cutoff=0.15, class_balanced_loss=True,
        weight_decay_stage2=0.0, max_epochs_stage1=20,
    )
    result = run_trial(config, fixed, log_checkpoint=None)
    # Degenerate partitions should gate-fail, not raise.
    assert result.objective == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/buddy_percept_sweep/pipeline.py
"""Wandb-agnostic orchestration of one full Stage1+Stage2 trial (spec §3).
`run_trial` is the single function the wandb entrypoint (Task 8) calls.
"""
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score

from scripts.buddy_percept_sweep.cache import FixedInputs
from scripts.buddy_percept_sweep.clustering import leiden_partition, merge_small_communities
from scripts.buddy_percept_sweep.objective import compute_objective
from scripts.buddy_percept_sweep.stage1 import ParameterizedLearnedStudent, train_stage1
from scripts.buddy_percept_sweep.stage2 import (
    ParameterizedAttentionPoolingMapper, auc_summary, evaluate_auc, train_stage2,
)
from scripts.buddy_percept_sweep.targets import (
    assign_to_train_communities, build_targets, cosine_vote_fractions, one_hot,
)


@dataclass
class TrialConfig:
    heads: str
    num_heads: int
    d_shared: int
    lr: float
    noise_std: float
    lambda_affect: float
    batch_size: int
    weight_decay: float
    teacher_graph_K: int
    teacher_graph_alpha: float
    leiden_resolution: float
    merge_small_threshold: float
    mapper_lr: float
    mapper_epochs: int
    num_queries: int
    mlp_head: str
    transfer_k: int
    target_cutoff: Union[str, float]
    class_balanced_loss: bool
    weight_decay_stage2: float
    max_epochs_stage1: int = 200
    seed: int = 42


@dataclass
class TrialResult:
    emotion_ami: float
    genre_ami: float
    stage2_macro_auc: float
    objective: float
    n_topics_after_merge: int
    stage1_seconds: float
    stage2_seconds: float


def run_trial(config: TrialConfig, fixed_inputs: FixedInputs,
              log_checkpoint: Optional[Callable[[int, float], None]] = None) -> TrialResult:
    stage1_start = time.monotonic()
    student = ParameterizedLearnedStudent(
        heads=config.heads, num_heads=config.num_heads, d_shared=config.d_shared,
        content_dim=fixed_inputs.train_content.shape[1], affect_dim=fixed_inputs.train_affect.shape[1],
    )
    train_embedding = train_stage1(
        student, fixed_inputs, lr=config.lr, noise_std=config.noise_std,
        lambda_affect=config.lambda_affect, batch_size=config.batch_size,
        weight_decay=config.weight_decay, seed=config.seed, max_epochs=config.max_epochs_stage1,
        log_checkpoint=log_checkpoint, teacher_graph_K=config.teacher_graph_K,
        teacher_graph_alpha=config.teacher_graph_alpha,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.eval()
    with torch.no_grad():
        heldout_embedding, _ = student(
            torch.as_tensor(fixed_inputs.heldout_content, dtype=torch.float32, device=device),
            torch.as_tensor(fixed_inputs.heldout_affect, dtype=torch.float32, device=device),
        )
    heldout_embedding = heldout_embedding.cpu().numpy()

    k_neighbors = max(2, min(20, len(train_embedding) - 1))
    raw_labels = leiden_partition(train_embedding, resolution=config.leiden_resolution,
                                   seed=config.seed, k_neighbors=k_neighbors)
    merged_labels, _ = merge_small_communities(train_embedding, raw_labels, config.merge_small_threshold)
    n_topics = int(merged_labels.max()) + 1

    transfer_k = min(config.transfer_k, len(train_embedding))
    heldout_hard = assign_to_train_communities(train_embedding, merged_labels, heldout_embedding, k=transfer_k)

    emotion_ami = float(adjusted_mutual_info_score(fixed_inputs.heldout_emotion, heldout_hard))
    genre_mask = fixed_inputs.heldout_genre != ""
    if genre_mask.any() and len(set(heldout_hard[genre_mask].tolist())) > 1:
        genre_ami = float(adjusted_mutual_info_score(fixed_inputs.heldout_genre[genre_mask], heldout_hard[genre_mask]))
    else:
        genre_ami = 0.0
    stage1_seconds = time.monotonic() - stage1_start

    stage2_start = time.monotonic()
    if n_topics < 2:
        # Degenerate partition: no valid multi-class target to train a
        # classifier on. Gate-fail via the objective rather than crash.
        return TrialResult(
            emotion_ami=emotion_ami, genre_ami=genre_ami, stage2_macro_auc=0.5,
            objective=compute_objective(emotion_ami, genre_ami, 0.5),
            n_topics_after_merge=n_topics, stage1_seconds=stage1_seconds,
            stage2_seconds=time.monotonic() - stage2_start,
        )

    if config.target_cutoff == "single_label":
        train_targets_np = one_hot(merged_labels, n_topics)
        baseline_heldout_targets = one_hot(heldout_hard, n_topics)
    else:
        train_fractions = cosine_vote_fractions(train_embedding, merged_labels, train_embedding,
                                                 n_topics, k=transfer_k)
        train_targets_np = build_targets(train_fractions, n_topics, config.target_cutoff)
        # Evaluation ALWAYS uses single-label held-out targets (Global
        # Constraints) regardless of the training target convention.
        baseline_heldout_targets = one_hot(heldout_hard, n_topics)

    mapper = ParameterizedAttentionPoolingMapper(
        n_topics=n_topics, num_queries=config.num_queries, mlp_head=config.mlp_head,
        d_model=fixed_inputs.train_patches.shape[-1],
    ).to(device)
    train_targets = torch.as_tensor(train_targets_np, dtype=torch.float32)
    train_stage2(
        mapper, fixed_inputs.train_patches, train_targets, mapper_lr=config.mapper_lr,
        mapper_epochs=config.mapper_epochs, weight_decay=config.weight_decay_stage2,
        class_balanced=config.class_balanced_loss, train_labels_for_weighting=merged_labels,
        seed=config.seed,
    )
    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(fixed_inputs.heldout_patches.to(device))).cpu().numpy()
    aucs, skipped = evaluate_auc(heldout_scores, baseline_heldout_targets)
    stage2_macro_auc = auc_summary(aucs)["macro"] if aucs else 0.5
    stage2_seconds = time.monotonic() - stage2_start

    objective = compute_objective(emotion_ami, genre_ami, stage2_macro_auc)
    return TrialResult(
        emotion_ami=emotion_ami, genre_ami=genre_ami, stage2_macro_auc=stage2_macro_auc,
        objective=objective, n_topics_after_merge=n_topics,
        stage1_seconds=stage1_seconds, stage2_seconds=stage2_seconds,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_pipeline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/buddy_percept_sweep/pipeline.py scripts/buddy_percept_sweep/tests/test_pipeline.py
git commit -m "feat(sweep): add end-to-end trial orchestration"
```

---

### Task 8: Config resolution, real raw-data loader, wandb entrypoint, sweep YAML

**Files:**
- Create: `scripts/buddy_percept_sweep/config.py`
- Create: `scripts/buddy_percept_sweep/real_data.py`
- Create: `scripts/run_buddy_percept_sweep_agent.py`
- Create: `scripts/sweep_config_buddy_percept.yaml`
- Test: `scripts/buddy_percept_sweep/tests/test_config.py`

**Interfaces:**
- Consumes: `TrialConfig` (Task 7), `wandb.config` (a dict-like object with the 22 keys from spec §5).
- Produces: `resolve_trial_config(raw: dict) -> TrialConfig` (validates and
  type-coerces every key, applying `target_cutoff` string/float
  disambiguation since W&B sweep values arrive as plain YAML-typed values).

- [ ] **Step 1: Write the failing test**

```python
# scripts/buddy_percept_sweep/tests/test_config.py
from scripts.buddy_percept_sweep.config import resolve_trial_config


def _full_raw_config(**overrides):
    base = dict(
        heads="attn1", num_heads=1, d_shared=32, lr=1e-3, noise_std=0.0,
        lambda_affect=1.0, batch_size=1024, weight_decay=0.0,
        teacher_graph_K=20, teacher_graph_alpha=0.5,
        leiden_resolution=1.0, merge_small_threshold=0.0,
        mapper_lr=1e-2, mapper_epochs=400, num_queries=1, mlp_head="linear",
        transfer_k=20, target_cutoff="single_label", class_balanced_loss=False,
        weight_decay_stage2=0.0,
    )
    base.update(overrides)
    return base


def test_resolve_trial_config_round_trips_all_core_fields():
    config = resolve_trial_config(_full_raw_config())
    assert config.heads == "attn1"
    assert config.mapper_epochs == 400
    assert config.target_cutoff == "single_label"


def test_resolve_trial_config_accepts_numeric_target_cutoff():
    config = resolve_trial_config(_full_raw_config(target_cutoff=0.15))
    assert config.target_cutoff == 0.15


def test_resolve_trial_config_coerces_bool_like_class_balanced():
    config = resolve_trial_config(_full_raw_config(class_balanced_loss=True))
    assert config.class_balanced_loss is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/buddy_percept_sweep/config.py
"""Translates a raw wandb.config dict (or plain dict, for tests) into a
validated TrialConfig (spec §5's 22 parameters)."""
from scripts.buddy_percept_sweep.pipeline import TrialConfig


def resolve_trial_config(raw: dict) -> TrialConfig:
    target_cutoff = raw["target_cutoff"]
    if target_cutoff != "single_label":
        target_cutoff = float(target_cutoff)
    return TrialConfig(
        heads=str(raw["heads"]),
        num_heads=int(raw["num_heads"]),
        d_shared=int(raw["d_shared"]),
        lr=float(raw["lr"]),
        noise_std=float(raw["noise_std"]),
        lambda_affect=float(raw["lambda_affect"]),
        batch_size=int(raw["batch_size"]),
        weight_decay=float(raw["weight_decay"]),
        teacher_graph_K=int(raw["teacher_graph_K"]),
        teacher_graph_alpha=float(raw["teacher_graph_alpha"]),
        leiden_resolution=float(raw["leiden_resolution"]),
        merge_small_threshold=float(raw["merge_small_threshold"]),
        mapper_lr=float(raw["mapper_lr"]),
        mapper_epochs=int(raw["mapper_epochs"]),
        num_queries=int(raw["num_queries"]),
        mlp_head=str(raw["mlp_head"]),
        transfer_k=int(raw["transfer_k"]),
        target_cutoff=target_cutoff,
        class_balanced_loss=bool(raw["class_balanced_loss"]),
        weight_decay_stage2=float(raw["weight_decay_stage2"]),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/buddy_percept_sweep/tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the real raw-data loader** (no test — this wraps
  real ArtELingo/GoEmotions loading identical to tonight's pilots; it is
  exercised by the Task 9 smoke test, not by fast unit tests, since it
  needs the real data files and HF model download)

```python
# scripts/buddy_percept_sweep/real_data.py
"""Real ArtELingo/GoEmotions/patch-feature loader -- the `raw_loader`
passed into `FixedInputCache.get` outside of tests. Reuses the exact
loading functions this investigation already validated, via the
established sibling-module-import pattern (never edits the originals).
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

_BUDDY_DIR = Path(__file__).resolve().parents[2] / "src/test/20260923_artelingo_buddy_analysis"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_real_raw_inputs():
    from scripts.buddy_percept_sweep.cache import RawInputs

    arch = _load_module("arch_for_sweep", _BUDDY_DIR / "run_learned_student_arch_sweep_pilot.py")
    pipeline = arch.load_sibling_module("pipeline_for_sweep", arch.PIPELINE_PATH)
    affect_pilot = arch.load_sibling_module("affect_for_sweep", arch.AFFECT_PILOT_PATH)
    single_modality = arch.load_sibling_module("single_modality_for_sweep", arch.SINGLE_MODALITY_PATH)
    cca_audit = arch.load_sibling_module("cca_for_sweep", arch.CCA_AUDIT_PATH)
    heldout_pipeline = arch.load_sibling_module("heldout_pipeline_for_sweep", arch.PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = arch.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = arch.HELDOUT_JSON

    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    heldout_paintings, heldout_img, heldout_txt, heldout_emotion_counts = heldout_pipeline.load_dedup_features()
    heldout_emotion = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]

    train_affect = np.asarray(affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, device), dtype=np.float32)
    heldout_affect = np.asarray(affect_pilot.extract_affect_nodes(arch.HELDOUT_JSON, heldout_paintings, device), dtype=np.float32)

    train_content_raw = cca_audit.content_features(img_nodes, txt_nodes, affect_pilot)
    heldout_content_raw = cca_audit.content_features(heldout_img, heldout_txt, affect_pilot)

    genre_map = pipeline.load_genre_map()
    train_genre = np.array([genre_map.get(p, "") for p in paintings], dtype=object)
    heldout_genre = np.array([genre_map.get(p, "") for p in heldout_paintings], dtype=object)

    from run_buddy_stage2_pilot import load_patch_features as _unused  # noqa: F401 (import-path sanity)
    stage2 = arch.load_sibling_module("stage2_for_sweep_patches", _BUDDY_DIR / "run_buddy_stage2_pilot.py")
    train_patches = stage2.load_patch_features(stage2.TRAIN_PATCH_FEATURE_PATH, len(paintings), "train")
    heldout_patches = stage2.load_patch_features(stage2.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out")

    return RawInputs(
        train_content_raw=train_content_raw.astype(np.float32),
        heldout_content_raw=heldout_content_raw.astype(np.float32),
        train_affect=train_affect, heldout_affect=heldout_affect,
        train_emotion=train_emotion, heldout_emotion=heldout_emotion,
        train_genre=train_genre, heldout_genre=heldout_genre,
        train_patches=train_patches, heldout_patches=heldout_patches,
    )
```

- [ ] **Step 6: Write the wandb entrypoint**

```python
# scripts/run_buddy_percept_sweep_agent.py
"""W&B agent entrypoint for the buddy-percept comprehensive sweep.
Usage (per spec §8): after `wandb sweep scripts/sweep_config_buddy_percept.yaml`,
run `wandb agent <sweep_id>` with this as the sweep's `program:` target.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb

from scripts.buddy_percept_sweep.cache import FixedInputCache
from scripts.buddy_percept_sweep.config import resolve_trial_config
from scripts.buddy_percept_sweep.pipeline import run_trial
from scripts.buddy_percept_sweep.real_data import load_real_raw_inputs

_CACHE = FixedInputCache()  # lives for the whole agent process (spec §3.1)


def main() -> None:
    run = wandb.init()
    config = resolve_trial_config(dict(wandb.config))
    fixed_inputs = _CACHE.get(content_pca_dim=getattr(wandb.config, "content_pca_dim", 50),
                              raw_loader=load_real_raw_inputs)

    def log_checkpoint(epoch: int, proxy: float) -> None:
        wandb.log({"checkpoint_proxy": proxy, "epoch": epoch})

    result = run_trial(config, fixed_inputs, log_checkpoint=log_checkpoint)
    wandb.log({
        "objective": result.objective,
        "stage1_emotion_ami": result.emotion_ami,
        "stage1_genre_ami": result.genre_ami,
        "stage2_macro_auc": result.stage2_macro_auc,
        "n_topics_after_merge": result.n_topics_after_merge,
        "stage1_seconds": result.stage1_seconds,
        "stage2_seconds": result.stage2_seconds,
    })
    run.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Write the sweep config**

```yaml
# scripts/sweep_config_buddy_percept.yaml
program: scripts/run_buddy_percept_sweep_agent.py
method: bayes
metric:
  name: objective
  goal: maximize
early_terminate:
  type: hyperband
  min_iter: 3
  eta: 2
project: CoSiR-buddy-percept-sweep

# Core (spec §5.1-5.2) -- always meaningful.
parameters:
  heads:
    values: ["mlp128", "attn1", "attn4"]
  lr:
    distribution: log_uniform_values
    min: 0.0003
    max: 0.003
  noise_std:
    values: [0.0, 0.02, 0.05, 0.1]
  leiden_resolution:
    distribution: log_uniform_values
    min: 0.05
    max: 2.0
  mapper_lr:
    distribution: log_uniform_values
    min: 0.0003
    max: 0.03
  mapper_epochs:
    values: [100, 200, 400, 800, 1600]
  merge_small_threshold:
    values: [0.0, 0.01, 0.02]
  class_balanced_loss:
    values: [true, false]
  target_cutoff:
    values: ["single_label", 0.5, 0.3, 0.15]

  # Extra (spec §5.3-5.4) -- included now per explicit approval of the
  # full 22-dim sweep. To fall back to core-only, regenerate this file
  # with just the block above (do not add a runtime toggle -- see plan
  # Task 8 note).
  num_heads:
    values: [1, 2, 4, 8, 16]
  d_shared:
    values: [16, 32, 64, 128]
  content_pca_dim:
    values: [30, 50, 80, 120]
  lambda_affect:
    distribution: log_uniform_values
    min: 0.5
    max: 2.0
  teacher_graph_K:
    values: [10, 15, 20, 30]
  teacher_graph_alpha:
    values: [0.3, 0.5, 0.7]
  batch_size:
    values: [512, 1024, 2048, 4096]
  weight_decay:
    values: [0.0, 0.00001, 0.0001]
  num_queries:
    values: [1, 2, 4, 8]
  mlp_head:
    values: ["linear", "one_hidden"]
  transfer_k:
    values: [10, 20, 30, 40]
  weight_decay_stage2:
    values: [0.0, 0.00001, 0.0001]
```

**Note on the spec's "env-var toggle" idea:** implementing a runtime
`SWEEP_EXTRAS` flag would still leave W&B's Bayesian model sampling all 22
dimensions even when 13 are ignored by the trial script — wasteful and
confusing to the optimizer. Since the approved sweep runs all 22 dims now,
this plan generates one complete YAML instead; a genuine core-only sweep
later is a second, smaller YAML (trivial to generate by deleting the
"Extra" block above), not a runtime toggle. This is a deliberate,
documented deviation from the spec's exact toggle mechanism in favor of a
simpler, correct one — flag if this should be reconciled back into the spec.

- [ ] **Step 8: Commit**

```bash
git add scripts/buddy_percept_sweep/config.py scripts/buddy_percept_sweep/real_data.py \
        scripts/buddy_percept_sweep/tests/test_config.py \
        scripts/run_buddy_percept_sweep_agent.py scripts/sweep_config_buddy_percept.yaml
git commit -m "feat(sweep): add config resolution, real-data loader, wandb entrypoint, sweep YAML"
```

---

### Task 9: Local smoke test (real data, real GPU, no cluster)

**Files:**
- Create: `src/test/20260928_buddy_percept_sweep/run_local_smoke_test.py`

**Interfaces:**
- Consumes: `real_data.load_real_raw_inputs`, `FixedInputCache`, `resolve_trial_config`, `run_trial` (all prior tasks).
- Produces: a printed report confirming the real pipeline runs end-to-end; no new library code.

- [ ] **Step 1: Write the smoke-test script**

```python
# src/test/20260928_buddy_percept_sweep/run_local_smoke_test.py
"""Manual verification (not a pytest): runs 3 hardcoded configs through the
real ArtELingo data on the local GPU, confirming the full sweep pipeline
(Tasks 1-8) produces sane, non-degenerate metrics before creating the real
W&B sweep and dispatching to DAS6 (spec §8 step 1).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.buddy_percept_sweep.cache import FixedInputCache
from scripts.buddy_percept_sweep.config import resolve_trial_config
from scripts.buddy_percept_sweep.pipeline import run_trial
from scripts.buddy_percept_sweep.real_data import load_real_raw_inputs

BASE_CONFIG = dict(
    heads="attn1", num_heads=1, d_shared=32, lr=1e-3, noise_std=0.0,
    lambda_affect=1.0, batch_size=1024, weight_decay=0.0,
    teacher_graph_K=20, teacher_graph_alpha=0.5,
    leiden_resolution=1.0, merge_small_threshold=0.01,
    mapper_lr=1e-2, mapper_epochs=400, num_queries=1, mlp_head="linear",
    transfer_k=20, target_cutoff="single_label", class_balanced_loss=True,
    weight_decay_stage2=0.0,
)


def main() -> None:
    cache = FixedInputCache()
    variants = [
        {},  # baseline core config
        {"heads": "mlp128", "num_heads": 1},
        {"target_cutoff": 0.15, "num_queries": 4, "mlp_head": "one_hidden"},
    ]
    for i, overrides in enumerate(variants):
        raw = {**BASE_CONFIG, **overrides}
        config = resolve_trial_config(raw)
        fixed_inputs = cache.get(content_pca_dim=50, raw_loader=load_real_raw_inputs)
        start = time.monotonic()
        result = run_trial(config, fixed_inputs)
        elapsed = time.monotonic() - start
        print(
            f"[variant {i}] emotion_ami={result.emotion_ami:.4f} genre_ami={result.genre_ami:.4f} "
            f"stage2_auc={result.stage2_macro_auc:.4f} objective={result.objective:.4f} "
            f"n_topics={result.n_topics_after_merge} elapsed={elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and manually verify**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260928_buddy_percept_sweep/run_local_smoke_test.py`

Expected: 3 lines printed, each with finite, non-NaN metrics; the first
variant's `emotion_ami`/`genre_ami` roughly in the range tonight's own
pilots produced (~0.10-0.20 / ~0.15-0.30) — a wildly different number
(e.g. exactly 0 or exactly 1) signals a wiring bug, not real variance.
`elapsed` for variant 0 should be noticeably longer than variants 1-2 (it
pays the one-time cache warmup); variants 1-2 should each take roughly the
~35-65s per-trial range measured in tonight's investigation.

- [ ] **Step 3: Fix any wiring bugs found, re-run until all 3 variants pass**

(No fixed code to write here — this step exists to catch real issues the
synthetic-data unit tests structurally cannot, per Task right-sizing: real
data has real class imbalance, real NaN-prone edge cases, real file
paths.)

- [ ] **Step 4: Commit**

```bash
git add src/test/20260928_buddy_percept_sweep/run_local_smoke_test.py
git commit -m "test(sweep): add local smoke test against real ArtELingo data"
```

---

### Task 10: Create the sweep and dispatch 9 agents on DAS6

**Files:** none (operational task, uses `cluster-run`'s existing CLI).

- [ ] **Step 1: Confirm all 3 nodes are ready**

Run: `cluster status --node node4XX` for each of the 3 reserved nodes
(substitute actual node names). Confirm `ok: true`, `shell_idle: true`,
and `alloc_gpus` shows 3 GPUs per node.

- [ ] **Step 2: Confirm data availability on each node**

Per spec §10 risk: check the ArtELingo CLIP-feature/patch-feature paths
this pipeline depends on are reachable from each node (via `cluster-run`'s
`DATA_MAP`, or already present). Do this before launching all 9 agents,
not after.

- [ ] **Step 3: Create the sweep once**

Run (from repo root, after committing Tasks 1-9):
```bash
wandb sweep scripts/sweep_config_buddy_percept.yaml
```
Note the printed `sweep_id`.

- [ ] **Step 4: Dispatch 9 agents, 3 per node**

For each of the 3 nodes, for `i` in `0, 1, 2`:
```bash
cluster launch --gpu-slots i -- wandb agent <sweep_id>
```
Record each returned `tag` (9 total).

- [ ] **Step 5: Monitor**

`cluster watch <tag>` (one per tag, `run_in_background: true`) plus the
W&B sweep dashboard for `objective` progress and parameter importance. Per
the cluster-run skill's own guidance, don't poll in the foreground.

- [ ] **Step 6: 24h check-in**

At the 24h soft deadline, check whether `objective` is still improving
(W&B dashboard). If yes, leave the 9 agents running (they keep pulling
from the same queue — no redesign needed to extend). If converged, proceed
to Task 11.

---

### Task 11: Post-sweep top-10 4-seed stress test

**Files:**
- Create: `src/test/20260928_buddy_percept_sweep/run_top10_stress.py`
- Test: `src/test/20260928_buddy_percept_sweep/test_top10_stress_selection.py`

**Interfaces:**
- Consumes: W&B sweep API (`wandb.Api().sweep(sweep_id).runs`), `TrialConfig`/`run_trial` (Task 7).
- Produces: `select_top_n(runs: list[dict], n: int) -> list[dict]` (pure,
  testable selection logic) plus a script that calls it against the real
  API and re-runs each selected config at 4 seeds.

- [ ] **Step 1: Write the failing test for selection logic**

```python
# src/test/20260928_buddy_percept_sweep/test_top10_stress_selection.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_top10_stress import select_top_n


def test_select_top_n_sorts_by_objective_descending():
    runs = [
        {"id": "a", "objective": 0.5, "config": {}},
        {"id": "b", "objective": 0.9, "config": {}},
        {"id": "c", "objective": -1.0, "config": {}},  # gate-failed, excluded
        {"id": "d", "objective": 0.7, "config": {}},
    ]
    top = select_top_n(runs, n=2)
    assert [r["id"] for r in top] == ["b", "d"]


def test_select_top_n_excludes_gate_failed_runs_even_if_fewer_than_n_remain():
    runs = [{"id": "a", "objective": -1.0, "config": {}}] * 5
    top = select_top_n(runs, n=10)
    assert top == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/test/20260928_buddy_percept_sweep/test_top10_stress_selection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the script (selection logic + real-API driver)**

```python
# src/test/20260928_buddy_percept_sweep/run_top10_stress.py
"""Post-sweep: pull the top-10 runs by `objective` from the completed W&B
sweep, re-run each at the established 4-seed stress convention
(42, 7, 123, 2024), and report the most robust final winner (spec §6).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from scripts.buddy_percept_sweep.cache import FixedInputCache
from scripts.buddy_percept_sweep.config import resolve_trial_config
from scripts.buddy_percept_sweep.pipeline import run_trial
from scripts.buddy_percept_sweep.real_data import load_real_raw_inputs

STRESS_SEEDS = (42, 7, 123, 2024)


def select_top_n(runs: list[dict], n: int) -> list[dict]:
    valid = [r for r in runs if r["objective"] > -1.0]
    valid.sort(key=lambda r: r["objective"], reverse=True)
    return valid[:n]


def fetch_sweep_runs(sweep_id: str) -> list[dict]:
    import wandb
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    return [
        {"id": run.id, "objective": run.summary.get("objective", -1.0), "config": dict(run.config)}
        for run in sweep.runs
    ]


def main(sweep_id: str) -> None:
    runs = fetch_sweep_runs(sweep_id)
    top10 = select_top_n(runs, n=10)
    if not top10:
        print("No gate-passing runs found in this sweep.")
        return
    cache = FixedInputCache()
    print(f"Stress-testing top {len(top10)} runs at seeds {STRESS_SEEDS}...")
    final_results = []
    for rank, run in enumerate(top10, start=1):
        base_config = resolve_trial_config(run["config"])
        seed_objectives = []
        for seed in STRESS_SEEDS:
            base_config.seed = seed
            fixed_inputs = cache.get(content_pca_dim=run["config"].get("content_pca_dim", 50),
                                      raw_loader=load_real_raw_inputs)
            result = run_trial(base_config, fixed_inputs)
            seed_objectives.append(result.objective)
        mean_objective = float(np.mean(seed_objectives))
        final_results.append((rank, run["id"], mean_objective, seed_objectives))
        print(f"rank {rank} (sweep run {run['id']}): mean={mean_objective:.4f}, seeds={seed_objectives}")
    final_results.sort(key=lambda entry: entry[2], reverse=True)
    winner = final_results[0]
    print(f"\nFinal winner: sweep run {winner[1]}, 4-seed mean objective {winner[2]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_top10_stress.py <sweep_id>")
    main(sys.argv[1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/test/20260928_buddy_percept_sweep/test_top10_stress_selection.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/test/20260928_buddy_percept_sweep/run_top10_stress.py \
        src/test/20260928_buddy_percept_sweep/test_top10_stress_selection.py
git commit -m "feat(sweep): add post-sweep top-10 4-seed stress test"
```

- [ ] **Step 6: Run for real once the sweep has enough completed runs**

Run: `python src/test/20260928_buddy_percept_sweep/run_top10_stress.py <sweep_id>`

Fold the winning configuration and its 4-seed stress result into
`docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md` as a
new subsection, matching this investigation's established reporting
convention.

---

## Self-Review

**1. Spec coverage:** §3 (trial architecture) → Task 7. §3.1 (caching) →
Task 2. §4 (reused blocks) → Tasks 3, 5, 6 note their pilot-provenance and
why each is a fresh reimplementation rather than an import of the pilot
script directly (the pilot scripts have CLI/report side effects and
aren't designed to be called as a library inside a long-lived process).
§5 (search space) → Task 8's YAML, all 22 params. §6 (seeding) → Global
Constraints + Task 11. §7 (objective/hyperband) → Tasks 1, 3
(`log_checkpoint`), 7. §8 (execution) → Tasks 9, 10. §9 (file layout) →
matches throughout. §10 (risks) → content_pca_dim cache (Task 2), W&B
volume (accepted, no code change needed), data availability (Task 10 step
2), hyperband min_iter (Task 8 YAML). §11 (deliverables) → every checklist
item has a task.

**2. Placeholder scan:** none found — every step has runnable code or a
concrete command.

**3. Type consistency:** `TrialConfig` fields match between Task 7's
definition and Task 8's `resolve_trial_config` construction call
(field-for-field checked). `FixedInputs`/`RawInputs` fields match between
Task 2's definition and Task 8's `real_data.py` construction. `TrialResult`
fields match between Task 7's definition and Task 8/9/11's consumption.

**4. Review Focus:** all five items map to a task's tests: degenerate
Leiden → Task 4 test 2 + Task 7 test 2; `num_heads` no-op on mlp128 → Task
3 test 1; two-topic edge case → Task 5 test 5; objective boundary → Task 1
tests 5; `content_pca_dim` cache correctness → Task 2 tests 2-3.

---

Plan complete and saved to `docs/superpowers/plans/2026-09-28-buddy-percept-sweep.md`. Please review the plan. Which execution approach would you prefer?

- **Subagent-driven** — a fresh subagent implements each task and a fresh reviewer checks it before the next one starts, then a whole-branch review at the end. Most thorough; costs a fresh context per task and per review.
- **Native** — I implement every task myself in this session, then one fresh reviewer on the most capable model checks the whole branch at the end. Cheapest and fastest; no independent review until the end.

For this plan I recommend **subagent-driven**: Tasks 1-7 form a strict dependency chain where a wrong interface early (e.g. a mismatched field name in `TrialConfig`) silently breaks everything downstream without necessarily failing loudly, and Task 10 dispatches real compute across 9 reserved DAS6 GPUs for up to 24+ hours — a mistake shipped there is expensive to notice late. The per-task review gate is worth the extra cost here. Does the plan capture what you want, and which approach should we use?
