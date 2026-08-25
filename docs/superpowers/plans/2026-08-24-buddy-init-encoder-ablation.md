# Buddy-Init Encoder-Pair Ablation (Experiment 8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer Experiment 8 from `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 — does the *choice* of (vision, text) encoder pair used to build the buddy graph/init matter for downstream retrieval, holding the frozen CLIP training backbone, gated combiner, and all training-time buddy terms fixed (off), on RedCaps-150k? And, as a secondary check, does a pair's C3 cross-VLM survival rate predict its Experiment 8 retrieval Δ?

**Architecture:** `compute_buddy_init(img_feats, txt_feats, ...)` (`src/conditional_buddy/compute_buddies.py`) is already a pure function over raw feature arrays. The only code gap is that `TrainableEmbeddingManager._buddy_init` always sources those arrays from the training run's own (CLIP) `FeatureManager` — there is no way to hand it a different encoder pair's features. This plan adds one small, backward-compatible `feature_override` parameter to `_buddy_init` (Task 1), a loader that fetches an arbitrary (vision, text) pair's cached features row-aligned to the training FeatureManager (Task 2, reusing the already-cached `heldout_feats/` from the cross-VLM survival study), and a new Hydra-visible `train.buddies.encoder_pair` override that wires the two together in `train_cosir.py` (Task 3). From there the ablation reuses the exact bash-loop-over-template-key-axis / Hydra-multirun-over-seed pattern already established by `scripts/run_init_ablation.sh` (Experiment 1), just swapping the swept axis from `initialization_strategy` to `encoder_pair` (Task 4), with a paired analysis script keyed the same way (Task 5).

**Tech Stack:** Python 3.10, Hydra/OmegaConf, PyTorch, wandb, numpy/pandas. Existing CoSiR training entrypoint `main_cosir.py`; existing held-out feature cache (`src/test/20260708_heldout_grid/extract_heldout.py`) and cross-VLM survival results (`docs/reports/assets/buddy_cross_vlm/grid_agreement.json`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 8 (added 2026-08-24).

## Global Constraints

- Always run Python/bash training or analysis commands with `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR` first (per `.claude/CLAUDE.md`).
- Statistical standard (spec §5): ≥3 seeds, paired-within-seed Δ, report mean ± std and mean/SEM (flag `|z| ≥ 2` with `*`), compare against the noise floor (~0.1–0.7 R1), never against zero.
- Fixed operating point (same as Experiment 1's confirmed strong cell): `lr=1e-3`, `lr_label=1e-4`, `embedding_dim=16`, `alpha=0.5`. `initialization_strategy=buddies` fixed throughout — only the graph-source encoder pair varies.
- Every training-time buddy term stays OFF: never pass `+loss.lambda_buddy`, `+loss.lambda_buddy_con`, or `+loss.buddy_refresh*`. Omitting them gives the code's own default of `0.0`/`False`.
- `train.buddies.encoder_pair` is a **template-compatibility key**, exactly like `initialization_strategy` in Experiment 1 (`src/hook/train_cosir.py:244-271`): give every pair its own `results_dir` — never share one `results_dir` across two different pairs, or a stale template silently serves the wrong pair's init.
- Scope: RedCaps-150k only (`dataset=redcaps_150k`), all 16 (vision × text) pairs from the C3 cross-VLM survival grid (`VISION = [clip_img, dinov2, siglip_v, vit_sup]`, `TEXT = [clip_txt, minilm, bge, e5]`), 3 seeds per pair = 48 runs. The CLIP backbone, gated combiner, and training-time buddy terms never change — only the init-source encoder pair does.
- Feature storage: the 150k held-out cache (`src/test/20260708_heldout_grid/heldout_feats/redcaps/<model>.npy`) is reused as-is, unmodified — it's already referenced by the C3 report, so nothing in this plan renames or moves it. This plan does not touch the 300k extension (spec's optional Task 8-equivalent) — that is a separate, explicitly-gated follow-up, not part of this plan's scope.
- Two existing `src/` files are modified in this plan (`src/utils/embedding_manager_nocache.py`, `src/hook/train_cosir.py`). Per CLAUDE.md, log both changes in `.claude/20260824_log.md` (one `# <path>` section per file, added when that file's task completes).
- wandb defaults: `entity=augustoxq`, `project=cosir_image` (`configs/config.yaml:18-19`).

---

### Task 1: `feature_override` on `TrainableEmbeddingManager._buddy_init`

**Files:**
- Modify: `src/utils/embedding_manager_nocache.py:345-401` (`_buddy_init`)
- Test: `src/test/20260824_buddy_init_encoder_ablation/test_feature_override.py`
- Modify: `.claude/20260824_log.md` (create; log this change)

**Interfaces:**
- Consumes: `src.conditional_buddy.compute_buddy_init` (unchanged).
- Produces: `_buddy_init(..., feature_override: Optional[Tuple[np.ndarray, np.ndarray, List[int]]] = None)` — when given, `_buddy_init` builds the graph from `feature_override`'s `(img_feats, txt_feats, input_sample_ids)` instead of loading from `feature_manager`. Consumed by Task 3's `train_cosir.py` wiring (via `_buddy_kwargs["feature_override"]`, which already flows through `initialize()` → `initialize_embeddings_buddies()`'s existing `**buddy_kwargs` passthrough — neither of those two functions needs to change).

- [ ] **Step 1: Write the failing tests**

Create `src/test/20260824_buddy_init_encoder_ablation/test_feature_override.py`:

```python
"""
Test: TrainableEmbeddingManager's buddies init accepts a feature_override triple,
bypassing FeatureManager entirely (Experiment 8 — buddy-init encoder-pair ablation,
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md).

Run:
    python src/test/20260824_buddy_init_encoder_ablation/test_feature_override.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch

from src.conditional_buddy.compute_buddies import compute_buddy_init
from src.utils.embedding_manager_nocache import TrainableEmbeddingManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()
TMP_DIR = "/tmp/test_feature_override_embeddings"


def _synthetic_pair(n=40, dim=32, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (n, dim)).astype(np.float32)
    txt = rng.normal(0, 1, (n, dim)).astype(np.float32)
    # Deliberately NOT range(n) - CLAUDE.md's sample-ID-consistency trap.
    sample_ids = list(range(100, 100 + n))
    return img, txt, sample_ids


def test_feature_override_bypasses_feature_manager():
    img, txt, sample_ids = _synthetic_pair()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    mgr = TrainableEmbeddingManager(
        sample_ids=sample_ids, embedding_dim=8, embeddings_dir=TMP_DIR,
        mode="ram", initialization_strategy="zeros", device=DEVICE,
    )
    # feature_manager=None: if the override branch didn't bypass it, this would raise
    # AttributeError on feature_manager.get_num_chunks().
    mgr.initialize(
        "buddies", feature_manager=None, model=None, device=DEVICE,
        k=10, use_half=USE_HALF if False else None,  # placeholder removed below
        feature_override=(img, txt, sample_ids),
    ) if False else mgr.initialize(
        "buddies", feature_manager=None, model=None, device=DEVICE,
        k=10, feature_override=(img, txt, sample_ids),
    )
    emb = mgr.get_embeddings(sample_ids)
    assert emb.shape == (len(sample_ids), 8), f"unexpected shape {emb.shape}"
    assert not torch.allclose(emb, torch.zeros_like(emb)), "embeddings were not initialized"
    print("PASS test_feature_override_bypasses_feature_manager")


def test_feature_override_matches_direct_compute_buddy_init():
    """The override path must be numerically identical to calling compute_buddy_init
    directly with the same inputs — Task 1 must not add any reordering/reprocessing
    beyond what compute_buddy_init itself does."""
    img, txt, sample_ids = _synthetic_pair(seed=2)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    mgr = TrainableEmbeddingManager(
        sample_ids=sample_ids, embedding_dim=8, embeddings_dir=TMP_DIR,
        mode="ram", initialization_strategy="zeros", device=DEVICE,
    )
    mgr.initialize(
        "buddies", feature_manager=None, model=None, device=DEVICE,
        k=10, seed=42, feature_override=(img, txt, sample_ids),
    )
    got = mgr.get_embeddings(sample_ids).cpu().numpy()

    want = compute_buddy_init(
        img, txt, n_dim=8, K=10, device=DEVICE, seed=42, use_half=USE_HALF,
        input_sample_ids=sample_ids, output_sample_ids=sample_ids,
    )
    np.testing.assert_allclose(got, want, atol=1e-4)
    print("PASS test_feature_override_matches_direct_compute_buddy_init")


if __name__ == "__main__":
    test_feature_override_bypasses_feature_manager()
    test_feature_override_matches_direct_compute_buddy_init()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("ALL TESTS PASSED")
```

(The `if False else` line in the first test is a mistake to fix in Step 1 — write the clean single call shown below instead. This note only exists so Step 2 fails for the *right* reason.) Actually write it clean the first time:

```python
    mgr.initialize(
        "buddies", feature_manager=None, model=None, device=DEVICE,
        k=10, feature_override=(img, txt, sample_ids),
    )
```

as the entire body of the `mgr.initialize(...)` call in `test_feature_override_bypasses_feature_manager` — no `if False`, no placeholder branch. (The paragraph above exists only to flag that this file must not literally contain dead branches; write it correctly the first time.)

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260824_buddy_init_encoder_ablation/test_feature_override.py
```
Expected: `TypeError: _buddy_init() got an unexpected keyword argument 'feature_override'` (raised from inside `initialize()`'s dispatch to `_buddy_init`).

- [ ] **Step 3: Implement `feature_override` in `_buddy_init`**

In `src/utils/embedding_manager_nocache.py`, replace the `_buddy_init` method (currently lines 345-401) with:

```python
    def _buddy_init(
        self,
        feature_manager,
        device: str,
        k: int = 30,
        alpha: float = 0.5,
        method: str = "spectral",
        knn_batch_size: int = 1024,
        normalize_method: str = "rank",
        seed: int = 42,
        b_weight: float = 1.0,
        feature_override: Optional[Tuple[np.ndarray, np.ndarray, List[int]]] = None,
    ) -> np.ndarray:
        """
        Conditional-buddies init: build the cross-modal mutual-KNN graph, embed it,
        and return a normalised [N, D] array reordered to self.sample_ids order.

        b_weight: B-lean affinity multiplier for strict-intersection buddies
                  (1.0 = off / union-only; >1 pulls strict buddies tighter).
        feature_override: optional (img_feats, txt_feats, input_sample_ids) triple to
                  build the graph from instead of feature_manager's own (CLIP) features
                  — e.g. a non-CLIP (vision, text) encoder pair for the buddy-init
                  encoder-pair ablation (Experiment 8, docs/superpowers/specs/
                  2026-08-04-buddy-publication-plan-design.md). img_feats/txt_feats must
                  be row-aligned to input_sample_ids. Only the GRAPH SOURCE changes —
                  the frozen training backbone is untouched either way. When given,
                  feature_manager is not accessed at all (may be None).
        """
        from src.conditional_buddy import compute_buddy_init

        print(f"[EmbeddingManager] Buddies init (method={method}, K={k}, alpha={alpha})…")

        if feature_override is not None:
            img, txt, fm_sample_ids = feature_override
            print(f"[EmbeddingManager] Using feature_override — {img.shape[0]} rows "
                  f"(bypassing feature_manager for graph source)")
        else:
            img_parts: List[np.ndarray] = []
            txt_parts: List[np.ndarray] = []
            num_shards = feature_manager.get_num_chunks()
            for shard_id in tqdm(range(num_shards), desc="Loading features"):
                feats = feature_manager.get_features_by_chunk(shard_id)
                img_parts.append(feats["img_features"].cpu().numpy().astype(np.float32))
                txt_parts.append(feats["txt_features"].cpu().numpy().astype(np.float32))
            img = np.concatenate(img_parts, axis=0)
            txt = np.concatenate(txt_parts, axis=0)
            del img_parts, txt_parts  # superseded by img/txt; dead weight at N~3M scale
            fm_sample_ids = feature_manager.get_all_sample_ids()

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
            b_weight=b_weight,
            input_sample_ids=fm_sample_ids,
            output_sample_ids=self.sample_ids,
            return_edges=True,
        )
        np.save(self.embeddings_dir / "buddy_edges.npy", edges.astype(np.int64))
        print(
            f"[EmbeddingManager] Buddies init done. "
            f"Mean norm: {np.linalg.norm(emb, axis=1).mean():.4f}"
        )
        return emb
```

Confirm `Tuple` is already imported at the top of the file (`from typing import Any, Dict, List, Optional, Tuple` — it is, per the file's existing imports); no import changes needed.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python src/test/20260824_buddy_init_encoder_ablation/test_feature_override.py
```
Expected: `ALL TESTS PASSED`.

- [ ] **Step 5: Log the change**

Create `.claude/20260824_log.md`:

```markdown
# /src/utils/embedding_manager_nocache.py

## `_buddy_init`: added `feature_override` parameter

**Before:** always loaded img/txt features from `feature_manager` (CLIP-backed).

**After:** accepts an optional `feature_override=(img_feats, txt_feats, input_sample_ids)`
triple; when given, builds the buddy graph from it instead, bypassing `feature_manager`
entirely. `feature_manager=None` is valid in that case. Backward-compatible — omitting the
argument reproduces the exact original code path.

**Why:** Experiment 8 (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
§4) needs to build the buddy graph from encoder pairs other than CLIP while keeping the
frozen training backbone on CLIP. See `docs/superpowers/plans/
2026-08-24-buddy-init-encoder-ablation.md` Task 1.
```

- [ ] **Step 6: Commit**

```bash
git add src/utils/embedding_manager_nocache.py src/test/20260824_buddy_init_encoder_ablation/test_feature_override.py .claude/20260824_log.md
git commit -m "feat: add feature_override to buddy-init for non-CLIP encoder pairs (Experiment 8)"
```

---

### Task 2: Encoder-pair feature loader — `src/conditional_buddy/heldout_encoder_features.py`

**Files:**
- Create: `src/conditional_buddy/heldout_encoder_features.py`
- Test: `src/test/20260824_buddy_init_encoder_ablation/test_heldout_encoder_features.py`

**Interfaces:**
- Consumes: `src/test/20260708_heldout_grid/extract_heldout.py`'s `cache_path(dataset, model, smoke)`; `src/test/20260623_redcaps_buddy/redcaps_buddy.py`'s `load_data()` (returns `Data.img`, `.txt`, `.sample_ids`, both L2-normalized, in the RedCaps-150k `FeatureManager`'s own row order); a live `feature_manager` instance (from `train_cosir.py`, RedCaps-150k config).
- Produces: `load_encoder_pair_features(dataset: str, vision: str, text: str, feature_manager) -> Tuple[np.ndarray, np.ndarray, List[int]]` — consumed by Task 3's `train_cosir.py` wiring as the `feature_override` triple for Task 1's new parameter.

This module requires the real RedCaps-150k `FeatureManager` on disk to test meaningfully (no synthetic substitute — the whole point is verifying alignment against the *actual* store), matching this codebase's existing convention for `redcaps_buddy.py`/`cross_vlm_buddy.py`'s own tests.

- [ ] **Step 1: Write the failing test**

Create `src/test/20260824_buddy_init_encoder_ablation/test_heldout_encoder_features.py`:

```python
"""
Test: load_encoder_pair_features loads clip_img x clip_txt identically to RedCaps'
own load_data(), and raises when sample-id order doesn't match feature_manager.

Requires the RedCaps-150k FeatureManager on disk (STORAGE in redcaps_buddy.py) and the
held-out feature cache from src/test/20260708_heldout_grid/extract_heldout.py (at least
one non-CLIP model, e.g. dinov2, already extracted — the survival study already did this).

Run:
    python src/test/20260824_buddy_init_encoder_ablation/test_heldout_encoder_features.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "20260623_redcaps_buddy")))

import numpy as np

from src.conditional_buddy.heldout_encoder_features import load_encoder_pair_features
from src.utils import FeatureManager
import redcaps_buddy as rb


def test_clip_pair_matches_load_data():
    fm = FeatureManager(rb.STORAGE)
    img, txt, sample_ids = load_encoder_pair_features("redcaps", "clip_img", "clip_txt", fm)
    data = rb.load_data()
    assert sample_ids == data.sample_ids
    np.testing.assert_allclose(img, data.img, atol=1e-5)
    np.testing.assert_allclose(txt, data.txt, atol=1e-5)
    print(f"PASS test_clip_pair_matches_load_data ({len(sample_ids)} rows)")


def test_nonclip_pair_shape_and_alignment():
    fm = FeatureManager(rb.STORAGE)
    img, txt, sample_ids = load_encoder_pair_features("redcaps", "dinov2", "clip_txt", fm)
    data = rb.load_data()
    assert sample_ids == data.sample_ids
    assert img.shape[0] == len(sample_ids)
    np.testing.assert_allclose(txt, data.txt, atol=1e-5)  # clip_txt side unchanged
    print(f"PASS test_nonclip_pair_shape_and_alignment (img dim={img.shape[1]})")


def test_mismatched_sample_ids_raises():
    class _FakeFM:
        def get_all_sample_ids(self):
            return [0, 1, 2]  # deliberately wrong length/order
    try:
        load_encoder_pair_features("redcaps", "clip_img", "clip_txt", _FakeFM())
        raise AssertionError("expected AssertionError on sample-id mismatch")
    except AssertionError as e:
        assert "sample" in str(e).lower()
        print("PASS test_mismatched_sample_ids_raises")


if __name__ == "__main__":
    test_clip_pair_matches_load_data()
    test_nonclip_pair_shape_and_alignment()
    test_mismatched_sample_ids_raises()
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260824_buddy_init_encoder_ablation/test_heldout_encoder_features.py
```
Expected: `ModuleNotFoundError: No module named 'src.conditional_buddy.heldout_encoder_features'`.

- [ ] **Step 3: Implement the loader**

Create `src/conditional_buddy/heldout_encoder_features.py`:

```python
"""
Load buddy-graph source features for an arbitrary (vision, text) encoder pair, row-aligned
to a given FeatureManager's sample-id order.

Used by Experiment 8 (buddy-init encoder-pair ablation,
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md) to swap which encoder
pair BUILDS the buddy graph/init while the frozen training backbone stays CLIP throughout.

'clip_img'/'clip_txt' come straight from the dataset's own load_data() (which itself reads
the CLIP FeatureManager). Every other name is read from the held-out feature cache built by
src/test/20260708_heldout_grid/extract_heldout.py, which is guaranteed row-aligned to the
SAME FeatureManager because both were built by calling that dataset's shared load_data().
"""
import os
import sys
from typing import List, Tuple

import numpy as np

_HELDOUT_GRID_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "test", "20260708_heldout_grid"
)
if _HELDOUT_GRID_DIR not in sys.path:
    sys.path.insert(0, _HELDOUT_GRID_DIR)

VISION_ENCODERS = ["clip_img", "dinov2", "siglip_v", "vit_sup"]
TEXT_ENCODERS = ["clip_txt", "minilm", "bge", "e5"]

# dataset key -> (module name, directory to import it from)
_DATASET_LOADERS = {
    "redcaps": ("redcaps_buddy", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "test", "20260623_redcaps_buddy")),
}


def _dataset_module(dataset: str):
    mod_name, mod_dir = _DATASET_LOADERS[dataset]
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    return __import__(mod_name)


def load_encoder_pair_features(
    dataset: str, vision: str, text: str, feature_manager
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Returns (img_feats, txt_feats, sample_ids) for the (vision, text) encoder pair, in
    feature_manager's own sample-id order.

    Raises AssertionError if the dataset's load_data() row order (which the held-out cache
    was built against) does not exactly match feature_manager.get_all_sample_ids() — the
    CLAUDE.md 'sample ID consistency' check, applied to this second cached feature source.
    """
    from extract_heldout import cache_path  # src/test/20260708_heldout_grid

    mod = _dataset_module(dataset)
    data = mod.load_data()
    fm_ids = list(feature_manager.get_all_sample_ids())
    assert data.sample_ids == fm_ids, (
        f"held-out cache row order does not match feature_manager for dataset={dataset}: "
        f"{len(data.sample_ids)} vs {len(fm_ids)} sample ids, or order differs. Re-run "
        "extract_heldout.py against the SAME FeatureManager storage_dir this training run "
        "uses before selecting this encoder pair."
    )

    def _load(name):
        if name == "clip_img":
            return np.ascontiguousarray(data.img, dtype=np.float32)
        if name == "clip_txt":
            return np.ascontiguousarray(data.txt, dtype=np.float32)
        path = cache_path(dataset, name, 0)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing held-out cache {path}; run:\n"
                f"  python src/test/20260708_heldout_grid/extract_heldout.py "
                f"--dataset {dataset} --model {name}"
            )
        return np.load(path).astype(np.float32)

    return _load(vision), _load(text), data.sample_ids
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python src/test/20260824_buddy_init_encoder_ablation/test_heldout_encoder_features.py
```
Expected: `ALL TESTS PASSED`. (Skip/flag `test_nonclip_pair_shape_and_alignment` if `heldout_feats/redcaps/dinov2.npy` doesn't exist yet in this environment — re-run `extract_heldout.py --dataset redcaps --model dinov2` first per the survival study's existing instructions.)

- [ ] **Step 5: Commit**

```bash
git add src/conditional_buddy/heldout_encoder_features.py src/test/20260824_buddy_init_encoder_ablation/test_heldout_encoder_features.py
git commit -m "feat: add encoder-pair feature loader for buddy-init ablation (Experiment 8)"
```

---

### Task 3: Wire `train.buddies.encoder_pair` into `train_cosir.py`

**Files:**
- Modify: `src/hook/train_cosir.py:250-266` (inside `_init_embedding_manager`)
- Modify: `.claude/20260824_log.md` (append second section)

**Interfaces:**
- Consumes: Task 1's `_buddy_init(..., feature_override=...)`, Task 2's `load_encoder_pair_features`.
- Produces: a new Hydra override `+train.buddies.encoder_pair=<vision>:<text>` (e.g. `dinov2:bge`) that, when set, makes a `buddies`-strategy run build its graph from that pair instead of CLIP. Absent by default — every existing sweep script and config is unaffected. Consumed by Task 4's sweep script.

- [ ] **Step 1: Modify `_init_embedding_manager`**

In `src/hook/train_cosir.py`, the block currently reads (lines 250-266):

```python
    _bud = getattr(cfg.train, "buddies", None)
    _buddy_kwargs = {
        "k": int(getattr(_bud, "k", 30)) if _bud is not None else 30,
        "alpha": float(getattr(_bud, "alpha", 0.5)) if _bud is not None else 0.5,
        "method": str(getattr(_bud, "method", "spectral")) if _bud is not None else "spectral",
        "knn_batch_size": int(getattr(_bud, "knn_batch_size", 1024)) if _bud is not None else 1024,
        "normalize_method": str(getattr(_bud, "normalize_method", "rank")) if _bud is not None else "rank",
        "seed": int(cfg.seed),
        "b_weight": float(getattr(_bud, "b_weight", 1.0)) if _bud is not None else 1.0,
    }
    _extra = None
    if strategy == "buddies":
        _extra = {"k": _buddy_kwargs["k"], "alpha": _buddy_kwargs["alpha"],
                  "method": _buddy_kwargs["method"]}
        # Only add b_weight to the template key when it departs from the default, so
        # existing (pre-b_weight) templates stay compatible for standard runs while a
        # changed lean still forces a rebuild (no silent template reuse across values).
        if _buddy_kwargs["b_weight"] != 1.0:
            _extra["b_weight"] = _buddy_kwargs["b_weight"]
```

Replace it with:

```python
    _bud = getattr(cfg.train, "buddies", None)
    _buddy_kwargs = {
        "k": int(getattr(_bud, "k", 30)) if _bud is not None else 30,
        "alpha": float(getattr(_bud, "alpha", 0.5)) if _bud is not None else 0.5,
        "method": str(getattr(_bud, "method", "spectral")) if _bud is not None else "spectral",
        "knn_batch_size": int(getattr(_bud, "knn_batch_size", 1024)) if _bud is not None else 1024,
        "normalize_method": str(getattr(_bud, "normalize_method", "rank")) if _bud is not None else "rank",
        "seed": int(cfg.seed),
        "b_weight": float(getattr(_bud, "b_weight", 1.0)) if _bud is not None else 1.0,
    }
    # Experiment 8 (buddy-init encoder-pair ablation, docs/superpowers/specs/
    # 2026-08-04-buddy-publication-plan-design.md §4): swap which (vision, text) encoder
    # pair BUILDS the buddy graph/init, while the frozen CLIP training backbone stays
    # untouched. Set via `+train.buddies.encoder_pair=<vision>:<text>` (e.g. "dinov2:bge");
    # absent by default, which preserves the exact original CLIP-FeatureManager code path.
    _encoder_pair = getattr(_bud, "encoder_pair", None) if _bud is not None else None
    if _encoder_pair:
        from src.conditional_buddy.heldout_encoder_features import load_encoder_pair_features
        _vision, _text = str(_encoder_pair).split(":")
        _img_ov, _txt_ov, _ids_ov = load_encoder_pair_features(
            cfg.data.dataset_type, _vision, _text, feature_manager
        )
        _buddy_kwargs["feature_override"] = (_img_ov, _txt_ov, _ids_ov)
    _extra = None
    if strategy == "buddies":
        _extra = {"k": _buddy_kwargs["k"], "alpha": _buddy_kwargs["alpha"],
                  "method": _buddy_kwargs["method"]}
        # Only add b_weight to the template key when it departs from the default, so
        # existing (pre-b_weight) templates stay compatible for standard runs while a
        # changed lean still forces a rebuild (no silent template reuse across values).
        if _buddy_kwargs["b_weight"] != 1.0:
            _extra["b_weight"] = _buddy_kwargs["b_weight"]
        if _encoder_pair:
            _extra["encoder_pair"] = _encoder_pair
```

- [ ] **Step 2: Smoke-test the new override, and its equivalence to the default CLIP path**

This has no pytest infra (matches this file's existing convention — verified by real invocation, per the Experiment 1 plan's Task 4 note). Run two 2-epoch smoke runs and confirm they land within noise of each other, since `encoder_pair=clip_img:clip_txt` must be numerically equivalent to the original no-override path (both ultimately feed CLIP's own img/txt features into the same `compute_buddy_init` call — see Task 1's `feature_override` docstring):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python main_cosir.py \
  dataset=redcaps_150k eval.evaluation_interval=1 eval.oracle_aggregation=max \
  eval.test_ratio=0.2 model=clip_base model.num_layers=6 model.embedding_dim=16 \
  optimizer.lr=1e-3 optimizer.lr_label=1e-4 seed=1 \
  train.initialization_strategy=buddies train.buddies.alpha=0.5 train.epochs=2 \
  experiment.results_dir=res/CoSiR_buddy_encoder_ablation/_smoke/default \
  wandb.group="buddy-encoder-ablation-smoke"

python main_cosir.py \
  dataset=redcaps_150k eval.evaluation_interval=1 eval.oracle_aggregation=max \
  eval.test_ratio=0.2 model=clip_base model.num_layers=6 model.embedding_dim=16 \
  optimizer.lr=1e-3 optimizer.lr_label=1e-4 seed=1 \
  train.initialization_strategy=buddies train.buddies.alpha=0.5 \
  +train.buddies.encoder_pair=clip_img:clip_txt train.epochs=2 \
  experiment.results_dir=res/CoSiR_buddy_encoder_ablation/_smoke/override \
  wandb.group="buddy-encoder-ablation-smoke"
```
Expected: both runs complete 2 epochs with no traceback; the override run's log shows `[EmbeddingManager] Using feature_override — <N> rows` (confirming the new branch actually fired) with `<N>` matching the default run's sample count; the two runs' `test_oracle/t2i_R1` and `test_oracle/i2t_R1` (check via wandb or each run's summary) agree within the measured noise floor (~0.1–0.7 R1) — if they diverge by more than that, stop and re-check Task 1/3 before proceeding to Task 4 (it would mean the override path isn't actually equivalent for the CLIP pair, invalidating the whole ablation's baseline).

- [ ] **Step 3: Log the change**

Append to `.claude/20260824_log.md`:

```markdown
# /src/hook/train_cosir.py

## `_init_embedding_manager`: added `train.buddies.encoder_pair` override

**Before:** the `buddies` strategy always built its graph from the run's own (CLIP)
`feature_manager`; no way to choose a different source encoder pair.

**After:** an optional `+train.buddies.encoder_pair=<vision>:<text>` Hydra override (e.g.
`dinov2:bge`) loads that pair's features via `load_encoder_pair_features` (Task 2) and
passes them through as `feature_override` (Task 1). Absent by default — behavior is
unchanged for every existing config/sweep. Also added to the buddies template-compatibility
`extra` dict so a stale template from a different pair is never silently reused.

**Why:** Experiment 8 (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
§4). See `docs/superpowers/plans/2026-08-24-buddy-init-encoder-ablation.md` Task 3.
```

- [ ] **Step 4: Commit**

```bash
git add src/hook/train_cosir.py .claude/20260824_log.md
git commit -m "feat: wire train.buddies.encoder_pair override into training (Experiment 8)"
```

---

### Task 4: Sweep script — `scripts/run_buddy_init_encoder_ablation.sh`

**Files:**
- Create: `scripts/run_buddy_init_encoder_ablation.sh`

**Interfaces:**
- Consumes: Task 3's `+train.buddies.encoder_pair` override; `main_cosir.py`.
- Produces: per-pair experiment directories under `${BASE_RESULTS_DIR}/pair_<safe_pair>/`, wandb runs tagged `${WANDB_TAG}` in group `buddy-init encoder-pair ablation` — consumed by Task 5's analysis script.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
set -euo pipefail
# Experiment 8 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §4):
# does the CHOICE of (vision, text) encoder pair used to BUILD the buddy graph/init matter
# for downstream retrieval, holding the frozen CLIP training backbone, gated combiner, and
# all training-time buddy terms OFF (same operating point as Experiment 1's 'buddies' arm)?
#
# train.buddies.encoder_pair is a TEMPLATE-COMPATIBILITY key, exactly like
# initialization_strategy in scripts/run_init_ablation.sh: each pair gets its OWN
# results_dir so its own template_embeddings/, avoiding template-reuse races. A bash loop
# over the template-key axis (encoder pair), Hydra multirun over the non-template axis
# (seed) — same pattern as run_init_ablation.sh.
#
#   SMOKE=1 bash scripts/run_buddy_init_encoder_ablation.sh                # 2 epochs, 1 pair
#   ENCODER_PAIR_SWEEP="clip_img:clip_txt dinov2:bge" bash scripts/run_buddy_init_encoder_ablation.sh
#   bash scripts/run_buddy_init_encoder_ablation.sh                        # full 16-pair sweep
#
# Requires the held-out feature cache for every non-CLIP encoder used to already exist:
#   python src/test/20260708_heldout_grid/extract_heldout.py --dataset redcaps --model <name>
# (already run for the C3 cross-VLM survival study — see docs/reports/2026-07-16_buddy_cross_vlm_survival.md)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Template-key axis (bash loop; each pair = its own template + results_dir) ───
if [ -z "${ENCODER_PAIR_SWEEP:-}" ]; then
  PAIRS=()
  for V in clip_img dinov2 siglip_v vit_sup; do
    for T in clip_txt minilm bge e5; do
      PAIRS+=("${V}:${T}")
    done
  done
  ENCODER_PAIR_SWEEP="${PAIRS[*]}"
fi

# ── Non-template axis (Hydra multirun; reuses each pair's template) ─────────────
SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

# ── Fixed operating point (same as Experiment 1's confirmed strong cell) ────────
LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

# ── Dataset + storage (RedCaps-150k only, per spec §4 Experiment 8 scope) ───────
DATASET="${DATASET:-redcaps_150k}"
TEST_RATIO="${TEST_RATIO:-0.2}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_buddy_encoder_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-buddy-encoder-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-buddy-init encoder-pair ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  ENCODER_PAIR_SWEEP="${ENCODER_PAIR_SWEEP_SMOKE:-clip_img:clip_txt}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, pair(s)={$ENCODER_PAIR_SWEEP} — template-build + pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

echo "==================================================================="
echo "Buddy-init encoder-pair ablation ($DATASET): {$ENCODER_PAIR_SWEEP} x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA, initialization_strategy=buddies"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "==================================================================="

for PAIR in $ENCODER_PAIR_SWEEP; do
  SAFE_PAIR="${PAIR/:/_x_}"
  RD="${BASE_RESULTS_DIR}/pair_${SAFE_PAIR}"
  echo ">>> encoder_pair=${PAIR}  ->  results_dir=${RD}"
  python main_cosir.py -m \
    dataset="$DATASET" \
    eval.evaluation_interval="$EVAL_INTERVAL" \
    eval.oracle_aggregation=max \
    eval.test_ratio="$TEST_RATIO" \
    model=clip_base \
    model.num_layers=6 \
    model.embedding_dim="$EMBEDDING_DIM" \
    optimizer.lr="$LR_SWEEP" \
    optimizer.lr_label="$LR_LABEL_SWEEP" \
    seed="$SEED_SWEEP" \
    train.initialization_strategy=buddies \
    train.buddies.alpha="$ALPHA" \
    +train.buddies.encoder_pair="$PAIR" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    +loss.log_buddy_preservation=true \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
    ${EXTRA_OVERRIDES:-}
done

echo "==================================================================="
echo "Done. Analyse (paired vs. clip_img:clip_txt, mean delta +/- std) with:"
echo "  python scripts/analyze_buddy_init_encoder_ablation.py --tag $WANDB_TAG"
echo "==================================================================="
```

- [ ] **Step 2: Smoke-test it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SMOKE=1 bash scripts/run_buddy_init_encoder_ablation.sh
```
Expected: one short run (`pair_clip_img_x_clip_txt`), 2 epochs, no traceback, ending with the `Done. Analyse ...` banner.

```bash
SMOKE=1 ENCODER_PAIR_SWEEP_SMOKE="dinov2:bge" bash scripts/run_buddy_init_encoder_ablation.sh
```
Expected: same, but `results_dir` ends in `pair_dinov2_x_bge`, and the log shows the `feature_override` branch firing with a non-CLIP feature dimension in the `[buddies] Step 1: mutual KNN` line's shapes.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_buddy_init_encoder_ablation.sh
git commit -m "feat: add buddy-init encoder-pair ablation sweep runner (Experiment 8)"
```

---

### Task 5: Analysis script — `scripts/analyze_buddy_init_encoder_ablation.py`

**Files:**
- Create: `scripts/analyze_buddy_init_encoder_ablation.py`

**Interfaces:**
- Consumes: wandb runs from Task 4 (group `buddy-init encoder-pair ablation`); `docs/reports/assets/buddy_cross_vlm/grid_agreement.json` (C3 survival data — keys `cells` (16 `"{vision}x{text}"` names), `E.lift` (16×16 chance-lift matrix for the union graph, the graph type buddy-init actually uses)).
- Produces: printed paired Δ tables (each pair vs. `clip_img:clip_txt`, mean ± std, mean/SEM) and a survival-rate-vs-Δ correlation — read directly by Task 7's report-writing step.

Same testing approach as `scripts/analyze_init_ablation.py` (Experiment 1): one pure, offline-testable core (`compute_paired_deltas`, `summarize`, `survival_rate_per_cell`), TDD'd via `--selftest`; the wandb-fetching half is exercised for real in Task 7 against live data.

- [ ] **Step 1: Write the failing selftest**

Create `scripts/analyze_buddy_init_encoder_ablation.py` with the imports, constants, and `_selftest()` only:

```python
"""
Paired analysis for Experiment 8 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
S4): does the (vision, text) encoder pair used to build the buddy graph/init matter for
downstream retrieval, holding the frozen CLIP backbone and all training-time buddy terms off?

Reads wandb runs from the 'buddy-init encoder-pair ablation' group
(scripts/run_buddy_init_encoder_ablation.sh), pairs every non-baseline encoder pair against
the clip_img:clip_txt baseline WITHIN each seed, and reports mean delta +/- std and mean/SEM
(spec S5). Also joins each pair's C3 cross-VLM survival rate (mean off-diagonal chance-lift
of its union graph E against the other 15 cells, from
docs/reports/assets/buddy_cross_vlm/grid_agreement.json) against its measured retrieval delta.

Usage
-----
  python scripts/analyze_buddy_init_encoder_ablation.py --tag buddy-encoder-ablation-redcaps_150k
  python scripts/analyze_buddy_init_encoder_ablation.py --selftest   # offline check, no wandb call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

T2I = "test_oracle/t2i_R1"
I2T = "test_oracle/i2t_R1"
BASELINE = "clip_img:clip_txt"
CELL = [("encoder_pair", ("train", "buddies", "encoder_pair")),
        ("seed", ("seed",))]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SURVIVAL_JSON = os.path.join(ROOT, "docs/reports/assets/buddy_cross_vlm/grid_agreement.json")


def _selftest():
    """Offline arithmetic check - no wandb call, no survival JSON needed."""
    df = pd.DataFrame([
        {"encoder_pair": "clip_img:clip_txt", "seed": 1, T2I: 50.0},
        {"encoder_pair": "dinov2:bge", "seed": 1, T2I: 52.0},
        {"encoder_pair": "clip_img:clip_txt", "seed": 2, T2I: 48.0},
        {"encoder_pair": "dinov2:bge", "seed": 2, T2I: 51.0},
        {"encoder_pair": "clip_img:clip_txt", "seed": 3, T2I: 49.0},
        {"encoder_pair": "dinov2:bge", "seed": 3, T2I: 49.0},
    ])
    deltas = compute_paired_deltas(df, T2I, "dinov2:bge")
    assert len(deltas) == 3, f"expected 3 paired cells, got {len(deltas)}"
    got = sorted(d for _, d in deltas)
    want = [0.0, 2.0, 3.0]
    assert got == want, f"expected deltas {want}, got {got}"
    s = summarize(deltas)
    assert s["n"] == 3
    assert abs(s["mean"] - (5.0 / 3)) < 1e-9
    assert s["wins"] == 2

    # survival_rate_per_cell: 3-cell toy grid, symmetric lift matrix.
    toy = {"cells": ["a", "b", "c"], "E": {"lift": [[1, 4, 6], [4, 1, 2], [6, 2, 1]]}}
    rates = survival_rate_per_cell(toy)
    assert abs(rates["a"] - 5.0) < 1e-9, rates   # mean(4, 6) off-diag
    assert abs(rates["b"] - 3.0) < 1e-9, rates   # mean(4, 2)
    assert abs(rates["c"] - 4.0) < 1e-9, rates   # mean(6, 2)
    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_buddy_init_encoder_ablation.py --selftest
```
Expected: `NameError: name 'compute_paired_deltas' is not defined`.

- [ ] **Step 3: Implement the core functions and CLI**

Add the following above `_selftest()` (after the `SURVIVAL_JSON` constant):

```python
def cget(cfg, path, default=None):
    d = cfg
    for p in path:
        if d is None:
            return default
        try:
            d = d.get(p) if hasattr(d, "get") else getattr(d, p, None)
        except Exception:
            return default
    return default if d is None else d


def sget(summ, key, default=np.nan):
    try:
        v = summ.get(key, default)
    except Exception:
        v = getattr(summ, key, default)
    return default if v is None else v


def fetch(entity, project, group, tag=None):
    import wandb
    api = wandb.Api()
    rows = []
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        if tag and tag not in (run.tags or []):
            continue
        cfg, summ = run.config, run.summary
        pair = cget(cfg, ("train", "buddies", "encoder_pair"))
        if not pair:
            continue
        row = {
            "run_id": run.id,
            "state": run.state,
            "encoder_pair": pair,
            T2I: float(sget(summ, T2I)) if not np.isnan(sget(summ, T2I)) else np.nan,
            I2T: float(sget(summ, I2T)) if not np.isnan(sget(summ, I2T)) else np.nan,
        }
        for cname, cpath in CELL:
            cv = cget(cfg, cpath)
            row[cname] = cv
        rows.append(row)
    return pd.DataFrame(rows)


def compute_paired_deltas(df, metric, treatment_pair):
    """Pair BASELINE vs treatment_pair within each seed. Returns list of (seed, delta) where
    delta = treatment - baseline. Pure function - no wandb, no I/O."""
    deltas = []
    for seed, cell in df.groupby("seed"):
        by_pair = cell.groupby("encoder_pair")[metric].max()
        if BASELINE not in by_pair.index or treatment_pair not in by_pair.index:
            continue
        b, t = by_pair[BASELINE], by_pair[treatment_pair]
        if np.isnan(b) or np.isnan(t):
            continue
        deltas.append((seed, t - b))
    return deltas


def summarize(deltas):
    """mean, std, sem, z=mean/sem, win-rate - matches analyze_buddy_families.py's convention."""
    n = len(deltas)
    if n == 0:
        return {"n": 0}
    arr = np.asarray([d for _, d in deltas], dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    z = mean / sem if (n > 1 and sem > 0) else float("nan")
    wins = int((arr > 0).sum())
    return {"n": n, "mean": mean, "std": std, "sem": sem, "z": z, "wins": wins}


def survival_rate_per_cell(grid: dict) -> dict:
    """Mean off-diagonal chance-lift of each cell's union graph E against the other 15 cells
    (docs/reports/assets/buddy_cross_vlm/grid_agreement.json's 'E'.'lift' matrix), keyed by
    cell name (e.g. 'dinov2xbge'). Higher = more cross-VLM-consensus buddy structure."""
    cells = grid["cells"]
    lift = np.asarray(grid["E"]["lift"], dtype=float)
    n = len(cells)
    rates = {}
    for i, name in enumerate(cells):
        off = [lift[i, j] for j in range(n) if j != i]
        rates[name] = float(np.mean(off))
    return rates


def to_pair_key(encoder_pair: str) -> str:
    """'dinov2:bge' -> 'dinov2xbge' to match grid_agreement.json's cell-name convention."""
    v, t = encoder_pair.split(":")
    return f"{v}x{t}"


def analyze(entity, project, group, tag=None):
    print(f"\n{'='*78}\nExperiment 8 - buddy-init encoder-pair ablation  group='{group}'"
          + (f"  tag='{tag}'" if tag else "") + f"\n{'='*78}")
    df = fetch(entity, project, group, tag=tag)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag)")
        return
    n_unfinished = (df["state"] != "finished").sum()
    pairs = sorted(df["encoder_pair"].unique())
    print(f"  {len(df)} run(s); pairs present: {pairs}."
          + (f"  [{n_unfinished} not finished -> best-so-far]" if n_unfinished else ""))

    with open(SURVIVAL_JSON) as f:
        grid = json.load(f)
    surv = survival_rate_per_cell(grid)

    summary_rows = []
    for metric in (T2I, I2T):
        print(f"\n  --- {metric} (vs. {BASELINE}) ---")
        for pair in pairs:
            if pair == BASELINE:
                continue
            deltas = compute_paired_deltas(df, metric, pair)
            s = summarize(deltas)
            if s["n"] == 0:
                continue
            sig = f"  mean/SEM={s['z']:+.1f}{' *' if not np.isnan(s['z']) and abs(s['z']) >= 2 else ''}" if s["n"] > 1 else ""
            print(f"    {pair:>20}: mean delta = {s['mean']:+.2f} (n={s['n']}, wins={s['wins']}){sig}")
            summary_rows.append({"metric": metric, "encoder_pair": pair, "mean_delta": s["mean"],
                                   "z": s.get("z", np.nan), "survival_rate": surv.get(to_pair_key(pair), np.nan)})

    corr_df = pd.DataFrame(summary_rows)
    for metric in (T2I, I2T):
        sub = corr_df[corr_df["metric"] == metric].dropna(subset=["mean_delta", "survival_rate"])
        if len(sub) >= 3:
            r = np.corrcoef(sub["mean_delta"], sub["survival_rate"])[0, 1]
            print(f"\n  Correlation(mean delta [{metric}], C3 survival rate) over {len(sub)} pairs: r={r:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="buddy-init encoder-pair ablation")
    ap.add_argument("--tag", default=None, help="only include runs carrying this wandb tag")
    ap.add_argument("--selftest", action="store_true", help="offline arithmetic check, no wandb call")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(args.entity, args.project, args.group, tag=args.tag)


if __name__ == "__main__":
    main()
```

Delete the old `if __name__ == "__main__":` block that only handled `--selftest` (Step 1's version) — it's superseded by `main()` above.

- [ ] **Step 4: Run the selftest again to verify it passes**

```bash
python scripts/analyze_buddy_init_encoder_ablation.py --selftest
```
Expected: `SELFTEST OK`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_buddy_init_encoder_ablation.py
git commit -m "feat: add paired analysis + survival-rate correlation for encoder-pair ablation (Experiment 8)"
```

---

### Task 6: Launch the full 150k sweep

**Files:** none (execution only).

**Interfaces:**
- Consumes: Task 4's sweep script.
- Produces: 48 finished wandb runs (16 pairs × 3 seeds) feeding Task 7's analysis.

- [ ] **Step 1: Confirm the held-out feature cache is complete**

```bash
for m in dinov2 siglip_v vit_sup minilm bge e5; do
  test -f src/test/20260708_heldout_grid/heldout_feats/redcaps/${m}.npy \
    && echo "OK  $m" || echo "MISSING  $m -- run: python src/test/20260708_heldout_grid/extract_heldout.py --dataset redcaps --model $m"
done
```
Expected: `OK` for all six (already produced by the C3 cross-VLM survival study). If any are missing, extract them before proceeding.

- [ ] **Step 2: Confirm Task 4's smoke tests and Task 3's equivalence check passed**

Verify Task 3 Step 2's two smoke runs agreed within noise, and Task 4 Step 2's smoke tests completed without error, before spending full compute.

- [ ] **Step 3: Launch the full sweep**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_buddy_init_encoder_ablation.sh
```
This runs 16 pairs × 3 seeds = 48 runs at 100 epochs each. Long-running — launch with `run_in_background` or `nohup ... &` if executing interactively.

- [ ] **Step 4: Verify all 48 runs finished**

Check the wandb UI (project `cosir_image`, group `buddy-init encoder-pair ablation`, tag `buddy-encoder-ablation-redcaps_150k`) or query via `wandb.Api()` that all 48 runs show `state == "finished"` with `test_oracle/t2i_R1` and `test_oracle/i2t_R1` present.

---

### Task 7: Analyze results and write the report

**Files:**
- Create: `docs/reports/2026-08-24_buddy_init_encoder_ablation.md` (adjust date to when Task 6 actually completes)
- Modify: `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` (add the Experiment 8 outcome, and re-check whether Experiment 6 is still needed per its "likely subsumed" note in §6/§8)

**Interfaces:**
- Consumes: `scripts/analyze_buddy_init_encoder_ablation.py` (Task 5) output.
- Produces: an input to the paper's headline-initializer decision and to the Experiment 6 go/no-go call.

- [ ] **Step 1: Run the analysis**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_buddy_init_encoder_ablation.py --tag buddy-encoder-ablation-redcaps_150k
```
Capture the full printed output (per-pair paired tables + mean Δ ± std + mean/SEM for both `t2i_R1`/`i2t_R1`, plus the two survival-rate correlation lines).

- [ ] **Step 2: Apply the spec's decision rule**

Per spec §4 Experiment 8's success criteria: if one or more non-CLIP pairs beat `clip_img:clip_txt` (seed-replicated, mean/SEM ≥ 2), that pair is a candidate stronger initializer to lead the paper with; if all 16 cluster near the baseline within the noise floor, that is itself evidence the graph *structure* (not the specific encoder) drives usefulness. Check the survival-rate correlation: a strong positive `r` answers Experiment 6's deferred question directly.

- [ ] **Step 3: Write the results report**

Create `docs/reports/2026-08-24_buddy_init_encoder_ablation.md` following the structure of `docs/reports/2026-08-16_buddy_init_ablation.md` (method, per-pair results table, survival-rate correlation, interpretation, caveats, reproduction commands).

- [ ] **Step 4: Update the spec**

In `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`: add the outcome as a new row in §2's claims table (next available letter), citing the new report; and in §6/§8, resolve the "Experiment 6 likely subsumed" note based on the actual correlation result (drop Experiment 6 from scope if the correlation is clear either way; keep it only if the result is ambiguous enough that Experiment 6's per-sample-level version would add real information).

- [ ] **Step 5: Commit**

```bash
git add docs/reports/2026-08-24_buddy_init_encoder_ablation.md docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
git commit -m "results: buddy-init encoder-pair ablation (Experiment 8)"
```

---

## Self-Review

- **Spec coverage:** Tasks 1–3 build the (previously nonexistent) capability to source the buddy graph from a non-CLIP encoder pair while keeping the CLIP backbone fixed — the mechanism spec §4 Experiment 8 assumed was reusable but wasn't quite there yet. Tasks 4–7 cover the sweep, analysis (including the survival-rate correlation that folds in Experiment 6's deferred question), execution, and report/spec update end-to-end, at the 150k scope this plan commits to.
- **Placeholder scan:** every code block is complete and runnable as written. Task 1's test file explicitly calls out and then removes its own placeholder dead branch in Step 1 — a one-time authoring note, not a plan placeholder left for an implementer to fill in.
- **Type/interface consistency:** `_buddy_init(..., feature_override: Optional[Tuple[np.ndarray, np.ndarray, List[int]]] = None)` (Task 1) is consumed identically in Task 3's `_buddy_kwargs["feature_override"] = (...)` and in Task 2's `load_encoder_pair_features(...) -> Tuple[np.ndarray, np.ndarray, List[int]]` return, whose three-element order matches. `compute_paired_deltas(df, metric, treatment_pair) -> list[(seed, float)]`, `summarize(deltas) -> dict`, and `survival_rate_per_cell(grid) -> dict[str, float]` (Task 5) are defined once and used identically in `analyze()`/`_selftest()`.
- **Scope check:** RedCaps-150k, 16 pairs, 3 seeds (48 runs) only — the 300k extension is explicitly out of scope for this plan (a separate follow-up plan once this one's report is in, per the spec's own gating). Experiments 0–7, 9 are untouched by this plan.
- **Sample-ID consistency (CLAUDE.md's flagged failure mode):** addressed directly — Task 2's loader asserts the held-out cache's row order matches `feature_manager.get_all_sample_ids()` before ever building a graph from it, and Task 1's test deliberately uses non-`range(n)` sample IDs to catch an accidental positional-only implementation.
