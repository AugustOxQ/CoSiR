# Matched PercepT-vs-buddy Head-to-Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One validated harness that runs buddy's and PercepT's Stage 1 + a shared Stage 2 at matched topic counts, with a val/test split of the held-out set, so an equal-budget W&B search can compare the two systems fairly.

**Architecture:** New `scripts/buddy_percept_sweep/h2h_*.py` modules sit beside the existing §6i harness (which stays unchanged and reproducible). Stage 1 for each system is a faithful port that calls the frozen pilots' own helper functions (knobs applied by setting the pilot modules' constants), outputs embeddings + topics, and hands them to a shared evaluation + Stage 2 path. A disk store caches every expensive, hyperparameter-independent input.

**Tech Stack:** Python 3.10, PyTorch, numpy, scikit-learn, igraph/leidenalg, W&B sweeps, DAS6 via the cluster-run CLI.

**Spec:** `docs/superpowers/specs/2026-09-30-matched-percept-buddy-h2h-design.md`

## Global Constraints

- Worktree `/project/CoSiR-buddy_prototype_conditioning`, branch `experiment/percept_topic_pipeline`. Env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR`.
- Never edit `src/test/20260922_percept_topic_pipeline/`, `src/test/20260923_artelingo_buddy_analysis/`, `src/test/20260927_deep_stage_analysis/` (load/import only), nor `src/hook/train_cosir.py`, `scripts/run_sweep_agent.py`, `scripts/sweep_config_v*.yaml`, `scripts/buddy_percept_sweep/real_data.py`.
- Existing harness modules (`pipeline.py`, `stage1.py`, `stage2.py`, `targets.py`, `clustering.py`, `cache.py`, `config.py`) keep their behaviour; only additive changes, and only where a task says so.
- Held-out evaluation labels: k-NN vote into the system's own train topics, **k = 20, fixed** (`EVAL_TRANSFER_K = 20`); never a sweep parameter.
- Seeds: search (1001, 1002); stress (42, 7, 123, 2024); test (11, 23, 57, 101, 211).
- Topic-count levels: `k_target` ∈ {16, 40}; buddy tolerance ±2.
- DAS6 wrappers: `#! /bin/bash`, `set -euo pipefail`, export `PERCEPT_FEATURE_ROOT=/local/wding/pre_extract`, `PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; no conda sourcing; never set `CUDA_VISIBLE_DEVICES`; file name `scripts/run_*.sh`.
- Implementers do **not** commit; the controller commits after review (parallel implementers share one git index).
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_014TPkEGSdVV1Lq7Krp5ARdC`.
- Test command: `python -m pytest scripts/buddy_percept_sweep src/test/20260928_buddy_percept_sweep src/test/20260930_harness_confirmation src/test/20260930_matched_h2h -q`. No test may need real data, a GPU, or W&B.

## Review Focus

1. A buddy Stage 1 whose embedding cannot reach the K band in 12 bisection steps → the trial returns `objective = -1.0` with `k_miss = True` logged; never an exception.
2. A held-out subset where some topic has no evaluation-label members → that topic is skipped in macro AUC, `skipped_topics` is logged, the trial still returns a finite AUC.
3. Several agents on one node start at once with no cache → exactly one builds the store; the others wait on the lock and load it; no process ever reads a partially written file.
4. PercepT DEC leaves a surviving center with no train members → `n_topics` reports the number of distinct train labels actually used, not `k_target`; Stage 2 target width = that number.
5. `k_target = 0` (fixed-resolution reference mode used for the §6i winner) → no bisection; `leiden_resolution` from the config is used as given.

---

## File structure

| File | Responsibility | Task |
|---|---|---|
| `scripts/buddy_percept_sweep/h2h_types.py` | `Stage1Output`, `H2HSplit` dataclasses, constants | 1 |
| `scripts/buddy_percept_sweep/h2h_store.py` | build/load disk store of all fixed inputs for both systems | 1 |
| `scripts/buddy_percept_sweep/h2h_split.py` | deterministic stratified val/test split of held-out | 1 |
| `scripts/buddy_percept_sweep/pilot_metrics.py` (modify) | add `arch`, `cca_audit` to `PilotModules` | 1 |
| `scripts/buddy_percept_sweep/h2h_buddy.py` | buddy Stage 1: pilot-faithful port + harness adapter | 2 |
| `scripts/buddy_percept_sweep/h2h_percept.py` | PercepT Stage 1: pilot-faithful port | 3 |
| `scripts/buddy_percept_sweep/h2h_topics.py` | graph building, Leiden, K-target bisection | 4 |
| `scripts/buddy_percept_sweep/h2h_eval.py` | AMIs (both yardsticks), Stage 2 train/eval on a subset | 4 |
| `scripts/buddy_percept_sweep/h2h_trial.py` | `H2HConfig`, `resolve_h2h_config`, `run_h2h_trial` | 4 |
| `src/test/20260930_matched_h2h/validate_v1_stage2.py` | V1 (local) | 5 |
| `src/test/20260930_matched_h2h/validate_v2_v3.py` + `scripts/run_h2h_validate.sh` | V2/V3 (DAS6) | 6 |
| `scripts/buddy_percept_sweep/h2h_agent.py`, `scripts/h2h_sweeps/*.yaml`, `scripts/run_h2h_agent.sh` | in-process W&B agent + 4 sweep configs | 7 |
| `scripts/buddy_percept_sweep/h2h_select.py`, `scripts/run_h2h_select.sh` | select / stress / test / summarize | 8 |

Tests for `scripts/buddy_percept_sweep/h2h_*.py` go in `scripts/buddy_percept_sweep/tests/test_h2h_*.py`; tests for validation scripts in `src/test/20260930_matched_h2h/test_*.py`.

---

### Task 1: Types, fixed-input store, val/test split

**Files:**
- Create: `scripts/buddy_percept_sweep/h2h_types.py`, `scripts/buddy_percept_sweep/h2h_store.py`, `scripts/buddy_percept_sweep/h2h_split.py`
- Modify: `scripts/buddy_percept_sweep/pilot_metrics.py` (add two fields + loading)
- Test: `scripts/buddy_percept_sweep/tests/test_h2h_store.py`, `scripts/buddy_percept_sweep/tests/test_h2h_split.py`

**Interfaces:**
- Consumes: `pilot_metrics.load_pilot_modules()`; pilot functions named below.
- Produces:

```python
# h2h_types.py
EVAL_TRANSFER_K = 20
SEARCH_SEEDS = (1001, 1002)
STRESS_SEEDS = (42, 7, 123, 2024)
TEST_SEEDS = (11, 23, 57, 101, 211)

@dataclass
class Stage1Output:
    train_embedding: np.ndarray            # (N_train, D) float32
    heldout_embedding: np.ndarray          # (N_heldout, D) float32
    train_labels: Optional[np.ndarray]     # PercepT: native train topic ids; buddy: None
    heldout_native: Optional[np.ndarray]   # PercepT: native held-out topic ids; buddy: None
    info: dict                              # stop_reason, epochs_run, seconds, ...

@dataclass
class H2HSplit:
    val_idx: np.ndarray    # int64, sorted, indices into held-out paintings
    test_idx: np.ndarray   # int64, sorted, disjoint from val_idx, union = all
    digest: str            # sha1 hex of val_idx.tobytes() + test_idx.tobytes()

# h2h_store.py
@dataclass
class H2HStore:
    train_paintings: np.ndarray; heldout_paintings: np.ndarray       # object
    train_img: np.ndarray; train_txt: np.ndarray                     # float32 dedup CLIP nodes
    heldout_img: np.ndarray; heldout_txt: np.ndarray
    train_content_raw: np.ndarray; heldout_content_raw: np.ndarray   # cca_audit.content_features
    train_affect28: np.ndarray; heldout_affect28: np.ndarray         # float32 (N, 28)
    train_percept_h: np.ndarray; heldout_percept_h: np.ndarray       # PercepT fused inputs, float32
    train_emotion: np.ndarray; heldout_emotion: np.ndarray           # object, majority emotion
    train_genre: np.ndarray; heldout_genre: np.ndarray               # object, "" when missing
    train_patches: torch.Tensor; heldout_patches: torch.Tensor       # loaded from the .pt files, never cached

def default_cache_dir() -> Path  # $H2H_CACHE_DIR, else f"{$PERCEPT_FEATURE_ROOT or /data/SSD2/pre_extract}/artelingo_h2h_cache"
def build_arrays(pilot) -> dict[str, np.ndarray]        # the expensive part (real data)
def load_patches(n_train: int, n_heldout: int) -> tuple[torch.Tensor, torch.Tensor]
def load_or_build_store(cache_dir: Path | None = None,
                        builder: Callable[[], dict] | None = None,
                        patch_loader: Callable[[int, int], tuple] | None = None) -> H2HStore

# h2h_split.py
def make_split(heldout_emotion: np.ndarray, heldout_genre: np.ndarray, seed: int = 0) -> H2HSplit
```

- [ ] **Step 1: Write failing tests** (`test_h2h_split.py`)

```python
import numpy as np
from scripts.buddy_percept_sweep.h2h_split import make_split

def _labels(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    emotion = rng.choice(np.array(["joy", "fear", "awe", "sad"], dtype=object), size=n)
    genre = np.where(rng.random(n) < 0.02, "portrait", "").astype(object)
    return emotion, genre

def test_split_is_disjoint_complete_sorted():
    e, g = _labels()
    s = make_split(e, g)
    assert len(np.intersect1d(s.val_idx, s.test_idx)) == 0
    assert np.array_equal(np.sort(np.concatenate([s.val_idx, s.test_idx])), np.arange(len(e)))
    assert np.all(np.diff(s.val_idx) > 0) and np.all(np.diff(s.test_idx) > 0)

def test_split_is_deterministic_and_seed_sensitive():
    e, g = _labels()
    assert make_split(e, g).digest == make_split(e, g).digest
    assert make_split(e, g, seed=1).digest != make_split(e, g).digest

def test_split_is_stratified_by_emotion_and_genre_presence():
    e, g = _labels()
    s = make_split(e, g)
    for label in np.unique(e):
        n_val = np.sum(e[s.val_idx] == label); n_all = np.sum(e == label)
        assert abs(n_val - n_all / 2) <= 1
    has_g = g != ""
    assert abs(has_g[s.val_idx].sum() - has_g.sum() / 2) <= 4   # one per emotion stratum at most
```

`test_h2h_store.py` (inject a fake builder; never touches real data):

```python
import numpy as np, torch, threading
from scripts.buddy_percept_sweep import h2h_store

def _fake_arrays():
    n, m = 12, 6
    f = lambda r, c: np.arange(r * c, dtype=np.float32).reshape(r, c)
    obj = lambda k, v: np.array([v] * k, dtype=object)
    return dict(train_paintings=obj(n, "p"), heldout_paintings=obj(m, "q"),
                train_img=f(n, 4), train_txt=f(n, 4), heldout_img=f(m, 4), heldout_txt=f(m, 4),
                train_content_raw=f(n, 8), heldout_content_raw=f(m, 8),
                train_affect28=f(n, 28), heldout_affect28=f(m, 28),
                train_percept_h=f(n, 10), heldout_percept_h=f(m, 10),
                train_emotion=obj(n, "joy"), heldout_emotion=obj(m, "fear"),
                train_genre=obj(n, ""), heldout_genre=obj(m, "portrait"))

def _patches(n, m):
    return torch.zeros(n, 3, 5), torch.zeros(m, 3, 5)

def test_build_then_load_round_trips(tmp_path):
    calls = []
    def builder():
        calls.append(1); return _fake_arrays()
    a = h2h_store.load_or_build_store(tmp_path, builder=builder, patch_loader=_patches)
    b = h2h_store.load_or_build_store(tmp_path, builder=builder, patch_loader=_patches)
    assert len(calls) == 1
    assert np.array_equal(a.train_content_raw, b.train_content_raw)
    assert list(b.heldout_genre) == ["portrait"] * 6
    assert b.train_patches.shape == (12, 3, 5)

def test_concurrent_callers_build_once(tmp_path):
    calls = []
    def builder():
        calls.append(1); return _fake_arrays()
    threads = [threading.Thread(target=h2h_store.load_or_build_store,
                                args=(tmp_path,), kwargs=dict(builder=builder, patch_loader=_patches))
               for _ in range(4)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(calls) == 1

def test_partial_file_is_never_loaded(tmp_path):
    (tmp_path / "h2h_store.npz.tmp").write_bytes(b"garbage")
    s = h2h_store.load_or_build_store(tmp_path, builder=_fake_arrays, patch_loader=_patches)
    assert s.train_img.shape == (12, 4)
```

- [ ] **Step 2: Run tests, confirm they fail** (`ImportError`).

- [ ] **Step 3: Implement.**
  - `make_split`: strata = `(emotion, genre != "")`. For each stratum in sorted order, permute its indices with `np.random.default_rng(seed)` (one generator for the whole call, strata visited in sorted order), put the first `ceil(n/2)` in val, the rest in test. Sort both. `digest = hashlib.sha1(val.tobytes() + test.tobytes()).hexdigest()`.
  - `pilot_metrics.py`: add fields `arch: object = None`, `cca_audit: object = None` (defaults keep existing tests valid); in `load_pilot_modules`, keep `arch`, load `cca_audit = arch.load_sibling_module("cca_for_metrics", arch.CCA_AUDIT_PATH)`, set `arch.cca_audit = cca_audit`, and route `arch.HELDOUT_STORAGE_DIR` / `arch.HELDOUT_JSON` with the same env formulas already used for `heldout_pipeline`.
  - `build_arrays(pilot)`: mirror `real_data.load_real_raw_inputs` (read it) for dedup features, majority emotion, 28-d affect (`affect_pilot.extract_affect_nodes`, cast float32), content (`cca_audit.content_features(img, txt, affect_pilot)`), genre (`pipeline.load_genre_map()`). For PercepT inputs load `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py` as `base` (read `run_percept_mapper_symmetric_sweep_pilot.fit_stage1_and_get_targets` in `src/test/20260927_deep_stage_analysis/` — it is the reference flow) and call `base.extract_affect_embedding_nodes(json_path, paintings, device, log)` for train (`pipeline.TRAIN_JSON`) and held-out (`base.HELDOUT_JSON`), then `base.fused_embeddings(img, txt, affect768, cca_audit, affect_pilot)`. Assert held-out count 9,365 and train count 61,402 only when not under test (the builder is injected in tests).
  - `load_patches`: load `src/test/20260922_percept_topic_pipeline/run_percept_stage2_pilot.py` like real_data does and call its `load_patch_features` for train and held-out.
  - `load_or_build_store`: `cache_dir.mkdir(parents=True, exist_ok=True)`; take an exclusive `fcntl.flock` on `cache_dir / "h2h_store.lock"` (open with `"a"`); inside the lock, if `h2h_store.npz` is missing, call `builder()` (default: `lambda: build_arrays(load_pilot_modules())`), write with `np.savez(tmp)` where `tmp = cache_dir / "h2h_store.npz.tmp"` (overwrite any stale tmp), then `os.replace(tmp, final)`. Release the lock, then `np.load(final, allow_pickle=True)` and build `H2HStore`, adding patches from `patch_loader(len(train_paintings), len(heldout_paintings))` (default `load_patches`). Threads in one process: also guard with a module-level `threading.Lock` around the flock section, because flock is per open-file and may not serialize threads.
- [ ] **Step 4: Run the test command; all pass.**
- [ ] **Step 5: Stop; report files changed + test summary to the controller (no commit).**

---

### Task 2: Buddy Stage 1 (pilot-faithful port + harness adapter)

**Files:**
- Create: `scripts/buddy_percept_sweep/h2h_buddy.py`
- Test: `scripts/buddy_percept_sweep/tests/test_h2h_buddy.py`

**Interfaces:**
- Consumes: `H2HStore`, `Stage1Output` (Task 1), `PilotModules` with `arch` (Task 1); existing `stage1.ParameterizedLearnedStudent`, `stage1.train_stage1`, `cache.FixedInputs`.
- Produces:

```python
@dataclass
class BuddyStage1Config:
    impl: str = "pilot"              # "pilot" | "harness"
    heads: str = "attn1"             # pilot: mlp128|attn1|attn4 ; harness: mlp128|attn
    num_heads: int = 1               # harness only
    d_shared: int = 32
    content_pca_dim: int = 50
    lr: float = 1e-3
    batch_size: int = 1024
    temperature: float = 0.1         # pilot only (arch.TEMPERATURE)
    max_epochs: int = 200
    plateau_window: int = 5          # pilot only
    plateau_rel_improvement: float = 0.01  # pilot only
    noise_std: float = 0.0           # harness only
    lambda_affect: float = 1.0       # harness only
    weight_decay: float = 0.0        # harness only
    teacher_graph_K: int = 20        # harness only

def fit_buddy_stage1(cfg: BuddyStage1Config, store: H2HStore, seed: int,
                     monitor_idx: np.ndarray, pilot: PilotModules, device: str) -> Stage1Output
```

- [ ] **Step 1: Write failing tests** — pure pieces only (real training needs data):

```python
import numpy as np, pytest
from scripts.buddy_percept_sweep.h2h_buddy import BuddyStage1Config, pilot_constants, monitor_subsample

def test_pilot_constants_map_config_to_arch_names():
    c = pilot_constants(BuddyStage1Config(d_shared=64, lr=3e-4, batch_size=2048, temperature=0.2,
                                          max_epochs=150, plateau_window=3,
                                          plateau_rel_improvement=0.02, content_pca_dim=80))
    assert c == {"D_SHARED": 64, "LEARNING_RATE": 3e-4, "BATCH_SIZE": 2048, "TEMPERATURE": 0.2,
                 "MAX_EPOCHS": 150, "PLATEAU_WINDOW": 3, "PLATEAU_REL_IMPROVEMENT": 0.02,
                 "CONTENT_PCA_DIM": 80}

def test_monitor_subsample_matches_pilot_draw_when_full():
    # pilot: rng = default_rng(seed); sampled = rng.choice(n, 2000, replace=False); rank = rng.choice(n, 5000, replace=False)
    n = 9365
    s, r = monitor_subsample(n_monitor=n, seed=42, edge_sample=2000, rank_sample=5000)
    rng = np.random.default_rng(42)
    assert np.array_equal(s, rng.choice(n, size=2000, replace=False))
    assert np.array_equal(r, rng.choice(n, size=5000, replace=False))

def test_monitor_subsample_caps_at_subset_size():
    s, r = monitor_subsample(n_monitor=1200, seed=0, edge_sample=2000, rank_sample=5000)
    assert len(s) == 1200 and len(r) == 1200

def test_invalid_impl_or_heads_raise():
    with pytest.raises(ValueError):
        pilot_constants(BuddyStage1Config(impl="nope"))
    with pytest.raises(ValueError):
        pilot_constants(BuddyStage1Config(heads="attn"))   # pilot impl needs attn1/attn4/mlp128
```

- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement.**
  - `pilot_constants(cfg)`: validate `cfg.impl == "pilot"` and `cfg.heads in {"mlp128","attn1","attn4"}` (else `ValueError`), return the dict shown in the test.
  - `monitor_subsample(n_monitor, seed, edge_sample, rank_sample)`: `rng = np.random.default_rng(seed)`; draw `min(edge_sample, n)` then `min(rank_sample, n)` without replacement, in that order.
  - `fit_buddy_stage1` with `impl == "pilot"` is a **line-by-line port of `src/test/20260923_artelingo_buddy_analysis/run_attention_h1_embedding_snapshot_pilot.py` lines 177–380** (read them), minus data loading (from `store`), minus the epoch-0 Leiden/metrics block (it consumes no RNG), minus reporting. Exactly:
    1. `torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)` if CUDA; `torch.backends.cudnn.deterministic=True; benchmark=False; torch.use_deterministic_algorithms(True, warn_only=True)`.
    2. Set every `pilot_constants(cfg)` entry with `setattr(pilot.arch, name, value)`; restore the previous values in a `finally` (the arch module is shared per process).
    3. `PCA(n_components=cfg.content_pca_dim, random_state=seed)` fit on `store.train_content_raw`, transform both (float32). **random_state is the run seed, as in the pilot.**
    4. Teacher graphs: `pilot.pipeline.build_buddy_graphs(store.train_img, store.train_txt, K=pilot.pipeline.K, alpha=pilot.pipeline.ALPHA, device=device, connect_components=True)` → content graph (third return); affect graph `pilot.single_modality.build_single_modality_graph("train-affect-teacher", store.train_affect28, pilot.pipeline, pilot.affect_pilot, device, expected_nodes=N_train)`; edges via `pilot.arch.upper_triangle_edges`. Cache both edge arrays per process in a module dict keyed by `("pilot_teacher",)` — they do not depend on any knob or seed.
    5. Tensors as in the pilot; then `torch.manual_seed(seed); np.random.seed(seed)`; `model = pilot.arch.LearnedStudent(cfg.heads).to(device)`; `optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)`.
    6. Monitor: `sampled, rank = monitor_subsample(len(monitor_idx), seed, pilot.arch.EDGE_SAMPLE_SIZE, pilot.arch.EFFECTIVE_RANK_SAMPLE_SIZE)`; monitor reference graphs = `build_single_modality_graph` on `content_heldout[monitor_idx]` and `store.heldout_affect28[monitor_idx]` with `pilot.heldout_pipeline`, `expected_nodes=len(monitor_idx)`; epoch-0 `pilot.arch.evaluate_checkpoint(...)` on the monitor tensors; then the pilot's epoch loop verbatim (epoch_rng = `np.random.default_rng(seed + epoch)`, content pairs then affect pairs, `content_batch_embeddings`, two `symmetric_infonce`, `content_loss + affect_loss`, checkpoint every `pilot.arch.CHECKPOINT_EVERY`, plateau rule).
    7. Final `model.eval()` embeddings for train and **all** held-out → `Stage1Output(train_embedding, heldout_embedding, None, None, info={"stop_reason", "epochs_run", "seconds"})`.
    With `monitor_idx = np.arange(9365)` this must reproduce the pilot's snapshot exactly (validated by V2 in Task 6).
  - `impl == "harness"`: build a `cache.FixedInputs` from the store (PCA with `random_state=seed`, affect = `store.*_affect28`, emotion/genre/patches from the store), construct `stage1.ParameterizedLearnedStudent(heads=("attn1" if cfg.heads=="attn" else cfg.heads), num_heads, d_shared, ...)` after seeding, call `stage1.train_stage1(...)` with `max_epochs=cfg.max_epochs`, `teacher_graph_K`, `lambda_affect`, `noise_std`, `weight_decay`, `lr`, `batch_size`; held-out embedding as in `pipeline.run_trial` lines 78–85. `monitor_idx` unused.
- [ ] **Step 4: Run test command; all pass.**
- [ ] **Step 5: Report to controller (no commit).**

---

### Task 3: PercepT Stage 1 (pilot-faithful port)

**Files:**
- Create: `scripts/buddy_percept_sweep/h2h_percept.py`
- Test: `scripts/buddy_percept_sweep/tests/test_h2h_percept.py`

**Interfaces:**
- Consumes: `H2HStore`, `Stage1Output` (Task 1).
- Produces:

```python
@dataclass
class PerceptStage1Config:
    pretrain_epochs: int = 100
    pretrain_lr: float = 1e-3
    dec_lr: float = 1e-4
    lambda_balance: float = 1000.0
    lambda_reconstruction: float = 1.0
    n_initial_factor: float = 1.5      # N_initial = max(k_target, round(factor * k_target))
    stability_threshold: float = 1e-3
    max_dec_epochs: int = 500

def load_percept_modules() -> SimpleNamespace   # s2 (fixed Stage-2 pilot) with .base and .sweep, loaded once per process
def n_initial_for(cfg: PerceptStage1Config, k_target: int) -> int
def percept_constants(cfg: PerceptStage1Config, k_target: int) -> dict[str, dict[str, object]]  # {"s2": {...}, "base": {...}, "sweep": {...}}
def fit_percept_stage1(cfg: PerceptStage1Config, store: H2HStore, seed: int, k_target: int,
                       mods: SimpleNamespace, device: str) -> Stage1Output
```

- [ ] **Step 1: Write failing tests** (pure pieces):

```python
from scripts.buddy_percept_sweep.h2h_percept import PerceptStage1Config, n_initial_for, percept_constants

def test_n_initial_never_below_k():
    assert n_initial_for(PerceptStage1Config(n_initial_factor=1.0), 40) == 40
    assert n_initial_for(PerceptStage1Config(n_initial_factor=1.5), 40) == 60
    assert n_initial_for(PerceptStage1Config(n_initial_factor=2.5), 16) == 40

def test_constants_target_the_modules_that_read_them():
    c = percept_constants(PerceptStage1Config(), 40)
    # defaults reproduce the §6g fixed pilot: N 60 -> 40, lambda_balance 1000, lambda_recon 1
    assert c["s2"]["N_INITIAL_CLUSTERS"] == 60 and c["s2"]["N_SURVIVING_CLUSTERS"] == 40
    assert c["s2"]["LAMBDA_BALANCE"] == 1000.0 and c["s2"]["LAMBDA_RECONSTRUCTION"] == 1.0
    assert c["base"]["PRETRAIN_EPOCHS"] == 100 and c["base"]["PRETRAIN_LEARNING_RATE"] == 1e-3
    assert c["sweep"]["DEC_LEARNING_RATE"] == 1e-4 and c["sweep"]["STABILITY_THRESHOLD"] == 1e-3
```

  Before writing `percept_constants`, **read** `src/test/20260922_percept_topic_pipeline/run_percept_stage2_fixed_pilot.py`, its `base` (`run_percept_stage1_pilot.py`) and `sweep` modules, and find which module each constant is actually read from at call time (e.g. `train_dec_until_stable_fixed` reads `sweep.DEC_LEARNING_RATE`, `sweep.MAX_DEC_EPOCHS`, `sweep.STABILITY_THRESHOLD`; λ_balance / λ_recon may be read from `s2` or `sweep`). If a constant is read from a different module than the test assumes, fix the **test's module key** to match reality and say so in your report. Also set the seed constants (`SEED`) on every module that reads one.

- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement.**
  - `load_percept_modules()`: load `src/test/20260922_percept_topic_pipeline/run_percept_stage2_fixed_pilot.py` as `s2` (importlib, unique name); it exposes `s2.base`, `s2.sweep` (see `fit_stage1_and_get_targets`). Cache in a module global.
  - `fit_percept_stage1`: apply `percept_constants(cfg, k_target)` plus `SEED=seed` with setattr, restoring old values in `finally`. Seed numpy/torch/cuda. Then follow `fit_stage1_and_get_targets` from `train_inputs = torch.from_numpy(store.train_percept_h)` onward: `build_autoencoder`, `pretrain_autoencoder`, `sweep.initialize_cluster_centers(encoder, train_inputs, device, N_initial, seed)`, `s2.train_dec_until_stable_fixed(...)`, `s2.prune_centers_fixed(centers, k_target)`. Latent embeddings: `encoder(inputs)` in eval/no_grad, batched by 8192, for train and held-out (float32 numpy). Native labels: the argmax of the DEC soft assignment to the **surviving** centers — use the same assignment function the pilot uses for `train_topic` / `heldout_topic` in `src/test/20260927_deep_stage_analysis/run_percept_fixed_snapshot_pilot.py` (read it; reuse the function, do not re-derive). Relabel native train labels to 0..n-1 by sorted unique id and apply the same map to held-out; held-out ids absent from train map to the nearest surviving center that has train members (record the count in `info["heldout_unmapped"]`). `info`: `n_initial`, `dec_epochs`, `dec_stop_reason`, `n_train_topics` (distinct), `seconds`.
- [ ] **Step 4: Run test command; all pass.**
- [ ] **Step 5: Report to controller (no commit).**

---

### Task 4: Topics, evaluation, trial orchestration

**Files:**
- Create: `scripts/buddy_percept_sweep/h2h_topics.py`, `scripts/buddy_percept_sweep/h2h_eval.py`, `scripts/buddy_percept_sweep/h2h_trial.py`
- Test: `scripts/buddy_percept_sweep/tests/test_h2h_topics.py`, `test_h2h_eval.py`, `test_h2h_trial.py`

**Interfaces:**
- Consumes: Tasks 1–3; existing `clustering.merge_small_communities`, `targets.assign_to_train_communities`, `targets.cosine_vote_fractions`, `targets.build_targets`, `targets.one_hot`, `stage2.ParameterizedAttentionPoolingMapper`, `stage2.train_stage2`, `stage2.evaluate_auc`, `stage2.auc_summary`, `pilot_metrics.independent_partition`, `pilot_metrics.ami_emotion_genre`.
- Produces:

```python
# h2h_topics.py
def build_topic_graph(embedding, kind: str, pilot, device: str, k_neighbors: int = 20) -> scipy.sparse.csr_matrix
    # kind "mknn": src.conditional_buddy.buddy_graph.mutual_knn(embedding, K=k_neighbors, backend="auto", device=device)
    # kind "pilot_repaired": pilot.single_modality.build_single_modality_graph("train-topics", embedding, pilot.pipeline, pilot.affect_pilot, device, expected_nodes=len(embedding))
def leiden_on_graph(graph, resolution: float, seed: int) -> np.ndarray   # RBConfigurationVertexPartition, 0..n-1 labels
def target_k_partition(embedding, graph, k_target: int, tolerance: int, merge_threshold: float,
                       seed: int, max_steps: int = 12, lo: float = 0.02, hi: float = 20.0) -> tuple[np.ndarray, dict]
    # bisection on log(resolution); each step: leiden -> merge_small_communities -> K; hit if |K-k_target| <= tolerance
    # info: resolution, k_raw, k_after_merge, steps, hit (bool); returns best-so-far labels if not hit

# h2h_eval.py
def eval_labels(train_embedding, train_labels, heldout_embedding, subset_idx) -> np.ndarray   # transfer, k=EVAL_TRANSFER_K
def stage1_metrics(train_embedding, train_labels, heldout_embedding, subset_idx, heldout_emotion,
                   heldout_genre, seed, pilot, device, native_subset=None) -> dict
    # keys: transfer_emo, transfer_genre, ind_emo, ind_genre, ind_k, [native_emo, native_genre]
def stage2_metrics(s2cfg, store, train_embedding, train_labels, subset_idx,
                   eval_label_sets: dict[str, np.ndarray], seed: int, device: str) -> dict
    # trains ONE mapper, evaluates it against each label set: keys auc_<name>, skipped_<name>

# h2h_trial.py
@dataclass
class Stage2Config:
    mapper_lr: float = 1e-2; mapper_epochs: int = 400; num_queries: int = 1; mlp_head: str = "linear"
    weight_decay_stage2: float = 0.0; class_balanced_loss: bool = False
    target_cutoff: Union[str, float] = "single_label"; train_target_k: int = 20

@dataclass
class H2HConfig:
    system: str                     # "buddy" | "percept"
    k_target: int                   # 16 | 40 ; 0 = fixed-resolution reference mode (buddy only)
    buddy: BuddyStage1Config
    percept: PerceptStage1Config
    stage2: Stage2Config
    leiden_graph: str = "mknn"      # buddy: "mknn" | "pilot_repaired"
    merge_small_threshold: float = 0.0
    leiden_resolution: float = 1.0  # used only when k_target == 0

def resolve_h2h_config(raw: dict) -> H2HConfig
    # flat keys: system, k_target, leiden_graph, merge_small_threshold, leiden_resolution,
    # buddy_<field>, percept_<field>, and Stage2Config field names; unknown keys starting with "_" ignored,
    # any other unknown key -> ValueError; missing keys -> dataclass defaults
def run_h2h_trial(cfg: H2HConfig, store: H2HStore, split: H2HSplit, subset: str, seeds: tuple[int, ...],
                  pilot, percept_mods, device: str, monitor: str = "val") -> dict
    # returns {"per_seed": [row...], "objective": float, "mean": {...}, "split_digest": str}
```

  Per-seed row keys: `seed, n_topics, k_miss, resolution, auc_primary, skipped_primary, auc_native (percept only), transfer_emo, transfer_genre, ind_emo, ind_genre, ind_k, native_emo, native_genre (percept), stage1_seconds, stage2_seconds`. `objective = mean(auc_primary)` over seeds, or `-1.0` if any seed has `k_miss`.

- [ ] **Step 1: Write failing tests**, synthetic only:

```python
# test_h2h_topics.py
import numpy as np
from scipy.sparse import csr_matrix
from scripts.buddy_percept_sweep.h2h_topics import leiden_on_graph, target_k_partition

def _cliques(n_cliques, size):
    rows, cols = [], []
    for c in range(n_cliques):
        base = c * size
        for i in range(size):
            for j in range(size):
                if i != j:
                    rows.append(base + i); cols.append(base + j)
        rows.append(base); cols.append(((c + 1) % n_cliques) * size)   # ring link keeps it connected
        cols.append(base); rows.append(((c + 1) % n_cliques) * size)
    n = n_cliques * size
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

def test_leiden_finds_cliques():
    labels = leiden_on_graph(_cliques(6, 10), resolution=1.0, seed=0)
    assert len(np.unique(labels)) == 6

def test_target_k_hits_band():
    emb = np.random.default_rng(0).normal(size=(60, 4)).astype(np.float32)
    labels, info = target_k_partition(emb, _cliques(6, 10), k_target=6, tolerance=0,
                                      merge_threshold=0.0, seed=0)
    assert info["hit"] and len(np.unique(labels)) == 6

def test_target_k_reports_miss_instead_of_raising():
    emb = np.random.default_rng(0).normal(size=(60, 4)).astype(np.float32)
    labels, info = target_k_partition(emb, _cliques(6, 10), k_target=40, tolerance=0,
                                      merge_threshold=0.0, seed=0, max_steps=4)
    assert info["hit"] is False and len(labels) == 60
```

```python
# test_h2h_eval.py
import numpy as np
from scripts.buddy_percept_sweep.h2h_eval import eval_labels

def test_eval_labels_follow_nearest_train_topic():
    rng = np.random.default_rng(0)
    a = rng.normal([20, 0], 0.1, size=(50, 2)); b = rng.normal([0, 20], 0.1, size=(50, 2))
    train = np.vstack([a, b]).astype(np.float32); labels = np.array([0] * 50 + [1] * 50)
    held = np.vstack([rng.normal([20, 0], 0.1, size=(10, 2)), rng.normal([0, 20], 0.1, size=(10, 2))]).astype(np.float32)
    out = eval_labels(train, labels, held, np.arange(20))
    assert list(out) == [0] * 10 + [1] * 10
```

```python
# test_h2h_trial.py
import pytest
from scripts.buddy_percept_sweep.h2h_trial import resolve_h2h_config

def test_resolve_routes_prefixed_keys():
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 16, "buddy_lr": 3e-4, "buddy_impl": "harness",
                              "percept_lambda_balance": 10.0, "mapper_lr": 3e-3, "target_cutoff": "0.3",
                              "_wandb": {}})
    assert cfg.buddy.lr == 3e-4 and cfg.buddy.impl == "harness"
    assert cfg.percept.lambda_balance == 10.0
    assert cfg.stage2.mapper_lr == 3e-3 and cfg.stage2.target_cutoff == 0.3

def test_resolve_rejects_unknown_key():
    with pytest.raises(ValueError):
        resolve_h2h_config({"system": "buddy", "k_target": 16, "bogus": 1})

def test_fixed_resolution_mode_is_buddy_only():
    with pytest.raises(ValueError):
        resolve_h2h_config({"system": "percept", "k_target": 0})
```

  Also a `test_h2h_trial.py` test for `run_h2h_trial` using monkeypatched `fit_buddy_stage1` / `fit_percept_stage1` / `stage2_metrics` / `stage1_metrics` fakes that returns a k_miss on seed 2 → `objective == -1.0` and both per-seed rows present (Review Focus 1), and one where a PercepT fake returns only 38 distinct train labels for k_target 40 → `n_topics == 38` (Review Focus 4). Add to `test_h2h_eval.py` a `stage2_metrics`-free test of the AUC skip path: call `stage2.evaluate_auc` via a helper `_auc_for(scores, labels, n_topics)` you add to `h2h_eval.py`, with one topic absent from `labels` → finite macro AUC and `skipped == 1` (Review Focus 2).

- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement.**
  - `leiden_on_graph`: igraph from upper-triangle edges (as `clustering.leiden_partition` does), `leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=seed)`, relabel 0..n-1.
  - `target_k_partition`: `log_lo, log_hi = log(lo), log(hi)`; start at `r = 1.0`; loop ≤ `max_steps`: labels = leiden; merged = `merge_small_communities(embedding, labels, merge_threshold)[0]`; K = n unique; if hit return; if K > target+tol: `log_hi = log(r)` else `log_lo = log(r)`; `r = exp((log_lo+log_hi)/2)`. Track the closest-K result for the miss return.
  - `eval_labels`: `targets.assign_to_train_communities(train_embedding, train_labels, heldout_embedding[subset_idx], EVAL_TRANSFER_K)`.
  - `stage1_metrics`: transfer AMIs via `ami_emotion_genre(eval_labels(...), heldout_emotion[subset_idx], heldout_genre[subset_idx])`; independent via `independent_partition(heldout_embedding[subset_idx], pilot, "heldout", seed, device)`; native if `native_subset` is given.
  - `stage2_metrics`: n_topics = `train_labels.max()+1`. Train targets: `single_label` → `targets.one_hot(train_labels, n)`; else `targets.build_targets(targets.cosine_vote_fractions(train_embedding, train_labels, train_embedding, n, k=s2cfg.train_target_k), n, float(cutoff))`. Seed torch/numpy/cuda, build `ParameterizedAttentionPoolingMapper(n_topics=n, num_queries, mlp_head, d_model=store.train_patches.shape[-1])`, `train_stage2(..., class_balanced=..., train_labels_for_weighting=train_labels, seed=seed)` (read `pipeline.run_trial` lines 105–150 and mirror it), scores on `store.heldout_patches[subset_idx]`, then for each label set `_auc_for(scores, labels, n)` = `evaluate_auc(scores, one_hot(labels, n))` → macro via `auc_summary`, and skipped count.
  - `run_h2h_trial`: subset indices from `split` (`"val"` or `"test"`); monitor indices = `split.val_idx` when `monitor == "val"`, `np.arange(N_heldout)` when `"all"`. Per seed: Stage 1 by system; buddy topics: `graph = build_topic_graph(train_embedding, cfg.leiden_graph, ...)`, then `target_k_partition` (or, if `k_target == 0`, `leiden_on_graph(graph, cfg.leiden_resolution, seed)` + merge); PercepT topics: `train_labels` from Stage 1. `n_topics` = distinct train labels. Then `stage1_metrics` and `stage2_metrics` with label sets `{"primary": eval_labels(...)}` plus `{"native": heldout_native[subset_idx]}` for PercepT. If `k_miss`, skip Stage 2 and set `auc_primary = nan` for that seed (objective becomes -1).
- [ ] **Step 4: Run test command; all pass.**
- [ ] **Step 5: Report to controller (no commit).**

---

### Task 5: V1 — Stage 2 equivalence on saved pilot topics (local GPU)

**Files:**
- Create: `src/test/20260930_matched_h2h/validate_v1_stage2.py`, `src/test/20260930_matched_h2h/test_validate_v1.py`

**Interfaces:**
- Consumes: `h2h_eval.stage2_metrics` path pieces (Task 4), `targets`, the pilots listed below.
- Produces: stdout lines `V1_RESULT {json}` for `buddy` and `percept`, and `src/test/20260930_matched_h2h/v1_results.json`.

- [ ] **Step 1: Read** `src/test/20260927_deep_stage_analysis/run_candidate4_fixed_stress.py` and `run_candidate4_rich_multilabel_pilot.py` (the §6f 0.8534 flow on `attention_h1_embedding_snapshot.npz`), and `run_percept_fixed_snapshot_pilot.py` (the 0.5925 flow; `percept_fixed_snapshot.npz` stores `heldout_stage2_targets`, `per_topic_auc`). Write down in the script docstring exactly which functions produce the pilots' train/held-out target matrices.
- [ ] **Step 2: Write a failing test** for the pure comparison helper:

```python
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_v1_stage2 import compare_targets

def test_compare_targets_exact_and_mismatch():
    a = np.eye(3, dtype=np.float32)
    assert compare_targets(a, a.copy()) == {"equal": True, "n_diff_rows": 0}
    b = a.copy(); b[1] = [0, 0, 1]
    assert compare_targets(a, b) == {"equal": False, "n_diff_rows": 1}
```

  `compare_targets` lives in `validate_v1_stage2.py`; heavy imports in that script must be lazy (inside functions) so the test imports it without data or a GPU.
- [ ] **Step 3: Implement** `validate_v1_stage2.py`:
  - **buddy:** compute the pilot's own train and held-out target matrices by importing the candidate-4 functions (never re-derive). Compute the harness's matrices from the same snapshot and the pilot's merged train labels with `targets.cosine_vote_fractions` / `build_targets` (cutoff 0.15, k=20) and `assign_to_train_communities` (k=20). Report `compare_targets` for both. Then train the harness mapper exactly as §6f (read the pilot for lr, epochs, class balancing, num_queries=1, linear head) at seeds 42/7/123/2024 on the **pilot's** targets and report mean macro AUC vs 0.8534.
  - **percept:** targets = the snapshot's own; compare the harness single-label/multi-hot construction from `train_topic`/`heldout_topic` against `heldout_stage2_targets`; train the harness mapper at lr 1e-3/100 epochs seed 42 → compare to the snapshot's macro AUC (0.5925); also report lr 1e-2/400 at seeds 42/7/123/2024 (informational; §6g's 0.9226 came from a different refit).
  - Print `V1_RESULT {json}` per system with `targets_equal`, `n_diff_rows`, `auc_mean`, `auc_std`, `reference`, `delta`, `pass` (`targets_equal and |delta| <= 0.003`).
- [ ] **Step 4: Run tests (all pass).** The controller runs the script itself on the local GPU.
- [ ] **Step 5: Report to controller (no commit).**

---

### Task 6: V2/V3 validation runner (DAS6)

**Files:**
- Create: `src/test/20260930_matched_h2h/validate_v2_v3.py`, `scripts/run_h2h_validate.sh`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `V2_RESULT {json}` (per seed: `ind_emo`, `ind_genre`, `ind_k`, `train_k`, `stop_reason`, `epochs_run`, `seconds`) and `V3_RESULT {json}` (native emo/genre AMI on all held-out, K, AUC native targets at lr 1e-3/100 and lr 1e-2/400 over seeds 42/7/123/2024, `stage1_seconds`), plus `TIMING {json}` lines (per-trial seconds for one default buddy-pilot, buddy-harness and PercepT `run_h2h_trial` on val with one seed — the controller uses these for R4 sizing).

- [ ] **Step 1: Implement** `validate_v2_v3.py --which v2|v3|timing --seeds 42,7,123,2024`:
  - v2: `fit_buddy_stage1(BuddyStage1Config(), store, seed, monitor_idx=np.arange(9365), pilot, device)`; `train_k` = distinct labels of `pilot.arch.detect_communities(build_topic_graph(train_emb, "pilot_repaired", ...), seed=seed)`; independent held-out labels via `independent_partition(heldout_emb, pilot, "heldout", seed, device)`; AMIs via `ami_emotion_genre` on all held-out.
  - v3: `fit_percept_stage1(PerceptStage1Config(), store, 42, 40, mods, device)` → native AMIs on all held-out; Stage 2 on native train labels with native held-out labels (single_label targets, num_queries 1, linear) at the two mapper settings.
  - timing: one `run_h2h_trial` per mode with default configs, `subset="val"`, `seeds=(1001,)`, printing `TIMING`.
- [ ] **Step 2:** `scripts/run_h2h_validate.sh <which> [seeds]` wrapper (Global Constraints), `bash -n`, `chmod +x`.
- [ ] **Step 3:** A unit test that `validate_v2_v3.py --help` parses and `--which bogus` exits non-zero (no data).
- [ ] **Step 4: Report to controller (no commit).**

---

### Task 7: In-process W&B agent and the four sweep configs

**Files:**
- Create: `scripts/buddy_percept_sweep/h2h_agent.py`, `scripts/h2h_sweeps/{buddy_k16,buddy_k40,percept_k16,percept_k40}.yaml`, `scripts/run_h2h_agent.sh`
- Test: `scripts/buddy_percept_sweep/tests/test_h2h_agent.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `python scripts/buddy_percept_sweep/h2h_agent.py --sweep <entity/project/id> [--count N]` runs `wandb.agent(sweep_id, function=_trial, project=..., entity=..., count=N)` in-process with one `H2HStore`, one split, one `PilotModules`, one PercepT module set per process. `_trial`: `wandb.init()`, `resolve_h2h_config(dict(wandb.config))`, `run_h2h_trial(cfg, store, split, "val", SEARCH_SEEDS, ...)`, `wandb.log` of every mean metric + `objective` + `split_digest` + per-seed rows flattened as `s{i}_<key>`; any exception → log `objective=-1.0`, `error=<repr>` and finish the run (never kill the agent). Free CUDA memory between runs (`torch.cuda.empty_cache()`).

- [ ] **Step 1: Write failing tests**: each YAML parses; `method: bayes`; `metric: {name: objective, goal: maximize}`; `run_cap` equals the value in a single constant `TRIALS_PER_CELL` written in all four files (initially 300; the controller edits it after timing); `system`/`k_target` are fixed values matching the file name; the Stage 2 parameter block is **identical** across all four files; every parameter name resolves through `resolve_h2h_config` (build a config dict from each parameter's first value / `min`). Test that `_trial`'s exception path logs `objective=-1.0` using a fake `wandb` module injected via `monkeypatch`.
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** the agent and YAMLs. Search spaces (spec §6, adapted): Stage 2 shared block — `mapper_lr` log_uniform_values 1e-3..3e-2; `mapper_epochs` [100, 200, 400, 800]; `num_queries` [1, 2, 4, 8]; `mlp_head` [linear, one_hidden]; `weight_decay_stage2` [0.0, 1e-5, 1e-4]; `class_balanced_loss` [false, true]; `target_cutoff` [single_label, 0.15, 0.3, 0.5]; `train_target_k` [10, 20, 40]. Buddy: `buddy_impl` [pilot, harness]; `buddy_heads` [mlp128, attn1, attn4] (harness maps attn1/attn4→attn with `buddy_num_heads` [1, 2, 4, 8]); `buddy_d_shared` [16, 32, 64, 128]; `buddy_lr` log_uniform 1e-4..1e-2; `buddy_batch_size` [512, 1024, 2048, 4096]; `buddy_content_pca_dim` [30, 50, 80, 120]; `buddy_temperature` [0.05, 0.1, 0.2]; `buddy_max_epochs` [100, 200, 400]; `buddy_plateau_window` [3, 5, 10]; `buddy_noise_std` [0, 0.05, 0.1, 0.2]; `buddy_lambda_affect` log_uniform 0.25..4; `buddy_weight_decay` [0, 1e-5, 1e-4]; `buddy_teacher_graph_K` [10, 15, 20, 30]; `leiden_graph` [mknn, pilot_repaired]; `merge_small_threshold` [0.0, 0.005, 0.01, 0.02]. PercepT: `percept_pretrain_epochs` [50, 100, 200]; `percept_pretrain_lr` log_uniform 3e-4..3e-3; `percept_dec_lr` log_uniform 3e-5..1e-3; `percept_lambda_balance` log_uniform 10..1e4; `percept_lambda_reconstruction` log_uniform 0.1..10; `percept_n_initial_factor` [1.0, 1.5, 2.0, 2.5]; `percept_stability_threshold` [0.001, 0.003]. W&B entity/project `polysemic/CoSiR-h2h`. `run_h2h_agent.sh <sweep_path> [count]` wrapper per Global Constraints plus `export WANDB_SILENT=true`.
  - Note: the heads mapping means `buddy_heads=attn4` under `impl=harness` still uses `buddy_num_heads`; under `impl=pilot`, `attn4` means the pilot's 4-head fusion and `buddy_num_heads` is ignored. Document this in the YAML header comment.
- [ ] **Step 4: Run test command; all pass.**
- [ ] **Step 5: Report to controller (no commit).**

---

### Task 8: Select, stress, test, summarize

**Files:**
- Create: `scripts/buddy_percept_sweep/h2h_select.py`, `scripts/run_h2h_select.sh`
- Test: `scripts/buddy_percept_sweep/tests/test_h2h_select.py`

**Interfaces:**
- Consumes: Tasks 1–4, 7.
- Produces CLI subcommands (model on `src/test/20260928_buddy_percept_sweep/run_top10_stress.py`: pure functions + thin CLI, heavy imports lazy, wandb only in `select`):
  - `select --sweep <path> --top 5 --out <json>`: finished runs with `objective > -1`, sorted desc, top N, each row `{rank, id, objective, config}` (config minus `_` keys).
  - `run --finalists <json> --ranks 1,2 --subset val|test --seeds 42,7,123,2024 --tag TEXT [--monitor val|all]`: prints `H2H_SEED {json}` per seed and `H2H_RESULT {json}` per rank (summary: `auc_mean`, `auc_std`, `auc_min`, `auc_max`, `ind_emo_mean`, `ind_genre_mean`, `transfer_emo_mean`, `transfer_genre_mean`, `n_topics` list, `k_miss_count`, and `auc_native_mean` when present).
  - `reference --which m8x7ifx4|percept_6g --subset test --seeds ...`: runs the two fixed reference configs (buddy harness impl with the §6i winner's config from `src/test/20260928_buddy_percept_sweep/finalists.json` rank 10, `k_target=0`; PercepT at the fixed-pilot defaults, `k_target=40`, `mapper_lr=1e-2`, `mapper_epochs=400`, `single_label`).
  - `summarize --in LOG... --out-md PATH`: markdown table grouped by `tag`, sorted by `auc_mean` desc within a tag, winner line per tag.
- [ ] **Step 1: Failing tests** for the pure functions: `select_top(runs, n)`, `summarize_rows(per_seed)` (nan-safe: k_miss rows excluded from AUC stats but counted), `parse_lines(paths, prefix)` (dedup on `(tag, rank, run_id)`, last wins), `render_markdown(summaries)`.
- [ ] **Step 2–4:** implement until green; `run_h2h_select.sh <subcommand args...>` wrapper.
- [ ] **Step 5: Report to controller (no commit).**

---

### Task 9 (controller): validate, size, sweep, select, test, report

Not dispatched. The controller:
1. Commits each reviewed task (`feat(h2h): cluster run: ...`).
2. Runs V1 locally; runs V2 (seeds 42/7/123/2024), V3 and timing on DAS6; compares V2 to QC2 `BASELINE_RESULT` lines; applies the spec §5 failure rule.
3. Sizes `TRIALS_PER_CELL` by R4 from `TIMING`, edits the four YAMLs, commits, creates the sweeps, and launches 9 agents with `cluster launch` (agent-to-cell assignment proportional to per-trial time so cells finish together).
4. Watches; when all four sweeps reach `run_cap`: `select` top 5 per cell, `run` stress on val, pick winners, `run` winners on test, `reference` on test.
5. Writes the full report `docs/reports/auto/percept/<date>_matched_percept_buddy_h2h.md`, master report §6j/§6k, updates memory, runs the final whole-branch review.
