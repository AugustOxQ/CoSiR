# CoSiR v2 Candidate A: affect-signal factor learning (cells E, SE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PercepT's GoEmotions affect signal as a source of training conditions for the factor encoders (cell E),
alone and mixed with the CLIP-image-cluster style source (cell SE). Pick a cell against the matched control C0 on
selection rows by a pre-registered emotion-gain rule, then confirm it once on fresh held episodes.

**Architecture:** Three small `src/` additions: a caption join, a GoEmotions probability extractor, and a multi-view
partition source. Everything else reuses the factor-learning 2×2's machinery: `train_factors` with the naive-rule
condition loss; `run_grid.py` for configs, gates and the loader; `run_posthoc.py` for diagnostics; the probe's
evaluation helpers. A new dated run folder holds prepare, smoke, runs, evaluation and replication. A second folder
holds the held test.

**Tech Stack:** Python 3.10 (conda env `CoSiR`), PyTorch, transformers 5.6 (local HF cache, offline), NumPy, SciPy,
scikit-learn 1.6, pytest; RTX 3090 locally (DAS6 via the `cluster-run` skill only if the timing smoke says heavy).

**Spec:** `docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-affect-factor-learning-design.md` (read §0 for every
term). Executors read the spec and this plan.

## Global Constraints

- Python: `/root/miniconda3/envs/CoSiR/bin/python`, run from `/project/CoSiR` (branch `main`). Tests:
  `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q <path>`; the full suite is `src/test` (227 tests before this plan).
- Seed 42 unless stated. No `cuml` / `cugraph`.
- **ArtELingo emotion and style labels are evaluation-only.** Diagnostic probes may be fit on them only to measure,
  never to train or choose.
- **The only external training signal is `SamLowe/roberta-base-go_emotions`**: 28 sigmoid probabilities per caption,
  loaded offline from the local HF cache.
- **Only scorer-train captions ever reach the affect model.** Val, selection and held captions never do.
- **Rows:** training, the affect extraction, the partitions and every fitted statistic use scorer-train rows; the
  selection rows are for choosing; held rows are read only by Task 5's `--run`, once. Val is never read.
- **No tuning:** R3_CONFIG weights, `painting_batches=True`, pair agreement, λ_condition 1.0, β 0.3, 64 episodes per
  step, 12 random negatives, 2,000 steps, k-means k=64, the −1.5 style margin, the 0.5 tie band and 4,096 selection
  episodes per label are all fixed.
- **Gates:** `evaluate_factor_gates` with `AMENDED_2026_09_29_THRESHOLDS`, fit = scorer-train, eval = selection,
  readout reference = `r0_readout_reference` in `src/test/20261016_factor_learning_grid/cache/grid_prepare.json`.
  Eight gates are **binding**: participation_ratio, redundancy, readout, dead, modality_private, usage_concentration,
  community_spanning, pair_retrieval. **sparsity** is reported only.
- At most **3** training processes share the GPU at once.
- Every modified `src/` file gets an entry in `.claude/20261018_log.md`: file path as header, before/after snippets,
  explanation. The file is gitignored: write it, never `git add` it.
- Run folders `src/test/20261018_affect_factor_learning/` and `src/test/20261019_affect_factor_learning_held/`, each with
  a `<folder>_log.md` and a `.gitignore` containing `*.npy *.npz *.json *.pt *.log cache/ checkpoints/ results/`.
- Reports:
  - `docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md` and
    `docs/reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md`.
  - Verdict first, in plain language. Every number sits next to its baseline: C0 for the criteria; naive on original
    R3 at β 0.3 as the current system.
  - Numbers carry analysis; figures in `docs/reports/assets/2026-10-18_affect_factor_learning/`, built by
    `docs/reports/assets/build_2026-10-18_affect_factor_learning_figures.py`; no em dashes.
  - One row in `docs/reports/reports_sum.md` plus the "Current work (v2)" line;
    `/root/miniconda3/envs/CoSiR/bin/python scripts/check_reports_sum.py` prints OK.
- **Another Claude session may have uncommitted edits in this tree** (e.g. `docs/reports/reports_sum.md`,
  `docs/reports/weekly/*`, `bin/`). Stage only the task's files, by explicit path.
  - For `reports_sum.md`, stage only your own lines: copy `git show HEAD:docs/reports/reports_sum.md`, apply your edit
    to the copy, run `git hash-object -w <copy>`, then
    `git update-index --cacheinfo 100644,<hash>,docs/reports/reports_sum.md`. Mirror the same edit in the working tree.
  - On `index.lock`, wait and retry.
- Commit locally, never push. Every commit message ends with a blank line, then:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP`
- **Stop points:**
  - Task 2: the timing smoke projects heavy runs. Use `cluster-run` if a DAS6 node is already reserved; otherwise ask
    the user to reserve one.
  - Task 3: C0 fails a binding gate, or no cell qualifies.
  - Any destructive action.

## Review Focus

- A caption row whose `caption` is missing, empty or not a string: `join_captions` must raise, not silently feed an
  empty string to the model. Pinned in Task 1 (`test_join_captions_rejects_missing_or_bad_captions`).
- The GoEmotions model is absent from the local cache (another machine, e.g. a DAS6 node): the extractor must fail with
  a clear error, and its tests must skip, not error. Pinned in Task 1 (the `loaded` fixture skip, and the
  `local_files_only` load).
- A multi-view source where one view has far fewer groups: each view must still get about half the conditions, not a
  share proportional to its group count. Pinned in Task 1
  (`test_multi_partition_source_draws_each_view_about_half_the_time`).
- A view whose groups are all smaller than `min_group_rows`: it must drop out, not crash sampling. Pinned in Task 1
  (`test_view_without_valid_groups_is_excluded`).
- A cell that fails only the sparsity gate must remain eligible. The rule must not count sparsity. Pinned in Task 3
  (`_check_affect_rule`'s sparsity-only case).

---

### Task 1: Caption join, GoEmotions extractor, multi-view partition source

**Files:**
- Modify: `src/data/artelingo.py` (add `join_captions` after `join_art_styles`)
- Create: `src/data/affect.py`
- Modify: `src/train/condition_sources.py` (add `MultiPartitionSource` after `CommunitySource`)
- Test: `src/test/test_artelingo_loader.py` (append), `src/test/test_affect.py` (new), `src/test/test_condition_sources.py` (append)

**Interfaces:**
- Consumes: `_rows_by_sample_id(sample_ids, annotations)` in `src/data/artelingo.py`; `_PartitionSource._setup`,
  `_condition`, `valid_keys`, `Condition` in `src/train/condition_sources.py`;
  `mine_condition_episodes(source, None, keys, n, rng, num_hard=0, num_random=12)`.
- Produces:
  - `join_captions(sample_ids: np.ndarray, annotations: list[dict]) -> np.ndarray` (object array of str, one per row)
  - `GOEMOTIONS_MODEL: str`, `GOEMOTIONS_NUM_LABELS = 28`,
    `load_goemotions(model_name=GOEMOTIONS_MODEL, device=None) -> (tokenizer, model)`,
    `goemotions_probabilities(texts, device=None, batch_size=256, max_length=64, model_name=GOEMOTIONS_MODEL, loaded=None) -> np.ndarray` of shape (n, 28), float32
  - `MultiPartitionSource(labels_by_view: dict[str, np.ndarray], rows, min_group_rows=200, max_tries=100)` with
    `.views: list[str]`, `.valid_keys`, `.sample_condition(rng) -> Condition`, `swap_capable = False`

- [ ] **Step 1: Write the failing caption-join tests** (append to `src/test/test_artelingo_loader.py`)

```python
from src.data.artelingo import join_captions  # noqa: E402


def test_join_captions_is_positional_by_sample_id():
    annotations = [{"caption": "calm sea", "painting": "p0"}, {"caption": "a dark storm", "painting": "p1"}]
    assert join_captions(np.array([1, 0]), annotations).tolist() == ["a dark storm", "calm sea"]


@pytest.mark.parametrize("bad", [{}, {"caption": ""}, {"caption": "   "}, {"caption": ["a list"]}, {"caption": None}])
def test_join_captions_rejects_missing_or_bad_captions(bad):
    with pytest.raises(ValueError, match="caption"):
        join_captions(np.array([0]), [bad])


def test_join_captions_rejects_bad_ids():
    with pytest.raises(ValueError):
        join_captions(np.array([0, 0]), [{"caption": "a"}, {"caption": "b"}])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_artelingo_loader.py -k captions`
Expected: FAIL with `ImportError: cannot import name 'join_captions'`.

- [ ] **Step 3: Implement `join_captions`** (in `src/data/artelingo.py`, after `join_art_styles`)

```python
def join_captions(sample_ids: np.ndarray, annotations: list[dict]) -> np.ndarray:
    """annotations[sample_id]["caption"] per feature row (the same positional join as join_annotations)."""
    rows = _rows_by_sample_id(sample_ids, annotations)
    if any(not isinstance(row.get("caption"), str) or not row["caption"].strip() for row in rows):
        raise ValueError("Every ArtELingo row needs a non-empty string caption")
    return np.asarray([row["caption"] for row in rows], dtype=object)
```

- [ ] **Step 4: Run the loader tests** — `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_artelingo_loader.py`. Expected: all PASS.

- [ ] **Step 5: Write the failing extractor tests** (`src/test/test_affect.py`)

```python
"""GoEmotions affect extraction (affect spec §4); skipped when the model is not in the local HF cache."""

import numpy as np
import pytest

from src.data.affect import GOEMOTIONS_NUM_LABELS, goemotions_probabilities, load_goemotions

TEXTS = ["I am so happy and grateful, this is wonderful!",
         "This is heartbreaking, I feel so sad and lonely.",
         "A bowl of pears on a wooden table."]


@pytest.fixture(scope="module")
def loaded():
    try:
        return load_goemotions(device="cpu")
    except OSError as err:                                  # not cached on this machine
        pytest.skip(f"GoEmotions model not in the local HF cache: {err}")


def test_shape_range_and_dtype(loaded):
    p = goemotions_probabilities(TEXTS, loaded=loaded)
    assert p.shape == (3, GOEMOTIONS_NUM_LABELS) and p.dtype == np.float32
    assert (p >= 0).all() and (p <= 1).all()


def test_joy_and_sadness_are_read_correctly(loaded):
    _, model = loaded
    index = {name: int(i) for i, name in model.config.id2label.items()}
    p = goemotions_probabilities(TEXTS, loaded=loaded)
    assert p[0, index["joy"]] > p[0, index["sadness"]]
    assert p[1, index["sadness"]] > p[1, index["joy"]]


def test_batch_size_does_not_change_the_output(loaded):
    one = goemotions_probabilities(TEXTS, loaded=loaded, batch_size=1)
    all_at_once = goemotions_probabilities(TEXTS, loaded=loaded, batch_size=3)
    assert np.allclose(one, all_at_once, atol=1e-5)


def test_empty_input_and_bad_arguments(loaded):
    assert goemotions_probabilities([], loaded=loaded).shape == (0, GOEMOTIONS_NUM_LABELS)
    with pytest.raises(ValueError, match="batch_size"):
        goemotions_probabilities(TEXTS, loaded=loaded, batch_size=0)
    with pytest.raises(TypeError, match="str"):
        goemotions_probabilities(["fine", 3], loaded=loaded)
```

- [ ] **Step 6: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_affect.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.affect'` (collection error).

- [ ] **Step 7: Implement `src/data/affect.py`**

```python
"""GoEmotions affect probabilities for captions.

The one external training signal the parent spec allows since its 2026-09-30 amendment: the RoBERTa model fine-tuned
on GoEmotions (Reddit comments, 28 emotion categories) that PercepT uses as its affect input. It was never trained on
ArtELingo. It is loaded offline from the local Hugging Face cache.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"
GOEMOTIONS_NUM_LABELS = 28


def load_goemotions(model_name: str = GOEMOTIONS_MODEL, device=None):
    """Tokenizer and eval-mode model from the local HF cache (``local_files_only``); raises OSError if not cached."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    if model.config.num_labels != GOEMOTIONS_NUM_LABELS:
        raise ValueError(f"{model_name} has {model.config.num_labels} labels, expected {GOEMOTIONS_NUM_LABELS}")
    return tokenizer, model.to(device or "cpu").eval()


def goemotions_probabilities(texts, device=None, batch_size: int = 256, max_length: int = 64,
                             model_name: str = GOEMOTIONS_MODEL, loaded=None) -> np.ndarray:
    """Sigmoid probabilities over the 28 GoEmotions labels, one float32 row per text, in input order.

    ``loaded`` is a ``load_goemotions`` result to reuse across calls; otherwise the model is loaded here.
    """
    texts = list(texts)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("every text must be a str")
    if not texts:
        return np.zeros((0, GOEMOTIONS_NUM_LABELS), dtype=np.float32)
    tokenizer, model = loaded if loaded is not None else load_goemotions(model_name, device)
    target = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(target)
            out.append(torch.sigmoid(model(**encoded).logits).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)
```

- [ ] **Step 8: Run the extractor tests** — `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_affect.py`. Expected: 4 PASS (the model is cached on this machine; `ls ~/.cache/huggingface/hub | grep go_emotions` shows it).

- [ ] **Step 9: Write the failing multi-view source tests** (append to `src/test/test_condition_sources.py`)

```python
from src.train.condition_episodes import mine_condition_episodes  # noqa: E402
from src.train.condition_sources import MultiPartitionSource  # noqa: E402


def test_multi_partition_source_draws_each_view_about_half_the_time():
    rows = np.arange(2000)
    src = MultiPartitionSource({"affect": rows % 4, "image": rows % 16}, rows, min_group_rows=50)
    rng = np.random.default_rng(0)
    views = [src.sample_condition(rng).key[0] for _ in range(4000)]
    assert src.views == ["affect", "image"]
    assert 0.46 < views.count("affect") / len(views) < 0.54      # uniform over keys would give 4/20 = 0.2


def test_multi_partition_conditions_and_outside_follow_their_view():
    rows = np.arange(2000)
    labels = {"affect": rows % 4, "image": (rows // 7) % 10}
    src = MultiPartitionSource(labels, rows, min_group_rows=50)
    assert len(set(src.valid_keys)) == len(src.valid_keys) == 14
    for key in src.valid_keys:
        cond = src._condition(key)
        view_labels = labels[key[0]]
        assert (view_labels[cond.inside] == key[1]).all() and (view_labels[cond.outside] != key[1]).all()
        assert len(np.union1d(cond.inside, cond.outside)) == 2000


def test_view_without_valid_groups_is_excluded():
    rows = np.arange(1000)
    src = MultiPartitionSource({"affect": rows % 4, "tiny": rows % 500}, rows, min_group_rows=100)
    assert src.views == ["affect"]
    assert all(src.sample_condition(np.random.default_rng(i)).key[0] == "affect" for i in range(20))


def test_multi_partition_source_feeds_the_miner_and_is_not_swap_capable():
    rows = np.arange(4000)
    keys = rows // 2
    src = MultiPartitionSource({"affect": rows % 5, "image": (rows // 3) % 6}, rows, min_group_rows=100)
    ep = mine_condition_episodes(src, None, keys, 20, np.random.default_rng(1), num_hard=0, num_random=12)
    assert ep.candidates.shape == (20, 16) and not src.swap_capable
```

If `numpy` is not imported at the top of `test_condition_sources.py`, add `import numpy as np`.

- [ ] **Step 10: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_sources.py -k multi_partition`
Expected: FAIL with `ImportError: cannot import name 'MultiPartitionSource'`.

- [ ] **Step 11: Implement `MultiPartitionSource`** (in `src/train/condition_sources.py`, after `CommunitySource`)

```python
class MultiPartitionSource(_PartitionSource):
    """Several partitions of the same rows at once (e.g. affect clusters and CLIP image clusters).

    A condition is one group of one view; its outside is the fit rows outside that group. ``sample_condition`` draws a
    view uniformly, then one of that view's valid groups uniformly, so each view supplies about the same share of
    conditions whatever its number of groups. A view with no valid group is dropped.
    """

    name = "multi_partition"

    def __init__(self, labels_by_view: dict, rows, min_group_rows=200, max_tries=100):
        if not labels_by_view:
            raise ValueError("labels_by_view needs at least one view")
        self._setup(dict(labels_by_view), rows, min_group_rows, max_tries)
        self._keys_by_view: dict = {}
        for key in self.valid_keys:
            self._keys_by_view.setdefault(key[0], []).append(key)
        self.views = sorted(self._keys_by_view)

    def sample_condition(self, rng) -> Condition:
        keys = self._keys_by_view[self.views[int(rng.integers(len(self.views)))]]
        return self._condition(keys[int(rng.integers(len(keys)))])
```

- [ ] **Step 12: Run the source tests, then the whole suite**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_sources.py` then `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test`
Expected: all PASS (227 + 15 new: 7 caption, 4 affect, 4 source).

- [ ] **Step 13: Change log and commit**

Append entries for `src/data/artelingo.py`, `src/data/affect.py` (new) and `src/train/condition_sources.py` to `.claude/20261018_log.md` (not staged).

```bash
git add src/data/artelingo.py src/data/affect.py src/train/condition_sources.py \
        src/test/test_artelingo_loader.py src/test/test_affect.py src/test/test_condition_sources.py
git commit -m "feat(v2): caption join, GoEmotions affect extractor, multi-view partition source

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP"
```

---

### Task 2: Run script: prepare (affect extraction, partition, diagnostics) and the timing smoke (STOP POINT: local vs DAS6)

**Files:**
- Create: `src/test/20261018_affect_factor_learning/run_affect.py`,
  `src/test/20261018_affect_factor_learning/20261018_affect_factor_learning_log.md`,
  `src/test/20261018_affect_factor_learning/.gitignore`

**Interfaces:**
- Consumes:
  - From Task 1: `join_captions`, `load_goemotions`, `goemotions_probabilities`, `MultiPartitionSource`.
  - From `src/test/20261016_factor_learning_grid/run_grid.py`, imported via importlib as `grid` (the pattern that file
    uses to import `run_final`): `grid.load_grid() -> (cache, prep, meta, graph)`, `grid.cell_config(cell, seed, steps)`,
    `grid.sha256_file`, `grid.fin`, `grid.sel`, `grid.probe`, `grid.CKPT`, `grid.FULL_STEPS`, `grid.SMOKE_STEPS`,
    `grid.HEAVY_*`, `grid.DEVICE`.
  - `load_artelingo`, `ANNOTATIONS_PATH`; `grouped_subsplit(groups, rows, second_fraction, seed)` from
    `src/data/splits.py`; `CommunitySource`; `train_factors`, `save_factor_checkpoint`, `load_factor_checkpoint`;
    `MiniBatchKMeans`, `adjusted_mutual_info_score`, `LogisticRegression`.
- Produces for Tasks 3-5:
  - `CELLS = ("E", "SE")`.
  - `affect_cell_config(cell, seed, steps) = grid.cell_config("S", seed, steps)` for E and SE (identical training
    config; only the source differs), and `grid.cell_config("C0", ...)` for C0.
  - `run_affect_cell(cell, seed, steps=2000, tag="", overwrite=False) -> dict` for cell in {"E", "SE", "C0"}. It
    writes `checkpoints/{cell}_seed{seed}{tag}.pt` and `results/history_{cell}_seed{seed}{tag}.json`, and refuses to
    overwrite a full checkpoint without `overwrite`.
  - `cache/affect_prepare.npz`: `affect_probs` (183,694 × 28 float32, scorer-train order) and `affect_local` (int64
    k-means labels).
  - `cache/affect_prepare.json`: SHA-256s of the affect array, the C0 and S reference checkpoints; diagnostics;
    timings.
  - `results/smoke_timing.json` with the `decision`.

- [ ] **Step 1: Write the script's constants, imports and prepare phase**

Module docstring as in `run_grid.py` (purpose, phases, how to run, the row-scope rule: only scorer-train captions
reach the affect model). Constants:

```python
SEED = 42
CELLS = ("E", "SE")
AFFECT_K = 64
KMEANS_SETTINGS = {"n_clusters": AFFECT_K, "random_state": SEED, "n_init": 3, "batch_size": 4096}
DIAG_HELDOUT_FRACTION = 0.2                       # painting-grouped split INSIDE scorer-train, diagnostic only
C0_REF = grid.CKPT / "C0_seed42.pt"               # the 2x2's matched control
S_REF = grid.CKPT / "S_seed42.pt"                 # the 2x2's style cell, reference row only
CACHE, CKPT, RESULTS = HERE / "cache", HERE / "checkpoints", HERE / "results"
```

`prepare()` does the following, in order:
1. `cache, prep, meta, _ = grid.load_grid()`; `data = load_artelingo()`; `st = cache["scorer_train"]`.
2. **Captions:** load the annotations JSON (`ANNOTATIONS_PATH`) and set
   `captions = join_captions(data.sample_ids[st], annotations)`. Assert `len(captions) == len(st)`. **Only these
   captions reach the model.**
3. **Extraction:** `loaded = load_goemotions(device=grid.DEVICE)`, then
   `affect = goemotions_probabilities(captions, loaded=loaded, batch_size=256, max_length=64)`. Assert the shape is
   (183,694, 28) and every value is finite and in [0, 1]. Log the time.
4. **Partition:** `affect_local = MiniBatchKMeans(**KMEANS_SETTINGS).fit_predict(affect)` on the raw vectors. Record the
   group sizes and how many groups have ≥ 200 rows.
5. **References:** SHA-256 of `C0_REF` and `S_REF`. Assert their stored configs equal `grid.cell_config("C0", 42)` and
   `grid.cell_config("S", 42)`.
6. **Diagnostics** (measured only; never used for a choice):
   - AMI of `affect_local` with ArtELingo emotion and art style on scorer-train rows. Put it next to the same AMIs for
     `prep["clip_image_local"]` and `cache["clip_caption"][st]`.
   - Split scorer-train by painting: `first, second = grouped_subsplit(cache["groups"], st, 0.2, seed=42)`.
   - On that split, fit `LogisticRegression(C=1.0, max_iter=1000)` on standardized inputs: affect-28 → emotion, and
     CLIP caption features → emotion. Report both accuracies on `second` with the majority-class accuracy.
7. Save `cache/affect_prepare.npz` and `cache/affect_prepare.json`. Log everything.

- [ ] **Step 2: Write `run_affect_cell` and the smoke phase**

```python
def affect_cell_config(cell: str, seed: int, steps: int = grid.FULL_STEPS):
    return grid.cell_config("C0" if cell == "C0" else "S", seed, steps)


def affect_source(cell: str, prep_grid: dict, affect_local: np.ndarray):
    n = len(affect_local)
    if cell == "E":
        return CommunitySource(affect_local, np.arange(n))
    if cell == "SE":
        return MultiPartitionSource({"affect": affect_local, "image": prep_grid["clip_image_local"]}, np.arange(n))
    if cell == "C0":
        return None
    raise ValueError(cell)
```

`run_affect_cell(cell, seed, steps, tag, overwrite)` mirrors `grid.run_cell`:
- the overwrite guard (full runs only);
- loads `grid.load_grid()`, `load_artelingo()` and `cache/affect_prepare.npz`;
- `config = affect_cell_config(cell, seed, steps)`; `source = affect_source(...)`;
- `train_factors(data.img_features[st], data.txt_features[st], graph, config, device=grid.DEVICE,
  group_ids=prep["local_groups"], condition_source=source, history=history)` with stdout redirected;
- finite check, peak GPU, checkpoint and history JSON (including `"source": cell` and the source's view list).

`smoke()` runs E and SE at `grid.SMOKE_STEPS` (10 and 60 steps, tags `_smoke10` / `_smoke60`) and computes the
per-step slope as `grid.smoke` does. It projects:
- the selection runs: E + SE;
- the replication runs: 2 × (the slower cell + C0), where C0's per-step time comes from
  `src/test/20261016_factor_learning_grid/results/smoke_timing.json`.

The `decision` is `"run_locally"` unless one run exceeds 45 min, the total exceeds 3 h, or the peak exceeds 20 GiB.
Write `results/smoke_timing.json`.

Add `main()` flags: `--prepare`, `--smoke`, `--run CELL --seed S [--overwrite]`, and `--evaluate`, `--replicate`,
`--tables`. The last three raise `NotImplementedError("Task 3")` for now.

- [ ] **Step 3: Run prepare** — `/root/miniconda3/envs/CoSiR/bin/python src/test/20261018_affect_factor_learning/run_affect.py --prepare 2>&1 | tee src/test/20261018_affect_factor_learning/run_prepare.log`. Expected: affect array (183,694, 28) finite in [0,1]; ≤ 64 groups; the AMI and probe diagnostics logged; the C0/S configs asserted.

- [ ] **Step 4: Run the smoke** — `... run_affect.py --smoke 2>&1 | tee .../run_smoke.log`. Expected: `results/smoke_timing.json` with a decision.

- [ ] **Step 5: STOP POINT: write the smoke table and the diagnostics into the log, and report to the controller.**
  - `run_locally`: Task 3 runs on the local RTX 3090.
  - Otherwise the controller checks `cluster-run` for a reserved DAS6 node. If there is one, the full runs go
    through the `cluster-run` skill; if not, the controller asks the user to reserve a node.

Smoke checkpoints are never evaluated.

- [ ] **Step 6: Commit**

```bash
git add src/test/20261018_affect_factor_learning/run_affect.py src/test/20261018_affect_factor_learning/.gitignore \
        src/test/20261018_affect_factor_learning/20261018_affect_factor_learning_log.md
git commit -m "feat(v2): affect factor-learning run script (GoEmotions extraction, affect partition, diagnostics, timing smoke)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP"
```

---

### Task 3: Runs E and SE, selection evaluation, selection report (STOP POINT: C0 gate / no qualifying cell)

**Files:**
- Modify: `src/test/20261018_affect_factor_learning/run_affect.py` (add `evaluate`, `tables`, the rule),
  `src/test/20261018_affect_factor_learning/20261018_affect_factor_learning_log.md`
- Create: `docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md`,
  `docs/reports/assets/build_2026-10-18_affect_factor_learning_figures.py`,
  `docs/reports/assets/2026-10-18_affect_factor_learning/*.png`
- Modify: `docs/reports/reports_sum.md` (own lines only)

**Interfaces:**
- Consumes (Task 2): `run_affect_cell`, `affect_cell_config`, `cache/affect_prepare.*`.
- Consumes (`grid`):
  - `grid.gate_report(img_fit, txt_fit, img_eval, txt_eval, data, cache, community_local, reference)`;
  - `grid.sel.masked`, `grid.fin.SELECTION_SHA256`, `grid._row_mask`;
  - `grid.probe`: `r1_points`, `r1_with_ci`, `r1_diff` (paired, per-episode, points with `"point"` and `"ci95"`),
    `fixed_weight_ranks(img, txt, ic, tc, episodes, weights, beta) -> (ranks, ties)`,
    `oracle_ranks(img, txt, ic, tc, episodes, beta, steps, device, targets=None)`, `BETAS`, `CHANCE_R1`.
- Consumes (`src/test/20261016_factor_learning_grid/run_posthoc.py` via importlib): `probe_analyses`, `ami_analysis`,
  `balance_matched`-style logic, `per_target`. Adapt calls to this folder's models; where a function hard-codes the
  2×2's models, reimplement the same computation here rather than editing that file.
- Also: `label_episode_weights`, `standard_label_episodes`, `label_episodes_sha256`.
- Produces for Tasks 4-5:
  - `results/selection_results.json` (`rule.picked`, the `d_emo` / `d_style` blocks, gates) and
    `results/selection_ranks.npz`.
  - `apply_affect_rule(gates_passed, d_emo, d_style) -> dict` with keys `stop`, `picked`, `qualifying`, `tie_band`,
    `binding_ok`.

- [ ] **Step 1: Train E and SE (seed 42) as two parallel processes**

```bash
cd /project/CoSiR
for c in E SE; do
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261018_affect_factor_learning/run_affect.py --run $c --seed 42 \
    > src/test/20261018_affect_factor_learning/run_${c}_seed42.log 2>&1 &
done
wait
```

Expected: two checkpoints, finite histories. If a process dies at start-up (e.g. out of memory), rerun it alone with the
same command and settings, and log it.

- [ ] **Step 2: Implement the rule with its hand checks**

```python
BINDING_GATES = ("participation_ratio", "redundancy", "readout", "dead", "modality_private",
                 "usage_concentration", "community_spanning", "pair_retrieval")      # sparsity: reported only
STYLE_MARGIN, TIE_POINTS = -1.5, 0.5
CHANGES = {"E": 1, "SE": 2}


def apply_affect_rule(gates_passed: dict, d_emo: dict, d_style: dict) -> dict:
    """Affect spec §6. Eligible = the 8 binding gates pass (sparsity is not counted); qualifies = eligible and
    D_emo lower bound > 0 and D_style lower bound > -1.5; pick = highest D_emo, cells within 0.5 tie, a tie goes to
    E (fewer changes), then to the higher D_emo. ``gates_passed`` maps model -> {gate: bool}; ``d_emo`` / ``d_style``
    map cell -> {"point": R@1 points, "ci95": [lo, hi]}."""
    binding_ok = {m: all(bool(g[name]) for name in BINDING_GATES) for m, g in gates_passed.items()}
    if not binding_ok["C0"]:
        return {"stop": "C0 fails a binding gate: the setup is broken", "picked": None, "qualifying": [],
                "tie_band": [], "binding_ok": binding_ok}
    qualifying = [c for c in CELLS if binding_ok[c] and d_emo[c]["ci95"][0] > 0
                  and d_style[c]["ci95"][0] > STYLE_MARGIN]
    if not qualifying:
        return {"stop": "no cell qualifies", "picked": None, "qualifying": [], "tie_band": [],
                "binding_ok": binding_ok}
    best = max(d_emo[c]["point"] for c in qualifying)
    tied = [c for c in qualifying if d_emo[c]["point"] >= best - TIE_POINTS]
    picked = min(tied, key=lambda c: (CHANGES[c], -d_emo[c]["point"]))
    return {"stop": None, "picked": picked, "qualifying": qualifying, "tie_band": tied, "binding_ok": binding_ok}


def _check_affect_rule() -> None:
    names = BINDING_GATES + ("sparsity",)
    ok = {m: {n: True for n in names} for m in ("C0", "E", "SE")}
    blk = lambda p, lo: {"point": p, "ci95": [lo, p + 1.0]}                        # noqa: E731
    style_fine = {c: blk(0.0, -0.8) for c in CELLS}
    bad_c0 = {**ok, "C0": {**ok["C0"], "readout": False}}
    assert apply_affect_rule(bad_c0, {c: blk(2, 1) for c in CELLS}, style_fine)["stop"].startswith("C0")
    sparse_only = {**ok, "E": {**ok["E"], "sparsity": False}}                     # sparsity is not binding
    assert apply_affect_rule(sparse_only, {"E": blk(1.0, 0.2), "SE": blk(0.1, -0.5)}, style_fine)["picked"] == "E"
    assert apply_affect_rule(ok, {c: blk(0.5, -0.1) for c in CELLS}, style_fine)["stop"] == "no cell qualifies"
    assert apply_affect_rule(ok, {"E": blk(1.0, 0.2), "SE": blk(1.4, 0.5)}, style_fine)["picked"] == "E"    # tie band
    assert apply_affect_rule(ok, {"E": blk(1.0, 0.2), "SE": blk(1.8, 0.9)}, style_fine)["picked"] == "SE"
    assert apply_affect_rule(ok, {"E": blk(1.0, 0.2), "SE": blk(1.8, 0.9)},
                             {**style_fine, "SE": blk(-1.0, -2.0)})["picked"] == "E"                     # style guard
```

- [ ] **Step 3: Implement `evaluate()`**

In order:
1. `_check_affect_rule()`.
2. Load `data`, `grid.load_grid()` and the affect prepare cache.
3. **Episodes:** `standard_label_episodes(data, cache["groups"], cache["selection"], label, 4096, seed=42)` per label.
   - Assert the SHA-256 of the first 2,048 (a `LabelEpisodes` built from each field's `[:2048]`) equals
     `grid.fin.SELECTION_SHA256[label]`.
   - Assert every episode row is a selection row.
   - Null targets: `np.random.default_rng(42).integers(1, 13, 4096)` per label, in (emotion, art_style) order.
4. CLIP features masked to selection rows (`grid.sel.masked`).
5. **Models:**
   - `"R3"`: the stage-(d) cache codes, as in `grid.model_codes("R3")`;
   - `"C0"`: `C0_REF`;
   - `"S"`: `S_REF`, reference only;
   - `"E"`, `"SE"`: this folder's `*_seed42.pt`.
   - Load each checkpoint with `load_factor_checkpoint` and assert its config (C0: `grid.cell_config("C0", 42)`; S, E,
     SE: `grid.cell_config("S", 42)`).
   - Encode scorer-train and selection rows only; NaN elsewhere (assert the row scope).
6. **Per model:**
   - gates (`grid.gate_report` with `community_local = cache["community"][st]` and
     `reference = meta["r0_readout_reference"]`);
   - naive ranks at every β in `grid.probe.BETAS`;
   - label oracle at β 0.3 and 0, and its null at β 0.
7. CLIP-only ranks.
8. **Scores:**
   - `d_emo[X] = r1_diff(naive[X]@0.3, naive[C0]@0.3)["emotion"]["mean"]`;
   - `d_style[X] = ...["art_style"]["mean"]`, for X in E, SE;
   - also pooled, vs original R3, and S vs C0 for reference.
9. `rule = apply_affect_rule({m: gates[m].passed for m in ("C0", "E", "SE")}, d_emo, d_style)`.
10. **Reported extras** (context, never gating): the measured style SE and the guard's power at true 0 and −0.5 (normal
    approximation); the balance-matched-β comparison of E and SE vs C0; per-modality code probes and the
    within-painting caption-residual emotion probe; the AMI of each model's argmax factor with emotion, style, the CLIP
    image clusters and the affect clusters; the per-target breakdown of E/SE − C0; the condition loss and τ histories.
11. Save `results/selection_results.json` and `results/selection_ranks.npz`, and log the verdict.

`--tables` reprints: gates (binding and sparsity) × models, the `D_emo` / `D_style` / pooled table with CIs, naive R@1
per scope and direction, the β grid, the oracle with its null, the extras, and the rule outcome.

- [ ] **Step 4: Run the evaluation** — `... run_affect.py --evaluate 2>&1 | tee .../run_evaluate.log`. Expected: asserts pass; the rule outcome is printed.

- [ ] **Step 5: STOP POINT check.** If `rule["stop"]` is not None, finish Steps 6-7 with the stop verdict, then stop and report; Tasks 4-5 are not run.

- [ ] **Step 6: Selection report and figures.**

The report goes in `docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md`, verdict first.
It covers:
- the rule outcome, plus `D_emo` and `D_style` for E and SE with CIs, against C0;
- plain R@1 next to C0, original R3 and S;
- the gates (binding, with sparsity reported);
- the §4 diagnostics (how well GoEmotions lines up with ArtELingo emotion);
- the label oracle;
- per direction and per target;
- the β grid and the balance-matched comparison;
- the code probes and AMI;
- the guard's power;
- the spec §11 caveats;
- the smoke timing and where the runs ran.

Build the figures with `docs/reports/assets/build_2026-10-18_affect_factor_learning_figures.py` from
`results/selection_results.json`:
1. naive R@1 per model and label with CIs, and the C0 / R3 / CLIP-only / chance lines;
2. `D_emo` and `D_style` per cell with CIs and the 0 and −1.5 reference lines.

Use slots `#2a78d6`, `#eb6834` and `#1baf7a` on white, and look at each PNG for label collisions. Add the
reports_sum row (`10-18`) and the "Current work (v2)" line, staging your own lines only. `check_reports_sum.py`
must print OK.

- [ ] **Step 7: Commit**

```bash
git add src/test/20261018_affect_factor_learning/run_affect.py \
        src/test/20261018_affect_factor_learning/20261018_affect_factor_learning_log.md \
        docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md \
        docs/reports/assets/build_2026-10-18_affect_factor_learning_figures.py docs/reports/assets/2026-10-18_affect_factor_learning/
# plus reports_sum.md own lines via git update-index (Global Constraints)
git commit -m "docs(v2): affect factor-learning selection (E / SE vs C0 on selection rows)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01M7g4w8Dpf9KCGqu4aCBfwP"
```

---

### Task 4: Replication seeds 43 and 44 (picked cell and C0)

**Files:**
- Modify: `run_affect.py` (implement `--replicate`), the folder log, and the selection report (a Replication section)

**Interfaces:**
- Consumes: `rule.picked` from `results/selection_results.json`; `run_affect_cell`; the evaluation code path.
- Produces: `checkpoints/{picked,C0}_seed{43,44}.pt`, `results/replication.json` (per seed: `d_emo`, `d_style`,
  pooled, gates for both models).

- [ ] **Step 1:** Train `run_affect_cell(picked, s)` and `run_affect_cell("C0", s)` for s in (43, 44) in this folder. That is four runs, at most three at a time.
- [ ] **Step 2:** Implement `--replicate`. It evaluates the four checkpoints on the same 4,096 + 4,096 selection episodes with the same code path: gates, naive at β 0.3, and picked seed s against C0 seed s. It writes `results/replication.json`. Run it. Expected: finite results; reported only (the verdict rests on seed 42).
- [ ] **Step 3:** Add the Replication section to the selection report (a table per seed) and commit, with message `docs(v2): affect factor-learning replication seeds 43/44 (picked cell and C0)` and the trailers.

---

### Task 5: Held test (power, smoke, one run) and the held report

**Files:**
- Create: `src/test/20261019_affect_factor_learning_held/run_held.py`, its log and `.gitignore`,
  `docs/reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md`
- Modify: the figure script (one held figure), `docs/reports/reports_sum.md` (own lines only)

**Interfaces:**
- Consumes: `selection_results.json` (picked cell and its `d_emo` block), `replication.json`, the checkpoints (picked
  cell seeds 42/43/44 from this plan's folder, C0 seed 42 = `C0_REF`, C0 seeds 43/44 from this plan's folder), and the
  original R3 checkpoint `src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt` (SHA-256
  `1c299fc008ca006ffd2de37315ac3557523bf06e0b755d4dcbf70b999b4e453f`).
- Produces: `results/power.json`, `results/smoke_held.json` (numbers discarded), `results/held_results.json` (written
  once).

- [ ] **Step 1: Power (`--power`)**

```python
from scipy.stats import norm

N_CHOICES, SHRINK, TARGET_POWER, N_SELECTION = (2048, 4096, 8192), 0.75, 0.8, 4096


def held_episode_count(d_point: float, d_ci95: list[float]) -> tuple[int, dict]:
    """Affect spec §7: SE from the selection D_emo CI (4,096 episodes), effect shrunk by 0.75, smallest n per label
    with power >= 0.8, else 8,192."""
    se = (d_ci95[1] - d_ci95[0]) / (2 * 1.959964)
    effect = SHRINK * d_point
    table = {n: float(norm.cdf(effect / (se * (N_SELECTION / n) ** 0.5) - 1.959964)) for n in N_CHOICES}
    chosen = next((n for n in N_CHOICES if table[n] >= TARGET_POWER), N_CHOICES[-1])
    return chosen, table
```

Hand check, asserted in the script: `d_point=2.0` and `d_ci95=[1.0, 3.0]` give SE 0.5102 and effect 1.5. Power is 0.547
at 2,048 and 0.836 at 4,096, so `chosen == 4096`. Write `results/power.json` before any held read.

- [ ] **Step 2: Smoke (`--smoke`)** runs the same path as `--run` with the selection rows in place of the held rows (seed 43, the chosen n). It writes `results/smoke_held.json`, whose numbers are discarded. Confirm it runs end to end with finite outputs.

- [ ] **Step 3: The held run (`--run`, once; refuses if `results/held_results.json` exists).**
  - Recompute `grouped_split(leakage_groups(...), seed=42)` and assert it equals the stage-(d) cache.
  - Encode only the held rows (picked cell and C0 at seeds 42/43/44, and original R3 with its SHA asserted), and mask
    the CLIP features to held rows. Affect vectors are never computed for held rows.
  - Episodes: `standard_label_episodes(data, groups, held, label, n, seed=43)`. Record their SHA-256s and assert every
    row is a held row.
  - Compute:
    - naive ranks at β 0.3 for all models, and CLIP-only;
    - the label oracle and its null for the picked cell, C0 and R3;
    - `D_emo,held` and `D_style,held` (picked vs C0, seed 42), plus pooled;
    - the seeds 43/44 differences and the context rows vs original R3.
  - The verdict is **"confirmed"** iff the `D_emo,held` lower bound is > 0 and the `D_style,held` lower bound is > −1.5.
  - Write `results/held_results.json` and `results/held_ranks.npz`.

- [ ] **Step 4: Held report, figure, index, commit.**
  - Report `docs/reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md`, verdict first. It covers
    confirmed or not; plain R@1 of the picked cell, C0 and original R3; CLIP-only; per label and direction; the oracle;
    the seeds; the power table; the held-row disclosure (spec §7); and the caveats.
  - Add one held figure.
  - Add the reports_sum row (`10-19`) and the Current-work line, staging your own lines only; `check_reports_sum.py`
    must print OK.
  - Commit with `docs(v2): affect factor-learning held test (picked cell vs C0 on fresh seed-43 held episodes)` and
    the trailers.

---

## After the last task (controller)

- An Opus whole-branch final review, then ONE fix wave. Held rows are never re-read in the fix wave. Post-hoc
  selection-row diagnostics are allowed if the review asks for them.
- Update the memory file `project_next-step-factor-learning.md` with the outcome and the next decision for the user.
