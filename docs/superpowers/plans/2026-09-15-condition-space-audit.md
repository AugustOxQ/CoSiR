# Condition-Space Steerability Audit (Experiment 17.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cheap, training-free audit that tests whether an interpretable semantic axis (emotional/tonal or registerial framing) is linearly present in raw CLIP space and/or already leaking into existing trained buddy-condition vectors, using only proxy labels already in the data (RedCaps subreddit, Impressions `caption_type`), gated against a matched random-prompt/random-relabeling control per axis — producing the positive/null/partial verdict that scopes Experiment 17.2.

**Architecture:** Two independent, pure-function libraries (`axis_definitions.py` for label-group extraction, `text_anchor.py` for CLIP-text-anchor direction construction) feed two orchestration scripts (`raw_clip_audit.py` for the raw-CLIP sanity check, `checkpoint_probe.py` for the existing-checkpoint probe). Both scripts write JSON result tables; a final aggregation step applies the spec's decision rule and writes the report. No training, no new model code — CLIP's frozen text tower is loaded read-only via the raw-HF-submodule pattern this project already uses for feature extraction (`src/test/20260623_redcaps_buddy/extract_features.py`), never via `CoSiRModel` (broken local `llvmlite`/`cuml` import chain).

**Tech Stack:** Python 3.10, numpy, torch, `transformers` (`AutoModel`/`AutoTokenizer`, `openai/clip-vit-base-patch32`), scikit-learn (`LogisticRegression`, `StratifiedKFold`, `roc_auc_score`), pytest; conda env `CoSiR`; existing `src.utils.FeatureManager`.

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4, Experiment 17.1.

**Execution mode:** Per this project's established convention (see `.ccg/tasks/buddy-k-scaling-stage-a/plan.md` for the pattern used on Experiment 16.1), prefer routing each task's implementation through Codex as a subagent, with Claude reviewing the diff and running verification directly. This is CPU-light, training-free work — no GPU-job caution needed, and Codex may run the scripts itself as part of its own verification, with Claude spot-checking final artifacts.

## Global Constraints

- Conda env for every command: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR`.
- New code folder: `src/test/20260915_condition_space_audit/` (repo dated-folder convention).
- Never `import src.model` or `CoSiRModel` — pulls in `src.model.clustering` → `cuml`/`cudf` → broken local `libllvmlite.so`. Use the raw HF CLIP submodules (`model.text_model(**x).pooler_output` → `model.text_projection(...)`), exactly as `src/test/20260623_redcaps_buddy/extract_features.py` already does.
- Datasets: RedCaps-150k (`/data/SSD2/pre_extract/redcaps_150k/features`, `/data/PDD/redcaps/redcaps_plus/redcaps_150k.json`) and Impressions (`/data/SSD2/pre_extract/impressions/features`, `/project/Impressions/metadata/impressions_train.json`) — both feature stores already exist, no extraction needed.
- Sample-id join is **positional**: `records = [meta[s] for s in sample_ids]` where `sample_ids` comes from `FeatureManager.load_all_to_ram(...)["sample_ids"]` (or, for a checkpoint, `final_embeddings/sample_ids.npy`) — same pattern as `src/test/20260623_redcaps_buddy/redcaps_buddy.py::load_data`. Never assume `sample_id == list index` without this join.
- No GPU training. CLIP text-tower encoding of a handful of short prompts runs fine on CPU (`device="cpu"` default throughout this plan) — do not add CUDA-only code paths.
- Every control comparison (direction-based and probe-based) uses `seed=42` for reproducibility, matching this project's existing convention (e.g. `2026-07-16-buddy-cross-vlm-survival.md`'s `seed=42` default).
- Artifacts dir for the final report: `docs/reports/2026-09-15_condition_space_audit.md` (this project's report-naming convention).

---

### Task 1: Scaffold folder and discover candidate condition-vector checkpoints

**Files:**
- Create: `src/test/20260915_condition_space_audit/20260915_condition_space_audit_log.md` (stub)
- Create: `src/test/20260915_condition_space_audit/discover_checkpoints.py`
- Test: `src/test/20260915_condition_space_audit/test_discover_checkpoints.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `find_candidate_checkpoints(res_globs: list[str]) -> list[str]` (list of experiment directory paths); `src/test/20260915_condition_space_audit/checkpoints.json` (`{"redcaps_150k": [...], "impressions": [...]}`), consumed by Task 5.

- [ ] **Step 1: Create the folder and log stub**

Create `src/test/20260915_condition_space_audit/20260915_condition_space_audit_log.md`:

```markdown
# Condition-Space Steerability Audit (Exp. 17.1) — Log

Spec: docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md (Experiment 17.1)
Plan: docs/superpowers/plans/2026-09-15-condition-space-audit.md

## Checkpoint discovery
(to be filled: which candidate checkpoints were found per dataset)

## Results
(to be filled after the real run)
```

- [ ] **Step 2: Write the failing test for checkpoint discovery**

Create `src/test/20260915_condition_space_audit/test_discover_checkpoints.py`:

```python
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import discover_checkpoints as dc


def _write_fake_checkpoint(root, combine_side, conditioning_mode, init_strategy, with_embeddings=True):
    exp_dir = tempfile.mkdtemp(dir=root)
    cfg = {
        "model": {"combine_side": combine_side, "conditioning_mode": conditioning_mode},
        "train": {"initialization_strategy": init_strategy},
    }
    with open(os.path.join(exp_dir, "experiment_metadata.json"), "w") as f:
        json.dump({"config": repr(cfg)}, f)
    if with_embeddings:
        emb_dir = os.path.join(exp_dir, "final_embeddings")
        os.makedirs(emb_dir)
        open(os.path.join(emb_dir, "embeddings.npy"), "wb").close()
    return exp_dir


def test_finds_matching_asymmetric_buddy_checkpoint():
    with tempfile.TemporaryDirectory() as root:
        good = _write_fake_checkpoint(root, "img", "asymmetric", "buddies")
        found = dc.find_candidate_checkpoints([os.path.join(root, "*")])
        assert found == [good]


def test_rejects_symmetric_and_wrong_init():
    with tempfile.TemporaryDirectory() as root:
        _write_fake_checkpoint(root, "img", "symmetric_shared", "buddies")
        _write_fake_checkpoint(root, "img", "asymmetric", "imgtxt")
        _write_fake_checkpoint(root, "txt", "asymmetric", "buddies")
        found = dc.find_candidate_checkpoints([os.path.join(root, "*")])
        assert found == []


def test_skips_dir_missing_embeddings():
    with tempfile.TemporaryDirectory() as root:
        _write_fake_checkpoint(root, "img", "asymmetric", "buddies", with_embeddings=False)
        found = dc.find_candidate_checkpoints([os.path.join(root, "*")])
        assert found == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_discover_checkpoints.py -v`
Expected: FAIL / ERROR with "No module named 'discover_checkpoints'" (file doesn't exist yet).

- [ ] **Step 4: Implement `discover_checkpoints.py`**

Create `src/test/20260915_condition_space_audit/discover_checkpoints.py`:

```python
"""Find completed CoSiR training runs whose condition-vector checkpoints match
the current default architecture (asymmetric, combine_side='img', buddy-init) —
candidates for Experiment 17.1(b)'s existing-checkpoint probe.
"""
import ast
import glob
import json
import os


def find_candidate_checkpoints(res_globs):
    candidates = []
    for pattern in res_globs:
        for exp_dir in sorted(glob.glob(pattern)):
            meta_path = os.path.join(exp_dir, "experiment_metadata.json")
            emb_path = os.path.join(exp_dir, "final_embeddings", "embeddings.npy")
            if not (os.path.isfile(meta_path) and os.path.isfile(emb_path)):
                continue
            meta = json.load(open(meta_path))
            try:
                cfg = ast.literal_eval(meta["config"])
            except (KeyError, ValueError, SyntaxError):
                continue
            model_cfg = cfg.get("model", {})
            train_cfg = cfg.get("train", {})
            if (
                model_cfg.get("combine_side") == "img"
                and model_cfg.get("conditioning_mode", "asymmetric") == "asymmetric"
                and train_cfg.get("initialization_strategy") == "buddies"
            ):
                candidates.append(exp_dir)
    return candidates


REDCAPS_150K_GLOBS = [
    "res/CoSiR_init_ablation/redcaps_150k/*_CoSiR_Experiment",
    "res/CoSiR_condition_freeze_ablation/redcaps_150k/*_CoSiR_Experiment",
]
IMPRESSIONS_GLOBS = [
    "res/CoSiR_init_ablation/impressions/*_CoSiR_Experiment",
]


def main():
    result = {
        "redcaps_150k": find_candidate_checkpoints(REDCAPS_150K_GLOBS),
        "impressions": find_candidate_checkpoints(IMPRESSIONS_GLOBS),
    }
    out_path = os.path.join(os.path.dirname(__file__), "checkpoints.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"redcaps_150k: {len(result['redcaps_150k'])} candidates")
    print(f"impressions: {len(result['impressions'])} candidates")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_discover_checkpoints.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run discovery against the real repo and record counts in the log**

Run (from repo root `/project/CoSiR`):
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260915_condition_space_audit/discover_checkpoints.py
```
Expected: prints nonzero counts for both `redcaps_150k` and `impressions` (both C5's `init_ablation` folders are known to contain completed buddy-init runs for both datasets — confirmed present during spec research, 2026-09-15). If either count is 0, stop and report — Task 5 cannot proceed for that dataset without a real fallback glob. Append the printed counts to the log's "Checkpoint discovery" section.

- [ ] **Step 7: Commit**

```bash
git add src/test/20260915_condition_space_audit/
git commit -m "feat(exp17.1): scaffold condition-space audit, discover candidate checkpoints"
```

---

### Task 2: Axis definitions — concrete label-group extraction for both datasets

**Files:**
- Create: `src/test/20260915_condition_space_audit/axis_definitions.py`
- Test: `src/test/20260915_condition_space_audit/test_axis_definitions.py`

**Interfaces:**
- Consumes: nothing (pure functions over `records: list[dict]`).
- Produces:
  - `REDCAPS_AXES: dict[str, dict]`, `IMPRESSIONS_AXES: dict[str, dict]` — each axis entry has `description`, `prompts_a`, `prompts_b`, plus dataset-specific pole-selection fields.
  - `redcaps_binary_labels(records: list[dict], axis_name: str) -> tuple[np.ndarray, np.ndarray]` — `(keep_idx, labels)`, `labels[i] in {0,1}` for `pole_a`/`pole_b`.
  - `impressions_binary_labels(records: list[dict], axis_name: str) -> tuple[np.ndarray, np.ndarray]` — same shape, keyed on `caption_type`.

- [ ] **Step 1: Write the failing tests**

Create `src/test/20260915_condition_space_audit/test_axis_definitions.py`:

```python
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import axis_definitions as ax


def test_redcaps_warmth_labels():
    records = [
        {"image": "redcaps/images2020/cats/a.jpg"},
        {"image": "redcaps/images2020/mildlyinteresting/b.jpg"},
        {"image": "redcaps/images2020/gardening/c.jpg"},  # neither pole — excluded
        {"image": "redcaps/images2020/rarepuppers/d.jpg"},
    ]
    keep, labels = ax.redcaps_binary_labels(records, "warmth")
    assert list(keep) == [0, 1, 3]
    assert list(labels) == [1, 0, 1]


def test_impressions_aesthetic_vs_description_labels():
    records = [
        {"caption_type": "aesthetic"},
        {"caption_type": "description"},
        {"caption_type": "impression"},  # neither pole — excluded
        {"caption_type": "aesthetic"},
    ]
    keep, labels = ax.impressions_binary_labels(records, "aesthetic_vs_description")
    assert list(keep) == [0, 1, 3]
    assert list(labels) == [1, 0, 1]


def test_all_axes_have_prompts():
    for axes in (ax.REDCAPS_AXES, ax.IMPRESSIONS_AXES):
        for name, spec in axes.items():
            assert spec["prompts_a"], name
            assert spec["prompts_b"], name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_axis_definitions.py -v`
Expected: FAIL with "No module named 'axis_definitions'".

- [ ] **Step 3: Implement `axis_definitions.py`**

Create `src/test/20260915_condition_space_audit/axis_definitions.py`:

```python
"""Concrete two-pole label-group and prompt-pair definitions for the Exp. 17.1
condition-space audit. Subreddit/caption_type pools were chosen from labels
verified as well-populated in redcaps_150k / impressions_train during spec
research (2026-09-15) — see docs/reports/2026-09-15_condition_space_audit.md.
"""
import numpy as np

REDCAPS_AXES = {
    "warmth": {
        "description": (
            "companion-animal/cute subreddits vs. curiosity-framed subreddits "
            "— proxy for positive emotional valence"
        ),
        "pole_a": ["cats", "rarepuppers", "blackcats", "dogpictures", "pitbulls", "guineapigs", "eyebleach"],
        "pole_b": ["mildlyinteresting", "interestingasfuck", "natureisfuckinglit"],
        "prompts_a": [
            "a photo of a cute, happy animal",
            "an adorable pet photo",
            "a heartwarming picture of a beloved animal",
        ],
        "prompts_b": [
            "a photo of a surprising, curious object",
            "an unusual and interesting scene",
            "a strange or unexpected sight",
        ],
    },
    "register": {
        "description": (
            "'porn'-tagged aesthetic-photography subreddits vs. casual snapshot "
            "subreddits — proxy for formal/descriptive vs. casual register"
        ),
        "pole_a": ["earthporn", "foodporn", "carporn"],
        "pole_b": ["mildlyinteresting", "itookapicture"],
        "prompts_a": [
            "a professionally composed, high-quality photograph",
            "a formal, polished piece of photography",
            "an artfully composed image",
        ],
        "prompts_b": [
            "a casual snapshot photo",
            "an informal, quick picture",
            "a plain everyday photo",
        ],
    },
}

IMPRESSIONS_AXES = {
    "aesthetic_vs_description": {
        "description": (
            "subjective aesthetic captions vs. factual descriptive captions, "
            "same images — proxy for framing register"
        ),
        "pole_a_caption_type": "aesthetic",
        "pole_b_caption_type": "description",
        "prompts_a": [
            "a subjective, evocative description of an image's aesthetic qualities",
            "an artful, impressionistic caption",
        ],
        "prompts_b": [
            "a plain, factual description of an image's contents",
            "an objective, literal caption",
        ],
    },
    "impression_vs_caption": {
        "description": (
            "interpretive clinical-impression captions vs. plain captions, same "
            "images — proxy for interpretive vs. literal framing"
        ),
        "pole_a_caption_type": "impression",
        "pole_b_caption_type": "caption",
        "prompts_a": [
            "an interpretive clinical impression of an image",
            "a diagnostic-style summary judgment",
        ],
        "prompts_b": [
            "a plain caption naming what is shown",
            "a simple literal label for an image",
        ],
    },
}


def _subreddit_of(record):
    parts = record["image"].split("/")
    return parts[2] if len(parts) > 2 else "?"


def redcaps_binary_labels(records, axis_name):
    axis = REDCAPS_AXES[axis_name]
    keep, labels = [], []
    for i, r in enumerate(records):
        sub = _subreddit_of(r)
        if sub in axis["pole_a"]:
            keep.append(i)
            labels.append(1)
        elif sub in axis["pole_b"]:
            keep.append(i)
            labels.append(0)
    return np.array(keep, dtype=np.int64), np.array(labels, dtype=np.int64)


def impressions_binary_labels(records, axis_name):
    axis = IMPRESSIONS_AXES[axis_name]
    keep, labels = [], []
    for i, r in enumerate(records):
        ct = r.get("caption_type")
        if ct == axis["pole_a_caption_type"]:
            keep.append(i)
            labels.append(1)
        elif ct == axis["pole_b_caption_type"]:
            keep.append(i)
            labels.append(0)
    return np.array(keep, dtype=np.int64), np.array(labels, dtype=np.int64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_axis_definitions.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/test/20260915_condition_space_audit/axis_definitions.py src/test/20260915_condition_space_audit/test_axis_definitions.py
git commit -m "feat(exp17.1): add axis definitions and label-group extraction"
```

---

### Task 3: CLIP text-anchor direction construction

**Files:**
- Create: `src/test/20260915_condition_space_audit/text_anchor.py`
- Test: `src/test/20260915_condition_space_audit/test_text_anchor.py`

**Interfaces:**
- Consumes: nothing new (loads `openai/clip-vit-base-patch32` via `transformers`).
- Produces:
  - `direction_from_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray` — pure, unit-normalized.
  - `load_clip_text_tower(device="cpu") -> tuple[model, tokenizer]`
  - `encode_prompts(model, tokenizer, prompts: list[str], device="cpu") -> np.ndarray`
  - `build_direction(model, tokenizer, prompts_a, prompts_b, device="cpu") -> np.ndarray`
  - `build_control_directions(model, tokenizer, n=20, seed=42, device="cpu") -> list[np.ndarray]`

- [ ] **Step 1: Write the failing test for the pure direction math**

Create `src/test/20260915_condition_space_audit/test_text_anchor.py`:

```python
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import text_anchor as ta


def test_direction_from_embeddings_is_unit_norm():
    emb_a = np.array([[1.0, 0.0], [1.0, 0.0]])
    emb_b = np.array([[0.0, 1.0], [0.0, 1.0]])
    d = ta.direction_from_embeddings(emb_a, emb_b)
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert np.allclose(d, [1 / np.sqrt(2), -1 / np.sqrt(2)])


def test_direction_from_embeddings_rejects_degenerate():
    emb_a = np.array([[1.0, 2.0]])
    emb_b = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        ta.direction_from_embeddings(emb_a, emb_b)


@pytest.mark.slow
def test_build_direction_live_clip_smoke():
    model, tokenizer = ta.load_clip_text_tower(device="cpu")
    d = ta.build_direction(model, tokenizer, ["a happy photo"], ["a sad photo"], device="cpu")
    assert d.shape == (512,)
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-4)


@pytest.mark.slow
def test_build_control_directions_live_clip_smoke():
    model, tokenizer = ta.load_clip_text_tower(device="cpu")
    dirs = ta.build_control_directions(model, tokenizer, n=3, seed=42, device="cpu")
    assert len(dirs) == 3
    assert all(d.shape == (512,) for d in dirs)
    # deterministic given the fixed seed
    dirs2 = ta.build_control_directions(model, tokenizer, n=3, seed=42, device="cpu")
    assert all(np.allclose(a, b) for a, b in zip(dirs, dirs2))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_text_anchor.py -v -m "not slow"`
Expected: FAIL with "No module named 'text_anchor'".

- [ ] **Step 3: Implement `text_anchor.py`**

Create `src/test/20260915_condition_space_audit/text_anchor.py`:

```python
"""CLIP-text-anchor semantic-direction construction (StyleCLIP/ActAdd-style
difference-of-embeddings) plus matched random-prompt control directions, for
the Exp. 17.1 condition-space audit. Uses the raw HF CLIP text tower only —
never `CoSiRModel` (see this plan's Global Constraints).
"""
import random

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

BACKBONE = "openai/clip-vit-base-patch32"

# Fixed neutral word pool for control (meaningless) prompt-pair directions —
# unrelated to any of this audit's target axes, sampled deterministically.
CONTROL_WORD_POOL = [
    "table", "cloud", "river", "engine", "pencil", "mountain", "bottle", "ladder",
    "window", "carpet", "hammer", "bicycle", "lantern", "curtain", "basket", "kettle",
    "anchor", "blanket", "compass", "drum", "shovel", "mirror", "ribbon", "pillow",
    "faucet", "trumpet", "sandal", "wrench", "candle", "barrel",
]


def direction_from_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    direction = emb_a.mean(axis=0) - emb_b.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("degenerate direction (pole A and pole B means are identical)")
    return direction / norm


def load_clip_text_tower(device="cpu"):
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    return model, tokenizer


def encode_prompts(model, tokenizer, prompts, device="cpu"):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.text_model(**inputs)
        emb = model.text_projection(out.pooler_output)
    return emb.cpu().numpy().astype(np.float32)


def build_direction(model, tokenizer, prompts_a, prompts_b, device="cpu"):
    emb_a = encode_prompts(model, tokenizer, prompts_a, device)
    emb_b = encode_prompts(model, tokenizer, prompts_b, device)
    return direction_from_embeddings(emb_a, emb_b)


def build_control_directions(model, tokenizer, n=20, seed=42, device="cpu"):
    rng = random.Random(seed)
    directions = []
    for _ in range(n):
        w1, w2 = rng.sample(CONTROL_WORD_POOL, 2)
        d = build_direction(
            model, tokenizer,
            [f"a photo of a {w1}"],
            [f"a photo of a {w2}"],
            device=device,
        )
        directions.append(d)
    return directions
```

- [ ] **Step 4: Run the fast tests to verify they pass**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_text_anchor.py -v -m "not slow"`
Expected: 2 passed.

- [ ] **Step 5: Run the live CLIP smoke tests**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_text_anchor.py -v -m slow`
Expected: 2 passed (downloads `openai/clip-vit-base-patch32` from HF on first run if not already cached — CPU-only, seconds to a couple minutes on first download, near-instant after).

- [ ] **Step 6: Register the `slow` marker (avoid a pytest warning)**

Check whether `pytest.ini`/`pyproject.toml`/`setup.cfg` already registers custom markers:
```bash
grep -rn "markers" pytest.ini pyproject.toml setup.cfg 2>/dev/null
```
If none exists, add a `src/test/20260915_condition_space_audit/pytest.ini`:
```ini
[pytest]
markers =
    slow: marks tests that load the live CLIP model (deselect with '-m "not slow"')
```
If the repo already has a root pytest config with a `markers` section, add the `slow` marker line there instead and skip creating a new file.

- [ ] **Step 7: Commit**

```bash
git add src/test/20260915_condition_space_audit/text_anchor.py src/test/20260915_condition_space_audit/test_text_anchor.py src/test/20260915_condition_space_audit/pytest.ini
git commit -m "feat(exp17.1): add CLIP text-anchor direction construction + control directions"
```

---

### Task 4: Raw-CLIP audit script (Phase 17.1(a))

**Files:**
- Create: `src/test/20260915_condition_space_audit/raw_clip_audit.py`
- Test: `src/test/20260915_condition_space_audit/test_raw_clip_audit.py`

**Interfaces:**
- Consumes: `axis_definitions.REDCAPS_AXES/IMPRESSIONS_AXES`, `*_binary_labels`; `text_anchor.load_clip_text_tower/build_direction/build_control_directions`.
- Produces: `evaluate_direction(features, labels, direction) -> float`; `fold_auc(auc) -> float`; `audit_axis(...) -> dict` (per-axis/per-modality result with `real_auc`, `real_auc_folded`, `control_mean`, `control_std`, `z`, `verdict`); `src/test/20260915_condition_space_audit/raw_clip_audit_results.json`, consumed by Task 6.

- [ ] **Step 1: Write the failing tests for the pure statistics**

Create `src/test/20260915_condition_space_audit/test_raw_clip_audit.py`:

```python
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import raw_clip_audit as rca


def test_evaluate_direction_perfect_separation():
    features = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    labels = np.array([1, 1, 0, 0])
    direction = np.array([1.0, 0.0])
    auc = rca.evaluate_direction(features, labels, direction)
    assert np.isclose(auc, 1.0)


def test_fold_auc_symmetric():
    assert np.isclose(rca.fold_auc(0.2), 0.8)
    assert np.isclose(rca.fold_auc(0.8), 0.8)
    assert np.isclose(rca.fold_auc(0.5), 0.5)


def test_decision_rule_positive():
    assert rca.decision_rule(real_auc_folded=0.75, z=3.0) == "positive"


def test_decision_rule_partial():
    assert rca.decision_rule(real_auc_folded=0.55, z=2.5) == "partial"


def test_decision_rule_null():
    assert rca.decision_rule(real_auc_folded=0.52, z=0.8) == "null"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_raw_clip_audit.py -v`
Expected: FAIL with "No module named 'raw_clip_audit'".

- [ ] **Step 3: Implement `raw_clip_audit.py`**

Create `src/test/20260915_condition_space_audit/raw_clip_audit.py`:

```python
"""Phase 17.1(a): does a CLIP-text-anchor semantic direction separate RedCaps
subreddit or Impressions caption_type proxy labels, beyond a matched
random-prompt control? Training-free — reuses already-cached CLIP features.
"""
import json
import os
import sys

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import axis_definitions as ax
import text_anchor as ta
from src.utils import FeatureManager

REDCAPS_STORAGE = "/data/SSD2/pre_extract/redcaps_150k/features"
REDCAPS_ANNOT = "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"
IMPRESSIONS_STORAGE = "/data/SSD2/pre_extract/impressions/features"
IMPRESSIONS_ANNOT = "/project/Impressions/metadata/impressions_train.json"

POSITIVE_AUC_FLOOR = 0.60
Z_BAR = 2.0


def load_features(storage_dir, annotation_path):
    fm = FeatureManager(storage_dir)
    d = fm.load_all_to_ram(["img_features", "txt_features"])
    img = F.normalize(d["img_features"].float(), dim=1).numpy().astype(np.float32)
    txt = F.normalize(d["txt_features"].float(), dim=1).numpy().astype(np.float32)
    sample_ids = [int(x) for x in d["sample_ids"]]
    meta = json.load(open(annotation_path))
    records = [meta[s] for s in sample_ids]
    return {"img": img, "txt": txt}, records


def evaluate_direction(features, labels, direction):
    scores = features @ direction
    return float(roc_auc_score(labels, scores))


def fold_auc(auc):
    return max(auc, 1.0 - auc)


def decision_rule(real_auc_folded, z):
    if z >= Z_BAR and real_auc_folded >= POSITIVE_AUC_FLOOR:
        return "positive"
    if z >= Z_BAR:
        return "partial"
    return "null"


def audit_axis(features_by_modality, keep, labels, prompts_a, prompts_b, model, tokenizer, n_control=20, seed=42):
    real_dir = ta.build_direction(model, tokenizer, prompts_a, prompts_b, device="cpu")
    control_dirs = ta.build_control_directions(model, tokenizer, n=n_control, seed=seed, device="cpu")

    result = {}
    for modality, features in features_by_modality.items():
        feats = features[keep]
        real_auc = evaluate_direction(feats, labels, real_dir)
        control_aucs = np.array([fold_auc(evaluate_direction(feats, labels, d)) for d in control_dirs])
        real_auc_folded = fold_auc(real_auc)
        z = float((real_auc_folded - control_aucs.mean()) / (control_aucs.std(ddof=1) + 1e-8))
        result[modality] = {
            "n_pos": int(labels.sum()),
            "n_neg": int((1 - labels).sum()),
            "real_auc": real_auc,
            "real_auc_folded": real_auc_folded,
            "control_mean": float(control_aucs.mean()),
            "control_std": float(control_aucs.std(ddof=1)),
            "z": z,
            "verdict": decision_rule(real_auc_folded, z),
        }
    return result


def main():
    model, tokenizer = ta.load_clip_text_tower(device="cpu")

    redcaps_feats, redcaps_records = load_features(REDCAPS_STORAGE, REDCAPS_ANNOT)
    impressions_feats, impressions_records = load_features(IMPRESSIONS_STORAGE, IMPRESSIONS_ANNOT)

    results = {"redcaps_150k": {}, "impressions": {}}

    for axis_name, spec in ax.REDCAPS_AXES.items():
        keep, labels = ax.redcaps_binary_labels(redcaps_records, axis_name)
        results["redcaps_150k"][axis_name] = audit_axis(
            redcaps_feats, keep, labels, spec["prompts_a"], spec["prompts_b"], model, tokenizer
        )

    for axis_name, spec in ax.IMPRESSIONS_AXES.items():
        keep, labels = ax.impressions_binary_labels(impressions_records, axis_name)
        results["impressions"][axis_name] = audit_axis(
            impressions_feats, keep, labels, spec["prompts_a"], spec["prompts_b"], model, tokenizer
        )

    out_path = os.path.join(os.path.dirname(__file__), "raw_clip_audit_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_raw_clip_audit.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the real audit end-to-end**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260915_condition_space_audit/raw_clip_audit.py`
Expected: prints a nested JSON with `redcaps_150k.{warmth,register}` and `impressions.{aesthetic_vs_description,impression_vs_caption}`, each with `img`/`txt` sub-results carrying `verdict` in `{positive, partial, null}`; writes `raw_clip_audit_results.json`. No errors, no NaNs.

- [ ] **Step 6: Commit**

```bash
git add src/test/20260915_condition_space_audit/raw_clip_audit.py src/test/20260915_condition_space_audit/test_raw_clip_audit.py src/test/20260915_condition_space_audit/raw_clip_audit_results.json
git commit -m "feat(exp17.1): raw-CLIP text-anchor audit (Phase 17.1a)"
```

---

### Task 5: Existing-checkpoint probe (Phase 17.1(b))

**Files:**
- Create: `src/test/20260915_condition_space_audit/checkpoint_probe.py`
- Test: `src/test/20260915_condition_space_audit/test_checkpoint_probe.py`

**Interfaces:**
- Consumes: `src/test/20260915_condition_space_audit/checkpoints.json` (Task 1); `axis_definitions.*_binary_labels` (Task 2).
- Produces: `probe_selectivity(X, y, seed=42, n_shuffles=20, n_folds=5) -> dict`; `load_checkpoint(run_dir) -> tuple[np.ndarray, np.ndarray]`; `src/test/20260915_condition_space_audit/checkpoint_probe_results.json`, consumed by Task 6.

- [ ] **Step 1: Write the failing tests**

Create `src/test/20260915_condition_space_audit/test_checkpoint_probe.py`:

```python
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import checkpoint_probe as cp


def test_probe_selectivity_separable_data():
    rng = np.random.RandomState(0)
    X_pos = rng.normal(loc=2.0, scale=0.5, size=(60, 4))
    X_neg = rng.normal(loc=-2.0, scale=0.5, size=(60, 4))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 60 + [0] * 60)
    result = cp.probe_selectivity(X, y, seed=42, n_shuffles=10, n_folds=5)
    assert result["real_acc"] > 0.9
    assert result["z"] > 2.0


def test_probe_selectivity_random_data_has_low_selectivity():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(120, 4))
    y = rng.randint(0, 2, size=120)
    result = cp.probe_selectivity(X, y, seed=42, n_shuffles=10, n_folds=5)
    assert abs(result["z"]) < 3.0


def test_load_checkpoint_roundtrip(tmp_path):
    emb_dir = tmp_path / "final_embeddings"
    emb_dir.mkdir()
    emb = np.random.rand(10, 16).astype(np.float32)
    sids = np.arange(10, dtype=np.int64)
    np.save(emb_dir / "embeddings.npy", emb)
    np.save(emb_dir / "sample_ids.npy", sids)
    loaded_emb, loaded_sids = cp.load_checkpoint(str(tmp_path))
    assert np.allclose(loaded_emb, emb)
    assert np.array_equal(loaded_sids, sids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_checkpoint_probe.py -v`
Expected: FAIL with "No module named 'checkpoint_probe'".

- [ ] **Step 3: Implement `checkpoint_probe.py`**

Create `src/test/20260915_condition_space_audit/checkpoint_probe.py`:

```python
"""Phase 17.1(b): does a linear probe on already-trained buddy-condition
vectors already recover a proxy-label axis, beyond a matched random-relabeling
control (Hewitt & Liang selectivity)? No retraining — reads existing
checkpoints' final_embeddings only.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import axis_definitions as ax

Z_BAR = 2.0


def load_checkpoint(run_dir):
    emb_dir = os.path.join(run_dir, "final_embeddings")
    emb = np.load(os.path.join(emb_dir, "embeddings.npy"))
    sample_ids = np.load(os.path.join(emb_dir, "sample_ids.npy"))
    return emb, sample_ids


def probe_selectivity(X, y, seed=42, n_shuffles=20, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    real_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=skf)
    real_acc = float(real_scores.mean())

    rng = np.random.RandomState(seed)
    shuffle_accs = []
    for i in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        skf_i = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + i + 1)
        s = cross_val_score(LogisticRegression(max_iter=1000), X, y_shuffled, cv=skf_i)
        shuffle_accs.append(s.mean())
    shuffle_accs = np.array(shuffle_accs)

    z = float((real_acc - shuffle_accs.mean()) / (shuffle_accs.std(ddof=1) + 1e-8))
    return {
        "n": int(len(y)),
        "real_acc": real_acc,
        "control_mean": float(shuffle_accs.mean()),
        "control_std": float(shuffle_accs.std(ddof=1)),
        "selectivity": float(real_acc - shuffle_accs.mean()),
        "z": z,
        "verdict": "positive" if z >= Z_BAR else "null",
    }


def probe_checkpoint(run_dir, records_by_sample_id, axes, binary_labels_fn):
    emb, sample_ids = load_checkpoint(run_dir)
    records = [records_by_sample_id[int(s)] for s in sample_ids]
    result = {}
    for axis_name in axes:
        keep, labels = binary_labels_fn(records, axis_name)
        if len(np.unique(labels)) < 2 or len(labels) < 20:
            result[axis_name] = {"verdict": "skipped", "reason": "insufficient samples for this axis in this checkpoint"}
            continue
        result[axis_name] = probe_selectivity(emb[keep], labels)
    return result


def main():
    checkpoints_path = os.path.join(os.path.dirname(__file__), "checkpoints.json")
    checkpoints = json.load(open(checkpoints_path))

    redcaps_meta = json.load(open("/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"))
    impressions_meta = json.load(open("/project/Impressions/metadata/impressions_train.json"))

    results = {"redcaps_150k": {}, "impressions": {}}
    for run_dir in checkpoints["redcaps_150k"]:
        results["redcaps_150k"][run_dir] = probe_checkpoint(
            run_dir, redcaps_meta, ax.REDCAPS_AXES.keys(), ax.redcaps_binary_labels
        )
    for run_dir in checkpoints["impressions"]:
        results["impressions"][run_dir] = probe_checkpoint(
            run_dir, impressions_meta, ax.IMPRESSIONS_AXES.keys(), ax.impressions_binary_labels
        )

    out_path = os.path.join(os.path.dirname(__file__), "checkpoint_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -m pytest src/test/20260915_condition_space_audit/test_checkpoint_probe.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the real probe end-to-end**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260915_condition_space_audit/checkpoint_probe.py`
Expected: one result block per discovered checkpoint (Task 1's `checkpoints.json`) × axis, each `verdict` in `{positive, null, skipped}`; writes `checkpoint_probe_results.json`. If Task 1 found 0 checkpoints for a dataset, that dataset's result dict is empty — not an error, just report it in the log.

- [ ] **Step 6: Commit**

```bash
git add src/test/20260915_condition_space_audit/checkpoint_probe.py src/test/20260915_condition_space_audit/test_checkpoint_probe.py src/test/20260915_condition_space_audit/checkpoint_probe_results.json
git commit -m "feat(exp17.1): existing-checkpoint control-task-gated probe (Phase 17.1b)"
```

---

### Task 6: Aggregate results, apply the spec's decision rule, write the report

**Files:**
- Create: `src/test/20260915_condition_space_audit/summarize.py`
- Create: `docs/reports/2026-09-15_condition_space_audit.md`

**Interfaces:**
- Consumes: `raw_clip_audit_results.json` (Task 4), `checkpoint_probe_results.json` (Task 5).
- Produces: overall gate verdict (`positive`/`null`/`partial`) plus, if not null, the winning axis — read by whoever scopes Experiment 17.2 next; no code consumes this programmatically, it's read by a human.

- [ ] **Step 1: Implement `summarize.py`**

Create `src/test/20260915_condition_space_audit/summarize.py`:

```python
"""Apply Experiment 17.1's spec decision rule across all axis/modality results
and print the overall gate verdict for Experiment 17.2's scoping."""
import json
import os


def main():
    here = os.path.dirname(__file__)
    raw = json.load(open(os.path.join(here, "raw_clip_audit_results.json")))

    rows = []  # (dataset, axis, modality, verdict, real_auc_folded, z)
    for dataset, axes in raw.items():
        for axis_name, modalities in axes.items():
            for modality, r in modalities.items():
                rows.append((dataset, axis_name, modality, r["verdict"], r["real_auc_folded"], r["z"]))

    positives = [r for r in rows if r[3] == "positive"]
    partials = [r for r in rows if r[3] == "partial"]

    print(f"{'dataset':<14} {'axis':<28} {'modality':<6} {'verdict':<9} {'auc_folded':>10} {'z':>8}")
    for row in sorted(rows, key=lambda r: -r[5]):
        print(f"{row[0]:<14} {row[1]:<28} {row[2]:<6} {row[3]:<9} {row[4]:>10.3f} {row[5]:>8.2f}")

    if positives:
        best = max(positives, key=lambda r: r[5])
        print(f"\nGATE VERDICT: positive — strongest axis: {best[1]} ({best[0]}, {best[2]}, z={best[5]:.2f})")
    elif partials:
        best = max(partials, key=lambda r: r[5])
        print(f"\nGATE VERDICT: partial — strongest axis: {best[1]} ({best[0]}, {best[2]}, z={best[5]:.2f})")
    else:
        print("\nGATE VERDICT: null — no axis cleared its control baseline on any dataset/modality")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and capture the verdict**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260915_condition_space_audit/summarize.py`
Expected: a table of every (dataset, axis, modality) row plus one `GATE VERDICT: positive|partial|null` line. Copy this output into the report below verbatim.

- [ ] **Step 3: Write the report**

Create `docs/reports/2026-09-15_condition_space_audit.md`, filling `<...>` markers with the actual run output from Step 2 (the summary table, the gate verdict, and — from `checkpoint_probe_results.json` — whether any existing checkpoint already shows a positive/null selectivity per axis):

```markdown
# Condition-Space Steerability Audit — Experiment 17.1

**Date:** 2026-09-15
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, Experiment 17.1
**Plan:** `docs/superpowers/plans/2026-09-15-condition-space-audit.md`

## What was tested

Two training-free checks, per the spec's Phase A: (a) whether a CLIP-text-anchor
semantic direction (built from contrastive prompt pairs per axis, StyleCLIP/ActAdd-style)
separates RedCaps subreddit or Impressions `caption_type` proxy labels in raw CLIP
image/text feature space, beyond a matched random-prompt control (20 controls,
seed=42); (b) whether the same proxy labels are already linearly decodable from
already-trained buddy-init condition vectors (any completed asymmetric,
`combine_side="img"`, buddy-init checkpoint — discovered via
`discover_checkpoints.py`), beyond a matched random-relabeling control
(Hewitt & Liang selectivity, 20 shuffles).

Axes tested: RedCaps `warmth` (companion-animal vs. curiosity-framed subreddits)
and `register` (aesthetic-photography vs. casual-snapshot subreddits); Impressions
`aesthetic_vs_description` and `impression_vs_caption` (both `caption_type` contrasts
— no valence proxy exists for Impressions, a genuine data-availability gap, not an
oversight; see `src/test/20260915_condition_space_audit/axis_definitions.py` for
exact pole definitions and prompts).

Decision rule (per axis/modality): **positive** if the real direction's folded AUC
(`max(auc, 1-auc)`) clears the control mean by `z ≥ 2` (this project's existing
significance convention) *and* the folded AUC itself is `≥ 0.60`; **partial** if
`z ≥ 2` but AUC `< 0.60`; **null** otherwise. Overall gate: positive if any
axis/modality is positive, partial if none are positive but some are partial,
null if all are null.

## Results — Phase A (raw CLIP)

```
<paste summarize.py's full table + GATE VERDICT line here>
```

## Results — Phase B (existing checkpoints)

<summarize checkpoint_probe_results.json here: per dataset, how many checkpoints
were probed, and for each axis, how many showed "positive" selectivity vs. "null"
vs. "skipped" (insufficient samples in that checkpoint's split) — this informs
17.2's initialization design (does buddy-init already carry proxy-label signal by
accident) but is not itself the primary gate, per the spec.>

## Verdict and effect on Experiment 17.2

<state the overall gate verdict (positive/null/partial) from Step 2's output, the
winning axis if any, and what this means for 17.2's scope per the spec's stated
routing: positive → scope 17.2 to the winning axis; null → scope 17.2 to
architecture-only fixes, write up the interpretability ambition as a negative
result matching C4/C7; partial → scope 17.2 to the single strongest axis,
flagged exploratory.>

## Caveats

- All four axes are proxy labels, not ground truth for "emotional tone" or
  "political framing" specifically — a positive result here means the proxy axis
  is present, not that the originally-envisioned target concept is (see the
  spec's §8 risk row on this).
- No verified published benchmark exists for emotion/political-framing detection
  on Reddit *image* content specifically (per this project's literature research,
  2026-09-15) — this audit's own result is the first evidence either way for this
  domain.
- Impressions has no valence-adjacent proxy label at all in this pass; both of its
  axes are registerial/framing contrasts within `caption_type`, not emotional-tone
  contrasts — stated as a data-availability constraint, not routed around.
```

- [ ] **Step 4: Commit**

```bash
git add src/test/20260915_condition_space_audit/summarize.py docs/reports/2026-09-15_condition_space_audit.md
git commit -m "docs(exp17.1): summarize condition-space audit results and gate verdict"
```

---

### Task 7: Append the Result paragraph to the spec, close out the log

**Files:**
- Modify: `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` (Experiment 17.1's entry in §4)
- Modify: `src/test/20260915_condition_space_audit/20260915_condition_space_audit_log.md`

**Interfaces:**
- Consumes: the gate verdict from Task 6.
- Produces: nothing further consumed by code — this closes out Experiment 17.1 and hands the verdict to whoever scopes 17.2 next.

- [ ] **Step 1: Append a Result paragraph to 17.1 in the spec**

In `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, find Experiment 17.1's bullet list (ends with its `**Cost:**` line) and append one more bullet, matching this spec's existing `**Result (YYYY-MM-DD, ...):**` convention (see 16.1/16.2 for the exact style — one paragraph, state the verdict, the winning axis if any, and link the report):

```markdown
- **Result (2026-09-15, `src/test/20260915_condition_space_audit/`):** <fill in from docs/reports/2026-09-15_condition_space_audit.md's Verdict section — state positive/null/partial, the winning axis and its z/AUC if positive or partial, and what this means for 17.2's scope per the decision rule above>. Full write-up: `docs/reports/2026-09-15_condition_space_audit.md`.
```

- [ ] **Step 2: Fill in the log's Results section**

In `src/test/20260915_condition_space_audit/20260915_condition_space_audit_log.md`, replace the `## Results` placeholder with a short summary (2-3 sentences) of the gate verdict and a link to the report.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md src/test/20260915_condition_space_audit/20260915_condition_space_audit_log.md
git commit -m "docs(exp17.1): record condition-space audit result in publication-plan spec"
```

---

## Self-review

- **Spec coverage:** Phase A(a) (CLIP-text-anchor sanity check vs. control) → Tasks 3+4; Phase A(b) (existing-checkpoint probe vs. control) → Tasks 1+5; the spec's stated 2-datasets × ≥2-axes-each × {raw CLIP, checkpoints} scope → Task 2's four axis definitions, exercised by both Task 4 and Task 5; the spec's positive/null/partial decision rule → `raw_clip_audit.decision_rule` (Task 4) and `summarize.py` (Task 6); the spec's required final report → Task 6; the spec's requirement that this feeds 17.2's scoping → Task 7's spec update.
- **Placeholder scan:** every task has real, complete code (no TBD/stub bodies); the only `<...>` markers left are in Task 6/7's report/spec templates, which are explicitly the *output* of running the scripts (cannot be known until the audit actually runs) — not unresolved design decisions.
- **Type consistency:** `redcaps_binary_labels`/`impressions_binary_labels` signature (`records, axis_name) -> (keep_idx, labels)`) is identical across Task 2's definition, Task 4's `audit_axis` caller, and Task 5's `probe_checkpoint` caller; `direction_from_embeddings`/`build_direction`/`build_control_directions` signatures from Task 3 are used unchanged in Task 4; `load_checkpoint` from Task 5 matches its own test's roundtrip in Task 5 Step 1.
