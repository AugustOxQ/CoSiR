# CoSiR v2 CVPR plan, experiments E0 to E5: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the aspect-episode evaluation stack and the baselines it needs. Train method A (aspect-trained shared
factors) on ArtELingo and decide the Fri Oct 9 go/no-go, together with the held-out-aspect test and the early
in-context MLLM probe. Extract the features the later experiments need, and replicate a GO winner at two more seeds.

**Architecture:** Three pipeline modules carry the protocol. `src/eval/aspect_episodes.py` builds value-disjoint
cross-item aspect episodes for any dataset with per-row integer labels. `src/eval/aspect_metrics.py` turns score
matrices into per-anchor R@1, condition gain and swap, plus painting-clustered bootstrap CIs.
`src/eval/aspect_scorers.py` and `src/eval/pair_metric_baselines.py` produce those score matrices.

Training reuses `train_factors` with a new pseudo-aspect episode loss. The episodes are built by the same builder
from k-means pseudo-partitions over scorer-train rows. Each experiment runs from a dated folder under `src/test/`
and ends with a report in `docs/reports/auto/v2/`.

**Tech Stack:** Python 3.10 (`/root/miniconda3/envs/CoSiR/bin/python`), numpy, torch, scikit-learn, pytest,
transformers 5.6.2 (Qwen3-VL, already installed), matplotlib for figures.

**Spec:** `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` (revision 2). Read §3 to §11
before starting. Section numbers below refer to it.

## Mechanism note (found while planning, 2026-10-03)

The agreement rule can only carry an aspect from the shown values to the anchor's value if the values share
dimensions: every value of an aspect must be a different pattern over the same factors.
- **Value-specific factors** (one factor per value) make the rule blind under value-disjoint conditions. The
  supports show other values, so the anchor's factor gets zero weight and every candidate ties. A simulation gave
  condition gain 0.00, against 0.20 to 0.32 for aspect-block codes.
- **Raw features that mix the aspects** across all dimensions defeat raw pair rules in the same way: 0.00 mixed,
  against 0.70 to 0.95 when each aspect has its own block.

This explains the aspect spike's failure (SE was trained on value episodes, raw CLIP mixes aspects) and states what
method A must learn: an aspect-block basis. The plan pins it with tests (Tasks 4 and 8), adds a denser grid variant
(A6) and reports a value-sharing diagnostic (Task 13).

## Global Constraints

- **Environment.**
  - Python is always `/root/miniconda3/envs/CoSiR/bin/python`.
  - Tests: `/root/miniconda3/envs/CoSiR/bin/python -m pytest <file> -q` from `/project/CoSiR`.
  - Never install into the CoSiR env. Use `pip install --target /data/SSD2/pyenvs/<name>/` and prepend that path in
    the script that needs it.
- **Shared machine.**
  - Before GPU work run `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`, and wrap every GPU
    command in `flock -n -o -E 75 /tmp/gpu0.lock <cmd>`. Exit 75 means another session holds the GPU: stop and
    report.
  - CPU work sets `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`.
  - Never kill a process you did not start.
- **Row scope.**
  - ArtELingo development reads only selection rows (32,413 rows, 6,451 paintings). Training reads only
    scorer-train rows (183,694 rows, 36,518 paintings).
  - Val and held rows are never read in this plan, and no episode, feature or code row outside the stated scope may
    be finite in an evaluation array (use NaN masking).
  - CUB development uses only the 150 zero-shot training species; the 50 test species are never read.
  - GeneCIS templates are never read.
- **No evaluation labels in training.** ArtELingo emotion, style and genre labels and CUB attributes are used only
  for episodes and diagnostics. GoEmotions affect clusters are allowed (distant supervision, spec §4 C2).
- **Frozen backbones.** CLIP ViT-B/32 features come from `load_artelingo()`, and Qwen3-VL-Embedding-2B features from
  Task 6's extractor. No fine-tuning.
- **Fixed numbers** (spec §5.1, §6, §10):
  - 4 support pairs, 4 contrast pairs, 13 candidates (p_a, p_b, 11 negatives);
  - at least 30 paintings per eligible value;
  - fusion grid λ ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16, inf}, extended once to {32, 64} if a pick lands on 16;
  - 5,000 bootstrap resamples, bootstrap seed 42;
  - GO picks on seed-42 episodes and tests on seed-43 episodes; the MLLM probe uses seed-44 episodes, 300 anchors
    per aspect pair;
  - training is 2,000 steps, β 0.3 in training, L = 32 unless stated.
- **Seeds.** Model seeds are 42, 43 and 44; episode seeds are as above.
- **Commits.**
  - One commit per task step marked Commit.
  - Stage files by explicit path. Other sessions share main and `bin/` stays untracked.
  - End each message with the two attribution lines: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_01BUKmxn46sHfmwsu6WMZiQQ`.
- **Change log.** Edits to existing source files get an entry in `.claude/20261003_log.md` (or the current day's
  file): a `# <path>` header, before/after snippets, and why.
- **Experiment folders.**
  - Folders are `src/test/<YYYYMMDD sequence>_<name>/`, each with a `.gitignore` copied from
    `src/test/20261023_aspect_episode_spike/.gitignore` (results, npz, json, logs and checkpoints stay local).
  - Each folder ends with a log `<YYYYMMDD>_<name>_log.md`: problem, steps, results, issues.
- **Reports.**
  - Every experiment ends with a report at `docs/reports/auto/v2/<YYYY-MM-DD sequence>_<topic>.md` plus one row in
    `docs/reports/reports_sum.md` (v2 table, after the last row), then
    `/root/miniconda3/envs/CoSiR/bin/python scripts/check_reports_sum.py` must print OK.
  - Style: paper-draft ("we", past tense), a real baseline beside every number, figures, no dashes as punctuation.
- **Long jobs** (over about 15 minutes: training grids, extraction, the MLLM probe) are launched by the main session
  with Bash `run_in_background: true`. A subagent writes and smoke-tests the script; the controller launches it.

## Review Focus

These inputs and conditions are not exercised by the happy-path tests but would bite a user of the stack. Each has
a test in its owning task.

1. **Unlabelled rows** (genre −1 on 19% of paintings, the CUB "no unique value" images) must never fill a role that
   needs that label. Task 2, `test_unlabelled_rows_never_used`.
2. **Tied scores**, e.g. all-zero agreement weights, so the factor term is constant. A tie must count as a miss, never
   as a hit. Task 3, `test_ties_are_misses`.
3. **A constant term inside z-fusion** must not produce NaN; the fused score falls back to the cosine. Task 4,
   `test_zfuse_constant_term_falls_back_to_cosine`.
4. **Non-finite rows** (NaN outside the read scope) reaching a score must make that episode a miss, never a crash or
   a silent hit. Task 3, `test_nonfinite_scores_are_misses`.
5. **An exhausted episode pool** (small datasets, strict constraints) must raise a clear `RuntimeError` naming the
   aspect pair, not loop forever. Task 2, `test_exhausted_pool_raises`.

---

## File map

| File | Responsibility | Task |
|---|---|---|
| `src/data/artelingo_splits.py` | recompute the stage (d) split; ArtELingo aspect label codes | 1 |
| `src/data/wikiart_genre.py` | ArtGAN genre labels joined to paintings | 1 |
| `src/eval/aspect_episodes.py` | aspect episode builder, validator, hashing, concatenation | 2 |
| `src/eval/aspect_metrics.py` | first-place hits, per-anchor metrics, clustered bootstrap | 3 |
| `src/model/aspect_rule.py` | agreement weights (torch) and z-fusion | 4 |
| `src/eval/aspect_scorers.py` | backbone, agreement and uniform scorers; λ cross-fitting | 4 |
| `docs/superpowers/held_ledger.md` | the held-read ledger (spec §10) | 5 |
| `src/data/cub.py` | CUB attributes, captions and zero-shot split | 5 |
| `src/data/feature_extract.py` | CLIP B/32 and Qwen3-VL-Embedding-2B encoders | 6 |
| `src/eval/pair_metric_baselines.py` | Tier-1 raw-feature baselines (spec §8) | 8 |
| `src/train/pseudo_partitions.py` | k-means partitions and the pseudo-aspect episode bank | 10 |
| `src/train/aspect_loss.py` | aspect episode scores and loss | 11 |
| `src/train/train_factors.py` (modify) | optional aspect loss branch | 11 |
| `src/eval/mllm_reranker.py` | Qwen3-VL-2B-Instruct in-context reranker | 14 |
| `scripts/extract_features.py` | dataset adapters for E4 extraction | 16 |
| `src/test/test_*.py` | unit tests for each module | 1 to 14 |

---

### Task 1: ArtELingo splits, aspect labels and WikiArt genre

**Files:**
- Create: `src/data/artelingo_splits.py`, `src/data/wikiart_genre.py`
- Test: `src/test/test_artelingo_splits.py`, `src/test/test_wikiart_genre.py`

**Interfaces:**
- Consumes: `src.data.splits.grouped_split`, `grouped_subsplit`, `leakage_groups`; `src.data.artelingo.ArtelingoData`
  (fields `img_features`, `txt_features`, `emotions`, `paintings`, `art_styles`).
- Produces:
  - `artelingo_splits(data) -> ArtelingoSplits`, with fields `groups`, `scorer_train`, `selection`, `val` and `held`
    (`np.ndarray`, int64 global rows);
  - `encode_labels(values, missing=("", None), exclude=()) -> tuple[np.ndarray, list[str]]`, giving int codes with −1
    for missing or excluded values, plus a name list;
  - `artelingo_aspect_labels(data) -> dict[str, np.ndarray]`, keyed `emotion`, `style` and `genre` (int, −1 missing);
  - `load_wikiart_genre(paintings, csv_paths=GENRE_CSVS) -> np.ndarray` (int, −1 missing);
  - `GENRE_NAMES: list[str]`.

- [ ] **Step 1: Write the failing tests**

`src/test/test_wikiart_genre.py`:
```python
import numpy as np

from src.data.wikiart_genre import GENRE_NAMES, load_wikiart_genre


def test_genre_join_by_stem_and_conflicts(tmp_path):
    a = tmp_path / "genre_train.csv"
    b = tmp_path / "genre_val.csv"
    a.write_text("Impressionism/monet_water-lilies.jpg,4\nBaroque/rubens_x.jpg,7\nCubism/conflict.jpg,0\n")
    b.write_text("Cubism/conflict.jpg,6\nRealism/only-in-val.jpg,9\n")
    paintings = np.array(["monet_water-lilies", "rubens_x", "conflict", "only-in-val", "unknown"], dtype=object)
    got = load_wikiart_genre(paintings, csv_paths=(a, b))
    assert got.tolist() == [4, 7, -1, 9, -1]                 # conflict and unknown are missing


def test_genre_names_match_artgan_class_file():
    assert GENRE_NAMES[0] == "abstract_painting" and GENRE_NAMES[5] == "nude_painting"
    assert GENRE_NAMES[9] == "still_life" and len(GENRE_NAMES) == 10
```

`src/test/test_artelingo_splits.py`:
```python
import numpy as np

from src.data.artelingo_splits import encode_labels


def test_encode_labels_missing_and_excluded():
    codes, names = encode_labels(np.array(["sad", "", "awe", "something else", None, "sad"], dtype=object),
                                 exclude=("something else",))
    assert names == ["awe", "sad"]
    assert codes.tolist() == [1, -1, 0, -1, -1, 1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_wikiart_genre.py src/test/test_artelingo_splits.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.wikiart_genre'`

- [ ] **Step 3: Implement**

`src/data/wikiart_genre.py`:
```python
"""WikiArt genre labels (ArtGAN `WikiArt Dataset/Genre/genre_{train,val}.csv`) joined to ArtELingo paintings.

Downloaded 2026-10-02 into /data/SSD/wikiart_genre/. The class names come from ArtGAN's `Genre/genre_class`
file (no .txt extension; see docs/reports/auto/v2/2026-10-28_citation_check.md, E1).
"""

import csv
from pathlib import Path

import numpy as np

GENRE_DIR = Path("/data/SSD/wikiart_genre")
GENRE_CSVS = (GENRE_DIR / "genre_train.csv", GENRE_DIR / "genre_val.csv")
GENRE_NAMES = ["abstract_painting", "cityscape", "genre_painting", "illustration", "landscape", "nude_painting",
               "portrait", "religious_painting", "sketch_and_study", "still_life"]


def load_wikiart_genre(paintings: np.ndarray, csv_paths=GENRE_CSVS) -> np.ndarray:
    """One genre id per entry of ``paintings`` (file stems); -1 when unknown or when the CSVs disagree."""
    ids: dict[str, int] = {}
    for path in csv_paths:
        with Path(path).open() as handle:
            for rel_path, label in csv.reader(handle):
                stem = Path(rel_path).stem
                value = int(label)
                if stem in ids and ids[stem] != value:
                    ids[stem] = -1
                else:
                    ids.setdefault(stem, value)
    return np.asarray([ids.get(str(p), -1) for p in paintings], dtype=np.int64)
```

`src/data/artelingo_splits.py`:
```python
"""The stage (d) ArtELingo split and per-row aspect label codes (CVPR plan spec §2.3, §5)."""

from dataclasses import dataclass

import numpy as np

from src.data.splits import grouped_split, grouped_subsplit, leakage_groups
from src.data.wikiart_genre import load_wikiart_genre

SPLIT_SEED = 42
SELECTION_FRACTION = 0.15
EXPECTED_SIZES = {"train": 216_107, "val": 30_872, "held": 61_744, "scorer_train": 183_694, "selection": 32_413}
EMOTION_CATCH_ALL = "something else"


@dataclass(frozen=True)
class ArtelingoSplits:
    groups: np.ndarray
    scorer_train: np.ndarray
    selection: np.ndarray
    val: np.ndarray
    held: np.ndarray


def artelingo_splits(data) -> ArtelingoSplits:
    """Recompute grouped_split(leakage_groups(...), 42) and the 15% selection sub-split; assert the known sizes."""
    groups = leakage_groups(data.paintings, data.img_features)
    split = grouped_split(groups, seed=SPLIT_SEED)
    scorer_train, selection = grouped_subsplit(groups, split.train, SELECTION_FRACTION, seed=SPLIT_SEED)
    got = {"train": len(split.train), "val": len(split.val), "held": len(split.held),
           "scorer_train": len(scorer_train), "selection": len(selection)}
    if got != EXPECTED_SIZES:
        raise AssertionError(f"split sizes {got} differ from the stage (d) split {EXPECTED_SIZES}")
    return ArtelingoSplits(groups, scorer_train, selection, np.sort(split.val), np.sort(split.held))


def encode_labels(values, missing=("", None), exclude=()) -> tuple[np.ndarray, list]:
    """Integer codes in sorted name order; -1 for missing or excluded values."""
    values = np.asarray(values, dtype=object)
    bad = set(missing) | set(exclude)
    names = sorted({v for v in values.tolist() if v not in bad})
    lookup = {name: i for i, name in enumerate(names)}
    return np.asarray([lookup.get(v, -1) for v in values.tolist()], dtype=np.int64), names


def artelingo_aspect_labels(data) -> dict[str, np.ndarray]:
    """Per-row codes for the three ArtELingo aspects; emotion excludes the catch-all, genre is -1 where unlabelled."""
    emotion, _ = encode_labels(data.emotions, exclude=(EMOTION_CATCH_ALL,))
    style, _ = encode_labels(data.art_styles)
    genre = load_wikiart_genre(np.asarray(data.paintings, dtype=object))
    return {"emotion": emotion, "style": style, "genre": genre}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_wikiart_genre.py src/test/test_artelingo_splits.py -q`
Expected: 3 passed.

- [ ] **Step 5: Real-data check, recorded in the E0 folder log in Task 7**

```bash
cd /project/CoSiR && OMP_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python - <<'EOF'
import numpy as np
from src.data.artelingo import load_artelingo
from src.data.artelingo_splits import artelingo_splits, artelingo_aspect_labels
d = load_artelingo(); s = artelingo_splits(d); lab = artelingo_aspect_labels(d)
for k in ("emotion", "style", "genre"):
    sel = lab[k][s.selection]
    print(k, "labelled share in selection", round(float((sel >= 0).mean()), 3), "values", len(np.unique(sel[sel >= 0])))
EOF
```
Expected: emotion share about 0.9 (the catch-all excluded), style 1.0, genre about 0.81; 8 or 9 emotion values, 23
or more style values and 10 genre values. Also confirm the split equals stage (d)'s: compare `s.scorer_train` and
`s.selection` with `grid.load_grid()` the way `src/test/20261026_genre_coverage/genre_coverage.py` does, and assert
array equality.

- [ ] **Step 6: Commit**

```bash
git add src/data/artelingo_splits.py src/data/wikiart_genre.py src/test/test_artelingo_splits.py src/test/test_wikiart_genre.py
git commit -m "feat(v2): ArtELingo split recompute, aspect label codes and WikiArt genre join"
```

---

### Task 2: Aspect episode builder

**Files:**
- Create: `src/eval/aspect_episodes.py`
- Test: `src/test/test_aspect_episodes.py`

**Interfaces:**
- Consumes: `src.data.sampling.draw_distinct(rng, pool, keys, used, count) -> list[int]` (raises `ValueError` when it
  cannot fill).
- Produces:
  - `AspectEpisodes`, a frozen dataclass with fields:
    - `aspect_a: str`, `aspect_b: str`;
    - `anchor (n,)`, `candidates (n, 13)`;
    - `pairs_a_img (n, 4)`, `pairs_a_txt (n, 4)`, `pairs_b_img (n, 4)`, `pairs_b_txt (n, 4)`;
    - method `condition(which) -> (sup_img, sup_txt, con_img, con_txt, target_col)`;
    - method `rows() -> np.ndarray` (every row used).
  - `PaintingValueIndex(labels, groups)` with `.lacks(aspect, value) -> np.ndarray[bool]` over all rows.
  - `eligible_values(labels_a, groups, rows, min_paintings) -> list[int]`.
  - `build_aspect_episodes(labels, groups, rows, aspect_a, aspect_b, n_episodes, seed, third=None,
    min_paintings=30, index=None) -> AspectEpisodes`.
  - `validate_aspect_episodes(ep, labels, groups, index, third=None) -> None` (raises `AssertionError`).
  - `episodes_sha256(ep) -> str`.
  - `concat_episodes(list[AspectEpisodes]) -> AspectEpisodes`.

The rules (spec §5.1), with a = the anchor's value on aspect A, b = its value on aspect B, and t = its value on the
third aspect when one is given:
- **Labels.** Every role uses only rows labelled (≥ 0) on A and B, and on the third aspect when `third` is given.
- **p_a:** label_A = a, and the row's painting has no row with label_B = b (and none with third = t).
- **p_b:** label_B = b, and the painting lacks a (and t).
- **Negatives (11):** the painting lacks a, b and t.
- **P_A (4 pairs):** 4 distinct eligible values v ≠ a. Each pair is (x, y) with label_A(x) = label_A(y) = v and
  label_B(x) ≠ label_B(y), x and y from different paintings, and both paintings lacking a and b.
- **P_B:** symmetric, sharing B, differing on A.
- **Distinct paintings:** all 30 rows of an episode come from distinct paintings.
- **Conditions:** condition "a" uses supports P_A and contrasts P_B, with target column 0; condition "b" swaps them,
  with target column 1.

- [ ] **Step 1: Write the failing tests**

`src/test/test_aspect_episodes.py`:
```python
import numpy as np
import pytest

from src.eval.aspect_episodes import (
    AspectEpisodes, PaintingValueIndex, build_aspect_episodes, concat_episodes, episodes_sha256,
    validate_aspect_episodes,
)


def _world(n_paintings=3000, seed=0, genre_missing=0.2):
    """Three aspects, 6 values each; two rows per painting; the first aspect varies per row like emotion."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_paintings), 2)
    style = np.repeat(rng.integers(0, 6, n_paintings), 2)
    genre = np.repeat(rng.integers(0, 6, n_paintings), 2)
    genre[np.repeat(rng.random(n_paintings) < genre_missing, 2)] = -1
    emotion = rng.integers(0, 6, 2 * n_paintings)
    return {"emotion": emotion, "style": style, "genre": genre}, groups


def test_roles_follow_the_rules_and_validate():
    labels, groups = _world()
    rows = np.arange(len(groups))
    index = PaintingValueIndex(labels, groups)
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 200, seed=1, third="genre", index=index)
    assert isinstance(ep, AspectEpisodes) and ep.candidates.shape == (200, 13) and ep.pairs_a_img.shape == (200, 4)
    validate_aspect_episodes(ep, labels, groups, index, third="genre")
    a, b = labels["emotion"][ep.anchor], labels["style"][ep.anchor]
    assert (labels["emotion"][ep.candidates[:, 0]] == a).all() and (labels["style"][ep.candidates[:, 1]] == b).all()
    assert (labels["emotion"][ep.pairs_a_img] == labels["emotion"][ep.pairs_a_txt]).all()
    assert (labels["style"][ep.pairs_a_img] != labels["style"][ep.pairs_a_txt]).all()
    assert (labels["emotion"][ep.pairs_a_img] != a[:, None]).all()


def test_condition_swaps_roles():
    labels, groups = _world()
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "emotion", "style", 20, seed=2)
    si, st, ci, ct, col = ep.condition("a")
    assert col == 0 and (si == ep.pairs_a_img).all() and (ci == ep.pairs_b_img).all()
    si, st, ci, ct, col = ep.condition("b")
    assert col == 1 and (si == ep.pairs_b_img).all() and (ci == ep.pairs_a_img).all()
    with pytest.raises(ValueError):
        ep.condition("c")


def test_seed_determinism_and_hash():
    labels, groups = _world()
    rows = np.arange(len(groups))
    e1 = build_aspect_episodes(labels, groups, rows, "emotion", "style", 50, seed=3)
    e2 = build_aspect_episodes(labels, groups, rows, "emotion", "style", 50, seed=3)
    e3 = build_aspect_episodes(labels, groups, rows, "emotion", "style", 50, seed=4)
    assert episodes_sha256(e1) == episodes_sha256(e2) != episodes_sha256(e3)


def test_rows_scope_respected():
    labels, groups = _world()
    rows = np.arange(0, len(groups), 1)[: len(groups) // 2]                   # first half of the paintings only
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 50, seed=5)
    assert np.isin(ep.rows(), rows).all()


def test_unlabelled_rows_never_used():                                   # Review Focus 1
    labels, groups = _world(genre_missing=0.5)
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "genre", "style", 100, seed=6)
    assert (labels["genre"][ep.rows()] >= 0).all()


def test_exhausted_pool_raises():                                         # Review Focus 5
    labels, groups = _world(n_paintings=40)
    with pytest.raises(RuntimeError, match="emotion.*style"):
        build_aspect_episodes(labels, groups, np.arange(len(groups)), "emotion", "style", 50, seed=7,
                              min_paintings=1)


def test_concat_keeps_order():
    labels, groups = _world()
    rows = np.arange(len(groups))
    e1 = build_aspect_episodes(labels, groups, rows, "emotion", "style", 10, seed=8)
    e2 = build_aspect_episodes(labels, groups, rows, "style", "genre", 10, seed=9)
    both = concat_episodes([e1, e2])
    assert both.anchor.tolist() == e1.anchor.tolist() + e2.anchor.tolist() and both.aspect_a == "mixed"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_episodes.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.aspect_episodes'`

- [ ] **Step 3: Implement `src/eval/aspect_episodes.py`**

```python
"""Aspect episodes (CVPR plan spec §5.1): the condition shows an aspect through value-disjoint cross-item example
pairs, contrasted with pairs that share another aspect. Generic over datasets: labels are per-row int codes, -1 =
unlabelled. Rows, groups and labels index the same global row space."""

import hashlib
from dataclasses import dataclass, fields

import numpy as np

from src.data.sampling import draw_distinct

NUM_PAIRS = 4
NUM_NEGATIVES = 11


@dataclass(frozen=True)
class AspectEpisodes:
    aspect_a: str
    aspect_b: str
    anchor: np.ndarray
    candidates: np.ndarray        # column 0 = p_a, column 1 = p_b, then 11 negatives
    pairs_a_img: np.ndarray       # example pair i of aspect a: the image of row pairs_a_img[:, i] ...
    pairs_a_txt: np.ndarray       # ... with the caption of row pairs_a_txt[:, i]
    pairs_b_img: np.ndarray
    pairs_b_txt: np.ndarray

    def condition(self, which: str):
        """(support_img, support_txt, contrast_img, contrast_txt, target_column) for condition 'a' or 'b'."""
        if which == "a":
            return self.pairs_a_img, self.pairs_a_txt, self.pairs_b_img, self.pairs_b_txt, 0
        if which == "b":
            return self.pairs_b_img, self.pairs_b_txt, self.pairs_a_img, self.pairs_a_txt, 1
        raise ValueError(f"condition must be 'a' or 'b', got {which!r}")

    def rows(self) -> np.ndarray:
        return np.concatenate([self.anchor.ravel(), self.candidates.ravel(), self.pairs_a_img.ravel(),
                               self.pairs_a_txt.ravel(), self.pairs_b_img.ravel(), self.pairs_b_txt.ravel()])


class PaintingValueIndex:
    """``lacks(aspect, value)``: True for rows whose painting (group) has no row labelled ``value`` on ``aspect``."""

    def __init__(self, labels: dict, groups: np.ndarray) -> None:
        self.labels = {k: np.asarray(v, dtype=np.int64) for k, v in labels.items()}
        self.groups = np.asarray(groups)
        self._cache: dict = {}

    def lacks(self, aspect: str, value: int) -> np.ndarray:
        key = (aspect, int(value))
        if key not in self._cache:
            having = np.unique(self.groups[self.labels[aspect] == value])
            mask = ~np.isin(self.groups, having)
            mask.flags.writeable = False
            self._cache[key] = mask
        return self._cache[key]


def eligible_values(labels_a: np.ndarray, groups: np.ndarray, rows: np.ndarray, min_paintings: int) -> list:
    rows = np.asarray(rows, dtype=np.int64)
    out = []
    for value in np.unique(labels_a[rows]):
        if value < 0:
            continue
        if len(np.unique(groups[rows[labels_a[rows] == value]])) >= min_paintings:
            out.append(int(value))
    return out


def _pairs(rng, base, share, differ, values, groups, used):
    """One cross-item pair per value: both rows share ``value`` on ``share`` and differ on ``differ``."""
    img, txt = [], []
    for value in values:
        pool = base[share[base] == value]
        x = draw_distinct(rng, pool, groups, used, 1)[0]
        y = draw_distinct(rng, pool[differ[pool] != differ[x]], groups, used, 1)[0]
        img.append(x)
        txt.append(y)
    return img, txt


def build_aspect_episodes(labels: dict, groups: np.ndarray, rows: np.ndarray, aspect_a: str, aspect_b: str,
                          n_episodes: int, seed: int, third: str | None = None, min_paintings: int = 30,
                          index: PaintingValueIndex | None = None) -> AspectEpisodes:
    groups = np.asarray(groups)
    index = index or PaintingValueIndex(labels, groups)
    la, lb = index.labels[aspect_a], index.labels[aspect_b]
    lt = index.labels[third] if third else None
    rows = np.asarray(rows, dtype=np.int64)
    known = (la[rows] >= 0) & (lb[rows] >= 0)
    if third:
        known &= lt[rows] >= 0
    pool = rows[known]
    ok_a = eligible_values(la, groups, pool, min_paintings)
    ok_b = eligible_values(lb, groups, pool, min_paintings)
    if len(ok_a) <= NUM_PAIRS or len(ok_b) <= NUM_PAIRS:
        raise RuntimeError(f"{aspect_a} x {aspect_b}: need more than {NUM_PAIRS} eligible values per aspect, "
                           f"got {len(ok_a)} and {len(ok_b)}")
    anchors = pool[np.isin(la[pool], ok_a) & np.isin(lb[pool], ok_b)]
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")}
    failures = 0
    while len(out["anchor"]) < n_episodes:
        anchor = int(anchors[rng.integers(len(anchors))])
        a, b = int(la[anchor]), int(lb[anchor])
        lack_a, lack_b = index.lacks(aspect_a, a)[pool], index.lacks(aspect_b, b)[pool]
        lack_t = index.lacks(third, int(lt[anchor]))[pool] if third else np.ones(len(pool), dtype=bool)
        used = {groups[anchor]}
        try:
            p_a = draw_distinct(rng, pool[(la[pool] == a) & lack_b & lack_t], groups, used, 1)
            p_b = draw_distinct(rng, pool[(lb[pool] == b) & lack_a & lack_t], groups, used, 1)
            negatives = draw_distinct(rng, pool[lack_a & lack_b & lack_t], groups, used, NUM_NEGATIVES)
            base = pool[lack_a & lack_b]
            va = rng.choice([v for v in ok_a if v != a], NUM_PAIRS, replace=False)
            vb = rng.choice([v for v in ok_b if v != b], NUM_PAIRS, replace=False)
            pa_img, pa_txt = _pairs(rng, base, la, lb, va, groups, used)
            pb_img, pb_txt = _pairs(rng, base, lb, la, vb, groups, used)
        except ValueError:
            failures += 1
            if failures > 10 * n_episodes:
                raise RuntimeError(f"{aspect_a} x {aspect_b}: could not fill {n_episodes} episodes "
                                   f"({failures} failed draws); the pools are too small for these constraints")
            continue
        for key, value in zip(out, (anchor, p_a + p_b + negatives, pa_img, pa_txt, pb_img, pb_txt)):
            out[key].append(value)
    return AspectEpisodes(aspect_a, aspect_b, *(np.asarray(out[k], dtype=np.int64) for k in out))


def validate_aspect_episodes(ep: AspectEpisodes, labels: dict, groups: np.ndarray, index: PaintingValueIndex,
                             third: str | None = None) -> None:
    """Assert every rule of spec §5.1 on the final arrays."""
    la, lb = index.labels[ep.aspect_a], index.labels[ep.aspect_b]
    for i in range(len(ep.anchor)):
        r = ep.anchor[i]
        a, b = la[r], lb[r]
        every = np.concatenate([[r], ep.candidates[i], ep.pairs_a_img[i], ep.pairs_a_txt[i], ep.pairs_b_img[i],
                                ep.pairs_b_txt[i]])
        assert len(np.unique(groups[every])) == len(every), f"episode {i}: a painting appears twice"
        assert (la[every] >= 0).all() and (lb[every] >= 0).all(), f"episode {i}: unlabelled row"
        p_a, p_b, neg = ep.candidates[i, 0], ep.candidates[i, 1], ep.candidates[i, 2:]
        assert la[p_a] == a and index.lacks(ep.aspect_b, b)[p_a], f"episode {i}: p_a"
        assert lb[p_b] == b and index.lacks(ep.aspect_a, a)[p_b], f"episode {i}: p_b"
        assert index.lacks(ep.aspect_a, a)[neg].all() and index.lacks(ep.aspect_b, b)[neg].all(), f"episode {i}: neg"
        if third:
            t = index.labels[third][r]
            assert t >= 0 and index.lacks(third, t)[ep.candidates[i]].all(), f"episode {i}: third aspect"
        for share, differ, xs, ys, own in ((la, lb, ep.pairs_a_img[i], ep.pairs_a_txt[i], a),
                                           (lb, la, ep.pairs_b_img[i], ep.pairs_b_txt[i], b)):
            assert (share[xs] == share[ys]).all() and (differ[xs] != differ[ys]).all(), f"episode {i}: pair"
            assert len(np.unique(share[xs])) == NUM_PAIRS and (share[xs] != own).all(), f"episode {i}: pair values"
        examples = np.concatenate([ep.pairs_a_img[i], ep.pairs_a_txt[i], ep.pairs_b_img[i], ep.pairs_b_txt[i]])
        assert index.lacks(ep.aspect_a, a)[examples].all() and index.lacks(ep.aspect_b, b)[examples].all(), \
            f"episode {i}: an example shows the anchor's value"


def episodes_sha256(ep: AspectEpisodes) -> str:
    digest = hashlib.sha256(f"{ep.aspect_a}|{ep.aspect_b}".encode())
    for f in fields(ep)[2:]:
        digest.update(np.ascontiguousarray(getattr(ep, f.name), dtype=np.int64).tobytes())
    return digest.hexdigest()


def concat_episodes(parts: list) -> AspectEpisodes:
    names = {(p.aspect_a, p.aspect_b) for p in parts}
    a, b = next(iter(names)) if len(names) == 1 else ("mixed", "mixed")
    return AspectEpisodes(a, b, *(np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(parts[0])[2:]))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_episodes.py -q`
Expected: 7 passed. If `test_exhausted_pool_raises` hangs instead of raising, the failure counter is not reached:
check that every `draw_distinct` failure path raises `ValueError` and is counted.

- [ ] **Step 5: Commit**

```bash
git add src/eval/aspect_episodes.py src/test/test_aspect_episodes.py
git commit -m "feat(v2): generic value-disjoint aspect episode builder with validator (spec 5.1)"
```

---

### Task 3: Aspect metrics and the clustered bootstrap

**Files:**
- Create: `src/eval/aspect_metrics.py`
- Test: `src/test/test_aspect_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Scores are `dict[str, dict[str, np.ndarray]]`, `scores[cond][dir]` with
  `cond ∈ {"a", "b"}` and `dir ∈ {"i2t", "t2i"}`, each of shape `(n, 13)` with candidates in the episode's column
  order.
- Produces:
  - `first_place(scores, column) -> np.ndarray[float]`;
  - `per_anchor(scores) -> dict[str, np.ndarray]` with keys `r1`, `gain`, `other`, `swap` and `strict` (each
    `(n,)`, averaged over directions);
  - `cluster_bootstrap(values, clusters, n_boot=5000, seed=42) -> dict` (`point`, `ci95`, `n_clusters`; in the
    units of `values`);
  - `summarize(per_anchor_values, clusters) -> dict[str, dict]` (every metric ×100, so in points);
  - `compare(per_a, per_b, clusters, metric) -> dict` (bootstrap of the per-anchor difference ×100).

- [ ] **Step 1: Write the failing tests**

`src/test/test_aspect_metrics.py`:
```python
import numpy as np
import pytest

from src.eval.aspect_metrics import cluster_bootstrap, compare, first_place, per_anchor, summarize


def _scores(a_i2t, b_i2t, a_t2i=None, b_t2i=None):
    a_t2i = a_i2t if a_t2i is None else a_t2i
    b_t2i = b_i2t if b_t2i is None else b_t2i
    return {"a": {"i2t": np.asarray(a_i2t, float), "t2i": np.asarray(a_t2i, float)},
            "b": {"i2t": np.asarray(b_i2t, float), "t2i": np.asarray(b_t2i, float)}}


def test_condition_blind_scorer_has_zero_gain():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(500, 13))
    m = per_anchor(_scores(s, s))                    # identical under both conditions
    assert np.allclose(m["gain"], 0.0) and m["swap"].sum() == 0


def test_perfect_conditional_scorer():
    s_a = np.zeros((10, 13)); s_a[:, 0] = 1.0
    s_b = np.zeros((10, 13)); s_b[:, 1] = 1.0
    m = per_anchor(_scores(s_a, s_b))
    assert np.allclose(m["r1"], 1) and np.allclose(m["gain"], 1) and np.allclose(m["strict"], 1)


def test_aspect_finder_ignoring_condition_gets_half_r1_zero_gain():
    s = np.zeros((10, 13)); s[:, 0] = 1.0; s[5:, 0] = 0.0; s[5:, 1] = 1.0      # always ranks p_a or p_b first
    m = per_anchor(_scores(s, s))
    assert np.isclose(m["r1"].mean(), 0.5) and np.allclose(m["gain"], 0.0)


def test_ties_are_misses():                                               # Review Focus 2
    s = np.zeros((4, 13))
    assert first_place(s, 0).sum() == 0


def test_nonfinite_scores_are_misses():                                   # Review Focus 4
    s = np.zeros((3, 13)); s[:, 0] = 1.0; s[1, 5] = np.nan
    assert first_place(s, 0).tolist() == [1.0, 0.0, 1.0]


def test_cluster_bootstrap_widens_with_clustering():
    rng = np.random.default_rng(1)
    cluster_effect = rng.normal(size=100)
    clusters = np.repeat(np.arange(100), 20)
    values = cluster_effect[clusters] + 0.1 * rng.normal(size=2000)
    clustered = cluster_bootstrap(values, clusters)
    naive = cluster_bootstrap(values, np.arange(2000))
    assert clustered["n_clusters"] == 100
    assert (clustered["ci95"][1] - clustered["ci95"][0]) > 2 * (naive["ci95"][1] - naive["ci95"][0])
    assert np.isclose(clustered["point"], values.mean())


def test_cluster_bootstrap_needs_two_clusters():
    with pytest.raises(ValueError):
        cluster_bootstrap(np.ones(5), np.zeros(5))


def test_summarize_and_compare_in_points():
    s_a = np.zeros((10, 13)); s_a[:, 0] = 1.0
    s_b = np.zeros((10, 13)); s_b[:, 1] = 1.0
    good = per_anchor(_scores(s_a, s_b))
    blind = per_anchor(_scores(s_a, s_a))
    clusters = np.arange(10)
    assert summarize(good, clusters)["r1"]["point"] == pytest.approx(100.0)
    assert compare(good, blind, clusters, "gain")["point"] == pytest.approx(100.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/eval/aspect_metrics.py`**

```python
"""Aspect-episode metrics (CVPR plan spec §5.1, §10): R@1, condition gain, other-aspect rate, swap, strict swap,
and a bootstrap that resamples whole clusters (paintings, species, reference images)."""

import numpy as np

CONDITIONS = ("a", "b")
DIRECTIONS = ("i2t", "t2i")
METRICS = ("r1", "gain", "other", "swap", "strict")


def first_place(scores: np.ndarray, column: int) -> np.ndarray:
    """1.0 where ``column`` scores strictly above every other candidate; ties and non-finite rows are misses."""
    s = np.asarray(scores, dtype=np.float64)
    target = s[:, column:column + 1]
    others = np.delete(s, column, axis=1)
    finite = np.isfinite(s).all(axis=1)
    return ((others < target).all(axis=1) & finite).astype(np.float64)


def per_anchor(scores: dict) -> dict:
    """Per-anchor metrics averaged over the two directions. gain = R@1 - other-aspect rate."""
    acc = {m: [] for m in METRICS}
    for d in DIRECTIONS:
        s_a, s_b = np.asarray(scores["a"][d], float), np.asarray(scores["b"][d], float)
        hit_aa, hit_bb = first_place(s_a, 0), first_place(s_b, 1)
        hit_ab, hit_ba = first_place(s_b, 0), first_place(s_a, 1)     # the other aspect's candidate wins
        r1 = 0.5 * (hit_aa + hit_bb)
        other = 0.5 * (hit_ba + hit_ab)
        finite = np.isfinite(s_a).all(1) & np.isfinite(s_b).all(1)
        swap = ((s_a[:, 0] > s_a[:, 1]) & (s_b[:, 1] > s_b[:, 0]) & finite).astype(np.float64)
        for name, value in (("r1", r1), ("gain", r1 - other), ("other", other), ("swap", swap),
                            ("strict", hit_aa * hit_bb)):
            acc[name].append(value)
    return {m: 0.5 * (v[0] + v[1]) for m, v in acc.items()}


def cluster_bootstrap(values, clusters, n_boot: int = 5000, seed: int = 42, chunk: int = 250) -> dict:
    values = np.asarray(values, dtype=np.float64)
    _, idx = np.unique(np.asarray(clusters), return_inverse=True)
    k = int(idx.max()) + 1
    if k < 2:
        raise ValueError("cluster_bootstrap needs at least two clusters")
    sums = np.bincount(idx, weights=values, minlength=k)
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, n_boot, chunk):
        draws = rng.integers(0, k, size=(min(chunk, n_boot - start), k))
        boots.append(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
    boots = np.concatenate(boots)
    return {"point": float(values.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                    float(np.percentile(boots, 97.5))], "n_clusters": k}


def _points(result: dict) -> dict:
    return {**result, "point": 100 * result["point"], "ci95": [100 * c for c in result["ci95"]]}


def summarize(per_anchor_values: dict, clusters) -> dict:
    return {m: _points(cluster_bootstrap(per_anchor_values[m], clusters)) for m in METRICS}


def compare(per_a: dict, per_b: dict, clusters, metric: str) -> dict:
    return _points(cluster_bootstrap(np.asarray(per_a[metric]) - np.asarray(per_b[metric]), clusters))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_metrics.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/eval/aspect_metrics.py src/test/test_aspect_metrics.py
git commit -m "feat(v2): aspect metrics (condition gain, swap) and cluster bootstrap"
```

---

### Task 4: Agreement rule, core scorers and λ cross-fitting

**Files:**
- Create: `src/model/aspect_rule.py`, `src/eval/aspect_scorers.py`
- Test: `src/test/test_aspect_rule.py`, `src/test/test_aspect_scorers.py`
- Check script: `src/test/20261029_aspect_eval_setup/reproduce_spike.py` (folder created here; `.gitignore` copied)

**Interfaces:**
- Consumes:
  - `src.model.conditioning.conditional_score(query_feat (Q,D), cand_feat (Q,K,D), query_codes (Q,F),
    cand_codes (Q,K,F), weights (Q,F), beta) -> (Q,K)`;
  - `AspectEpisodes` (Task 2) and `per_anchor`, `summarize` (Task 3).
- Produces:
  - in `src/model/aspect_rule.py`:
    - `agreement_weights(sup_img_codes (E,S,F), sup_txt_codes, con_img_codes (E,C,F), con_txt_codes) -> (E,F)`
      (torch);
    - `zscore_rows(x) -> x` (per-row z-score with a zero row for constant rows);
    - `zfuse(cos, term, lam) -> tensor` (torch);
  - in `src/eval/aspect_scorers.py`:
    - `EvalInputs(img, txt, img_codes=None, txt_codes=None)`, where features are L2-normalized on construction;
    - `cosine_scores(inputs, ep) -> scores`;
    - `agreement_term(inputs, ep, uniform=False) -> scores` (factor term only);
    - `fused_scores(cos_scores, term_scores, lam) -> scores`;
    - `LAMBDA_GRID`;
    - `crossfit_lambda(cos_scores, term_scores, parity) -> (scores, picks)`, which picks λ on each parity half by
      the mean of R@1 and condition gain, applies it to the other half, and extends the grid once at the edge;
    - `fixed_beta_scores(inputs, ep, beta, uniform=False) -> scores`, i.e. `β·cos + factor term` (the training-time
      form).

- [ ] **Step 1: Write the failing tests**

`src/test/test_aspect_rule.py`:
```python
import torch

from src.model.aspect_rule import agreement_weights, zfuse, zscore_rows


def test_agreement_weights_pick_coactive_factors_and_normalize():
    sup_i = torch.zeros(1, 4, 3); sup_t = torch.zeros(1, 4, 3)
    sup_i[..., 0] = 1.0; sup_t[..., 0] = 2.0                  # factor 0 co-active in supports
    con_i = torch.zeros(1, 4, 3); con_t = torch.zeros(1, 4, 3)
    con_i[..., 1] = 1.0; con_t[..., 1] = 1.0                  # factor 1 co-active in contrasts
    w = agreement_weights(sup_i, sup_t, con_i, con_t)
    assert torch.allclose(w, torch.tensor([[1.0, 0.0, 0.0]]))


def test_agreement_weights_all_zero_when_no_positive_gap():
    z = torch.zeros(2, 4, 3)
    assert torch.equal(agreement_weights(z, z, z + 1, z + 1), torch.zeros(2, 3))


def test_zfuse_constant_term_falls_back_to_cosine():                  # Review Focus 3
    cos = torch.tensor([[0.3, 0.1, 0.2]])
    term = torch.tensor([[5.0, 5.0, 5.0]])
    out = zfuse(cos, term, 4.0)
    assert torch.isfinite(out).all() and torch.equal(out.argsort(), zscore_rows(cos).argsort())


def test_zfuse_inf_is_term_only_and_zero_is_cos_only():
    cos = torch.tensor([[0.3, 0.1, 0.2]]); term = torch.tensor([[0.0, 1.0, 2.0]])
    assert torch.equal(zfuse(cos, term, float("inf")), zscore_rows(term))
    assert torch.equal(zfuse(cos, term, 0.0), zscore_rows(cos))
```

`src/test/test_aspect_scorers.py`:
```python
import numpy as np

from src.eval.aspect_episodes import build_aspect_episodes
from src.eval.aspect_metrics import per_anchor
from src.eval.aspect_scorers import (
    LAMBDA_GRID, EvalInputs, agreement_term, cosine_scores, crossfit_lambda, fixed_beta_scores,
)


def _world(n_paintings=2500, seed=0):
    """Aspect-block codes: every value of an aspect is a different positive pattern over the SAME 6 factors
    (aspect a uses factors 0-5, aspect b factors 6-11). This is the code shape the agreement rule needs."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_paintings), 2)
    a = rng.integers(0, 6, 2 * n_paintings)
    b = np.repeat(rng.integers(0, 6, n_paintings), 2)
    pattern_a, pattern_b = rng.random((6, 6)) ** 3, rng.random((6, 6)) ** 3
    code = np.concatenate([pattern_a[a], pattern_b[b]], axis=1).astype(np.float32)
    img = code + 0.1 * rng.normal(size=code.shape).astype(np.float32)
    txt = code + 0.1 * rng.normal(size=code.shape).astype(np.float32)
    return {"a": a, "b": b}, groups, img, txt, code


def test_cosine_is_condition_blind_and_agreement_is_not():
    labels, groups, img, txt, code = _world()
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "a", "b", 300, seed=1)
    inputs = EvalInputs(img, txt, code, code)
    assert np.allclose(per_anchor(cosine_scores(inputs, ep))["gain"], 0.0)
    gain = per_anchor(agreement_term(inputs, ep))["gain"].mean()
    assert gain > 0.15                     # controller's simulation on this world: about 0.32
    assert np.allclose(per_anchor(agreement_term(inputs, ep, uniform=True))["gain"], 0.0)
    assert per_anchor(fixed_beta_scores(inputs, ep, 0.3))["gain"].mean() > 0.05


def test_value_onehot_codes_are_blind_under_value_disjoint_conditions():
    """Pins the mechanism: if each value has its own factor, supports showing OTHER values give the anchor's value
    zero weight, so every candidate ties and the condition gain is exactly 0 (simulation 2026-10-03)."""
    labels, groups, img, txt, _ = _world()
    onehot = np.zeros((len(groups), 12), np.float32)
    onehot[np.arange(len(groups)), labels["a"]] = 1.0
    onehot[np.arange(len(groups)), 6 + labels["b"]] = 1.0
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "a", "b", 200, seed=3)
    m = per_anchor(agreement_term(EvalInputs(img, txt, onehot, onehot), ep))
    assert np.allclose(m["gain"], 0.0) and np.allclose(m["r1"], 0.0)


def test_crossfit_returns_grid_picks_and_valid_scores():
    labels, groups, img, txt, code = _world()
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "a", "b", 200, seed=2)
    inputs = EvalInputs(img, txt, code, code)
    scores, picks = crossfit_lambda(cosine_scores(inputs, ep), agreement_term(inputs, ep), np.arange(200) % 2)
    assert set(picks) == {0, 1} and all(p in LAMBDA_GRID + [32.0, 64.0] for p in picks.values())
    assert scores["a"]["i2t"].shape == (200, 13)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_rule.py src/test/test_aspect_scorers.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/model/aspect_rule.py`**

```python
"""The training-free agreement rule for aspect conditions (CVPR plan spec §6) and per-episode z-fusion."""

import torch
from torch import Tensor
from torch.nn import functional as F


def agreement_weights(sup_img: Tensor, sup_txt: Tensor, con_img: Tensor, con_txt: Tensor) -> Tensor:
    """(E,S,F) x4 -> (E,F): ReLU(mean_S img*txt - mean_C img*txt), L1-normalized; an all-zero row stays zero."""
    gap = (sup_img * sup_txt).mean(dim=-2) - (con_img * con_txt).mean(dim=-2)
    w = F.relu(gap)
    total = w.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, w / total.clamp_min(1e-12), torch.zeros_like(w))


def zscore_rows(x: Tensor) -> Tensor:
    """Per-row z-score over candidates (population std); a constant row becomes all zeros."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return torch.where(std > 0, (x - mean) / std.clamp_min(1e-12), torch.zeros_like(x))


def zfuse(cos: Tensor, term: Tensor, lam: float) -> Tensor:
    """z(cos) + lam * z(term); lam = inf means z(term) alone, lam = 0 means z(cos) alone."""
    if lam == float("inf"):
        return zscore_rows(term)
    if lam == 0:
        return zscore_rows(cos)
    return zscore_rows(cos) + lam * zscore_rows(term)
```

- [ ] **Step 4: Implement `src/eval/aspect_scorers.py`**

```python
"""Score matrices for aspect episodes: backbone cosine, the agreement rule on factor codes (and its uniform-weight
control), z-fusion and lambda cross-fitting (CVPR plan spec §5.1, §8)."""

from dataclasses import dataclass

import numpy as np
import torch

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.model.aspect_rule import agreement_weights, zfuse

LAMBDA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float("inf")]
EDGE_EXTENSION = [32.0, 64.0]


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@dataclass
class EvalInputs:
    img: np.ndarray
    txt: np.ndarray
    img_codes: np.ndarray | None = None
    txt_codes: np.ndarray | None = None

    def __post_init__(self):
        self.img, self.txt = _unit(self.img), _unit(self.txt)


def _query_side(inputs, ep, d):
    """(query features, candidate features, query codes, candidate codes) for direction d."""
    cand = ep.candidates
    if d == "i2t":
        return (inputs.img[ep.anchor], inputs.txt[cand],
                None if inputs.img_codes is None else inputs.img_codes[ep.anchor],
                None if inputs.txt_codes is None else inputs.txt_codes[cand])
    return (inputs.txt[ep.anchor], inputs.img[cand],
            None if inputs.txt_codes is None else inputs.txt_codes[ep.anchor],
            None if inputs.img_codes is None else inputs.img_codes[cand])


def cosine_scores(inputs: EvalInputs, ep) -> dict:
    out = {c: {} for c in CONDITIONS}
    for d in DIRECTIONS:
        q, c, _, _ = _query_side(inputs, ep, d)
        s = np.einsum("nd,nkd->nk", q, c)
        for cond in CONDITIONS:
            out[cond][d] = s                                   # identical under both conditions by construction
    return out


def _weights(inputs, ep, cond, uniform):
    si, st, ci, ct, _ = ep.condition(cond)
    if uniform:
        f = inputs.img_codes.shape[1]
        return torch.full((len(ep.anchor), f), 1.0 / f)
    return agreement_weights(_t(inputs.img_codes[si]), _t(inputs.txt_codes[st]), _t(inputs.img_codes[ci]),
                             _t(inputs.txt_codes[ct]))


def agreement_term(inputs: EvalInputs, ep, uniform: bool = False) -> dict:
    """Factor term sum_l w_l q_l c_l with agreement-rule weights (or uniform weights: the condition removed)."""
    out = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        w = _weights(inputs, ep, cond, uniform)
        for d in DIRECTIONS:
            _, _, qc, cc = _query_side(inputs, ep, d)
            out[cond][d] = (w[:, None, :] * _t(qc)[:, None, :] * _t(cc)).sum(-1).numpy()
    return out


def fixed_beta_scores(inputs: EvalInputs, ep, beta: float, uniform: bool = False) -> dict:
    """beta * cos + factor term: the score used in training (spec §6)."""
    cos, term = cosine_scores(inputs, ep), agreement_term(inputs, ep, uniform)
    return {c: {d: beta * cos[c][d] + term[c][d] for d in DIRECTIONS} for c in CONDITIONS}


def fused_scores(cos: dict, term: dict, lam: float) -> dict:
    return {c: {d: zfuse(_t(cos[c][d]), _t(term[c][d]), lam).numpy() for d in DIRECTIONS} for c in CONDITIONS}


def _criterion(scores, rows):
    m = per_anchor({c: {d: scores[c][d][rows] for d in DIRECTIONS} for c in CONDITIONS})
    return 0.5 * (m["r1"].mean() + m["gain"].mean())


def crossfit_lambda(cos: dict, term: dict, parity: np.ndarray):
    """Pick lambda on each parity half by mean(R@1, condition gain), apply it to the other half; one edge extension."""
    parity = np.asarray(parity)
    fused = {lam: fused_scores(cos, term, lam) for lam in LAMBDA_GRID}
    picks = {}
    out = {c: {d: np.empty_like(cos[c][d]) for d in DIRECTIONS} for c in CONDITIONS}
    for half in (0, 1):
        tune, apply = parity == half, parity != half
        best = max(fused, key=lambda lam: _criterion(fused[lam], tune))
        if best == LAMBDA_GRID[-2]:                                  # 16 picked: extend the grid once
            for lam in EDGE_EXTENSION:
                fused[lam] = fused_scores(cos, term, lam)
            best = max(fused, key=lambda lam: _criterion(fused[lam], tune))
        picks[half] = best
        for c in CONDITIONS:
            for d in DIRECTIONS:
                out[c][d][apply] = fused[best][c][d][apply]
    return out, picks
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_rule.py src/test/test_aspect_scorers.py -q`
Expected: 7 passed.

- [ ] **Step 6: Regression against the aspect spike**

The new metrics and scorers must reproduce the spike's stored numbers on the spike's own episodes.

Create the folder `src/test/20261029_aspect_eval_setup/` (copy `.gitignore` from
`src/test/20261023_aspect_episode_spike/`) and the script `reproduce_spike.py`:
```python
"""E0 check: the new aspect modules reproduce the aspect spike's CLIP and SE numbers on the spike's episodes."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.eval.aspect_episodes import AspectEpisodes  # noqa: E402
from src.eval.aspect_metrics import per_anchor  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores, fixed_beta_scores  # noqa: E402

spec = importlib.util.spec_from_file_location("ra", ROOT / "src/test/20261018_affect_factor_learning/run_affect.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
data = ra.load_artelingo(); cache, *_ = ra.grid.load_grid()
prep_record = json.loads((ra.CACHE / "affect_prepare.json").read_text())
z = np.load(ROOT / "src/test/20261023_aspect_episode_spike/results/aspect_episodes.npz")
# spike layout: cands = [p_emo, p_style, negs]; condition emotion = supports P_emo; spike pairs: pe_x image, pe_y caption
ep = AspectEpisodes("emotion", "style", z["anchor"], z["cands"], z["pe_x"], z["pe_y"], z["ps_x"], z["ps_y"])
sl = cache["selection"]
img = ra.sel.masked(data.img_features, sl); txt = ra.sel.masked(data.txt_features, sl)
se_ic, se_tc, _ = ra.model_codes("SE", data, cache, prep_record)
clip = per_anchor(cosine_scores(EvalInputs(img, txt), ep))
se = per_anchor(fixed_beta_scores(EvalInputs(img, txt, se_ic, se_tc), ep, 0.3))
got = {"clip_r1": 100 * clip["r1"].mean(), "se_b03_r1": 100 * se["r1"].mean(), "se_b03_swap": 100 * se["swap"].mean()}
print(json.dumps(got, indent=1))
want = {"clip_r1": 11.13, "se_b03_r1": 10.95, "se_b03_swap": 16.25}
assert all(abs(got[k] - want[k]) < 0.01 for k in want), (got, want)
print("REPRODUCED")
```

Run: `cd /project/CoSiR && OMP_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261029_aspect_eval_setup/reproduce_spike.py`
Expected: `REPRODUCED`, with CLIP 11.13, SE β 0.3 R@1 10.95 and swap 16.25.
- The spike counted R@1 with `tie_aware_rank`, where a tie is half a rank. If a difference beyond 0.01 traces to
  ties, report the tie count and stop; do not change `first_place`.
- The spike's swap value was a mean over directions computed the same way.

- [ ] **Step 7: Commit**

```bash
git add src/model/aspect_rule.py src/eval/aspect_scorers.py src/test/test_aspect_rule.py src/test/test_aspect_scorers.py src/test/20261029_aspect_eval_setup/.gitignore src/test/20261029_aspect_eval_setup/reproduce_spike.py
git commit -m "feat(v2): agreement rule, aspect scorers and lambda cross-fitting; reproduces the aspect spike"
```

---

### Task 5: Held ledger, CUB data and the CUB third aspect

**Files:**
- Create: `docs/superpowers/held_ledger.md`, `src/data/cub.py`, `src/test/20261029_aspect_eval_setup/cub_third_aspect.py`
- Test: `src/test/test_cub.py`

**Interfaces:**
- Produces:
  - `load_cub(root=CUB_ROOT) -> CubData` with fields:
    - `image_ids (N,)`, `paths (N,)`, `species (N,)`;
    - `captions` (`list[list[str]]`, 10 per image);
    - `attributes: dict[str, np.ndarray]` (group name to a per-image value code, −1 when not exactly one value is
      present at certainty ≥ 3);
    - `attribute_values: dict[str, list[str]]`;
  - `zero_shot_split(species, class_list_dir) -> tuple[np.ndarray, np.ndarray]` (train-species and test-species
    image indices, from xlsa17 `trainvalclasses.txt` and `testclasses.txt`);
  - `dev_species(train_species_ids, n_dev=30, seed=42) -> np.ndarray`.

- [ ] **Step 1: Get the zero-shot class lists (needs the user)**

The xlsa17 package is not on disk, and a hook blocks outbound network calls without the user's approval. Ask the
user to run:
```
! mkdir -p /data/SSD/cub/xlsa17 && cd /data/SSD/cub/xlsa17 && curl -sSfL -o xlsa17.zip https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip && unzip -o -j xlsa17.zip 'xlsa17/data/CUB/trainvalclasses.txt' 'xlsa17/data/CUB/testclasses.txt' && wc -l *.txt
```
Expected: `trainvalclasses.txt` with 150 lines and `testclasses.txt` with 50 lines.
- If the URL fails, ask the user for a mirror, then stop this task.
- The rest of E0 continues meanwhile.

- [ ] **Step 2: Write the failing test** (`src/test/test_cub.py`; it builds a tiny fake CUB tree in `tmp_path`)

```python
import numpy as np

from src.data.cub import dev_species, load_cub, zero_shot_split


def _fake_cub(root):
    (root / "CUB_200_2011" / "attributes").mkdir(parents=True)
    base = root / "CUB_200_2011"
    (base / "images.txt").write_text("1 001.A/a1.jpg\n2 001.A/a2.jpg\n3 002.B/b1.jpg\n")
    (base / "image_class_labels.txt").write_text("1 1\n2 1\n3 2\n")
    (base / "classes.txt").write_text("1 001.A\n2 002.B\n")
    (root / "attributes.txt").write_text("1 has_primary_color::red\n2 has_primary_color::blue\n3 has_size::small\n")
    rows = ["1 1 1 4 0.0", "1 2 0 4 0.0", "1 3 1 3 0.0",       # image 1: red only, small
            "2 1 1 4 0.0", "2 2 1 3 0.0", "2 3 0 3 0.0",       # image 2: two colours -> -1
            "3 1 0 4 0.0", "3 2 1 2 0.0", "3 3 1 4 0.0"]       # image 3: blue at certainty 2 -> -1
    (base / "attributes" / "image_attribute_labels.txt").write_text("\n".join(rows) + "\n")
    for cls, name in (("001.A", "a1"), ("001.A", "a2"), ("002.B", "b1")):
        d = root / "captions" / "extracted" / "text_c10" / cls
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.txt").write_text("\n".join(f"caption {i}" for i in range(10)) + "\n")
    (root / "xlsa17").mkdir()
    (root / "xlsa17" / "trainvalclasses.txt").write_text("001.A\n")
    (root / "xlsa17" / "testclasses.txt").write_text("002.B\n")


def test_load_cub_attributes_and_split(tmp_path):
    _fake_cub(tmp_path)
    cub = load_cub(tmp_path)
    assert cub.attributes["has_primary_color"].tolist() == [0, -1, -1]      # red (code 0) only for image 1
    assert cub.attribute_values["has_primary_color"] == ["red", "blue"]      # file order, not alphabetical
    assert len(cub.captions[0]) == 10 and cub.species.tolist() == [1, 1, 2]
    train_idx, test_idx = zero_shot_split(cub.species, tmp_path / "xlsa17", tmp_path / "CUB_200_2011")
    assert train_idx.tolist() == [0, 1] and test_idx.tolist() == [2]
    assert len(dev_species(np.arange(1, 151), n_dev=30, seed=42)) == 30
```

Value codes follow the order of `attributes.txt`, so red = 0 and blue = 1. Image 1 is red only, giving code 0.

- [ ] **Step 3: Run it to verify it fails**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_cub.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/data/cub.py`**

```python
"""CUB-200-2011 with Reed et al. captions, per-image attribute groups and the xlsa17 zero-shot species split."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CUB_ROOT = Path("/data/SSD/cub")
MIN_CERTAINTY = 3                                   # CUB certainty 3 = "probably", 4 = "definitely"


@dataclass(frozen=True)
class CubData:
    image_ids: np.ndarray
    paths: np.ndarray
    species: np.ndarray
    captions: list
    attributes: dict
    attribute_values: dict


def _pairs(path):
    return [line.split(maxsplit=1) for line in Path(path).read_text().splitlines() if line.strip()]


def load_cub(root=CUB_ROOT) -> CubData:
    root = Path(root)
    base = root / "CUB_200_2011"
    images = _pairs(base / "images.txt")
    image_ids = np.array([int(i) for i, _ in images])
    paths = np.array([p for _, p in images], dtype=object)
    species = np.array([int(c) for _, c in _pairs(base / "image_class_labels.txt")])
    attr_file = root / "attributes.txt" if (root / "attributes.txt").exists() else base / "attributes" / "attributes.txt"
    group_of, value_of, values = {}, {}, {}
    for aid, name in _pairs(attr_file):
        group, value = name.split("::")
        group_of[int(aid)] = group
        values.setdefault(group, []).append(value)
        value_of[int(aid)] = values[group].index(value)
    present = {g: [set() for _ in image_ids] for g in values}
    pos = {int(i): k for k, i in enumerate(image_ids)}
    for line in (base / "attributes" / "image_attribute_labels.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        img, aid, is_present, certainty = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        if is_present == 1 and certainty >= MIN_CERTAINTY:
            present[group_of[aid]][pos[img]].add(value_of[aid])
    attributes = {g: np.array([next(iter(s)) if len(s) == 1 else -1 for s in sets], dtype=np.int64)
                  for g, sets in present.items()}
    captions = []
    for p in paths:
        cls, name = Path(p).parent.name, Path(p).stem
        lines = (root / "captions" / "extracted" / "text_c10" / cls / f"{name}.txt").read_text().splitlines()
        captions.append([l.strip() for l in lines if l.strip()])
    return CubData(image_ids, paths, species, captions, attributes, values)


def zero_shot_split(species: np.ndarray, class_list_dir, cub_base=CUB_ROOT / "CUB_200_2011"):
    """Indices of images whose species is in xlsa17 trainvalclasses.txt / testclasses.txt."""
    name_to_id = {name: int(i) for i, name in _pairs(Path(cub_base) / "classes.txt")}
    train = {name_to_id[n.strip()] for n in Path(class_list_dir, "trainvalclasses.txt").read_text().split()}
    test = {name_to_id[n.strip()] for n in Path(class_list_dir, "testclasses.txt").read_text().split()}
    if train & test:
        raise ValueError("zero-shot train and test species overlap")
    return np.flatnonzero(np.isin(species, list(train))), np.flatnonzero(np.isin(species, list(test)))


def dev_species(train_species_ids, n_dev: int = 30, seed: int = 42) -> np.ndarray:
    ids = np.unique(np.asarray(train_species_ids))
    return np.sort(np.random.default_rng(seed).choice(ids, n_dev, replace=False))
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_cub.py -q`
Expected: 1 passed.

- [ ] **Step 6: Write `docs/superpowers/held_ledger.md`**

```markdown
# Held-read ledger (CVPR plan spec §10)

Every read of a final test split is one row. Final scripts check this file and refuse to run a second time.
Budget: 1 main + 1 reserve read per dataset for the CVPR paper.

| # | Date | Dataset and split | Purpose | Episode / data SHA-256 | Script SHA-256 | Report |
|---|---|---|---|---|---|---|
| H1 | 2026-09-29 | ArtELingo held rows, value (label) episodes, seed 42, 1,024 per label | repaired-factor condition eval | emotion e62ab41f…, style 3a58cf9d… | see report | [condition eval](../reports/auto/v2/2026-10-12_candidate_a_condition_eval_repaired_factors.md) |
| H2 | 2026-09-30 | ArtELingo held rows, same episodes as H1 | stage (d) final test | as H1 | see report | [stage (d) final](../reports/auto/v2/2026-10-14_candidate_a_stage_d_final.md) |
| H3 | 2026-10-01 | ArtELingo held rows, value episodes, seed 43, 8,192 per label | affect factor-learning held test | emotion abd1ca38…, style ee87686c… | 11d5c73f… | [affect held](../reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md) |
| H4 | 2026-10-02 | CUB standard test split (5,794 images, all 200 species; includes the 50 zero-shot test species) | backbone-check attribute probes and retrieval (diagnostic, no CoSiR model) | n/a | see report | [backbone check](../reports/auto/v2/2026-10-25_backbone_check.md) |

**CVPR budget status:** ArtELingo 0 of 2 used (aspect episodes are new; H1 to H3 were value episodes); CUB 0 of 2
(H4 disclosed); SemArt 0 of 2; GeneCIS 0 of 2.
```

Check each date and SHA-256 prefix against the linked report before committing, and correct any that differ.

- [ ] **Step 7: CUB third aspect** (spec §11 E0; development species only)

Write `src/test/20261029_aspect_eval_setup/cub_third_aspect.py`. It must:
1. Load `load_cub()`, then `zero_shot_split` with `class_list_dir=/data/SSD/cub/xlsa17`.
2. Restrict to train species and split them by `dev_species`: the 30 dev species are the scoring set, and the other
   120 are the probe-training set.
3. Load CLIP B/32 features from `/data/SSD2/pre_extract/backbone_check/cub/clip/{img.npy,txt.npy,index.json}`.
   Images are in `index.json["image_paths"]` order; `txt` has shape (N, 10, 512); take the mean over the 10 captions.
   Map paths back to CUB image ids via `images.txt`.
4. For each group in (`has_shape`, `has_wing_pattern`, `has_breast_pattern`, `has_wing_color`), fit
   `LogisticRegression(C=1.0, max_iter=2000)` on the 120-species images, separately on image features and on caption
   features (labelled images only). Score accuracy on the 30 dev species and compute the majority-class rate there.
5. Pick the group with the highest `min(image_acc, caption_acc) - majority`. Print a table and write
   `results/cub_third_aspect.json`.

Run: `cd /project/CoSiR && OMP_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261029_aspect_eval_setup/cub_third_aspect.py`
Expected: four rows and one picked group. CUB's three aspects become `has_primary_color`, `has_bill_shape` and the
pick. Record it in the E0 log; the spec §5.2 table is updated in Task 7.

- [ ] **Step 8: Commit**

```bash
git add src/data/cub.py src/test/test_cub.py docs/superpowers/held_ledger.md src/test/20261029_aspect_eval_setup/cub_third_aspect.py
git commit -m "feat(v2): CUB loader with zero-shot split, held-read ledger, CUB third-aspect choice"
```

---

### Task 6: Feature extraction library and the Qwen fidelity check

**Files:**
- Create: `src/data/feature_extract.py`, `src/test/20261029_aspect_eval_setup/qwen_fidelity.py`
- Test: `src/test/test_feature_extract.py`
- Reference (read only): `src/test/20261025_backbone_check/extract.py`

**Interfaces:**
- Produces:
  - `load_encoder(name, device) -> Encoder`, for name `"clip_b32"` or `"qwen3vl_emb_2b"`;
  - `Encoder.encode_images(list[PIL.Image], batch_size) -> np.ndarray` and
    `Encoder.encode_texts(list[str], batch_size) -> np.ndarray` (both float32, L2-normalized);
  - `Encoder.dim: int`;
  - `QWEN_INSTRUCTION_DEFAULT = "Represent the user's input."`.

- [ ] **Step 1: Write the failing test** (CPU, tiny inputs, CLIP only; Qwen is checked in Step 5)

```python
import numpy as np
from PIL import Image

from src.data.feature_extract import load_encoder


def test_clip_encoder_shapes_and_norm():
    enc = load_encoder("clip_b32", device="cpu")
    img = enc.encode_images([Image.new("RGB", (64, 64), (255, 0, 0))] * 2, batch_size=2)
    txt = enc.encode_texts(["a red square", "a blue circle"], batch_size=2)
    assert img.shape == (2, 512) and txt.shape == (2, 512) and enc.dim == 512
    assert np.allclose(np.linalg.norm(img, axis=1), 1, atol=1e-4)
    assert float(img[0] @ txt[0]) > float(img[0] @ txt[1])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_feature_extract.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/data/feature_extract.py`**

Port the two encoders from `src/test/20261025_backbone_check/extract.py`: CLIP from `HFClipLike` and Qwen from
`QwenEmb` (both selected by its `build(name)`), plus the helpers `_l2` and `_batches`.
- **CLIP B/32:** `CLIPModel.get_image_features` and `get_text_features` (`openai/clip-vit-base-patch32`, fp16 on
  CUDA, fp32 on CPU), each L2-normalized. It must match the cached ArtELingo features: Step 5 checks that.
- **Qwen3-VL-Embedding-2B:** the backbone check's recipe:
  - `Qwen3VLForConditionalGeneration` loaded from `Qwen/Qwen3-VL-Embedding-2B` in bf16;
  - chat template with the default instruction `QWEN_INSTRUCTION_DEFAULT`;
  - last-token pooling of the final hidden state, then L2 normalization;
  - `max_pixels = 512 * 32 * 32`.

Set `LD_LIBRARY_PATH` to include `/root/miniconda3/envs/CoSiR/lib/python3.10/site-packages/nvidia/cu13/lib` before
the import, as the backbone check did. Signatures:
```python
QWEN_INSTRUCTION_DEFAULT = "Represent the user's input."


class Encoder:
    name: str
    dim: int

    def encode_images(self, images, batch_size: int = 64) -> np.ndarray: ...
    def encode_texts(self, texts, batch_size: int = 256) -> np.ndarray: ...


def load_encoder(name: str, device: str = "cuda", instruction: str = QWEN_INSTRUCTION_DEFAULT) -> Encoder: ...
```
Copy the working code paths from `extract.py` rather than rewriting them. Keep the `instruction` argument: Task 9's
"aspect named in the instruction" baseline (E10, later plan) reuses it.

- [ ] **Step 4: Run the test**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_feature_extract.py -q`
Expected: 1 passed.

- [ ] **Step 5: Fidelity checks** (GPU, under the lock)

`qwen_fidelity.py` does two checks:
1. **CLIP:** re-encode 64 ArtELingo selection captions and 64 images with `load_encoder("clip_b32")`. Assert cosine
   ≥ 0.999 to the cached `load_artelingo()` rows.
2. **Qwen:** install the official stack into an isolated directory with
   `pip install --target /data/SSD2/pyenvs/qwen_official "transformers==4.57.*" qwen-vl-utils`. Run the model card's
   official embedding code (in a subprocess with `PYTHONPATH=/data/SSD2/pyenvs/qwen_official`) on 50 CUB images and
   50 captions. Compare with `load_encoder("qwen3vl_emb_2b")` on the same inputs.
   - **Pass:** every per-item cosine ≥ 0.98, and the 50×50 image–caption retrieval rankings agree at top-1 on at
     least 48 of 50 queries.
   - **Report:** the min, median and max cosine. If it fails, try `max_pixels` at the model-card default (1,310,720)
     once. If it still fails, record FAIL. Spec §7's fallback (PE-Core L/14) is then decided by the user in Task 7.

Run: `flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python src/test/20261029_aspect_eval_setup/qwen_fidelity.py`
Expected: `CLIP PASS` and `QWEN PASS` (or `QWEN FAIL` with the numbers).

- [ ] **Step 6: Commit**

```bash
git add src/data/feature_extract.py src/test/test_feature_extract.py src/test/20261029_aspect_eval_setup/qwen_fidelity.py
git commit -m "feat(v2): feature extraction library (CLIP B/32, Qwen3-VL-Embedding-2B) with fidelity check"
```

---

### Task 7: E0 report and spec sync

**Files:**
- Create: `docs/reports/auto/v2/2026-10-29_aspect_eval_setup.md`, `src/test/20261029_aspect_eval_setup/20261029_aspect_eval_setup_log.md`
- Modify: `docs/reports/reports_sum.md` (one row); `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` (§5.2 CUB row: the chosen third group; §7: Qwen fidelity outcome)

- [ ] **Step 1: Write the folder log and the report.**
  - The report covers the module design (aspect episodes, condition gain, clustered bootstrap) and the spike
    reproduction (Task 4 Step 6).
  - It also covers the real-data label coverage (Task 1 Step 5), the CUB third-aspect table, the Qwen fidelity
    numbers and the ledger.
  - Baseline beside every number: the spike's stored values for the reproduction, and the majority rate for the CUB
    probes.
- [ ] **Step 2: Update the spec.**
  - §5.2: replace "a third attribute group chosen in E0" with the chosen group and its numbers.
  - §7: add one line with the fidelity result.
  - If Qwen fidelity failed, stop and ask the user to confirm the PE-Core fallback before Task 16.
- [ ] **Step 3: Check and commit**

Run: `/root/miniconda3/envs/CoSiR/bin/python scripts/check_reports_sum.py` (expect OK), then:
```bash
git add docs/reports/auto/v2/2026-10-29_aspect_eval_setup.md docs/reports/reports_sum.md docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md src/test/20261029_aspect_eval_setup/20261029_aspect_eval_setup_log.md
git commit -m "docs(v2): E0 report: aspect evaluation stack, spike reproduction, CUB third aspect, Qwen fidelity"
```

---

### Task 8: Tier-1 raw-feature baselines

**Files:**
- Create: `src/eval/pair_metric_baselines.py`
- Test: `src/test/test_pair_metric_baselines.py`

**Interfaces:**
- Consumes: `EvalInputs`, `cosine_scores` and `CONDITIONS`/`DIRECTIONS` (Tasks 3 and 4); `AspectEpisodes.condition`.
- Produces: every function below returns a term-score dict `scores[cond][dir] -> (n, 13)`, to be fused with the
  cosine by `crossfit_lambda`:
  - `fit_pca_basis(img_rows, txt_rows, r=32, seed=42) -> PcaBasis`, an unsupervised basis fit on training rows:
    per-modality centring, then a shared PCA, whitened;
  - `diag_agreement_term(inputs, ep, relu: bool)`;
  - `bilinear_agreement_term(inputs, ep, basis)`;
  - `kissme_term(inputs, ep, basis)`;
  - `rca_term(inputs, ep, basis)`;
  - `xing_term(inputs, ep, basis, steps=100)`;
  - `wang_term(inputs, ep, steps=50)`;
  - `pair_probe_term(inputs, ep, steps=200, l2=1.0)`;
  - `tip_adapter_term(inputs, ep, gamma=5.0)`;
  - `value_prototype_term(inputs, ep)`.

Definitions. Let x be a pair's image feature and y its caption feature (unit-normalized), q the query feature and c
a candidate feature (opposite modality). S are the support pairs and C the contrast pairs.
- **diag agreement:** `w = mean_S(x⊙y) − mean_C(x⊙y)` (ReLU if `relu`); term = `Σ_k w_k q_k c_k`.
- **bilinear:** in PCA space (x', y', q', c'), `M = mean_S(x' y'ᵀ) − mean_C(x' y'ᵀ)`; term = `q'ᵀ ((M+Mᵀ)/2) c'`.
- **KISSME:** in PCA space with pair differences `δ = x' − y'`, `Σ_S = mean_S(δδᵀ) + I` and `Σ_C = mean_C(δδᵀ) + I`
  (identity shrinkage works because the basis is whitened); `M = Σ_S⁻¹ − Σ_C⁻¹`; term = `−(q'−c')ᵀ M (q'−c')`.
- **RCA:** `W = (Σ_S)^(−1/2)`; term = `cos(W q', W c')`.
- **Xing:** a diagonal metric `a ≥ 0` minimizing `Σ_S a·δ² − log(Σ_C sqrt(a·δ² + 1e-8))`, by projected gradient
  (Adam, lr 0.05, initialized at ones, clamped at 0); term = `−Σ_k a_k (q'−c')_k²`.
- **Wang et al.:** weights `w ∈ R^D` (raw features), initialized at ones, minimizing the mean of
  `softplus(sim(C) − sim(S))` over all support–contrast pairs, with `sim(x, y) = Σ_k w_k² x_k y_k` (Adam, lr 0.01);
  term = `sim(q, c)`.
- **Pair probe:** logistic regression on `z = x⊙y` (D-dim), labels S = 1 and C = 0, L2 penalty `l2·‖β‖²/2`, 200
  gradient steps (lr 0.5, batched over episodes); term = the logit of `q⊙c`.
- **Tip-Adapter:** `A(q,c) = Σ_S exp(−γ(1 − cos(q⊙c, x⊙y))) − Σ_C exp(−γ(1 − cos(q⊙c, x⊙y)))`.
- **Value prototype:** `cos(c, mean of support members in c's modality) − cos(c, mean of contrast members in c's
  modality)`.

- [ ] **Step 1: Write the failing tests** (the synthetic world from Task 4's test, with features carrying the
  aspects)

```python
import numpy as np
import pytest

from src.eval.aspect_episodes import build_aspect_episodes
from src.eval.aspect_metrics import per_anchor
from src.eval.aspect_scorers import EvalInputs
from src.eval import pair_metric_baselines as B


def _world(n_paintings=2500, seed=0):
    """Each aspect occupies its own block of feature dimensions (a: 0-23, b: 24-47); each value is a pattern over
    its block. If the aspects are instead mixed across all dimensions, raw-feature pair rules show no gain
    (controller's simulation on 2026-10-03: 0.00 mixed vs 0.70 to 0.95 block); that is exactly what raw CLIP does."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_paintings), 2)
    a = rng.integers(0, 6, 2 * n_paintings)
    b = np.repeat(rng.integers(0, 6, n_paintings), 2)
    pattern_a = rng.normal(size=(6, 24)).astype(np.float32)
    pattern_b = rng.normal(size=(6, 24)).astype(np.float32)
    x = np.concatenate([pattern_a[a], pattern_b[b]], axis=1)
    img = x + 0.4 * rng.normal(size=x.shape).astype(np.float32)
    txt = x + 0.4 * rng.normal(size=x.shape).astype(np.float32)
    return {"a": a, "b": b}, groups, img, txt


@pytest.fixture(scope="module")
def setup():
    labels, groups, img, txt = _world()
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "a", "b", 200, seed=1)
    inputs = EvalInputs(img, txt)
    basis = B.fit_pca_basis(inputs.img, inputs.txt, r=16)
    return ep, inputs, basis


@pytest.mark.parametrize("name", ["diag", "diag_relu", "bilinear", "kissme", "rca", "xing", "wang", "probe", "tip"])
def test_pair_baselines_use_the_condition(setup, name):
    ep, inputs, basis = setup
    term = {"diag": lambda: B.diag_agreement_term(inputs, ep, relu=False),
            "diag_relu": lambda: B.diag_agreement_term(inputs, ep, relu=True),
            "bilinear": lambda: B.bilinear_agreement_term(inputs, ep, basis),
            "kissme": lambda: B.kissme_term(inputs, ep, basis),
            "rca": lambda: B.rca_term(inputs, ep, basis),
            "xing": lambda: B.xing_term(inputs, ep, basis),
            "wang": lambda: B.wang_term(inputs, ep),
            "probe": lambda: B.pair_probe_term(inputs, ep),
            "tip": lambda: B.tip_adapter_term(inputs, ep)}[name]()
    assert term["a"]["i2t"].shape == (200, 13) and np.isfinite(term["a"]["i2t"]).all()
    assert per_anchor(term)["gain"].mean() > 0.05                 # the condition changes the ranking


def test_value_prototype_has_little_gain_on_aspect_episodes(setup):
    ep, inputs, _ = setup
    assert per_anchor(B.value_prototype_term(inputs, ep))["gain"].mean() < 0.05
```

The `> 0.05` threshold is a sanity floor: in this synthetic world each aspect has its own block of dimensions, so
any estimator that uses the pairs should show some condition gain (the signed diagonal rule reached 0.70 in the
controller's simulation). The value prototype should not, because no support
shows the anchor's value.

- [ ] **Step 2: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_pair_metric_baselines.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/eval/pair_metric_baselines.py`**

```python
"""Tier-1 raw-feature baselines for aspect episodes (CVPR plan spec §8): similarity estimated directly from the 4+4
example pairs, plus few-shot classics adapted to pairs. Each returns a term-score dict scores[cond][dir] (n, 13)."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


@dataclass
class PcaBasis:
    mean_img: np.ndarray
    mean_txt: np.ndarray
    components: np.ndarray     # (r, D)
    scale: np.ndarray          # (r,), so that projections have unit variance on the fit rows

    def project(self, x, modality):
        mean = self.mean_img if modality == "img" else self.mean_txt
        return ((np.asarray(x, np.float32) - mean) @ self.components.T) / self.scale


def fit_pca_basis(img_rows, txt_rows, r: int = 32, seed: int = 42) -> PcaBasis:
    """Unsupervised: centre each modality, PCA on the stacked rows, whiten. Fit on training rows only."""
    mi, mt = img_rows.mean(0), txt_rows.mean(0)
    stacked = np.concatenate([img_rows - mi, txt_rows - mt]).astype(np.float64)
    rng = np.random.default_rng(seed)
    if len(stacked) > 50_000:
        stacked = stacked[rng.choice(len(stacked), 50_000, replace=False)]
    _, s, vt = np.linalg.svd(stacked, full_matrices=False)
    comps = vt[:r].astype(np.float32)
    scale = (s[:r] / np.sqrt(len(stacked) - 1)).astype(np.float32)
    return PcaBasis(mi.astype(np.float32), mt.astype(np.float32), comps, scale)


def _sides(inputs, ep, d):
    if d == "i2t":
        return inputs.img[ep.anchor], inputs.txt[ep.candidates], "img", "txt"
    return inputs.txt[ep.anchor], inputs.img[ep.candidates], "txt", "img"


def _pairs(inputs, ep, cond):
    si, st, ci, ct, _ = ep.condition(cond)
    return inputs.img[si], inputs.txt[st], inputs.img[ci], inputs.txt[ct]   # (n, 4, D) each


def _each(fn):
    """Build scores[cond][dir] by calling fn(cond, d) -> (n, 13) numpy."""
    return {c: {d: fn(c, d) for d in DIRECTIONS} for c in CONDITIONS}


def diag_agreement_term(inputs, ep, relu: bool):
    def fn(cond, d):
        sx, sy, cx, cy = _pairs(inputs, ep, cond)
        w = (sx * sy).mean(1) - (cx * cy).mean(1)
        if relu:
            w = np.maximum(w, 0.0)
        q, c, _, _ = _sides(inputs, ep, d)
        return np.einsum("nd,nkd->nk", q * w, c)
    return _each(fn)


def _proj_pairs(inputs, ep, cond, basis):
    sx, sy, cx, cy = _pairs(inputs, ep, cond)
    p = lambda a, m: basis.project(a.reshape(-1, a.shape[-1]), m).reshape(a.shape[0], a.shape[1], -1)  # noqa: E731
    return p(sx, "img"), p(sy, "txt"), p(cx, "img"), p(cy, "txt")


def _proj_sides(inputs, ep, d, basis):
    q, c, qm, cm = _sides(inputs, ep, d)
    n, k, dim = c.shape
    return basis.project(q, qm), basis.project(c.reshape(-1, dim), cm).reshape(n, k, -1)


def bilinear_agreement_term(inputs, ep, basis):
    def fn(cond, d):
        sx, sy, cx, cy = _proj_pairs(inputs, ep, cond, basis)
        m = np.einsum("nsi,nsj->nij", sx, sy) / sx.shape[1] - np.einsum("nsi,nsj->nij", cx, cy) / cx.shape[1]
        m = 0.5 * (m + m.transpose(0, 2, 1))
        q, c = _proj_sides(inputs, ep, d, basis)
        return np.einsum("ni,nij,nkj->nk", q, m, c)
    return _each(fn)


def _cov(delta):                      # (n, s, r) -> (n, r, r) + I
    r = delta.shape[-1]
    return np.einsum("nsi,nsj->nij", delta, delta) / delta.shape[1] + np.eye(r, dtype=np.float32)


def kissme_term(inputs, ep, basis):
    def fn(cond, d):
        sx, sy, cx, cy = _proj_pairs(inputs, ep, cond, basis)
        m = np.linalg.inv(_cov(sx - sy)) - np.linalg.inv(_cov(cx - cy))
        q, c = _proj_sides(inputs, ep, d, basis)
        diff = q[:, None, :] - c
        return -np.einsum("nki,nij,nkj->nk", diff, m, diff)
    return _each(fn)


def rca_term(inputs, ep, basis):
    def fn(cond, d):
        sx, sy, _, _ = _proj_pairs(inputs, ep, cond, basis)
        vals, vecs = np.linalg.eigh(_cov(sx - sy))
        w = np.einsum("nij,nj,nkj->nik", vecs, 1.0 / np.sqrt(vals), vecs)       # Sigma^(-1/2)
        q, c = _proj_sides(inputs, ep, d, basis)
        qw, cw = np.einsum("nij,nj->ni", w, q), np.einsum("nij,nkj->nki", w, c)
        qw /= np.linalg.norm(qw, axis=-1, keepdims=True)
        cw /= np.linalg.norm(cw, axis=-1, keepdims=True)
        return np.einsum("ni,nki->nk", qw, cw)
    return _each(fn)


def xing_term(inputs, ep, basis, steps: int = 100):
    def fn(cond, d):
        sx, sy, cx, cy = (_t(a) for a in _proj_pairs(inputs, ep, cond, basis))
        ds, dc = (sx - sy) ** 2, (cx - cy) ** 2
        a = torch.ones(ds.shape[0], ds.shape[-1], requires_grad=True)
        opt = torch.optim.Adam([a], lr=0.05)
        for _ in range(steps):
            loss = ((ds * a[:, None]).sum(-1).sum(-1)
                    - torch.log(torch.sqrt((dc * a[:, None]).sum(-1) + 1e-8).sum(-1))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                a.clamp_(min=0.0)
        q, c = (_t(v) for v in _proj_sides(inputs, ep, d, basis))
        return (-(((q[:, None] - c) ** 2) * a.detach()[:, None]).sum(-1)).numpy()
    return _each(fn)


def wang_term(inputs, ep, steps: int = 50):
    def fn(cond, d):
        sx, sy, cx, cy = (_t(a) for a in _pairs(inputs, ep, cond))
        ps, pc = sx * sy, cx * cy                                        # (n, 4, D)
        w = torch.ones(ps.shape[0], ps.shape[-1], requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.01)
        for _ in range(steps):
            sim_s = (ps * (w ** 2)[:, None]).sum(-1)                     # (n, 4)
            sim_c = (pc * (w ** 2)[:, None]).sum(-1)
            loss = F.softplus(sim_c[:, None, :] - sim_s[:, :, None]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        q, c, _, _ = _sides(inputs, ep, d)
        return (_t(q)[:, None] * _t(c) * (w.detach() ** 2)[:, None]).sum(-1).numpy()
    return _each(fn)


def pair_probe_term(inputs, ep, steps: int = 200, l2: float = 1.0):
    def fn(cond, d):
        sx, sy, cx, cy = (_t(a) for a in _pairs(inputs, ep, cond))
        z = torch.cat([sx * sy, cx * cy], dim=1)                          # (n, 8, D)
        y = torch.cat([torch.ones(sx.shape[:2]), torch.zeros(cx.shape[:2])], dim=1)
        beta = torch.zeros(z.shape[0], z.shape[-1], requires_grad=True)
        bias = torch.zeros(z.shape[0], requires_grad=True)
        opt = torch.optim.SGD([beta, bias], lr=0.5)
        for _ in range(steps):
            logits = (z * beta[:, None]).sum(-1) + bias[:, None]
            loss = F.binary_cross_entropy_with_logits(logits, y) + 0.5 * l2 * (beta ** 2).sum(-1).mean() / z.shape[1]
            opt.zero_grad(); loss.backward(); opt.step()
        q, c, _, _ = _sides(inputs, ep, d)
        return ((_t(q)[:, None] * _t(c) * beta.detach()[:, None]).sum(-1) + bias.detach()[:, None]).numpy()
    return _each(fn)


def tip_adapter_term(inputs, ep, gamma: float = 5.0):
    def fn(cond, d):
        sx, sy, cx, cy = _pairs(inputs, ep, cond)
        q, c, _, _ = _sides(inputs, ep, d)
        qc = q[:, None] * c                                               # (n, 13, D)
        qc /= np.linalg.norm(qc, axis=-1, keepdims=True) + 1e-12
        def aff(px, py):
            k = px * py
            k /= np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12
            return np.exp(-gamma * (1.0 - np.einsum("nkd,nsd->nks", qc, k))).sum(-1)
        return aff(sx, sy) - aff(cx, cy)
    return _each(fn)


def value_prototype_term(inputs, ep):
    def fn(cond, d):
        si, st, ci, ct, _ = ep.condition(cond)
        q, c, _, cm = _sides(inputs, ep, d)
        feats = inputs.txt if cm == "txt" else inputs.img
        sup = np.concatenate([feats[si], feats[st]], axis=1).mean(1)
        con = np.concatenate([feats[ci], feats[ct]], axis=1).mean(1)
        sup /= np.linalg.norm(sup, axis=1, keepdims=True)
        con /= np.linalg.norm(con, axis=1, keepdims=True)
        return np.einsum("nkd,nd->nk", c, sup) - np.einsum("nkd,nd->nk", c, con)
    return _each(fn)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_pair_metric_baselines.py -q`
Expected: 10 passed. If one parametrized baseline fails the 0.05 floor on this world, print its gain and check
the sign convention: higher must mean more similar. Never lower the threshold below 0.05; tell the controller.

- [ ] **Step 5: Commit**

```bash
git add src/eval/pair_metric_baselines.py src/test/test_pair_metric_baselines.py
git commit -m "feat(v2): tier-1 raw-feature aspect baselines (agreement, KISSME, RCA, Xing, Wang, probe, Tip-Adapter)"
```

---

### Task 9: E1, Tier-1 baselines on ArtELingo selection aspect episodes

**Files:**
- Create: `src/test/20261030_aspect_baselines/run_baselines.py` (with `.gitignore`); log
  `20261030_aspect_baselines_log.md`; report `docs/reports/auto/v2/2026-10-30_aspect_baselines.md`
- Modify: `docs/reports/reports_sum.md`

**Interfaces:**
- Consumes: Tasks 1 to 4 and 8, plus the SE, C0 and R3 codes via the `run_affect.model_codes` helper (as in Task 4
  Step 6).
- Produces:
  - `results/episodes_seed{42,43}.npz`: per aspect pair, every `AspectEpisodes` field. These are reused by E3.
  - `results/baselines_seed{42,43}.json`: per scorer, the cross-fitted per-anchor metrics summarized with the
    painting-clustered bootstrap, the λ picks, and the per-pair table.
  - `results/per_anchor_seed{42,43}.npz`: per-anchor arrays for every scorer, for paired comparisons in E3.

- [ ] **Step 1: Write the runner.** `run_baselines.py --episodes-seed {42|43} [--n 4096]`:
  1. Load data, splits (Task 1) and labels. Mask features to selection rows, with NaN elsewhere, and assert the mask
     the way `run_aspect.py` does.
  2. For each aspect pair (`emotion`/`style`, `emotion`/`genre`, `style`/`genre`), the third aspect is the remaining
     one. Build `build_aspect_episodes(labels, groups, selection, a, b, n, seed, third=..., index=index)`, run
     `validate_aspect_episodes` on it, and save the arrays and SHA-256 hashes.
  3. Fit `fit_pca_basis` on 60,000 scorer-train rows (`np.random.default_rng(0).choice(scorer_train, 60000,
     replace=False)`). These are training rows, with no labels.
  4. Scorers:
     - `cosine_scores` (backbone only);
     - `diag_agreement_term` (signed, relu), `bilinear`, `kissme`, `rca`, `xing`, `wang`, `pair_probe`,
       `tip_adapter` and `value_prototype`;
     - the agreement rule on SE, C0 and R3 codes, plus SE's uniform-weight control.
  5. Every term goes through `crossfit_lambda(cos, term, parity=np.arange(n_total) % 2)`, computed over the pooled
     episodes of all three pairs.
  6. Metrics: `per_anchor`; clusters = `groups[anchor]` (the anchor painting); `summarize`.
  7. Write the JSON and npz outputs. Print a table: scorer, R@1, condition gain (each with CI), other-aspect rate,
     swap, and the λ picks.
  8. Assert that cosine's condition gain is exactly 0.
The Tier-1 row "agreement rule on unsupervised bases" (PCA, NMF, SpLiCE, SAE: claim K7) belongs to E11 in the
next plan; it is not part of the GO bar.

- [ ] **Step 2: Smoke test.** Run with `--n 64 --episodes-seed 42`. Expected: completes, cosine gain 0.00, all
  scores finite.
- [ ] **Step 3: Launch the full runs.** The controller runs both seeds in the background (CPU-heavy, so set
  `OMP_NUM_THREADS=8`). Expected: about 20 to 40 minutes each.
- [ ] **Step 4: Report.** E1 report with the tables for both seeds and the per-pair breakdown. Name the **best raw
  metric-from-pairs baseline** on seed 42, by the mean of R@1 and condition gain among diag/bilinear/KISSME/RCA/Xing/
  Wang/probe/Tip-Adapter. That baseline is the GO bar in Task 13. Add the reports_sum row and run the checker.
- [ ] **Step 5: Commit**

```bash
git add src/test/20261030_aspect_baselines/.gitignore src/test/20261030_aspect_baselines/run_baselines.py src/test/20261030_aspect_baselines/20261030_aspect_baselines_log.md docs/reports/auto/v2/2026-10-30_aspect_baselines.md docs/reports/reports_sum.md
git commit -m "docs(v2): E1 tier-1 baselines on ArtELingo aspect episodes; names the GO baseline bar"
```

---

### Task 10: E2, pseudo-partitions and the training episode bank

**Files:**
- Create: `src/train/pseudo_partitions.py`, `src/test/20261031_pseudo_partitions/build_partitions.py` (with `.gitignore`), log, report `docs/reports/auto/v2/2026-10-31_pseudo_partitions.md`
- Test: `src/test/test_pseudo_partitions.py`
- Modify: `docs/reports/reports_sum.md`

**Interfaces:**
- Consumes: `build_aspect_episodes`, `concat_episodes`, `PaintingValueIndex` (Task 2).
- Produces:
  - `kmeans_partition(features, rows, k=64, seed=42) -> np.ndarray`: labels over all rows, −1 outside `rows`;
    features are L2-normalized; `MiniBatchKMeans(n_init=3, batch_size=4096)`.
  - `build_episode_bank(partitions: dict[str, np.ndarray], groups, rows, n_per_pair, seed, min_paintings=30) ->
    AspectEpisodes`: one block per unordered partition pair, with the third partition as the third aspect when there
    are exactly 3.
  - Cache `results/partitions.npz` (local scorer-train arrays: `affect`, `image`, `caption`, `local_groups`) and
    `results/bank_<name>.npz` for each partition set in {`AIC` = affect+image+caption, `AI` = affect+image (the
    held-out-genre set), `IC` = image+caption (the supervision ablation)}.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from src.train.pseudo_partitions import build_episode_bank, kmeans_partition


def test_kmeans_partition_scope_and_bank():
    rng = np.random.default_rng(0)
    n_paint = 3000
    groups = np.repeat(np.arange(n_paint), 2)
    feats = rng.normal(size=(2 * n_paint, 8)).astype(np.float32)
    rows = np.arange(0, 2 * n_paint - 200)
    p1 = kmeans_partition(feats, rows, k=8, seed=0)
    assert (p1[rows] >= 0).all() and (p1[len(rows):] == -1).all()
    p2 = kmeans_partition(feats[:, ::-1].copy(), rows, k=8, seed=1)
    p3 = np.repeat(rng.integers(0, 8, n_paint), 2); p3[len(rows):] = -1
    bank = build_episode_bank({"x": p1, "y": p2, "z": p3}, groups, rows, n_per_pair=20, seed=2, min_paintings=5)
    assert len(bank.anchor) == 60 and np.isin(bank.rows(), rows).all()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_pseudo_partitions.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/train/pseudo_partitions.py`**

```python
"""Pseudo-partitions over training rows and the pseudo-aspect episode bank (CVPR plan spec §6). No evaluation label
is used: partitions are k-means clusters of features, or of GoEmotions affect probabilities (distant supervision)."""

from itertools import combinations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.eval.aspect_episodes import PaintingValueIndex, build_aspect_episodes, concat_episodes


def kmeans_partition(features, rows, k: int = 64, seed: int = 42) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    x = np.asarray(features, dtype=np.float32)[rows]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    fit = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096).fit_predict(x)
    labels = np.full(len(features), -1, dtype=np.int64)
    labels[rows] = fit
    return labels


def build_episode_bank(partitions: dict, groups, rows, n_per_pair: int, seed: int,
                       min_paintings: int = 30) -> "AspectEpisodes":
    names = sorted(partitions)
    index = PaintingValueIndex(partitions, groups)
    blocks = []
    for i, (a, b) in enumerate(combinations(names, 2)):
        rest = [n for n in names if n not in (a, b)]
        third = rest[0] if len(rest) == 1 else None
        blocks.append(build_aspect_episodes(partitions, groups, rows, a, b, n_per_pair, seed + i, third=third,
                                            min_paintings=min_paintings, index=index))
    return concat_episodes(blocks)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_pseudo_partitions.py -q`
Expected: 1 passed.

- [ ] **Step 5: Build the real partitions and banks.** `build_partitions.py` works on scorer-train local rows
  `0..n-1`:
  - `affect`: copy `affect_local` from `src/test/20261018_affect_factor_learning/cache/affect_prepare.npz`.
  - `image`: copy `clip_image_local` from `src/test/20261016_factor_learning_grid/cache/grid_prepare.npz`.
  - `local_groups`: copy from the same file.
  - `caption`: the new partition, `kmeans_partition(data.txt_features[scorer_train], np.arange(n), k=64, seed=42)`.
  - The content graph: copy `graph.npz` from the grid cache, asserting shape `(183694, 183694)`.

  Record the SHA-256 of every copied array. Banks (65,536 episodes, `n_per_pair` = 21,846 for three pairs, trimmed
  to 65,536; two-partition sets use one pair of 65,536):
  - `AIC`;
  - `AI` (held-out genre);
  - `IC` (no affect).

  Diagnostic only: the adjusted mutual information (AMI) of each partition with emotion, style and genre on
  scorer-train rows. This is the one place training-row labels are read, as a report-only diagnostic.

  Run in the background (the controller launches it): about 15 to 40 minutes.
- [ ] **Step 6: Report** (E2): partition sizes, the AMI table (emotion vs affect, style vs image, genre vs caption,
  plus the off-diagonals), bank sizes and validity (`validate_aspect_episodes` on 1,000 sampled bank episodes per
  set), and SHA-256 hashes. Add the reports_sum row and run the checker.
- [ ] **Step 7: Commit**

```bash
git add src/train/pseudo_partitions.py src/test/test_pseudo_partitions.py src/test/20261031_pseudo_partitions/.gitignore src/test/20261031_pseudo_partitions/build_partitions.py src/test/20261031_pseudo_partitions/20261031_pseudo_partitions_log.md docs/reports/auto/v2/2026-10-31_pseudo_partitions.md docs/reports/reports_sum.md
git commit -m "feat(v2): pseudo-partitions (caption-content k-means) and pseudo-aspect episode banks AIC/AI/IC (E2)"
```

---

### Task 11: Aspect episode loss in factor training

**Files:**
- Create: `src/train/aspect_loss.py`
- Modify: `src/train/train_factors.py` (config fields, one branch, validation); change log `.claude/<date>_log.md`
- Test: `src/test/test_aspect_loss.py`; existing `src/test/test_train_factors.py` must still pass

**Interfaces:**
- Consumes: `agreement_weights` (Task 4); `conditional_score`; `AspectEpisodes` (Task 2); the existing
  `train_factors` internals (`encode_rows`, the `log_tau` pattern).
- Produces:
  - `aspect_episode_scores(img_feat, txt_feat, img_codes, txt_codes, idx: dict, beta) -> dict[(cond, dir)] -> (E,13)`
    (torch);
  - `aspect_episode_loss(scores, log_tau, lambda_swap) -> Tensor`;
  - new `FactorTrainingConfig` fields (defaults keep current behaviour): `lambda_aspect: float = 0.0`,
    `aspect_episodes_per_step: int = 32`, `aspect_beta: float = 0.3`, `lambda_swap: float = 1.0`;
  - `train_factors(..., aspect_bank=None)`: required exactly when `lambda_aspect > 0`; bank rows are local
    `0..n-1`; incompatible with `lambda_condition > 0`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import torch
from scipy.sparse import csr_matrix

from src.eval.aspect_episodes import build_aspect_episodes
from src.train.aspect_loss import aspect_episode_loss, aspect_episode_scores
from src.train.train_factors import R3_CONFIG, train_factors


def _tiny():
    rng = np.random.default_rng(0)
    n_paint = 600
    groups = np.repeat(np.arange(n_paint), 2)
    a = rng.integers(0, 6, 2 * n_paint)
    b = np.repeat(rng.integers(0, 6, n_paint), 2)
    proto = rng.normal(size=(12, 16)).astype(np.float32)
    img = proto[a] + proto[6 + b] + 0.3 * rng.normal(size=(2 * n_paint, 16)).astype(np.float32)
    txt = proto[a] + proto[6 + b] + 0.3 * rng.normal(size=(2 * n_paint, 16)).astype(np.float32)
    bank = build_aspect_episodes({"a": a, "b": b}, groups, np.arange(2 * n_paint), "a", "b", 256, seed=1,
                                 min_paintings=5)
    nbr = np.arange(2 * n_paint) ^ 1                                  # pair rows of the same painting
    graph = csr_matrix((np.ones(2 * n_paint), (np.arange(2 * n_paint), nbr)), shape=(2 * n_paint, 2 * n_paint))
    return img, txt, groups, bank, graph


def test_loss_is_finite_and_prefers_correct_target():
    s_good = {("a", d): torch.tensor([[2.0] + [0.0] * 12]) for d in ("i2t", "t2i")}
    s_good |= {("b", d): torch.tensor([[0.0, 2.0] + [0.0] * 11]) for d in ("i2t", "t2i")}
    s_bad = {("a", d): torch.tensor([[0.0, 2.0] + [0.0] * 11]) for d in ("i2t", "t2i")}
    s_bad |= {("b", d): torch.tensor([[2.0] + [0.0] * 12]) for d in ("i2t", "t2i")}
    zero = torch.tensor(0.0)
    assert aspect_episode_loss(s_good, zero, 1.0) < aspect_episode_loss(s_bad, zero, 1.0)


def test_train_factors_with_aspect_bank_runs_and_validates():
    img, txt, groups, bank, graph = _tiny()
    import dataclasses
    cfg = dataclasses.replace(R3_CONFIG, epochs=5, batch_size=64, lambda_aspect=1.0, aspect_episodes_per_step=8,
                              num_factors=8, painting_batches=True)
    history = {}
    model, ic, tc = train_factors(img, txt, graph, cfg, device="cpu", group_ids=groups, aspect_bank=bank,
                                  history=history, log_every=1)
    assert np.isfinite(ic).all() and "aspect_loss" in history and len(history["aspect_loss"]) == 5


def test_aspect_bank_required_iff_lambda():
    img, txt, groups, bank, graph = _tiny()
    import dataclasses, pytest
    with pytest.raises(ValueError):
        train_factors(img, txt, graph, dataclasses.replace(R3_CONFIG, epochs=1, lambda_aspect=1.0), device="cpu",
                      group_ids=groups)
    with pytest.raises(ValueError):
        train_factors(img, txt, graph, dataclasses.replace(R3_CONFIG, epochs=1), device="cpu", group_ids=groups,
                      aspect_bank=bank)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_loss.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train.aspect_loss'`.

- [ ] **Step 3: Implement `src/train/aspect_loss.py`**

```python
"""Pseudo-aspect episode loss (CVPR plan spec §6): cross-entropy on the correct target under each condition and
direction, plus a swap term that the target outranks the other aspect's candidate."""

import torch
from torch import Tensor
from torch.nn import functional as F

from src.model.aspect_rule import agreement_weights
from src.model.conditioning import conditional_score

CONDITIONS = ("a", "b")
DIRECTIONS = ("i2t", "t2i")
TARGET = {"a": 0, "b": 1}


def aspect_episode_scores(img_feat, txt_feat, img_codes, txt_codes, idx: dict, beta: float) -> dict:
    """idx holds row-local index tensors: anchor (E,), candidates (E,13), pa_img/pa_txt/pb_img/pb_txt (E,4)."""
    roles = {"a": (idx["pa_img"], idx["pa_txt"], idx["pb_img"], idx["pb_txt"]),
             "b": (idx["pb_img"], idx["pb_txt"], idx["pa_img"], idx["pa_txt"])}
    out = {}
    for cond, (si, st, ci, ct) in roles.items():
        w = agreement_weights(img_codes[si], txt_codes[st], img_codes[ci], txt_codes[ct])
        a, c = idx["anchor"], idx["candidates"]
        out[(cond, "i2t")] = conditional_score(img_feat[a], txt_feat[c], img_codes[a], txt_codes[c], w, beta)
        out[(cond, "t2i")] = conditional_score(txt_feat[a], img_feat[c], txt_codes[a], img_codes[c], w, beta)
    return out


def aspect_episode_loss(scores: dict, log_tau: Tensor, lambda_swap: float) -> Tensor:
    tau = log_tau.exp()
    ce, swap = 0.0, 0.0
    for cond in CONDITIONS:
        t, o = TARGET[cond], 1 - TARGET[cond]
        for d in DIRECTIONS:
            s = scores[(cond, d)] / tau
            target = torch.full((s.shape[0],), t, dtype=torch.long, device=s.device)
            ce = ce + F.cross_entropy(s, target)
            swap = swap + F.softplus(-(s[:, t] - s[:, o])).mean()
    return (ce + lambda_swap * swap) / 4.0
```

- [ ] **Step 4: Modify `src/train/train_factors.py`**

Write the change-log entry first, then make these changes:
1. Add the four config fields after `condition_beta`:
```python
    lambda_aspect: float = 0.0
    aspect_episodes_per_step: int = 32
    aspect_beta: float = 0.3
    lambda_swap: float = 1.0
```
2. Add the imports `from src.train.aspect_loss import aspect_episode_loss, aspect_episode_scores`, and add an
   `aspect_bank=None` parameter to `train_factors` after `condition_source`.
3. Validation, after the existing condition checks:
```python
    if config.lambda_aspect < 0:
        raise ValueError("lambda_aspect must be >= 0")
    if (config.lambda_aspect > 0) != (aspect_bank is not None):
        raise ValueError("pass an aspect_bank exactly when lambda_aspect > 0")
    if config.lambda_aspect > 0 and config.lambda_condition > 0:
        raise ValueError("the value-condition loss and the aspect loss are not combined")
    if aspect_bank is not None and (aspect_bank.rows().min() < 0 or aspect_bank.rows().max() >= len(img_features)):
        raise ValueError("aspect_bank rows must index the training rows (0 .. n-1)")
```
4. Add a helper next to `_episode_tensors`:
```python
def _aspect_tensors(model, img, txt, bank, take: np.ndarray, device):
    parts = (bank.anchor[take][:, None], bank.candidates[take], bank.pairs_a_img[take], bank.pairs_a_txt[take],
             bank.pairs_b_img[take], bank.pairs_b_txt[take])
    table = np.concatenate(parts, axis=1)
    rows, inverse = np.unique(table, return_inverse=True)
    inv = torch.as_tensor(inverse.reshape(table.shape), device=device)
    idx = {"anchor": inv[:, 0], "candidates": inv[:, 1:14], "pa_img": inv[:, 14:18], "pa_txt": inv[:, 18:22],
           "pb_img": inv[:, 22:26], "pb_txt": inv[:, 26:30]}
    img_rows, txt_rows = img[rows].to(device), txt[rows].to(device)
    return img_rows, txt_rows, model.encode_image(img_rows), model.encode_text(txt_rows), idx
```
5. Initialize `log_tau` the same way as the condition branch, from the score std of step-0 episodes, using
   `aspect_rng = np.random.default_rng([config.seed, 2])` and
   `take = aspect_rng.choice(len(aspect_bank.anchor), config.aspect_episodes_per_step, replace=False)`. Concatenate
   all eight score tensors before taking the std.
6. In the epoch loop, after the condition-loss block:
```python
        aspect_loss = None
        if config.lambda_aspect > 0:
            take = first_take if epoch == 1 else aspect_rng.choice(len(aspect_bank.anchor),
                                                                    config.aspect_episodes_per_step, replace=False)
            img_rows, txt_rows, ic, tc, idx = _aspect_tensors(model, img, txt, aspect_bank, take, selected_device)
            aspect_loss = aspect_episode_loss(aspect_episode_scores(img_rows, txt_rows, ic, tc, idx,
                                                                    config.aspect_beta), log_tau, config.lambda_swap)
            loss = loss + config.lambda_aspect * aspect_loss
```
7. In the history block, add:
```python
            if aspect_loss is not None:
                history.setdefault("aspect_loss", []).append(aspect_loss.item())
                history.setdefault("tau", []).append(float(log_tau.detach().exp()))
```

- [ ] **Step 5: Run the new and existing tests**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_loss.py src/test/test_train_factors.py src/test/test_factor_condition_loss.py -q`
Expected: all pass (the three new tests plus every existing one, unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/train/aspect_loss.py src/train/train_factors.py src/test/test_aspect_loss.py .claude/20261003_log.md
git commit -m "feat(v2): pseudo-aspect episode loss in train_factors (lambda_aspect, swap term), default off"
```

---

### Task 12: E3 training grid and selection pick (seed-42 episodes)

**Files:**
- Create: `src/test/20261101_aspect_factor_gonogo/run_gonogo.py` (with `.gitignore`), `PREREGISTRATION.md`, log

**Interfaces:**
- Consumes: Task 10 caches (partitions, graph, banks), Task 9 episodes (`episodes_seed42.npz`,
  `episodes_seed43.npz`), Tasks 3, 4 and 11.
- Produces: `checkpoints/<run>_seed<s>.pt`; `results/select_seed42.json`; `results/picked.json` (`{"run": ...}`).

The grid, at most about 10 runs: seed 42, 2,000 steps, `dataclasses.replace(R3_CONFIG, painting_batches=True, ...)`
(the C0 recipe plus the aspect loss).

| Run | Bank | Changes | Eligible to be picked |
|---|---|---|---|
| A1 | AIC | `lambda_aspect=1, lambda_swap=1` | yes |
| A2 | AIC | A1 with `num_factors=64` | yes |
| A3 | AIC | A1 with `lambda_aspect=3` | yes |
| A4 | AIC | A1 with `lambda_swap=0` | yes |
| A5 | AIC | A1 with `aspect_beta=0.0` | yes |
| A6 | AIC | A1 with `lambda_sparsity=0.0, lambda_decorrelation=0.1` (denser codes, so values can share factors) | yes |
| H1 | AI | A1's settings: the **held-out-genre model** (K8) | no (test only) |
| S1 | IC | A1's settings: the **supervision ablation** (no affect) | no (descriptive) |

- [ ] **Step 1: Write `PREREGISTRATION.md` before any run,** and commit it. Copy spec §6 "Go/no-go" and
  "Held-out-aspect test" verbatim and add:
  - the grid table above;
  - the picking criterion: on seed-42 selection episodes, pooled over the three aspect pairs, maximize the mean of
    cross-fitted R@1 and condition gain among A1 to A6;
  - the GO comparisons (Task 13);
  - the K8 rule;
  - the MLLM probe rule (Task 14);
  - the clustered bootstrap: anchor painting, 5,000 resamples, seed 42.
```bash
git add src/test/20261101_aspect_factor_gonogo/.gitignore src/test/20261101_aspect_factor_gonogo/PREREGISTRATION.md
git commit -m "docs(v2): E3 go/no-go pre-registration (grid, pick rule, GO and K8 rules) before any run"
```
- [ ] **Step 2: Write the runner.**
  - `--train RUN --seed S`: trains on scorer-train rows with the run's bank, saves the checkpoint and history, and
    refuses to overwrite.
  - `--select`: encodes the selection rows of every run (codes NaN elsewhere) and loads `episodes_seed42.npz`. For
    each run it computes `crossfit_lambda(cosine, agreement_term)`, the uniform-weight control, `per_anchor` and
    `summarize`. It writes `select_seed42.json` and `picked.json` by the pre-registered rule.
- [ ] **Step 3: Smoke test.** `--train A1 --seed 42 --smoke` (50 steps), then `--select --smoke` on 64 episodes.
  Expected: finite codes; picked.json written.
- [ ] **Step 4: Launch the grid.** The controller launches the 8 runs in the background under the GPU lock, at most
  3 in parallel (about 10 minutes each).
- [ ] **Step 5: Run `--select`.** Record the pick in the log; it is not reported as a result yet.
- [ ] **Step 6: Commit** the runner and log:
```bash
git add src/test/20261101_aspect_factor_gonogo/run_gonogo.py src/test/20261101_aspect_factor_gonogo/20261101_aspect_factor_gonogo_log.md
git commit -m "feat(v2): E3 aspect-factor grid runner and seed-42 selection pick"
```

---

### Task 13: GO test, held-out aspect and the ablation rows (seed-43 episodes)

**Files:**
- Modify: `src/test/20261101_aspect_factor_gonogo/run_gonogo.py` (add `--gonogo`)

**Interfaces:**
- Consumes: `picked.json`; `episodes_seed43.npz` and `per_anchor_seed43.npz` from Task 9 (the best raw
  metric-from-pairs baseline named in the E1 report); H1 and S1 checkpoints; SE, C0 and R3 codes.
- Produces: `results/gonogo.json`, with these sections:
  - `go`: R@1 and gain of the picked run against backbone-only, the GO baseline and the picked run's uniform-weight
    control, each a clustered paired difference with CI, plus `GO: true|false` and `strong_go: true|false`;
  - `k8`: H1's genre-pair condition gain against H1's uniform-weight control, with the CI and `K8: true|false`;
  - `ablation`: S1 against the picked run on emotion pairs, plus SE, C0 and R3 rows (descriptive).
  - `value_sharing` (diagnostic, using selection-row labels): for each aspect and model, the share of factors
    whose mean code is at least 10% of the factor's maximum mean for at least half of the aspect's values. The
    rule needs values to share factors (Task 4's one-hot test), so this explains a GO or a NO-GO.

- [ ] **Step 1: Implement `--gonogo`.**
  - Every comparison uses `compare(per_a, per_b, clusters=groups[anchor], metric)` on the seed-43 episodes, with
    each method's λ cross-fitted within those episodes (parity), as pre-registered.
  - **GO:** for each of the three comparators, `compare(...)["ci95"][0] > 0` on both `r1` and `gain`.
  - **Strong GO:** the R@1 point is at least 4 points above backbone-only and the gain point at least 4.
  - **K8:** restrict to the episodes whose pair includes genre (`emotion`/`genre`, `style`/`genre`); require
    `compare(H1, H1_uniform)["gain"]["ci95"][0] > 0`.
  - Write a clear verdict block to stdout.
- [ ] **Step 2: Smoke test** on 64 episodes (`--smoke`). Expected: the JSON has every section; no NaN.
- [ ] **Step 3: Run once,** after Task 12's grid. The controller runs it.
- [ ] **Step 4: Commit:**
```bash
git add src/test/20261101_aspect_factor_gonogo/run_gonogo.py
git commit -m "feat(v2): E3 GO test, held-out-genre K8 test and ablation rows on seed-43 episodes"
```

---

### Task 14: In-context MLLM reranker and the early probe

**Files:**
- Create: `src/eval/mllm_reranker.py`, `src/test/20261102_mllm_probe/run_probe.py` (with `.gitignore`), log
- Test: `src/test/test_mllm_reranker.py` (prompt construction only, no model)

**Interfaces:**
- Consumes: `AspectEpisodes`; ArtELingo image paths (`/data/PDD/wikiart_proj/wikiart/<annotation "image">`) and
  captions (`src.data.artelingo.join_captions` with `ANNOTATIONS_PATH`); the metrics from Task 3.
- Produces:
  - `build_messages(query, candidates, supports, contrasts, direction) -> list[dict]`, a chat message list for the
    Qwen3-VL processor;
  - `LETTERS = "ABCDEFGHIJKLM"`;
  - `QwenReranker(model_id="Qwen/Qwen3-VL-2B-Instruct", device="cuda", max_pixels=256*28*28)`, with
    `.score(messages) -> np.ndarray (13,)` (next-token logits of the 13 letters).

The prompt never names the aspect:

> "Each example pair shows an image and a caption of two different artworks that are alike in one respect. The
> counter-example pairs are alike in a different respect. Pick the candidate that is alike to the query in the same
> respect as the example pairs. Answer with one letter."

Then come the example pairs (image + caption), the counter-example pairs, the query (an image for i2t, a caption for
t2i), the candidates labelled A to M (captions for i2t, images for t2i) and "Answer:".

- [ ] **Step 1: Write the failing test** (prompt structure)

```python
from src.eval.mllm_reranker import LETTERS, build_messages


def test_messages_hide_the_aspect_and_label_13_candidates():
    sup = [("img_s%d.jpg" % i, "support caption %d" % i) for i in range(4)]
    con = [("img_c%d.jpg" % i, "contrast caption %d" % i) for i in range(4)]
    msgs = build_messages(("query.jpg", None), ["cand %d" % i for i in range(13)], sup, con, "i2t")
    text = " ".join(part.get("text", "") for m in msgs for part in m["content"] if part["type"] == "text")
    for word in ("emotion", "style", "genre", "colour", "color"):
        assert word not in text.lower()
    assert all(f"{L}." in text for L in LETTERS) and text.rstrip().endswith("Answer:")
    images = [p for m in msgs for p in m["content"] if p["type"] == "image"]
    assert len(images) == 1 + 4 + 4                                  # query + example images
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_mllm_reranker.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/eval/mllm_reranker.py`**

```python
"""In-context MLLM reranker for aspect episodes (CVPR plan spec §6 early probe, §8 tier 2). The model sees the
example and counter-example pairs, the query and 13 lettered candidates; its next-token logits over the letters
are the scores. The aspect is never named."""

import numpy as np
import torch

LETTERS = "ABCDEFGHIJKLM"
INSTRUCTION = ("Each example pair shows an image and a caption of two different artworks that are alike in one "
               "respect. The counter-example pairs are alike in a different respect. Pick the candidate that is "
               "alike to the query in the same respect as the example pairs. Answer with one letter.")


def _pair_block(title, pairs):
    parts = [{"type": "text", "text": title}]
    for i, (image, caption) in enumerate(pairs, 1):
        parts += [{"type": "text", "text": f"Pair {i}: image"}, {"type": "image", "image": image},
                  {"type": "text", "text": f"caption: {caption}"}]
    return parts


def build_messages(query, candidates, supports, contrasts, direction):
    """query = (image_path, None) for i2t or (None, caption) for t2i; candidates are captions (i2t) or image paths
    (t2i); supports / contrasts are lists of (image_path, caption)."""
    content = [{"type": "text", "text": INSTRUCTION}]
    content += _pair_block("Example pairs:", supports)
    content += _pair_block("Counter-example pairs:", contrasts)
    content.append({"type": "text", "text": "Query:"})
    content.append({"type": "image", "image": query[0]} if direction == "i2t"
                   else {"type": "text", "text": query[1]})
    content.append({"type": "text", "text": "Candidates:"})
    for letter, cand in zip(LETTERS, candidates):
        if direction == "i2t":
            content.append({"type": "text", "text": f"{letter}. {cand}"})
        else:
            content += [{"type": "text", "text": f"{letter}."}, {"type": "image", "image": cand}]
    content.append({"type": "text", "text": "Answer:"})
    return [{"role": "user", "content": content}]


class QwenReranker:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", device: str = "cuda",
                 max_pixels: int = 256 * 28 * 28):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()
        tok = self.processor.tokenizer
        self.letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
        self.device = device

    @torch.no_grad()
    def score(self, messages) -> np.ndarray:
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1]
        return logits[self.letter_ids].float().cpu().numpy()
```

When run, confirm on two real prompts that the letter tokens are single tokens and that `apply_chat_template` accepts
`{"type": "image", "image": <path>}`. Adjust to the processor's documented content keys if it raises, and record the
change in the log.

- [ ] **Step 4: Run the test**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_mllm_reranker.py -q`
Expected: 1 passed.

- [ ] **Step 5: Probe runner.** `run_probe.py --n 300 --seed 44`:
  - Build 300 selection episodes per aspect pair with seed 44 (Task 2), validate them, and save them with their
    SHA-256.
  - For each episode, condition (a, b) and direction (i2t, t2i), score with `QwenReranker` into `scores[cond][dir]`.
  - Write `per_anchor` and `summarize` results to `results/probe.json`, alongside cosine on the same episodes.
  - **Pre-registered rule:** the MLLM works if `compare(mllm, cosine)` has a CI lower bound > 0 on both `r1` and
    `gain`.
  - Checkpoint progress every 50 episodes to `results/probe_partial.npz` so a crash resumes.
  - Smoke with `--n 4`, then the controller launches the full run in the background under the GPU lock. Expected
    time: 1 to 3 hours on the 3090.
- [ ] **Step 6: Commit**

```bash
git add src/eval/mllm_reranker.py src/test/test_mllm_reranker.py src/test/20261102_mllm_probe/.gitignore src/test/20261102_mllm_probe/run_probe.py src/test/20261102_mllm_probe/20261102_mllm_probe_log.md
git commit -m "feat(v2): in-context Qwen3-VL reranker and the early MLLM probe (branch 2 vs 3)"
```

---

### Task 15: E3 report and the Oct 9 decision package

**Files:**
- Create: `docs/reports/auto/v2/2026-11-01_aspect_factor_gonogo.md` (includes the MLLM probe), figures under `docs/reports/assets/2026-11-01_aspect_factor_gonogo/`
- Modify: `docs/reports/reports_sum.md`; spec §4 claims table status (K2, K7 bar, K8)

- [ ] **Step 1: Report.**
  - It covers: the grid and the pick (seed 42); the GO verdict with every comparison and CI (seed 43); strong GO;
    K8; the supervision ablation; SE, C0 and R3 rows; the MLLM probe and its verdict.
  - It ends with **which branch the numbers point to** (spec §4) as a recommendation; the user decides.
  - Figures: (a) R@1 against condition gain for every method, with CIs; (b) per-aspect-pair bars.
  - Baseline beside every number: backbone-only and the GO bar.
- [ ] **Step 2: Update the spec's claims table statuses.** Add the reports_sum row, run the checker, and commit:
```bash
git add docs/reports/auto/v2/2026-11-01_aspect_factor_gonogo.md docs/reports/assets/2026-11-01_aspect_factor_gonogo docs/reports/reports_sum.md docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md
git commit -m "docs(v2): E3 go/no-go report with held-out aspect, ablation and MLLM probe; branch recommendation"
```
- [ ] **Step 3: Stop and hand the decision to the user** (spec §4: GO, branch 2 or branch 3). Do not start E6 or
  later until the user has chosen.

---

### Task 16: E4, feature extraction for the later experiments

**Files:**
- Create: `scripts/extract_features.py`
- Test: `src/test/test_extract_features_adapters.py` (adapter listing only, on tiny fixtures)

**Interfaces:**
- Consumes: `load_encoder` (Task 6), `load_cub` (Task 5).
- Produces `/data/SSD2/pre_extract/<dataset>/<backbone>/{img.npy, txt.npy, index.json}` (float16 arrays,
  L2-normalized) for:
  - `artelingo_full` (Qwen only: 61,402 painting images + 308,723 captions; B/32 already cached);
  - `semart` (B/32 and Qwen; captions scrubbed by `scrub_semart`);
  - `coco_train2014` (B/32 and Qwen; 5 captions per image from `captions_train2014.json`; for GeneCIS factor
    training);
  - `genecis_vg_crops` (B/32 and Qwen; the focus-attribute crops made exactly as `/project/genecis/datasets/
    vaw_dataset.py` does, with dilation 0.7 and padding to a square);
  - `genecis_coco` (Qwen; B/32 already in the GeneCIS feature store).

  CUB features for both backbones exist from the backbone check; reuse them.

Adapters, as `scripts/extract_features.py --dataset NAME --backbone {clip_b32,qwen3vl_emb_2b} [--limit N]`:
- **`semart`:** read `semart_{train,val,test}.csv` (tab-separated, latin-1). Description = `DESCRIPTION` passed
  through `scrub_semart(description, author, title, date)`, which removes the author's name tokens, the title string
  and every 3- or 4-digit year. `index.json` keeps the split, type, school and timeframe columns.
- **`coco_train2014`:** images from `/data/SSD/coco/images/train2014` (check the path; the GeneCIS check used
  `/data/SSD/coco/images/val2014`). Captions are grouped by image id.
- **`genecis_vg_crops`:** read `/project/genecis/genecis/focus_attribute.json`. Collect every (image_id, bbox) used
  as reference, target or gallery item, crop with GeneCIS's `load_cropped_image` logic (copy the function and cite
  it), and use the images from `/data/SSD/visual_genome/VG_100K_all`. There is no caption, so `txt.npy` is empty.
  This only prepares features; no template is scored.

- [ ] **Step 1: Write a failing test for `scrub_semart`:**
```python
from scripts.extract_features import scrub_semart


def test_scrub_semart_removes_author_title_years():
    out = scrub_semart("Holbein painted the Darmstadt Madonna in 1526, late in his career.",
                       "HOLBEIN, Hans the Younger", "Darmstadt Madonna", "1526-28")
    assert "Holbein" not in out and "Darmstadt Madonna" not in out and "1526" not in out
```
- [ ] **Step 2: Implement the CLI and adapters.**
  - Each adapter yields `(id, image_path_or_crop, [captions])`.
  - Images are loaded with PIL and converted to RGB.
  - Write in batches with a resume file (`progress.json`).
  - `index.json` maps rows to ids and metadata.
- [ ] **Step 3: Test, then smoke** each adapter with `--limit 32` on CPU or GPU.
- [ ] **Step 4: Launch.** The controller launches the full extractions in the background, local GPU first, under
  the lock. If the local GPU is busy or the queue exceeds about 10 hours, use DAS6 through the `cluster-run` skill:
  sync only the needed image folders to `/local/wding/`, and follow `feedback_cluster-data-sync-unreliable-new-paths`
  (verify with a real job).

  Expected local time: ArtELingo Qwen about 75 + 15 min, COCO train both backbones about 2 h, VG crops about 40 min,
  SemArt about 30 min.
- [ ] **Step 5: Commit** the script and test, plus a short log
  `src/test/20261103_feature_extraction/20261103_feature_extraction_log.md` with counts, timings and the SHA-256 of
  each `index.json`:
```bash
git add scripts/extract_features.py src/test/test_extract_features_adapters.py src/test/20261103_feature_extraction/20261103_feature_extraction_log.md
git commit -m "feat(v2): E4 feature extraction adapters (ArtELingo Qwen, SemArt, COCO train, GeneCIS VG crops)"
```

---

### Task 17: E5, replication seeds (only after a GO)

**Files:**
- Modify: `src/test/20261101_aspect_factor_gonogo/run_gonogo.py` (add `--replicate`)
- Create: `docs/reports/auto/v2/2026-11-04_aspect_factor_replication.md`

- [ ] **Step 1: Train the picked run at seeds 43 and 44** (the controller launches them; about 10 minutes each).
- [ ] **Step 2: `--replicate`.** Evaluate each seed on the seed-43 episodes exactly as `--gonogo` does for the picked
  run. Report:
  - the per-seed R@1 and gain against the same three comparators;
  - the 3-seed mean, with its CI from the clustered bootstrap over per-anchor seed means (spec §10).
- [ ] **Step 3: Report and commit.** The report also covers seed variance and whether every seed clears the GO
  comparisons (descriptive; the GO decision was seed 42's). Add the reports_sum row and run the checker:
```bash
git add src/test/20261101_aspect_factor_gonogo/run_gonogo.py docs/reports/auto/v2/2026-11-04_aspect_factor_replication.md docs/reports/reports_sum.md
git commit -m "docs(v2): E5 replication of the GO winner at seeds 43 and 44"
```

---

### Final step: whole-branch review (user rule `final-review.md`)

After Task 17 (or after Task 15 on a NO-GO), dispatch one final whole-branch review on the most capable model. It
must:
- re-derive the load-bearing numbers from the stored arrays: the spike reproduction, the GO comparisons, K8 and the
  MLLM verdict;
- check the row-scope assertions and the held ledger;
- check that the pre-registration preceded the runs (commit order).

Then run one fix wave and a scoped re-review of the fixes.
