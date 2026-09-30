# CoSiR v2 Candidate A — Stage (d): trained conditional scorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development, with
> Claude Code subagents as implementers and Claude as controller and reviewer. Steps use checkboxes
> (`- [ ]`). The user has pre-approved automatic execution of this plan overnight. Stop only at the
> Task 6 stop point, or for a destructive or irreversible action.

**Goal:** train a scorer that reads a condition from support and contrast examples on **frozen** R3
factors. Select it among five self-generated-condition runs, then test on held-out human-label
episodes whether it uses the condition better than the naive rule, in both retrieval directions.

**Architecture:**
- Three condition sources (factor combinations, CLIP image/caption clusters, Block 1 communities)
  generate multi-positive training episodes from scorer-train rows only.
- A naive-initialized residual interface, a tiny tied per-factor MLP, produces condition weights
  inside the existing `conditional_score`.
- Training uses a multi-positive ranking loss, with an optional swap term.
- Selection runs on a 15% carve-out of the train paintings. The final test is on the untouched
  held split.

**Tech Stack:** Python 3.10, PyTorch, NumPy, scikit-learn 1.6 (`MiniBatchKMeans`), SciPy, pytest.
Conda env `CoSiR`.

**Spec:** `docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-stage-d-design.md`. Read it; it
is the binding authority.

## Global Constraints

- Repo `/project/CoSiR`, branch `main`.
  - Python: `/root/miniconda3/envs/CoSiR/bin/python`.
  - Tests: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q <path>`, from the repo root.
- `seed=42` unless a step names another seed. No `cuml`/`cugraph`.
- **R3 factors are frozen.** Load `src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt`
  (SHA-256 `1c299fc008ca006ffd2de37315ac3557523bf06e0b755d4dcbf70b999b4e453f`) with
  `load_factor_checkpoint`, encode with `encode_rows`, and never update it.
- **Human labels (emotion, art style) are evaluation-only.** They never feed training, a condition
  source, or the episode miner.
- **Held rows are touched only in Task 7.** Val rows are not used at all.
- Mining, clustering, graphs and communities use **scorer-train rows only**.
- No painting repeats inside an episode. Distinctness is keyed on leakage-group ids from
  `leakage_groups`.
- Function/class-formal code in `src/`, with tests. Real runs go in dated folders
  `src/test/yyyymmdd_<name>/`, each with a `yyyymmdd_<name>_log.md` log and a local `.gitignore`
  (`*.npy`, `*.json`, `*.pt`, `*.log`, `cache/`, `checkpoints/`). Nothing cached is committed.
- **Modifying an existing `src/` file:** add an entry in `.claude/yyyymmdd_log.md`. It is
  gitignored, so write it but do not git-add it.
- **Reports:** written to `docs/reports/auto/v2/YYYY-MM-DD_<topic>.md`, with no `cosir_v2_` prefix.
  Continue the v2 date sequence: Task 6 uses `2026-10-13_`, Task 7 uses `2026-10-14_`, and the
  dated test folders use `20261013_` / `20261014_`.
  - Add one row per report to the v2 table in `docs/reports/reports_sum.md`, in the existing format,
    and update its "Start here → Current work" line.
  - Then run `/root/miniconda3/envs/CoSiR/bin/python scripts/check_reports_sum.py`; it must print
    OK.
  - Each report puts a plain-language verdict first. Every number comes from a real run.
- No configuration beyond those named in this plan (the G1-G5 runs, and seeds 43/44 for the
  selected run). No hyperparameter tuning, and no early stopping.
- **Commits:**
  - scope `feat(v2): …`, `fix(v2): …` or `docs(v2): …`;
  - never push;
  - every message ends with these two trailer lines, after a blank line:
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
    `Claude-Session: https://claude.ai/code/session_01VWeYJ1t4s2D4jk4EY73Zht`
- If one run would exceed about 60 minutes, stop and hand back to the controller. Run long jobs in
  the background with a log, and kill any monitor process you start.

## Review Focus

1. **A condition source or the miner returns a row outside the rows it was fitted on.** That would
   leak selection or held rows into training. Every returned id must be a subset of the fit rows.
   Test owned by Tasks 1 and 2.
2. **Degenerate groups:** a factor combination whose top decile is all zeros, a tiny k-means
   cluster, or a small community. These must be skipped or resampled, never used for episodes that
   cannot fill with distinct paintings, and never cause a crash. Test owned by Task 1.
3. **The step-0 scorer differs from the naive rule by rounding.** The whole Δ comparison assumes
   naive is exactly the step-0 model. Pin it with `torch.equal`. Test owned by Task 3.
4. **The wrong-condition control pairs an episode with itself** (no derangement), which would
   silently shrink Δ. Test owned by Task 5.
5. **Human swap episodes carry ambiguous items.** Examples: a `p_style` painting that another
   annotator labelled with the anchor's emotion, or an emotion support that shares the anchor's
   style. Both conditions must stay one-aspect clean. Test owned by Task 5.

---

### Task 1: grouped sub-split, shared distinct-draw helper, and the three condition sources

**Files:**
- Modify: `src/data/splits.py` (add `grouped_subsplit`).
- Create: `src/data/sampling.py`. Move `_draw_distinct` here as the public `draw_distinct`, code
  unchanged.
- Modify: `src/eval/label_episodes.py`. Import `draw_distinct` from `src/data/sampling.py`, and
  keep `_draw_distinct = draw_distinct` as a module alias so behaviour is identical.
- Create: `src/train/condition_sources.py`.
- Tests: `src/test/test_splits.py` (append), `src/test/test_condition_sources.py`.

**Interfaces:**
- Produces:
  - `grouped_subsplit(groups, rows, second_fraction, seed=42) -> tuple[np.ndarray, np.ndarray]`
  - `draw_distinct(rng, pool, keys, used, count) -> list[int]`
  - `Condition(source: str, key: tuple, inside: np.ndarray, outside: np.ndarray)`
  - `FactorComboSource(pair_codes, rows, top_frac=0.10, bottom_frac=0.50, max_factors=3, min_group_rows=200, max_tries=100)`
  - `ClipClusterSource(img_features, txt_features, rows, n_clusters=64, seed=42, min_group_rows=200, max_tries=100)`
  - `CommunitySource(community_labels, rows, min_group_rows=200)`
  - Every source has `.name: str`, `.swap_capable: bool`, `.sample_condition(rng) -> Condition`,
    and `.sample_swap(rng) -> tuple[Condition, Condition]`.
    - `sample_swap` raises `NotImplementedError` when the source is not swap-capable.
    - For swap-capable sources, `len(np.intersect1d(A.inside, B.inside)) >= 20` is guaranteed.

- [ ] **Step 1: Write the failing tests** — `src/test/test_condition_sources.py`

```python
import numpy as np
import pytest

from src.data.sampling import draw_distinct
from src.data.splits import grouped_subsplit
from src.train.condition_sources import ClipClusterSource, CommunitySource, Condition, FactorComboSource


def _blobs(n_per=400, dim=16, seed=0):
    """4 well-separated blobs in image space and 4 differently arranged blobs in caption space."""
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[:4] * 10
    img_lab = np.repeat(np.arange(4), n_per)
    txt_lab = (img_lab + np.tile(np.arange(4), n_per)) % 4          # caption blob differs from image blob
    img = centers[img_lab] + rng.normal(size=(4 * n_per, dim))
    txt = centers[txt_lab][:, ::-1] + rng.normal(size=(4 * n_per, dim))
    return img.astype(np.float32), txt.astype(np.float32), img_lab, txt_lab


def test_grouped_subsplit_keeps_groups_whole_and_hits_fraction():
    groups = np.repeat(np.arange(3000), 3)
    rows = np.arange(0, 9000, 2)                                    # a subset, like split.train
    first, second = grouped_subsplit(groups, rows, 0.15, seed=42)
    assert not set(groups[first]) & set(groups[second])
    assert set(first) | set(second) == set(rows) and not set(first) & set(second)
    assert abs(len(second) / len(rows) - 0.15) < 0.01
    again = grouped_subsplit(groups, rows, 0.15, seed=42)
    assert np.array_equal(again[1], second)
    with pytest.raises(ValueError):
        grouped_subsplit(groups, rows, 1.0)


def test_draw_distinct_never_repeats_a_key():
    rng = np.random.default_rng(0)
    keys = np.repeat(np.arange(50), 4)
    used: set = set()
    picked = draw_distinct(rng, np.arange(200), keys, used, 30)
    assert len({keys[r] for r in picked}) == 30 and used == {keys[r] for r in picked}


def test_factor_combo_conditions_are_disjoint_nonzero_and_inside_fit_rows():
    rng = np.random.default_rng(1)
    codes = np.maximum(0.0, rng.normal(size=(5000, 8)) - 0.3).astype(np.float32)
    codes[:, 7] = 0.0                                               # a dead factor: its top decile is all zero
    rows = np.arange(0, 5000, 2)
    src = FactorComboSource(codes, rows, min_group_rows=50)
    for _ in range(30):
        c = src.sample_condition(rng)
        assert isinstance(c, Condition) and c.source == "factor_combo"
        assert set(c.inside) <= set(rows) and set(c.outside) <= set(rows)
        assert not set(c.inside) & set(c.outside)
        factors, weights = np.array(c.key[0]), np.array(c.key[1])
        score_in = codes[c.inside][:, factors] @ weights
        score_out = codes[c.outside][:, factors] @ weights
        assert (score_in > 0).all() and score_in.min() >= score_out.max()
    a, b = src.sample_swap(rng)
    assert not set(a.key[0]) & set(b.key[0])
    assert len(np.intersect1d(a.inside, b.inside)) >= 20


def test_factor_combo_raises_when_no_valid_condition_exists():
    codes = np.zeros((1000, 4), dtype=np.float32)
    with pytest.raises(RuntimeError):
        FactorComboSource(codes, np.arange(1000), min_group_rows=10, max_tries=5).sample_condition(
            np.random.default_rng(0))


def test_clip_clusters_recover_blobs_and_swap_uses_both_views():
    img, txt, img_lab, _ = _blobs()
    rows = np.arange(len(img))
    src = ClipClusterSource(img, txt, rows, n_clusters=4, seed=42, min_group_rows=50)
    rng = np.random.default_rng(2)
    c = src.sample_condition(rng)
    assert c.key[0] in ("image", "caption") and set(c.inside) <= set(rows)
    if c.key[0] == "image":
        assert len(np.unique(img_lab[c.inside])) == 1               # a recovered blob
    a, b = src.sample_swap(rng)
    assert a.key[0] == "image" and b.key[0] == "caption"
    assert len(np.intersect1d(a.inside, b.inside)) >= 20
    assert not set(a.inside) & set(a.outside)


def test_sources_never_return_rows_outside_the_fit_rows():
    img, txt, _, _ = _blobs()
    rows = np.arange(0, len(img), 3)
    src = ClipClusterSource(img, txt, rows, n_clusters=4, seed=42, min_group_rows=20)
    rng = np.random.default_rng(3)
    for _ in range(20):
        c = src.sample_condition(rng)
        assert set(c.inside) <= set(rows) and set(c.outside) <= set(rows)


def test_community_source_skips_small_groups_and_cannot_swap():
    labels = np.full(3000, -1)
    rows = np.arange(0, 3000, 2)
    labels[rows] = np.repeat(np.arange(5), len(rows) // 5 + 1)[: len(rows)]
    labels[rows[:10]] = 99                                          # a tiny community
    src = CommunitySource(labels, rows, min_group_rows=50)
    rng = np.random.default_rng(4)
    keys = {src.sample_condition(rng).key for _ in range(50)}
    assert ("community", 99) not in keys
    assert src.swap_capable is False
    with pytest.raises(NotImplementedError):
        src.sample_swap(rng)
```

Append to `src/test/test_splits.py`:

```python
def test_grouped_subsplit_rejects_empty_parts():
    import pytest
    from src.data.splits import grouped_subsplit
    with pytest.raises(ValueError):
        grouped_subsplit(np.zeros(10, dtype=int), np.arange(10), 0.5)   # one group cannot be split
```

- [ ] **Step 2: Run to verify failure**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_sources.py src/test/test_splits.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.sampling'` (and
`src.train.condition_sources`).

- [ ] **Step 3: Implement.**

`src/data/splits.py`, append:

```python
def grouped_subsplit(groups: np.ndarray, rows: np.ndarray, second_fraction: float,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split ``rows`` into (first, second) by whole leakage groups; ``second`` gets ~second_fraction of rows."""
    if not 0.0 < second_fraction < 1.0:
        raise ValueError("second_fraction must be in (0, 1)")
    rows = np.asarray(rows, dtype=np.int64)
    unique, inverse, counts = np.unique(np.asarray(groups)[rows], return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    shares = counts[order] / counts.sum()
    start = np.cumsum(shares) - shares
    in_second = np.empty(len(unique), dtype=bool)
    in_second[order] = start >= 1.0 - second_fraction
    mask = in_second[inverse]
    first, second = np.sort(rows[~mask]), np.sort(rows[mask])
    if len(first) == 0 or len(second) == 0:
        raise ValueError("A sub-split part is empty; too few groups for this fraction")
    return first, second
```

`src/data/sampling.py`: move the body of `_draw_distinct` from `src/eval/label_episodes.py`
verbatim into a public function:

```python
"""Sampling helpers shared by label episodes and condition episodes."""

import numpy as np


def draw_distinct(rng, pool, keys, used, count):
    """Draw `count` rows from `pool` whose keys (leakage groups / paintings) are not yet in `used`.

    Updates `used`. Rejection sampling first, then an exact fallback over one row per unused key.
    Raises ValueError when `pool` cannot supply `count` distinct keys.
    """
    if count > 0 and len(pool) == 0:
        raise ValueError("Not enough distinct paintings to fill an episode")
    picked = []
    for _ in range(50 * count):
        if len(picked) == count:
            break
        row = int(pool[rng.integers(len(pool))])
        if keys[row] not in used:
            used.add(keys[row])
            picked.append(row)
    if len(picked) < count:
        eligible = pool[~np.isin(keys[pool], list(used))]
        _, first = np.unique(keys[eligible], return_index=True)
        eligible = eligible[np.sort(first)]
        if len(eligible) < count - len(picked):
            raise ValueError("Not enough distinct paintings to fill an episode")
        for row in rng.choice(eligible, count - len(picked), replace=False):
            used.add(keys[row])
            picked.append(int(row))
    return picked
```

In `src/eval/label_episodes.py`, delete the old function body and add
`from src.data.sampling import draw_distinct` plus `_draw_distinct = draw_distinct`. Leave the
call sites unchanged. The existing label-episode tests must pass unchanged. This is a pure move,
with identical RNG consumption.

`src/train/condition_sources.py`:

```python
"""Self-generated condition sources for stage (d) (spec §3).

A source defines groups of rows that "share a condition". Every group is built only from the rows the
source was fitted on (scorer-train rows): no human label and no row outside ``rows`` is ever used.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans

MIN_SWAP_OVERLAP = 20


@dataclass(frozen=True)
class Condition:
    source: str
    key: tuple
    inside: np.ndarray            # sorted global row ids sharing the condition (subset of fit rows)
    outside: np.ndarray           # sorted global row ids clearly lacking it (subset of fit rows)


class FactorComboSource:
    """Condition = high on a random positive mix of 1-3 R3 factors (top decile vs bottom half)."""

    name = "factor_combo"
    swap_capable = True

    def __init__(self, pair_codes, rows, top_frac=0.10, bottom_frac=0.50, max_factors=3,
                 min_group_rows=200, max_tries=100):
        self.rows = np.sort(np.asarray(rows, dtype=np.int64))
        self.fit_codes = np.asarray(pair_codes, dtype=np.float32)[self.rows]
        self.num_factors = self.fit_codes.shape[1]
        self.top_frac, self.bottom_frac, self.max_factors = top_frac, bottom_frac, max_factors
        self.min_group_rows, self.max_tries = min_group_rows, max_tries

    def _random_factors(self, rng, exclude=()):
        available = np.setdiff1d(np.arange(self.num_factors), np.asarray(exclude, dtype=np.int64))
        k = int(rng.integers(1, min(self.max_factors, len(available)) + 1))
        return np.sort(rng.choice(available, k, replace=False)), rng.dirichlet(np.ones(k))

    def _condition(self, factors, weights):
        score = self.fit_codes[:, factors] @ weights
        hi = np.quantile(score, 1.0 - self.top_frac)
        lo = np.quantile(score, self.bottom_frac)
        inside_mask = (score >= hi) & (score > 0)
        outside_mask = (score <= lo) & ~inside_mask
        if inside_mask.sum() < self.min_group_rows or outside_mask.sum() < self.min_group_rows:
            return None
        key = (tuple(int(f) for f in factors), tuple(round(float(w), 6) for w in weights))
        return Condition(self.name, key, self.rows[inside_mask], self.rows[outside_mask])

    def sample_condition(self, rng) -> Condition:
        for _ in range(self.max_tries):
            condition = self._condition(*self._random_factors(rng))
            if condition is not None:
                return condition
        raise RuntimeError("FactorComboSource found no valid condition in max_tries")

    def sample_swap(self, rng) -> tuple[Condition, Condition]:
        for _ in range(self.max_tries):
            a = self.sample_condition(rng)
            for _ in range(self.max_tries):
                b = self._condition(*self._random_factors(rng, exclude=a.key[0]))
                if b is not None and len(np.intersect1d(a.inside, b.inside)) >= MIN_SWAP_OVERLAP:
                    return a, b
        raise RuntimeError("FactorComboSource found no valid swap pair in max_tries")


class _PartitionSource:
    """Condition = one group of a partition (per view); outside = the other groups of that view."""

    swap_capable = False

    def _setup(self, labels_by_view: dict, rows, min_group_rows, max_tries):
        self.rows = np.sort(np.asarray(rows, dtype=np.int64))
        self.labels_by_view = {v: np.asarray(l, dtype=np.int64) for v, l in labels_by_view.items()}
        self.max_tries = max_tries
        self._members, self._outside = {}, {}
        for view, labels in self.labels_by_view.items():
            fit_labels = labels[self.rows]
            for group in np.unique(fit_labels):
                if group < 0:
                    continue
                inside = self.rows[fit_labels == group]
                if len(inside) >= min_group_rows and len(self.rows) - len(inside) >= min_group_rows:
                    self._members[(view, int(group))] = inside
        if not self._members:
            raise RuntimeError(f"{self.name}: no group has at least {min_group_rows} rows")
        self.valid_keys = sorted(self._members)

    def _condition(self, key) -> Condition:
        if key not in self._outside:
            self._outside[key] = np.setdiff1d(self.rows, self._members[key], assume_unique=True)
        return Condition(self.name, key, self._members[key], self._outside[key])

    def sample_condition(self, rng) -> Condition:
        return self._condition(self.valid_keys[int(rng.integers(len(self.valid_keys)))])

    def sample_swap(self, rng) -> tuple[Condition, Condition]:
        raise NotImplementedError(f"{self.name} has one group per item; it cannot form swap pairs")


class ClipClusterSource(_PartitionSource):
    """k-means on L2-normalized raw CLIP image features and, separately, caption features."""

    name = "clip_cluster"
    swap_capable = True

    def __init__(self, img_features, txt_features, rows, n_clusters=64, seed=42, min_group_rows=200,
                 max_tries=100):
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        labels_by_view = {}
        for view, feats in (("image", img_features), ("caption", txt_features)):
            x = np.asarray(feats, dtype=np.float32)[rows]
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            fit = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3,
                                  batch_size=4096).fit_predict(x)
            labels = np.full(len(feats), -1, dtype=np.int64)
            labels[rows] = fit
            labels_by_view[view] = labels
        self._setup(labels_by_view, rows, min_group_rows, max_tries)

    def sample_swap(self, rng) -> tuple[Condition, Condition]:
        for _ in range(self.max_tries):
            anchor = int(self.rows[rng.integers(len(self.rows))])
            a_key = ("image", int(self.labels_by_view["image"][anchor]))
            b_key = ("caption", int(self.labels_by_view["caption"][anchor]))
            if a_key in self._members and b_key in self._members:
                a, b = self._condition(a_key), self._condition(b_key)
                if len(np.intersect1d(a.inside, b.inside)) >= MIN_SWAP_OVERLAP:
                    return a, b
        raise RuntimeError("ClipClusterSource found no valid swap pair in max_tries")


class CommunitySource(_PartitionSource):
    """Block 1 Stage-1 communities (labels computed on scorer-train rows; -1 elsewhere)."""

    name = "community"

    def __init__(self, community_labels, rows, min_group_rows=200, max_tries=100):
        self._setup({"community": community_labels}, rows, min_group_rows, max_tries)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_sources.py src/test/test_splits.py src/test/test_label_episodes.py`
Expected: all pass. The label-episode tests are unchanged and still pass after the move. Then run
the full suite once: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/`.

- [ ] **Step 5: Change log and commit.** Write `.claude/20261013_log.md` entries for `src/data/splits.py`
  and `src/eval/label_episodes.py`.

```bash
git add src/data/splits.py src/data/sampling.py src/eval/label_episodes.py src/train/condition_sources.py src/test/test_condition_sources.py src/test/test_splits.py
git commit -m "feat(v2): stage-(d) condition sources, grouped sub-split, shared distinct-draw helper"
```

---

### Task 2: the condition-episode miner (multi-positive episodes and swap episodes)

**Files:**
- Create: `src/train/condition_episodes.py`
- Test: `src/test/test_condition_episodes.py`

**Interfaces:**
- Consumes (Task 1): `Condition`, the sources' `sample_condition` / `sample_swap` / `swap_capable`,
  and `draw_distinct`.
- Produces:
  - `pair_feature_units(img_features, txt_features) -> np.ndarray`: L2-normalized
    `0.5·(img+txt)`, float32.
  - `ConditionEpisodes(anchor (E,), supports (E,4), contrasts (E,4), candidates (E,16), positive_mask (E,16) bool)`:
    candidates are 4 positives, then 6 hard negatives, then 6 random negatives.
  - `SwapEpisodes(anchor (E,), supports_a, contrasts_a, supports_b, contrasts_b (E,4), candidates (E,16), positive_mask_a, positive_mask_b (E,16) bool)`:
    candidates are 3 A-only, then 3 B-only, then 5 hard and 5 random negatives outside both.
  - `mine_condition_episodes(source, units, keys, n_episodes, rng, episodes_per_condition=4, num_support=4, num_contrast=4, num_positive=4, num_hard=6, num_random=6, hard_pool=2048, max_failures=1000) -> ConditionEpisodes`
  - `mine_swap_episodes(source, units, keys, n_episodes, rng, episodes_per_pair=4, num_support=4, num_contrast=4, num_each=3, num_hard=5, num_random=5, hard_pool=2048, max_failures=1000) -> SwapEpisodes`

- [ ] **Step 1: Write the failing tests** — `src/test/test_condition_episodes.py`

```python
import numpy as np
import pytest

from src.train.condition_episodes import (
    ConditionEpisodes, SwapEpisodes, mine_condition_episodes, mine_swap_episodes, pair_feature_units,
)
from src.train.condition_sources import ClipClusterSource, CommunitySource, FactorComboSource


def _world(n=4000, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(n, dim)).astype(np.float32)
    txt = (img + 0.3 * rng.normal(size=(n, dim))).astype(np.float32)
    codes = np.maximum(0.0, rng.normal(size=(n, 8)) - 0.2).astype(np.float32)
    keys = np.arange(n) // 2                                        # two annotation rows per painting
    return img, txt, codes, keys


def _all_rows(ep, i, fields):
    out = []
    for f in fields:
        v = getattr(ep, f)[i]
        out.extend(np.atleast_1d(v).tolist())
    return out


def test_condition_episodes_roles_and_distinct_paintings():
    img, txt, codes, keys = _world()
    rows = np.arange(0, 4000)
    src = FactorComboSource(codes, rows, min_group_rows=100)
    units = pair_feature_units(img, txt)
    ep = mine_condition_episodes(src, units, keys, 40, np.random.default_rng(1))
    assert isinstance(ep, ConditionEpisodes) and ep.candidates.shape == (40, 16)
    assert ep.positive_mask[:, :4].all() and not ep.positive_mask[:, 4:].any()
    for i in range(40):
        members = _all_rows(ep, i, ("anchor", "supports", "contrasts", "candidates"))
        assert len({keys[r] for r in members}) == len(members)      # no painting twice


def test_positives_inside_negatives_outside_the_same_condition():
    img, txt, codes, keys = _world()
    labels = np.repeat(np.arange(8), 500)
    rows = np.arange(4000)
    src = CommunitySource(labels, rows, min_group_rows=100)
    ep = mine_condition_episodes(src, pair_feature_units(img, txt), keys, 30, np.random.default_rng(2))
    for i in range(30):
        group = labels[ep.anchor[i]]
        assert (labels[ep.supports[i]] == group).all()
        assert (labels[ep.candidates[i, :4]] == group).all()
        assert (labels[ep.contrasts[i]] != group).all()
        assert (labels[ep.candidates[i, 4:]] != group).all()


def test_hard_negatives_are_the_nearest_outside_items_when_the_pool_is_exhaustive():
    img, txt, codes, _ = _world(n=1200)
    keys = np.arange(1200)                                          # one row per painting: exact top-6
    labels = np.repeat(np.arange(4), 300)
    rows = np.arange(1200)
    units = pair_feature_units(img, txt)
    src = CommunitySource(labels, rows, min_group_rows=50)
    ep = mine_condition_episodes(src, units, keys, 10, np.random.default_rng(3), hard_pool=10_000)
    for i in range(10):
        anchor, hard = ep.anchor[i], ep.candidates[i, 4:10]
        used = {keys[r] for r in _all_rows(ep, i, ("anchor", "supports", "contrasts")) + ep.candidates[i, :4].tolist()}
        outside = rows[(labels != labels[anchor]) & ~np.isin(keys, list(used))]
        sims = units[outside] @ units[anchor]
        best = np.sort(sims)[::-1][:6]
        assert np.allclose(np.sort(units[hard] @ units[anchor])[::-1], best, atol=1e-6)


def test_miner_only_uses_fit_rows():
    img, txt, codes, keys = _world()
    rows = np.arange(0, 4000, 2)
    src = FactorComboSource(codes, rows, min_group_rows=50)
    ep = mine_condition_episodes(src, pair_feature_units(img, txt), keys, 20, np.random.default_rng(4))
    used = np.concatenate([ep.anchor, ep.supports.ravel(), ep.contrasts.ravel(), ep.candidates.ravel()])
    assert set(used.tolist()) <= set(rows.tolist())


def test_swap_episodes_have_disjoint_roles_and_anchor_in_both():
    img, txt, codes, keys = _world()
    rows = np.arange(4000)
    src = FactorComboSource(codes, rows, min_group_rows=100)
    rng = np.random.default_rng(5)
    ep = mine_swap_episodes(src, pair_feature_units(img, txt), keys, 20, rng)
    assert isinstance(ep, SwapEpisodes) and ep.candidates.shape == (20, 16)
    assert ep.positive_mask_a[:, :3].all() and not ep.positive_mask_a[:, 3:].any()
    assert ep.positive_mask_b[:, 3:6].all() and not ep.positive_mask_b[:, :3].any() and not ep.positive_mask_b[:, 6:].any()
    for i in range(20):
        members = _all_rows(ep, i, ("anchor", "supports_a", "contrasts_a", "supports_b", "contrasts_b", "candidates"))
        assert len({keys[r] for r in members}) == len(members)


def test_swap_mining_refuses_a_non_swap_source():
    img, txt, codes, keys = _world()
    src = CommunitySource(np.repeat(np.arange(8), 500), np.arange(4000), min_group_rows=100)
    with pytest.raises(ValueError):
        mine_swap_episodes(src, pair_feature_units(img, txt), keys, 4, np.random.default_rng(6))


def test_mining_is_deterministic_for_a_seed():
    img, txt, codes, keys = _world()
    src = FactorComboSource(codes, np.arange(4000), min_group_rows=100)
    units = pair_feature_units(img, txt)
    a = mine_condition_episodes(src, units, keys, 12, np.random.default_rng(7))
    b = mine_condition_episodes(src, units, keys, 12, np.random.default_rng(7))
    assert np.array_equal(a.candidates, b.candidates) and np.array_equal(a.supports, b.supports)
```

- [ ] **Step 2: Run to verify failure**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_episodes.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train.condition_episodes'`.

- [ ] **Step 3: Implement `src/train/condition_episodes.py`**

```python
"""Group-aware multi-positive condition episodes and swap episodes for stage (d) (spec §3)."""

from dataclasses import dataclass

import numpy as np

from src.data.sampling import draw_distinct


@dataclass(frozen=True)
class ConditionEpisodes:
    anchor: np.ndarray
    supports: np.ndarray
    contrasts: np.ndarray
    candidates: np.ndarray        # positives, then hard negatives, then random negatives
    positive_mask: np.ndarray


@dataclass(frozen=True)
class SwapEpisodes:
    anchor: np.ndarray
    supports_a: np.ndarray
    contrasts_a: np.ndarray
    supports_b: np.ndarray
    contrasts_b: np.ndarray
    candidates: np.ndarray        # A-only, then B-only, then hard and random negatives outside both
    positive_mask_a: np.ndarray
    positive_mask_b: np.ndarray


def pair_feature_units(img_features, txt_features) -> np.ndarray:
    pair = 0.5 * (np.asarray(img_features, dtype=np.float32) + np.asarray(txt_features, dtype=np.float32))
    return pair / np.maximum(np.linalg.norm(pair, axis=1, keepdims=True), 1e-12)


def _hard_negatives(rng, anchor, pool, units, keys, used, count, hard_pool):
    """The `count` rows of `pool` most CLIP-similar to the anchor (among a random `hard_pool` sample)."""
    sample = pool if len(pool) <= hard_pool else rng.choice(pool, hard_pool, replace=False)
    sample = sample[~np.isin(keys[sample], list(used))]
    order = np.argsort(-(units[sample] @ units[anchor]), kind="stable")
    picked = []
    for idx in order:
        row = int(sample[idx])
        if keys[row] in used:
            continue
        used.add(keys[row])
        picked.append(row)
        if len(picked) == count:
            return picked
    raise ValueError("Not enough distinct outside paintings for hard negatives")


def mine_condition_episodes(source, units, keys, n_episodes, rng, episodes_per_condition=4, num_support=4,
                            num_contrast=4, num_positive=4, num_hard=6, num_random=6, hard_pool=2048,
                            max_failures=1000) -> ConditionEpisodes:
    fields = {k: [] for k in ("anchor", "supports", "contrasts", "candidates")}
    failures = 0
    while len(fields["anchor"]) < n_episodes:
        condition = source.sample_condition(rng)
        for _ in range(episodes_per_condition):
            if len(fields["anchor"]) == n_episodes:
                break
            used: set = set()
            try:
                anchor = draw_distinct(rng, condition.inside, keys, used, 1)[0]
                supports = draw_distinct(rng, condition.inside, keys, used, num_support)
                positives = draw_distinct(rng, condition.inside, keys, used, num_positive)
                contrasts = draw_distinct(rng, condition.outside, keys, used, num_contrast)
                hard = _hard_negatives(rng, anchor, condition.outside, units, keys, used, num_hard, hard_pool)
                random_neg = draw_distinct(rng, condition.outside, keys, used, num_random)
            except ValueError:
                failures += 1
                if failures > max_failures:
                    raise RuntimeError("Too many conditions could not fill an episode")
                break                                   # resample a condition
            fields["anchor"].append(anchor)
            fields["supports"].append(supports)
            fields["contrasts"].append(contrasts)
            fields["candidates"].append(positives + hard + random_neg)
    mask = np.zeros((n_episodes, num_positive + num_hard + num_random), dtype=bool)
    mask[:, :num_positive] = True
    return ConditionEpisodes(*(np.asarray(fields[k], dtype=np.int64) for k in fields), positive_mask=mask)


def mine_swap_episodes(source, units, keys, n_episodes, rng, episodes_per_pair=4, num_support=4,
                       num_contrast=4, num_each=3, num_hard=5, num_random=5, hard_pool=2048,
                       max_failures=1000) -> SwapEpisodes:
    if not source.swap_capable:
        raise ValueError(f"{source.name} cannot form swap pairs")
    names = ("anchor", "supports_a", "contrasts_a", "supports_b", "contrasts_b", "candidates")
    fields = {k: [] for k in names}
    failures = 0
    while len(fields["anchor"]) < n_episodes:
        a, b = source.sample_swap(rng)
        both = np.intersect1d(a.inside, b.inside, assume_unique=True)
        a_only = np.setdiff1d(a.inside, b.inside, assume_unique=True)
        b_only = np.setdiff1d(b.inside, a.inside, assume_unique=True)
        neither = np.intersect1d(a.outside, b.outside, assume_unique=True)
        for _ in range(episodes_per_pair):
            if len(fields["anchor"]) == n_episodes:
                break
            used: set = set()
            try:
                anchor = draw_distinct(rng, both, keys, used, 1)[0]
                row = [anchor,
                       draw_distinct(rng, a_only, keys, used, num_support),
                       draw_distinct(rng, a.outside, keys, used, num_contrast),
                       draw_distinct(rng, b_only, keys, used, num_support),
                       draw_distinct(rng, b.outside, keys, used, num_contrast)]
                cands = (draw_distinct(rng, a_only, keys, used, num_each)
                         + draw_distinct(rng, b_only, keys, used, num_each)
                         + _hard_negatives(rng, anchor, neither, units, keys, used, num_hard, hard_pool)
                         + draw_distinct(rng, neither, keys, used, num_random))
            except ValueError:
                failures += 1
                if failures > max_failures:
                    raise RuntimeError("Too many swap pairs could not fill an episode")
                break
            for key, value in zip(names, row + [cands]):
                fields[key].append(value)
    width = 2 * num_each + num_hard + num_random
    mask_a = np.zeros((n_episodes, width), dtype=bool)
    mask_b = np.zeros((n_episodes, width), dtype=bool)
    mask_a[:, :num_each] = True
    mask_b[:, num_each:2 * num_each] = True
    return SwapEpisodes(*(np.asarray(fields[k], dtype=np.int64) for k in names),
                        positive_mask_a=mask_a, positive_mask_b=mask_b)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_episodes.py`
Expected: all pass. Then run the full suite once.

- [ ] **Step 5: Commit**

```bash
git add src/train/condition_episodes.py src/test/test_condition_episodes.py
git commit -m "feat(v2): multi-positive condition episodes and swap episodes with painting-distinct roles"
```

---

### Task 3: the residual condition interface and the scorer

**Files:**
- Create: `src/model/condition_interface.py`
- Test: `src/test/test_condition_interface.py`

**Interfaces:**
- Consumes: `naive_condition_weights`, `conditional_score` (`src/model/conditioning.py`).
- Produces:
  - `EVIDENCE_FEATURES` (6 names)
  - `factor_evidence(support_pair (E,S,F), contrast_pair (E,C,F), factor_scale (F,)) -> (E,F,6)`
  - `ResidualConditionInterface(factor_scale, hidden=16)`, with `forward(support_pair, contrast_pair) -> (E,F)`
  - `ConditionalScorer(interface, beta_init=0.3, tau_init=1.0)`, with:
    - `.beta` and `.tau` as tensors;
    - `.set_tau(value)`;
    - `.weights(support_pair, contrast_pair)`;
    - `.score(query_feat, cand_feat, query_codes, cand_codes, weights) -> (E,M)`.

- [ ] **Step 1: Write the failing tests** — `src/test/test_condition_interface.py`

```python
import torch

from src.model.condition_interface import (
    EVIDENCE_FEATURES, ConditionalScorer, ResidualConditionInterface, factor_evidence,
)
from src.model.conditioning import conditional_score, naive_condition_weights


def _pairs(seed=0, e=32, f=8):
    g = torch.Generator().manual_seed(seed)
    s = torch.relu(torch.randn(e, 4, f, generator=g))
    c = torch.relu(torch.randn(e, 4, f, generator=g))
    return s, c


def test_step_zero_interface_equals_naive_rule_exactly():
    s, c = _pairs()
    interface = ResidualConditionInterface(torch.rand(8) + 0.1)
    assert torch.equal(interface(s, c), naive_condition_weights(s, c))


def test_weights_are_nonnegative_and_sum_to_one_or_zero():
    s, c = _pairs(1)
    interface = ResidualConditionInterface(torch.ones(8))
    with torch.no_grad():
        interface.mlp[-1].bias.fill_(0.3)                           # a non-zero correction
    w = interface(s, c)
    total = w.sum(dim=-1)
    assert (w >= 0).all() and torch.all((total - 1).abs() < 1e-5)
    zero = interface(torch.zeros(2, 4, 8), torch.ones(2, 4, 8) * 5)  # every corrected gap still negative
    assert torch.equal(zero, torch.zeros(2, 8))


def test_evidence_features_have_the_documented_layout():
    s = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])
    c = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
    ev = factor_evidence(s, c, torch.tensor([2.0, 1.0]))
    assert len(EVIDENCE_FEATURES) == 6 and ev.shape == (1, 2, 6)
    assert torch.allclose(ev[0, 0], torch.tensor([1.0, 1.0, 0.0, 0.5, 1.0, 0.0]))   # gap 2/2, mean 2/2, std 1/2
    assert torch.allclose(ev[0, 1], torch.tensor([-2.0, 0.0, 2.0, 0.0, 0.0, 1.0]))


def test_interface_is_equivariant_to_factor_permutation():
    s, c = _pairs(2)
    scale = torch.rand(8) + 0.1
    interface = ResidualConditionInterface(scale)
    with torch.no_grad():
        for p in interface.mlp.parameters():
            p.normal_(0, 0.5)
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(3))
    permuted = ResidualConditionInterface(scale[perm])
    permuted.load_state_dict({**interface.state_dict(), "factor_scale": scale[perm]})
    assert torch.allclose(interface(s, c)[:, perm], permuted(s[..., perm], c[..., perm]), atol=1e-6)


def test_scorer_uses_conditional_score_with_learned_beta_and_gradients_reach_the_mlp():
    s, c = _pairs(4, e=6)
    scorer = ConditionalScorer(ResidualConditionInterface(torch.ones(8)), beta_init=0.3)
    assert abs(scorer.beta.item() - 0.3) < 1e-6
    q, k = torch.randn(6, 5), torch.randn(6, 7, 5)
    qc, kc = torch.rand(6, 8), torch.rand(6, 7, 8)
    w = scorer.weights(s, c)
    out = scorer.score(q, k, qc, kc, w)
    assert torch.allclose(out, conditional_score(q, k, qc, kc, w, scorer.beta), atol=1e-6)
    (out / scorer.tau).sum().backward()
    assert scorer.interface.mlp[-1].weight.grad is not None
    assert scorer.interface.mlp[-1].weight.grad.abs().sum() > 0
    scorer.set_tau(0.25)
    assert abs(scorer.tau.item() - 0.25) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_interface.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.condition_interface'`.

- [ ] **Step 3: Implement `src/model/condition_interface.py`**

```python
"""Stage (d) condition interface: the naive rule plus a learned per-factor correction (spec §4).

The correction MLP is tied across factors (the same few hundred parameters for every factor) and its
last layer starts at zero, so a freshly built interface reproduces ``naive_condition_weights`` exactly.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model.conditioning import conditional_score

EVIDENCE_FEATURES = ("gap", "mean_support", "mean_contrast", "std_support", "active_support", "active_contrast")


def factor_evidence(support_pair: Tensor, contrast_pair: Tensor, factor_scale: Tensor) -> Tensor:
    """(E,S,F), (E,C,F), (F,) -> (E,F,6). Code-valued features are divided by the per-factor scale."""
    mean_s, mean_c = support_pair.mean(dim=-2), contrast_pair.mean(dim=-2)
    std_s = support_pair.std(dim=-2, unbiased=False)
    act_s = (support_pair > 0).float().mean(dim=-2)
    act_c = (contrast_pair > 0).float().mean(dim=-2)
    scale = factor_scale.clamp_min(1e-6)
    return torch.stack([(mean_s - mean_c) / scale, mean_s / scale, mean_c / scale, std_s / scale,
                        act_s, act_c], dim=-1)


class ResidualConditionInterface(nn.Module):
    def __init__(self, factor_scale, hidden: int = 16):
        super().__init__()
        self.register_buffer("factor_scale", torch.as_tensor(factor_scale, dtype=torch.float32).clamp_min(1e-6))
        self.mlp = nn.Sequential(nn.Linear(len(EVIDENCE_FEATURES), hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, support_pair: Tensor, contrast_pair: Tensor) -> Tensor:
        gap = support_pair.mean(dim=-2) - contrast_pair.mean(dim=-2)
        evidence = factor_evidence(support_pair, contrast_pair, self.factor_scale)
        correction = self.mlp(evidence).squeeze(-1) * self.factor_scale
        weights = F.relu(gap + correction)
        total = weights.sum(dim=-1, keepdim=True)
        return torch.where(total > 0, weights / total.clamp_min(1e-12), torch.zeros_like(weights))


class ConditionalScorer(nn.Module):
    """s = beta * cos(CLIP) + sum_l w_l(c) q_l c_l, with a learned beta (softplus) and temperature tau."""

    def __init__(self, interface: ResidualConditionInterface, beta_init: float = 0.3, tau_init: float = 1.0):
        super().__init__()
        self.interface = interface
        self.beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(beta_init))))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    @property
    def beta(self) -> Tensor:
        return F.softplus(self.beta_raw)

    @property
    def tau(self) -> Tensor:
        return self.log_tau.exp()

    def set_tau(self, value: float) -> None:
        with torch.no_grad():
            self.log_tau.fill_(math.log(max(float(value), 1e-6)))

    def weights(self, support_pair: Tensor, contrast_pair: Tensor) -> Tensor:
        return self.interface(support_pair, contrast_pair)

    def score(self, query_feat, cand_feat, query_codes, cand_codes, weights) -> Tensor:
        return conditional_score(query_feat, cand_feat, query_codes, cand_codes, weights, self.beta)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_interface.py`
Expected: all pass. Then run the full suite once.

- [ ] **Step 5: Commit**

```bash
git add src/model/condition_interface.py src/test/test_condition_interface.py
git commit -m "feat(v2): naive-initialized residual condition interface and conditional scorer"
```

---

### Task 4: the scorer trainer and checkpoints

**Files:**
- Create: `src/train/train_scorer.py`
- Test: `src/test/test_train_scorer.py`

**Interfaces:**
- Consumes:
  - Task 1: the sources.
  - Task 2: `pair_feature_units`, `mine_condition_episodes`, `mine_swap_episodes`.
  - Task 3: `ResidualConditionInterface`, `ConditionalScorer`.
- Produces:
  - `ScorerTrainingConfig`, a dataclass with fields `steps=3000, batch_episodes=64, episodes_per_condition=4, lr=1e-3, swap=False, lambda_swap=1.0, beta_init=0.3, hidden=16, hard_pool=2048, seed=42`.
  - `multi_positive_nce(logits (E,M), positive_mask (E,M) bool) -> Tensor`
  - `episode_logits(scorer, anchor, supports, contrasts, candidates, img_feat, txt_feat, img_codes, txt_codes, device) -> dict[str, Tensor]`, keyed `"i2t"` and `"t2i"`.
  - `train_scorer(source, img_feat, txt_feat, img_codes, txt_codes, keys, factor_scale, config, device=None, log_every=100) -> tuple[ConditionalScorer, dict]`
  - `save_scorer_checkpoint(scorer, config, path)`
  - `load_scorer_checkpoint(path, device="cpu") -> tuple[ConditionalScorer, ScorerTrainingConfig]`

- [ ] **Step 1: Write the failing tests** — `src/test/test_train_scorer.py`

```python
import dataclasses

import numpy as np
import pytest
import torch

from src.train.condition_sources import CommunitySource, FactorComboSource
from src.train.train_scorer import (
    ScorerTrainingConfig, load_scorer_checkpoint, multi_positive_nce, save_scorer_checkpoint, train_scorer,
    episode_logits,
)
from src.train.condition_episodes import mine_condition_episodes, pair_feature_units


def test_multi_positive_nce_matches_cross_entropy_for_one_positive_and_is_zero_when_all_positive():
    logits = torch.randn(5, 7)
    mask = torch.zeros(5, 7, dtype=torch.bool)
    mask[:, 0] = True
    expected = torch.nn.functional.cross_entropy(logits, torch.zeros(5, dtype=torch.long))
    assert torch.allclose(multi_positive_nce(logits, mask), expected, atol=1e-6)
    assert multi_positive_nce(logits, torch.ones(5, 7, dtype=torch.bool)).abs() < 1e-6
    with pytest.raises(ValueError):
        multi_positive_nce(logits, torch.zeros(5, 7, dtype=torch.bool))


def _noisy_factor_world(n=3000, seed=0):
    """Naive is misled: each group's true factor has a small consistent gap; factor 7 is large random noise."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(6), n // 6)
    codes = np.zeros((n, 8), dtype=np.float32)
    codes[np.arange(n), labels] = 0.3                               # consistent, small true signal
    noisy = rng.random(n) < 0.5
    codes[noisy, 7] = rng.exponential(5.0, size=noisy.sum())        # large, inconsistent, uninformative
    img = rng.normal(size=(n, 8)).astype(np.float32)                # CLIP term carries no signal
    txt = rng.normal(size=(n, 8)).astype(np.float32)
    keys = np.arange(n)
    return labels, codes, img, txt, keys


def _r1(scorer, source, img, txt, codes, keys, seed):
    ep = mine_condition_episodes(source, pair_feature_units(img, txt), keys, 256, np.random.default_rng(seed))
    with torch.no_grad():
        logits = episode_logits(scorer, ep.anchor, ep.supports, ep.contrasts, ep.candidates, img, txt, codes, codes, "cpu")
    hits = [float(ep.positive_mask[np.arange(256), logits[d].argmax(dim=1).numpy()].mean()) for d in ("i2t", "t2i")]
    return float(np.mean(hits))


def test_training_beats_the_naive_start_when_naive_is_misled():
    labels, codes, img, txt, keys = _noisy_factor_world()
    source = CommunitySource(labels, np.arange(len(labels)), min_group_rows=50)
    scale = torch.as_tensor(codes.std(axis=0) + 1e-3)
    # beta ~ 0 so the (uninformative) CLIP term cannot be the thing training improves; the interface must.
    config = ScorerTrainingConfig(steps=400, batch_episodes=32, lr=3e-3, beta_init=1e-3, hard_pool=512, seed=0)
    naive, _ = train_scorer(source, img, txt, codes, codes, keys, scale, dataclasses.replace(config, steps=0),
                            device="cpu")
    trained, history = train_scorer(source, img, txt, codes, codes, keys, scale, config, device="cpu")
    assert np.isfinite(history["loss"]).all()
    assert _r1(trained, source, img, txt, codes, keys, 99) > _r1(naive, source, img, txt, codes, keys, 99) + 0.10


def test_swap_training_runs_and_non_swap_source_rejects_swap():
    rng = np.random.default_rng(1)
    codes = np.maximum(0.0, rng.normal(size=(4000, 8)) - 0.2).astype(np.float32)
    img = rng.normal(size=(4000, 8)).astype(np.float32)
    txt = rng.normal(size=(4000, 8)).astype(np.float32)
    keys = np.arange(4000)
    scale = torch.as_tensor(codes.std(axis=0) + 1e-3)
    src = FactorComboSource(codes, np.arange(4000), min_group_rows=50)
    config = ScorerTrainingConfig(steps=5, batch_episodes=8, swap=True, hard_pool=256)
    _, history = train_scorer(src, img, txt, codes, codes, keys, scale, config, device="cpu")
    assert len(history["loss_swap"]) > 0 and np.isfinite(history["loss_swap"]).all()
    community = CommunitySource(np.repeat(np.arange(4), 1000), np.arange(4000), min_group_rows=50)
    with pytest.raises(ValueError):
        train_scorer(community, img, txt, codes, codes, keys, scale, config, device="cpu")


def test_checkpoint_round_trip_and_determinism(tmp_path):
    labels, codes, img, txt, keys = _noisy_factor_world(n=1200)
    source = CommunitySource(labels, np.arange(1200), min_group_rows=50)
    scale = torch.as_tensor(codes.std(axis=0) + 1e-3)
    config = ScorerTrainingConfig(steps=20, batch_episodes=8, hard_pool=256, seed=3)
    a, hist_a = train_scorer(source, img, txt, codes, codes, keys, scale, config, device="cpu")
    b, hist_b = train_scorer(source, img, txt, codes, codes, keys, scale, config, device="cpu")
    assert hist_a["loss"] == hist_b["loss"]
    save_scorer_checkpoint(a, config, tmp_path / "s.pt")
    loaded, loaded_config = load_scorer_checkpoint(tmp_path / "s.pt")
    assert loaded_config == config
    for (name, p), (_, q) in zip(a.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(p, q), name
```

- [ ] **Step 2: Run to verify failure**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_train_scorer.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train.train_scorer'`.

- [ ] **Step 3: Implement `src/train/train_scorer.py`**

```python
"""Train the stage-(d) conditional scorer on frozen factor codes (spec §4).

Episodes are mined fresh every step from a condition source built on scorer-train rows, so no episode is
reused. The loss is a multi-positive ranking loss in both directions, plus an optional swap term.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from src.model.condition_interface import ConditionalScorer, ResidualConditionInterface
from src.train.condition_episodes import mine_condition_episodes, mine_swap_episodes, pair_feature_units


@dataclass
class ScorerTrainingConfig:
    steps: int = 3000
    batch_episodes: int = 64
    episodes_per_condition: int = 4
    lr: float = 1e-3
    swap: bool = False
    lambda_swap: float = 1.0
    beta_init: float = 0.3
    hidden: int = 16
    hard_pool: int = 2048
    seed: int = 42


def multi_positive_nce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("every episode needs at least one positive")
    positives = logits.masked_fill(~positive_mask, float("-inf"))
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positives, dim=1)).mean()


def _t(values, device):
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)


def episode_logits(scorer, anchor, supports, contrasts, candidates, img_feat, txt_feat, img_codes, txt_codes,
                   device) -> dict:
    support_pair = 0.5 * (_t(img_codes[supports], device) + _t(txt_codes[supports], device))
    contrast_pair = 0.5 * (_t(img_codes[contrasts], device) + _t(txt_codes[contrasts], device))
    weights = scorer.weights(support_pair, contrast_pair)
    out = {}
    for direction, qf, cf, qc, cc in (("i2t", img_feat, txt_feat, img_codes, txt_codes),
                                      ("t2i", txt_feat, img_feat, txt_codes, img_codes)):
        scores = scorer.score(_t(qf[anchor], device), _t(cf[candidates], device),
                              _t(qc[anchor], device), _t(cc[candidates], device), weights)
        out[direction] = scores / scorer.tau
    return out


def _rank_loss(scorer, ep, data, device):
    logits = episode_logits(scorer, ep.anchor, ep.supports, ep.contrasts, ep.candidates, *data, device)
    mask = torch.as_tensor(ep.positive_mask, device=device)
    return 0.5 * (multi_positive_nce(logits["i2t"], mask) + multi_positive_nce(logits["t2i"], mask))


def _swap_loss(scorer, sw, data, device):
    total = 0.0
    for supports, contrasts, mask in ((sw.supports_a, sw.contrasts_a, sw.positive_mask_a),
                                      (sw.supports_b, sw.contrasts_b, sw.positive_mask_b)):
        logits = episode_logits(scorer, sw.anchor, supports, contrasts, sw.candidates, *data, device)
        m = torch.as_tensor(mask, device=device)
        total = total + 0.5 * (multi_positive_nce(logits["i2t"], m) + multi_positive_nce(logits["t2i"], m))
    return total / 2


def train_scorer(source, img_feat, txt_feat, img_codes, txt_codes, keys, factor_scale, config: ScorerTrainingConfig,
                 device=None, log_every=100):
    if config.swap and not source.swap_capable:
        raise ValueError(f"{source.name} cannot form swap pairs; train it without the swap term")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    units = pair_feature_units(img_feat, txt_feat)
    data = (img_feat, txt_feat, img_codes, txt_codes)
    scorer = ConditionalScorer(ResidualConditionInterface(factor_scale, config.hidden),
                               beta_init=config.beta_init).to(device)

    def mine():
        return mine_condition_episodes(source, units, keys, config.batch_episodes, rng,
                                       config.episodes_per_condition, hard_pool=config.hard_pool)

    first = mine()
    with torch.no_grad():                                   # tau := std of step-0 scores (unit-scale logits)
        logits = episode_logits(scorer, first.anchor, first.supports, first.contrasts, first.candidates,
                                *data, device)
        scorer.set_tau(float(torch.cat([logits["i2t"].ravel(), logits["t2i"].ravel()]).std()))
    optimizer = torch.optim.Adam(scorer.parameters(), lr=config.lr)
    history = {"step": [], "loss": [], "loss_rank": [], "loss_swap": [], "beta": [], "tau": []}
    for step in range(config.steps):
        episodes = first if step == 0 else mine()
        loss_rank = _rank_loss(scorer, episodes, data, device)
        loss = loss_rank
        loss_swap = None
        if config.swap:
            swaps = mine_swap_episodes(source, units, keys, config.batch_episodes, rng,
                                       config.episodes_per_condition, hard_pool=config.hard_pool)
            loss_swap = _swap_loss(scorer, swaps, data, device)
            loss = loss + config.lambda_swap * loss_swap
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == config.steps - 1:
            history["step"].append(step)
            history["loss"].append(float(loss))
            history["loss_rank"].append(float(loss_rank))
            if loss_swap is not None:
                history["loss_swap"].append(float(loss_swap))
            history["beta"].append(float(scorer.beta))
            history["tau"].append(float(scorer.tau))
    return scorer.eval(), history


def save_scorer_checkpoint(scorer: ConditionalScorer, config: ScorerTrainingConfig, path) -> None:
    torch.save({"state_dict": scorer.state_dict(), "config": asdict(config),
                "num_factors": int(scorer.interface.factor_scale.numel())}, path)


def load_scorer_checkpoint(path, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    config = ScorerTrainingConfig(**payload["config"])
    scorer = ConditionalScorer(ResidualConditionInterface(torch.ones(payload["num_factors"]), config.hidden),
                               beta_init=config.beta_init)
    scorer.load_state_dict(payload["state_dict"])
    return scorer.to(device).eval(), config
```

If `test_training_beats_the_naive_start_when_naive_is_misled` fails after a faithful
implementation, do not loosen it and do not change the model to force a pass. Diagnose it (is the
fixture's premise wrong, or the trainer?) and report DONE_WITH_CONCERNS with the measured values.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_train_scorer.py`
Expected: all pass. Then run the full suite once.

- [ ] **Step 5: Commit**

```bash
git add src/train/train_scorer.py src/test/test_train_scorer.py
git commit -m "feat(v2): stage-(d) scorer trainer (multi-positive ranking + optional swap term) and checkpoints"
```

---

### Task 5: evaluation additions — wrong-condition control, condition-use gain, human swap test, ceiling, episode hash

**Files:**
- Create: `src/eval/condition_eval.py`
- Modify: `src/eval/label_episodes.py` (add `label_episodes_sha256`)
- Test: `src/test/test_condition_eval.py`

**Interfaces:**
- Consumes: `LabelEpisodes`, `tie_aware_rank`, `standard_label_episodes`
  (`src/eval/label_episodes.py`); `ConditionalScorer` (Task 3); `pair_codes`, `conditional_score`;
  `draw_distinct` (Task 1).
- Produces:
  - `label_episodes_sha256(episodes) -> str`. It must equal Task 7-of-the-repair-plan's
    `label_sha`: sha256 over the int64 bytes of anchor, positive, supports, contrasts and
    distractors, then the newline-joined labels.
  - `wrong_condition(episodes, seed=42) -> LabelEpisodes` (a derangement of supports and contrasts)
  - `label_ranks(scorer, img_feat, txt_feat, img_codes, txt_codes, episodes, device=None) -> dict[str, np.ndarray]`
  - `paired_bootstrap(values, n_boot=5000, seed=42) -> {"point": float, "ci95": [lo, hi]}`
  - `condition_use_gain(model_ranks, model_wrong_ranks, naive_ranks, naive_wrong_ranks, n_boot=5000, seed=42) -> dict`,
    keyed `"i2t"`, `"t2i"` and `"mean"`.
  - `HumanSwapEpisodes(anchor, supports_emo, contrasts_emo, supports_style, contrasts_style, candidates (E,13), emotions, styles)`
  - `build_human_swap_episodes(data, groups, rows, n_episodes, seed=42, num_support=4, num_contrast=4, num_negatives=11, min_paintings=30) -> HumanSwapEpisodes`
  - `human_swap_success(scorer, img_feat, txt_feat, img_codes, txt_codes, episodes, device=None) -> dict[str, np.ndarray]`
  - `swap_success_difference(model_success, naive_success, n_boot=5000, seed=42) -> dict`
  - `ceiling_ranks(img_feat, txt_feat, img_codes, txt_codes, episodes, beta=0.3, steps=100, lr=0.1, device=None) -> dict[str, np.ndarray]`

The human swap construction makes each condition **one-aspect clean**. This is a controller ruling
that tightens spec §6:
- Emotion supports share the anchor's emotion but NOT its style.
- Style supports share the anchor's style, from paintings with NO annotation of the anchor's
  emotion.
- `p_emo` has the anchor's emotion and a different style. `p_style` has the anchor's style and its
  painting has no annotation of the anchor's emotion.
- Negatives have a different style and no annotation of the anchor's emotion.
- Contrasts: for emotion, paintings with no annotation of the emotion; for style, rows of other
  styles.
- The anchor's emotion is never "something else". "Painting" means the leakage-group key.
- Only `rows` are sampled. Annotation cleanliness is checked over the full data arrays.

- [ ] **Step 1: Write the failing tests** — `src/test/test_condition_eval.py`

```python
from types import SimpleNamespace

import numpy as np
import torch

from src.eval.condition_eval import (
    build_human_swap_episodes, ceiling_ranks, condition_use_gain, human_swap_success, label_ranks,
    paired_bootstrap, swap_success_difference, wrong_condition,
)
from src.eval.label_episodes import LabelEpisodes, build_label_episodes, label_episodes_sha256
from src.model.condition_interface import ConditionalScorer, ResidualConditionInterface


def _episodes(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return LabelEpisodes(anchor=np.arange(n), positive=np.arange(n) + 100,
                         supports=rng.integers(0, 1000, (n, 4)), contrasts=rng.integers(0, 1000, (n, 4)),
                         distractors=rng.integers(0, 1000, (n, 12)), labels=np.array(["a"] * n))


def test_wrong_condition_is_a_derangement_and_keeps_everything_else():
    ep = _episodes()
    wrong = wrong_condition(ep, seed=42)
    assert np.array_equal(wrong.anchor, ep.anchor) and np.array_equal(wrong.distractors, ep.distractors)
    same = [(wrong.supports[i] == ep.supports[i]).all() for i in range(50)]
    assert not any(same)
    assert np.array_equal(wrong_condition(ep, seed=42).supports, wrong.supports)


def test_paired_bootstrap_and_condition_use_gain_arithmetic():
    out = paired_bootstrap(np.full(40, 0.25))
    assert out["point"] == 0.25 and out["ci95"] == [0.25, 0.25]
    ranks = {"i2t": np.array([1, 1, 2, 1.0]), "t2i": np.array([1, 2, 2, 2.0])}
    wrong = {"i2t": np.array([2, 2, 2, 2.0]), "t2i": np.array([2, 2, 2, 2.0])}
    naive = {"i2t": np.array([1, 2, 2, 2.0]), "t2i": np.array([2, 2, 2, 2.0])}
    gain = condition_use_gain(ranks, wrong, naive, wrong, n_boot=200)
    assert gain["i2t"]["point"] == 0.5 and gain["t2i"]["point"] == 0.25 and gain["mean"]["point"] == 0.375


def test_label_episode_hash_is_stable_and_sensitive():
    ep = _episodes()
    h = label_episodes_sha256(ep)
    assert h == label_episodes_sha256(_episodes()) and len(h) == 64
    changed = LabelEpisodes(**{**ep.__dict__, "positive": ep.positive + 1})
    assert label_episodes_sha256(changed) != h


def _art_world(seed=0):
    """120 paintings x 3 annotations; style per painting; emotion per annotation (mixed within paintings)."""
    rng = np.random.default_rng(seed)
    n_paint = 120
    styles_p = np.array(["s0", "s1", "s2", "s3"])[np.arange(n_paint) % 4]
    groups = np.repeat(np.arange(n_paint), 3)
    styles = styles_p[groups]
    emotions = np.array(["awe", "fear", "sadness", "something else"])[rng.integers(0, 4, len(groups))]
    return SimpleNamespace(emotions=emotions, art_styles=styles), groups


def test_human_swap_episodes_are_one_aspect_clean():
    data, groups = _art_world()
    rows = np.arange(len(groups))
    ep = build_human_swap_episodes(data, groups, rows, 60, seed=42, min_paintings=5)
    emo_paintings = {e: set(groups[data.emotions == e]) for e in np.unique(data.emotions)}
    for i in range(60):
        a = ep.anchor[i]
        e, s = data.emotions[a], data.art_styles[a]
        assert e != "something else" and ep.emotions[i] == e and ep.styles[i] == s
        p_emo, p_style, negs = ep.candidates[i, 0], ep.candidates[i, 1], ep.candidates[i, 2:]
        assert data.emotions[p_emo] == e and data.art_styles[p_emo] != s
        assert data.art_styles[p_style] == s and groups[p_style] not in emo_paintings[e]
        assert all(data.art_styles[r] != s and groups[r] not in emo_paintings[e] for r in negs)
        assert all(data.emotions[r] == e and data.art_styles[r] != s for r in ep.supports_emo[i])
        assert all(data.art_styles[r] == s and groups[r] not in emo_paintings[e] for r in ep.supports_style[i])
        assert all(groups[r] not in emo_paintings[e] for r in ep.contrasts_emo[i])
        assert all(data.art_styles[r] != s for r in ep.contrasts_style[i])
        members = [a, *ep.supports_emo[i], *ep.contrasts_emo[i], *ep.supports_style[i], *ep.contrasts_style[i],
                   *ep.candidates[i]]
        assert len({groups[r] for r in members}) == len(members)


def test_human_swap_success_counts_ties_as_failure_and_detects_a_correct_flip():
    data, groups = _art_world(1)
    rows = np.arange(len(groups))
    ep = build_human_swap_episodes(data, groups, rows, 20, seed=0, min_paintings=5)
    n = len(groups)
    feats = np.random.default_rng(2).normal(size=(n, 6)).astype(np.float32)
    zero = np.zeros((n, 8), dtype=np.float32)
    naive = ConditionalScorer(ResidualConditionInterface(torch.ones(8)), beta_init=1e-6)
    zero_feats = np.zeros_like(feats)                              # all scores exactly 0 -> every pair tied
    tied = human_swap_success(naive, zero_feats, zero_feats, zero, zero, ep)
    assert not tied["i2t"].any() and not tied["t2i"].any()
    codes = np.zeros((n, 8), dtype=np.float32)                      # factor 0-3: emotion, 4-7: style
    for j, e in enumerate(["awe", "fear", "sadness", "something else"]):
        codes[data.emotions == e, j] = 1.0
    for j, s in enumerate(["s0", "s1", "s2", "s3"]):
        codes[data.art_styles == s, 4 + j] = 1.0
    good = human_swap_success(naive, feats, feats, codes, codes, ep)
    assert good["i2t"].mean() > 0.9 and good["t2i"].mean() > 0.9
    diff = swap_success_difference(good, tied, n_boot=200)
    assert diff["pooled"]["point"] > 0.9


def test_ceiling_reaches_perfect_recall_when_one_factor_separates_the_positive():
    n = 400
    rng = np.random.default_rng(3)
    codes = rng.random((n, 8)).astype(np.float32) * 0.1
    ep = _episodes(n=30, seed=4)
    ep = LabelEpisodes(**{**ep.__dict__, "positive": np.arange(30) + 200, "distractors": rng.integers(250, 400, (30, 12))})
    codes[ep.anchor, 0] = 1.0
    codes[ep.positive, 0] = 1.0
    feats = rng.normal(size=(n, 6)).astype(np.float32)
    ranks = ceiling_ranks(feats, feats, codes, codes, ep, beta=0.0)
    assert (ranks["i2t"] == 1).all() and (ranks["t2i"] == 1).all()


def test_label_ranks_step_zero_scorer_matches_naive_recall():
    from src.eval.label_episodes import label_episode_recall, label_episode_weights
    rng = np.random.default_rng(5)
    n = 500
    labels = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    paintings = np.arange(n)
    ep = build_label_episodes(labels, paintings, np.arange(n), 40, seed=1, min_paintings_per_label=5)
    codes = rng.random((n, 8)).astype(np.float32)
    feats = rng.normal(size=(n, 6)).astype(np.float32)
    scorer = ConditionalScorer(ResidualConditionInterface(torch.ones(8)), beta_init=0.3)
    ours = label_ranks(scorer, feats, feats, codes, codes, ep)
    ref = label_episode_recall(feats, feats, codes, codes, ep, label_episode_weights(codes, codes, ep), 0.3)
    assert np.allclose(ours["i2t"], ref["i2t"]["ranks"]) and np.allclose(ours["t2i"], ref["t2i"]["ranks"])
```

- [ ] **Step 2: Run to verify failure**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_eval.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.condition_eval'`.

- [ ] **Step 3: Implement.**

In `src/eval/label_episodes.py`, add:

```python
import hashlib


def label_episodes_sha256(episodes: LabelEpisodes) -> str:
    """Identity of a label-episode set; equals the repair plan's Task 7 ``label_sha`` (held hashes recorded there)."""
    digest = hashlib.sha256()
    for field in ("anchor", "positive", "supports", "contrasts", "distractors"):
        digest.update(np.ascontiguousarray(getattr(episodes, field), dtype=np.int64).tobytes())
    digest.update("\n".join(map(str, episodes.labels.tolist())).encode())
    return digest.hexdigest()
```

`src/eval/condition_eval.py`:

```python
"""Stage (d) evaluation: wrong-condition control, condition-use gain, human swap test, ceiling (spec §5-6)."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from src.data.sampling import draw_distinct
from src.eval.label_episodes import EMOTION_CATCH_ALL, LabelEpisodes, tie_aware_rank
from src.model.conditioning import conditional_score, pair_codes


def _t(values, device=None):
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)


def wrong_condition(episodes: LabelEpisodes, seed: int = 42) -> LabelEpisodes:
    """Give every episode another episode's supports and contrasts (a seeded derangement)."""
    n = len(episodes.anchor)
    if n < 2:
        raise ValueError("need at least two episodes for a derangement")
    order = np.random.default_rng(seed).permutation(n)
    source = np.empty(n, dtype=np.int64)
    source[order] = np.roll(order, 1)
    assert np.all(source != np.arange(n))
    return replace(episodes, supports=episodes.supports[source], contrasts=episodes.contrasts[source])


def _weights(scorer, img_codes, txt_codes, supports, contrasts, device):
    support = pair_codes(_t(img_codes[supports], device), _t(txt_codes[supports], device))
    contrast = pair_codes(_t(img_codes[contrasts], device), _t(txt_codes[contrasts], device))
    return scorer.weights(support, contrast)


def _scores(scorer, weights, direction, anchor, candidates, img_feat, txt_feat, img_codes, txt_codes, device):
    if direction == "i2t":
        qf, cf, qc, cc = img_feat, txt_feat, img_codes, txt_codes
    else:
        qf, cf, qc, cc = txt_feat, img_feat, txt_codes, img_codes
    return scorer.score(_t(qf[anchor], device), _t(cf[candidates], device), _t(qc[anchor], device),
                        _t(cc[candidates], device), weights)


def label_ranks(scorer, img_feat, txt_feat, img_codes, txt_codes, episodes: LabelEpisodes, device=None) -> dict:
    candidates = np.concatenate([episodes.positive[:, None], episodes.distractors], axis=1)
    with torch.no_grad():
        w = _weights(scorer, img_codes, txt_codes, episodes.supports, episodes.contrasts, device)
        return {d: tie_aware_rank(_scores(scorer, w, d, episodes.anchor, candidates, img_feat, txt_feat,
                                          img_codes, txt_codes, device)).cpu().numpy()
                for d in ("i2t", "t2i")}


def paired_bootstrap(values, n_boot: int = 5000, seed: int = 42) -> dict:
    values = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(values), (n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return {"point": float(values.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def condition_use_gain(model_ranks, model_wrong_ranks, naive_ranks, naive_wrong_ranks, n_boot=5000, seed=42) -> dict:
    """Per episode: [hit(model) - hit(model | wrong c)] - [hit(naive) - hit(naive | wrong c)], R@1."""
    diffs = {}
    for d in ("i2t", "t2i"):
        hit = lambda r: (np.asarray(r[d]) <= 1).astype(np.float64)
        diffs[d] = (hit(model_ranks) - hit(model_wrong_ranks)) - (hit(naive_ranks) - hit(naive_wrong_ranks))
    out = {d: paired_bootstrap(v, n_boot, seed) for d, v in diffs.items()}
    out["mean"] = paired_bootstrap(0.5 * (diffs["i2t"] + diffs["t2i"]), n_boot, seed)
    return out


@dataclass(frozen=True)
class HumanSwapEpisodes:
    anchor: np.ndarray
    supports_emo: np.ndarray
    contrasts_emo: np.ndarray
    supports_style: np.ndarray
    contrasts_style: np.ndarray
    candidates: np.ndarray        # column 0 = p_emo, column 1 = p_style, then negatives
    emotions: np.ndarray
    styles: np.ndarray


def build_human_swap_episodes(data, groups, rows, n_episodes, seed=42, num_support=4, num_contrast=4,
                              num_negatives=11, min_paintings=30) -> HumanSwapEpisodes:
    emotions, styles = np.asarray(data.emotions), np.asarray(data.art_styles)
    groups, rows = np.asarray(groups), np.asarray(rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    has_emotion = {e: np.isin(groups, np.unique(groups[emotions == e])) for e in np.unique(emotions)}
    row_em, row_st = emotions[rows], styles[rows]

    def eligible(values, value):
        return len(np.unique(groups[rows[values == value]])) >= min_paintings

    ok_em = {e for e in np.unique(row_em) if e != EMOTION_CATCH_ALL and eligible(row_em, e)}
    ok_st = {s for s in np.unique(row_st) if eligible(row_st, s)}
    anchors = rows[np.isin(row_em, list(ok_em)) & np.isin(row_st, list(ok_st))]
    if len(anchors) == 0:
        raise ValueError("no eligible anchors")
    names = ("anchor", "supports_emo", "contrasts_emo", "supports_style", "contrasts_style", "candidates",
             "emotions", "styles")
    fields = {k: [] for k in names}
    failures = 0
    while len(fields["anchor"]) < n_episodes:
        a = int(anchors[rng.integers(len(anchors))])
        e, s = emotions[a], styles[a]
        clean_e = ~has_emotion[e][rows]                     # painting carries no annotation of e
        pools = {
            "sup_emo": rows[(row_em == e) & (row_st != s)],
            "con_emo": rows[clean_e],
            "sup_style": rows[(row_st == s) & clean_e],
            "con_style": rows[row_st != s],
            "p_emo": rows[(row_em == e) & (row_st != s)],
            "p_style": rows[(row_st == s) & clean_e],
            "neg": rows[(row_st != s) & clean_e],
        }
        used = {groups[a]}
        try:
            picked = [draw_distinct(rng, pools["sup_emo"], groups, used, num_support),
                      draw_distinct(rng, pools["con_emo"], groups, used, num_contrast),
                      draw_distinct(rng, pools["sup_style"], groups, used, num_support),
                      draw_distinct(rng, pools["con_style"], groups, used, num_contrast)]
            candidates = (draw_distinct(rng, pools["p_emo"], groups, used, 1)
                          + draw_distinct(rng, pools["p_style"], groups, used, 1)
                          + draw_distinct(rng, pools["neg"], groups, used, num_negatives))
        except ValueError:
            failures += 1
            if failures > 10 * n_episodes:
                raise RuntimeError("could not build enough human swap episodes")
            continue
        for key, value in zip(names, [a, *picked, candidates, e, s]):
            fields[key].append(value)
    return HumanSwapEpisodes(*(np.asarray(fields[k], dtype=np.int64) for k in names[:6]),
                             emotions=np.asarray(fields["emotions"]), styles=np.asarray(fields["styles"]))


def human_swap_success(scorer, img_feat, txt_feat, img_codes, txt_codes, episodes: HumanSwapEpisodes,
                       device=None) -> dict:
    """Success = p_emo above p_style under the emotion condition AND p_style above p_emo under the style one."""
    out = {}
    with torch.no_grad():
        w_emo = _weights(scorer, img_codes, txt_codes, episodes.supports_emo, episodes.contrasts_emo, device)
        w_style = _weights(scorer, img_codes, txt_codes, episodes.supports_style, episodes.contrasts_style, device)
        for d in ("i2t", "t2i"):
            s_emo = _scores(scorer, w_emo, d, episodes.anchor, episodes.candidates, img_feat, txt_feat,
                            img_codes, txt_codes, device)
            s_style = _scores(scorer, w_style, d, episodes.anchor, episodes.candidates, img_feat, txt_feat,
                              img_codes, txt_codes, device)
            ok = (s_emo[:, 0] > s_emo[:, 1]) & (s_style[:, 1] > s_style[:, 0])
            finite = torch.isfinite(s_emo).all(dim=1) & torch.isfinite(s_style).all(dim=1)
            out[d] = (ok & finite).cpu().numpy()
    return out


def swap_success_difference(model_success, naive_success, n_boot=5000, seed=42) -> dict:
    diff = {d: np.asarray(model_success[d], float) - np.asarray(naive_success[d], float) for d in ("i2t", "t2i")}
    out = {d: paired_bootstrap(v, n_boot, seed) for d, v in diff.items()}
    out["pooled"] = paired_bootstrap(0.5 * (diff["i2t"] + diff["t2i"]), n_boot, seed)
    out["rates"] = {"model": {d: float(np.mean(model_success[d])) for d in diff},
                    "naive": {d: float(np.mean(naive_success[d])) for d in diff}}
    return out


def ceiling_ranks(img_feat, txt_feat, img_codes, txt_codes, episodes: LabelEpisodes, beta=0.3, steps=100, lr=0.1,
                  device=None) -> dict:
    """Oracle upper bound: per-episode simplex weights optimised on the episode's own positive (diagnostic only)."""
    candidates = np.concatenate([episodes.positive[:, None], episodes.distractors], axis=1)
    out = {}
    for d in ("i2t", "t2i"):
        if d == "i2t":
            qf, cf, qc, cc = img_feat, txt_feat, img_codes, txt_codes
        else:
            qf, cf, qc, cc = txt_feat, img_feat, txt_codes, img_codes
        q, c = _t(qf[episodes.anchor], device), _t(cf[candidates], device)
        qcode, ccode = _t(qc[episodes.anchor], device), _t(cc[candidates], device)
        theta = torch.zeros(len(episodes.anchor), qcode.shape[1], device=q.device, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            s = conditional_score(q, c, qcode, ccode, torch.softmax(theta, dim=1), beta)
            margin = s[:, 0] - 0.01 * torch.logsumexp(s[:, 1:] / 0.01, dim=1)
            optimizer.zero_grad()
            (-margin.mean()).backward()
            optimizer.step()
        with torch.no_grad():
            s = conditional_score(q, c, qcode, ccode, torch.softmax(theta, dim=1), beta)
            out[d] = tie_aware_rank(s).cpu().numpy()
    return out
```

(`test_paired_bootstrap_and_condition_use_gain_arithmetic` asserts the hand-computed points:
i2t hits (m, m_wrong, n, n_wrong) give per-episode diffs `[0,1,0,1] → 0.5` and t2i `[1,0,0,0] → 0.25`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_condition_eval.py src/test/test_label_episodes.py`
Expected: all pass. Then run the full suite once.

- [ ] **Step 5: Change log and commit.** Add a `.claude/20261013_log.md` entry for
  `src/eval/label_episodes.py`.

```bash
git add src/eval/condition_eval.py src/eval/label_episodes.py src/test/test_condition_eval.py
git commit -m "feat(v2): stage-(d) evaluation — wrong-condition control, condition-use gain, human swap test, ceiling"
```

---

### Task 6: selection run (G1-G5 on the selection set) — STOP POINT

**Files:**
- Create: `src/test/20261013_stage_d_selection/run_selection.py`, the log
  `src/test/20261013_stage_d_selection/20261013_stage_d_selection_log.md`, and a local `.gitignore`.
- Create: `docs/reports/auto/v2/2026-10-13_candidate_a_stage_d_selection.md`.
- Modify: `docs/reports/reports_sum.md` (one v2 row, and the "Current work" line).

**Interfaces:** consumes everything from Tasks 1-5, plus:
- `load_artelingo`, `leakage_groups`, `grouped_split`, `load_factor_checkpoint`, `encode_rows`;
- `build_content_graph`, `GraphConfig`, `train_stage1`, `Stage1Config`, `detect_communities`;
- `standard_label_episodes`.

**Procedure** (write it as one script with `--tables` to reprint from saved JSON):

- [ ] **Step 1: Setup.**
  - Load data. Build `groups = leakage_groups(data.paintings, data.img_features)` and
    `split = grouped_split(groups, seed=42)`. Assert split sizes of 216,107 / 30,872 / 61,744.
  - Build the sub-split: `scorer_train, selection = grouped_subsplit(groups, split.train, 0.15, seed=42)`.
  - Assert the SHA-256 of the R3 checkpoint file equals the Global Constraints value. Load it and
    encode all rows (`encode_rows`). Assert the codes are finite.
  - `factor_scale = pair codes of scorer_train rows .std(axis=0) + 1e-6` (as a tensor).
- [ ] **Step 2: Fit the sources on `scorer_train` only.**
  - `FactorComboSource(pair_codes, scorer_train)`.
  - `ClipClusterSource(data.img_features, data.txt_features, scorer_train, n_clusters=64, seed=42)`.
  - Communities: `build_content_graph` on scorer-train features (`GraphConfig()`), then
    `train_stage1(..., Stage1Config())`, then `detect_communities(embeddings)`. Put the labels in a
    global array (-1 elsewhere) and build `CommunitySource(labels, scorer_train)`.
  - Log each source's number of valid groups and the size range.
- [ ] **Step 3: Train G1-G5.** Use `ScorerTrainingConfig()` defaults: 3,000 steps, batch 64, seed 42.

  | Run | Source | Config |
  |---|---|---|
  | G1 | factor_combo | `swap=False` |
  | G2 | factor_combo | `swap=True` |
  | G3 | clip_cluster | `swap=False` |
  | G4 | clip_cluster | `swap=True` |
  | G5 | community | `swap=False` |

  - `keys = groups`.
  - Save `checkpoints/G{k}.pt` (gitignored) and each run's history.
  - Build naive as `train_scorer(..., dataclasses.replace(ScorerTrainingConfig(), steps=0))`, the
    step-0 model with β=0.3. Assert that naive's weights on 16 selection episodes `torch.equal`
    `label_episode_weights`.
- [ ] **Step 4: Evaluate on the selection set.**
  - Episodes: `standard_label_episodes(data, groups, selection, "emotion", 2048, seed=42)`, and the
    same for `"art_style"`.
  - For naive and every run: `label_ranks` on the episodes and on `wrong_condition(episodes, seed=42)`.
  - Per run, compute `condition_use_gain(run, run_wrong, naive, naive_wrong)`, pooled over the two
    label types by concatenating episodes. Also compute it per label type.
  - **The selection score is `gain["mean"]["point"]`, in R@1 points (×100).**
  - Also report per run: R@1 and R@3 per direction; learned β and τ; the loss curve summary;
    CLIP-only rows (weights all zero, β=0.3) and uniform rows (weights 1/32, β=0.3).
  - Report `ceiling_ranks` on the selection episodes (β=0.3) as a diagnostic.
- [ ] **Step 5: Apply the pre-registered rule.**
  - The highest score wins.
  - Runs within 1.0 point of the best are tied. A tie goes to a no-swap run, then to the earlier run
    in the table.
  - **STOP POINT:** if the best score is ≤ +0.5 points, write the report with verdict "no trained
    run beats the naive rule on the selection set". Commit, then report STOP_POINT to the
    controller. Task 7 is **not** run.
  - Otherwise record the selected run.
- [ ] **Step 6: Report, index, commit.**
  - Report `docs/reports/auto/v2/2026-10-13_candidate_a_stage_d_selection.md`:
    1. Verdict first, in plain language: which condition source works, whether the swap term
       helps, and whether the scorer uses the condition better than naive.
    2. Then the tables: selection score with CIs per run, per label type and per direction; the
       R@1/R@3 table; β and τ; the ceiling.
    3. Then the caveats, including the spec §10 items.
  - Add the `reports_sum.md` row and run `check_reports_sum.py`.

```bash
git add src/test/20261013_stage_d_selection/ docs/reports/auto/v2/2026-10-13_candidate_a_stage_d_selection.md docs/reports/reports_sum.md
git commit -m "docs(v2): stage (d) selection — five self-generated-condition scorers vs the naive rule"
```

Expected runtime is about 30-60 minutes; InfoNCE-style training is cheap, and mining dominates.

---

### Task 7: replication and the final held-out test (only if Task 6 selected a run)

**Files:**
- Create: `src/test/20261014_stage_d_final/run_final.py`, its log, and a local `.gitignore`.
- Create: `docs/reports/auto/v2/2026-10-14_candidate_a_stage_d_final.md`.
- Modify: `docs/reports/reports_sum.md`.

**Procedure:**

- [ ] **Step 1: Rebuild the setup deterministically,** exactly as in Task 6 Steps 1-2: the same
  split, sub-split, R3 codes, and the selected run's source. Load the selected seed-42 checkpoint.
  Retrain the selected config with `seed=43` and `seed=44`, save them, and report their selection
  scores on the selection set, as in Task 6 Step 4.
- [ ] **Step 2: Held label episodes** (the first time held rows are touched).
  - `standard_label_episodes(data, groups, split.held, label, 1024, seed=42)` for emotion and for
    art style.
  - Assert `label_episodes_sha256` equals the repair plan's recorded held hashes:
    - emotion `e62ab41f7b54b500b25900a2e82c22d50f662c3843e767aeef3a8511c16a8c85`;
    - art style `3a58cf9dc3cb670f39a5084f786968d2186bd511f036565a6f825116422e167f`.
- [ ] **Step 3: Criterion 1.**
  - Compute `condition_use_gain` for the selected seed-42 scorer against naive, pooled over the
    label types.
  - **Met iff `ci95[0] > 0` for BOTH `"i2t"` AND `"t2i"`.** Also report the per-label-type values
    and seeds 43/44.
- [ ] **Step 4: Criterion 2.**
  - Build `build_human_swap_episodes(data, groups, split.held, 1024, seed=42)`.
  - Compute `human_swap_success` for the selected scorer and for naive, then `swap_success_difference`.
  - **Met iff `pooled.ci95[0] > 0`.** Report per direction, and report both success rates.
- [ ] **Step 5: Context rows.** CLIP-only, uniform, and R3 naive's numbers from the repair plan's
  Task 7 report, cited and not recomputed.
- [ ] **Step 6: Report, index, commit.**
  - Report `docs/reports/auto/v2/2026-10-14_candidate_a_stage_d_final.md`: verdict first, then
    each criterion met or not met with its numbers, then the seeds, the context, and the caveats.
    **The user makes the stage (e) decision.**
  - Add the `reports_sum.md` row, update "Current work", and run `check_reports_sum.py`.

```bash
git add src/test/20261014_stage_d_final/ docs/reports/auto/v2/2026-10-14_candidate_a_stage_d_final.md docs/reports/reports_sum.md
git commit -m "docs(v2): stage (d) final held-out test — condition use vs naive, human emotion-vs-style swap test"
```

---

## Self-review

- **Spec coverage:**
  - §3 data, splits, sources and episodes → Tasks 1, 2 and 6.
  - §4 model and loss → Tasks 3 and 4.
  - §5 selection and stop point → Task 6.
  - §6 final test and diagnostics → Tasks 5 and 7.
  - §7 constraints → Global Constraints.
  - §8 modules → Tasks 1-5.
  - §10 caveats → the report steps.
- **Deviation, a controller ruling:** the human swap test's supports are one-aspect clean. Emotion
  supports differ from the anchor's style, and style supports carry no annotation of its emotion.
  This tightens spec §6 without changing any criterion.
- **Placeholder scan:** the only prose-only steps are the two real-run scripts. Every computation
  in them calls a function defined and tested in Tasks 1-5.
- **Type consistency:**
  - `Condition`, `ConditionEpisodes`, `SwapEpisodes`, `ScorerTrainingConfig`, `ConditionalScorer`,
    `LabelEpisodes` and `HumanSwapEpisodes` field names are identical wherever they are used.
  - `train_scorer`'s argument order `(source, img_feat, txt_feat, img_codes, txt_codes, keys, factor_scale, config)`
    is the same in the tests and in Tasks 6 and 7.
- **Review Focus coverage:**
  - #1: `test_sources_never_return_rows_outside_the_fit_rows`, `test_miner_only_uses_fit_rows`.
  - #2: `test_factor_combo_conditions_are_disjoint_nonzero_and_inside_fit_rows` (dead factor),
    `test_community_source_skips_small_groups_and_cannot_swap`,
    `test_factor_combo_raises_when_no_valid_condition_exists`.
  - #3: `test_step_zero_interface_equals_naive_rule_exactly`, plus Task 6's runtime assert.
  - #4: `test_wrong_condition_is_a_derangement_and_keeps_everything_else`.
  - #5: `test_human_swap_episodes_are_one_aspect_clean`.
