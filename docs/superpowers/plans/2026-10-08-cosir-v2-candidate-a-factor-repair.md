# CoSiR v2 Candidate A — factor-discovery repair + painting-grouped split

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (Claude
> Code subagents implement; Claude is controller/reviewer — see Global Constraints) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** make Candidate A's 32-factor space genuinely multi-dimensional, cross-modal and sparse,
with gates that would have caught the collapse. Move every evaluation to a painting-grouped split.
Then re-measure the condition interface on the repaired factors, before any stage (d) planning.

**Architecture:** A new gate module (`src/eval/factor_gates.py`) adds pass/fail checks for
dimensionality, redundancy, information, sparsity and pair specificity to the existing usage
checks. A new data module (`src/data/`) loads ArtELingo once and builds a leakage-free
train/val/held split grouped by painting (and by identical image vector). A one-variable-at-a-time
diagnosis locates the collapse. Then default-off fix mechanisms are added to the existing
factor code: contrastive agreement, decorrelation, TopK activation and input centering. A
pre-registered grid runs on training rows. The hard geometry gates are the floor. Among the
configurations that pass, the one selected gives the largest condition-specific benefit on
validation episodes defined by ArtELingo's human labels (emotion and art style), which is the
project's aim, not reconstruction. The naive condition rule and the conditional score move into
`src/model/conditioning.py`. Everything is re-evaluated on held-out paintings, with R0 and the
repaired model paired on identical label-defined episodes.

**Tech Stack:** Python 3.10, PyTorch, NumPy, SciPy (`scipy.sparse`, `scipy.optimize`), existing
`FeatureManager` (`src.utils`), pytest. Conda env `CoSiR`.

**Spec:** `docs/superpowers/specs/2026-09-28-cosir-v2-ground-up-redesign.md` ("Candidate A
architecture" §1–3). **Motivating evidence** (read before any task):
- `docs/reports/2026-09-29_cosir_v2_code_review.md`: the collapse and split-leakage findings.
- `src/test/20260929_factor_collapse_verification/20260929_factor_collapse_verification_log.md`:
  the controller's independent check. Codes have participation ratio (PR) 1.32 against CLIP's
  ~40. See that log's correction section: the codes still linearly carry about as much as CLIP
  PCA rank ~11-12 (image) / ~8 (text), but in directions with ~10^4 times less variance than the
  dominant axis. The collapse is one of scale/geometry more than of information.
  99.54% of held rows' images appear in training.
- `docs/reports/2026-10-07_cosir_v2_candidate_a_naive_rule_mechanism.md`: what the collapse did to
  the condition interface.

This plan **supersedes the "validated" verdict of the factor-discovery plan's Task 6**
(`docs/superpowers/plans/2026-09-28-cosir-v2-candidate-a-factor-discovery.md`). That plan's gates
measured usage, not independence.

## Global Constraints

- Repository `/project/CoSiR-v2`, branch `cosir-v2`. Python: `/root/miniconda3/envs/CoSiR/bin/python`,
  run from the repository root. Tests: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q <path>`.
- `seed=42` for every stochastic step unless a task names another seed. No `cuml`/`cugraph`.
- Function/class-formal code in `src/` with unit tests. One-off real-data runs go in dated folders
  `src/test/yyyymmdd_<name>/`, each with a `yyyymmdd_<name>_log.md`. Caches and checkpoints in those
  folders are gitignored via a local `.gitignore` and are never committed.
- Modifying an existing `src/` file requires a change-log entry in `.claude/yyyymmdd_log.md`: a
  header per file path, before/after snippets, and the reason.
- Reports are `docs/reports/yyyy-mm-dd_cosir_v2_candidate_a_<name>.md`. Put a plain-language
  verdict first, then evidence, then caveats. Every number comes from a real run. Never tune
  until the numbers look good and then report only the last run.
- **Split discipline:** from Task 3 on, every factor fit uses the painting-grouped `train` rows
  only. Selection uses `val` rows only. `held` rows are touched exactly twice: Task 6's final gate
  check and Task 7's report.
- **No unregistered configurations.** Tasks 3, 6 and 7 list every configuration they run. Anything
  else needs controller and user approval first. The gate thresholds in Task 1 are pre-registered
  judgment calls; if they prove binding, report it. Never relax one silently.
- Implementers are Claude Code subagents, with the model sized to each task's difficulty. **No
  code goes to Codex unless the user explicitly says so** (user direction, 2026-09-29). Claude
  reviews every task. Never use `.ccg/tasks/`. Real GPU runs on the local machine are
  pre-authorized. Every run in this plan is expected to take under about 15 minutes; if one would
  exceed 30 minutes, stop and hand back to the controller.
- Commits end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push.

## Review Focus

1. **Same-painting rows inside one contrastive batch.** Graph-edge sampling makes near-duplicate
   annotations of one painting common. `cross_modal_infonce_loss` must exclude same-group
   off-diagonal pairs as negatives, not push them apart. Test owned by Task 4.
2. **Two paintings sharing an identical image vector.** There are 61,901 image vectors but 61,402
   paintings, so the painting key alone is not enough. Such rows must land in the same split part.
   Leakage must be zero by both the painting key and the image-hash key. Test owned by Task 2.
3. **Dead or constant factor columns.** Gates and the decorrelation penalty must not produce NaN.
   A constant column must be reported and must fail the redundancy gate, never silently dropped
   to make max |r| look good. Tests owned by Task 1 and Task 4.
4. **A centered model encoding held rows after a checkpoint reload.** Codes must be bit-identical
   to in-session encoding, so the input means are persisted buffers. Test owned by Task 4.
5. **All-zero codes, zero weight vectors, or all-tied scores.** Retrieval and ranking metrics must
   count ties against the positive (retrieval) or use tie-aware ranks (ranking), never as
   successes. Tests owned by Task 1 (retrieval) and Task 5 (ranking).

---

### Task 1: factor-geometry gates

**Files:**
- Create: `src/eval/__init__.py` (empty), `src/eval/factor_gates.py`
- Test: `src/test/test_factor_gates.py`
- Create (real check): `src/test/20261008_factor_gates_collapsed_check/run_check.py`,
  `src/test/20261008_factor_gates_collapsed_check/20261008_factor_gates_collapsed_check_log.md`

**Interfaces:**
- Consumes: nothing new. The dead / modality-private / community-spanning definitions are copied
  verbatim from `src/test/20260928_factor_discovery_validation/run_validation.py:26-91`
  (`NEAR_ZERO_MEAN=1e-4`, `PRIVATE_RATIO=0.05`, `SINGLE_COMMUNITY_MASS=0.50`).
- Produces (used by Tasks 3, 6 and 7):
  - `FactorGateThresholds` (frozen dataclass, fields below)
  - `relative_l2(target, reconstruction) -> float`
  - `participation_ratio(codes) -> float`
  - `factor_correlation_summary(codes, threshold=0.9, near_zero_std=1e-8) -> dict`
  - `active_fraction(codes) -> float`
  - `top2_mass_share(img_codes, txt_codes) -> float`
  - `dead_and_private(img_codes, txt_codes, near_zero_mean, private_min_over_max) -> tuple[np.ndarray, np.ndarray]`
  - `community_spanning(img_codes, txt_codes, labels, near_zero_mean, single_community_mass) -> dict`
  - `linear_readout_rel_l2(fit_codes, fit_features, eval_codes, eval_features, ridge=1e-3) -> float`
  - `pca_rel_l2(fit_features, eval_features, rank) -> float`
  - `paired_retrieval_recall(query, gallery, pool=1000, k=10, seed=42) -> float`
  - `FactorGateReport(values: dict, passed: dict[str, bool])` with property `all_passed`
  - `evaluate_factor_gates(*, fit_img_codes, fit_txt_codes, fit_img_features, fit_txt_features, eval_img_codes, eval_txt_codes, eval_img_features, eval_txt_features, community_img_codes, community_txt_codes, community_labels, thresholds=FactorGateThresholds()) -> FactorGateReport`

- [ ] **Step 1: Write the failing tests** in `src/test/test_factor_gates.py`

```python
import numpy as np

from src.eval.factor_gates import (
    FactorGateThresholds, community_spanning, dead_and_private, evaluate_factor_gates,
    factor_correlation_summary, linear_readout_rel_l2, paired_retrieval_recall,
    participation_ratio, relative_l2,
)


def _collapsed_codes(n=4000, factors=32, seed=0):
    """32 near-copies / anti-copies of one latent axis: the observed failure mode."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 1))
    signs = np.where(np.arange(factors) % 3 == 0, -1.0, 1.0)
    return np.maximum(0.0, 1.0 + z * signs + 0.05 * rng.normal(size=(n, factors)))


def _independent_codes(n=4000, factors=32, seed=0):
    rng = np.random.default_rng(seed)
    return np.maximum(0.0, rng.normal(size=(n, factors)) - 0.5)


def test_participation_ratio_separates_collapsed_from_independent():
    assert participation_ratio(_collapsed_codes()) < 2.0
    assert participation_ratio(_independent_codes()) > 20.0


def test_redundancy_summary_flags_copies_and_reports_constant_columns():
    codes = _collapsed_codes()
    codes[:, 5] = 0.7
    summary = factor_correlation_summary(codes)
    assert np.isfinite(summary["max_abs"]) and summary["max_abs"] > 0.9
    assert summary["constant_factors"] == [5]
    independent = factor_correlation_summary(_independent_codes())
    assert independent["max_abs"] < 0.2 and independent["pairs_at_or_above"] == 0


def test_all_zero_codes_are_degenerate_not_errors():
    zeros = np.zeros((2000, 32))
    assert participation_ratio(zeros) == 0.0
    assert paired_retrieval_recall(zeros, zeros, pool=500, k=10) == 0.0


def test_ties_count_against_the_positive_in_retrieval():
    tied = np.ones((2000, 32))
    assert paired_retrieval_recall(tied, tied, pool=500, k=10) == 0.0
    codes = _independent_codes(n=2000) + 1e-3
    assert paired_retrieval_recall(codes, codes, pool=500, k=1) == 1.0


def test_linear_readout_recovers_linear_features_and_not_noise():
    rng = np.random.default_rng(1)
    codes = _independent_codes(n=6000)
    features = codes @ rng.normal(size=(32, 64)) + 3.0
    assert linear_readout_rel_l2(codes[:4000], features[:4000], codes[4000:], features[4000:]) < 0.05
    noise = rng.normal(size=codes.shape)
    mean_only = relative_l2(features[4000:], np.broadcast_to(features[:4000].mean(0), features[4000:].shape))
    assert linear_readout_rel_l2(noise[:4000], features[:4000], noise[4000:], features[4000:]) > 0.95 * mean_only


def test_dead_private_spanning_match_task3_definitions():
    img, txt = np.zeros((6, 4)), np.zeros((6, 4))
    img[:, 0] = txt[:, 0] = 1.0          # shared, spread over 3 communities -> spanning
    img[:3, 1] = txt[:3, 1] = 1.0        # shared, 2/3 of mass in community 0 -> topic-like
    img[:, 2] = 1.0                      # image-only -> private (still spanning)
    labels = np.array([0, 0, 1, 1, 2, 2])  # factor 3 all zero -> dead
    dead, private = dead_and_private(img, txt, 1e-4, 0.05)
    assert dead.tolist() == [3] and private.tolist() == [2]
    span = community_spanning(img, txt, labels, 1e-4, 0.50)
    assert span["spanning_indices"] == [0, 2] and span["topic_like_indices"] == [1]
    assert span["spanning_fraction"] == 0.5


def test_evaluate_factor_gates_fails_collapsed_and_passes_independent():
    rng = np.random.default_rng(2)
    latent_img = _independent_codes(n=6000, seed=3)
    # same support as the image codes, so noise cannot switch on extra factors (sparsity gate)
    latent_txt = np.where(latent_img > 0,
                          np.maximum(0.0, latent_img + 0.05 * rng.normal(size=latent_img.shape)), 0.0)
    mixing = rng.normal(size=(32, 64))
    img_feat = latent_img @ mixing + 0.1 * rng.normal(size=(6000, 64)) + 2.0
    txt_feat = latent_txt @ mixing + 0.1 * rng.normal(size=(6000, 64)) + 2.0
    labels = rng.integers(0, 8, size=4000)
    thresholds = FactorGateThresholds(retrieval_pool=500)

    def gates(img_codes, txt_codes):
        return evaluate_factor_gates(
            fit_img_codes=img_codes[:4000], fit_txt_codes=txt_codes[:4000],
            fit_img_features=img_feat[:4000], fit_txt_features=txt_feat[:4000],
            eval_img_codes=img_codes[4000:], eval_txt_codes=txt_codes[4000:],
            eval_img_features=img_feat[4000:], eval_txt_features=txt_feat[4000:],
            community_img_codes=img_codes[:4000], community_txt_codes=txt_codes[:4000],
            community_labels=labels, thresholds=thresholds,
        )

    good = gates(latent_img, latent_txt)
    assert good.all_passed, good.passed
    collapsed = _collapsed_codes(n=6000, seed=4)
    bad = gates(collapsed, collapsed.copy())
    assert not bad.all_passed
    for name in ("participation_ratio", "redundancy", "readout", "sparsity"):
        assert bad.passed[name] is False, name
```

- [ ] **Step 2: Run to verify failure**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_factor_gates.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval'`.

- [ ] **Step 3: Implement `src/eval/factor_gates.py`**

```python
"""Pass/fail geometry gates for a learned shared factor space.

The factor-discovery plan's gates (top-2 mass, dead, modality-private,
community spanning) measure usage; a dictionary of 32 copies of one axis
passes all of them. These gates add dimensionality, redundancy, linear
information, sparsity and pair-specificity checks. Dead / private / spanning
definitions are copied from
src/test/20260928_factor_discovery_validation/run_validation.py.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorGateThresholds:
    min_participation_ratio: float = 8.0      # per modality, eval rows; collapsed codes: 1.32
    max_pair_abs_corr: float = 0.90           # eval pair codes; collapsed: 374/496 pairs >= .9
    readout_pca_rank: int = 10                # code readout must beat CLIP PCA-10 per modality
    readout_ridge: float = 1e-3
    readout_fit_rows: int = 100_000
    max_active_fraction: float = 0.375        # <= 12 of 32 active per row; collapsed: ~0.71
    near_zero_mean: float = 1e-4              # Task 3 definition
    private_min_over_max: float = 0.05        # Task 3 definition
    single_community_mass: float = 0.50       # Task 3 definition
    max_dead: int = 0
    max_private: int = 1
    max_top2_mass_share: float = 0.20
    min_spanning_fraction: float = 0.75
    retrieval_pool: int = 1000
    retrieval_k: int = 10
    min_retrieval_ratio: float = 0.5          # code R@k / CLIP R@k, same pools
    seed: int = 42


def relative_l2(target: np.ndarray, reconstruction: np.ndarray) -> float:
    """Mean per-row ||target - reconstruction|| / ||target|| (Task 3/6 metric)."""
    target = np.asarray(target, dtype=np.float64)
    error = np.linalg.norm(target - reconstruction, axis=1)
    return float((error / np.maximum(np.linalg.norm(target, axis=1), 1e-12)).mean())


def participation_ratio(codes: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2 of the centered covariance; 0.0 if no variance."""
    codes = np.asarray(codes, dtype=np.float64)
    energy = np.linalg.svd(codes - codes.mean(axis=0), compute_uv=False) ** 2
    total = energy.sum()
    if total <= 0:
        return 0.0
    share = energy / total
    return float(1.0 / (share**2).sum())


def factor_correlation_summary(codes: np.ndarray, threshold: float = 0.9,
                               near_zero_std: float = 1e-8) -> dict:
    """Max off-diagonal |Pearson r| over varying factors; constant factors are listed."""
    codes = np.asarray(codes, dtype=np.float64)
    std = codes.std(axis=0)
    varying = np.flatnonzero(std > near_zero_std)
    constant = np.flatnonzero(std <= near_zero_std).tolist()
    if len(varying) < 2:
        return {"max_abs": 0.0, "pairs_at_or_above": 0, "pairs_total": 0,
                "constant_factors": constant}
    upper = np.abs(np.corrcoef(codes[:, varying].T)[np.triu_indices(len(varying), 1)])
    return {"max_abs": float(upper.max()), "pairs_at_or_above": int((upper >= threshold).sum()),
            "pairs_total": int(upper.size), "constant_factors": constant}


def active_fraction(codes: np.ndarray) -> float:
    return float((np.asarray(codes) > 0).mean())


def top2_mass_share(img_codes: np.ndarray, txt_codes: np.ndarray) -> float:
    mean = 0.5 * (np.asarray(img_codes, np.float64).mean(0) + np.asarray(txt_codes, np.float64).mean(0))
    total = mean.sum()
    return 1.0 if total <= 0 else float(np.sort(mean)[-2:].sum() / total)


def dead_and_private(img_codes, txt_codes, near_zero_mean, private_min_over_max):
    img_mean = np.asarray(img_codes).mean(axis=0, dtype=np.float64)
    txt_mean = np.asarray(txt_codes).mean(axis=0, dtype=np.float64)
    maximum, minimum = np.maximum(img_mean, txt_mean), np.minimum(img_mean, txt_mean)
    dead = maximum < near_zero_mean
    private = ~dead & (minimum / np.maximum(maximum, 1e-12) <= private_min_over_max)
    return np.flatnonzero(dead), np.flatnonzero(private)


def community_spanning(img_codes, txt_codes, labels, near_zero_mean, single_community_mass) -> dict:
    combined = 0.5 * (np.asarray(img_codes, np.float64) + np.asarray(txt_codes, np.float64))
    labels = np.asarray(labels, dtype=np.int64)
    n_communities = int(labels.max()) + 1
    largest = np.zeros(combined.shape[1])
    for factor in range(combined.shape[1]):
        mass = np.bincount(labels, weights=combined[:, factor], minlength=n_communities)
        if mass.sum() > 0:
            largest[factor] = mass.max() / mass.sum()
    dead_idx, _ = dead_and_private(img_codes, txt_codes, near_zero_mean, 0.0)
    dead = np.zeros(combined.shape[1], dtype=bool)
    dead[dead_idx] = True
    topic_like = ~dead & (largest >= single_community_mass)
    spanning = ~dead & ~topic_like
    return {"spanning_indices": np.flatnonzero(spanning).tolist(),
            "topic_like_indices": np.flatnonzero(topic_like).tolist(),
            "spanning_fraction": float(spanning.sum() / combined.shape[1])}


def linear_readout_rel_l2(fit_codes, fit_features, eval_codes, eval_features, ridge=1e-3) -> float:
    """Best affine map codes -> features (scale-invariant ridge), scored on eval rows."""
    x = np.asarray(fit_codes, np.float64)
    y = np.asarray(fit_features, np.float64)
    x_mean, y_mean = x.mean(0), y.mean(0)
    xc = x - x_mean
    gram = xc.T @ xc
    trace = np.trace(gram)
    if trace <= 0:
        prediction = np.broadcast_to(y_mean, np.shape(eval_features))
    else:
        penalty = ridge * trace / gram.shape[0] * np.eye(gram.shape[0])
        weights = np.linalg.solve(gram + penalty, xc.T @ (y - y_mean))
        prediction = (np.asarray(eval_codes, np.float64) - x_mean) @ weights + y_mean
    return relative_l2(eval_features, prediction)


def pca_rel_l2(fit_features, eval_features, rank: int) -> float:
    fit = np.asarray(fit_features, np.float64)
    mean = fit.mean(0)
    basis = np.linalg.svd(fit - mean, full_matrices=False)[2][:rank]
    centered = np.asarray(eval_features, np.float64) - mean
    return relative_l2(eval_features, centered @ basis.T @ basis + mean)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def paired_retrieval_recall(query, gallery, pool=1000, k=10, seed=42) -> float:
    """Row i of query must retrieve row i of gallery among `pool` rows; ties count against."""
    n = len(query)
    n_pools = n // pool
    if n_pools == 0 or len(gallery) != n:
        raise ValueError("Need matching query/gallery with at least one full pool")
    order = np.random.default_rng(seed).permutation(n)
    q, g = _unit_rows(query), _unit_rows(gallery)
    hits = 0
    for index in range(n_pools):
        rows = order[index * pool:(index + 1) * pool]
        scores = q[rows] @ g[rows].T
        rank = (scores >= np.diag(scores)[:, None]).sum(axis=1)
        hits += int((rank <= k).sum())
    return hits / (n_pools * pool)


@dataclass
class FactorGateReport:
    values: dict
    passed: dict

    @property
    def all_passed(self) -> bool:
        return all(self.passed.values())


def evaluate_factor_gates(*, fit_img_codes, fit_txt_codes, fit_img_features, fit_txt_features,
                          eval_img_codes, eval_txt_codes, eval_img_features, eval_txt_features,
                          community_img_codes, community_txt_codes, community_labels,
                          thresholds: FactorGateThresholds = FactorGateThresholds()) -> FactorGateReport:
    t = thresholds
    fit_rows = np.arange(len(fit_img_codes))
    if len(fit_rows) > t.readout_fit_rows:
        fit_rows = np.sort(np.random.default_rng(t.seed).choice(fit_rows, t.readout_fit_rows, replace=False))
    values: dict = {}
    for name, fit_codes, fit_feat, eval_codes, eval_feat in (
        ("img", fit_img_codes, fit_img_features, eval_img_codes, eval_img_features),
        ("txt", fit_txt_codes, fit_txt_features, eval_txt_codes, eval_txt_features),
    ):
        values[f"participation_ratio_{name}"] = participation_ratio(eval_codes)
        values[f"active_fraction_{name}"] = active_fraction(eval_codes)
        values[f"readout_{name}"] = linear_readout_rel_l2(
            np.asarray(fit_codes)[fit_rows], np.asarray(fit_feat)[fit_rows], eval_codes, eval_feat, t.readout_ridge)
        values[f"pca{t.readout_pca_rank}_{name}"] = pca_rel_l2(
            np.asarray(fit_feat)[fit_rows], eval_feat, t.readout_pca_rank)
    pair = 0.5 * (np.asarray(eval_img_codes, np.float64) + np.asarray(eval_txt_codes, np.float64))
    values["correlation"] = factor_correlation_summary(pair)
    dead, private = dead_and_private(eval_img_codes, eval_txt_codes, t.near_zero_mean, t.private_min_over_max)
    values["dead_indices"], values["private_indices"] = dead.tolist(), private.tolist()
    values["top2_mass_share"] = top2_mass_share(eval_img_codes, eval_txt_codes)
    values["community"] = community_spanning(community_img_codes, community_txt_codes, community_labels,
                                             t.near_zero_mean, t.single_community_mass)
    code_recall = 0.5 * (paired_retrieval_recall(eval_img_codes, eval_txt_codes, t.retrieval_pool, t.retrieval_k, t.seed)
                         + paired_retrieval_recall(eval_txt_codes, eval_img_codes, t.retrieval_pool, t.retrieval_k, t.seed))
    clip_recall = 0.5 * (paired_retrieval_recall(eval_img_features, eval_txt_features, t.retrieval_pool, t.retrieval_k, t.seed)
                         + paired_retrieval_recall(eval_txt_features, eval_img_features, t.retrieval_pool, t.retrieval_k, t.seed))
    if clip_recall <= 0:
        raise ValueError("CLIP pair retrieval is zero; the retrieval gate is undefined")
    values["code_retrieval_recall"], values["clip_retrieval_recall"] = code_recall, clip_recall
    values["retrieval_ratio"] = code_recall / clip_recall
    rank = t.readout_pca_rank
    passed = {
        "participation_ratio": bool(min(values["participation_ratio_img"], values["participation_ratio_txt"]) >= t.min_participation_ratio),
        "redundancy": bool(values["correlation"]["max_abs"] <= t.max_pair_abs_corr and not values["correlation"]["constant_factors"]),
        "readout": bool(values["readout_img"] <= values[f"pca{rank}_img"] and values["readout_txt"] <= values[f"pca{rank}_txt"]),
        "sparsity": bool(max(values["active_fraction_img"], values["active_fraction_txt"]) <= t.max_active_fraction),
        "dead": len(dead) <= t.max_dead,
        "modality_private": len(private) <= t.max_private,
        "usage_concentration": values["top2_mass_share"] <= t.max_top2_mass_share,
        "community_spanning": values["community"]["spanning_fraction"] >= t.min_spanning_fraction,
        "pair_retrieval": values["retrieval_ratio"] >= t.min_retrieval_ratio,
    }
    return FactorGateReport(values=values, passed=passed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_factor_gates.py`
Expected: 7 passed. Then run the full suite, `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/`.
Every previously passing test still passes.

- [ ] **Step 5: Real sanity check on the known-collapsed codes.** Write `run_check.py`:
  - Load `src/test/20261007_naive_rule_mechanism_analysis/cache/factor42_{img,txt}.npy`.
  - Load features with `load_real_features` and the row split with `split_items`, both imported
    from `src/test/20261007_naive_rule_mechanism_analysis/run_mechanism.py`.
  - Fit rows are the row-split train rows; eval rows are the held rows.
  - Community labels: rebuild the train-row graph (`build_content_graph(..., GraphConfig())`), run
    `train_stage1(..., Stage1Config())` on the train rows, then `detect_communities`, exactly as
    `prepare_item_disjoint_codes` in `src/test/20261005_condition_ranking_evaluation/run_ranking_eval.py:51-94`
    does.
  - Run `evaluate_factor_gates` with default thresholds. Print every value and gate. Save the
    JSON as `gates_collapsed.json` (gitignored).

  **Expected:** `participation_ratio`, `redundancy`, `readout` and `sparsity` all FAIL. If any of
  those four passes, stop: that is a gate bug, and the controller reviews before Task 2. Record
  every value, including the retrieval ratio and CLIP retrieval baseline (reference numbers for
  Tasks 3 and 5), in the log file.

- [ ] **Step 6: Commit**

```bash
git add src/eval/__init__.py src/eval/factor_gates.py src/test/test_factor_gates.py src/test/20261008_factor_gates_collapsed_check/
git commit -m "feat(cosir-v2): factor-geometry pass/fail gates (dimensionality, redundancy, readout, sparsity, retrieval)"
```

---

### Task 2: ArtELingo loader + painting-grouped split

**Files:**
- Create: `src/data/__init__.py` (empty), `src/data/artelingo.py`, `src/data/splits.py`
- Test: `src/test/test_artelingo_loader.py`, `src/test/test_splits.py`
- Create (real check): `src/test/20261008_painting_grouped_split/check_split.py`,
  `src/test/20261008_painting_grouped_split/20261008_painting_grouped_split_log.md`

**Interfaces:**
- Consumes: `FeatureManager` from `src.utils`. Copy the loading pattern from
  `src/test/20260928_factor_discovery_validation/run_validation.py:94-110`.
- Produces (used by Tasks 3, 6 and 7):
  - `ArtelingoData` (frozen dataclass: `img_features`, `txt_features`: `(N, 512) float32`;
    `sample_ids: (N,) int64`; `emotions`, `paintings`: `(N,) str`)
  - `join_annotations(sample_ids, annotations) -> tuple[np.ndarray, np.ndarray]`
  - `load_artelingo(feature_dir=FEATURE_DIR, annotations_path=ANNOTATIONS_PATH, expected_samples=308_723) -> ArtelingoData`
  - `image_hashes(img_features) -> np.ndarray`
  - `leakage_groups(paintings, img_features) -> np.ndarray`: int64 group id per row
  - `GroupedSplit(train, val, held)`: sorted int64 row indices
  - `grouped_split(groups, fractions=(0.7, 0.1, 0.2), seed=42) -> GroupedSplit`
  - `split_leakage(split, paintings, img_features) -> dict[str, int]`

- [ ] **Step 1: Write the failing tests**

`src/test/test_splits.py`:

```python
import numpy as np
import pytest

from src.data.splits import GroupedSplit, grouped_split, leakage_groups, split_leakage


def _features(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 8)).astype(np.float32)


def test_rows_sharing_an_image_vector_join_one_group_across_paintings():
    features = _features(4)
    features[3] = features[0]                      # different painting, identical image vector
    paintings = np.array(["a", "a", "b", "c"])
    groups = leakage_groups(paintings, features)
    assert groups[0] == groups[1] == groups[3] and groups[2] != groups[0]


def test_grouped_split_never_splits_a_group_and_hits_fractions():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(2000), rng.integers(1, 9, size=2000))
    split = grouped_split(groups, (0.7, 0.1, 0.2), seed=42)
    parts = [set(groups[idx]) for idx in (split.train, split.val, split.held)]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert len(split.train) + len(split.val) + len(split.held) == len(groups)
    shares = np.array([len(split.train), len(split.val), len(split.held)]) / len(groups)
    assert np.allclose(shares, (0.7, 0.1, 0.2), atol=0.01)


def test_grouped_split_is_deterministic_and_seed_sensitive():
    groups = np.repeat(np.arange(500), 3)
    a, b = grouped_split(groups, seed=42), grouped_split(groups, seed=42)
    assert np.array_equal(a.held, b.held)
    assert not np.array_equal(a.held, grouped_split(groups, seed=43).held)


def test_grouped_split_rejects_bad_fractions_and_empty_parts():
    with pytest.raises(ValueError):
        grouped_split(np.arange(10), (0.5, 0.5))
    with pytest.raises(ValueError):
        grouped_split(np.arange(10), (0.7, 0.2, 0.2))
    with pytest.raises(ValueError):
        grouped_split(np.zeros(10, dtype=int), (0.7, 0.1, 0.2))   # one group -> empty parts


def test_split_leakage_is_zero_for_grouped_split_and_nonzero_for_row_split():
    features = _features(300)
    paintings = np.array([f"p{i // 3}" for i in range(300)])
    features[1::3] = features[0::3]                # each painting's rows share one image vector
    split = grouped_split(leakage_groups(paintings, features), seed=42)
    assert all(value == 0 for value in split_leakage(split, paintings, features).values())
    perm = np.random.default_rng(0).permutation(300)
    row_split = GroupedSplit(np.sort(perm[:210]), np.sort(perm[210:240]), np.sort(perm[240:]))
    assert split_leakage(row_split, paintings, features)["held_rows_painting_in_train"] > 0
```

`src/test/test_artelingo_loader.py`:

```python
import numpy as np
import pytest

from src.data.artelingo import join_annotations


def test_join_annotations_is_positional_by_sample_id():
    annotations = [{"emotion": "awe", "painting": "p0"}, {"emotion": "fear", "painting": "p1"}]
    emotions, paintings = join_annotations(np.array([1, 0]), annotations)
    assert emotions.tolist() == ["fear", "awe"] and paintings.tolist() == ["p1", "p0"]


@pytest.mark.parametrize("ids", [[0, 0], [0, 2], [-1, 0]])
def test_join_annotations_rejects_duplicate_or_out_of_range_ids(ids):
    annotations = [{"emotion": "awe", "painting": "p0"}, {"emotion": "fear", "painting": "p1"}]
    with pytest.raises(ValueError):
        join_annotations(np.array(ids), annotations)


def test_join_annotations_rejects_missing_or_empty_fields():
    with pytest.raises(ValueError):
        join_annotations(np.array([0]), [{"emotion": "awe"}])
    with pytest.raises(ValueError):
        join_annotations(np.array([0]), [{"emotion": "awe", "painting": ""}])
```

- [ ] **Step 2: Run to verify failure**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_splits.py src/test/test_artelingo_loader.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 3: Implement `src/data/artelingo.py`**

```python
"""Load cached ArtELingo CLIP features with their positional annotation join."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils import FeatureManager

FEATURE_DIR = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")


@dataclass(frozen=True)
class ArtelingoData:
    img_features: np.ndarray
    txt_features: np.ndarray
    sample_ids: np.ndarray
    emotions: np.ndarray
    paintings: np.ndarray


def join_annotations(sample_ids: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """annotations[sample_id] per feature row; ids must be unique and in range."""
    ids = np.asarray(sample_ids, dtype=np.int64)
    if len(ids) == 0 or ids.min() < 0 or ids.max() >= len(annotations) or len(np.unique(ids)) != len(ids):
        raise ValueError("Sample IDs must uniquely index the annotation list")
    rows = [annotations[int(i)] for i in ids]
    if any(not row.get("emotion") or not row.get("painting") for row in rows):
        raise ValueError("Every ArtELingo row needs non-empty emotion and painting fields")
    return (np.asarray([row["emotion"] for row in rows]),
            np.asarray([row["painting"] for row in rows]))


def load_artelingo(feature_dir: str = FEATURE_DIR, annotations_path: Path = ANNOTATIONS_PATH,
                   expected_samples: int = 308_723) -> ArtelingoData:
    manager = FeatureManager(storage_dir=feature_dir)
    with Path(annotations_path).open() as file:
        annotations = json.load(file)
    if not (len(annotations) == manager.total_samples == expected_samples):
        raise ValueError("Annotation/feature count mismatch")
    features = manager.load_all_to_ram(["img_features", "txt_features"])
    sample_ids = np.asarray(manager.get_all_sample_ids(), dtype=np.int64)
    img = features["img_features"].numpy().astype(np.float32, copy=False)
    txt = features["txt_features"].numpy().astype(np.float32, copy=False)
    if not (len(sample_ids) == len(img) == len(txt) == expected_samples):
        raise ValueError("Feature rows and sample IDs do not match metadata")
    emotions, paintings = join_annotations(sample_ids, annotations)
    return ArtelingoData(img, txt, sample_ids, emotions, paintings)
```

- [ ] **Step 4: Implement `src/data/splits.py`**

```python
"""Leakage-free train/val/held splits grouped by painting and identical image vector."""

import hashlib
from dataclasses import dataclass

import numpy as np


def image_hashes(img_features: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(img_features, dtype=np.float32)
    return np.asarray([hashlib.md5(row.tobytes()).hexdigest() for row in rows])


def leakage_groups(paintings: np.ndarray, img_features: np.ndarray) -> np.ndarray:
    """Union rows that share a painting or an exact image vector; return dense group ids."""
    n = len(paintings)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for keys in (np.asarray(paintings).tolist(), image_hashes(img_features).tolist()):
        first: dict = {}
        for row, key in enumerate(keys):
            if key in first:
                a, b = find(row), find(first[key])
                if a != b:
                    parent[a] = b
            else:
                first[key] = row
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    return np.unique(roots, return_inverse=True)[1].astype(np.int64)


@dataclass(frozen=True)
class GroupedSplit:
    train: np.ndarray
    val: np.ndarray
    held: np.ndarray


def grouped_split(groups: np.ndarray, fractions=(0.7, 0.1, 0.2), seed: int = 42) -> GroupedSplit:
    """Shuffle whole groups, then cut by cumulative row share at the fraction boundaries."""
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be three positive numbers summing to 1")
    groups = np.asarray(groups)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    shares = counts[order] / counts.sum()
    start = np.cumsum(shares) - shares
    group_part = np.empty(len(unique), dtype=np.int64)
    group_part[order] = np.searchsorted(np.cumsum(fractions)[:2], start, side="right")
    row_part = group_part[inverse]
    parts = [np.flatnonzero(row_part == p).astype(np.int64) for p in range(3)]
    if any(len(p) == 0 for p in parts):
        raise ValueError("A split part is empty; too few groups for these fractions")
    return GroupedSplit(*parts)


def split_leakage(split: GroupedSplit, paintings: np.ndarray, img_features: np.ndarray) -> dict:
    """Rows of val/held whose painting or exact image vector also appears in an earlier part."""
    paintings = np.asarray(paintings)
    hashes = image_hashes(img_features)
    checks = (("val", "train", split.val, split.train),
              ("held", "train", split.held, split.train),
              ("held", "val", split.held, split.val))
    out = {}
    for part, reference_name, rows, reference in checks:
        seen_paintings, seen_hashes = set(paintings[reference]), set(hashes[reference])
        out[f"{part}_rows_painting_in_{reference_name}"] = int(sum(p in seen_paintings for p in paintings[rows]))
        out[f"{part}_rows_image_in_{reference_name}"] = int(sum(h in seen_hashes for h in hashes[rows]))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_splits.py src/test/test_artelingo_loader.py`
Expected: all pass. Then run the full suite, `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/`.

- [ ] **Step 6: Real split check.** Write `check_split.py`: run `load_artelingo()`,
  `leakage_groups`, `grouped_split(seed=42)`, `split_leakage`. Print and log:
  - row, painting and group counts per part;
  - row share per part;
  - emotion distribution per part (a counts table);
  - the size of the largest group;
  - the leakage dict, where every value must be 0.

  **Expected:** every leakage value is 0, and shares are within 1 point of 70/10/20. If a leakage
  value is non-zero, stop: it is a bug.

- [ ] **Step 7: Commit**

```bash
git add src/data/ src/test/test_splits.py src/test/test_artelingo_loader.py src/test/20261008_painting_grouped_split/
git commit -m "feat(cosir-v2): ArtELingo loader + painting/image-grouped train/val/held split"
```

---

### Task 3: locate the collapse (one variable at a time, existing code only)

**Files:**
- Create: `src/test/20261009_factor_collapse_diagnosis/run_diagnosis.py`,
  `src/test/20261009_factor_collapse_diagnosis/20261009_factor_collapse_diagnosis_log.md`,
  `src/test/20261009_factor_collapse_diagnosis/.gitignore` (`*.npy`, `*.json`, `*.pt`, `cache/`)
- Create: `docs/reports/2026-10-09_cosir_v2_candidate_a_factor_collapse_diagnosis.md`
- No `src/` changes. If a variant seems to need one, stop and report instead.

**Interfaces:**
- Consumes:
  - From Task 2: `load_artelingo`, `leakage_groups`, `grouped_split`.
  - From Task 1: `evaluate_factor_gates`, `FactorGateThresholds`.
  - Existing: `build_content_graph`, `GraphConfig` (`src/model/graph.py`); `train_stage1`,
    `Stage1Config` (`src/train/stage1.py`); `detect_communities` (`src/model/communities.py`);
    `train_factors`, `FactorTrainingConfig` (`src/train/train_factors.py`).
- Produces: a diagnosis report the controller uses to confirm or amend Task 4's mechanism list.

**Why this task exists:** the code review's causes are plausible but untested:
- a non-contrastive agreement loss;
- no decorrelation term;
- a balance loss that copies of one axis can satisfy;
- weak sparsity;
- uncentered inputs.

Centering is a reparameterization of an affine encoder (`W(x−μ)+b = Wx + (b−Wμ)`), so it can only
matter through optimization. Its prior is low, but it is cheap to check.

- [ ] **Step 1: Setup, shared by all variants.**
  - Load data. Build groups and the split (seed 42).
  - Build the graph on `train` rows only, then Stage 1 and communities on `train` rows.
  - Every variant trains on `train` rows with `train_factors(...)`. Codes for `val` rows come from
    the returned model's `encode_image` / `encode_text` in eval mode (8,192-row batches, as
    `prepare_item_disjoint_codes` does).
  - Gates: fit rows = `train`, eval rows = `val`, community codes/labels = `train`.

- [ ] **Step 2: Run exactly these variants** (seed 42, 32 factors, 2,000 epochs, everything else at
  `FactorTrainingConfig` defaults unless listed):

| ID | Change from D0 | Question |
|---|---|---|
| D0 | `lambda_usage_balance=0.1` (the Task 6 / Task 7 recipe) | Does the collapse reproduce on the painting split? |
| D1 | reconstruction only: `lambda_paired=lambda_graph=lambda_sparsity=lambda_anti_split=lambda_usage_balance=0` | Does a plain ReLU autoencoder already collapse? |
| D2 | D1 + `lambda_sparsity=0.01` | Does L1 alone cause it? |
| D3 | D0 with `lambda_paired=0` | Is the cosine agreement loss the cause? |
| D4 | D0 with `lambda_usage_balance=0` (the Task 3 recipe) | Is the balance loss the cause? |
| D5 | D0 with `lambda_graph=0` | Is the graph term the cause? |
| D6 | D0 with only the **encoder input** centered by its `train`-row means (pass centered arrays to `train_factors`; gates still use the original uncentered features, which is valid because the readout has an intercept, so values stay comparable across variants) | Does centering change the optimum? |

- [ ] **Step 3: Pre-registered reading.** For every variant, report every gate value and pass/fail.
  - A term is **implicated** if removing it (D3–D5 vs D0) at least doubles the smaller of the two
    per-modality participation ratios, or brings max |r| from above 0.9 to below 0.9.
  - If D1 is itself collapsed (PR < 3), the collapse does not need any auxiliary loss. Say so
    plainly: in that case the fix must add pressure (decorrelation, contrastive agreement, TopK),
    not only remove a term.
  - Also report the D0 row next to Task 1's collapsed-code numbers (row split) as a split-effect
    comparison.

- [ ] **Step 4: Write the report** with the verdict first: which terms are implicated, and which of
  Task 4's mechanisms (InfoNCE agreement, decorrelation, TopK, centering) the evidence supports.
  **Controller checkpoint:** if the evidence points to a cause Task 4 does not address, the
  controller amends Task 4 and Task 6's grid with the user before Task 4 starts.

- [ ] **Step 5: Commit**

```bash
git add src/test/20261009_factor_collapse_diagnosis/ docs/reports/2026-10-09_cosir_v2_candidate_a_factor_collapse_diagnosis.md
git commit -m "docs(cosir-v2): one-variable-at-a-time diagnosis of the factor-space collapse"
```

---

### Task 4: fix mechanisms (default-off) + checkpoints

**Files:**
- Modify: `src/model/factors.py` (activation option, persisted input means)
- Modify: `src/train/factors.py` (add `cross_modal_infonce_loss`, `decorrelation_penalty`)
- Modify: `src/train/train_factors.py` (new config fields, `group_ids`, checkpoint helpers)
- Test: `src/test/test_factors.py`, `src/test/test_factor_losses.py`, `src/test/test_train_factors.py` (additions only)
- Create: `.claude/20261010_log.md` (change log for the three modified files; `.claude/` is
  gitignored in this repo, so the log is written but not committed)

**Interfaces:**
- Consumes: nothing from earlier tasks. These are code-only changes.
- Produces (used by Tasks 6 and 7):
  - `SharedFactorEncoder(feature_dim, num_factors, dropout=0.1, activation="relu", topk=None, image_mean=None, text_mean=None)`.
    `activation` is `"relu"` or `"topk"`. `image_mean` and `text_mean` are `(feature_dim,)`
    arrays or `None`, stored as buffers `image_mean` / `text_mean` (zeros when `None`).
  - `cross_modal_infonce_loss(img_codes, txt_codes, temperature=0.1, group_ids=None) -> Tensor`
  - `decorrelation_penalty(codes, eps=1e-6) -> Tensor`
  - New `FactorTrainingConfig` fields:
    - `agreement: str = "cosine"` (`"cosine"` or `"infonce"`)
    - `infonce_temperature: float = 0.1`
    - `lambda_decorrelation: float = 0.0`
    - `activation: str = "relu"`
    - `topk: int | None = None`
    - `center_inputs: bool = False`
  - `train_factors(img_features, txt_features, graph, config, device=None, group_ids=None)`
  - `save_factor_checkpoint(model, config, path) -> None`
  - `load_factor_checkpoint(path, device="cpu") -> tuple[SharedFactorEncoder, FactorTrainingConfig]`
- **Invariant:** with default config the training computation is unchanged. Old scripts reproduce.

- [ ] **Step 1: Pin the pre-change behavior first.** Before editing anything, add to
  `src/test/test_train_factors.py` a golden test:
  - Build a tiny synthetic run on CPU: `N=64`, `D=16`, `L=8`, `epochs=5`, `batch_size=16`, a ring
    graph `csr_matrix` over 64 nodes, `FactorTrainingConfig(num_factors=8, epochs=5, batch_size=16)`.
  - Store `np.round(img_codes[:4], 6)` as a literal in the test, taken from the **current**
    code's output.
  - Assert equality after the change. Run it now; it must pass on unmodified code.

- [ ] **Step 2: Write the failing tests** (append to the existing test files):

```python
# src/test/test_factor_losses.py
import torch
from src.train.factors import cross_modal_infonce_loss, decorrelation_penalty, paired_agreement_loss


def test_infonce_rejects_the_collapsed_solution_that_cosine_agreement_accepts():
    collapsed = torch.ones(32, 8)                                   # every item on one direction
    assert paired_agreement_loss(collapsed, collapsed).item() < 1e-6
    assert abs(cross_modal_infonce_loss(collapsed, collapsed).item() - torch.log(torch.tensor(32.0)).item()) < 1e-4
    distinct = torch.eye(32)                                        # every item its own direction
    assert cross_modal_infonce_loss(distinct, distinct).item() < 0.05


def test_infonce_masks_same_group_false_negatives():
    torch.manual_seed(0)
    codes = torch.rand(6, 8)
    codes[1] = codes[0]                                             # same painting, near-duplicate
    groups = torch.tensor([0, 0, 1, 2, 3, 4])
    masked = cross_modal_infonce_loss(codes, codes, group_ids=groups)
    unmasked = cross_modal_infonce_loss(codes, codes)
    assert masked.item() < unmasked.item()


def test_infonce_validates_inputs():
    import pytest
    with pytest.raises(ValueError):
        cross_modal_infonce_loss(torch.rand(4, 3), torch.rand(4, 3), temperature=0.0)
    with pytest.raises(ValueError):
        cross_modal_infonce_loss(torch.rand(4, 3), torch.rand(4, 3), group_ids=torch.tensor([0, 1]))


def test_decorrelation_separates_copies_from_independent_and_ignores_constants():
    torch.manual_seed(0)
    z = torch.randn(512, 1)
    copies = torch.relu(1 + z.repeat(1, 8) + 0.01 * torch.randn(512, 8))
    independent = torch.relu(torch.randn(512, 8))
    assert decorrelation_penalty(copies).item() > 0.8
    assert decorrelation_penalty(independent).item() < 0.05
    with_constant = independent.clone()
    with_constant[:, 3] = 0.5
    value = decorrelation_penalty(with_constant)
    assert torch.isfinite(value) and value.item() < 0.05
    assert decorrelation_penalty(torch.rand(1, 8)).item() == 0.0     # one row: no variance, no NaN
```

```python
# src/test/test_factors.py
import numpy as np
import pytest
import torch
from src.model.factors import SharedFactorEncoder


def test_topk_keeps_at_most_k_nonnegative_activations_per_row():
    torch.manual_seed(0)
    model = SharedFactorEncoder(16, 8, dropout=0.0, activation="topk", topk=3).eval()
    codes = model.encode_image(torch.randn(50, 16))
    assert ((codes > 0).sum(dim=1) <= 3).all() and (codes >= 0).all()


@pytest.mark.parametrize("kwargs", [dict(activation="topk", topk=0), dict(activation="topk", topk=9),
                                    dict(activation="topk", topk=None), dict(activation="relu", topk=2),
                                    dict(activation="gelu")])
def test_invalid_activation_settings_raise(kwargs):
    with pytest.raises(ValueError):
        SharedFactorEncoder(16, 8, **kwargs)


def test_input_mean_is_subtracted_before_encoding():
    torch.manual_seed(0)
    mean = np.linspace(-1, 1, 16).astype(np.float32)
    centered = SharedFactorEncoder(16, 8, dropout=0.0, image_mean=mean, text_mean=mean).eval()
    plain = SharedFactorEncoder(16, 8, dropout=0.0).eval()
    plain.load_state_dict({k: v for k, v in centered.state_dict().items()
                           if not k.endswith("_mean")}, strict=False)
    x = torch.randn(10, 16)
    assert torch.allclose(centered.encode_image(x), plain.encode_image(x - torch.as_tensor(mean)))
```

```python
# src/test/test_train_factors.py (in addition to the Step 1 golden test)
def test_checkpoint_round_trip_gives_identical_codes(tmp_path):
    # tiny centered TopK + InfoNCE run, then save/load, then re-encode
    ...  # build the same ring-graph fixture as the golden test
    config = FactorTrainingConfig(num_factors=8, epochs=3, batch_size=16, agreement="infonce",
                                  activation="topk", topk=3, center_inputs=True, lambda_decorrelation=1.0)
    model, img_codes, _ = train_factors(img, txt, graph, config, device="cpu", group_ids=np.arange(64) // 2)
    save_factor_checkpoint(model, config, tmp_path / "f.pt")
    loaded, loaded_config = load_factor_checkpoint(tmp_path / "f.pt")
    with torch.no_grad():
        again = loaded.encode_image(torch.as_tensor(img, dtype=torch.float32)).numpy()
    assert loaded_config == config and np.array_equal(again, img_codes)


def test_group_ids_length_mismatch_and_unknown_agreement_raise():
    ...  # same fixture; FactorTrainingConfig(agreement="bogus") -> ValueError;
         # group_ids of length 63 with agreement="infonce" -> ValueError
```

Write out the elided fixture lines (`...`) in full in the real test file. They repeat the golden
test's ring-graph fixture.

- [ ] **Step 3: Run to verify failure**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_factors.py src/test/test_factor_losses.py src/test/test_train_factors.py`
Expected: the new tests FAIL with `ImportError` or `TypeError`. The golden test passes.

- [ ] **Step 4: Implement.**

`src/train/factors.py`, add:

```python
def cross_modal_infonce_loss(img_codes: Tensor, txt_codes: Tensor, temperature: float = 0.1,
                             group_ids: Tensor | None = None) -> Tensor:
    """Symmetric image<->text InfoNCE on L2-normalized codes.

    Unlike paired_agreement_loss, a single code direction shared by every item
    scores log(B), not 0: matched pairs must beat in-batch negatives. Off-diagonal
    pairs with equal group_ids (same painting / image) are excluded as negatives.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = F.normalize(img_codes, dim=1) @ F.normalize(txt_codes, dim=1).T / temperature
    if group_ids is not None:
        group_ids = torch.as_tensor(group_ids, device=logits.device)
        if group_ids.shape != (len(img_codes),):
            raise ValueError("group_ids must have one entry per row")
        same = group_ids[:, None] == group_ids[None, :]
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))
    targets = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def decorrelation_penalty(codes: Tensor, eps: float = 1e-6) -> Tensor:
    """Mean squared off-diagonal batch correlation over factors with non-zero std.

    Constant factors are excluded (no NaN); the dead/constant gates in
    src/eval/factor_gates.py catch a model that escapes by going constant.
    """
    if len(codes) < 2:
        return codes.sum() * 0.0
    std = codes.std(dim=0)
    varying = std > eps
    k = int(varying.sum())
    if k < 2:
        return codes.sum() * 0.0
    z = (codes[:, varying] - codes[:, varying].mean(dim=0)) / std[varying]
    corr = z.T @ z / (len(codes) - 1)
    off = corr - torch.diag(torch.diagonal(corr))
    return (off**2).sum() / (k * (k - 1))
```

`src/model/factors.py`: the constructor validates:
- `activation in {"relu", "topk"}`;
- for `"topk"`, `1 <= topk <= num_factors`;
- for `"relu"`, `topk is None`;
- each mean has shape `(feature_dim,)`.

Register buffers with `self.register_buffer("image_mean", ...)` and `self.register_buffer("text_mean", ...)`
(zeros if `None`). Encoding becomes
`self._activate(self.image_encoder(self.image_dropout(img_feat - self.image_mean)))`, and likewise for text. The activation:

```python
def _activate(self, pre: Tensor) -> Tensor:
    if self.activation == "relu":
        return F.relu(pre)
    values, indices = pre.topk(self.topk, dim=1)
    return torch.zeros_like(pre).scatter(1, indices, F.relu(values))
```

Subtracting a zero buffer is exact, so default behavior is bit-identical. Decoders are unchanged:
they reconstruct the original features, and their bias absorbs the mean.

`src/train/train_factors.py`:
- Add the six config fields.
- In `train_factors`:
  - Validate `config.agreement`, and `group_ids` length when given.
  - When `center_inputs`, pass `image_mean=img.mean(0)` and `text_mean=txt.mean(0)`.
  - Build the model with `activation` / `topk`.
  - The agreement term is `paired_agreement_loss(...)` for `"cosine"`, else
    `cross_modal_infonce_loss(img_codes, txt_codes, config.infonce_temperature, None if group_ids is None else torch.as_tensor(group_ids[node_ids], device=selected_device))`.
  - Add `config.lambda_decorrelation * 0.5 * (decorrelation_penalty(img_codes) + decorrelation_penalty(txt_codes))`
    **only when** `lambda_decorrelation > 0`, so the default graph is unchanged.
- Checkpoint helpers:

```python
def save_factor_checkpoint(model: SharedFactorEncoder, config: FactorTrainingConfig, path) -> None:
    torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                "feature_dim": model.image_encoder.in_features}, path)


def load_factor_checkpoint(path, device: str = "cpu") -> tuple[SharedFactorEncoder, FactorTrainingConfig]:
    payload = torch.load(path, map_location=device, weights_only=True)
    config = FactorTrainingConfig(**payload["config"])
    model = SharedFactorEncoder(payload["feature_dim"], config.num_factors,
                                activation=config.activation, topk=config.topk)
    model.load_state_dict(payload["state_dict"])     # restores image_mean / text_mean buffers
    return model.to(device).eval(), config
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/`
Expected: all pass, including the Step 1 golden test (default behavior unchanged) and every
pre-existing test.

- [ ] **Step 6: Write `.claude/20261010_log.md`:** one `# /src/...` section per modified file,
  with before/after snippets and the reason (cite Task 3's report and the code review).

- [ ] **Step 7: Commit**

```bash
git add src/model/factors.py src/train/factors.py src/train/train_factors.py src/test/test_factors.py src/test/test_factor_losses.py src/test/test_train_factors.py
git commit -m "feat(cosir-v2): default-off factor-collapse fixes (InfoNCE agreement, decorrelation, TopK, centering) + checkpoints"
```

---

### Task 5: conditioning module + label-defined episode evaluator

**Why this comes before the repair grid:** Task 6 selects a factor recipe by the goal it serves,
conditional cross-item matching, not by reconstruction. The only non-circular conditions
available are ArtELingo's two human labels:
- **`emotion`**: 9 values, set per annotation, subjective and often expressed in the caption.
- **`art_style`**: 27 WikiArt styles, set per painting, largely visual.

Factor-mined episodes can't be used for selection: each model mines episodes from its own codes,
so they reward self-consistency, not aspects humans recognize.

**Files:**
- Create: `src/model/conditioning.py`; test `src/test/test_conditioning.py`
- Create: `src/eval/label_episodes.py`; test `src/test/test_label_episodes.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. These are code-only changes.
- Produces (used by Tasks 6 and 7, and later stage (d)):
  - `pair_codes(img_codes, txt_codes) -> Tensor`
  - `naive_condition_weights(support_pair_codes, contrast_pair_codes=None, top_k=None) -> Tensor`
    - input shapes `[..., S, L]` and `[..., C, L]`; returns `[..., L]`;
    - computes `ReLU(mean_S − mean_C)`, optionally keeps the `top_k` largest, then L1-normalizes;
    - an all-zero result stays zero.
  - `conditional_score(query_feat, cand_feat, query_codes, cand_codes, weights, beta) -> Tensor`
    - shapes: `[B, D]`, `[B, M, D]`, `[B, L]`, `[B, M, L]`, `[B, L]` → `[B, M]`;
    - computes `beta·cos(query, cand) + Σ_l w_l·q_l·c_l`.
  - `LabelEpisodes`: frozen dataclass. `anchor: (E,)`, `positive: (E,)`, `supports: (E, S)`,
    `contrasts: (E, C)`, `distractors: (E, M)` (global row indices, int64), and `labels: (E,)`
    (the condition label of each episode).
  - `build_label_episodes(labels, paintings, rows, n_episodes, seed=42, num_support=4, num_contrast=4, num_distractors=12, min_paintings_per_label=30, exclude_target_labels=()) -> LabelEpisodes`
  - `tie_aware_rank(scores) -> Tensor`: positive in column 0; returns
    `1 + #(score > positive) + 0.5·#(score == positive)` over the other columns.
  - `label_episode_recall(img_feat, txt_feat, img_codes, txt_codes, episodes, weights, beta) -> dict`
    - keys `"i2t"` / `"t2i"`, each holding `ranks`, `recall1`, `recall3` and `tied_episodes`.
  - `condition_lift(img_feat, txt_feat, img_codes, txt_codes, episodes) -> dict`
    - scored at β=0: the ranking then depends on the factor space alone and is invariant to
      weight scale, so naive vs uniform is fair with no β to tune;
    - keys `"naive"`, `"uniform"`, `"lift"` (per direction: naive R@1 − uniform R@1) and
      `"lift_mean"`;
    - this is **Task 6's selection metric**.

- [ ] **Step 1: Write the failing tests**

`src/test/test_conditioning.py`:

```python
import torch

from src.model.conditioning import conditional_score, naive_condition_weights, pair_codes


def test_naive_weights_are_l1_normalized_relu_gap_and_zero_stays_zero():
    support = torch.tensor([[[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]]])
    contrast = torch.tensor([[[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]])
    w = naive_condition_weights(support, contrast)
    assert torch.allclose(w, torch.tensor([[0.5, 0.0, 0.5]]))
    assert torch.equal(naive_condition_weights(support, support), torch.zeros(1, 3))


def test_top_k_keeps_only_the_largest_gaps():
    support = torch.tensor([[[3.0, 2.0, 1.0, 0.5]]])
    w = naive_condition_weights(support, None, top_k=2)
    assert torch.allclose(w, torch.tensor([[0.6, 0.4, 0.0, 0.0]]))


def test_zero_weights_reduce_score_to_scaled_clip_cosine():
    torch.manual_seed(0)
    q, c = torch.randn(2, 5), torch.randn(2, 4, 5)
    qc, cc = torch.rand(2, 3), torch.rand(2, 4, 3)
    score = conditional_score(q, c, qc, cc, torch.zeros(2, 3), beta=0.3)
    expected = 0.3 * torch.nn.functional.cosine_similarity(q[:, None, :], c, dim=-1)
    assert torch.allclose(score, expected, atol=1e-6)


def test_beta_zero_ranking_is_invariant_to_weight_scale():
    torch.manual_seed(1)
    q, c = torch.randn(3, 5), torch.randn(3, 6, 5)
    qc, cc, w = torch.rand(3, 4), torch.rand(3, 6, 4), torch.rand(3, 4)
    a = conditional_score(q, c, qc, cc, w, beta=0.0).argsort(dim=1)
    b = conditional_score(q, c, qc, cc, 7.0 * w, beta=0.0).argsort(dim=1)
    assert torch.equal(a, b)


def test_pair_codes_is_the_modality_mean():
    assert torch.equal(pair_codes(torch.ones(2, 3), torch.zeros(2, 3)), torch.full((2, 3), 0.5))
```

`src/test/test_label_episodes.py`:

```python
import numpy as np
import pytest
import torch

from src.eval.label_episodes import (
    build_label_episodes, condition_lift, label_episode_recall, tie_aware_rank,
)


def _fixture(seed=0):
    """3 labels with unequal sizes; 3 annotation rows per painting; label is per painting."""
    rng = np.random.default_rng(seed)
    painting_labels = np.repeat(np.array(["a", "b", "c"]), [400, 120, 60])
    paintings = np.repeat(np.arange(len(painting_labels)), 3).astype(str)
    labels = np.repeat(painting_labels, 3)
    return labels, paintings, rng


def test_episodes_respect_labels_and_never_repeat_a_painting():
    labels, paintings, _ = _fixture()
    rows = np.arange(len(labels))
    ep = build_label_episodes(labels, paintings, rows, 300, seed=42)
    for i in range(300):
        same = [ep.anchor[i], ep.positive[i], *ep.supports[i]]
        other = [*ep.contrasts[i], *ep.distractors[i]]
        assert all(labels[r] == ep.labels[i] for r in same)
        assert all(labels[r] != ep.labels[i] for r in other)
        members = same + other
        assert len({paintings[r] for r in members}) == len(members)


def test_target_labels_are_balanced_and_rows_are_respected():
    labels, paintings, _ = _fixture()
    ep = build_label_episodes(labels, paintings, np.arange(len(labels)), 900, seed=42)
    shares = np.array([(ep.labels == x).mean() for x in ("a", "b", "c")])
    assert np.allclose(shares, 1 / 3, atol=0.06)                        # balanced despite 400/120/60
    subset = np.arange(0, len(labels), 2)                                # every label still present
    ep_subset = build_label_episodes(labels, paintings, subset, 50, seed=42)
    used = np.concatenate([ep_subset.anchor, ep_subset.positive, ep_subset.supports.ravel(),
                           ep_subset.contrasts.ravel(), ep_subset.distractors.ravel()])
    assert set(used.tolist()) <= set(subset.tolist())


def test_small_and_excluded_labels_are_never_targets_and_two_are_required():
    labels, paintings, _ = _fixture()
    rows = np.arange(len(labels))
    ep = build_label_episodes(labels, paintings, rows, 200, seed=42, min_paintings_per_label=100)
    assert set(ep.labels.tolist()) == {"a", "b"}                       # "c" has 60 paintings
    ep2 = build_label_episodes(labels, paintings, rows, 200, seed=42, exclude_target_labels=("a",))
    assert "a" not in set(ep2.labels.tolist())
    assert any(labels[r] == "a" for r in ep2.distractors.ravel())       # still usable as a negative
    with pytest.raises(ValueError):
        build_label_episodes(labels, paintings, rows, 10, seed=42, min_paintings_per_label=100,
                             exclude_target_labels=("b",))


def test_builder_is_deterministic():
    labels, paintings, _ = _fixture()
    rows = np.arange(len(labels))
    a = build_label_episodes(labels, paintings, rows, 40, seed=42)
    b = build_label_episodes(labels, paintings, rows, 40, seed=42)
    assert np.array_equal(a.distractors, b.distractors) and np.array_equal(a.labels, b.labels)


def test_tie_aware_rank_counts_ties_against_the_positive():
    assert tie_aware_rank(torch.zeros(2, 13)).tolist() == [7.0, 7.0]
    best = torch.tensor([[5.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
    assert tie_aware_rank(best).tolist() == [1.0, 2.5]


def _label_onehot(labels):
    return (labels[:, None] == np.array(["a", "b", "c"])[None, :]).astype(np.float32)


def test_recall_uses_clip_when_weights_are_zero_and_ties_when_everything_is_zero():
    labels, paintings, rng = _fixture()
    n = len(labels)
    # CLIP features encode the label; the positive is the only same-label candidate.
    img = np.concatenate([3 * _label_onehot(labels), 0.1 * rng.normal(size=(n, 13))], axis=1).astype(np.float32)
    txt = np.concatenate([3 * _label_onehot(labels), 0.1 * rng.normal(size=(n, 13))], axis=1).astype(np.float32)
    codes = np.zeros((n, 4), dtype=np.float32)
    ep = build_label_episodes(labels, paintings, np.arange(n), 100, seed=42)
    zeros = torch.zeros(100, 4)
    clip = label_episode_recall(img, txt, codes, codes, ep, zeros, beta=1.0)
    assert clip["i2t"]["recall1"] >= 0.99 and clip["t2i"]["recall1"] >= 0.99
    tied = label_episode_recall(img, txt, codes, codes, ep, zeros, beta=0.0)
    assert tied["i2t"]["recall1"] == 0.0 and tied["i2t"]["tied_episodes"] == 100


def test_condition_lift_is_positive_when_a_factor_encodes_the_label_and_near_zero_for_noise():
    labels, paintings, rng = _fixture()
    n = len(labels)
    img = rng.normal(size=(n, 16)).astype(np.float32)
    txt = rng.normal(size=(n, 16)).astype(np.float32)
    # 3 label factors + 20 irrelevant factors: uniform weighting is swamped by the irrelevant
    # ones (R@1 roughly 0.1-0.2); naive weighting concentrates on the label factor (R@1 near 1).
    informative = np.concatenate([_label_onehot(labels), rng.random((n, 20)).astype(np.float32)], axis=1)
    noise = rng.random((n, 8)).astype(np.float32)
    ep = build_label_episodes(labels, paintings, np.arange(n), 400, seed=42)
    assert condition_lift(img, txt, informative, informative, ep)["lift_mean"] > 0.3
    assert abs(condition_lift(img, txt, noise, noise, ep)["lift_mean"]) < 0.1
```

- [ ] **Step 2: Run to verify failure**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_conditioning.py src/test/test_label_episodes.py`
Expected: FAIL with `ModuleNotFoundError` for `src.model.conditioning` and `src.eval.label_episodes`.

- [ ] **Step 3: Implement `src/model/conditioning.py`**

```python
"""Nonparametric condition weights and the conditional score s(I, T | c).

Task 9 of the condition-interface plan showed the L1-normalized naive rule is
the scale-fair baseline and that the learned recovery head adds nothing; this
module is the reusable form of both for stage (d).
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def pair_codes(img_codes: Tensor, txt_codes: Tensor) -> Tensor:
    return 0.5 * (img_codes + txt_codes)


def naive_condition_weights(support_pair_codes: Tensor, contrast_pair_codes: Tensor | None = None,
                            top_k: int | None = None) -> Tensor:
    gap = support_pair_codes.mean(dim=-2)
    if contrast_pair_codes is not None:
        gap = gap - contrast_pair_codes.mean(dim=-2)
    weights = F.relu(gap)
    if top_k is not None:
        if not 1 <= top_k <= weights.shape[-1]:
            raise ValueError("top_k must be in [1, num_factors]")
        kept = torch.zeros_like(weights).scatter(-1, weights.topk(top_k, dim=-1).indices, 1.0)
        weights = weights * kept
    total = weights.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, weights / total.clamp_min(1e-12), torch.zeros_like(weights))


def conditional_score(query_feat: Tensor, cand_feat: Tensor, query_codes: Tensor, cand_codes: Tensor,
                      weights: Tensor, beta: float) -> Tensor:
    cosine = F.cosine_similarity(query_feat[:, None, :], cand_feat, dim=-1)
    factor = (weights[:, None, :] * query_codes[:, None, :] * cand_codes).sum(dim=-1)
    return beta * cosine + factor
```

- [ ] **Step 4: Implement `src/eval/label_episodes.py`**

```python
"""Episodes whose condition is a human label (emotion, art style), not a factor.

Label-defined episodes are identical for every factor model, so two models can
be compared on the very same episodes (paired), and they do not reward a model
for agreeing with its own mined structure.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.model.conditioning import conditional_score, naive_condition_weights, pair_codes


@dataclass(frozen=True)
class LabelEpisodes:
    anchor: np.ndarray
    positive: np.ndarray
    supports: np.ndarray
    contrasts: np.ndarray
    distractors: np.ndarray
    labels: np.ndarray


def _draw_distinct(rng, pool, paintings, used, count):
    """Draw `count` rows from `pool` whose paintings are not yet in `used` (updates `used`)."""
    picked = []
    for _ in range(50 * count):
        if len(picked) == count:
            break
        row = int(pool[rng.integers(len(pool))])
        if paintings[row] not in used:
            used.add(paintings[row])
            picked.append(row)
    if len(picked) < count:
        eligible = pool[~np.isin(paintings[pool], list(used))]
        _, first = np.unique(paintings[eligible], return_index=True)    # one row per painting
        eligible = eligible[np.sort(first)]
        if len(eligible) < count - len(picked):
            raise ValueError("Not enough distinct paintings to fill an episode")
        for row in rng.choice(eligible, count - len(picked), replace=False):
            used.add(paintings[row])
            picked.append(int(row))
    return picked


def build_label_episodes(labels, paintings, rows, n_episodes, seed=42, num_support=4, num_contrast=4,
                         num_distractors=12, min_paintings_per_label=30,
                         exclude_target_labels=()) -> LabelEpisodes:
    labels, paintings = np.asarray(labels), np.asarray(paintings)
    rows = np.asarray(rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    row_labels = labels[rows]
    members = {label: rows[row_labels == label] for label in np.unique(row_labels)}
    others = {label: rows[row_labels != label] for label in members}
    eligible = sorted(label for label, m in members.items()
                      if label not in set(exclude_target_labels)
                      and len(np.unique(paintings[m])) >= min_paintings_per_label)
    if len(eligible) < 2:
        raise ValueError("Need at least two eligible target labels")
    fields = {k: [] for k in ("anchor", "positive", "supports", "contrasts", "distractors", "labels")}
    for _ in range(n_episodes):
        label = eligible[rng.integers(len(eligible))]       # label-balanced targets
        used: set = set()
        same = _draw_distinct(rng, members[label], paintings, used, 2 + num_support)
        other = _draw_distinct(rng, others[label], paintings, used, num_contrast + num_distractors)
        fields["anchor"].append(same[0])
        fields["positive"].append(same[1])
        fields["supports"].append(same[2:])
        fields["contrasts"].append(other[:num_contrast])
        fields["distractors"].append(other[num_contrast:])
        fields["labels"].append(label)
    return LabelEpisodes(**{k: np.asarray(v, dtype=np.int64 if k != "labels" else None)
                            for k, v in fields.items()})


def tie_aware_rank(scores: torch.Tensor) -> torch.Tensor:
    positive, others = scores[:, :1], scores[:, 1:]
    return 1.0 + (others > positive).sum(dim=1) + 0.5 * (others == positive).sum(dim=1)


def _t(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def label_episode_recall(img_feat, txt_feat, img_codes, txt_codes, episodes: LabelEpisodes,
                         weights: torch.Tensor, beta: float) -> dict:
    candidates = np.concatenate([episodes.positive[:, None], episodes.distractors], axis=1)
    out = {}
    for direction, qf, cf, qc, cc in (("i2t", img_feat, txt_feat, img_codes, txt_codes),
                                      ("t2i", txt_feat, img_feat, txt_codes, img_codes)):
        scores = conditional_score(_t(qf[episodes.anchor]), _t(cf[candidates]),
                                   _t(qc[episodes.anchor]), _t(cc[candidates]), weights, beta)
        ranks = tie_aware_rank(scores)
        out[direction] = {
            "ranks": ranks.numpy(), "recall1": float((ranks <= 1).float().mean()),
            "recall3": float((ranks <= 3).float().mean()),
            "tied_episodes": int((scores[:, 1:] == scores[:, :1]).any(dim=1).sum()),
        }
    return out


def condition_lift(img_feat, txt_feat, img_codes, txt_codes, episodes: LabelEpisodes) -> dict:
    """Naive minus uniform R@1 at beta=0: the condition-specific benefit of the factor space."""
    support = pair_codes(_t(img_codes[episodes.supports]), _t(txt_codes[episodes.supports]))
    contrast = pair_codes(_t(img_codes[episodes.contrasts]), _t(txt_codes[episodes.contrasts]))
    naive = naive_condition_weights(support, contrast)
    uniform = torch.full_like(naive, 1.0 / naive.shape[-1])
    result = {name: label_episode_recall(img_feat, txt_feat, img_codes, txt_codes, episodes, w, 0.0)
              for name, w in (("naive", naive), ("uniform", uniform))}
    lift = {d: result["naive"][d]["recall1"] - result["uniform"][d]["recall1"] for d in ("i2t", "t2i")}
    return {**result, "lift": lift, "lift_mean": 0.5 * (lift["i2t"] + lift["t2i"])}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/test_conditioning.py src/test/test_label_episodes.py src/test/`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/model/conditioning.py src/test/test_conditioning.py src/eval/label_episodes.py src/test/test_label_episodes.py
git commit -m "feat(cosir-v2): conditioning module + human-label episode evaluator (paired, non-circular)"
```

---

### Task 6: repair run — pre-registered grid, gates, goal-based selection, seed replication

**Files:**
- Create: `src/test/20261011_factor_repair_grid/run_grid.py`,
  `src/test/20261011_factor_repair_grid/20261011_factor_repair_grid_log.md`,
  `src/test/20261011_factor_repair_grid/.gitignore` (`*.npy`, `*.json`, `*.pt`, `cache/`, `checkpoints/`)
- Create: `docs/reports/2026-10-11_cosir_v2_candidate_a_factor_repair.md`

**Interfaces:**
- Consumes: Task 1 (gates), Task 2 (data and split), Task 4 (config fields, `group_ids`,
  checkpoints), Task 5 (`build_label_episodes`, `condition_lift`). Setup is identical to Task 3
  Step 1.
- Produces, for Task 7:
  - `src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt`
  - `src/test/20261011_factor_repair_grid/checkpoints/R0_seed42.pt`
  - The selected recipe's exact `FactorTrainingConfig`, recorded in the report and the log.

- [ ] **Step 1: Run exactly this grid** (seed 42, 32 factors, 2,000 epochs,
  `lambda_usage_balance=0.1`, `group_ids = leakage_groups(...)[train]`, other fields at defaults).
  If Task 3's checkpoint amended this grid, use the amended one.

| ID | Config beyond the base |
|---|---|
| R0 | none: the collapsed recipe on the new split, the baseline |
| R1 | `agreement="infonce"` |
| R2 | `lambda_decorrelation=1.0` |
| R3 | `agreement="infonce", lambda_decorrelation=1.0` |
| R4 | `agreement="infonce", activation="topk", topk=8, lambda_sparsity=0.0` |
| R5 | R4 + `lambda_decorrelation=1.0` |
| R6 | R3 + `center_inputs=True` |
| R7 | R3 + `lambda_sparsity=0.1` |

- [ ] **Step 2: Gates on `val` for every run.** Report a table: one row per run, one column per gate
  value, with pass/fail marks.

- [ ] **Step 3: Condition lift on `val` for every run, including gate failures.**
  - Build the validation label episodes **once**, from `val` rows (seed 42), and share them across
    all runs:
    - 2,048 emotion episodes, with `exclude_target_labels=("something else",)`. A catch-all
      label is not a condition, but it may still appear as a negative.
    - 2,048 art-style episodes.
  - For each run, compute `condition_lift` on both sets.
  - The selection score is the mean of the two `lift_mean` values: the condition-specific R@1
    benefit, averaged over the two condition types and both retrieval directions.

- [ ] **Step 4: Pre-registered selection.**
  1. **Necessary:** the run passes all gates on `val`. The gates are the hard floor for a healthy
     space.
  2. **Choose:** among passing runs, pick the highest selection score. It measures the aim
     directly: does telling the model the condition improve cross-item matching on
     human-defined conditions it wasn't selected for?
  3. **Tie-break:** scores within 1.0 R@1 point are treated as equal. Prefer the lower mean
     `readout` error, since it keeps more CLIP information for conditions we haven't measured.
     If still tied, take the earlier run in the table.
  4. **If no run passes:** report which gates bind for each run, plus each run's selection score.
     Stop, and the user decides. Do not add runs.
  5. **Sanity check on the gates themselves:** if a gate-failing run's score beats every passing
     run by more than 2 R@1 points, flag it in the verdict. That is evidence that the failed
     gate (for example sparsity) may not be necessary for the aim, and it goes to the user. Do
     not change the selection.

- [ ] **Step 5: Seed replication.**
  - Retrain the selected recipe with seeds 43 and 44. Both must pass all gates on `val`;
    otherwise the recipe is **not** selected. Report it and stop, as in Step 4.
  - Report each seed's selection score. If a replication seed scores at or below R0, record it as
    a caveat on how stable the benefit is.
  - Report cross-seed factor alignment: Hungarian matching (`scipy.optimize.linear_sum_assignment`
    on −|r| between seed-42 and seed-43/44 `val` pair codes), with mean, median and minimum
    matched |r|. Low alignment with passing gates means a stable subspace but unstable axes; that
    is a caveat, not a failure.

- [ ] **Step 6: Final held check, once.** Run the gates on `held` rows for the selected seed-42
  model and for R0. Save both checkpoints with `save_factor_checkpoint`, then verify that
  `load_factor_checkpoint` re-encodes 1,000 `held` rows bit-identically.

- [ ] **Step 7: Write the report.** Verdict first: is the space repaired, which mechanism did it,
  what condition-specific benefit does it buy over R0, and at what cost to reconstruction?
  Include:
  - the grid table (gates plus selection score);
  - the replication results;
  - the held check;
  - a before/after row against Task 1's collapsed-code values.

- [ ] **Step 8: Commit**

```bash
git add src/test/20261011_factor_repair_grid/ docs/reports/2026-10-11_cosir_v2_candidate_a_factor_repair.md
git commit -m "docs(cosir-v2): factor-discovery repair grid with pre-registered gates and goal-based selection"
```

---

### Task 7: condition interface on the repaired factors (held-out evaluation)

**Files:**
- Create: `src/test/20261012_condition_eval_repaired_factors/run_eval.py`,
  `src/test/20261012_condition_eval_repaired_factors/20261012_condition_eval_repaired_factors_log.md`,
  `src/test/20261012_condition_eval_repaired_factors/.gitignore`
- Create: `docs/reports/2026-10-12_cosir_v2_candidate_a_condition_eval_repaired_factors.md`

**Interfaces:**
- Consumes:
  - Task 6's two checkpoints, via `load_factor_checkpoint`.
  - Task 2's data and split.
  - Task 5's conditioning module and label-episode evaluator.
  - `mine_episodes`, `EpisodeMiningConfig` (`src/train/episodes.py`, unchanged).
  - `choose_swap_pairs` and `score_pool` from
    `src/test/20261005_condition_ranking_evaluation/run_ranking_eval.py` (imported, as Tasks 8–9
    did; `score_pool` is only for the parity check).
- Produces: the report on which the user makes the stage (d) decision.

- [ ] **Step 1: Human-label episodes (primary).**
  - Build the held label episodes **once** from `held` rows (seed 42), and use the **identical**
    episodes for R0 and the selected model:
    - 1,024 emotion episodes, with `exclude_target_labels=("something else",)`;
    - 1,024 art-style episodes.
  - Variants: naive (L1), uniform, CLIP-only. Choose β per variant on the Task 6 validation label
    episodes (both label types pooled) from `{0, .001, .003, .01, .03, .1, .3, 1, 3}`, by mean
    bidirectional R@1. Report β=0 as well.
  - Report R@1 / R@3 per label type, per direction and pooled, with a chance line of 1/13 and 3/13.
  - Paired bootstrap (5,000 resamples, seed 42, over episodes) for:
    - (a) selected naive − R0 naive, on the same episodes;
    - (b) selected naive − selected uniform;
    - (c) selected naive − CLIP-only.

- [ ] **Step 2: Factor-mined episodes (secondary; protocol unchanged from Task 7 of the
  condition-interface plan, for comparability).** For each model separately:
  - Mine 4,096 episodes on `val` rows and 1,024 on `held` rows. Use `mine_episodes` with
    `EpisodeMiningConfig(seed=42)` under single-threaded BLAS, then remap split-local indices to
    global rows as that task did.
  - Variants:
    - naive (L1);
    - naive top-k, with `k ∈ {1, 3, 5}` chosen on `val`;
    - uniform;
    - oracle (one-hot on the target factor);
    - CLIP-only.
  - β for each variant is chosen on `val` from the same grid.
  - Score with `conditional_score`. Assert parity with `score_pool` on the first 16 episodes
    (max abs difference < 1e-5).
  - Report on `held`:
    - R@1 / R@3 with `tie_aware_rank`, and the tie count;
    - a paired bootstrap for naive − uniform;
    - swap reversal on `choose_swap_pairs` pairs;
    - the per-role breakdown from Task 9 Q2 of the condition-interface plan.

  These episodes differ between models (each mines its own), so R0-vs-selected comparisons here are
  unpaired and descriptive only.

- [ ] **Step 3: Write the report.** Verdict first, then a comparison table with three columns:
  - Task 9 (collapsed factors, row split);
  - R0 (collapsed factors, painting split): the split effect;
  - selected (repaired factors, painting split): the factor effect.

  Then state which pre-registered readiness criteria hold. **The user makes the stage (d)
  decision.**
  1. **Primary:** on the identical held label episodes, pooled over emotion and art style, the
     selected model's naive R@1 beats R0's naive R@1 with a paired 95% CI above 0, in both
     directions. On the same episodes, the selected naive also beats the selected uniform with a
     CI above 0 in both directions. Per-label-type results are reported alongside.
  2. **Secondary:** naive swap reversal on factor-mined held episodes is higher than R0's, in both
     directions.
  3. **Floor:** naive − uniform on factor-mined held episodes has a CI above 0 in both directions.
     The collapsed factors already met this in Task 9, so it cannot establish repair on its own.

  **Stated caveat:** selection (Task 6) used validation episodes of the same two condition types.
  The held episodes use unseen paintings, so criterion 1 is a fair test for *emotion and style*
  conditions, but not evidence about arbitrary human-stated conditions. That is stage (e)'s
  human-judged set.

- [ ] **Step 4: Commit**

```bash
git add src/test/20261012_condition_eval_repaired_factors/ docs/reports/2026-10-12_cosir_v2_candidate_a_condition_eval_repaired_factors.md
git commit -m "docs(cosir-v2): condition-interface re-evaluation on repaired factors (label-defined, paired)"
```

---

## Deferred to the stage (d) plan (explicitly out of scope here)

- **Episode-role redesign.** Condition-only distractors are as correct as the positive under the
  condition (code review #5). Task 7 keeps the condition-interface plan's Task 7 protocol for the
  secondary factor-mined evaluation, only for comparability.
- **Whether and how Block 1's Stage 1 output enters the scorer** (code review #3).
- **Retiring `ConditionEncoder`** (`src/model/condition.py`), and removing `src/model/whitening.py`
  or marking it as a documented negative result.
- **Ranking-loss training of a learned interface.** Task 9 Q4c: 41/44 swaps from scratch.

## Self-review

- **Spec coverage:**
  - Spec §1 "shared sparse factor space": the sparsity, redundancy and dimensionality gates plus
    the fix mechanisms (Tasks 1, 4, 6).
  - "Group-sparse/CCA control against split dictionary": kept via the dead / modality-private
    gates (Task 1).
  - §2 condition interface: the naive rule in `src` (Task 5).
  - §3 scoring function: `conditional_score` (Task 5).
  - The project aim (conditional cross-item matching): the selection metric (Task 6) and the
    primary readiness criterion (Task 7) both use human-label-defined episodes.
  - Code-review critical #1: Tasks 1, 3, 4 and 6. Critical #2: Task 2 and the split discipline.
    High #4 (working pieces in test scripts): Task 5. Medium #6 (no checkpoints): Tasks 4 and 6.
    High #3, medium #5 and the low items: deferred, listed above.
- **Placeholder scan:** the only elisions are Task 4 Step 2's fixture lines. They are explicitly
  defined as repeating Step 1's ring-graph fixture in full.
- **Type consistency:**
  - `evaluate_factor_gates` keyword names are identical in Tasks 1, 3 and 6.
  - `GroupedSplit(train, val, held)` is used consistently.
  - `FactorTrainingConfig` field names in Task 4 match Task 6's grid (`agreement`,
    `infonce_temperature`, `lambda_decorrelation`, `activation`, `topk`, `center_inputs`).
  - `load_factor_checkpoint` returns `(model, config)` in Tasks 4, 6 and 7.
  - `LabelEpisodes`, `build_label_episodes` and `condition_lift` (Task 5) are used with the same
    signatures in Tasks 6 and 7.
  - `naive_condition_weights` and `conditional_score` shapes match between the interface block and
    the tests.
- **Review Focus:** all five lines have an owning test:
  - #1: `test_infonce_masks_same_group_false_negatives`
  - #2: `test_rows_sharing_an_image_vector_join_one_group_across_paintings`
  - #3: `test_redundancy_summary_flags_copies_and_reports_constant_columns` and
    `test_decorrelation_separates_copies_from_independent_and_ignores_constants`
  - #4: `test_checkpoint_round_trip_gives_identical_codes`
  - #5: `test_ties_count_against_the_positive_in_retrieval`,
    `test_all_zero_codes_are_degenerate_not_errors`, `test_tie_aware_rank_counts_ties_against_the_positive`
    and `test_recall_uses_clip_when_weights_are_zero_and_ties_when_everything_is_zero` (Task 5)
