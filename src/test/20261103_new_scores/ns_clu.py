"""r17 ticket 06: clustering scorer core: metric table, paired cell (joint bootstrap), seed ranges; gate (i) replays.

Spec docs/superpowers/specs/2026-11-03-new-scores-design.md §3.4 (Intervals, Outcomes), §3.5 stage 0 (i) a to d, §3.7,
D3, D12, D16, D20.

Frozen interface (index, "Clustering scorer (06)"; a later need adds a function, never changes a signature):
  metric_table(arm, aspect, parts, labels, k, metrics=CLASS_METRICS) -> list of {arm, aspect, k, metric, per_seed, mean}
  paired_cell(cand_parts, comp_parts, labels, k, r=None, metrics=CLASS_METRICS, draws=2000, rng_seed=0)
      -> {metric: {cand, comp, d, lo, hi, r, outcome, n_draws}}
  ranges(parts_by_seed, labels, k, metrics=CLASS_METRICS) -> {metric: {r, per_seed}}

The bootstrap generalises r14's `rb_score.draw_bootstrap` / `paired_bootstrap` (code map A) to four metrics at any K:
  - per cell one fresh `default_rng(rng_seed)`; per draw `rows = integers(0, n, size=n)` over ALL n rows (unlabelled
    rows included), then c*'s five seeds `choice([0..4], size=5, replace=True)`, then, if the comparator is seeded,
    its five seeds the same way; an unseeded comparator draws nothing (pseudo-seed 0);
  - rows are "indexed with repeats, then masked": a draw's multiplicity of every row is its weight in the contingency
    table, and rows without a label (-1) are dropped from the table afterwards (`ns_metrics.contingency`);
  - every metric of the cell reads the same draws, through `ns_metrics.from_contingency`;
  - arm value = mean over the drawn seeds (repeats count) of the mean over the 3 states; d = cand - comparator; interval
    = `np.percentile(d, [2.5, 97.5])` of the DIFFERENCE; point = rows 0..n-1 and every seed once; outcome by
    `rt_stats.outcome(lo, r)`.
  - partitions are used as given, never normalised.
The two-reading comparator (r14's `b_matched`: the larger of the matched arm's r20 and own readings, taken again in
every draw, both readings on the SAME seed draw) is the private `_boot` with two readings; `paired_cell` is its
one-reading case, which draws the comparator's five seeds in the same slot, so its draws equal `b_matched`'s.

Gate (i) replays: `replay_i_a`, `replay_i_b`, `replay_i_c`, `replay_i_d_emoset`, `replay_i_d_class_count`; each returns
{'pass', 'max_abs_diff', 'items': [...]} with the stored value cited by path and key. A `draws` argument below 2,000
compares the point values only (the intervals need the stored 2,000 draws); `pass` is True only for the full replay.
"""
import json
import multiprocessing
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import ns_paths
import ns_metrics

CLASS_METRICS = tuple(ns_paths.CLASS_METRICS)
SEEDS = tuple(ns_paths.SEEDS)
N_STATES = len(ns_paths.KMEANS_STATES)
WORKERS = 8                      # bootstrap processes (one thread each, as r14); set to 1 in small tests
_CTX = {}                        # the fork-shared inputs of one bootstrap (set before the pool starts)


# ------------------------------------------------------------------------------------------------ input checks
def _labels(labels):
    labels = np.asarray(labels)
    assert labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer), (labels.shape, labels.dtype)
    assert (labels >= 0).any(), 'no labelled row'
    return labels.astype(np.int64)


def _parts(parts, n, name, k, seeded=None):
    """int [S, P, n] from int [P, n] (unseeded, S = 1) or [S, P, n]; ids in [0, k). `seeded` forces the expected kind."""
    p = np.asarray(parts)
    assert np.issubdtype(p.dtype, np.integer), f'{name}: partitions must be integers, got {p.dtype}'
    assert p.ndim in (2, 3), f'{name}: need [P, n] or [S, P, n], got {p.shape}'
    if seeded is not None:
        assert (p.ndim == 3) == seeded, f'{name}: expected a {"seeded [S, P, n]" if seeded else "unseeded [P, n]"} array, got {p.shape}'
    if p.ndim == 2:
        p = p[None]
    assert p.shape[2] == n, f'{name}: {p.shape[2]} rows, labels have {n}'
    assert p.shape[1] >= 1 and p.shape[0] >= 1, f'{name}: empty {p.shape}'
    assert p.min() >= 0 and p.max() < k, f'{name}: cluster ids {p.min()}..{p.max()} outside [0, {k})'
    return p.astype(np.int64)


def _metrics(metrics):
    metrics = tuple(metrics)
    assert metrics and len(set(metrics)) == len(metrics) and set(metrics) <= set(CLASS_METRICS), metrics
    return metrics


# ------------------------------------------------------------------------------------------------ metric table
def metric_table(arm, aspect, parts, labels, k, metrics=CLASS_METRICS):
    """One row per metric: {arm, aspect, k, metric, per_seed (floats, one per seed; one for an unseeded arm), mean}.
    parts int [3, n] (unseeded) or [S, 3, n]; labels int [n], -1 = no label (masked after clustering). Value of a seed =
    mean over its states; mean = mean over seeds (r14's `ac_cluster.ami` form)."""
    metrics = _metrics(metrics)
    y = _labels(labels)
    p = _parts(parts, len(y), 'parts', k)
    rows = []
    for m in metrics:
        per = [float(ns_metrics.score_partitions(m, y, seed_parts)) for seed_parts in p]
        rows.append(dict(arm=arm, aspect=aspect, k=int(k), metric=m, per_seed=per, mean=float(np.mean(per))))
    return rows


def ranges(parts_by_seed, labels, k, metrics=CLASS_METRICS):
    """Seed range per metric: {metric: {'r': max - min over the 5 seeds, unrounded, 'per_seed': {'0': v0, ... '4': v4}}}.
    parts_by_seed: {seed: int [3, n]} for seeds 0 to 4, or int [5, 3, n]. Validation labels in the caller's hands."""
    metrics = _metrics(metrics)
    y = _labels(labels)
    if isinstance(parts_by_seed, dict):
        assert sorted(parts_by_seed) == list(SEEDS), f'seeds {sorted(parts_by_seed)}, expected {list(SEEDS)}'
        arr = np.stack([np.asarray(parts_by_seed[s]) for s in SEEDS])
    else:
        arr = np.asarray(parts_by_seed)
        assert arr.ndim == 3 and arr.shape[0] == len(SEEDS), f'need [5, 3, n], got {arr.shape}'
    p = _parts(arr, len(y), 'parts_by_seed', k, seeded=True)
    out = {}
    for m in metrics:
        vals = [ns_metrics.score_partitions(m, y, seed_parts) for seed_parts in p]
        assert np.isfinite(vals).all()
        out[m] = dict(r=float(max(vals) - min(vals)), per_seed={str(s): float(v) for s, v in zip(SEEDS, vals)})
    return out


# ------------------------------------------------------------------------------------------------ draws
def draws_iter(n_rows, n_cand_seeds, comp_seeded, n_draws, rng_seed=0):
    """The whole RNG transcript of one cell, r14's `rb_score.draw_bootstrap` order: a fresh `default_rng(rng_seed)`; per
    draw `rows = integers(0, n, size=n)`, then c*'s seeds `choice([0..S-1], size=S, replace=True)`, then (seeded
    comparator only) the comparator's seeds the same way. Yields (rows, cand_seeds, comp_seeds or None)."""
    assert n_rows > 0 and n_draws > 0 and n_cand_seeds > 0
    rng = np.random.default_rng(rng_seed)
    eligible = list(range(n_cand_seeds))
    for _ in range(n_draws):
        rows = rng.integers(0, n_rows, size=n_rows)
        cand = rng.choice(eligible, size=len(eligible), replace=True)
        comp = rng.choice(eligible, size=len(eligible), replace=True) if comp_seeded else None
        yield rows, cand, comp


# ------------------------------------------------------------------------------------------------ the bootstrap
def _arm_values(ctx_parts, y, weights, chosen, ky, k, metrics):
    """float [M]: mean over the chosen seeds (repeats count) of the mean over states of each metric, on the weighted
    contingency table of the labelled rows (weights = the draw's row multiplicities)."""
    per_seed = {}
    for s in dict.fromkeys(int(c) for c in chosen):
        tables = [ns_metrics.contingency(y, p, weights, (ky, k)) for p in ctx_parts[s]]
        per_seed[s] = np.mean([[ns_metrics.from_contingency(t, m) for m in metrics] for t in tables], axis=0)
    return np.mean([per_seed[int(c)] for c in chosen], axis=0)


def _draw_d(weights, cand_seeds, comp_seeds):
    c = _CTX
    cand = _arm_values(c['cand'], c['y'], weights, cand_seeds, c['ky'], c['k'], c['metrics'])
    comp = np.max([_arm_values(parts, c['y'], weights, comp_seeds, c['ky'], c['k'], c['metrics'])
                   for parts in c['comps'].values()], axis=0)
    return cand - comp


def _chunk(span):
    lo, hi = span
    c = _CTX
    with threadpool_limits(limits=1):
        return [_draw_d(c['weights'][i][c['mask']], c['cand_seeds'][i], c['comp_seeds'][i]) for i in range(lo, hi)]


def _boot(cand, comps, labels, k, metrics, comp_seeded, n_draws, rng_seed, workers):
    """Joint bootstrap of one cell. cand int [S, 3, n]; comps {reading: int [S', P, n]} (S' = S if comp_seeded, else 1;
    several readings share one seed draw and the larger is taken in every draw, r14's `b_matched`). Returns, per metric
    (arrays [M] in the order of `metrics`): point cand, point comp (largest reading), its reading, d, and the draws' d."""
    n = len(labels)
    y_all = labels
    mask = y_all >= 0
    ky = int(y_all.max()) + 1
    cand = _parts(cand, n, 'cand', k, seeded=True)
    assert cand.shape[0] == len(SEEDS) and cand.shape[1] == N_STATES, f'cand {cand.shape}'
    comps = {name: _parts(p, n, f'comp[{name}]', k) for name, p in comps.items()}
    for name, p in comps.items():
        assert p.shape[0] == (cand.shape[0] if comp_seeded else 1), f'comp[{name}] {p.shape} (seeded {comp_seeded})'
    ctx = dict(y=y_all[mask], k=k, ky=ky, metrics=metrics, mask=mask,
               cand=[[sp[mask] for sp in seed] for seed in cand],
               comps={name: [[sp[mask] for sp in seed] for seed in p] for name, p in comps.items()})
    # draws first (schedule-independent), as r14: the whole transcript before any metric is computed
    weights = np.zeros((n_draws, n), dtype=np.uint16)
    cand_seeds, comp_seeds = [], []
    for i, (rows, cs, ms) in enumerate(draws_iter(n, cand.shape[0], comp_seeded, n_draws, rng_seed)):
        counts = np.bincount(rows, minlength=n)
        assert counts.max() < 2 ** 16 and counts.sum() == n
        weights[i] = counts
        cand_seeds.append(cs)
        comp_seeds.append(ms if comp_seeded else np.array([0]))
    ctx.update(weights=weights, cand_seeds=cand_seeds, comp_seeds=comp_seeds)
    # the point estimate: rows 0..n-1, every seed once
    ones = np.ones(int(mask.sum()), dtype=np.int64)
    all_c = np.arange(cand.shape[0])
    all_m = np.arange(cand.shape[0]) if comp_seeded else np.array([0])
    cand_pt = _arm_values(ctx['cand'], ctx['y'], ones, all_c, ky, k, metrics)
    comp_pt = np.array([_arm_values(parts, ctx['y'], ones, all_m, ky, k, metrics) for parts in ctx['comps'].values()])
    best = comp_pt.argmax(axis=0)
    names = list(comps)
    _CTX.clear()
    _CTX.update(ctx)
    try:
        spans = [(lo, min(lo + 25, n_draws)) for lo in range(0, n_draws, 25)]
        if workers == 1:
            parts = [_chunk(s) for s in spans]
        else:
            with multiprocessing.get_context('fork').Pool(workers) as pool:
                parts = pool.map(_chunk, spans)
    finally:
        _CTX.clear()
    d = np.array([row for chunk in parts for row in chunk])
    assert d.shape == (n_draws, len(metrics)) and np.isfinite(d).all()
    return dict(cand=cand_pt, comp=comp_pt.max(axis=0), reading=[names[b] for b in best], d_draws=d)


def _cell(res, metrics, r, n_draws):
    from rt_stats import outcome
    out = {}
    for j, m in enumerate(metrics):
        lo, hi = np.percentile(res['d_draws'][:, j], [2.5, 97.5])
        rm = None if r is None else r[m]
        out[m] = dict(cand=float(res['cand'][j]), comp=float(res['comp'][j]), d=float(res['cand'][j] - res['comp'][j]),
                      lo=float(lo), hi=float(hi), r=None if rm is None else float(rm),
                      outcome=None if rm is None else outcome(float(lo), float(rm)), n_draws=int(n_draws))
    return out


def paired_cell(cand_parts, comp_parts, labels, k, r=None, metrics=CLASS_METRICS, draws=2000, rng_seed=0):
    """Candidate against comparator on one cell, all `metrics` on the same draws. cand_parts int [5, 3, n]; comp_parts
    int [3, n] (unseeded: the draws are rows and c*'s seeds only) or [5, 3, n] (seeded: its seeds drawn apart, after
    c*'s); labels int [n], -1 = no label; k = the cluster count of the partitions. `r`: {metric: seed range} or None
    (outcome None). Returns {metric: {cand, comp, d, lo, hi, r, outcome, n_draws}} with d = cand - comp, [lo, hi] the
    2.5 and 97.5 percentiles of d over `draws` joint draws, outcome by `rt_stats.outcome(lo, r)`."""
    ns_paths.add_rounds()
    metrics = _metrics(metrics)
    y = _labels(labels)
    comp = np.asarray(comp_parts)
    assert comp.ndim in (2, 3), f'comp_parts {comp.shape}'
    assert r is None or set(metrics) <= set(r), f'r lacks {set(metrics) - set(r)}'
    res = _boot(np.asarray(cand_parts), {'comp': comp}, y, k, metrics, comp.ndim == 3, draws, rng_seed, WORKERS)
    return _cell(res, metrics, r, draws)
