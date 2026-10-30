"""Clustering for r12 (spec 2026-10-30 §3.4 "Clustering", §3.5 0a(iii) r10 part): r10's partitions, mean AMI and paired
bootstrap, on any arrays and labels. Every statistical piece is a wrapper of r10's code, not a copy; r10 and r11 are
imported read-only.

  partitions(x)              rs_vh.r20_partition       MiniBatchKMeans k = 20, states 0, 1, 2, n_init 3, batch 4,096, one
                                                       thread, float32, unit norm checked (not repaired); cast to int32
  percept_partitions(prob)   rs_vh._partition_task     the two readings of PercepT core: r20 on the L2-normalised
                                                       probabilities (rs_vh.l2_probabilities) and own (argmax)
  ami(parts, labels)         rs_vh.point_table,        the mean over states (and over seeds for a seeded side) of
                             rs_vh._mean_over          scikit-learn's AMI; labels < 0 are masked inside the AMI
  paired(cand, comp, labels) rg_held.bootstrap_d_t,    d = A_cand - A_comp with the 95% percentile interval. Seeded
                             rs_stage4.d_g_block       candidate against an unseeded comparator: rg_held.draws_d_t
                                                       (rows I_b = integers, then S_b = choice of 5 seeds, the
                                                       comparator draws nothing). Both seeded: rg_held.draws_d_r
                                                       (rows, S_b, then the comparator's own S_R,b); with
                                                       shared_seeds=True rs_stage4.draws_d_g (rows, then one S_b shared
                                                       by both). A fresh default_rng(0) per
                                                       call, hence per statistic. Partitions are fitted once.
                                                       rs_stage4.d_t_block is tied to ArtELingo's aspects (its outcome
                                                       uses RANGES[aspect]); its core rg_held.bootstrap_d_t is used.
  cached_partitions(x, name) partitions + a hash-checked cache under OUTPUTS/partitions/<name>.npz (key `labels`)
  r10_emotion_replay()       spec §3.5 0a(iii): r10's held emotion block (c* against the teacher) through this module

Seeded sides are int32[5, 3, n] (seed, state, row), unseeded sides int32[3, n].
"""
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import ac_paths
import rs_stage4
import rs_vh
import rg_held

K, STATES, DRAWS = rs_vh.K, tuple(rs_vh.STATES), 2000
SEEDS = tuple(ac_paths.SEEDS)
assert K == 20 and STATES == (0, 1, 2) and SEEDS == tuple(rg_held.SEEDS), 'r10 constants changed'


def partitions(x, k=K, states=STATES):
    """r10's MiniBatchKMeans partitions of unit rows `x`: int32 [3, n] (`rs_vh.r20_partition`, which fixes the settings;
    other k or states are refused rather than approximated)."""
    assert k == K and tuple(states) == STATES, f'r10 fixes k = {K} and states {STATES}, got {k}, {tuple(states)}'
    return rs_vh.r20_partition(x).astype(np.int32)


def percept_partitions(prob):
    """PercepT core's two readings of its probabilities [n, 40], as r10 (`rs_vh._partition_task`, last line):
    {'r20': int32 [3, n] on the L2-normalised probabilities, 'own': int32 [1, n], the argmax}."""
    return {'r20': partitions(rs_vh.l2_probabilities(prob)), 'own': rs_vh.own_partition(prob).astype(np.int32)}


def _seeded(parts):
    parts = np.asarray(parts)
    assert np.issubdtype(parts.dtype, np.integer), parts.dtype
    assert parts.ndim in (2, 3), parts.shape
    return parts.ndim == 3


def _fixed(parts, n, own=False):
    """r10's `fixed` form of one side: {seed: {'r20': int [3, n]}} (an unseeded side is {0: ...}). `own=True` also
    accepts PercepT core's own reading, one partition per set ([1, n], seeded [5, 1, n]; r10's `{'own': [1, n]}`
    form, `rs_vh.own_partition`); the key stays 'r20' because the AMI does not care which reading it scores."""
    parts = np.asarray(parts)
    width = (len(STATES), 1) if own else (len(STATES),)
    seeded = _seeded(parts)
    assert parts.shape[-2] in width, f'{parts.shape[-2]} partitions per set, expected {" or ".join(map(str, width))}'
    if seeded:
        assert parts.shape == (len(SEEDS), parts.shape[1], n), parts.shape
        return {seed: {'r20': parts[i]} for i, seed in enumerate(SEEDS)}
    assert parts.shape == (parts.shape[0], n), parts.shape
    return {0: {'r20': parts}}


def ami(parts, labels):
    """Mean AMI against `labels`: over the states, and for a seeded side over the seeds of the state means. `parts` is
    1 or 3 partitions per set, so PercepT core's own reading ([1, n] or [5, 1, n]) is scored too."""
    labels = np.asarray(labels)
    fixed = _fixed(parts, len(labels), own=True)
    table = rs_vh.point_table(labels, {'x': fixed})
    return rs_vh._mean_over(rs_vh._point_val(table), 'x', sorted(fixed))


def paired(cand_parts, comp_parts, labels, draws=DRAWS, workers=8, shared_seeds=False):
    """d = A_cand - A_comp over `labels` with r10's paired bootstrap (partitions fixed, rows then seeds).
    The candidate is seeded (int32 [5, 3, n]); the comparator is unseeded ([3, n]) or seeded. A seeded comparator
    draws its own seeds (rows, candidate seeds, comparator seeds; `rg_held.bootstrap_d_r`, as r10's d_R for an
    unrelated arm); `shared_seeds=True` gives both the one seed draw (`rs_stage4.d_g_block`, r10's d_G, for two runs of
    one recipe). Only 3-partition sets are scored here: the own reading (one partition) goes through `ami`.
    Returns {'A_cand', 'A_comp', 'd', 'interval': [lo, hi], 'bootstrap_d'}."""
    labels = np.asarray(labels)
    assert labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer) and (labels >= 0).any()
    assert _seeded(cand_parts), 'the candidate must be seeded: int32 [5, 3, n]'
    cand, comp = _fixed(cand_parts, len(labels)), _fixed(comp_parts, len(labels))
    assert not shared_seeds or _seeded(comp_parts), 'shared_seeds needs a seeded comparator'
    if _seeded(comp_parts) and shared_seeds:
        res = rs_stage4.d_g_block(cand, comp, labels, draws, workers)
        a_cand, a_comp, interval = res['A_cand'], res['A_G'], [res['lo'], res['hi']]
    elif _seeded(comp_parts):
        res = rg_held.bootstrap_d_r(cand, comp, labels, draws, workers)
        a_cand, a_comp, interval = res['A_G'], res['A_R'], list(res['interval'])
    else:
        res = rg_held.bootstrap_d_t(cand, comp, labels, draws, workers)
        a_cand, a_comp, interval = res['A_G'], res['A_T'], list(res['interval'])
    return dict(A_cand=a_cand, A_comp=a_comp, d=res['d'], interval=interval, bootstrap_d=res['bootstrap_d'])


def _replay_fit(task):
    """One fit of the replay (in a worker): the held emotion rows of c* at a seed, or of the teacher (seed None).
    c* is read in its stored float16 and the teacher as unit float32, as `rt_stage0._partition` does."""
    arm, seed, limit = task
    import rt_data
    with threadpool_limits(1):
        if arm == 'cstar':
            x = rt_data.raw('cstar', 'held', 'emotion', seed)
        else:
            x = rt_data.embed('instructed', 'held', 'emotion')
        return task, partitions(x[:limit])


def r10_emotion_replay(workers=8, draws=DRAWS, limit=None):
    """Spec §3.5 0a(iii), r10 part: r10's stage 4 emotion block on its held inputs, through this module. c*'s full-scale
    held emotion outputs (seeds 0 to 4; `rt_data.raw('cstar', 'held', ...)`, the files `rs_stage4` scores) and the
    teacher's held store (`rt_data.embed('instructed', ...)`) are clustered by `partitions` and compared by `paired`;
    labels are `rt_data.labels('held')['emotion']` (< 0 masked inside the AMI). The teacher's partitions are fitted
    afresh here (r10 scored 12-24's frozen file of the same fit); the spec expects A_cand 0.10658729378204021, A_T
    0.09207439052277983, d 0.014512903259260385, interval [0.011288090988306501, 0.017671180133450672] within 1e-9.
    `limit` keeps the first rows only (a plumbing smoke). Returns {'A_cand', 'A_T', 'd', 'interval'}."""
    import rt_data
    labels = rt_data.labels('held')['emotion'][:limit]
    tasks = [('cstar', seed, limit) for seed in SEEDS] + [('instructed', None, limit)]
    with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
        fitted = dict(pool.map(_replay_fit, tasks, chunksize=1))
    cand = np.stack([fitted[('cstar', seed, limit)] for seed in SEEDS])
    res = paired(cand, fitted[('instructed', None, limit)], labels, draws, workers)
    return dict(A_cand=res['A_cand'], A_T=res['A_comp'], d=res['d'], interval=res['interval'])


def _tmp_path(path):
    """A temporary name for `path` that is unique per writer (the process id), so two writers never share one."""
    return path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')


def _cache_root(root):
    return Path(root) if root is not None else ac_paths.OUTPUTS / 'partitions'


def cached_partitions(x, name, root=None):
    """`partitions(x)` cached as `<root>/<name>.npz` (key `labels`, int32 [3, n]; root defaults to OUTPUTS/partitions,
    name as the index's `<dataset>__<split>__<arm>__s<seed>`) with a sidecar `<name>.json` holding the sha256 of the
    float32 rows (`rs_vh.array_sha`), the settings and the sha256 of the partition file. A hit needs the sidecar's row
    hash to equal the hash of `x`; the name alone never decides. A name whose rows changed is recomputed and rewritten;
    a file that no longer matches its own sidecar hash, or has none, is refused."""
    x32 = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
    row_sha = rs_vh.array_sha(x32)
    path = _cache_root(root) / f'{name}.npz'
    side_path = path.with_suffix('.json')
    settings = dict(k=K, states=list(STATES), n_init=rs_vh.N_INIT, batch_size=rs_vh.BATCH)
    if path.exists() or side_path.exists():
        assert path.is_file() and side_path.is_file(), f'{path}: partition file and sidecar are not both there'
        side = json.loads(side_path.read_text())
        assert side['partitions_sha256'] == rs_vh.file_sha(path), f'{path}: partition file changed'
        if side['array_sha256'] == row_sha and side['settings'] == settings and side.get('shape') == list(x32.shape):
            with np.load(path, allow_pickle=False) as z:
                assert z.files == ['labels']
                labels = z['labels']
            assert labels.dtype == np.int32 and labels.shape == (len(STATES), len(x32)), (path, labels.shape)
            return labels
    labels = partitions(x32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    np.savez(tmp, labels=labels)
    os.replace(tmp, path)
    side_path.write_text(json.dumps(dict(array_sha256=row_sha, shape=list(x32.shape), settings=settings,
                                         partitions_sha256=rs_vh.file_sha(path)), indent=2) + '\n')
    return labels
