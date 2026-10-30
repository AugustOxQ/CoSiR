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
                                                       comparator draws nothing). Both seeded: rs_stage4.draws_d_g
                                                       (rows, then one S_b shared by both). A fresh default_rng(0) per
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


def _fixed(parts, n):
    """r10's `fixed` form of one side: {seed: {'r20': int [3, n]}} (an unseeded side is {0: ...})."""
    parts = np.asarray(parts)
    if _seeded(parts):
        assert parts.shape == (len(SEEDS), len(STATES), n), parts.shape
        return {seed: {'r20': parts[i]} for i, seed in enumerate(SEEDS)}
    assert parts.shape == (len(STATES), n), parts.shape
    return {0: {'r20': parts}}


def ami(parts, labels):
    """Mean AMI against `labels`: over the states, and for a seeded side over the seeds of the state means."""
    labels = np.asarray(labels)
    fixed = _fixed(parts, len(labels))
    table = rs_vh.point_table(labels, {'x': fixed})
    return rs_vh._mean_over(rs_vh._point_val(table), 'x', sorted(fixed))


def paired(cand_parts, comp_parts, labels, draws=DRAWS, workers=8):
    """d = A_cand - A_comp over `labels` with r10's paired bootstrap (partitions fixed, rows then seeds).
    The candidate is seeded (int32 [5, 3, n]); the comparator is unseeded ([3, n]) or seeded (draws shared with the
    candidate, as r10's d_G). Returns {'A_cand', 'A_comp', 'd', 'interval': [lo, hi], 'bootstrap_d'}."""
    labels = np.asarray(labels)
    assert labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer) and (labels >= 0).any()
    assert _seeded(cand_parts), 'the candidate must be seeded: int32 [5, 3, n]'
    cand, comp = _fixed(cand_parts, len(labels)), _fixed(comp_parts, len(labels))
    if _seeded(comp_parts):
        res = rs_stage4.d_g_block(cand, comp, labels, draws, workers)
        a_cand, a_comp, interval = res['A_cand'], res['A_G'], [res['lo'], res['hi']]
    else:
        res = rg_held.bootstrap_d_t(cand, comp, labels, draws, workers)
        a_cand, a_comp, interval = res['A_G'], res['A_T'], list(res['interval'])
    return dict(A_cand=a_cand, A_comp=a_comp, d=res['d'], interval=interval, bootstrap_d=res['bootstrap_d'])


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
        if side['array_sha256'] == row_sha and side['settings'] == settings:
            with np.load(path, allow_pickle=False) as z:
                assert z.files == ['labels']
                labels = z['labels']
            assert labels.dtype == np.int32 and labels.shape == (len(STATES), len(x32)), (path, labels.shape)
            return labels
    labels = partitions(x32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{name}.tmp.npz')
    np.savez(tmp, labels=labels)
    os.replace(tmp, path)
    side_path.write_text(json.dumps(dict(array_sha256=row_sha, shape=list(x32.shape), settings=settings,
                                         partitions_sha256=rs_vh.file_sha(path)), indent=2) + '\n')
    return labels
