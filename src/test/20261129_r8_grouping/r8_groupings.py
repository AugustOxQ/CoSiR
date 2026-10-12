"""Round-8 groupings (DECISION_RULE.md M3, M7 labels, M8, M9, M13, M16; ticket 07).

One CLI step per grouping ID (`part --id X`, run in parallel by the controller) and one `collect` that writes
`results/groupings/r8_groupings.{npz,json}` once (key = grouping ID, int64 compact labels over the scorer-train rows,
constant within painting, plus `scorer_train`, `g_st`; the json has `groupings: {ID: M3 record}` and
`dropped: {ID: reason}`). NBC is triggered after the gate looks at the NB scorers, so it goes into a second file
(`collect --ids NBC --name r8_groupings_nbc`).

Pure functions (search, merge, controls, TP map, NRK choice) do not touch r8_common; the file steps import it lazily.
CPU only. Label-free AMIs with the content grouping may be recorded (rule 7); no style or genre label is read.
"""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

TARGET_K = 17
MIN_ROWS = 200
GRAPH_K = 20
LEIDEN_SEED = 42
STAB_SEEDS = (43, 44)
GRID = [j / 10 for j in range(1, 31)]            # M3 (2): r = j/10, j = 1 to 30
MAX_BISECT = 10
CLUSTPY_VERSION = '0.0.3'
TIMEOUT_SECONDS = 30 * 60

SEARCH_IDS = ('V100', 'V67', 'V33', 'NB100', 'NB67', 'NB33', 'NBC', 'AL', 'DI', 'SG')
FC_IDS = tuple(f'FC{k}' for k in range(1, 7))
REF_IDS = ('R0own', 'NRK', 'TP')
ALL_IDS = SEARCH_IDS[:6] + FC_IDS + SEARCH_IDS[6:] + REF_IDS
# M13: index i of X (seed 7100 + i for the labels, 7200 + i and 7300 + i for the image-side draws)
CONTROL_INDEX = {name: i for i, name in enumerate(
    ('V100', 'V67', 'V33', 'NB100', 'NB67', 'NB33') + FC_IDS + ('NBC', 'AL', 'DI', 'SG'), start=1)}
assert CONTROL_INDEX['V100'] == 1 and CONTROL_INDEX['FC1'] == 7 and CONTROL_INDEX['NBC'] == 13 \
    and CONTROL_INDEX['SG'] == 16
ENCODER_DIR = {'AL': 'aladin', 'DI': 'dinov2l', 'SG': 'siglip2'}
ENCODER_ROOT = Path('/data/SSD2/pre_extract/artelingo')
N_IMAGES = 49_121


# ---------------------------------------------------------------- small helpers

def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    out = x / np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(out).all(), 'non-finite unit vectors (zero vector?)'
    return out


def compact(labels):
    """Labels renumbered 0..k-1 by sorted value (empty clusters vanish); int64."""
    return np.unique(np.asarray(labels), return_inverse=True)[1].astype(np.int64).reshape(-1)


def painting_labels(local, g):
    """The per-painting labels of constant-within-painting row labels; asserts constancy and compactness (R7)."""
    first = np.unique(g, return_index=True)[1]
    p = local[first]
    assert np.array_equal(np.unique(g), np.arange(len(first))), 'painting codes must be 0..n-1'
    assert np.array_equal(p[g], local), 'labels differ within a painting'
    assert np.array_equal(np.unique(p), np.arange(p.max() + 1)), 'labels are not compact'
    return p


def _sweep():
    import r8_common as C
    return C.load_prior('run_sweep')


def _told():
    import r8_common as C
    return C.load_prior('run_told_oracle')


def _ami(a, b):
    from sklearn.metrics import adjusted_mutual_info_score
    return float(adjusted_mutual_info_score(a, b))


def describe(local, g, content_p=None):
    p = painting_labels(local, g)
    rec = {'n_groups': int(p.max()) + 1, 'sizes_rows': np.bincount(local).tolist(),
           'sizes_paintings': np.bincount(p).tolist(), 'partition_sha256': sha_array(local)}
    if content_p is not None:        # label-free (rule 7): AMI with the content grouping
        rec['ami_content_paintings'] = _ami(content_p, p)
        rec['ami_content_rows'] = _ami(content_p[g], local)
    return rec


# ---------------------------------------------------------------- M3: Leiden at r and the resolution search

def leiden_labels(graph, r, seed):
    """leidenalg RBConfigurationVertexPartition at resolution r (R7 run_sweep.leiden's call, with r and seed free)."""
    import leidenalg
    p = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=r, seed=seed)
    return compact(np.asarray(p.membership, dtype=np.int64))


def _key(r):
    return round(float(r), 9)


def search_resolution(k_of_r, target=TARGET_K):
    """M3's search on k_of_r(r) -> K or None (None = merge_small raised SystemExit: that r is skipped and recorded).
    (1) r = 1.0, stop if K = target. (2) grid r = j/10, j = 1..30: among K = target the point nearest 1.0, ties to the
    smaller r. (3) else the adjacent grid pair whose K values bracket the target with midpoint nearest 1.0 (ties to the
    smaller) is bisected at most 10 times, keeping the half that still brackets; stop at the first K = target.
    (4) else the nearest count: smallest |K - target| over everything evaluated, ties to r nearest 1.0, then smaller r.
    Returns {'evaluated': [{r, K, [reason]}] in call order, 'chosen_r', 'K', 'stage', 'dropped', 'reason'}."""
    seen, order = {}, []

    def ev(r):
        k = _key(r)
        if k not in seen:
            seen[k] = k_of_r(k)
            order.append(k)
        return seen[k]

    def result(stage, r=None):
        evaluated = [{'r': k, 'K': None if seen[k] is None else int(seen[k]),
                      **({'reason': 'K undefined: merge_small raised SystemExit (no group reaches 200 rows)'}
                         if seen[k] is None else {})} for k in order]
        valid = [k for k in order if seen[k] is not None]
        if r is None and not valid:
            return {'evaluated': evaluated, 'chosen_r': None, 'K': None, 'stage': 'none', 'dropped': True,
                    'reason': 'merge_small is undefined at every evaluated resolution'}
        if r is None:                                                           # (4)
            r = min(valid, key=lambda k: (abs(int(seen[k]) - target), abs(k - 1.0), k))
            stage = 'nearest'
        return {'evaluated': evaluated, 'chosen_r': r, 'K': int(seen[r]), 'stage': stage, 'dropped': False,
                'reason': None}

    if ev(1.0) == target:                                                       # (1)
        return result('r1', _key(1.0))
    grid = [_key(r) for r in GRID]
    for r in grid:
        ev(r)
    hits = [r for r in grid if seen[r] == target]                               # (2)
    if hits:
        return result('grid', min(hits, key=lambda k: (abs(k - 1.0), k)))
    # (3) rule gap: "adjacent grid pair" is read as adjacent among the grid points whose K is defined
    valid = [r for r in grid if seen[r] is not None]
    brackets = [(a, b) for a, b in zip(valid, valid[1:])
                if (seen[a] - target) * (seen[b] - target) < 0]
    if brackets:
        lo, hi = min(brackets, key=lambda ab: (abs((ab[0] + ab[1]) / 2 - 1.0), ab[0]))
        for _ in range(MAX_BISECT):
            mid = _key((lo + hi) / 2)
            k = ev(mid)
            if k == target:
                return result('bisect', mid)
            if k is None:        # rule gap: no half can be kept when K(mid) is undefined; bisection stops
                break
            if (seen[lo] - target) * (k - target) < 0:
                hi = mid
            else:
                lo = mid
    return result('nearest')


def group_space(phi_p, g, content_p=None, target=TARGET_K, min_rows=MIN_ROWS):
    """M3 recipe on painting vectors phi_p (rows inherit their painting's vector for merge_small): unit vectors,
    knn_union_graph k = 20 built once, Leiden RBConfiguration seed 42 at the searched r, merge_small at min_rows.
    Seeds 43 and 44 at the chosen r are descriptive. Returns (row labels int64 or None if dropped, record)."""
    sw, to = _sweep(), _told()
    xp = unit(phi_p)
    graph = sw.knn_union_graph(xp, GRAPH_K)
    rows64 = xp[g].astype(np.float64)               # merge_small converts to float64 anyway; do it once
    cache, why = {}, {}

    def merged(r, seed):
        raw = leiden_labels(graph, r, seed)
        return to.merge_small(raw[g], rows64, min_rows)

    def k_of_r(r):
        try:
            cache[r] = merged(r, LEIDEN_SEED)
        except SystemExit as exc:
            why[r] = str(exc)
            return None
        return int(cache[r][0].max()) + 1

    rec = search_resolution(k_of_r, target)
    for e in rec['evaluated']:
        if e['K'] is None:
            e['reason'] = f"K undefined: merge_small raised SystemExit: {why[e['r']]}"
    rec.update(seed=LEIDEN_SEED, target_K=target, merge_min_rows=min_rows,
               graph={'k': GRAPH_K, 'vertices': graph.vcount(), 'edges': graph.ecount()})
    if rec['dropped']:
        return None, rec
    local, moves = cache[rec['chosen_r']]
    rec.update(describe(local, g, content_p), merge_moves=moves, exact_17=rec['K'] == target)
    labels = {LEIDEN_SEED: local}
    rec['stability_seeds'] = {}
    for s in STAB_SEEDS:
        try:
            labels[s], _ = merged(rec['chosen_r'], s)
            rec['stability_seeds'][str(s)] = {'n_groups': int(labels[s].max()) + 1,
                                              'partition_sha256': sha_array(labels[s])}
        except SystemExit as exc:
            rec['stability_seeds'][str(s)] = {'n_groups': None, 'reason': str(exc)}
    rec['stability_ami'] = {f'{a}_{b}': _ami(labels[a], labels[b]) for a, b in ((42, 43), (42, 44), (43, 44))
                            if a in labels and b in labels}
    if rec['n_groups'] < 2:
        rec.update(dropped=True, reason='fewer than 2 groups after merge')
        return None, rec
    return local, rec


# ---------------------------------------------------------------- M7 labels and M8 merge: native labels

def native_group(labels_p, phi_p, g, content_p=None, min_rows=MIN_ROWS):
    """Native painting labels (FCMI, NRK, TP): empty clusters removed and labels compacted, then merge_small at
    min_rows rows in the unit phi space (rows inherit their painting's vector). Returns (row labels or None, record)."""
    to = _told()
    raw = compact(labels_p)
    rec = {'raw_n_clusters': int(raw.max()) + 1, 'raw_sizes_paintings': np.bincount(raw).tolist(),
           'raw_sizes_rows': np.bincount(raw[g]).tolist(), 'merge_min_rows': min_rows, 'dropped': False,
           'reason': None}
    xp = unit(phi_p)
    try:
        local, moves = to.merge_small(raw[g], xp[g], min_rows)
    except SystemExit as exc:
        rec.update(dropped=True, reason=f'merge_small: {exc}')
        return None, rec
    rec.update(describe(local, g, content_p), merge_moves=moves)
    if rec['n_groups'] < 2:
        rec.update(dropped=True, reason='fewer than 2 groups after merge')
        return None, rec
    return local, rec


# ---------------------------------------------------------------- R0own

R0_KEY = '20261116_grouping_step1_style/results/step1_group_style.npz'
R7_GROUPINGS_KEY = '20261126_design_l_style_source/results/groupings/r7_groupings.npz'


def r0_frozen_labels(scorer_train, g_st, content_p=None):
    """R0's frozen 17 groups (round 1's `style_csd`), asserted equal to R7's `r7_groupings.npz` key R0; no search."""
    import r8_common as C
    C.assert_inputs(keys=(R0_KEY, R7_GROUPINGS_KEY), r7=False)
    with np.load(C.path_of(R0_KEY), allow_pickle=False) as z:
        assert {'scorer_train', 'g_st', 'style_csd', 'style_csd__seed43', 'style_csd__seed44'} <= set(z.files)
        assert np.array_equal(z['scorer_train'], scorer_train) and np.array_equal(z['g_st'], g_st), 'R0 alignment'
        labels = np.asarray(z['style_csd'])
        seeds = {42: labels, 43: z['style_csd__seed43'], 44: z['style_csd__seed44']}
    with np.load(C.path_of(R7_GROUPINGS_KEY), allow_pickle=False) as z:
        assert np.array_equal(z['R0'], labels), 'r7_groupings R0 differs from round 1 style_csd'
    assert labels.dtype == np.int64 and labels.shape == scorer_train.shape and len(np.unique(labels)) == 17
    rec = describe(labels, g_st, content_p)
    rec.update(dropped=False, reason=None, source=R0_KEY, source_sha256=C.INPUTS[R0_KEY],
               stability_ami={f'{a}_{b}': _ami(seeds[a], seeds[b]) for a, b in ((42, 43), (42, 44), (43, 44))})
    return labels, rec


# ---------------------------------------------------------------- M8: NrKmeans (disposable child)

def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def worker(directory):
    """Only the child imports ClustPy: fit NrKmeans(n_clusters=[17, 17], random_state=0), save labels, V_, P_."""
    directory = Path(directory)
    started = time.monotonic()
    rec = {'dropped': True, 'reason': None}
    try:
        assert package_version('clustpy') == CLUSTPY_VERSION, 'ClustPy version differs from pin'
        from clustpy.alternative import NrKmeans
        x = np.load(directory / 'vectors.npy', allow_pickle=False)
        assert x.dtype == np.float32 and x.ndim == 2 and np.isfinite(x).all() and len(x) >= 17
        model = NrKmeans(n_clusters=[17, 17], random_state=0).fit(x)
        labels = np.asarray(model.labels_)
        assert labels.shape == (len(x), 2) and labels.dtype.kind == 'i' and (labels >= 0).all(), 'labels_ shape'
        assert len(model.P_) == 2, 'NrKmeans did not keep two subspaces'
        np.savez(directory / 'fit.npz', labels=labels.astype(np.int64), V=np.asarray(model.V_, dtype=np.float64),
                 P0=np.asarray(model.P_[0], dtype=np.int64), P1=np.asarray(model.P_[1], dtype=np.int64))
        rec.update(dropped=False)
    except (Exception, SystemExit) as exc:
        rec['reason'] = f'{type(exc).__name__}: {exc}'
    rec['fit_seconds'] = time.monotonic() - started
    (directory / 'worker.json').write_text(json.dumps(rec))


def fit_child(x, out, timeout_seconds=TIMEOUT_SECONDS, command=None):
    """Import and fit in a child with a time limit (M8: 30 minutes, dropped past it). Returns (fit dict or None, record).
    `command` replaces the child command line (tests)."""
    started = time.monotonic()
    rec = {'dropped': True, 'reason': None, 'timeout_seconds': timeout_seconds,
           'clustpy_version': package_version('clustpy'), 'clustpy_version_pin': CLUSTPY_VERSION,
           'parameters': {'n_clusters': [17, 17], 'random_state': 0}, 'other_parameters': 'ClustPy defaults'}
    fit = None
    with tempfile.TemporaryDirectory(prefix='nrkmeans_', dir=out) as tmp:
        directory = Path(tmp)
        np.save(directory / 'vectors.npy', x, allow_pickle=False)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}
        cmd = command or [sys.executable, str(Path(__file__).resolve()), '--worker', str(directory)]
        try:
            with open(directory / 'child.log', 'w+') as log:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=log, timeout=timeout_seconds, check=False)
                log.seek(0)
                rec['child_log_tail'] = log.read()[-8000:]
            rec['child_returncode'] = result.returncode
            assert result.returncode == 0, f'child exited with code {result.returncode}'
            status = json.loads((directory / 'worker.json').read_text())
            assert {'dropped', 'reason', 'fit_seconds'} <= status.keys(), 'worker record keys'
            rec.update(status)
            if not rec['dropped']:
                with np.load(directory / 'fit.npz', allow_pickle=False) as z:
                    fit = {k: np.asarray(z[k]) for k in z.files}
                assert fit['labels'].dtype == np.int64 and fit['labels'].shape == (len(x), 2), 'child labels'
                assert fit['V'].shape == (x.shape[1], x.shape[1]), 'child V_ shape'
        except subprocess.TimeoutExpired:
            rec.update(dropped=True, reason=f'NrKmeans exceeded {timeout_seconds} seconds')
        except (Exception, SystemExit) as exc:
            fit = None
            rec.update(dropped=True, reason=f'{type(exc).__name__}: {exc}')
    rec['run_seconds'] = time.monotonic() - started
    return fit, rec


def nrk_choose(labels2, content_p, g):
    """The kept clustering: the one with the lower AMI with the content grouping over the scorer-train rows (M8; label
    free; ties to the first). Returns (index, info)."""
    cols = [compact(labels2[:, k]) for k in (0, 1)]
    rows = [_ami(content_p[g], c[g]) for c in cols]
    paintings = [_ami(content_p, c) for c in cols]
    chosen = 0 if rows[0] <= rows[1] else 1
    return chosen, {'ami_content_rows_both': rows, 'ami_content_paintings_both': paintings, 'chosen_clustering': chosen,
                    'choice_level': 'rows',        # R7 chose at painting level; M8 says rows
                    'choice_painting_level_would_pick': 0 if paintings[0] <= paintings[1] else 1}


def nrk_space(x, V, P, s):
    """Unit transform_subspace(x, s) = x @ V_[:, P_[s]] (nrkmeans.py l.944), float32."""
    return unit(np.asarray(x, dtype=np.float64) @ np.asarray(V)[:, np.asarray(P[s])])


# ---------------------------------------------------------------- M9: told prompts

def tp_prompts(paintings, expect=27):
    """Style names = the sorted top-level folders of paintings.json image paths; prompts 'a {style} painting' with
    underscores as spaces (exact strings recorded)."""
    names = sorted({v['image'].split('/')[0] for v in paintings.values()})
    assert len(names) == expect, f'{len(names)} style folders, not {expect}'
    return names, [f"a {n.replace('_', ' ')} painting" for n in names]


def tp_argmax(x, t):
    """argmax_s x_i . t_s in float64, ties to the lowest index."""
    return np.argmax(np.asarray(x, dtype=np.float64) @ np.asarray(t, dtype=np.float64).T, axis=1).astype(np.int64)


def tp_group(x_p, t, g, min_rows=MIN_ROWS, content_p=None):
    """TP: zero-shot class of each painting (x_p unit CLIP), groups under min_rows scorer-train rows merged by
    merge_small in x's space. Returns (row labels or None, record, merge map int64 [n_classes], -1 = class without a
    scorer-train painting)."""
    raw = tp_argmax(x_p, t)
    local, rec = native_group(raw, x_p, g, content_p, min_rows)
    mmap = np.full(len(t), -1, dtype=np.int64)
    if local is not None:
        p = painting_labels(local, g)
        mmap[raw] = p                               # merged label is a function of the raw class (asserted)
        assert np.array_equal(mmap[raw], p), 'TP merge map is not a function of the class'
        rec['merge_map'] = mmap.tolist()
        rec['raw_class_sizes_paintings'] = np.bincount(raw, minlength=len(t)).tolist()
    return local, rec, mmap


def tp_centres(x_p, labels_p):
    """Unit mean of x_p over each merged TP group (only used for gallery images of a class without paintings)."""
    k = int(labels_p.max()) + 1
    mu = np.stack([np.asarray(x_p, dtype=np.float64)[labels_p == c].mean(axis=0) for c in range(k)])
    return mu / np.linalg.norm(mu, axis=1, keepdims=True)


def tp_apply(x, t, mmap, centres=None):
    """The TP map of images x (unit CLIP): argmax class, then the merge map (the gallery goes through the same map).
    Returns (group per image, number of images that took the fallback)."""
    cls = tp_argmax(x, t)
    out = np.asarray(mmap)[cls]
    missing = out < 0
    # rule gap: a class that no scorer-train painting has as its argmax has no merged group; M9 does not say where a
    # gallery image of that class goes. With `centres` the image takes the nearest merged-group centre in x's space
    # (merge_small's own criterion, applied to the image); without them the call stops.
    if missing.any():
        assert centres is not None, 'a gallery image has a TP class without scorer-train paintings'
        out = out.copy()
        out[missing] = np.argmax(np.asarray(x, dtype=np.float64)[missing] @ np.asarray(centres).T, axis=1)
    return out, int(missing.sum())


# ---------------------------------------------------------------- M13: random controls

def random_control_labels(local, g, i):
    """R7 L8 with default_rng(7100 + i): the painting indices are permuted, consecutive blocks of the permutation form
    groups 0, 1, ... with X's group sizes in paintings (X's group order); rows inherit their painting's group."""
    p = painting_labels(local, g)
    sizes = np.bincount(p)
    perm = np.random.default_rng(7100 + i).permutation(len(p))
    z = np.empty(len(p), dtype=np.int64)
    z[perm] = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    assert np.array_equal(np.bincount(z), sizes), 'M13 painting sizes'
    return z[g]


def control_draws(labels_p, i, n_sel):
    """M13's image-side draws (for ticket 08's banks; same calls): shares = bincount(labels_p, minlength=K) / n
    (float64); scorer-train paintings (the same draw in both halves) default_rng(7200 + i).choice(K, size=n, p=shares)
    in M1 order; selection-row paintings default_rng(7300 + i).choice(K, size=n_sel, p=shares), in ascending
    artelingo_splits().groups code over the selection rows."""
    n = len(labels_p)
    k = int(labels_p.max()) + 1
    shares = np.bincount(labels_p, minlength=k) / n
    train = np.random.default_rng(7200 + i).choice(k, size=n, p=shares)
    sel = np.random.default_rng(7300 + i).choice(k, size=n_sel, p=shares)
    return train, sel


# ---------------------------------------------------------------- collect (M16)

def assemble(parts, scorer_train, g, ids):
    """parts: {ID: {'labels': row labels or None, 'record': dict}} for the requested ids. Returns (arrays, record):
    labels per non-dropped ID plus Z<X> for each X with a control index, `dropped` {ID: reason}."""
    arrays = {'scorer_train': np.asarray(scorer_train), 'g_st': np.asarray(g)}
    groupings, dropped, dropped_records = {}, {}, {}
    for name in ids:
        part = parts[name]
        rec, labels = dict(part['record']), part['labels']
        if labels is not None:
            assert labels.dtype == np.int64 and labels.shape == g.shape, f'{name}: label shape/dtype'
            painting_labels(labels, g)
            assert labels.max() >= 1, f'{name}: fewer than 2 groups must be dropped before collect'
        if labels is None or rec.get('dropped'):
            assert rec.get('reason'), f'{name}: dropped without a reason'
            dropped[name], dropped_records[name] = str(rec['reason']), rec
        else:
            arrays[name] = labels
            groupings[name] = rec
        if name in CONTROL_INDEX:
            zname, i = f'Z{name}', CONTROL_INDEX[name]
            if name in dropped:
                dropped[zname] = f'no control: {name} is dropped ({dropped[name]})'
            else:
                z = random_control_labels(labels, g, i)
                arrays[zname] = z
                groupings[zname] = {**describe(z, g), 'dropped': False, 'reason': None, 'seed': 7100 + i,
                                    'matched_grouping': name, 'index': i}
    return arrays, {'groupings': groupings, 'dropped': dropped, 'dropped_records': dropped_records,
                    'ids': list(ids)}


# ====================================================================== file steps (import r8_common lazily)

def _paths(smoke):
    import r8_common as C
    root = C.SMOKE if smoke else C.RESULTS
    return root, root / 'groupings' / 'parts', root / 'spaces'


def read_space(path, smoke=False):
    """A space file (05, 06, 07's own): phi float32 [n, d] unit, image_index int32 (the non-held images; a subset in
    smoke). Returns phi, image_index."""
    import r8_common as C
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) >= {'phi', 'image_index'}, f'{path}: keys {z.files}'
        phi, idx = np.asarray(z['phi']), np.asarray(z['image_index'])
    assert phi.dtype == np.float32 and phi.ndim == 2 and idx.dtype == np.int32 and idx.shape == (len(phi),), \
        f'{path}: phi/image_index dtype or shape'
    assert np.isfinite(phi).all(), f'{path}: non-finite phi'
    assert np.allclose(np.linalg.norm(phi, axis=1), 1, atol=1e-4), f'{path}: phi is not unit'
    nh = C.nonheld_images()
    if smoke:
        assert np.all(np.diff(idx) > 0) and np.isin(idx, nh).all(), f'{path}: smoke image_index'
    else:
        assert np.array_equal(idx, nh), f'{path}: image_index differs from nonheld_images()'
    return phi, idx


def painting_vectors(phi, idx, n):
    """phi of the first n paintings (M1 order) through their images; every one must be in the file."""
    import r8_common as C
    img = C.painting_image()[:n]
    pos = np.searchsorted(idx, img)
    assert (pos < len(idx)).all() and np.array_equal(idx[pos], img), 'a painting image is not in the space file'
    return phi[pos]


def subset(n):
    """(scorer_train, g_st, content, n_paintings) restricted to the first n paintings (n=None: all)."""
    import r8_common as C
    c = C.ctx()
    n = c.n_paintings if n is None else int(n)
    mask = c.g_st < n
    return c.scorer_train[mask], c.g_st[mask], c.content[:n], n


def write_space(path, record_path, phi, image_index, record):
    import r8_common as C
    C.write_npz_once(path, phi=phi, image_index=image_index)
    return C.write_json_once(record_path, {**record, 'space_sha256': C.sha_file(path), 'phi_shape': list(phi.shape)})


def encoder_space(name, smoke, spaces):
    """AL/DI/SG: raw embeddings -> unit space file (written once; reused when it is there and its input matches)."""
    import r8_common as C
    d = ENCODER_ROOT / f'r8_enc_{ENCODER_DIR[name]}'
    emb_path, meta_path = d / 'embeddings.npy', d / 'meta.json'
    assert emb_path.is_file() and meta_path.is_file(), f'{name}: {d} is not extracted (ticket 04)'
    emb = np.load(emb_path, mmap_mode='r', allow_pickle=False)
    assert emb.ndim == 2 and emb.shape[0] == N_IMAGES and emb.dtype == np.float32, f'{name}: embeddings shape/dtype'
    assert np.isfinite(emb).all(), f'{name}: embeddings not finite'
    sha, meta_sha = C.sha_file(emb_path), C.sha_file(meta_path)
    space, rec_path = spaces / f'{name}.npz', spaces / f'{name}.json'
    if space.exists():
        assert json.loads(rec_path.read_text())['input_sha256']['embeddings.npy'] == sha, f'{space}: other input'
        return space
    phi = unit(emb)
    write_space(space, rec_path, phi, C.nonheld_images(),
                {'builder': 'r8_groupings', 'setting': name, 'smoke': smoke, 'source': str(emb_path),
                 'input_sha256': {'embeddings.npy': sha, 'meta.json': meta_sha}, 'transform': 'unit'})
    return space


def read_record(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def record_pass(rec):
    """The `pass` of a stage/D21 record (ticket 06 writes `passed`, the gate index says `pass`): None if absent."""
    if rec is None:
        return None
    return rec['pass'] if 'pass' in rec else rec.get('passed')


def fail_reason(rec, label):
    """Why a stage/D21 record failed: its failed tests and its error text."""
    failed = [k for k, v in (rec.get('tests') or {}).items() if not v]
    return f"{label} failed: tests {failed or 'n/a'}; {rec.get('error') or rec.get('reason') or 'no error text'}"


def stage_pass(name, smoke):
    """The stage record's pass (stageE_<AL|DI|SG>.json, stage1_fcmi.json): None when the record does not exist."""
    import r8_common as C
    return record_pass(read_record((C.SMOKE / 'gate' if smoke else C.RESULTS) / name))


def part_search(name, smoke, n, strength_note=None):
    import r8_common as C
    root, parts, spaces = _paths(smoke)
    st, g, content, n = subset(n)
    if name in ENCODER_DIR:
        if not smoke and stage_pass(f'stageE_{name}.json', smoke) is not True:
            return None, {'dropped': True, 'reason': f'ESC1 not passed for {name} (stageE_{name}.json: '
                                                    f'{stage_pass(f"stageE_{name}.json", smoke)})'}, n
        encoder_space(name, smoke, spaces)
    phi, idx = read_space(spaces / f'{name}.npz', smoke)
    local, rec = group_space(painting_vectors(phi, idx, n), g, content)
    rec['space'] = str(spaces / f'{name}.npz')
    rec['space_sha256'] = C.sha_file(spaces / f'{name}.npz')
    return local, rec, n


def fc_paths(name, smoke):
    import r8_common as C
    w = FC_IDS.index(name)
    if smoke:
        d = C.SMOKE / 'fcmi' / name
        return d / ('stage1_fcmi_labels.npz' if w == 2 else 'd21_labels.npz'), d / 'spaces' / f'{name}.npz', \
            d / ('stage1_fcmi.json' if w == 2 else 'd21.json')
    return (C.RESULTS / 'stage1_fcmi_labels.npz' if w == 2 else C.RESULTS / 'fcmi' / name / 'd21_labels.npz'), \
        C.RESULTS / 'spaces' / f'{name}.npz', (C.RESULTS / 'stage1_fcmi.json' if w == 2 else
                                               C.RESULTS / 'fcmi' / name / 'd21.json')


def part_fc(name, smoke, labels_path=None, space_path=None):
    """FC<n>: ticket 06's space file and native labels (path read from the space JSON's `native_labels`), then
    compaction and the 200-row merge_small. Drops: FC3 failed SC2 -> all of FC1 to FC6 (reason from stage1_fcmi.json);
    another weight failed D21 (no space file) -> that FC only (reason from fcmi/FC<n>/d21.json)."""
    import r8_common as C
    lp, sp, vp = fc_paths(name, smoke)
    sp = Path(space_path or sp)
    if not smoke:
        stage1 = read_record(C.RESULTS / 'stage1_fcmi.json')
        assert stage1 is not None, 'stage1_fcmi.json is missing: run stage 1 (ticket 06) first'
        if record_pass(stage1) is False:
            return None, {'dropped': True, 'reason': fail_reason(stage1, 'SC2 (FC3, stage1_fcmi.json)') +
                          '; FC1 to FC6 drop out'}, None
    if not sp.is_file():
        v = read_record(vp)
        if v is not None and record_pass(v) is False:
            return None, {'dropped': True, 'reason': fail_reason(v, f'D21 test of {name} ({vp})')}, None
    assert sp.is_file(), f'{name}: space file {sp} missing (ticket 06)'
    meta = read_record(sp.with_suffix('.json'))
    assert meta is not None and 'native_labels' in meta, f'{sp.with_suffix(".json")}: no native_labels field'
    lp = Path(labels_path or meta['native_labels'])
    assert lp.is_file(), f'{name}: native labels {lp} missing'
    assert 'native_labels_sha256' not in meta or C.sha_file(lp) == meta['native_labels_sha256'], 'native labels SHA'
    ctx = C.ctx()
    with np.load(lp, allow_pickle=False) as z:
        assert {'labels', 'painting_index', 'painting_ids', 'painting_image'} <= set(z.files)
        labels, pidx, pids, pimg = (np.asarray(z[k]) for k in ('labels', 'painting_index', 'painting_ids',
                                                               'painting_image'))
    n = len(labels)
    assert labels.dtype == np.int64 and n <= ctx.n_paintings and (smoke or n == ctx.n_paintings)
    assert np.array_equal(pidx, np.arange(n)), 'native labels: painting_index'
    assert np.array_equal(pids.astype(str), ctx.painting_ids[:n]), 'native labels: painting ids differ (C6)'
    assert np.array_equal(pimg, C.painting_image()[:n]), 'native labels: painting images differ (C6)'
    st, g, content, n = subset(n)
    phi, idx = read_space(sp, smoke)
    local, rec = native_group(labels, painting_vectors(phi, idx, n), g, content)
    rec.update(space=str(sp), space_sha256=C.sha_file(sp), native_labels=str(lp), native_labels_sha256=C.sha_file(lp))
    return local, rec, n


def part_r0own(smoke, n):
    import r8_common as C
    root, parts, spaces = _paths(smoke)
    c = C.ctx()
    st, g, content, n = subset(n)
    full = c.n_paintings
    labels, rec = r0_frozen_labels(c.scorer_train, c.g_st, c.content)
    if n != full:                                   # smoke: the first n paintings of R0's labels
        labels = compact(labels[c.g_st < n])
        rec = {**rec, 'smoke_subset_paintings': n}
    sp = spaces / 'R0own.npz'
    if not sp.exists():
        C.assert_inputs(keys=('/data/SSD2/pre_extract/artelingo/style_csd_vitl/embeddings.npy',), r7=False)
        csd = np.load(C.STYLE / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        nh = C.nonheld_images()
        write_space(sp, spaces / 'R0own.json', unit(csd[nh]), nh,
                    {'builder': 'r8_groupings', 'setting': 'R0own', 'smoke': smoke, 'transform': 'unit CSD',
                     'input_sha256': {'embeddings.npy': C.INPUTS['/data/SSD2/pre_extract/artelingo/style_csd_vitl/'
                                                                 'embeddings.npy']}})
    rec['space'] = str(sp)
    return labels, rec, n


def part_nrk(smoke, n, timeout_seconds=TIMEOUT_SECONDS):
    import r8_common as C
    root, parts, spaces = _paths(smoke)
    assert timeout_seconds == TIMEOUT_SECONDS or smoke, 'a shortened NRK time limit is for --smoke only'
    st, g, content, n = subset(n)
    x = np.ascontiguousarray(C.x_paintings()[:n])
    out = parts
    out.mkdir(parents=True, exist_ok=True)
    fit, rec = fit_child(x, out, timeout_seconds)
    if fit is None:
        return None, rec, n
    P = [fit['P0'], fit['P1']]
    s, info = nrk_choose(fit['labels'], content, g)
    rec.update(info, subspace_dims=[len(p) for p in P])
    phi_p = nrk_space(x, fit['V'], P, s)
    local, nrec = native_group(fit['labels'][:, s], phi_p, g, content)
    rec.update(nrec)
    if rec['dropped']:
        return None, rec, n
    sp = spaces / 'NRK.npz'
    nh = C.nonheld_images()
    write_space(sp, spaces / 'NRK.json', nrk_space(C.x_images(), fit['V'], P, s), nh,
                {'builder': 'r8_groupings', 'setting': 'NRK', 'smoke': smoke, 'subspace': s, 'P': [p.tolist() for p in P],
                 'V_sha256': sha_array(fit['V']), 'transform': 'unit x @ V_[:, P_[s]]'})
    rec.update(space=str(sp), space_sha256=C.sha_file(sp))
    return local, rec, n


def clip_text(texts, device='cpu'):
    """Unit CLIP B/32 text features of the cached features' weights (float32, tokenizer as the extraction:
    padding, truncation, max_length 77). Returns (features, weight record)."""
    import torch
    import r8_common as C
    from transformers import CLIPModel, CLIPTokenizer
    snap = next(Path(k).parent for k in C.CLIP_WEIGHTS if Path(k).is_file() and (Path(k).parent / 'config.json').is_file())
    model = CLIPModel.from_pretrained(snap, dtype=torch.float32).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(snap)
    loaded = next(snap / f for f in ('model.safetensors', 'pytorch_model.bin') if (snap / f).is_file())
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            t = tok(list(texts[i:i + 64]), return_tensors='pt', padding=True, truncation=True, max_length=77).to(device)
            f = model.get_text_features(**t)
            f = getattr(f, 'pooler_output', f)
            out.append(f.float().cpu().numpy())
    return unit(np.concatenate(out)), C.clip_weight_record(loaded)


def text_check(n_rows=256, device='cpu'):
    """M9: the captions of the first 256 scorer-train rows, re-encoded, against their cached txt_features."""
    import r8_common as C
    art = importlib.import_module('src.data.artelingo')
    c = C.ctx()
    rows = c.scorer_train[:n_rows]
    ann = json.loads(Path(art.ANNOTATIONS_PATH).read_text())
    caps = art.join_captions(c.data.sample_ids[rows], ann)
    feats, weight = clip_text([str(s) for s in caps], device)
    cached = unit(c.data.txt_features[rows])
    cos = (feats.astype(np.float64) * cached).sum(axis=1)
    return {'n': int(n_rows), 'min_cosine': float(cos.min()), 'passed': bool(cos.min() >= 0.9999),
            'weights': weight}


def part_tp(smoke, n, device='cpu'):
    import r8_common as C
    root, parts, spaces = _paths(smoke)
    C.assert_inputs(keys=('/data/SSD2/pre_extract/artelingo_clip224/paintings.json',), r7=False)
    paintings = json.loads((C.path_of('/data/SSD2/pre_extract/artelingo_clip224/paintings.json')).read_text())['paintings']
    names, prompts = tp_prompts(paintings)
    t, weight = clip_text(prompts, device)
    check = text_check(256, device)
    rec = {'style_names': names, 'prompts': prompts, 'text_check': check, 'clip_weights': weight}
    if not check['passed']:
        rec.update(dropped=True, reason=f"text check failed: min cosine {check['min_cosine']:.6f} < 0.9999")
        return None, rec, None
    st, g, content, n = subset(n)
    x = C.x_paintings()[:n]
    local, trec, mmap = tp_group(x, t, g, content_p=content)
    rec.update(trec)
    if local is None:
        return None, rec, n
    images, n_fallback = tp_apply(C.x_images(), t, mmap, tp_centres(x, painting_labels(local, g)))
    rec['gallery_images_without_class_paintings'] = n_fallback
    k = int(local.max()) + 1
    phi = np.zeros((len(images), k), dtype=np.float32)
    phi[np.arange(len(images)), images] = 1.0                 # one-hot: M11's own map gives exactly the TP map
    sp = spaces / 'TP.npz'
    write_space(sp, spaces / 'TP.json', phi, C.nonheld_images(),
                {'builder': 'r8_groupings', 'setting': 'TP', 'smoke': smoke, 'transform': 'one-hot of the TP map',
                 'prompts': prompts, 'merge_map': mmap.tolist(), 'text_check': check, 'clip_weights': weight,
                 'gallery_group_sizes': np.bincount(images, minlength=k).tolist()})
    rec.update(space=str(sp), space_sha256=C.sha_file(sp), dropped=False, reason=None)
    return local, rec, n


def write_part(name, labels, rec, n, smoke):
    import r8_common as C
    root, parts, _ = _paths(smoke)
    rec = {**rec, 'id': name, 'smoke': smoke, 'n_paintings': n}
    if labels is not None:
        C.write_npz_once(parts / f'{name}.npz', labels=labels, n_paintings=np.int64(n))
    else:
        assert rec.get('dropped') and rec.get('reason'), f'{name}: no labels and no drop reason'
    return C.write_json_once(parts / f'{name}.json', rec)


def run_part(name, smoke=False, n=None, labels=None, space=None, timeout_seconds=TIMEOUT_SECONDS, device='cpu'):
    import r8_common as C
    C.assert_rule()
    C.assert_threads()
    assert name in ALL_IDS, f'unknown grouping ID {name}'
    n = (n or 2048) if smoke else None
    assert smoke or n is None
    if name in FC_IDS:
        lab, rec, n = part_fc(name, smoke, labels, space)
    elif name == 'R0own':
        lab, rec, n = part_r0own(smoke, n)
    elif name == 'NRK':
        lab, rec, n = part_nrk(smoke, n, timeout_seconds)
    elif name == 'TP':
        lab, rec, n = part_tp(smoke, n, device)
    else:
        lab, rec, n = part_search(name, smoke, n)
    if n is None:
        n = C.ctx().n_paintings
    if lab is not None and lab.max() < 1:
        rec = {**rec, 'dropped': True, 'reason': 'fewer than 2 groups'}
        lab = None
    return write_part(name, lab, rec, n, smoke)


def collect(ids, smoke=False, name='r8_groupings'):
    import r8_common as C
    C.assert_rule()
    root, parts, _ = _paths(smoke)
    out = root / 'groupings'
    targets = [out / f'{name}.npz', out / f'{name}.json']
    assert all(not p.exists() or C.in_smoke(p) for p in targets), f'{name}: output exists (write once)'
    loaded, ns = {}, set()
    for i in ids:
        assert (parts / f'{i}.json').is_file(), f'{i}: no part; run `part --id {i}` first'
        rec = json.loads((parts / f'{i}.json').read_text())
        lab = None
        if (parts / f'{i}.npz').is_file():
            with np.load(parts / f'{i}.npz', allow_pickle=False) as z:
                lab = np.asarray(z['labels'])
                ns.add(int(z['n_paintings']))
        loaded[i] = {'labels': lab, 'record': rec}
        if rec.get('n_paintings') is not None:
            ns.add(int(rec['n_paintings']))
    assert len(ns) == 1, f'parts cover different painting counts: {ns}'
    st, g, _, n = subset(ns.pop() if smoke else None)
    assert smoke or n == C.ctx().n_paintings
    arrays, rec = assemble(loaded, st, g, ids)
    C.write_npz_once(targets[0], **arrays)
    return C.write_json_once(targets[1], {**rec, 'smoke': smoke, 'name': name, 'n_rows': len(st), 'n_paintings': n,
                                          'scorer_train_sha256': sha_array(st), 'g_st_sha256': sha_array(g),
                                          'part_sha256': {i: {e: C.sha_file(parts / f'{i}.{e}') for e in ('json', 'npz')
                                                              if (parts / f'{i}.{e}').is_file()} for i in ids}})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('part', help='one grouping ID')
    p.add_argument('--id', required=True, choices=ALL_IDS)
    p.add_argument('--smoke', nargs='?', const=2048, type=int, metavar='N',
                   help='first N paintings (default 2048), outputs under results/smoke/')
    p.add_argument('--labels', type=Path, help='FC*: native labels npz (default: ticket 06 paths)')
    p.add_argument('--space', type=Path, help='FC*: space file (default: ticket 06 paths)')
    p.add_argument('--timeout-seconds', type=int, default=TIMEOUT_SECONDS, help='NRK; shorter only with --smoke')
    p.add_argument('--device', default='cpu', help='TP text encoder')
    c = sub.add_parser('collect', help='write r8_groupings.npz/.json once')
    c.add_argument('--ids', nargs='+', default=[i for i in ALL_IDS if i != 'NBC'])
    c.add_argument('--name', default='r8_groupings')
    c.add_argument('--smoke', action='store_true')
    ap.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.worker)
    elif args.cmd == 'part':
        run_part(args.id, args.smoke is not None, args.smoke, args.labels, args.space, args.timeout_seconds, args.device)
    elif args.cmd == 'collect':
        collect(args.ids, args.smoke, args.name)
    else:
        ap.error('part or collect')


if __name__ == '__main__':
    main()
