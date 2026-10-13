"""Rule §8's descriptive scopes; no verdict, fit, pick, or new statistic.

Metrics are the existing per_anchor metrics, including each direction by passing that
same direction in both slots. Points/intervals use r9_stats' cluster bootstrap. All
scopes retain anchor-painting clusters and the held runner's seed/episode order.
"""
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R
import r9_score as S
import r9_stats as ST
import numpy as np
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor

AFF = 'aff_fused'
ROW_METRICS = ('r1', 'gain', 'swap')


class BoundaryError(AssertionError):
    """An owned, number-free file boundary name, safe for the runner's console."""


def require(ok, boundary):
    if not ok:
        raise BoundaryError(boundary)


def absent(reason):
    return {'status': 'absent', 'reason': reason}


def load_arrays(path, seeds, n_per_pair):
    """Strict primary-only held layout, including the saved raw gates/reader inputs."""
    n_seed = len(R.PAIRS) * n_per_pair
    n = len(seeds) * n_seed
    keys = {f'{s}__{m}' for s in S.PRIMARY_SCORERS for m in METRICS}
    keys |= {'cl', 'pair_index', 'seed_index'}
    keys |= {f'{prefix}__{c}' for prefix in ('aff_gate', 'P', 'margin', 'pick') for c in CONDITIONS}
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == keys, 'arrays keys')
        arrays = {k: z[k] for k in z.files}
    for k, x in arrays.items():
        shape = (S.N_TAU, n) if k.startswith('aff_gate__') else (n, len(R.R3.A0)) if k.startswith('P__') else (n,)
        require(x.shape == shape and np.isfinite(x).all(), 'arrays shape/finite: ' + k)
        dtype = np.int64 if k in ('cl', 'pair_index', 'seed_index') or k.startswith('pick__') else np.float32 if k.startswith('aff_gate__') else np.float64
        require(x.dtype == dtype, 'arrays dtype: ' + k)
    require(np.array_equal(arrays['seed_index'], np.repeat(np.arange(len(seeds)), n_seed)), 'arrays seed order')
    pi = np.repeat(np.arange(len(R.PAIRS)), n_per_pair)
    out = []
    for i, seed in enumerate(seeds):
        sl = slice(i * n_seed, (i + 1) * n_seed)
        require(np.array_equal(arrays['pair_index'][sl], pi), 'arrays pair order')
        core = {s: {m: arrays[f'{s}__{m}'][sl] for m in METRICS} for s in S.PRIMARY_SCORERS}
        core.update({k: arrays[k][sl] for k in ('cl', 'pair_index')})
        out.append(core)
    return arrays, out


def check_pass(core, record, a28):
    """Reproduce every statistical field, including both Holm levels, before describing."""
    extra = {k: record[k] for k in ('episodes_sha256', 'runner_sha256', 'module_sha256')}
    if a28:
        got = ST.pass_record_a28(core, 'smoke' if record['mode'] == 'smoke' else 'a28', record['seeds'], extra,
                                 **{k: record[k] for k in ('set_aside', 'flagged_languages', 'near_duplicates')})
    else:
        got = ST.pass_record(core, record['mode'], record['seeds'], extra, secondary=False)
    for k, value in got.items():
        if k != 'time':
            require(value == record.get(k), 'arrays reproduce pass: ' + k)
    return {c: True for c in ST.CHECKS}


def direction_rows():
    """Return a score_seed callback and its collected direction metrics."""
    rows = {}

    def collect(name, scores):
        require(name not in rows, 'direction scorer repeated')
        rows[name] = {d: per_anchor({c: {other: scores[c][d] for other in DIRECTIONS} for c in CONDITIONS})
                      for d in DIRECTIONS}
    return rows, collect


@lru_cache(maxsize=128)
def _point_ci_bytes(values, clusters):
    # Repeated condition-free zero gains and identical scopes reuse the exact
    # existing computation. Bound memory, and key by bytes rather than a hash.
    values = np.frombuffer(values, dtype=np.float64)
    cl = np.frombuffer(clusters, dtype=np.int64)
    # As in R6's descriptive rows: no n_j is needed here. The pass reproduction
    # above uses bootstrap_draws and its integer counts; this is its same
    # anchor-cluster point/interval implementation without computing counts twice.
    return ST.C.point_ci(values, cl)


def point_ci(values, cl):
    return _point_ci_bytes(np.asarray(values, dtype=np.float64).tobytes(),
                           np.asarray(cl, dtype=np.int64).tobytes())


def row(values, aff, cl, with_minus=True):
    out = {'n_episodes': int(len(cl)), 'n_paintings': int(len(np.unique(cl)))}
    if not len(cl) or len(np.unique(cl)) < 2:
        return {**out, **absent('fewer than two anchor paintings; no interval')}
    require(all(np.asarray(x).shape == cl.shape for x in [*values.values(), *aff.values()]), 'row alignment')
    out.update({m: point_ci(values[m], cl) for m in ROW_METRICS})
    if with_minus:
        out['aff_minus'] = {m: point_ci(aff[m] - values[m], cl) for m in ROW_METRICS}
    return out


def pooled(inputs):
    require(bool(inputs), 'no descriptive seed inputs')
    return {'cl': np.concatenate([x['cl'] for x in inputs]),
            'pair_index': np.concatenate([x['pair_index'] for x in inputs]),
            'pa': {s: {m: np.concatenate([x['pa'][s][m] for x in inputs]) for m in ROW_METRICS}
                   for s in inputs[0]['pa']}}


def rows_for_scope(data, mask=None):
    mask = np.ones(len(data['cl']), dtype=bool) if mask is None else np.asarray(mask)
    require(mask.dtype == bool and mask.shape == data['cl'].shape, 'scope mask')
    cl = data['cl'][mask]
    aff = {m: x[mask] for m, x in data['pa'][AFF].items() if m in ROW_METRICS}
    return {s: row({m: x[mask] for m, x in values.items() if m in ROW_METRICS}, aff, cl, s != AFF)
            for s, values in data['pa'].items()}


def scope(inputs, masks=None):
    """A subset from each seed's own episodes, without rebuilding or reweighting."""
    if masks is not None:
        require(len(masks) == len(inputs), 'subset seed alignment')
        inputs = [{**x, 'cl': x['cl'][msk], 'pair_index': x['pair_index'][msk],
                   'pa': {s: {m: v[msk] for m, v in pa.items()} for s, pa in x['pa'].items()}}
                  for x, msk in zip(inputs, masks)]
    data = pooled(inputs)
    return {'pooled': rows_for_scope(data),
            'per_pair': {p: rows_for_scope(data, data['pair_index'] == i) for i, p in enumerate(R.PAIR_NAMES)}}


def describe(inputs, cells):
    data = pooled(inputs)
    out = scope(inputs)
    out['per_seed'] = {str(x['seed']): scope([x]) for x in inputs}
    out['per_direction'] = {}
    for d in DIRECTIONS:
        ds = [{**x, 'pa': {s: dirs[d] for s, dirs in x['directions'].items()}} for x in inputs]
        out['per_direction'][d] = {**scope(ds), 'per_seed': {str(x['seed']): scope([x]) for x in ds}}
    gates = {c: np.concatenate([x['gates']['aff'][c] for x in inputs], axis=1) for c in CONDITIONS}
    pi = data['pair_index']
    all_taus = [{c: gates[c][t] for c in CONDITIONS} for t in range(S.N_TAU)]
    # The chosen tune-half cell scores parity 1-h, just as score_seed does.
    chosen = []
    for x in inputs:
        parity = np.arange(len(x['cl'])) % 2
        tau = np.array([R.F2.cell_values(cells['aff']['fused'][1 - int(p)])[1] for p in parity], dtype=int)
        chosen.append({c: x['gates']['aff'][c][tau, np.arange(len(parity))] for c in CONDITIONS})
    chosen = {c: np.concatenate([g[c] for g in chosen]) for c in CONDITIONS}
    picks = {c: np.concatenate([x['reader']['pick'][c] for x in inputs]) for c in CONDITIONS}
    return {'rows': out, 'gate_share': {'all_taus': R.RF.open_shares(all_taus, pi),
                                      'chosen_cells': R.RF.open_shares([chosen], pi)['tau_0']},
            'reader_affect_pick_rate': R.C.pick_shares(picks, R.R3.A0, pi),
            'reported_scorers': {'B1': 'B′(A1)', 'r1_fused': 'R1', 'r1_cf': 'R1 counterpart',
                                 **{s: 'PM ' + s for s in S.PM_SCORERS}}}


def compare_backbones(current, other):
    a, b = pooled(current), pooled(other)
    require(np.array_equal(a['cl'], b['cl']) and np.array_equal(a['pair_index'], b['pair_index']), 'paired anchor alignment')
    # Caller must establish equal episode hashes BEFORE reaching this calculation.
    def one(mask):
        r = row({m: b['pa'][AFF][m][mask] for m in ROW_METRICS},
                {m: a['pa'][AFF][m][mask] for m in ROW_METRICS}, a['cl'][mask])
        return {**{k: v for k, v in r.items() if k not in ROW_METRICS and k != 'aff_minus'},
                'other_aff': {m: r[m] for m in ROW_METRICS if m in r}, 'aff_minus': r.get('aff_minus', {})}
    return {'labels': {'other_aff': "AFF on the paired other read",
                       'aff_minus': "This read's AFF minus the other read's AFF, paired by episode"},
            'pooled': one(np.ones(len(a['cl']), dtype=bool)),
            'per_pair': {p: one(a['pair_index'] == i) for i, p in enumerate(R.PAIR_NAMES)}}


def emotion_distribution(episodes, emotion):
    anchor = np.concatenate([ep.pooled.anchor for ep in episodes])
    pi = np.concatenate([ep.pair_index for ep in episodes])
    require((anchor >= 0).all() and (anchor < len(emotion)).all(), 'emotion anchor row join')
    def one(mask):
        values, counts = np.unique(np.asarray(emotion)[anchor[mask]], return_counts=True)
        total = int(counts.sum())
        return {'n_episodes': total, 'counts': {str(v): int(n) for v, n in zip(values, counts)},
                'shares': {str(v): int(n) / total for v, n in zip(values, counts)}}
    return {'pooled': one(np.ones(len(anchor), dtype=bool)),
            'per_pair': {p: one(pi == i) for i, p in enumerate(R.PAIR_NAMES)}}
