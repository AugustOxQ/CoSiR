"""Stage 3 of r13: SC1 to SC5, outcomes, the verdict, SC5's claim (spec 2026-10-31 §3.4 "Test scoring" and "Outcomes", §6;
D3, D7, D14). It inherits r12's stage 2 and 3 protocol (spec 2026-10-30 §3.4, §6) and reuses r12's code unchanged:

  ac_cluster.cached_partitions / ami / paired   the photo model's partitions, its AMI, the paired bootstrap
                                                (2,000 draws: images, then seeds)
  ac_stats.arm_retrieval                        per-query AP under a picked read-out (W fitted on all validation photos)
  rt_stats.paired                               the paired bootstrap of AP (10,000 joint draws over queries and seeds)

What differs from r12 and is built here: the candidate is any seeded arm given as arrays, its r is the photo model's own
range r' (passed in), the comparators are read from r12's saved files (picks, partitions, per-query AP) instead of being
recomputed, and the table is SC1 to SC5 (r13's SC5 is r12's SC8, the CRL comparison on Affection).

  SC1 AMI  against plain                 WIN           r'_clu
  SC2 mAP  against plain                 WIN           r'_mAP    (each side under its own picked read-out)
  SC3 AMI  against the teacher           LEVEL or WIN  r'_clu
  SC4 mAP  against the teacher           LEVEL or WIN  r'_mAP
  SC5 mAP  against CRL (picked version)  WIN           r'_mAP    (its own claim; no verdict)

Nothing is written under r12's outputs (`pf_paths.R12_OUT`); `write_stage3` writes under `pf_paths.OUTPUTS` only.
"""
import json
import multiprocessing as mp
from pathlib import Path
from typing import NamedTuple

import numpy as np
from threadpoolctl import threadpool_limits

import pf_paths  # first: puts r12's round on sys.path
import ac_cluster
import ac_stats
import rs_vh
import rt_metrics
import rt_stats

SC_IDS = ('SC1', 'SC2', 'SC3', 'SC4', 'SC5')
VERDICT_IDS = ('SC1', 'SC2', 'SC3', 'SC4')
OUTCOMES = rt_stats.OUTCOMES
BARS = rt_stats.BARS                    # {'WIN': {'WIN'}, 'LEVEL_OR_WIN': {'WIN', 'LEVEL'}}

# id -> (metric, comparator, bar, which of the photo model's ranges r' applies)
SC = {
    'SC1': ('ami', 'plain', 'WIN', 'clu'),
    'SC2': ('map', 'plain', 'WIN', 'map'),
    'SC3': ('ami', 'teacher', 'LEVEL_OR_WIN', 'clu'),
    'SC4': ('map', 'teacher', 'LEVEL_OR_WIN', 'map'),
    'SC5': ('map', 'crl', 'WIN', 'map'),
}


def outcome(lower, r):
    """WIN when the lower 95% bound > 0; LEVEL when not WIN and the bound > -r (r: the photo model's range r' of the SC);
    BELOW otherwise (r10's T-WIN / T-LEVEL / T-BELOW, `rt_stats.outcome`)."""
    return rt_stats.outcome(lower, r)


def sc_pass(sc_id, outcome_):
    """True when the outcome meets the SC's bar (WIN for a WIN bar; LEVEL or WIN for a LEVEL_OR_WIN bar)."""
    assert outcome_ in OUTCOMES, outcome_
    return outcome_ in BARS[SC[sc_id][2]]


def verdict(outcomes):
    """Spec §6: 'holds' when SC1 to SC4 pass; 'qualified' when SC1 and SC2 pass and exactly one of SC3 and SC4 does;
    'fails' otherwise. outcomes: {sc_id: outcome} for exactly SC1 to SC4 (SC5 is its own claim, not part of this)."""
    assert set(outcomes) == set(VERDICT_IDS), f'need exactly {VERDICT_IDS}, got {sorted(outcomes)}'
    ok = {i: sc_pass(i, outcomes[i]) for i in VERDICT_IDS}
    if all(ok.values()):
        return 'holds'
    if ok['SC1'] and ok['SC2'] and ok['SC3'] != ok['SC4']:
        return 'qualified'
    return 'fails'


# ---- the comparators: r12's outputs, read and checked, never recomputed or written ---------------------------------------

def comparator_arms(picks):
    """{'plain': 'plain', 'teacher': teacher_<wording>, 'crl': crl_<version>} from r12's `stage1/picks.json['affection']`.
    SC1 and SC3 use the clustering wording pick, SC2 and SC4 the retrieval one (both 'artwork' in r12; checked equal
    here because the teacher row of the document is one arm)."""
    wording = picks['teacher_wording']
    assert wording['clustering'] == wording['retrieval'], f'teacher wording differs between tasks: {wording}'
    return {'plain': 'plain', 'teacher': f'teacher_{wording["retrieval"]}', 'crl': f'crl_{picks["crl_version"]}'}


def _load_ap(path, n_rows=None):
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['ap', 'rows'], (path, z.files)
        ap, rows = z['ap'], z['rows']
    assert ap.dtype == np.float64 and rows.dtype == np.int64 and ap.ndim == 1 and ap.shape == rows.shape, (path, ap.shape)
    assert np.isfinite(ap).all() and len(np.unique(rows)) == len(rows), path
    assert n_rows is None or len(rows) <= n_rows, (path, len(rows), n_rows)
    return ap, rows


def _load_partition(r12, arm, n):
    """r12's saved test partitions of an unseeded arm (int32 [3, n]), checked against the sidecar's file hash and size."""
    path = r12 / 'partitions' / f'affection__test__{arm}__s-.npz'
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['partitions_sha256'] == rs_vh.file_sha(path), f'{path}: partition file changed'
    assert side['shape'][0] == n, (path, side['shape'], n)
    with np.load(path, allow_pickle=False) as z:
        assert z.files == ['labels'], (path, z.files)
        labels = z['labels']
    assert labels.dtype == np.int32 and labels.shape == (3, n), (path, labels.dtype, labels.shape)
    return labels


def load_comparators(picks, n_test, r12=None):
    """The three comparators of SC1 to SC5 from r12's saved outputs under `r12` (default `pf_paths.R12_OUT`, read only).
    picks: r12's `stage1/picks.json['affection']`. Returns {'plain'|'teacher'|'crl': {arm, readout, ap, rows, point,
    labels}}: `ap`, `rows` from `stage2/perquery/affection__<arm>__emotion__s-.npz` (CRL's from `stage3/perquery`, and
    checked equal to its stage 2 file); `point` = ap.mean() (r12's `arm_retrieval` point of an unseeded arm); `labels`
    the arm's r12 test partitions, hash-checked against their JSON (None for CRL: it is only an SC5 comparator).
    AP means must match r12's SC2/SC4/SC8 comparator points; partition AMI must match its saved arm row.
    """
    import ac_data
    r12 = Path(pf_paths.R12_OUT if r12 is None else r12)
    ref2 = json.loads((r12 / 'stage2/verdict.json').read_text())
    ref3 = json.loads((r12 / 'stage3/verdict.json').read_text())
    labels = ac_data.labels('test')
    out = {}
    for key, arm in comparator_arms(picks).items():
        stage = 3 if key == 'crl' else 2
        ap, rows = _load_ap(r12 / f'stage{stage}/perquery/affection__{arm}__emotion__s-.npz', n_test)
        if key == 'crl':
            ap2, rows2 = _load_ap(r12 / f'stage2/perquery/affection__{arm}__emotion__s-.npz', n_test)
            assert np.array_equal(ap, ap2) and np.array_equal(rows, rows2), f'{arm}: AP differs between stages 2 and 3'
        out[key] = dict(arm=arm, readout=picks['readout'][arm], ap=ap, rows=rows, point=float(ap.mean()),
                        labels=None if key == 'crl' else _load_partition(r12, arm, n_test))
        sid = {'plain': 'SC2', 'teacher': 'SC4', 'crl': 'SC8'}[key]
        reference = ref3 if key == 'crl' else ref2
        assert abs(out[key]['point'] - reference['sc'][sid]['point_t']) <= pf_paths.TOL['ii'], \
            f'{arm}: AP point differs from r12 {sid}'
        if out[key]['labels'] is not None:
            assert abs(ac_cluster.ami(out[key]['labels'], labels) - ref2['rows'][arm]['ami']) <= pf_paths.TOL['ii'], \
                f'{arm}: partition AMI differs from r12 row'
    return out


# ---- the candidate and its scores ---------------------------------------------------------------------------------------

class Candidate(NamedTuple):
    """A seeded arm as arrays, in `ac_data.keys('test')` / `keys('val')` order, rows of seed s at index s.
    test_stored: float16 [5, n, d] as saved (clustering input; `ac_data.stored`); test_embed: float32 [5, n, d] unit
    (retrieval input; `ac_data.embed`, i.e. the stored rows cast to float32 and normalized); val_embed: float32
    [5, nv, d] unit (the read-out is fitted on all validation photos). readout: the committed pick, e.g. 'center+k10'."""
    name: str
    readout: str
    test_stored: np.ndarray
    test_embed: np.ndarray
    val_embed: np.ndarray


def _check_candidate(cand):
    assert cand.test_stored.dtype == np.float16 and cand.test_embed.dtype == np.float32 \
        and cand.val_embed.dtype == np.float32, 'candidate dtypes'
    assert cand.test_stored.ndim == 3 and cand.test_stored.shape[0] == 5, cand.test_stored.shape
    assert cand.test_embed.shape == cand.test_stored.shape, 'candidate test shapes'
    assert cand.val_embed.ndim == 3 and cand.val_embed.shape[0] == 5 and cand.val_embed.shape[2] == cand.test_stored.shape[2]
    for name, x in (('test_embed', cand.test_embed), ('val_embed', cand.val_embed)):
        assert np.isfinite(x).all() and np.abs(np.linalg.norm(x, axis=2) - 1).max() < 1e-3, f'{name} is not unit'
    f = cand.test_stored.astype(np.float32)
    again = f / np.linalg.norm(f, axis=2, keepdims=True)
    assert np.abs(again - cand.test_embed).max() < 1e-6, 'test_embed is not the normalized stored rows'


def _fit_one(args):
    x, name, root = args
    with threadpool_limits(1):
        return ac_cluster.cached_partitions(x, name, root)


def candidate_partitions(cand, root, workers=8):
    """int32 [5, 3, n]: r10's partitions of each seed's stored rows, cached under `root` as
    `affection__test__<name>__s<seed>.npz` (root must not be under r12's outputs)."""
    root = Path(root)
    assert pf_paths.R12_OUT.resolve() not in (root.resolve(), *root.resolve().parents), f'{root} is under r12 outputs'
    tasks = [(cand.test_stored[s], f'affection__test__{cand.name}__s{s}', root) for s in range(5)]
    with mp.get_context('fork').Pool(min(workers, 5)) as pool:
        return np.stack(pool.map(_fit_one, tasks, chunksize=1))


def candidate_retrieval(cand, readout, labels, workers=8):
    """`ac_stats.arm_retrieval` on the test photos under `readout`: {'ap': [5, nq], 'rows', 'dropped', 'point',
    'per_seed'} (W fitted on all validation photos, k-query expansion on the test gallery)."""
    res = ac_stats.arm_retrieval(lambda s: cand.val_embed[s], lambda s: cand.test_embed[s], readout, labels,
                                 seeds=ac_stats.SEEDS, workers=workers)
    assert res['dropped'] == 0, (readout, 'queries without a relevant item', res['dropped'])
    return res


def _entry(sc_id, cand, comp, point_c, point_t, d, interval, r):
    """One `sc` entry, in r12's verdict form (`ac_stats.sc_entry`); candidate is this arm's name."""
    metric, _, _, _ = SC[sc_id]
    interval = [float(interval[0]), float(interval[1])]
    out = outcome(interval[0], r)
    clustering = metric == 'ami'
    return {'candidate': cand.name, 'comparator': comp['arm'], 'dataset': 'affection', 'aspect': 'emotion',
            'metric': metric, 'readout_c': None if clustering else cand.readout,
            'readout_t': None if clustering else comp['readout'],
            'point_c': float(point_c), 'point_t': float(point_t), 'd': float(d), 'interval': interval,
            'r': float(r), 'outcome': out, 'pass': bool(sc_pass(sc_id, out))}


def _check_r(r):
    for k in ('clu', 'map'):
        assert np.isfinite(r[k]) and r[k] > 0, f'seed range r\'_{k} = {r[k]} must be finite and positive'


def score(cand, r, labels, dist, comparators, picks, partition_root, draws_clu=2000, draws_ret=10000, workers=8,
          frozen_rows=None):
    """SC1 to SC5 for a seeded candidate on the test photos, the verdict and SC5's claim.
    r: {'clu': r'_clu, 'map': r'_mAP} (the candidate's own seed ranges); labels: int64 [n] (`ac_data.labels('test')`);
    dist: float64 [n, 9] (`ac_data.emotion_dist('test')`, for the nDCG diagnostic row); comparators: `load_comparators`;
    picks: r12's `picks['affection']`; partition_root: where the candidate's partitions are cached (not r12's);
    frozen_rows: {arm: row} copied into the document's rows (r12's rows of plain, teacher, CRL and frozen c*).
    Clustering: `ac_cluster.paired` (draws_clu draws, images then seeds) against the comparator's r12 partitions;
    retrieval: `rt_stats.paired` (draws_ret joint draws over queries and seeds) on the two arms' per-query AP, the rows
    of the two checked equal. Returns {'doc': the verdict document, 'perquery': {'photo__s0': (ap, rows), ...,
    'photo_R0__s4': ...}, 'partitions': int32 [5, 3, n]}."""
    _check_candidate(cand)
    _check_r(r)
    labels = np.asarray(labels)
    n = len(labels)
    assert cand.test_stored.shape[1] == n, ('candidate rows', cand.test_stored.shape, n)
    parts = candidate_partitions(cand, partition_root, workers)
    picked = candidate_retrieval(cand, cand.readout, labels, workers)
    r0 = candidate_retrieval(cand, 'none+k0', labels, workers)
    ami = ac_cluster.ami(parts, labels)
    sc = {}
    for sid, (metric, which, _, rkey) in SC.items():
        comp = comparators[which]
        if metric == 'ami':
            res = ac_cluster.paired(parts, comp['labels'], labels, draws_clu, workers)
            sc[sid] = _entry(sid, cand, comp, res['A_cand'], res['A_comp'], res['d'], res['interval'], r[rkey])
        else:
            assert np.array_equal(picked['rows'], comp['rows']), (sid, 'the two arms were scored on different queries')
            p = rt_stats.paired(picked['ap'], comp['ap'], draws_ret)
            sc[sid] = _entry(sid, cand, comp, picked['point'], comp['point'], p['d'], p['interval'], r[rkey])
    nd = [float(rt_metrics.graded(cand.test_embed[s], dist, workers=workers).mean()) for s in range(5)]
    rows = {cand.name: dict(ami=ami, readout=cand.readout, map_picked=dict(map=picked['point'], per_seed=picked['per_seed']),
                            map_r0=dict(map=r0['point'], per_seed=r0['per_seed']), ndcg_r0=float(np.mean(nd)))}
    assert cand.name not in (frozen_rows or {}), 'a frozen row would overwrite the candidate\'s'
    rows.update(frozen_rows or {})
    outcomes = {sid: e['outcome'] for sid, e in sc.items() if sid in VERDICT_IDS}
    doc = {'sc': sc, 'verdict': verdict(outcomes), 'claims': claims(sc), 'rows': rows, 'r': {k: float(r[k]) for k in ('clu', 'map')},
           'smoke': pf_paths.SMOKE, 'draws_clu': draws_clu, 'draws_ret': draws_ret}
    doc = json.loads(json.dumps(doc, allow_nan=False))
    perquery = {}
    for suffix, res in (('', picked), ('_R0', r0)):
        for s in range(5):
            perquery[f'{cand.name}{suffix}__s{s}'] = (res['ap'][s], res['rows'])
    return {'doc': doc, 'perquery': perquery, 'partitions': parts}


def claims(sc):
    """{'SC5': bool}: the CRL bar is its own claim; a failed one gives no claim and keeps its entry and row."""
    return {'SC5': bool(sc['SC5']['pass'])}


# ---- real data: r12's rows, writing, the stage, gate (ii) ----------------------------------------------------------------

def frozen_rows(picks, r12=None):
    """r12's rows of plain, the teacher, CRL and frozen c* for the verdict document, copied unchanged: plain, teacher and
    c* from `stage2/verdict.json['rows']`, CRL's from `stage3/verdict.json['rows']['affection']`."""
    r12 = Path(pf_paths.R12_OUT if r12 is None else r12)
    stage2 = json.loads((r12 / 'stage2/verdict.json').read_text())
    stage3 = json.loads((r12 / 'stage3/verdict.json').read_text())
    arms = comparator_arms(picks)
    rows = {arm: stage2['rows'][arm] for arm in (arms['plain'], arms['teacher'], 'cstar')}
    rows[arms['crl']] = stage3['rows']['affection'][arms['crl']]
    return rows


def write_stage3(result, out=None):
    """Write `<out>/verdict.json` and `<out>/perquery/affection__<arm>__emotion__s<seed>.npz` (`ap`, `rows`; r12's naming,
    arm `photo` and `photo_R0`). out defaults to `pf_paths.OUTPUTS/stage3`; never under r12's outputs."""
    out = Path(pf_paths.OUTPUTS / 'stage3' if out is None else out)
    assert pf_paths.R12_OUT.resolve() not in (out.resolve(), *out.resolve().parents), f'{out} is under r12 outputs'
    (out / 'perquery').mkdir(parents=True, exist_ok=True)
    for key, (ap, rows) in result['perquery'].items():
        arm, seed = key.split('__')
        np.savez(out / 'perquery' / f'affection__{arm}__emotion__{seed}.npz', ap=ap, rows=rows)
    (out / 'verdict.json').write_text(json.dumps(result['doc'], indent=1, allow_nan=False) + '\n')
    return out / 'verdict.json'


def candidate_from_data(name, readout, loader):
    """A `Candidate` from a loader: loader.stored(split, seed) -> float16 rows, loader.embed(split, seed) -> float32 unit
    rows, split 'val' or 'test' (r12's `ac_data` for c*; ticket 06's `pf_export` for the photo model, bound to a run set)."""
    return Candidate(name=name, readout=readout,
                     test_stored=np.stack([loader.stored('test', s) for s in range(5)]),
                     test_embed=np.stack([loader.embed('test', s) for s in range(5)]),
                     val_embed=np.stack([loader.embed('val', s) for s in range(5)]))


def stage3(cand, r, workers=8, draws_clu=2000, draws_ret=10000, partition_root=None, r12=None, frozen=True):
    """The stage 3 scoring on the real test photos and r12's comparators; returns `score`'s result (the runner writes
    it with `write_stage3`). r: the candidate's r' ({'clu', 'map'}); picks: r12's committed picks; frozen=False leaves
    r12's rows out of the document (the replay, where the candidate is r12's c* itself)."""
    import ac_data
    r12 = Path(pf_paths.R12_OUT if r12 is None else r12)
    picks = json.loads((r12 / 'stage1/picks.json').read_text())['affection']
    labels = ac_data.labels('test')
    comps = load_comparators(picks, len(labels), r12)
    return score(cand, r, labels, ac_data.emotion_dist('test'), comps, picks,
                 partition_root=pf_paths.OUTPUTS / 'partitions' if partition_root is None else partition_root,
                 draws_clu=draws_clu, draws_ret=draws_ret, workers=workers, frozen_rows=frozen_rows(picks, r12) if frozen else None)


# Gate (ii), SC part (spec §3.5 0a(ii)): r12's c* through this module must give r12's stage 2 SC1 to SC4 and stage 3 SC8.
REPLAY_PAIRS = (('SC1', 'SC1'), ('SC2', 'SC2'), ('SC3', 'SC3'), ('SC4', 'SC4'), ('SC5', 'SC8'))   # (mine, r12's)
REPLAY_NUMBERS = ('point_c', 'point_t', 'd', 'r')


def compare_entry(mine, ref, tol):
    """Differences between a replayed `sc` entry and r12's: the numbers and interval ends within tol, the rest equal.
    Returns {'max_abs_diff': float, 'same': bool (outcome, pass, names, read-outs), 'ok': bool}."""
    diffs = [abs(mine[k] - ref[k]) for k in REPLAY_NUMBERS] + [abs(a - b) for a, b in zip(mine['interval'], ref['interval'])]
    same = all(mine[k] == ref[k] for k in ('candidate', 'comparator', 'dataset', 'aspect', 'metric', 'readout_c',
                                           'readout_t', 'outcome', 'pass'))
    worst = float(max(diffs))
    return {'max_abs_diff': worst, 'same': bool(same), 'ok': bool(same and worst <= tol)}


def gate_ii_sc(workers=8, draws_clu=2000, draws_ret=10000, out=None, r12=None):
    """Spec §3.5 0a(ii), SC part: r12's frozen c* test exports (`ac_data.stored/embed`, the five seeds), its picks and its
    ranges go through `score`; the five entries must reproduce r12's SC1 to SC4 (`stage2/verdict.json`) and SC8
    (`stage3/verdict.json`, r13's SC5 comparison): point values, d, r and interval ends within 1e-9, same outcomes.
    Also checked: c*'s refitted partitions equal r12's saved ones, its per-query AP equals r12's `stage2/perquery`
    within 1e-9, and its row (AMI, mAP, R0 mAP, nDCG) equals r12's. Writes `OUTPUTS/stage0a/sc_replay.json`
    ({"ok", "items": {name: {"value", "reference", "ok"}}, ...}) and returns it. Slow: the bootstraps (draws_clu,
    draws_ret) dominate; draws below the defaults give a plumbing check only (the result says so)."""
    import ac_data
    r12 = Path(pf_paths.R12_OUT if r12 is None else r12)
    tol = pf_paths.TOL['ii']
    out = Path(pf_paths.OUTPUTS / 'stage0a' if out is None else out)
    picks = json.loads((r12 / 'stage1/picks.json').read_text())['affection']
    ranges = ac_stats.load_ranges(r12 / 'stage0b/ranges.json')
    r = {'clu': ranges['clu']['r'], 'map': ranges['map']['r']}
    cand = candidate_from_data('cstar', picks['readout']['cstar'], _CstarLoader)
    res = stage3(cand, r, workers, draws_clu, draws_ret, partition_root=out / 'sc_replay_partitions', r12=r12, frozen=False)
    ref2 = json.loads((r12 / 'stage2/verdict.json').read_text())
    ref3 = json.loads((r12 / 'stage3/verdict.json').read_text())
    items = {}
    for mine, theirs in REPLAY_PAIRS:
        ref = (ref2 if theirs in ref2['sc'] else ref3)['sc'][theirs]
        c = compare_entry(res['doc']['sc'][mine], ref, tol)
        items[f'{mine}={theirs}'] = dict(value=res['doc']['sc'][mine], reference=ref, **c)
    same_parts = True
    for s in range(5):
        with np.load(r12 / f'partitions/affection__test__cstar__s{s}.npz', allow_pickle=False) as z:
            same_parts &= bool(np.array_equal(z['labels'], res['partitions'][s]))
    items['partitions'] = dict(value=None, reference='r12 partitions/affection__test__cstar__s{0..4}.npz', ok=same_parts)
    ap_diff = 0.0
    for s in range(5):
        with np.load(r12 / f'stage2/perquery/affection__cstar__emotion__s{s}.npz', allow_pickle=False) as z:
            ap, rows = res['perquery'][f'cstar__s{s}']
            assert np.array_equal(z['rows'], rows), ('per-query rows', s)
            ap_diff = max(ap_diff, float(np.abs(z['ap'] - ap).max()))
    items['perquery'] = dict(value=ap_diff, reference=0.0, ok=ap_diff <= tol)
    row, ref_row = res['doc']['rows']['cstar'], ref2['rows']['cstar']
    row_diff = max(abs(row['ami'] - ref_row['ami']), abs(row['ndcg_r0'] - ref_row['ndcg_r0']),
                   abs(row['map_picked']['map'] - ref_row['map_picked']['map']),
                   abs(row['map_r0']['map'] - ref_row['map_r0']['map']),
                   *(abs(a - b) for a, b in zip(row['map_picked']['per_seed'], ref_row['map_picked']['per_seed'])))
    items['row'] = dict(value=row_diff, reference=0.0, ok=row_diff <= tol)
    doc = {'ok': all(i['ok'] for i in items.values()), 'tol': tol, 'draws_clu': draws_clu, 'draws_ret': draws_ret,
           'full_draws': (draws_clu, draws_ret) == (2000, 10000), 'items': items}
    doc = json.loads(json.dumps(doc, allow_nan=False))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sc_replay.json').write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    return doc


class _CstarLoader:
    """r12's frozen c* exports for `candidate_from_data` (emotion head)."""

    @staticmethod
    def stored(split, seed):
        import ac_data
        return ac_data.stored('cstar', split, seed)

    @staticmethod
    def embed(split, seed):
        import ac_data
        return ac_data.embed('cstar', split, seed)
