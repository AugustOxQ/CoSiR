"""r17 ticket 06: stage 0 gate (i) replays of the clustering scorer (spec §3.5 (i) a to d).

Each function feeds the stored inputs of an older round, in that round's order and through its loaders, to `ns_clu`
(and `ns_metrics` for k-means), and compares with the stored values. Each returns
  {'pass': bool, 'max_abs_diff': float, 'wiring_ok': bool, 'draws': int, 'complete': bool, 'items': [...]}
where an item is {name, source: 'path::key', ours, theirs, abs_diff, ok, kind}. `draws` below 2,000 compares the point
values only (the stored intervals come from 2,000 draws); `pass` needs the full draws, every item within `TOL_REPLAY` and
no restriction (aspects, seeds, arms or encoders left out sets `complete` False). A partition check counts the differing
entries as its abs_diff.

  replay_a_r14(draws, aspects)       r14 SigLIP 2 ArtELingo SC1 and SC6 AMI values and intervals (stored partitions)
  replay_a_r15(draws, aspects)       r15 CLIP L/14 MMArt SC1 and SC4 (partitions recomputed by `ns_metrics.partitions`)
  replay_a_partitions(aspects, seeds) `ns_metrics.partitions(x, 20, 'artelingo')` equals r10's stage 4 c* held partitions
  replay_b(draws, aspects)           r14 SigLIP 2 ArtELingo SC3 (two readings, `b_matched`)
  replay_c_affection(draws)          r14 SigLIP 2 Affection SC4, SC6
  replay_c_emoset(draws)             r14 EmoSet SC7, SC9
  replay_d_emoset(arms, workers)     r14 EmoSet K = 8 (`sheet_emoset.json`)
  replay_d_class_count(encs, aspects, seeds)  r15's class-count diagnostic (MMArt test, K = 9 and 10)
  gate_i(draws)                      all of them, keys a, b, c, d
CLI: python ns_clu_replay.py {a|b|c|d_emoset|d_class_count|all} [--draws N] [--workers W] [--out DIR]
"""
import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import ns_clu  # noqa: E402
import ns_metrics  # noqa: E402
import rs_vh  # noqa: E402
import rg_held  # noqa: E402
import bs_artclu  # noqa: E402
import ac_data  # noqa: E402

FULL = ns_paths.N_DRAWS['clu']
TOL = ns_paths.TOL_REPLAY
ASPECTS = tuple(ns_paths.ASPECTS)
SEEDS = tuple(ns_paths.SEEDS)
O14, O15, O10, R12 = ns_paths.R14_OUT, ns_paths.R15_OUT, ns_paths.R10_OUT, ns_paths.R12_OUT


# ------------------------------------------------------------------------------------------------ items and results
def _item(name, source, ours, theirs, kind='value'):
    diff = float(abs(ours - theirs))
    return dict(name=name, source=str(source), kind=kind, ours=float(ours), theirs=float(theirs), abs_diff=diff,
                ok=bool(diff <= TOL))


def _same(name, source, ours, theirs, kind='label'):
    ok = bool(ours == theirs)
    return dict(name=name, source=str(source), kind=kind, ours=str(ours), theirs=str(theirs), abs_diff=0.0 if ok else 1.0,
                ok=ok)


def _result(items, draws, complete=True):
    assert items, 'a replay with no item'
    worst = max(i['abs_diff'] for i in items)
    ok = all(i['ok'] for i in items)
    return dict(**{'pass': bool(ok and draws == FULL and complete)}, max_abs_diff=float(worst), wiring_ok=bool(ok),
                draws=int(draws), complete=bool(complete), items=items)


def _merge(*results):
    items = [i for r in results for i in r['items']]
    return _result(items, min(r['draws'] for r in results), all(r['complete'] for r in results))


def _cell_items(tag, cell, stored, path, draws, comp_reading=None):
    """Compare one `paired_cell`-style result {cand, comp, d, lo, hi} (AMI) with a stored cell dict."""
    items = []
    pairs = [('cand', 'candidate_value'), ('comp', 'comparator_value'), ('d', 'd')]
    if draws == FULL:
        pairs += [('lo', 'lo'), ('hi', 'hi')]
    for ours_key, key in pairs:
        items.append(_item(f'{tag}/{key}', f'{path}::{stored["sc"]}/{stored["encoder"]}/{stored["aspect"]}/{key}',
                           cell[ours_key], stored[key]))
    return items


def _stored_cells(path, **match):
    cells = json.loads(Path(path).read_text())
    assert isinstance(cells, list)
    return {c['sc'] + '/' + c['aspect']: c for c in cells if all(c.get(k) == v for k, v in match.items())}


def _ami_cell(cand, comp, y, k, draws):
    return ns_clu.paired_cell(cand, comp, y, k, metrics=('ami',), draws=draws)['ami']


def _read_parts(path, n):
    """A stored partition file written by r12/r13/r14 (key `labels`, int32 [3, n]; sidecar `partitions_sha256`), read only."""
    path = Path(path)
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['partitions_sha256'] == rs_vh.file_sha(path), f'{path}: partition file differs from its sidecar'
    with np.load(path, allow_pickle=False) as z:
        assert z.files == ['labels'], path
        labels = z['labels']
    assert labels.shape == (3, n) and np.issubdtype(labels.dtype, np.integer), (path, labels.shape, labels.dtype)
    return labels


# ------------------------------------------------------------------------------------------------ (a) paintings
def replay_a_r14(draws=FULL, aspects=ASPECTS):
    """r14 SigLIP 2 ArtELingo: SC1 (c* against the frozen image) and SC6 (against the teacher) AMI values and intervals,
    from r14's stored partitions (`O14/stage3/partitions/{cstar,frozen}/siglip2/`, r10/r14's stored teacher)."""
    names, _sha, labels = bs_artclu.held_setup()
    n = len(names)
    cells_path = O14 / 'stage3/cells_artelingo_clu.json'
    stored = _stored_cells(cells_path, encoder='siglip2')
    root = O14 / 'stage3/partitions'
    frozen = rg_held.read_frozen_partitions(root / 'frozen/siglip2/frozen__s0__r20.npz', n)
    items = []
    for a in aspects:
        cand = np.stack([rg_held.read_frozen_partitions(root / f'cstar/siglip2/{a}__s{s}__r20.npz', n) for s in SEEDS])
        teacher = bs_artclu.stored_teacher(bs_artclu.TEACHER_ROOT, a, n)[0]['r20']
        for sc, comp in (('SC1', frozen), ('SC6', teacher)):
            cell = _ami_cell(cand, comp, labels[a], ns_paths.K20, draws)
            items += _cell_items(f'r14/siglip2/artelingo/{a}/{sc}', cell, stored[f'{sc}/{a}'], cells_path, draws)
    return _result(items, draws, tuple(aspects) == ASPECTS)


def _unit32(x):
    x = np.asarray(x).astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(norm).all() and (norm > 0).all()
    return (x / norm).astype(np.float32)


def replay_a_r15(draws=FULL, aspects=ASPECTS, enc='clip_l14'):
    """r15 CLIP L/14 MMArt test: SC1 (c* against the frozen image) and SC4 (against the teacher) AMI values and intervals.
    The partitions are recomputed with `ns_metrics.partitions` (8 threads) from r15's exports and features, in r15's
    order (`mt_clu.score_encoder`'s loaders)."""
    import importlib
    import mt_clu
    import mt_paths
    md, ex = importlib.import_module('mt_data'), importlib.import_module('mt_export')
    keys = mt_clu._ordered_keys('test')
    cells_path = O15 / 'stage2/cells_clu.json'
    stored = _stored_cells(cells_path, encoder=enc, data='mmart_test')
    fit = lambda x: ns_metrics.partitions(x, ns_paths.K20, 'mmart')   # noqa: E731
    frozen = fit(_unit32(md.features(mt_paths.DEFAULT_VIEW[enc], keys)))
    items = []
    for a in aspects:
        y = np.asarray(md.labels(keys, a)).astype(np.int64)
        cand = np.stack([fit(np.asarray(ex.load('new', enc, a, s, keys)).astype(np.float32)) for s in SEEDS])
        teacher = fit(_unit32(md.features(mt_paths.TEACHER_VIEW[a], keys)))
        for sc, comp in (('SC1', frozen), ('SC4', teacher)):
            cell = _ami_cell(cand, comp, y, ns_paths.K20, draws)
            items += _cell_items(f'r15/{enc}/mmart/{a}/{sc}', cell, stored[f'{sc}/{a}'], cells_path, draws)
    return _result(items, draws, tuple(aspects) == ASPECTS)


def replay_a_partitions(aspects=ASPECTS, seeds=SEEDS):
    """`ns_metrics.partitions(x, 20, 'artelingo')` on r10's stage 4 c* held exports gives the stored partitions
    `O10/stage4/partitions/full/k11=40/<aspect>__s<seed>__r20.npz` exactly (array equality)."""
    from pathlib import Path as P
    S0 = bs_artclu.S0
    held_dir = O10 / 'full/held/k11=40'
    cand = S0.Candidate('C', held_dir, 'k11=40')
    items = []
    for a in aspects:
        for s in seeds:
            x = S0.load_candidate(cand, 'full', a, s)[0]
            ours = ns_metrics.partitions(np.asarray(x, dtype=np.float32), ns_paths.K20, 'artelingo')
            path = O10 / f'stage4/partitions/full/k11=40/{a}__s{s}__r20.npz'
            theirs = rg_held.read_frozen_partitions(path, len(x))
            assert ours.shape == theirs.shape
            diff = int((ours != theirs).sum())
            items.append(dict(name=f'r10/artelingo/{a}/s{s}/partitions', source=f'{path}::labels', kind='partitions',
                              ours=f'{diff} entries differ', theirs='equal', abs_diff=float(diff), ok=diff == 0))
    return _result(items, FULL, tuple(aspects) == ASPECTS and tuple(seeds) == SEEDS)


# ------------------------------------------------------------------------------------------------ (b) matched arm
def replay_b(draws=FULL, aspects=ASPECTS):
    """r14 SigLIP 2 ArtELingo SC3, the matched-arm interval: M2's r20 and own readings on one seed draw, the larger taken in
    every draw (`ns_clu._boot` with two readings, r14's `b_matched`). Compares value, comparator value (larger reading),
    the reading's name, d and the interval. The one-reading path against r14 with the own reading removed is a unit test
    (`test_ns_clu.py`)."""
    names, _sha, labels = bs_artclu.held_setup()
    n = len(names)
    spec = bs_artclu.ENC['siglip2']
    cells_path = O14 / 'stage3/cells_artelingo_clu.json'
    stored = _stored_cells(cells_path, encoder='siglip2')
    root = O14 / 'stage3/partitions'
    matched = bs_artclu.stored_m(spec['root'], spec['matched'], n)
    items = []
    for a in aspects:
        cand = np.stack([rg_held.read_frozen_partitions(root / f'cstar/siglip2/{a}__s{s}__r20.npz', n) for s in SEEDS])
        comps = {'r20': np.stack([matched[s]['r20'] for s in SEEDS]), 'own': np.stack([matched[s]['own'] for s in SEEDS])}
        k_all = max(int(cand.max()), int(comps['r20'].max()), int(comps['own'].max())) + 1
        res = ns_clu._boot(cand, comps, labels[a].astype(np.int64), k_all, ('ami',), True, draws, 0, ns_clu.WORKERS)
        d = res['cand'][0] - res['comp'][0]
        lo, hi = np.percentile(res['d_draws'][:, 0], [2.5, 97.5])
        cell = dict(cand=res['cand'][0], comp=res['comp'][0], d=d, lo=lo, hi=hi)
        st = stored[f'SC3/{a}']
        items += _cell_items(f'r14/siglip2/artelingo/{a}/SC3', cell, st, cells_path, draws)
        reading = f'{spec["matched"]}-{res["reading"][0]}'
        items.append(_same(f'r14/siglip2/artelingo/{a}/SC3/comparator_reading',
                           f'{cells_path}::SC3/siglip2/{a}/comparator_reading', reading, st['comparator_reading']))
    return _result(items, draws, tuple(aspects) == ASPECTS)


# ------------------------------------------------------------------------------------------------ (c) photo sets
def replay_c_affection(draws=FULL):
    """r14 SigLIP 2 Affection test: SC4 (photo model against the plain image) and SC6 (against the teacher), AMI values
    and intervals, from r14's cached partitions (`O14/stage3/partitions/affection__test__*`) and r12's teacher partition."""
    labels = np.asarray(ac_data.labels('test')).astype(np.int64)
    n = len(labels)
    cells_path = O14 / 'stage3/cells_affection.json'
    stored = _stored_cells(cells_path, encoder='siglip2', task='clu')
    root = O14 / 'stage3/partitions'
    cand = np.stack([_read_parts(root / f'affection__test__photo_siglip2__s{s}.npz', n) for s in SEEDS])
    frozen = _read_parts(root / 'affection__test__frozen_siglip2__s-.npz', n)
    teacher = _read_parts(R12 / 'partitions/affection__test__teacher_artwork__s-.npz', n)
    items = []
    for sc, comp in (('SC4', frozen), ('SC6', teacher)):
        cell = _ami_cell(cand, comp, labels, ns_paths.K20, draws)
        items += _cell_items(f'r14/siglip2/affection/emotion/{sc}', cell, stored[f'{sc}/emotion'], cells_path, draws)
    return _result(items, draws)


def _emoset_labels():
    import bs_emoset
    import bs_emoset_score as ES
    src = ES.RealSource()
    keys = bs_emoset.keys('test')
    ES.check_order(keys)
    return src, keys, src.labels(keys)


def replay_c_emoset(draws=FULL):
    """r14 EmoSet test (Qwen): SC7 (photo model against the plain image) and SC9 (against the teacher, artwork wording),
    AMI values and intervals, from r14's cached partitions `O14/stage3/partitions_emoset/`."""
    _src, keys, labels = _emoset_labels()
    n = len(labels)
    cells_path = O14 / 'stage3/cells_emoset.json'
    stored = _stored_cells(cells_path, encoder='qwen', task='clu')
    root = O14 / 'stage3/partitions_emoset'
    cand = np.stack([_read_parts(root / f'emoset__test__photo_qwen__s{s}.npz', n) for s in SEEDS])
    items = []
    for sc, arm in (('SC7', 'plain_qwen'), ('SC9', 'teacher_emotion_focus')):
        comp = _read_parts(root / f'emoset__test__{arm}__s-.npz', n)
        cell = _ami_cell(cand, comp, labels, ns_paths.K20, draws)
        items += _cell_items(f'r14/qwen/emoset/emotion/{sc}', cell, stored[f'{sc}/emotion'], cells_path, draws)
    return _result(items, draws)


def gate_i_c(draws=FULL):
    return _merge(replay_c_affection(draws), replay_c_emoset(draws))


# ------------------------------------------------------------------------------------------------ (d) K != 20
_TASKS = {}


def _fit8(i):
    x, k, name = _TASKS['tasks'][i]
    return ns_metrics.partitions(x, k, 'emoset')          # 1 thread, set in code (D20)


def replay_d_emoset(arms=None, workers=8):
    """r14 EmoSet K = 8: the AMI at 20 (from r14's cached partitions) and at K = 8 (recomputed with `ns_metrics.partitions`
    at 1 thread, from the stored float16 rows of a seeded arm and the unit float32 rows of a frozen one, as r14's
    `score_arm`) of each `sheet_emoset.json` row, as `ac_cluster.ami` (mean over states, then seeds). `arms`: a subset of
    the sheet's rows (default all 12)."""
    import bs_emoset_score as ES
    src, keys, labels = _emoset_labels()
    n = len(labels)
    sheet_path = O14 / 'stage3/sheet_emoset.json'
    rows = {r['arm']: r for r in json.loads(sheet_path.read_text())['rows']}
    arms = tuple(rows) if arms is None else tuple(arms)
    assert set(arms) <= set(rows), sorted(set(arms) - set(rows))
    root = O14 / 'stage3/partitions_emoset'
    tasks, index = [], {}
    for arm in arms:
        seeded = arm in ES.SEEDED
        for s in (SEEDS if seeded else (None,)):
            x = src.stored(arm, 'test', s, keys) if seeded else src.frozen(arm, keys)
            index[(arm, s)] = len(tasks)
            tasks.append((x, 8, f'{arm}/{s}'))
    _TASKS['tasks'] = tasks
    try:
        if workers <= 1:
            fitted = [_fit8(i) for i in range(len(tasks))]
        else:
            with multiprocessing.get_context('fork').Pool(min(workers, len(tasks))) as pool:
                fitted = pool.map(_fit8, range(len(tasks)), chunksize=1)
    finally:
        _TASKS.clear()
    items = []
    for arm in arms:
        seeded = arm in ES.SEEDED
        seeds = SEEDS if seeded else (None,)
        p8 = np.stack([fitted[index[(arm, s)]] for s in seeds]) if seeded else fitted[index[(arm, None)]]
        p20 = np.stack([_read_parts(root / f'emoset__test__{arm}__s{"-" if s is None else s}.npz', n) for s in seeds])
        p20 = p20 if seeded else p20[0]
        for k, parts, key in ((8, p8, 'ami_k8'), (20, p20, 'ami_k20')):
            ours = ns_clu.metric_table(arm, 'emotion', parts, labels, k, metrics=('ami',))[0]['mean']
            items.append(_item(f'r14/emoset/{arm}/{key}', f'{sheet_path}::rows[{arm}]/{key}', ours, rows[arm][key]))
    return _result(items, FULL, set(arms) == set(rows))


def replay_d_class_count(encs=ns_paths.ENCODERS, aspects=('emotion', 'genre'), seeds=SEEDS):
    """r15's class-count diagnostic (`O15/diag/clustering_k_class.json`; MMArt test, K = 9 emotion and 10 genre; style is
    K = 20 and repeats its AMI at 20, covered by (a)): per encoder and aspect X-c* (per seed and mean), the frozen image,
    the matched arm (k-means on the L2-normalised probabilities, `ns_metrics.percept_partitions`) and the teacher, each AMI
    at K recomputed with `ns_metrics.partitions` (8 threads) from r15's exports, features and probabilities in r15's order
    (`mt_diag.task_clu_test`'s loaders). A subset of `seeds` compares the per-seed values only."""
    import importlib
    import mt_diag
    import mt_paths
    md, ex, mm, ret = (importlib.import_module(m) for m in ('mt_data', 'mt_export', 'mt_matched', 'mt_ret'))
    keys = mt_diag.ordered_keys('test')
    y = mt_diag.aspect_labels(keys)
    diag_path = O15 / 'diag/clustering_k_class.json'
    sections = json.loads(diag_path.read_text())['sections']
    items = []
    fit = lambda x, k: ns_metrics.partitions(x, k, 'mmart')            # noqa: E731
    ami = lambda arm, a, parts, k: ns_clu.metric_table(arm, a, parts, y[a], k, ('ami',))[0]   # noqa: E731
    teacher = {}
    for a in aspects:
        k = len(mt_paths.CLASSES[a])
        assert k != ns_paths.K20, f'{a} has K = 20: nothing new to replay'
        teacher[a] = ami('teacher', a, fit(ret.view_rows(mt_paths.TEACHER_VIEW[a], keys), k), k)['mean']
    for enc in encs:
        frozen_x = _unit32(md.features(mt_paths.DEFAULT_VIEW[enc], keys))
        for a in aspects:
            k = len(mt_paths.CLASSES[a])
            st = sections[enc]['aspects'][a]
            assert st['k'] == k
            src = f'{diag_path}::sections.{enc}.aspects.{a}'
            per = []
            for s in seeds:
                x = np.asarray(ex.load('new', enc, a, s, keys)).astype(np.float32)
                per.append(ami('cstar', a, fit(x, k), k)['mean'])
                items.append(_item(f'r15/{enc}/{a}/cstar_per_seed[{s}]', f'{src}.cstar_per_seed.kclass[{s}]', per[-1],
                                   st['cstar_per_seed']['kclass'][s]))
            if tuple(seeds) == SEEDS:
                items.append(_item(f'r15/{enc}/{a}/cstar_kclass', f'{src}.cstar_kclass', np.mean(per), st['cstar_kclass']))
                mk = [ami('matched', a, ns_metrics.percept_partitions(mm.load_probs(enc, s, keys), k, 'mmart'), k)['mean']
                      for s in SEEDS]
                items.append(_item(f'r15/{enc}/{a}/matched_kclass', f'{src}.matched_kclass', np.mean(mk), st['matched_kclass']))
                items.append(_item(f'r15/{enc}/{a}/frozen_kclass', f'{src}.frozen_kclass',
                                   ami('frozen', a, fit(frozen_x, k), k)['mean'], st['frozen_kclass']))
                items.append(_item(f'r15/{enc}/{a}/teacher_kclass', f'{src}.teacher_kclass', teacher[a], st['teacher_kclass']))
    complete = tuple(encs) == tuple(ns_paths.ENCODERS) and tuple(aspects) == ('emotion', 'genre') and tuple(seeds) == SEEDS
    return _result(items, FULL, complete)


# ------------------------------------------------------------------------------------------------ all
def gate_i_a(draws=FULL):
    return _merge(replay_a_r14(draws), replay_a_r15(draws), replay_a_partitions())


def gate_i_d():
    return _merge(replay_d_emoset(), replay_d_class_count())


def gate_i(draws=FULL):
    """{'a', 'b', 'c', 'd': result} of stage 0 (i), the status keys of the index (`i_a`, `i_b`, `i_c` per set, `i_d`)."""
    return dict(a=gate_i_a(draws), b=replay_b(draws), c=dict(affection=replay_c_affection(draws),
                                                              emoset=replay_c_emoset(draws)), d=gate_i_d())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('part', choices=['a_r14', 'a_r15', 'a_partitions', 'b', 'c_affection', 'c_emoset', 'd_emoset',
                                    'd_class_count', 'all'])
    p.add_argument('--draws', type=int, default=FULL)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--aspects', nargs='+', default=None)
    p.add_argument('--out', default=None, help='write the result JSON here (default: print the summary only)')
    a = p.parse_args(argv)
    ns_clu.WORKERS = a.workers
    t0 = time.monotonic()
    kw = {} if a.aspects is None else dict(aspects=tuple(a.aspects))
    fn = {'a_r14': lambda: replay_a_r14(a.draws, **kw), 'a_r15': lambda: replay_a_r15(a.draws, **kw),
          'a_partitions': lambda: replay_a_partitions(**kw), 'b': lambda: replay_b(a.draws, **kw),
          'c_affection': lambda: replay_c_affection(a.draws), 'c_emoset': lambda: replay_c_emoset(a.draws),
          'd_emoset': lambda: replay_d_emoset(workers=a.workers),
          'd_class_count': lambda: replay_d_class_count(**kw), 'all': lambda: gate_i(a.draws)}[a.part]
    res = fn()
    flat = [res] if 'items' in res else [r for v in res.values() for r in (v.values() if 'items' not in v else [v])]
    for r in flat:
        print(f'{a.part}: pass={r["pass"]} wiring_ok={r["wiring_ok"]} max_abs_diff={r["max_abs_diff"]:.3g} '
              f'draws={r["draws"]} complete={r["complete"]} items={len(r["items"])}')
    print(f'{time.monotonic() - t0:.0f}s')
    if a.out:
        ns_paths.write_json(a.out, res)
    return 0 if all(r['wiring_ok'] for r in flat) else 1


if __name__ == '__main__':
    raise SystemExit(main())
