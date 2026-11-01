"""r13 diagnostics after the verdict (spec 2026-10-31 §3.6). They decide nothing. `main()` is called by the stage runner's
`diag` subcommand and writes `OUTPUTS/diag/diag.json`, one key per §3.6 item:

  r0_rows                 (a) plain-cosine (R0) mAP and nDCG: the photo model from its exports (per seed), the comparators
                          carried from r12's `diag/diag.json`
  ndcg                    (b) graded emotion nDCG, R0 and under the picked read-out: photo computed, comparators carried
  untied                  (c) SC1 to SC4 and the verdict again on the test images with no plurality tie (as r12's untied)
  per_source_map,         (d) per-query AP grouped by source / by class (r12's `ac_diag._per_source`, `_per_class`)
  per_class_ap
  photo_vs_cstar          (e) photo model minus frozen c* (r12's exports), paired, clustering and retrieval
  teacher_gain_share      (f) (arm - plain) / (teacher - plain), point values, as report 12-31 (c* beside it)
  seed_range              (g) r' beside r12's ranges and the held (test) per-seed spread of AMI and mAP
  sweep_sheet             (h) the stage 1 sheet, with row 8 against row 1
  edge_plurality_share    (i) per sweep row, share of validation graph edges whose two photos share the plurality emotion
  descriptive_rows        (j) r12's PercepT core and CLIP L/14 rows, carried

The functions take arrays; `main()` is the thin loader (photo model through `pf_export`, comparators and frozen c* from
r12's outputs, read only). Reused from r12's `ac_diag`: `_per_source`, `_per_class`, `_perquery` (r12's per-query files).
"""
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import pf_paths  # first: puts r12's round on sys.path
import ac_cluster
import ac_data
import ac_diag
import ac_select
import ac_stats
import rt_metrics
import rt_readout
import rt_stats
import pf_stage3

SEEDS = tuple(pf_paths.SEEDS)
PHOTO = 'photo'


def _f(x):
    return float(x)


def _read(path):
    return json.loads(Path(path).read_text())


# ---- (a), (b): R0 rows and graded nDCG ---------------------------------------------------------------------------------

def r0_row(test_embed, labels, dist, workers=8):
    """Plain-cosine rows of a seeded arm: float32 unit [5, n, d] -> {'map_r0': {'map', 'per_seed'}, 'ndcg_r0': {'mean',
    'per_seed'}} (`rt_metrics.retrieval` with qe_k = 0, `rt_metrics.graded`)."""
    maps = [_f(rt_metrics.retrieval(np.ascontiguousarray(test_embed[s]), labels, qe_k=0, workers=workers)['ap'].mean())
            for s in range(len(test_embed))]
    nd = [_f(rt_metrics.graded(np.ascontiguousarray(test_embed[s]), dist, workers=workers).mean()) for s in range(len(test_embed))]
    return dict(map_r0=dict(map=_f(np.mean(maps)), per_seed=maps), ndcg_r0=dict(mean=_f(np.mean(nd)), per_seed=nd))


def ndcg_picked(val_embed, test_embed, readout, dist, workers=8):
    """Graded nDCG per seed after the picked read-out's transform (fitted on validation) then plain cosine, as r12's
    `ac_diag._ndcg` (overlap gain, no query expansion). Arrays float32 unit [5, nv|n, d]."""
    ro = ac_select.readout_from_name(readout)
    out = []
    for s in range(len(test_embed)):
        x = rt_readout.stage5_apply(ro, val_embed[s], test_embed[s])
        out.append(_f(rt_metrics.graded(np.ascontiguousarray(x, dtype=np.float32), dist, workers=workers).mean()))
    return dict(readout=readout, mean=_f(np.mean(out)), per_seed=out)


# ---- (d): per source and per class -------------------------------------------------------------------------------------

def per_source_class(ap, rows, src, labels):
    """(per-source mAP, per-class AP) of one per-query AP array ([nq] or [5, nq]) with its query rows (r12's functions)."""
    return ac_diag._per_source(ap, rows, src), ac_diag._per_class(ap, rows, labels)


# ---- (e): photo minus frozen c*, paired ----------------------------------------------------------------------------------

def paired_vs_cstar(parts_photo, parts_cstar, labels, ap_photo, ap_cstar, rows_photo, rows_cstar, draws_clu, draws_ret,
                    workers=8):
    """Photo model minus frozen c* on the same test photos. Clustering: int32 [5, 3, n] each, `ac_cluster.paired` with
    one shared seed draw (photo seed s starts from c* seed s). Retrieval: per-query AP [5, nq] each (each under its own
    picked read-out), `rt_stats.paired` (its seed draw is shared by both sides). Returns d, interval and both points."""
    assert np.array_equal(rows_photo, rows_cstar), 'the two arms were scored on different queries'
    clu = ac_cluster.paired(parts_photo, parts_cstar, labels, draws_clu, workers, shared_seeds=True)
    ret = rt_stats.paired(ap_photo, ap_cstar, draws_ret)
    return dict(
        clustering=dict(metric='ami', photo=_f(clu['A_cand']), cstar=_f(clu['A_comp']), d=_f(clu['d']),
                        interval=[_f(v) for v in clu['interval']], draws=draws_clu, shared_seeds=True),
        retrieval=dict(metric='map', photo=_f(ap_photo.mean(axis=1).mean()), cstar=_f(ap_cstar.mean(axis=1).mean()),
                       d=_f(ret['d']), interval=[_f(v) for v in ret['interval']], draws=draws_ret))


# ---- (f): share of the teacher's gain ------------------------------------------------------------------------------------

def gain_share(point, plain, teacher):
    """(point - plain) / (teacher - plain), as report 12-31 (point values). The teacher must be above plain."""
    assert teacher > plain, (teacher, plain)
    return _f((point - plain) / (teacher - plain))


def teacher_gain_share(rows, teacher):
    """rows: verdict rows {arm: row} with 'ami' and 'map_picked'['map']; the share for each arm in {photo, cstar} in both
    tasks. Returns {arm: {'ami', 'map'}}."""
    plain, tea = rows['plain'], rows[teacher]
    return {arm: dict(ami=gain_share(rows[arm]['ami'], plain['ami'], tea['ami']),
                      map=gain_share(rows[arm]['map_picked']['map'], plain['map_picked']['map'], tea['map_picked']['map']))
            for arm in (PHOTO, 'cstar')}


# ---- (g): r' beside r12's ranges and the held spread -------------------------------------------------------------------------

def spread(per_seed, r_prime, r12):
    """Held per-seed values, their spread (max minus min) beside r' (validation, this round) and r12's r."""
    v = [_f(x) for x in per_seed]
    return dict(per_seed=v, spread=max(v) - min(v), r_prime=_f(r_prime), r12=_f(r12))


def seed_range(stage2, ami_per_seed, map_picked_per_seed, map_r0_per_seed, ndcg_r0_per_seed, cstar_diag):
    """stage2: STAGE2_FILE's dict. The photo model's held spreads beside r' and r12's ranges; c*'s r12 spread carried."""
    rp, r12 = stage2['ranges'], stage2['r12_ranges']
    return dict(
        r_prime={k: _f(rp[k]['r']) for k in ('clu', 'map', 'ndcg')},
        r12={k: _f(r12[k]['r']) for k in ('clu', 'map', 'ndcg')},
        validation_per_seed={k: [_f(x) for x in rp[k]['per_seed']] for k in ('clu', 'map', 'ndcg')},
        held=dict(ami=spread(ami_per_seed, rp['clu']['r'], r12['clu']['r']),
                  map_picked=spread(map_picked_per_seed, rp['map']['r'], r12['map']['r']),
                  map_r0=spread(map_r0_per_seed, rp['map']['r'], r12['map']['r']),
                  ndcg_r0=spread(ndcg_r0_per_seed, rp['ndcg']['r'], r12['ndcg']['r'])),
        cstar_r12_held=cstar_diag)


# ---- (h): the sheet --------------------------------------------------------------------------------------------------------

SHEET_FIELDS = ('ami', 'map', 'g_clu', 'g_map', 'score')


def sheet_summary(sheet):
    """The full sheet (rows 0 to 9, row 9's settings, the pick) and row 8 (from random weights) against row 1: the
    difference row 8 minus row 1 in each score field (None when either row has no score)."""
    r1, r8 = sheet['rows']['1'], sheet['rows']['8']
    diff = {f: (None if r1[f] is None or r8[f] is None else _f(r8[f] - r1[f])) for f in SHEET_FIELDS}
    return dict(sheet=sheet, row8_vs_row1=dict(row1_status=r1['status'], row8_status=r8['status'],
                                               row1={f: r1[f] for f in SHEET_FIELDS}, row8={f: r8[f] for f in SHEET_FIELDS},
                                               row8_minus_row1=diff))


# ---- (i): validation graph edges by plurality emotion -----------------------------------------------------------------------

def edge_share(pairs, labels):
    """Share of edges (int64 [M, 2], local rows) whose two photos have the same plurality emotion (labels int [n])."""
    pairs = np.asarray(pairs)
    assert pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.max() < len(labels)
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    return dict(n_edges=int(len(pairs)), share=_f(same.mean()))


def chance_share(labels):
    """Share of random pairs with equal plurality emotion (sum of squared class shares), for scale."""
    p = np.bincount(labels) / len(labels)
    return _f((p ** 2).sum())


def sweep_edge_shares(sheet, load_dev_pairs, labels):
    """Per sweep row, `edge_share` of the validation graph of the row's (k, correction); rows without a graph
    (reference row 0, skipped rows) are None. load_dev_pairs(k, correction) -> int64 [M, 2]."""
    cache, out = {}, {}
    for n in sorted(sheet['rows'], key=int):
        row = sheet['rows'][n]
        if row['status'] not in ('ok', 'failed'):
            out[n] = None
            continue
        k, corr = int(row['settings']['k']), bool(row['settings']['correction'])
        if (k, corr) not in cache:
            cache[(k, corr)] = edge_share(load_dev_pairs(k, corr), labels)
        out[n] = dict(k=k, correction=corr, status=row['status'], **cache[(k, corr)])
    return dict(rows=out, chance=chance_share(labels))


# ---- (c): the verdict on untied images ----------------------------------------------------------------------------------------

def _partition_one(x):
    with threadpool_limits(1):
        return ac_cluster.partitions(x)


def _partitions_many(xs, workers):
    with mp.get_context('fork').Pool(min(workers, len(xs))) as pool:
        return pool.map(_partition_one, xs, chunksize=1)


def untied_scores(cand, comps, r, labels, ties, draws_clu, draws_ret, workers=8):
    """SC1 to SC4 and the verdict on the test photos whose plurality has no tie. Evaluation set = the untied photos:
    partitions refit on it (the photo model's stored rows, the comparators' as r12 scored them), queries and gallery are
    that set, read-outs (fitted on validation) and r' unchanged; same draws as stage 3.
    cand: `pf_stage3.Candidate`; comps: {'plain'|'teacher': dict(arm, readout, test_embed [n, d], val_embed [nv, d])};
    ties: bool [n]; r: {'clu', 'map'} (r'); labels int64 [n]. Comparators' clustering input is their embed rows
    (r12's `_untied`)."""
    keep = np.flatnonzero(~np.asarray(ties))
    assert 0 < len(keep) <= len(labels)
    lab = np.asarray(labels)[keep]
    stored = [cand.test_stored[s][keep] for s in range(5)] + [comps[k]['test_embed'][keep] for k in ('plain', 'teacher')]
    parts = _partitions_many(stored, workers)
    parts_c = np.stack(parts[:5])
    comp_parts = dict(plain=parts[5], teacher=parts[6])
    ret_c = ac_stats.arm_retrieval(lambda s: cand.val_embed[s], lambda s: np.ascontiguousarray(cand.test_embed[s][keep]),
                                   cand.readout, lab, seeds=ac_stats.SEEDS, workers=workers)
    sc, dropped = {}, dict(photo=int(ret_c['dropped']))
    ret_t = {}
    for sid, (metric, which, _, rkey) in pf_stage3.SC.items():
        if sid == 'SC5':
            continue
        comp = comps[which]
        if metric == 'ami':
            res = ac_cluster.paired(parts_c, comp_parts[which], lab, draws_clu, workers)
            sc[sid] = pf_stage3._entry(sid, cand, comp, res['A_cand'], res['A_comp'], res['d'], res['interval'], r[rkey])
        else:
            if which not in ret_t:
                ret_t[which] = ac_stats.arm_retrieval(
                    lambda s, c=comp: c['val_embed'], lambda s, c=comp: np.ascontiguousarray(c['test_embed'][keep]),
                    comp['readout'], lab, seeds=None, workers=workers)
                dropped[which] = int(ret_t[which]['dropped'])
            t = ret_t[which]
            assert np.array_equal(ret_c['rows'], t['rows']), (sid, 'different queries')
            p = rt_stats.paired(ret_c['ap'], t['ap'], draws_ret)
            sc[sid] = pf_stage3._entry(sid, cand, comp, ret_c['point'], t['point'], p['d'], p['interval'], r[rkey])
    outcomes = {sid: e['outcome'] for sid, e in sc.items()}
    return json.loads(json.dumps(dict(
        n=int(len(keep)), n_tied_dropped=int(len(labels) - len(keep)), draws_clu=draws_clu, draws_ret=draws_ret,
        dropped_queries=dropped, sc=sc, verdict=pf_stage3.verdict(outcomes)), allow_nan=False))


# ---- the loader ---------------------------------------------------------------------------------------------------------------

class _PhotoLoader:
    """The photo model's exports of runs final/s0..s4, through `pf_export` (sidecars checked there)."""

    @staticmethod
    def stored(split, seed):
        import pf_export
        return pf_export.stored(f'final/s{seed}', split)

    @staticmethod
    def embed(split, seed):
        import pf_export
        return pf_export.embed(f'final/s{seed}', split)


def _photo_perquery(out, r0):
    aps, rows = [], None
    for s in SEEDS:
        with np.load(out / f'stage3/perquery/affection__{PHOTO}{"_R0" if r0 else ""}__emotion__s{s}.npz', allow_pickle=False) as z:
            ap, r = z['ap'], z['rows']
        assert rows is None or np.array_equal(rows, r)
        rows = r
        aps.append(ap)
    return np.stack(aps), rows


def _load_parts(path, labels):
    with np.load(path, allow_pickle=False) as z:
        p = z['labels']
    assert p.shape == (3, len(labels)) and p.dtype == np.int32, (path, p.shape)
    return p


def main(workers=8):
    t0 = time.monotonic()
    out, r12 = Path(pf_paths.OUTPUTS), Path(pf_paths.R12_OUT)
    assert (out / 'stage3/verdict.json').is_file(), 'diag runs after the verdict'
    v3 = _read(out / 'stage3/verdict.json')
    assert v3['smoke'] == pf_paths.SMOKE, ('the verdict is from another smoke setting', v3['smoke'])
    s2 = _read(pf_paths.STAGE2_FILE)
    sheet = _read(pf_paths.SHEET_FILE)
    picks = _read(r12 / 'stage1/picks.json')['affection']
    r12v2, r12diag = _read(r12 / 'stage2/verdict.json'), _read(r12 / 'diag/diag.json')
    arms = pf_stage3.comparator_arms(picks)
    teacher, crl = arms['teacher'], arms['crl']
    labels, dist, src = ac_data.labels('test'), ac_data.emotion_dist('test'), ac_data.sources('test')
    n = len(labels)
    draws_clu, draws_ret = v3['draws_clu'], v3['draws_ret']
    rows = v3['rows']
    assert set(rows) == {PHOTO, 'plain', teacher, 'cstar', crl}, sorted(rows)
    cand = pf_stage3.candidate_from_data(PHOTO, rows[PHOTO]['readout'], _PhotoLoader)
    pf_stage3._check_candidate(cand)
    assert cand.test_stored.shape[1] == n
    doc = dict(smoke=pf_paths.SMOKE, reading=dict(
        r0_rows='photo model computed from its exports; comparators carried from r12 diag/diag.json (r12 outputs)',
        ndcg='overlap gain, no query expansion; photo computed, comparators carried from r12 diag/diag.json',
        untied='test images with no plurality tie; partitions refit; queries and gallery are that set; read-outs (fit on '
               'validation) and r\' unchanged; stage 3 draws; SC5 not repeated',
        photo_vs_cstar='photo minus frozen c*, each under its own picked read-out; shared seed draw (photo seed s starts '
                       'from c* seed s)',
        teacher_gain_share='(arm - plain) / (teacher - plain), point values (AMI; mAP under each arm\'s picked read-out)',
        edge_plurality_share='validation graph of the row\'s (k, correction); plurality = ac_data.labels(val); chance = '
                             'share for random pairs'))
    # (a) R0 rows
    mine = r0_row(cand.test_embed, labels, dist, workers)
    assert abs(mine['map_r0']['map'] - rows[PHOTO]['map_r0']['map']) < 1e-9, 'R0 mAP differs from the verdict row'
    assert abs(mine['ndcg_r0']['mean'] - rows[PHOTO]['ndcg_r0']) < 1e-9, 'R0 nDCG differs from the verdict row'
    carried = {arm: r for arm, r in r12diag['r0_rows'].items() if arm in ('cstar', 'plain', teacher, crl)}
    doc['r0_rows'] = {PHOTO: dict(readout=cand.readout, map_picked=rows[PHOTO]['map_picked'], ami=rows[PHOTO]['ami'], **mine),
                      **{'r12/' + arm: r for arm, r in carried.items()}}
    # (b) nDCG
    pk = ndcg_picked(cand.val_embed, cand.test_embed, cand.readout, dist, workers)
    doc['ndcg'] = {PHOTO: dict(readout=cand.readout, r0=mine['ndcg_r0'], picked=pk),
                   **{'r12/' + arm: r12diag['ndcg'][arm] for arm in carried}}
    # (d) per source and per class
    src_map, cls_ap = {}, {}
    for arm in (PHOTO, 'cstar', 'plain', teacher, crl):
        src_map[arm], cls_ap[arm] = {}, {}
        for kind, r0 in (('picked', False), ('r0', True)):
            ap, qrows = _photo_perquery(out, r0) if arm == PHOTO else ac_diag._perquery(2, arm, r0)
            src_map[arm][kind], cls_ap[arm][kind] = per_source_class(ap, qrows, src, labels)
    doc['per_source_map'], doc['per_class_ap'] = src_map, cls_ap
    # (e) photo minus frozen c*
    parts_p = np.stack([_load_parts(out / f'partitions/affection__test__{PHOTO}__s{s}.npz', labels) for s in SEEDS])
    parts_c = np.stack([_load_parts(r12 / f'partitions/affection__test__cstar__s{s}.npz', labels) for s in SEEDS])
    ap_p, rows_p = _photo_perquery(out, False)
    ap_c, rows_c = ac_diag._perquery(2, 'cstar', False)
    doc['photo_vs_cstar'] = paired_vs_cstar(parts_p, parts_c, labels, ap_p, ap_c, rows_p, rows_c, draws_clu, draws_ret, workers)
    # (f) share of the teacher's gain
    doc['teacher_gain_share'] = teacher_gain_share(rows, teacher)
    # (g) r' and spreads
    ami_p = [_f(ac_cluster.ami(parts_p[s], labels)) for s in range(5)]
    assert abs(np.mean(ami_p) - rows[PHOTO]['ami']) < 1e-9, (np.mean(ami_p), rows[PHOTO]['ami'])
    doc['seed_range'] = seed_range(s2, ami_p, rows[PHOTO]['map_picked']['per_seed'], mine['map_r0']['per_seed'],
                                   mine['ndcg_r0']['per_seed'], r12diag['cstar_seed_spread'])
    # (h) sheet
    doc['sweep_sheet'] = sheet_summary(sheet)
    # (i) validation graph edges
    import pf_data
    import pf_graphs
    fit_keys, val_keys = pf_data.fit_keys(), pf_data.keys('val')
    doc['edge_plurality_share'] = sweep_edge_shares(
        sheet, lambda k, c: pf_graphs.load(k, c, fit_keys, val_keys)['dev_pairs'], ac_data.labels('val'))
    # (j) r12's descriptive rows
    doc['descriptive_rows'] = dict(
        percept_core=dict(row=r12v2['rows']['percept'], **r12diag['percept_core']),
        clip_l14=r12v2['rows']['clip'], source='r12 stage2/verdict.json rows and diag/diag.json (archived, carried)')
    # (c) untied (the slow item, last)
    comps = {}
    for key, arm in (('plain', 'plain'), ('teacher', teacher)):
        comps[key] = dict(arm=arm, readout=picks['readout'][arm], test_embed=ac_data.embed(arm, 'test'),
                          val_embed=ac_data.embed(arm, 'val'))
    r = {k: _f(s2['ranges'][k]['r']) for k in ('clu', 'map')}
    doc['untied'] = untied_scores(cand, comps, r, labels, ac_data.ties('test'), draws_clu, draws_ret, workers)
    doc['untied']['full_set_verdict'] = v3['verdict']
    doc['seconds'] = round(time.monotonic() - t0, 1)
    (out / 'diag').mkdir(parents=True, exist_ok=True)
    (out / 'diag/diag.json').write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    return doc


if __name__ == '__main__':
    main()
