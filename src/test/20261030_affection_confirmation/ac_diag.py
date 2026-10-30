"""r12 diagnostics (`--stage diag`) and the Impressions figure (`--stage 4`). Spec 2026-10-30, 3.5 stage 4, 3.6, D1, D8, D13.

Neither stage decides anything. Both are called from ac_run.py (`stagediag`, `stage4`); this module reads ac_run's helpers
lazily (ac_run imports nothing from here at import time).

diag reads the three verdict stages' files (stage1/scores.json and picks, stage2/verdict.json, stage3/verdict.json and
the per-query arrays and partitions stage 2 wrote) and writes OUTPUTS/diag/diag.json:
  r0_rows, ndcg, switch_column, untied, per_source_map, per_class_ap, cstar_seed_spread, rows_not_picked, chance_map,
  percept_core, plus `reading` (the wording of each reading) and `inputs`.
stage 4 writes OUTPUTS/stage4/impressions.png and impressions.json.
"""
import json
import re
from pathlib import Path

import numpy as np

import ac_paths
import ac_cluster
import ac_data
import ac_select
import ac_stats
import rs_vh
import rt_metrics
import rt_readout
import rt_stats

SEEDS = tuple(ac_paths.SEEDS)
ASPECTS = tuple(ac_paths.ASPECTS)
PERCEPT_PUBLISHED = (0.18, 0.21, 0.25)          # PercepT's published numbers, a different protocol (spec D8)
SPEC_UUIDS = 827                                # D13's count in the spec
N_QUERIES, TOP = 6, 5


def _run():
    import ac_run                               # lazy: ac_run is the caller (a second import when run as __main__)
    return ac_run


def _smoke():
    return int(getattr(ac_paths, 'SMOKE', 0) or 0)


def _out():
    return Path(ac_paths.OUTPUTS)


def _f(x):
    return float(x)


# ---- diag: reading the stage outputs ------------------------------------------------------------------------------------

def _perquery(stage, arm, r0=False):
    """Stage `stage`'s per-query arrays of an Affection arm (picked read-out, or R0): (ap, rows). ap float64 [5, nq]
    (seeded) or [nq]; rows int64 [nq], identical across seeds."""
    seeded = arm in ac_data.SEEDED
    aps, rows = [], None
    for seed in (SEEDS if seeded else (None,)):
        p = _out() / f'stage{stage}/perquery/affection__{arm}{"_R0" if r0 else ""}__emotion__s{"-" if seed is None else seed}.npz'
        with np.load(p, allow_pickle=False) as z:
            assert set(z.files) >= {'ap', 'rows'}, (p, z.files)
            ap, r = z['ap'], z['rows']
        assert ap.ndim == 1 and ap.shape == r.shape and ap.dtype == np.float64 and r.dtype == np.int64, (p, ap.shape)
        assert rows is None or np.array_equal(rows, r), (p, 'queries differ between seeds')
        rows = r
        aps.append(ap)
    return (np.stack(aps) if seeded else aps[0]), rows


def _seed_mean(ap):
    return ap.mean(axis=0) if ap.ndim == 2 else ap


def _need_files():
    o = _out()
    for p in ('stage1/scores.json', 'stage1/picks.json', 'stage2/verdict.json', 'stage3/verdict.json'):
        assert (o / p).is_file(), f'diag runs after the verdicts: {p} is missing'


def _per_source(ap, rows, src):
    """Mean AP per source over the queries of that source (mean over seeds first for a seeded arm); gallery is unchanged
    (all test images, as scored). src: list[str] in keys('test') order, grouped through `rows`, never by position."""
    s = np.asarray(src)[rows]
    m = _seed_mean(ap)
    return {name: dict(map=_f(m[s == name].mean()), n_queries=int((s == name).sum())) for name in sorted(set(s))}


def _per_class(ap, rows, labels):
    lab = labels[rows]
    m = _seed_mean(ap)
    return {str(int(c)): dict(name=ac_paths.EMOTIONS[int(c)], ap=_f(m[lab == c].mean()), n_queries=int((lab == c).sum()))
            for c in np.unique(lab)}


def _ndcg(arm, ro, dist, workers):
    """Per-seed mean nDCG (overlap gain, no query expansion) of the arm's test rows under read-out `ro` (None: R0)."""
    out = []
    for seed in (SEEDS if arm in ac_data.SEEDED else (None,)):
        x = ac_data.embed(arm, 'test', seed)
        if ro is not None:
            x = rt_readout.stage5_apply(ro, ac_data.embed(arm, 'val', seed), x)
        v = rt_metrics.graded(np.ascontiguousarray(x, dtype=np.float32), dist, workers=workers)
        assert v.shape == (len(dist),)
        out.append(_f(v.mean()))
    return out


def _switch(labels, workers):
    """c*'s heads scored with emotion labels on test, mAP at R0, per seed and the seed mean."""
    doc = {}
    for head in ASPECTS:
        per = []
        for seed in SEEDS:
            x = ac_data.embed('cstar', 'test', seed, head=head)
            per.append(_f(rt_metrics.retrieval(x, labels, qe_k=0, workers=workers)['ap'].mean()))
        doc[head] = dict(map_r0=_f(np.mean(per)), per_seed=per)
    return doc


def _untied(picks, ranges, draws_clu, draws_ret, workers):
    """SC1 to SC4 on the untied test images only. The evaluation set is the test images whose plurality has no tie:
    partitions refit on it, queries and gallery are that set, read-outs (fit on validation) and r unchanged."""
    R = _run()
    keep = np.flatnonzero(~ac_data.ties('test'))
    labels = ac_data.labels('test')[keep]
    assert len(keep) < ac_paths.N['test'] or _smoke(), 'no tied image'
    get = lambda arm: (lambda s: np.ascontiguousarray(ac_data.embed(arm, 'test', s)[keep]))
    get_cluster = lambda arm, s: (ac_data.stored(arm, 'test', s) if arm == 'cstar' else
                                  ac_data.embed(arm, 'test', s))[keep]
    aff = picks['affection']
    parts, ret, sc = {}, {}, {}

    def parts_of(arm):
        if arm not in parts:
            if arm in ac_data.SEEDED:
                parts[arm] = np.stack([ac_cluster.partitions(get_cluster(arm, s)) for s in SEEDS])
            else:
                parts[arm] = ac_cluster.partitions(get_cluster(arm, None))
        return parts[arm]

    def ret_of(arm):
        if arm not in ret:
            seeds = SEEDS if arm in ac_data.SEEDED else None
            ret[arm] = ac_stats.arm_retrieval(lambda s: ac_data.embed(arm, 'val', s), get(arm), aff['readout'][arm],
                                              labels, seeds=seeds, workers=workers)
        return ret[arm]

    for sid in ('SC1', 'SC3'):
        comp = ac_stats.comparator_arm(sid, picks)
        res = ac_cluster.paired(parts_of('cstar'), parts_of(comp), labels, draws_clu, workers)
        sc[sid] = ac_stats.cluster_sc(sid, res, picks, ranges)
    for sid in ('SC2', 'SC4'):
        comp = ac_stats.comparator_arm(sid, picks)
        sc[sid] = ac_stats.retrieval_sc(sid, ret_of('cstar'), ret_of(comp), picks, ranges, None, draws_ret)
    assert list(sc) == ['SC1', 'SC3', 'SC2', 'SC4']
    return dict(n=int(len(keep)), n_tied_dropped=int(ac_paths.N['test'] - len(keep)) if not _smoke() else None,
                draws_clu=draws_clu, draws_ret=draws_ret,
                dropped_queries={arm: int(r['dropped']) for arm, r in ret.items()},
                sc={k: R.sc_dump(sc[k]) for k in ('SC1', 'SC2', 'SC3', 'SC4')})


def _stage1_tables(scores, picks):
    """Every stage 1 validation row with a `picked` flag: read-outs (per arm), teacher wordings, CRL versions."""
    aff = picks['affection']
    out = {'readout': {}, 'teacher_wording': {}, 'crl_version': {}}
    for ds, arms in scores['readout'].items():
        out['readout'][ds] = {}
        for arm, table in arms.items():
            pick = picks[ds]['readout'][arm]
            out['readout'][ds][arm] = {name: dict(row, picked=bool(name == pick)) for name, row in table.items()}
    for task, by in scores['teacher_wording'].items():
        out['teacher_wording'][task] = {w: dict(value=v, picked=bool(w == aff['teacher_wording'][task])) for w, v in by.items()}
    for ds, by in scores['crl_version'].items():
        out['crl_version'][ds] = {v: dict(score=s, picked=bool(v == picks[ds]['crl_version'])) for v, s in by.items()}
    return out


def diag(args):
    R = _run()
    _need_files()
    o = _out()
    s2, s3 = R.read_json(o / 'stage2/verdict.json'), R.read_json(o / 'stage3/verdict.json')
    scores, picks = R.read_json(o / 'stage1/scores.json'), R.read_json(o / 'stage1/picks.json')
    assert s2['smoke'] == s3['smoke'] == _smoke(), ('the verdict files are from another smoke setting', s2['smoke'], s3['smoke'])
    ranges = R.load_ranges()
    aff = picks['affection']
    rows2 = s2['rows']
    assert set(rows2) == set(ac_data.ARMS), list(rows2)
    labels, dist = ac_data.labels('test'), ac_data.emotion_dist('test')
    src = ac_data.sources('test')
    assert len(src) == len(labels) == len(dist) and dist.shape[1] == 9
    workers = args.workers
    doc = dict(smoke=_smoke(), reading=dict(
        untied='the evaluation set is the test images with no plurality tie; partitions refit on it; queries and gallery '
               'are that set; read-outs (fit on validation) and r unchanged; same draws as the stage 2 run',
        per_source_map='stage 2 per-query AP grouped by the query rows (rows), gallery = all test images; mean over seeds '
                       'first for a seeded arm',
        ndcg='overlap gain, no query expansion; picked = the arm\'s picked read-out transform applied, then plain cosine',
        rows_not_picked='all stage 1 validation rows (read-outs, wordings, CRL versions) with a picked flag; test rows '
                        'stage 2 and 3 scored for the wordings and CRL versions not picked; test numbers for read-outs not '
                        'picked are not computed',
        percept_core='AMI in both readings (r20 = r10 k=20 partitions of the L2-normalised probabilities; own = argmax) '
                     'beside PercepT\'s published numbers, which use another protocol'))
    # R0 rows, every arm (stage 2 files)
    doc['r0_rows'] = {arm: dict(map_r0=r['map_r0'], map_picked=r['map_picked'], readout=r['readout'], ami=r['ami'],
                                ndcg_r0=r['ndcg_r0']) for arm, r in rows2.items()}
    # graded nDCG, R0 and picked
    nd = {}
    for arm in rows2:
        r0 = _ndcg(arm, None, dist, workers)
        assert abs(np.mean(r0) - rows2[arm]['ndcg_r0']) < 1e-9, (arm, np.mean(r0), rows2[arm]['ndcg_r0'])
        pk = _ndcg(arm, ac_select.readout_from_name(aff['readout'][arm]), dist, workers)
        nd[arm] = dict(readout=aff['readout'][arm], r0=dict(mean=_f(np.mean(r0)), per_seed=r0),
                       picked=dict(mean=_f(np.mean(pk)), per_seed=pk))
    doc['ndcg'] = nd
    # the switch column
    doc['switch_column'] = _switch(labels, workers)
    assert np.allclose(doc['switch_column']['emotion']['per_seed'], rows2['cstar']['map_r0']['per_seed'], atol=1e-9)
    # per-source mAP and per-class AP from stage 2's per-query arrays, by rows
    teacher = f'teacher_{aff["teacher_wording"]["retrieval"]}'
    doc['per_source_map'], doc['per_class_ap'] = {}, {}
    for arm in rows2:
        doc['per_source_map'][arm] = {}
        for kind, r0 in (('picked', False), ('r0', True)):
            ap, rows = _perquery(2, arm, r0)
            doc['per_source_map'][arm][kind] = _per_source(ap, rows, src)
    for arm in ('cstar', teacher):
        doc['per_class_ap'][arm] = {}
        for kind, r0 in (('picked', False), ('r0', True)):
            ap, rows = _perquery(2, arm, r0)
            doc['per_class_ap'][arm][kind] = _per_class(ap, rows, labels)
    # c*'s per-seed spread of AMI and mAP beside r
    ami = []
    for s in SEEDS:
        with np.load(o / f'partitions/affection__test__cstar__s{s}.npz', allow_pickle=False) as z:
            p = z['labels']
        assert p.shape == (3, len(labels)) and p.dtype == np.int32, p.shape
        ami.append(_f(ac_cluster.ami(p, labels)))
    spread = lambda v, r: dict(per_seed=v, spread=max(v) - min(v), r=_f(r))
    doc['cstar_seed_spread'] = dict(
        ami=spread(ami, ranges['clu']['r']), map_picked=spread(rows2['cstar']['map_picked']['per_seed'], ranges['map']['r']),
        map_r0=spread(rows2['cstar']['map_r0']['per_seed'], ranges['map']['r']))
    assert abs(np.mean(ami) - rows2['cstar']['ami']) < 1e-9, (np.mean(ami), rows2['cstar']['ami'])
    # rows not picked
    doc['rows_not_picked'] = dict(
        stage1_validation=_stage1_tables(scores, picks),
        teacher_wordings_test={w: dict(picked=bool(w == aff['teacher_wording']['clustering']),
                                       picked_for_retrieval=bool(w == aff['teacher_wording']['retrieval']),
                                       **rows2[f'teacher_{w}']) for w in ('artwork', 'photo')},
        crl_versions_test={v: dict(picked=bool(v == aff['crl_version']), **s3['rows']['affection'][f'crl_{v}'])
                           for v in ac_paths.CRL_VERSIONS})
    doc['chance_map'] = rt_stats.chance(labels)
    pc = rows2['percept']
    doc['percept_core'] = dict(ami_r20=pc['ami'], ami_own=pc['ami_own'], published=list(PERCEPT_PUBLISHED),
                               protocol='other protocol (published numbers, not our test split)')
    # untied test images: SC1 to SC4 again, with stage 2's draws
    doc['untied'] = _untied(picks, ranges, s2['draws_clu'], s2['draws_ret'], workers)
    doc['inputs'] = dict(stage2_draws=dict(clu=s2['draws_clu'], ret=s2['draws_ret']), verdict_stage2=s2['verdict'])
    R.write_json(o / 'diag/diag.json', doc)
    return doc


# ---- stage 4: Impressions -----------------------------------------------------------------------------------------------

def impressions_choice():
    """D13's rule over the store's images. Returns (chosen {uuid: file}, store {files, rows, info}). One image per UUID:
    the file without a headline overlay, else the first overlay file in file-name order. The files are the Impressions
    store's (every image with a stored embedding), checked against /project/Impressions/media."""
    store = Path(ac_paths.IMPRESSIONS_STORE)
    index = json.loads((store / 'index.json').read_text())
    assert index['dataset'] == 'impressions' and index['dtype'] == 'float16' and index['dim'] == 2048
    imgs = index['images']
    files = [r['image'] for r in imgs]
    assert len(files) == len(set(files)) == index['n_images']
    media = Path(ac_paths.IMPRESSIONS_DIR) / 'media'
    for r in imgs:                                                        # the field and the file name agree
        assert re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(_cap\d+_\d+)?\.png', r['image']), r
        assert r['image'].startswith(r['uuid']) and r['has_overlay'] == ('_cap' in r['image']), r
    missing = [f for f in files if not (media / f).is_file()]
    assert not missing, ('files in the store but not under', str(media), missing[:3])
    by = {}
    for r in imgs:
        by.setdefault(r['uuid'], []).append(r)
    return by, index


def _choose(by, available):
    chosen, n_plain = {}, 0
    for u in sorted(by):
        cand = sorted(r['image'] for r in by[u] if r['image'] in available)
        if not cand:
            continue
        plain = [f for f in cand if '_cap' not in f]
        n_plain += bool(plain)
        chosen[u] = plain[0] if plain else cand[0]
    return chosen, n_plain


def _cos_rank(q, g, top):
    sim = q @ g.T
    assert sim.shape == (len(q), len(g))
    order = np.argsort(-sim, axis=1, kind='stable')[:, :top]            # ties keep gallery (sorted UUID) order
    return order, np.take_along_axis(sim, order, axis=1)


def _figure(doc, path, media):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    nq = len(doc['queries'])
    fig = plt.figure(figsize=(2.6 * (TOP + 1) + 0.6, 2.35 * 2 * nq + 0.9), dpi=100)
    gs = fig.add_gridspec(2 * nq, TOP + 1, left=0.06, right=0.99, top=1 - 0.7 / (2 * 2.35 * nq + 0.9),
                          bottom=0.005, wspace=0.06, hspace=0.34)
    fig.suptitle('Impressions: top 5 by R0 cosine, c* (emotion head, seed 0) and the plain image embedding', fontsize=15,
                 y=1 - 0.12 / (2 * 2.35 * nq + 0.9))

    def show(ax, file, title, color='black'):
        im = Image.open(media / file).convert('RGB')
        ax.imshow(im)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(color), sp.set_linewidth(2)
        ax.set_title(title, fontsize=11, pad=3)

    for i, q in enumerate(doc['queries']):
        ax = fig.add_subplot(gs[2 * i:2 * i + 2, 0])
        show(ax, q['file'], f'query {i + 1}: {q["uuid"][:8]}', 'tab:red')
        for j, name in enumerate(('cstar', 'plain')):
            for k, hit in enumerate(q[name]):
                ax = fig.add_subplot(gs[2 * i + j, k + 1])
                show(ax, hit['file'], f'{hit["uuid"][:8]}  cos {hit["cos"]:.3f}')
                if k == 0:
                    ax.text(0.03, 0.97, 'c*' if name == 'cstar' else 'plain', transform=ax.transAxes, fontsize=13, va='top', ha='left', bbox=dict(fc='white', ec='black', pad=2))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def stage4(args):
    R = _run()
    smoke = bool(_smoke())
    by, index = impressions_choice()
    all_files = [r['image'] for r in index['images']]
    exp = _out() / 'exports/cstar/impressions/emotion__s0'
    side = json.loads(exp.with_suffix('.json').read_text())
    with np.load(exp.with_suffix('.npz'), allow_pickle=False) as z:
        assert z.files == ['img'], z.files
        img = z['img']
    names = side['keys']
    assert side['smoke'] is smoke and side['split'] == 'impressions' and side['seed'] == 0 and side['aspect'] == 'emotion', side
    assert img.shape == (len(names), 2048) and img.dtype == np.float16 and len(names) == len(set(names)), img.shape
    assert side['keys_sha256'] == rs_vh.names_sha(names) and side['array_sha256'] == rs_vh.array_sha(img), 'export hashes'
    assert set(names) <= set(all_files)
    assert smoke or names == all_files, 'a full export holds every store image, in store order'
    # c* rows by file name
    xc = {f: i for i, f in enumerate(names)}
    chosen, n_plain = _choose(by, set(names))
    uuids = sorted(chosen)
    found = len(uuids)
    assert found > N_QUERIES, ('too few UUIDs', found)
    assert smoke or found == len(by), (found, len(by))
    qu, gu = uuids[:N_QUERIES], uuids[N_QUERIES:]
    # plain rows from the default store, by file name
    store = Path(ac_paths.IMPRESSIONS_STORE)
    raw = np.load(store / 'img.npy', mmap_mode='r', allow_pickle=False)
    assert raw.shape == (index['n_images'], 2048) and raw.dtype == np.float16
    pos = {f: i for i, f in enumerate(all_files)}
    unit = lambda x: ac_data._unit(np.asarray(x, dtype=np.float32))
    files = [chosen[u] for u in uuids]
    plain = unit(raw[np.asarray([pos[f] for f in files])])
    cstar = unit(img[np.asarray([xc[f] for f in files])])
    assert plain.shape == cstar.shape == (found, 2048)
    qi, gi = np.arange(N_QUERIES), np.arange(N_QUERIES, found)
    doc = dict(smoke=smoke, rule='D13: one image per UUID, the file without a headline overlay, else the first overlay file '
                                'in file-name order; queries the first 6 UUIDs sorted; gallery the others; R0 cosine; top 5',
               n_store_files=len(all_files), n_export_files=len(names), n_uuids=found, spec_n_uuids=SPEC_UUIDS,
               matches_spec=bool(found == SPEC_UUIDS), n_queries=len(qu), n_gallery=len(gu),
               n_uuids_with_plain_file=int(n_plain), n_uuids_overlay_only=int(found - n_plain),
               media_dir=str(Path(ac_paths.IMPRESSIONS_DIR) / 'media'), queries=[])
    ranks = {}
    for name, x in (('cstar', cstar), ('plain', plain)):
        order, sim = _cos_rank(x[qi], x[gi], TOP)
        ranks[name] = ([[gu[j] for j in row] for row in order], sim)
    for i, u in enumerate(qu):
        q = dict(uuid=u, file=chosen[u])
        for name in ('cstar', 'plain'):
            q[name] = [dict(uuid=ranks[name][0][i][k], file=chosen[ranks[name][0][i][k]], cos=_f(ranks[name][1][i, k]))
                       for k in range(TOP)]
            assert len(q[name]) == TOP and len({h['uuid'] for h in q[name]}) == TOP and u not in {h['uuid'] for h in q[name]}
        doc['queries'].append(q)
    assert len(doc['queries']) == N_QUERIES and len(gu) == found - N_QUERIES
    o = _out() / 'stage4'
    R.write_json(o / 'impressions.json', doc)
    _figure(doc, o / 'impressions.png', Path(doc['media_dir']))
    print(f'  stage 4: {found} UUIDs (spec says {SPEC_UUIDS}), {len(qu)} queries, {len(gu)} gallery', flush=True)
    return doc
