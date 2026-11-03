"""r17 ticket 11: stage 3 diagnostics (spec §3.6, D2, D3, D5, D10, D11). Descriptive: nothing here decides anything.

One CLI item per diagnostic, so the runner queues them separately (each is its own plain process; `ns_clu.paired_cell`
opens a fork pool, so never call an item from a pool worker). Outputs are JSON under OUT/stage3/.

  sweep --set S --enc E [--arms ..] [--aspects ..] [--counts ..]
        AMI and ARI at `ns_paths.SWEEP_COUNTS` (D2) for c* (photo model on the photo sets), the plain image, matched
        PercepT (k-means on the 40 probabilities, painting sets), and (once per set) the teacher (in the Qwen item) and
        CRL (in the CLIP L/14 item). Point values: the mean over 5 seeds with the minimum and maximum over seeds; an
        unseeded arm has one value (mean over its 3 k-means states). At the class count K, c* also gets NMI and ACC
        (the same partitions) for the seed spreads. -> OUT/stage3/sweep/<set>__<enc>.json
  figure --set S           one PNG per set from the sweep files -> OUT/stage3/figures/sweep_<set>.png
  per-class --set S --enc E     AP per emotion class, c* minus the teacher, paired intervals over that class's queries and
        c*'s seeds (10,000 draws), class sizes beside (D11) -> OUT/stage3/per_class/<set>__<enc>.json
  qwen-crl                  Qwen c* against CRL on CLIP L/14, MMArt test (D10): clustering at 20 and at K, retrieval;
        intervals, no outcome word -> OUT/stage3/qwen_crl_mmart.json
  told-untold --set S --enc E   untold minus told c* per aspect, one shared seed draw (D8) -> OUT/stage3/told_untold/
  r0                        R0 rows of the new retrieval arms (from the stage 1 picks record) -> OUT/stage3/r0_rows.json
  tables --set S            each class-count cell beside its AMI-at-20 value, and the test seed spreads beside the
        validation ranges -> OUT/stage3/tables_<set>.json (reads stage 2 cells, the sweep files and the stage 1 ranges)

Wording rule (D5): a lower point value at a count is "lower point value", no verdict word. No item writes an outcome.
Smoke (NS_SMOKE=1, NS_OUTPUTS=<folder>) uses `ns_paths.SMOKE_DRAWS` and accepts --counts / --aspects to cut the work.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as P  # noqa: E402

P.add_rounds()
import ns_clu  # noqa: E402
import ns_clu_cells  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402
import ns_photo  # noqa: E402
import ns_ret  # noqa: E402

STAGE3 = 'stage3'
SWEEP_ARMS_PAINTING = ('cstar', 'plain', 'matched', 'teacher', 'crl')
SWEEP_ARMS_PHOTO = ('photo', 'plain', 'teacher')
ARM_LABEL = {'cstar': 'c*', 'photo': 'photo model', 'plain': 'plain image', 'matched': 'matched PercepT',
             'teacher': 'teacher', 'crl': 'CRL'}


def _out(*parts):
    return P.OUT.joinpath(STAGE3, *parts)


def _draws(task):
    return (P.SMOKE_DRAWS if P.SMOKE else P.N_DRAWS)[task]


def _check_set_enc(set_, enc):
    assert set_ in P.HELD_SPLIT, set_
    assert enc in P.ENCODERS, enc
    if set_ in P.PHOTO_SETS:
        assert enc in P.PHOTO_ENCODERS[set_], f'{set_} is scored for {P.PHOTO_ENCODERS[set_]}'


def default_arms(set_, enc):
    """The arms one sweep item computes: teacher only in the Qwen item (it is encoder-free), CRL only in CLIP L/14's."""
    arms = list(SWEEP_ARMS_PHOTO if set_ in P.PHOTO_SETS else SWEEP_ARMS_PAINTING)
    if enc != 'qwen':
        arms.remove('teacher')
    if 'crl' in arms and enc != P.CRL_ENCODER:
        arms.remove('crl')
    return tuple(arms)


# ================================================================================================ count sweep
def arm_parts(set_, enc, aspect, arm, k):
    """int32 [5, 3, n] (c*, photo, matched: seeds) or [3, n] (plain, teacher, CRL). The verdict code's own paths:
    ns_clu_cells builds the rows in each earlier round's form and calls `ns_metrics.partitions` / `percept_partitions`."""
    split = P.HELD_SPLIT[set_]
    if arm in ('cstar', 'photo'):
        return ns_clu_cells._cand_parts(set_, split, enc, aspect, k)
    return ns_clu_cells._comp_parts(set_, split, enc, aspect, arm, k)


def _row(arm, aspect, k, metric, per_seed, seeded):
    v = [float(x) for x in per_seed]
    return dict(arm=arm, aspect=aspect, k=int(k), metric=metric, per_seed=v, mean=float(np.mean(v)), min=float(min(v)),
                max=float(max(v)), seeded=bool(seeded))


def sweep(set_, enc, arms=None, aspects=None, counts=None, out_path=None, log=print):
    """Count sweep of one (set, encoder) item (module docstring). `counts` {aspect: tuple} overrides the grid (smoke)."""
    _check_set_enc(set_, enc)
    arms = tuple(arms or default_arms(set_, enc))
    aspects = tuple(aspects or P.SET_ASPECTS[set_])
    counts = {a: tuple((counts or {}).get(a, P.SWEEP_COUNTS[a])) for a in aspects}
    split = P.HELD_SPLIT[set_]
    path = Path(out_path) if out_path else _out('sweep', f'{set_}__{enc}.json')
    rows, fits, t0 = [], 0, time.monotonic()
    doc = dict(set=set_, encoder=enc, split=split, arms=list(arms), counts={a: list(counts[a]) for a in aspects},
               class_count={a: P.CLASS_COUNT[(set_, a)] for a in aspects}, kmeans_threads=P.KMEANS_THREADS[set_],
               rows=rows, complete=False, smoke=bool(P.SMOKE))
    for aspect in aspects:
        y = ns_clu_cells._labels(set_, split, aspect)
        K = P.CLASS_COUNT[(set_, aspect)]
        assert ns_metrics.class_count(y) == K, (set_, aspect, ns_metrics.class_count(y), K)
        for k in counts[aspect]:
            for arm in arms:
                t1 = time.monotonic()
                parts = arm_parts(set_, enc, aspect, arm, k)
                seeded = parts.ndim == 3
                fits += (parts.shape[0] if seeded else 1) * parts.shape[-2]
                metrics = P.CLASS_METRICS if (k == K and arm in ('cstar', 'photo')) else P.SWEEP_METRICS
                for t in ns_clu.metric_table(arm, aspect, parts, y, k, metrics=metrics):
                    rows.append(_row(arm, aspect, k, t['metric'], t['per_seed'], seeded))
                log(f'  {set_}/{enc}/{aspect} k={k} {arm}: {time.monotonic() - t1:.0f}s')
        doc.update(kmeans_runs=fits, seconds=time.monotonic() - t0)
        P.write_json(path, doc)                       # partial file after every aspect
    doc.update(complete=True, kmeans_runs=fits, seconds=time.monotonic() - t0)
    P.write_json(path, doc)
    return doc


def figure(set_, out_path=None):
    """One PNG per set from the sweep files: rows (aspect, metric), columns encoders; lines per arm, band min to max over
    seeds; teacher (Qwen item) and CRL (CLIP L/14 item) drawn in every column; dotted line at the class count."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    encs = P.PHOTO_ENCODERS[set_] if set_ in P.PHOTO_SETS else P.ENCODERS
    docs, missing = {}, []
    for e in encs:
        f = _out('sweep', f'{set_}__{e}.json')
        if f.exists():
            docs[e] = json.loads(f.read_text())
        else:
            missing.append(e)
    aspects = P.SET_ASPECTS[set_]
    metrics = P.SWEEP_METRICS
    rows_ax = [(a, m) for a in aspects for m in metrics]
    fig, axes = plt.subplots(len(rows_ax), len(encs), figsize=(4.0 * len(encs) + 1.5, 2.6 * len(rows_ax) + 1.2),
                             squeeze=False, sharex='row')
    colour = {'cstar': 'tab:blue', 'photo': 'tab:blue', 'plain': 'tab:gray', 'matched': 'tab:green',
              'teacher': 'tab:red', 'crl': 'tab:purple'}
    shared = {}
    for d in docs.values():
        for a in ('teacher', 'crl'):
            if a in d['arms']:
                shared[a] = d

    def series(doc, arm, aspect, metric):
        r = sorted((x for x in doc['rows'] if x['arm'] == arm and x['aspect'] == aspect and x['metric'] == metric),
                   key=lambda x: x['k'])
        return r
    for i, (aspect, metric) in enumerate(rows_ax):
        for j, e in enumerate(encs):
            ax = axes[i][j]
            if i == 0:
                ax.set_title(P.ENCODER_NAME[e], fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{aspect} {metric.upper()}', fontsize=8)
            if i == len(rows_ax) - 1:
                ax.set_xlabel('clusters', fontsize=8)
            if e not in docs:
                ax.text(0.5, 0.5, 'not run', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            for arm in docs[e]['arms'] + [a for a in shared if a not in docs[e]['arms']]:
                src = docs[e] if arm in docs[e]['arms'] else shared[arm]
                r = series(src, arm, aspect, metric)
                if not r:
                    continue
                ks = [x['k'] for x in r]
                ax.plot(ks, [x['mean'] for x in r], marker='o', ms=3, color=colour[arm],
                        ls='--' if arm in ('teacher', 'crl') else '-', label=ARM_LABEL[arm])
                if r[0]['seeded']:
                    ax.fill_between(ks, [x['min'] for x in r], [x['max'] for x in r], color=colour[arm], alpha=0.25, lw=0)
            ax.axvline(P.CLASS_COUNT[(set_, aspect)], color='k', ls=':', lw=0.8)
            ax.set_xticks(P.SWEEP_COUNTS[aspect])
            ax.tick_params(labelsize=7)
    handles, labels = {}, {}
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles), loc='lower center', ncol=len(handles), fontsize=8)
    title = f'Count sweep, {set_} ({P.HELD_SPLIT[set_]}): mean over 5 seeds, band min to max; dotted: class count'
    if missing:
        title += f'; not run: {", ".join(missing)}'
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path = Path(out_path) if out_path else _out('figures', f'sweep_{set_}.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ================================================================================================ retrieval helpers
def _teacher_pick(set_):
    if set_ == 'artelingo':
        return json.loads((P.R11_OUT / 'stage4/picks.json').read_text())['instructed']
    return json.loads((P.R15_OUT / 'stage1/picks_teacher.json').read_text())['arms']['teacher']['pick']


def _teacher_getter(set_, aspect, split):
    def get(s):
        x = ns_data.teacher(set_, split, aspect)
        return ns_ret._unit(x) if set_ == 'mmart' else x            # r15's retrieval form divides once more
    return get


def _picks_record():
    return P.read_committed_record('stage1_picks.json')


def _ap(rows, arm, set_, aspect, y, box, workers):
    pick = _teacher_pick(set_) if arm == 'teacher' else ns_ret._pick(arm, set_, rows.enc, box)
    if arm == 'teacher':
        split = rows.held
        return ns_ret.arm_ap(_teacher_getter(set_, aspect, 'val'), _teacher_getter(set_, aspect, split), pick, y, None,
                             workers), pick
    return rows.ap(arm, aspect, pick, y, workers), pick


# ================================================================================================ emotion per class
def per_class(set_, enc, workers=8, draws=None):
    """AP per emotion class (D11): c* minus the teacher; per class a paired interval over that class's queries and c*'s
    seeds (`ns_ret.paired_ret`, comparator unseeded, r None). Class sizes: labelled rows and queries of that class."""
    assert set_ in P.PAINTING_SETS and enc in P.ENCODERS
    draws = draws or _draws('ret')
    aspect = 'emotion'
    y = ns_data.labels(set_, P.HELD_SPLIT[set_], aspect)
    rows, box = ns_ret._Rows(set_, enc), [None]
    cand, pc = _ap(rows, 'cstar', set_, aspect, y, box, workers)
    teach, pt = _ap(rows, 'teacher', set_, aspect, y, box, workers)
    assert np.array_equal(cand['rows'], teach['rows']), 'c* and the teacher were scored on other queries'
    qlab = y[cand['rows']]
    out = []
    for c in sorted(set(qlab.tolist())):
        sel = qlab == c
        p = ns_ret.paired_ret(cand['ap'][:, sel], teach['ap'][sel], r=None, draws=draws, comp_seeds='none')
        out.append(dict(label=int(c), n_labelled=int((y == c).sum()), n_queries=int(sel.sum()), cstar_ap=p['cand'],
                        teacher_ap=p['comp'], d=p['d'], lo=p['lo'], hi=p['hi'], n_draws=p['n_draws']))
    doc = dict(set=set_, encoder=enc, aspect=aspect, cstar_pick=pc, teacher_pick=pt, cstar_map=cand['point'],
               teacher_map=teach['point'], n_queries=int(len(cand['rows'])), classes=out, smoke=bool(P.SMOKE))
    P.write_json(_out('per_class', f'{set_}__{enc}.json'), doc)
    return doc


# ================================================================================================ Qwen c* against CRL
def qwen_crl(workers=8, clu_draws=None, ret_draws=None, aspects=None):
    """Qwen c* against CRL on CLIP L/14, MMArt test (D10). Clustering at 20 (AMI, ARI) and at the class count K (all four
    metrics) per aspect through `ns_clu.paired_cell` (r None); retrieval through `ns_ret.paired_ret` (r None, unseeded
    comparator). Intervals only: no outcome word is written."""
    set_, enc, split = 'mmart', 'qwen', 'test'
    clu_draws, ret_draws = clu_draws or _draws('clu'), ret_draws or _draws('ret')
    res = dict(set=set_, split=split, candidate='Qwen c*', comparator='CRL on CLIP L/14', clustering=[], retrieval=[],
               smoke=bool(P.SMOKE))
    cache, aspects = {}, tuple(aspects or P.ASPECTS)
    for aspect in aspects:
        y = ns_clu_cells._labels(set_, split, aspect)
        K = P.CLASS_COUNT[(set_, aspect)]
        for k, metrics in ((20, P.SWEEP_METRICS), (K, P.CLASS_METRICS)):
            if k == 20 and K == 20:
                continue                                              # MMArt style: K = 20, the K row covers it
            cand = cache.setdefault(('c', aspect, k), arm_parts(set_, enc, aspect, 'cstar', k))
            comp = cache.setdefault(('m', aspect, k), arm_parts(set_, P.CRL_ENCODER, aspect, 'crl', k))
            cell = ns_clu.paired_cell(cand, comp, y, k, r=None, metrics=metrics, draws=clu_draws)
            for m in metrics:
                v = cell[m]
                res['clustering'].append(dict(aspect=aspect, k=int(k), at='K' if k == K else '20', metric=m, cand=v['cand'],
                                              comp=v['comp'], d=v['d'], lo=v['lo'], hi=v['hi'], n_draws=v['n_draws']))
    rows_q, rows_c, box = ns_ret._Rows(set_, enc), ns_ret._Rows(set_, P.CRL_ENCODER), [None]
    for aspect in aspects:
        y = ns_data.labels(set_, split, aspect)
        cq, pq = _ap(rows_q, 'cstar', set_, aspect, y, box, workers)
        cc, pc = _ap(rows_c, 'crl', set_, aspect, y, box, workers)
        assert np.array_equal(cq['rows'], cc['rows'])
        p = ns_ret.paired_ret(cq['ap'], cc['ap'], r=None, draws=ret_draws, comp_seeds='none')
        res['retrieval'].append(dict(aspect=aspect, metric='map', cand=p['cand'], comp=p['comp'], d=p['d'], lo=p['lo'],
                                     hi=p['hi'], n_draws=p['n_draws'], cand_readout=pq, comp_readout=pc,
                                     n_queries=int(len(cq['rows']))))
    P.write_json(_out('qwen_crl_mmart.json'), res)
    return res


# ================================================================================================ told beside untold
def told_untold(set_, enc, workers=8, draws=None):
    """Told c* beside untold c* per aspect (D8): untold minus told with its paired interval; one shared seed draw (untold
    seed s is built from told seed s's runs), `ns_ret.paired_ret(..., comp_seeds='shared')`, r None."""
    assert set_ in P.PAINTING_SETS and enc in P.ENCODERS
    draws = draws or _draws('ret')
    rows, box, out = ns_ret._Rows(set_, enc), [None], []
    for aspect in P.ASPECTS:
        y = ns_data.labels(set_, P.HELD_SPLIT[set_], aspect)
        told, pt = _ap(rows, 'cstar', set_, aspect, y, box, workers)
        untold, pu = _ap(rows, 'untold', set_, aspect, y, box, workers)
        assert np.array_equal(told['rows'], untold['rows'])
        p = ns_ret.paired_ret(untold['ap'], told['ap'], r=None, draws=draws, comp_seeds='shared')
        out.append(dict(aspect=aspect, told=p['comp'], untold=p['cand'], untold_minus_told=p['d'], lo=p['lo'], hi=p['hi'],
                        n_draws=p['n_draws'], told_readout=pt, untold_readout=pu, n_queries=int(len(told['rows']))))
    doc = dict(set=set_, encoder=enc, cells=out, smoke=bool(P.SMOKE))
    P.write_json(_out('told_untold', f'{set_}__{enc}.json'), doc)
    return doc


# ================================================================================================ R0 rows
def r0_rows(record=None):
    """R0 (no map, no expansion: `none+k0`) validation mAP of each new retrieval arm per aspect, with the pick, its
    cross-fit mAP and the unit, from the stage 1 picks record. A missing pick is listed, not skipped."""
    record = _picks_record() if record is None else record
    import ns_picks
    rows, missing = [], []
    for arm, s, e in ns_picks.PICKS:
        key = ns_picks.key_of(arm, s, e)
        if key not in record:
            missing.append(key)
            continue
        rec = record[key]
        rows.append(dict(key=key, pick=rec['pick'], r0_map=rec['scores']['r0_map'], map_at_pick=rec['scores']['map'],
                         unit=rec['unit']))
    doc = dict(rows=rows, missing=missing, smoke=bool(P.SMOKE))
    P.write_json(_out('r0_rows.json'), doc)
    return doc


# ================================================================================================ tables
def _read_sweeps(set_):
    out = {}
    for e in (P.PHOTO_ENCODERS[set_] if set_ in P.PHOTO_SETS else P.ENCODERS):
        f = _out('sweep', f'{set_}__{e}.json')
        if f.exists():
            out[e] = json.loads(f.read_text())
    return out


def tables(set_, cells=None, ranges=None):
    """(a) each class-count cell of stage 2 beside the AMI at 20 clusters: the cell's own candidate and comparator arms'
    AMI at 20 from the sweep (point value, mean over seeds) and, for SC5, the stage 2 AMI-at-20 cell; (b) test seed
    spreads of the candidate (max minus min over 5 seeds at the class count, from the sweep files) beside the stage 1
    validation range r of the same key. Missing inputs are listed under `missing`."""
    cells = json.loads((P.OUT / 'stage2/cells.json').read_text()) if cells is None else cells
    ranges = P.read_committed_record('stage1_ranges.json') if ranges is None else ranges
    sweeps = _read_sweeps(set_)
    def at20(enc, arm, aspect, metric='ami'):
        for src in sweeps.values():                      # teacher and CRL live in one item per set
            if arm in src['arms'] and (src['encoder'] == enc or arm in ('teacher', 'crl')):
                for x in src['rows']:
                    if (x['arm'], x['aspect'], x['metric'], x['k']) == (arm, aspect, metric, 20):
                        return x['mean']
        return None
    beside, spreads, missing = [], [], []
    ami20 = {(c['set'], c['encoder'], c['aspect']): c for c in cells if c['set'] == set_ and '/ami@20/' in c['id']}
    for c in cells:
        if c['set'] != set_ or c['task'] != 'clu' or '/ami@20/' in c['id']:
            continue
        a20_cand, a20_comp = at20(c['encoder'], c['candidate'], c['aspect']), at20(c['encoder'], c['comparator'], c['aspect'])
        s5 = ami20.get((c['set'], c['encoder'], c['aspect'])) if c['sc'] == 'SC5' else None
        beside.append(dict(id=c['id'], metric=c['metric'], k=c['k'], cand=c['cand'], comp=c['comp'], d=c['d'],
                           lo=c['lo'], hi=c['hi'], ami_at_20_cand=a20_cand, ami_at_20_comp=a20_comp,
                           sc5_ami_at_20_cell=None if s5 is None else dict(cand=s5['cand'], comp=s5['comp'], d=s5['d'],
                                                                           lo=s5['lo'], hi=s5['hi'])))
        if a20_cand is None:
            missing.append(f'sweep AMI at 20: {set_}/{c["encoder"]}/{c["aspect"]}/{c["candidate"]}')
    cand_arm = 'photo' if set_ in P.PHOTO_SETS else 'cstar'
    for enc, doc in sweeps.items():
        for aspect in doc['counts']:
            K = P.CLASS_COUNT[(set_, aspect)]
            for m in P.CLASS_METRICS:
                r = [x for x in doc['rows'] if (x['arm'], x['aspect'], x['metric'], x['k']) == (cand_arm, aspect, m, K)]
                key = f'{"photo" if set_ in P.PHOTO_SETS else "cstar"}/{set_}/{enc}/{aspect}/{m}'
                if not r:
                    missing.append(f'sweep {key} at K')
                    continue
                spread = r[0]['max'] - r[0]['min']
                val = ranges.get(key)
                spreads.append(dict(key=key, k=K, test_per_seed=r[0]['per_seed'], test_spread=spread,
                                    validation_range=None if val is None else val['r'],
                                    ratio=None if val is None or not val['r'] else spread / val['r']))
                if val is None:
                    missing.append(f'stage 1 range {key}')
    doc = dict(set=set_, class_count_beside_ami20=beside, seed_spreads=spreads, missing=missing, smoke=bool(P.SMOKE))
    P.write_json(_out(f'tables_{set_}.json'), doc)
    return doc


# ================================================================================================ CLI
def _counts(arg):
    """'emotion=5,9;genre=10' -> {aspect: tuple}; None -> None."""
    if not arg:
        return None
    return {kv.split('=')[0]: tuple(int(v) for v in kv.split('=')[1].split(',')) for kv in arg.split(';')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sweep')
    s.add_argument('--set', required=True, choices=sorted(P.HELD_SPLIT))
    s.add_argument('--enc', required=True, choices=P.ENCODERS)
    s.add_argument('--arms', nargs='+', default=None)
    s.add_argument('--aspects', nargs='+', default=None)
    s.add_argument('--counts', default=None, help="smoke only: 'emotion=5,9;genre=10'")
    f = sub.add_parser('figure')
    f.add_argument('--set', required=True, choices=sorted(P.HELD_SPLIT))
    for name in ('per-class', 'told-untold'):
        c = sub.add_parser(name)
        c.add_argument('--set', required=True, choices=P.PAINTING_SETS)
        c.add_argument('--enc', required=True, choices=P.ENCODERS)
    q = sub.add_parser('qwen-crl')
    q.add_argument('--aspects', nargs='+', default=None, choices=P.ASPECTS, help='smoke only')
    sub.add_parser('r0')
    t = sub.add_parser('tables')
    t.add_argument('--set', required=True, choices=sorted(P.HELD_SPLIT))
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    if args.cmd == 'sweep':
        if args.counts:
            assert P.SMOKE, '--counts is for smoke runs'
        d = sweep(args.set, args.enc, args.arms, args.aspects, _counts(args.counts))
        print(f'sweep {args.set}/{args.enc}: {d["kmeans_runs"]} k-means runs, {d["seconds"]:.0f}s')
    elif args.cmd == 'figure':
        print(figure(args.set))
    elif args.cmd == 'per-class':
        d = per_class(args.set, args.enc)
        print(f'per-class {args.set}/{args.enc}: {len(d["classes"])} classes')
    elif args.cmd == 'told-untold':
        told_untold(args.set, args.enc)
    elif args.cmd == 'qwen-crl':
        assert not args.aspects or P.SMOKE, '--aspects is for smoke runs'
        qwen_crl(aspects=args.aspects)
    elif args.cmd == 'r0':
        d = r0_rows()
        print(f'r0 rows: {len(d["rows"])}, missing {d["missing"]}')
    elif args.cmd == 'tables':
        d = tables(args.set)
        print(f'tables {args.set}: {len(d["class_count_beside_ami20"])} beside, {len(d["seed_spreads"])} spreads, missing {len(d["missing"])}')
    print(f'{args.cmd} done in {time.monotonic() - t0:.0f}s')


if __name__ == '__main__':
    main()
