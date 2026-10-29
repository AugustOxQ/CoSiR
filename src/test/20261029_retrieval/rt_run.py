"""r11 retrieval: the stage runner (spec 2026-10-29-retrieval-design.md, 3.4 to 3.6 and 6, amendment A1).

  python rt_run.py stage1 stage2 stage3 stage4 stage5 diag [--outputs DIR] [--draws 10000] [--workers 8]
                   [--smoke N [--arms a,b,...]]

Wiring only: the metrics come from rt_metrics, the read-outs from rt_readout, the bootstrap, outcomes and the SC table
from rt_stats, the data from rt_data. Inputs (rt_data arrays, the PercepT validation export, the CRL files) come from
rt_paths.OUTPUTS; every file written goes under --outputs. Held stages (1, 2, 3, 5 and diag) refuse to start unless
the committed seed ranges are tracked by git, unchanged against HEAD and equal to <outputs>/stage0a/ranges.json
(skipped only with --smoke). Stage 4 reads validation data only.

Files (under --outputs): stage{1,2,3,5}/verdict.json, stage2/switch.json, stage{1,2,3,5}/perquery/*.npz,
stage4/scores.json, stage4/picks.json, diag/diag.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import rt_paths
import rt_data
import rt_metrics
import rt_readout
import rt_stats
import rs_vh

ASPECTS, SEEDS = rt_paths.ASPECTS, rt_paths.SEEDS
STAGES = ('stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'diag')
HELD_STAGES = ('stage1', 'stage2', 'stage3', 'stage5', 'diag')
REQUIRED_ARMS = ('cstar', 'plain', 'instructed')       # the SC table's arms


# ---- inputs ---------------------------------------------------------------------------------------------------------

class Loader:
    """Every rt_data call goes here: smoke slicing, the stage's split rule and the input hashes."""

    def __init__(self, smoke, splits):
        self.smoke, self.splits, self.hashes, self._labels, self._dist = smoke, splits, {}, {}, {}

    def _cut(self, x):
        return x if self.smoke is None else x[:self.smoke]

    def _check(self, split):
        assert split in self.splits, f'this stage may read only {self.splits}, not {split!r}'

    def labels(self, split):
        self._check(split)
        if split not in self._labels:
            lab = {a: self._cut(x) for a, x in rt_data.labels(split).items()}
            for a, x in lab.items():
                self.hashes[f'labels/{split}/{a}'] = rs_vh.array_sha(x)
            self._labels[split] = lab
        return self._labels[split]

    def emotion_dist(self, split):
        self._check(split)
        if split not in self._dist:
            self._dist[split] = self._cut(rt_data.emotion_dist(split))
            self.hashes[f'emotion_dist/{split}'] = rs_vh.array_sha(self._dist[split])
        return self._dist[split]

    def embed(self, split, arm, head, seed):
        """Float32 L2-normalized rows of the arm at the head (None for an aspect-free arm), seed None if unseeded."""
        self._check(split)
        assert (head is None) == (arm in rt_data.ASPECT_FREE), (arm, head)
        assert (seed is None) == (arm not in rt_data.SEEDED), (arm, seed)
        x = self._cut(rt_data.embed(arm, split, head or 'emotion', seed))
        n = len(self.labels(split)['emotion'])
        assert x.dtype == np.float32 and x.ndim == 2 and len(x) == n, (arm, x.shape, x.dtype)
        key = f'embed/{split}/{arm}' + ('' if head is None else f'/{head}') + ('' if seed is None else f'/s{seed}')
        self.hashes[key] = rs_vh.array_sha(x)
        return x


# ---- guard and files ------------------------------------------------------------------------------------------------

def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def guard(args, need_picks=False):
    """Returns '' when a held stage may run, else the reason it must be refused."""
    ranges_out = Path(args.outputs) / 'stage0a/ranges.json'
    if not ranges_out.is_file():
        return f'{ranges_out} is missing (run rt_stage0.py 0a first)'
    if need_picks and not (Path(args.outputs) / 'stage4/picks.json').is_file():
        return f'{Path(args.outputs) / "stage4/picks.json"} is missing (run stage4 first)'
    if args.smoke is not None:
        return ''
    f = rt_paths.RANGES_FILE
    if not f.is_file():
        return f'{f} does not exist (stage 0a has not written it)'
    cwd = str(f.parent)
    if subprocess.run(['git', 'ls-files', '--error-unmatch', str(f)], cwd=cwd, capture_output=True).returncode:
        return f'{f} is not tracked by git: commit the seed ranges before any held stage'
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', str(f)], cwd=cwd, capture_output=True).returncode:
        return f'{f} differs from HEAD: commit the seed ranges before any held stage'
    if read_json(f) != read_json(ranges_out):
        return f'{f} and {ranges_out} differ'
    return ''


def load_ranges(args):
    doc = read_json(Path(args.outputs) / 'stage0a/ranges.json')
    assert doc.get('smoke') == args.smoke, f'ranges.json was written with smoke={doc.get("smoke")}, run has {args.smoke}'
    r = {a: float(doc['map'][a]['r']) for a in ASPECTS}
    assert all(v > 0 for v in r.values()), r
    return r, float(doc['ndcg']['emotion']['r']), rs_vh.file_sha(Path(args.outputs) / 'stage0a/ranges.json')


# ---- scoring cells --------------------------------------------------------------------------------------------------

def head_of(arm, aspect):
    """The arm's head for a query of `aspect`: the aspect itself for an aspect-specific arm, None otherwise."""
    return None if arm in rt_data.ASPECT_FREE else aspect


def cell_list(stage, arm):
    """(head, column) pairs of an arm: stages 1 and 5 the own head per aspect; stage 2 every head by every column."""
    if stage == 2:
        heads = (None,) if arm in rt_data.ASPECT_FREE else ASPECTS
        return [(h, c) for h in heads for c in ASPECTS]
    return [(head_of(arm, a), a) for a in ASPECTS]


def perquery_path(outputs, stage, arm, column, head, seed):
    name = f'{arm}__{column}' + ('' if head is None else f'__h{head}') + f'__s{"-" if seed is None else seed}'
    return Path(outputs) / f'stage{stage}/perquery/{name}.npz'


def score_arm(args, stage, arm, cells, get_x, labels, qe_k=0):
    """Per-query AP and P@10 of every (head, column) cell of an arm, seeds 0 to 4 for a seeded arm.
    get_x(head, seed) -> float32 rows. Returns {(head, column): dict(ap, p10, rows, dropped)}: ap and p10 are
    float64[nq] (unseeded) or float64[5, nq] (rows = seeds 0 to 4)."""
    seeds = SEEDS if arm in rt_data.SEEDED else (None,)
    acc = {c: dict(ap=[], p10=[], rows=None, dropped=None) for c in cells}
    for head in dict.fromkeys(h for h, _ in cells):
        for seed in seeds:
            x = get_x(head, seed)
            for (h, column) in cells:
                if h != head:
                    continue
                r = rt_metrics.retrieval(x, labels[column], qe_k=qe_k, workers=args.workers)
                a = acc[(h, column)]
                assert r['ap'].dtype == r['p10'].dtype == np.float64 and r['rows'].dtype == np.int64
                if a['rows'] is None:
                    a['rows'], a['dropped'] = r['rows'], r['dropped']
                assert np.array_equal(a['rows'], r['rows']) and a['dropped'] == r['dropped']
                a['ap'].append(r['ap'])
                a['p10'].append(r['p10'])
                p = perquery_path(args.outputs, stage, arm, column, h, seed)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(p, ap=r['ap'], p10=r['p10'], rows=r['rows'])
    out = {}
    for c, a in acc.items():
        out[c] = dict(ap=np.stack(a['ap']) if seeds != (None,) else a['ap'][0],
                      p10=np.stack(a['p10']) if seeds != (None,) else a['p10'][0],
                      rows=a['rows'], dropped=a['dropped'])
        shape = (len(seeds), len(a['rows'])) if seeds != (None,) else (len(a['rows']),)
        assert out[c]['ap'].shape == shape and out[c]['ap'].dtype == np.float64
        if args.smoke is None:
            assert a['dropped'] == 0, (arm, c, 'queries without a relevant painting')
    return out


def run_cells(args, stage, arms, get_x_for, labels, qe_ks=None):
    """{(arm, head, column): cell} over the arms; get_x_for(arm) -> get_x(head, seed)."""
    res = {}
    for arm in arms:
        t = time.perf_counter()
        cells = score_arm(args, stage, arm, cell_list(stage, arm), get_x_for(arm), labels,
                          qe_k=0 if qe_ks is None else qe_ks[arm])
        for (h, c), v in cells.items():
            res[(arm, h, c)] = v
        print(f'  stage{stage} {arm}: {time.perf_counter() - t:.1f}s', flush=True)
    return res


def mean_map(cell):
    """{'mean': mean over seeds of the mAP, 'per_seed': [5 mAPs] or None}."""
    ap = cell['ap']
    if ap.ndim == 1:
        return dict(mean=float(ap.mean()), per_seed=None)
    per = [float(ap[s].mean()) for s in range(5)]
    return dict(mean=float(np.mean(per)), per_seed=per)


def map_block(res, arms):
    return {arm: {a: mean_map(res[(arm, head_of(arm, a), a)]) for a in ASPECTS} for arm in arms}


# ---- the SC table ---------------------------------------------------------------------------------------------------

def sc_block(res, ids, r_map, draws):
    """verdict 'sc' entries for the SC ids, and {id: [outcomes]}."""
    doc, outs = {}, {}
    for sid in ids:
        sc = rt_stats.SC[sid]
        comps, outcomes = [], []
        for cmp in sc.comparisons:
            cand, comp = res[(*cmp.candidate, sc.aspect)], res[(*cmp.comparator, sc.aspect)]
            assert np.array_equal(cand['rows'], comp['rows']), (sid, 'query rows differ')
            p = rt_stats.paired(cand['ap'], comp['ap'], draws)
            o = rt_stats.outcome(p['interval'][0], r_map[sc.aspect])
            outcomes.append(o)
            comps.append(dict(candidate=list(cmp.candidate), comparator=list(cmp.comparator), aspect=sc.aspect,
                              d=p['d'], interval=[float(v) for v in p['interval']], r=r_map[sc.aspect], outcome=o,
                              bar=cmp.bar))
        outs[sid] = outcomes
        doc[sid] = dict(comparisons=comps, **{'pass': bool(rt_stats.sc_pass(sid, outcomes))})
    return doc, outs


def n_queries(res):
    return {f'{arm}/{h}/{c}': [len(v['rows']), v['dropped']] for (arm, h, c), v in res.items()}


def verdict_common(args, loader, draws, t0, **extra):
    return dict(inputs=loader.hashes, draws=draws, smoke=args.smoke, workers=args.workers,
                seconds=time.perf_counter() - t0, **extra)


def held_loader(args):
    return Loader(args.smoke, ('held',))


# ---- stages 1, 2 ----------------------------------------------------------------------------------------------------

def stage1(args):
    t0 = time.perf_counter()
    r_map, _, ranges_sha = load_ranges(args)
    L = held_loader(args)
    labels = L.labels('held')
    res = run_cells(args, 1, args.arms, lambda arm: (lambda h, s: L.embed('held', arm, h, s)), labels)
    sc_doc, outs = sc_block(res, rt_stats.TABLE_IDS[1], r_map, args.draws)
    L.hashes['ranges'] = ranges_sha
    doc = verdict_common(args, L, args.draws, t0, sc=sc_doc, go=bool(rt_stats.table_go(outs, 1)),
                         map=map_block(res, args.arms), arms=list(args.arms), n_queries=n_queries(res))
    write_json(Path(args.outputs) / 'stage1/verdict.json', doc)
    return doc


def stage2(args):
    t0 = time.perf_counter()
    r_map, _, ranges_sha = load_ranges(args)
    L = held_loader(args)
    labels = L.labels('held')
    res = run_cells(args, 2, args.arms, lambda arm: (lambda h, s: L.embed('held', arm, h, s)), labels)
    sc_doc, outs = sc_block(res, rt_stats.SWITCH_IDS, r_map, args.draws)
    switch = {}
    for arm in args.arms:
        switch[arm] = {}
        for (h, c) in cell_list(2, arm):
            switch[arm].setdefault('-' if h is None else h, {})[c] = mean_map(res[(arm, h, c)])['mean']
    switch['chance'] = {c: rt_stats.chance(labels[c]) for c in ASPECTS}
    write_json(Path(args.outputs) / 'stage2/switch.json', switch)
    L.hashes['ranges'] = ranges_sha
    doc = verdict_common(args, L, args.draws, t0, sc=sc_doc, switch_go=bool(rt_stats.switch_go(outs)),
                         map=map_block(res, args.arms), arms=list(args.arms), n_queries=n_queries(res))
    write_json(Path(args.outputs) / 'stage2/verdict.json', doc)
    return doc


# ---- stage 3 --------------------------------------------------------------------------------------------------------

def stage3(args):
    t0 = time.perf_counter()
    _, r_nd, ranges_sha = load_ranges(args)
    s1 = read_json(Path(args.outputs) / 'stage1/verdict.json')
    assert s1['smoke'] == args.smoke, 'stage 1 verdict was written with another smoke setting'
    L = held_loader(args)
    dist = L.emotion_dist('held')
    nd = {}
    for arm in ('cstar', 'plain', 'instructed'):
        seeds = SEEDS if arm in rt_data.SEEDED else (None,)
        head = head_of(arm, 'emotion')
        vals = []
        for seed in seeds:
            v = rt_metrics.graded(L.embed('held', arm, head, seed), dist, workers=args.workers)
            assert v.shape == (len(dist),) and v.dtype == np.float64
            p = perquery_path(args.outputs, 3, arm, 'emotion', head, seed)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, ndcg=v, rows=np.arange(len(v), dtype=np.int64))
            vals.append(v)
        nd[arm] = np.stack(vals) if len(seeds) > 1 else vals[0]
        print(f'  stage3 {arm}: {time.perf_counter() - t0:.1f}s', flush=True)
    sc_doc, outs = {}, {}
    for sid, comp_arm in (('ND-plain', 'plain'), ('ND-instructed', 'instructed')):
        p = rt_stats.paired(nd['cstar'], nd[comp_arm], args.draws)
        o = rt_stats.outcome(p['interval'][0], r_nd)
        outs[sid] = o
        sc_doc[sid] = dict(comparisons=[dict(candidate=['cstar', 'emotion'],
                                             comparator=[comp_arm, head_of(comp_arm, 'emotion')], aspect='emotion',
                                             d=p['d'], interval=[float(v) for v in p['interval']], r=r_nd, outcome=o)])
    sc1 = s1['sc']['SC1']['comparisons'][0]['outcome']
    sc4 = s1['sc']['SC4']['comparisons'][0]['outcome']
    reading = rt_stats.stage3_reading(outs['ND-plain'], outs['ND-instructed'], sc1, sc4)

    def block(v):
        per = None if v.ndim == 1 else [float(v[s].mean()) for s in range(5)]
        return dict(mean=float(v.mean()) if per is None else float(np.mean(per)), per_seed=per)

    L.hashes['ranges'] = ranges_sha
    doc = verdict_common(args, L, args.draws, t0, sc=sc_doc, reading=reading,
                         ndcg={arm: {'emotion': block(v)} for arm, v in nd.items()},
                         stage1_outcomes=dict(SC1=sc1, SC4=sc4))
    write_json(Path(args.outputs) / 'stage3/verdict.json', doc)
    return doc


# ---- stage 4 --------------------------------------------------------------------------------------------------------

def stage4(args):
    t0 = time.perf_counter()
    r_map, _, ranges_sha = load_ranges(args)
    if args.smoke is None:
        assert read_json(rt_paths.RANGES_FILE) == read_json(Path(args.outputs) / 'stage0a/ranges.json')
    L = Loader(args.smoke, ('val',))                       # every data call below is validation only
    labels = L.labels('val')
    scores = {}
    picks = {}
    for arm in args.arms:
        ta = time.perf_counter()
        xs = {}
        for s in (SEEDS if arm in rt_data.SEEDED else (None,)):
            if arm in rt_data.ASPECT_FREE:
                x = L.embed('val', arm, None, s)
                xs.update({(a, s): x for a in ASPECTS})      # one matrix: its eigendecomposition is shared
            else:
                xs.update({(a, s): L.embed('val', arm, a, s) for a in ASPECTS})
        widths = {x.shape[1] for x in xs.values()}
        assert len(widths) == 1, (arm, 'the three aspects must have one width', widths)
        width = widths.pop()
        if arm != 'crl':
            assert width == rt_data.WIDTHS[arm], (arm, width)
        ros = rt_readout.menu(width)
        rows = rt_readout.crossfit_all(xs, labels, ros, workers=args.workers)
        sc, keep = rt_readout.score(rows, r_map), rt_readout.guard(rows, r_map)
        scores[arm] = {rt_readout.name(ro): dict(map={a: float(m[a]) for a in ASPECTS}, score=float(sc[ro]),
                                                 kept=bool(keep[ro])) for ro, m in rows.items()}
        picks[arm] = rt_readout.name(rt_readout.select(rows, r_map))
        print(f'  stage4 {arm}: width {width}, {len(ros)} read-outs, pick {picks[arm]}, '
              f'{time.perf_counter() - ta:.1f}s', flush=True)
    L.hashes['ranges'] = ranges_sha
    scores['dropped'] = rt_readout.half_dropped(labels)
    scores.update(inputs=L.hashes, smoke=args.smoke, ranges={a: r_map[a] for a in ASPECTS},
                  seconds=time.perf_counter() - t0)
    write_json(Path(args.outputs) / 'stage4/scores.json', scores)
    write_json(Path(args.outputs) / 'stage4/picks.json', picks)
    return picks


# ---- stage 5 --------------------------------------------------------------------------------------------------------

def stage5(args):
    t0 = time.perf_counter()
    r_map, _, ranges_sha = load_ranges(args)
    picks = read_json(Path(args.outputs) / 'stage4/picks.json')
    assert set(args.arms) <= set(picks), ('picks missing for', set(args.arms) - set(picks))
    L = Loader(args.smoke, ('held', 'val'))
    labels = L.labels('held')
    ros = {}

    def get_x_for(arm):
        def get_x(head, seed):
            return rt_readout.stage5_apply(ros[arm], L.embed('val', arm, head, seed), L.embed('held', arm, head, seed))
        return get_x

    # the pick's k is needed before the first retrieval: resolve each arm's read-out from its menu first
    for arm in args.arms:
        width = rt_data.WIDTHS[arm] if arm != 'crl' else rt_data_crl_width()
        by_name = {rt_readout.name(ro): ro for ro in rt_readout.menu(width)}
        assert picks[arm] in by_name, (arm, picks[arm], 'not in the arm\'s menu')
        ros[arm] = by_name[picks[arm]]
    res = run_cells(args, 5, args.arms, get_x_for, labels, qe_ks={arm: ros[arm].k for arm in args.arms})
    sc_doc, outs = sc_block(res, rt_stats.TABLE_IDS[5], r_map, args.draws)
    L.hashes['ranges'] = ranges_sha
    doc = verdict_common(args, L, args.draws, t0, sc=sc_doc, go=bool(rt_stats.table_go(outs, 5)),
                         map=map_block(res, args.arms), arms=list(args.arms), readouts=dict(picks),
                         n_queries=n_queries(res))
    write_json(Path(args.outputs) / 'stage5/verdict.json', doc)
    return doc


def rt_data_crl_width():
    import rt_crl
    widths = {rt_crl.text_array(a).shape[0] for a in ASPECTS}
    assert len(widths) == 1, ('CRL needs one width K for the three aspects', widths)
    return widths.pop()


# ---- diagnostics ----------------------------------------------------------------------------------------------------

def read_perquery(args, stage, arm, column):
    """Seed-stacked per-query arrays of an arm's own head from a stage's files: dict(ap, p10, rows)."""
    seeds = SEEDS if arm in rt_data.SEEDED else (None,)
    zs = [np.load(perquery_path(args.outputs, stage, arm, column, head_of(arm, column), s)) for s in seeds]
    rows = zs[0]['rows']
    assert all(np.array_equal(z['rows'], rows) for z in zs)
    return dict(ap=np.stack([z['ap'] for z in zs]), p10=np.stack([z['p10'] for z in zs]), rows=rows)


def diag(args):
    t0 = time.perf_counter()
    r_map, _, _ = load_ranges(args)
    outdir = Path(args.outputs)
    for s in ('stage1', 'stage2', 'stage3', 'stage5'):
        assert (outdir / s / 'verdict.json').is_file(), f'diag runs after the verdicts: {s} is missing'
    L = held_loader(args)
    labels = L.labels('held')
    names = {'emotion': list(rs_vh.EMOTIONS), 'style': None, 'genre': None}
    doc = dict(smoke=args.smoke, inputs=L.hashes)
    # per-class AP (stage 1, R0): mean over seeds, then over the queries of each label
    pc = {}
    for arm in ('cstar', 'instructed'):
        pc[arm] = {}
        for a in ASPECTS:
            q = read_perquery(args, 1, arm, a)
            ap, lab = q['ap'].mean(axis=0), labels[a][q['rows']]
            pc[arm][a] = {str(int(c)): dict(ap=float(ap[lab == c].mean()), n=int((lab == c).sum()),
                                            name=names[a][c] if names[a] else None)
                          for c in np.unique(lab)}
    doc['per_class_ap'] = pc
    # c*'s held per-seed mAP spread beside the validation range
    doc['cstar_seed_spread'] = {}
    for a in ASPECTS:
        per = [float(m) for m in read_perquery(args, 1, 'cstar', a)['ap'].mean(axis=1)]
        doc['cstar_seed_spread'][a] = dict(per_seed=per, spread=max(per) - min(per), r=r_map[a])
    doc['chance'] = {a: rt_stats.chance(labels[a]) for a in ASPECTS}
    # P@10 of every arm: stage 1 (R0) and stage 5 (picks); mean over queries and seeds
    doc['p10'] = {f'stage{st}': {arm: {a: float(read_perquery(args, st, arm, a)['p10'].mean()) for a in ASPECTS}
                                 for arm in args.arms} for st in (1, 5)}
    if 'crl' in args.arms:
        lists = read_json(rt_paths.OUTPUTS / 'crl/lists.json')['aspects']
        vals = {a: set(lists[a]['values']) for a in ASPECTS}
        doc['crl'] = {a: dict(count=lists[a]['count'], label_name_matches=lists[a]['label_name_matches'],
                              n_label_name_matches=len(lists[a]['label_name_matches']), note=lists[a]['note'])
                      for a in ASPECTS}
        doc['crl_overlap_between_aspects'] = {f'{a}&{b}': sorted(vals[a] & vals[b])
                                              for i, a in enumerate(ASPECTS) for b in ASPECTS[i + 1:]}
    doc['seconds'] = time.perf_counter() - t0
    write_json(outdir / 'diag/diag.json', doc)
    return doc


# ---- command line ---------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stages', nargs='+', choices=STAGES)
    ap.add_argument('--outputs', type=Path, default=rt_paths.OUTPUTS)
    ap.add_argument('--draws', type=int)
    ap.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    ap.add_argument('--smoke', type=int)
    ap.add_argument('--arms', help='comma list; only with --smoke (a smoke without the arms that are not ready)')
    args = ap.parse_args(argv)
    if args.smoke is not None:
        assert 0 < args.smoke <= min(rt_paths.N_VAL, rt_paths.N_HELD)
        assert args.outputs.resolve() != rt_paths.OUTPUTS.resolve(), 'smoke needs separate outputs'
    args.draws = args.draws or (200 if args.smoke is not None else 10000)
    assert args.arms is None or args.smoke is not None, '--arms is for --smoke only'
    args.arms = tuple(args.arms.split(',')) if args.arms else rt_data.ARMS
    assert set(args.arms) <= set(rt_data.ARMS) and set(REQUIRED_ARMS) <= set(args.arms), args.arms
    for stage in args.stages:
        if stage in HELD_STAGES:
            why = guard(args, need_picks=(stage == 'stage5'))
            if why:
                print(f'REFUSED {stage}: {why}', file=sys.stderr)
                return 1
        t = time.perf_counter()
        print(f'{stage} ...', flush=True)
        out = globals()[stage](args)
        extra = '' if stage in ('stage4', 'diag') else (
            f' go={out["go"]}' if 'go' in out else f' switch_go={out["switch_go"]}' if 'switch_go' in out
            else f' reading={out["reading"]}')
        print(f'{stage}: done in {time.perf_counter() - t:.1f}s{extra}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
