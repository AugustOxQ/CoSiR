"""r19 ticket 13: diagnostics (spec docs/superpowers/specs/2026-11-03-past-picks-design.md §3.6, D14), stage 4.

Descriptive rows after the verdicts. Nothing here decides, picks, sets a range or changes a record: every number is a scorer
call of tickets 07 to 10 (with `subset=` where the scorer has one), a stored file, or a difference, share, spread or count of
those. Stage 1 picks and units are read only through `pp_paths.read_committed_record` (inside the scorers). r17's NMI, ARI,
ACC and cluster counts are ticket 14's.

Row kinds (one function each; each runs alone with `--only`). A missing input gives `{"skipped": <reason>}` for the kind (or,
inside a kind, for one encoder or set, with the other parts kept); an unexpected error gives `{"failed": <message>}` and
exit code 1 at the end.

  r0                 R0 (plain cosine, no read-out map) mAP of every arm: X-c* (5 seeds), X's plain image, X's matched arm
                     (L2-normalised topic probabilities, 5 seeds), the teacher. Sets: SemArt (all seven), MMArt test (new
                     three), ArtELingo held (c* per seed, plain image, validation R0) and Affection test (photo model's validation R0).
  clustering_kclass  AMI of k-means with K = the class count (SemArt 10; MMArt 9, 20, 10) beside K = 20, same arms and sets
                     (route: `ct_score.partitions(x, K)`, states 0 to 2; matched also its own 40-topic argmax).
  semart_close       D14: SC11 to SC14 again on the SemArt set without the paintings at nearest cosine >= 0.93
                     (`pp_semart_score.score(..., subset=)`), beside the stored cells of the whole set, with the outcome each
                     cell and the SemArt verdict would get ("verdict would change"). Reporting only.
  semart_ap_type     mean per-query AP per SemArt type of X-c*, the plain image and the teacher at their picks.
  semart_inner       SemArt pairs inside the set at cosine >= 0.95 (default Qwen rows) and the md5 duplicates removed.
  r6_5_held          10-25's and 10-26's stored held AMI (20 clusters) of the R6-5 student on the three new encoders beside the
                     c* held AMI of this round (cells SC1).
  gain_share         (X-c* - plain) / (teacher - plain) per encoder, set, aspect, task, from the stage 2 cells.
  seed_spreads       max - min over seeds 0 to 4 of X-c* on SemArt, MMArt, ArtELingo held and Affection test (the last two: r14's
                     `bs_diag.artelingo_held` / `affection_held`, new three, against OUT/stage1 ranges) (AMI at 20 clusters; mAP at R0) beside the
                     validation range the teacher cell carries (its LEVEL unit r).
  percept_strongest  per set and aspect the largest matched-arm value of the cells SC3, SC9, SC13 (C1).
  cost_rows          the cost table's other rows from the probe's files (`OUT/stage3/cost_table.json`, `cells_cost.json`).
  node_time          node402 seconds per run from `OUT/node/units.jsonl`.
  r17_metrics        ticket 14 (D13, context only): r17's NMI, ARI, ACC and cluster counts (`ns_metrics`, imported inside the kind)
                     at K = 20 and K = the class count on the same arms and sets as clustering_kclass; held ArtELingo and Affection
                     X-c* at K = 20.

Output: `OUT/stage4/diag/<kind>.json` (with a `markdown` list) and `OUT/stage4/diag/tables.md` (built from every kind file
present, in the order above). CLI: `python pp_diag.py run [--only KIND ...] [--workers N] [--draws N]`. Smoke: PP_OUTPUTS and
PP_SMOKE=1 (fewer draws).
"""
import pp_paths as PP  # noqa: I001  (first: sets the older rounds' output roots)
import pp_register  # noqa: F401  (all seven encoders in r14's and r15's tables)
import bs_diag  # noqa: E402  (r14's held readers; after pp_paths and pp_register)

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

KINDS = ('r0', 'clustering_kclass', 'semart_close', 'semart_ap_type', 'semart_inner', 'r6_5_held', 'gain_share',
         'seed_spreads', 'percept_strongest', 'cost_rows', 'node_time', 'r17_metrics')
ENCODERS, NEW, ASPECTS, SEEDS = tuple(PP.ENCODERS), tuple(PP.NEW_ENCODERS), tuple(PP.ASPECTS), tuple(PP.SEEDS)
AE, AF, MM, SA = PP.DATA
PLAIN_SC = {(AE, 'clu'): 'SC1', (AE, 'ret'): 'SC2', (AF, 'clu'): 'SC4', (AF, 'ret'): 'SC5', (MM, 'clu'): 'SC7',
            (MM, 'ret'): 'SC8', (SA, 'clu'): 'SC11', (SA, 'ret'): 'SC12'}
TEACHER_SC = {AE: 'SC6', AF: 'SC6', MM: 'SC10', SA: 'SC14'}
MATCHED_SC = {AE: 'SC3', MM: 'SC9', SA: 'SC13'}


class Skip(Exception):
    """An input is missing: the kind (or the section) is recorded as skipped with the reason."""


def diag_dir():
    d = Path(PP.OUT) / 'stage4' / 'diag'
    d.mkdir(parents=True, exist_ok=True)
    return d


def f4(v, n=4):
    return 'n/a' if v is None else f'{v:.{n}f}'


def mean(v):
    return float(np.mean(v))


def spread(v):
    return float(max(v) - min(v))


def table(head, rows):
    return ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]


def need(path):
    path = Path(path)
    if not path.exists():
        raise Skip(f'missing input {path}')
    return path


def read_json(path):
    return json.loads(need(path).read_text())


def section(fn):
    """Run one section: its value, or {'skipped': reason} for a missing input (other sections go on)."""
    try:
        return fn()
    except Skip as e:
        return {'skipped': str(e)}
    except (FileNotFoundError, KeyError, RuntimeError) as e:       # a missing store, export, record (RuntimeError: uncommitted)
        if isinstance(e, KeyError) or isinstance(e, RuntimeError) and 'committed' not in str(e):
            raise
        return {'skipped': f'{type(e).__name__}: {e}'}


def n_draws(task):
    return (PP.SMOKE_DRAWS if PP.SMOKE else PP.N_DRAWS)[task]


def load_cells():
    """All stage 2 cells and the cost cells under OUT (`pp_outcome.load_cells`); Skip when there are none."""
    import pp_outcome
    cells, cost, _files = pp_outcome.load_cells(PP.OUT)
    if not cells:
        raise Skip(f'no stage 2 cells under {Path(PP.OUT) / "stage2"}')
    return cells, cost


def cell_index(cells):
    return {(c['encoder'], c['data'], c['aspect'], c['task'], c['sc']): c for c in cells}


# ============================================================================================ data sets (rows of one scored set)
class DataSet:
    """The rows the arm tables need, all joined by key in `keys` order. Subclasses wire a set to the scorers' readers."""
    name = None
    encoders = ()
    aspects = ()

    def keys(self):
        raise NotImplementedError

    def labels(self, aspect):                    # int64 [n], -1 = none
        raise NotImplementedError

    def class_count(self, aspect):
        raise NotImplementedError

    def cstar(self, enc, aspect, seed, task):    # float32 [n, d]; task 'clu' (cast only) or 'ret' (unit rows)
        raise NotImplementedError

    def frozen(self, enc, task):
        raise NotImplementedError

    def teacher(self, aspect, task):
        raise NotImplementedError

    def probs(self, enc, seed):                  # float32 [n, 40]
        raise NotImplementedError


class SemartSet(DataSet):
    name, encoders, aspects = 'semart', ENCODERS, ('genre',)

    def __init__(self):
        import pp_semart
        import pp_semart_score as SS
        self.pp_semart, self.SS = pp_semart, SS
        self._keys = SS.production_order()
        self.inp = {t: SS.SemartInputs(t) for t in ('clu', 'ret')}

    def keys(self):
        return self._keys

    def labels(self, aspect):
        return np.asarray(self.pp_semart.labels(self._keys), dtype=np.int64)

    def class_count(self, aspect):
        return len(PP.SEMART_TYPES)

    def cstar(self, enc, aspect, seed, task):
        return self.inp[task].cstar_eval(enc, seed, self._keys)

    def frozen(self, enc, task):
        return self.inp[task].frozen_eval(enc, self._keys)

    def teacher(self, aspect, task):
        return self.inp[task].teacher_eval(self._keys)

    def probs(self, enc, seed):
        import pp_matched
        return pp_matched.load_probs('semart', enc, seed, self._keys)


class MmartSet(DataSet):
    name, encoders, aspects = 'mmart', NEW, ASPECTS

    def __init__(self):
        import importlib
        import mt_clu
        import mt_paths
        import mt_ret
        self.clu, self.ret, self.MP = mt_clu, mt_ret, mt_paths
        self.md, self.ex = importlib.import_module('mt_data'), importlib.import_module('mt_export')
        self._keys = list(mt_clu._ordered_keys('test'))
        mt_ret.check_order(self._keys)

    def keys(self):
        return self._keys

    def labels(self, aspect):
        return np.asarray(self.md.labels(self._keys, aspect), dtype=np.int64)

    def class_count(self, aspect):
        return len(self.MP.CLASSES[aspect])

    def cstar(self, enc, aspect, seed, task):
        if task == 'ret':
            return self.ret.cstar_rows(enc, aspect, seed, 'test', self._keys)
        return np.asarray(self.ex.load(self.clu.EXPORT_SET['test'], enc, aspect, seed, self._keys)).astype(np.float32)

    def frozen(self, enc, task):
        view = self.MP.DEFAULT_VIEW[enc]
        return self.ret.view_rows(view, self._keys) if task == 'ret' else self.clu._unit32(self.md.features(view, self._keys))

    def teacher(self, aspect, task):
        view = self.MP.TEACHER_VIEW[aspect]
        return self.ret.view_rows(view, self._keys) if task == 'ret' else self.clu._unit32(self.md.features(view, self._keys))

    def probs(self, enc, seed):
        import pp_matched
        return pp_matched.load_probs('mmart_new', enc, seed, self._keys)


SETS = {'semart': SemartSet, 'mmart': MmartSet}
_SET_CACHE, _TABLE_CACHE = {}, {}


def get_set(name):
    if name not in _SET_CACHE:
        _SET_CACHE[name] = SETS[name]()
    return _SET_CACHE[name]


# ============================================================================================ arm tables (R0, AMI at 20 and K)
def ami_mean(parts, y):
    """Mean over the partitions (rows of `parts`) of the AMI against `y` on the labelled rows."""
    import mt_clu
    parts = np.asarray(parts)
    y = np.asarray(y)
    assert parts.ndim == 2 and parts.shape[1] == len(y)
    m = y >= 0
    return float(np.mean([float(mt_clu.rb_score.ami(y[m], p[m])) for p in parts]))


def kmeans(x, k):
    """Three k-means partitions with k clusters (states 0 to 2): k = 20 is the scorers' route (`mt_clu.kmeans_partitions`)."""
    import mt_clu
    if k == 20:
        return mt_clu.kmeans_partitions(x)
    from threadpoolctl import threadpool_limits
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2 and np.isfinite(x).all()
    with threadpool_limits(limits=8):
        return np.asarray(mt_clu.ct_score.partitions(x, k))


def r0_map(x, y, workers=8):
    import mt_ret
    return float(mt_ret.rt_metrics.retrieval(x, y, workers=workers)['ap'].mean())


def unit32(x):
    x = np.asarray(x).astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(n).all() and (n > 0).all()
    return (x / n).astype(np.float32)


def reading_values(x, y, k):
    return dict(ami20=ami_mean(kmeans(x, 20), y), amik=ami_mean(kmeans(x, k), y))


def arm_table(ds, enc, aspect, workers=8):
    """Every arm's R0 mAP, AMI at 20 clusters and AMI at K = the class count on `ds` (memoised in the process)."""
    key = (ds.name, enc, aspect)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    import mt_clu
    import pp_matched
    y, k = ds.labels(aspect), ds.class_count(aspect)
    out = dict(n_rows=int(len(y)), n_labelled=int((y >= 0).sum()), k=k, cstar=dict(r0=[], ami20=[], amik=[]),
               matched=dict(r0=[], ami20=[], amik=[], ami_own=[]))
    for s in SEEDS:
        xc = ds.cstar(enc, aspect, s, 'clu')
        p20 = kmeans(xc, 20)
        out['cstar']['ami20'].append(ami_mean(p20, y))
        out['cstar']['amik'].append(out['cstar']['ami20'][-1] if k == 20 else ami_mean(kmeans(xc, k), y))
        out['cstar']['r0'].append(r0_map(ds.cstar(enc, aspect, s, 'ret'), y, workers))
        prob = ds.probs(enc, s)
        rd = pp_matched.readings(prob)
        out['matched']['ami20'].append(ami_mean(rd['r20'], y))
        out['matched']['ami_own'].append(ami_mean(np.asarray(rd['own']), y))
        out['matched']['amik'].append(out['matched']['ami20'][-1] if k == 20 else ami_mean(kmeans(mt_clu.ct_score.l2(prob.astype(np.float32)), k), y))
        out['matched']['r0'].append(r0_map(unit32(prob), y, workers))
    for arm, getter in (('frozen', lambda t: ds.frozen(enc, t)), ('teacher', lambda t: ds.teacher(aspect, t))):
        xc = getter('clu')
        p20 = kmeans(xc, 20)
        out[arm] = dict(ami20=ami_mean(p20, y), amik=ami_mean(p20, y) if k == 20 else ami_mean(kmeans(xc, k), y),
                        r0=r0_map(getter('ret'), y, workers))
    _TABLE_CACHE[key] = out
    return out


def each_set(ctx, fn, sets=None):
    """{set: {enc: {aspect: fn(ds, enc, aspect)}}} with skips per (set, enc); Skip when nothing ran."""
    res, ran = {}, 0
    for name in sets or ctx['sets']:
        try:
            ds = get_set(name)
        except (Skip, FileNotFoundError) as e:
            res[name] = {'skipped': f'{type(e).__name__}: {e}'}
            continue
        res[name] = {}
        for enc in ds.encoders:
            def one(enc=enc):
                return {a: fn(ds, enc, a) for a in ds.aspects}
            res[name][enc] = section(one)
            ran += 'skipped' not in res[name][enc]
    if not ran:
        raise Skip('no encoder had its inputs on any set: ' + json.dumps(res)[:300])
    return res



# ---------------------------------------------------------------------------------------------- held sets (r14's code, driven by registration)
def held_art(enc, workers):
    """ArtELingo held per-seed values of the new encoder `enc` through r14's `bs_diag.artelingo_held` (OUT = r19's OUT by
    `pp_paths`; c* exports `OUT/cstar/<enc>/held`, stage 1 files `OUT/stage1/*_<enc>.json`). A missing input is a skip."""
    try:
        return bs_diag.artelingo_held(enc, workers)
    except bs_diag.NotRun as e:
        raise Skip(f'{e}: {e.missing}')


def held_aff(enc, workers):
    """Affection test per-seed values of the X photo model (`bs_diag.affection_held`; the photo model's rows, as r14)."""
    try:
        return bs_diag.affection_held(enc, workers)
    except bs_diag.NotRun as e:
        raise Skip(f'{e}: {e.missing}')


def stage1_file(name):
    try:
        return bs_diag.read_json(Path(PP.OUT) / 'stage1' / name)
    except bs_diag.NotRun as e:
        raise Skip(f'{e}: {e.missing}')


def per_encoder(fn):
    return {enc: section(lambda enc=enc: fn(enc)) for enc in NEW}


def r0_held(ctx):
    w = ctx['workers']

    def art(enc):
        h = held_art(enc, w)['ret']
        val = stage1_file(f'artret_ranges_{enc}.json')['per_seed']
        return dict(cstar_r0_per_seed=h['cstar_r0'], frozen_r0=h['frozen_r0'],
                    validation_cstar_r0_mean={a: float(np.mean(list(v.values()))) for a, v in val.items()})

    def aff(enc):
        rec = stage1_file(f'affection_ranges_{enc}.json')
        ps = rec['ranges_doc']['map']['per_seed']
        ps = list(ps.values()) if isinstance(ps, dict) else list(ps)
        return dict(validation_photo_r0_per_seed=ps, validation_photo_r0_mean=float(np.mean(ps)))
    return dict(artelingo_held=per_encoder(art), affection_test=per_encoder(aff))


def spreads_held(ctx):
    w = ctx['workers']

    def ratio(sp, r):
        return dict(held_spread=sp, validation_range=r, ratio=sp / r)

    def art(enc):
        h = held_art(enc, w)
        rc, rr = stage1_file(f'artclu_ranges_{enc}.json')['ranges'], stage1_file(f'artret_ranges_{enc}.json')['ranges']
        return dict(clustering={a: ratio(spread(h['clu'][a]), rc[a]) for a in ASPECTS},
                    retrieval_r0={a: ratio(spread(h['ret']['cstar_r0'][a]), rr[a]) for a in ASPECTS})

    def aff(enc):
        h = held_aff(enc, w)
        return dict(ami=ratio(spread(h['ami']), h['ranges']['clu']), map_at_pick=ratio(spread(h['map']), h['ranges']['map']))
    return dict(artelingo_held=per_encoder(art), affection_test=per_encoder(aff))


def held_ok(res):
    return any('skipped' not in v for sec in res.values() for v in sec.values())


def skip_lines(res):
    return [f'- {sname} {e}: skipped ({v["skipped"][:80]})' for sname, sec in res.items() for e, v in sec.items() if 'skipped' in v]


# ============================================================================================ row kinds
def kind_r0(ctx):
    try:
        res = each_set(ctx, lambda ds, e, a: arm_table(ds, e, a, ctx['workers']))
    except Skip as e:
        res = {'skipped': str(e)}
    held = r0_held(ctx)
    if 'skipped' in res and not held_ok(held):
        raise Skip('no set had its inputs: ' + json.dumps([res, held])[:300])
    res = {k: v for k, v in res.items() if k != 'skipped'}
    md = ['R0 = plain cosine retrieval mAP (no read-out map; every labelled row is a query and a gallery row).', '']
    for sname, encs in res.items():
        rows = []
        for e, aspects in encs.items() if 'skipped' not in encs else ():
            if 'skipped' in aspects:
                rows.append([e, '-', 'skipped', aspects['skipped'][:60], '', '', ''])
                continue
            for a, t in aspects.items():
                rows.append([e, a, f4(mean(t['cstar']['r0'])), f4(t['frozen']['r0']), f4(mean(t['matched']['r0'])), f4(t['teacher']['r0']),
                             f'{f4(min(t["cstar"]["r0"]))} to {f4(max(t["cstar"]["r0"]))}'])
        md += [f'### {sname}', '', *table(['encoder', 'aspect', 'X-c*', 'plain image', 'matched (PercepT)', 'teacher', 'c* seed range'], rows), '']
    rows = []
    for e, v in held['artelingo_held'].items():
        for a in ASPECTS if 'skipped' not in v else ():
            rows.append([e, a, f4(mean(v['cstar_r0_per_seed'][a])), f4(v['frozen_r0'][a]), f4(v['validation_cstar_r0_mean'][a])])
    md += ["### artelingo_held (new three; r14's `bs_diag.artelingo_held`)", '',
           *table(['encoder', 'aspect', 'X-c* held', 'plain image held', 'X-c* validation'], rows), '']
    rows = [[e, f4(v['validation_photo_r0_mean'])] for e, v in held['affection_test'].items() if 'skipped' not in v]
    md += ['### affection_test (X photo model; validation R0 as r14 reports it)', '', *table(['encoder', 'photo validation R0 mean'], rows), '',
           *skip_lines(held), '']
    return dict(sets=res, **held), md


def kind_clustering_kclass(ctx):
    res = each_set(ctx, lambda ds, e, a: arm_table(ds, e, a, ctx['workers']))
    md = ['AMI of k-means, K = the class count beside K = 20 (mean over seeds and the three k-means states).', '']
    for sname, encs in res.items():
        rows = []
        for e, aspects in encs.items() if 'skipped' not in encs else ():
            if 'skipped' in aspects:
                rows.append([e, '-', '-', 'skipped', aspects['skipped'][:60], '', '', '', '', '', ''])
                continue
            for a, t in aspects.items():
                rows.append([e, a, t['k'], f4(mean(t['cstar']['amik'])), f4(mean(t['cstar']['ami20'])), f4(t['frozen']['amik']),
                             f4(t['frozen']['ami20']), f4(mean(t['matched']['amik'])), f4(mean(t['matched']['ami20'])),
                             f4(t['teacher']['amik']), f4(t['teacher']['ami20'])])
        md += [f'### {sname}', '', *table(['encoder', 'aspect', 'K', 'c* K', 'c* 20', 'plain K', 'plain 20', 'matched K', 'matched 20',
                                         'teacher K', 'teacher 20'], rows), '']
    return dict(sets=res), md


class record_encoders:
    """Context manager: `pp_semart_score.ENCODERS` = the encoders the committed SemArt stage 1 record holds (as `pp_run.py`
    semart-score does for a partial record), restored on exit. `.have` is that tuple. A committed-record problem is raised
    as `read_committed_record` raises it (the kind then skips or fails as before)."""

    def __enter__(self):
        import pp_semart_score as SS
        self.SS, self.old = SS, SS.ENCODERS
        self.have = tuple(PP.read_committed_record(SS.STAGE1_NAME))
        SS.ENCODERS = self.have
        return self

    def __exit__(self, *exc):
        self.SS.ENCODERS = self.old
        return False


def missing_encoder(have):
    return {'skipped': f'no entry in the committed SemArt stage 1 record (it holds {", ".join(have)})'}


def semart_outcomes(cells):
    """{sc: outcome} of SC11 to SC14 cells of one encoder and the SemArt verdict (`pp_outcome`)."""
    import pp_outcome as O
    out = {}
    for c in cells:
        out[f'{c["sc"]}_{c["task"]}'] = O.cell_outcome(c)
    verdict = None
    if all(k in out for k in ('SC11_clu', 'SC12_ret', 'SC13_clu')):
        verdict = O.semart_verdict(out['SC11_clu'], out['SC12_ret'], out['SC13_clu'])
    return out, verdict


def kind_semart_close(ctx):
    import pp_semart
    import pp_semart_score as SS
    keys = pp_semart.keys('set')
    close, far = set(pp_semart.keys('close')), pp_semart.keys('far')
    left = [k for k in keys if k not in close]
    assert left == far, 'set minus close is not pp_semart.keys("far")'
    cells_all = []
    try:
        cells_all = load_cells()[0]
    except Skip:
        pass
    res, md_rows, changed = {}, [], []
    with record_encoders() as guard:
        for enc in ENCODERS:
            if enc not in guard.have:
                res[enc] = missing_encoder(guard.have)
                continue

            def one(enc=enc):
                got = SS.score('semart_clu', enc, n_draws('clu'), ctx['workers'], subset=far) + \
                    SS.score('semart_ret', enc, n_draws('ret'), ctx['workers'], subset=far)
                full_clu_sc14 = [c for c in cells_all if c['encoder'] == enc and c['data'] == SA]
                o_far, v_far = semart_outcomes(got)
                o_full, v_full = semart_outcomes(full_clu_sc14) if full_clu_sc14 else (None, None)
                changes = None if o_full is None else {k: [o_full.get(k), o_far[k]] for k in o_far if o_full.get(k) != o_far[k]}
                return dict(n_close=len(close), n_far=len(far), cells_without_close=got, outcome_without_close=o_far,
                            verdict_without_close=v_far, outcome_whole_set=o_full, verdict_whole_set=v_full,
                            outcome_changes=changes, verdict_changes=None if v_full is None else v_full != v_far)
            res[enc] = section(one)
    for enc, r in res.items():
        if 'skipped' in r:
            md_rows.append([enc, 'skipped', r['skipped'][:70], '', '', ''])
            continue
        whole = {(c['sc'], c['task']): c for c in cells_all if c['encoder'] == enc and c['data'] == SA}
        for c in r['cells_without_close']:
            w = whole.get((c['sc'], c['task']))
            md_rows.append([enc, f'{c["sc"]} {c["task"]}', f'{c["d"]:+.4f} [{c["lo"]:+.4f}, {c["hi"]:+.4f}]',
                            f'{w["d"]:+.4f} [{w["lo"]:+.4f}, {w["hi"]:+.4f}]' if w else 'n/a',
                            f'{r["outcome_whole_set"].get(c["sc"] + "_" + c["task"]) if r["outcome_whole_set"] else "n/a"} -> '
                            f'{r["outcome_without_close"][c["sc"] + "_" + c["task"]]}', c['n_rows']])
        if r['verdict_changes']:
            changed.append(enc)
    ran = [e for e, r in res.items() if 'skipped' not in r]
    if not ran:
        raise Skip('no encoder could be scored without the close paintings: ' + json.dumps(res)[:300])
    any_change = None if any(res[e]['verdict_changes'] is None for e in ran) else bool(changed)
    md = [f'D14: the SemArt set without the {len(close)} paintings at nearest cosine >= {PP.SEMART_CLOSE} ({len(far)} left). Reporting only: '
          'no bar changes and no verdict is replaced.', '',
          f'Any SemArt verdict would change: {"unknown (the whole-set cells are missing)" if any_change is None else ("yes: " + ", ".join(changed) if any_change else "no")}.',
          '', *table(['encoder', 'cell', 'without close: d [95% interval]', 'whole set: d [95% interval]', 'outcome whole -> without',
                      'rows'], md_rows), '']
    return dict(n_close=len(close), n_far=len(far), encoders=res, any_verdict_changes=any_change, verdict_changed_for=changed), md


def per_type(ap, rows, y):
    a = np.asarray(ap, dtype=np.float64)
    a = a.mean(axis=0) if a.ndim == 2 else a
    yy = np.asarray(y)[np.asarray(rows)]
    return {t: dict(n_queries=int((yy == i).sum()), ap=float(a[yy == i].mean())) for i, t in enumerate(PP.SEMART_TYPES) if (yy == i).any()}


def kind_semart_ap_type(ctx):
    import pp_semart_score as SS
    ds = get_set('semart') if 'semart' in SETS else None
    keys = ds.keys()
    y = ds.labels('genre')
    res = {}
    with record_encoders() as guard:
        for enc in ENCODERS:
            if enc not in guard.have:
                res[enc] = missing_encoder(guard.have)
                continue

            def one(enc=enc):
                units, picks = SS.stage1_for(enc)
                arms = SS.ret_arms(enc, SS.SemartInputs('ret'), keys, picks, ctx['workers'])
                return {arm: dict(point=float(arms[arm]['point']), per_type=per_type(arms[arm]['ap'], arms[arm]['rows'], y))
                        for arm in SS.ARMS}
            res[enc] = section(one)
    if all('skipped' in r for r in res.values()):
        raise Skip('no encoder had its picks or rows: ' + json.dumps(res)[:300])
    md = ['Mean per-query AP per SemArt type at each arm\'s pick (c*: mean of seeds 0 to 4); queries are the paintings of the type.', '']
    for enc, r in res.items():
        if 'skipped' in r:
            md += [f'{enc}: skipped ({r["skipped"][:80]})', '']
            continue
        rows = [[t, r['cstar']['per_type'][t]['n_queries'], f4(r['cstar']['per_type'][t]['ap']), f4(r['frozen']['per_type'][t]['ap']),
                 f4(r['teacher']['per_type'][t]['ap'])] for t in r['cstar']['per_type']]
        md += [f'### {enc}', '', *table(['type', 'queries', 'X-c*', 'plain image', 'teacher'], rows), '']
    return dict(encoders=res), md


def kind_semart_inner(ctx):
    import pp_semart
    keys = pp_semart.keys('set')
    n_pairs, n_members = pp_semart.inner_pairs(pp_semart._qwen_rows(keys), PP.SEMART_CUT)
    gate = pp_semart.gate_vi()
    groups = gate['md5_groups']
    res = dict(n_set=len(keys), cut=PP.SEMART_CUT, pairs=int(n_pairs), paintings_in_pairs=int(n_members), rows='Qwen default image',
               md5_groups=len(groups), md5_dropped=sum(len(g['dropped']) for g in groups), n_gate=gate['paintings'])
    md = ['SemArt pairs inside the set (default Qwen rows) and the md5 step.', '',
          *table(['set paintings', 'pairs at cosine >= 0.95', 'paintings in a pair', 'md5 groups', 'md5 duplicates removed', 'before md5'],
                 [[res['n_set'], n_pairs, n_members, res['md5_groups'], res['md5_dropped'], res['n_gate']]]), '']
    return res, md


R6_5 = {'clip_b32': (PP.RB_PART2, 'K3', 'reported_stage4.json'), 'clip_b16': (PP.EG_PART2, 'K6', 'reported_stage4.json'),
        'dinov2_large': (PP.EG_PART2, 'K5', 'reported_stage3.json')}


def kind_r6_5_held(ctx):
    try:
        cells = cell_index(load_cells()[0])
    except Skip:
        cells = {}
    res, rows = {}, []
    for enc, (root, block, fname) in R6_5.items():
        def one(enc=enc, root=root, block=block, fname=fname):
            doc = read_json(Path(root) / fname)
            b = doc['blocks'][block]
            out = {}
            for a in ASPECTS:
                c = cells.get((enc, AE, a, 'clu', 'SC1'))
                out[a] = dict(r6_5_ami20=float(b['aspects'][a]['margins']['c_frozen']['A_cand']),
                              cstar_ami20=None if c is None else c['candidate_value'])
            return dict(source=str(Path(root) / fname), scale=doc.get('scale'), block=block, outcome_id=b['verdict']['outcome_id'],
                        aspects=out)
        res[enc] = section(one)
        r = res[enc]
        for a in ASPECTS:
            rows.append([enc, a, 'skipped' if 'skipped' in r else f4(r['aspects'][a]['r6_5_ami20']),
                         'n/a' if 'skipped' in r else f4(r['aspects'][a]['cstar_ami20']),
                         '-' if 'skipped' in r else f'{r["scale"]} {r["block"]} {r["outcome_id"]}'])
    if all('skipped' in r for r in res.values()):
        raise Skip('no stored 10-25 or 10-26 file: ' + json.dumps(res)[:300])
    md = ['R6-5 student (10-25 and 10-26 rounds; stored files) beside c* (this round, SC1 candidate value): ArtELingo held, AMI with 20 '
          'clusters, five seeds. DINOv2 R6-5 is the sample-scale run (the full run was skipped).', '',
          *table(['encoder', 'aspect', 'R6-5', 'c*', 'stored block'], rows), '']
    return dict(encoders=res), md


def kind_gain_share(ctx):
    cells, _ = load_cells()
    ix = cell_index(cells)
    rows, out = [], []
    for (enc, data, a, task, sc), t in sorted(ix.items()):
        if sc != TEACHER_SC.get(data):
            continue
        p = ix.get((enc, data, a, task, PLAIN_SC[(data, task)]))
        if p is None:
            continue
        plain, teacher, model = p['comparator_value'], t['comparator_value'], t['candidate_value']
        gap = teacher - plain
        row = dict(encoder=enc, data=data, aspect=a, task=task, plain=plain, teacher=teacher, cstar=model, cstar_in_plain_cell=p['candidate_value'],
                   gap=gap, share=(model - plain) / gap if gap > 0 else None,
                   note=None if gap > 0 else 'teacher not above the plain image: no share')
        out.append(row)
        rows.append([enc, data, a, task, f4(plain), f4(model), f4(teacher), f4(gap), f4(row['share'], 3)])
    if not out:
        raise Skip('no plain-image cell has a teacher cell beside it')
    md = ['Share of the teacher\'s gain over the plain image that X-c* keeps: (c* - plain) / (teacher - plain). A gap of zero or below gives '
          'no share.', '', *table(['encoder', 'set', 'aspect', 'task', 'plain', 'X-c*', 'teacher', 'gap', 'share'], rows), '']
    return dict(rows=out), md


def kind_seed_spreads(ctx):
    try:
        ix = cell_index(load_cells()[0])
    except Skip:
        ix = {}

    def one_cell(ds, enc, a):
        t = arm_table(ds, enc, a, ctx['workers'])
        data = MM if ds.name == 'mmart' else SA
        r_clu, r_ret = (ix.get((enc, data, a, task, TEACHER_SC[data])) for task in ('clu', 'ret'))
        rc, rr = (None if c is None else c['r'] for c in (r_clu, r_ret))
        sc, sr = spread(t['cstar']['ami20']), spread(t['cstar']['r0'])
        return dict(ami20=dict(per_seed=t['cstar']['ami20'], test_spread=sc, validation_range=rc, ratio=None if not rc else sc / rc),
                    map_r0=dict(per_seed=t['cstar']['r0'], test_spread=sr, validation_range=rr, ratio=None if not rr else sr / rr))
    try:
        res = each_set(ctx, one_cell)
    except Skip as e:
        res = {'skipped': str(e)}
    held = spreads_held(ctx)
    if 'skipped' in res and not held_ok(held):
        raise Skip('no set had its inputs: ' + json.dumps([res, held])[:300])
    res = {k: v for k, v in res.items() if k != 'skipped'}
    rows = []
    for sname, encs in res.items():
        for e, aspects in encs.items() if 'skipped' not in encs else ():
            if 'skipped' in aspects:
                continue
            for a, v in aspects.items():
                rows.append([sname, e, a, f4(v['ami20']['test_spread']), f4(v['ami20']['validation_range']), f4(v['ami20']['ratio'], 2),
                             f4(v['map_r0']['test_spread']), f4(v['map_r0']['validation_range']), f4(v['map_r0']['ratio'], 2)])
    md = ['Seed spread (max - min over seeds 0 to 4) of X-c* on test beside the validation range (the LEVEL unit r of the teacher cell, '
          'unrounded; "n/a" when the cell file is missing). ArtELingo held and Affection test: r14\'s `bs_diag` values against the stage 1 '
          'ranges.', '',
          *table(['set', 'encoder', 'aspect', 'AMI spread (test)', 'AMI range (val)', 'ratio', 'mAP R0 spread (test)', 'mAP range (val)', 'ratio'], rows), '']
    rows = []
    for e, v in held['artelingo_held'].items():
        for a in ASPECTS if 'skipped' not in v else ():
            rows.append(['artelingo_held', e, a, f4(v['clustering'][a]['held_spread']), f4(v['clustering'][a]['validation_range']),
                         f4(v['clustering'][a]['ratio'], 2), f4(v['retrieval_r0'][a]['held_spread']), f4(v['retrieval_r0'][a]['validation_range']),
                         f4(v['retrieval_r0'][a]['ratio'], 2)])
    for e, v in held['affection_test'].items():
        if 'skipped' not in v:
            rows.append(['affection_test', e, 'emotion', f4(v['ami']['held_spread']), f4(v['ami']['validation_range']), f4(v['ami']['ratio'], 2),
                         f'{f4(v["map_at_pick"]["held_spread"])} (at pick)', f4(v['map_at_pick']['validation_range']), f4(v['map_at_pick']['ratio'], 2)])
    md += ['', *table(['set', 'encoder', 'aspect', 'AMI spread', 'AMI range (val)', 'ratio', 'mAP spread (R0; Affection at pick)',
                       'mAP range (val)', 'ratio'], rows), '', *skip_lines(held), '']
    return dict(sets=res, **held), md


def kind_percept_strongest(ctx):
    cells, _ = load_cells()
    best, rows = {}, []
    for c in cells:
        if c['sc'] != MATCHED_SC.get(c['data']):
            continue
        k = (c['data'], c['aspect'])
        if k not in best or c['comparator_value'] > best[k]['value']:
            best[k] = dict(value=c['comparator_value'], encoder=c['encoder'], entry=PP.MATCHED[c['encoder']], sc=c['sc'])
    if not best:
        raise Skip('no SC3, SC9 or SC13 cell')
    for (data, a), v in sorted(best.items()):
        rows.append([data, a, v['entry'], v['encoder'], v['sc'], f4(v['value'])])
    md = ['Strongest PercepT entry per set and aspect (C1): the largest matched-arm value (the cell\'s comparator value, the larger of '
          'its two readings) over the encoders that have an SC3, SC9 or SC13 cell.', '',
          *table(['set', 'aspect', 'entry', 'encoder', 'cell', 'AMI'], rows), '']
    return dict(strongest={f'{d}/{a}': v for (d, a), v in best.items()}), md


def kind_cost_rows(ctx):
    root = Path(PP.OUT) / 'stage3'
    ct = read_json(root / 'cost_table.json')
    cells = read_json(root / 'cells_cost.json')
    rows = []
    for name, p in ct['passes'].items():
        vt = p.get('vision_tower')
        rows.append([name, p.get('kind', ''), f4(p.get('ms_per_image_mean'), 2), f4(p.get('images_per_second'), 1), f4(p.get('ms_per_image_e2e'), 2),
                     f4(p.get('images_per_second_e2e'), 1), f4(p.get('peak_memory_allocated_mib'), 0), p.get('params_loaded', 'n/a'),
                     'n/a' if vt is None else json.dumps(vt)[:60]])
    whole = [[e, f4(w['cstar_s'], 1), f4(w['teacher_s'], 1), f4(w['cstar_e2e_s'], 1), f4(w['teacher_e2e_s'], 1)] for e, w in ct['whole_set_16384'].items()]
    md = ['Cost table, other rows (from the probe: `stage3/cost_table.json`).', '',
          *table(['pass', 'kind', 'ms/image', 'images/s', 'ms/image e2e', 'images/s e2e', 'peak MiB', 'parameters', 'vision tower share'], rows), '',
          f'Passes per aspect: {json.dumps(ct["passes_per_aspect"])} (c* shares one encoder pass among three aspects; the teacher runs one '
          'instructed Qwen pass per aspect).', '',
          *table(['encoder', 'c* s (16,384 paintings, model)', 'teacher s', 'c* s e2e', 'teacher s e2e'], whole), '',
          'Prep log Qwen (store extraction): ' + json.dumps(ct.get('prep_log_qwen')), '']
    return dict(cost_table=ct, cells=cells), md


def kind_node_time(ctx):
    path = need(Path(PP.OUT) / 'node' / 'units.jsonl')
    units = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not units:
        raise Skip(f'{path} is empty')
    groups = {}
    for u in units:
        groups.setdefault((u.get('kind'), u.get('enc', u.get('encoder'))), []).append(u)
    rows, out = [], []
    for (kind, enc), us in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        secs = [float(u['seconds']) for u in us if u.get('status', 'ok') == 'ok' and u.get('seconds') is not None]
        row = dict(kind=kind, encoder=enc, runs=len(us), ok=len(secs), failed=sum(u.get('status') not in (None, 'ok') for u in us),
                   attempts=sum(int(u.get('attempts', 1)) for u in us), seconds=secs, mean_s=mean(secs) if secs else None,
                   min_s=min(secs) if secs else None, max_s=max(secs) if secs else None, total_s=float(sum(secs)),
                   hosts=sorted({str(u.get('host')) for u in us}))
        out.append(row)
        rows.append([kind, enc, row['runs'], row['ok'], row['failed'], f4(row['mean_s'], 0), f4(row['min_s'], 0), f4(row['max_s'], 0),
                     f4(row['total_s'], 0), ','.join(row['hosts'])])
    md = ['node402 time per run (`node/units.jsonl`; seconds of the unit, including its export).', '',
          *table(['kind', 'encoder', 'runs', 'ok', 'failed', 'mean s', 'min s', 'max s', 'total s', 'host'], rows), '']
    return dict(units=out, n_lines=len(units)), md


R17_FOLDER = Path(PP.__file__).resolve().parents[1] / '20261103_new_scores'      # r17's `ns_*` files (read only)
R17_METRICS = ('nmi', 'ari', 'acc')


def r17():
    """r17's `ns_metrics` imported read-only, inside the kind only (it asserts scikit-learn 1.6.1). Its folder goes at the END of
    sys.path, so `pp_paths` and `pp_register` stay first. Importing writes nothing."""
    if str(R17_FOLDER) not in sys.path:
        sys.path.append(str(R17_FOLDER))
    try:
        import ns_metrics
    except ImportError as e:
        raise Skip(f'r17 module not importable: {e}')
    return ns_metrics


def r17_values(NM, y, parts):
    """{metric: mean over the states, 'clusters': mean number of non-empty clusters}; labelled rows only (r17's array forms)."""
    parts = np.asarray(parts)
    assert parts.ndim == 2 and parts.shape[1] == len(y) and (parts >= 0).all(), 'partitions need ids >= 0 and one row per key'
    out = {m: NM.score_partitions(m, np.asarray(y), parts) for m in R17_METRICS}
    out['clusters'] = float(np.mean([len(np.unique(p)) for p in parts]))
    return out


def avg(rows):
    return {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}


def r17_arm_values(NM, ds, enc, aspect):
    """Per arm and K (20 and the class count) the mean over seeds (c*, matched) and k-means states of NMI, ARI, ACC, clusters; the
    partitions are `arm_table`'s (`kmeans`; matched: k-means of the L2-normalised probabilities)."""
    import mt_clu
    y, k = ds.labels(aspect), ds.class_count(aspect)
    out = {}
    for kk in sorted({20, k}):
        out[str(kk)] = dict(
            cstar=avg([r17_values(NM, y, kmeans(ds.cstar(enc, aspect, s, 'clu'), kk)) for s in SEEDS]),
            matched=avg([r17_values(NM, y, kmeans(mt_clu.ct_score.l2(ds.probs(enc, s).astype(np.float32)), kk)) for s in SEEDS]),
            frozen=r17_values(NM, y, kmeans(ds.frozen(enc, 'clu'), kk)),
            teacher=r17_values(NM, y, kmeans(ds.teacher(aspect, 'clu'), kk)))
    return dict(k=k, values=out)


def r17_held(ctx, NM):
    """ArtELingo held X-c* and Affection test X-c* at 20 clusters, from the partitions r14's readers fix (cached under
    `bs_diag.out_dir()`). The class-count K is not stored there, so the held sets give K = 20 only."""
    def art(enc):
        import bs_artclu
        names, sha, labels = bs_artclu.held_setup()
        held_dir = need(Path(PP.OUT) / 'cstar' / enc / 'held/k11=40')
        res = {}
        for a in ASPECTS:
            fixed, _ = bs_artclu.cstar_fixed(held_dir, bs_diag.out_dir() / 'cache/artclu' / enc, a)
            res[a] = avg([r17_values(NM, labels[a], np.asarray(fixed[s]['r20'])) for s in SEEDS])
        return res

    def aff(enc):
        import ac_data
        import bs_affection
        import pf_stage3
        rec_file = need(bs_affection.ranges_path(enc, Path(PP.OUT)))
        need(bs_affection.export_path(f'photo_{enc}', 'test', 0, Path(PP.OUT)))
        rec = bs_affection.read_record(rec_file, enc)
        cand = bs_affection.PhotoArm(f'photo_{enc}', Path(PP.OUT)).candidate(rec['photo_readout']['pick'])
        parts = pf_stage3.candidate_partitions(cand, bs_diag.out_dir() / 'cache/affection_partitions', ctx['workers'])
        labels = np.asarray(ac_data.labels('test'))
        return avg([r17_values(NM, labels, parts[s]) for s in range(5)])
    return dict(artelingo_held=per_encoder(art), affection_test=per_encoder(aff))


def kind_r17_metrics(ctx):
    """D13: context rows only. No claim, no word of verdict; nothing here reaches `pp_outcome`."""
    NM = r17()
    try:
        res = each_set(ctx, lambda ds, e, a: r17_arm_values(NM, ds, e, a))
    except Skip as e:
        res = {'skipped': str(e)}
    held = r17_held(ctx, NM)
    if 'skipped' in res and not held_ok(held):
        raise Skip('no set had its inputs: ' + json.dumps([res, held])[:300])
    res = {k: v for k, v in res.items() if k != 'skipped'}
    md = ["r17's NMI, ARI, ACC and the number of non-empty clusters (`ns_metrics`, read only) on this round's clustering cells; mean over "
          'seeds (c*, matched) and the three k-means states. Context only (D13): no claim and no bar.', '']
    for sname, encs in res.items():
        rows = []
        for e, aspects in encs.items() if 'skipped' not in encs else ():
            if 'skipped' in aspects:
                rows.append([e, '-', '-', '-', 'skipped', aspects['skipped'][:60], '', ''])
                continue
            for a, t in aspects.items():
                for kk, arms in t['values'].items():
                    for arm, v in arms.items():
                        rows.append([e, a, kk, arm, f4(v['nmi']), f4(v['ari']), f4(v['acc']), f'{v["clusters"]:.1f}'])
        md += [f'### {sname}', '', *table(['encoder', 'aspect', 'K', 'arm', 'NMI', 'ARI', 'ACC', 'clusters'], rows), '']
    rows = []
    for e, v in held['artelingo_held'].items():
        for a in ASPECTS if 'skipped' not in v else ():
            rows.append([e, a, *(f4(v[a][m]) for m in R17_METRICS), f'{v[a]["clusters"]:.1f}'])
    md += ['### artelingo_held (X-c*, K = 20)', '', *table(['encoder', 'aspect', 'NMI', 'ARI', 'ACC', 'clusters'], rows), '']
    rows = [[e, *(f4(v[m]) for m in R17_METRICS), f'{v["clusters"]:.1f}'] for e, v in held['affection_test'].items() if 'skipped' not in v]
    md += ['### affection_test (X photo model, K = 20)', '', *table(['encoder', 'NMI', 'ARI', 'ACC', 'clusters'], rows), '', *skip_lines(held), '']
    return dict(sets=res, **held), md


KIND_FN = {'r0': kind_r0, 'clustering_kclass': kind_clustering_kclass, 'semart_close': kind_semart_close,
           'semart_ap_type': kind_semart_ap_type, 'semart_inner': kind_semart_inner, 'r6_5_held': kind_r6_5_held,
           'gain_share': kind_gain_share, 'seed_spreads': kind_seed_spreads, 'percept_strongest': kind_percept_strongest,
           'cost_rows': kind_cost_rows, 'node_time': kind_node_time, 'r17_metrics': kind_r17_metrics}
assert tuple(KIND_FN) == KINDS


# ============================================================================================ driver
def write_json(path, doc):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(doc, indent=1, default=float, allow_nan=True) + '\n')
    os.replace(tmp, path)


def write_tables():
    """`tables.md` from every kind file present, in KINDS order."""
    d = diag_dir()
    out = ['# r19 diagnostics (stage 4; descriptive, nothing here decides a verdict)', '']
    if PP.SMOKE:
        out += ['SMOKE RUN: reduced draws; the numbers are not results.', '']
    for k in KINDS:
        f = d / f'{k}.json'
        if not f.exists():
            continue
        doc = json.loads(f.read_text())
        out += [f'## {k}', '']
        if 'skipped' in doc:
            out += [f'Skipped: {doc["skipped"]}', '']
        elif 'failed' in doc:
            out += [f'Failed: {doc["failed"]}', '']
        else:
            out += doc.get('markdown', [])
    (d / 'tables.md').write_text('\n'.join(out) + '\n')
    return d / 'tables.md'


def run_kind(kind, ctx):
    t0 = time.monotonic()
    doc = dict(kind=kind, smoke=bool(PP.SMOKE))
    try:
        data, md = KIND_FN[kind](ctx)
        doc.update(data)
        doc['markdown'] = md
    except Skip as e:
        doc['skipped'] = str(e)
    except (FileNotFoundError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'committed' not in str(e):
            traceback.print_exc()
            doc['failed'] = f'{type(e).__name__}: {e}'
        else:
            doc['skipped'] = f'{type(e).__name__}: {e}'
    except Exception as e:  # noqa: BLE001  (one kind's bug must not stop the others)
        traceback.print_exc()
        doc['failed'] = f'{type(e).__name__}: {e}'
    doc['seconds'] = round(time.monotonic() - t0, 1)
    write_json(diag_dir() / f'{kind}.json', doc)
    return doc


def run(only=None, workers=8, sets=('semart', 'mmart')):
    kinds = list(only or KINDS)
    assert all(k in KIND_FN for k in kinds), kinds
    ctx = dict(workers=workers, sets=tuple(sets))
    status = {}
    for k in kinds:
        doc = run_kind(k, ctx)
        status[k] = 'skipped' if 'skipped' in doc else 'failed' if 'failed' in doc else 'ok'
        print(f'diag {k}: {status[k]} ({doc["seconds"]} s)' + (f' - {doc.get("skipped") or doc.get("failed")}'[:200] if status[k] != 'ok' else ''),
              flush=True)
    write_tables()
    return status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--only', nargs='+', choices=KINDS, default=None)
    r.add_argument('--workers', type=int, default=8)
    a = p.parse_args(argv)
    status = run(a.only, a.workers)
    return 1 if 'failed' in status.values() else 0


if __name__ == '__main__':
    sys.exit(main())
