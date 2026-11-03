"""r15 diagnostics (ticket 08, spec §3.6, D14, D19): descriptive read-outs after the verdicts. They decide nothing: no item
feeds a verdict, a pick, a range or a record. Every number is a function of tickets 01 to 06 (the scorers `mt_clu`, `mt_ret`,
the data door `mt_data`, the exports, the matched arms) or an older round; this module adds only differences, shares,
spreads and counts. Each item writes `OUT/diag/<item>.{json,md}` (pattern of r14's `bs_diag.py`).

Items (one function each; the sets are `mt_data.keys(name)`):
  r0_rows               R0 (plain cosine) mAP on MMArt test: X-c* (five seeds), X's frozen image, X's matched arm (L2-normalised
                        topic probabilities), the teacher. Reads exports and features only (no stage 1 record).
  clustering_k_class    AMI with k-means of K = the class count (9, 20, 10) beside K = 20, same arms (matched: also its
                        own 40-topic argmax). Partitions: `ct_score.partitions(x, K)`, three states, the route of `mt_clu`.
  per_class_ap          mean per-query AP per class at each arm's stage 1 pick (X-c*, X's frozen image, the teacher).
  far                   the far subset (D19, the cut at 0.85): SC1, SC2, SC4 cells (`mt_clu.score_encoder`,
                        `mt_ret.score_encoder` with keyset `far`; stage 1 picks and ranges; full draws).
  unseen_artist         D14: the count and the artists only; no cells (fewer paintings than the 20 clusters).
  shared_held           the 11,528 shared ArtELingo held paintings: SC1, SC2, SC4 cells with MMArt labels beside the same cells
                        with ArtELingo labels (r14's reader `bs_artclu.held_setup`), same paintings, same picks.
                        The same calls as the scorers' `score_encoder`, with the labels passed in (this module's loop; the
                        k-means partitions are made once and scored against both label sets).
  near_copy             the count, the 0.80 to 1.0 histogram of the nearest cosine, the removed paintings per class at each
                        cut, and the test pairs with default-Qwen cosine >= 0.95 inside the cut set.
  teacher_gain_share    (X-c* - frozen) / (teacher - frozen) per encoder, aspect and task, from the stage 2 test cells.
  seed_spreads          max - min over seeds 0 to 4 of X-c* on test (AMI at 20 clusters; mAP at R0 and at the pick) beside the
                        validation ranges of the stage 1 record.
  percept_strongest     per aspect the largest of the four matched arms' larger readings (own, 20 clusters) on test (C1).

A missing input gives the item (or the section of it) status "not run" with the missing path; the other items go on. An
unexpected error in a task gives the section status "failed" with the message, and the exit code 1 at the end.
Status of an item: "ok", "partial", "not run" or "failed".

CLI: `python mt_diag.py all|<item> [--workers N]`, N <= 3 processes of 8 threads (default 3; 1 runs in this process). Smoke mode
is the environment's: `MT_SMOKE=1 MT_OUTPUTS=<scratch>` (draws `mt_paths.SMOKE_DRAWS`; stage 1 records read from
`OUT/records`). Cell files for far and shared_held go to `mt_clu.cells_path` / `mt_ret.cells_path` (OUT/diag/cells_*).
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mt_paths as MP  # noqa: E402

if str(MP.BS_ROUND) not in sys.path:
    sys.path.insert(0, str(MP.BS_ROUND))     # r14's bs_emoset (histogram, removed_mask)

ITEMS = ('r0_rows', 'clustering_k_class', 'per_class_ap', 'far', 'unseen_artist', 'shared_held', 'near_copy',
         'teacher_gain_share', 'seed_spreads', 'percept_strongest')
ASPECTS = tuple(MP.ASPECTS)
SEEDS = tuple(MP.SEEDS)
ENCODERS = tuple(MP.ENCODERS)
CUTS = (0.85, 0.90, 0.95, 0.99)             # the cuts of the near-copy counts (0.95 = the test cut, 0.85 = the far subset)
PAIR_CUT = 0.95                             # test pairs inside the cut set
DATA_SHARED_AE = 'mmart_shared_held_artelingo_labels'
MAX_WORKERS = 3


def n_draws(task):
    """Bootstrap draws: the protocol's (2,000 clustering, 10,000 retrieval), or `mt_paths.SMOKE_DRAWS` under smoke."""
    return (MP.SMOKE_DRAWS if MP.SMOKE else MP.N_DRAWS)[task]


# ============================================================================================ lazy imports
def _m(name):
    return importlib.import_module(name)


def _data():
    return _m('mt_data')


def _clu():
    return _m('mt_clu')


def _ret():
    return _m('mt_ret')


class NotRun(Exception):
    """An input is missing: the section is reported as "not run" with the paths."""

    def __init__(self, why, missing=()):
        super().__init__(why)
        self.missing = [str(m) for m in missing]


def diag_dir():
    d = Path(MP.OUT) / 'diag'
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path):
    if not Path(path).exists():
        raise NotRun('missing input', [path])
    return json.loads(Path(path).read_text())


def committed(name):
    """A stage 1 record through `mt_paths.read_committed_record` (smoke: the scratch record)."""
    try:
        return MP.read_committed_record(name)
    except FileNotFoundError as e:
        raise NotRun(f'record {name} is missing', [e.filename])
    except RuntimeError as e:
        raise NotRun(str(e), [name])


def f4(v, n=4):
    return 'n/a' if v is None else f'{v:.{n}f}'


def table(head, rows):
    return ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + \
        ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]


def ordered_keys(name):
    d = _data()
    keys = list(d.keys(name))
    assert keys and len(set(keys)) == len(keys) and keys == list(d.order(keys)), f'{name}: not in mt_data.order'
    return keys


def unit32(x):
    x = np.asarray(x).astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(n).all() and (n > 0).all()
    return (x / n).astype(np.float32)


def aspect_labels(keys):
    d = _data()
    return {a: np.asarray(d.labels(keys, a), dtype=np.int64) for a in ASPECTS}


def ami_mean(parts, y):
    """Mean over the partitions (rows of `parts` [m, n]) of the AMI against `y` on the labelled rows (-1 masked)."""
    rb_score = _clu().rb_score
    parts = np.asarray(parts)
    assert parts.ndim == 2 and parts.shape[1] == len(y)
    m = np.asarray(y) >= 0
    return float(np.mean([float(rb_score.ami(y[m], p[m])) for p in parts]))


def kmeans_k(x, k):
    """Three k-means partitions with k clusters on all rows (r14 part a's route: `ct_score.partitions`, states 0 to 2, 8
    threads); k = 20 is `mt_clu.kmeans_partitions`. int [3, n]."""
    clu = _clu()
    if k == 20:
        return clu.kmeans_partitions(x)
    from threadpoolctl import threadpool_limits
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2 and np.isfinite(x).all() and np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-3
    with threadpool_limits(limits=8):
        return np.asarray(clu.ct_score.partitions(x, k))


def r0_map(x, y, workers=8):
    return float(_ret().rt_metrics.retrieval(x, y, workers=workers)['ap'].mean())


def per_class(ap, rows, y, aspect):
    """Mean per-query AP per class (seeds averaged first) over the queries of the class."""
    a = np.asarray(ap, dtype=np.float64)
    a = a.mean(axis=0) if a.ndim == 2 else a
    yy = np.asarray(y)[np.asarray(rows)]
    return {cl: dict(n_queries=int((yy == i).sum()), ap=float(a[yy == i].mean()))
            for i, cl in enumerate(MP.CLASSES[aspect]) if (yy == i).any()}


def spread(v):
    return float(max(v) - min(v))


def artelingo_labels(keys):
    """ArtELingo's labels of the shared held paintings, by painting name, through r14's reader (`bs_artclu.held_setup`: the
    12,281 held paintings in the CT's list). int64 per aspect, -1 = none."""
    d, clu = _data(), _clu()
    names, _sha, labels = clu.bs_artclu.held_setup()
    pos = {n: i for i, n in enumerate(names)}
    assert len(pos) == len(names)
    idx = [pos[d.artelingo_name(k)] for k in keys]
    out = {a: np.asarray(labels[a])[idx].astype(np.int64) for a in ASPECTS}
    assert all(v.shape == (len(keys),) for v in out.values())
    return out


# ============================================================================================ tasks (one process each)
def _teacher_pick():
    doc = committed('stage1_teacher.json')
    return doc.get('picks', doc)['arms']['teacher']


def _ranges_block(block):
    r = block.get('ranges', block) if isinstance(block.get('ranges'), dict) else block
    assert set(r) == set(ASPECTS), sorted(r)
    return {a: float(r[a]) for a in ASPECTS}


def task_clu_test(enc):
    """Test AMI per arm and seed: X-c* (k-means 20 clusters, and K = the class count), the frozen image, the matched arm (20
    clusters, K, and its own reading). Exports and features only."""
    d, clu, ex, mm = _data(), _clu(), _m('mt_export'), _m('mt_matched')
    keys = ordered_keys('test')
    y = aspect_labels(keys)
    frozen = unit32(d.features(MP.DEFAULT_VIEW[enc], keys))
    fz20 = clu.kmeans_partitions(frozen)
    matched20 = mm.partitions(enc, keys)
    out = dict(n_rows=len(keys), aspects={})
    for a in ASPECTS:
        k = len(MP.CLASSES[a])
        c20, ck = [], []
        for s in SEEDS:
            x = np.asarray(ex.load('new', enc, a, s, keys)).astype(np.float32)
            p20 = clu.kmeans_partitions(x)
            c20.append(ami_mean(p20, y[a]))
            ck.append(c20[-1] if k == 20 else ami_mean(kmeans_k(x, k), y[a]))
        m20, mown, mk = [], [], []
        for s in SEEDS:
            m20.append(ami_mean(matched20[s]['k20'], y[a]))
            mown.append(ami_mean(np.asarray(matched20[s]['own'])[None, :], y[a]))
            if k == 20:
                mk.append(m20[-1])
            else:
                prob = mm.load_probs(enc, s, keys)
                mk.append(ami_mean(kmeans_k(clu.ct_score.l2(prob.astype(np.float32)), k), y[a]))
        out['aspects'][a] = dict(
            k=k, n_labelled=int((y[a] >= 0).sum()),
            cstar=dict(k20=c20, kclass=ck), frozen=dict(k20=ami_mean(fz20, y[a]),
                                                        kclass=ami_mean(fz20 if k == 20 else kmeans_k(frozen, k), y[a])),
            matched=dict(k20=m20, own=mown, kclass=mk))
        print(f'clu_test {enc} {a}: c* K20 {np.mean(c20):.4f} K{k} {np.mean(ck):.4f}', flush=True)
    return out


def task_ret_r0(enc):
    """Test R0 mAP (plain cosine, labelled rows are queries and gallery) of X-c* per seed, the frozen image, the matched arm per
    seed (L2-normalised probabilities)."""
    d, ret, ex, mm = _data(), _ret(), _m('mt_export'), _m('mt_matched')
    keys = ordered_keys('test')
    ret.check_order(keys)
    y = aspect_labels(keys)
    fz = ret.view_rows(MP.DEFAULT_VIEW[enc], keys)
    probs = {s: ret.unit(mm.load_probs(enc, s, keys)) for s in SEEDS}
    out = dict(n_rows=len(keys), aspects={})
    for a in ASPECTS:
        cs = [r0_map(ret.cstar_rows(enc, a, s, 'test', keys), y[a]) for s in SEEDS]
        out['aspects'][a] = dict(n_labelled=int((y[a] >= 0).sum()), cstar=cs, frozen=r0_map(fz, y[a]),
                                 matched=[r0_map(probs[s], y[a]) for s in SEEDS])
        print(f'ret_r0 {enc} {a}: c* {np.mean(cs):.4f}', flush=True)
    return out


def task_ret_pick(enc):
    """Test mAP at the stage 1 picks, per class: X-c* (five seeds) and X's frozen image (`mt_ret.arm_ap`, the scorer's call)."""
    d, ret = _data(), _ret()
    _ranges, arms, _teach = ret.load_stage1(enc)
    val_keys, keys = ordered_keys('val'), ordered_keys('test')
    ret.check_order(val_keys)
    ret.check_order(keys)
    y = aspect_labels(keys)
    view = MP.DEFAULT_VIEW[enc]
    fz_val, fz_eval = ret.view_rows(view, val_keys), ret.view_rows(view, keys)
    out = dict(n_rows=len(keys), picks=dict(cstar=arms['cstar']['pick'], frozen=arms['frozen']['pick']), aspects={})
    for a in ASPECTS:
        cand = ret.arm_ap(lambda s: ret.cstar_rows(enc, a, s, 'val', val_keys),
                          lambda s: ret.cstar_rows(enc, a, s, 'test', keys), arms['cstar']['pick'], y[a], True, 8)
        fz = ret.arm_ap(lambda s: fz_val, lambda s: fz_eval, arms['frozen']['pick'], y[a], False, 8)
        out['aspects'][a] = dict(cstar=dict(point=cand['point'], per_seed=cand['per_seed'],
                                            per_class=per_class(cand['ap'], cand['rows'], y[a], a)),
                                 frozen=dict(point=fz['point'], per_class=per_class(fz['ap'], fz['rows'], y[a], a)))
        print(f'ret_pick {enc} {a}: c* {cand["point"]:.4f} frozen {fz["point"]:.4f}', flush=True)
    return out


def task_teacher_test(_=None):
    """The teacher's test R0 mAP and AMI (20 clusters and K = class count) per aspect (no record read)."""
    d, clu, ret = _data(), _clu(), _ret()
    keys = ordered_keys('test')
    y = aspect_labels(keys)
    out = dict(n_rows=len(keys), aspects={})
    for a in ASPECTS:
        x = ret.view_rows(MP.TEACHER_VIEW[a], keys)
        k = len(MP.CLASSES[a])
        p20 = clu.kmeans_partitions(x)
        out['aspects'][a] = dict(k=k, r0=r0_map(x, y[a]), k20=ami_mean(p20, y[a]),
                                 kclass=ami_mean(p20 if k == 20 else kmeans_k(x, k), y[a]))
        print(f'teacher_test {a}: R0 {out["aspects"][a]["r0"]:.4f}', flush=True)
    return out


def task_teacher_pick(_=None):
    """The teacher's test mAP at its stage 1 pick, per class."""
    d, ret = _data(), _ret()
    pick = _teacher_pick()['pick']
    val_keys, keys = ordered_keys('val'), ordered_keys('test')
    y = aspect_labels(keys)
    out = dict(pick=pick, aspects={})
    for a in ASPECTS:
        tv, te = ret.view_rows(MP.TEACHER_VIEW[a], val_keys), ret.view_rows(MP.TEACHER_VIEW[a], keys)
        res = ret.arm_ap(lambda s: tv, lambda s: te, pick, y[a], False, 8)
        out['aspects'][a] = dict(point=res['point'], per_class=per_class(res['ap'], res['rows'], y[a], a))
        print(f'teacher_pick {a}: {res["point"]:.4f}', flush=True)
    return out


def task_far_clu(enc):
    cells = _clu().score_encoder(enc, 'far', ('SC1', 'SC4'), n_draws('clu'), 8, write=False)
    return dict(cells=cells)


def task_far_ret(enc):
    ret = _ret()
    cells = ret.score_encoder(enc, 'far', ('SC2', 'SC4'), n_draws('ret'), 8)
    return dict(cells=cells)


def task_shared_clu(enc):
    """SC1 and SC4 (AMI) cells on the shared held paintings, MMArt labels and ArtELingo labels. The calls are
    `mt_clu.score_encoder`'s, aspect by aspect (c* partitions, frozen and teacher partitions made once), with the labels
    of each variant passed to `mt_clu.score_aspect`."""
    d, clu, ex = _data(), _clu(), _m('mt_export')
    keys = ordered_keys('shared_held')
    rec = committed(f'stage1_{enc}.json')
    ranges = clu.record_ranges(rec, enc)
    variants = {clu.DATA_NAME['shared_held']: aspect_labels(keys), DATA_SHARED_AE: artelingo_labels(keys)}
    frozen = clu.single_fixed(d.features(MP.DEFAULT_VIEW[enc], keys))
    cells = []
    for a in ASPECTS:
        c_fixed = clu.cstar_fixed({s: ex.load('shared_held', enc, a, s, keys) for s in SEEDS})
        teacher = clu.single_fixed(d.features(MP.TEACHER_VIEW[a], keys))
        for tag, y in variants.items():
            got = clu.score_aspect(enc, a, y[a], c_fixed, frozen=frozen, teacher=teacher, scs=('SC1', 'SC4'), r=ranges[a],
                                   n_draws=n_draws('clu'), workers=8, data=tag, matched_name=MP.MATCHED[enc])
            cells += got
            print(f'shared_clu {enc} {a} {tag}: ' + ', '.join(f'{c["sc"]} d={c["d"]:+.4f}' for c in got), flush=True)
    return dict(cells=cells)


def task_shared_ret(enc):
    """SC2 and SC4 (mAP) cells on the shared held paintings, MMArt labels and ArtELingo labels: `mt_ret.score_encoder`'s loop
    with the labels of each variant (per-query AP depends on the labels, so every arm is scored once per variant)."""
    d, ret = _data(), _ret()
    ranges, arms, teach = ret.load_stage1(enc)
    val_keys, keys = ordered_keys('val'), ordered_keys('shared_held')
    ret.check_order(val_keys)
    ret.check_order(keys)
    variants = {ret.cell_data('shared_held'): aspect_labels(keys), DATA_SHARED_AE: artelingo_labels(keys)}
    ek = ret.export_keyset('shared_held')
    view = MP.DEFAULT_VIEW[enc]
    fz_val, fz_eval = ret.view_rows(view, val_keys), ret.view_rows(view, keys)
    readouts = dict(cstar=arms['cstar']['pick'], frozen=arms['frozen']['pick'], teacher=teach['pick'])
    cells = []
    for a in ASPECTS:
        tv, te = ret.view_rows(MP.TEACHER_VIEW[a], val_keys), ret.view_rows(MP.TEACHER_VIEW[a], keys)
        for tag, y in variants.items():
            cand = ret.arm_ap(lambda s: ret.cstar_rows(enc, a, s, 'val', val_keys),
                              lambda s: ret.cstar_rows(enc, a, s, ek, keys), readouts['cstar'], y[a], True, 8)
            fz = ret.arm_ap(lambda s: fz_val, lambda s: fz_eval, readouts['frozen'], y[a], False, 8)
            th = ret.arm_ap(lambda s: tv, lambda s: te, readouts['teacher'], y[a], False, 8)
            got = ret.aspect_cells(enc, a, cand, fz, th, ranges[a], n_draws('ret'), readouts, len(keys), tag, ('SC2', 'SC4'))
            cells += got
            print(f'shared_ret {enc} {a} {tag}: ' + ', '.join(f'{c["sc"]} d={c["d"]:+.4f}' for c in got), flush=True)
    return dict(cells=cells)


TASKS = dict(clu_test=task_clu_test, ret_r0=task_ret_r0, ret_pick=task_ret_pick, teacher_test=task_teacher_test,
             teacher_pick=task_teacher_pick, far_clu=task_far_clu, far_ret=task_far_ret, shared_clu=task_shared_clu,
             shared_ret=task_shared_ret)
PER_ENCODER = ('clu_test', 'ret_r0', 'ret_pick', 'far_clu', 'far_ret', 'shared_clu', 'shared_ret')
ITEM_TASKS = dict(
    r0_rows=('ret_r0', 'teacher_test'), clustering_k_class=('clu_test', 'teacher_test'),
    per_class_ap=('ret_pick', 'teacher_pick'), far=('far_clu', 'far_ret'), unseen_artist=(), shared_held=('shared_clu', 'shared_ret'),
    near_copy=(), teacher_gain_share=(), seed_spreads=('clu_test', 'ret_r0', 'ret_pick'), percept_strongest=('clu_test',))
assert tuple(ITEM_TASKS) == ITEMS


def run_task(key):
    """One task in this process; returns a picklable {'status': 'ok'|'not run'|'failed', ...}."""
    name, enc = key
    t0 = time.monotonic()
    try:
        res = TASKS[name](enc) if enc else TASKS[name]()
        return dict(status='ok', result=res, seconds=round(time.monotonic() - t0, 1))
    except NotRun as e:
        return dict(status='not run', reason=str(e), missing=e.missing, seconds=round(time.monotonic() - t0, 1))
    except FileNotFoundError as e:
        return dict(status='not run', reason=f'missing input: {e}', missing=[str(e.filename)],
                    seconds=round(time.monotonic() - t0, 1))
    except Exception as e:  # noqa: BLE001  (reported, never swallowed: the others go on and the exit code is 1)
        traceback.print_exc()
        return dict(status='failed', reason=f'{type(e).__name__}: {e}', missing=[], seconds=round(time.monotonic() - t0, 1))


def task_keys(items):
    keys = []
    for it in items:
        for t in ITEM_TASKS[it]:
            for k in ((t, e) for e in ENCODERS) if t in PER_ENCODER else [(t, None)]:
                if k not in keys:
                    keys.append(k)
    # the longest first
    rank = {n: i for i, n in enumerate(('shared_ret', 'shared_clu', 'far_clu', 'clu_test', 'far_ret', 'ret_pick', 'ret_r0',
                                        'teacher_pick', 'teacher_test'))}
    return sorted(keys, key=lambda k: rank[k[0]])


def run_tasks(keys, workers):
    """Run the tasks; `workers` processes (fork, each with its own 8 threads), or in this process when workers is 1."""
    assert 1 <= workers <= MAX_WORKERS, f'workers must be 1 to {MAX_WORKERS}'
    res = {}
    if workers == 1 or len(keys) <= 1:
        for k in keys:
            res[k] = run_task(k)
            print(f'task {k}: {res[k]["status"]} ({res[k]["seconds"]} s)', flush=True)
        return res
    ctx = multiprocessing.get_context('fork')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(run_task, k): k for k in keys}
        for f in concurrent.futures.as_completed(futs):
            k = futs[f]
            res[k] = f.result()
            print(f'task {k}: {res[k]["status"]} ({res[k]["seconds"]} s)', flush=True)
    return res


# ============================================================================================ item documents
class Failed(NotRun):
    """A task failed unexpectedly; the section is marked "failed"."""


class Doc:
    """An item's document: sections with a status each; the markdown lines beside."""

    def __init__(self, item, note):
        self.d = dict(item=item, note=note, smoke=bool(MP.SMOKE), sections={})
        self.md = []

    def section(self, name, fn):
        try:
            self.d['sections'][name] = dict(status='ok', **fn())
        except Failed as e:
            self.d['sections'][name] = dict(status='failed', reason=str(e), missing=e.missing)
        except NotRun as e:
            self.d['sections'][name] = dict(status='not run', reason=str(e), missing=e.missing)
        except Exception as e:  # noqa: BLE001  (a bug in one section must not stop the others)
            traceback.print_exc()
            self.d['sections'][name] = dict(status='failed', reason=f'{type(e).__name__}: {e}', missing=[])

    def add(self, *lines):
        self.md.extend(lines)

    def finish(self):
        st = [s['status'] for s in self.d['sections'].values()]
        if all(s == 'ok' for s in st):
            status = 'ok'
        elif any(s == 'ok' for s in st):
            status = 'partial'
        else:
            status = 'failed' if 'failed' in st else 'not run'
        self.d['status'] = status
        self.d['missing'] = sorted({m for s in self.d['sections'].values() for m in s.get('missing', [])})
        return self.d

    def write(self):
        d = diag_dir()
        item = self.d['item']
        clean = json.loads(json.dumps(self.d, default=float))
        (d / f'{item}.json').write_text(json.dumps(clean, indent=1) + '\n')
        head = [f'# diag: {item}', '', f'Status: {self.d["status"]}. {self.d["note"]}', '']
        if MP.SMOKE:
            head += ['SMOKE RUN: reduced draws; the numbers are not results.', '']
        head += [f'- missing: `{m}`' for m in self.d['missing']]
        for name, s in self.d['sections'].items():
            if s['status'] != 'ok':
                head.append(f'- section {name}: {s["status"]} ({s["reason"]})')
        (d / f'{item}.md').write_text('\n'.join(head + [''] + self.md) + '\n')


def got(res, name, enc=None):
    """The result of a task, or NotRun/failed for the section that needs it."""
    r = res.get((name, enc))
    if r is None:
        raise NotRun(f'task {name} {enc or ""} was not run')
    if r['status'] == 'ok':
        return r['result']
    if r['status'] == 'not run':
        raise NotRun(r['reason'], r['missing'])
    raise Failed(r['reason'])


def item_r0_rows(res):
    doc = Doc('r0_rows', 'R0 (plain cosine) mAP on MMArt test per arm; descriptive')
    teacher = {}
    try:
        teacher = got(res, 'teacher_test')['aspects']
    except NotRun:
        pass

    def make(enc):
        def sec():
            r = got(res, 'ret_r0', enc)['aspects']
            rows, out = [], {}
            for a in ASPECTS:
                x = r[a]
                out[a] = dict(cstar_per_seed=x['cstar'], cstar_mean=float(np.mean(x['cstar'])), frozen=x['frozen'],
                              matched_per_seed=x['matched'], matched_mean=float(np.mean(x['matched'])),
                              teacher=teacher.get(a, {}).get('r0'), n_labelled=x['n_labelled'])
                rows.append([a, f4(out[a]['cstar_mean']), f4(x['frozen']), f4(out[a]['matched_mean']),
                             f4(out[a]['teacher']), x['n_labelled']])
            doc.add(f'## {enc}', '', *table(['aspect', 'X-c* (mean of 5)', 'frozen image', 'matched arm (mean of 5)', 'teacher',
                                              'labelled'], rows), '')
            return dict(aspects=out)
        return sec
    for enc in ENCODERS:
        doc.section(enc, make(enc))
    if not teacher:
        doc.add('The teacher column is empty: its task did not run.', '')
    return doc


def item_clustering_k_class(res):
    doc = Doc('clustering_k_class', 'AMI of k-means with K = the class count (9, 20, 10) beside K = 20, MMArt test; descriptive')
    teacher = {}
    try:
        teacher = got(res, 'teacher_test')['aspects']
    except NotRun:
        pass

    def make(enc):
        def sec():
            r = got(res, 'clu_test', enc)['aspects']
            rows, out = [], {}
            for a in ASPECTS:
                x = r[a]
                t = teacher.get(a, {})
                out[a] = dict(k=x['k'], cstar_k20=float(np.mean(x['cstar']['k20'])), cstar_kclass=float(np.mean(x['cstar']['kclass'])),
                              frozen_k20=x['frozen']['k20'], frozen_kclass=x['frozen']['kclass'],
                              matched_k20=float(np.mean(x['matched']['k20'])), matched_kclass=float(np.mean(x['matched']['kclass'])),
                              matched_own=float(np.mean(x['matched']['own'])), teacher_k20=t.get('k20'), teacher_kclass=t.get('kclass'),
                              cstar_per_seed=x['cstar'], n_labelled=x['n_labelled'])
                o = out[a]
                rows.append([a, x['k'], f4(o['cstar_k20']), f4(o['cstar_kclass']), f4(o['frozen_k20']), f4(o['frozen_kclass']),
                             f4(o['matched_k20']), f4(o['matched_kclass']), f4(o['matched_own']), f4(o['teacher_k20']),
                             f4(o['teacher_kclass'])])
            doc.add(f'## {enc}', '', *table(['aspect', 'K', 'X-c* K=20', 'X-c* K', 'frozen K=20', 'frozen K', 'matched K=20',
                                              'matched K', 'matched own (40 topics)', 'teacher K=20', 'teacher K'], rows), '')
            return dict(aspects=out)
        return sec
    for enc in ENCODERS:
        doc.section(enc, make(enc))
    return doc


def item_per_class_ap(res):
    doc = Doc('per_class_ap', 'mean per-query AP per class at each arm\'s stage 1 pick, MMArt test (X-c* mean over seeds); descriptive')
    teacher = {}
    try:
        teacher = got(res, 'teacher_pick')['aspects']
        doc.add(f'Teacher pick: `{got(res, "teacher_pick")["pick"]}` (one pick for all encoders, D12).', '')
    except NotRun:
        doc.add('The teacher column is empty: its task did not run.', '')

    def make(enc):
        def sec():
            r = got(res, 'ret_pick', enc)
            out = {}
            doc.add(f'## {enc} (picks: X-c* `{r["picks"]["cstar"]}`, frozen `{r["picks"]["frozen"]}`)', '')
            for a in ASPECTS:
                x = r['aspects'][a]
                t = teacher.get(a, {}).get('per_class', {})
                out[a] = dict(cstar=x['cstar'], frozen=x['frozen'], teacher=teacher.get(a))
                rows = [[cl, v['n_queries'], f4(v['ap']), f4(x['frozen']['per_class'][cl]['ap']),
                         f4(t.get(cl, {}).get('ap'))] for cl, v in x['cstar']['per_class'].items()]
                doc.add(f'### {a}: mAP X-c* {f4(x["cstar"]["point"])}, frozen {f4(x["frozen"]["point"])}, '
                        f'teacher {f4(teacher.get(a, {}).get("point"))}', '',
                        *table(['class', 'queries', 'X-c*', 'frozen', 'teacher'], rows), '')
            return dict(picks=r['picks'], aspects=out)
        return sec
    for enc in ENCODERS:
        doc.section(enc, make(enc))
    return doc


def test_cells():
    """The verdict cells of stage 2 ({(encoder, sc, aspect, task): cell}); empty when the files are absent."""
    out = {}
    for path in (_clu().cells_path('test'), _ret().cells_path('test')):
        if Path(path).exists():
            for c in json.loads(Path(path).read_text()):
                out[(c['encoder'], c['sc'], c['aspect'], c['task'])] = c
    return out


def cell_rows(cells, with_test=True):
    tc = test_cells() if with_test else {}
    rows = []
    for c in cells:
        t = tc.get((c['encoder'], c['sc'], c['aspect'], c['task']))
        rows.append([c['encoder'], c['aspect'], c['sc'], c['task'], f4(c['candidate_value']), f4(c['comparator_value']),
                     f'{c["d"]:+.4f}', f'[{c["lo"]:+.4f}, {c["hi"]:+.4f}]', f4(c['r']) if c['r'] else '-',
                     f'{t["d"]:+.4f}' if t else 'n/a', c['n_draws'], c['n_rows']])
    return table(['encoder', 'aspect', 'SC', 'task', 'candidate', 'comparator', 'd', '95% interval', 'r (SC4)',
                  'd on test', 'draws', 'rows'], rows)


def item_far(res):
    doc = Doc('far', f'SC1, SC2, SC4 cells on the far subset (D19: the cut at {MP.CUT_FAR}, {MP.N_FAR} paintings); no outcome is drawn; descriptive')
    doc.add(f'Cells as the scorers write them (`mmart_far`); "d on test" is the verdict cell of the same SC, aspect and encoder, '
            'when the stage 2 files exist. The matched arm is not part of these cells (SC3 is not asked).', '')
    cells_all = []

    def make(enc):
        def sec():
            cells = got(res, 'far_clu', enc)['cells'] + got(res, 'far_ret', enc)['cells']
            cells_all.extend(cells)
            doc.add(f'## {enc}', '', *cell_rows(cells), '')
            return dict(cells=cells)
        return sec
    for enc in ENCODERS:
        doc.section(enc, make(enc))
    _write_cell_files('far', cells_all)
    return doc


def _write_cell_files(keyset, cells):
    """Merge the keyset's cells into `mt_clu.cells_path(keyset)` / `mt_ret.cells_path(keyset)` (the main process only: the
    tasks run in parallel and must not write one file)."""
    clu, ret = _clu(), _ret()
    for task, path in (('clu', clu.cells_path(keyset)), ('ret', ret.cells_path(keyset))):
        part = [c for c in cells if c['task'] == task]
        if part:
            clu.write_cells(path, part)


def item_shared_held(res):
    doc = Doc('shared_held', 'SC1, SC2, SC4 cells on the 11,528 shared ArtELingo held paintings: MMArt labels beside ArtELingo labels; descriptive')
    doc.add('Same paintings, same picks, same ranges (MMArt validation), same draws; only the labels differ, which separates new '
            'labels from new images. There is no SC3 (the matched arm\'s MMArt probabilities exist for `new` only). No outcome is drawn.', '')
    cells_all = []

    def make(enc):
        def sec():
            cells = got(res, 'shared_clu', enc)['cells'] + got(res, 'shared_ret', enc)['cells']
            cells_all.extend(cells)
            by = {(c['sc'], c['aspect'], c['task'], c['data']): c for c in cells}
            rows = []
            for a in ASPECTS:
                for sc, task in (('SC1', 'clu'), ('SC2', 'ret'), ('SC4', 'clu'), ('SC4', 'ret')):
                    m, e = by.get((sc, a, task, 'mmart_shared_held')), by.get((sc, a, task, DATA_SHARED_AE))
                    if m is None or e is None:
                        continue
                    rows.append([a, sc, m['task'], f4(m['candidate_value']), f4(m['comparator_value']), f'{m["d"]:+.4f}',
                                 f'[{m["lo"]:+.4f}, {m["hi"]:+.4f}]', f4(e['candidate_value']), f4(e['comparator_value']),
                                 f'{e["d"]:+.4f}', f'[{e["lo"]:+.4f}, {e["hi"]:+.4f}]', m.get('n_labelled', m.get('n_queries', '')), e.get('n_labelled', e.get('n_queries', ''))])
            doc.add(f'## {enc}', '', *table(['aspect', 'SC', 'task', 'cand (MMArt)', 'comp (MMArt)', 'd (MMArt)', 'interval (MMArt)',
                                              'cand (ArtELingo)', 'comp (ArtELingo)', 'd (ArtELingo)', 'interval (ArtELingo)',
                                              'labelled (queries) MMArt', 'labelled (queries) ArtELingo'], rows), '')
            return dict(cells=cells)
        return sec
    for enc in ENCODERS:
        doc.section(enc, make(enc))
    _write_cell_files('shared_held', cells_all)
    return doc


def item_unseen_artist(res):
    doc = Doc('unseen_artist', 'D14: test paintings whose artist has no ArtELingo train painting: the count and the artists only; descriptive')

    def sec():
        d = _data()
        keys = d.keys('unseen_artist')
        test = set(d.keys('test'))
        assert set(keys) <= test and len(set(keys)) == len(keys), 'unseen_artist must be a subset of test'
        train = d.artelingo_train_names()
        artists = {}
        for k in keys:
            artists.setdefault(d.artist_of(k), []).append(k)
        train_artists = {d.artist_of(n) for n in train}
        assert not (set(artists) & train_artists)
        n_clusters = MP.K
        out = dict(n_paintings=len(keys), n_artists=len(artists), n_test=len(test), n_train_names=len(train),
                   n_train_artists=len(train_artists), expected_paintings=15, matches_expected=len(keys) == 15,
                   artists={a: len(v) for a, v in sorted(artists.items())}, keys=keys,
                   n_clusters=n_clusters, cells='none: fewer paintings than clusters')
        doc.add(f'{len(keys)} test paintings by {len(artists)} artists (of {len(test)} test paintings) have an artist with no '
                f'painting among the {len(train)} ArtELingo train paintings ({len(train_artists)} artists; list '
                f'`ct2_data.split2("full")` fit). Expected from the build notes: 15 paintings by 12 artists '
                f'({"matches" if out["matches_expected"] else "DIFFERS"}).', '',
                f'No SC1, SC2 or SC4 cells are given: {len(keys)} paintings are fewer than the {n_clusters} clusters of the k-means, '
                'so any AMI or mAP cell on them would be noise (D14, agent default).', '',
                *table(['artist', 'paintings'], [[a, n] for a, n in out['artists'].items()]), '')
        return out
    doc.section('subset', sec)
    return doc


def pairs_inside(x, cut, block=2048):
    """Pairs i < j of the unit float32 rows with cosine >= cut (inclusive, float32 value compared in float64). Returns
    (i, j, cos) int64, int64, float32 arrays."""
    ii, jj, cc = [], [], []
    for a in range(0, len(x), block):
        sim = x[a:a + block] @ x.T
        r, c = np.nonzero(sim.astype(np.float64) >= cut)
        keep = (c > r + a)
        ii.append(r[keep] + a)
        jj.append(c[keep])
        cc.append(sim[r[keep], c[keep]])
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(cc)


def item_near_copy(res):
    doc = Doc('near_copy', 'near-copy count and histogram of the nearest ArtELingo cosine, removed paintings per class at each cut, '
              'test pairs inside the cut set; descriptive')

    def count_sec():
        import pandas as pd
        d = _data()
        bs_emoset = _m('bs_emoset')
        new = ordered_keys('new')
        nc = pd.read_parquet(MP.MMART_NEARCOPY)
        assert nc.key.is_unique and set(nc.key) == set(new) and len(nc) == len(new), 'near-copy file: one row per new key'
        best = nc.set_index('key').max_cos.loc[new].to_numpy().astype(np.float32)
        assert np.isfinite(best).all()
        hist = bs_emoset.histogram(best)
        stored = read_json(MP.PREP_NEARCOPY)['counts'] if MP.PREP_NEARCOPY.exists() else {}
        counts = {f'{c:.2f}': int(bs_emoset.removed_mask(best, c).sum()) for c in (0.80, 0.85, 0.90, 0.95, 0.99, 1.0)}
        diff = {k: (counts[k], stored[k]) for k in stored if k in counts and counts[k] != stored[k]}
        edges = hist['edges']
        doc.add(f'{len(new)} new MMArt paintings; nearest ArtELingo train/validation painting by default-Qwen cosine '
                f'(`{MP.MMART_NEARCOPY}`). Paintings with cosine below 0.80: {hist["n_below"]}.', '',
                *table(['cosine >=', 'paintings'], [[k, v] for k, v in counts.items()]), '',
                *table(['bin', 'paintings'], [[f'[{edges[i]:.2f}, {edges[i + 1]:.2f}' + (']' if i == 19 else ')'), c]
                                              for i, c in enumerate(hist['counts'])]), '')
        if diff:
            doc.add(f'The prep round\'s stored counts differ from the file at: {diff}.', '')
        return dict(n_new=len(new), counts_ge=counts, histogram=hist, prep_counts_differ=diff)

    def removed_sec():
        d = _data()
        new = ordered_keys('new')
        y = aspect_labels(new)
        where = {k: i for i, k in enumerate(new)}
        out, lines = {}, []
        for cut in CUTS:
            c = d.cut(cut)
            sets = dict(exact=c['exact'], near=c['near'], md5=c['md5'])
            removed = c['exact'] + c['near']
            out[f'{cut:.2f}'] = dict(n_new=c['n_new'], n_kept=len(c['kept']), n_exact=len(c['exact']), n_near=len(c['near']),
                                     n_md5=len(c['md5']), n_removed_exact_and_near=len(removed), per_class={})
            for a in ASPECTS:
                idx = np.array([where[k] for k in removed], dtype=np.int64)
                ya = y[a][idx] if len(idx) else np.zeros(0, dtype=np.int64)
                out[f'{cut:.2f}']['per_class'][a] = {cl: int((ya == i).sum()) for i, cl in enumerate(MP.CLASSES[a])}
                out[f'{cut:.2f}']['per_class'][a]['no label'] = int((ya < 0).sum())
        doc.add('Removed = byte copies (exact) plus near-copies at the cut (cosine >= cut to an ArtELingo train or validation '
                'painting); the md5-duplicate removal (second and later paintings of an md5 group among the rest) is counted separately.', '',
                *table(['cut', 'kept', 'exact', 'near', 'md5 duplicates'],
                       [[k, v['n_kept'], v['n_exact'], v['n_near'], v['n_md5']] for k, v in out.items()]), '')
        for a in ASPECTS:
            classes = list(MP.CLASSES[a]) + ['no label']
            total = {cl: int((y[a] == i).sum()) if cl != 'no label' else int((y[a] < 0).sum()) for i, cl in enumerate(classes)}
            doc.add(f'### removed per class: {a}', '',
                    *table(['class', 'new'] + [f'cut {k}' for k in out], [[cl, total[cl]] + [out[k]['per_class'][a][cl] for k in out]
                                                                         for cl in classes]), '')
        return dict(cuts=out)

    def pairs_sec():
        d = _data()
        keys = ordered_keys('test')
        x = unit32(d.features('qwen_default', keys))
        i, j, c = pairs_inside(x, PAIR_CUT)
        y = aspect_labels(keys)
        inv = sorted(set(i.tolist()) | set(j.tolist()))
        same = {}
        for a in ASPECTS:
            both = (y[a][i] >= 0) & (y[a][j] >= 0)
            same[a] = dict(pairs_both_labelled=int(both.sum()),
                           same_class=int((y[a][i][both] == y[a][j][both]).sum()))
        top = np.argsort(-c)[:20]
        examples = [dict(a=keys[i[t]], b=keys[j[t]], cosine=float(c[t])) for t in top]
        ge99 = int((c.astype(np.float64) >= 0.99).sum())
        doc.add(f'Test set (cut at {MP.CUT_TEST}, {len(keys)} paintings): {len(i)} pairs of test paintings with default-Qwen '
                f'cosine >= {PAIR_CUT} (inclusive), involving {len(inv)} paintings; {ge99} of the pairs at >= 0.99.', '',
                *table(['aspect', 'pairs with both labelled', 'of these, same class'],
                       [[a, v['pairs_both_labelled'], v['same_class']] for a, v in same.items()]), '',
                *table(['painting', 'painting', 'cosine'], [[e['a'], e['b'], f4(e['cosine'])] for e in examples]), '')
        return dict(n_test=len(keys), cut=PAIR_CUT, n_pairs=int(len(i)), n_paintings=len(inv), n_pairs_ge_099=ge99,
                    agreement=same, top_pairs=examples)
    doc.section('count_and_histogram', count_sec)
    doc.section('removed_per_class', removed_sec)
    doc.section('pairs_inside_test', pairs_sec)
    return doc


def item_teacher_gain_share(res):
    doc = Doc('teacher_gain_share', 'share of the teacher\'s gain over the plain image kept by X-c*: (X-c* - frozen) / (teacher - frozen); '
              'MMArt test; descriptive')

    def sec():
        clu, ret = _clu(), _ret()
        paths = [clu.cells_path('test'), ret.cells_path('test')]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise NotRun('the stage 2 cells files are missing', missing)
        cells = {}
        for p in paths:
            for c in json.loads(Path(p).read_text()):
                assert c['data'] == 'mmart_test', c['data']
                cells[(c['encoder'], c['aspect'], c['task'], c['sc'])] = c
        rows, out = [], []
        for enc in ENCODERS:
            for a in ASPECTS:
                for task, plain_sc in (('clu', 'SC1'), ('ret', 'SC2')):
                    p, t = cells.get((enc, a, task, plain_sc)), cells.get((enc, a, task, 'SC4'))
                    if p is None or t is None:
                        continue
                    plain, teacher, model = p['comparator_value'], t['comparator_value'], t['candidate_value']
                    gap = teacher - plain
                    row = dict(encoder=enc, aspect=a, task=task, plain=plain, teacher=teacher, model=model,
                               model_in_plain_cell=p['candidate_value'], gap=gap,
                               share=(model - plain) / gap if gap > 0 else None,
                               note=None if gap > 0 else 'teacher not above the plain image: no share')
                    out.append(row)
                    rows.append([enc, a, task, f4(plain), f4(model), f4(teacher), f4(gap), f4(row['share'], 3)])
        if not out:
            raise NotRun('no encoder has an SC1 or SC2 cell with an SC4 cell', paths)
        doc.add(*table(['encoder', 'aspect', 'task', 'frozen (plain)', 'X-c*', 'teacher', 'gap', 'share kept'], rows), '',
                'Clustering: AMI at 20 clusters; retrieval: mAP at each arm\'s own pick. A gap of zero or below gives no share.', '')
        return dict(rows=out)
    doc.section('cells', sec)
    return doc


def item_seed_spreads(res):
    doc = Doc('seed_spreads', 'max - min over the five seeds of X-c* on MMArt test beside the stage 1 validation ranges; '
              'ratio = test spread / validation range; descriptive')

    def make(enc):
        def sec():
            clu_t = got(res, 'clu_test', enc)['aspects']
            ret_r = got(res, 'ret_r0', enc)['aspects']
            rec = committed(f'stage1_{enc}.json')
            r_clu, r_ret = _clu().record_ranges(rec, enc), _ranges_block(rec['ret_ranges'])
            try:
                ret_p = got(res, 'ret_pick', enc)['aspects']
            except NotRun:
                ret_p = None
            out, rows = {}, []
            for a in ASPECTS:
                s_clu, s_r0 = spread(clu_t[a]['cstar']['k20']), spread(ret_r[a]['cstar'])
                s_pick = spread(ret_p[a]['cstar']['per_seed']) if ret_p else None
                out[a] = dict(ami_k20=dict(per_seed=clu_t[a]['cstar']['k20'], test_spread=s_clu, validation_range=r_clu[a],
                                           ratio=s_clu / r_clu[a]),
                              map_r0=dict(per_seed=ret_r[a]['cstar'], test_spread=s_r0, validation_range=r_ret[a], ratio=s_r0 / r_ret[a]),
                              map_at_pick=None if ret_p is None else dict(per_seed=ret_p[a]['cstar']['per_seed'], test_spread=s_pick,
                                                                          validation_range=r_ret[a], ratio=s_pick / r_ret[a]))
                rows.append([a, f4(s_clu), f4(r_clu[a]), f4(s_clu / r_clu[a], 2), f4(s_r0), f4(r_ret[a]), f4(s_r0 / r_ret[a], 2),
                             f4(s_pick), f4(s_pick / r_ret[a], 2) if s_pick is not None else 'n/a'])
            doc.add(f'## {enc}', '', *table(['aspect', 'AMI spread (test)', 'AMI range (val)', 'ratio', 'mAP R0 spread (test)',
                                              'mAP range (val, R0)', 'ratio', 'mAP at pick spread (test)', 'ratio'], rows), '')
            return dict(aspects=out)
        return sec
    for enc in ENCODERS:
        doc.section(enc, make(enc))
    doc.add('The validation range of mAP is that of R0 (stage 1); the spread at the pick is shown beside it for the unit the SC4 '
            'cells use.', '')
    return doc


def item_percept_strongest(res):
    doc = Doc('percept_strongest', 'strongest PercepT entry per aspect: the largest of the four matched arms\' larger readings '
              '(own topics, 20 clusters) on MMArt test, AMI, five-seed mean (C1); descriptive')

    def sec():
        per, out = {}, {}
        for enc in ENCODERS:
            r = got(res, 'clu_test', enc)['aspects']
            for a in ASPECTS:
                own, r20 = float(np.mean(r[a]['matched']['own'])), float(np.mean(r[a]['matched']['k20']))
                reading = 'own' if own >= r20 else 'r20'
                per.setdefault(a, {})[enc] = dict(entry=MP.MATCHED[enc], own=own, r20=r20, larger_reading=reading, larger=max(own, r20))
        cells = test_cells()
        for a in ASPECTS:
            best = max(per[a], key=lambda e: per[a][e]['larger'])
            out[a] = dict(strongest_encoder=best, strongest_entry=per[a][best]['entry'], value=per[a][best]['larger'],
                          reading=per[a][best]['larger_reading'], arms=per[a])
            for enc in ENCODERS:                      # beside the SC3 cell's comparator value, when the stage 2 file exists
                c = cells.get((enc, 'SC3', a, 'clu'))
                per[a][enc]['sc3_cell_comparator_value'] = None if c is None else c['comparator_value']
        doc.add(*table(['aspect', 'strongest', 'encoder', 'larger reading', 'AMI'],
                       [[a, v['strongest_entry'], v['strongest_encoder'], v['reading'], f4(v['value'])] for a, v in out.items()]), '')
        for a in ASPECTS:
            doc.add(f'### {a}', '', *table(['encoder', 'entry', 'own', '20 clusters', 'larger', 'SC3 cell comparator value'],
                                           [[e, v['entry'], f4(v['own']), f4(v['r20']), f4(v['larger']), f4(v['sc3_cell_comparator_value'])]
                                            for e, v in per[a].items()]), '')
        doc.add('Retrieval: the matched arm has no mAP result in the verdict cells; its R0 mAP is in the item r0_rows.', '')
        return dict(aspects=out)
    doc.section('clustering', sec)
    return doc


FUNCS = dict(r0_rows=item_r0_rows, clustering_k_class=item_clustering_k_class, per_class_ap=item_per_class_ap, far=item_far,
             unseen_artist=item_unseen_artist, shared_held=item_shared_held, near_copy=item_near_copy,
             teacher_gain_share=item_teacher_gain_share, seed_spreads=item_seed_spreads,
             percept_strongest=item_percept_strongest)
assert tuple(FUNCS) == ITEMS


# ============================================================================================ driver
def run_items(items, workers=MAX_WORKERS):
    """Run the tasks the items need, then write each item's files. Returns {item: {'status', 'missing', 'seconds'}}."""
    for it in items:
        assert it in FUNCS, it
    t0 = time.monotonic()
    res = run_tasks(task_keys(items), workers) if any(ITEM_TASKS[i] for i in items) else {}
    t_tasks = time.monotonic() - t0
    summary = {}
    for it in items:
        t1 = time.monotonic()
        try:
            doc = FUNCS[it](res)
            d = doc.finish()
        except Exception as e:  # noqa: BLE001  (an item bug must not stop the others)
            traceback.print_exc()
            doc = Doc(it, FUNCS[it].__doc__ or '')
            doc.d['sections']['item'] = dict(status='failed', reason=f'{type(e).__name__}: {e}', missing=[])
            d = doc.finish()
        d['seconds'] = round(time.monotonic() - t1, 1)
        d['task_seconds'] = {f'{k[0]}:{k[1]}' if k[1] else k[0]: v['seconds'] for k, v in res.items()
                             if k[0] in ITEM_TASKS[it]}
        doc.write()
        summary[it] = dict(status=d['status'], missing=d['missing'], seconds=d['seconds'])
        print(f'diag {it}: {d["status"]} ({d["seconds"]} s)' + (f', missing {d["missing"][:2]}' if d['missing'] else ''), flush=True)
    summary['_tasks_seconds'] = round(t_tasks, 1)
    (diag_dir() / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('item', choices=('all',) + ITEMS)
    p.add_argument('--workers', type=int, default=MAX_WORKERS, choices=range(1, MAX_WORKERS + 1),
                   help='processes of 8 threads each (default 3)')
    a = p.parse_args(argv)
    summary = run_items(list(ITEMS) if a.item == 'all' else [a.item], a.workers)
    bad = [k for k, v in summary.items() if not k.startswith('_') and v['status'] == 'failed']
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
