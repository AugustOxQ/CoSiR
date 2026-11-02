"""r14 diagnostics (ticket 12b, spec §3.6): descriptive read-outs after the verdicts. Deciding nothing: no item feeds a verdict,
a pick or a record. Every number is a function of tickets 04 to 11 or an older round; this module adds only differences,
shares and spreads. Each item writes `OUT/diag/<item>.{json,md}`; the knob curves also `OUT/diag/knob_curves.png`.

Items (one function each, source function in its docstring): r0_rows, emoset_k8, emoset_per_class_ap, photo_minus_cstar,
teacher_gain_share, held_spreads, r_e_vs_spread, emoset_uncut, node_vs_3090, knob_curves, photo_length.

A missing input gives the item (or the section of it) status "not run" with the missing path; the other items go on.
Status of an item: "ok" (all sections ran), "partial" (some did not), "not run" (none did).

CLI: `python bs_diag.py all|<item> [--smoke] [--workers N]`. `--smoke` uses 40 and 100 bootstrap draws and the synthetic
EmoSet source `SmokeSource`; outputs go where BS_OUTPUTS says (the smoke generator writes the file inputs there).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bs_paths as BP  # noqa: E402

ITEMS = ('r0_rows', 'emoset_k8', 'emoset_per_class_ap', 'photo_minus_cstar', 'teacher_gain_share', 'held_spreads',
         'r_e_vs_spread', 'emoset_uncut', 'node_vs_3090', 'knob_curves', 'photo_length')
N_CLU, N_RET = 2000, 10000          # the draw counts of r12 (stage 3)
SMOKE_DRAWS = (40, 100)


class NotRun(Exception):
    """An input is missing: the section is reported as "not run" with the paths."""

    def __init__(self, why, missing=()):
        super().__init__(why)
        self.missing = [str(m) for m in missing]


def out_dir():
    d = Path(BP.OUT) / 'diag'
    d.mkdir(parents=True, exist_ok=True)
    return d


def need(*paths):
    gone = [p for p in paths if not Path(p).exists()]
    if gone:
        raise NotRun('missing input', gone)


def read_json(path):
    need(path)
    return json.loads(Path(path).read_text())


def section(doc, name, fn):
    """Run one section of an item; a missing input marks only this section."""
    try:
        doc['sections'][name] = dict(status='ok', **fn())
    except NotRun as e:
        doc['sections'][name] = dict(status='not run', reason=str(e), missing=e.missing)


def new_doc(item, note):
    return dict(item=item, note=note, sections={})


def finish(doc):
    st = [s['status'] for s in doc['sections'].values()]
    doc['status'] = 'ok' if all(s == 'ok' for s in st) else ('not run' if not any(s == 'ok' for s in st) else 'partial')
    doc['missing'] = sorted({m for s in doc['sections'].values() for m in s.get('missing', [])})
    return doc


def f4(v, n=4):
    return 'n/a' if v is None else f'{v:.{n}f}'


def table(head, rows):
    return ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]


def write_item(item, doc, md_lines):
    d = out_dir()
    clean = json.loads(json.dumps(doc, default=float, allow_nan=True))
    (d / f'{item}.json').write_text(json.dumps(clean, indent=1) + '\n')
    head = [f'# diag: {item}', '', f'Status: {doc["status"]}. {doc["note"]}', '']
    for m in doc['missing']:
        head.append(f'- missing: `{m}`')
    for name, s in doc['sections'].items():
        if s['status'] != 'ok':
            head.append(f'- section {name}: not run ({s["reason"]})')
    (d / f'{item}.md').write_text('\n'.join(head + [''] + md_lines) + '\n')


# ============================================================================================ synthetic source (smoke)
class SmokeSource:
    """The source interface of `bs_emoset_score` on synthetic features (class centres plus noise), for the smoke run, with an
    uncut test set (`keys_uncut`): the cut test keys plus `extra` removed photos."""
    D = 16
    NOISE = {'photo_qwen': 0.9, 'cstar_qwen': 1.1, 'photo_siglip2': 1.3, 'photo_siglip1': 1.2, 'photo_clip_l14': 1.4,
             'plain_qwen': 1.6, 'teacher_emotion_focus': 1.0, 'teacher_emotion_focus_photo': 1.05,
             'frozen_siglip2': 1.5, 'frozen_siglip1': 1.5, 'frozen_clip_l14': 1.5, 'crl_clip_l14': 1.4}

    def __init__(self, n_val=240, n_test=320, extra=24, seed=0):
        import bs_emoset
        self.E = bs_emoset
        rng = np.random.default_rng(seed)
        self.centres = rng.standard_normal((8, self.D))
        self.label, self.key = {}, {}
        for split, n in (('val', n_val), ('test', n_test), ('removed', extra)):
            cls = rng.permutation(np.arange(n) % 8)
            self.key[split] = [f'image/{BP.EMOSET_CLASSES[c]}/{split}_{i:05d}.jpg' for i, c in enumerate(cls)]
            self.label.update({k: int(c) for k, c in zip(self.key[split], cls)})

    def keys(self, split):
        return self.E.order(self.key[split])

    def keys_uncut(self, split):
        return self.E.order(self.key[split] + (self.key['removed'] if split == 'test' else []))

    def labels(self, keys):
        return np.asarray([self.label[k] for k in keys], dtype=np.int64)

    def _x(self, arm, keys, seed):
        import zlib
        rs = np.random.default_rng(zlib.crc32(f'{arm}/{seed}'.encode()))
        x = self.centres[[self.label[k] for k in keys]] + self.NOISE[arm] * rs.standard_normal((len(keys), self.D))
        x = x.astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def stored(self, arm, split, seed, keys):
        return self._x(arm, keys, seed).astype(np.float16)

    def frozen(self, arm, keys):
        return self._x(arm, keys, None)

    def export_sha(self, arm):
        return {f'{sp}__s{s}': f'{arm}-{sp}-{s}'.ljust(64, '0') for sp in ('val', 'test') for s in range(5)}

    def descriptive_available(self):
        import bs_emoset_score as S
        return list(S.DESCRIPTIVE_ARMS), {}


# ============================================================================================ EmoSet context
class Ctx:
    """The stage 2 record, the cut keys and labels, and the per-arm test scoring, memoized (`bs_emoset_score.score_arm`)."""

    def __init__(self, source, smoke, workers):
        import bs_emoset_score as S
        self.S, self.source, self.workers = S, source, workers
        self.n_clu, self.n_ret = SMOKE_DRAWS if smoke else (N_CLU, N_RET)
        rec_file = S.record_file()
        alt = Path(BP.OUT) / 'stage2' / S.STAGE2_NAME
        path = alt if smoke or not rec_file.exists() else rec_file
        self.rec = S.check_record(read_json(path))
        self.val_keys, self.test_keys = source.keys('val'), source.keys('test')
        S.check_order(self.val_keys)
        S.check_order(self.test_keys)
        self.labels = S._check_labels(source.labels(self.test_keys), len(self.test_keys))
        stage3 = Path(BP.OUT) / 'stage3/partitions_emoset'
        self.cache = stage3 if stage3.exists() else out_dir() / 'cache/partitions_emoset'
        self.picks = {**{a: d['pick'] for a, d in self.rec['readout'].items()},
                      **{a: d['pick'] for a, d in self.rec['descriptive'].items()}}
        self.res = {}

    def arm(self, arm):
        if arm not in self.picks:
            raise NotRun(f'{arm}: no read-out in the stage 2 record (arm not run)', [Path(BP.OUT) / 'stage2'])
        if arm not in self.res:
            try:
                self.res[arm] = self.S.score_arm(self.source, arm, self.picks[arm], self.val_keys, self.test_keys,
                                                 self.labels, self.cache, self.workers)
            except FileNotFoundError as e:
                raise NotRun(f'{arm}: {e}', [e.filename])
        return self.res[arm]

    def arms(self):
        return list(self.S.BAR_ARMS) + list(self.rec['descriptive'])


_CTX = {}


def ctx(smoke, workers):
    if 'c' not in _CTX:
        _CTX['c'] = Ctx(SmokeSource() if smoke else __import__('bs_emoset_score').RealSource(), smoke, workers)
    return _CTX['c']


# ============================================================================================ items
def emoset_rows(c):
    """Sheet rows from `OUT/stage3/sheet_emoset.json` (ticket 08) when it exists, else `bs_emoset_score.sheet_row` over `score_arm`."""
    sheet = Path(BP.OUT) / 'stage3' / c.S.SHEET_NAME
    if sheet.exists() and not _CTX.get('smoke'):
        return json.loads(sheet.read_text())['rows'], str(sheet)
    rows = [c.S.sheet_row(c.arm(a), 'bar' if a in c.S.BAR_ARMS else 'descriptive') for a in c.arms()]
    return rows, 'computed with bs_emoset_score.score_arm and sheet_row'


def per_class(ap, rows, labels):
    """Mean AP per class over the queries of the class; ap [nq] or [5, nq] (seeds averaged)."""
    a = np.asarray(ap, dtype=np.float64)
    a = a.mean(axis=0) if a.ndim == 2 else a
    y = labels[np.asarray(rows)]
    return {cl: float(a[y == i].mean()) for i, cl in enumerate(BP.EMOSET_CLASSES) if (y == i).any()}


def r0_rows(smoke, workers):
    """R0 rows for every arm. EmoSet test: R0 mAP of every arm (`bs_emoset_score.sheet_row`, ticket 08's sheet). ArtELingo: held
    R0 mAP per seed of X-c* and of X's frozen image (`bs_artret.score_cstar_held` / `score_frozen_held` at R0) and X-c*'s
    validation R0 mAP (`bs_artret.validation_ranges` file). Affection: the photo models' validation R0 mAP (`pf_stage2.ranges`
    in the stage 1 file). Teachers on ArtELingo and Affection keep only their picked read-out in the stored outputs."""
    doc = new_doc('r0_rows', 'plain cosine retrieval (R0) mAP; descriptive')
    md = []

    def emo():
        c = ctx(smoke, workers)
        rows, src = emoset_rows(c)
        md.extend(['## EmoSet test, R0 mAP', ''] + table(['arm', 'role', 'R0 mAP', 'picked read-out', 'picked mAP'],
                   [[r['arm'], r['role'], f4(r['map_r0']), r['readout'], f4(r['map_picked'])] for r in rows]) + [''])
        return dict(source=src, rows=[dict(arm=r['arm'], role=r['role'], map_r0=r['map_r0'], map_r0_per_seed=r.get('map_r0_per_seed'))
                                      for r in rows])

    def art():
        out, lines = {}, ['## ArtELingo, R0 mAP (held: X-c* and frozen; validation: X-c*)', '']
        for enc in BP.NEW_ENCODERS:
            try:
                held = artelingo_held(enc, workers)
                val = read_json(Path(BP.OUT) / f'stage1/artret_ranges_{enc}.json')['per_seed']
            except NotRun as e:
                out[enc] = dict(status='not run', reason=str(e), missing=e.missing)
                continue
            out[enc] = dict(status='ok', held=dict(cstar_r0=held['ret']['cstar_r0'], frozen_r0=held['ret']['frozen_r0']),
                            validation_cstar_r0_mean={a: float(np.mean(list(v.values()))) for a, v in val.items()})
            for a in BP.ASPECTS:
                lines.append(f'- {enc} {a}: held X-c* R0 mean {np.mean(held["ret"]["cstar_r0"][a]):.4f}, frozen R0 '
                             f'{held["ret"]["frozen_r0"][a]:.4f}; validation X-c* R0 mean {out[enc]["validation_cstar_r0_mean"][a]:.4f}')
        if not any(v['status'] == 'ok' for v in out.values()):
            raise NotRun('no encoder has its ArtELingo inputs', [m for v in out.values() for m in v.get('missing', [])])
        md.extend(lines + [''])
        return dict(encoders=out)

    def aff():
        out, lines = {}, ['## Affection, validation R0 mAP of the X photo models (mean over seeds)', '']
        for enc in BP.NEW_ENCODERS:
            try:
                rec = read_json(Path(BP.OUT) / f'stage1/affection_ranges_{enc}.json')
            except NotRun as e:
                out[enc] = dict(status='not run', reason=str(e), missing=e.missing)
                continue
            ps = rec['ranges_doc']['map']['per_seed']
            ps = list(ps.values()) if isinstance(ps, dict) else list(ps)
            out[enc] = dict(status='ok', validation_photo_r0_mean=float(np.mean(ps)), per_seed=ps)
            lines.append(f'- photo_{enc}: {np.mean(ps):.4f}')
        if not any(v['status'] == 'ok' for v in out.values()):
            raise NotRun('no stage 1 Affection file', [m for v in out.values() for m in v.get('missing', [])])
        md.extend(lines + [''])
        return dict(encoders=out)

    for n, fn in (('emoset', emo), ('artelingo', art), ('affection', aff)):
        section(doc, n, fn)
    return finish(doc), md


def emoset_k8(smoke, workers):
    """EmoSet with 8 clusters: ticket 08's descriptive row (`OUT/stage3/sheet_emoset.json`, `ami_k8` beside `ami_k20`) if it
    exists, else its function (`bs_emoset_score.score_arm`, which fits `partitions_k(x, 8)`)."""
    doc = new_doc('emoset_k8', 'AMI of k-means with 8 clusters (the class count) beside the 20-cluster AMI; EmoSet test; descriptive')
    md = []

    def sec():
        c = ctx(smoke, workers)
        rows, src = emoset_rows(c)
        md.extend(table(['arm', 'role', 'AMI k=20', 'AMI k=8'], [[r['arm'], r['role'], f4(r['ami_k20']), f4(r['ami_k8'])] for r in rows]))
        return dict(source=src, rows=[dict(arm=r['arm'], role=r['role'], ami_k20=r['ami_k20'], ami_k8=r['ami_k8']) for r in rows])
    section(doc, 'emoset', sec)
    return finish(doc), md


def emoset_per_class_ap(smoke, workers):
    """EmoSet per-class AP: per-query AP of `bs_emoset_score.score_arm` (R0 and the arm's pick), averaged within each class
    (queries joined to the class labels by row index; seeds averaged). EmoSet test, cut set."""
    doc = new_doc('emoset_per_class_ap', 'mean per-query AP per class, test photos (cut set); descriptive')
    md = []

    def sec():
        c = ctx(smoke, workers)
        out = {}
        for a in c.arms():
            r = c.arm(a)
            out[a] = dict(readout=r['pick'], at_pick=per_class(r['picked']['ap'], r['picked']['rows'], c.labels),
                          r0=per_class(r['r0']['ap'], r['r0']['rows'], c.labels))
        classes = list(BP.EMOSET_CLASSES)
        for key, title in (('at_pick', 'at the picked read-out'), ('r0', 'at R0')):
            md.extend([f'## {title}', ''] + table(['arm'] + classes, [[a] + [f4(out[a][key].get(k)) for k in classes] for a in out]) + [''])
        return dict(arms=out)
    section(doc, 'emoset', sec)
    return finish(doc), md


def photo_minus_cstar(smoke, workers):
    """The photo model minus frozen c* on EmoSet test, paired: AMI (20 clusters) by `ac_cluster.paired` with `shared_seeds=True`
    (the photo model of seed s starts from c* seed s, so the seeds are one set), mAP at each arm's pick and at R0 by `rt_stats.paired`
    on the per-query AP [5, nq]. Differences also in units of r_E (stage 2 record)."""
    doc = new_doc('photo_minus_cstar', 'photo_qwen minus cstar_qwen, EmoSet test, paired bootstrap; descriptive, no outcome')
    md = []

    def sec():
        import ac_cluster
        import rt_stats
        c = ctx(smoke, workers)
        ph, cs = c.arm('photo_qwen'), c.arm('cstar_qwen')
        r_clu, r_map = c.rec['ranges']['clu']['r'], c.rec['ranges']['map']['r']
        b = ac_cluster.paired(ph['parts20'], cs['parts20'], c.labels, c.n_clu, c.workers, shared_seeds=True)
        out = dict(ami=dict(photo=b['A_cand'], cstar=b['A_comp'], d=b['d'], lo=b['interval'][0], hi=b['interval'][1],
                            d_in_r_E=b['d'] / r_clu, n_draws=c.n_clu))
        for name, key in (('map_at_pick', 'picked'), ('map_r0', 'r0')):
            assert np.array_equal(ph[key]['rows'], cs[key]['rows']), 'different queries'
            p = rt_stats.paired(ph[key]['ap'], cs[key]['ap'], c.n_ret)
            out[name] = dict(photo=ph[key]['point'], cstar=cs[key]['point'], d=p['d'], lo=p['interval'][0], hi=p['interval'][1],
                             d_in_r_E=p['d'] / r_map, n_draws=c.n_ret)
        out['readouts'] = dict(photo=ph['pick'], cstar=cs['pick'])
        md.extend(table(['measure', 'photo', 'frozen c*', 'd', '95% interval', 'd / r_E', 'draws'],
                        [[k, f4(v['photo']), f4(v['cstar']), f4(v['d']), f'[{f4(v["lo"])}, {f4(v["hi"])}]', f4(v['d_in_r_E'], 2),
                          v['n_draws']] for k, v in out.items() if k != 'readouts']))
        return out
    section(doc, 'emoset', sec)
    return finish(doc), md


def teacher_gain_share(smoke, workers):
    """The share of the teacher's gain over the plain image kept by the model: (model - plain) / (teacher - plain), from the
    cells files of tickets 05 to 08 (`OUT/stage3/cells_*.json`, the values `candidate_value` / `comparator_value` of the cells that
    compare one candidate with the plain or frozen image (SC1, SC2, SC4, SC5, SC7, SC8) and with the teacher (SC6, SC9,
    SC10)). Covers each X-c* on ArtELingo, each X photo model on Affection, and the photo model on EmoSet. A gap of zero or
    below gives no share."""
    doc = new_doc('teacher_gain_share', 'kept share = (model - plain) / (teacher - plain), per candidate, data, aspect and task; descriptive')
    md = []

    def sec():
        files = sorted((Path(BP.OUT) / 'stage3').glob('cells_*.json'))
        need(Path(BP.OUT) / 'stage3')
        if not files:
            raise NotRun('no cells file', [Path(BP.OUT) / 'stage3/cells_*.json'])
        groups = {}
        for f in files:
            for cell in json.loads(f.read_text()):
                g = groups.setdefault((cell['encoder'], cell['data'], cell['aspect'], cell['task'], cell['candidate']), {})
                comp = cell['comparator']
                if comp.startswith('teacher'):
                    g['teacher'] = cell
                elif comp.startswith('frozen_') or comp == 'plain_qwen':
                    g['plain'] = cell
        rows = []
        for (enc, data, aspect, task, cand), g in sorted(groups.items()):
            if 'teacher' not in g or 'plain' not in g:
                continue
            t, p = g['teacher'], g['plain']
            plain, teacher, model = p['comparator_value'], t['comparator_value'], t['candidate_value']
            gap = teacher - plain
            rows.append(dict(encoder=enc, data=data, aspect=aspect, task=task, candidate=cand, plain=plain, teacher=teacher,
                             model=model, model_in_plain_cell=p['candidate_value'], gap=gap,
                             share=(model - plain) / gap if gap > 0 else None,
                             note=None if gap > 0 else 'teacher not above the plain image: no share'))
        if not rows:
            raise NotRun('no candidate has both a plain/frozen cell and a teacher cell', files)
        md.extend(table(['encoder', 'data', 'aspect', 'task', 'candidate', 'plain', 'teacher', 'model', 'share'],
                        [[r['encoder'], r['data'], r['aspect'], r['task'], r['candidate'], f4(r['plain']), f4(r['teacher']),
                          f4(r['model']), f4(r['share'], 3)] for r in rows]))
        return dict(rows=rows)
    section(doc, 'cells', sec)
    return finish(doc), md


def artelingo_held(enc, workers):
    """Held per-seed values of X-c* on ArtELingo: clustering AMI per aspect and seed (`bs_artclu.cstar_fixed` partitions scored
    by `rb_score.ami`, the mean of three states) and retrieval mAP per aspect and seed at R0 and at X-c*'s pick
    (`bs_artret.score_cstar_held`), plus X's frozen image at R0 (`bs_artret.score_frozen_held`). Memoized."""
    key = ('art', enc)
    if key in _CTX:
        return _CTX[key]
    root = Path(BP.OUT) / 'cstar' / enc
    held_dir = root / 'held/k11=40'
    picks_file = Path(BP.OUT) / f'stage1/artret_picks_{enc}.json'
    need(held_dir, picks_file)
    import bs_artclu
    import bs_artret
    import rt_readout
    picks = read_json(picks_file)
    names, sha, labels = bs_artclu.held_setup()
    import rb_score
    clu = {}
    for a in BP.ASPECTS:
        fixed, _ = bs_artclu.cstar_fixed(held_dir, out_dir() / 'cache/artclu' / enc, a)
        y = labels[a]
        m = y >= 0
        clu[a] = [float(np.mean([float(rb_score.ami(y[m], p[m])) for p in fixed[s]['r20']])) for s in bs_artclu.SEEDS]
    r0 = rt_readout.name(rt_readout.R0)
    outs = out_dir() / 'cache/artret' / enc
    pick = bs_artret.score_cstar_held(root, picks['cstar']['pick'], outs / 'pick', workers)
    base = bs_artret.score_cstar_held(root, r0, outs / 'r0', workers)
    fz = bs_artret.score_frozen_held(bs_artret.ENC[enc]['view'], f'frozen_{enc}_diag', r0, outs / 'frozen', workers)
    ret = dict(cstar_pick=dict(readout=picks['cstar']['pick'], **{a: [float(x.mean()) for x in pick[a]['ap']] for a in BP.ASPECTS}),
               cstar_r0={a: [float(x.mean()) for x in base[a]['ap']] for a in BP.ASPECTS},
               frozen_r0={a: float(np.mean(fz[a]['ap'])) for a in BP.ASPECTS})
    _CTX[key] = dict(clu=clu, ret=ret)
    return _CTX[key]


def affection_held(enc, workers):
    """Held (test) per-seed values of an X photo model on Affection: AMI per seed (`ac_cluster.ami` of
    `pf_stage3.candidate_partitions`) and mAP per seed at its stage 1 pick (`pf_stage3.candidate_retrieval`), through
    `bs_affection.PhotoArm`. Memoized."""
    key = ('aff', enc)
    if key in _CTX:
        return _CTX[key]
    import bs_affection
    rec_file = bs_affection.ranges_path(enc, Path(BP.OUT))
    need(rec_file, bs_affection.export_path(f'photo_{enc}', 'test', 0, Path(BP.OUT)))
    import ac_cluster
    import ac_data
    import pf_stage3
    rec = bs_affection.read_record(rec_file, enc)
    pa = bs_affection.PhotoArm(f'photo_{enc}', Path(BP.OUT))
    labels = np.asarray(ac_data.labels('test'))
    cand = pa.candidate(rec['photo_readout']['pick'])
    parts = pf_stage3.candidate_partitions(cand, out_dir() / 'cache/affection_partitions', workers)
    ret = pf_stage3.candidate_retrieval(cand, cand.readout, labels, workers)
    _CTX[key] = dict(ami=[float(ac_cluster.ami(parts[s], labels)) for s in range(5)], map=ret['per_seed'],
                     readout=cand.readout, ranges=rec['ranges'])
    return _CTX[key]


def held_spreads(smoke, workers):
    """Held seed spreads (max - min over seeds 0 to 4) beside the validation ranges. ArtELingo: `artelingo_held` against
    `OUT/stage1/artclu_ranges_<enc>.json` and `artret_ranges_<enc>.json`. Affection: `affection_held` against
    `OUT/stage1/affection_ranges_<enc>.json`. EmoSet: the photo model's test AMI and mAP at R0 per seed from `score_arm`
    against r_E (stage 2 record)."""
    doc = new_doc('held_spreads', 'seed spread on held / test data beside the validation seed range; ratio = held spread / validation range; descriptive')
    md = []
    sp = lambda v: float(max(v) - min(v))

    def art():
        out, lines = {}, ['## ArtELingo held (X-c*)', '']
        for enc in BP.NEW_ENCODERS:
            try:
                h = artelingo_held(enc, workers)
                rc = read_json(Path(BP.OUT) / f'stage1/artclu_ranges_{enc}.json')['ranges']
                rr = read_json(Path(BP.OUT) / f'stage1/artret_ranges_{enc}.json')['ranges']
            except NotRun as e:
                out[enc] = dict(status='not run', reason=str(e), missing=e.missing)
                continue
            out[enc] = dict(status='ok', clustering={a: dict(held_spread=sp(h['clu'][a]), validation_range=rc[a], ratio=sp(h['clu'][a]) / rc[a])
                                                      for a in BP.ASPECTS},
                            retrieval_r0={a: dict(held_spread=sp(h['ret']['cstar_r0'][a]), validation_range=rr[a],
                                                  ratio=sp(h['ret']['cstar_r0'][a]) / rr[a]) for a in BP.ASPECTS})
            for a in BP.ASPECTS:
                lines.append(f'- {enc} {a}: AMI held {out[enc]["clustering"][a]["held_spread"]:.4f} vs validation {rc[a]:.4f}; '
                             f'mAP(R0) held {out[enc]["retrieval_r0"][a]["held_spread"]:.4f} vs validation {rr[a]:.4f}')
        if not any(v['status'] == 'ok' for v in out.values()):
            raise NotRun('no encoder has its ArtELingo inputs', [m for v in out.values() for m in v.get('missing', [])])
        md.extend(lines + [''])
        return dict(encoders=out)

    def aff():
        out, lines = {}, ['## Affection test (X photo models)', '']
        for enc in BP.NEW_ENCODERS:
            try:
                h = affection_held(enc, workers)
            except NotRun as e:
                out[enc] = dict(status='not run', reason=str(e), missing=e.missing)
                continue
            out[enc] = dict(status='ok', ami=dict(held_spread=sp(h['ami']), validation_range=h['ranges']['clu'], ratio=sp(h['ami']) / h['ranges']['clu']),
                            map_at_pick=dict(held_spread=sp(h['map']), validation_range=h['ranges']['map'], ratio=sp(h['map']) / h['ranges']['map']))
            lines.append(f'- photo_{enc}: AMI {sp(h["ami"]):.4f} vs {h["ranges"]["clu"]:.4f}; mAP {sp(h["map"]):.4f} vs {h["ranges"]["map"]:.4f}')
        if not any(v['status'] == 'ok' for v in out.values()):
            raise NotRun('no X photo model has its Affection inputs', [m for v in out.values() for m in v.get('missing', [])])
        md.extend(lines + [''])
        return dict(encoders=out)

    def emo():
        e = emoset_spread(ctx(smoke, workers))
        md.extend(['## EmoSet test (photo_qwen), against r_E', ''] + table(['measure', 'test spread', 'r_E (validation)', 'ratio'],
                  [[k, f4(v['test_spread']), f4(v['r_E']), f4(v['ratio'], 2)] for k, v in e.items()]) + [''])
        return dict(photo_qwen=e)
    for n, fn in (('artelingo', art), ('affection', aff), ('emoset', emo)):
        section(doc, n, fn)
    return finish(doc), md


def emoset_spread(c):
    """The photo model's EmoSet test spread: AMI (20 clusters) per seed by `ac_cluster.ami` of the seed's partitions and mAP at
    R0 per seed from `score_arm`, against r_E,clu and r_E,mAP of the stage 2 record."""
    import ac_cluster
    r = c.arm('photo_qwen')
    ami = [float(ac_cluster.ami(r['parts20'][s], c.labels)) for s in range(5)]
    mp = [float(v) for v in r['r0']['per_seed']]
    out = {}
    for k, v, rng_ in (('AMI (20 clusters)', ami, c.rec['ranges']['clu']['r']), ('mAP at R0', mp, c.rec['ranges']['map']['r'])):
        s = float(max(v) - min(v))
        out[k] = dict(per_seed=v, test_spread=s, r_E=float(rng_), ratio=s / float(rng_))
    return out


def r_e_vs_spread(smoke, workers):
    """r_E (stage 2 record, `bs_emoset_score.seed_ranges`) beside the photo model's spread over its five runs on EmoSet test
    (`emoset_spread`: `ac_cluster.ami` per seed; `score_arm` mAP at R0 per seed)."""
    doc = new_doc('r_e_vs_spread', 'r_E (validation, max - min over the five runs) beside the same spread on test; descriptive')
    md = []

    def sec():
        e = emoset_spread(ctx(smoke, workers))
        md.extend(table(['measure', 'r_E (validation)', 'test spread', 'test / r_E'], [[k, f4(v['r_E']), f4(v['test_spread']), f4(v['ratio'], 2)] for k, v in e.items()]))
        return e
    section(doc, 'emoset', sec)
    return finish(doc), md


def removed_classes(keys):
    """Class name of each removed photo, from the split files (`bs_emoset.labels`, D17: no code reads the class from a key)."""
    import bs_emoset
    return list(bs_emoset.labels(keys))


def emoset_uncut(smoke, workers):
    """SC7 to SC10 on the uncut EmoSet test set: the same calls as `bs_emoset_score.stage3` (`score_arm`, `ac_cluster.paired`,
    `rt_stats.paired`, `make_cell`) with the test keys of `bs_emoset.keys('test', cut=False)`; read-outs, wordings and r_E from the
    stage 2 record, read-out maps fitted on the validation cut set as in the main run. Plus the removed photos per class from
    `RECORDS/nearcopy_emoset.json` (`bs_emoset.real_record`'s file)."""
    doc = new_doc('emoset_uncut', 'SC7 to SC10 recomputed on the uncut test set (no outcome is drawn); removed photos per class; descriptive')
    md = []

    def removed():
        f = BP.RECORDS / 'nearcopy_emoset.json'
        rec = read_json(f)
        out = {}
        for sp_, part in rec['splits'].items():
            cnt = {cl: 0 for cl in BP.EMOSET_CLASSES}
            for cl in removed_classes(part['removed_keys']):
                cnt[cl] += 1
            out[sp_] = dict(n=part['n'], n_removed=part['n_removed'], n_kept=part['n_kept'], removed_per_class=cnt)
        md.extend(['## Removed photos per class (near-copy record)', ''] + table(['split', 'n', 'removed'] + list(BP.EMOSET_CLASSES),
                  [[s, v['n'], v['n_removed']] + [v['removed_per_class'][cl] for cl in BP.EMOSET_CLASSES] for s, v in out.items()]) + [''])
        return dict(source=str(f), splits=out)

    def cells():
        import ac_cluster
        import rt_stats
        c = ctx(smoke, workers)
        S = c.S
        uncut = c.source.keys_uncut('test') if hasattr(c.source, 'keys_uncut') else __import__('bs_emoset').keys('test', cut=False)
        S.check_order(uncut)
        labels = S._check_labels(c.source.labels(uncut), len(uncut))
        w_clu, w_ret = (S.TEACHER_ARM[c.rec['wording'][t]] for t in ('clustering', 'retrieval'))
        arms = list(dict.fromkeys(['photo_qwen', 'plain_qwen', w_clu, w_ret]))
        res = {}
        for a in arms:
            try:
                res[a] = S.score_arm(c.source, a, c.picks[a], c.val_keys, uncut, labels, out_dir() / 'cache/partitions_emoset_uncut', c.workers)
            except FileNotFoundError as e:
                raise NotRun(f'{a}: {e}', [e.filename])
            except AssertionError as e:
                if 'not in the file' in str(e):
                    raise NotRun(f'{a}: the exports hold the cut set only ({e})', [])
                raise
        ph = res['photo_qwen']
        r_clu, r_map = c.rec['ranges']['clu']['r'], c.rec['ranges']['map']['r']
        out = []
        for sc, comp, r in (('SC7', 'plain_qwen', None), ('SC9', w_clu, r_clu)):
            b = ac_cluster.paired(ph['parts20'], res[comp]['parts20'], labels, c.n_clu, c.workers)
            out.append(S.make_cell(sc, 'clu', 'photo_qwen', comp, b['A_cand'], b['A_comp'], b['d'], b['interval'], r,
                                   None if r is None else S.R_SOURCE['clu'], c.n_clu))
        for sc, comp, r in (('SC8', 'plain_qwen', None), ('SC10', w_ret, r_map)):
            x = res[comp]['picked']
            assert np.array_equal(ph['picked']['rows'], x['rows'])
            p = rt_stats.paired(ph['picked']['ap'], x['ap'], c.n_ret)
            out.append(S.make_cell(sc, 'ret', 'photo_qwen', comp, ph['picked']['point'], x['point'], p['d'], p['interval'], r,
                                   None if r is None else S.R_SOURCE['ret'], c.n_ret))
        out.sort(key=lambda z: ('SC7', 'SC8', 'SC9', 'SC10').index(z['sc']))
        # beside the cut-set cells, when ticket 08's file exists
        cut = {}
        f = Path(BP.OUT) / 'stage3' / S.CELLS_NAME
        if f.exists():
            cut = {z['sc']: z for z in json.loads(f.read_text())}
        md.extend([f'## SC7 to SC10 on the uncut test set (n = {len(uncut)})', ''] + table(
            ['SC', 'candidate vs comparator', 'd', '95% interval', 'cut-set d (stage 3)', 'draws'],
            [[z['sc'], f'{z["candidate"]} vs {z["comparator"]}', f4(z['d']), f'[{f4(z["lo"])}, {f4(z["hi"])}]',
              f4(cut[z['sc']]['d']) if z['sc'] in cut else 'n/a', z['n_draws']] for z in out]) + [''])
        return dict(n_test_uncut=len(uncut), cells=out, cut_set_cells_file=str(f) if cut else None)
    section(doc, 'removed_per_class', removed)
    section(doc, 'uncut_cells', cells)
    return finish(doc), md


def node_vs_3090(smoke, workers):
    """The node402 minus 3090 difference of 0(viii): `OUT/stage0/viii.json` (`bs_node.viii`: validation AMI of c* seed 1 per
    head, node against the 3090's stored export, and the export difference). Shown as stored."""
    doc = new_doc('node_vs_3090', 'c* seed 1, node402 against the 3090 (0(viii)); recorded, no gate; descriptive')
    md = []

    def sec():
        v = read_json(Path(BP.OUT) / 'stage0/viii.json')
        if v.get('status') != 'ok':
            raise NotRun(f'viii.json status {v.get("status")}', [Path(BP.OUT) / 'stage0/viii.json'])
        md.extend(table(['head', 'AMI node', 'AMI 3090', 'difference', 'max abs export diff', 'mean abs export diff'],
                        [[a, f4(i['ami_node']), f4(i['ami_3090']), f4(i['ami_diff'], 5), f'{i["export_max_abs_diff"]:.3g}',
                          f'{i["export_mean_abs_diff"]:.3g}'] for a, i in v['items'].items()]) + ['', f'host {v.get("host")}, device {v.get("device")}, {v.get("seconds")} s'])
        return dict(items=v['items'], host=v.get('host'), device=v.get('device'), seconds=v.get('seconds'))
    section(doc, 'viii', sec)
    return finish(doc), md


def knob_png(curves, path):
    """The figure: one panel per knob, x = the knob's values (in value order; the dashed line marks c*'s own value), y = the
    effect in seed ranges (`d_r`) of the mean over aspects, AMI (M, with its interval) and mAP (M). Not-run points are left out."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    knobs = list(curves['curves'])
    ncol = 4
    nrow = int(np.ceil(len(knobs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, knob in zip(axes.ravel(), knobs):
        ax.set_visible(True)
        cv = curves['curves'][knob]
        pts = cv['points']
        done = [(i, p) for i, p in enumerate(pts) if p['status'] == 'done']
        ax.axhline(0, color='grey', lw=0.8)
        if done:
            xs = [i for i, _ in done]
            m = [p['M_ami'] for _, p in done]
            ax.errorbar(xs, m, yerr=[[p['M_ami'] - p['M_ami_lo'] for _, p in done], [p['M_ami_hi'] - p['M_ami'] for _, p in done]],
                        marker='o', color='tab:blue', capsize=3, label='AMI (M, interval)')
            ax.plot(xs, [p['M_map'] for _, p in done], marker='s', color='tab:orange', label='mAP (M)')
        labs = [str(p['value']) for p in pts]
        ax.set_xticks(range(len(pts)))
        ax.set_xticklabels(labs, fontsize=7, rotation=30)
        for i, p in enumerate(pts):
            if str(p['value']) == str(cv['star']):
                ax.axvline(i, color='k', ls='--', lw=0.8)
            if p['status'] != 'done':
                ax.annotate('not run', (i, 0), fontsize=6, color='red', ha='center', va='bottom')
        ax.set_title(knob, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle('Knob sweep: row minus reference, in seed ranges (mean over aspects). Analysis only: no point is a better setting.',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def knob_curves(smoke, workers):
    """The knob sweep's curves per knob: `OUT/stage4/partb_knob_curves.json` (`bs_partb.knob_curves`, ticket 11), drawn to
    `OUT/diag/knob_curves.png` and tabulated. Nothing is recomputed; no point is called better."""
    doc = new_doc('knob_curves', 'effect of each knob value against the reference, in seed ranges; analysis only (D6); descriptive')
    md = []

    def sec():
        f = Path(BP.OUT) / 'stage4/partb_knob_curves.json'
        cur = read_json(f)
        png = out_dir() / 'knob_curves.png'
        knob_png(cur, png)
        for knob, cv in cur['curves'].items():
            md.extend([f'## {knob} (c* value: {cv["star"]})', ''] + table(['value', 'status', 'M AMI [lo, hi]', 'M mAP'],
                      [[p['value'], p['status'], f'{f4(p["M_ami"], 2)} [{f4(p["M_ami_lo"], 2)}, {f4(p["M_ami_hi"], 2)}]' if p['status'] == 'done' else p.get('reason', ''),
                        f4(p['M_map'], 2) if p['status'] == 'done' else ''] for p in cv['points']]) + [''])
        md.insert(0, '![knob curves](knob_curves.png)\n')
        return dict(source=str(f), figure=str(png), knobs=len(cur['curves']),
                    points_done=sum(p['status'] == 'done' for cv in cur['curves'].values() for p in cv['points']),
                    points_total=sum(len(cv['points']) for cv in cur['curves'].values()))
    section(doc, 'curves', sec)
    return finish(doc), md


def photo_length(smoke, workers):
    """The photo sweep's training-length rows (`epochs=2`, `epochs=5`, `epochs=20`; `bs_photo.ROWS`) against the reference
    (the photo model's settings rerun on node402): `OUT/stage4/partb_photo.json` (`bs_partb.photo_sheet`), shown as stored."""
    doc = new_doc('photo_length', 'fixed training of 2, 5 and 20 epochs against the reference, Affection validation, seed ranges of r13; descriptive')
    md = []

    def sec():
        f = Path(BP.OUT) / 'stage4/partb_photo.json'
        s = read_json(f)
        rows = {k: v for k, v in s['rows'].items() if k.startswith('epochs=')}
        if not rows:
            raise NotRun('no epochs rows in the sheet', [f])
        lines = []
        for k, v in sorted(rows.items(), key=lambda kv: float(kv[0].split('=')[1])):
            if v['status'] == 'done':
                lines.append([k, 'done', f4(v['ami']['d']), f'[{f4(v["ami"]["lo"])}, {f4(v["ami"]["hi"])}]', f4(v['ami']['d_r'], 2),
                              f4(v['map']['d']), f'[{f4(v["map"]["lo"])}, {f4(v["map"]["hi"])}]', f4(v['map']['d_r'], 2)])
            else:
                lines.append([k, f'not run ({v.get("reason")})', '', '', '', '', '', ''])
        md.extend([f'Reference state: {s.get("reference_state")}; reference values {s.get("reference_values")}', ''] +
                  table(['row', 'status', 'AMI d', 'AMI interval', 'AMI d / r', 'mAP d', 'mAP interval', 'mAP d / r'], lines))
        return dict(source=str(f), reference_state=s.get('reference_state'), reference_values=s.get('reference_values'),
                    units=s.get('units'), rows=rows)
    section(doc, 'photo_sweep', sec)
    return finish(doc), md


FUNCS = dict(r0_rows=r0_rows, emoset_k8=emoset_k8, emoset_per_class_ap=emoset_per_class_ap, photo_minus_cstar=photo_minus_cstar,
             teacher_gain_share=teacher_gain_share, held_spreads=held_spreads, r_e_vs_spread=r_e_vs_spread,
             emoset_uncut=emoset_uncut, node_vs_3090=node_vs_3090, knob_curves=knob_curves, photo_length=photo_length)
assert tuple(FUNCS) == ITEMS


def run_item(item, smoke=False, workers=8):
    """Run one item and write `OUT/diag/<item>.{json,md}`. Returns its JSON document."""
    assert item in FUNCS, item
    _CTX['smoke'] = bool(smoke)
    t0 = time.monotonic()
    try:
        doc, md = FUNCS[item](smoke, workers)
    except NotRun as e:                     # a context (record, folder) missing for the whole item
        doc = finish(dict(item=item, note=FUNCS[item].__doc__.split('\n')[0],
                          sections={'item': dict(status='not run', reason=str(e), missing=e.missing)}))
        md = []
    doc['seconds'] = round(time.monotonic() - t0, 1)
    write_item(item, doc, md)
    return doc


def run_all(smoke=False, workers=8):
    """All items in order; each writes its files; one item's missing input never stops the others. Returns
    {item: {'status', 'missing', 'seconds'}}. Ticket 12a's `bs_run.py diag` calls this."""
    _CTX.clear()
    out = {}
    for item in ITEMS:
        d = run_item(item, smoke, workers)
        out[item] = dict(status=d['status'], missing=d['missing'], seconds=d['seconds'])
        print(f'diag {item}: {d["status"]} ({d["seconds"]} s)' + (f', missing {d["missing"][:2]}' if d['missing'] else ''), flush=True)
    out_f = out_dir() / 'summary.json'
    out_f.write_text(json.dumps(out, indent=1) + '\n')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('item', choices=('all',) + ITEMS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    a = p.parse_args(argv)
    if a.item == 'all':
        run_all(a.smoke, a.workers)
    else:
        _CTX.clear()
        d = run_item(a.item, a.smoke, a.workers)
        print(f'diag {a.item}: {d["status"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
