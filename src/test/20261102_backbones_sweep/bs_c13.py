"""Independent r14 C13, from §3.3–3.5 and §6 and frozen r10–r13 functions.

Only compare() opens r14 result records. No r14 scorer is imported. The module
docstrings of bs_artclu, bs_artret, bs_affection, bs_emoset, bs_emoset_score,
bs_outcome, bs_photo and bs_cstar were consulted for file interfaces only;
bs_paths and bs_stores are the two permitted r14 imports.

--sections artclu:siglip2 is a full-data subset smoke (nine clustering cells).
Subset compares report their scope explicitly; they do not certify full C13.
Intermediate numerical arrays are cached under --out, bound to their input
bytes, keys, labels, and the frozen Python sources. Never reuse r14 partitions.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

import bs_paths as P
import bs_stores

for _folder in (P.R10_ROUND, P.R11_ROUND, P.R12_ROUND, P.R13_ROUND,
                P.RB_ROUND, P.EG_ROUND):
    sys.path.insert(0, str(_folder))
import pf_paths  # installs the frozen rounds' own transitive import paths
import ac_cluster
import ac_crl
import ac_data
import ac_select
import ac_stats
import eg_paths
import pf_nearcopy
import rb_paths
import rb_score
import rs_vh
import rt_data
import rt_metrics
import rt_readout
import rt_stats

IDENTITY = ('sc', 'encoder', 'aspect', 'task', 'data', 'candidate', 'comparator')
SEEDS = tuple(P.SEEDS)
TOL = 1e-6
SCRATCH = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/c13_smoke')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')


def join_rows(stored, requested):
    """C6: require unique identities on both sides, including subset joins."""
    stored, requested = list(stored), list(requested)
    assert all(isinstance(k, str) and k for k in stored + requested)
    assert len(set(stored)) == len(stored), 'duplicate stored keys'
    assert len(set(requested)) == len(requested), 'duplicate requested keys'
    lookup = {k: i for i, k in enumerate(stored)}
    missing = [k for k in requested if k not in lookup]
    assert not missing, f'missing keys: {missing[:3]}'
    rows = np.asarray([lookup[k] for k in requested], np.int64)
    assert [stored[i] for i in rows] == requested
    return rows


def hashed_order(keys):
    keys = list(keys)
    join_rows(keys, keys)
    return sorted(keys, key=lambda k: hashlib.sha256(k.encode('utf-8')).hexdigest())


def cell_id(cell):
    return tuple(cell[k] for k in IDENTITY)


def index_cells(cells):
    result = {}
    for cell in cells:
        key = cell_id(cell)
        assert key not in result, f'duplicate cell: {key}'
        result[key] = cell
    return result


def cell_outcome(lo, r):
    assert np.isfinite(lo)
    return rt_stats.outcome(lo, r)


def verdicts(cells, status=None, partial=False):
    """§6, D18. Status is external evidence, never a numerical input.

    Missing live-path cells are errors. In subset smoke mode they remain pending.
    A failed paintings path removes all that encoder's SC6 claims; a failed
    photos path removes only its photo claims (the shared status interface).
    """
    index_cells(cells)
    if status is None:
        status = {'encoders': {e: {'cstar_runs': 'ok', 'photo_runs': 'ok',
                                  'stage0': {'paintings': 'ok', 'photos': 'ok'}}
                               for e in P.NEW_ENCODERS},
                  'part_c': {'stage0': 'ok', 'stage2': 'ok'}}
    assert set(status['encoders']) == set(P.NEW_ENCODERS)
    out, claims = {}, []

    def require(rows, count):
        if len(rows) != count:
            assert partial, f'need {count} cells, got {len(rows)}'
            return False
        return True

    for enc in P.NEW_ENCODERS:
        st = status['encoders'][enc]
        assert {st['cstar_runs'], st['photo_runs'], *st['stage0'].values()} <= {'ok', 'failed'}
        assert not (st['cstar_runs'] == 'failed' and st['photo_runs'] == 'ok')
        bad_art = st['cstar_runs'] == 'failed' or st['stage0']['paintings'] == 'failed'
        bad_photo = (st['cstar_runs'] == 'failed' or st['photo_runs'] == 'failed'
                     or st['stage0']['photos'] == 'failed')
        art = [c for c in cells if c['encoder'] == enc and c['sc'] in ('SC1', 'SC2', 'SC3')]
        photo = [c for c in cells if c['encoder'] == enc and c['sc'] in ('SC4', 'SC5')]
        painting_verdict, photo_verdict = 'inconclusive', 'inconclusive'
        if not bad_art:
            if require(art, 9):
                assert {(c['sc'], c['aspect']) for c in art} == {
                    (sc, a) for sc in ('SC1', 'SC2', 'SC3') for a in P.ASPECTS}
                base = all(c['outcome'] == 'WIN' for c in art if c['sc'] != 'SC3')
                painting_verdict = ('holds' if all(c['outcome'] == 'WIN' for c in art)
                                    else 'qualified' if base else 'fails')
            else:
                painting_verdict = 'pending'
        if not bad_photo:
            if require(photo, 2):
                assert {c['sc'] for c in photo} == {'SC4', 'SC5'}
                photo_verdict = 'holds' if all(c['outcome'] == 'WIN' for c in photo) else 'fails'
            else:
                photo_verdict = 'pending'
        out[enc] = {'paintings': painting_verdict, 'photos': photo_verdict}
        sc6 = [c for c in cells if c['encoder'] == enc and c['sc'] == 'SC6']
        art6 = [c for c in sc6 if c['data'] == 'artelingo_held']
        photo6 = [c for c in sc6 if c['data'] == 'affection_test']
        if not bad_art:
            require(art6, 6)
        if not bad_photo:
            require(photo6, 2)
        for c in sc6:
            bad = bad_art or (c['data'] == 'affection_test' and bad_photo)
            claims.append({**{k: c[k] for k in IDENTITY},
                           'claim': 'inconclusive' if bad else c['outcome'] in ('WIN', 'LEVEL')})
    st = status['part_c']
    assert set(st) >= {'stage0', 'stage2'} and {st['stage0'], st['stage2']} <= {'ok', 'failed'}
    emo = [c for c in cells if c['sc'] in ('SC7', 'SC8', 'SC9', 'SC10')]
    if 'failed' in (st['stage0'], st['stage2']):
        part_c = 'inconclusive'
    elif not require(emo, 4):
        part_c = 'pending'
    else:
        assert {c['sc'] for c in emo} == {'SC7', 'SC8', 'SC9', 'SC10'}
        v = {c['sc']: c['outcome'] for c in emo}
        base = v['SC7'] == v['SC8'] == 'WIN'
        teacher = [v['SC9'], v['SC10']]
        part_c = ('holds' if base and teacher == ['WIN', 'WIN'] else
                  'qualified' if base and sorted(teacher) == ['LEVEL', 'WIN'] else 'fails')
    return {'encoders': out, 'part_c': part_c, 'claims': claims}


@lru_cache(maxsize=1)
def frozen_identity():
    """Bind local caches to the actual frozen code and this re-derivation."""
    files = {Path(__file__)}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and str(path).endswith('.py'):
            p = Path(path)
            if p.is_relative_to(P.TEST) and p.parent != P.ROUND:
                files.add(p)
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(str(p).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


class Computation:
    def __init__(self, out, n_clu, n_ret, aspects=None, validation_only=False):
        self.out, self.n_clu, self.n_ret = Path(out), n_clu, n_ret
        self.aspects = tuple(P.ASPECTS if aspects is None else aspects)
        self.validation_only = validation_only
        self.cells, self.stage1 = [], {}
        self.nearcopy, self.stage2 = None, None

    def array(self, name, inputs, calculate):
        """Own cache only; all statistical calculations below call frozen code."""
        h = hashlib.sha256(frozen_identity().encode())
        for x in inputs:
            if isinstance(x, np.ndarray):
                h.update(str((x.shape, str(x.dtype))).encode())
                h.update(np.ascontiguousarray(x).tobytes())
            else:
                h.update(json.dumps(x, sort_keys=True).encode())
        path = self.out / 'cache' / f'{name}__{h.hexdigest()}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                return z['value']
        start = time.monotonic()
        value = np.asarray(calculate())
        assert np.isfinite(value).all(), name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, value=value)
        print(f'{name}: {time.monotonic() - start:.1f}s', flush=True)
        return value

    def parts(self, x):
        return self.array('partitions', [x], lambda: ac_cluster.partitions(x))

    def raw_ap(self, x, labels):
        def calc():
            result = rt_metrics.retrieval(x, labels, qe_k=0, workers=8)
            expected = np.flatnonzero(labels >= 0)
            assert np.array_equal(result['rows'], expected), 'dropped or reordered queries'
            return result['ap']
        return self.array('r0_ap', [x, labels], calc)

    def ranges(self, getter, labels):
        clu, mp = [], []
        for seed in SEEDS:
            x = getter(seed)
            clu.append(ac_cluster.ami(self.parts(x), labels))
            mp.append(float(self.raw_ap(ac_data._unit(x), labels).mean()))
        return {'clu': {'per_seed': clu, 'r': rt_stats.seed_range(clu)},
                'map': {'per_seed': mp, 'r': rt_stats.seed_range(mp)}}

    def menu(self, getter, labels, ranges, seeded, art=False):
        """Frozen cross-fit per seed, then the frozen rule on seed means.

        None is r11's permitted single-matrix form. Rebinding a single training
        seed as None leaves every fit, row, half and score unchanged. Splitting
        calls this way bounds cache granularity and preserves the arithmetic.
        """
        scores = []
        aspects = tuple(labels)
        for seed in SEEDS if seeded else (None,):
            per_aspect = []
            for a in aspects:
                x = getter(a, seed)
                ros = rt_readout.menu(x.shape[1])
                def calc():
                    rows = rt_readout.crossfit_all({(a, None): x}, {a: labels[a]}, ros, workers=8)
                    return np.asarray([rows[ro][a] for ro in ros])
                per_aspect.append(self.array('crossfit', [x, labels[a]], calc))
            scores.append(np.stack(per_aspect, axis=1))
        means = np.mean(scores, axis=0)
        rows = {ro: {a: float(means[i, j]) for j, a in enumerate(aspects)}
                for i, ro in enumerate(ros)}
        selector = rt_readout if art else ac_select
        pick = selector.select(rows, ranges) if art else selector.pick_readout(rows, ranges)
        sc = rt_readout.score(rows, ranges) if art else ac_select.readout_scores(rows, ranges)
        keep = rt_readout.guard(rows, ranges) if art else ac_select.readout_guard(rows, ranges)
        return {'pick': rt_readout.name(pick), 'scores': {
            rt_readout.name(ro): {'map': rows[ro], 'score': sc[ro], 'kept': keep[ro]} for ro in ros}}

    def retrieval(self, getter, readout, labels, seeded):
        aps = []
        for seed in SEEDS if seeded else (None,):
            val, test = getter('val', seed), getter('test', seed)
            def calc():
                res = ac_stats.arm_retrieval(lambda _: val, lambda _: test,
                                             readout, labels, workers=8)
                assert np.array_equal(res['rows'], np.flatnonzero(labels >= 0))
                return res['ap']
            aps.append(self.array('picked_ap', [val, test, readout, labels], calc))
        return np.stack(aps) if seeded else aps[0]

    def cell(self, sc, enc, aspect, task, data, candidate, comparator, stat, r, source=None):
        if task == 'clu':
            cv, tv, d, interval = stat
        else:
            cand, comp = stat
            result = rt_stats.paired(cand, comp, draws=self.n_ret)
            cv, tv = float(cand.mean()), float(comp.mean())
            d, interval = result['d'], result['interval']
        lo, hi = map(float, interval)
        # The spec supplies a candidate range even for WIN-only criteria. The
        # shared serialization leaves their `r` null; retain the actual unit
        # separately so the requested three-way outcome remains available.
        serialized_r = r if sc in ('SC6', 'SC9', 'SC10') else None
        self.cells.append(dict(sc=sc, encoder=enc, aspect=aspect, task=task, data=data,
                               candidate=candidate, comparator=comparator,
                               candidate_value=cv, comparator_value=tv, d=d, lo=lo, hi=hi,
                               r=serialized_r, level_unit=r, r_source=source, outcome=cell_outcome(lo, r),
                               n_draws=self.n_clu if task == 'clu' else self.n_ret, seeds=list(SEEDS)))

    def derive_nearcopy(self):
        import ijson
        # Image membership only; no Affection training annotations or text rows.
        with (P.AFF / P.ENCODERS['qwen']['aff'][0] / 'index.json').open('rb') as f:
            images = ijson.items(f, 'images.item')
            train = sorted(f"{r['source']}/{r['image_name']}" for r in images
                           if set(r['splits']) - {'ignored'} == {'train'})
        assert len(train) == len(set(train)) == 72255
        reference = ac_data._unit(bs_stores.load('affection', 'qwen_default', train))
        splits = {}
        for split in ('val', 'test'):
            keys = emo_keys(split)
            x = ac_data._unit(bs_stores.load('emoset', 'qwen_default', keys))
            best = self.array('nearcopy', [train, reference, keys, x],
                              lambda: pf_nearcopy._nearest(x, reference, pf_nearcopy.BLOCK)[0])
            removed = best.astype(np.float64) >= P.NEAR_COPY_CUT
            gone = sorted(k for k, flag in zip(keys, removed) if flag)
            kept = hashed_order(k for k, flag in zip(keys, removed) if not flag)
            hist, _ = np.histogram(np.minimum(best.astype(np.float64), 1.), bins=pf_nearcopy.HIST_EDGES)
            splits[split] = dict(n_total=len(keys), n_removed=len(gone), n_kept=len(kept),
                                 removed_keys=gone, keys=kept, order_sha256=rs_vh.names_sha(kept),
                                 hist={'edges': pf_nearcopy.HIST_EDGES, 'counts': hist.tolist()})
        self.nearcopy = {'cut': P.NEAR_COPY_CUT, 'n_reference': len(train), 'splits': splits}
        return self.nearcopy

    def artclu(self, enc):
        val_labels, held_labels = rt_data.labels('val'), rt_data.labels('held')
        root = rb_paths.OUTPUTS if enc == 'siglip2' else eg_paths.OUTPUTS
        old = root / 'part2/partitions/full'
        frozen = load_old_parts(old / P.FROZEN_ARM[enc] / 'frozen__s0__r20.npz')
        matched = {s: {r: load_old_parts(old / P.MATCHED[enc] / f'percept__s{s}__{r}.npz')
                       for r in ('r20', 'own')} for s in SEEDS}
        doc = self.stage1.setdefault(enc, {})
        doc['artclu'] = {}
        for aspect in self.aspects:
            values = [ac_cluster.ami(self.parts(cstar(enc, 'val', aspect, s)), val_labels[aspect])
                      for s in SEEDS]
            r = rt_stats.seed_range(values)
            doc['artclu'][aspect] = {'per_seed': values, 'r': r}
            if self.validation_only:
                continue
            cand = np.stack([self.parts(cstar(enc, 'held', aspect, s)) for s in SEEDS])
            y = held_labels[aspect]
            k, m, f = f'cstar_{enc}', P.MATCHED[enc], P.FROZEN_ARM[enc]
            cf = {s: {'r20': cand[s]} for s in SEEDS}
            for sc, arm, other, seedless in (
                    ('SC1', f, {0: {'r20': frozen}}, (f,)),
                    ('SC3', m, matched, ())):
                eligible = {k: list(SEEDS), **({arm: list(SEEDS)} if not seedless else {})}
                result = rb_score.paired_bootstrap({k: cf, arm: other}, y, eligible, (k, arm),
                                                   self.n_clu, workers=8, seedless=seedless)
                self.cell(sc, enc, aspect, 'clu', 'artelingo_held', k,
                          f'frozen_{enc}' if sc == 'SC1' else m,
                          (result['A_cand'], max(result['A_e'].values()), result['d'], result['interval']), r)
            teacher = load_old_parts(old / 'T' / f'{aspect}__s0__r20.npz')
            res = ac_cluster.paired(cand, teacher, y, draws=self.n_clu, workers=8)
            self.cell('SC6', enc, aspect, 'clu', 'artelingo_held', k, f'teacher_{aspect}',
                      (res['A_cand'], res['A_comp'], res['d'], res['interval']), r)

    def artret(self, enc):
        labels = rt_data.labels('val')
        ranges = {}
        for a in P.ASPECTS:
            values = [float(self.raw_ap(ac_data._unit(cstar(enc, 'val', a, s)), labels[a]).mean())
                      for s in SEEDS]
            ranges[a] = {'per_seed': values, 'r': rt_stats.seed_range(values)}
        pick = self.menu(lambda a, s: ac_data._unit(cstar(enc, 'val', a, s)), labels,
                         {a: ranges[a]['r'] for a in P.ASPECTS}, True, art=True)
        old_picks = read_json(P.R11_OUT / 'stage4/picks.json')
        if enc == 'siglip1':
            r11 = read_json(P.R11_ROUND / 'stage0a_ranges.json')['map']
            frozen_pick = self.menu(lambda a, s: art_image(enc, 'val'), labels,
                                    {a: r11[a]['r'] for a in P.ASPECTS}, False, art=True)
        else:
            frozen_pick = {'pick': old_picks['clip' if enc == 'clip_l14' else enc]}
        self.stage1.setdefault(enc, {})['artret'] = {
            'ranges': ranges, 'cstar': pick, 'frozen': frozen_pick,
            'teacher': {'pick': old_picks['instructed']}}
        if self.validation_only:
            return
        for a in self.aspects:
            y = rt_data.labels('held')[a]
            cand = self.retrieval(lambda sp, s: ac_data._unit(cstar(enc, 'val' if sp == 'val' else 'held', a, s)),
                                  pick['pick'], y, True)
            if enc == 'siglip1':
                frozen = self.retrieval(lambda sp, s: art_image(enc, 'val' if sp == 'val' else 'held'),
                                        frozen_pick['pick'], y, False)
            else:
                frozen = r11_ap('clip' if enc == 'clip_l14' else enc, a, y)
            teacher = r11_ap('instructed', a, y)
            for sc, other, name in (('SC2', frozen, f'frozen_{enc}'), ('SC6', teacher, f'teacher_{a}')):
                self.cell(sc, enc, a, 'ret', 'artelingo_held', f'cstar_{enc}', name,
                          (cand, other), ranges[a]['r'])

    def affection(self, enc):
        valkeys, testkeys = ac_data.keys('val'), ac_data.keys('test')
        getter = lambda s, sp: exported('affection', f'photo_{enc}', sp, s,
                                        valkeys if sp == 'val' else testkeys)
        # r13 also records nDCG; reproduce it from the frozen measurement.
        ranges = ac_stats.measure_seed_ranges(lambda s: ac_data._unit(getter(s, 'val')),
                                              lambda s: getter(s, 'val'), ac_data.labels('val'),
                                              ac_data.emotion_dist('val'), workers=8)
        pick = self.menu(lambda a, s: ac_data._unit(getter(s, 'val')),
                         {'emotion': ac_data.labels('val')}, {'emotion': ranges['map']['r']}, True)
        p12 = read_json(P.R12_OUT / 'stage1/picks.json')['affection']
        if enc == 'clip_l14':
            frozen_pick = {'pick': p12['readout']['clip']}
        else:
            r12 = read_json(P.R12_OUT / 'stage0b/ranges.json')['map']['r']
            frozen_pick = self.menu(lambda a, s: ac_data._unit(bs_stores.load('affection', enc, valkeys)),
                                    {'emotion': ac_data.labels('val')}, {'emotion': r12}, False)
        self.stage1.setdefault(enc, {})['affection'] = {
            'ranges': ranges, 'photo': pick, 'frozen': frozen_pick}
        if self.validation_only:
            return
        y = ac_data.labels('test')
        cand = np.stack([self.parts(getter(s, 'test')) for s in SEEDS])
        frozen = self.parts(ac_data._unit(bs_stores.load('affection', enc, testkeys)))
        teacher_arm = 'teacher_' + p12['teacher_wording']['clustering']
        teacher = load_old_parts(P.R12_OUT / 'partitions' / f'affection__test__{teacher_arm}__s-.npz')
        for sc, other, name in (('SC4', frozen, f'frozen_{enc}'), ('SC6', teacher, 'teacher_emotion_focus')):
            result = ac_cluster.paired(cand, other, y, draws=self.n_clu, workers=8)
            self.cell(sc, enc, 'emotion', 'clu', 'affection_test', f'photo_{enc}', name,
                      (result['A_cand'], result['A_comp'], result['d'], result['interval']), ranges['clu']['r'])
        cand_ap = self.retrieval(lambda sp, s: ac_data._unit(getter(s, sp)), pick['pick'], y, True)
        if enc == 'clip_l14':
            frozen_ap = saved_ap(P.R12_OUT / 'stage2/perquery/affection__clip__emotion__s-.npz', y)
        else:
            frozen_ap = self.retrieval(lambda sp, s: ac_data._unit(bs_stores.load(
                'affection', enc, valkeys if sp == 'val' else testkeys)), frozen_pick['pick'], y, False)
        teacher_arm = 'teacher_' + p12['teacher_wording']['retrieval']
        teacher_ap = saved_ap(P.R12_OUT / f'stage2/perquery/affection__{teacher_arm}__emotion__s-.npz', y)
        for sc, other, name in (('SC5', frozen_ap, f'frozen_{enc}'),
                                 ('SC6', teacher_ap, 'teacher_emotion_focus')):
            self.cell(sc, enc, 'emotion', 'ret', 'affection_test', f'photo_{enc}', name,
                      (cand_ap, other), ranges['map']['r'])

    def emoset(self):
        if self.nearcopy is None:
            self.derive_nearcopy()
        keys = {sp: self.nearcopy['splits'][sp]['keys'] for sp in ('val', 'test')}
        getter = lambda arm, sp, s: emo_vector(arm, sp, s, keys[sp])
        yval = emo_labels('val', keys['val'])
        ranges = self.ranges(lambda s: getter('photo_qwen', 'val', s), yval)
        bar_arms = ('photo_qwen', 'plain_qwen', 'teacher_emotion_focus', 'teacher_emotion_focus_photo')
        descriptive = ('cstar_qwen', 'crl_clip_l14', *(f'photo_{e}' for e in P.NEW_ENCODERS),
                       *(f'frozen_{e}' for e in P.NEW_ENCODERS))
        menus = {}
        for arm in (*bar_arms, *descriptive):
            menus[arm] = self.menu(lambda a, s: retrieval_vector(arm, getter(arm, 'val', s)),
                                    {'emotion': yval}, {'emotion': ranges['map']['r']}, emo_seeded(arm))
        ami = {w: ac_cluster.ami(self.parts(getter('teacher_' + w, 'val', None)), yval)
               for w in ('emotion_focus', 'emotion_focus_photo')}
        mp = {w: menus['teacher_' + w]['scores'][menus['teacher_' + w]['pick']]['map']['emotion'] for w in ami}
        wording = {task: ac_select.pick_wording(vals['emotion_focus'], vals['emotion_focus_photo'], ranges[k]['r'])
                   for task, vals, k in (('clustering', ami, 'clu'), ('retrieval', mp, 'map'))}
        self.stage2 = {'ranges': ranges, 'readouts': menus, 'teacher_wording': wording,
                       'wording_scores': {'clustering': ami, 'retrieval': mp}}
        if self.validation_only:
            return
        y = emo_labels('test', keys['test'])
        cand = np.stack([self.parts(getter('photo_qwen', 'test', s)) for s in SEEDS])
        teacher_clu = 'teacher_emotion_focus' + ('_photo' if wording['clustering'] == 'photo' else '')
        for sc, arm in (('SC7', 'plain_qwen'), ('SC9', teacher_clu)):
            other = self.parts(getter(arm, 'test', None))
            result = ac_cluster.paired(cand, other, y, draws=self.n_clu, workers=8)
            self.cell(sc, 'qwen', 'emotion', 'clu', 'emoset_test', 'photo_qwen', arm,
                      (result['A_cand'], result['A_comp'], result['d'], result['interval']), ranges['clu']['r'])
        cand_ap = self.retrieval(lambda sp, s: ac_data._unit(getter('photo_qwen', sp, s)),
                                 menus['photo_qwen']['pick'], y, True)
        teacher_ret = 'teacher_emotion_focus' + ('_photo' if wording['retrieval'] == 'photo' else '')
        for sc, arm in (('SC8', 'plain_qwen'), ('SC10', teacher_ret)):
            other = self.retrieval(lambda sp, s: retrieval_vector(arm, getter(arm, sp, s)),
                                   menus[arm]['pick'], y, False)
            self.cell(sc, 'qwen', 'emotion', 'ret', 'emoset_test', 'photo_qwen', arm,
                      (cand_ap, other), ranges['map']['r'])


def load_old_parts(path):
    """Read-only frozen partition reader; never call a cache writer in an old round."""
    path = Path(path)
    side = read_json(path.with_suffix('.json'))
    assert side['partitions_sha256'] == rs_vh.file_sha(path), path
    identity = side.get('identity', {})
    if 'held_sha256' in identity:
        assert identity['held_sha256'] == rs_vh.names_sha(rt_data.names('held'))
    with np.load(path, allow_pickle=False) as z:
        x = z['labels']
    assert x.ndim == 2 and np.issubdtype(x.dtype, np.integer)
    return x


def cstar(enc, split, aspect, seed):
    """r11's checked reader, changing only its input root as the interface specifies."""
    old = rt_data.CSTAR_ROOT
    try:
        rt_data.CSTAR_ROOT = P.OUT / 'cstar' / enc
        return rt_data.raw('cstar', split, aspect, seed)
    finally:
        rt_data.CSTAR_ROOT = old


def exported(dataset, arm, split, seed, keys):
    path = P.OUT / 'exports' / dataset / arm / f'{split}__s{seed}.npz'
    side = read_json(path.with_suffix('.json'))
    assert side['sha256'] == rs_vh.file_sha(path), f'{path}: file sha256'
    assert (side['dataset'], side['arm'], side['split'], side['seed']) == (dataset, arm, split, seed)
    with np.load(path, allow_pickle=False) as z:
        stored, x = z['keys'].tolist(), z['emb']
    assert rs_vh.names_sha(stored) == side['keys_sha256']
    assert rs_vh.array_sha(x) == side['emb_sha256']
    ac_data._array(x, len(stored), 2048, np.float16, unit=True)
    return x[join_rows(stored, keys)]


@lru_cache(maxsize=3)
def art_image_keys(enc):
    import ijson
    with (P.ART / P.ENCODERS[enc]['art'] / 'index.json').open('rb') as f:
        rows = list(ijson.items(f, 'images.item'))
    names = [r['painting'] for r in rows]
    join_rows(names, names)
    return dict(zip(names, (r['image'] for r in rows)))


def art_image(enc, split):
    mapping = art_image_keys(enc)
    keys = [mapping[n] for n in rt_data.names(split)]
    return ac_data._unit(bs_stores.load('artelingo', enc, keys))


def saved_ap(path, labels):
    with np.load(path, allow_pickle=False) as z:
        ap, rows = z['ap'], z['rows']
    assert ap.dtype == np.float64 and ap.ndim == 1 and np.isfinite(ap).all()
    assert np.array_equal(rows, np.flatnonzero(labels >= 0)), f'{path}: query identity'
    return ap


def r11_ap(arm, aspect, labels):
    head = f'__h{aspect}' if arm == 'instructed' else ''
    return saved_ap(P.R11_OUT / f'stage5/perquery/{arm}__{aspect}{head}__s-.npz', labels)


def emo_keys(split):
    # Project only the image path. The first field is read solely by emo_labels.
    import ijson
    with (P.EMOSET_ROOT / f'{split}.json').open('rb') as f:
        keys = [row[1] for row in ijson.items(f, 'item')]
    join_rows(keys, keys)
    assert len(keys) == P.EMOSET_SPLITS[split]
    return keys


def emo_labels(split, keys):
    rows = read_json(P.EMOSET_ROOT / f'{split}.json')
    positions = join_rows([r[1] for r in rows], keys)
    codes = {name: i for i, name in enumerate(P.EMOSET_CLASSES)}
    return np.asarray([codes[rows[i][0]] for i in positions], np.int64)


def emo_seeded(arm):
    return arm.startswith(('photo_', 'cstar_'))


def emo_vector(arm, split, seed, keys):
    if emo_seeded(arm):
        return exported('emoset', arm, split, seed, keys)
    view = {'plain_qwen': 'qwen_default', 'teacher_emotion_focus': 'emotion_focus',
            'teacher_emotion_focus_photo': 'emotion_focus_photo'}.get(arm, arm.removeprefix('frozen_'))
    if arm == 'crl_clip_l14':
        pick = read_json(P.R12_OUT / 'stage1/picks.json')['affection']['crl_version']
        encoder, wording = pick.split('_')
        assert encoder == 'clip', 'D10 freezes r12 CLIP CRL'
        img = ac_data._unit(bs_stores.load('emoset', 'clip_l14', keys))
        return ac_crl.vectors(img, ac_data._crl_text(encoder, wording, 'emotion'))
    return ac_data._unit(bs_stores.load('emoset', view, keys))


def retrieval_vector(arm, x):
    # emo_vector already normalizes every frozen view exactly once (CRL in
    # float64). Only the stored float16 exports still need normalization.
    return ac_data._unit(x) if emo_seeded(arm) else x


def derive(n_clu=2000, n_ret=10000, out=None, sections=None, aspects=None, validation_only=False):
    assert 1 <= n_clu <= 2000 and 1 <= n_ret <= 10000
    out = P.OUT / 'c13' if out is None else Path(out)
    all_sections = [f'{part}:{enc}' for enc in P.NEW_ENCODERS
                    for part in ('artclu', 'artret', 'affection')] + ['stage2']
    sections = all_sections if sections is None else list(sections)
    assert sections and len(set(sections)) == len(sections) and set(sections) <= set(all_sections + ['nearcopy'])
    partial = set(sections) != set(all_sections) or aspects is not None or validation_only
    work = Computation(out, n_clu, n_ret, aspects, validation_only)
    start = time.monotonic()
    unavailable = []
    for section in sections:
        print(f'C13 derive {section}', flush=True)
        try:
            if section == 'nearcopy':
                work.derive_nearcopy()
            elif section == 'stage2':
                work.emoset()
            else:
                part, enc = section.split(':')
                getattr(work, part)(enc)
        except FileNotFoundError as exc:
            # D18 cannot be inferred from missing vectors. Preserve the evidence
            # and continue other paths; only compare may consult status.json to
            # excuse a missing path. Any missing live-path input still fails C13.
            unavailable.append({'section': section, 'missing': str(exc.filename)})
            print(f'C13 unavailable {section}: {exc.filename}', flush=True)
        # Checkpoint completed sections even if a subsequent input is unavailable.
        doc = {'schema': 1, 'partial': partial, 'sections': sections, 'aspects': list(work.aspects),
               'validation_only': validation_only,
               'unavailable': list(unavailable),
               'n_clu': n_clu, 'n_ret': n_ret, 'nearcopy': work.nearcopy,
               'stage1': work.stage1, 'stage2': work.stage2, 'cells': work.cells,
               'verdicts': verdicts(work.cells, partial=True),
               'seconds': time.monotonic() - start, 'complete': False}
        write_json(out / 'derived.json', doc)
    doc['complete'] = True
    doc['verdicts'] = verdicts(work.cells, partial=partial or bool(unavailable))
    write_json(out / 'derived.json', doc)
    print(f"C13 derived cells={len(work.cells)} seconds={doc['seconds']:.1f} partial={partial}")
    return doc


class Comparison:
    def __init__(self):
        self.items, self.mismatches, self.max_diff = [], [], {}

    def add(self, name, derived, recorded, kind='exact', sets=False):
        diff = None
        if kind != 'exact' and all(isinstance(v, (float, int)) and not isinstance(v, bool)
                                   for v in (derived, recorded)):
            diff = abs(float(derived) - float(recorded))
            ok = bool(np.isfinite(diff) and diff <= TOL)
            self.max_diff[kind] = max(self.max_diff.get(kind, 0.), diff) if np.isfinite(diff) else None
        elif sets and isinstance(derived, list) and isinstance(recorded, list):
            ok = (len(derived) == len(set(derived)) and len(recorded) == len(set(recorded))
                  and set(derived) == set(recorded))
        else:
            ok = type(derived) is type(recorded) and derived == recorded
        entry = {'name': name, 'derived': derived, 'recorded': recorded, 'ok': ok}
        if diff is not None:
            entry['abs_diff'] = diff
        self.items.append(entry)
        if not ok:
            self.mismatches.append(entry)

    def tree(self, name, a, b, kind='measurement'):
        if isinstance(a, dict):
            for k, value in a.items():
                self.tree(f'{name}.{k}', value, b.get(k) if isinstance(b, dict) else None, kind)
        elif isinstance(a, list):
            self.add(name + '.length', len(a), len(b) if isinstance(b, list) else None)
            for i, value in enumerate(a):
                self.tree(f'{name}.{i}', value, b[i] if isinstance(b, list) and i < len(b) else None, kind)
        else:
            self.add(name, a, b, kind if isinstance(a, float) else 'exact')


def compare(derived=None, points_only=False, report=None):
    """The sole r14 result-record boundary. Layout adapters live only here."""
    doc = read_json(P.OUT / 'c13/derived.json' if derived is None else derived)
    result = Comparison()
    result.add('derive.complete', doc.get('complete'), True)
    status_path = P.RECORDS / 'status.json'
    status = read_json(status_path) if status_path.exists() else None
    for missing in doc.get('unavailable', []):
        result.add('missing_input.' + missing['section'],
                   missing_path_excused(missing['section'], status), True)
    # The record adapters are intentionally explicit; a missing field fails.
    compare_records(result, doc)
    recorded = []
    for path in sorted((P.OUT / 'stage3').glob('cells_*.json')):
        if not path.name.endswith('.pending.json'):
            recorded.extend(read_json(path))
    ours, theirs = index_cells(doc['cells']), index_cells([recorded_identity(c) for c in recorded])
    for key, cell in ours.items():
        name = '/'.join(key)
        ref = theirs.get(key)
        result.add(name + '.present', ref is not None, True)
        if ref is None:
            continue
        for field in ('candidate_value', 'comparator_value', 'd', 'lo', 'hi', 'r'):
            if points_only and field in ('lo', 'hi'):
                continue
            expected = ref.get(field)
            result.add(name + '.' + field, cell[field], expected, field)
        result.add(name + '.seeds', cell['seeds'], ref.get('seeds'))
        if not points_only:
            result.add(name + '.n_draws', cell['n_draws'], ref.get('n_draws'))
            result.add(name + '.outcome', cell['outcome'] if ref.get('r') is not None else
                       ('WIN' if cell['lo'] > 0 else 'not WIN'),
                       cell_outcome(ref['lo'], ref['r']) if ref.get('r') is not None else
                       ('WIN' if ref['lo'] > 0 else 'not WIN'))
    if not doc['partial']:
        result.add('cells.identities', sorted(ours), sorted(theirs))
    outcomes = P.OUT / 'stage3/outcomes.json'
    availability = 'not yet written'
    status_verdicts = None
    if status is not None:
        try:
            status_verdicts = verdicts(doc['cells'], status, partial=doc['partial'])
        except AssertionError as exc:
            result.add('status.live_paths_complete', str(exc), 'all required cells present')
    if outcomes.exists() and not points_only and status_verdicts is not None:
        availability = 'compared'
        compare_outcomes(result, doc, read_json(outcomes), status)
    elif outcomes.exists():
        availability = 'skipped in points-only smoke' if points_only else 'status unavailable'
        if not points_only:
            result.add('outcomes.status_present', status is not None, True)
    payload = {'items_compared': len(result.items), 'mismatches': result.mismatches,
               'max_absolute_difference': result.max_diff, 'items': result.items,
               'partial': doc['partial'], 'points_only': points_only,
               'outcomes': availability,
               'status_verdicts': status_verdicts}
    write_json(P.OUT / 'c13/compare.json' if report is None else report, payload)
    mx = max((v for v in result.max_diff.values() if v is not None), default=0.)
    print(f'C13 compare items={len(result.items)} mismatches={len(result.mismatches)} max_diff={mx:.12g}')
    return payload


def recorded_identity(cell):
    """Compare-only name adapter (run chat, 2026-11-02): the round's cells name two comparators differently from C13.
    SC3: 'matched_M2' is C13's 'M2'. Affection SC6: 'teacher_artwork' (r12's artwork-wording outputs, bs_affection) is
    C13's 'teacher_emotion_focus' (the artwork wording's view; EmoSet cells already use that name). Values untouched."""
    comp = cell['comparator']
    if cell['sc'] == 'SC3' and comp.startswith('matched_'):
        return {**cell, 'comparator': comp[len('matched_'):]}
    if cell['data'] == 'affection_test' and comp == 'teacher_artwork':
        return {**cell, 'comparator': 'teacher_emotion_focus'}
    return cell


def missing_path_excused(section, status):
    """D18 excuses missing inputs only on an explicitly failed status path."""
    if status is None:
        return False
    if section in ('nearcopy', 'stage2'):
        return 'failed' in status['part_c'].values()
    part, enc = section.split(':')
    st = status['encoders'][enc]
    if part in ('artclu', 'artret'):
        return st['cstar_runs'] == 'failed' or st['stage0']['paintings'] == 'failed'
    return st['cstar_runs'] == 'failed' or st['photo_runs'] == 'failed' or st['stage0']['photos'] == 'failed'


def compare_records(result, doc):
    """Record-format projections, populated solely from compare-mode evidence."""
    for enc, entry in doc['stage1'].items():
        path = P.RECORDS / f'stage1_{enc}.json'
        record = read_json(path) if path.exists() else {}
        result.add(f'stage1.{enc}.encoder', enc, record.get('encoder'))
        for part in ('artclu', 'artret'):
            if part not in entry:
                continue
            ranges = entry[part] if part == 'artclu' else entry[part]['ranges']
            ref = record.get('artclu' if part == 'artclu' else 'artret_ranges', {})
            for aspect, values in ranges.items():
                name = f'stage1.{enc}.{part}.{aspect}'
                result.add(name + '.r', values['r'], ref.get('ranges', {}).get(aspect), 'range')
                per_seed = ref.get('per_seed', {}).get(aspect, {})
                for seed, value in enumerate(values['per_seed']):
                    result.add(name + f'.s{seed}', value, per_seed.get(str(seed)), 'seed_value')
        if 'artret' in entry:
            picks = record.get('artret_picks', {})
            for arm in ('cstar', 'frozen', 'teacher'):
                menu = entry['artret'][arm]
                ref = picks.get(arm, {})
                result.add(f'stage1.{enc}.artret.{arm}.pick', menu['pick'], ref.get('pick'))
                if 'scores' in menu:
                    result.tree(f'stage1.{enc}.artret.{arm}.scores', menu['scores'], ref.get('table'))
        if 'affection' in entry:
            ref = record.get('affection', {})
            result.tree(f'stage1.{enc}.affection.ranges', entry['affection']['ranges'], ref.get('ranges_doc'))
            for arm, target in (('photo', 'photo_readout'), ('frozen', 'frozen')):
                menu, other = entry['affection'][arm], ref.get(target, {})
                result.add(f'stage1.{enc}.affection.{arm}.pick', menu['pick'], other.get('pick'))
                if 'scores' in menu:
                    result.tree(f'stage1.{enc}.affection.{arm}.scores', menu['scores'], other.get('scores'))
    if doc['nearcopy'] is not None:
        path = P.RECORDS / 'nearcopy_emoset.json'
        record = read_json(path) if path.exists() else {}
        result.add('nearcopy.cut', doc['nearcopy']['cut'], record.get('cut'), 'cut')
        result.add('nearcopy.n_reference', doc['nearcopy']['n_reference'], record.get('reference', {}).get('n'))
        for split, entry in doc['nearcopy']['splits'].items():
            ref = record.get('splits', {}).get(split, {})
            for k in ('n_removed', 'n_kept'):
                result.add(f'nearcopy.{split}.{k}', entry[k], ref.get(k))
            result.add(f'nearcopy.{split}.n', entry['n_total'], ref.get('n'))
            result.add(f'nearcopy.{split}.removed_keys', entry['removed_keys'], ref.get('removed_keys'), sets=True)
            # The nearcopy record stores only the removed set. Reconstruct its
            # implied order independently; if a stage2 order is saved, check it too.
            if isinstance(ref.get('removed_keys'), list):
                gone = set(ref['removed_keys'])
                order = hashed_order(k for k in emo_keys(split) if k not in gone)
            else:
                order = None
            result.add(f'nearcopy.{split}.order', entry['keys'], order)
            result.tree(f'nearcopy.{split}.hist', entry['hist'], ref.get('hist'))
    if doc['stage2'] is not None:
        path = P.RECORDS / 'stage2_emoset.json'
        record = read_json(path) if path.exists() else {}
        compare_stage2(result, doc['stage2'], record)


def named_node(record, names):
    """Compare-only schema lookup; names are interface aliases, never values."""
    if not isinstance(record, dict):
        return None
    for name in names:
        if name in record:
            return record[name]
    for value in record.values():
        if isinstance(value, dict):
            found = named_node(value, names)
            if found is not None:
                return found
    return None


def scalar(node, names=('verdict', 'outcome')):
    if isinstance(node, dict):
        return next((node[k] for k in names if k in node), None)
    return node


def compare_stage2(result, derived, record):
    """Project the stage-2 interface, accepting nested measurement documents.

    An unrecognized or absent required measurement always compares as missing;
    metadata such as timestamps and source file names is outside C13's scope.
    """
    for metric in ('clu', 'map'):
        aliases = ('clu', 'ami', 'clustering') if metric == 'clu' else ('map', 'mAP', 'retrieval')
        ranges = named_node(record, ('ranges_doc', 'r_E_doc', 'seed_ranges', 'ranges', 'r_E')) or {}
        ranges = ranges if isinstance(ranges, dict) else {}
        entry = next((ranges[k] for k in aliases if k in ranges), None)
        if isinstance(entry, dict) and 'per_seed' in entry:
            expected = dict(entry)
            if isinstance(expected['per_seed'], dict):
                expected['per_seed'] = [expected['per_seed'].get(str(s)) for s in SEEDS]
        else:
            r = scalar(entry, ('r', 'range'))
            if r is None:
                r = named_node(record, ('r_E_clu',) if metric == 'clu' else ('r_E_map', 'r_E_mAP'))
            seeds = named_node(record, ('per_seed',)) or {}
            per_seed = next((seeds[k] for k in aliases if k in seeds), None) if isinstance(seeds, dict) else None
            if per_seed is None:
                per_seed = named_node(ranges, (f'{metric}_per_seed', f'per_seed_{metric}'))
            if isinstance(per_seed, dict):
                per_seed = [per_seed.get(str(s)) for s in SEEDS]
            expected = {'r': r, 'per_seed': per_seed}
        result.tree(f'stage2.ranges.{metric}', derived['ranges'][metric], expected)
    menus = named_node(record, ('readouts', 'readout', 'menus')) or {}
    menus = menus if isinstance(menus, dict) else {}
    if isinstance(record.get('descriptive'), dict):    # the record keeps §3.4 step 4's rows apart (run chat, 2026-11-02)
        menus = {**menus, **record['descriptive']}
    for arm, menu in derived['readouts'].items():
        target = menus.get(arm, {})
        if not isinstance(target, dict):
            target = {'pick': target, 'scores': (named_node(record, ('tables', 'readout_scores', 'scores')) or {}).get(arm)}
        projected = {'pick': target.get('pick'), 'scores': target.get('scores', target.get('table'))}
        result.tree(f'stage2.readouts.{arm}', menu, projected)
    wordings = named_node(record, ('teacher_wording', 'wording_picks', 'wording')) or {}
    for task, pick in derived['teacher_wording'].items():
        aliases = (task, 'clu', 'ami') if task == 'clustering' else (task, 'ret', 'map', 'mAP')
        node = next((wordings[k] for k in aliases if k in wordings), None)
        target = scalar(node, ('pick', 'wording'))
        result.add(f'stage2.teacher_wording.{task}', pick, target)
    scores = named_node(record, ('wording_scores', 'teacher_scores', 'teacher_wording_scores')) or wordings
    teacher = record.get('teacher')
    if isinstance(teacher, dict) and 'ami' in teacher and 'map_at_own_pick' in teacher:   # the record's layout (run chat)
        scores = {'clustering': teacher['ami'], 'retrieval': teacher['map_at_own_pick']}
    for task, values in derived['wording_scores'].items():
        aliases = (task, 'clu', 'ami') if task == 'clustering' else (task, 'ret', 'map', 'mAP')
        target = next((scores[k] for k in aliases if k in scores), {})
        target = target if isinstance(target, dict) else {}
        target = target.get('scores', target)
        for wording, value in values.items():
            key = 'photo' if wording.endswith('_photo') else 'artwork'
            other = next((target[k] for k in (wording, 'teacher_' + wording, key) if k in target), None)
            result.add(f'stage2.wording_scores.{task}.{key}', value,
                       scalar(other, ('ami', 'map', 'mAP', 'value', 'score')), 'wording_score')


def compare_outcomes(result, doc, record, status):
    expected = verdicts(doc['cells'], status, partial=doc['partial'])
    root = record.get('verdicts', record)
    encoders = named_node(root, ('encoders', 'part_a')) or root
    for enc, parts in expected['encoders'].items():
        target = encoders.get(enc, {})
        for part, value in parts.items():
            if value != 'pending':
                result.add(f'verdicts.{enc}.{part}', value, scalar(target.get(part)))
    if expected['part_c'] != 'pending':
        result.add('verdicts.part_c', expected['part_c'], scalar(named_node(root, ('part_c',))))
    # Cell identity, outcome and SC6 claims are checked independently of sentences.
    def cell_rows(node):
        if isinstance(node, dict):
            if all(k in node for k in IDENTITY):
                yield node
            else:
                for v in node.values():
                    yield from cell_rows(v)
        elif isinstance(node, list):
            for v in node:
                yield from cell_rows(v)
    recorded_cells = index_cells(list(cell_rows(record.get('cells', []))))
    recorded_claims = index_cells(list(cell_rows(record.get('claims', []))))
    for c in doc['cells']:
        other = recorded_cells.get(cell_id(c), {})
        outcome = c['outcome'] if c['r'] is not None else ('WIN' if c['lo'] > 0 else 'not WIN')
        result.add('/'.join(cell_id(c)) + '.written_outcome', outcome, other.get('outcome'))
    for c in expected['claims']:
        other = recorded_claims.get(cell_id(c), recorded_cells.get(cell_id(c), {}))
        result.add('/'.join(cell_id(c)) + '.claim', c['claim'],
                   scalar(other, ('claim', 'licensed', 'pass')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('derive')
    d.add_argument('--n-clu', type=int, default=2000)
    d.add_argument('--n-ret', type=int, default=10000)
    d.add_argument('--out', type=Path, default=P.OUT / 'c13')
    d.add_argument('--sections', nargs='+', help='subset smoke, e.g. artclu:siglip2 or nearcopy')
    d.add_argument('--aspects', nargs='+', choices=P.ASPECTS, help='subset smoke; artclu/artret cells only')
    d.add_argument('--validation-only', action='store_true', help='subset smoke: ranges and picks, no held scoring')
    c = sub.add_parser('compare')
    c.add_argument('--derived', type=Path, default=P.OUT / 'c13/derived.json')
    c.add_argument('--points-only', action='store_true')
    c.add_argument('--report', type=Path, help='override report path for a scratch-only smoke')
    args = parser.parse_args(argv)
    with threadpool_limits(8):
        if args.command == 'derive':
            derive(args.n_clu, args.n_ret, args.out, args.sections, args.aspects, args.validation_only)
            return 0
        return int(bool(compare(args.derived, args.points_only, args.report)['mismatches']))


if __name__ == '__main__':
    raise SystemExit(main())
