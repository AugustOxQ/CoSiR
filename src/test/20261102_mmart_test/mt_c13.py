"""Independent r15 C13 from the binding MMArt rule and saved keyed inputs.

No mt_* or r14 scorer is imported. Frozen r10--r13 numerical functions are
accessed through the permitted bs_c13 module; mp_stores supplies keyed joins.
Only compare() opens round results (records/, stage1/, stage2/). Derivation
reads only the index, near-copy table, stores and checked export bytes.

Store retrieval rows use unit(unit(raw)) in float32. This follows the round's
implementation, not r11's once-normalised protocol. Clustering uses unit(raw)
for stores and unnormalised float16->float32 c* exports. Retrieval c* rows are
normalised once. All populations and parity halves use SHA256 key order.
K-means defaults to 8 threads: the round runner sets OMP_NUM_THREADS=8; the rule fixes no thread count.
Smoke: one thread changes 212/16,384 assignments in seed 2, state 1, emotion; mean AMI differs by 1.3e-5.

Sections: cut, ranges:ENC, picks:ENC (or picks:ENC:cstar/frozen), picks:teacher,
cells:ENC, outcomes. Dependencies are derived too, never read from results.
Example smoke: --sections cut ranges:siglip2 picks:siglip2 picks:teacher
cells:siglip2 --draws-clu 40 --draws-ret 100. Full scope requires all sections,
all 60 cells, complete score tables and protocol draw counts. Caches include
input arrays/identities and the frozen source bytes. No old partitions reused.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

ROUND = Path('/project/CoSiR/src/test/20261102_mmart_test')
BASE = Path('/data/SSD2/cosir_r15/20261102_mmart_test')
DEFAULT_ROOT = BASE / 'outputs'
DEFAULT_OUT = BASE / 'c13'
SMOKE_OUT = BASE / 'c13_smoke'
INDEX = Path('/data/SSD/mmart/mmart_index.parquet')
NEARCOPY = Path('/data/SSD/mmart/mmart_nearcopy.parquet')
MMART = Path('/data/SSD2/pre_extract/mmart')
ART = Path('/data/SSD2/pre_extract/artelingo_full')
for _p in ('20261102_backbones_sweep', '20261102_mmart_prep'):
    sys.path.insert(0, '/project/CoSiR/src/test/' + _p)
import bs_c13 as frozen
import mp_stores

ac_cluster, ac_select, ac_stats = frozen.ac_cluster, frozen.ac_select, frozen.ac_stats
rt_metrics, rt_readout, rt_stats = frozen.rt_metrics, frozen.rt_readout, frozen.rt_stats
rb_score = frozen.rb_score
ENCODERS = ('qwen', 'siglip2', 'siglip1', 'clip_l14')
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_VIEW = dict(qwen='qwen_default', siglip2='siglip2', siglip1='siglip1', clip_l14='clip_l14')
TEACHER_VIEW = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
MATCHED = dict(qwen='P3', siglip2='M2', siglip1='M8', clip_l14='M7')
D7 = {e: {('emotion', 'clu'), ('style', 'clu'), ('genre', 'ret')} |
      ({('style', 'ret')} if e != 'qwen' else set()) for e in ENCODERS}
TOL = 1e-6
IDENTITY = ('sc', 'encoder', 'aspect', 'task', 'data', 'candidate', 'comparator')
FULL_SECTIONS = ('cut',) + tuple(f'{s}:{e}' for e in ENCODERS for s in ('ranges', 'picks', 'cells')) + ('picks:teacher', 'outcomes')
STAGE2_FILES = ('stage2/outcomes.json', 'stage2/cells_clu.json', 'stage2/cells_ret.json')


def read_json(path):
    return json.loads(Path(path).read_text())


def check_output(path):
    path = Path(path).resolve()
    assert any(path.is_relative_to(p) for p in (DEFAULT_OUT, SMOKE_OUT)), f'output outside C13 directories: {path}'
    return path


def write_json(path, value):
    path = check_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


join_rows = frozen.join_rows
hashed_order = frozen.hashed_order


def keys_sha(keys):
    return hashlib.sha256('\n'.join(keys).encode('utf-8')).hexdigest()


def unit(x, floor=False):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and (norms > 0).all(), 'zero/nonfinite row'
    return x / (np.maximum(norms, 1e-12) if floor else norms)


def store_retrieval(x):
    return unit(unit(x, floor=True))


def kmeans_partitions(x, threads=8):
    """r14's fit settings, with an explicit limit replacing its forced one thread."""
    assert isinstance(threads, int) and not isinstance(threads, bool) and threads >= 1
    x = np.asarray(x).astype(np.float32)
    assert np.isfinite(x).all()
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < frozen.rs_vh.UNIT_TOL, 'rows are not unit norm'
    with threadpool_limits(limits=threads):  # both BLAS and OpenMP
        parts = [MiniBatchKMeans(n_clusters=20, random_state=state, n_init=3, batch_size=4096).fit(x).labels_
                 for state in (0, 1, 2)]
    return np.stack(parts).astype(np.int32)


def cut(index, nearcopy, threshold):
    """Ordered filters; counts are incremental, never overlapping removals."""
    assert index.key.is_unique and nearcopy.key.is_unique
    rows = index[index.split == 'new'].set_index('key')
    order = hashed_order(rows.index)
    cos = nearcopy.set_index('key').loc[order, 'max_cos'].to_numpy()
    assert set(nearcopy.key) == set(order) and np.isfinite(cos).all()
    kept, seen = [], set()
    dropped = dict(exact=0, near=0, md5=0)
    for k, c in zip(order, cos):
        row = rows.loc[k]
        if row.exact_dup_trainval:
            dropped['exact'] += 1
        elif float(c) >= threshold:
            dropped['near'] += 1
        elif row.md5 in seen:
            dropped['md5'] += 1
        else:
            assert isinstance(row.md5, str) and row.md5
            seen.add(row.md5)
            kept.append(k)
    return dict(threshold=threshold, n=len(kept), n_new=len(order), dropped=dropped,
                sha256=keys_sha(kept), keys=kept)


def cell_id(c):
    return tuple(c[k] for k in IDENTITY)


def index_cells(cells):
    result = {}
    for c in cells:
        k = cell_id(c)
        assert k not in result, f'duplicate cell {k}'
        result[k] = c
    return result


def expected_ids(enc):
    out = set()
    for a in ASPECTS:
        for sc, task, comp in [('SC1', 'clu', f'frozen_{enc}'),
                               ('SC2', 'ret', f'frozen_{enc}'),
                               ('SC3', 'clu', f'matched_{MATCHED[enc]}'),
                               ('SC4', 'clu', f'teacher_{a}'),
                               ('SC4', 'ret', f'teacher_{a}')]:
            out.add((sc, enc, a, task, 'mmart_test', f'cstar_{enc}', comp))
    return out


def cell_outcome(lo, r=None):
    assert np.isfinite(lo)
    if r is None:
        return 'WIN' if lo > 0 else 'not WIN'
    assert np.isfinite(r) and r >= 0
    return 'WIN' if lo > 0 else 'LEVEL' if lo > -r else 'BELOW'


def good_status():
    return dict(gates={**{k: 'ok' for k in ('i_nearcopy', 'iii', 'vi')},
                       **{k: dict.fromkeys(ENCODERS, 'ok') for k in ('i_default', 'ii', 'iv')},
                       **{k: dict.fromkeys(ASPECTS, 'ok') for k in ('i_teacher', 'v')}},
                stage1={**{k: dict.fromkeys(ENCODERS, 'ok') for k in ('ranges', 'picks')}, 'teacher_pick': 'ok'})


def validate_status(status):
    expected = good_status()
    for top, fields in expected.items():
        for k, value in fields.items():
            actual = status[top][k]
            if top == 'stage1' and k == 'ranges' and isinstance(actual, str):
                actual = dict.fromkeys(ENCODERS, actual)
            if isinstance(value, dict):
                assert set(actual) == set(value), f'{top}.{k}'
                assert set(actual.values()) <= {'ok', 'failed'}, f'{top}.{k}'
            else:
                assert actual in ('ok', 'failed'), f'{top}.{k}'


def voids(status, enc, sc=None, aspect=None, task=None):
    """D17. sc=None asks whether the encoder verdict is void."""
    g, s = status['gates'], status['stage1']
    ranges = s['ranges'] if isinstance(s['ranges'], dict) else dict.fromkeys(ENCODERS, s['ranges'])
    reasons = []
    def add(bad, name):
        if bad:
            reasons.append(name)
    add(g['i_nearcopy'] == 'failed', 'gates.i_nearcopy')
    add(g['iii'] == 'failed', 'gates.iii')
    add(g['i_default']['qwen'] == 'failed', 'gates.i_default.qwen')
    if enc != 'qwen':
        add(g['i_default'][enc] == 'failed', 'gates.i_default.' + enc)
    add(g['ii'][enc] == 'failed', 'gates.ii.' + enc)
    if sc is None or sc in ('SC2', 'SC4'):
        add(ranges[enc] == 'failed', 'stage1.ranges.' + enc)
        add(s['picks'][enc] == 'failed', 'stage1.picks.' + enc)
    if sc is None or sc == 'SC3':
        add(g['iv'][enc] == 'failed', 'gates.iv.' + enc)
    if sc is None or sc == 'SC2' or (sc == 'SC4' and task == 'ret'):
        add(g['vi'] == 'failed', 'gates.vi')
    if sc == 'SC4':
        for gate in ('i_teacher', 'v'):
            add(g[gate][aspect] == 'failed', f'gates.{gate}.{aspect}')
        if task == 'ret':
            add(s['teacher_pick'] == 'failed', 'stage1.teacher_pick')
            add(ranges['qwen'] == 'failed', 'stage1.ranges.qwen.teacher')
    return reasons


def sc4_claim(enc, aspect, task, outcome, void=False):
    if void:
        return 'inconclusive'
    if outcome == 'WIN' and (aspect, task) in D7[enc]:
        return 'better than'
    return 'level with or better than' if outcome in ('WIN', 'LEVEL') else 'no claim'


def decisions(cells, status=None, encoders=ENCODERS, partial=False):
    status = good_status() if status is None else status
    validate_status(status)
    indexed = index_cells(cells)
    result = dict(verdicts={}, sc4={}, cells=[])
    for enc in encoders:
        missing = expected_ids(enc) - set(indexed)
        live_missing = [k for k in missing if not voids(status, enc, k[0], k[2], k[3])]
        assert partial or not live_missing, f'missing live cells: {live_missing[:2]}'
        rows = [c for c in cells if c['encoder'] == enc]
        decorated = []
        for c in rows:
            why = voids(status, enc, c['sc'], c['aspect'], c['task'])
            decorated.append({**c, 'outcome': cell_outcome(c['lo'], c['r']),
                              'status': 'inconclusive' if why else 'ok', 'void': why})
        result['cells'].extend(decorated)
        bars = [c for c in decorated if c['sc'] != 'SC4']
        why = voids(status, enc)
        if why:
            verdict = 'inconclusive'
        elif len(bars) != 9:
            assert partial
            verdict = 'pending'
        elif all(c['outcome'] == 'WIN' for c in bars):
            verdict = 'holds'
        elif all(c['outcome'] == 'WIN' for c in bars if c['sc'] in ('SC1', 'SC2')):
            verdict = 'qualified'
        else:
            verdict = 'fails'
        result['verdicts'][enc] = verdict
        result['sc4'][enc] = [dict(aspect=c['aspect'], task=c['task'],
                                  d7=(c['aspect'], c['task']) in D7[enc],
                                  outcome=c['outcome'], status=c['status'], void=bool(c['void']),
                                  claim=sc4_claim(enc, c['aspect'], c['task'], c['outcome'], bool(c['void'])))
                               for c in decorated if c['sc'] == 'SC4']
    return result


@lru_cache(maxsize=1)
def source_identity():
    return hashlib.sha256((frozen.frozen_identity() + file_sha(__file__) + file_sha(mp_stores.__file__)).encode()).hexdigest()


class Computation:
    def __init__(self, root, out, n_clu, n_ret, workers, kmeans_threads=8):
        assert isinstance(kmeans_threads, int) and not isinstance(kmeans_threads, bool) and kmeans_threads >= 1
        self.root, self.out = Path(root).resolve(), check_output(out)
        self.n_clu, self.n_ret, self.workers = n_clu, n_ret, workers
        self.kmeans_threads = kmeans_threads
        self.operation_seconds, self.cache_hits = {}, {}
        self.index = pd.read_parquet(INDEX)
        assert self.index.key.is_unique
        self.valkeys = hashed_order(self.index.loc[self.index.split == 'val', 'key'])
        assert len(self.valkeys) == 5769
        self.cutsets, self.ranges, self.picks, self.cells = {}, {}, {}, []
        self.inputs = {str(p): file_sha(p) for p in (INDEX, NEARCOPY)}

    def array(self, name, inputs, calculate):
        h = hashlib.sha256(source_identity().encode())
        for x in inputs:
            if isinstance(x, np.ndarray):
                h.update(str((x.shape, x.dtype.str)).encode())
                h.update(np.ascontiguousarray(x).tobytes())
            else:
                h.update(json.dumps(x, sort_keys=True).encode())
        path = self.out / 'cache' / f'{name}__{h.hexdigest()}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                self.cache_hits[name] = self.cache_hits.get(name, 0) + 1
                return z['value']
        start = time.monotonic()
        value = np.asarray(calculate())
        assert np.isfinite(value).all(), name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, value=value)
        seconds = time.monotonic() - start
        self.operation_seconds[name] = self.operation_seconds.get(name, 0.) + seconds
        print(f'{name}: {seconds:.1f}s', flush=True)
        return value

    def derive_cut(self):
        if not self.cutsets:
            near = pd.read_parquet(NEARCOPY)
            self.cutsets = {name: cut(self.index, near, t) for name, t in (('test', .95), ('far', .85))}
            assert self.cutsets['test']['n'] == 16384 and self.cutsets['far']['n'] == 10403
        return self.cutsets

    def keys(self, split):
        return self.valkeys if split == 'val' else self.derive_cut()['test']['keys']

    def labels(self, split):
        rows = self.index.set_index('key').loc[self.keys(split)]
        labels = {}
        for a in ASPECTS:
            # Label identities need only preserve equality; missing stays -1.
            values = rows['genre_name' if a == 'genre' else a].fillna('')
            classes = sorted(set(values) - {''})
            codes = {name: i for i, name in enumerate(classes)}
            labels[a] = np.asarray([codes.get(v, -1) for v in values], dtype=np.int64)
        return labels

    def exported(self, enc, split, a, seed, matched=False):
        keyset = 'val' if split == 'val' else 'new'
        path = self.root / 'exports' / keyset / enc / ('matched' if matched else 'cstar') / (
            f'probs__s{seed}.npz' if matched else f'{a}__s{seed}.npz')
        digest = file_sha(path)
        side = read_json(path.with_suffix('.json'))
        assert side['sha256'] == digest, f'export checksum: {path}'
        self.inputs[str(path)] = digest
        self.inputs[str(path.with_suffix('.json'))] = file_sha(path.with_suffix('.json'))
        if not matched:
            assert (side['encoder'], side['aspect'], side['seed'], side['keyset']) == (enc, a, seed, keyset)
            assert side['view'] == DEFAULT_VIEW[enc]
        with np.load(path, allow_pickle=False) as z:
            stored, x = z['keys'].tolist(), z['probs' if matched else 'emb']
        assert x.dtype == (np.float32 if matched else np.float16)
        assert x.shape == (len(stored), 40 if matched else 2048) and np.isfinite(x).all()
        if 'keys_sha256' in side:
            assert keys_sha(stored) == side['keys_sha256']
        return x[join_rows(stored, self.keys(split))].astype(np.float32)

    def features(self, view, split):
        keys = self.keys(split)
        if split == 'val' and view in TEACHER_VIEW.values():
            # D8 overrides mp_stores' held-only teacher route.
            folder = ART / (mp_stores.FOLDERS[view] + '_s2img')
            names = self.index.set_index('key').loc[keys, 'painting'].tolist()
            x = mp_stores.join(folder, names, view, teacher=True, reference=True)
        else:
            x = mp_stores.load(view, keys, index=self.index, root=MMART, art_root=ART)
        return unit(x, floor=True)

    def vector(self, enc, arm, split, a, seed=None, retrieval=False):
        if arm == 'cstar':
            x = self.exported(enc, split, a, seed)
            return unit(x) if retrieval else x
        view = TEACHER_VIEW[a] if arm == 'teacher' else DEFAULT_VIEW[enc]
        x = self.features(view, split)
        return unit(x) if retrieval else x

    def parts(self, x, keys):
        """All fits, including matched arms, share this thread-bound cache path."""
        return self.array('partitions', [keys, x, {'kmeans_threads': self.kmeans_threads}],
                          lambda: kmeans_partitions(x, self.kmeans_threads))

    def derive_ranges(self, enc, tasks=('clu', 'ret')):
        entry = self.ranges.setdefault(enc, {})
        labels = self.labels('val')
        for task in tasks:
            if task in entry:
                continue
            entry[task] = {}
            for a in ASPECTS:
                values = []
                for s in SEEDS:
                    x = self.vector(enc, 'cstar', 'val', a, s, retrieval=task == 'ret')
                    if task == 'clu':
                        value = ac_cluster.ami(self.parts(x, self.valkeys), labels[a])
                    else:
                        def calc():
                            res = rt_metrics.retrieval(x, labels[a], qe_k=0, workers=self.workers)
                            assert np.array_equal(res['rows'], np.flatnonzero(labels[a] >= 0))
                            return res['ap']
                        value = float(self.array('r0_ap', [self.valkeys, x, labels[a]], calc).mean())
                    values.append(float(value))
                entry[task][a] = dict(per_seed={str(s): v for s, v in zip(SEEDS, values)}, r=rt_stats.seed_range(values))
        return entry

    def derive_pick(self, enc, arm):
        key = 'teacher' if arm == 'teacher' else f'{enc}:{arm}'
        if key in self.picks:
            return self.picks[key]
        ranges = {a: v['r'] for a, v in self.derive_ranges(enc, ('ret',))['ret'].items()}
        labels = self.labels('val')
        scores = []
        for s in SEEDS if arm == 'cstar' else (None,):
            per_aspect = []
            for a in ASPECTS:
                x = self.vector(enc, arm, 'val', a, s, retrieval=True)
                ros = rt_readout.menu(x.shape[1])
                def calc():
                    rows = rt_readout.crossfit_all({(a, None): x}, {a: labels[a]}, ros, workers=self.workers)
                    return np.asarray([rows[ro][a] for ro in ros])
                per_aspect.append(self.array('crossfit', [self.valkeys, x, labels[a]], calc))
            scores.append(np.stack(per_aspect, axis=1))
        means = np.mean(scores, axis=0)
        rows = {ro: {a: float(means[i, j]) for j, a in enumerate(ASPECTS)} for i, ro in enumerate(ros)}
        score, keep = ac_select.readout_scores(rows, ranges), ac_select.readout_guard(rows, ranges)
        pick = ac_select.pick_readout(rows, ranges)
        entry = dict(pick=rt_readout.name(pick), table={rt_readout.name(ro): dict(map=rows[ro], score=float(score[ro]), kept=bool(keep[ro])) for ro in ros},
                     ranges=ranges, dropped=rt_readout.half_dropped(labels))
        self.picks[key] = entry
        return entry

    def retrieval(self, enc, arm, a, y):
        pick = self.derive_pick('qwen' if arm == 'teacher' else enc, arm)['pick']
        aps = []
        for s in SEEDS if arm == 'cstar' else (None,):
            val = self.vector(enc, arm, 'val', a, s, retrieval=True)
            test = self.vector(enc, arm, 'test', a, s, retrieval=True)
            def calc():
                res = ac_stats.arm_retrieval(lambda _: val, lambda _: test, pick, y, workers=self.workers)
                assert np.array_equal(res['rows'], np.flatnonzero(y >= 0))
                return res['ap']
            aps.append(self.array('picked_ap', [self.valkeys, self.keys('test'), val, test, pick, y], calc))
        return (np.stack(aps) if arm == 'cstar' else aps[0]), pick

    def derive_cells(self, enc):
        if any(c['encoder'] == enc for c in self.cells):
            return
        ranges = self.derive_ranges(enc)
        keys, labels = self.keys('test'), self.labels('test')
        frozen_parts = self.parts(self.vector(enc, 'frozen', 'test', 'emotion'), keys)
        matched = {}
        for s in SEEDS:
            prob = self.exported(enc, 'test', None, s, matched=True)
            matched[s] = dict(own=np.argmax(prob, axis=1).astype(np.int32)[None, :],
                              r20=self.parts(unit(prob), keys))
        for a in ASPECTS:
            y = labels[a]
            cand = np.stack([self.parts(self.vector(enc, 'cstar', 'test', a, s), keys) for s in SEEDS])
            cf = {s: {'r20': cand[s]} for s in SEEDS}
            for sc, comp, other, seedless in [('SC1', f'frozen_{enc}', {0: {'r20': frozen_parts}}, True),
                                               ('SC3', f'matched_{MATCHED[enc]}', matched, False)]:
                def calc():
                    result = rb_score.paired_bootstrap({'c': cf, 'b': other}, y,
                        {'c': list(SEEDS), **({} if seedless else {'b': list(SEEDS)})}, ('c', 'b'),
                        self.n_clu, workers=self.workers, seedless=('b',) if seedless else ())
                    return [result['A_cand'], max(result['A_e'].values()), result['d'], *result['interval']]
                inputs = [keys, y, cand, sc, self.n_clu] + [other[s][r] for s in sorted(other) for r in sorted(other[s])]
                stats = self.array('clu_cell', inputs, calc)
                self.cell(sc, enc, a, 'clu', comp, stats, None, y)
            teacher = self.parts(self.vector(enc, 'teacher', 'test', a), keys)
            def teacher_calc():
                res = ac_cluster.paired(cand, teacher, y, draws=self.n_clu, workers=self.workers)
                return [res['A_cand'], res['A_comp'], res['d'], *res['interval']]
            stats = self.array('clu_cell', [keys, y, cand, teacher, 'SC4', self.n_clu], teacher_calc)
            self.cell('SC4', enc, a, 'clu', f'teacher_{a}', stats, ranges['clu'][a]['r'], y)
            ap, pick = self.retrieval(enc, 'cstar', a, y)
            for sc, arm, comp in [('SC2', 'frozen', f'frozen_{enc}'), ('SC4', 'teacher', f'teacher_{a}')]:
                other, other_pick = self.retrieval(enc, arm, a, y)
                def ret_calc():
                    res = rt_stats.paired(ap, other, draws=self.n_ret)
                    return [float(ap.mean()), float(other.mean()), res['d'], *res['interval']]
                stats = self.array('ret_cell', [keys, y, ap, other, self.n_ret], ret_calc)
                self.cell(sc, enc, a, 'ret', comp, stats, ranges['ret'][a]['r'] if sc == 'SC4' else None, y,
                          candidate_readout=pick, comparator_readout=other_pick, n_queries=ap.shape[-1])

    def cell(self, sc, enc, a, task, comparator, stats, r, y, **extra):
        cv, tv, d, lo, hi = map(float, stats)
        self.cells.append(dict(sc=sc, encoder=enc, aspect=a, task=task, data='mmart_test',
            candidate=f'cstar_{enc}', comparator=comparator, candidate_value=cv, comparator_value=tv,
            d=d, lo=lo, hi=hi, r=r, outcome=cell_outcome(lo, r),
            n_draws=self.n_clu if task == 'clu' else self.n_ret, seeds=list(SEEDS),
            n_rows=len(y), **({'n_labelled': int((y >= 0).sum())} if task == 'clu' else {}), **extra))


def derive(root=DEFAULT_ROOT, out=DEFAULT_OUT, n_clu=2000, n_ret=10000, sections=None, workers=8, kmeans_threads=8):
    assert 1 <= workers <= 8 and 1 <= n_clu <= 2000 and 1 <= n_ret <= 10000
    sections = list(FULL_SECTIONS if sections is None else sections)
    allowed = set(FULL_SECTIONS) | {f'picks:{e}:{a}' for e in ENCODERS for a in ('cstar', 'frozen')}
    assert sections and len(set(sections)) == len(sections) and set(sections) <= allowed
    work = Computation(root, out, n_clu, n_ret, workers, kmeans_threads)
    start, unavailable, section_seconds = time.monotonic(), [], {}
    doc = {}
    for section in sections:
        section_start = time.monotonic()
        print(f'C13 derive {section}', flush=True)
        try:
            bits = section.split(':')
            if section == 'cut':
                work.derive_cut()
            elif bits[0] == 'ranges':
                work.derive_ranges(bits[1])
            elif bits[0] == 'picks':
                if bits[1] == 'teacher':
                    work.derive_pick('qwen', 'teacher')
                else:
                    for arm in (bits[2],) if len(bits) == 3 else ('cstar', 'frozen'):
                        work.derive_pick(bits[1], arm)
            elif bits[0] == 'cells':
                work.derive_cells(bits[1])
            # outcomes uses only our cells; actual D17 evidence is compare-only.
        except FileNotFoundError as exc:
            unavailable.append(dict(section=section, missing=str(exc.filename)))
            print(f'C13 unavailable {section}: {exc.filename}', flush=True)
        section_seconds[section] = time.monotonic() - section_start
        doc = dict(schema=1, root=str(work.root), sections=sections, complete=False,
                   n_clu=n_clu, n_ret=n_ret, workers=workers, unavailable=unavailable,
                   kmeans_threads=kmeans_threads, section_seconds=section_seconds,
                   operation_seconds=work.operation_seconds, cache_hits=work.cache_hits,
                   cutsets=work.cutsets, ranges=work.ranges, picks=work.picks, cells=work.cells,
                   decisions=decisions(work.cells, partial=True), inputs=work.inputs,
                   source_sha256=source_identity(), seconds=time.monotonic() - start)
        write_json(work.out / 'derived.json', doc)
    doc['complete'] = True
    doc['scope'] = 'full' if full_scope(doc) else 'subset:' + ','.join(sections)
    write_json(work.out / 'derived.json', doc)
    print(f"C13 derived cells={len(work.cells)} seconds={doc['seconds']:.1f} scope={doc['scope']}", flush=True)
    return doc


def full_scope(doc):
    if not doc.get('complete') or doc.get('unavailable') or set(doc.get('sections', [])) != set(FULL_SECTIONS):
        return False
    if (doc.get('n_clu'), doc.get('n_ret')) != (2000, 10000):
        return False
    if set(doc.get('cutsets', {})) != {'test', 'far'}:
        return False
    if set(index_cells(doc.get('cells', []))) != set().union(*(expected_ids(e) for e in ENCODERS)):
        return False
    for e in ENCODERS:
        for task in ('clu', 'ret'):
            if set(doc.get('ranges', {}).get(e, {}).get(task, {})) != set(ASPECTS):
                return False
    wanted = {'teacher'} | {f'{e}:{a}' for e in ENCODERS for a in ('cstar', 'frozen')}
    if set(doc.get('picks', {})) != wanted:
        return False
    for key, entry in doc['picks'].items():
        width = 2048 if key == 'teacher' or key.endswith(':cstar') or key.startswith('qwen:') else 768 if key.startswith('clip_l14:') else 1152
        if set(entry.get('table', {})) != {rt_readout.name(ro) for ro in rt_readout.menu(width)}:
            return False
    return True


class Comparison:
    def __init__(self):
        self.items = []

    def add(self, name, derived, recorded, numeric=False):
        diff = None
        if numeric and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (derived, recorded)):
            d = abs(float(derived) - float(recorded))
            ok = bool(np.isfinite(d) and d <= TOL)
            diff = d if np.isfinite(d) else None
        else:
            ok = type(derived) is type(recorded) and derived == recorded
        self.items.append(dict(name=name, derived=derived, recorded=recorded, abs_diff=diff, pass_=ok))

    def tree(self, name, derived, recorded):
        if isinstance(derived, dict):
            self.add(name + '.keys', sorted(derived), sorted(recorded) if isinstance(recorded, dict) else None)
            for k, v in derived.items():
                self.tree(name + '.' + k, v, recorded.get(k) if isinstance(recorded, dict) else None)
        elif isinstance(derived, list):
            self.add(name + '.length', len(derived), len(recorded) if isinstance(recorded, list) else None)
            for i, v in enumerate(derived):
                self.tree(f'{name}.{i}', v, recorded[i] if isinstance(recorded, list) and i < len(recorded) else None)
        else:
            self.add(name, derived, recorded, numeric=isinstance(derived, float))

    def report(self):
        sections = {}
        for entry in self.items:
            section = entry['name'].split('/')[0]
            stats = sections.setdefault(section, dict(items=0, failed=0, max_abs=0.))
            stats['items'] += 1
            stats['failed'] += not entry['pass_']
            stats['max_abs'] = max(stats['max_abs'], entry['abs_diff'] or 0.)
        return dict(items=self.items, items_compared=len(self.items),
                    failed=sum(not x['pass_'] for x in self.items),
                    max_abs=max((x['abs_diff'] or 0. for x in self.items), default=0.), sections=sections)


def compare_claim_value(value):
    """The round's CLAIM_NONE is JSON null; no other claim maps to null."""
    if value == 'no claim':
        return None
    return {'invalid_derived_claim': None} if value is None else value


def compare_matched_identity(enc, seed, side, npz_sha256, input_sha256):
    """One exact integrity item per export, including its saved-input binding."""
    arm = MATCHED[enc]
    prefix = ('percept', 'core', 'stage2', 'k40') if enc == 'qwen' else ('matched', arm)
    suffix = (*prefix, f'P3__s{seed}.npz')
    mapper = side.get('mapper')
    actual_suffix = '/'.join(Path(mapper).parts[-len(suffix):]) if isinstance(mapper, str) else mapper
    expected = dict(encoder=enc, seed=seed, seed_type='int', view=DEFAULT_VIEW[enc],
                    mapper_suffix='/'.join(suffix), npz_present=True,
                    sidecar_sha256=npz_sha256, input_sha256=npz_sha256)
    actual = dict(encoder=side.get('encoder'), seed=side.get('seed'), seed_type=type(side.get('seed')).__name__,
                  view=side.get('view'), mapper_suffix=actual_suffix, npz_present=npz_sha256 is not None,
                  sidecar_sha256=side.get('sha256'), input_sha256=input_sha256)
    return expected, actual


def compare_code_provenance(doc, derive_commit):
    """Check the requested literal git-file digest without altering derive's schema."""
    import subprocess
    command = ['git', '-C', '/project/CoSiR', 'show', '--end-of-options',
               f'{derive_commit}:src/test/20261102_mmart_test/mt_c13.py']
    error, committed = None, None
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.returncode:
            error = completed.stderr.decode('utf-8', errors='replace').strip()[:1000]
        else:
            # derive's source_identity() is a composite (frozen sources + this file + mp_stores); rebuild it with the
            # committed bytes of this file in place of the current ones (controller fix, 2026-11-03).
            committed_file = hashlib.sha256(completed.stdout).hexdigest()
            committed = hashlib.sha256((frozen.frozen_identity() + committed_file
                                        + file_sha(mp_stores.__file__)).encode()).hexdigest()
    except OSError as exc:
        error = str(exc)
    derived = doc.get('source_sha256')
    return dict(derive_commit=derive_commit, committed_source_sha256=committed,
                derived_source_sha256=derived, current_source_sha256=file_sha(__file__), error=error,
                matches=committed is not None and isinstance(derived, str) and derived == committed)


def record_refusal(doc, report, cells):
    if not full_scope(doc):
        return 'scope is a subset or lacks full protocol coverage/draw counts'
    if doc.get('kmeans_threads') != 8 or isinstance(doc.get('kmeans_threads'), bool):
        return 'derived.json must specify kmeans_threads 8'
    if report.get('code_provenance', {}).get('matches') is not True:
        return 'derive source hash does not match --derive-commit'
    if report['failed']:
        return 'one or more comparisons failed'
    if len(cells) != 60 or any(c.get('n_draws') != (2000 if c.get('task') == 'clu' else 10000) for c in cells):
        return 'round cells do not contain all 60 cells at 2000/10000 draws'
    return None


def compare(root=DEFAULT_ROOT, records=None, out=DEFAULT_OUT, write_record=False, derive_commit='f093fb7'):
    """Sole result boundary: read exact bytes once, adapt layouts here only."""
    root, out = Path(root).resolve(), check_output(out)
    records = ROUND / 'records' if records is None else Path(records)
    doc = read_json(out / 'derived.json')
    result = Comparison()
    result.add('integrity/complete', doc.get('complete'), True)
    result.add('integrity/root', doc.get('root'), str(root))
    code = compare_code_provenance(doc, derive_commit)
    result.add('integrity/derive_source_commit', code['derived_source_sha256'],
               code['committed_source_sha256'] if code['error'] is None else {'git_error': code['error']})
    result.add('integrity/unavailable', doc.get('unavailable'), [])
    # Export checksums were checked during derive; check the same bytes still exist.
    for path, digest in doc.get('inputs', {}).items():
        result.add('inputs/' + path, digest, file_sha(path) if Path(path).exists() else None)
    def load(path):
        path = Path(path)
        return read_json(path) if path.exists() else {}
    for enc in ENCODERS:
        for seed in SEEDS:
            npz = root / 'exports' / 'new' / enc / 'matched' / f'probs__s{seed}.npz'
            try:
                side = load(npz.with_suffix('.json'))
                if not isinstance(side, dict):
                    side = {}
            except (ValueError, OSError):
                side = {}
            digest = file_sha(npz) if npz.exists() else None
            expected_identity, recorded_identity = compare_matched_identity(
                enc, seed, side, digest, doc.get('inputs', {}).get(str(npz)))
            result.add(f'integrity/matched.{enc}.s{seed}', expected_identity, recorded_identity)
    status = load(records / 'status.json')
    result.add('integrity/status_present', bool(status), True)
    stage2, hashes = {}, {}
    for relative in STAGE2_FILES:
        path = root / relative
        if path.exists():
            data = path.read_bytes()
            stage2[relative] = json.loads(data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
        else:
            stage2[relative] = [] if 'cells_' in relative else {}
            result.add('integrity/' + relative, 'present', 'missing')
    round_cells = stage2['stage2/cells_clu.json'] + stage2['stage2/cells_ret.json']
    theirs = index_cells(round_cells)
    outcomes = stage2['stage2/outcomes.json']
    outcome_cells = index_cells(outcomes.get('cells', []))
    # The adapters compare both committed stage1 records and their output copies.
    if doc['cutsets']:
        ref = load(records / 'cutsets.json')
        for name, entry in doc['cutsets'].items():
            projection = {k: v for k, v in entry.items() if k != 'keys'}
            # The round serializes the kept SET in lexical order for this
            # digest. Its scoring order (and ours) remains SHA256 order.
            projection['sha256'] = keys_sha(sorted(entry['keys']))
            result.tree('cut/' + name, projection, ref.get(name))
    for enc, tasks in doc['ranges'].items():
        record = load(records / f'stage1_{enc}.json')
        for task, aspects in tasks.items():
            for location, ref in [('record', record.get(f'{task}_ranges', {})),
                                  ('stage1', load(root / 'stage1' / f'{task}_ranges_{enc}.json'))]:
                name = f'ranges:{enc}/{task}.{location}'
                result.add(name + '.n_val', 5769, ref.get('n_val'))
                for a, values in aspects.items():
                    result.add(name + '.' + a + '.r', values['r'], ref.get('ranges', {}).get(a), True)
                    result.tree(name + '.' + a + '.per_seed', values['per_seed'], ref.get('per_seed', {}).get(a))
    for key, entry in doc['picks'].items():
        enc, arm = ('teacher', 'teacher') if key == 'teacher' else key.split(':')
        record = load(records / f'stage1_{enc}.json')
        for location, ref in [('record', record if enc == 'teacher' else record.get('picks', {})),
                              ('stage1', load(root / 'stage1' / f'picks_{enc}.json'))]:
            result.add(f'picks:{key}/{location}.n_val', 5769, ref.get('n_val'))
            result.tree(f'picks:{key}/{location}', entry, ref.get('arms', {}).get(arm))
    encoders = tuple(e for e in ENCODERS if f'cells:{e}' in doc['sections'])
    for enc in encoders:
        result.add(f'cells:{enc}/identities', sorted(expected_ids(enc)), sorted(cell_id(c) for c in doc['cells'] if c['encoder'] == enc))
    for c in doc['cells']:
        name = f"cells:{c['encoder']}/" + '/'.join(cell_id(c))
        for location, index in [('raw', theirs), ('outcomes', outcome_cells)]:
            ref = index.get(cell_id(c), {})
            for k, v in c.items():
                if k == 'outcome' and location == 'raw':
                    continue
                result.add(name + f'.{location}.' + k, v, ref.get(k), isinstance(v, float))
    if set(doc['sections']) == set(FULL_SECTIONS):
        result.add('integrity/all_cell_ids', sorted(set().union(*(expected_ids(e) for e in ENCODERS))), sorted(theirs))
        result.add('integrity/all_outcome_ids', sorted(theirs), sorted(outcome_cells))
    if status:
        result.tree('outcomes/status', status, outcomes.get('status'))
        expected = decisions(doc['cells'], status, encoders=encoders)
        for enc, verdict in expected['verdicts'].items():
            ref = outcomes.get('verdicts', {}).get(enc, {})
            result.add(f'outcomes:{enc}/verdict', verdict, ref.get('verdict'))
            result.add(f'outcomes:{enc}/void_verdict', bool(voids(status, enc)), ref.get('verdict') == 'inconclusive')
        for c in expected['cells']:
            ref = outcome_cells.get(cell_id(c), {})
            name = f"outcomes:{c['encoder']}/" + '/'.join(cell_id(c))
            # Round status vocabulary: a live cell is "decided".
            result.add(name + '.status', 'decided' if c['status'] == 'ok' else c['status'], ref.get('status'))
            result.add(name + '.void', bool(c['void']), bool(ref.get('void')))
        for enc, claims in expected['sc4'].items():
            recorded_claims = outcomes.get('sc4', {}).get(enc, [])
            refs = {(c['aspect'], c['task']): c for c in recorded_claims}
            result.add(f'claims:{enc}/unique_identities', len(refs), len(recorded_claims))
            result.add(f'claims:{enc}/identities', sorted((c['aspect'], c['task']) for c in claims), sorted(refs))
            for c in claims:
                ref = refs.get((c['aspect'], c['task']), {})
                for k, v in c.items():
                    if k == 'status' and v == 'ok':
                        v = 'decided'
                    if k == 'claim':
                        v = compare_claim_value(v)
                    # Missing is different from an explicit CLAIM_NONE/null.
                    value = (bool(ref.get('void')) if k == 'void' else
                             ref[k] if k in ref else {'missing_field': k})
                    result.add(f"claims:{enc}/{c['aspect']}.{c['task']}.{k}", v, value)
    payload = result.report()
    payload.update(scope='full' if full_scope(doc) else doc.get('scope', 'subset'),
                   draws=dict(clu=doc['n_clu'], ret=doc['n_ret']), sha256=hashes,
                   kmeans_threads=doc.get('kmeans_threads'),
                   code_provenance=code,
                   derive_source_sha256=doc.get('source_sha256'), compare_source_sha256=source_identity(),
                   pass_=payload['failed'] == 0)
    refusal = record_refusal(doc, payload, round_cells) if write_record else None
    payload['record'] = 'refused: ' + refusal if refusal else 'not requested' if not write_record else 'written'
    # Public JSON uses the requested literal "pass" key.
    for entry in payload['items']:
        entry['pass'] = entry.pop('pass_')
    payload['pass'] = payload.pop('pass_')
    write_json(out / 'compare.json', payload)
    if write_record and not refusal:
        # This is the sole explicitly requested write outside our output directories.
        records.mkdir(parents=True, exist_ok=True)
        (records / 'c13.json').write_text(json.dumps({'pass': True, 'sha256': hashes}, indent=1) + '\n')
    print(f"C13 {'PASS' if payload['pass'] else 'FAIL'} items={payload['items_compared']} failed={payload['failed']} max_abs={payload['max_abs']:.12g} scope={'full' if payload['scope'] == 'full' else 'subset'}", flush=True)
    if refusal:
        raise ValueError('refusing --write-record: ' + refusal)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('derive', 'compare'):
        p = sub.add_parser(name)
        p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
        p.add_argument('--out', type=Path, default=DEFAULT_OUT)
        if name == 'derive':
            p.add_argument('--draws-clu', type=int, default=2000)
            p.add_argument('--draws-ret', type=int, default=10000)
            p.add_argument('--sections', nargs='+')
            p.add_argument('--workers', type=int, default=8)
            p.add_argument('--kmeans-threads', type=int, default=8)
        else:
            p.add_argument('--records', type=Path, default=ROUND / 'records')
            p.add_argument('--write-record', action='store_true')
            p.add_argument('--derive-commit', default='f093fb7')
    args = parser.parse_args()
    with threadpool_limits(limits=8):
        if args.command == 'derive':
            derive(args.root, args.out, args.draws_clu, args.draws_ret, args.sections, args.workers,
                   kmeans_threads=args.kmeans_threads)
        else:
            result = compare(args.root, args.records, args.out, args.write_record, derive_commit=args.derive_commit)
            return 0 if result['pass'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
