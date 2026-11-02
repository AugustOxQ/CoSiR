"""Tests of mt_ret (ticket 05): synthetic arrays, mt_data and mt_export replaced by stubs. Verdict code: each test
fails when the logic it guards is broken (parity halves, guard, tie, teacher ranges, fit on validation only)."""
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mt_paths as MP  # noqa: E402
import mt_ret  # noqa: E402
import rt_metrics  # noqa: E402
import rt_readout  # noqa: E402

ASPECTS = ('emotion', 'style', 'genre')
D = 16


def _order(keys):
    return sorted(keys, key=lambda k: hashlib.sha256(k.encode('utf-8')).hexdigest())


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class World:
    """A tiny MMArt: 'val' and 'test' keys in sha256 order, labels per aspect (with -1), and every view and export as a
    class-clustered cloud whose noise depends on (view, aspect, seed)."""

    def __init__(self, n_val=200, n_test=260, seed=7):
        rng = np.random.default_rng(seed)
        self.key_sets = {}
        self.lab = {}
        self.centers = {a: rng.normal(size=(4, D)) for a in ASPECTS}
        self.offset = rng.normal(size=D) * 2          # a shift that centring removes
        for name, n in (('val', n_val), ('test', n_test)):
            keys = _order([f'Style_{name}/artist_{name}_{i}.jpg' for i in range(n)])
            self.key_sets[name] = keys
            for a in ASPECTS:
                y = rng.integers(0, 4, size=n).astype(np.int64)
                y[rng.random(n) < 0.08] = -1
                self.lab[(name, a)] = dict(zip(keys, y))
        self.calls = []
        self._cache = {}

    def _cloud(self, tag, name, aspect, noise):
        """Rows of `name`'s keys: the aspect's class centre plus noise seeded by tag."""
        keys = self.key_sets[name]
        h = int(hashlib.sha256(repr((tag, name, aspect)).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        y = np.array([self.lab[(name, aspect)][k] for k in keys])
        c = self.centers[aspect][np.where(y < 0, 0, y)]
        return _unit(c + noise * rng.normal(size=(len(keys), D)) + self.offset)

    def rows(self, tag, name, aspect, noise):
        key = (tag, name, aspect, noise)
        if key not in self._cache:
            self._cache[key] = self._cloud(tag, name, aspect, noise)
        return self._cache[key]

    def which(self, keys):
        keys = list(keys)
        for name, ks in self.key_sets.items():
            if keys == ks:
                return name
        raise AssertionError('keys are no known set')

    # ---- the stub modules
    def mt_data(self):
        m = types.ModuleType('mt_data')
        m.order = _order

        def keys(name):
            self.calls.append(('keys', name))
            return list(self.key_sets['val' if name == 'val' else 'test'])

        def labels(ks, aspect):
            name = self.which(ks)
            self.calls.append(('labels', name, aspect))
            return np.array([self.lab[(name, aspect)][k] for k in ks], dtype=np.int64)

        def features(view, ks):
            name = self.which(ks)
            self.calls.append(('features', view, name))
            teacher = {v: a for a, v in MP.TEACHER_VIEW.items()}
            if view in teacher:
                return self.rows(view, name, teacher[view], 0.9)
            return self.rows(view, name, 'emotion', 1.6)

        m.keys, m.labels, m.features = keys, labels, features
        return m

    def mt_export(self):
        m = types.ModuleType('mt_export')

        def load(keyset, enc, aspect, seed, ks):
            name = self.which(ks)
            self.calls.append(('export', keyset, enc, aspect, seed, name))
            assert (keyset == 'val') == (name == 'val'), (keyset, name)
            return self.rows((enc, 'cstar', seed), name, aspect, 0.6 + 0.05 * seed).astype(np.float16)

        m.load = load
        return m


@pytest.fixture
def world(monkeypatch, tmp_path):
    w = World()
    monkeypatch.setitem(sys.modules, 'mt_data', w.mt_data())
    monkeypatch.setitem(sys.modules, 'mt_export', w.mt_export())
    monkeypatch.setattr(MP, 'OUT', tmp_path / 'out')
    return w


# ============================================================================================ order and parity halves
def test_check_order_accepts_sha256_and_rejects_sorted_keys(world):
    keys = world.key_sets['val']
    mt_ret.check_order(keys)
    with pytest.raises(AssertionError):
        mt_ret.check_order(sorted(keys))                   # r12's sorted-key order would group labels in the halves
    with pytest.raises(AssertionError):
        mt_ret.check_order(keys[::-1])


def test_crossfit_halves_follow_position_parity_in_the_given_order(world):
    """R0's cross-fit mAP is the mean of the two halves' mAP, halves = even / odd POSITION in the rows as given."""
    x = world.rows('plain', 'val', 'style', 1.6)
    y = np.array([world.lab[('val', 'style')][k] for k in world.key_sets['val']], dtype=np.int64)
    ranges = {a: 0.01 for a in ASPECTS}
    labels = {a: np.array([world.lab[('val', a)][k] for k in world.key_sets['val']], dtype=np.int64) for a in ASPECTS}
    doc = mt_ret.readout_doc({(a, None): x for a in ASPECTS}, labels, ranges, workers=1)
    by_parity = np.mean([rt_metrics.retrieval(x[idx], y[idx], workers=1)['ap'].mean()
                         for idx in (np.arange(0, len(y), 2), np.arange(1, len(y), 2))])
    assert doc['table']['none+k0']['map']['style'] == pytest.approx(by_parity, abs=1e-12)
    # the same rows in another order give another number: the halves are positions, not keys
    perm = np.random.default_rng(0).permutation(len(y))
    labels2 = {a: v[perm] for a, v in labels.items()}
    doc2 = mt_ret.readout_doc({(a, None): x[perm] for a in ASPECTS}, labels2, ranges, workers=1)
    assert doc2['table']['none+k0']['map']['style'] != pytest.approx(by_parity, abs=1e-6)
    # and the halves of the SORTED keys would differ from the sha256 order's (so passing sorted keys would change picks)
    srt = np.argsort(np.array(world.key_sets['val']))
    by_sorted = np.mean([rt_metrics.retrieval(x[srt][idx], y[srt][idx], workers=1)['ap'].mean()
                         for idx in (np.arange(0, len(y), 2), np.arange(1, len(y), 2))])
    assert by_sorted != pytest.approx(by_parity, abs=1e-6)


# ============================================================================================ guard, tie, score
def _fake_crossfit(monkeypatch, table):
    """Replace r11's cross-fit by fixed mAPs {read-out name: {aspect: mAP}} (every other read-out: R0's)."""
    def fake(xs, labels, ros, workers=8):
        base = table['none+k0']
        return {ro: dict(table.get(rt_readout.name(ro), base)) for ro in ros}
    monkeypatch.setattr(rt_readout, 'crossfit_all', fake)


R = 0.01
TABLE = {
    'none+k0': dict(emotion=.5, style=.5, genre=.5),
    'center+k10': dict(emotion=.52, style=.51, genre=.5),                 # score (2 + 1 + 0) / 3 = 1.0
    'whiten16+k10': dict(emotion=.6, style=.6, genre=.489),               # genre 1.1 r below R0: guard drops it
    'center+k3': dict(emotion=.52 - 0.00002, style=.51, genre=.5),        # 0.00067 below the best: a tie, k3 < k10
    'none+k1': dict(emotion=.5199, style=.5, genre=.5),                   # 0.0033 below the best, no tie... (see below)
    'center+k1': dict(emotion=.52, style=.51, genre=.49),                 # exactly r below R0 in genre: kept
}


def _doc(xs_world, ranges, monkeypatch, table):
    _fake_crossfit(monkeypatch, table)
    x = xs_world.rows('plain', 'val', 'style', 1.6)
    lab = {a: np.array([xs_world.lab[('val', a)][k] for k in xs_world.key_sets['val']], dtype=np.int64) for a in ASPECTS}
    return mt_ret.readout_doc({(a, None): x for a in ASPECTS}, lab, ranges, workers=1)


def test_score_is_mean_gain_in_units_of_range(world, monkeypatch):
    doc = _doc(world, {a: R for a in ASPECTS}, monkeypatch, TABLE)
    assert doc['table']['center+k10']['score'] == pytest.approx(1.0, abs=1e-12)
    assert doc['table']['whiten16+k10']['score'] == pytest.approx((10 + 10 - 1.1) / 3, abs=1e-9)
    big = _doc(world, {'emotion': 0.02, 'style': 0.01, 'genre': 0.01}, monkeypatch, TABLE)
    assert big['table']['center+k10']['score'] == pytest.approx((1 + 1 + 0) / 3, abs=1e-12)    # each aspect its own r


def test_guard_drops_a_readout_more_than_r_below_r0_and_keeps_exactly_r(world, monkeypatch):
    doc = _doc(world, {a: R for a in ASPECTS}, monkeypatch, TABLE)
    assert doc['table']['whiten16+k10']['kept'] is False
    assert doc['table']['center+k1']['kept'] is True                     # exactly r below: kept
    assert doc['pick'] != 'whiten16+k10'                                  # best score, but guarded out


def test_tie_rule_takes_the_first_in_tie_order_within_0001(world, monkeypatch):
    doc = _doc(world, {a: R for a in ASPECTS}, monkeypatch, TABLE)
    assert doc['table']['center+k3']['score'] >= doc['table']['center+k10']['score'] - 0.001
    assert doc['pick'] == 'center+k3'                                     # tied with center+k10; smaller k first
    t2 = dict(TABLE, **{'center+k3': dict(emotion=.51, style=.51, genre=.5)})    # 0.0033 below: no tie
    assert _doc(world, {a: R for a in ASPECTS}, monkeypatch, t2)['pick'] == 'center+k10'


def test_pick_depends_on_the_ranges_given(world, monkeypatch):
    t = {'none+k0': TABLE['none+k0'], 'whiten16+k10': TABLE['whiten16+k10'], 'center+k10': TABLE['center+k10']}
    small = _doc(world, {a: 0.005 for a in ASPECTS}, monkeypatch, t)['pick']
    large = _doc(world, {a: 0.02 for a in ASPECTS}, monkeypatch, t)['pick']
    assert (small, large) == ('center+k10', 'whiten16+k10')


# ============================================================================================ ranges
def test_ranges_are_max_minus_min_of_validation_r0_map_labelled_rows(world):
    res = mt_ret.validation_ranges('siglip2', workers=1)
    keys = world.key_sets['val']
    for a in ASPECTS:
        y = np.array([world.lab[('val', a)][k] for k in keys], dtype=np.int64)
        per_seed = [float(rt_metrics.retrieval(
            mt_ret.unit(world.rows(('siglip2', 'cstar', s), 'val', a, 0.6 + 0.05 * s).astype(np.float16)), y,
            workers=1)['ap'].mean()) for s in range(5)]
        assert [res['per_seed'][a][s] for s in range(5)] == pytest.approx(per_seed, abs=1e-12)
        assert res['ranges'][a] == pytest.approx(max(per_seed) - min(per_seed), abs=1e-12)
        assert res['ranges'][a] > 0
        assert res['labelled'][a] == int((y >= 0).sum()) < len(y)
    assert all(c[1] == 'val' and c[5] == 'val' for c in world.calls if c[0] == 'export')


def test_write_ranges_file_and_reader(world):
    path = mt_ret.write_ranges('siglip2', mt_ret.validation_ranges('siglip2', workers=1))
    assert path == MP.OUT / 'stage1/ret_ranges_siglip2.json'
    rec = json.loads(path.read_text())
    assert rec['encoder'] == 'siglip2' and set(rec['ranges']) == set(ASPECTS)
    assert mt_ret.read_ranges('siglip2') == rec['ranges']
    path.write_text(json.dumps(dict(rec, encoder='qwen')))
    with pytest.raises(AssertionError):
        mt_ret.read_ranges('siglip2')                       # another encoder's file is refused


def test_ranges_refuse_a_zero_range():
    with pytest.raises(AssertionError):
        mt_ret.own_ranges({a: {s: 0.3 for s in range(5)} for a in ASPECTS})
    with pytest.raises(AssertionError):
        mt_ret.own_ranges({a: {s: 0.3 + s for s in range(4)} for a in ASPECTS})


# ============================================================================================ picks
def test_picks_read_validation_only_and_use_the_encoders_own_ranges(world, monkeypatch):
    mt_ret.write_ranges('siglip2', dict(ranges={a: 0.005 for a in ASPECTS}, per_seed={}, labelled={}, n_val=0, sources={}))
    t = {'none+k0': TABLE['none+k0'], 'whiten16+k10': TABLE['whiten16+k10'], 'center+k10': TABLE['center+k10']}
    _fake_crossfit(monkeypatch, t)
    rec = mt_ret.pick_encoder('siglip2', workers=1)
    assert set(rec['arms']) == {'cstar', 'frozen'}                       # no teacher in an encoder's file
    for arm in ('cstar', 'frozen'):
        assert rec['arms'][arm]['ranges'] == {a: 0.005 for a in ASPECTS}
        assert rec['arms'][arm]['pick'] == 'center+k10'
    assert all(c[0] != 'labels' or c[1] == 'val' for c in world.calls)
    assert all(c[0] != 'export' or c[1] == 'val' for c in world.calls)
    assert {c[2] for c in world.calls if c[0] == 'features'} == {'val'}
    assert {c[1] for c in world.calls if c[0] == 'features'} == {'siglip2'}


def test_teacher_pick_uses_qwens_ranges_for_every_encoder(world, monkeypatch):
    mt_ret.write_ranges('qwen', dict(ranges={a: 0.02 for a in ASPECTS}, per_seed={}, labelled={}, n_val=0, sources={}))
    mt_ret.write_ranges('siglip2', dict(ranges={a: 0.005 for a in ASPECTS}, per_seed={}, labelled={}, n_val=0, sources={}))
    t = {'none+k0': TABLE['none+k0'], 'whiten16+k10': TABLE['whiten16+k10'], 'center+k10': TABLE['center+k10']}
    _fake_crossfit(monkeypatch, t)
    teacher = mt_ret.pick_teacher(workers=1)
    own = mt_ret.pick_encoder('siglip2', workers=1)
    assert teacher['arms']['teacher']['ranges'] == {a: 0.02 for a in ASPECTS}
    assert teacher['arms']['teacher']['pick'] == 'whiten16+k10'          # Qwen's wide ranges keep it
    assert own['arms']['cstar']['pick'] == 'center+k10'                  # siglip2's narrow ones drop it
    assert {c[1] for c in world.calls if c[0] == 'features'} >= set(MP.TEACHER_VIEW.values())


def test_teacher_pick_refuses_without_qwens_ranges(world):
    with pytest.raises(FileNotFoundError):
        mt_ret.pick_teacher(workers=1)


def test_pick_files_and_committed_record_shapes(world, monkeypatch):
    mt_ret.write_ranges('qwen', dict(ranges={a: 0.02 for a in ASPECTS}, per_seed={}, labelled={}, n_val=0, sources={}))
    _fake_crossfit(monkeypatch, TABLE)
    p = mt_ret.write_picks('qwen', mt_ret.pick_encoder('qwen', workers=1))
    q = mt_ret.write_teacher_picks(mt_ret.pick_teacher(workers=1))
    assert p == MP.OUT / 'stage1/picks_qwen.json' and q == MP.OUT / 'stage1/picks_teacher.json'
    assert json.loads(p.read_text())['encoder'] == 'qwen'
    assert set(json.loads(q.read_text())['arms']) == {'teacher'}


# ============================================================================================ the fit
def test_map_is_fitted_on_all_validation_rows_and_applied_to_the_eval_rows(world):
    """Centering at a pick: mu comes from the validation rows only; the result equals the manual centred retrieval."""
    xv = world.rows('plain', 'val', 'style', 1.6)
    xe = world.rows('plain', 'test', 'style', 1.6)
    y = np.array([world.lab[('test', 'style')][k] for k in world.key_sets['test']], dtype=np.int64)
    res = mt_ret.arm_ap(lambda s: xv, lambda s: xe, 'center+k3', y, seeded=False, workers=1)
    z = xe.astype(np.float64) - xv.astype(np.float64).mean(axis=0)
    z = (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)
    want = rt_metrics.retrieval(z, y, qe_k=3, workers=1)
    assert np.array_equal(res['rows'], want['rows'])
    assert np.allclose(res['ap'], want['ap'], atol=1e-12, rtol=0)
    z_own = xe.astype(np.float64) - xe.astype(np.float64).mean(axis=0)             # fitting on the eval rows differs
    z_own = (z_own / np.linalg.norm(z_own, axis=1, keepdims=True)).astype(np.float32)
    assert not np.allclose(rt_metrics.retrieval(z_own, y, qe_k=3, workers=1)['ap'], want['ap'], atol=1e-6)


def test_arm_ap_seeded_gives_five_rows_and_checks_the_seed_set(world):
    y = np.array([world.lab[('test', 'genre')][k] for k in world.key_sets['test']], dtype=np.int64)
    res = mt_ret.arm_ap(lambda s: world.rows(('e', 'cstar', s), 'val', 'genre', 0.7),
                        lambda s: world.rows(('e', 'cstar', s), 'test', 'genre', 0.7), 'none+k0', y, seeded=True, workers=1)
    assert res['ap'].shape[0] == 5 and len(res['per_seed']) == 5


# ============================================================================================ cells
def _record_docs(enc='siglip2'):
    ranges = {'emotion': 0.011, 'style': 0.022, 'genre': 0.033}
    picks = dict(encoder=enc, arms={'cstar': dict(pick='center+k3'), 'frozen': dict(pick='none+k0')})
    teacher = dict(arms={'teacher': dict(pick='center+k10')})
    return ranges, picks, teacher


def test_score_encoder_cells_sc2_and_sc4(world):
    ranges, picks, teacher = _record_docs()
    cells = mt_ret.score_encoder('siglip2', keyset='test', n_draws=60, workers=1, ranges=ranges, picks=picks,
                                 teacher=teacher)
    assert [(c['sc'], c['aspect']) for c in cells] == [(sc, a) for a in ASPECTS for sc in ('SC2', 'SC4')]
    keys = world.key_sets['test']
    for c in cells:
        a = c['aspect']
        assert c['encoder'] == 'siglip2' and c['task'] == 'ret' and c['data'] == 'mmart_test' and c['n_draws'] == 60
        assert c['seeds'] == [0, 1, 2, 3, 4] and c['n_rows'] == len(keys)
        if c['sc'] == 'SC2':
            assert c['r'] is None and c['comparator'] == 'frozen_siglip2' and c['candidate'] == 'cstar_siglip2'
            assert (c['candidate_readout'], c['comparator_readout']) == ('center+k3', 'none+k0')
        else:
            assert c['r'] == ranges[a] and c['comparator'] == f'teacher_{a}' and c['comparator_readout'] == 'center+k10'
            assert 'seed range' in c['r_source']
        y = np.array([world.lab[('test', a)][k] for k in keys], dtype=np.int64)
        assert np.isclose(c['d'], c['candidate_value'] - c['comparator_value'], atol=1e-12)
    # the candidate value is the mean over five seeds of the manual fit-on-validation, apply-to-test mAP at center+k3
    a = 'style'
    y = np.array([world.lab[('test', a)][k] for k in keys], dtype=np.int64)
    maps = []
    for s in range(5):
        xv = mt_ret.unit(world.rows(('siglip2', 'cstar', s), 'val', a, 0.6 + 0.05 * s).astype(np.float16))
        xe = mt_ret.unit(world.rows(('siglip2', 'cstar', s), 'test', a, 0.6 + 0.05 * s).astype(np.float16))
        z = xe.astype(np.float64) - xv.astype(np.float64).mean(axis=0)
        z = (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)
        maps.append(rt_metrics.retrieval(z, y, qe_k=3, workers=1)['ap'].mean())
    cell = next(c for c in cells if c['aspect'] == a and c['sc'] == 'SC2')
    assert cell['candidate_value'] == pytest.approx(np.mean(maps), abs=1e-12)


def test_score_encoder_reads_no_validation_labels_and_fits_on_validation_features(world):
    ranges, picks, teacher = _record_docs()
    world.calls.clear()
    mt_ret.score_encoder('siglip2', keyset='test', n_draws=20, workers=1, ranges=ranges, picks=picks, teacher=teacher)
    assert not [c for c in world.calls if c[0] == 'labels' and c[1] == 'val']          # picks/ranges come from records
    exp = [c for c in world.calls if c[0] == 'export']
    assert {(c[1], c[5]) for c in exp} == {('val', 'val'), ('new', 'test')}            # test rows read the 'new' exports
    feats = {(c[1], c[2]) for c in world.calls if c[0] == 'features'}
    assert feats >= {('siglip2', 'val'), ('siglip2', 'test')} | {(v, s) for v in MP.TEACHER_VIEW.values()
                                                                   for s in ('val', 'test')}


def test_score_encoder_sc_selection_and_keyset_mapping(world):
    ranges, picks, teacher = _record_docs()
    only2 = mt_ret.score_encoder('siglip2', n_draws=20, workers=1, ranges=ranges, picks=picks, teacher=teacher,
                                 scs=('SC2',))
    assert {c['sc'] for c in only2} == {'SC2'} and len(only2) == 3
    assert [mt_ret.export_keyset(k) for k in ('val', 'test', 'far', 'unseen_artist', 'new', 'shared_held')] == \
        ['val', 'new', 'new', 'new', 'new', 'shared_held']
    assert mt_ret.cell_data('test') == 'mmart_test' and mt_ret.cell_data('far') == 'mmart_far'
    with pytest.raises(AssertionError):
        mt_ret.score_encoder('siglip2', n_draws=20, workers=1, ranges=ranges, picks=picks, teacher=teacher,
                             scs=('SC1',))


def test_score_encoder_reads_the_committed_records_by_name(world, monkeypatch):
    ranges, picks, teacher = _record_docs()
    seen = []

    def fake(name):
        seen.append(name)
        return {'stage1_siglip2.json': dict(ret_ranges=dict(encoder='siglip2', ranges=ranges), picks=picks),
                'stage1_teacher.json': dict(picks=teacher)}[name]
    monkeypatch.setattr(MP, 'read_committed_record', fake)
    cells = mt_ret.score_encoder('siglip2', n_draws=20, workers=1)
    assert sorted(seen) == ['stage1_siglip2.json', 'stage1_teacher.json'] and len(cells) == 6
    assert {c['r'] for c in cells if c['sc'] == 'SC4'} == set(ranges.values())


def test_make_cell_asserts_the_bars():
    kw = dict(encoder='qwen', aspect='style', candidate='c', comparator='t', candidate_value=.3, comparator_value=.2,
              d=.1, lo=.05, hi=.15, r_source='x', n_draws=10, n_rows=5)
    mt_ret.make_cell('SC2', r=None, **kw)
    mt_ret.make_cell('SC4', r=0.01, **kw)
    with pytest.raises(AssertionError):
        mt_ret.make_cell('SC2', r=0.01, **kw)                # SC2 is WIN-only
    with pytest.raises(AssertionError):
        mt_ret.make_cell('SC4', r=None, **kw)                # SC4 needs the unit
    with pytest.raises(AssertionError):
        mt_ret.make_cell('SC4', r=0.0, **kw)
    with pytest.raises(AssertionError):
        mt_ret.make_cell('SC4', r=0.01, **dict(kw, lo=float('nan')))


# ============================================================================================ replay comparison
def test_replay_items_fail_beyond_1e9():
    ok = mt_ret.replay_result([mt_ret.replay_item('a', 1.0, 1.0 + 5e-10)])
    bad = mt_ret.replay_result([mt_ret.replay_item('a', 1.0, 1.0 + 2e-9)])
    assert ok['passed'] and not bad['passed'] and bad['max_abs_diff'] == pytest.approx(2e-9, rel=1e-3)
