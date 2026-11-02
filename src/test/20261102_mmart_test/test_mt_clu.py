"""Tests of mt_clu (r15 ticket 04): the clustering scorer on synthetic arrays and the MMArt glue on stub data modules.

Each expected value is a hand-written bootstrap that follows the spec's draw order with `default_rng(0)` (r14 §3.4); a
scorer that changes the draws, the seed handling, the reading choice or the masking gives other numbers.
"""
import hashlib
import json
import sys
import types

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score as ami

import mt_clu as M
import mt_paths

N, K, B = 240, 6, 60
SEEDS = list(range(5))


# ------------------------------------------------------------------------------------------------ synthetic partitions
def _labels(rng, masked=0, n=N):
    y = rng.integers(0, K, n)
    y[:masked] = -1
    return y


def _noisy(rng, y, noise, sets=3):
    base = np.where(y < 0, rng.integers(0, K, len(y)), y)
    out = []
    for _ in range(sets):
        p = base.copy()
        move = rng.random(len(y)) < noise
        p[move] = rng.integers(0, K, int(move.sum()))
        out.append(p)
    return np.array(out, dtype=np.int64)


def _fixed(rng, y, noise, readings=('r20',), seeds=SEEDS):
    return {s: {r: _noisy(rng, y, noise, 1 if r == 'own' else 3) for r in readings} for s in seeds}


def _score(fixed_list, y, rows, reading):
    """Mean over drawn seeds of the mean over a set's partitions of the AMI on `rows`, labels < 0 masked AFTER indexing."""
    yy = y[rows]
    m = yy >= 0
    return float(np.mean([np.mean([ami(yy[m], p[rows][m]) for p in fs[reading]]) for fs in fixed_list]))


def _interval(diffs):
    return np.percentile(diffs, [2.5, 97.5]).tolist()


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(123)
    y = _labels(rng, masked=17)
    return dict(y=y, cand=_fixed(rng, y, 0.45), frozen=_fixed(rng, y, 0.6, seeds=[0]),
                matched=_fixed(rng, y, 0.5, ('r20', 'own')), teacher=_fixed(rng, y, 0.55, seeds=[0]))


def _cells(d, scs=('SC1', 'SC3', 'SC4'), r=0.0123, **kw):
    return M.score_aspect('siglip2', 'genre', d['y'], d['cand'], frozen=d['frozen'], matched=d['matched'],
                          teacher=d['teacher'], scs=scs, r=r, n_draws=B, workers=1, matched_name='M2', **kw)


def _by_sc(cells):
    return {c['sc']: c for c in cells}


def test_statistics_are_r14s_own_functions():
    """The replay proves the reuse; here the identity: no copy of the statistic code lives in this round."""
    import bs_artclu
    import rb_score
    assert M.frozen_margin is bs_artclu.frozen_margin and M.matched_margin is bs_artclu.matched_margin
    assert M.teacher_margin is bs_artclu.teacher_margin and M.seed_range is bs_artclu.seed_range
    assert M.validation_ami is bs_artclu.validation_ami and M.rb_score is rb_score


# ------------------------------------------------------------------------------------------------ SC1
def test_sc1_frozen_arm_draws_no_seeds(data):
    """Per draw: rows, then five seeds for X-c* only. A frozen image that drew seeds too would shift the stream."""
    y = data['y']
    c = _by_sc(_cells(data))['SC1']
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(B):
        rows = rng.integers(0, N, size=N)
        s = rng.choice(SEEDS, size=5, replace=True)
        diffs.append(_score([data['cand'][int(i)] for i in s], y, rows, 'r20') - _score([data['frozen'][0]], y, rows, 'r20'))
    assert [c['lo'], c['hi']] == pytest.approx(_interval(diffs), abs=1e-12, rel=0)
    rows0 = np.arange(N)
    point = _score([data['cand'][s] for s in SEEDS], y, rows0, 'r20') - _score([data['frozen'][0]], y, rows0, 'r20')
    assert c['d'] == pytest.approx(point, abs=1e-12, rel=0)
    assert c['candidate_value'] - c['comparator_value'] == pytest.approx(c['d'], abs=1e-12)


def test_unlabelled_rows_are_masked_after_indexing_not_dropped_before(data):
    """The fixture has 17 unlabelled rows. The hand bootstrap draws over all N rows (the unlabelled ones included) and masks
    after indexing; dropping them first changes n, the draws and the intervals."""
    y = data['y']
    keep = y >= 0
    assert (~keep).sum() == 17
    dropped = M.score_aspect('siglip2', 'genre', y[keep], {s: {'r20': v['r20'][:, keep]} for s, v in data['cand'].items()},
                             frozen={0: {'r20': data['frozen'][0]['r20'][:, keep]}}, scs=('SC1',), n_draws=B, workers=1)
    full = _by_sc(_cells(data, scs=('SC1',)))['SC1']
    assert full['n_rows'] == N and full['n_labelled'] == N - 17 and dropped[0]['n_rows'] == N - 17
    assert [full['lo'], full['hi']] != pytest.approx([dropped[0]['lo'], dropped[0]['hi']], abs=1e-9)
    # the full-set cell equals the masked-after-indexing hand bootstrap (test_sc1_frozen_arm_draws_no_seeds)


# ------------------------------------------------------------------------------------------------ SC3
def _hand_matched(d, fixed_max=False, shared_seeds=False):
    y = d['y']
    rng = np.random.default_rng(0)
    r20_all = _score([d['matched'][s] for s in SEEDS], y, np.arange(N), 'r20')
    own_all = _score([d['matched'][s] for s in SEEDS], y, np.arange(N), 'own')
    point_reading = 'r20' if r20_all >= own_all else 'own'
    diffs, flips = [], 0
    for _ in range(B):
        rows = rng.integers(0, N, size=N)
        s_c = rng.choice(SEEDS, size=5, replace=True)
        s_m = s_c if shared_seeds else rng.choice(SEEDS, size=5, replace=True)
        a = _score([d['cand'][int(i)] for i in s_c], y, rows, 'r20')
        m = {r: _score([d['matched'][int(i)] for i in s_m], y, rows, r) for r in ('r20', 'own')}
        best = point_reading if fixed_max else max(m, key=m.get)
        flips += best != point_reading
        diffs.append(a - m[best])
    return diffs, flips, point_reading, r20_all, own_all


def test_sc3_larger_matched_reading_is_retaken_in_each_draw():
    """Readings close enough to cross in some draws: a reading fixed from the point values gives another interval."""
    rng = np.random.default_rng(7)
    y = _labels(rng)
    d = dict(y=y, cand=_fixed(rng, y, 0.40))
    d['matched'] = {s: {'r20': _noisy(rng, y, 0.50, 3), 'own': _noisy(rng, y, 0.50, 1)} for s in SEEDS}
    d['frozen'] = _fixed(rng, y, 0.6, seeds=[0])
    d['teacher'] = _fixed(rng, y, 0.55, seeds=[0])
    c = _by_sc(_cells(d))['SC3']
    retaken, flips, reading, r20_all, own_all = _hand_matched(d)
    fixed, _, _, _, _ = _hand_matched(d, fixed_max=True)
    assert flips > 0, 'the synthetic readings must cross over in some draws'
    assert _interval(retaken) != pytest.approx(_interval(fixed), abs=1e-9)
    assert [c['lo'], c['hi']] == pytest.approx(_interval(retaken), abs=1e-12, rel=0)
    assert c['comparator_reading'] == f'M2-{reading}'
    assert c['comparator_readings'] == {'M2-r20': pytest.approx(r20_all, abs=1e-12), 'M2-own': pytest.approx(own_all, abs=1e-12)}
    assert c['comparator_value'] == pytest.approx(max(r20_all, own_all), abs=1e-12)


def test_sc3_seeds_are_drawn_independently_per_arm(data):
    c = _by_sc(_cells(data))['SC3']
    independent, *_ = _hand_matched(data)
    shared, *_ = _hand_matched(data, shared_seeds=True)
    assert _interval(shared) != pytest.approx(_interval(independent), abs=1e-9)
    assert [c['lo'], c['hi']] == pytest.approx(_interval(independent), abs=1e-12, rel=0)


# ------------------------------------------------------------------------------------------------ SC4
def test_sc4_teacher_draws_nothing_and_unit_is_the_ranges_r(data):
    y = data['y']
    c = _by_sc(_cells(data, r=0.0123))['SC4']
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(B):
        rows = rng.integers(0, N, size=N)
        s = rng.choice(SEEDS, size=5, replace=True)
        diffs.append(_score([data['cand'][int(i)] for i in s], y, rows, 'r20') - _score([data['teacher'][0]], y, rows, 'r20'))
    assert [c['lo'], c['hi']] == pytest.approx(_interval(diffs), abs=1e-12, rel=0)
    assert c['r'] == 0.0123 and c['comparator'] == 'teacher_genre' and c['candidate'] == 'cstar_siglip2'
    assert 'validation' in c['r_source']


def test_cells_follow_the_index_format(data):
    cells = _cells(data, data='mmart_test')
    assert [c['sc'] for c in cells] == ['SC1', 'SC3', 'SC4']
    keys = ('sc encoder aspect task data candidate comparator candidate_value comparator_value d lo hi r r_source '
            'n_draws seeds n_rows').split()
    for c in cells:
        assert set(keys) <= set(c) and c['task'] == 'clu' and c['data'] == 'mmart_test' and c['seeds'] == SEEDS
        assert c['encoder'] == 'siglip2' and c['aspect'] == 'genre' and c['n_draws'] == B and c['n_rows'] == N
        json.dumps(c, allow_nan=False)
    sc1, sc3, sc4 = cells
    assert sc1['r'] is None and sc3['r'] is None                       # WIN-only bars carry no unit
    assert sc1['comparator'] == 'frozen_siglip2' and sc3['comparator'] == 'matched_M2'


def test_sc4_needs_a_unit_and_win_only_cells_must_not_have_one(data):
    with pytest.raises(AssertionError):
        _cells(data, scs=('SC4',), r=None)
    with pytest.raises(AssertionError):
        M.make_cell('SC1', 'siglip2', 'genre', 'c', 'f', 0.2, 0.1, 0.1, 0.0, 0.2, 0.01, 'x', B, 'mmart_test', N, N)
    with pytest.raises(AssertionError):
        M.make_cell('SC4', 'siglip2', 'genre', 'c', 't', 0.2, 0.1, 0.1, 0.0, 0.2, 0.0, 'x', B, 'mmart_test', N, N)


def test_only_the_requested_scs_are_scored(data):
    assert [c['sc'] for c in _cells(data, scs=('SC4',))] == ['SC4']
    cells = M.score_aspect('siglip2', 'genre', data['y'], data['cand'], frozen=data['frozen'], scs=('SC1',), n_draws=B,
                           workers=1)
    assert [c['sc'] for c in cells] == ['SC1']
    with pytest.raises(AssertionError):                                 # SC3 without its matched partitions
        M.score_aspect('siglip2', 'genre', data['y'], data['cand'], scs=('SC3',), n_draws=B, workers=1)


# ------------------------------------------------------------------------------------------------ cell file
def test_write_cells_replaces_the_same_encoder_sc_and_keeps_others(tmp_path, data):
    path = tmp_path / 'stage2' / 'cells_clu.json'
    c2 = _cells(data)
    other = [dict(c, encoder='siglip1') for c in c2]
    M.write_cells(path, c2)
    M.write_cells(path, other)
    again = [dict(c, d=c['d'] + 1) for c in c2 if c['sc'] == 'SC1']
    out = M.write_cells(path, again)
    saved = json.loads(path.read_text())
    assert saved == out and len(saved) == 6
    assert [c['d'] for c in saved if c['encoder'] == 'siglip2' and c['sc'] == 'SC1'] == [again[0]['d']]
    assert [c['d'] for c in saved if c['encoder'] == 'siglip2' and c['sc'] == 'SC3'] == [c2[1]['d']]
    assert [c['d'] for c in saved if c['encoder'] == 'siglip1'] == [c['d'] for c in other]


# ------------------------------------------------------------------------------------------------ matched conversion
def test_matched_fixed_takes_mt_matched_readings():
    rng = np.random.default_rng(1)
    parts = {s: {'own': rng.integers(0, 40, 50).astype(np.int32), 'k20': rng.integers(0, 20, (3, 50)).astype(np.int32)}
             for s in SEEDS}
    fixed = M.matched_fixed(parts)
    assert sorted(fixed) == SEEDS and set(fixed[0]) == {'r20', 'own'}
    assert fixed[2]['own'].shape == (1, 50) and fixed[2]['r20'].shape == (3, 50)
    assert np.array_equal(fixed[2]['r20'], parts[2]['k20']) and np.array_equal(fixed[2]['own'][0], parts[2]['own'])
    with pytest.raises(AssertionError):
        M.matched_fixed({s: parts[s] for s in (0, 1, 2)})


# ------------------------------------------------------------------------------------------------ stub data modules
NK, D = 240, 16


def _keys():
    ks = [f'Style_{i % 3}/painting-{i:04d}.jpg' for i in range(NK)]
    return sorted(ks, key=lambda k: hashlib.sha256(k.encode('utf-8')).hexdigest())


class Stub:
    """mt_data, mt_export, mt_matched and the committed record, on tiny synthetic arrays keyed by the key string."""

    def __init__(self, tmp_path):
        self.keys_all = _keys()
        self.calls = []
        self.tmp = tmp_path
        self.k = {name: self.keys_all for name in ('val', 'new', 'test', 'far', 'unseen_artist', 'shared_held')}
        self.k['test'] = self.keys_all[:200]
        rng = np.random.default_rng(3)
        self.lab = {a: rng.integers(0, 4, NK) for a in mt_paths.ASPECTS}
        for a in self.lab:
            self.lab[a][[3, 11, 40, 41]] = -1                          # unlabelled rows
        self.view = {}

    def _unit(self, tag, d=D):
        rng = np.random.default_rng(int(hashlib.md5(tag.encode()).hexdigest()[:8], 16))
        x = rng.normal(size=(NK, d)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def by_key(self, x, keys):
        pos = {k: i for i, k in enumerate(self.keys_all)}
        return x[[pos[k] for k in keys]]

    # mt_data
    def mt_data(self):
        s = self

        def keys(name):
            return list(s.k[name])

        def order(ks):
            return sorted(ks, key=lambda k: hashlib.sha256(k.encode('utf-8')).hexdigest())

        def labels(ks, aspect):
            return s.by_key(s.lab[aspect], ks).astype(np.int64)

        def features(view, ks):
            s.calls.append(('features', view, len(ks)))
            return s.by_key(s._unit('view:' + view), ks)
        return types.SimpleNamespace(keys=keys, order=order, labels=labels, features=features)

    def mt_export(self):
        s = self

        def load(keyset, enc, aspect, seed, ks):
            s.calls.append(('export', keyset, enc, aspect, seed, len(ks)))
            x = s.by_key(s._unit(f'cstar:{enc}:{aspect}:{seed}'), ks)
            return x.astype(np.float16)
        return types.SimpleNamespace(load=load)

    def mt_matched(self):
        s = self

        def partitions(enc, ks):
            s.calls.append(('matched', enc, len(ks)))
            rng = np.random.default_rng(len(ks))
            return {sd: {'own': rng.integers(0, 40, len(ks)).astype(np.int32),
                         'k20': rng.integers(0, 20, (3, len(ks))).astype(np.int32)} for sd in SEEDS}
        return types.SimpleNamespace(partitions=partitions)


@pytest.fixture
def stub(tmp_path, monkeypatch):
    s = Stub(tmp_path)
    monkeypatch.setitem(sys.modules, 'mt_data', s.mt_data())
    monkeypatch.setitem(sys.modules, 'mt_export', s.mt_export())
    monkeypatch.setitem(sys.modules, 'mt_matched', s.mt_matched())
    monkeypatch.setattr(mt_paths, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(M, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(M, '_export_sha', lambda keyset, enc, aspect, seed: f'sha:{keyset}:{enc}:{aspect}:{seed}')
    rec = {'clu_ranges': {'ranges': {'emotion': 0.011, 'style': 0.022, 'genre': 0.033}}}
    monkeypatch.setattr(mt_paths, 'read_committed_record', lambda name: rec)
    return s


# ------------------------------------------------------------------------------------------------ clustering_ranges
def test_ranges_are_max_minus_min_over_five_seeds_on_validation_only(stub, monkeypatch, tmp_path):
    per = {a: {s: 0.2 + 0.001 * ((s * 7 + len(a)) % 5) + 1e-7 * s for s in SEEDS} for a in mt_paths.ASPECTS}
    seen = []
    order = iter([per[a][s] for a in mt_paths.ASPECTS for s in SEEDS])

    def fake_ami(x, y):
        seen.append((x.dtype, x.shape, y.shape, int((y < 0).sum())))
        return next(order)
    monkeypatch.setattr(M, 'validation_ami', fake_ami)
    res = M.clustering_ranges('qwen')
    for a in mt_paths.ASPECTS:
        vals = [per[a][s] for s in SEEDS]
        assert res['ranges'][a] == max(vals) - min(vals)               # unrounded
        assert res['per_seed'][a] == per[a]
    assert len(seen) == 15
    assert all(dt == np.float32 and shx == (NK, D) and shy == (NK,) for dt, shx, shy, _ in seen)
    assert all(u > 0 for *_, u in seen)                                # unlabelled rows are in the matrix, not dropped
    exports = [c for c in stub.calls if c[0] == 'export']
    assert {c[1] for c in exports} == {'val'} and len(exports) == 15  # validation only
    assert all(c[5] == NK for c in exports)                            # every row of the set
    assert res['n_val'] == NK


def test_ranges_use_the_validation_ami_of_all_rows_with_masking(stub):
    """No monkeypatch of the AMI: the range of real k-means on stub exports equals max - min of a hand computation."""
    import ct_score
    res = M.clustering_ranges('siglip2')
    keys = stub.k['val']
    for a in mt_paths.ASPECTS:
        y = stub.lab[a]
        vals = []
        for s in SEEDS:
            x = stub.by_key(stub._unit(f'cstar:siglip2:{a}:{s}'), keys).astype(np.float16).astype(np.float32)
            ps = ct_score.output_partitions(x, 'k1')['r20']
            vals.append(np.mean([ami(y[y >= 0], p[y >= 0]) for p in ps]))
        assert res['ranges'][a] == pytest.approx(max(vals) - min(vals), abs=1e-12, rel=0)


def test_write_ranges_goes_to_stage1_and_reads_back(stub, tmp_path):
    res = M.clustering_ranges('clip_l14')
    path = M.write_ranges('clip_l14', res)
    assert path == tmp_path / 'out' / 'stage1' / 'clu_ranges_clip_l14.json'
    rec = json.loads(path.read_text())
    assert rec['encoder'] == 'clip_l14' and set(rec['ranges']) == set(mt_paths.ASPECTS)
    assert rec['ranges'] == res['ranges']
    assert rec['sources']['emotion']['0'] == 'sha:val:clip_l14:emotion:0'


# ------------------------------------------------------------------------------------------------ score_encoder
def test_score_encoder_wires_rows_views_and_r(stub, tmp_path, monkeypatch):
    import ct_score
    sizes = []
    real = ct_score.output_partitions

    def spy(x, kind):
        sizes.append((len(x), kind))
        return real(x, kind)
    monkeypatch.setattr(ct_score, 'output_partitions', spy)
    cells = M.score_encoder('siglip2', keyset='test', n_draws=8, workers=1)
    assert len(cells) == 9 and [c['sc'] for c in cells[:3]] == ['SC1', 'SC3', 'SC4']
    assert {c['data'] for c in cells} == {'mmart_test'} and {c['n_rows'] for c in cells} == {200}
    sc4 = {c['aspect']: c for c in cells if c['sc'] == 'SC4'}
    assert {a: c['r'] for a, c in sc4.items()} == {'emotion': 0.011, 'style': 0.022, 'genre': 0.033}   # from the record
    assert all(c['r'] is None for c in cells if c['sc'] != 'SC4')
    # k-means on every row of the set (200), never on labelled rows only; tests labels hold -1 inside the set
    assert set(n for n, _ in sizes) == {200}
    k = stub.k['test']
    assert any((stub.lab[a][[stub.keys_all.index(x) for x in k]] < 0).any() for a in mt_paths.ASPECTS)
    # exports of the 'new' keyset for test keys; matched partitions on exactly the cut set's rows
    ex = [c for c in stub.calls if c[0] == 'export']
    assert {c[1] for c in ex} == {'new'} and len(ex) == 15 and {c[5] for c in ex} == {200}
    assert ('matched', 'siglip2', 200) in stub.calls
    feats = {c[1] for c in stub.calls if c[0] == 'features'}
    assert feats == {'siglip2', 'emotion_focus', 'style_word', 'genre_word'}
    path = tmp_path / 'out' / 'stage2' / 'cells_clu.json'
    assert json.loads(path.read_text()) == cells


def test_score_encoder_values_equal_a_hand_run_on_the_same_arrays(stub):
    """One aspect, by hand through ct_score and the statistic functions, equals the glue's cell."""
    import ct_score
    import bs_artclu
    keys = stub.k['test']
    y = stub.by_key(stub.lab['genre'], keys)
    cstar = {s: {'r20': ct_score.output_partitions(
        stub.by_key(stub._unit(f'cstar:siglip2:genre:{s}'), keys).astype(np.float16).astype(np.float32), 'k1')['r20']}
        for s in SEEDS}
    fz = {0: {'r20': ct_score.output_partitions(stub.by_key(stub._unit('view:siglip2'), keys), 'k1')['r20']}}
    want = bs_artclu.frozen_margin(cstar, fz, y, 8, 1)
    cells = M.score_encoder('siglip2', keyset='test', scs=('SC1',), n_draws=8, workers=1)
    c = [c for c in cells if c['aspect'] == 'genre'][0]
    assert (c['d'], c['lo'], c['hi']) == (want['d'], want['interval'][0], want['interval'][1])
    assert c['candidate_value'] == want['A_cand']


def test_diagnostic_keysets_name_their_data_and_file(stub, tmp_path):
    cells = M.score_encoder('qwen', keyset='unseen_artist', scs=('SC1',), n_draws=8, workers=1)
    assert {c['data'] for c in cells} == {'mmart_unseen_artist'}
    assert (tmp_path / 'out' / 'diag' / 'cells_clu_unseen_artist.json').exists()
    assert not (tmp_path / 'out' / 'stage2').exists()
    cells = M.score_encoder('qwen', keyset='shared_held', scs=('SC1', 'SC4'), n_draws=8, workers=1)
    assert {c['data'] for c in cells} == {'mmart_shared_held'}
    assert {c[1] for c in stub.calls if c[0] == 'export'} >= {'shared_held'}
    with pytest.raises(AssertionError):                                # no matched arm on the ArtELingo held paintings
        M.score_encoder('qwen', keyset='shared_held', scs=('SC3',), n_draws=8, workers=1)


def test_score_encoder_refuses_keys_not_in_the_sha256_order(stub, monkeypatch):
    data_mod = stub.mt_data()
    bad = list(reversed(stub.k['test']))
    data_mod.keys = lambda name: bad
    monkeypatch.setitem(sys.modules, 'mt_data', data_mod)
    with pytest.raises(AssertionError):
        M.score_encoder('qwen', keyset='test', scs=('SC1',), n_draws=8, workers=1)


def test_stage2_scoring_reads_the_committed_record_for_every_set_and_every_sc(stub, monkeypatch):
    asked = []
    monkeypatch.setattr(mt_paths, 'read_committed_record', lambda name: asked.append(name) or
                        {'clu_ranges': {'ranges': {a: 0.01 for a in mt_paths.ASPECTS}}})
    M.score_encoder('clip_l14', keyset='test', scs=('SC4',), n_draws=8, workers=1)
    assert asked == ['stage1_clip_l14.json']
    M.score_encoder('clip_l14', keyset='test', scs=('SC1',), n_draws=8, workers=1)
    assert asked == ['stage1_clip_l14.json'] * 2                       # SC1 needs no range, but stage 1 must be committed
    M.score_encoder('clip_l14', keyset='far', scs=('SC1', 'SC3'), n_draws=8, workers=1)
    assert asked == ['stage1_clip_l14.json'] * 3


def test_sc1_on_test_refuses_without_the_committed_record_before_any_row_is_read(stub, monkeypatch):
    def refuse(name):
        raise RuntimeError(f'record {name} is not committed unchanged in HEAD')
    monkeypatch.setattr(mt_paths, 'read_committed_record', refuse)
    stub.calls.clear()
    for scs in (('SC1',), ('SC1', 'SC3'), ('SC3',)):
        with pytest.raises(RuntimeError):
            M.score_encoder('siglip2', keyset='test', scs=scs, n_draws=8, workers=1)
    assert stub.calls == []                                            # no export, feature or label was read


def test_record_of_another_encoder_is_refused(stub, monkeypatch):
    other = {'encoder': 'qwen', 'clu_ranges': {'encoder': 'qwen', 'ranges': {a: 0.01 for a in mt_paths.ASPECTS}}}
    monkeypatch.setattr(mt_paths, 'read_committed_record', lambda name: other)
    with pytest.raises(AssertionError):
        M.score_encoder('siglip2', keyset='test', scs=('SC4',), n_draws=8, workers=1)
    with pytest.raises(AssertionError):
        M.score_encoder('siglip2', keyset='test', scs=('SC1',), n_draws=8, workers=1)
    with pytest.raises(AssertionError):
        M.record_ranges({'clu_ranges': {'encoder': 'qwen', 'ranges': {a: 0.1 for a in mt_paths.ASPECTS}}}, 'siglip2')
    assert M.record_ranges({'encoder': 'qwen', 'clu_ranges': {'ranges': {a: 0.1 for a in mt_paths.ASPECTS}}}, 'qwen')


def test_range_of_a_record_must_be_finite_and_positive():
    with pytest.raises(AssertionError):
        M.record_ranges({'clu_ranges': {'ranges': {'emotion': 0.0, 'style': 0.1, 'genre': 0.1}}})
    with pytest.raises(AssertionError):
        M.record_ranges({'clu_ranges': {'ranges': {'emotion': 0.1, 'style': 0.1}}})
    assert M.record_ranges({'clu_ranges': {'ranges': {'emotion': 0.1, 'style': 0.2, 'genre': 0.3}}})['style'] == 0.2


# ------------------------------------------------------------------------------------------------ replay comparison
def test_replay_compare_uses_1e9_and_skips_intervals_on_reduced_draws():
    ours = dict(candidate_value=0.1, comparator_value=0.05, d=0.05, lo=0.04, hi=0.06)
    theirs = dict(ours)
    full = M.compare_cell(ours, theirs, 'x', full_draws=True)
    assert full['x/lo']['ok'] and set(full) == {'x/candidate_value', 'x/comparator_value', 'x/d', 'x/lo', 'x/hi'}
    off = dict(theirs, hi=0.06 + 5e-9)
    assert not M.compare_cell(ours, off, 'x', full_draws=True)['x/hi']['ok']
    assert M.compare_cell(ours, dict(theirs, hi=0.06 + 5e-10), 'x', full_draws=True)['x/hi']['ok']
    reduced = M.compare_cell(ours, off, 'x', full_draws=False)           # intervals depend on the draw count
    assert set(reduced) == {'x/candidate_value', 'x/comparator_value', 'x/d'}
    assert not M.compare_cell(ours, dict(theirs, d=0.05 + 1e-8), 'x', full_draws=False)['x/d']['ok']
