"""Tests of om_cells (ticket 10): the cell list, the repeated TURTLE candidate, labels, voids, pins, sentences, the C13 guard.
Synthetic data only; nothing reads a real store, `records/` or `om_paths.DEFAULT_OUT`.

Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q
     src/test/20261103_outside_methods/test_om_cells.py
"""
import copy
import json
from collections import Counter

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

import om_data  # noqa: F401  (imported before any test patches things: it asserts N_ROWS)
import om_paths as op
import om_cells as oc
import om_stage1 as s1

import ns_clu  # noqa: E402  (om_paths put r17's folder on sys.path)
import ns_outcome  # noqa: E402
import ns_paths  # noqa: E402
import ns_ret  # noqa: E402


@pytest.fixture(autouse=True)
def _one_worker(monkeypatch):
    monkeypatch.setattr(ns_clu, 'WORKERS', 1)


# ------------------------------------------------------------------------------------------------ the cell list
def test_cell_counts_per_sc():
    specs = oc.cell_specs()
    got = Counter(c['sc'] for c in specs)
    assert got == {'SC1': 29, 'SC2': 6, 'SC3': 58, 'SC4': 12, 'SC5': 116, 'SC6': 116}
    assert len({c['id'] for c in specs}) == len(specs) == 337


def test_cell_shape_and_encoders():
    specs = oc.cell_specs()
    by = lambda sc: [c for c in specs if c['sc'] == sc]
    assert {c['encoder'] for c in by('SC1')} == {'qwen'} == {c['encoder'] for c in by('SC2')}
    assert {c['encoder'] for c in by('SC3')} == {'qwen', 'clip_l14'} == {c['encoder'] for c in by('SC4')}
    assert {c['encoder'] for c in by('SC5')} == set(op.ENCODERS) == {c['encoder'] for c in by('SC6')}
    assert {c['comparator'] for c in by('SC1')} == {'tpips'} and {c['comparator'] for c in by('SC3')} == {'clay'}
    assert {c['comparator'] for c in by('SC5') + by('SC6')} == {'plain_turtle'}
    assert {c['candidate'] for c in by('SC5')} == {'cstar_turtle'} and {c['candidate'] for c in by('SC6')} == {'cstar'}
    assert {c['task'] for c in by('SC2') + by('SC4')} == {'ret'} and {c['task'] for c in specs} == {'clu', 'ret'}
    for c in specs:                                         # id = SC/set/encoder/aspect/metric/k/comparator, k '-' for map
        k = '-' if c['k'] is None else str(c['k'])
        assert c['id'] == '/'.join((c['sc'], c['set'], c['encoder'], c['aspect'], c['metric'], k, c['comparator']))
    # MMArt style: the class count is 20, so one K and no separate AMI-at-20 cell
    style = [c for c in by('SC1') if (c['set'], c['aspect']) == ('mmart', 'style')]
    assert sorted((c['metric'], c['k']) for c in style) == [('acc', 20), ('ami', 20), ('ari', 20), ('nmi', 20)]
    emo = [c for c in by('SC1') if (c['set'], c['aspect']) == ('artelingo', 'emotion')]
    assert sorted((c['metric'], c['k']) for c in emo) == [('acc', 9), ('ami', 9), ('ami', 20), ('ari', 9), ('nmi', 9)]


def test_every_range_key_exists_in_stage1():
    keys = {k['key']: k for k in s1.range_keys()}
    for c in oc.cell_specs():
        rk = oc.range_key(c)
        assert rk in keys, (c['id'], rk)
        assert keys[rk]['metric'] == c['metric'] and keys[rk]['k'] == c['k'] and keys[rk]['enc'] == c['encoder']
        assert (c['task'] == 'ret') == rk.endswith('/map')


def test_k_values_come_from_the_label_ks():
    for c in oc.cell_specs():
        if c['task'] == 'clu':
            assert c['k'] in op.label_ks(c['set'], c['aspect'])


# ------------------------------------------------------------------------------------------------ the repeated candidate
def _toy(n=40, k=4, seed=0, mask=True):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    if mask:
        y[::7] = -1
    seeds = np.stack([np.where(rng.random(n) < 0.6, np.maximum(y, 0), rng.integers(0, k, size=n)) for _ in range(5)])
    plain = np.stack([np.where(rng.random(n) < 0.4, np.maximum(y, 0), rng.integers(0, k, size=n)) for _ in range(3)])
    return y.astype(np.int64), seeds[:, None, :].astype(np.int32), plain.astype(np.int32), k


def _hand(fn, y, a, b, draws, rng_seed=0):
    """The cell by hand: r14's draw order (rows, then c*'s five seeds), arm value = mean over drawn seeds of the mean over
    states, unlabelled rows dropped after the resample. a [5, S, n] candidate, b [S', n] comparator."""
    n = len(y)

    def val(parts, rows):
        keep = y[rows] >= 0
        return np.mean([fn(y[rows][keep], p[rows][keep]) for p in parts])

    rng = np.random.default_rng(rng_seed)
    d = []
    for _ in range(draws):
        rows = rng.integers(0, n, size=n)
        seeds = rng.choice(list(range(5)), size=5, replace=True)
        d.append(np.mean([val(a[s], rows) for s in seeds]) - val(b, rows))
    allrows = np.arange(n)
    point = np.mean([val(a[s], allrows) for s in range(5)]), val(b, allrows)
    return point, np.percentile(d, [2.5, 97.5])


def test_repeated_turtle_candidate_matches_hand_mean_and_seed_draws():
    y, parts, plain, k = _toy()
    cand = oc.repeat_states(parts)
    assert cand.shape == (5, 3, len(y)) and cand.dtype == parts.dtype
    for s in range(5):
        for st in range(3):
            assert np.array_equal(cand[s, st], parts[s, 0])
    res = ns_clu.paired_cell(cand, plain, y, k, r=None, metrics=('ari', 'ami'), draws=30)
    (c_pt, m_pt), (lo, hi) = _hand(adjusted_rand_score, y, cand, plain, 30)
    assert res['ari']['cand'] == pytest.approx(c_pt, abs=1e-12) and res['ari']['comp'] == pytest.approx(m_pt, abs=1e-12)
    assert res['ari']['lo'] == pytest.approx(lo, abs=1e-9) and res['ari']['hi'] == pytest.approx(hi, abs=1e-9)
    (c_pt, m_pt), (lo, hi) = _hand(adjusted_mutual_info_score, y, cand, plain, 30)
    assert res['ami']['cand'] == pytest.approx(c_pt, abs=1e-9) and res['ami']['lo'] == pytest.approx(lo, abs=1e-9)
    # the point value is the plain mean over the five seeds (one partition each), not any single seed
    one = [adjusted_rand_score(y[y >= 0], parts[s, 0][y >= 0]) for s in range(5)]
    assert res['ari']['cand'] == pytest.approx(np.mean(one), abs=1e-12)
    assert max(one) - min(one) > 1e-3        # the seeds differ, so a draw of seeds matters


def test_unrepeated_turtle_candidate_is_refused_by_the_scorer():
    y, parts, plain, k = _toy()
    with pytest.raises(AssertionError):
        ns_clu.paired_cell(parts, plain, y, k, r=None, metrics=('ari',), draws=5)
    with pytest.raises(AssertionError):
        oc.repeat_states(parts[:, 0])                       # [5, n]: not a TURTLE c* array
    with pytest.raises(AssertionError):
        oc.repeat_states(parts[:4])                         # four seeds


# ------------------------------------------------------------------------------------------------ fake loaders
class FakeLoaders:
    """Synthetic stand-ins for `oc.Loaders`: partitions and per-query AP built from the labels, deterministic."""

    def __init__(self, n=36, missing=()):
        self.n, self.missing, self.calls = n, set(missing), []
        self.rng = np.random.default_rng(1)

    def _y(self, set_, aspect):
        k = op.CLASS_COUNT[(set_, aspect)]
        y = (np.arange(self.n) * 7 % self.n) % k
        y = y.astype(np.int64)
        y[::9] = -1
        return y

    def labels(self, set_, aspect):
        return self._y(set_, aspect)

    def _noisy(self, set_, aspect, k, p, states, seeds=None):
        y = np.maximum(self._y(set_, aspect), 0) % k
        shape = (states, self.n) if seeds is None else (seeds, states, self.n)
        return np.where(self.rng.random(shape) < p, y, self.rng.integers(0, k, size=shape)).astype(np.int32)

    def cstar_parts(self, set_, enc, aspect, k):
        self.calls.append(('cstar_parts', set_, enc, aspect, k))
        return self._noisy(set_, aspect, k, 0.7, 3, seeds=5)

    def tpips_parts(self, set_, aspect, k):
        self.calls.append(('tpips_parts', set_, aspect, k))
        if 'tpips' in self.missing:
            raise FileNotFoundError('no TPIPS store')
        return self._noisy(set_, aspect, k, 0.5, 3)

    def clay_parts(self, set_, aspect, k, text_set):
        self.calls.append(('clay_parts', set_, aspect, k, text_set))
        return self._noisy(set_, aspect, k, 0.5, 3)

    def turtle_plain(self, set_, enc, k):
        self.calls.append(('turtle_plain', set_, enc, k))
        return self._noisy(set_, 'emotion', k, 0.5, 3)

    def turtle_cstar(self, set_, enc, aspect, k):
        self.calls.append(('turtle_cstar', set_, enc, aspect, k))
        return self._noisy(set_, aspect, k, 0.7, 1, seeds=5)          # [5, 1, n] as om_data.turtle_parts

    def ap(self, arm, set_, enc, aspect, pick, text_set=None):
        self.calls.append(('ap', arm, set_, enc, aspect, pick, text_set))
        nq = int((self._y(set_, aspect) >= 0).sum())
        base = {'cstar': 0.6, 'tpips': 0.4, 'clay': 0.4}[arm]
        shape = (5, nq) if arm == 'cstar' else (nq,)
        ap = np.clip(base + 0.1 * self.rng.standard_normal(shape), 0, 1).astype(np.float64)
        return {'ap': ap, 'rows': np.arange(nq, dtype=np.int64), 'point': float(ap.mean()), 'per_seed': None, 'dropped': 0}


def good_picks():
    sha = lambda ts: {a: f'sha-{ts}-{a}' for a in op.ASPECTS}
    p = {s1.PIN_KEY: {'A': sha('A'), 'B': sha('B')}}
    for s in op.SETS:
        p[f'tpips/{s}'] = {'pick': 'center+k10'}
        p[f'clay/{s}'] = {'pick': 'whiten49+k0', 'text_set': 'A'}
        p[f'clay_textset/clu/{s}'] = {'set': 'A', 'scores': {}, 'rule': 'x'}
        p[f'clay_textset/ret/{s}'] = {'set': 'A', 'scores': {}, 'rule': 'x'}
    return p


@pytest.fixture
def patch_sha(monkeypatch):
    monkeypatch.setattr(oc, 'text_sha', lambda ts, aspect: f'sha-{ts}-{aspect}')
    monkeypatch.setattr(ns_ret, 'reused_pick', lambda arm, set_, enc: f'reused-{arm}-{set_}-{enc}')


# ------------------------------------------------------------------------------------------------ scoring wiring
def test_sc5_unit_has_15_cells_on_artelingo_and_scores_the_aspects_labels(patch_sha):
    ld = FakeLoaders()
    raw = oc.score_unit('SC5', 'artelingo', 'siglip1', ld, picks=good_picks(), draws=25)
    assert len(raw) == 15 and len(oc.score_unit('SC5', 'mmart', 'siglip1', ld, picks=good_picks(), draws=5)) == 14
    cell = next(c for c in raw if c['aspect'] == 'emotion' and c['metric'] == 'ari' and c['k'] == 9)
    y = ld.labels('artelingo', 'emotion')
    assert cell['n_draws'] == 25 and cell['n_rows'] == ld.n and cell['n_labelled'] == int((y >= 0).sum())
    assert {c[0] for c in ld.calls} == {'turtle_cstar', 'turtle_plain'}


def test_sc5_numbers_equal_the_direct_scorer_call(patch_sha, monkeypatch):
    y, parts, plain, k = _toy(n=36, k=9, mask=True)

    class L(FakeLoaders):
        def labels(self, set_, aspect):
            return y

        def turtle_plain(self, set_, enc, k_):
            return plain % k_

        def turtle_cstar(self, set_, enc, aspect, k_):
            return parts % k_

    raw = oc.score_unit('SC5', 'artelingo', 'qwen', L(n=36), picks=good_picks(), draws=20)
    cell = next(c for c in raw if (c['aspect'], c['metric'], c['k']) == ('emotion', 'ami', 9))
    direct = ns_clu.paired_cell(np.repeat(parts % 9, 3, axis=1), plain % 9, y, 9, r=None, metrics=('ami',), draws=20)['ami']
    for key in ('cand', 'comp', 'd', 'lo', 'hi'):
        assert cell[key] == pytest.approx(direct[key], abs=1e-12), key


def test_sc6_uses_cstar_kmeans_candidate_against_plain_turtle(patch_sha):
    ld = FakeLoaders()
    raw = oc.score_unit('SC6', 'mmart', 'clip_l14', ld, picks=good_picks(), draws=10)
    assert len(raw) == len([c for c in oc.cell_specs() if (c['sc'], c['set'], c['encoder']) == ('SC6', 'mmart', 'clip_l14')])
    kinds = {c[0] for c in ld.calls}
    assert kinds == {'cstar_parts', 'turtle_plain'}


def test_sc1_and_sc3_use_kmeans_on_tpips_and_clay_with_the_picked_text_set(patch_sha):
    picks = good_picks()
    picks['clay_textset/clu/mmart'] = {'set': 'B', 'scores': {}, 'rule': 'x'}
    ld = FakeLoaders()
    raw1 = oc.score_unit('SC1', 'artelingo', 'qwen', ld, picks=picks, draws=10)
    assert len(raw1) == 15 and all(c['comparator'] == 'tpips' for c in raw1)
    raw3 = oc.score_unit('SC3', 'mmart', 'clip_l14', ld, picks=picks, draws=10)
    assert {c['text_set'] for c in raw3} == {'B'}
    assert {call[4] for call in ld.calls if call[0] == 'clay_parts'} == {'B'}


def test_sc3_refuses_a_text_set_whose_pin_changed(patch_sha, monkeypatch):
    picks = good_picks()
    monkeypatch.setattr(oc, 'text_sha', lambda ts, aspect: 'something else')
    with pytest.raises(AssertionError, match='pin'):
        oc.score_unit('SC3', 'artelingo', 'qwen', FakeLoaders(), picks=picks, draws=5)
    # a missing pin (set B fell back) is not a pass either
    picks = good_picks()
    picks['clay_textset/clu/artelingo']['set'] = 'B'
    picks[s1.PIN_KEY]['B']['emotion'] = None
    with pytest.raises(AssertionError, match='pin'):
        oc.score_unit('SC3', 'artelingo', 'qwen', FakeLoaders(), picks=picks, draws=5)


def test_check_pin_ignores_other_aspects_and_compares_exactly(monkeypatch):
    monkeypatch.setattr(oc, 'text_sha', lambda ts, aspect: 'abc' if aspect == 'emotion' else 'zzz')
    pins = {s1.PIN_KEY: {'A': {'emotion': 'abc', 'style': 'q', 'genre': 'q'}, 'B': {a: None for a in op.ASPECTS}}}
    oc.check_pin(pins, 'A', 'emotion')
    with pytest.raises(AssertionError):
        oc.check_pin(pins, 'A', 'style')


def test_retrieval_cells_sc2_and_sc4(patch_sha):
    ld = FakeLoaders()
    raw = oc.score_unit('SC2', 'artelingo', 'qwen', ld, picks=good_picks(), draws=40)
    assert len(raw) == 3 and all(c['k'] is None and c['metric'] == 'map' and c['id'].endswith('/-/tpips') for c in raw)
    ap_calls = [c for c in ld.calls if c[0] == 'ap']
    assert ('ap', 'cstar', 'artelingo', 'qwen', 'emotion', 'reused-cstar-artelingo-qwen', None) in ap_calls
    assert ('ap', 'tpips', 'artelingo', 'qwen', 'emotion', 'center+k10', None) in ap_calls
    assert all(c['n_draws'] == 40 and c['candidate_readout'] == 'reused-cstar-artelingo-qwen' for c in raw)
    assert all(c['comparator_readout'] == 'center+k10' for c in raw)
    raw4 = oc.score_unit('SC4', 'mmart', 'clip_l14', ld, picks=good_picks(), draws=40)
    assert len(raw4) == 3 and all(c['text_set'] == 'A' and c['comparator_readout'] == 'whiten49+k0' for c in raw4)
    assert ('ap', 'clay', 'mmart', 'clip_l14', 'genre', 'whiten49+k0', 'A') in ld.calls


def test_retrieval_refuses_arms_scored_on_other_queries(patch_sha):
    class L(FakeLoaders):
        def ap(self, arm, *a, **k):
            r = super().ap(arm, *a, **k)
            if arm == 'tpips':
                r['rows'] = r['rows'] + 1
            return r

    with pytest.raises(AssertionError, match='queries'):
        oc.score_unit('SC2', 'artelingo', 'qwen', L(), picks=good_picks(), draws=5)


def test_retrieval_text_set_must_agree_between_the_pick_and_the_choice(patch_sha):
    picks = good_picks()
    picks['clay/mmart']['text_set'] = 'B'
    with pytest.raises(AssertionError, match='text set'):
        oc.score_unit('SC4', 'mmart', 'qwen', FakeLoaders(), picks=picks, draws=5)


def test_missing_input_gives_cells_without_numbers_not_a_crash(patch_sha):
    ld = FakeLoaders(missing=('tpips',))
    raw = oc.score_unit('SC1', 'artelingo', 'qwen', ld, picks=good_picks(), draws=5)
    assert len(raw) == 15
    assert all(c['d'] is None and c['lo'] is None and 'no TPIPS store' in c['missing'] for c in raw)
    picks = good_picks()
    del picks['tpips/mmart']
    raw = oc.score_unit('SC2', 'mmart', 'qwen', FakeLoaders(), picks=picks, draws=5)
    assert len(raw) == 3 and all(c['d'] is None and 'tpips/mmart' in c['missing'] for c in raw)


# ------------------------------------------------------------------------------------------------ labels
def test_label_margins():
    r = 0.01
    assert oc.label_of(1e-9, r) == 'WIN'
    assert oc.label_of(0.0, r) == 'LEVEL'                    # not WIN, lo > -r
    assert oc.label_of(-r + 1e-9, r) == 'LEVEL'
    assert oc.label_of(-r, r) == 'BELOW'                     # strict: lo > -r only
    assert oc.label_of(-r - 1e-9, r) == 'BELOW'
    assert oc.label_of(0.5, r) == 'WIN'


def test_missing_range_gives_inconclusive_unless_win():
    assert oc.label_of(0.2, None) == 'WIN'
    assert oc.label_of(0.0, None) == 'inconclusive'
    assert oc.label_of(-0.3, None) == 'inconclusive'


def ok_status():
    return {'gates': {g: 'ok' for g in ('i', 'ii', 'iii_a', 'iii_b', 'iii_c', 'iv')}, 'stage1': {},
            'r17': {'path': 'x', 'sha256': 'y'}}


def ok_r17():
    g = {k: 'ok' for k in ns_outcome.GATE_SINGLE}
    g.update(i_c={s: 'ok' for s in ns_paths.PHOTO_SETS}, iii_metric={m: 'ok' for m in ns_paths.CLASS_METRICS},
             iii_count={f'{s}/{a}': 'ok' for (s, a) in ns_paths.CLASS_COUNT},
             iv={e: 'ok' for e in ns_paths.ENCODERS}, v={e: 'ok' for e in ns_paths.ENCODERS})
    ranges = \
        {k: 'ok' for k in {ns_outcome.range_key(c) for c in ns_outcome.claim_cells()} if k is not None}
    return {'gates': g, 'stage1': {'ranges': ranges, 'picks': {}}}


def ranges_record(r=0.01, missing=()):
    out = {}
    for k in s1.range_keys():
        out[k['key']] = {'r': None, 'source': f'missing:{k["key"]}'} if k['key'] in missing else {'r': r, 'source': 'test'}
    return out


def raw_cell(spec, lo=0.02, hi=0.05, d=0.03):
    return dict(spec, cand=0.5, comp=0.5 - d, d=d, lo=lo, hi=hi, n_draws=7, n_rows=10, missing=None)


def assemble_specs(specs, status=None, r17=None, ranges=None, picks=None, **raw_kw):
    raw = [raw_cell(s, **raw_kw) for s in specs]
    return oc.assemble(raw, ranges or ranges_record(), picks or good_picks(), status or ok_status(), r17 or ok_r17(),
                       expected=specs)


def spec_of(sc, set_='artelingo', enc=None, aspect='emotion', metric=None, k=None):
    for c in oc.cell_specs():
        if c['sc'] == sc and c['set'] == set_ and (enc is None or c['encoder'] == enc) and c['aspect'] == aspect \
                and (metric is None or c['metric'] == metric) and (k is None or c['k'] == k):
            return c
    raise KeyError((sc, set_, enc, aspect, metric, k))


def test_assemble_labels_and_r_source():
    sp = spec_of('SC1', metric='ami', k=9)
    out = assemble_specs([sp], lo=0.001)
    c = out[0]
    assert c['label'] == 'WIN' and c['void'] is None and c['r'] == 0.01 and c['r_source'] == 'test'
    assert c['sentence'].startswith('On ArtELingo held, c* on ')
    out = assemble_specs([sp], lo=-0.005, hi=0.02, d=0.01)
    assert out[0]['label'] == 'LEVEL'
    out = assemble_specs([sp], lo=-0.02, hi=0.02, d=0.0)
    assert out[0]['label'] == 'BELOW'
    assert set(oc.CELL_KEYS) <= set(out[0])


def test_assemble_missing_range_is_inconclusive_but_win_stands():
    sp = spec_of('SC1', metric='ami', k=9)
    rk = oc.range_key(sp)
    rec = ranges_record(missing=(rk,))
    assert assemble_specs([sp], ranges=rec, lo=0.001)[0]['label'] == 'WIN'
    c = assemble_specs([sp], ranges=rec, lo=-0.001)[0]
    assert c['label'] == 'inconclusive' and c['r'] is None and c['void'] is None and c['sentence'] is None
    # a range key absent from the record is the same as a null range
    rec = ranges_record()
    del rec[rk]
    assert assemble_specs([sp], ranges=rec, lo=-0.001)[0]['label'] == 'inconclusive'


def test_assemble_r17_failed_range_status_makes_r_missing():
    sp = spec_of('SC1', metric='ami', k=9)
    r17 = ok_r17()
    r17['stage1']['ranges'][f'cstar/artelingo/qwen/emotion/ami'] = 'failed'
    c = assemble_specs([sp], r17=r17, lo=-0.001)[0]
    assert c['label'] == 'inconclusive' and c['r'] is None
    assert assemble_specs([sp], r17=r17, lo=0.5)[0]['label'] == 'WIN'


def test_failed_gate_voids_exactly_its_cells():
    specs = oc.cell_specs()
    table = {'i': {'SC3', 'SC4'}, 'ii': {'SC5', 'SC6'}, 'iii_a': {'SC1', 'SC3', 'SC5', 'SC6'}, 'iii_b': {'SC5', 'SC6'},
             'iii_c': {'SC2', 'SC4'}, 'iv': {'SC1', 'SC2'}}
    for gate, scs in table.items():
        st = ok_status()
        st['gates'][gate] = 'failed'
        out = assemble_specs(specs, status=st)
        voided = {c['sc'] for c in out if c['void']}
        assert voided == scs, gate
        assert all((c['label'] is None) == bool(c['void']) for c in out)
        assert all(c['sentence'] is None for c in out if c['void'])
        assert all(f'gate {gate}' in c['void'] for c in out if c['void'])
    st = ok_status()
    st['gates']['iv'] = 'missing'
    assert {c['sc'] for c in assemble_specs(specs, status=st) if c['void']} == {'SC1', 'SC2'}


def test_a_voided_cell_keeps_no_label_even_for_a_win():
    sp = spec_of('SC3', enc='qwen', metric='ami', k=9)
    st = ok_status()
    st['gates']['i'] = 'failed'
    c = assemble_specs([sp], status=st, lo=0.5)[0]
    assert c['label'] is None and c['void'] and c['sentence'] is None and c['lo'] == 0.5


def test_stage1_picks_void_per_set_and_arm():
    specs = oc.cell_specs()
    picks = good_picks()
    del picks['tpips/mmart']
    out = assemble_specs(specs, picks=picks)
    assert {(c['sc'], c['set']) for c in out if c['void']} == {('SC2', 'mmart')}
    picks = good_picks()
    del picks['clay_textset/ret/artelingo']
    out = assemble_specs(specs, picks=picks)
    assert {(c['sc'], c['set']) for c in out if c['void']} == {('SC4', 'artelingo')}
    picks = good_picks()
    del picks['clay/mmart']
    assert {(c['sc'], c['set']) for c in assemble_specs(specs, picks=picks) if c['void']} == {('SC4', 'mmart')}
    picks = good_picks()
    del picks['clay_textset/clu/artelingo']
    assert {(c['sc'], c['set']) for c in assemble_specs(specs, picks=picks) if c['void']} == {('SC3', 'artelingo')}
    # the status can fail a stage 1 item too
    st = ok_status()
    st['stage1']['tpips/artelingo'] = 'failed'
    assert {(c['sc'], c['set']) for c in assemble_specs(specs, status=st) if c['void']} == {('SC2', 'artelingo')}


def test_r17_stage0_voids_the_same_cells_here():
    specs = oc.cell_specs()

    def voided(mut):
        r17 = ok_r17()
        mut(r17['gates'])
        return [c for c in assemble_specs(specs, r17=r17) if c['void']]

    assert {c['sc'] for c in voided(lambda g: g.update(i_a='failed'))} == {'SC1', 'SC3', 'SC5', 'SC6'}
    assert {c['sc'] for c in voided(lambda g: g.update(ii_scorer='failed'))} == {'SC2', 'SC4'}
    assert voided(lambda g: g.update(i_c={'affection': 'failed', 'emoset': 'failed'})) == []        # photo sets: not ours
    assert voided(lambda g: g.update(ii_crl='failed', ii_percept='failed', vi='failed')) == []
    out = voided(lambda g: g['iii_metric'].update(nmi='failed'))
    assert out and {c['metric'] for c in out} == {'nmi'} and {c['sc'] for c in out} == {'SC1', 'SC3', 'SC5', 'SC6'}
    out = voided(lambda g: g['iii_count'].update({'artelingo/genre': 'failed'}))
    # only the class-count cells of that set and aspect (genre's AMI at 20 is not one)
    assert out and all((c['set'], c['aspect']) == ('artelingo', 'genre') and c['k'] == 10 for c in out)
    out = voided(lambda g: g.update(i_d='failed'))
    assert out and all(c['k'] == op.CLASS_COUNT[(c['set'], c['aspect'])] for c in out)
    assert any(c['k'] == 20 and c['aspect'] == 'style' and c['set'] == 'mmart' for c in out)        # class count 20
    assert not any(c['k'] == 20 and (c['set'], c['aspect']) == ('artelingo', 'emotion') for c in out)


def test_status_shape_is_checked():
    st = ok_status()
    del st['gates']['iv']
    with pytest.raises(AssertionError):
        assemble_specs(oc.cell_specs()[:1], status=st)
    st = ok_status()
    st['gates']['i'] = 'maybe'
    with pytest.raises(AssertionError):
        assemble_specs(oc.cell_specs()[:1], status=st)


def test_a_cell_not_scored_is_void_and_unlabelled():
    specs = oc.cell_specs()[:3]
    out = oc.assemble([raw_cell(specs[0])], ranges_record(), good_picks(), ok_status(), ok_r17(), expected=specs)
    assert [c['id'] for c in out] == [s['id'] for s in specs]
    assert out[0]['label'] is not None and out[1]['label'] is None and out[2]['label'] is None
    assert 'not scored' in out[1]['void'] and out[1]['d'] is None
    # an id outside the list is refused
    bogus = dict(raw_cell(specs[0]), id='SC1/x')
    with pytest.raises(AssertionError):
        oc.assemble([bogus], ranges_record(), good_picks(), ok_status(), ok_r17(), expected=specs)


def test_missing_input_cell_is_void():
    sp = spec_of('SC1', metric='ami', k=9)
    raw = dict(raw_cell(sp), cand=None, comp=None, d=None, lo=None, hi=None, missing='FileNotFoundError: no TPIPS store')
    c = oc.assemble([raw], ranges_record(), good_picks(), ok_status(), ok_r17(), expected=[sp])[0]
    assert c['label'] is None and 'no TPIPS store' in c['void']


# ------------------------------------------------------------------------------------------------ sentences
def test_sentences():
    sp = spec_of('SC1', metric='ami', k=9)
    c = assemble_specs([sp], lo=0.001)[0]
    assert c['sentence'] == ('On ArtELingo held, c* on Qwen3-VL-Embedding-2B clusters paintings by emotion better than '
                             'TPIPS (AMI, 9 clusters).')
    c = assemble_specs([sp], lo=-0.005, hi=0.02, d=0.01)[0]
    assert c['label'] == 'LEVEL' and c['sentence'].endswith('level with TPIPS (AMI, 9 clusters).')
    c = assemble_specs([sp], lo=-0.05, hi=0.02, d=-0.01)[0]
    assert c['label'] == 'BELOW' and c['sentence'] is None                       # upper bound not below 0: no claim
    c = assemble_specs([sp], lo=-0.05, hi=-0.01, d=-0.03)[0]
    assert c['label'] == 'BELOW' and c['sentence'] == (
        'On ArtELingo held, TPIPS clusters paintings by emotion better than c* on Qwen3-VL-Embedding-2B (AMI, 9 clusters).')
    sp2 = spec_of('SC2', metric='map')
    c = assemble_specs([sp2], lo=0.001)[0]
    assert c['sentence'].startswith('On ArtELingo held, c* on ') and 'retrieves paintings by emotion better than TPIPS (mAP)' in c['sentence']
    sp5 = spec_of('SC5', enc='siglip2', metric='acc', k=9)
    c = assemble_specs([sp5], lo=0.001)[0]
    assert "TURTLE (its released code's default settings, without its hyperparameter selection) on both sides" in c['sentence']
    assert 'better than the plain SigLIP 2 image' in c['sentence']
    sp6 = spec_of('SC6', enc='siglip2', metric='acc', k=9)
    c = assemble_specs([sp6], lo=0.001)[0]
    assert ("the plain SigLIP 2 image clustered with TURTLE at its released code's default settings, without its "
            "hyperparameter selection") in c['sentence']
    sp3 = spec_of('SC3', enc='clip_l14', metric='ami', k=20)
    c = assemble_specs([sp3], lo=0.001)[0]
    assert 'better than CLAY (AMI, 20 clusters)' in c['sentence']


# ------------------------------------------------------------------------------------------------ the C13 guard
def _cells_for_c13():
    specs = [spec_of('SC1', metric='ami', k=9), spec_of('SC2', metric='map'), spec_of('SC4', set_='mmart', enc='qwen', aspect='genre')]
    raw = [dict(raw_cell(s), candidate_readout='rc', comparator_readout='rt') if s['task'] == 'ret' else raw_cell(s)
           for s in specs]
    return oc.assemble(raw, ranges_record(), good_picks(), ok_status(), ok_r17(), expected=specs)


def _c13_doc(cells, picks):
    return {'picks': oc.pick_summary(picks),
            'cells': [{k: c.get(k) for k in ('id', 'cand', 'comp', 'd', 'lo', 'hi', 'label', 'candidate_readout',
                                             'comparator_readout') if k in c} for c in cells]}


def test_c13_agrees_within_tolerance():
    cells, picks = _cells_for_c13(), good_picks()
    doc = _c13_doc(cells, picks)
    assert oc.c13_disagreements(cells, picks, doc) == []
    doc['cells'][0]['d'] += 0.9e-6
    doc['cells'][1]['lo'] -= 0.9e-6
    assert oc.c13_disagreements(cells, picks, doc) == []


@pytest.mark.parametrize('field', ['cand', 'comp', 'd', 'lo', 'hi'])
def test_c13_refuses_a_2e6_difference(field):
    cells, picks = _cells_for_c13(), good_picks()
    doc = _c13_doc(cells, picks)
    doc['cells'][1][field] += 2e-6
    bad = oc.c13_disagreements(cells, picks, doc)
    assert len(bad) == 1 and field in bad[0]


def test_c13_refuses_label_pick_and_id_differences():
    cells, picks = _cells_for_c13(), good_picks()
    doc = _c13_doc(cells, picks)
    doc['cells'][0]['label'] = 'LEVEL'
    assert any('label' in b for b in oc.c13_disagreements(cells, picks, doc))
    doc = _c13_doc(cells, picks)
    doc['picks']['tpips/artelingo'] = 'none+k0'
    assert any('tpips/artelingo' in b for b in oc.c13_disagreements(cells, picks, doc))
    doc = _c13_doc(cells, picks)
    doc['cells'][1]['comparator_readout'] = 'none+k0'
    assert any('comparator_readout' in b for b in oc.c13_disagreements(cells, picks, doc))
    doc = _c13_doc(cells, picks)
    doc['cells'].pop()
    assert any('missing' in b for b in oc.c13_disagreements(cells, picks, doc))
    doc = _c13_doc(cells, picks)
    doc['cells'].append(dict(doc['cells'][0], id='SC1/bogus'))
    assert any('bogus' in b for b in oc.c13_disagreements(cells, picks, doc))


def test_c13_void_cells_must_be_void_there_too():
    st = ok_status()
    st['gates']['i'] = 'failed'
    specs = [spec_of('SC3', enc='qwen', metric='ami', k=9)]
    cells = assemble_specs(specs, status=st)
    doc = _c13_doc(cells, good_picks())
    assert oc.c13_disagreements(cells, good_picks(), doc) == []
    doc['cells'][0]['label'] = 'WIN'
    assert oc.c13_disagreements(cells, good_picks(), doc)


def test_labels_step_refuses_without_c13_and_on_disagreement(tmp_path):
    cells, picks = _cells_for_c13(), good_picks()
    cells_path, c13_path, out_path = tmp_path / 'cells.json', tmp_path / 'c13' / 'c13_cells.json', tmp_path / 'rec' / 'labels.json'
    cells_path.write_text(json.dumps(cells))
    with pytest.raises(SystemExit) as e:
        oc.run_labels(cells_path, c13_path, out_path, picks)
    assert 'c13_cells.json' in str(e.value) and not out_path.exists()
    c13_path.parent.mkdir()
    doc = _c13_doc(cells, picks)
    doc['cells'][2]['hi'] += 2e-6
    c13_path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        oc.run_labels(cells_path, c13_path, out_path, picks)
    assert not out_path.exists()
    doc = _c13_doc(cells, picks)
    c13_path.write_text(json.dumps(doc))
    oc.run_labels(cells_path, c13_path, out_path, picks)
    rec = json.loads(out_path.read_text())
    assert [c['id'] for c in rec['cells']] == [c['id'] for c in cells]
    assert rec['cells_sha256'] == op.sha256_file(cells_path) and rec['c13_sha256'] == op.sha256_file(c13_path)
    assert [c['label'] for c in rec['cells']] == [c['label'] for c in cells]


def test_labels_step_refuses_a_cells_file_that_changed_the_label(tmp_path):
    cells, picks = _cells_for_c13(), good_picks()
    cells_path, c13_path = tmp_path / 'cells.json', tmp_path / 'c13_cells.json'
    c13_path.write_text(json.dumps(_c13_doc(cells, picks)))
    tampered = copy.deepcopy(cells)
    tampered[0]['label'] = 'BELOW' if cells[0]['label'] != 'BELOW' else 'WIN'
    cells_path.write_text(json.dumps(tampered))
    with pytest.raises(SystemExit):
        oc.run_labels(cells_path, c13_path, tmp_path / 'labels.json', picks)


def test_labels_step_never_defaults_to_the_real_records_in_tests():
    # the labels path comes from om_paths.record_path; under OM_SMOKE it is OUT/records, else RECORDS (committed)
    assert oc.labels_path().name == 'labels.json'
