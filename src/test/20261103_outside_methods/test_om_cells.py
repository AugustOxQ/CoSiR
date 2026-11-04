"""Tests of om_cells (ticket 10): the cell list, the repeated TURTLE candidate, labels, voids, pins, sentences, the C13 guard.
Synthetic data only; nothing reads a real store, `records/` or `om_paths.DEFAULT_OUT`.
Includes C13 stub/refusal cleanup and r17 range consistency checks at assembly.

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


@pytest.fixture(autouse=True)
def r17_ranges(tmp_path, monkeypatch):
    """A fake r17 record, independent of our ranges; keep all assembly tests off real r17 files."""
    monkeypatch.setattr(ns_paths, 'RECORDS', tmp_path / 'r17_reference')
    ns_paths.RECORDS.mkdir()
    path = ns_paths.RECORDS / 'stage1_ranges.json'
    keys = {f"cstar/{k['set']}/{k['enc']}/{k['aspect']}/{k['metric']}" for k in s1.range_keys() if k['source'] == 'r17'}
    path.write_text(json.dumps({k: {'r': 0.01} for k in keys}))
    return path


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
    """Synthetic stand-ins for `oc.Loaders`: partitions and per-query AP built from the labels. Every call is memoised, so a
    test can ask the stand-in again for the very arrays a cell was scored on and call r17's scorer on them directly."""

    def __init__(self, n=60, missing=()):
        self.n, self.missing, self.calls, self.memo = n, set(missing), [], {}
        self.rng = np.random.default_rng(1)

    def _memo(self, key, make):
        if key not in self.memo:
            self.memo[key] = make()
        return self.memo[key]

    def _y(self, set_, aspect):
        k = op.CLASS_COUNT[(set_, aspect)]
        y = (np.arange(self.n) % k).astype(np.int64)         # every class twice or more; the masked rows leave all present
        y[[3, 10, 22, 25]] = -1
        return y

    def labels(self, set_, aspect):
        return self._y(set_, aspect)

    def _noisy(self, set_, aspect, k, p, states, seeds=None):
        y = np.maximum(self._y(set_, aspect), 0) % k
        shape = (states, self.n) if seeds is None else (seeds, states, self.n)
        return np.where(self.rng.random(shape) < p, y, self.rng.integers(0, k, size=shape)).astype(np.int32)

    def cstar_parts(self, set_, enc, aspect, k):
        self.calls.append(('cstar_parts', set_, enc, aspect, k))
        return self._memo(('cstar_parts', set_, enc, aspect, k), lambda: self._noisy(set_, aspect, k, 0.7, 3, seeds=5))

    def tpips_parts(self, set_, aspect, k):
        self.calls.append(('tpips_parts', set_, aspect, k))
        if 'tpips' in self.missing:
            raise FileNotFoundError('no TPIPS store')
        return self._memo(('tpips_parts', set_, aspect, k), lambda: self._noisy(set_, aspect, k, 0.5, 3))

    def clay_parts(self, set_, aspect, k, text_set):
        self.calls.append(('clay_parts', set_, aspect, k, text_set))
        return self._memo(('clay_parts', set_, aspect, k, text_set), lambda: self._noisy(set_, aspect, k, 0.5, 3))

    def turtle_plain(self, set_, enc, k):
        self.calls.append(('turtle_plain', set_, enc, k))
        return self._memo(('turtle_plain', set_, enc, k), lambda: self._noisy(set_, 'emotion', k, 0.5, 3))

    def turtle_cstar(self, set_, enc, aspect, k):
        self.calls.append(('turtle_cstar', set_, enc, aspect, k))
        return self._memo(('turtle_cstar', set_, enc, aspect, k),
                          lambda: self._noisy(set_, aspect, k, 0.7, 1, seeds=5))          # [5, 1, n] as om_data.turtle_parts

    def ap(self, arm, set_, enc, aspect, pick, text_set=None):
        self.calls.append(('ap', arm, set_, enc, aspect, pick, text_set))

        def make():
            nq = int((self._y(set_, aspect) >= 0).sum())
            base = {'cstar': 0.6, 'tpips': 0.4, 'clay': 0.4}[arm]
            shape = (5, nq) if arm == 'cstar' else (nq,)
            ap = np.clip(base + 0.1 * self.rng.standard_normal(shape), 0, 1).astype(np.float64)
            return {'ap': ap, 'rows': np.arange(nq, dtype=np.int64), 'point': float(ap.mean()), 'per_seed': None, 'dropped': 0}
        return self._memo(('ap', arm, set_, enc, aspect, pick, text_set), make)


def good_picks():
    sha = lambda ts: {a: f'sha-{ts}-{a}' for a in op.ASPECTS}
    p = {s1.PIN_KEY: {'A': sha('A'), 'B': sha('B')}}
    for s in op.SETS:
        p[f'tpips/{s}'] = {'pick': 'center+k10'}
        p[f'clay/{s}'] = {'pick': 'whiten49+k0', 'text_set': 'A'}
        p[f'clay_textset/clu/{s}'] = {'set': 'A', 'scores': {}, 'rule': 'x'}
        p[f'clay_textset/ret/{s}'] = {'set': 'A', 'scores': {}, 'rule': 'x'}
    return p


@pytest.fixture(autouse=True)
def _reused_pick(monkeypatch):
    monkeypatch.setattr(ns_ret, 'reused_pick', lambda arm, set_, enc: f'reused-{arm}-{set_}-{enc}')


@pytest.fixture
def patch_sha(monkeypatch):
    monkeypatch.setattr(oc, 'text_sha', lambda ts, aspect: f'sha-{ts}-{aspect}')


# ------------------------------------------------------------------------------------------------ scoring wiring
def test_sc5_unit_has_15_cells_on_artelingo_and_scores_the_aspects_labels(patch_sha):
    ld = FakeLoaders()
    raw = oc.score_unit('SC5', 'artelingo', 'siglip1', ld, picks=good_picks(), draws=25)
    assert len(raw) == 15 and len(oc.score_unit('SC5', 'mmart', 'siglip1', ld, picks=good_picks(), draws=5)) == 14
    cell = next(c for c in raw if c['aspect'] == 'emotion' and c['metric'] == 'ari' and c['k'] == 9)
    y = ld.labels('artelingo', 'emotion')
    assert cell['n_draws'] == 25 and cell['n_rows'] == ld.n and cell['n_labelled'] == int((y >= 0).sum())
    assert {c[0] for c in ld.calls} == {'turtle_cstar', 'turtle_plain'}


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
    """A scored cell as `score_unit` writes it: the protocol's draws and the read-outs / text set of `good_picks()`."""
    c = dict(spec, cand=0.5, comp=0.5 - d, d=d, lo=lo, hi=hi, n_draws=op.N_DRAWS[spec['task']], n_rows=10, missing=None)
    if spec['task'] == 'ret':
        c.update(candidate_readout=f"reused-cstar-{spec['set']}-{spec['encoder']}",
                 comparator_readout='center+k10' if spec['sc'] == 'SC2' else 'whiten49+k0')
    if spec['sc'] in ('SC3', 'SC4'):
        c['text_set'] = 'A'
    return c


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


def test_absent_r17_range_status_is_missing_even_with_a_numeric_range():
    sp = spec_of('SC1', metric='ami', k=9)
    r17 = ok_r17()
    del r17['stage1']['ranges']['cstar/artelingo/qwen/emotion/ami']
    cell = assemble_specs([sp], r17=r17, lo=-0.001)[0]
    assert cell['r'] is None and cell['label'] == 'inconclusive'
    assert 'r17 range cstar/artelingo/qwen/emotion/ami missing' in cell['r_source']


@pytest.mark.parametrize('r,ref,refuse', [(0.005, 0.01, True), (0.01, 0.02, True),
                                        (0.01, 0.01 + 5e-13, False), (0.01, None, True)])
def test_assemble_compares_r17_family_with_its_record(r17_ranges, r, ref, refuse):
    sp = spec_of('SC1', metric='ami', k=9)
    key = 'cstar/artelingo/qwen/emotion/ami'
    doc = json.loads(r17_ranges.read_text())
    if ref is None:
        doc.pop(key)
    else:
        doc[key]['r'] = ref
    r17_ranges.write_text(json.dumps(doc))
    ranges = ranges_record()
    ranges[oc.range_key(sp)]['r'] = r
    if refuse:
        with pytest.raises(SystemExit, match='assemble.*' + oc.range_key(sp)):
            assemble_specs([sp], ranges=ranges)
    else:
        assert assemble_specs([sp], ranges=ranges)[0]['r'] == r


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


def _voided_ids(mut):
    r17 = ok_r17()
    mut(r17['gates'])
    return {c['id'] for c in assemble_specs(oc.cell_specs(), r17=r17) if c['void']}


def _ids(pred):
    """Hand-written selections of the 337 cells, by the cell fields only."""
    return {c['id'] for c in oc.cell_specs() if pred(c)}


CLU_SCS = ('SC1', 'SC3', 'SC5', 'SC6')


def test_r17_scorer_gates_void_the_cells_of_their_scorer():
    assert _voided_ids(lambda g: g.update(i_a='failed')) == _ids(lambda c: c['sc'] in CLU_SCS)
    assert _voided_ids(lambda g: g.update(ii_scorer='failed')) == _ids(lambda c: c['sc'] in ('SC2', 'SC4'))
    # r17's read-out rule (vi) voids a cell with a new pick: every SC2 and SC4 cell has one (TPIPS, CLAY)
    assert _voided_ids(lambda g: g.update(vi='failed')) == _ids(lambda c: c['sc'] in ('SC2', 'SC4'))
    assert len(_voided_ids(lambda g: g.update(vi='failed'))) == 18
    # not ours: photo sets, CRL, PercepT, the retrieval rows of the old arms, the other r17 gate keys
    assert _voided_ids(lambda g: g.update(i_c={'affection': 'failed', 'emoset': 'failed'})) == set()
    assert _voided_ids(lambda g: g.update(i_b='failed', ii_crl='failed', ii_percept='failed')) == set()
    assert _voided_ids(lambda g: g.update(iv={e: 'failed' for e in ns_paths.ENCODERS}, v={e: 'failed' for e in ns_paths.ENCODERS})) == set()


@pytest.mark.parametrize('metric', ['acc', 'ari', 'nmi', 'ami'])
def test_r17_metric_gate_voids_exactly_the_clustering_cells_of_that_metric(metric):
    got = _voided_ids(lambda g: g['iii_metric'].update({metric: 'failed'}))
    assert got == _ids(lambda c: c['sc'] in CLU_SCS and c['metric'] == metric)
    assert got                                              # AMI at 20 included for ami, retrieval never


@pytest.mark.parametrize('set_,aspect', sorted(op.CLASS_COUNT))
def test_r17_count_gate_voids_exactly_the_class_count_cells_of_its_set_and_aspect(set_, aspect):
    got = _voided_ids(lambda g: g['iii_count'].update({f'{set_}/{aspect}': 'failed'}))
    want = _ids(lambda c: c['sc'] in CLU_SCS and (c['set'], c['aspect']) == (set_, aspect) and c['k'] == op.CLASS_COUNT[(set_, aspect)])
    assert got == want and got
    assert not any(i.split('/')[5] == '20' for i in got) or (set_, aspect) == ('mmart', 'style')     # AMI at 20 is not a class count


def test_r17_class_count_gate_i_d_voids_the_class_count_cells_everywhere():
    got = _voided_ids(lambda g: g.update(i_d='failed'))
    assert got == _ids(lambda c: c['sc'] in CLU_SCS and c['k'] == op.CLASS_COUNT[(c['set'], c['aspect'])])
    assert any(i.split('/')[1:5] == ['mmart', 'qwen', 'style', 'ami'] and i.split('/')[5] == '20' for i in got)     # MMArt style: 20 is the count
    assert not any(i.split('/')[1] == 'artelingo' and i.split('/')[5] == '20' for i in got)


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
    raw = [raw_cell(s) for s in specs]
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


def _rebuild_of(cells):
    return lambda: copy.deepcopy(cells)


def test_labels_step_refuses_without_c13_and_on_disagreement(tmp_path):
    cells, picks = _cells_for_c13(), good_picks()
    specs = [{'id': c['id']} for c in cells]
    cells_path, c13_path, out_path = tmp_path / 'cells.json', tmp_path / 'c13' / 'c13_cells.json', tmp_path / 'rec' / 'labels.json'
    cells_path.write_text(json.dumps(cells))
    with pytest.raises(SystemExit) as e:
        oc.run_labels(cells_path, c13_path, out_path, picks, _rebuild_of(cells), expected=specs)
    assert 'c13_cells.json' in str(e.value) and not out_path.exists()
    c13_path.parent.mkdir()
    doc = _c13_doc(cells, picks)
    doc['cells'][2]['hi'] += 2e-6
    c13_path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        oc.run_labels(cells_path, c13_path, out_path, picks, _rebuild_of(cells), expected=specs)
    assert not out_path.exists()
    doc = _c13_doc(cells, picks)
    c13_path.write_text(json.dumps(doc))
    oc.run_labels(cells_path, c13_path, out_path, picks, _rebuild_of(cells), expected=specs)
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
        oc.run_labels(cells_path, c13_path, tmp_path / 'labels.json', picks, _rebuild_of(cells), expected=[{'id': c['id']} for c in cells])


@pytest.mark.parametrize('smoke', [0, 1])
def test_labels_stub_only_allowed_in_smoke(tmp_path, monkeypatch, smoke):
    monkeypatch.setattr(op, 'SMOKE', smoke)
    cells, picks = _cells_for_c13(), good_picks()
    cells_path, c13_path, out = tmp_path / 'cells.json', tmp_path / 'c13.json', tmp_path / 'labels.json'
    cells_path.write_text(json.dumps(cells))
    c13_path.write_text(json.dumps(dict(_c13_doc(cells, picks), stub=True)))
    call = lambda: oc.run_labels(cells_path, c13_path, out, picks, _rebuild_of(cells), expected=cells)
    if smoke:
        assert call() == out
    else:
        with pytest.raises(SystemExit, match='stub'):
            call()
        assert not out.exists()


@pytest.mark.parametrize('reason', ['absent_c13', 'stub', 'partial', 'stale', 'unscored', 'disagrees'])
def test_labels_refusal_removes_an_older_record(tmp_path, monkeypatch, reason):
    monkeypatch.setattr(op, 'SMOKE', 0)
    cells, picks = _cells_for_c13(), good_picks()
    expected = copy.deepcopy(cells)
    fresh = copy.deepcopy(cells)
    c13 = _c13_doc(cells, picks)
    if reason == 'stub':
        c13['stub'] = True
    elif reason == 'partial':
        cells = cells[:-1]
    elif reason == 'stale':
        fresh[0]['r'] = 0.5
    elif reason == 'unscored':
        cells[0]['void'] = fresh[0]['void'] = 'not scored'
    elif reason == 'disagrees':
        c13['cells'][0]['d'] += 0.1
    cells_path, c13_path, out = tmp_path / 'cells.json', tmp_path / 'c13.json', tmp_path / 'labels.json'
    cells_path.write_text(json.dumps(cells))
    if reason != 'absent_c13':
        c13_path.write_text(json.dumps(c13))
    out.write_text('{"old": true}')
    with pytest.raises(SystemExit):
        oc.run_labels(cells_path, c13_path, out, picks, _rebuild_of(fresh), expected=expected)
    assert not out.exists()


def test_labels_step_refuses_a_cells_file_that_is_not_what_assemble_gives_now(tmp_path):
    cells, picks = _cells_for_c13(), good_picks()
    cells_path, c13_path = tmp_path / 'cells.json', tmp_path / 'c13_cells.json'
    cells_path.write_text(json.dumps(cells))
    c13_path.write_text(json.dumps(_c13_doc(cells, picks)))          # C13 agrees with the old file
    now = copy.deepcopy(cells)
    now[0]['r'] = 0.5                                                # a range changed since: assemble gives another file
    with pytest.raises(SystemExit, match='assemble'):
        oc.run_labels(cells_path, c13_path, tmp_path / 'labels.json', picks, _rebuild_of(now), expected=[{'id': c['id']} for c in cells])
    assert not (tmp_path / 'labels.json').exists()


def test_labels_step_refuses_an_empty_or_partial_cell_list(tmp_path):
    picks = good_picks()
    c13_path = tmp_path / 'c13_cells.json'
    for cells in ([], _cells_for_c13()[:2]):
        cells_path = tmp_path / 'cells.json'
        cells_path.write_text(json.dumps(cells))
        c13_path.write_text(json.dumps(_c13_doc(cells, picks)))      # C13 agrees with whatever is there
        with pytest.raises(SystemExit, match='expected'):
            oc.run_labels(cells_path, c13_path, tmp_path / 'labels.json', picks, _rebuild_of(cells))        # default: all 337
    assert not (tmp_path / 'labels.json').exists()


# ------------------------------------------------------------------------------------------------ numbers against r17's scorer
KEYS = ('cand', 'comp', 'd', 'lo', 'hi')


def _same(cell, direct):
    for key in KEYS:
        assert cell[key] == pytest.approx(direct[key], abs=1e-12, rel=0), key


def test_bootstrap_seed_is_zero():
    assert op.BOOT_SEED == 0                  # the direct calls below name seed 0 themselves


def _clu_arms(ld, sc, set_, enc, aspect, k, text_set='A'):
    """The two partition arrays a clustering cell of `sc` must be scored on, asked of the stand-in directly."""
    if sc == 'SC1':
        return ld.cstar_parts(set_, enc, aspect, k), ld.tpips_parts(set_, aspect, k)
    if sc == 'SC3':
        return ld.cstar_parts(set_, enc, aspect, k), ld.clay_parts(set_, aspect, k, text_set)
    if sc == 'SC5':
        return np.repeat(ld.turtle_cstar(set_, enc, aspect, k), 3, axis=1), ld.turtle_plain(set_, enc, k)
    return ld.cstar_parts(set_, enc, aspect, k), ld.turtle_plain(set_, enc, k)


@pytest.mark.parametrize('sc,set_,enc', [('SC1', 'artelingo', 'qwen'), ('SC3', 'artelingo', 'clip_l14'), ('SC3', 'mmart', 'qwen'),
                                         ('SC5', 'artelingo', 'siglip2'), ('SC5', 'mmart', 'qwen'), ('SC6', 'artelingo', 'qwen'),
                                         ('SC6', 'mmart', 'clip_l14')])
def test_clustering_cells_equal_r17s_paired_cell(patch_sha, sc, set_, enc):
    ld = FakeLoaders()
    picks = good_picks()
    picks['clay_textset/clu/mmart'] = {'set': 'B', 'scores': {}, 'rule': 'x'}
    raw = oc.score_unit(sc, set_, enc, ld, picks=picks, draws=24)
    ts = 'B' if set_ == 'mmart' else 'A'
    assert len(raw) > 10
    for cell in raw:
        cand, comp = _clu_arms(ld, sc, set_, enc, cell['aspect'], cell['k'], ts)
        y = ld.labels(set_, cell['aspect'])
        direct = ns_clu.paired_cell(cand, comp, y, cell['k'], r=None, metrics=(cell['metric'],), draws=24, rng_seed=0)[cell['metric']]
        _same(cell, direct)
        assert cell['n_draws'] == 24 and cell['n_rows'] == len(y)


@pytest.mark.parametrize('sc,set_,enc,comp_arm,pick,ts', [('SC2', 'artelingo', 'qwen', 'tpips', 'center+k10', None),
                                                          ('SC2', 'mmart', 'qwen', 'tpips', 'center+k10', None),
                                                          ('SC4', 'artelingo', 'clip_l14', 'clay', 'whiten49+k0', 'A'),
                                                          ('SC4', 'mmart', 'qwen', 'clay', 'whiten49+k0', 'A')])
def test_retrieval_cells_equal_r17s_paired_ret(patch_sha, sc, set_, enc, comp_arm, pick, ts):
    ld = FakeLoaders()
    raw = oc.score_unit(sc, set_, enc, ld, picks=good_picks(), draws=60)
    assert len(raw) == 3
    for cell in raw:
        cand = ld.ap('cstar', set_, enc, cell['aspect'], f'reused-cstar-{set_}-{enc}')['ap']
        comp = ld.ap(comp_arm, set_, enc, cell['aspect'], pick, ts)['ap']
        assert cand.shape[0] == 5 and comp.ndim == 1
        _same(cell, ns_ret.paired_ret(cand, comp, r=None, draws=60, comp_seeds='none'))
        assert cell['n_draws'] == 60 and cell['n_queries'] == comp.shape[0]


def test_retrieval_numbers_are_not_a_bootstrap_of_another_seed_or_order(patch_sha):
    ld = FakeLoaders()
    cell = oc.score_unit('SC2', 'artelingo', 'qwen', ld, picks=good_picks(), draws=60)[0]
    cand = ld.ap('cstar', 'artelingo', 'qwen', cell['aspect'], 'reused-cstar-artelingo-qwen')['ap']
    comp = ld.ap('tpips', 'artelingo', 'qwen', cell['aspect'], 'center+k10')['ap']
    other = ns_ret.paired_ret(cand, comp, r=None, draws=60, comp_seeds='none', rng_seed=1)
    assert (cell['lo'], cell['hi']) != (other['lo'], other['hi'])          # the guarded seed matters here
    rev = ns_ret.paired_ret(cand[..., ::-1], comp, r=None, draws=60, comp_seeds='none')
    assert (cell['lo'], cell['hi']) != (rev['lo'], rev['hi'])


class PerfectLoaders(FakeLoaders):
    """Candidate partitions equal the labels, every comparator is one cluster; cstar AP is 0.8, the comparators' 0.2."""

    def _perfect(self, set_, aspect, k, shape):
        y = np.maximum(self._y(set_, aspect), 0) % k
        return np.broadcast_to(y, shape + (self.n,)).astype(np.int32).copy()

    def cstar_parts(self, set_, enc, aspect, k):
        return self._perfect(set_, aspect, k, (5, 3))

    def tpips_parts(self, set_, aspect, k):
        return np.zeros((3, self.n), np.int32)

    def turtle_plain(self, set_, enc, k):
        return np.zeros((3, self.n), np.int32)

    def ap(self, arm, set_, enc, aspect, pick, text_set=None):
        nq = int((self._y(set_, aspect) >= 0).sum())
        a = np.full((5, nq) if arm == 'cstar' else (nq,), 0.8 if arm == 'cstar' else 0.2)
        return {'ap': a, 'rows': np.arange(nq, dtype=np.int64), 'point': float(a.mean()), 'per_seed': None, 'dropped': 0}


@pytest.mark.parametrize('sc,set_,enc', [('SC1', 'artelingo', 'qwen'), ('SC6', 'mmart', 'siglip1')])
def test_clustering_hand_value_perfect_candidate_against_one_cluster(patch_sha, sc, set_, enc):
    raw = oc.score_unit(sc, set_, enc, PerfectLoaders(), picks=good_picks(), draws=12)
    for c in raw:
        if c['metric'] in ('ari', 'nmi', 'ami') and c['k'] == op.CLASS_COUNT[(set_, c['aspect'])]:     # perfect scores 1, one cluster 0
            assert (c['cand'], c['comp'], c['d'], c['lo'], c['hi']) == pytest.approx((1.0, 0.0, 1.0, 1.0, 1.0)), c['id']


@pytest.mark.parametrize('sc,set_,enc', [('SC2', 'artelingo', 'qwen'), ('SC4', 'mmart', 'clip_l14')])
def test_retrieval_hand_value_constant_ap(patch_sha, sc, set_, enc):
    raw = oc.score_unit(sc, set_, enc, PerfectLoaders(), picks=good_picks(), draws=30)
    for c in raw:
        assert (c['cand'], c['comp'], c['d'], c['lo'], c['hi']) == pytest.approx((0.8, 0.2, 0.6, 0.6, 0.6))


def test_sc4_reads_the_retrieval_text_set_and_sc3_the_clustering_one(patch_sha):
    picks = good_picks()
    picks['clay_textset/clu/mmart'] = {'set': 'B', 'scores': {}, 'rule': 'x'}           # clustering chose B, retrieval A
    ld = FakeLoaders()
    raw4 = oc.score_unit('SC4', 'mmart', 'qwen', ld, picks=picks, draws=5)
    assert {c['text_set'] for c in raw4} == {'A'}
    assert {call[6] for call in ld.calls if call[:3] == ('ap', 'clay', 'mmart')} == {'A'}
    raw3 = oc.score_unit('SC3', 'mmart', 'qwen', ld, picks=picks, draws=5)
    assert {c['text_set'] for c in raw3} == {'B'}
    picks['clay_textset/clu/mmart'] = {'set': 'A', 'scores': {}, 'rule': 'x'}           # and the other way round
    picks['clay_textset/ret/mmart'] = {'set': 'B', 'scores': {}, 'rule': 'x'}
    picks['clay/mmart']['text_set'] = 'B'
    ld = FakeLoaders()
    assert {c['text_set'] for c in oc.score_unit('SC4', 'mmart', 'qwen', ld, picks=picks, draws=5)} == {'B'}
    assert {c['text_set'] for c in oc.score_unit('SC3', 'mmart', 'qwen', ld, picks=picks, draws=5)} == {'A'}


def test_class_count_groups_assert_the_labels_have_that_many_classes(patch_sha):
    class L(FakeLoaders):
        def labels(self, set_, aspect):
            y = super().labels(set_, aspect)
            if (set_, aspect) == ('artelingo', 'emotion'):
                y[y == 8] = -1                                    # 8 classes left, the count is 9
            return y

    with pytest.raises(AssertionError, match='classes'):
        oc.score_unit('SC6', 'artelingo', 'qwen', L(), picks=good_picks(), draws=5)
    oc.score_unit('SC6', 'artelingo', 'qwen', FakeLoaders(), picks=good_picks(), draws=5)        # whole labels: fine


def test_sentence_needs_an_upper_bound_below_zero_for_below():
    sp = spec_of('SC1', metric='ami', k=9)
    c = assemble_specs([sp], lo=-0.05, hi=0.0, d=-0.02)[0]                  # hi exactly 0: not below
    assert c['label'] == 'BELOW' and c['sentence'] is None
    c = assemble_specs([sp], lo=-0.05, hi=-1e-9, d=-0.02)[0]
    assert c['sentence'] is not None and 'better than c*' in c['sentence']


# ------------------------------------------------------------------------------------------------ stale parts, statuses
def test_stale_parts_are_refused_at_assembly(monkeypatch):
    sp = spec_of('SC2', metric='map')
    good = raw_cell(sp)
    run = lambda raw, picks=None: oc.assemble([raw], ranges_record(), picks or good_picks(), ok_status(), ok_r17(), expected=[sp])
    assert run(good)[0]['label'] == 'WIN'
    with pytest.raises(AssertionError, match='n_draws'):
        run(dict(good, n_draws=5))
    with pytest.raises(AssertionError, match='comparator_readout'):
        run(dict(good, comparator_readout='none+k0'))
    with pytest.raises(AssertionError, match='candidate_readout'):
        run(dict(good, candidate_readout='old'))
    picks = good_picks()
    picks['tpips/artelingo'] = {'pick': 'whiten49+k10'}             # the stage 1 pick moved after the parts were written
    with pytest.raises(AssertionError, match='comparator_readout'):
        run(good, picks)
    monkeypatch.setattr(op, 'SMOKE', 1)                              # smoke draws are allowed other counts
    assert run(dict(good, n_draws=5))[0]['label'] == 'WIN'
    with pytest.raises(AssertionError, match='comparator_readout'):
        run(dict(good, comparator_readout='none+k0'))               # but never another read-out


def test_stale_text_set_is_refused_at_assembly():
    for sc, key in (('SC3', 'clay_textset/clu/artelingo'), ('SC4', 'clay_textset/ret/artelingo')):
        sp = spec_of(sc, enc='qwen', metric='map' if sc == 'SC4' else 'ami', k=None if sc == 'SC4' else 9)
        picks = good_picks()
        picks[key] = {'set': 'B', 'scores': {}, 'rule': 'x'}
        with pytest.raises(AssertionError, match='text_set'):
            oc.assemble([raw_cell(sp)], ranges_record(), picks, ok_status(), ok_r17(), expected=[sp])


def test_unscored_cells_need_no_freshness():
    sp = spec_of('SC2', metric='map')
    gone = dict(raw_cell(sp, ), cand=None, comp=None, d=None, lo=None, hi=None, n_draws=None, missing='FileNotFoundError: x')
    assert oc.assemble([gone], ranges_record(), good_picks(), ok_status(), ok_r17(), expected=[sp])[0]['label'] is None


def test_range_status_not_ok_makes_r_missing_whatever_the_record_says():
    sp = spec_of('SC1', metric='ami', k=9)
    rk = oc.range_key(sp)
    for bad in ('failed', 'missing'):
        st = ok_status()
        st['stage1'][rk] = bad
        c = assemble_specs([sp], status=st, lo=-0.001)[0]
        assert c['r'] is None and c['label'] == 'inconclusive' and c['void'] is None, bad
        assert assemble_specs([sp], status=st, lo=0.3)[0]['label'] == 'WIN'      # a WIN stands without r
    st = ok_status()
    st['stage1'][rk] = 'ok'
    assert assemble_specs([sp], status=st, lo=-0.001)[0]['label'] == 'LEVEL'
    st['stage1'][oc.range_key(spec_of('SC1', metric='ari', k=9))] = 'failed'     # another cell's range: this one keeps its r
    assert assemble_specs([sp], status=st, lo=-0.001)[0]['r'] == 0.01


def test_status_keys_outside_the_picks_and_ranges_are_refused():
    st = ok_status()
    st['stage1']['tpips/artelingo'] = 'ok'
    st['stage1']['clay_textset/clu/mmart'] = 'failed'
    st['stage1'][oc.range_key(spec_of('SC1', metric='ami', k=9))] = 'ok'
    oc.check_status(st)
    for bad in ('tpip/artelingo', 'artelingo/qwen/emotion/ami/10', 'clay_textset/mmart', 'tpips/emoset'):
        st = ok_status()
        st['stage1'][bad] = 'failed'
        with pytest.raises(AssertionError, match='keys no cell uses'):
            oc.check_status(st)


# ------------------------------------------------------------------------------------------------ the cells step
def _parts(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw))


def test_unit_with_missing_cells_is_scored_again_but_a_complete_one_is_skipped(tmp_path):
    sp = spec_of('SC1', metric='ami', k=9)
    todo = [('SC1', 'artelingo', 'qwen')]
    path = oc.parts_dir(tmp_path) / 'SC1__artelingo__qwen.json'
    calls = []
    score = lambda sc, set_, enc, ld, picks, draws: calls.append(sc) or [raw_cell(sp)]
    _parts(path, [dict(raw_cell(sp), cand=None, d=None, lo=None, hi=None, missing='FileNotFoundError: no store')])
    assert oc.run_cells(todo, {}, None, 5, False, out=tmp_path, score=score) == [] and calls == ['SC1']      # input arrived
    assert json.loads(path.read_text())[0]['missing'] is None
    oc.run_cells(todo, {}, None, 5, False, out=tmp_path, score=score)
    assert calls == ['SC1']                                         # done: skipped
    oc.run_cells(todo, {}, None, 5, True, out=tmp_path, score=score)
    assert calls == ['SC1', 'SC1']                                  # --force


def test_one_failing_unit_does_not_stop_the_others_and_the_run_fails_at_the_end(tmp_path, capsys):
    sp = spec_of('SC1', metric='ami', k=9)
    todo = [('SC1', 'artelingo', 'qwen'), ('SC2', 'artelingo', 'qwen'), ('SC3', 'artelingo', 'qwen')]

    def score(sc, set_, enc, ld, picks, draws):
        if sc == 'SC2':
            raise AssertionError('partial store at unit 3')
        return [raw_cell(sp)]

    failed = oc.run_cells(todo, {}, None, 5, False, out=tmp_path, score=score)
    assert failed == [('SC2', 'artelingo', 'qwen')]
    names = sorted(p.name for p in oc.parts_dir(tmp_path).iterdir())
    assert names == ['SC1__artelingo__qwen.json', 'SC3__artelingo__qwen.json']          # no file for the failed unit
    assert 'FAILED SC2/artelingo/qwen' in capsys.readouterr().err


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    """OUT and the records folder in tmp; committed records served from a dict (the real function is tested apart)."""
    records, out = tmp_path / 'records', tmp_path / 'out'
    records.mkdir()
    monkeypatch.setattr(op, 'RECORDS', records)
    monkeypatch.setattr(op, 'OUT', out)
    monkeypatch.setattr(op, 'SMOKE', 0)
    store = {}
    monkeypatch.setattr(op, 'read_committed_record', lambda name: copy.deepcopy(store[name]))
    return type('Env', (), dict(tmp=tmp_path, records=records, out=out, store=store))


def test_cmd_cells_exit_code_is_nonzero_when_a_unit_fails_and_the_rest_is_written(run_env, monkeypatch):
    sp = spec_of('SC1', metric='ami', k=9)
    run_env.store.update({'stage1_om_ranges.json': ranges_record(), 'stage1_om_picks.json': good_picks()})
    monkeypatch.setattr(oc, 'Loaders', lambda: None)

    def score(sc, set_, enc, ld, picks, draws):
        if (sc, set_, enc) == ('SC2', 'artelingo', 'qwen'):
            raise AssertionError('boom')
        return [raw_cell(sp)]

    monkeypatch.setattr(oc, 'score_unit', score)
    args = oc.argparse.Namespace(sc=None, set='artelingo', enc='qwen', draws=5, force=False)
    assert oc.cmd_cells(args) == 1
    names = sorted(p.name for p in oc.parts_dir().iterdir())
    assert names == [f'{sc}__artelingo__qwen.json' for sc in ('SC1', 'SC3', 'SC4', 'SC5', 'SC6')]       # all but the failed SC2
    assert oc.cmd_cells(oc.argparse.Namespace(sc='SC1', set='artelingo', enc='qwen', draws=5, force=False)) == 0


def _git_repo(tmp_path, monkeypatch, committed, uncommitted):
    import subprocess
    repo = (tmp_path / 'repo').resolve()
    (repo / 'records').mkdir(parents=True)
    run = lambda *a: subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', *a], check=True,
                                    capture_output=True)
    run('init', '-q')
    for name, rec in committed.items():
        (repo / 'records' / name).write_text(json.dumps(rec))
    run('add', '-A')
    run('commit', '-q', '--allow-empty', '-m', 'x')
    for name, rec in uncommitted.items():
        (repo / 'records' / name).write_text(json.dumps(rec))
    monkeypatch.setattr(op, 'REPO', repo)
    monkeypatch.setattr(op, 'RECORDS', repo / 'records')
    monkeypatch.setattr(op, 'SMOKE', 0)
    monkeypatch.setattr(op, 'OUT', tmp_path / 'out')


@pytest.mark.parametrize('committed', ['picks', 'ranges', 'neither'])
def test_cmd_cells_refuses_uncommitted_stage1_records_and_scores_nothing(tmp_path, monkeypatch, committed):
    recs = {'stage1_om_ranges.json': ranges_record(), 'stage1_om_picks.json': good_picks()}
    keep = {'picks': ['stage1_om_picks.json'], 'ranges': ['stage1_om_ranges.json'], 'neither': []}[committed]
    _git_repo(tmp_path, monkeypatch, {k: recs[k] for k in keep}, {k: v for k, v in recs.items() if k not in keep})

    def boom(*a, **k):
        raise AssertionError('scored although stage 1 is not committed')

    monkeypatch.setattr(oc, 'score_unit', boom)
    monkeypatch.setattr(oc, 'Loaders', boom)
    with pytest.raises(RuntimeError, match='not committed'):
        oc.cmd_cells(oc.argparse.Namespace(sc=None, set=None, enc=None, draws=5, force=False))
    assert not oc.parts_dir().exists()


def test_cmd_cells_runs_when_stage1_is_committed(tmp_path, monkeypatch):
    recs = {'stage1_om_ranges.json': ranges_record(), 'stage1_om_picks.json': good_picks()}
    _git_repo(tmp_path, monkeypatch, recs, {})
    seen = []
    monkeypatch.setattr(oc, 'Loaders', lambda: None)
    monkeypatch.setattr(oc, 'score_unit', lambda sc, set_, enc, ld, picks, draws: seen.append((sc, set_, enc)) or [])
    assert oc.cmd_cells(oc.argparse.Namespace(sc='SC2', set='mmart', enc='qwen', draws=5, force=False)) == 0
    assert seen == [('SC2', 'mmart', 'qwen')]


# ------------------------------------------------------------------------------------------------ the labels step, whole
@pytest.fixture
def labelled(run_env, monkeypatch):
    """Three cells in three units, parts written, records served; `labels` can run against it."""
    specs = [spec_of('SC1', metric='ami', k=9), spec_of('SC2', metric='map'),
             spec_of('SC4', set_='mmart', enc='qwen', aspect='genre')]
    monkeypatch.setattr(oc, 'cell_specs', lambda: specs)
    r17_path = run_env.tmp / 'r17_status.json'
    r17_path.write_text(json.dumps(ok_r17()))
    status = ok_status()
    status['r17'] = {'path': str(r17_path), 'sha256': op.sha256_file(r17_path)}
    run_env.store.update({'stage1_om_ranges.json': ranges_record(), 'stage1_om_picks.json': good_picks(), 'status.json': status})
    for sp in specs:
        _parts(oc.parts_dir() / f"{sp['sc']}__{sp['set']}__{sp['encoder']}.json", [raw_cell(sp)])
    run_env.specs = specs
    run_env.cells_path = run_env.out / 'stage2' / 'cells.json'
    run_env.c13_path = run_env.out / 'c13' / 'c13_cells.json'
    run_env.labels = run_env.records / 'labels.json'
    return run_env


def _write_c13(env, mutate=None):
    cells = json.loads(env.cells_path.read_text())
    doc = _c13_doc(cells, good_picks())
    if mutate:
        mutate(doc)
    env.c13_path.parent.mkdir(parents=True, exist_ok=True)
    env.c13_path.write_text(json.dumps(doc))


def test_cmd_assemble_then_cmd_labels_writes_the_record(labelled):
    assert oc.cmd_assemble(None) == 0
    _write_c13(labelled)
    assert oc.cmd_labels(None) == 0
    rec = json.loads(labelled.labels.read_text())
    assert [c['id'] for c in rec['cells']] == [s['id'] for s in labelled.specs]
    assert {c['label'] for c in rec['cells']} == {'WIN'}


def test_cmd_labels_refuses_without_the_c13_file(labelled):
    oc.cmd_assemble(None)
    with pytest.raises(SystemExit, match='c13_cells.json'):
        oc.cmd_labels(None)
    assert not labelled.labels.exists()


def test_cmd_labels_refuses_when_c13_disagrees(labelled):
    oc.cmd_assemble(None)
    _write_c13(labelled, lambda doc: doc['cells'][0].update(label='LEVEL'))
    with pytest.raises(SystemExit, match='disagrees'):
        oc.cmd_labels(None)
    _write_c13(labelled, lambda doc: doc['cells'][1].update(lo=doc['cells'][1]['lo'] + 3e-6))
    with pytest.raises(SystemExit, match='disagrees'):
        oc.cmd_labels(None)
    _write_c13(labelled, lambda doc: doc['picks'].update({'tpips/artelingo': 'none+k0'}))
    with pytest.raises(SystemExit, match='disagrees'):
        oc.cmd_labels(None)
    assert not labelled.labels.exists()


def test_cmd_labels_refuses_a_cells_file_older_than_the_records(labelled):
    oc.cmd_assemble(None)
    _write_c13(labelled)
    status = labelled.store['status.json']                           # a gate fails after cells.json was written
    status['gates']['iv'] = 'failed'
    labelled.store['status.json'] = status
    with pytest.raises(SystemExit, match='assemble'):
        oc.cmd_labels(None)
    assert not labelled.labels.exists()
    ranges = ranges_record()                                         # or a range changes
    labelled.store['status.json']['gates']['iv'] = 'ok'
    ranges[oc.range_key(labelled.specs[0])]['r'] = 0.5
    labelled.store['stage1_om_ranges.json'] = ranges
    with pytest.raises(SystemExit, match='assemble'):
        oc.cmd_labels(None)


def test_cmd_labels_refuses_a_partial_cells_file(labelled):
    oc.cmd_assemble(None)
    cells = json.loads(labelled.cells_path.read_text())
    labelled.cells_path.write_text(json.dumps(cells[:2]))
    _write_c13(labelled)
    with pytest.raises(SystemExit):
        oc.cmd_labels(None)
    assert not labelled.labels.exists()


# ------------------------------------------------------------------------------------------------ Loaders on a stand-in om_data
import zlib  # noqa: E402

import ns_data  # noqa: E402
import ns_metrics  # noqa: E402

LN, LD = 30, 24
SPLITS = {'artelingo': 'held', 'mmart': 'test'}


def rows_of(*tag):
    """float32 unit rows [LN, LD], different for every tag: the test's own record of what each store holds."""
    rng = np.random.default_rng(zlib.crc32('|'.join(map(str, tag)).encode()))
    x = rng.standard_normal((LN, LD)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def labels_of(set_, aspect):
    y = (np.arange(LN) % 3).astype(np.int64)
    y[[2, 11]] = -1
    return y


class StubData:
    """The `om_data` calls Loaders makes, recording each call; rows come from `rows_of` keyed by what was asked for."""

    def __init__(self):
        self.calls = []
        self.fail_turtle = False

    def labels(self, set_, split, aspect):
        self.calls.append(('labels', set_, split, aspect))
        return labels_of(set_, aspect)

    def cstar(self, set_, split, enc, aspect, seed, unit=True):
        self.calls.append(('cstar', set_, split, enc, aspect, seed, unit))
        x = rows_of('cstar', set_, split, enc, aspect, seed)
        return x if unit else x * np.float32(3.0)

    def tpips(self, set_, split, aspect):
        self.calls.append(('tpips', set_, split, aspect))
        return rows_of('tpips', set_, split, aspect)

    def clay(self, set_, split, aspect, text_set):
        self.calls.append(('clay', set_, split, aspect, text_set))
        return rows_of('clay', set_, split, aspect, text_set)

    def turtle_parts(self, set_, enc, arm, k, aspect=None):
        self.calls.append(('turtle_parts', set_, enc, arm, k, aspect))
        if self.fail_turtle:
            raise FileNotFoundError('no fit')
        return np.full((3, LN) if arm == 'plain' else (5, 1, LN), k % 7, np.int32)


def fake_partitions(x, k, set_name):
    """Stands in for r10's k-means: three 'states' that depend on the rows, the k and the set (and record nothing)."""
    x = np.asarray(x)
    return np.stack([(np.argmax(x[:, :k], axis=1) + j + len(set_name)) % k for j in range(3)]).astype(np.int32)


@pytest.fixture
def stub(monkeypatch):
    sd = StubData()
    monkeypatch.setattr(oc, 'om_data', sd)
    monkeypatch.setattr(ns_data, 'cstar', sd.cstar)               # `ns_clu_cells._cand_rows` reads c* through ns_data
    monkeypatch.setattr(ns_metrics, 'partitions', fake_partitions)
    return sd


@pytest.mark.parametrize('set_', ['artelingo', 'mmart'])
def test_loaders_read_the_held_split_of_the_set(stub, set_):
    ld = oc.Loaders()
    y = ld.labels(set_, 'emotion')
    assert y.dtype == np.int64 and np.array_equal(y, labels_of(set_, 'emotion'))
    assert stub.calls == [('labels', set_, SPLITS[set_], 'emotion')]


@pytest.mark.parametrize('set_', ['artelingo', 'mmart'])
def test_loaders_cstar_partitions_are_the_five_seeds_of_the_held_split_in_clustering_form(stub, set_):
    got = oc.Loaders().cstar_parts(set_, 'qwen', 'genre', 10)
    calls = [c for c in stub.calls if c[0] == 'cstar']
    assert [c[5] for c in calls] == [0, 1, 2, 3, 4]               # seeds in order
    assert {c[2] for c in calls} == {SPLITS[set_]} and {c[6] for c in calls} == {False}        # held, not unit rows
    want = np.stack([fake_partitions(rows_of('cstar', set_, SPLITS[set_], 'qwen', 'genre', s) * np.float32(3.0), 10, set_)
                     for s in range(5)])
    assert got.shape == (5, 3, LN) and np.array_equal(got, want)


@pytest.mark.parametrize('set_', ['artelingo', 'mmart'])
def test_loaders_tpips_and_clay_partitions_are_k_means_of_the_held_rows(stub, set_):
    ld = oc.Loaders()
    got = ld.tpips_parts(set_, 'style', 9)
    assert np.array_equal(got, fake_partitions(rows_of('tpips', set_, SPLITS[set_], 'style'), 9, set_))
    got = ld.clay_parts(set_, 'style', 9, 'B')
    assert np.array_equal(got, fake_partitions(rows_of('clay', set_, SPLITS[set_], 'style', 'B'), 9, set_))
    assert [c for c in stub.calls if c[0] in ('tpips', 'clay')] == [('tpips', set_, SPLITS[set_], 'style'),
                                                                    ('clay', set_, SPLITS[set_], 'style', 'B')]


def test_loaders_turtle_arms_and_missing_fit(stub):
    ld = oc.Loaders()
    assert ld.turtle_plain('mmart', 'siglip1', 9).shape == (3, LN)
    assert ld.turtle_cstar('mmart', 'siglip1', 'genre', 9).shape == (5, 1, LN)
    assert [c for c in stub.calls if c[0] == 'turtle_parts'] == [('turtle_parts', 'mmart', 'siglip1', 'plain', 9, None),
                                                                 ('turtle_parts', 'mmart', 'siglip1', 'cstar', 9, 'genre')]
    stub.fail_turtle = True
    with pytest.raises(oc.MissingInput, match='no fit'):
        ld.turtle_plain('mmart', 'siglip1', 9)
    with pytest.raises(oc.MissingInput):
        ld.turtle_cstar('mmart', 'siglip1', 'genre', 9)


def _direct_ap(get_val, get_eval, pick, y, seeds):
    return ns_ret.arm_ap(get_val, get_eval, pick, y, seeds)


@pytest.mark.parametrize('set_', ['artelingo', 'mmart'])
def test_loaders_ap_fits_on_validation_scores_on_held_with_the_five_seeds_of_unit_rows(stub, set_):
    held, y = SPLITS[set_], labels_of(set_, 'genre')
    got = oc.Loaders().ap('cstar', set_, 'qwen', 'genre', 'center+k10')
    cs = [c for c in stub.calls if c[0] == 'cstar']
    assert {c[6] for c in cs} == {True}                              # unit rows (default), as r17's retrieval arm
    assert sorted({(c[2], c[5]) for c in cs}) == sorted({(sp, s) for sp in ('val', held) for s in range(5)})
    want = _direct_ap(lambda s: rows_of('cstar', set_, 'val', 'qwen', 'genre', s),
                      lambda s: rows_of('cstar', set_, held, 'qwen', 'genre', s), 'center+k10', y, (0, 1, 2, 3, 4))
    assert got['ap'].shape == want['ap'].shape == (5, int((y >= 0).sum()))
    assert np.array_equal(got['ap'], want['ap']) and np.array_equal(got['rows'], want['rows'])
    swapped = _direct_ap(lambda s: rows_of('cstar', set_, held, 'qwen', 'genre', s),
                         lambda s: rows_of('cstar', set_, 'val', 'qwen', 'genre', s), 'center+k10', y, (0, 1, 2, 3, 4))
    assert not np.array_equal(swapped['ap'], want['ap'])             # the order of the splits matters here


@pytest.mark.parametrize('set_', ['artelingo', 'mmart'])
@pytest.mark.parametrize('arm,ts', [('tpips', None), ('clay', 'B')])
def test_loaders_ap_of_the_unseeded_arms(stub, set_, arm, ts):
    held, y = SPLITS[set_], labels_of(set_, 'genre')
    got = oc.Loaders().ap(arm, set_, 'qwen', 'genre', 'center+k10', ts)
    tag = lambda split: (arm, set_, split, 'genre') + (() if ts is None else (ts,))
    want = _direct_ap(lambda s: rows_of(*tag('val')), lambda s: rows_of(*tag(held)), 'center+k10', y, None)
    assert got['ap'].ndim == 1 and np.array_equal(got['ap'], want['ap'])
    splits = [c[2] for c in stub.calls if c[0] == arm]
    assert set(splits) == {'val', held}
    swapped = _direct_ap(lambda s: rows_of(*tag(held)), lambda s: rows_of(*tag('val')), 'center+k10', y, None)
    assert not np.array_equal(swapped['ap'], want['ap'])


def test_labels_step_never_defaults_to_the_real_records_in_tests():
    # the labels path comes from om_paths.record_path; under OM_SMOKE it is OUT/records, else RECORDS (committed)
    assert oc.labels_path().name == 'labels.json'


def test_labels_step_refuses_a_cell_that_was_never_scored(tmp_path):
    cells, picks = _cells_for_c13(), good_picks()
    cells[1] = dict(cells[1], void='not scored', label=None, cand=None, comp=None, d=None, lo=None, hi=None)
    specs = [{'id': c['id']} for c in cells]
    cells_path, c13_path, out_path = tmp_path / 'cells.json', tmp_path / 'c13_cells.json', tmp_path / 'labels.json'
    cells_path.write_text(json.dumps(cells))
    c13_path.write_text(json.dumps(_c13_doc(cells, picks)))
    with pytest.raises(SystemExit) as e:
        oc.run_labels(cells_path, c13_path, out_path, picks, _rebuild_of(cells), expected=specs)
    assert 'never scored' in str(e.value) and not out_path.exists()


def test_a_failed_rerun_removes_the_older_parts_file(tmp_path):
    todo = [('SC1', 'artelingo', 'qwen')]
    path = oc.parts_dir(tmp_path) / 'SC1__artelingo__qwen.json'
    path.parent.mkdir(parents=True)
    path.write_text('[]')

    def score(*a):
        raise AssertionError('arms were scored on other queries')

    assert oc.run_cells(todo, {}, None, 5, True, out=tmp_path, score=score) == todo and not path.exists()
