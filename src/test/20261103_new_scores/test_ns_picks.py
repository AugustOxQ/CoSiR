"""Tests of ns_picks (r17 ticket 09). Logic tests use small fakes and a stand-in for the cross-fit scores; two read the
real recorded range files (milliseconds).

Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q
     src/test/20261103_new_scores/test_ns_picks.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import ns_picks  # noqa: E402
import ac_select  # noqa: E402
import mt_data  # noqa: E402
import rt_data  # noqa: E402
import rt_readout  # noqa: E402

ASPECTS = ns_paths.ASPECTS
N, WIDTH_UNTOLD = 40, 24


# ------------------------------------------------------------------------------------------------ fakes
class FakeData:
    """Stands in for ns_data on N validation rows; records every split it is asked for."""

    def __init__(self, set_order='ok'):
        self.splits, self.set_order = [], set_order
        self.rng = np.random.default_rng(0)

    def keys(self, set_, split):
        self.splits.append(split)
        names = [f'p{i:03d}' for i in range(N)]
        if set_ == 'artelingo':
            return sorted(names, reverse=self.set_order == 'bad')
        ordered = list(mt_data.order(names))
        return ordered[::-1] if self.set_order == 'bad' else ordered

    def labels(self, set_, split, aspect):
        self.splits.append(split)
        return np.arange(N, dtype=np.int64) % 3

    def _rows(self, d, split):
        self.splits.append(split)
        x = self.rng.normal(size=(N, d)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def untold(self, set_, split, enc, seed):
        return self._rows(WIDTH_UNTOLD, split)

    def percept_probs(self, set_, split, enc, seed):
        self.splits.append(split)
        p = np.abs(self.rng.normal(size=(N, ns_paths.N_TOPICS))).astype(np.float32)
        return p / p.sum(axis=1, keepdims=True)

    def crl(self, set_, split, aspect):
        return self._rows(ns_paths.N_CRL_VALUES, split)


@pytest.fixture
def small(monkeypatch):
    for key in (('artelingo', 'val'), ('mmart', 'val')):
        monkeypatch.setitem(ns_paths.N_ROWS, key, N)
    monkeypatch.setattr(ns_paths, 'UNTOLD_WIDTH', WIDTH_UNTOLD)


def fake_ranges(tmp_path, monkeypatch, r=0.01):
    """Recorded range files in r11's, r14's and r15's formats under tmp_path, with unit r (+ a distinct value per file)."""
    per = {s: r + 0.001 * s for s in ns_paths.SEEDS}                 # max - min = 0.004
    rng = 0.004

    def seeds():
        return {a: {str(s): per[s] for s in ns_paths.SEEDS} for a in ASPECTS}
    files = {}
    r11 = {'map': {a: {'r': rng, 'per_seed': [per[s] for s in ns_paths.SEEDS]} for a in ASPECTS}}
    (tmp_path / 'r11.json').write_text(json.dumps(r11))
    for kind, key in (('r14', 'artret_ranges'), ('r15', 'ret_ranges')):
        for enc in ns_paths.ENCODERS:
            rec = {key: {'encoder': enc, 'ranges': {a: rng for a in ASPECTS}, 'per_seed': seeds()}}
            (tmp_path / f'{kind}_{enc}.json').write_text(json.dumps(rec))

    def _range_file(set_, enc):
        if set_ == 'artelingo' and enc == 'qwen':
            return tmp_path / 'r11.json', 'r11'
        kind = 'r14' if set_ == 'artelingo' else 'r15'
        return tmp_path / f'{kind}_{enc}.json', kind
    monkeypatch.setattr(ns_picks, '_range_file', _range_file)
    return rng


# ------------------------------------------------------------------------------------------------ units (D17)
def test_units_are_the_recorded_ranges():
    r, src = ns_picks.ranges('artelingo', 'qwen')
    assert r['emotion'] == 0.0014116107779215126 and 'stage0a_ranges.json' in src          # r11, unrounded
    r, _ = ns_picks.ranges('mmart', 'clip_l14')
    assert r == {'emotion': 0.000633751040052366, 'style': 0.0038390840430087003, 'genre': 0.0019367407309804907}  # r15
    r, _ = ns_picks.ranges('artelingo', 'siglip2')
    assert r['genre'] == 0.004259334686874894                                                # r14


def test_units_come_from_the_record_not_from_scores(tmp_path, monkeypatch, small):
    """A recorded range is used as written: nothing is recomputed from the validation rows."""
    import rt_metrics
    monkeypatch.setattr(rt_metrics, 'retrieval', lambda *a, **k: pytest.fail('a range was recomputed'))
    rng = fake_ranges(tmp_path, monkeypatch)
    seen = {}

    def fake_crossfit(xs, labels, ros, workers=8):
        return {ro: {a: 0.2 + (0.0 if ro == rt_readout.R0 else 0.002) for a in ASPECTS} for ro in ros}

    def spy(xs, labels, ranges, **kw):
        seen['ranges'] = dict(ranges)
        return real(xs, labels, ranges, **kw)
    real = ac_select.arm_table
    monkeypatch.setattr(rt_readout, 'crossfit_all', fake_crossfit)
    monkeypatch.setattr(ac_select, 'arm_table', spy)
    rec = ns_picks.pick('matched', 'mmart', 'siglip2', out=tmp_path, data=FakeData())
    assert seen['ranges'] == {a: rng for a in ASPECTS} and rec['unit'] == seen['ranges']
    assert rec['scores']['score'] == pytest.approx(0.002 / rng)                    # (mAP - mAP(R0)) / r


def test_range_with_inconsistent_per_seed_refused(tmp_path, monkeypatch):
    fake_ranges(tmp_path, monkeypatch)
    rec = json.loads((tmp_path / 'r15_clip_l14.json').read_text())
    rec['ret_ranges']['ranges']['style'] = 0.5
    (tmp_path / 'r15_clip_l14.json').write_text(json.dumps(rec))
    with pytest.raises(AssertionError):
        ns_picks.ranges('mmart', 'clip_l14')


# ------------------------------------------------------------------------------------------------ validation only
def test_inputs_refuse_held_and_test_splits(small):
    for split in ('held', 'test'):
        with pytest.raises(AssertionError):
            ns_picks.inputs('untold', 'artelingo', 'qwen', split=split, data=FakeData())


def test_only_validation_is_requested(small):
    d = FakeData()
    for arm, set_, enc in (('untold', 'mmart', 'qwen'), ('matched', 'artelingo', 'siglip1'), ('crl', 'mmart', 'clip_l14')):
        ns_picks.inputs(arm, set_, enc, data=d)
    assert set(d.splits) == {'val'} and len(d.splits) > 10


def test_only_the_16_new_picks_exist():
    assert len(ns_picks.PICKS) == 16 and len(set(ns_picks.PICKS)) == 16
    assert ('matched', 'artelingo', 'qwen') not in ns_picks.PICKS                          # keeps r11's pick
    with pytest.raises(AssertionError):
        ns_picks.check_arm('crl', 'artelingo', 'clip_l14')                                 # keeps r11's pick
    keys = {ns_picks.key_of(*p) for p in ns_picks.PICKS}
    import ns_outcome
    assert keys == {ns_outcome.pick_key(c) for c in ns_outcome.claim_cells()} - {None}      # the keys the verdict code reads


# ------------------------------------------------------------------------------------------------ row order
def test_mmart_order_is_sha256_and_no_keys_passed(small, monkeypatch):
    calls = []
    real = ac_select.arm_table

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)
    monkeypatch.setattr(ac_select, 'arm_table', spy)
    xs, labels = ns_picks.inputs('crl', 'mmart', 'clip_l14', data=FakeData())
    ns_picks.readout_doc(xs, labels, {a: 0.01 for a in ASPECTS}, workers=1)
    assert calls and all('keys' not in kw for kw in calls)                                  # r15 trap 1


@pytest.mark.parametrize('set_', ['mmart', 'artelingo'])
def test_rows_out_of_order_refused(small, set_):
    with pytest.raises(AssertionError):
        ns_picks.inputs('matched', set_, 'siglip2', data=FakeData(set_order='bad'))


def test_mmart_sorted_order_is_refused(small):
    class Sorted(FakeData):
        def keys(self, set_, split):
            return sorted(super().keys(set_, split))
    assert list(mt_data.order(Sorted().keys('mmart', 'val'))) != Sorted().keys('mmart', 'val')   # sha256 order differs
    with pytest.raises(AssertionError):
        ns_picks.inputs('matched', 'mmart', 'siglip2', data=Sorted())


# ------------------------------------------------------------------------------------------------ input forms
def test_aspect_free_arms_share_one_array_per_seed(small):
    for arm in ('untold', 'matched'):
        xs, _ = ns_picks.inputs(arm, 'mmart', 'siglip2', data=FakeData())
        assert set(xs) == {(a, s) for a in ASPECTS for s in ns_paths.SEEDS}
        for s in ns_paths.SEEDS:
            assert xs[('emotion', s)] is xs[('style', s)] is xs[('genre', s)]
        assert xs[('emotion', 0)] is not xs[('emotion', 1)]
    xs, _ = ns_picks.inputs('crl', 'mmart', 'clip_l14', data=FakeData())
    assert set(xs) == {(a, None) for a in ASPECTS}


def test_percept_form_is_probabilities_over_their_float32_norm(small):
    d = FakeData()
    p = d.percept_probs('mmart', 'val', 'qwen', 0)
    want = (p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    assert np.array_equal(ns_picks.unit32(p), want) and ns_picks.unit32(p).dtype == np.float32
    assert not np.allclose(ns_picks.unit32(p), p)                                           # not the raw probabilities
    assert np.allclose(np.linalg.norm(ns_picks.unit32(p), axis=1), 1, atol=1e-6)
    d2 = FakeData()
    xs, _ = ns_picks.inputs('matched', 'mmart', 'qwen', data=d2)
    assert np.array_equal(xs[('emotion', 0)], ns_picks.unit32(FakeData().percept_probs('mmart', 'val', 'qwen', 0)))


def test_wrong_width_or_rows_refused(small):
    class Wide(FakeData):
        def crl(self, set_, split, aspect):
            return self._rows(49, split)
    with pytest.raises(AssertionError):
        ns_picks.inputs('crl', 'mmart', 'clip_l14', data=Wide())


# ------------------------------------------------------------------------------------------------ r11's rule
def _rule(monkeypatch, scores_in_r, r=0.01, base=0.2, k0_maps=None):
    """Run readout_doc with stand-in cross-fit mAPs: read-out name -> gain in units of r (all aspects alike), or a dict
    {aspect: gain}. Unlisted read-outs gain 0."""
    def fake_crossfit(xs, labels, ros, workers=8):
        out = {}
        for ro in ros:
            g = scores_in_r.get(rt_readout.name(ro), 0.0)
            g = g if isinstance(g, dict) else {a: g for a in ASPECTS}
            out[ro] = {a: base + g[a] * r for a in ASPECTS}
        return out
    monkeypatch.setattr(rt_readout, 'crossfit_all', fake_crossfit)
    xs = {(a, None): np.zeros((N, WIDTH_UNTOLD), np.float32) for a in ASPECTS}
    labels = {a: np.arange(N, dtype=np.int64) % 3 for a in ASPECTS}
    return ns_picks.readout_doc(xs, labels, {a: r for a in ASPECTS})


def test_tie_resolved_by_r11_order(monkeypatch):
    # none+k10 is within 0.001 (in units of r) of center+k10: the tie order takes none before center
    doc = _rule(monkeypatch, {'center+k10': 5.0, 'none+k10': 4.9995, 'center+k5': 4.0})
    assert doc['pick'] == 'none+k10'


def test_gap_beyond_tolerance_takes_the_best(monkeypatch):
    doc = _rule(monkeypatch, {'center+k10': 5.0, 'none+k10': 4.998})
    assert doc['pick'] == 'center+k10'


def test_smaller_k_wins_a_tie_and_no_expansion_first(monkeypatch):
    assert _rule(monkeypatch, {'center+k10': 3.0, 'center+k5': 3.0})['pick'] == 'center+k5'
    assert _rule(monkeypatch, {'center+k0': 3.0, 'center+k1': 3.0})['pick'] == 'center+k0'


def test_guard_drops_a_readout_that_loses_on_one_aspect(monkeypatch):
    # center+k10 has the best mean (10, 10, -3 -> 5.67) but loses 3 r on genre (more than r): dropped
    doc = _rule(monkeypatch, {'center+k10': {'emotion': 10.0, 'style': 10.0, 'genre': -3.0}, 'none+k10': 2.0})
    assert doc['table']['center+k10']['kept'] is False and doc['pick'] == 'none+k10'


def test_nothing_better_than_r0_picks_r0(monkeypatch):
    assert _rule(monkeypatch, {})['pick'] == 'none+k0'


def test_menu_at_the_arms_width(monkeypatch):
    names = set(_rule(monkeypatch, {})['table'])
    assert names == {rt_readout.name(ro) for ro in rt_readout.menu(WIDTH_UNTOLD)} and len(names) == 15


def test_rule_runs_on_real_crossfit_and_halves_follow_row_order(monkeypatch):
    """Without the stand-in: tiny real data. Picks are read-out names of the menu; scores equal (mAP - mAP(R0)) / r."""
    rng = np.random.default_rng(3)
    y = (np.arange(N) % 3).astype(np.int64)
    x = rng.normal(size=(N, WIDTH_UNTOLD)).astype(np.float32)
    x[:, 0] += 4.0 * y
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    labels = {a: y for a in ASPECTS}
    doc = ns_picks.readout_doc({(a, None): x for a in ASPECTS}, labels, {a: 0.01 for a in ASPECTS}, workers=1)
    base = doc['table']['none+k0']['map']
    for name, row in doc['table'].items():
        assert row['score'] == pytest.approx(np.mean([(row['map'][a] - base[a]) / 0.01 for a in ASPECTS]), abs=1e-12)
    assert doc['pick'] in doc['table']


# ------------------------------------------------------------------------------------------------ records
def test_pick_record_format_and_table_file(tmp_path, monkeypatch, small):
    fake_ranges(tmp_path, monkeypatch)
    monkeypatch.setattr(rt_readout, 'crossfit_all', lambda xs, labels, ros, workers=8: {
        ro: {a: 0.2 + (0.003 if rt_readout.name(ro) == 'whiten24+k3' else 0.0) for a in ASPECTS} for ro in ros})
    rec = ns_picks.pick('untold', 'artelingo', 'siglip1', out=tmp_path, data=FakeData())
    assert rec['pick'] == 'whiten24+k3' and rec['w'] == 'whiten24' and rec['k'] == 3
    assert set(rec) >= {'pick', 'w', 'k', 'scores', 'unit', 'table_path'}
    assert set(rec['scores']['map']) == set(ASPECTS) and rec['scores']['score'] == pytest.approx(0.75)
    table = json.loads((tmp_path / rec['table_path']).read_text())
    assert table['pick'] == 'whiten24+k3' and len(table['table']) == 15


def test_write_pick_merges_arms(tmp_path):
    ns_picks.write_pick('crl/mmart/clip_l14', {'pick': 'none+k10'}, out=tmp_path)
    path = ns_picks.write_pick('untold/mmart/qwen', {'pick': 'center+k10'}, out=tmp_path)
    ns_picks.write_pick('crl/mmart/clip_l14', {'pick': 'center+k0'}, out=tmp_path)           # a rerun replaces its own key only
    assert json.loads(path.read_text()) == {'crl/mmart/clip_l14': {'pick': 'center+k0'}, 'untold/mmart/qwen': {'pick': 'center+k10'}}
    with pytest.raises(AssertionError):
        ns_picks.write_pick('crl/artelingo/clip_l14', {}, out=tmp_path)


# ------------------------------------------------------------------------------------------------ gate (vi)
class R11Fake:
    """A stand-in for rt_data: two arms, one seeded and aspect-free (percept), one not (crl), on N validation rows."""
    ARMS = ('percept', 'crl')
    SEEDED = {'percept'}
    ASPECT_FREE = {'percept'}

    @staticmethod
    def labels(split):
        assert split == 'val'
        return {a: np.arange(N, dtype=np.int64) % 3 for a in ASPECTS}

    @staticmethod
    def embed(arm, split, aspect, seed=None):
        assert split == 'val'
        d = 8 if arm == 'percept' else 12
        x = np.random.default_rng(100 + (seed or 0) + 7 * (arm == 'crl') + 3 * ASPECTS.index(aspect)).normal(size=(N, d))
        x[:, 0] += 3.0 * (np.arange(N) % 3)
        x = x.astype(np.float32)
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def stored(tmp_path):
    """r11-style stage4 files made by the rule on R11Fake, and a ranges file."""
    r_a = {a: 0.01 for a in ASPECTS}
    scores, picks = {}, {}
    for arm in R11Fake.ARMS:
        xs = {}
        for s in (ns_paths.SEEDS if arm in R11Fake.SEEDED else (None,)):
            for a in ASPECTS:
                xs[(a, s)] = R11Fake.embed(arm, 'val', 'emotion' if arm in R11Fake.ASPECT_FREE else a, s)
        doc = ns_picks.readout_doc(xs, {a: v for a, v in R11Fake.labels('val').items()}, r_a, workers=1)
        scores[arm], picks[arm] = doc['table'], doc['pick']
    (tmp_path / 'scores.json').write_text(json.dumps(scores))
    (tmp_path / 'picks.json').write_text(json.dumps(picks))
    (tmp_path / 'ranges.json').write_text(json.dumps({'map': {a: {'r': 0.01} for a in ASPECTS}}))
    return tmp_path


def _gate(stored, arms=None):
    return ns_picks.gate_vi(arms, workers=1, data=R11Fake, stored_dir=stored, ranges_file=stored / 'ranges.json')


def test_gate_vi_passes_on_matching_stored_outputs(stored):
    res = _gate(stored)
    assert res['pass'] is True and res['max_abs_diff'] <= ns_paths.TOL_REPLAY and res['complete']
    assert res['picks'] == json.loads((stored / 'picks.json').read_text())


def test_gate_vi_fails_on_a_changed_score(stored):
    s = json.loads((stored / 'scores.json').read_text())
    s['percept']['center+k10']['score'] += 1e-6
    (stored / 'scores.json').write_text(json.dumps(s))
    res = _gate(stored)
    assert res['pass'] is False and res['max_abs_diff'] == pytest.approx(1e-6, rel=0.01)


def test_gate_vi_fails_on_a_changed_pick(stored):
    p = json.loads((stored / 'picks.json').read_text())
    p['crl'] = 'whiten12+k1' if p['crl'] != 'whiten12+k1' else 'none+k0'
    (stored / 'picks.json').write_text(json.dumps(p))
    assert _gate(stored)['pass'] is False


def test_gate_vi_fails_on_a_changed_guard_flag(stored):
    s = json.loads((stored / 'scores.json').read_text())
    s['crl']['center+k3']['kept'] = not s['crl']['center+k3']['kept']
    (stored / 'scores.json').write_text(json.dumps(s))
    res = _gate(stored)
    assert res['pass'] is False and res['max_abs_diff'] <= ns_paths.TOL_REPLAY            # only the flag differs


def test_gate_vi_on_a_subset_of_arms_cannot_pass(stored):
    res = _gate(stored, arms=('crl',))
    assert res['values_ok'] is True and res['pass'] is False and res['complete'] is False


def test_gate_vi_fed_the_real_r11_arm_set():
    assert set(rt_data.ARMS) == {'cstar', 'plain', 'instructed', 'clip', 'siglip2', 'percept', 'crl'}
    assert ns_picks.gate_vi.__defaults__[0] is None                                          # default: every arm
