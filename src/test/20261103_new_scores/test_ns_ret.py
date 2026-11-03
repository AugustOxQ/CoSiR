"""Tests of ns_ret (r17 ticket 08): the paired bootstrap, the arm scorer and the cell assembly. Synthetic data only
(no stores read); the replays of gate (ii) read real stores and run through the CLI, not here.

Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q
     src/test/20261103_new_scores/test_ns_ret.py
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import ns_data  # noqa: E402
import ns_outcome  # noqa: E402
import ns_ret  # noqa: E402
import rt_metrics  # noqa: E402
import rt_readout  # noqa: E402
import rt_stats  # noqa: E402

DRAWS = 300


def ap(shape, seed):
    return np.random.default_rng(seed).uniform(0.1, 0.9, size=shape)


def reference_boot(cand, comp, draws, mode):
    """Plain loop, written from the rule: one default_rng(0); per draw I, then c*'s seeds, then (apart) the
    comparator's seeds (shared: the same seeds for both)."""
    rng = np.random.default_rng(0)
    n = cand.shape[-1]
    out = np.empty(draws)
    for b in range(draws):
        I = rng.integers(0, n, size=n)
        sc = rng.choice([0, 1, 2, 3, 4], size=5, replace=True) if cand.ndim == 2 else None
        if comp.ndim == 2:
            sm = rng.choice([0, 1, 2, 3, 4], size=5, replace=True) if mode == 'apart' else sc
        mc = cand[sc][:, I].mean() if cand.ndim == 2 else cand[I].mean()
        mm = comp[sm][:, I].mean() if comp.ndim == 2 else comp[I].mean()
        out[b] = mc - mm
    return out


# ------------------------------------------------------------------------------------------------ paired_ret
@pytest.mark.parametrize('seeded_cand', [True, False])
def test_none_equals_rt_stats_paired_bit_for_bit(seeded_cand):
    cand = ap((5, 120) if seeded_cand else (120,), 1)
    comp = ap((120,), 2)
    ref = rt_stats.paired(cand, comp, DRAWS)
    got = ns_ret.paired_ret(cand, comp, r=0.01, draws=DRAWS, comp_seeds='none')
    assert got['d'] == ref['d']
    assert (got['lo'], got['hi']) == tuple(ref['interval'])
    assert got['cand'] == cand.mean() and got['comp'] == comp.mean()
    assert got['n_draws'] == DRAWS and got['r'] == 0.01


def test_none_other_rng_seed_uses_the_same_loop():
    cand, comp = ap((5, 90), 3), ap((90,), 4)
    a = ns_ret.paired_ret(cand, comp, draws=DRAWS, comp_seeds='none', rng_seed=0)
    b = ns_ret.paired_ret(cand, comp, draws=DRAWS, comp_seeds='none', rng_seed=7)
    assert (a['lo'], a['hi']) != (b['lo'], b['hi'])
    # rng_seed 7, checked against a plain loop with default_rng(7)
    rng = np.random.default_rng(7)
    boot = []
    for _ in range(DRAWS):
        I = rng.integers(0, 90, size=90)
        S = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        boot.append(cand[S][:, I].mean() - comp[I].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert abs(b['lo'] - lo) < 1e-12 and abs(b['hi'] - hi) < 1e-12


def test_shared_equals_rt_stats_paired_with_two_seeded_arms():
    cand, comp = ap((5, 100), 5), ap((5, 100), 6)
    ref = rt_stats.paired(cand, comp, DRAWS)
    got = ns_ret.paired_ret(cand, comp, draws=DRAWS, comp_seeds='shared')
    assert got['d'] == ref['d'] and (got['lo'], got['hi']) == tuple(ref['interval'])


def test_apart_draws_the_comparators_seeds_after_cstars():
    cand, comp = ap((5, 100), 7), ap((5, 100), 8)
    got = ns_ret.paired_ret(cand, comp, draws=DRAWS, comp_seeds='apart')
    boot = reference_boot(cand, comp, DRAWS, 'apart')
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert abs(got['lo'] - lo) < 1e-12 and abs(got['hi'] - hi) < 1e-12
    assert got['d'] == cand.mean() - comp.mean()


def test_apart_differs_from_shared():
    cand, comp = ap((5, 100), 9), ap((5, 100), 10)
    a = ns_ret.paired_ret(cand, comp, draws=DRAWS, comp_seeds='apart')
    s = ns_ret.paired_ret(cand, comp, draws=DRAWS, comp_seeds='shared')
    assert (a['lo'], a['hi']) != (s['lo'], s['hi'])
    assert a['d'] == s['d']


def test_apart_with_equal_seed_rows_is_wider_than_shared():
    # two arms that are the same five rows: a shared draw cancels the seed variation, apart draws do not
    base = np.random.default_rng(11).uniform(0.2, 0.8, size=(5, 150))
    base = base + np.arange(5)[:, None] * 0.05            # seeds differ in level
    a = ns_ret.paired_ret(base, base.copy(), draws=2000, comp_seeds='apart')
    s = ns_ret.paired_ret(base, base.copy(), draws=2000, comp_seeds='shared')
    assert (a['hi'] - a['lo']) > 3 * (s['hi'] - s['lo'])


def test_float64_enforced():
    good = ap((5, 50), 1)
    with pytest.raises(AssertionError):
        ns_ret.paired_ret(good.astype(np.float32), ap((50,), 2), comp_seeds='none', draws=10)
    with pytest.raises(AssertionError):
        ns_ret.paired_ret(good, ap((50,), 2).astype(np.float32), comp_seeds='none', draws=10)
    with pytest.raises(AssertionError):
        ns_ret.paired_ret(good, ap((5, 50), 2).astype(np.float32), comp_seeds='apart', draws=10)


def test_query_counts_must_match():
    for mode, comp in (('none', ap((49,), 2)), ('apart', ap((5, 49), 2)), ('shared', ap((5, 49), 2))):
        with pytest.raises(AssertionError):
            ns_ret.paired_ret(ap((5, 50), 1), comp, comp_seeds=mode, draws=10)


def test_seed_shape_must_fit_the_mode():
    with pytest.raises(AssertionError):                       # 'none' is an unseeded comparator
        ns_ret.paired_ret(ap((5, 50), 1), ap((5, 50), 2), comp_seeds='none', draws=10)
    for mode in ('apart', 'shared'):                          # two seeded arms
        with pytest.raises(AssertionError):
            ns_ret.paired_ret(ap((5, 50), 1), ap((50,), 2), comp_seeds=mode, draws=10)
    with pytest.raises(AssertionError):
        ns_ret.paired_ret(ap((3, 50), 1), ap((50,), 2), comp_seeds='none', draws=10)
    with pytest.raises(AssertionError):
        ns_ret.paired_ret(ap((5, 50), 1), ap((50,), 2), comp_seeds='sideways', draws=10)


def test_outcome_words():
    cand = np.full(40, 0.5)
    win = ns_ret.paired_ret(cand + 0.1, cand, r=0.01, draws=50, comp_seeds='none')
    assert win['outcome'] == 'WIN' and win['lo'] > 0
    level = ns_ret.paired_ret(cand - 0.005, cand, r=0.01, draws=50, comp_seeds='none')
    assert level['outcome'] == 'LEVEL'
    below = ns_ret.paired_ret(cand - 0.05, cand, r=0.01, draws=50, comp_seeds='none')
    assert below['outcome'] == 'BELOW'
    none_r = ns_ret.paired_ret(cand - 0.005, cand, draws=50, comp_seeds='none')
    assert none_r['outcome'] == 'not WIN' and none_r['r'] is None
    assert ns_ret.paired_ret(cand + 0.1, cand, draws=50, comp_seeds='none')['outcome'] == 'WIN'


# ------------------------------------------------------------------------------------------------ arm_ap
def synth_rows(n, d, seed, classes=4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=n)
    y[::9] = -1
    centers = rng.normal(size=(classes, d))
    x = centers[np.maximum(y, 0)] + 1.0 * rng.normal(size=(n, d))
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True), y.astype(np.int64)


def test_arm_ap_unseeded_equals_the_stage5_steps():
    xv, _ = synth_rows(70, 12, 1)
    xe, y = synth_rows(90, 12, 2)
    res = ns_ret.arm_ap(lambda s: xv, lambda s: xe, 'center+k3', y, None, workers=1)
    ro = rt_readout.ReadOut('center', None, 3)
    ref = rt_metrics.retrieval(rt_readout.stage5_apply(ro, xv, xe), y, qe_k=3, workers=1)
    assert np.array_equal(res['ap'], ref['ap']) and np.array_equal(res['rows'], ref['rows'])
    assert res['ap'].dtype == np.float64 and res['per_seed'] is None and res['point'] == float(ref['ap'].mean())


def test_arm_ap_map_is_fitted_on_validation_rows_and_applied_to_test():
    xv, _ = synth_rows(70, 12, 1)
    xe, y = synth_rows(90, 12, 2)
    shifted = (xv + 0.5).astype(np.float32)               # a different validation set: a different centre
    a = ns_ret.arm_ap(lambda s: xv, lambda s: xe, 'center+k0', y, None, workers=1)
    b = ns_ret.arm_ap(lambda s: shifted, lambda s: xe, 'center+k0', y, None, workers=1)
    assert not np.array_equal(a['ap'], b['ap'])


def test_arm_ap_seeded_five_seeds_and_rows_shared():
    seen = []

    def gv(s):
        seen.append(('v', s))
        return synth_rows(60, 10, 10 + s)[0]

    def ge(s):
        seen.append(('e', s))
        return synth_rows(80, 10, 20 + s)[0]

    y = synth_rows(80, 10, 3)[1]
    res = ns_ret.arm_ap(gv, ge, 'none+k0', y, ns_paths.SEEDS, workers=1)
    assert res['ap'].shape == (5, len(res['rows'])) and len(res['per_seed']) == 5
    assert sorted({s for _, s in seen}) == [0, 1, 2, 3, 4]
    assert abs(res['point'] - np.mean(res['per_seed'])) < 1e-15
    with pytest.raises(AssertionError):
        ns_ret.arm_ap(gv, ge, 'none+k0', y, (0, 1, 2), workers=1)


def test_arm_ap_checks_labels():
    xv, _ = synth_rows(70, 12, 1)
    xe, y = synth_rows(90, 12, 2)
    with pytest.raises(AssertionError):
        ns_ret.arm_ap(lambda s: xv, lambda s: xe, 'none+k0', y[:-1], None, workers=1)
    with pytest.raises(AssertionError):
        ns_ret.arm_ap(lambda s: xv, lambda s: xe, 'none+k0', y.astype(np.int32), None, workers=1)


# ------------------------------------------------------------------------------------------------ score_cells
N_V, N_E, D = 50, 70, 8


class Fake:
    """Replaces the data and record readers of ns_ret with synthetic arrays; counts the calls."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.picks_read = []
        mp = monkeypatch
        mp.setattr(ns_data, 'labels', self.labels)
        mp.setattr(ns_data, 'cstar', self.cstar)
        mp.setattr(ns_data, 'untold', self.untold)
        mp.setattr(ns_data, 'plain', self.plain)
        mp.setattr(ns_data, 'crl', self.crl)
        mp.setattr(ns_data, 'percept_probs', self.probs)
        mp.setattr(ns_ret, 'reused_pick', self.reused_pick)
        mp.setattr(ns_ret, 'recorded_range', self.recorded_range)
        mp.setattr(ns_paths, 'read_committed_record', self.record)
        mp.setattr(ns_paths, 'SMOKE', 0)

    def n(self, split):
        return N_V if split == 'val' else N_E

    def labels(self, set_, split, aspect):
        return synth_rows(self.n(split), D, 5 + ns_paths.ASPECTS.index(aspect))[1]

    def _x(self, split, tag, d=D, seed=0):
        return synth_rows(self.n(split), d, 100 * (split == 'val') + zlib.crc32(repr(tag).encode()) % 50 + seed)[0]

    def cstar(self, set_, split, enc, aspect, seed, unit=True):
        self.calls.append(('cstar', split))
        return self._x(split, ('c', aspect), seed=seed)

    def untold(self, set_, split, enc, seed):
        self.calls.append(('untold', split))
        return self._x(split, 'u', 3 * D, seed)

    def plain(self, set_, split, enc):
        self.calls.append(('plain', split))
        return self._x(split, 'p')

    def crl(self, set_, split, aspect, raw=False):
        self.calls.append(('crl', split))
        return self._x(split, ('r', aspect), 5)

    def probs(self, set_, split, enc, seed):
        self.calls.append(('probs', split))
        return np.abs(self._x(split, 'm', 6, seed)) / 3

    def reused_pick(self, arm, set_, enc):
        return {'cstar': 'center+k3', 'plain': 'none+k0'}.get(arm)

    def recorded_range(self, set_, enc, aspect):
        return 0.002, 'fake range'

    def record(self, name):
        self.picks_read.append(name)
        assert name == 'stage1_picks.json'
        return {f'{arm}/mmart/clip_l14': {'pick': 'center+k1'} for arm in ('untold', 'matched', 'crl')}


@pytest.fixture
def fake(monkeypatch):
    return Fake(monkeypatch)


def by_sc(cells):
    out = {}
    for c in cells:
        out.setdefault(c['sc'], []).append(c)
    return out


def test_score_cells_gives_the_claim_cells_of_the_scs(fake):
    cells = ns_ret.score_cells('mmart', 'clip_l14', ('SC6', 'SC7', 'SC8'), None, 60, workers=1)
    want = {c['id'] for c in ns_outcome.claim_cells()
            if c['set'] == 'mmart' and c['encoder'] == 'clip_l14' and c['sc'] in ('SC6', 'SC7', 'SC8')}
    assert {c['id'] for c in cells} == want and len(cells) == 9
    for c in cells:
        cc = ns_outcome.check_cell(c, smoke=True)          # the format 02 reads
        assert c['task'] == 'ret' and c['metric'] == 'map' and c['k'] is None
        assert c['n_draws'] == 60 and c['r'] == 0.002 and c['r_source'] == 'fake range'
        assert abs(c['d'] - (c['cand'] - c['comp'])) < 1e-15
        assert c['n_rows'] == N_E and 0 < c['n_queries'] <= N_E
    assert fake.picks_read                                   # new picks come through the committed-record reader
    assert set(fake.picks_read) == {'stage1_picks.json'}


def test_score_cells_reads_held_for_scoring_and_val_for_the_fit(fake):
    ns_ret.score_cells('mmart', 'clip_l14', ('SC7',), None, 30, workers=1)
    assert {('untold', 'val'), ('untold', 'test'), ('plain', 'val'), ('plain', 'test')} <= set(fake.calls)
    assert not any(k in ('crl', 'probs', 'cstar') for k, _ in fake.calls)


def test_score_cells_uses_the_intended_draws_per_sc(fake):
    cells = by_sc(ns_ret.score_cells('mmart', 'clip_l14', ('SC6', 'SC7', 'SC8'), None, 80, workers=1))
    # reproduce SC8 and SC7 from the arms by hand
    a = 'emotion'
    y = fake.labels('mmart', 'test', a)

    def run(get_v, get_e, pick, seeds):
        return ns_ret.arm_ap(get_v, get_e, pick, y, seeds, workers=1)

    cs = run(lambda s: fake.cstar('mmart', 'val', 'clip_l14', a, s), lambda s: fake.cstar('mmart', 'test', 'clip_l14', a, s),
             'center+k3', ns_paths.SEEDS)
    pc = run(lambda s: ns_ret._unit(fake.probs('mmart', 'val', 'clip_l14', s)),
             lambda s: ns_ret._unit(fake.probs('mmart', 'test', 'clip_l14', s)), 'center+k1', ns_paths.SEEDS)
    want = ns_ret.paired_ret(cs['ap'], pc['ap'], r=0.002, draws=80, comp_seeds='apart')
    sc8 = next(c for c in cells['SC8'] if c['aspect'] == a)
    assert (sc8['lo'], sc8['hi'], sc8['d']) == (want['lo'], want['hi'], want['d'])
    shared = ns_ret.paired_ret(cs['ap'], pc['ap'], r=0.002, draws=80, comp_seeds='shared')
    assert (sc8['lo'], sc8['hi']) != (shared['lo'], shared['hi'])
    # SC6: CRL is unseeded -> r11's draws
    crl = run(lambda s: fake.crl('mmart', 'val', a), lambda s: fake.crl('mmart', 'test', a), 'center+k1', None)
    want6 = rt_stats.paired(cs['ap'], crl['ap'], 80)
    sc6 = next(c for c in cells['SC6'] if c['aspect'] == a)
    assert (sc6['d'], sc6['lo'], sc6['hi']) == (want6['d'], want6['interval'][0], want6['interval'][1])
    # SC7: plain, the aspect's own labels
    un = run(lambda s: fake.untold('mmart', 'val', 'clip_l14', s), lambda s: fake.untold('mmart', 'test', 'clip_l14', s),
             'center+k1', ns_paths.SEEDS)
    pl = run(lambda s: fake.plain('mmart', 'val', 'clip_l14'), lambda s: fake.plain('mmart', 'test', 'clip_l14'),
             'none+k0', None)
    want7 = rt_stats.paired(un['ap'], pl['ap'], 80)
    sc7 = next(c for c in cells['SC7'] if c['aspect'] == a)
    assert (sc7['d'], sc7['lo'], sc7['hi']) == (want7['d'], want7['interval'][0], want7['interval'][1])
    assert sc7['cand'] == un['ap'].mean() and sc7['comp'] == pl['ap'].mean()


def test_score_cells_untold_is_scored_with_each_aspects_labels(fake):
    cells = by_sc(ns_ret.score_cells('mmart', 'clip_l14', ('SC7',), None, 20, workers=1))['SC7']
    assert len({c['n_queries'] for c in cells}) >= 1
    assert [c['aspect'] for c in cells] == list(ns_paths.ASPECTS)
    # the untold rows are the same for the three aspects; the per-aspect numbers differ because the labels do
    assert len({round(c['cand'], 12) for c in cells}) == 3


def test_score_cells_percept_rows_are_l2_normalised(fake, monkeypatch):
    seen = []
    orig = ns_ret.arm_ap

    def spy(get_val, get_eval, pick, labels, seeds, workers=8):
        x = get_eval(0 if seeds else None)
        seen.append(x)
        return orig(get_val, get_eval, pick, labels, seeds, workers)

    monkeypatch.setattr(ns_ret, 'arm_ap', spy)
    ns_ret.score_cells('mmart', 'clip_l14', ('SC8',), None, 20, workers=1)
    probs = [x for x in seen if x.shape[1] == 6]
    assert probs and all(np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6) for x in probs)
    assert all(x.dtype == np.float32 for x in probs)


def test_score_cells_sc6_only_on_clip_l14_and_unknown_sc_refused(fake):
    with pytest.raises(AssertionError):
        ns_ret.score_cells('mmart', 'qwen', ('SC6',), None, 20, workers=1)
    with pytest.raises(AssertionError):
        ns_ret.score_cells('mmart', 'clip_l14', ('SC2',), None, 20, workers=1)
    with pytest.raises(AssertionError):
        ns_ret.score_cells('affection', 'clip_l14', ('SC7',), None, 20, workers=1)


def test_score_cells_explicit_picks_need_smoke(fake, monkeypatch):
    with pytest.raises(AssertionError):                       # a given record is for smoke runs and tests only
        ns_ret.score_cells('mmart', 'clip_l14', ('SC8',), {'matched/mmart/clip_l14': {'pick': 'none+k0'}}, 20, workers=1)
    monkeypatch.setattr(ns_paths, 'SMOKE', 1)
    cells = ns_ret.score_cells('mmart', 'clip_l14', ('SC8',), {'matched/mmart/clip_l14': {'pick': 'none+k0'}}, 20, workers=1)
    assert len(cells) == 3 and {c['comparator_readout'] for c in cells} == {'none+k0'}


@pytest.mark.parametrize('arm, sc', [('untold', 'SC7'), ('matched', 'SC8'), ('crl', 'SC6')])
def test_score_cells_missing_pick_skips_only_its_sc(fake, monkeypatch, capsys, arm, sc):
    scs = ('SC6', 'SC7', 'SC8')
    full = ns_ret.score_cells('mmart', 'clip_l14', scs, None, 20, workers=1)
    record = fake.record('stage1_picks.json')
    key = f'{arm}/mmart/clip_l14'
    del record[key]
    monkeypatch.setattr(ns_paths, 'read_committed_record', lambda name: record)
    calls = []
    original = ns_ret._Rows.ap

    def spy(self, arm, aspect, *args):
        calls.append((arm, aspect))
        return original(self, arm, aspect, *args)

    monkeypatch.setattr(ns_ret._Rows, 'ap', spy)
    got = ns_ret.score_cells('mmart', 'clip_l14', scs, None, 20, workers=1)
    assert got == [c for c in full if c['sc'] != sc]
    assert len(calls) == len(set(calls))               # a cached arm survives a missing comparator
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and sc in lines[0] and key in lines[0]


@pytest.mark.parametrize('source', ['reused_pick', 'arm_ap', 'record', 'file'])
def test_score_cells_other_errors_still_raise(fake, monkeypatch, source):
    def fail(*args, **kwargs):
        raise FileNotFoundError('stage1_picks.json') if source == 'file' else KeyError('other source')

    if source == 'reused_pick':
        monkeypatch.setattr(ns_ret, 'reused_pick', fail)
    elif source == 'arm_ap':
        monkeypatch.setattr(ns_ret, 'arm_ap', fail)
    elif source == 'record':
        monkeypatch.setattr(ns_paths, 'read_committed_record', lambda name: {'untold/mmart/clip_l14': {}})
    else:
        monkeypatch.setattr(ns_paths, 'read_committed_record', fail)
    with pytest.raises(FileNotFoundError if source == 'file' else KeyError):
        ns_ret.score_cells('mmart', 'clip_l14', ('SC7',), None, 20, workers=1)


def test_score_cells_qwen_artelingo_sc8_keeps_r11_probabilities(fake, monkeypatch):
    called = []
    monkeypatch.setattr(ns_ret, 'r11_percept_val', lambda seed: called.append(seed) or fake._x('val', 'q', 6, seed))
    monkeypatch.setattr(ns_paths, 'read_committed_record', lambda name: pytest.fail('Qwen on ArtELingo has no new pick'))
    monkeypatch.setattr(ns_ret, 'reused_pick', lambda arm, s, e: {'cstar': 'center+k3', 'matched': 'center+k10'}.get(arm))
    cells = ns_ret.score_cells('artelingo', 'qwen', ('SC8',), None, 20, workers=1)
    assert sorted(set(called)) == [0, 1, 2, 3, 4] and len(cells) == 3
    assert not any(k == 'probs' and s == 'val' for k, s in fake.calls)


def test_mmart_plain_rows_get_r15s_retrieval_unit_and_artelingo_rows_do_not(monkeypatch):
    x = synth_rows(N_E, D, 1)[0] * np.float32(1.0000001)           # unit within 1e-7, not exactly
    monkeypatch.setattr(ns_data, 'plain', lambda set_, split, enc: x)
    mm = ns_ret._Rows('mmart', 'clip_l14')._plain_rows('test')
    assert np.array_equal(mm, ns_ret._unit(x)) and not np.array_equal(mm, x)
    ae = ns_ret._Rows('artelingo', 'qwen')._plain_rows('held')
    assert np.array_equal(ae, x)


# ------------------------------------------------------------------------------------------------ record reads
# Values copied from the records themselves (opened 2026-11-03); exact equality. These pin the reads the cells make
# (SC6 to SC8 r, the reused plain / CRL / c* picks), which no gate checks (gate (ii) pins only the c* pick).
@pytest.mark.parametrize('set_, enc, aspect, want', [
    # r14 records/stage1_<enc>.json ['artret_ranges']['ranges'][aspect] (ArtELingo validation): CLIP L/14 emotion
    # 0.26528545 - 0.26359618; r15's ret_ranges for the same encoder differs (0.000633751040052366)
    ('artelingo', 'clip_l14', 'emotion', 0.0016892742932269056),
    ('artelingo', 'siglip1', 'emotion', 0.0021643605637413077),     # r14 artret_ranges (r15: 0.0005943409765497565)
    ('artelingo', 'siglip2', 'emotion', 0.0015141856668823839),     # r14 artret_ranges (r15: 0.0010930056213613037)
    # r15 records/stage1_<enc>.json ['ret_ranges']['ranges'][aspect] (MMArt validation)
    ('mmart', 'siglip1', 'emotion', 0.0005943409765497565),
    ('mmart', 'siglip2', 'emotion', 0.0010930056213613037),
    ('mmart', 'clip_l14', 'genre', 0.0019367407309804907),
    ('mmart', 'qwen', 'style', 0.00258267282747926),
    # r11 stage0a_ranges.json ['map'][aspect]['r'] (Qwen c*, ArtELingo validation)
    ('artelingo', 'qwen', 'emotion', 0.0014116107779215126),
    ('artelingo', 'qwen', 'genre', 0.0038237977312404192),
])
def test_recorded_range_reads_the_recorded_value(set_, enc, aspect, want):
    r, src = ns_ret.recorded_range(set_, enc, aspect)
    assert r == want and isinstance(src, str)


@pytest.mark.parametrize('arm, set_, enc, want', [
    # r15 records/stage1_<enc>.json ['picks']['arms']: cstar 'center+k10', frozen (= plain) 'none+k10' for Qwen and
    # SigLIP 1 on MMArt. No ArtELingo encoder has a plain pick that differs from its c* pick (r14 artret_picks cstar and
    # frozen are 'center+k10' for all three; r11 stage4/picks.json cstar and plain 'center+k10'), so the plain-versus-c*
    # key mapping is pinned on MMArt; the same line serves both sets.
    ('plain', 'mmart', 'qwen', 'none+k10'),
    ('plain', 'mmart', 'siglip1', 'none+k10'),
    ('cstar', 'mmart', 'siglip1', 'center+k10'),
    ('cstar', 'mmart', 'qwen', 'center+k10'),
    ('plain', 'artelingo', 'siglip1', 'center+k10'),                # r14 stage1_siglip1.json artret_picks.frozen.pick
    ('plain', 'artelingo', 'qwen', 'center+k10'),                   # r11 stage4/picks.json plain
    ('cstar', 'artelingo', 'clip_l14', 'center+k10'),               # r14 stage1_clip_l14.json artret_picks.cstar.pick
    ('matched', 'artelingo', 'qwen', 'center+k10'),                 # r11 stage4/picks.json percept
    ('crl', 'artelingo', 'clip_l14', 'none+k10'),                   # r11 stage4/picks.json crl
])
def test_reused_pick_reads_the_recorded_pick(arm, set_, enc, want):
    assert ns_ret.reused_pick(arm, set_, enc) == want
