"""Tests of om_stage1 (ticket 08): ranges, CLAY's text sets, read-out picks. Synthetic data; a few tests read the real
recorded range files (milliseconds).

Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q
     src/test/20261103_outside_methods/test_om_stage1.py
"""
import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score

import om_data  # noqa: F401  (imported before any test patches N_ROWS: it asserts them)
import om_paths as op
import om_stage1 as s1

import ac_select  # noqa: E402  (om_paths put the older rounds on sys.path)
import mt_data  # noqa: E402
import ns_metrics  # noqa: E402
import ns_picks  # noqa: E402
import ns_paths  # noqa: E402
import rt_readout  # noqa: E402

ASPECTS = op.ASPECTS
N = 40


def unit(a):
    a = np.asarray(a, dtype=np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ fake data
class FakeVal:
    """Stands in for ValData on N validation rows of either set. `calls` records what was read."""

    def __init__(self, n_texts=None, held_overlap=False, fallback_b=False):
        self.calls, self.rng = [], np.random.default_rng(0)
        self.n_texts = n_texts or {'A': 50, 'B': 100}
        self.held_overlap, self.fallback_b = held_overlap, fallback_b

    def keys(self, set_):
        names = [f'p{i:03d}' for i in range(N)]
        return sorted(names) if set_ == 'artelingo' else list(mt_data.order(names))

    def held_keys(self, set_):
        if self.held_overlap:
            return self.keys(set_)[:3]
        return [f'h{i:03d}' for i in range(N)]

    def labels(self, set_, aspect):
        self.calls.append(('labels', set_, aspect))
        return np.arange(N, dtype=np.int64) % 3

    def cstar(self, set_, enc, aspect, seed):
        self.calls.append(('cstar', set_, enc, aspect, seed))
        return np.full((N, 4), float(seed), dtype=np.float32)

    def _rows(self, d):
        return unit(self.rng.normal(size=(N, d)))

    def tpips(self, set_, aspect):
        self.calls.append(('tpips', set_, aspect))
        return self._rows(16)

    def clay(self, set_, aspect, text_set):
        self.calls.append(('clay', set_, aspect, text_set))
        assert not (text_set == 'B' and self.fallback_b), 'set B has no vectors on fallback'
        return self._rows(16)

    def text_count(self, text_set, aspect):
        if text_set == 'B' and self.fallback_b:
            return None
        return self.n_texts[text_set]

    def text_sha(self, text_set, aspect):
        return None if self.text_count(text_set, aspect) is None else f'sha-{text_set}-{aspect}'


@pytest.fixture(autouse=True)
def never_write_the_real_out(tmp_path, monkeypatch):
    """Every writer here defaults to `om_paths.OUT`: point it at a scratch folder so no test can touch the real outputs."""
    monkeypatch.setattr(op, 'OUT', tmp_path / 'guard_out')


@pytest.fixture
def small(monkeypatch):
    for key in (('artelingo', 'val'), ('mmart', 'val')):
        monkeypatch.setitem(ns_paths.N_ROWS, key, N)


# ------------------------------------------------------------------------------------------------ the range
def flipped(y, frac, seed):
    """A partition equal to y with a fraction of its rows moved to a random class (a known, graded AMI)."""
    rng = np.random.default_rng(seed)
    p = y.copy()
    idx = rng.choice(len(y), int(frac * len(y)), replace=False)
    p[idx] = rng.integers(0, 3, size=len(idx))
    return p


def test_range_is_max_minus_min_over_seeds_of_state_means(small):
    y = np.arange(N, dtype=np.int64) % 3
    fracs = {0: (0.1, 0.4, 0.2), 1: (0.3, 0.3, 0.3), 2: (0.0, 0.5, 0.6), 3: (0.2, 0.2, 0.9), 4: (0.4, 0.1, 0.1)}
    parts = {s: np.stack([flipped(y, f, 10 * s + i) for i, f in enumerate(fracs[s])]).astype(np.int32) for s in fracs}
    means = {s: float(np.mean([adjusted_mutual_info_score(y, p) for p in parts[s]])) for s in parts}
    assert len(set(round(v, 6) for v in means.values())) == 5          # five different seed means
    first_state = [adjusted_mutual_info_score(y, parts[s][0]) for s in parts]

    def stub(x, k, set_name):
        assert k == 20 and set_name == 'artelingo'
        return parts[int(x[0, 0])]
    got = s1.cstar_range(FakeVal(), 'artelingo', 'qwen', 'emotion', 20, 'ami', partitions=stub)
    want = max(means.values()) - min(means.values())
    assert got['r'] == pytest.approx(want, abs=1e-12)
    assert abs(want - (max(first_state) - min(first_state))) > 1e-3     # a first-state-only range would differ
    assert got['per_seed'] == {str(s): pytest.approx(means[s], abs=1e-12) for s in means}
    assert got['r'] == float(got['r']) and got['r'] != round(got['r'], 4)  # unrounded


def test_range_reads_seeds_zero_to_four_of_validation_only(small):
    y = np.arange(N, dtype=np.int64) % 3
    d = FakeVal()
    s1.cstar_range(d, 'artelingo', 'qwen', 'style', 20, 'ami', partitions=lambda x, k, s: np.stack([y] * 3))
    seeds = [c[4] for c in d.calls if c[0] == 'cstar']
    assert seeds == [0, 1, 2, 3, 4] and all(c[2] == 'qwen' for c in d.calls if c[0] == 'cstar')


# ------------------------------------------------------------------------------------------------ the r table
def test_range_keys_cover_the_labelled_cell_families():
    ks = s1.range_keys()
    assert len(ks) == len(set(k['key'] for k in ks))
    clu = [k for k in ks if k['metric'] != 'map']
    ret = [k for k in ks if k['metric'] == 'map']
    assert len(ret) == 2 * 4 * 3 and all(k['key'] == f"{k['set']}/{k['enc']}/{k['aspect']}/map" for k in ret)
    assert len(clu) == 4 * (15 + 14)       # ArtELingo 4 metrics at the class count + AMI at 20; MMArt style: class count is 20
    have = {k['key'] for k in clu}
    assert 'artelingo/qwen/emotion/ami/9' in have and 'artelingo/qwen/emotion/ami/20' in have
    assert 'artelingo/qwen/emotion/nmi/20' not in have
    assert 'mmart/siglip1/style/acc/20' in have and 'mmart/siglip1/style/ami/20' in have
    assert 'mmart/siglip1/style/ami/27' not in have


def write_fake_sources(tmp_path, r17=True, drop17=None):
    dirs = {}
    for rnd, inner, kind in (('r14', 'artclu', 'artelingo_clustering_validation_ranges'),
                             ('r15', 'clu_ranges', 'mmart_clustering_validation_ranges')):
        d = tmp_path / rnd
        d.mkdir()
        for i, enc in enumerate(op.ENCODERS):
            if rnd == 'r14' and enc == 'qwen':
                continue
            per = {a: {str(s): 0.1 + 0.01 * s * (1 + i) for s in range(5)} for a in ASPECTS}
            rng = {a: 0.04 * (1 + i) for a in ASPECTS}
            (d / f'stage1_{enc}.json').write_text(json.dumps({'encoder': enc, inner: {
                'encoder': enc, 'kind': kind, 'ranges': rng, 'per_seed': per}}))
        dirs[rnd] = d
    f17 = tmp_path / 'stage1_ranges.json'
    if r17:
        doc = {}
        for k in s1.range_keys():
            if k['metric'] != 'map' and k['k'] == op.CLASS_COUNT[(k['set'], k['aspect'])]:
                doc[f"cstar/{k['set']}/{k['enc']}/{k['aspect']}/{k['metric']}"] = {'r': 0.5, 'per_seed': {}, 'k': k['k']}
        for key in drop17 or ():
            doc.pop(key)
        f17.write_text(json.dumps(doc))
    return dirs, f17


def computed_qwen():
    return {a: {'r': 0.3 + 0.01 * i, 'per_seed': {str(s): 0.1 for s in range(5)}, 'n_val': N, 'n_labelled': N}
            for i, a in enumerate(ASPECTS)}


def test_table_routes_every_family_to_its_source(tmp_path, monkeypatch):
    dirs, f17 = write_fake_sources(tmp_path)
    monkeypatch.setattr(s1, 'RECORD_DIRS', dirs)
    t = s1.build_ranges(computed_qwen(), r17_path=f17)
    assert len(t) == len(s1.range_keys()) and all(set(v) >= {'r', 'source'} for v in t.values())
    # class count: r17
    assert t['artelingo/siglip2/genre/nmi/10']['r'] == 0.5 and 'stage1_ranges.json' in t['artelingo/siglip2/genre/nmi/10']['source']
    assert t['mmart/qwen/emotion/ami/9']['r'] == 0.5
    # MMArt: AMI at 20 and the style class-count AMI come from r15; the other metrics at 20 from r17
    assert t['mmart/clip_l14/genre/ami/20']['r'] == pytest.approx(0.04 * 4) and 'r15' in t['mmart/clip_l14/genre/ami/20']['source']
    assert t['mmart/siglip1/style/ami/20']['r'] == pytest.approx(0.04 * 3) and 'r15' in t['mmart/siglip1/style/ami/20']['source']
    assert t['mmart/siglip1/style/ari/20']['r'] == 0.5 and 'stage1_ranges' in t['mmart/siglip1/style/ari/20']['source']
    # ArtELingo AMI at 20: r14 for the other three encoders, this stage for Qwen
    assert t['artelingo/siglip2/style/ami/20']['r'] == pytest.approx(0.08) and 'r14' in t['artelingo/siglip2/style/ami/20']['source']
    assert t['artelingo/qwen/style/ami/20']['r'] == pytest.approx(0.31) and t['artelingo/qwen/style/ami/20']['per_seed']
    assert 'r14' not in t['artelingo/qwen/style/ami/20']['source']
    # a class-count AMI on ArtELingo is r17's even though AMI at 20 is not
    assert t['artelingo/qwen/style/ami/27']['r'] == 0.5


def test_mmart_qwen_ami20_is_r15s_record_not_the_computed_artelingo_range():
    """MMArt's AMI at 20 comes from r15 for every encoder, Qwen too; `qwen_ami20` is ArtELingo's and must not reach it."""
    rec = json.loads((s1.RECORD_DIRS['r15'] / 'stage1_qwen.json').read_text())['clu_ranges']['ranges']
    t = s1.build_ranges(computed_qwen(), r17_path=tmp_none())
    for a in ASPECTS:
        e = t[f'mmart/qwen/{a}/ami/20']
        assert e['r'] == float(rec[a]) and 'r15' in e['source'] and 'per_seed' not in e
        assert e['r'] not in {v['r'] for v in computed_qwen().values()}
    assert t['artelingo/qwen/emotion/ami/20']['r'] == computed_qwen()['emotion']['r']      # ArtELingo's is the computed one


def tmp_none():
    return '/nonexistent/stage1_ranges.json'


def test_missing_r17_record_gives_null_with_source_and_no_crash(tmp_path, monkeypatch):
    dirs, f17 = write_fake_sources(tmp_path, r17=False)
    monkeypatch.setattr(s1, 'RECORD_DIRS', dirs)
    t = s1.build_ranges(computed_qwen(), r17_path=f17)
    nulls = {k for k, v in t.items() if v['r'] is None}
    want = {k['key'] for k in s1.range_keys() if k['source'] == 'r17'}
    assert nulls == want and len(nulls) > 50
    assert all('absent' in t[k]['source'] and 'stage1_ranges.json' in t[k]['source'] for k in nulls)
    assert t['artelingo/siglip2/style/ami/20']['r'] is not None and t['mmart/qwen/style/ami/20']['r'] is not None


def test_one_missing_r17_key_gives_null_for_that_key_only(tmp_path, monkeypatch):
    drop = 'cstar/artelingo/siglip1/emotion/acc'
    dirs, f17 = write_fake_sources(tmp_path, drop17=[drop])
    monkeypatch.setattr(s1, 'RECORD_DIRS', dirs)
    t = s1.build_ranges(computed_qwen(), r17_path=f17)
    assert t['artelingo/siglip1/emotion/acc/9']['r'] is None and drop in t['artelingo/siglip1/emotion/acc/9']['source']
    assert sum(v['r'] is None for v in t.values()) == 1


def test_missing_r14_r15_r11_source_gives_null_for_that_key_only(tmp_path, monkeypatch):
    dirs, f17 = write_fake_sources(tmp_path)
    (dirs['r14'] / 'stage1_siglip2.json').unlink()                                         # file absent
    p = dirs['r15'] / 'stage1_siglip1.json'
    doc = json.loads(p.read_text())
    del doc['clu_ranges']
    p.write_text(json.dumps(doc))                                                          # key absent
    monkeypatch.setattr(s1, 'RECORD_DIRS', dirs)
    real_map = s1.map_range

    def map_range(set_, enc, aspect):
        if (set_, enc) == ('mmart', 'clip_l14'):
            raise FileNotFoundError('r15 stage1_clip_l14.json')                            # r11 / r14 / r15 mAP record absent
        return real_map(set_, enc, aspect)
    monkeypatch.setattr(s1, 'map_range', map_range)
    t = s1.build_ranges(computed_qwen(), r17_path=f17)
    nulls = {k for k, v in t.items() if v['r'] is None}
    want = ({f'artelingo/siglip2/{a}/ami/20' for a in ASPECTS} | {f'mmart/siglip1/{a}/ami/20' for a in ASPECTS}
            | {f'mmart/clip_l14/{a}/map' for a in ASPECTS})
    assert nulls == want and all(t[k]['source'].startswith('missing:') for k in nulls)
    assert t['artelingo/siglip1/style/ami/20']['r'] is not None and t['mmart/siglip2/style/ami/20']['r'] is not None


def test_a_bug_in_a_source_still_crashes(tmp_path, monkeypatch):
    dirs, f17 = write_fake_sources(tmp_path)
    monkeypatch.setattr(s1, 'RECORD_DIRS', dirs)
    monkeypatch.setattr(s1, 'map_range', lambda *a: (_ for _ in ()).throw(ValueError('bug')))
    with pytest.raises(ValueError):
        s1.build_ranges(computed_qwen(), r17_path=f17)


def test_rerun_picks_up_r17_when_it_arrives(tmp_path, monkeypatch):
    (tmp_path / 'first').mkdir()
    (tmp_path / 'later').mkdir()
    dirs, f17 = write_fake_sources(tmp_path / 'first', r17=False)
    monkeypatch.setattr(s1, 'RECORD_DIRS', dirs)
    before = s1.build_ranges(computed_qwen(), r17_path=f17)
    _, f17_later = write_fake_sources(tmp_path / 'later', r17=True)
    after = s1.build_ranges(computed_qwen(), r17_path=f17_later)
    assert sum(v['r'] is None for v in before.values()) > 50 and not any(v['r'] is None for v in after.values())
    same = [k for k in before if before[k]['r'] is not None]
    assert all(before[k] == after[k] for k in same)                       # only the r17 entries change


def test_recorded_range_with_inconsistent_per_seed_refused(tmp_path, monkeypatch):
    dirs, f17 = write_fake_sources(tmp_path)
    p = dirs['r14'] / 'stage1_siglip2.json'
    doc = json.loads(p.read_text())
    doc['artclu']['ranges']['style'] = 0.9
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(s1, 'RECORD_DIRS', dirs)
    with pytest.raises(AssertionError):
        s1.build_ranges(computed_qwen(), r17_path=f17)


def test_real_recorded_ranges():
    r, _ = s1.recorded_ami20('artelingo', 'siglip2', 'style')
    assert r == 0.01604545569202437                                     # r14's artclu
    r, src = s1.recorded_ami20('mmart', 'qwen', 'emotion')
    assert r == 0.004760010300125482 and 'r15' in src
    r, src = s1.map_range('mmart', 'clip_l14', 'emotion')
    assert r == 0.000633751040052366
    with pytest.raises(AssertionError):
        s1.recorded_ami20('artelingo', 'qwen', 'style')                 # no recorded range exists for it


# ------------------------------------------------------------------------------------------------ tie rules
@pytest.mark.parametrize('margin', [s1.CLU_MARGIN, s1.RET_MARGIN])
def test_text_set_tie_rule_at_the_margin(margin):
    a = 0.5
    assert s1.choose_text_set(a, a + margin - 1e-6, margin)[0] == 'A'            # just inside: A
    assert s1.choose_text_set(a, a + margin, margin)[0] == 'A'                   # exactly the margin: still a tie
    assert s1.choose_text_set(a, a + margin + 1e-6, margin)[0] == 'B'            # just outside: B
    assert s1.choose_text_set(a, a - 1.0, margin)[0] == 'A'                      # A far higher
    assert s1.choose_text_set(a, a + 5.0, margin)[0] == 'B'


def test_margins_are_the_spec_ones():
    assert s1.CLU_MARGIN == 0.001 and s1.RET_MARGIN == 0.1


def test_text_set_fallback_is_a_with_no_pick():
    which, reason = s1.choose_text_set(0.5, None, s1.RET_MARGIN)
    assert (which, reason) == ('A', 'fallback')


# ------------------------------------------------------------------------------------------------ the menu clip
@pytest.mark.parametrize('rank', [49, 50])
def test_whitening_sizes_are_clipped_to_the_rank(rank):
    m = s1.clay_menu(rank)
    assert len(m) == 15                                                         # 5 + 5 + 5
    sizes = {ro.size for ro in m if ro.w == 'whiten'}
    assert sizes == {rank}
    assert all(ro.size <= rank for ro in m if ro.w == 'whiten')
    assert {rt_readout.name(ro) for ro in m} >= {'none+k0', 'center+k10', f'whiten{rank}+k3'}
    assert m[0] == rt_readout.R0                                                # r11's tie order kept
    full = {ro.size for ro in rt_readout.menu(768) if ro.w == 'whiten'}
    assert full == {64, 256, 768}                                               # the unclipped menu differs


def test_menu_follows_the_smallest_aspect_rank():
    assert {ro.size for ro in s1.clay_menu(20) if ro.w == 'whiten'} == {20}
    assert {ro.size for ro in s1.clay_menu(100) if ro.w == 'whiten'} == {64, 100}


def test_the_clay_pick_is_scored_on_the_clipped_menu(small, monkeypatch, tmp_path):
    seen = {}
    real = ac_select.arm_table

    def spy(xs, labels, ranges, ros=None, **kw):
        seen['ros'] = ros
        return real(xs, labels, ranges, ros=ros, **kw)
    monkeypatch.setattr(ac_select, 'arm_table', spy)
    monkeypatch.setattr(rt_readout, 'crossfit_all', lambda xs, labels, ros, workers=8: {
        ro: {a: 0.3 for a in ASPECTS} for ro in ros})
    s1.clay_pick(FakeVal(), 'artelingo', 'A', unit={a: 0.01 for a in ASPECTS}, out=tmp_path)
    assert seen['ros'] is not None and len(seen['ros']) == 15
    assert max(ro.size for ro in seen['ros'] if ro.w == 'whiten') == 49
    s1.clay_pick(FakeVal(), 'artelingo', 'B', unit={a: 0.01 for a in ASPECTS}, out=tmp_path)
    assert max(ro.size for ro in seen['ros'] if ro.w == 'whiten') == 50


# ------------------------------------------------------------------------------------------------ the rule
def test_readout_doc_equals_r17s_when_the_menu_is_the_default(small):
    d = FakeVal()
    xs = {(a, None): d.tpips('artelingo', a) for a in ASPECTS}
    labels = {a: d.labels('artelingo', a) for a in ASPECTS}
    unit_ = {a: 0.01 * (i + 1) for i, a in enumerate(ASPECTS)}
    mine = s1.readout_doc(xs, labels, unit_, workers=2)
    theirs = ns_picks.readout_doc(xs, labels, unit_, workers=2)
    assert mine['pick'] == theirs['pick'] and mine['table'] == theirs['table']


def test_readout_doc_uses_the_given_menu(small):
    d = FakeVal()
    xs = {(a, None): d.tpips('artelingo', a) for a in ASPECTS}
    labels = {a: d.labels('artelingo', a) for a in ASPECTS}
    ros = s1.clay_menu(8)
    doc = s1.readout_doc(xs, labels, {a: 0.01 for a in ASPECTS}, ros=ros, workers=2)
    assert set(doc['table']) == {rt_readout.name(r) for r in ros} and len(doc['table']) == 15


def fake_crossfit(gain_by_name):
    """A stand-in for the cross-fit: mAP = 0.2 + gain of the read-out's name (same for the three aspects)."""
    def f(xs, labels, ros, workers=8):
        return {ro: {a: 0.2 + gain_by_name.get(rt_readout.name(ro), 0.0) for a in ASPECTS} for ro in ros}
    return f


def test_tpips_pick_is_one_pick_across_aspects_with_qwen_units(small, monkeypatch, tmp_path):
    monkeypatch.setattr(rt_readout, 'crossfit_all', fake_crossfit({'center+k10': 0.02, 'whiten64+k5': 0.001}))
    rec = s1.tpips_pick(FakeVal(), 'artelingo', unit={a: 0.01 for a in ASPECTS}, out=tmp_path)
    assert rec['pick'] == 'center+k10' and rec['w'] == 'center' and rec['k'] == 10
    assert rec['scores']['score'] == pytest.approx(2.0)                         # (0.02) / 0.01
    assert set(rec['scores']['map']) == set(ASPECTS)
    assert (tmp_path / rec['table_path']).exists()


def test_tpips_menu_is_the_full_one(small, monkeypatch, tmp_path):
    seen = {}
    real = ac_select.arm_table

    def spy(xs, labels, ranges, ros=None, **kw):
        seen['ros'] = ros
        return real(xs, labels, ranges, ros=ros, **kw)
    monkeypatch.setattr(ac_select, 'arm_table', spy)
    monkeypatch.setattr(rt_readout, 'crossfit_all', fake_crossfit({}))
    s1.tpips_pick(FakeVal(), 'artelingo', unit={a: 0.01 for a in ASPECTS}, out=tmp_path)
    assert seen['ros'] is None                                                  # arm_table's default menu at the width


def test_units_are_qwen_for_tpips_and_clip_for_clay(monkeypatch):
    got = []
    monkeypatch.setattr(ns_picks, 'ranges', lambda set_, enc: got.append((set_, enc)) or ({a: 0.01 for a in ASPECTS}, 'src'))
    assert s1.units('tpips', 'mmart')[0] == {a: 0.01 for a in ASPECTS}
    s1.units('clay', 'artelingo')
    assert got == [('mmart', 'qwen'), ('artelingo', 'clip_l14')]


# ------------------------------------------------------------------------------------------------ CLAY's text sets
def stub_ami(values):
    """Replace the clustering score: `values` {(text_set, aspect): mean-over-states AMI at 20}."""
    def f(data, set_, aspect, text_set, partitions=None):
        return values[(text_set, aspect)]
    return f


def test_clustering_text_set_by_mean_ami20_with_the_margin(small, monkeypatch):
    base = {('A', a): 0.30 for a in ASPECTS}
    for delta, want in ((0.001 - 1e-6, 'A'), (0.001 + 1e-5, 'B'), (-0.2, 'A')):
        vals = {**base, **{('B', a): 0.30 + delta for a in ASPECTS}}
        monkeypatch.setattr(s1, 'clay_ami20', stub_ami(vals))
        rec = s1.clay_clustering_choice(FakeVal(), 'artelingo')
        assert rec['set'] == want
        assert rec['scores']['A']['mean_ami20'] == pytest.approx(0.30)
        assert rec['scores']['B']['mean_ami20'] == pytest.approx(0.30 + delta)


def test_clustering_choice_averages_over_aspects(small, monkeypatch):
    vals = {('A', 'emotion'): 0.3, ('A', 'style'): 0.3, ('A', 'genre'): 0.3,
            ('B', 'emotion'): 0.6, ('B', 'style'): 0.2, ('B', 'genre'): 0.2}           # mean 0.333: 0.0333 above A
    monkeypatch.setattr(s1, 'clay_ami20', stub_ami(vals))
    rec = s1.clay_clustering_choice(FakeVal(), 'mmart')
    assert rec['set'] == 'B' and rec['scores']['B']['mean_ami20'] == pytest.approx(1 / 3)
    vals[('B', 'emotion')] = 0.4                                                          # mean 0.2667: below A
    assert s1.clay_clustering_choice(FakeVal(), 'mmart')['set'] == 'A'


def test_clustering_choice_fallback_never_touches_set_b(small, monkeypatch):
    seen = []

    def f(data, set_, aspect, text_set, partitions=None):
        seen.append(text_set)
        return 0.3
    monkeypatch.setattr(s1, 'clay_ami20', f)
    rec = s1.clay_clustering_choice(FakeVal(fallback_b=True), 'artelingo')
    assert rec['set'] == 'A' and rec['scores']['B'] is None and set(seen) == {'A'} and rec['reason'] == 'fallback'


def test_clay_ami20_is_the_mean_over_states_of_k20_on_labelled_rows(small):
    y = np.arange(N, dtype=np.int64) % 3
    y[:5] = -1
    parts = np.stack([flipped(np.arange(N) % 3, f, i) for i, f in enumerate((0.1, 0.4, 0.2))]).astype(np.int32)

    class D(FakeVal):
        def labels(self, set_, aspect):
            return y.copy()

    got = s1.clay_ami20(D(), 'artelingo', 'emotion', 'A', partitions=lambda x, k, s: (k == 20 and s == 'artelingo') and parts)
    want = np.mean([adjusted_mutual_info_score(y[5:], p[5:]) for p in parts])
    assert got == pytest.approx(want, abs=1e-12)


def ret_pick_stub(monkeypatch, gains):
    """Per text set, a fixed best read-out and mAP: gains {text_set: (name, gain)}. Text set read from the call's rows."""
    state = {}
    real_clay = FakeVal.clay

    def clay(self, set_, aspect, text_set):
        state['ts'] = text_set
        return real_clay(self, set_, aspect, text_set)
    monkeypatch.setattr(FakeVal, 'clay', clay)

    def cf(xs, labels, ros, workers=8):
        name, gain = gains[state['ts']]
        return {ro: {a: 0.2 + (gain if rt_readout.name(ro) == name else 0.0) for a in ASPECTS} for ro in ros}
    monkeypatch.setattr(rt_readout, 'crossfit_all', cf)


def test_retrieval_text_set_picked_per_set_then_compared_in_units_of_clip_cstar(small, monkeypatch, tmp_path):
    unit_ = {a: 0.01 for a in ASPECTS}
    # A: pick center+k10 at mAP 0.2 + 0.02; B: pick whiten50+k5 at 0.2 + 0.0005 (+0.0 below A in score units)
    ret_pick_stub(monkeypatch, {'A': ('center+k10', 0.02), 'B': ('whiten50+k5', 0.0005)})
    out = s1.clay_retrieval(FakeVal(), 'artelingo', unit=unit_, out=tmp_path)
    ch = out['clay_textset/ret/artelingo']
    assert ch['set'] == 'A'
    assert ch['scores']['A']['score'] == pytest.approx(22.0) and ch['scores']['A']['pick'] == 'center+k10'
    assert ch['scores']['B']['pick'] == 'whiten50+k5' and ch['scores']['B']['score'] == pytest.approx(20.05)
    assert out['clay/artelingo']['text_set'] == 'A' and out['clay/artelingo']['pick'] == 'center+k10'


def test_retrieval_score_is_map_over_range_not_the_rule_gain(small, monkeypatch, tmp_path):
    """B has a higher mAP at its pick but a lower gain over its own R0: the units (mAP / range) decide."""
    unit_ = {a: 0.1 for a in ASPECTS}
    state = {}
    real_clay = FakeVal.clay

    def clay(self, set_, aspect, text_set):
        state['ts'] = text_set
        return real_clay(self, set_, aspect, text_set)
    monkeypatch.setattr(FakeVal, 'clay', clay)
    base = {'A': 0.20, 'B': 0.40}

    def cf(xs, labels, ros, workers=8):
        b = base[state['ts']]
        return {ro: {a: b + (0.001 if rt_readout.name(ro) == 'center+k0' else 0.0) for a in ASPECTS} for ro in ros}
    monkeypatch.setattr(rt_readout, 'crossfit_all', cf)
    out = s1.clay_retrieval(FakeVal(), 'artelingo', unit=unit_, out=tmp_path)
    ch = out['clay_textset/ret/artelingo']
    assert ch['scores']['A']['score'] == pytest.approx(2.01) and ch['scores']['B']['score'] == pytest.approx(4.01)
    assert ch['set'] == 'B' and out['clay/artelingo']['text_set'] == 'B'
    assert out['clay/artelingo']['scores']['r0_map']['emotion'] == pytest.approx(0.40)


def test_retrieval_margin_goes_to_a(small, monkeypatch, tmp_path):
    unit_ = {a: 0.1 for a in ASPECTS}
    for diff, want in ((0.1 - 1e-4, 'A'), (0.1 + 1e-4, 'B')):                    # in units of the range: diff / 0.1 ... see below
        # score = mAP / 0.1; a score gap of 0.1 needs an mAP gap of 0.01
        gap = diff * 0.1
        ret_pick_stub(monkeypatch, {'A': ('none+k0', 0.0), 'B': ('none+k0', gap)})
        out = s1.clay_retrieval(FakeVal(), 'artelingo', unit=unit_, out=tmp_path)
        # the stub puts the gain on every read-out named none+k0, i.e. R0 itself: mAP(B) = 0.2 + gap
        assert out['clay_textset/ret/artelingo']['set'] == want


def test_retrieval_fallback_gives_a_and_no_set_b_pick(small, monkeypatch, tmp_path):
    ret_pick_stub(monkeypatch, {'A': ('center+k10', 0.02)})
    out = s1.clay_retrieval(FakeVal(fallback_b=True), 'mmart', unit={a: 0.01 for a in ASPECTS}, out=tmp_path)
    ch = out['clay_textset/ret/mmart']
    assert ch['set'] == 'A' and ch['scores']['B'] is None and ch['reason'] == 'fallback'
    assert out['clay/mmart']['text_set'] == 'A'
    assert not list((tmp_path / 'stage1' / 'tables').glob('*__B.json'))


# ------------------------------------------------------------------------------------------------ validation only
class FakeOd:
    """Stands in for om_data: records the split of every call."""

    def __init__(self):
        self.splits = []

    def keys(self, set_, split):
        self.splits.append(split)
        return [f'{set_}{split}{i}' for i in range(3)]

    def labels(self, set_, split, aspect):
        self.splits.append(split)
        return np.zeros(3, dtype=np.int64)

    def cstar(self, set_, split, enc, aspect, seed, unit=True):
        self.splits.append(split)
        assert unit is False, 'ranges use the clustering form of c* (r10, r14, r15)'
        return np.zeros((3, 2), dtype=np.float32)

    def tpips(self, set_, split, aspect, roots=None, name_of=None):
        self.splits.append(split)
        return np.zeros((3, 2), dtype=np.float32)

    def clay(self, set_, split, aspect, text_set, out=None):
        self.splits.append(split)
        return np.zeros((3, 2), dtype=np.float32)


def test_valdata_reads_only_the_validation_split():
    od = FakeOd()
    d = s1.ValData(od=od)
    d.keys('artelingo'), d.labels('mmart', 'style'), d.cstar('artelingo', 'qwen', 'emotion', 0)
    d.tpips('mmart', 'genre'), d.clay('artelingo', 'style', 'A')
    assert set(od.splits) == {'val'} and len(od.splits) == 5
    d.held_keys('mmart')                                                         # keys only, to check disjointness
    assert od.splits[-1] == 'test'


def test_valdata_methods_take_no_split_argument():
    import inspect
    for name in ('keys', 'labels', 'cstar', 'tpips', 'clay'):
        assert 'split' not in inspect.signature(getattr(s1.ValData, name)).parameters, name


def test_audit_refuses_keys_shared_with_the_held_split(small):
    s1.audit_validation_only(FakeVal(), 'artelingo')
    with pytest.raises(AssertionError, match='held'):
        s1.audit_validation_only(FakeVal(held_overlap=True), 'artelingo')
    with pytest.raises(AssertionError, match='held'):
        s1.audit_validation_only(FakeVal(held_overlap=True), 'mmart')


def test_audit_refuses_a_wrong_row_count(monkeypatch):
    d = FakeVal()
    with pytest.raises(AssertionError):
        s1.audit_validation_only(d, 'artelingo')                               # N=40 against the real 6,152


# ------------------------------------------------------------------------------------------------ the steps
class Unsorted(FakeVal):
    """Validation keys not in the evaluation order of the set."""

    def keys(self, set_):
        k = super().keys(set_)
        return k[::-1]


def stub_clay_step(monkeypatch, tmp_path):
    ret_pick_stub(monkeypatch, {'A': ('center+k10', 0.02), 'B': ('whiten50+k5', 0.0005)})
    monkeypatch.setattr(s1, 'units', lambda arm, set_: ({a: 0.01 for a in ASPECTS}, 'stub'))
    monkeypatch.setattr(s1, 'clay_ami20', stub_ami({**{('A', a): 0.30 for a in ASPECTS}, **{('B', a): 0.40 for a in ASPECTS}}))


def test_step_clay_writes_every_key_and_the_pins(small, monkeypatch, tmp_path):
    stub_clay_step(monkeypatch, tmp_path)
    recs = s1.step_clay(FakeVal(), 'mmart', 2, out=tmp_path)
    keys = {'clay/mmart', 'clay_textset/ret/mmart', 'clay_textset/clu/mmart'}
    assert set(recs) == keys
    doc = json.loads((tmp_path / 'stage1' / 'om_picks.json').read_text())
    assert set(doc) == keys | {s1.PIN_KEY}
    assert doc[s1.PIN_KEY] == s1.text_pins(FakeVal()) and doc[s1.PIN_KEY]['A']['style'] == 'sha-A-style'
    assert doc['clay_textset/clu/mmart']['set'] == 'B' and doc['clay_textset/ret/mmart']['set'] == 'A'   # each its own rule
    assert doc['clay/mmart']['pick'] == 'center+k10' and doc['clay/mmart']['text_set'] == 'A'
    for ts in ('A', 'B'):
        assert (tmp_path / 'stage1' / 'tables' / f'clay__mmart__{ts}.json').exists()


def test_step_clay_refuses_validation_keys_in_the_held_split_in_both_places(small, monkeypatch, tmp_path):
    stub_clay_step(monkeypatch, tmp_path)
    bad = FakeVal(held_overlap=True)
    with pytest.raises(AssertionError, match='held'):
        s1.clay_pick(bad, 'artelingo', 'A', unit={a: 0.01 for a in ASPECTS}, out=tmp_path)      # the pick's own audit
    with pytest.raises(AssertionError, match='held'):
        s1.clay_clustering_choice(bad, 'artelingo')                                              # the choice's own audit
    with pytest.raises(AssertionError, match='held'):
        s1.step_clay(bad, 'artelingo', 2, out=tmp_path)
    assert not list(tmp_path.rglob('om_picks.json'))


@pytest.mark.parametrize('set_', ['artelingo', 'mmart'])
def test_step_clay_refuses_keys_out_of_the_evaluation_order(small, monkeypatch, tmp_path, set_):
    stub_clay_step(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match='order'):
        s1.step_clay(Unsorted(), set_, 2, out=tmp_path)
    assert not list(tmp_path.rglob('om_picks.json'))


class NoQwenGenre(FakeVal):
    """c* exports of Qwen on the `genre` aspect are missing; the others are random rows (k-means as r10 runs on them)."""

    def cstar(self, set_, enc, aspect, seed):
        if aspect == 'genre':
            raise FileNotFoundError(f'cstar {set_} {enc} {aspect} {seed}')
        return unit(np.random.default_rng(seed).normal(size=(N, 4)))


def test_step_ranges_carries_a_null_qwen_entry_when_its_export_is_missing(small, tmp_path, monkeypatch):
    (tmp_path / 'src').mkdir()
    dirs, _ = write_fake_sources(tmp_path / 'src', r17=False)
    monkeypatch.setattr(s1, 'RECORD_DIRS', dirs)
    monkeypatch.setattr(s1, 'R17_FILE', 'no_such_stage1_ranges.json')                  # absent: r17 keys null as well
    path, table = s1.step_ranges(NoQwenGenre(), out=tmp_path)
    assert json.loads(path.read_text()) == table and path == tmp_path / 'stage1' / 'om_ranges.json'
    g = table['artelingo/qwen/genre/ami/20']
    assert g['r'] is None and g['source'].startswith('missing:') and 'FileNotFoundError' in g['source']
    e = table['artelingo/qwen/emotion/ami/20']
    assert e['r'] is not None and np.isfinite(e['r']) and len(e['per_seed']) == 5      # the other aspects survive
    assert table['artelingo/qwen/style/ami/20']['r'] is not None
    assert table['artelingo/siglip2/genre/ami/20']['r'] is not None                    # r14 untouched
    with pytest.raises(AssertionError, match='held'):
        s1.step_ranges(FakeVal(held_overlap=True), out=tmp_path)


# ------------------------------------------------------------------------------------------------ TPIPS store missing
def test_missing_tpips_store_crashes_loudly_and_writes_nothing(small, tmp_path, monkeypatch):
    names = [f'p{i:03d}' for i in range(N)]
    monkeypatch.setattr(om_data, 'keys', lambda s, sp: list(names))
    monkeypatch.setattr(om_data, 'labels', lambda s, sp, a: np.arange(N) % 3)
    d = s1.ValData(tpips_roots={'artelingo': tmp_path / 'nowhere'})
    monkeypatch.setattr(d, 'held_keys', lambda set_: [f'h{i}' for i in range(N)])
    with pytest.raises(FileNotFoundError):
        s1.step_tpips(d, 'artelingo', 1, out=tmp_path / 'out')
    assert not (tmp_path / 'out').exists() or not list((tmp_path / 'out').rglob('om_picks.json'))


# ------------------------------------------------------------------------------------------------ the index file
def test_merge_pick_keeps_other_keys(tmp_path):
    p = tmp_path / 'om_picks.json'
    s1.merge_picks(p, {'tpips/artelingo': {'pick': 'none+k0'}})
    s1.merge_picks(p, {'clay/artelingo': {'pick': 'center+k10'}, 'clay_textset/clu/artelingo': {'set': 'A'}})
    s1.merge_picks(p, {'tpips/artelingo': {'pick': 'center+k3'}})
    doc = json.loads(p.read_text())
    assert doc['tpips/artelingo']['pick'] == 'center+k3' and doc['clay/artelingo']['pick'] == 'center+k10'
    assert set(doc) == {'tpips/artelingo', 'clay/artelingo', 'clay_textset/clu/artelingo'}


def test_text_pins_cover_both_sets_and_all_aspects(small):
    pins = s1.text_pins(FakeVal())
    assert set(pins) == {'A', 'B'} and set(pins['A']) == set(ASPECTS) == set(pins['B'])
    assert pins['B']['style'] == 'sha-B-style'
    assert s1.text_pins(FakeVal(fallback_b=True))['B'] == {a: None for a in ASPECTS}


def test_pins_are_frozen_once_written(tmp_path):
    p = tmp_path / 'om_picks.json'
    pins = {'A': {a: f'a-{a}' for a in ASPECTS}, 'B': {a: f'b-{a}' for a in ASPECTS}}
    s1.merge_picks(p, {}, pins=pins)
    assert json.loads(p.read_text())[s1.PIN_KEY] == pins
    s1.merge_picks(p, {'tpips/mmart': {'pick': 'none+k0'}}, pins=pins)           # same pins: fine
    changed = {'A': pins['A'], 'B': dict(pins['B'], style='b-other')}
    with pytest.raises(AssertionError, match='frozen'):
        s1.merge_picks(p, {}, pins=changed)
    assert json.loads(p.read_text())[s1.PIN_KEY] == pins                          # nothing was written


@pytest.mark.skipif(not (op.CLAY_B_DIR / 'lists.json').exists(), reason='set B not generated under this OUT')
def test_real_text_pins_match_the_cache_hash_form():
    """The pin is om_data's text_sha256 (array_sha256 of the float32 rows): ticket 10 and the CLAY cache use the same."""
    import om_clay
    d = s1.ValData()
    pins = s1.text_pins(d)
    t = om_clay.text_set('A', 'style')
    assert pins['A']['style'] == om_data.array_sha256(t) and len(pins['A']['style']) == 64
