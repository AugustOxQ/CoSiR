"""Tests for ac_stats (ticket 08): seed ranges, the SC1 to SC8 table, outcomes, the verdict, the yes/no line and the CRL
claims of spec 2026-10-30, sections 3.4 and 6. Verdict code: written test first, through the public functions.

Seams: outcome / seed_range / ranges_doc (numbers -> outcome, ranges), SC / r_for / comparator_arm / readouts (the
table), sc_pass / verdict / repeats_line (outcomes -> decisions), sc_entry / stage2_doc / stage3_doc (the verdict
files), arm_retrieval / retrieval_sc (embeddings -> d, interval, outcome), measure_seed_ranges (embeddings -> r),
replay_r11_stage5 (the slow test replays r11's stage 5).
"""
import itertools
import json

import numpy as np
import pytest

import ac_paths  # noqa: F401  (puts r11's round on sys.path)
import ac_stats

OUT = ('WIN', 'LEVEL', 'BELOW')


# ---- outcome and seed ranges ---------------------------------------------------------------------------------------

def test_outcome_boundaries():
    r = 0.01
    assert ac_stats.outcome(1e-12, r) == 'WIN'
    assert ac_stats.outcome(0.0, r) == 'LEVEL'              # lower = 0 is not a WIN (strict)
    assert ac_stats.outcome(-0.005, r) == 'LEVEL'
    assert ac_stats.outcome(-r, r) == 'BELOW'               # lower = -r is not a LEVEL (strict)
    assert ac_stats.outcome(-r - 1e-12, r) == 'BELOW'
    assert ac_stats.outcome(0.3, r) == 'WIN'


def test_outcome_refuses_a_bad_seed_range():
    for r in (-0.1, float('nan'), float('inf')):
        with pytest.raises(AssertionError):
            ac_stats.outcome(0.0, r)


def test_seed_range_is_max_minus_min_over_five_seeds_unrounded():
    assert ac_stats.seed_range([0.30, 0.34, 0.31, 0.29, 0.33]) == pytest.approx(0.05, abs=1e-15)
    v = [0.1234567890123, 0.2, 0.3, 0.4, 0.4567890123457]
    assert ac_stats.seed_range(v) == 0.4567890123457 - 0.1234567890123
    with pytest.raises(AssertionError):
        ac_stats.seed_range([0.1, 0.2, 0.3])


def test_ranges_doc_keeps_the_per_seed_values_and_each_range():
    clu = [0.10, 0.12, 0.11, 0.105, 0.115]
    mp = [0.20, 0.21, 0.205, 0.2, 0.19]
    nd = [0.5, 0.52, 0.51, 0.5, 0.49]
    doc = ac_stats.ranges_doc(clu, mp, nd)
    assert set(doc) == {'clu', 'map', 'ndcg'}
    assert doc['clu'] == {'per_seed': clu, 'r': 0.12 - 0.10}
    assert doc['map']['r'] == 0.21 - 0.19 and doc['ndcg']['r'] == 0.52 - 0.49
    json.dumps(doc, allow_nan=False)


def test_ranges_doc_refuses_a_zero_range():
    with pytest.raises(AssertionError):
        ac_stats.ranges_doc([0.1] * 5, [0.2, 0.21, 0.2, 0.2, 0.2], [0.5, 0.52, 0.5, 0.5, 0.5])


def brute_ap(x, lab):
    """AP per query by loops: gallery = all other rows, stable descending cosine, queries with a relevant item."""
    x = x.astype(np.float64)
    out = []
    for q in range(len(x)):
        g = [i for i in range(len(x)) if i != q]
        order = sorted(g, key=lambda i: -(x[q] @ x[i]))        # sorted is stable: ties keep gallery order
        rel = [lab[i] == lab[q] for i in order]
        if not any(rel):
            continue
        hits, s = 0, 0.0
        for rank, r in enumerate(rel, 1):
            if r:
                hits += 1
                s += hits / rank
        out.append(s / hits)
    return np.array(out)


def brute_ndcg(x, dist):
    x = x.astype(np.float64)
    out = []
    for q in range(len(x)):
        g = [i for i in range(len(x)) if i != q]
        gain = {i: np.minimum(dist[q], dist[i]).sum() for i in g}
        order = sorted(g, key=lambda i: -(x[q] @ x[i]))
        ideal = sorted(g, key=lambda i: -gain[i])
        disc = [1 / np.log2(r + 2) for r in range(len(g))]
        out.append(sum(gain[i] * d for i, d in zip(order, disc)) / sum(gain[i] * d for i, d in zip(ideal, disc)))
    return np.array(out)


def test_measure_seed_ranges_gives_the_per_seed_ami_map_and_ndcg_and_their_ranges():
    import ac_cluster
    from sklearn.metrics import adjusted_mutual_info_score as sk_ami
    n = 120
    xs = {s: blobs(80 + s, n=n, classes=4, noise=0.6 + 0.15 * s)[0] for s in range(5)}
    lab = blobs(80, n=n, classes=4)[1]
    dist = np.random.default_rng(5).dirichlet(np.ones(9), size=n)
    doc = ac_stats.measure_seed_ranges(lambda s: xs[s], lambda s: xs[s], lab, dist, workers=1)
    assert set(doc) == {'clu', 'map', 'ndcg'}
    want_clu = [np.mean([sk_ami(lab, p) for p in ac_cluster.partitions(xs[s])]) for s in range(5)]
    want_map = [brute_ap(xs[s], lab).mean() for s in range(5)]
    want_ndcg = [brute_ndcg(xs[s], dist).mean() for s in range(5)]
    for key, want in (('clu', want_clu), ('map', want_map), ('ndcg', want_ndcg)):
        np.testing.assert_allclose(doc[key]['per_seed'], want, atol=1e-10, rtol=0)
        assert doc[key]['r'] == max(doc[key]['per_seed']) - min(doc[key]['per_seed'])
    assert doc['map']['per_seed'][0] > doc['map']['per_seed'][4]          # the noise grows with the seed index


def test_validation_seed_ranges_read_c_stars_emotion_head_on_all_validation_images(monkeypatch):
    import ac_data
    import ac_cluster
    import rt_metrics
    n = 100
    raw = {s: (blobs(90 + s, n=n, noise=0.6 + 0.1 * s)[0] * 1.0005).astype(np.float16) for s in range(5)}
    xs = {s: ac_data._unit(raw[s]) for s in range(5)}
    lab = blobs(90, n=n)[1]
    dist = np.random.default_rng(6).dirichlet(np.ones(9), size=n)
    calls, stored_calls, clustered, retrieved, graded = [], [], [], [], []

    def embed(arm, split, seed=None, *, head='emotion'):
        calls.append((arm, split, seed, head))
        return xs[seed]

    def stored(arm, split, seed, *, head='emotion'):
        stored_calls.append((arm, split, seed, head))
        return raw[seed]

    partition, retrieval, ndcg = ac_cluster.partitions, rt_metrics.retrieval, rt_metrics.graded

    def cluster(x):
        assert x.dtype == np.float16, 'clustering needs stored rows'
        np.testing.assert_array_equal(x, raw[len(clustered)])
        clustered.append(x)
        return partition(x)

    def metric(real, seen):
        def score(x, *a, **kw):
            assert x.dtype == np.float32, 'retrieval needs unit float32 rows'
            np.testing.assert_array_equal(x, xs[len(seen)])
            seen.append(x)
            return real(x, *a, **kw)
        return score

    monkeypatch.setattr(ac_data, 'embed', embed)
    monkeypatch.setattr(ac_data, 'stored', stored)
    monkeypatch.setattr(ac_cluster, 'partitions', cluster)
    monkeypatch.setattr(rt_metrics, 'retrieval', metric(retrieval, retrieved))
    monkeypatch.setattr(rt_metrics, 'graded', metric(ndcg, graded))
    monkeypatch.setattr(ac_data, 'labels', lambda split: lab if split == 'val' else 1 / 0)
    monkeypatch.setattr(ac_data, 'emotion_dist', lambda split: dist if split == 'val' else 1 / 0)
    doc = ac_stats.validation_seed_ranges(workers=1)
    assert set(calls) == {('cstar', 'val', s, 'emotion') for s in range(5)}      # c*, validation only, emotion head
    assert calls == stored_calls and len(clustered) == len(retrieved) == len(graded) == 5, 'separate inputs per seed'
    assert doc['map']['per_seed'][0] == pytest.approx(brute_ap(xs[0], lab).mean(), abs=1e-10)
    want = [ac_cluster.ami(partition(raw[s]), lab) for s in range(5)]
    np.testing.assert_allclose(doc['clu']['per_seed'], want, atol=1e-12, rtol=0)


@pytest.mark.real_data
def test_r11_r_a_is_the_frozen_map_range_per_aspect():
    r_a = ac_stats.r11_r_a()
    assert set(r_a) == {'emotion', 'style', 'genre'}
    assert r_a['emotion'] == 0.0014116107779215126 and r_a['style'] == 0.002229334648849224


def test_write_ranges_freezes_the_committed_copy(tmp_path):
    doc = ac_stats.ranges_doc([0.1, 0.12, 0.11, 0.105, 0.115], [0.2, 0.21, 0.205, 0.2, 0.19],
                              [0.5, 0.52, 0.51, 0.5, 0.49])
    copy = tmp_path / 'stage0b_ranges.json'
    ac_stats.write_ranges(doc, tmp_path / 'out', copy_to=copy)
    assert json.loads((tmp_path / 'out/stage0b/ranges.json').read_text()) == doc
    assert copy.read_bytes() == (tmp_path / 'out/stage0b/ranges.json').read_bytes()
    assert ac_stats.load_ranges(copy) == doc
    ac_stats.write_ranges(doc, tmp_path / 'out', copy_to=copy)                 # the same content again: allowed
    other = ac_stats.ranges_doc([0.1, 0.13, 0.11, 0.105, 0.115], doc['map']['per_seed'], doc['ndcg']['per_seed'])
    with pytest.raises(AssertionError):
        ac_stats.write_ranges(other, tmp_path / 'out2', copy_to=copy)
    assert json.loads(copy.read_text()) == doc and not (tmp_path / 'out2').exists()


def test_load_ranges_refuses_a_document_whose_r_is_not_the_per_seed_range(tmp_path):
    doc = ac_stats.ranges_doc([0.1, 0.12, 0.11, 0.105, 0.115], [0.2, 0.21, 0.205, 0.2, 0.19],
                              [0.5, 0.52, 0.51, 0.5, 0.49])
    doc['clu']['r'] = 0.5
    f = tmp_path / 'r.json'
    f.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        ac_stats.load_ranges(f)


def test_load_ranges_refuses_a_zero_range_even_when_r_matches_the_per_seed_values(tmp_path):
    # five equal per-seed values: r = 0 = max - min, so only the positivity guard can refuse it
    doc = {k: {'per_seed': v, 'r': ac_stats.seed_range(v)} for k, v in
           (('clu', [0.1] * 5), ('map', [0.2, 0.21, 0.205, 0.2, 0.19]), ('ndcg', [0.5, 0.52, 0.51, 0.5, 0.49]))}
    f = tmp_path / 'r.json'
    f.write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match='clu'):
        ac_stats.load_ranges(f)
    doc['clu'] = {'per_seed': [0.1, 0.12, 0.11, 0.105, 0.115], 'r': 0.12 - 0.1}
    f.write_text(json.dumps(doc))
    assert ac_stats.load_ranges(f)['clu']['r'] == 0.12 - 0.1                  # the same file with a positive range loads


# ---- the replay of r11's stage 5 -----------------------------------------------------------------------------------

ASP = ('emotion', 'style', 'genre')


def fake_artelingo(noise=None):
    """A stand-in for rt_data: validation and held rows; c* seeded with a head per aspect, plain aspect-free,
    instructed with a head per aspect. noise: the blob noise per arm (default: c* the best but plain)."""
    from types import SimpleNamespace
    n = 90
    lab = {a: blobs(100 + i, n=n, classes=3 + i)[1] for i, a in enumerate(ASP)}
    noise = noise or {'cstar': 0.5, 'instructed': 0.6, 'plain': 1.4}

    def embed(arm, split, aspect, seed=None):
        i = 0 if arm == 'plain' else ASP.index(aspect)           # an aspect-free arm ignores the aspect
        k = (0 if split == 'val' else 500) + 10 * i + (seed or 0) + {'cstar': 0, 'plain': 3, 'instructed': 7}[arm]
        return blobs(k, n=n, classes=3 + i, noise=noise[arm])[0]

    return SimpleNamespace(labels=lambda split: lab, embed=embed, ARMS=('cstar', 'plain', 'instructed'),
                           SEEDED={'cstar'}, ASPECT_FREE={'plain'}), lab


R11_PICKS = {'cstar': 'center+k3', 'plain': 'center+k3', 'instructed': 'none+k1'}


def test_replay_of_r11_stage5_gives_the_arm_maps_and_sc10_to_sc15_in_r11s_format():
    data, lab = fake_artelingo()
    import ac_select
    import rt_readout
    menu = rt_readout.menu(16)                       # the fake embeddings are 16 wide (`blobs`, d = 16)
    assert all(ac_select.readout_from_name(name) in menu for name in R11_PICKS.values())
    got = ac_stats.replay_r11_stage5(workers=1, draws=60, data=data, picks=R11_PICKS, ranges=R_A)
    assert set(got) == {'map', 'sc'} and set(got['map']) == {'cstar', 'plain', 'instructed'}
    assert list(got['sc']) == [f'SC{i}' for i in range(10, 16)]
    # per-arm mAP: mean over seeds and the five per-seed values for the seeded arm, None for the others
    c = got['map']['cstar']['style']
    assert len(c['per_seed']) == 5 and c['mean'] == pytest.approx(np.mean(c['per_seed']), abs=1e-15)
    assert got['map']['plain']['genre']['per_seed'] is None
    # by hand for one cell: plain, genre: centre on validation (the aspect-free arm has one matrix), k = 3
    import rt_metrics
    xv, xh = data.embed('plain', 'val', 'genre'), data.embed('plain', 'held', 'genre')
    z = xh.astype(np.float64) - xv.astype(np.float64).mean(axis=0)
    z = (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)
    want = rt_metrics.retrieval(z, lab['genre'], qe_k=3, workers=1)['ap'].mean()
    assert got['map']['plain']['genre']['mean'] == pytest.approx(want, abs=1e-12)
    # the table: SC10 to SC12 against plain, SC13 to SC15 against instructed, one comparison each
    for i, a in enumerate(ASP):
        for sid, comp in ((f'SC{10 + i}', 'plain'), (f'SC{13 + i}', 'instructed')):
            comp_ = got['sc'][sid]['comparisons']
            assert len(comp_) == 1
            e = comp_[0]
            assert e['candidate'] == ['cstar', a] and e['aspect'] == a and e['r'] == R_A[a]
            assert e['comparator'][0] == comp
            assert e['d'] == pytest.approx(got['map']['cstar'][a]['mean'] - got['map'][comp][a]['mean'], abs=1e-12)
            assert e['outcome'] == ac_stats.outcome(e['interval'][0], R_A[a])
            bar = ('WIN',) if comp == 'plain' else ('WIN', 'LEVEL')       # r11's bars: plain WIN, instructed LEVEL or WIN
            assert got['sc'][sid]['pass'] == (e['outcome'] in bar)


def test_replay_refuses_an_arm_without_a_pick():
    data, _ = fake_artelingo()
    with pytest.raises(AssertionError):
        ac_stats.replay_r11_stage5(workers=1, draws=10, data=data, picks={'cstar': 'center+k3'}, ranges=R_A)


def hand_ap(x_val, x_held, labels, centre, k):
    """mAP of one cell by hand: centre on the validation mean (float64) or just re-normalise, one cast to float32,
    then r11's retrieval with the expansion k (the read-out arithmetic is written out here, not taken from ac_stats)."""
    import rt_metrics
    z = x_held.astype(np.float64) - (x_val.astype(np.float64).mean(axis=0) if centre else 0.0)
    z = (z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    return float(rt_metrics.retrieval(z, labels, qe_k=k, workers=1)['ap'].mean())


def test_replay_scores_each_arms_own_head_per_aspect_by_hand():
    # Mutation guard: a replay that always read c*'s (or any arm's) emotion head would pass the structure checks above;
    # each cell's per-seed mAP is recomputed here from that aspect's own head and that aspect's own labels.
    data, lab = fake_artelingo()
    got = ac_stats.replay_r11_stage5(workers=1, draws=20, data=data, picks=R11_PICKS, ranges=R_A)
    for a in ASP:
        want = [hand_ap(data.embed('cstar', 'val', a, s), data.embed('cstar', 'held', a, s), lab[a], True, 3)
                for s in range(5)]
        np.testing.assert_allclose(got['map']['cstar'][a]['per_seed'], want, atol=1e-12, rtol=0)
        want_i = hand_ap(data.embed('instructed', 'val', a), data.embed('instructed', 'held', a), lab[a], False, 1)
        assert got['map']['instructed'][a]['mean'] == pytest.approx(want_i, abs=1e-12)        # 'none+k1', own head
    # the three c* heads really are different data (else the check above could not tell them apart)
    cells = [got['map']['cstar'][a]['per_seed'] for a in ASP]
    assert len({tuple(c) for c in cells}) == 3
    for a in ('style', 'genre'):
        emo = [hand_ap(data.embed('cstar', 'val', 'emotion', s), data.embed('cstar', 'held', 'emotion', s),
                       lab[a], True, 3) for s in range(5)]
        assert not np.allclose(got['map']['cstar'][a]['per_seed'], emo, atol=1e-6)


def test_replay_scores_each_sc_against_its_own_aspects_range():
    # Mutation guard: c* is worse than the instructed arm everywhere, so SC13 to SC15 are LEVEL or BELOW depending on
    # r; a replay that scored every SC with the emotion range would flip SC14 (and SC15) under these two range sets.
    data, _ = fake_artelingo(noise={'cstar': 1.0, 'instructed': 0.7, 'plain': 1.6})
    cases = [({'emotion': 0.05, 'style': 0.2, 'genre': 0.05}, {'SC13': 'BELOW', 'SC14': 'LEVEL', 'SC15': 'BELOW'}),
             ({'emotion': 0.2, 'style': 0.05, 'genre': 0.2}, {'SC13': 'LEVEL', 'SC14': 'BELOW', 'SC15': 'LEVEL'})]
    for r_a, want in cases:
        got = ac_stats.replay_r11_stage5(workers=1, draws=60, data=data, picks=R11_PICKS, ranges=r_a)
        for sid, outcome_ in want.items():
            e = got['sc'][sid]['comparisons'][0]
            assert -0.2 < e['interval'][0] < -0.05, (sid, e['interval'])        # the lower bound sits between the r's
            assert e['r'] == r_a[e['aspect']] and e['outcome'] == outcome_, (sid, e)
            assert got['sc'][sid]['pass'] == (outcome_ == 'LEVEL'), sid          # SC13 to SC15: LEVEL or WIN
        for sid in ('SC10', 'SC11', 'SC12'):                                     # c* beats plain by a wide margin
            assert got['sc'][sid]['comparisons'][0]['outcome'] == 'WIN', sid


R11_STAGE5 = ac_paths.R11_OUT / 'stage5/verdict.json'


@pytest.mark.slow
@pytest.mark.real_data
def test_replay_matches_r11s_saved_stage5_verdict():
    """Stage 0a(iii), r11 stage 5 part (run by the controller): every arm's mAP (mean and per-seed values) and
    SC10 to SC15's d, interval, r and outcome within 1e-9 of r11's saved stage5/verdict.json. r11's stage 5 took
    235 s with 8 workers (c* and the other arms, 10,000 draws); the replay does the same work."""
    saved = json.loads(R11_STAGE5.read_text())
    got = ac_stats.replay_r11_stage5(workers=8, draws=saved['draws'])
    assert list(got['map']) == saved['arms']
    for arm, by_aspect in saved['map'].items():
        for a, cell in by_aspect.items():
            assert abs(got['map'][arm][a]['mean'] - cell['mean']) < 1e-9, (arm, a)
            if cell['per_seed'] is None:
                assert got['map'][arm][a]['per_seed'] is None, (arm, a)
            else:
                assert len(got['map'][arm][a]['per_seed']) == len(cell['per_seed']) == 5, (arm, a)
                assert max(abs(g - s) for g, s in zip(got['map'][arm][a]['per_seed'], cell['per_seed'])) < 1e-9, (arm, a)
    assert list(got['sc']) == list(saved['sc'])
    for sid, doc in saved['sc'].items():
        assert got['sc'][sid]['pass'] == doc['pass'], sid
        assert len(got['sc'][sid]['comparisons']) == len(doc['comparisons']), sid
        for g, s in zip(got['sc'][sid]['comparisons'], doc['comparisons']):
            assert abs(g['d'] - s['d']) < 1e-9 and abs(g['r'] - s['r']) < 1e-12, sid
            assert max(abs(x - y) for x, y in zip(g['interval'], s['interval'])) < 1e-9, sid
            assert (g['outcome'], g['candidate'], g['comparator'], g['aspect']) == \
                (s['outcome'], s['candidate'], s['comparator'], s['aspect']), sid


# ---- the SC table --------------------------------------------------------------------------------------------------

RANGES = {'clu': {'per_seed': [0] * 5, 'r': 0.011}, 'map': {'per_seed': [0] * 5, 'r': 0.0022},
          'ndcg': {'per_seed': [0] * 5, 'r': 0.0033}}
R_A = {'emotion': 0.0014, 'style': 0.0025, 'genre': 0.0033}


def test_the_table_has_eight_rows_with_the_specs_data_metric_and_bar():
    assert list(ac_stats.SC) == [f'SC{i}' for i in range(1, 9)]
    want = {  # id: (stage, dataset, aspect, metric, bar)
        'SC1': (2, 'affection', 'emotion', 'ami', 'WIN'),
        'SC2': (2, 'affection', 'emotion', 'map', 'WIN'),
        'SC3': (2, 'affection', 'emotion', 'ami', 'LEVEL_OR_WIN'),
        'SC4': (2, 'affection', 'emotion', 'map', 'LEVEL_OR_WIN'),
        'SC5': (3, 'artelingo', 'emotion', 'map', 'WIN'),
        'SC6': (3, 'artelingo', 'style', 'map', 'WIN'),
        'SC7': (3, 'artelingo', 'genre', 'map', 'WIN'),
        'SC8': (3, 'affection', 'emotion', 'map', 'WIN'),
    }
    for sid, (stage, dataset, aspect, metric, bar) in want.items():
        sc = ac_stats.SC[sid]
        assert (sc.stage, sc.dataset, sc.aspect, sc.metric, sc.bar) == (stage, dataset, aspect, metric, bar), sid
        assert sc.candidate == 'cstar'


def test_r_mapping_per_sc():
    # SC1 and SC3 clustering: r_clu; SC2, SC4 and SC8: r_ret,mAP; SC5 to SC7: r11's r_a of their aspect (never r_ret)
    got = {sid: ac_stats.r_for(sid, RANGES, R_A) for sid in ac_stats.SC}
    assert got == {'SC1': 0.011, 'SC2': 0.0022, 'SC3': 0.011, 'SC4': 0.0022,
                   'SC5': 0.0014, 'SC6': 0.0025, 'SC7': 0.0033, 'SC8': 0.0022}


def test_r_for_refuses_a_range_that_is_zero_negative_or_not_finite():
    # r11's `outcome` accepts r = 0 (LEVEL impossible); a range read from a file must be finite and positive here
    for bad in (0.0, -0.001, float('nan'), float('inf')):
        for sid, ranges, r_a in (('SC1', {**RANGES, 'clu': {'r': bad}}, R_A), ('SC2', {**RANGES, 'map': {'r': bad}}, R_A),
                                 ('SC6', RANGES, {**R_A, 'style': bad})):
            with pytest.raises(AssertionError):
                ac_stats.r_for(sid, ranges, r_a)
    assert ac_stats.r_for('SC6', RANGES, R_A) == 0.0025              # the good values still pass


def picks_fixture(wc='artwork', wr='photo', affection_crl='qwen_photo', art_crl='clip_painting'):
    return {
        'affection': {
            'readout': {'cstar': 'center+k10', 'plain': 'none+k3', 'teacher_artwork': 'center+k5',
                        'teacher_photo': 'whiten256+k1', 'clip': 'none+k0', 'percept': 'none+k0',
                        'crl_clip_painting': 'none+k0', 'crl_clip_photo': 'none+k1', 'crl_qwen_painting': 'none+k5',
                        'crl_qwen_photo': 'whiten64+k10'},
            'teacher_wording': {'clustering': wc, 'retrieval': wr}, 'crl_version': affection_crl},
        'artelingo': {'readout': {'cstar': 'center+k10', 'crl_clip_painting': 'none+k10',
                                  'crl_qwen_painting': 'none+k3'}, 'crl_version': art_crl},
    }


def test_comparators_follow_the_picks():
    p = picks_fixture(wc='artwork', wr='photo', affection_crl='qwen_photo', art_crl='clip_painting')
    got = {sid: ac_stats.comparator_arm(sid, p) for sid in ac_stats.SC}
    assert got == {'SC1': 'plain', 'SC2': 'plain',
                   'SC3': 'teacher_artwork',       # the clustering wording pick
                   'SC4': 'teacher_photo',         # the retrieval wording pick
                   'SC5': 'crl_clip_painting', 'SC6': 'crl_clip_painting', 'SC7': 'crl_clip_painting',
                   'SC8': 'crl_qwen_photo'}
    p = picks_fixture(wc='photo', wr='artwork', art_crl='qwen_painting')
    assert ac_stats.comparator_arm('SC3', p) == 'teacher_photo'
    assert ac_stats.comparator_arm('SC4', p) == 'teacher_artwork'
    assert ac_stats.comparator_arm('SC6', p) == 'crl_qwen_painting'


def test_readouts_of_each_side_are_the_picked_ones_and_none_for_clustering():
    p = picks_fixture()
    assert ac_stats.readouts('SC1', p) == (None, None)
    assert ac_stats.readouts('SC3', p) == (None, None)
    assert ac_stats.readouts('SC2', p) == ('center+k10', 'none+k3')
    assert ac_stats.readouts('SC4', p) == ('center+k10', 'whiten256+k1')              # retrieval wording: photo
    assert ac_stats.readouts('SC5', p) == ('center+k10', 'none+k10')                  # ArtELingo's c* and CLIP CRL
    assert ac_stats.readouts('SC8', p) == ('center+k10', 'whiten64+k10')              # Affection's c* and Qwen photo CRL
    p['affection']['readout']['cstar'] = 'none+k1'
    assert ac_stats.readouts('SC8', p)[0] == 'none+k1'
    assert ac_stats.readouts('SC5', p)[0] == 'center+k10'                             # ArtELingo has its own pick


# ---- sc_pass, verdict, repeats_line --------------------------------------------------------------------------------

def test_sc_pass_applies_each_rows_bar():
    for sid in ('SC1', 'SC2', 'SC5', 'SC6', 'SC7', 'SC8'):                            # WIN bars
        assert [ac_stats.sc_pass(sid, o) for o in OUT] == [True, False, False], sid
    for sid in ('SC3', 'SC4'):                                                        # LEVEL or WIN bars
        assert [ac_stats.sc_pass(sid, o) for o in OUT] == [True, True, False], sid
    with pytest.raises(AssertionError):
        ac_stats.sc_pass('SC1', 'win')


def expected_verdict(o1, o2, o3, o4):
    """Spec 6 by hand: SC1 and SC2 need WIN, SC3 and SC4 need LEVEL or WIN."""
    first_two = o1 == 'WIN' and o2 == 'WIN'
    level_or_win = [o in ('WIN', 'LEVEL') for o in (o3, o4)]
    if first_two and all(level_or_win):
        return 'holds'
    if first_two and sum(level_or_win) == 1:
        return 'qualified'
    return 'fails'


@pytest.mark.parametrize('combo', list(itertools.product(OUT, repeat=4)))
def test_verdict_over_all_81_outcome_combinations_of_sc1_to_sc4(combo):
    outcomes = dict(zip(('SC1', 'SC2', 'SC3', 'SC4'), combo))
    assert ac_stats.verdict(outcomes) == expected_verdict(*combo)


def test_verdict_counts_per_class():
    counts = {'holds': 0, 'qualified': 0, 'fails': 0}
    for combo in itertools.product(OUT, repeat=4):
        counts[ac_stats.verdict(dict(zip(('SC1', 'SC2', 'SC3', 'SC4'), combo)))] += 1
    # SC1 = SC2 = WIN: SC3, SC4 in {WIN, LEVEL} both: 4 holds; exactly one BELOW: 2 x 2 = 4 qualified; the rest fail
    assert counts == {'holds': 4, 'qualified': 4, 'fails': 73}


def test_verdict_named_cases():
    base = {'SC1': 'WIN', 'SC2': 'WIN', 'SC3': 'LEVEL', 'SC4': 'LEVEL'}
    assert ac_stats.verdict(base) == 'holds'
    assert ac_stats.verdict({**base, 'SC4': 'BELOW'}) == 'qualified'
    assert ac_stats.verdict({**base, 'SC3': 'BELOW'}) == 'qualified'
    assert ac_stats.verdict({**base, 'SC3': 'BELOW', 'SC4': 'BELOW'}) == 'fails'
    assert ac_stats.verdict({**base, 'SC1': 'LEVEL'}) == 'fails'          # beating the plain image is required
    assert ac_stats.verdict({**base, 'SC2': 'BELOW', 'SC4': 'BELOW'}) == 'fails'


def test_verdict_needs_exactly_sc1_to_sc4():
    with pytest.raises(AssertionError):
        ac_stats.verdict({'SC1': 'WIN', 'SC2': 'WIN', 'SC3': 'WIN'})
    with pytest.raises(AssertionError):
        ac_stats.verdict({'SC1': 'WIN', 'SC2': 'WIN', 'SC3': 'WIN', 'SC4': 'WIN', 'SC5': 'WIN'})


def test_repeats_line_is_yes_only_when_sc3_is_a_win():
    assert [ac_stats.repeats_line(o) for o in OUT] == ['yes', 'no', 'no']
    with pytest.raises(AssertionError):
        ac_stats.repeats_line('holds')


# ---- the verdict files' `sc` entries, stage 2 and stage 3 documents ------------------------------------------------

ENTRY_KEYS = {'candidate', 'comparator', 'dataset', 'aspect', 'metric', 'readout_c', 'readout_t', 'point_c', 'point_t',
              'd', 'interval', 'r', 'outcome', 'pass'}


def entry(sid, lower, picks=None, r=None, d=0.01):
    picks = picks or picks_fixture()
    r = r if r is not None else ac_stats.r_for(sid, RANGES, R_A)
    return ac_stats.sc_entry(sid, picks, point_c=0.30, point_t=0.29, d=d, interval=[lower, lower + 0.02], r=r)


def test_sc_entry_has_the_index_format_and_names_the_picked_arms():
    e = entry('SC4', 0.001)
    assert set(e) == ENTRY_KEYS
    assert (e['candidate'], e['comparator'], e['dataset'], e['aspect'], e['metric']) == \
        ('cstar', 'teacher_photo', 'affection', 'emotion', 'map')
    assert (e['readout_c'], e['readout_t']) == ('center+k10', 'whiten256+k1')
    assert (e['point_c'], e['point_t'], e['d'], e['interval'], e['r']) == (0.30, 0.29, 0.01, [0.001, 0.021], 0.0022)
    e = entry('SC1', 0.001)
    assert (e['readout_c'], e['readout_t'], e['metric'], e['comparator']) == (None, None, 'ami', 'plain')
    json.dumps(e, allow_nan=False)


def test_sc_entry_outcome_and_pass_follow_the_lower_bound_and_the_scs_r():
    r = 0.0022
    cases = [  # lower bound -> outcome, pass for SC2 (WIN bar), pass for SC4 (LEVEL or WIN bar)
        (0.0001, 'WIN', True, True), (0.0, 'LEVEL', False, True), (-0.002, 'LEVEL', False, True),
        (-r, 'BELOW', False, False), (-0.01, 'BELOW', False, False)]
    for lower, out, p2, p4 in cases:
        e2, e4 = entry('SC2', lower), entry('SC4', lower)
        assert (e2['outcome'], e2['pass']) == (out, p2), lower
        assert (e4['outcome'], e4['pass']) == (out, p4), lower
    # SC6 uses r_a of style (0.0025), not r_ret,mAP (0.0022): -0.0023 is LEVEL for it, BELOW for SC2
    assert entry('SC6', -0.0023)['outcome'] == 'LEVEL' and entry('SC2', -0.0023)['outcome'] == 'BELOW'


def stage2_entries(o):
    """Entries whose outcomes are the given ones, by choosing the lower bound."""
    lower = {'WIN': 0.002, 'LEVEL': -0.001, 'BELOW': -0.5}
    return {sid: entry(sid, lower[o[i]]) for i, sid in enumerate(ac_stats.STAGE2_IDS)}


def test_stage2_doc_carries_the_verdict_and_the_yes_no_line():
    doc = ac_stats.stage2_doc(stage2_entries(['WIN', 'WIN', 'LEVEL', 'BELOW']), rows={'plain': {'map': 0.25}})
    assert set(doc) == {'sc', 'verdict', 'repeats_artelingo_win', 'rows'}
    assert doc['verdict'] == 'qualified' and doc['repeats_artelingo_win'] == 'no'
    assert list(doc['sc']) == ['SC1', 'SC2', 'SC3', 'SC4'] and doc['rows'] == {'plain': {'map': 0.25}}
    doc = ac_stats.stage2_doc(stage2_entries(['WIN', 'WIN', 'WIN', 'LEVEL']), rows={})
    assert doc['verdict'] == 'holds' and doc['repeats_artelingo_win'] == 'yes'
    doc = ac_stats.stage2_doc(stage2_entries(['LEVEL', 'WIN', 'WIN', 'WIN']), rows={})
    assert doc['verdict'] == 'fails'
    json.dumps(doc, allow_nan=False)


def test_stage2_doc_refuses_an_entry_whose_outcome_does_not_match_its_interval():
    sc = stage2_entries(['WIN', 'WIN', 'WIN', 'WIN'])
    sc['SC2'] = dict(sc['SC2'], outcome='BELOW')
    with pytest.raises(AssertionError):
        ac_stats.stage2_doc(sc, rows={})
    sc = stage2_entries(['WIN', 'WIN', 'WIN', 'WIN'])
    sc['SC3'] = dict(sc['SC3'], interval=[-1.0, 1.0])                  # lower bound now says BELOW
    with pytest.raises(AssertionError):
        ac_stats.stage2_doc(sc, rows={})
    with pytest.raises(AssertionError):
        ac_stats.stage2_doc({k: v for k, v in stage2_entries(['WIN'] * 4).items() if k != 'SC4'}, rows={})


@pytest.mark.parametrize('field, wrong', [('dataset', 'artelingo'), ('aspect', 'style'), ('metric', 'ami')])
def test_stage2_doc_refuses_an_entry_that_is_about_another_dataset_aspect_or_metric(field, wrong):
    # outcome and pass still follow from the interval, so only the table check can notice that SC2's entry is not SC2's
    sc = stage2_entries(['WIN', 'WIN', 'WIN', 'WIN'])
    sc['SC2'] = dict(sc['SC2'], **{field: wrong})
    with pytest.raises(AssertionError):
        ac_stats.stage2_doc(sc, rows={})


def test_stage3_doc_refuses_an_entry_that_is_about_another_aspect():
    sc = stage3_entries([0.01] * 4)
    sc['SC7'] = dict(sc['SC7'], aspect='style')               # SC7 is ArtELingo genre
    with pytest.raises(AssertionError):
        ac_stats.stage3_doc(sc, rows={})


def stage3_entries(lowers):
    return {sid: entry(sid, lowers[i]) for i, sid in enumerate(ac_stats.STAGE3_IDS)}


def test_stage3_each_crl_bar_is_its_own_claim():
    sc = stage3_entries([0.01, 0.01, 0.01, 0.01])
    doc = ac_stats.stage3_doc(sc, rows={})
    assert doc['claims'] == {'SC5': True, 'SC6': True, 'SC7': True, 'SC8': True}
    doc = ac_stats.stage3_doc(stage3_entries([0.01, 0.01, 0.01, -0.001]), rows={})
    assert doc['claims'] == {'SC5': True, 'SC6': True, 'SC7': True, 'SC8': False}     # one failure leaves the others


def test_a_failed_crl_claim_keeps_its_row():
    rows = {'crl_clip_painting': {'emotion': 0.2}, 'crl_qwen_photo': {'emotion': 0.27}}
    sc = stage3_entries([0.01, -0.0001, 0.01, -0.5])
    doc = ac_stats.stage3_doc(sc, rows=rows)
    assert doc['claims'] == {'SC5': True, 'SC6': False, 'SC7': True, 'SC8': False}
    assert list(doc['sc']) == ['SC5', 'SC6', 'SC7', 'SC8']                            # every row kept, failed or not
    assert doc['sc']['SC6']['outcome'] == 'LEVEL' and doc['sc']['SC6']['pass'] is False
    assert doc['sc']['SC8']['outcome'] == 'BELOW' and doc['sc']['SC8']['d'] == 0.01
    assert doc['rows'] == rows
    assert 'verdict' not in doc                                                     # stage 3 has claims, no verdict
    json.dumps(doc, allow_nan=False)


def test_stage3_doc_refuses_a_wrong_id_set():
    sc = stage3_entries([0.01] * 4)
    del sc['SC8']
    with pytest.raises(AssertionError):
        ac_stats.stage3_doc(sc, rows={})


# ---- the retrieval comparison with picked read-outs ------------------------------------------------------------------

ANGLES = np.radians([0.0, 10.0, 80.0, 90.0])
TOY = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1).astype(np.float32)   # four unit rows in the plane


def unseeded(x):
    return lambda seed: x


def test_arm_retrieval_gives_per_query_ap_by_hand_on_four_points():
    # labels 0,0,1,1: every query's nearest neighbour is its class mate, AP = 1 for all four
    good = ac_stats.arm_retrieval(unseeded(TOY), unseeded(TOY), 'none+k0', np.array([0, 0, 1, 1]), workers=1)
    np.testing.assert_allclose(good['ap'], [1, 1, 1, 1])
    assert good['ap'].dtype == np.float64 and list(good['rows']) == [0, 1, 2, 3] and good['dropped'] == 0
    assert good['point'] == 1.0 and good['per_seed'] is None
    # labels 0,1,0,1: the mate comes second (queries 0 and 3) or third (queries 1 and 2): AP = 1/2, 1/3, 1/3, 1/2
    bad = ac_stats.arm_retrieval(unseeded(TOY), unseeded(TOY), 'none+k0', np.array([0, 1, 0, 1]), workers=1)
    np.testing.assert_allclose(bad['ap'], [1 / 2, 1 / 3, 1 / 3, 1 / 2])
    assert bad['point'] == pytest.approx(5 / 12)


def blobs(seed, n=240, classes=4, d=16, noise=0.5):
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(99).standard_normal((classes, d))
    lab = np.arange(n) % classes
    x = centers[lab] + noise * rng.standard_normal((n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), lab.astype(np.int64)


def test_arm_retrieval_fits_the_read_out_on_validation_and_applies_it_to_the_scored_split():
    xv, _ = blobs(1)
    xe, lab = blobs(2)
    xe = xe + np.float32(0.5)                                  # the scored split is offset from validation
    xe = (xe / np.linalg.norm(xe, axis=1, keepdims=True)).astype(np.float32)
    got = ac_stats.arm_retrieval(unseeded(xv), unseeded(xe), 'center+k3', lab, workers=1)
    # by hand: centre on the VALIDATION mean (float64), L2-normalise, one cast to float32; expansion k = 3 in retrieval
    z = xe.astype(np.float64) - xv.astype(np.float64).mean(axis=0)
    z = (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)
    import rt_metrics
    want = rt_metrics.retrieval(z, lab, qe_k=3, workers=1)
    np.testing.assert_allclose(got['ap'], want['ap'], atol=1e-12, rtol=0)
    # fitting on the scored split itself would give another answer
    own = ac_stats.arm_retrieval(unseeded(xe), unseeded(xe), 'center+k3', lab, workers=1)
    assert not np.allclose(own['ap'], got['ap'], atol=1e-6)
    # the read-out's k is used: no expansion gives other APs
    plain = ac_stats.arm_retrieval(unseeded(xv), unseeded(xe), 'center+k0', lab, workers=1)
    assert not np.allclose(plain['ap'], got['ap'], atol=1e-6)


def test_a_seeded_arm_scores_each_seed_on_its_own_rows():
    xv = {s: blobs(10 + s)[0] for s in range(5)}
    xe = {s: blobs(20 + s, noise=0.4 + 0.1 * s)[0] for s in range(5)}
    lab = blobs(20)[1]
    got = ac_stats.arm_retrieval(lambda s: xv[s], lambda s: xe[s], 'none+k0', lab, seeds=(0, 1, 2, 3, 4), workers=1)
    assert got['ap'].shape == (5, len(lab)) and len(got['per_seed']) == 5
    assert got['per_seed'] == [float(got['ap'][s].mean()) for s in range(5)]
    assert got['point'] == pytest.approx(np.mean(got['per_seed']), abs=1e-15)
    assert got['per_seed'][0] > got['per_seed'][4]                       # noisier seeds retrieve worse: seeds are separate
    with pytest.raises(AssertionError):
        ac_stats.arm_retrieval(lambda s: xv[s], lambda s: xe[s], 'none+k0', lab, seeds=(0, 1, 2), workers=1)


def test_retrieval_sc_compares_two_arms_with_the_paired_bootstrap():
    xv = {s: blobs(10 + s)[0] for s in range(5)}
    cand = ac_stats.arm_retrieval(lambda s: xv[s], lambda s: blobs(30 + s, noise=0.4)[0], 'none+k0',
                                  blobs(30)[1], seeds=(0, 1, 2, 3, 4), workers=1)
    comp = ac_stats.arm_retrieval(unseeded(xv[0]), unseeded(blobs(40, noise=1.5)[0]), 'none+k0', blobs(30)[1],
                                  workers=1)
    p = picks_fixture()
    e = ac_stats.retrieval_sc('SC2', cand, comp, p, RANGES, R_A, draws=300)
    assert e['outcome'] == 'WIN' and e['pass'] is True and e['interval'][0] > 0
    assert e['point_c'] == cand['point'] and e['point_t'] == comp['point']
    assert e['d'] == pytest.approx(cand['point'] - comp['point'], abs=1e-12)
    assert e['interval'][0] < e['d'] < e['interval'][1]
    assert (e['readout_c'], e['readout_t'], e['comparator']) == ('center+k10', 'none+k3', 'plain')
    swapped = ac_stats.retrieval_sc('SC2', comp, cand, p, RANGES, R_A, draws=300)
    assert swapped['outcome'] == 'BELOW' and swapped['d'] == pytest.approx(-e['d'], abs=1e-12)


def test_retrieval_sc_level_and_below_depend_on_the_scs_r():
    lab = blobs(51)[1]
    clean = ac_stats.arm_retrieval(unseeded(blobs(50)[0]), unseeded(blobs(51, noise=0.5)[0]), 'none+k0', lab, workers=1)
    noisy = ac_stats.arm_retrieval(unseeded(blobs(50)[0]), unseeded(blobs(51, noise=1.2)[0]), 'none+k0', lab, workers=1)
    big = {'clu': {'r': 1.0}, 'map': {'r': 1.0}}
    tiny = {'clu': {'r': 1e-9}, 'map': {'r': 1e-9}}
    p = picks_fixture()
    worse_big = ac_stats.retrieval_sc('SC4', noisy, clean, p, big, R_A, draws=200)
    worse_tiny = ac_stats.retrieval_sc('SC4', noisy, clean, p, tiny, R_A, draws=200)
    assert worse_big['d'] < 0 and worse_big['interval'][1] < 0
    assert (worse_big['outcome'], worse_big['pass']) == ('LEVEL', True)           # SC4's bar is LEVEL or WIN
    assert (worse_tiny['outcome'], worse_tiny['pass']) == ('BELOW', False)
    # identical arms: every draw is 0, so the lower bound is 0 exactly: LEVEL, not WIN, even with a tiny r
    twin = ac_stats.retrieval_sc('SC4', clean, clean, p, tiny, R_A, draws=200)
    assert twin['d'] == 0.0 and twin['interval'] == [0.0, 0.0] and twin['outcome'] == 'LEVEL'


def test_cluster_sc_takes_ac_clusters_paired_result_and_uses_r_clu():
    import ac_cluster
    x, lab = blobs(60, n=300)
    parts5 = np.stack([ac_cluster.partitions(blobs(60 + s, n=300, noise=0.5)[0]) for s in range(5)])
    plain = ac_cluster.partitions(blobs(70, n=300, noise=3.0)[0])
    res = ac_cluster.paired(parts5, plain, lab, draws=50, workers=1)
    e = ac_stats.cluster_sc('SC1', res, picks_fixture(), RANGES, R_A)
    assert (e['metric'], e['comparator'], e['readout_c'], e['readout_t']) == ('ami', 'plain', None, None)
    assert (e['point_c'], e['point_t'], e['d'], e['interval']) == (res['A_cand'], res['A_comp'], res['d'], res['interval'])
    assert e['r'] == 0.011 and e['outcome'] == 'WIN' and e['pass']
    e3 = ac_stats.cluster_sc('SC3', res, picks_fixture(wc='photo'), RANGES, R_A)
    assert e3['comparator'] == 'teacher_photo' and e3['r'] == 0.011
    with pytest.raises(AssertionError):
        ac_stats.cluster_sc('SC2', res, picks_fixture(), RANGES, R_A)


def test_retrieval_sc_refuses_arms_scored_on_different_queries():
    a = ac_stats.arm_retrieval(unseeded(TOY), unseeded(TOY), 'none+k0', np.array([0, 0, 1, 1]), workers=1)
    b = ac_stats.arm_retrieval(unseeded(TOY), unseeded(TOY), 'none+k0', np.array([0, 0, 1, -1]), workers=1)
    assert list(b['rows']) == [0, 1]
    with pytest.raises(AssertionError):
        ac_stats.retrieval_sc('SC2', a, b, picks_fixture(), RANGES, R_A, draws=10)
    with pytest.raises(AssertionError):          # a clustering SC is not scored here
        ac_stats.retrieval_sc('SC1', a, a, picks_fixture(), RANGES, R_A, draws=10)


def test_retrieval_sc_refuses_arms_with_the_same_number_of_queries_but_different_rows():
    # Mutation guard: with different counts rt_stats.paired raises first, so this needs equal counts, unequal rows.
    ang = np.radians([0.0, 10.0, 80.0, 90.0, 45.0])
    x = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
    a = ac_stats.arm_retrieval(unseeded(x), unseeded(x), 'none+k0', np.array([0, 0, 1, 1, -1]), workers=1)
    b = ac_stats.arm_retrieval(unseeded(x), unseeded(x), 'none+k0', np.array([-1, 0, 0, 1, 1]), workers=1)
    assert list(a['rows']) == [0, 1, 2, 3] and list(b['rows']) == [1, 2, 3, 4]
    assert a['ap'].shape == b['ap'].shape
    with pytest.raises(AssertionError, match='different queries'):
        ac_stats.retrieval_sc('SC2', a, b, picks_fixture(), RANGES, R_A, draws=10)
    ac_stats.retrieval_sc('SC2', a, a, picks_fixture(), RANGES, R_A, draws=10)        # the same rows are accepted


@pytest.mark.parametrize('what', ['rows', 'dropped'])
def test_arm_retrieval_refuses_a_seed_scored_on_other_queries(monkeypatch, what):
    # r11's retrieval chooses the query rows from the labels alone, so a seed can differ only if that function
    # changes; the guard (`ac_stats.py` "queries differ between seeds") is exercised by making the third seed differ.
    import rt_metrics
    real = rt_metrics.retrieval
    calls = []

    def other_on_third(x, labels, qe_k=0, workers=8):
        res = real(x, labels, qe_k=qe_k, workers=workers)
        calls.append(1)
        if len(calls) == 3:
            res = dict(res, rows=res['rows'] + 1) if what == 'rows' else dict(res, dropped=res['dropped'] + 1)
        return res

    x, lab = blobs(7, n=40)
    ok = ac_stats.arm_retrieval(unseeded(x), unseeded(x), 'none+k0', lab, seeds=(0, 1, 2, 3, 4), workers=1)
    assert ok['ap'].shape == (5, 40)                                   # unchanged function: every seed accepted
    monkeypatch.setattr(rt_metrics, 'retrieval', other_on_third)
    with pytest.raises(AssertionError, match='queries differ between seeds'):
        ac_stats.arm_retrieval(unseeded(x), unseeded(x), 'none+k0', lab, seeds=(0, 1, 2, 3, 4), workers=1)
