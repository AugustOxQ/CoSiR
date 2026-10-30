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
