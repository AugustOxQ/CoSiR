"""Tests for ac_select (ticket 07): the stage 1 picks of spec 2026-10-30, section 6 "Selection rule (stage 1)".

Seams: pick_readout (cross-fit rows -> read-out), pick_wording, version_score / pick_crl_version, arm_table
(embeddings -> table and pick), write_picks (files). The slow test replays r11's stage 4 through this module.
"""
import json

import numpy as np
import pytest

import ac_paths  # noqa: F401  (puts r11's round on sys.path)
import ac_select
import rt_readout
from rt_readout import R0, ReadOut

EMO = ('emotion',)
ABC = ('emotion', 'style', 'genre')


def ro(w='none', size=None, k=0):
    return ReadOut(w, size, k)


def rows_from_scores(r0, r, scores, aspect='emotion'):
    """Single-aspect rows whose score (mAP - mAP(R0)) / r is the given number: {ReadOut: {aspect: mAP}}."""
    rows = {R0: {aspect: r0}}
    rows.update({x: {aspect: r0 + r * s} for x, s in scores.items()})
    return rows


# ---- read-out pick: tie rule at its boundary --------------------------------------------------------------------

R, M0 = 0.5, 0.25   # powers of two keep the arithmetic exact enough; ties still rely on the slack at the boundary


def pick(scores, **kw):
    return ac_select.pick_readout(rows_from_scores(M0, R, scores), {'emotion': R}, **kw)


def test_clear_winner_is_picked():
    assert pick({ro('center', k=0): 0.2, ro('whiten', 64, 0): 1.5}) == ro('whiten', 64, 0)


def test_all_scores_below_r0_gives_r0():
    assert pick({ro('center'): -0.3, ro('none', k=3): -2.0}) == R0


def test_tie_boundary_exactly_0_001_below_best_is_tied():
    # best 1.0 (late in the tie order); the earlier read-out is exactly 0.001 below: tied, so the earlier wins
    assert pick({ro('whiten', 64, 0): 1.0, ro('center'): 0.999}) == ro('center')


def test_just_outside_the_tie_boundary_loses():
    assert pick({ro('whiten', 64, 0): 1.0, ro('center'): 0.9989}) == ro('whiten', 64, 0)


def test_r0_is_first_in_the_tie_order():
    assert pick({ro('center'): 0.0005, ro('none', k=1): 0.0004}) == R0


@pytest.mark.parametrize('tied,winner', [
    ({ro('center'): 1.0, ro('none', k=1): 1.0}, ro('center')),                           # no expansion before expansion
    ({ro('whiten', 64, 0): 1.0, ro('none', k=5): 1.0}, ro('whiten', 64, 0)),             # k = 0 beats any expansion
    ({ro('none', k=1): 1.0, ro('center', k=0): 1.0}, ro('center', k=0)),
    ({ro('whiten', 64, 1): 1.0, ro('center', k=3): 1.0}, ro('center', k=3)),             # none < center < whiten
    ({ro('none', k=10): 1.0, ro('none', k=3): 1.0, ro('none', k=5): 1.0}, ro('none', k=3)),   # smaller k
    ({ro('whiten', 64, 0): 1.0, ro('whiten', 256, 0): 1.0, ro('whiten', 1024, 0): 1.0}, ro('whiten', 1024, 0)),
])
def test_tie_order_after_r0(tied, winner):
    assert pick(tied) == winner


# ---- three aspects and the guard (ArtELingo's Qwen CRL) ------------------------------------------------------------

RANGES3 = {'emotion': 0.002, 'style': 0.004, 'genre': 0.008}


def rows3(table):
    """{ReadOut: (emotion, style, genre)} -> rows; R0 is (0.30, 0.40, 0.50)."""
    out = {R0: dict(zip(ABC, (0.30, 0.40, 0.50)))}
    out.update({x: dict(zip(ABC, v)) for x, v in table.items()})
    return out


def test_guard_drops_a_read_out_more_than_r_below_r0_on_one_aspect():
    # genre is 0.0081 below R0 (r = 0.008): dropped although the other two aspects gain a lot
    table = {ro('center'): (0.31, 0.41, 0.4919), ro('none', k=1): (0.302, 0.402, 0.502)}
    assert ac_select.pick_readout(rows3(table), RANGES3) == ro('none', k=1)


def test_guard_keeps_a_read_out_exactly_r_below_r0():
    table = {ro('center'): (0.31, 0.41, 0.50 - 0.008)}
    assert ac_select.pick_readout(rows3(table), RANGES3) == ro('center')


# ---- an arm's table from embeddings: cross-fit, seeded averaging, pick ------------------------------------------

N_SYN = 80


def synthetic(seed, n=N_SYN, d=8, classes=3):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, classes, n).astype(np.int64)
    centers = rng.standard_normal((classes, d))
    x = centers[lab] + 1.5 * rng.standard_normal((n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32), lab


def r0_crossfit_map(x, lab):
    """R0's cross-fit mAP by hand: parity halves of the position, plain cosine inside each half, mean of the two."""
    import rt_metrics
    halves = [np.arange(0, len(x), 2), np.arange(1, len(x), 2)]
    return float(np.mean([rt_metrics.retrieval(x[h], lab[h], workers=1)['ap'].mean() for h in halves]))


def test_readout_by_name_round_trips_the_menu():
    for width in (8, 50, 768, 2048):
        for x in rt_readout.menu(width):
            assert ac_select.readout_by_name(rt_readout.name(x), width) == x


def test_readout_from_name_round_trips_the_menu_without_the_width():
    for width in (8, 50, 768, 2048):
        for x in rt_readout.menu(width):
            assert ac_select.readout_from_name(rt_readout.name(x)) == x


def test_arm_table_has_the_menu_and_r0_scores_zero():
    x, lab = synthetic(0)
    table, picked = ac_select.arm_table({('emotion', None): x}, {'emotion': lab}, {'emotion': 0.01}, workers=1)
    assert list(table) == [rt_readout.name(m) for m in rt_readout.menu(8)]     # 15 read-outs, tie order
    assert table['none+k0']['score'] == 0.0 and table['none+k0']['kept']
    assert table['none+k0']['map']['emotion'] == pytest.approx(r0_crossfit_map(x, lab), abs=1e-12)
    assert isinstance(picked, ReadOut) and rt_readout.name(picked) in table


def test_arm_table_pick_is_the_pick_of_its_own_scores():
    x, lab = synthetic(1)
    r = {'emotion': 0.01}
    table, picked = ac_select.arm_table({('emotion', None): x}, {'emotion': lab}, r, workers=1)
    rows = {ac_select.readout_by_name(n, 8): v['map'] for n, v in table.items()}
    assert picked == ac_select.pick_readout(rows, r)
    for n, v in table.items():
        assert v['score'] == pytest.approx((v['map']['emotion'] - table['none+k0']['map']['emotion']) / 0.01)


def test_seeded_arm_is_the_mean_over_the_five_seeds():
    xs, lab = {}, synthetic(0)[1]
    per_seed = []
    for s in range(5):
        x, _ = synthetic(10 + s)
        xs[('emotion', s)] = x
        per_seed.append(r0_crossfit_map(x, lab))
    assert max(per_seed) - min(per_seed) > 1e-3        # the seeds differ, so a wrong average would show
    table, _ = ac_select.arm_table(xs, {'emotion': lab}, {'emotion': 0.01}, workers=1)
    assert table['none+k0']['map']['emotion'] == pytest.approx(np.mean(per_seed), abs=1e-12)


def test_a_fixed_read_out_is_returned_with_r0_beside_it():
    x, lab = synthetic(2)
    fixed = ro('none', k=10)
    table, picked = ac_select.arm_table({('emotion', None): x}, {'emotion': lab}, {'emotion': 0.01},
                                        fixed=fixed, workers=1)
    assert picked == fixed and list(table) == ['none+k0', 'none+k10']


def test_keys_must_be_the_sorted_validation_list_with_one_per_row():
    x, lab = synthetic(2)
    args = ({('emotion', None): x}, {'emotion': lab}, {'emotion': 0.01})
    keys = [f'img{i:03d}' for i in range(N_SYN)]
    ac_select.arm_table(*args, ros=[R0], keys=keys, workers=1)                          # sorted, one per row: fine
    with pytest.raises(AssertionError):
        ac_select.arm_table(*args, ros=[R0], keys=keys[::-1], workers=1)                # not sorted
    with pytest.raises(AssertionError):
        ac_select.arm_table(*args, ros=[R0], keys=keys[:-1], workers=1)                 # one short
    with pytest.raises(AssertionError):
        ac_select.arm_table(*args, ros=[R0], keys=keys + ['img999'], workers=1)         # one too many


# ---- teacher wording ----------------------------------------------------------------------------------------------

def test_photo_wording_needs_more_than_0_001_r():
    r = 0.5
    assert ac_select.pick_wording(0.25, 0.25 + 0.001 * r * 1.01, r) == 'photo'     # just over the margin
    assert ac_select.pick_wording(0.25, 0.25 + 0.001 * r * 0.99, r) == 'artwork'   # just under
    assert ac_select.pick_wording(0.25, 0.25 + 0.001 * r, r) == 'artwork'          # exactly on it: not "more than"


def test_wording_margin_is_in_units_of_r_not_of_the_value():
    # a gap of 0.002 is 4 r when r = 0.0005 (photo); a gap of 0.0004 is 0.0008 r when r = 0.5 (artwork)
    assert ac_select.pick_wording(0.1, 0.102, 0.0005) == 'photo'
    assert ac_select.pick_wording(0.1, 0.1004, 0.5) == 'artwork'
    # the readings disagree: a gap of 0.0008 is under an unscaled 0.001 but 1.6 r when r = 0.0005, so photo
    assert ac_select.pick_wording(0.1, 0.1008, 0.0005) == 'photo'


def test_artwork_wins_when_photo_is_worse_or_equal():
    assert ac_select.pick_wording(0.2, 0.1, 0.01) == 'artwork'
    assert ac_select.pick_wording(0.2, 0.2, 0.01) == 'artwork'


# ---- CRL version --------------------------------------------------------------------------------------------------

def test_version_score_is_the_mean_over_aspects_of_map_over_r():
    s = ac_select.version_score({'emotion': 0.3, 'style': 0.4}, {'emotion': 0.1, 'style': 0.2, 'genre': 9.0})
    assert s == pytest.approx((3.0 + 2.0) / 2)
    assert ac_select.version_score({'emotion': 0.3}, {'emotion': 0.0015}) == pytest.approx(200.0)


def test_crl_version_highest_score_wins():
    scores = {'clip_painting': 100.0, 'clip_photo': 101.0, 'qwen_painting': 103.0, 'qwen_photo': 102.0}
    assert ac_select.pick_crl_version(scores) == 'qwen_painting'


@pytest.mark.parametrize('scores,winner', [
    # the score is already in units of r: 0.001 is the margin as it stands
    ({'clip_painting': 99.999, 'clip_photo': 100.0, 'qwen_painting': 99.5, 'qwen_photo': 99.0}, 'clip_painting'),
    ({'clip_painting': 99.9989, 'clip_photo': 100.0, 'qwen_painting': 99.5, 'qwen_photo': 99.0}, 'clip_photo'),
    ({'clip_painting': 90.0, 'clip_photo': 99.9995, 'qwen_painting': 99.9992, 'qwen_photo': 100.0}, 'clip_photo'),
    ({'clip_painting': 90.0, 'clip_photo': 90.0, 'qwen_painting': 100.0, 'qwen_photo': 100.0}, 'qwen_painting'),
    ({'clip_painting': 100.0, 'clip_photo': 100.0, 'qwen_painting': 100.0, 'qwen_photo': 100.0}, 'clip_painting'),
])
def test_crl_version_ties_go_to_the_first_in_the_fixed_order(scores, winner):
    assert ac_select.pick_crl_version(scores) == winner


def test_crl_version_on_the_two_painting_versions_only_for_artelingo():
    assert ac_select.pick_crl_version({'clip_painting': 5.0, 'qwen_painting': 5.0005}) == 'clip_painting'
    assert ac_select.pick_crl_version({'clip_painting': 5.0, 'qwen_painting': 5.002}) == 'qwen_painting'


def test_crl_version_refuses_an_unknown_name():
    with pytest.raises(AssertionError):
        ac_select.pick_crl_version({'clip_painting': 1.0, 'siglip_photo': 2.0})


# ---- the stage 1 documents: picks.json and scores.json -------------------------------------------------------------

def tbl(maps, picked, ranges, r0=None, extra=None):
    """A table with R0 and the picked read-out, scores as arm_table writes them (mean over aspects of
    (mAP - mAP(R0)) / r_a) and the guard. maps = {aspect: mAP at the pick}, ranges = {aspect: r_a},
    r0 = {aspect: mAP at R0} (default 0.01 below the pick), extra = {name: maps}: further rows."""
    r0 = r0 or {a: v - 0.01 for a, v in maps.items()}
    rows = {'none+k0': dict(r0), **({} if picked == 'none+k0' else {picked: dict(maps)}), **(extra or {})}
    return {n: {'map': dict(m),
                'score': sum((m[a] - r0[a]) / ranges[a] for a in r0) / len(r0),
                'kept': all(m[a] >= r0[a] - ranges[a] - 1e-12 for a in r0)} for n, m in rows.items()}


AFF_ARMS = ('cstar', 'plain', 'teacher_artwork', 'teacher_photo', 'clip', 'percept',
            'crl_clip_painting', 'crl_clip_photo', 'crl_qwen_painting', 'crl_qwen_photo')
R_CLU, R_RET = 0.004, 0.002
R_ART = {'emotion': 0.0014, 'style': 0.0022, 'genre': 0.003}


def stage1_inputs(aff_map=None, art_qwen_maps=None, ami=(0.20, 0.30)):
    """Affection tables where every arm picks center+k3 with mAP 0.30, except those overridden in `aff_map`."""
    aff_map = aff_map or {}
    tables = {a: tbl({'emotion': aff_map.get(a, 0.30)}, 'center+k3', {'emotion': R_RET}) for a in AFF_ARMS}
    picks = {a: ro('center', k=3) for a in AFF_ARMS}
    art_tables = {
        'crl_clip_painting': tbl({'emotion': 0.26, 'style': 0.29, 'genre': 0.51}, 'none+k10', R_ART),
        'crl_qwen_painting': tbl(art_qwen_maps or {'emotion': 0.25, 'style': 0.28, 'genre': 0.50}, 'center+k5', R_ART),
    }
    art_picks = {'crl_clip_painting': ro('none', k=10), 'crl_qwen_painting': ro('center', k=5)}
    return dict(aff_tables=tables, aff_picks=picks, ami=dict(artwork=ami[0], photo=ami[1]), r_clu=R_CLU, r_ret=R_RET,
                art_tables=art_tables, art_picks=art_picks, art_ranges=R_ART)


def test_stage1_documents_have_the_index_format():
    picks, scores = ac_select.build_stage1(**stage1_inputs())
    assert set(picks) == {'affection', 'artelingo'}
    assert set(picks['affection']) == {'readout', 'teacher_wording', 'crl_version'}
    assert picks['affection']['readout'] == {a: 'center+k3' for a in AFF_ARMS}
    assert picks['affection']['teacher_wording'] == {'clustering': 'photo', 'retrieval': 'artwork'}   # AMI gap 25 r; mAP equal
    assert picks['artelingo'] == {'readout': {'cstar': 'center+k10', 'crl_clip_painting': 'none+k10',
                                              'crl_qwen_painting': 'center+k5'},
                                  'crl_version': 'clip_painting'}
    assert set(scores) == {'readout', 'teacher_wording', 'crl_version'}
    assert set(scores['readout']) == {'affection', 'artelingo'}
    assert scores['readout']['affection']['cstar']['center+k3']['map'] == {'emotion': 0.30}
    assert scores['teacher_wording'] == {'clustering': {'artwork': 0.20, 'photo': 0.30},
                                         'retrieval': {'artwork': 0.30, 'photo': 0.30}}
    assert set(scores['crl_version']) == {'affection', 'artelingo'}
    assert set(scores['crl_version']['affection']) == set(ac_paths.CRL_VERSIONS)
    assert set(scores['crl_version']['artelingo']) == {'clip_painting', 'qwen_painting'}


def test_stage1_teacher_retrieval_wording_uses_the_map_at_each_wordings_own_pick():
    # photo's mAP is 0.3012 against 0.30: 0.0012 / r_ret(0.002) = 0.6 r > 0.001 r
    inp = stage1_inputs(aff_map={'teacher_photo': 0.3012})
    # a worse row that is not the pick: its mAP must not be read
    inp['aff_tables']['teacher_photo'] = tbl({'emotion': 0.3012}, 'center+k3', {'emotion': R_RET},
                                             extra={'whiten64+k0': {'emotion': 0.2}})
    picks, scores = ac_select.build_stage1(**inp)
    assert picks['affection']['teacher_wording']['retrieval'] == 'photo'
    assert scores['teacher_wording']['retrieval'] == {'artwork': 0.30, 'photo': 0.3012}


def test_stage1_retrieval_wording_margin_is_in_units_of_r_ret_not_r_clu():
    # R_CLU (0.004) is twice R_RET (0.002): 0.0011 r_ret is only 0.00055 r_clu, so r_clu would give artwork
    for gap, wording in ((0.0011 * R_RET, 'photo'), (0.001 * R_RET, 'artwork')):
        picks, _ = ac_select.build_stage1(**stage1_inputs(aff_map={'teacher_photo': 0.30 + gap}))
        assert picks['affection']['teacher_wording']['retrieval'] == wording, gap


def test_stage1_clustering_wording_uses_the_ami_inputs_and_r_clu():
    picks, _ = ac_select.build_stage1(**stage1_inputs(ami=(0.2000, 0.2000 + 0.001 * R_CLU)))   # exactly 0.001 r
    assert picks['affection']['teacher_wording']['clustering'] == 'artwork'
    picks, _ = ac_select.build_stage1(**stage1_inputs(ami=(0.2000, 0.2000 + 0.0011 * R_CLU)))
    assert picks['affection']['teacher_wording']['clustering'] == 'photo'


def test_stage1_affection_crl_version_follows_the_score_and_the_fixed_order():
    # mAP / r_ret: 0.30 / 0.002 = 150 for all four: tie, so the first
    picks, scores = ac_select.build_stage1(**stage1_inputs())
    assert picks['affection']['crl_version'] == 'clip_painting'
    assert scores['crl_version']['affection'] == {v: pytest.approx(150.0) for v in ac_paths.CRL_VERSIONS}
    picks, scores = ac_select.build_stage1(**stage1_inputs(aff_map={'crl_qwen_photo': 0.3004, 'crl_clip_photo': 0.3002}))
    assert picks['affection']['crl_version'] == 'qwen_photo'     # 150.2 against 150.1 and 150
    assert scores['crl_version']['affection']['qwen_photo'] == pytest.approx(0.3004 / R_RET)


def test_stage1_artelingo_crl_version_is_the_three_aspect_mean_of_map_over_r11s_r():
    # clip: (0.26/0.0014 + 0.29/0.0022 + 0.51/0.003) / 3 ; qwen picked a better emotion
    picks, scores = ac_select.build_stage1(**stage1_inputs(art_qwen_maps={'emotion': 0.30, 'style': 0.28, 'genre': 0.50}))
    clip = (0.26 / 0.0014 + 0.29 / 0.0022 + 0.51 / 0.003) / 3
    qwen = (0.30 / 0.0014 + 0.28 / 0.0022 + 0.50 / 0.003) / 3
    assert scores['crl_version']['artelingo'] == {'clip_painting': pytest.approx(clip), 'qwen_painting': pytest.approx(qwen)}
    assert picks['artelingo']['crl_version'] == 'qwen_painting' and qwen > clip


def test_stage1_refuses_a_fixed_artelingo_pick_that_differs_from_r11s():
    inp = stage1_inputs()
    inp['art_picks']['crl_clip_painting'] = ro('center', k=10)
    with pytest.raises(AssertionError):
        ac_select.build_stage1(**inp)


def test_stage1_refuses_an_affection_score_that_is_not_the_gain_over_r0_in_units_of_r_ret():
    inp = stage1_inputs()
    inp['aff_tables']['clip']['center+k3']['score'] += 1e-9
    with pytest.raises(AssertionError):
        ac_select.build_stage1(**inp)


def test_stage1_refuses_an_affection_pick_that_is_not_the_pick_of_its_table():
    inp = stage1_inputs()
    inp['aff_tables']['plain'] = tbl({'emotion': 0.30}, 'center+k3', {'emotion': R_RET},
                                     extra={'center+k5': {'emotion': 0.31}})     # center+k5 is better
    with pytest.raises(AssertionError):
        ac_select.build_stage1(**inp)                                             # picks['plain'] is center+k3
    inp['aff_picks']['plain'] = ro('center', k=5)
    ac_select.build_stage1(**inp)


def test_stage1_refuses_an_artelingo_qwen_pick_that_is_not_the_pick_of_its_table():
    inp = stage1_inputs()
    inp['art_tables']['crl_qwen_painting'] = tbl({'emotion': 0.25, 'style': 0.28, 'genre': 0.50}, 'center+k5', R_ART,
                                                 extra={'center+k7': {'emotion': 0.26, 'style': 0.29, 'genre': 0.51}})
    with pytest.raises(AssertionError):
        ac_select.build_stage1(**inp)


def test_stage1_refuses_an_artelingo_score_that_is_not_the_three_aspect_gain_in_units_of_r_a():
    for arm in ('crl_clip_painting', 'crl_qwen_painting'):
        inp = stage1_inputs()
        row = next(v for n, v in inp['art_tables'][arm].items() if n != 'none+k0')
        row['score'] += 1e-9
        with pytest.raises(AssertionError):
            ac_select.build_stage1(**inp)


def test_stage1_refuses_an_artelingo_table_that_lacks_an_aspect():
    inp = stage1_inputs()
    inp['art_tables']['crl_qwen_painting'] = tbl({'emotion': 0.25, 'style': 0.28}, 'center+k5', R_ART)
    with pytest.raises(AssertionError):
        ac_select.build_stage1(**inp)
    inp = stage1_inputs()
    inp['art_tables']['crl_clip_painting'] = tbl({'emotion': 0.26, 'style': 0.29, 'genre': 0.51, 'x': 0.1}, 'none+k10',
                                                 {**R_ART, 'x': 0.01})
    with pytest.raises(AssertionError):
        ac_select.build_stage1(**inp)


def test_stage1_refuses_an_affection_table_with_other_aspects_than_emotion():
    inp = stage1_inputs()
    inp['aff_tables']['clip'] = tbl({'emotion': 0.30, 'style': 0.3}, 'center+k3', {'emotion': R_RET, 'style': 0.01})
    with pytest.raises(AssertionError):
        ac_select.build_stage1(**inp)


def test_write_picks_writes_both_files_and_the_committed_copy(tmp_path):
    picks, scores = ac_select.build_stage1(**stage1_inputs())
    copy = tmp_path / 'round' / 'stage1_picks.json'
    copy.parent.mkdir()
    ac_select.write_picks(picks, scores, tmp_path / 'out', copy_to=copy)
    assert json.loads((tmp_path / 'out/stage1/picks.json').read_text()) == picks
    assert json.loads((tmp_path / 'out/stage1/scores.json').read_text()) == scores
    assert copy.read_bytes() == (tmp_path / 'out/stage1/picks.json').read_bytes()
    ac_select.write_picks(picks, scores, tmp_path / 'out', copy_to=copy)       # the same picks again: allowed


def test_write_picks_refuses_to_change_committed_picks(tmp_path):
    picks, scores = ac_select.build_stage1(**stage1_inputs())
    copy = tmp_path / 'stage1_picks.json'
    ac_select.write_picks(picks, scores, tmp_path / 'out', copy_to=copy)
    other = json.loads(json.dumps(picks))
    other['affection']['crl_version'] = 'qwen_photo'
    with pytest.raises(AssertionError):
        ac_select.write_picks(other, scores, tmp_path / 'out2', copy_to=copy)
    assert json.loads(copy.read_text()) == picks


def test_write_picks_refuses_an_invalid_document(tmp_path):
    picks, scores = ac_select.build_stage1(**stage1_inputs())
    picks['artelingo']['crl_version'] = 'qwen_photo'          # ArtELingo has the painting versions only
    with pytest.raises(AssertionError):
        ac_select.write_picks(picks, scores, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


# ---- the replay of r11's stage 4 -----------------------------------------------------------------------------------

def fake_data():
    """A stand-in for rt_data: a seeded arm with one head per aspect, an aspect-free arm, a 'crl' arm (width 5)."""
    from types import SimpleNamespace
    lab = {a: synthetic(0, classes=3 + i)[1] for i, a in enumerate(ABC)}

    def embed(arm, split, aspect, seed=None):
        assert split == 'val'
        k = {'cstar': 100 + 7 * ABC.index(aspect) + (seed or 0), 'plain': 50, 'crl': 60 + ABC.index(aspect)}[arm]
        return synthetic(k, d=5 if arm == 'crl' else 8)[0]

    return SimpleNamespace(labels=lambda split: lab, embed=embed, SEEDED={'cstar'}, ASPECT_FREE={'plain'},
                           WIDTHS={'cstar': 8, 'plain': 8}), lab


def test_replay_of_r11_stage4_gives_each_arms_table_in_r11s_format():
    import rt_readout
    data, lab = fake_data()
    scores, picks = ac_select.replay_r11_stage4(('cstar', 'plain', 'crl'), workers=1, data=data, ranges=RANGES3)
    assert set(scores) == set(picks) == {'cstar', 'plain', 'crl'}
    assert list(scores['cstar']) == [rt_readout.name(m) for m in rt_readout.menu(8)]
    assert list(scores['crl']) == [rt_readout.name(m) for m in rt_readout.menu(5)]
    assert set(scores['plain']['none+k0']) == {'map', 'score', 'kept'} and set(scores['plain']['none+k0']['map']) == set(ABC)
    # the seeded arm averages the five seeds' R0 values per aspect; the aspect-free arm uses one matrix for all
    for i, a in enumerate(ABC):
        by_hand = np.mean([r0_crossfit_map(data.embed('cstar', 'val', a, s), lab[a]) for s in range(5)])
        assert scores['cstar']['none+k0']['map'][a] == pytest.approx(by_hand, abs=1e-12)
        assert scores['plain']['none+k0']['map'][a] == pytest.approx(r0_crossfit_map(data.embed('plain', 'val', a), lab[a]),
                                                                    abs=1e-12)
    assert all(picks[arm] in scores[arm] for arm in picks)


R11_ARMS = ('cstar', 'plain', 'instructed', 'clip', 'siglip2', 'percept', 'crl')


@pytest.mark.slow
@pytest.mark.real_data
@pytest.mark.parametrize('arm', R11_ARMS)
def test_replay_matches_r11s_saved_stage4_scores_and_pick(arm):
    """Stage 0a(iii), r11 stage 4 part (run by the controller): every score within 1e-9 of r11's saved
    stage4/scores.json, and the arm's pick of stage4/picks.json. The seven arms together took about 17 minutes in
    r11 (c* the longest); pass `-k <arm>` to run one."""
    saved = json.loads((ac_paths.R11_OUT / 'stage4/scores.json').read_text())[arm]
    saved_pick = json.loads((ac_paths.R11_OUT / 'stage4/picks.json').read_text())[arm]
    scores, picks = ac_select.replay_r11_stage4((arm,))
    assert list(scores[arm]) == list(saved)
    for name, row in saved.items():
        got = scores[arm][name]
        assert got['kept'] == row['kept'], (arm, name)
        assert abs(got['score'] - row['score']) < 1e-9, (arm, name)
        for a in ABC:
            assert abs(got['map'][a] - row['map'][a]) < 1e-9, (arm, name, a)
    assert picks[arm] == saved_pick


def test_three_aspect_pick_and_scores_equal_r11s_own_functions_on_random_tables():
    rng = np.random.default_rng(3)
    menu = rt_readout.menu(2048)
    for _ in range(20):
        rows = {x: dict(zip(ABC, 0.3 + 0.01 * rng.standard_normal(3))) for x in menu}
        assert ac_select.pick_readout(rows, RANGES3) == rt_readout.select(rows, RANGES3)
        mine, theirs = ac_select.readout_scores(rows, RANGES3), rt_readout.score(rows, RANGES3)
        assert all(abs(mine[x] - theirs[x]) < 1e-12 for x in rows)
        assert ac_select.readout_guard(rows, RANGES3) == rt_readout.guard(rows, RANGES3)
