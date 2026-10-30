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


def test_three_aspect_pick_and_scores_equal_r11s_own_functions_on_random_tables():
    rng = np.random.default_rng(3)
    menu = rt_readout.menu(2048)
    for _ in range(20):
        rows = {x: dict(zip(ABC, 0.3 + 0.01 * rng.standard_normal(3))) for x in menu}
        assert ac_select.pick_readout(rows, RANGES3) == rt_readout.select(rows, RANGES3)
        mine, theirs = ac_select.readout_scores(rows, RANGES3), rt_readout.score(rows, RANGES3)
        assert all(abs(mine[x] - theirs[x]) < 1e-12 for x in rows)
        assert ac_select.readout_guard(rows, RANGES3) == rt_readout.guard(rows, RANGES3)
