"""Tests for r28_valpick: the pick, the cell statistics, the join check, the gate difference. Synthetic data only.
Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_r28_valpick.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r28_valpick as V  # noqa: E402


def cells_with(values):
    """{cell: {'mean': v}} from a list of six cell means in V.CELLS order."""
    assert len(values) == len(V.CELLS) == 6
    return {c: dict(mean=v) for c, v in zip(V.CELLS, values)}


def test_pick_takes_highest_mean_of_six():
    table = dict(siglip1=cells_with([0.1, 0.4, 0.5, 0.3, 0.3, 0.6]),    # mean 0.3667
                 siglip2=cells_with([0.1, 0.4, 0.5, 0.3, 0.3, 0.5]),    # lower
                 clip_l14=cells_with([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
    r = V.pick(table)
    assert r['pick'] == 'siglip1' and r['ranking'] == ['siglip1', 'siglip2', 'clip_l14']
    assert r['scores']['siglip1'] == pytest.approx(np.mean([0.1, 0.4, 0.5, 0.3, 0.3, 0.6]))
    assert r['margin'] == pytest.approx(0.1 / 6)


def test_pick_is_not_a_per_cell_vote():
    # clip wins 4 of 6 cells by a little; siglip1 wins 2 cells by a lot: the mean decides
    table = dict(siglip1=cells_with([0.0, 0.0, 0.0, 0.0, 0.9, 0.9]),
                 siglip2=cells_with([0.0] * 6),
                 clip_l14=cells_with([0.1, 0.1, 0.1, 0.1, 0.0, 0.0]))
    assert V.pick(table)['pick'] == 'siglip1'


def test_pick_ignores_information_row_and_flags_ties():
    table = dict(siglip1=cells_with([0.5] * 6), siglip2=cells_with([0.4] * 6), clip_l14=cells_with([0.3] * 6),
                 qwen=cells_with([0.9] * 6))
    assert V.pick(table)['pick'] == 'siglip1'                    # qwen's higher score is not a candidate
    table['siglip2'] = cells_with([0.5] * 6)
    r = V.pick(table)
    assert r['pick'] is None and r['tie'] == ['siglip1', 'siglip2']


def test_pick_refuses_a_missing_candidate():
    with pytest.raises(AssertionError):
        V.pick(dict(siglip1=cells_with([0.5] * 6), siglip2=cells_with([0.4] * 6)))


def test_encoder_score_needs_exactly_the_six_cells():
    c = cells_with([0.5] * 6)
    del c['map/genre']
    with pytest.raises(AssertionError):
        V.encoder_score(c)


def test_cell_stats_mean_and_range():
    s = V.cell_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    assert s['mean'] == pytest.approx(0.3) and s['min'] == 0.1 and s['max'] == 0.5
    with pytest.raises(AssertionError):
        V.cell_stats([0.1, 0.2])                                 # not five seeds


def test_join_check_catches_id_and_count_mismatch():
    V.join_check('abc', 'abc', 6152, 6152, 6152, 'f')
    with pytest.raises(AssertionError):
        V.join_check('abc', 'abd', 6152, 6152, 6152, 'f')        # export written for another name order
    with pytest.raises(AssertionError):
        V.join_check('abc', 'abc', 6151, 6152, 6152, 'f')        # row count differs


def test_gate_diff_is_max_abs():
    assert V.gate_diff([0.1, 0.2], [0.1, 0.25]) == pytest.approx(0.05)


def test_merge_joins_encoder_files_and_picks(tmp_path):
    import json
    for e, v in dict(siglip1=0.5, siglip2=0.4, clip_l14=0.3, qwen=0.9).items():
        V.write_json(tmp_path / f'valpick_{e}.json', dict(encoder=e, cells=cells_with([v] * 6), score=v))
    doc = V.merge(tmp_path, tmp_path / "rec")
    assert doc['complete'] and doc['pick']['pick'] == 'siglip1'
    assert json.loads((tmp_path / 'rec' / 'valpick.json').read_text())['pick']['pick'] == 'siglip1'
    (tmp_path / 'valpick_clip_l14.json').unlink()
    assert not V.merge(tmp_path, tmp_path / "rec2")['complete']
