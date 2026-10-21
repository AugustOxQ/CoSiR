"""Fast tests of ab_analysis on synthetic arrays (no stores, no sample)."""
import json
from pathlib import Path

import numpy as np
import pytest

import ab_analysis as A
import ab_metrics as M


def fake_sample(n=60, m=20, seed=0):
    rng = np.random.default_rng(seed)
    return dict(first=np.arange(n), subset_pid=rng.permutation(n)[:m])  # not sorted, as the real one


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_offset_is_instructed_minus_default_per_row_in_one_order():
    rng = np.random.default_rng(1)
    inst, default = unit(rng.normal(size=(60, 8))), unit(rng.normal(size=(60, 8)))
    s = fake_sample()
    off = A.instruction_offset(A.eval_rows(inst, s), A.eval_rows(default, s))
    for i, p in enumerate(s['subset_pid']):
        assert np.allclose(off[i], inst[p] - default[p])
    # a default in another painting order gives a different offset, and shape mismatches are refused
    shuffled = A.eval_rows(default, s)[::-1]
    assert not np.allclose(off, A.instruction_offset(A.eval_rows(inst, s), shuffled))
    with pytest.raises(AssertionError):
        A.instruction_offset(inst, default[:50])


def test_evaluation_join_asserts_fire_on_shuffled_order():
    rng = np.random.default_rng(2)
    full = unit(rng.normal(size=(60, 8)))
    s = fake_sample()
    got = A.eval_rows(full, s)
    assert np.array_equal(got, full[s['subset_pid']])
    A.check_order(s, s['subset_pid'])
    bad = s['subset_pid'].copy()
    bad[[0, 1]] = bad[[1, 0]]
    with pytest.raises(AssertionError):
        A.check_order(s, bad)
    with pytest.raises(AssertionError):
        A.eval_rows(full, s, pid=bad)
    with pytest.raises(AssertionError):
        A.eval_rows(full[:50], s)  # row count of the per-painting array
    # the positional check against an independently stored evaluation file
    A.positional_check(got, got.astype(np.float16))
    with pytest.raises(AssertionError):
        A.positional_check(got, got[::-1])


def test_check_array_guards():
    x = unit(np.random.default_rng(3).normal(size=(10, 4))).astype(np.float32)
    A.check_array(x, 10, d=4)
    with pytest.raises(AssertionError):
        A.check_array(x * 1.5, 10)
    with pytest.raises(AssertionError):
        A.check_array(x, 11)
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(AssertionError):
        A.check_array(bad, 10)


def test_angles_from_bases_match_ab_metrics():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(200, 12)) @ rng.normal(size=(12, 12))
    b = rng.normal(size=(200, 12)) + a * .5
    (va, _), (vb, _) = A.subspace_basis(a), A.subspace_basis(b)
    for r in (2, 4):
        cos, msq = A.angles_from_bases(va, vb, r)
        cos0, msq0 = M.principal_angles(a, b, r)
        assert np.allclose(cos, cos0) and np.isclose(msq, msq0)


def test_offset_geometry_shapes_and_self_angles():
    rng = np.random.default_rng(5)
    default = rng.normal(size=(300, 16))
    base = rng.normal(size=(300, 3)) @ rng.normal(size=(3, 16))
    offs = {'emotion': base, 'style': base.copy(), 'genre': rng.normal(size=(300, 16))}
    g = A.offset_geometry(offs, default, ranks=(2, 3))
    assert set(g['effective_rank']) == {'offset_emotion', 'offset_style', 'offset_genre', 'default'}
    assert np.allclose(g['angles']['emotion|style']['3']['cosines'], 1.0)  # same subspace
    assert g['angles']['emotion|genre']['3']['chance'] == 3 / 16
    assert g['angles']['emotion|genre']['3']['mean_sq_cosine'] < 1.0
    assert 0 < g['top_energy']['emotion']['3'] <= 1.0 + 1e-12


def test_pairwise_identical_views_and_floor():
    rng = np.random.default_rng(6)
    x = unit(rng.normal(size=(120, 10)))
    views = {v: x if v != 'genre_word' else unit(rng.normal(size=(120, 10))) for v in A.VIEWS}
    out = A.pairwise(views, k=5)
    assert len(out) == 6
    same = out['default|emotion_focus']
    assert np.isclose(same['cka'], 1.0) and np.isclose(same['knn_overlap'], 1.0)
    diff = out['default|genre_word']
    assert diff['cka'] < .5 and diff['knn_overlap'] < .3 and diff['cka_shuffled'] < .5
    assert np.isclose(diff['knn_chance'], 5 / 119)


def test_class_to_cluster_counts():
    y = np.array([0, 0, 1, 1, 1, 2, -1, -1])
    part = np.array([3, 3, 0, 0, 1, 2, 5, 5])
    t = A.class_to_cluster(y, part, k=6)
    assert list(t['classes']) == [0, 1, 2] and t['counts'].sum() == 6
    assert list(t['top_cluster']) == [3, 0, 2] and np.allclose(t['top_share'], [1, 2 / 3, 1])


def test_scores_reader(tmp_path):
    assert A.read_scores(tmp_path / 'scores.json') is None
    cell = dict(ami=dict(mean=.1, sd=.0), probe=dict(balanced_accuracy=.3, per_class_recall=[.3], classes=[0]))
    path = tmp_path / 'scores.json'
    path.write_text(json.dumps(dict(rows={'R0': {'emotion': {'img': {'0': cell}}}})))
    d = A.read_scores(path)
    ami, probe = A.score_entry(d, 'R0', 'emotion', 0)
    assert ami == .1 and probe['balanced_accuracy'] == .3
    assert A.score_entry(d, 'R0', 'style', 0) is None
    path.write_text(json.dumps(dict(nothing=1)))
    with pytest.raises(AssertionError):
        A.read_scores(path)


def test_row_meta_reader(tmp_path):
    folder = tmp_path / 'R3'
    folder.mkdir()
    (folder / 'ab_row.json').write_text(json.dumps(dict(name='R3', switches=dict(
        teacher='default', buddy=1.0, own=1.0, endpoints='both'), store='aspect', caption_side='scored')))
    m = A.read_row_meta(folder)
    assert m['switches']['teacher'] == 'default'
    (folder / 'ab_row.json').write_text(json.dumps(dict(name='R4', switches={})))
    with pytest.raises((AssertionError, KeyError)):
        A.read_row_meta(folder)


def test_atomic_json_refuses_nan(tmp_path):
    A.atomic_json(tmp_path / 'a.json', dict(x=np.float32(1.5), y=np.arange(3)))
    assert json.loads((tmp_path / 'a.json').read_text()) == dict(x=1.5, y=[0, 1, 2])
    with pytest.raises(ValueError):
        A.atomic_json(tmp_path / 'b.json', dict(x=float('nan')))
    assert not (tmp_path / 'b.json').exists()
