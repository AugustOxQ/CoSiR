"""Tests for r28_ablation: roots, the arm-name map onto t5's, the join check on a tiny export, the deltas.
Synthetic data only. Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_r28_ablation.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r28_ablation as A  # noqa: E402


def test_roots_c_star_from_r14_removals_from_r28():
    assert A.export_path('k11=40', 'emotion', 3) == A.REF_ROOT / 'held/k11=40/emotion__s3.npz'
    assert A.REF_ROOT == Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1')
    for r in A.REMOVALS:
        assert A.export_path(r, 'genre', 0) == A.R28_ROOT / f'held/{r}/genre__s0.npz'
    assert A.R28_ROOT == Path('/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1')
    with pytest.raises(AssertionError):
        A.export_path('k11=80', 'genre', 0)          # a knob row is not an ablation arm


def test_arm_names_map_onto_t5_names():
    assert A.T5_NAME['k11=40'] == 'k11=40'
    assert A.T5_NAME['buddy'] == 'k11=40-no_buddy' and set(A.T5_NAME.values()) <= set(A.t5.ARMS)
    assert A.accepted_sidecar_arms('relational') == {'relational', 'k11=40-no_relational'}
    assert A.accepted_sidecar_arms('k11=40') == {'k11=40'}


def _write(tmp, dir_arm, file_aspect, file_seed, n=8, w=6, list_sha='abc', **over):
    arm, aspect, seed = dir_arm, file_aspect, file_seed
    d = tmp / 'held' / arm
    d.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(seed).normal(size=(n, w)).astype(np.float16)
    np.savez(d / f'{aspect}__s{seed}.npz', img=x)
    side = dict(kind='held', arm=arm, aspect=aspect, seed=seed, rows=n, list_sha256=list_sha,
                arrays=dict(img=dict(shape=[n, w], dtype='float16', sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())))
    side.update(over)
    (d / f'{aspect}__s{seed}.json').write_text(json.dumps(side))
    return x


def test_load_rows_joins_by_sidecar(tmp_path):
    x = _write(tmp_path, 'buddy', 'style', 1)
    got = A.load_rows('buddy', 'style', 1, 'abc', {}, n=8, width=6, ref_root=tmp_path, r28_root=tmp_path)
    assert got.dtype == np.float32 and np.array_equal(got, x.astype(np.float32))


@pytest.mark.parametrize('over', [dict(list_sha256='zzz'), dict(rows=7), dict(aspect='genre'), dict(seed=2),
                                  dict(arm='relational'), dict(kind='train')])
def test_load_rows_refuses_a_bad_join(tmp_path, over):
    _write(tmp_path, 'buddy', 'style', 1, **over)
    with pytest.raises(AssertionError):
        A.load_rows('buddy', 'style', 1, 'abc', {}, n=8, width=6, ref_root=tmp_path, r28_root=tmp_path)


def test_load_rows_refuses_a_changed_array(tmp_path):
    _write(tmp_path, 'buddy', 'style', 1)
    p = tmp_path / 'held/buddy/style__s1.npz'
    np.savez(p, img=np.ones((8, 6), dtype=np.float16))
    with pytest.raises(AssertionError):
        A.load_rows('buddy', 'style', 1, 'abc', {}, n=8, width=6, ref_root=tmp_path, r28_root=tmp_path)


def test_t5_legacy_arm_name_is_accepted(tmp_path):
    _write(tmp_path, 'buddy', 'style', 1, arm='k11=40-no_buddy')
    A.load_rows('buddy', 'style', 1, 'abc', {}, n=8, width=6, ref_root=tmp_path, r28_root=tmp_path)


def _cells(ami, mp):
    return {a: dict(ami_per_seed=list(ami[i]), map_per_seed=list(mp[i])) for i, a in enumerate(A.ASPECTS)}


def test_deltas_are_removal_minus_cstar_and_seed_paired():
    ref = _cells([[0.1, 0.2], [0.4, 0.4], [0.5, 0.6]], [[0.3, 0.3], [0.3, 0.5], [0.6, 0.6]])
    arm = _cells([[0.0, 0.1], [0.3, 0.4], [0.6, 0.6]], [[0.2, 0.2], [0.3, 0.3], [0.6, 0.7]])
    d = A.deltas(arm, ref)
    assert d['emotion']['ami']['delta'] == pytest.approx(-0.1) and d['emotion']['ami']['per_seed'] == pytest.approx([-0.1, -0.1])
    assert d['style']['map']['delta'] == pytest.approx(-0.1) and d['style']['map']['per_seed'] == pytest.approx([0.0, -0.2])
    assert d['genre']['ami']['delta'] == pytest.approx(0.05)
    z = A.deltas(ref, ref)
    assert all(z[a][m]['delta'] == 0.0 for a in A.ASPECTS for m in ('ami', 'map'))


def test_plain_replay_target_is_the_papers_qwen_row():
    assert A.TABLE5_QWEN_PLAIN['ami'] == dict(emotion=0.070, style=0.235, genre=0.425)
    assert A.TABLE5_QWEN_PLAIN['map'] == dict(emotion=0.242, style=0.180, genre=0.476)


def test_r17_plain_rows_exist_for_both_encoders():
    for enc in ('qwen', 'siglip1'):
        rows = A.r17_rows(enc, 'plain')
        assert set(rows) == set(A.ASPECTS) and all(len(r['per_seed']) == 1 for r in rows.values())
    assert [round(A.r17_rows('siglip1', 'plain')[a]['per_seed'][0], 3) for a in A.ASPECTS] == [0.061, 0.382, 0.402]
