"""Tests of r29_image_cells (ticket 06): the outcome rule at its edges, the wiring (names, row order, C6), the order rule."""
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_image_cells as ic  # noqa: E402

import rt_stats  # noqa: E402  (r11: the rule the older cells used)
import ns_data  # noqa: E402
import bs_artret  # noqa: E402
import rs_vh  # noqa: E402

W = 2048


# ------------------------------------------------------------------------------------------------ outcome rule
@pytest.mark.parametrize('lo, r, want', [
    (1e-12, 0.01, 'WIN'),            # just above 0
    (0.0, 0.01, 'LEVEL'),            # exactly 0 is not above 0
    (-0.0099999, 0.01, 'LEVEL'),     # just above -r
    (-0.01, 0.01, 'BELOW'),          # exactly -r is not above -r
    (-0.5, 0.01, 'BELOW'),
    (0.0, 0.0, 'BELOW'),             # r = 0: LEVEL needs lower > 0, which is WIN
    (1e-9, 0.0, 'WIN'),
])
def test_outcome_edges(lo, r, want):
    assert ic.outcome(lo, r) == want


def test_outcome_agrees_with_r11_rule_on_a_grid():
    for r in (0.0, 0.002, 0.0150):
        for lo in np.linspace(-0.03, 0.03, 61):
            assert ic.outcome(float(lo), r) == rt_stats.outcome(float(lo), r)


@pytest.mark.parametrize('lo, r', [(0.1, None), (0.1, -1.0), (float('nan'), 0.01), (0.1, float('nan'))])
def test_outcome_refuses_bad_input(lo, r):
    with pytest.raises(AssertionError):
        ic.outcome(lo, r)


# ------------------------------------------------------------------------------------------------ wiring
NAMES = {'held': [f'h{i:02d}' for i in range(7)], 'val': [f'v{i:02d}' for i in range(5)]}


def _unit16(n, seed):
    x = np.random.default_rng(seed).normal(size=(n, W)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def _write(root, arm, split, aspect, seed, names, x16, over=None):
    if split == 'held':
        p = Path(root) / f'held/{arm}/{aspect}__s{seed}.npz'
        side = dict(kind='held', arm=arm, aspect=aspect, seed=seed, rows=len(names), width=W, dtype='float16',
                    list_sha256=rs_vh.names_sha(names))
        keys = dict(img=x16)
    else:
        p = Path(root) / f'val/{arm}/{aspect}__s{seed}.npz'
        side = dict(kind='val', arm=arm, aspect=aspect, seed=seed, rows=len(names), width=W, dtype='float16',
                    names_sha256=rs_vh.names_sha(names))
        keys = dict(img=x16, cap=x16)
    side.update(over or {})
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **keys)
    p.with_suffix('.json').write_text(json.dumps(side))
    return p


@pytest.fixture
def root(tmp_path):
    for split in ('held', 'val'):
        for a in ('emotion', 'style', 'genre'):
            for s in range(5):
                _write(tmp_path, 'siglip1', split, a, s, NAMES[split], _unit16(len(NAMES[split]), hash((split, a, s)) % 1000))
    return tmp_path


def _wiring(root, **kw):
    return ic.Wiring(root, 'siglip1', 'siglip1', names=lambda split: NAMES[split], **kw)


def test_head_rows_follow_the_name_order_and_values_are_stored_ones(root):
    w = _wiring(root)
    got = w.head('held', 'style', 3)
    with np.load(root / 'held/siglip1/style__s3.npz') as z:
        assert got.dtype == np.float16 and np.array_equal(got, z['img'])


def test_head_refuses_rows_exported_for_another_name_order(root):
    perm = list(reversed(NAMES['held']))                      # same names, other order: a position-based join would pass
    _write(root, 'siglip1', 'held', 'emotion', 0, perm, _unit16(len(perm), 1))
    with pytest.raises(AssertionError, match='names'):
        _wiring(root).head('held', 'emotion', 0)


def test_head_refuses_other_names_other_row_count_other_arm_aspect_seed(root):
    w = _wiring(root)
    _write(root, 'siglip1', 'val', 'emotion', 0, ['x'] * 5, _unit16(5, 1))
    with pytest.raises(AssertionError):
        w.head('val', 'emotion', 0)
    _write(root, 'siglip1', 'held', 'emotion', 1, NAMES['held'], _unit16(6, 1))            # 6 rows, 7 names
    with pytest.raises(AssertionError):
        w.head('held', 'emotion', 1)
    for over in (dict(arm='k2'), dict(aspect='genre'), dict(seed=4)):
        _write(root, 'siglip1', 'held', 'emotion', 2, NAMES['held'], _unit16(7, 1), over=over)
        with pytest.raises(AssertionError):
            w.head('held', 'emotion', 2)


def test_head_refuses_not_float16_and_not_unit_rows(root):
    w = _wiring(root)
    _write(root, 'siglip1', 'held', 'emotion', 3, NAMES['held'], _unit16(7, 1).astype(np.float32))
    with pytest.raises(AssertionError):
        w.head('held', 'emotion', 3)
    _write(root, 'siglip1', 'held', 'emotion', 4, NAMES['held'], (2 * _unit16(7, 1)).astype(np.float16))
    with pytest.raises(AssertionError):
        w.head('held', 'emotion', 4)


def test_validation_only_wiring_refuses_held_rows(root):
    w = _wiring(root, splits=('val',))
    w.head('val', 'emotion', 0)
    with pytest.raises(AssertionError, match='validation'):
        w.head('held', 'emotion', 0)


def test_val_falls_back_to_r14_training_file_layout(tmp_path):
    names = NAMES['val']
    x = _unit16(len(names), 5)
    p = tmp_path / f'train/k11=40/R6-5/emotion__{ic.GRAPHS["emotion"]}__s0.npz'
    p.parent.mkdir(parents=True)
    np.savez(p, img=x, cap=x)
    p.with_suffix('.json').write_text(json.dumps(dict(config=dict(aspect='emotion', seed=0, names_sha256=dict(dev=rs_vh.names_sha(names)),
                                                                   rg=dict(arm='k11=40')))))
    w = ic.Wiring(tmp_path, 'k11=40', 'siglip1', names=lambda split: NAMES[split])
    assert np.array_equal(w.head('val', 'emotion', 0), x)


def test_wired_scorers_read_the_new_root_and_are_restored(root):
    w = _wiring(root)
    orig_c, orig_e = ns_data.cstar, bs_artret.cstar_embed
    with w:
        x = ns_data.cstar('artelingo', 'held', 'siglip1', 'emotion', 2, unit=False)
        assert x.dtype == np.float32 and x.shape == (7, W)
        assert np.array_equal(x, w.head('held', 'emotion', 2).astype(np.float32))        # clustering form: not renormalised
        u = bs_artret.cstar_embed('ignored-root', 'val', 'style', 1)
        assert u.dtype == np.float32 and u.shape == (5, W)                                 # retrieval form: unit float32
        assert np.allclose(np.linalg.norm(u, axis=1), 1, atol=1e-6)
        with pytest.raises(AssertionError):
            ns_data.cstar('artelingo', 'held', 'siglip2', 'emotion', 2)                    # another encoder than wired
    assert ns_data.cstar is orig_c and bs_artret.cstar_embed is orig_e


def test_wired_untold_joins_three_heads_of_one_seed(root):
    w = _wiring(root)
    with w, mock.patch.dict(ns_data.ns_paths.N_ROWS, {('artelingo', 'val'): 5}):
        u = ns_data.untold('artelingo', 'val', 'siglip1', 0)
    assert u.shape == (5, 3 * W) and u.dtype == np.float32
    emo = ns_data._unit32(w.head('val', 'emotion', 0)) / np.float32(np.sqrt(3))
    assert np.allclose(u[:, :W], emo, atol=1e-6)


# ------------------------------------------------------------------------------------------------ order rule
def test_cells_refuse_without_the_picks_record(tmp_path):
    with pytest.raises(FileNotFoundError, match='picks'):
        ic.read_picks('siglip1', records=tmp_path, require_committed=False)


def test_picks_record_must_name_its_arm_and_encoder(tmp_path):
    (tmp_path / 'image_picks_siglip1.json').write_text(json.dumps(dict(arm='k2', encoder='siglip1')))
    with pytest.raises(AssertionError):
        ic.read_picks('siglip1', records=tmp_path, require_committed=False)


def test_picks_record_must_be_committed_unchanged(tmp_path):
    def git(*a):
        subprocess.run(['git', '-C', str(tmp_path), *a], check=True, capture_output=True)
    git('init')
    git('config', 'user.email', 'a@b')
    git('config', 'user.name', 'n')
    path = tmp_path / 'records' / 'image_picks_siglip1.json'
    path.parent.mkdir()
    path.write_text(json.dumps(dict(arm='siglip1', encoder='siglip1')))
    with pytest.raises(RuntimeError, match='commit'):
        ic.read_picks('siglip1', records=path.parent, require_committed=True, repo=tmp_path)
    git('add', '.')
    git('commit', '-m', 'x')
    assert ic.read_picks('siglip1', records=path.parent, require_committed=True, repo=tmp_path)['arm'] == 'siglip1'
    path.write_text(json.dumps(dict(arm='siglip1', encoder='siglip1', changed=1)))
    with pytest.raises(RuntimeError, match='commit'):
        ic.read_picks('siglip1', records=path.parent, require_committed=True, repo=tmp_path)


def test_encoder_of_arm():
    assert ic.encoder_of_arm('siglip1') == 'siglip1' and ic.encoder_of_arm('siglip1_prefix') == 'siglip1'
    assert ic.encoder_of_arm('siglip2') == 'siglip2' and ic.encoder_of_arm('clipl14_prefix') == 'clip_l14'
    with pytest.raises(AssertionError):
        ic.encoder_of_arm('qwen')
