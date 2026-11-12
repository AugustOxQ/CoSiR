"""Tests for r28_photos: the key join of exports, the re-pick under another r, the Affection seed range, the gate rule.
No real data: tiny synthetic exports in tmp_path. Run: pytest -q src/test/20261112_siglip_main/test_r28_photos.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r28_photos as P  # noqa: E402
import ns_photo  # noqa: E402
import rt_readout  # noqa: E402
import rs_vh  # noqa: E402
import ac_stats  # noqa: E402


def _write_export(root, set_name, arm, split, seed, keys, emb, side_over=None):
    d = Path(root) / set_name / arm
    d.mkdir(parents=True, exist_ok=True)
    f = d / f'{split}__s{seed}.npz'
    np.savez(f, keys=np.asarray(keys), emb=emb)
    side = dict(dataset=set_name, arm=arm, split=split, seed=seed, sha256=rs_vh.file_sha(f))
    side.update(side_over or {})
    f.with_suffix('.json').write_text(json.dumps(side))
    return f


def _unit16(n, d=2048, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


@pytest.fixture
def world(tmp_path, monkeypatch):
    keys = [f'img/{i:03d}' for i in range(12)]
    monkeypatch.setattr(ns_photo, 'keys', lambda s, sp: list(keys))
    monkeypatch.setattr(ns_photo, '_uncut_keys', lambda s, sp: set(keys))
    monkeypatch.setitem(P.ARMS, 'cstar_siglip1', dict(seeded=True, root=tmp_path))
    return keys, tmp_path


def test_stored_joins_by_key_not_by_position(world):
    keys, root = world
    emb = _unit16(12)
    perm = np.random.default_rng(1).permutation(12)          # the file holds the rows in another order
    _write_export(root, 'emoset', 'cstar_siglip1', 'val', 0, [keys[i] for i in perm], emb[perm])
    out = P.stored('emoset', 'val', 'cstar_siglip1', 0)
    assert out.shape == emb.shape and not np.array_equal(out, emb[perm])
    # the output row of key i is the row the file stored for key i, which is emb[i]
    assert np.array_equal(out, emb)


def test_stored_refuses_a_missing_key_a_changed_file_and_a_wrong_sidecar(world):
    keys, root = world
    emb = _unit16(12)
    _write_export(root, 'emoset', 'cstar_siglip1', 'val', 0, keys[:-1], emb[:-1])
    with pytest.raises(AssertionError):                         # key set differs from the split's
        P.stored('emoset', 'val', 'cstar_siglip1', 0)
    f = _write_export(root, 'emoset', 'cstar_siglip1', 'val', 1, keys, emb)
    side = json.loads(f.with_suffix('.json').read_text())
    side['sha256'] = '0' * 64
    f.with_suffix('.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError):                         # file differs from its sidecar
        P.stored('emoset', 'val', 'cstar_siglip1', 1)
    _write_export(root, 'emoset', 'cstar_siglip1', 'val', 2, keys, emb, side_over=dict(seed=3))
    with pytest.raises(AssertionError):                         # sidecar names another seed
        P.stored('emoset', 'val', 'cstar_siglip1', 2)


def test_stored_refuses_rows_that_are_not_unit(world):
    keys, root = world
    _write_export(root, 'emoset', 'cstar_siglip1', 'val', 0, keys, (_unit16(12).astype(np.float32) * 1.5).astype(np.float16))
    with pytest.raises(AssertionError):
        P.stored('emoset', 'val', 'cstar_siglip1', 0)


def _table(maps):
    """A recorded table in arm_table's form from {readout name: mAP}; 'none+k0' is R0."""
    return {n: {'map': {'emotion': m}, 'score': 0.0, 'kept': True} for n, m in maps.items()}


def test_pick_under_follows_the_tie_rule_and_the_guard_for_the_given_r():
    t = _table({'none+k0': 0.30, 'center+k0': 0.302, 'center+k10': 0.31, 'none+k3': 0.3099, 'whiten64+k0': 0.10})
    # r = 0.01: center+k10 gains 1.0 r, none+k3 0.99 r (not within 0.001 r of the best), whiten64 is dropped by the guard
    assert P.pick_under(t, 0.01) == 'center+k10'
    # r = 10: every gain is below 0.001 r of the best, all tie, the first in the tie order (R0) wins
    assert P.pick_under(t, 10.0) == 'none+k0'
    # r = 0.5: the gains are 0.0200 and 0.0198, within 0.001 of each other, so they tie and the smaller k (none+k3) wins
    assert P.pick_under(t, 0.5) == 'none+k3'


def test_pick_under_equals_the_menu_pick_name_format():
    for n in ('none+k0', 'center+k10', 'whiten256+k3'):
        assert rt_readout.name(P.ac_select.readout_from_name(n)) == n


def test_affection_range_is_max_minus_min_of_the_five_validation_maps(monkeypatch):
    seen = iter([0.30, 0.32, 0.29, 0.31, 0.305])

    def fake_retrieval(x, labels, qe_k, workers):
        assert qe_k == 0                                        # plain cosine, R0
        return dict(dropped=0, rows=np.arange(len(labels)), ap=np.full(len(labels), next(seen)))
    monkeypatch.setattr(P.rt_metrics, 'retrieval', fake_retrieval)
    xs = {s: np.zeros((6, 4), dtype=np.float32) for s in P.SEEDS}
    r, src = P.r_map_for('affection', xs, np.zeros(6, dtype=np.int64), 1)
    assert r == pytest.approx(0.32 - 0.29)
    with pytest.raises(AssertionError):                         # an unseeded arm has no own range on Affection
        P.r_map_for('affection', {None: xs[0]}, np.zeros(6, dtype=np.int64), 1)


def test_emoset_r_is_the_committed_stage_2_value():
    r, src = P.r_map_for('emoset', {}, None, 1)
    rec = json.loads(P.R14_STAGE2.read_text())
    assert r == rec['ranges']['map']['r'] and 0 < r < 0.05 and 'stage2_emoset.json' in src


def test_gate_passes_only_with_equal_numbers_and_equal_pick(monkeypatch):
    monkeypatch.setattr(P, 'recorded', lambda s, a: (0.4, 0.5, 'center+k10'))
    cell = dict(set='emoset', arm='photo_siglip1', readout='center+k10', ami=dict(mean=0.4), map=dict(mean=0.5))
    assert P.gate(cell)['pass']
    assert not P.gate(dict(cell, ami=dict(mean=0.4 + 1e-6)))['pass']
    assert not P.gate(dict(cell, readout='center+k3'))['pass']
