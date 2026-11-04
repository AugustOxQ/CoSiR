"""Tests of om_gate_iii (ticket 07): the comparison and pass logic, and the (b) partition-file path, on synthetic data.

The real replays read the older rounds' stores and take minutes; they are run by the CLI, not here.

Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q
     src/test/20261103_outside_methods/test_om_gate_iii.py
"""
import json

import numpy as np
import pytest

import om_data as od
import om_gate_iii as g
import om_paths as op

N = 60
K = op.K20


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Fake keys (60 rows per split) so the partition files can be written without the real stores; one bootstrap process."""
    ks = [f'p{i:03d}' for i in range(N)]
    monkeypatch.setattr(od, 'keys', lambda s, sp: list(ks))
    g.ns_clu.WORKERS = 1
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, N).astype(np.int64)
    y[:4] = -1
    cand = rng.integers(0, K, (5, 3, N)).astype(np.int32)
    cand[:, :, :] = np.where(rng.random((5, 3, N)) < 0.5, y.clip(0) * 3, cand)       # some signal
    comp = rng.integers(0, K, (3, N)).astype(np.int32)
    return dict(keys=ks, y=y, cand=cand, comp=comp, out=tmp_path / 'out')


def stored_from(got):
    return dict(candidate_value=got['cand'], comparator_value=got['comp'], d=got['d'], lo=got['lo'], hi=got['hi'])


# ------------------------------------------------------------------------------------------------ comparison and pass
def test_a_perturbed_expected_value_of_1e8_fails_and_1e10_passes():
    got = dict(cand=0.3, comp=0.2, d=0.1, lo=0.05, hi=0.15)
    stored = stored_from(got)
    ok = g.finish(g.cell_items('t', 'src', got, stored, True), 2000, 2000)
    assert ok['pass'] and ok['max_abs_diff'] == 0.0
    for key in ('candidate_value', 'comparator_value', 'd', 'lo', 'hi'):
        bad = dict(stored, **{key: stored[key] + 1e-8})
        res = g.finish(g.cell_items('t', 'src', got, bad, True), 2000, 2000)
        assert not res['pass'] and not res['wiring_ok'], key
        assert res['max_abs_diff'] == pytest.approx(1e-8, rel=1e-3)
        near = dict(stored, **{key: stored[key] + 1e-10})
        assert g.finish(g.cell_items('t', 'src', got, near, True), 2000, 2000)['pass'], key


def test_pass_needs_full_draws_and_every_cell():
    got = dict(cand=0.3, comp=0.2, d=0.1, lo=0.05, hi=0.15)
    items = g.cell_items('t', 'src', got, stored_from(got), True)
    assert not g.finish(items, 100, 2000)['pass']                       # fewer draws: values only
    assert g.finish(items, 100, 2000)['wiring_ok']
    assert not g.finish(items, 2000, 2000, complete=False)['pass']      # a cell or aspect left out
    assert len(g.cell_items('t', 'src', got, stored_from(got), False)) == 3   # no interval below the full draws


def test_a_partition_difference_counts_entries_and_fails():
    a = np.arange(10, dtype=np.int32)
    same = g.partition_item('p', 'src', a, a.copy())
    assert same['ok'] and same['abs_diff'] == 0
    b = a.copy()
    b[3] = 99
    diff = g.partition_item('p', 'src', a, b)
    assert not diff['ok'] and diff['abs_diff'] == 1
    with pytest.raises(AssertionError):
        g.partition_item('p', 'src', a, a[:5])


def test_gate_file_merges_parts_and_passes_only_with_all_three(tmp_path):
    path = tmp_path / 'stage0' / 'gate_iii.json'
    part = lambda ok: {'pass': ok, 'code_sha256': g.code_sha256()}          # noqa: E731
    assert g.write_gate(path, 'a', part(True))['pass'] is False
    assert g.write_gate(path, 'b', part(True))['pass'] is False
    res = g.write_gate(path, 'c', part(True))
    assert res['pass'] is True and set(res) >= {'a', 'b', 'c', 'pass'}
    assert json.loads(path.read_text())['pass'] is True
    assert g.write_gate(path, 'b', part(False))['pass'] is False             # a rerun replaces its part
    res = g.write_gate(path, 'b', {'pass': True, 'code_sha256': 'older code'})
    assert res['pass'] is False                                              # a part from other code does not count


# ------------------------------------------------------------------------------------------------ (b) files
def test_b_files_give_the_same_cell_as_memory(fake):
    mem = g.clu_cell(fake['cand'], fake['comp'], fake['y'], 40)
    got, back, _ = g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'], fake['y'], 40, fake['out'])
    assert np.array_equal(back, fake['comp'])
    for key in ('cand', 'comp', 'd', 'lo', 'hi'):
        assert abs(got[key] - mem[key]) <= op.TOL_REPLAY
    side = json.loads((fake['out'] / 'parts/replay/artelingo/qwen/plain/k20__start1.json').read_text())
    assert side['method'] == 'replay' and side['K'] == K and side['start'] == 1
    assert (fake['out'] / 'parts/replay/artelingo/qwen/cstar/emotion__s4__k20.npz').exists()


def test_b_c_star_round_trip_is_state_zero_of_each_seed(fake):
    g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'], fake['y'], 20, fake['out'], aspect='style')
    back = od.turtle_parts('artelingo', 'qwen', 'cstar', K, aspect='style', method='replay', out=fake['out'])
    assert back.shape == (5, 1, N) and np.array_equal(back[:, 0], fake['cand'][:, 0])


def test_b_refuses_partitions_in_a_wrong_order(fake):
    g.write_replay_parts('artelingo', 'qwen', fake['cand'], fake['comp'], 'emotion', fake['out'])
    path = fake['out'] / 'parts/replay/artelingo/qwen/plain/k20__start2.npz'
    with np.load(path) as z:
        ks, y = z['keys'], z['labels']
    perm = np.random.default_rng(1).permutation(N)
    np.savez(path, keys=ks[perm], labels=y[perm])                             # keys and labels permuted together
    side = path.with_suffix('.json')
    meta = json.loads(side.read_text())
    meta['sha256'] = op.sha256_file(path)
    side.write_text(json.dumps(meta))                                          # a consistent sidecar: only the order is wrong
    with pytest.raises(AssertionError, match='keys differ'):
        g.read_replay_parts('artelingo', 'qwen', fake['out'])


def test_b_rows_in_a_wrong_order_change_the_values(fake):
    mem = g.clu_cell(fake['cand'], fake['comp'], fake['y'], 40)
    perm = np.random.default_rng(2).permutation(N)
    got, _, _ = g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'][:, perm], fake['y'], 40, fake['out'])
    res = g.finish(g.cell_items('t', 'src', got, stored_from(mem), True), 2000, 2000)
    assert not res['wiring_ok']


@pytest.mark.parametrize('arm', ['plain', 'cstar'])
def test_b_refuses_a_file_that_is_not_what_was_written(fake, monkeypatch, arm):
    real = od.turtle_parts

    def stale(set_, enc, arm_, *a, **kw):
        y = real(set_, enc, arm_, *a, **kw).copy()
        if arm_ == arm:
            y[0, 0] = (y[0, 0] + 1) % K
        return y
    monkeypatch.setattr(od, 'turtle_parts', stale)
    with pytest.raises(AssertionError, match='read back'):
        g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'], fake['y'], 20, fake['out'])


def test_b_second_run_overwrites_its_own_replay_files(fake):
    for _ in range(2):
        g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'], fake['y'], 20, fake['out'])


# ------------------------------------------------------------------------------------------------ (c) picks
def test_c_refuses_a_pick_that_differs_from_the_recorded_one():
    g.check_pick('center+k10', 'center+k10', 'c*')
    with pytest.raises(AssertionError, match='pick'):
        g.check_pick('center+k10', 'none+k10', 'c*')
