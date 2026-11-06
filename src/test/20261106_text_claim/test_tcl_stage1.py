"""Fixture-file checks of text-claim design §3.4, §3.5 and §6 selection."""
import json

import numpy as np
import pytest

import tcl_paths as p
import tcl_stage1 as s1


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUT', tmp_path)
    monkeypatch.setattr(p, 'RECORDS', tmp_path / 'records')
    monkeypatch.setattr(p, 'SMOKE', 0)
    monkeypatch.setitem(p.N_PAINTINGS, 'art_val', 3)
    monkeypatch.setitem(p.N_PAINTINGS, 'mm_val', 3)
    monkeypatch.setitem(p.N_CAPTIONS, 'art_val', 4)


def score(tmp_path, *, set_name='art_val', kind='caption', row='plain', head='all', seed='x',
          direction='t2i', readout='r0', ap=None, ranks=None, pid=None):
    path = p.score_path(set_name, kind, row, head, seed, direction, readout, out=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = set_name.startswith('art') and direction != 'i2t'
    ids = np.array([0, 0, 1, 2] if text else [0, 1, 2], dtype=np.int64)
    aspects = (head,) if row in p.HEAD_ROWS else (('emotion',) if row == 'roberta' else p.ASPECTS)
    fields = dict(q_pid=ids if pid is None else pid,
                  first_rank=np.full(len(ids), 20, dtype=np.float32) if ranks is None
                  else np.asarray(ranks, dtype=np.float32))
    for aspect in aspects:
        values = np.full(len(ids), 0.25, dtype=np.float32) if ap is None else np.asarray(ap, dtype=np.float32)
        for suffix in ('', '__no_leak', '__cross_painting'):
            fields[f'ap__{aspect}{suffix}'] = values
    np.savez(path, **fields)
    return path


def all_scores(tmp_path, set_name='art_val'):
    group = p.GROUP[set_name]
    rows = p.BAR_ROWS + (p.SIDE_ROWS if group == 'art' else ())
    for kind in p.kinds(group):
        for row in rows:
            for head in p.ASPECTS if row in p.HEAD_ROWS else ('all',):
                for seed in p.SEEDS if row in p.SEEDED_ROWS else ('x',):
                    for direction in ('t2t',) if row == 'roberta' else p.DIRECTIONS:
                        for readout in p.READOUTS:
                            score(tmp_path, set_name=set_name, kind=kind, row=row, head=head,
                                  seed=seed, direction=direction, readout=readout)


def test_label_picks_seed_mean_exact_ties_and_unrounded_range(tmp_path):
    all_scores(tmp_path)
    # r0 has the best individual seed (1), but center's seed mean is 0.5.
    for seed, value in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
        score(tmp_path, row='cstar', head='emotion', seed=seed, ap=[1 if seed == 0 else 0] * 4)
        score(tmp_path, row='cstar', head='emotion', seed=seed, readout='center',
              ap=[0.25, np.nan, 0.75, np.nan])
        score(tmp_path, row='cstar', head='emotion', seed=seed, readout='center_k10', ap=[0.49] * 4)
        # A different direction has a nonzero range, measured at its own pick.
        score(tmp_path, row='cstar', head='emotion', seed=seed, direction='i2t',
              readout='center_k10', ap=[value + 0.3] * 3)
    score(tmp_path, readout='center', ap=[0.75, np.nan, 0.25, np.nan])
    score(tmp_path, readout='center_k10', ap=[0.5] * 4)
    # One float32 step above r0 must win: §6 allows no tie band.
    score(tmp_path, direction='t2t', readout='center', ap=[0.2500000298023224] * 4)

    got = s1.run('art_val')

    assert got['group'] == 'art'
    assert got['picks']['caption']['cstar']['emotion']['t2i'] == 'center'
    assert got['values']['caption']['cstar']['emotion']['t2i'] == pytest.approx(
        dict(r0=0.2, center=0.5, center_k10=0.49))
    assert got['ranges']['caption']['cstar']['emotion']['t2i'] == 0
    assert got['picks']['caption']['cstar']['emotion']['i2t'] == 'center_k10'
    assert got['ranges']['caption']['cstar']['emotion']['i2t'] == 0.4000000059604645
    assert got['picks']['caption']['plain']['emotion']['t2i'] == 'center'
    assert got['picks']['caption']['plain']['style']['i2t'] == 'r0'
    assert got['picks']['caption']['plain']['style']['t2t'] == 'center'
    assert set(got['ranges']['caption']) == set(p.SEEDED_ROWS)
    assert got['picks']['caption']['roberta'] == {'emotion': {'t2t': 'r0'}}


def test_pair_picks_mrr_then_r10_with_seed_means_and_exact_ties(tmp_path):
    all_scores(tmp_path)
    for seed in p.SEEDS:
        score(tmp_path, row='late', head='emotion', seed=seed, ranks=[1, np.nan, np.nan, np.nan])
        score(tmp_path, row='late', head='emotion', seed=seed, readout='center', ranks=[10] * 4)
        score(tmp_path, row='late', head='style', seed=seed,
              ranks=[1] * 4 if seed == 0 else [3, 3, np.nan, np.nan])
        for readout in ('center', 'center_k10'):
            score(tmp_path, row='late', head='style', seed=seed, readout=readout,
                  ranks=[2, 2, 2, np.nan])
        score(tmp_path, row='late', head='genre', seed=seed, ranks=[2, 2, 2, np.nan])
        score(tmp_path, row='late', head='genre', seed=seed, readout='center',
              ranks=[1] * 4 if seed == 0 else [20] * 4)
        # In i2t, style wins on its best seed but genre wins on the five-seed mean.
        score(tmp_path, row='late', head='style', seed=seed, direction='i2t',
              ranks=[1] * 3 if seed == 0 else [20] * 3)
        score(tmp_path, row='late', head='genre', seed=seed, direction='i2t', ranks=[1, 20, 20])
    for readout in ('r0', 'center_k10'):
        score(tmp_path, readout=readout, ranks=[1, np.nan, np.nan, np.nan])
    score(tmp_path, readout='center', ranks=[5] * 4)
    for readout in ('center', 'center_k10'):
        score(tmp_path, direction='i2t', readout=readout, ranks=[2] * 3)

    got = s1.run('art_val')

    assert got['pair_picks']['caption'] == {
        't2i': {'late': {'head': 'style', 'readout': 'center'}, 'plain': {'readout': 'r0'}},
        'i2t': {'late': {'head': 'genre', 'readout': 'r0'}, 'plain': {'readout': 'center'}},
    }
    # Label and pair read-outs are selected independently.
    assert got['picks']['caption']['late']['style']['t2i'] == 'r0'


@pytest.mark.parametrize('set_name,group', [('art_val', 'art'), ('mm_val', 'mm')])
def test_record_covers_group_and_is_byte_identical_on_rerun(tmp_path, set_name, group):
    all_scores(tmp_path, set_name)
    if group == 'mm':
        score(tmp_path, set_name=set_name, kind='historical', readout='center', ap=[0.75] * 3)
    got = s1.run(set_name)
    path = p.record_path(f'stage1_{group}.json')
    first = path.read_bytes()

    assert json.loads(first) == got
    assert first.decode() == json.dumps(got, sort_keys=True, indent=2, allow_nan=False) + '\n'
    assert set(got) == {'group', 'picks', 'values', 'ranges', 'pair_picks'}
    assert got['group'] == group
    for key in ('picks', 'values', 'ranges', 'pair_picks'):
        assert set(got[key]) == set(p.kinds(group))
    assert set(got['picks'][p.kinds(group)[0]]) == set(
        p.BAR_ROWS + (p.SIDE_ROWS if group == 'art' else ()))
    if group == 'mm':
        assert got['picks']['historical']['plain']['genre']['t2i'] == 'center'
        assert got['picks']['narrative']['plain']['genre']['t2i'] == 'r0'
    assert s1.run(set_name) == got
    assert path.read_bytes() == first


@pytest.mark.parametrize('set_name', ['art_held', 'mm_test'])
def test_held_sets_are_rejected_before_any_read(tmp_path, set_name):
    with pytest.raises(AssertionError, match='validation'):
        s1.run(set_name)
    assert not list((tmp_path / 'records').glob('*'))


def test_missing_file_crashes_without_writing_record(tmp_path):
    # Leave a hole by writing just the first seed/read-out; delete nothing.
    score(tmp_path, row='cstar', head='emotion', seed=0)
    with pytest.raises(FileNotFoundError):
        s1.run('art_val')
    assert not p.record_path('stage1_art.json').exists()


@pytest.mark.parametrize('bad', ['ap_shape', 'ap_dtype', 'ap_infinite', 'ap_empty', 'ap_bounds',
                                 'pid_order', 'pid_dtype', 'row_count', 'ranks_shape', 'ranks_zero'])
def test_invalid_score_input_crashes_before_writing_record(tmp_path, bad):
    all_scores(tmp_path)
    path = p.score_path('art_val', 'caption', 'cstar', 'emotion', 1, 't2i', 'center')
    with np.load(path, allow_pickle=False) as file:
        fields = dict(file)
    replacements = {
        'ap_shape': ('ap__emotion', np.zeros((4, 1), dtype=np.float32)),
        'ap_dtype': ('ap__emotion', np.zeros(4, dtype=np.float64)),
        'ap_infinite': ('ap__emotion', np.array([0, 0, 0, np.inf], dtype=np.float32)),
        'ap_empty': ('ap__emotion', np.full(4, np.nan, dtype=np.float32)),
        'ap_bounds': ('ap__emotion', np.full(4, 1.1, dtype=np.float32)),
        'pid_order': ('q_pid', np.array([0, 1, 0, 2], dtype=np.int64)),
        'pid_dtype': ('q_pid', np.array([0, 0, 1, 2], dtype=np.int32)),
        'row_count': ('q_pid', np.array([0, 0, 1], dtype=np.int64)),
        'ranks_shape': ('first_rank', np.ones(3, dtype=np.float32)),
        'ranks_zero': ('first_rank', np.zeros(4, dtype=np.float32)),
    }
    key, value = replacements[bad]
    fields[key] = value
    np.savez(path, **fields)

    with pytest.raises(AssertionError):
        s1.run('art_val')
    assert not p.record_path('stage1_art.json').exists()


def test_a_held_score_path_is_rejected(tmp_path, monkeypatch):
    held = score(tmp_path, set_name='art_held', row='cstar', head='emotion', seed=0)
    monkeypatch.setattr(p, 'score_path', lambda *args: held)
    with pytest.raises(AssertionError, match='validation'):
        s1.run('art_val')
    assert not p.record_path('stage1_art.json').exists()
