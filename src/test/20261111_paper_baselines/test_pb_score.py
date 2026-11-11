"""Stand-in data only: loader contracts, r17 score direction, gates and write guards."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_score as s
from pb_io import write_json
from rt_stats import outcome


def make_vlm(tmp_path, names, rows=None):
    root = tmp_path / 'emotion'
    root.mkdir()
    write_json(root / 'keys.json', names)
    np.save(root / 'emb.npy', np.eye(len(names), dtype=np.float32) if rows is None else rows)
    return root


def test_vlm_names_and_rows(tmp_path):
    names = ['a', 'b', 'c']
    root = make_vlm(tmp_path, names)
    rows, hashes = s.load_vlm2vec(tmp_path, 'emotion', names)
    assert rows.dtype == np.float32 and rows.shape == (3, 3) and len(hashes) == 2
    with pytest.raises(ValueError, match='names.*order'):
        s.load_vlm2vec(tmp_path, 'emotion', names[::-1])
    np.save(root / 'emb.npy', rows * 1.01)
    with pytest.raises(ValueError, match='unit norm'):
        s.load_vlm2vec(tmp_path, 'emotion', names)
    rows[0, 0] = np.nan
    np.save(root / 'emb.npy', rows)
    with pytest.raises(ValueError, match='finite'):
        s.load_vlm2vec(tmp_path, 'emotion', names)


@pytest.mark.parametrize('rows', [np.ones((4, 2)), np.ones(3), np.ones((3, 0))])
def test_vlm_refuses_wrong_shape(tmp_path, rows):
    make_vlm(tmp_path, ['a', 'b', 'c'], rows)
    with pytest.raises(ValueError, match='embedding rows'):
        s.load_vlm2vec(tmp_path, 'emotion', ['a', 'b', 'c'])


def make_topics(root, names, topics):
    for seed in s.P.UMAP_SEEDS:
        np.savez(root / f'topics_seed{seed}.npz', names=names, topics=topics)


def test_bertopic_names_values_and_distinct_count(tmp_path):
    names = [f'p{i}' for i in range(40)]
    good = np.tile(np.arange(20, dtype=np.int32), 2)
    make_topics(tmp_path, names, good)
    parts, hashes = s.load_bertopic(tmp_path, names)
    assert parts.dtype == np.int32 and parts.shape == (3, 40) and len(hashes) == 3
    with pytest.raises(ValueError, match='names.*order'):
        s.load_bertopic(tmp_path, names[::-1])
    for bad in (-1, 20):
        topics = good.copy()
        topics[0] = bad
        make_topics(tmp_path, names, topics)
        with pytest.raises(ValueError, match='outside'):
            s.load_bertopic(tmp_path, names)
    make_topics(tmp_path, names, good % 19)
    with pytest.raises(ValueError, match='20 distinct'):
        s.load_bertopic(tmp_path, names)
    make_topics(tmp_path, names, good.astype(float))
    with pytest.raises(ValueError, match='integer topics'):
        s.load_bertopic(tmp_path, names)


def test_missing_comparator_is_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='vlm2vec/emotion.*missing'):
        s.load_vlm2vec(tmp_path, 'emotion', ['a'])
    with pytest.raises(FileNotFoundError, match='bertopic.*missing'):
        s.load_bertopic(tmp_path, ['a'])


def standins():
    rng = np.random.default_rng(41)
    y = np.tile(np.arange(4), 20).astype(np.int64)
    y[-7:] = -1  # all rows clustered; unlabelled rows excluded only for AMI
    strong = np.tile(np.arange(4), 20).astype(np.int32)
    weak = rng.integers(0, 20, size=(3, len(y)), dtype=np.int32)
    cand = np.tile(strong, (5, 3, 1))
    data = {a: (cand, y) for a in s.P.ASPECTS}
    mean = s.ns_clu.metric_table('cstar', 'emotion', cand, y, 20, metrics=('ami',))[0]['mean']
    reference = dict(cells=[dict(id=f'SC1/artelingo/qwen/{a}/ami/20/tpips', cand=mean) for a in s.P.ASPECTS])
    return cand, weak, y, data, reference


@pytest.mark.parametrize('reverse,margin,label', [(False, .1, 'WIN'), (True, 2., 'LEVEL'), (True, 0., 'BELOW')])
def test_real_r17_direction_and_verdict(monkeypatch, reverse, margin, label):
    cand, comp, y, _, _ = standins()
    if reverse:
        cand, comp = np.tile(comp, (5, 1, 1)), cand[0]
    monkeypatch.setattr(s.P, 'SMOKE', 1)
    monkeypatch.setattr(s.ns_clu, 'WORKERS', 1)
    row = s.ns_clu.metric_table('cstar', 'emotion', cand, y, 20, metrics=('ami',))[0]
    gate = dict(cand=row['mean'], per_seed=row['per_seed'])
    cell = s.score_cell('bertopic', 'emotion', cand, comp, y, margin, gate, {})
    expected = s.ns_clu.paired_cell(cand, comp, y, 20, r={'ami': margin}, metrics=('ami',),
                                    draws=s.NS.SMOKE_DRAWS['clu'], rng_seed=s.NS.BOOT_SEED)['ami']
    assert {k: cell[k] for k in expected} == expected
    assert cell['d'] == pytest.approx(cell['cand'] - cell['comp'])
    assert cell['label'] == outcome(cell['lo'], margin) == label
    assert cell['n_labelled'] == 73 and cell['n_rows'] == 80
    assert len(cell['cand_per_seed_ami']) == 5 and len(cell['comp_per_state_ami']) == 3


@pytest.mark.parametrize('lo,r,label', [(1e-8, 0., 'WIN'), (0., .1, 'LEVEL'), (-.1, .1, 'BELOW'), (0., 0., 'BELOW')])
def test_r10_boundaries(lo, r, label):
    assert outcome(lo, r) == label


def test_gate_checks_all_aspects_before_any_output(tmp_path, monkeypatch):
    _, _, _, data, reference = standins()
    assert set(s.candidate_gate(data, reference)) == set(s.P.ASPECTS)
    reference['cells'][-1]['cand'] += 1e-5
    ref = tmp_path / 'r18.json'
    write_json(ref, reference)
    monkeypatch.setattr(s.P, 'R18_LABELS', ref)
    monkeypatch.setattr(s.P, 'OUT', tmp_path)
    monkeypatch.setattr(s, 'candidate_inputs', lambda: (['stand-in'], data))
    with pytest.raises(ValueError, match='gate FAILED for genre'):
        s.main(['--only', 'bertopic'])
    assert not (tmp_path / 'score' / 'cells.json').exists()


def test_smoke_refuses_committed_record(monkeypatch):
    monkeypatch.setattr(s.P, 'SMOKE', 1)
    with pytest.raises(SystemExit, match='2'):
        s.main(['--record'])


def test_smoke_end_to_end_uses_scratch_and_real_scorer(tmp_path, monkeypatch):
    cand, comp, y, data, reference = standins()
    ref, ranges_path = tmp_path / 'r18.json', tmp_path / 'ranges.json'
    write_json(ref, reference)
    write_json(ranges_path, {f'artelingo/qwen/{a}/ami/20': dict(r=.1) for a in s.P.ASPECTS})
    records = tmp_path / 'committed'
    monkeypatch.setattr(s.P, 'R18_LABELS', ref)
    monkeypatch.setattr(s.P, 'RANGES', ranges_path)
    monkeypatch.setattr(s.P, 'OUT', tmp_path / 'scratch')
    monkeypatch.setattr(s.P, 'RECORDS', records)
    monkeypatch.setattr(s.P, 'SMOKE', 1)
    monkeypatch.setattr(s.ns_clu, 'WORKERS', 1)
    monkeypatch.setattr(s, 'candidate_inputs', lambda: ([f'p{i}' for i in range(len(y))], data))
    monkeypatch.setattr(s, 'load_bertopic', lambda *args: (comp, {'stand-in': 'hash'}))
    assert s.main(['--only', 'bertopic']) == 0
    cells = json.loads((s.P.OUT / 'score' / 'cells.json').read_text())
    assert [c['id'] for c in cells] == [f'bertopic/{a}/ami/20' for a in s.P.ASPECTS]
    assert all(c['n_draws'] == 40 and c['label'] == 'WIN' for c in cells)
    assert not records.exists()
