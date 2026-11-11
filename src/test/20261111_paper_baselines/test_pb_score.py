"""Stand-in data only: loader contracts, r17 score direction, gates and write guards."""
import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_score as s
from pb_io import file_sha, write_json
from rt_stats import outcome


def make_vlm(tmp_path, names, rows=None):
    root = tmp_path / 'emotion'
    root.mkdir()
    write_json(root / 'keys.json', names)
    np.save(root / 'emb.npy', np.eye(len(names), dtype=np.float32) if rows is None else rows)
    write_json(root / 'meta.json', dict(aspect='emotion', instruction=s.P.INSTRUCTIONS['emotion'],
                                       emb_sha256=file_sha(root / 'emb.npy'), n=s.P.N_HELD, limit=None))
    return root


def refresh_emb_hash(root):
    meta = json.loads((root / 'meta.json').read_text())
    meta['emb_sha256'] = file_sha(root / 'emb.npy')
    write_json(root / 'meta.json', meta)


def test_vlm_names_and_rows(tmp_path):
    names = ['a', 'b', 'c']
    root = make_vlm(tmp_path, names)
    rows, hashes = s.load_vlm2vec(tmp_path, 'emotion', names)
    assert rows.dtype == np.float32 and rows.shape == (3, 3) and len(hashes) == 3
    with pytest.raises(ValueError, match='names.*order'):
        s.load_vlm2vec(tmp_path, 'emotion', names[::-1])
    np.save(root / 'emb.npy', rows * 1.01)
    refresh_emb_hash(root)
    with pytest.raises(ValueError, match='unit norm'):
        s.load_vlm2vec(tmp_path, 'emotion', names)
    rows[0, 0] = np.nan
    np.save(root / 'emb.npy', rows)
    refresh_emb_hash(root)
    with pytest.raises(ValueError, match='finite'):
        s.load_vlm2vec(tmp_path, 'emotion', names)


@pytest.mark.parametrize('rows', [np.ones((4, 2)), np.ones(3), np.ones((3, 0))])
def test_vlm_refuses_wrong_shape(tmp_path, rows):
    make_vlm(tmp_path, ['a', 'b', 'c'], rows)
    with pytest.raises(ValueError, match='embedding rows'):
        s.load_vlm2vec(tmp_path, 'emotion', ['a', 'b', 'c'])


def make_topics(root, names, topics, k=20):
    for seed in s.P.UMAP_SEEDS:
        path = s.topic_path(root, seed, k)
        np.savez(path, names=names, topics=topics)
        meta = dict(npz_sha256=file_sha(path), smoke=False, seed=seed)
        if k != 20:
            meta['k'] = k
        write_json(path.with_suffix('.json'), meta)


def test_bertopic_names_values_and_distinct_count(tmp_path):
    names = [f'p{i}' for i in range(40)]
    good = np.tile(np.arange(20, dtype=np.int32), 2)
    make_topics(tmp_path, names, good)
    parts, hashes = s.load_bertopic(tmp_path, names)
    assert parts.dtype == np.int32 and parts.shape == (3, 40) and len(hashes) == 6
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


@pytest.mark.parametrize('field,value', [('aspect', 'style'), ('instruction', 'wrong instruction'),
                                        ('emb_sha256', 'old-file'), ('n', 2000), ('limit', 2000)])
def test_vlm_refuses_wrong_metadata(tmp_path, field, value):
    root = make_vlm(tmp_path, ['a', 'b', 'c'])
    meta = json.loads((root / 'meta.json').read_text())
    meta[field] = value
    write_json(root / 'meta.json', meta)
    with pytest.raises(ValueError, match=field):
        s.load_vlm2vec(tmp_path, 'emotion', ['a', 'b', 'c'])


@pytest.mark.parametrize('field', ['aspect', 'instruction', 'emb_sha256', 'n', 'limit'])
def test_vlm_requires_metadata_fields(tmp_path, field):
    root = make_vlm(tmp_path, ['a', 'b', 'c'])
    meta = json.loads((root / 'meta.json').read_text())
    meta.pop(field)
    write_json(root / 'meta.json', meta)
    with pytest.raises(ValueError, match=field):
        s.load_vlm2vec(tmp_path, 'emotion', ['a', 'b', 'c'])


def test_vlm_requires_metadata_file(tmp_path):
    root = tmp_path / 'emotion'
    root.mkdir()
    np.save(root / 'emb.npy', np.eye(3, dtype=np.float32))
    write_json(root / 'keys.json', ['a', 'b', 'c'])
    with pytest.raises(FileNotFoundError, match='meta.json'):
        s.load_vlm2vec(tmp_path, 'emotion', ['a', 'b', 'c'])


def test_vlm_hashes_and_loads_one_snapshot(tmp_path, monkeypatch):
    names = ['a', 'b', 'c']
    root = make_vlm(tmp_path, names)
    path = root / 'emb.npy'
    original_bytes = path.read_bytes()
    original_read_bytes = Path.read_bytes
    original_load = np.load
    reads = []

    def read_then_replace(p):
        data = original_read_bytes(p)
        if p == path:
            reads.append(p)
            np.save(path, np.eye(3, dtype=np.float32)[::-1])
        return data

    def load_snapshot(source, *args, **kwargs):
        assert hasattr(source, 'getvalue'), 'np.load must consume the checked byte snapshot'
        assert source.getvalue() == original_bytes
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_bytes', read_then_replace)
    monkeypatch.setattr(s.np, 'load', load_snapshot)
    rows, hashes = s.load_vlm2vec(tmp_path, 'emotion', names)
    assert reads == [path]
    assert np.array_equal(rows, np.eye(3, dtype=np.float32))
    assert hashes[str(path)] == hashlib.sha256(original_bytes).hexdigest()
    assert hashes[str(path)] != file_sha(path)


@pytest.mark.parametrize('field,value', [('smoke', True), ('seed', 1), ('smoke', None), ('seed', None)])
def test_bertopic_checks_sidecar_seed_and_smoke(tmp_path, field, value):
    names = [f'p{i}' for i in range(20)]
    make_topics(tmp_path, names, np.arange(20, dtype=np.int32))
    side = tmp_path / 'topics_seed0.json'
    meta = json.loads(side.read_text())
    if value is None:
        meta.pop(field)
    else:
        meta[field] = value
    write_json(side, meta)
    with pytest.raises(ValueError, match=field):
        s.load_bertopic(tmp_path, names)


def test_bertopic_requires_sidecar(tmp_path):
    names = [f'p{i}' for i in range(20)]
    np.savez(tmp_path / 'topics_seed0.npz', names=names, topics=np.arange(20, dtype=np.int32))
    with pytest.raises(FileNotFoundError, match='topics_seed0.json'):
        s.load_bertopic(tmp_path, names)


def standins():
    rng = np.random.default_rng(41)
    y = np.tile(np.arange(4), 20).astype(np.int64)
    y[-7:] = -1  # all rows clustered; unlabelled rows excluded only for AMI
    strong = np.tile(np.arange(4), 20).astype(np.int32)
    weak = rng.integers(0, 20, size=(3, len(y)), dtype=np.int32)
    cand = np.tile(strong, (5, 3, 1))
    data = {(a, k): (cand, y) for a in s.P.ASPECTS for k in s.aspect_ks(a)}
    mean = s.ns_clu.metric_table('cstar', 'emotion', cand, y, 20, metrics=('ami',))[0]['mean']
    reference = dict(cells=[dict(id=f'SC1/artelingo/qwen/{a}/ami/{k}/tpips', cand=mean)
                            for a in s.P.ASPECTS for k in s.aspect_ks(a)])
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


@pytest.mark.parametrize('bad_k', [10, 20])
def test_gate_checks_all_aspects_before_any_output(tmp_path, monkeypatch, bad_k):
    _, _, _, data, reference = standins()
    assert set(s.candidate_gate(data, reference)) == {f'{a}/{k}' for a in s.P.ASPECTS for k in s.aspect_ks(a)}
    bad = next(c for c in reference['cells'] if c['id'] == f'SC1/artelingo/qwen/genre/ami/{bad_k}/tpips')
    bad['cand'] += 1e-5
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


@pytest.mark.parametrize('only', [None, 'bertopic', 'vlm2vec'])
def test_smoke_end_to_end_uses_scratch_and_real_scorer(tmp_path, monkeypatch, only):
    cand, comp, y, data, reference = standins()
    ref, ranges_path = tmp_path / 'r18.json', tmp_path / 'ranges.json'
    write_json(ref, reference)
    write_json(ranges_path, {f'artelingo/qwen/{a}/ami/{k}': dict(r=.1 + k / 1000)
                            for a in s.P.ASPECTS for k in s.aspect_ks(a)})
    records = tmp_path / 'committed'
    monkeypatch.setattr(s.P, 'R18_LABELS', ref)
    monkeypatch.setattr(s.P, 'RANGES', ranges_path)
    monkeypatch.setattr(s.P, 'OUT', tmp_path / 'scratch')
    monkeypatch.setattr(s.P, 'RECORDS', records)
    monkeypatch.setattr(s.P, 'SMOKE', 1)
    monkeypatch.setattr(s.ns_clu, 'WORKERS', 1)
    monkeypatch.setattr(s, 'candidate_inputs', lambda: ([f'p{i}' for i in range(len(y))], data))
    monkeypatch.setattr(s, 'load_bertopic', lambda root, names, k: (comp % k, {'stand-in': f'hash{k}'}))
    monkeypatch.setattr(s, 'load_vlm2vec', lambda *args: (np.eye(len(y), dtype=np.float32), {'stand-in': 'hash'}))
    fits = []

    def partitions(rows, k, set_name):
        fits.append((k, set_name))
        return comp % k

    monkeypatch.setattr(s.ns_metrics, 'partitions', partitions)
    assert s.main([] if only is None else ['--only', only]) == 0
    cells = json.loads((s.P.OUT / 'score' / 'cells.json').read_text())
    comparators = s.COMPARATORS if only is None else (only,)
    assert [c['id'] for c in cells] == [f'{c}/{a}/ami/{k}' for c in comparators for a in s.P.ASPECTS for k in s.aspect_ks(a)]
    assert all(c['n_draws'] == 40 and c['label'] == 'WIN' for c in cells)
    assert all(c['r'] == .1 + c['k'] / 1000 for c in cells)
    assert fits == ([] if only == 'bertopic' else [(k, s.P.SET) for a in s.P.ASPECTS for k in s.aspect_ks(a)])
    assert not records.exists()


@pytest.mark.parametrize('k', [9, 10, 27])
def test_bertopic_loads_class_count_files(tmp_path, k):
    names = [f'p{i}' for i in range(2 * k)]
    make_topics(tmp_path, names, np.tile(np.arange(k, dtype=np.int32), 2), k)
    parts, _ = s.load_bertopic(tmp_path, names, k)
    assert parts.shape == (3, len(names)) and np.array_equal(np.unique(parts), np.arange(k))
    with pytest.raises(ValueError, match='names.*order'):
        s.load_bertopic(tmp_path, names[::-1], k)
    bad = np.arange(len(names), dtype=np.int32) % k
    bad[0] = k
    make_topics(tmp_path, names, bad, k)
    with pytest.raises(ValueError, match='outside'):
        s.load_bertopic(tmp_path, names, k)


def test_bertopic_does_not_substitute_k20_for_class_count(tmp_path):
    names = [f'p{i}' for i in range(40)]
    make_topics(tmp_path, names, np.arange(40, dtype=np.int32) % 20)
    with pytest.raises(FileNotFoundError, match='topics_k9_seed0.npz'):
        s.load_bertopic(tmp_path, names, 9)
    make_topics(tmp_path, names, np.arange(40, dtype=np.int32) % 9, 9)
    side = tmp_path / 'topics_k9_seed0.json'
    meta = json.loads(side.read_text())
    meta['k'] = 20
    write_json(side, meta)
    with pytest.raises(ValueError, match='k must be 9'):
        s.load_bertopic(tmp_path, names, 9)


@pytest.mark.parametrize('wrong_count', [False, True])
def test_candidate_loads_both_counts_and_checks_label_classes(monkeypatch, wrong_count):
    n = s.P.N_HELD
    names = [f'p{i}' for i in range(n)]

    def labels(set_name, split, aspect):
        k = 8 if wrong_count and aspect == 'emotion' else s.CLASS_COUNTS[aspect]
        y = np.arange(n, dtype=np.int64) % k
        if aspect == 'genre':
            y[9949:] = -1
        return y

    fits = []

    def parts(set_name, split, enc, aspect, k):
        fits.append((set_name, split, enc, aspect, k))
        return np.tile(np.arange(n, dtype=np.int32) % k, (5, 3, 1))

    monkeypatch.setitem(sys.modules, 'ns_data', SimpleNamespace(keys=lambda *args: names, labels=labels))
    monkeypatch.setitem(sys.modules, 'ns_clu_cells', SimpleNamespace(_cand_parts=parts))
    if wrong_count:
        with pytest.raises(AssertionError, match='emotion'):
            s.candidate_inputs()
        assert not fits
    else:
        loaded, data = s.candidate_inputs()
        assert loaded == names
        assert list(data) == [(a, k) for a in s.P.ASPECTS for k in s.aspect_ks(a)]
        assert fits == [(s.P.SET, s.P.SPLIT, s.P.CSTAR_ENC, a, k) for a, k in data]


def test_gate_only_checks_six_cells_without_comparator_inputs(tmp_path, monkeypatch, capsys):
    _, _, _, data, reference = standins()
    ref = tmp_path / 'r18.json'
    write_json(ref, reference)
    monkeypatch.setattr(s.P, 'R18_LABELS', ref)
    monkeypatch.setattr(s.P, 'OUT', tmp_path)
    monkeypatch.setattr(s, 'candidate_inputs', lambda: (['stand-in'], data))
    monkeypatch.setattr(s, 'load_bertopic', lambda *args: pytest.fail('gate must not load comparators'))
    monkeypatch.setattr(s, 'load_vlm2vec', lambda *args: pytest.fail('gate must not load comparators'))
    assert s.main(['--gate-only']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['gate'] == 'PASS' and len(report['aspects']) == 6
    assert not (tmp_path / 'score' / 'cells.json').exists()
