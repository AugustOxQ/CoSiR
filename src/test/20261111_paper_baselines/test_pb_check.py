"""Independent checker paths with small stand-in stores; no real inputs or clustering jobs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_check as c
from pb_io import write_json


@pytest.mark.parametrize('wrapped', [False, True])
def test_check_compares_all_twelve_cells_with_own_sklearn_fits(tmp_path, monkeypatch, wrapped):
    names = [f'p{i}' for i in range(60)]
    bt = tmp_path / 'bertopic'
    bt.mkdir()
    store = tmp_path / 'vlm2vec'
    labels = {a: np.arange(len(names), dtype=np.int64) % c.CLASS_COUNTS[a] for a in c.P.ASPECTS}
    cells = []
    fits = []
    for k in c.TARGET_KS:
        for seed in c.P.UMAP_SEEDS:
            np.savez(c.topic_path(bt, seed, k), names=names, topics=np.arange(len(names), dtype=np.int32) % k)
    for aspect in c.P.ASPECTS:
        root = store / aspect
        root.mkdir(parents=True)
        np.save(root / 'emb.npy', np.ones((len(names), 2), dtype=np.float32))
        write_json(root / 'keys.json', names)
        for k in c.aspect_ks(aspect):
            mean = c.ami_mean(labels[aspect], [np.arange(len(names)) % k] * 3)
            cells.extend(dict(id=f'{comp}/{aspect}/ami/{k}', comp=mean) for comp in ('vlm2vec', 'bertopic'))
    write_json(tmp_path / 'score' / 'cells.json', dict(cells=cells) if wrapped else cells)

    class KMeans:
        def __init__(self, **kwargs):
            fits.append(kwargs)
            self.k = kwargs['n_clusters']

        def fit(self, rows):
            assert rows.dtype == np.float32
            self.labels_ = np.arange(len(rows), dtype=np.int32) % self.k
            return self

    monkeypatch.setattr(c, 'MiniBatchKMeans', KMeans)
    monkeypatch.setattr(c.P, 'OUT', tmp_path)
    monkeypatch.setattr(c.P, 'BERTOPIC_OUT', bt)
    monkeypatch.setattr(c.P, 'VLM2VEC_STORE', store)
    ns = SimpleNamespace(CLASS_COUNT={(c.P.SET, a): k for a, k in c.CLASS_COUNTS.items()})
    monkeypatch.setattr(c.P, 'add_scorer_paths', lambda: ns)
    monkeypatch.setitem(sys.modules, 'ns_data', SimpleNamespace(keys=lambda *args: names,
                         labels=lambda set_name, split, aspect: labels[aspect]))
    c.main([])
    result = json.loads((tmp_path / 'score' / 'check.json').read_text())
    assert len(result['rows']) == 12 and result['max_gap'] == 0
    assert {r['id'] for r in result['rows']} == {r['id'] for r in cells}
    assert fits == [dict(n_clusters=k, random_state=s, n_init=3, batch_size=4096)
                    for a in c.P.ASPECTS for k in c.aspect_ks(a) for s in (0, 1, 2)]


def test_check_refuses_invalid_class_count_partition(tmp_path):
    names = [f'p{i}' for i in range(20)]
    for seed in c.P.UMAP_SEEDS:
        np.savez(c.topic_path(tmp_path, seed, 9), names=names, topics=np.arange(20, dtype=np.int32) % 8)
    with pytest.raises(AssertionError, match='9 real topics'):
        c.bertopic_parts(tmp_path, names, 9)
