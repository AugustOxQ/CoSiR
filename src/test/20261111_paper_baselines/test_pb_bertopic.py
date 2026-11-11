"""Pure runner helpers: no BERTopic import, model download or real stores."""
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_bertopic as b
from pb_io import file_sha, names_sha, write_json


def test_relabel_preserves_partition():
    topics = np.tile(np.arange(20) * 7 + 3, 3)[::-1]
    final, mapping = b.relabel(topics)
    assert final.dtype == np.int32
    assert np.array_equal(np.unique(final), np.arange(20))
    assert np.array_equal(topics[:, None] == topics, final[:, None] == final)
    assert mapping == {i * 7 + 3: i for i in range(20)}


@pytest.mark.parametrize('topics', [[0, -1, 3], [0.0, 1.0], [0, 1], []])
def test_relabel_rejects_invalid(topics):
    with pytest.raises(ValueError):
        b.relabel(topics)


def test_nr_topics_counts_outlier_and_smoke_does_not_invent_topics():
    assert b.reduction_count(list(range(30)) + [-1]) == (21, 20)
    assert b.reduction_count(list(range(30))) == (20, 20)
    with pytest.raises(ValueError, match='without fallback'):
        b.reduction_count([0, 1, -1])
    assert b.reduction_count([0, 1, -1], smoke=True) == (3, 2)
    with pytest.raises(ValueError):
        b.reduction_count([-1], smoke=True)


def test_docs_checks_full_hash_before_limit(tmp_path):
    path = tmp_path / 'docs.json'
    names = ['a', 'b', 'c']
    data = dict(names=names, docs=['aa', 'bb', 'cc'], n_captions=[1, 2, 3], names_sha256=names_sha(names))
    write_json(path, data)
    assert b.load_docs(path, limit=2, expected_n=3) == (names[:2], ['aa', 'bb'], [1, 2])
    data['names'] = ['a', 'c', 'b']
    write_json(path, data)
    with pytest.raises(ValueError, match='names_sha256'):
        b.load_docs(path, limit=2, expected_n=3)


def test_completed_seed_needs_both_files_and_valid_hash(tmp_path):
    path = tmp_path / 'topics_seed0.npz'
    names = [f'p{i}' for i in range(20)]
    assert not b.seed_done(path, names, 'input', False)
    np.savez(path, names=names, topics=np.arange(20, dtype=np.int32))
    assert not b.seed_done(path, names, 'input', False)
    np.save(tmp_path / 'minilm.npy', np.eye(20, dtype=np.float32))
    meta = dict(npz_sha256=file_sha(path), input_sha256='input', smoke=False, n_topics=20,
                embeddings_sha256=file_sha(tmp_path / 'minilm.npy'))
    write_json(path.with_suffix('.json'), meta)
    assert b.seed_done(path, names, 'input', False)
    original_sha = meta['embeddings_sha256']
    meta['embeddings_sha256'] = 'old-embeddings'
    write_json(path.with_suffix('.json'), meta)
    assert not b.seed_done(path, names, 'input', False)
    meta['embeddings_sha256'] = original_sha
    write_json(path.with_suffix('.json'), meta)
    with pytest.raises(ValueError, match='names'):
        b.seed_done(path, names[::-1], 'input', False)
    meta['npz_sha256'] = 'bad'
    write_json(path.with_suffix('.json'), meta)
    with pytest.raises(ValueError, match='stale/corrupt'):
        b.seed_done(path, names, 'input', False)


def test_infinite_default_is_recorded_as_string():
    class Model:
        def get_params(self, deep):
            return dict(epsilon=float('inf'), count=10, metric='euclidean')
    assert json.loads(json.dumps(b.json_params(Model()), allow_nan=False))['epsilon'] == 'inf'


def test_runner_uses_default_then_embedding_outliers(tmp_path, monkeypatch):
    names = [f'p{i}' for i in range(22)]
    emb = np.zeros((22, 384), dtype=np.float32)

    class Model:
        def fit_transform(self, docs, embeddings):
            assert embeddings is emb
            self.topics_ = list(range(20)) + [-1, -1]
            return self.topics_, None

        def reduce_topics(self, docs, nr_topics):
            assert nr_topics == 21

        def reduce_outliers(self, docs, topics, **kwargs):
            if not kwargs:
                assert topics[-2:] == [-1, -1]
                return topics[:-2] + [0, -1]
            assert kwargs['strategy'] == 'embeddings' and kwargs['embeddings'] is emb
            assert topics[-2:] == [0, -1]
            return topics[:-1] + [1]

    monkeypatch.setattr(b, 'make_model', lambda seed: (Model(), {}))
    monkeypatch.setattr(b, 'versions', lambda: {'stand-in': '1'})
    np.save(tmp_path / 'minilm.npy', emb)
    meta = b.run_seed(['doc'] * 22, names, emb, {'sha256': file_sha(tmp_path / 'minilm.npy')},
                      tmp_path, 0, 'input', False)
    assert meta['outlier_counts'] == dict(after_hdbscan=2, after_reduce_topics=2,
                                          after_default_outliers=1, after_embedding_outliers=0)
    assert sum(meta['topic_sizes']) == 22
    assert b.seed_done(tmp_path / 'topics_seed0.npz', names, 'input', False)


def test_runner_stops_if_hdbscan_has_fewer_than_20(tmp_path, monkeypatch):
    class Model:
        def fit_transform(self, docs, embeddings):
            return [0, 1, -1], None

        def reduce_topics(self, *args, **kwargs):
            pytest.fail('must stop before reduction')

    monkeypatch.setattr(b, 'make_model', lambda seed: (Model(), {}))
    with pytest.raises(ValueError, match='fewer than 20'):
        b.run_seed(['doc'] * 3, ['a', 'b', 'c'], None, {}, tmp_path, 0, 'input', False)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def minilm(monkeypatch):
    """A model with the installed library's public module classes, without importing Torch."""
    class Transformer:
        pass

    class Pooling:
        pooling_mode = 'mean'

    class Normalize:
        pass

    class Model:
        max_seq_length = 256

        def __init__(self):
            self.modules = [Transformer(), Pooling(), Normalize()]
            self.rows = np.eye(2, dtype=np.float32)
            self.encode_calls = []

        def __iter__(self):
            return iter(self.modules)

        def tokenizer(self, docs, **kwargs):
            return {'length': [4] * len(docs)}

        def encode(self, docs, **kwargs):
            self.encode_calls.append(kwargs)
            return self.rows

    model = Model()
    packages = {}
    for name in ('sentence_transformers', 'sentence_transformers.base', 'sentence_transformers.base.modules',
                 'sentence_transformers.sentence_transformer', 'sentence_transformers.sentence_transformer.modules'):
        packages[name] = ModuleType(name)
        packages[name].__path__ = []
        monkeypatch.setitem(sys.modules, name, packages[name])
    packages['sentence_transformers'].SentenceTransformer = lambda *args, **kwargs: model
    packages['sentence_transformers.base.modules'].Transformer = Transformer
    packages['sentence_transformers.base.modules'].Normalize = Normalize
    packages['sentence_transformers.sentence_transformer.modules'].Pooling = Pooling
    monkeypatch.setattr(b, 'versions', lambda: {'stand-in': '1'})
    return model


@pytest.mark.parametrize('defect', ['missing_normalize', 'wrong_order', 'wrong_type', 'pooling', 'length'])
def test_minilm_refuses_fallback_before_encoding(tmp_path, minilm, defect):
    if defect == 'missing_normalize':
        minilm.modules.pop()
    elif defect == 'wrong_order':
        minilm.modules.reverse()
    elif defect == 'wrong_type':
        class OtherTransformer(type(minilm.modules[0])):
            pass
        minilm.modules[0] = OtherTransformer()
    elif defect == 'pooling':
        minilm.modules[1].pooling_mode = ('mean', 'max')
    else:
        minilm.max_seq_length = 512
    with pytest.raises(ValueError, match='MiniLM'):
        b.document_embeddings(['a', 'b'], ['p0', 'p1'], tmp_path, 'input')
    assert not minilm.encode_calls
    assert not (tmp_path / 'minilm.npy').exists()


def test_minilm_checks_unit_rows_and_records_model(tmp_path, minilm):
    rows, meta = b.document_embeddings(['a', 'b'], ['p0', 'p1'], tmp_path, 'input')
    assert np.allclose(np.linalg.norm(rows, axis=1), 1)
    assert meta['modules'] == ['Transformer', 'Pooling', 'Normalize']
    assert meta['pooling_mode'] == 'mean' and meta['max_seq_length'] == 256
    assert 'normalize_embeddings' not in minilm.encode_calls[0]
    cached, _ = b.document_embeddings(['a', 'b'], ['p0', 'p1'], tmp_path, 'input')
    assert np.array_equal(rows, cached) and len(minilm.encode_calls) == 1
    # Even a self-consistent file hash must not bypass unit validation on cache reuse.
    np.save(tmp_path / 'minilm.npy', rows * 2)
    meta['sha256'] = file_sha(tmp_path / 'minilm.npy')
    write_json(tmp_path / 'minilm.json', meta)
    with pytest.raises(ValueError, match='unit'):
        b.document_embeddings(['a', 'b'], ['p0', 'p1'], tmp_path, 'input')


@pytest.mark.parametrize('scale', [0., 1.0002, 2., float('nan')])
def test_minilm_refuses_nonunit_encoding(tmp_path, minilm, scale):
    minilm.rows *= scale
    with pytest.raises(ValueError, match='MiniLM'):
        b.document_embeddings(['a', 'b'], ['p0', 'p1'], tmp_path, 'input')
    assert not (tmp_path / 'minilm.npy').exists()


@pytest.mark.parametrize('legacy_field', ['modules', 'max_seq_length', 'pooling_mode'])
def test_minilm_recomputes_legacy_cache(tmp_path, minilm, legacy_field):
    _, meta = b.document_embeddings(['a', 'b'], ['p0', 'p1'], tmp_path, 'input')
    meta.pop(legacy_field, None)
    np.save(tmp_path / 'minilm.npy', minilm.rows * 2)
    meta['sha256'] = file_sha(tmp_path / 'minilm.npy')
    write_json(tmp_path / 'minilm.json', meta)
    rows, repaired = b.document_embeddings(['a', 'b'], ['p0', 'p1'], tmp_path, 'input')
    assert len(minilm.encode_calls) == 2 and np.allclose(np.linalg.norm(rows, axis=1), 1)
    assert repaired['modules'] == ['Transformer', 'Pooling', 'Normalize']
    assert repaired['max_seq_length'] == 256


def test_main_validates_embedding_cache_before_skipping_seeds(tmp_path, monkeypatch):
    names = ['a', 'b']
    events = []
    monkeypatch.setattr(b, 'load_docs', lambda *args: (names, ['doc'] * 2, [1, 1]))
    monkeypatch.setattr(b, 'file_sha', lambda path: 'input')

    def embeddings(*args):
        events.append('embeddings')
        return np.eye(2), {}

    def done(*args):
        assert events == ['embeddings']
        return True

    monkeypatch.setattr(b, 'document_embeddings', embeddings)
    monkeypatch.setattr(b, 'seed_done', done)
    assert b.main(['--seed', '0', '--out', str(tmp_path)]) == 0
