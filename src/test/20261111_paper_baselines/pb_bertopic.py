"""r25 BERTopic partitions; run with /root/venvs/bertopic/bin/python.

Production: all held documents, seeds 0/1/2, exactly 20 non-outlier topics.
Smoke: --limit 2000 --seed 0 --out <scratch> (may retain fewer real topics).
The installed source audit is pinned to BERTopic 0.17.4; see ticket_b_notes.md.
"""
import argparse
import inspect
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_paths as P
from pb_io import atomic_write, file_sha, names_sha, write_json

AUDITED_VERSION = '0.17.4'
# _bertopic.py __init__: low_memory defaults to False, min_topic_size to 10.
UMAP_ARGS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', low_memory=False)
PACKAGES = ('bertopic', 'sentence-transformers', 'transformers', 'torch', 'numpy', 'scipy',
            'scikit-learn', 'umap-learn', 'hdbscan', 'numba', 'pynndescent')
MINILM_MODULES = ['Transformer', 'Pooling', 'Normalize']
MINILM_MAX_SEQ_LENGTH = 256
MINILM_UNIT_TOL = 1e-4


def versions():
    return {p: version(p) for p in PACKAGES}


def load_docs(path, limit=None, expected_n=P.N_HELD):
    data = json.loads(Path(path).read_text())
    names, docs, caps = data['names'], data['docs'], data['n_captions']
    if (len(names) != expected_n or len(set(names)) != expected_n
            or len(docs) != expected_n or len(caps) != expected_n):
        raise ValueError(f'{path}: expected {expected_n} unique names and aligned docs/n_captions')
    if data['names_sha256'] != names_sha(names):
        raise ValueError(f'{path}: names_sha256 mismatch')
    if not all(isinstance(d, str) and d.strip() for d in docs):
        raise ValueError(f'{path}: empty or non-string document')
    if not all(isinstance(c, int) and c > 0 for c in caps):
        raise ValueError(f'{path}: invalid caption counts')
    if limit is not None and not 1 <= limit <= expected_n:
        raise ValueError(f'--limit must be in [1, {expected_n}]')
    return names[:limit], docs[:limit], caps[:limit]


def relabel(topics, n_topics=P.N_TOPICS):
    """Sorted original id -> contiguous int32 ids; refuse outliers and wrong topic count."""
    t = np.asarray(topics)
    if t.ndim != 1 or not np.issubdtype(t.dtype, np.integer) or not len(t) or (t < 0).any():
        raise ValueError('final topics must be a nonempty integer vector without outliers')
    ids, inverse = np.unique(t, return_inverse=True)
    if len(ids) != n_topics:
        raise ValueError(f'expected {n_topics} real topics, got {len(ids)}')
    return inverse.astype(np.int32), {int(old): i for i, old in enumerate(ids)}


def reduction_count(topics, target=P.N_TOPICS, smoke=False):
    """0.17.4 subtracts self._outliers from nr_topics in _reduce_to_n_topics."""
    ids = set(int(t) for t in topics)
    real = len(ids - {-1})
    if real < 1 or (real < target and not smoke):
        raise ValueError(f'HDBSCAN found {real} real topics, fewer than {target}; stopping without fallback')
    keep = min(real, target)
    return keep + int(-1 in ids), keep


def json_params(model):
    # get_params includes class-valued defaults (e.g. CountVectorizer.dtype).
    return {k: v if (v is None or isinstance(v, (str, bool, int, list, tuple, dict))
                     or isinstance(v, float) and np.isfinite(v)) else str(v)
            for k, v in model.get_params(deep=False).items()}


def make_model(seed):
    from bertopic import BERTopic
    from umap import UMAP
    if version('bertopic') != AUDITED_VERSION:
        raise RuntimeError('BERTopic version changed; audit constructor/reduction/outlier source before running')
    model = BERTopic()
    if type(model.umap_model) is not UMAP or type(model.hdbscan_model).__module__.split('.')[0] != 'hdbscan':
        raise RuntimeError('BERTopic must use its UMAP and hdbscan package defaults')
    default = model.umap_model.get_params()
    if any(default[k] != v for k, v in UMAP_ARGS.items()):
        raise RuntimeError('installed BERTopic UMAP defaults differ from the source audit')
    model.umap_model = UMAP(**UMAP_ARGS, random_state=seed)
    source = Path(inspect.getfile(BERTopic))
    audit = dict(bertopic_source=str(source), bertopic_source_sha256=file_sha(source),
                 umap_constructor=dict(UMAP_ARGS, random_state=seed),
                 umap_arguments=json_params(model.umap_model),
                 hdbscan_arguments=json_params(model.hdbscan_model),
                 vectorizer_arguments=json_params(model.vectorizer_model),
                 ctfidf_arguments=json_params(model.ctfidf_model),
                 nr_topics_rule='nr_topics includes -1 when present; _reduce_to_n_topics uses nr_topics - self._outliers',
                 default_outlier_strategy=inspect.signature(BERTopic.reduce_outliers).parameters['strategy'].default)
    return model, audit


def check_minilm_model(model):
    """Reject SentenceTransformer's silent fallback when the cached model files are incomplete."""
    from sentence_transformers.base.modules import Normalize, Transformer
    from sentence_transformers.sentence_transformer.modules import Pooling
    modules = list(model)
    if [type(m) for m in modules] != [Transformer, Pooling, Normalize]:
        raise ValueError(f'MiniLM requires modules {MINILM_MODULES}; got {[type(m).__name__ for m in modules]}')
    if modules[1].pooling_mode != 'mean':
        raise ValueError('MiniLM requires mean-only Pooling')
    if model.max_seq_length != MINILM_MAX_SEQ_LENGTH:
        raise ValueError(f'MiniLM requires max_seq_length={MINILM_MAX_SEQ_LENGTH}; got {model.max_seq_length}')
    return dict(modules=[type(m).__name__ for m in modules], pooling_mode='mean',
                max_seq_length=model.max_seq_length)


def check_minilm_rows(emb, n):
    if (emb.ndim != 2 or emb.shape[0] != n or emb.shape[1] < 1
            or emb.dtype != np.float32 or not np.isfinite(emb).all()):
        raise ValueError('MiniLM returned invalid embeddings')
    deviation = float(np.abs(np.linalg.norm(emb.astype(np.float64), axis=1) - 1).max())
    if not deviation < MINILM_UNIT_TOL:
        raise ValueError(f'MiniLM rows must be unit norm (max deviation {deviation}, tolerance {MINILM_UNIT_TOL})')


def document_embeddings(docs, names, out, input_sha, force=False):
    """Require the complete MiniLM model, with its own Normalize module, on CPU once for all seeds."""
    path, side = out / 'minilm.npy', out / 'minilm.json'
    provenance = dict(model=P.MINILM, names_sha256=names_sha(names), input_sha256=input_sha,
                      n_docs=len(docs), versions=versions(), device='cpu', normalization='model Normalize module')
    if path.exists() and side.exists() and not force:
        meta = json.loads(side.read_text())
        # Old fallback caches have no verified module specification: rebuild them automatically.
        checked_model = (meta.get('modules') == MINILM_MODULES and meta.get('pooling_mode') == 'mean'
                         and meta.get('max_seq_length') == MINILM_MAX_SEQ_LENGTH)
        if checked_model:
            if any(meta.get(k) != v for k, v in provenance.items()) or meta['sha256'] != file_sha(path):
                raise ValueError(f'{path}: stale/corrupt embedding cache; use --force or a separate --out')
            emb = np.load(path, allow_pickle=False)
            check_minilm_rows(emb, len(docs))
            if emb.shape[1] != meta['embedding_dimension']:
                raise ValueError(f'{path}: invalid cached embedding dimension')
            return emb, meta
    from sentence_transformers import SentenceTransformer
    start = time.monotonic()
    model = SentenceTransformer(P.MINILM, device='cpu', local_files_only=True)
    model_spec = check_minilm_model(model)
    lengths = []
    for i in range(0, len(docs), 128):
        batch = model.tokenizer(docs[i:i + 128], truncation=False, padding=False,
                                return_length=True, verbose=False)
        lengths.extend(batch['length'])
    emb = model.encode(docs, device='cpu', convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    check_minilm_rows(emb, len(docs))
    atomic_write(path, lambda f: np.save(f, emb, allow_pickle=False), binary=True)
    meta = dict(provenance, **model_spec, sha256=file_sha(path), embedding_dimension=int(emb.shape[1]),
                n_docs_exceed_max_seq_length=int(np.sum(np.asarray(lengths) > model.max_seq_length)),
                token_count_rule='untruncated tokenizer length including special tokens',
                max_document_tokens=int(max(lengths)), seconds=time.monotonic() - start)
    write_json(side, meta)
    return emb, meta


def seed_done(path, names, input_sha, smoke):
    side = path.with_suffix('.json')
    if not path.exists() or not side.exists():
        return False
    meta = json.loads(side.read_text())
    if meta['npz_sha256'] != file_sha(path) or meta['input_sha256'] != input_sha or meta['smoke'] != smoke:
        raise ValueError(f'{path}: stale/corrupt completed seed; use --force')
    embeddings = path.parent / 'minilm.npy'
    if not embeddings.is_file() or meta.get('embeddings_sha256') != file_sha(embeddings):
        return False
    with np.load(path, allow_pickle=False) as z:
        if z['names'].tolist() != names:
            raise ValueError(f'{path}: names differ from input order')
        t = z['topics']
        k = meta['n_topics'] if smoke else P.N_TOPICS
        if t.dtype != np.int32 or t.shape != (len(names),) or not np.array_equal(np.unique(t), np.arange(k)):
            raise ValueError(f'{path}: invalid completed partition')
    return True


def run_seed(docs, names, emb, emb_meta, out, seed, input_sha, smoke):
    start = time.monotonic()
    model, audit = make_model(seed)
    topics, _ = model.fit_transform(docs, emb)
    fit_seconds = time.monotonic() - start
    found = len(set(topics) - {-1})
    counts = dict(after_hdbscan=topics.count(-1))
    nr_topics, target = reduction_count(topics, smoke=smoke)
    model.reduce_topics(docs, nr_topics=nr_topics)
    topics = list(model.topics_)
    if len(set(topics) - {-1}) != target:
        raise ValueError(f'reduce_topics did not leave {target} real topics')
    counts['after_reduce_topics'] = topics.count(-1)
    if -1 in topics:
        topics = model.reduce_outliers(docs, topics)
    counts['after_default_outliers'] = topics.count(-1)
    if -1 in topics:
        topics = model.reduce_outliers(docs, topics, strategy='embeddings', embeddings=emb)
    counts['after_embedding_outliers'] = topics.count(-1)
    final, mapping = relabel(topics, target)
    path = out / f'topics_seed{seed}.npz'
    atomic_write(path, lambda f: np.savez_compressed(f, names=np.asarray(names), topics=final), binary=True)
    meta = dict(audit, seed=seed, smoke=smoke, n_docs=len(names), names_sha256=names_sha(names),
                input_sha256=input_sha, embeddings_sha256=emb_meta['sha256'], versions=versions(),
                topics_found_hdbscan=found, nr_topics_passed=nr_topics, n_topics=target,
                outlier_counts=counts, original_to_final_topic=mapping,
                topic_sizes=np.bincount(final, minlength=target).tolist(),
                fit_seconds=fit_seconds, seconds=time.monotonic() - start, npz_sha256=file_sha(path))
    write_json(path.with_suffix('.json'), meta)
    print(f'seed {seed}: HDBSCAN={found}, final={target}, outliers={counts}, {meta["seconds"]:.1f}s', flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--out', type=Path, default=P.BERTOPIC_OUT, help='BERTopic output directory itself')
    ap.add_argument('--seed', type=int, choices=P.UMAP_SEEDS, help='default: all three UMAP seeds')
    a = ap.parse_args(argv)
    if a.limit is not None and a.out.resolve() == P.BERTOPIC_OUT.resolve():
        ap.error('--limit requires a separate --out to protect production partitions')
    input_path = P.INPUTS / 'held_docs.json'
    names, docs, _ = load_docs(input_path, a.limit)
    input_sha = file_sha(input_path)
    smoke = a.limit is not None
    seeds = P.UMAP_SEEDS if a.seed is None else (a.seed,)
    # Repair legacy embedding caches before deciding whether their partitions can be reused.
    emb, meta = document_embeddings(docs, names, a.out, input_sha, a.force)
    todo = []
    for seed in seeds:
        if not a.force and seed_done(a.out / f'topics_seed{seed}.npz', names, input_sha, smoke):
            print(f'skip seed {seed} (validated complete)', flush=True)
        else:
            todo.append(seed)
    if not todo:
        return 0
    for seed in todo:
        run_seed(docs, names, emb, meta, a.out, seed, input_sha, smoke)
    return 0


if __name__ == '__main__':
    sys.exit(main())
