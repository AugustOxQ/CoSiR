"""Frozen r11 data in sorted split order; all joins and positional hashes are checked.

`raw` preserves r10's representation (including normalized float32 plain Qwen).
`embed` L2 normalizes once, preserving the plain loader's existing float32 normalization.
Arrays are loaded per call rather than cached across arms/seeds. CRL cosines and their normalization use float64,
then cast once to float32. See "Numeric convention" in 20261029_retrieval_log.md.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import rt_paths
import ct2_data
import rb_encoders
import rs_vh

ARMS = ('cstar', 'plain', 'instructed', 'clip', 'siglip2', 'percept', 'crl')
SEEDED = {'cstar', 'percept'}
ASPECT_FREE = {'plain', 'clip', 'siglip2', 'percept'}
CSTAR_ROOT = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full')
FULL_ROOT = rt_paths.PRE / 'artelingo_full'
PROMPTS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
GRAPHS = dict(emotion='Gres', style='Fres05', genre='R0o')
WIDTHS = dict(cstar=2048, plain=2048, instructed=2048, clip=768, siglip2=1152, percept=40)


def _json(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=2)
def _names(split):
    assert split in ('held', 'val'), split
    if split == 'held':
        doc = _json(rt_paths.CT_OUT / 'held/held_paintings.json')
        assert set(doc) == {'names', 'sha256'}
        ns = doc['names']
        assert rs_vh.names_sha(ns) == doc['sha256'] == rs_vh.HELD_SHA256, 'held names hash'
        count = rt_paths.N_HELD
    else:
        s = ct2_data.split2('full')
        ns = s['all_painting_ids'][s['dev']].tolist()
        assert rs_vh.names_sha(ns) == s['names_sha256']['dev'] == ct2_data.FULL_NAMES_SHA256[1]
        count = rt_paths.N_VAL
    assert len(ns) == len(set(ns)) == count and ns == sorted(ns), 'split count/order'
    assert all(isinstance(n, str) and n for n in ns)
    return tuple(ns)


def names(split):
    """Sorted, unique held (12,281) or validation (6,152) painting names."""
    return list(_names(split))


@lru_cache(maxsize=1)
def _annotations():
    rows = _json(rs_vh.ANNOTATIONS)
    assert isinstance(rows, list) and len(rows) == 308723
    assert {r['emotion'] for r in rows} == set(rs_vh.EMOTIONS)
    return rows


@lru_cache(maxsize=2)
def _labels(split):
    ns, rows = names(split), _annotations()
    result = dict(emotion=rs_vh.emotion_labels(ns, rows), style=rs_vh.style_labels(ns, rows),
                  genre=rs_vh.genre_labels(ns, rows, rs_vh.GENRE_CSVS))
    for aspect, x in result.items():
        assert x.shape == (len(ns),) and x.dtype == np.int64
        upper = dict(emotion=9, style=27, genre=10)[aspect]
        assert ((x >= (-1 if aspect == 'genre' else 0)) & (x < upper)).all()
    assert int((result['genre'] >= 0).sum()) == rt_paths.N_GENRE[split], 'labelled genre count'
    return result


def labels(split):
    """r10's sorted-name emotion/style codes; genre -1 marks unlabelled rows."""
    return {a: x.copy() for a, x in _labels(split).items()}


@lru_cache(maxsize=2)
def _emotion_dist(split):
    ns = names(split)
    lookup = {n: i for i, n in enumerate(ns)}
    codes = {e: i for i, e in enumerate(sorted(rs_vh.EMOTIONS))}
    counts = np.zeros((len(ns), 9), dtype=np.int64)
    for r in _annotations():
        if r['painting'] in lookup:
            counts[lookup[r['painting']], codes[r['emotion']]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    assert (totals > 0).all(), 'painting without captions'
    dist = counts / totals
    assert dist.shape == (len(ns), 9) and dist.dtype == np.float64
    assert np.allclose(dist.sum(axis=1), 1, rtol=0, atol=1e-15), 'emotion distribution sums'
    untied = (counts == counts.max(axis=1, keepdims=True)).sum(axis=1) == 1
    assert np.array_equal(dist.argmax(axis=1)[untied], _labels(split)['emotion'][untied])
    return dist


def emotion_dist(split):
    """All-caption emotion shares, float64 [n, 9], columns sorted by emotion name."""
    return _emotion_dist(split).copy()


def _gather(stored, requested):
    assert len(set(stored)) == len(stored), 'duplicate stored painting names'
    assert len(set(requested)) == len(requested), 'duplicate requested painting names'
    lookup = {name: i for i, name in enumerate(stored)}
    missing = [n for n in requested if n not in lookup]
    assert not missing, f'missing painting names: {missing[:3]}'
    rows = np.asarray([lookup[n] for n in requested], dtype=np.int64)
    assert [stored[i] for i in rows] == list(requested), 'joined name order'
    return rows


def _store(path, requested, *, width, count, kind):
    """A raw float16 name-keyed store, gathered in exactly the caller's order."""
    path = Path(path)
    assert kind in ('index', 'ids')
    if kind == 'index':
        index = _json(path / 'index.json')
        assert index['backbone'] == path.name and index['dataset'] == path.parent.name
        assert not index['smoke'] and index['dtype'] == 'float16' and index['dim'] == width
        assert index['n_images'] == count
        stored = [r['painting'] for r in index['images']]
    else:
        with np.load(path / 'ids.npz', allow_pickle=False) as z:
            ids = z['painting_names']
            assert ids.ndim == 1 and ids.dtype.kind == 'U', 'painting_names dtype/shape'
            stored = ids.tolist()
    assert len(stored) == count, (path, len(stored), count)
    assert all(isinstance(n, str) and n for n in stored)
    x = np.load(path / 'img.npy', mmap_mode='r', allow_pickle=False)
    assert x.shape == (count, width) and x.dtype == np.float16, (path, x.shape, x.dtype)
    out = x[_gather(stored, requested)]
    assert out.shape == (len(requested), width) and out.dtype == np.float16
    assert np.isfinite(out).all(), (path, 'nonfinite features')
    return out


def _view(x, count, width, dtype):
    assert x.shape == (count, width) and x.dtype == dtype, (x.shape, x.dtype)
    assert np.isfinite(x).all(), 'nonfinite features'
    return x


def _cstar(split, aspect, seed, ns):
    if split == 'held':
        path = CSTAR_ROOT / f'held/k11=40/{aspect}__s{seed}.npz'
    else:
        path = CSTAR_ROOT / f'train/k11=40/R6-5/{aspect}__{GRAPHS[aspect]}__s{seed}.npz'
    side = _json(path.with_suffix('.json'))
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == ({'img'} if split == 'held' else {'img', 'cap'}), path
        x = _view(z['img'], len(ns), 2048, np.float16)
        if split == 'val':
            _view(z['cap'], len(ns), 2048, np.float16)
    digest = rs_vh.names_sha(ns)
    if split == 'held':
        rs_vh.check_sidecar(side, x, aspect, seed, len(ns), digest, 'k11=40')
        assert side['width'] == 2048 and side['dtype'] == 'float16'
        assert side['arrays']['img'] == dict(shape=list(x.shape), dtype='float16', sha256=rs_vh.array_sha(x))
    else:
        config = side['config']
        assert config['mode'] == 'full' and not config['smoke']
        assert config['aspect'] == aspect and config['seed'] == seed
        assert config['graph_row'] == GRAPHS[aspect] and config['rg']['arm'] == 'k11=40'
        assert config['output_rows'] == len(ns) and config['names_sha256']['dev'] == digest
        assert side['output_sha256'] == rs_vh.file_sha(path), 'validation output file hash'
    assert np.max(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1)) < 1e-3
    return x


def _instructed(split, aspect, ns):
    prompt = PROMPTS[aspect]
    path = FULL_ROOT / f'qwen3vl_emb_2b_aspect_prompts/{prompt}_{"held" if split == "held" else "s2img"}'
    manifest = _json(path / ('meta.json' if split == 'held' else 'done.json'))
    assert manifest['instruction'] == ct2_data.itp_data.PROMPTS[prompt], 'instruction mismatch'
    assert manifest['model_id'] == 'Qwen/Qwen3-VL-Embedding-2B'
    assert manifest['model_revision'] == ct2_data.QWEN_REVISION and manifest['dtype'] == 'float16'
    with np.load(path / 'ids.npz', allow_pickle=False) as z:
        stored = z['painting_names']
        assert stored.ndim == 1 and stored.dtype.kind == 'U'
    count = rt_paths.N_HELD if split == 'held' else 29232
    assert manifest['shape'] == [count, 2048]
    if split == 'held':
        assert rs_vh.names_sha(stored.tolist()) == manifest['held_sha256'] == rs_vh.names_sha(ns)
    else:
        assert not manifest['smoke'] and manifest['parity_gate']['passed'] is True
        assert manifest['count'] == manifest['n_images'] == count
        assert manifest['ids_sha256'] == ct2_data.ids_sha256({'painting_names': stored})
    x = _store(path, ns, width=2048, count=count, kind='ids')
    if split == 'held':
        assert manifest['img_sha256'] == rs_vh.array_sha(x), 'held instructed image array hash'
    return x


def _percept(split, seed, ns):
    # Amendment 1: both splits use the held comparator's sample-scale mapper.
    held = rt_paths.CT_OUT / f'held/percept/core__k40/P3__s{seed}.npz'
    path = held if split == 'held' else rt_paths.OUTPUTS / f'percept_val/P3__s{seed}.npz'
    if split == 'val' and (not path.is_file() or not path.with_suffix('.json').is_file()):
        raise FileNotFoundError(f'Missing PercepT validation export {path}; run rt_percept_val.py')
    side = _json(path.with_suffix('.json'))
    source = side if split == 'held' else _json(held.with_suffix('.json'))
    config = source['config']
    assert (config['settings'], config['cut'], config['arm'], config['seed']) == ('core', 'k40', 'P3', seed)
    assert not config['smoke'] and source['K'] == 40
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'prob'}, path
        x = _view(z['prob'], len(ns), 40, np.float32)
    assert side['rows'] == len(ns), 'PercepT row count'
    if split == 'held':
        assert side['held_sha256'] == rs_vh.names_sha(ns)
    else:
        assert side['val_sha256'] == rs_vh.names_sha(ns), 'PercepT validation names hash'
        assert side['weights_sha256'] == source['weights_sha256'], 'PercepT held mapper weights hash'
        assert side['device'] == source['inference_device'] == config['device'] == 'cuda', 'PercepT device'
        assert side['seed'] == seed, 'PercepT seed'
    assert side['array_sha256s'] == {'prob': rs_vh.array_sha(x)}, 'PercepT array hash'
    assert ((x >= 0) & (x <= 1)).all(), 'PercepT probability range'
    return x


def raw(arm, split, aspect, seed=None):
    """Stored representation in `names(split)` order; seeded arms require seed 0..4."""
    assert arm in ARMS and split in ('held', 'val') and aspect in rt_paths.ASPECTS, (arm, split, aspect)
    assert (seed in rt_paths.SEEDS if arm in SEEDED else seed is None), (arm, seed)
    ns = names(split)
    if arm == 'crl':
        import rt_crl
        text = rt_crl.text_array(aspect)
        x = embed('clip', split, aspect).astype(np.float64) @ text.astype(np.float64).T
        assert x.shape == (len(ns), text.shape[0]) and x.dtype == np.float64
        return _view(x, len(ns), text.shape[0], np.float64)
    if arm == 'cstar':
        x = _cstar(split, aspect, seed, ns)
    elif arm == 'instructed':
        x = _instructed(split, aspect, ns)
    elif arm == 'percept':
        x = _percept(split, seed, ns)
    else:
        backbone = dict(plain='qwen3vl_emb_2b', clip='clip_l14', siglip2='siglip2_so400m')[arm]
        x = _store(FULL_ROOT / backbone, ns, width=WIDTHS[arm], count=61402, kind='index')
        if arm == 'plain':
            # Keep the frozen CT normalization/gate path, including its exact bytes.
            x = rb_encoders.full_images(backbone, ns)
    return _view(x, len(ns), WIDTHS[arm], np.float32 if arm in ('plain', 'percept') else np.float16)


def embed(arm, split, aspect, seed=None):
    """Unit float32 rows; see the round log's Numeric convention (CRL normalizes before its single cast)."""
    x = raw(arm, split, aspect, seed)
    if arm == 'plain':
        return x  # raw preserves the frozen loader's already normalized float32 rows (stage 0b's hash).
    x = x.astype(np.float64 if arm == 'crl' else np.float32)
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
