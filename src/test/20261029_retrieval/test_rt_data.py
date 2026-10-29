"""Join contracts first, then the frozen held/validation file boundaries."""
import json
import os
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import rt_data
import rt_paths
import rs_vh


@pytest.fixture
def store_path():
    # Keep every test write in this worktree, and retain fixtures (delete nothing).
    path = rt_paths.ROUND.parents[2] / '.scratch/r11-t01-tests' / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.mark.parametrize('kind', ['index', 'ids'])
def test_name_join_preserves_caller_order_and_stored_dtype(store_path, kind):
    x = np.asarray([[3, 4], [0, 2], [5, 0]], dtype=np.float16)
    np.save(store_path / 'img.npy', x)
    if kind == 'index':
        index = dict(backbone=store_path.name, dataset=store_path.parent.name,
                     smoke=False, dtype='float16', dim=2, n_images=3,
                     images=[dict(painting=n) for n in ('b', 'c', 'a')])
        (store_path / 'index.json').write_text(json.dumps(index))
    else:
        np.savez(store_path / 'ids.npz', painting_names=np.asarray(['b', 'c', 'a']))
    actual = rt_data._store(store_path, ['a', 'b'], width=2, count=3, kind=kind)
    np.testing.assert_array_equal(actual, x[[2, 0]])
    assert actual.dtype == np.float16
    with pytest.raises(AssertionError, match='missing'):
        rt_data._store(store_path, ['absent'], width=2, count=3, kind=kind)
    with pytest.raises(AssertionError, match='duplicate requested'):
        rt_data._store(store_path, ['a', 'a'], width=2, count=3, kind=kind)
    if kind == 'index':
        index['images'][0]['painting'] = 'a'
        (store_path / 'index.json').write_text(json.dumps(index))
    else:
        np.savez(store_path / 'ids.npz', painting_names=np.asarray(['a', 'c', 'a']))
    with pytest.raises(AssertionError, match='duplicate stored'):
        rt_data._store(store_path, ['a'], width=2, count=3, kind=kind)


def test_embed_float32_normalization_and_zero_floor(monkeypatch):
    x = np.asarray([[3, 4], [0, 0]], dtype=np.float16)
    monkeypatch.setattr(rt_data, 'raw', lambda *args, **kwargs: x)
    actual = rt_data.embed('clip', 'held', 'emotion')
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.asarray([[0.6, 0.8], [0, 0]], dtype=np.float32))
    np.testing.assert_array_equal(x, [[3, 4], [0, 0]])


@pytest.mark.parametrize('split', ['held', 'val'])
def test_crl_reserved(split):
    with pytest.raises(NotImplementedError):
        rt_data.raw('crl', split, 'emotion')


@pytest.mark.parametrize('args', [('unknown', 'held', 'emotion', None),
                                 ('plain', 'train', 'emotion', None),
                                 ('plain', 'held', 'unknown', None),
                                 ('cstar', 'held', 'emotion', None),
                                 ('percept', 'held', 'emotion', 5),
                                 ('clip', 'held', 'emotion', 0)])
def test_invalid_requests_assert_before_io(args):
    with pytest.raises(AssertionError):
        rt_data.raw(*args)


@pytest.mark.parametrize('split', ['held', 'val'])
@pytest.mark.parametrize('corruption', ['hash', 'names', 'rows', 'dtype'])
def test_cstar_rejects_corrupt_boundary(store_path, monkeypatch, split, corruption):
    requested = ['a', 'b', 'c']
    monkeypatch.setattr(rt_data, 'names', lambda role: requested)
    monkeypatch.setattr(rt_data, 'CSTAR_ROOT', store_path)
    directory = store_path / ('held/k11=40' if split == 'held' else 'train/k11=40/R6-5')
    directory.mkdir(parents=True)
    path = directory / ('emotion__s0.npz' if split == 'held' else 'emotion__Gres__s0.npz')
    x = np.full((3, 2048), 1 / np.sqrt(2048), dtype=np.float16)
    if corruption == 'dtype':
        x = x.astype(np.float32)
    np.savez(path, img=x, **({'cap': x} if split == 'val' else {}))
    digest = rs_vh.names_sha(requested)
    if split == 'held':
        side = dict(rows=3, list_sha256=digest, aspect='emotion', seed=0, arm='k11=40',
                    width=2048, dtype='float16', sha256=rs_vh.array_sha(x),
                    arrays={'img': dict(shape=[3, 2048], dtype='float16', sha256=rs_vh.array_sha(x))})
        if corruption == 'hash':
            side['sha256'] = 'bad'
        elif corruption == 'names':
            side['list_sha256'] = 'bad'
        elif corruption == 'rows':
            side['rows'] = 2
    else:
        side = dict(config=dict(mode='full', aspect='emotion', seed=0, smoke=False,
                                output_rows=3, names_sha256={'dev': digest},
                                graph_row='Gres', rg={'arm': 'k11=40'}),
                    output_sha256=rs_vh.file_sha(path))
        if corruption == 'hash':
            side['output_sha256'] = 'bad'
        elif corruption == 'names':
            side['config']['names_sha256']['dev'] = 'bad'
        elif corruption == 'rows':
            side['config']['output_rows'] = 2
    path.with_suffix('.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError):
        rt_data.raw('cstar', split, 'emotion', 0)


@pytest.mark.real_data
@pytest.mark.parametrize('split,n,genre', [('held', 12281, 9949), ('val', 6152, 4960)])
def test_real_names_labels_and_caption_distributions(split, n, genre):
    ns = rt_data.names(split)
    assert isinstance(ns, list) and len(ns) == len(set(ns)) == n and ns == sorted(ns)
    if split == 'held':
        assert rs_vh.names_sha(ns) == rs_vh.HELD_SHA256
    else:
        import ct2_data
        s = ct2_data.split2('full')
        assert ns == s['all_painting_ids'][s['dev']].tolist()
    labs, dist = rt_data.labels(split), rt_data.emotion_dist(split)
    rows = json.loads(rs_vh.ANNOTATIONS.read_text())
    expected = dict(emotion=rs_vh.emotion_labels(ns, rows), style=rs_vh.style_labels(ns, rows),
                    genre=rs_vh.genre_labels(ns, rows, rs_vh.GENRE_CSVS))
    for aspect in rt_paths.ASPECTS:
        assert labs[aspect].shape == (n,) and labs[aspect].dtype == np.int64
        np.testing.assert_array_equal(labs[aspect], expected[aspect])
    assert (labs['genre'] >= 0).sum() == genre
    assert dist.shape == (n, 9) and dist.dtype == np.float64
    np.testing.assert_allclose(dist.sum(axis=1), 1, rtol=0, atol=1e-15)
    assert (dist >= 0).all()
    counts = Counter(r['emotion'] for r in rows if r['painting'] == ns[0])
    np.testing.assert_array_equal(dist[0], [counts[e] / sum(counts.values()) for e in sorted(rs_vh.EMOTIONS)])
    untied = (dist == dist.max(axis=1, keepdims=True)).sum(axis=1) == 1
    np.testing.assert_array_equal(dist.argmax(axis=1)[untied], labs['emotion'][untied])


def _source_row(arm, split, aspect, seed, painting, row):
    """Independently open a reference row at the documented source boundary."""
    pre = rt_paths.PRE / 'artelingo_full'
    if arm == 'cstar':
        root = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full')
        graphs = dict(emotion='Gres', style='Fres05', genre='R0o')
        path = (root / f'held/k11=40/{aspect}__s{seed}.npz' if split == 'held' else
                root / f'train/k11=40/R6-5/{aspect}__{graphs[aspect]}__s{seed}.npz')
        with np.load(path) as z:
            return z['img'][row]
    if arm == 'percept':
        rel = (f'held/percept/core__k40/P3__s{seed}.npz' if split == 'held' else
               f'stage2/full/percept/core/stage2/k40/P3__s{seed}.npz')
        with np.load(rt_paths.CT_OUT / rel) as z:
            return z['prob' if split == 'held' else 'eval_prob'][row]
    if arm == 'instructed':
        prompt = dict(emotion='emotion_focus', style='style_word', genre='genre_word')[aspect]
        path = pre / f'qwen3vl_emb_2b_aspect_prompts/{prompt}_{"held" if split == "held" else "s2img"}'
        with np.load(path / 'ids.npz') as z:
            stored = z['painting_names'].tolist()
    else:
        path = pre / dict(plain='qwen3vl_emb_2b', clip='clip_l14', siglip2='siglip2_so400m')[arm]
        stored = [r['painting'] for r in json.loads((path / 'index.json').read_text())['images']]
    x = np.load(path / 'img.npy', mmap_mode='r')[stored.index(painting)]
    if arm == 'plain':
        x = x.astype(np.float32)[None, :]
        x = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12))[0]
    return x


CASES = ([(arm, aspect, seed) for arm in ('cstar', 'percept') for seed in range(5)
          for aspect in (rt_paths.ASPECTS if arm == 'cstar' else ('emotion',))]
         + [('instructed', aspect, None) for aspect in rt_paths.ASPECTS]
         + [(arm, 'emotion', None) for arm in ('plain', 'clip', 'siglip2')])


@pytest.mark.real_data
@pytest.mark.parametrize('split', ['held', 'val'])
@pytest.mark.parametrize('arm,aspect,seed', CASES)
def test_real_arm_shape_dtype_and_known_row(split, arm, aspect, seed):
    ns = rt_data.names(split)
    x = rt_data.raw(arm, split, aspect, seed)
    width = dict(cstar=2048, plain=2048, instructed=2048, clip=768, siglip2=1152, percept=40)[arm]
    assert x.shape == (len(ns), width)
    assert x.dtype == (np.float32 if arm in ('plain', 'percept') else np.float16)
    assert np.isfinite(x).all()
    row = 17
    expected = _source_row(arm, split, aspect, seed, ns[row], row)
    np.testing.assert_array_equal(x[row], expected)
    if arm == 'percept':
        assert ((x >= 0) & (x <= 1)).all()
    else:
        np.testing.assert_allclose(np.linalg.norm(x.astype(np.float32), axis=1), 1, rtol=0, atol=3e-4)


@pytest.mark.real_data
def test_round_paths_and_plain_held_regression_hash():
    default = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
    assert rt_paths.OUTPUTS == Path(os.environ.get('RT_OUTPUTS', str(default)))
    assert default.is_dir() and not default.is_symlink()
    assert rt_paths.MAIN_TEST == Path('/project/CoSiR/src/test')
    x = rt_data.raw('plain', 'held', 'emotion')
    path = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held/k11=40/emotion__s0.json')
    assert rs_vh.array_sha(x) == json.loads(path.read_text())['image_sha256']


@pytest.mark.parametrize('corruption', [None, 'hash', 'names', 'revision'])
def test_instructed_held_checks_payload_hash_and_names(store_path, monkeypatch, corruption):
    ns = ['a', 'b', 'c']
    monkeypatch.setattr(rt_data, 'names', lambda role: ns)
    monkeypatch.setattr(rt_data, 'FULL_ROOT', store_path)
    monkeypatch.setattr(rt_paths, 'N_HELD', 3)
    path = store_path / 'qwen3vl_emb_2b_aspect_prompts/emotion_focus_held'
    path.mkdir(parents=True)
    x = np.eye(3, 2048, dtype=np.float16)
    np.save(path / 'img.npy', x)
    np.savez(path / 'ids.npz', painting_names=np.asarray(ns))
    import ct2_data
    side = dict(instruction=ct2_data.itp_data.PROMPTS['emotion_focus'],
                model_id='Qwen/Qwen3-VL-Embedding-2B', model_revision=ct2_data.QWEN_REVISION,
                dtype='float16', shape=[3, 2048], held_sha256=rs_vh.names_sha(ns),
                img_sha256=rs_vh.array_sha(x))
    if corruption:
        side[dict(hash='img_sha256', names='held_sha256', revision='model_revision')[corruption]] = 'bad'
    (path / 'meta.json').write_text(json.dumps(side))
    if corruption:
        with pytest.raises(AssertionError):
            rt_data.raw('instructed', 'held', 'emotion')
    else:
        np.testing.assert_array_equal(rt_data.raw('instructed', 'held', 'emotion'), x)
