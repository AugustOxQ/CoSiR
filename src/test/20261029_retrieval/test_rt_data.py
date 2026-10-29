"""Join contracts, frozen file boundaries, and the round log's Numeric convention."""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import rt_data
import rt_paths
import rs_vh


@pytest.fixture
def store_path(tmp_path):
    return tmp_path


@pytest.fixture
def percept_source(tmp_path, monkeypatch):
    ns = ['a', 'b', 'c']
    monkeypatch.setattr(rt_paths, 'CT_OUT', tmp_path / 'frozen')
    monkeypatch.setattr(rt_paths, 'OUTPUTS', tmp_path / 'outputs')
    monkeypatch.setattr(rt_paths, 'N_VAL', 3)
    monkeypatch.setattr(rt_data, 'names', lambda split: ns if split == 'val' else pytest.fail(split))
    for seed in rt_paths.SEEDS:
        weights = rt_paths.CT_OUT / f'percept/core/stage2/k40/P3__s{seed}.pt'
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(f'sample mapper seed {seed}'.encode())
        config = dict(settings='core', cut='k40', arm='P3', seed=seed, smoke=False, device='cuda')
        source = dict(config=config, K=40, weights_sha256=rs_vh.file_sha(weights))
        weights.with_suffix('.json').write_text(json.dumps(source))
        held = rt_paths.CT_OUT / f'held/percept/core__k40/P3__s{seed}.json'
        held.parent.mkdir(parents=True, exist_ok=True)
        held.write_text(json.dumps(dict(source, rows=12281, held_sha256=rs_vh.HELD_SHA256,
                                        inference_device='cuda')))
    return np.full((3, 40), 0.25, dtype=np.float32)


@pytest.mark.parametrize('corruption', [None, 'rows', 'names', 'weights', 'hash', 'device', 'dtype', 'shape'])
def test_percept_validation_checks_exported_file(percept_source, corruption):
    x = percept_source
    path = rt_paths.OUTPUTS / 'percept_val/P3__s0.npz'
    path.parent.mkdir(parents=True)
    weights = rt_paths.CT_OUT / 'percept/core/stage2/k40/P3__s0.pt'
    side = dict(rows=3, val_sha256=rs_vh.names_sha(['a', 'b', 'c']),
                weights_sha256=rs_vh.file_sha(weights), array_sha256s={'prob': rs_vh.array_sha(x)},
                device='cuda', seed=0)
    if corruption in ('rows', 'names', 'weights', 'device'):
        side[dict(rows='rows', names='val_sha256', weights='weights_sha256', device='device')[corruption]] = 'bad'
    elif corruption == 'hash':
        side['array_sha256s']['prob'] = 'bad'
    elif corruption == 'dtype':
        x = x.astype(np.float16)
        side['array_sha256s']['prob'] = rs_vh.array_sha(x)
    elif corruption == 'shape':
        x = x[:, :39]
        side['array_sha256s']['prob'] = rs_vh.array_sha(x)
    np.savez(path, prob=x)
    path.with_suffix('.json').write_text(json.dumps(side))
    if corruption:
        with pytest.raises(AssertionError):
            rt_data.raw('percept', 'val', 'emotion', 0)
    else:
        np.testing.assert_array_equal(rt_data.raw('percept', 'val', 'emotion', 0), x)


def test_missing_percept_validation_names_export_command(percept_source):
    with pytest.raises(FileNotFoundError, match='rt_percept_val.py'):
        rt_data.raw('percept', 'val', 'emotion', 0)


def test_percept_export_cli_uses_held_mapper_and_preserves_matching_files(percept_source, monkeypatch):
    import ct_percept
    import rt_percept_val
    calls = []
    original_root = ct_percept.OUTPUTS

    def infer(settings, cut, arm, seed, ns):
        assert ct_percept.OUTPUTS == rt_paths.CT_OUT, 'must read frozen inputs from main checkout'
        calls.append((settings, cut, arm, seed, ns))
        return percept_source.copy()

    monkeypatch.setattr(ct_percept, 'infer_stage2', infer)
    assert rt_percept_val.main([]) == 0
    assert calls == [('core', 'k40', 'P3', seed, ['a', 'b', 'c']) for seed in range(5)]
    assert ct_percept.OUTPUTS == original_root
    for seed in range(5):
        np.testing.assert_array_equal(rt_data.raw('percept', 'val', 'emotion', seed), percept_source)
        path = rt_paths.OUTPUTS / f'percept_val/P3__s{seed}.json'
        side = json.loads(path.read_text())
        held = rt_paths.CT_OUT / f'held/percept/core__k40/P3__s{seed}.json'
        assert side == dict(rows=3, val_sha256=rs_vh.names_sha(['a', 'b', 'c']), seed=seed,
                            weights_sha256=json.loads(held.read_text())['weights_sha256'],
                            array_sha256s={'prob': rs_vh.array_sha(percept_source)}, device='cuda')
    directory = rt_paths.OUTPUTS / 'percept_val'
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    assert rt_percept_val.main([]) == 0
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()} == before


def test_percept_export_rejects_wrong_weights_before_inference(percept_source, monkeypatch):
    import ct_percept
    import rt_percept_val
    path = rt_paths.CT_OUT / 'percept/core/stage2/k40/P3__s0.pt'
    path.write_bytes(b'wrong full-scale weights')
    monkeypatch.setattr(ct_percept, 'infer_stage2', lambda *a, **k: pytest.fail('must check weights first'))
    with pytest.raises(AssertionError, match='weights'):
        rt_percept_val.export(0)
    assert not rt_paths.OUTPUTS.exists()


@pytest.mark.parametrize('existing', ['array', 'sidecar'])
def test_percept_export_refuses_different_existing_content(percept_source, monkeypatch, existing):
    import ct_percept
    import rt_percept_val
    monkeypatch.setattr(ct_percept, 'infer_stage2', lambda *args: percept_source.copy())
    path = rt_paths.OUTPUTS / 'percept_val/P3__s0.npz'
    path.parent.mkdir(parents=True)
    if existing == 'array':
        np.savez(path, prob=np.full((3, 40), 0.75, dtype=np.float32))
    else:
        path.with_suffix('.json').write_text(json.dumps({'weights_sha256': 'wrong mapper'}))
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.parent.iterdir()}
    with pytest.raises(FileExistsError, match='different content'):
        rt_percept_val.export(0)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.parent.iterdir()} == before


@pytest.mark.parametrize('invalid', ['dtype', 'shape', 'nonfinite', 'probability'])
def test_percept_export_checks_inference_boundary(percept_source, monkeypatch, invalid):
    import ct_percept
    import rt_percept_val
    x = percept_source.copy()
    if invalid == 'dtype':
        x = x.astype(np.float16)
    elif invalid == 'shape':
        x = x[:, :39]
    elif invalid == 'nonfinite':
        x[0, 0] = np.nan
    else:
        x[0, 0] = 1.01
    monkeypatch.setattr(ct_percept, 'infer_stage2', lambda *args: x)
    with pytest.raises(AssertionError):
        rt_percept_val.export(0)
    assert not rt_paths.OUTPUTS.exists()


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


def test_plain_embed_preserves_the_frozen_loaders_single_normalization(monkeypatch):
    x = np.random.default_rng(23).normal(size=(12, 17)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    monkeypatch.setattr(rt_data, 'raw', lambda *args: x)
    np.testing.assert_array_equal(rt_data.embed('plain', 'val', 'emotion'), x)


def test_crl_cosines_and_normalization_stay_float64_until_unit_rows(monkeypatch):
    import rt_crl
    rng = np.random.default_rng(24)
    image = rng.normal(size=(12, 768)).astype(np.float32)
    text = rng.normal(size=(50, 768)).astype(np.float32)
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    embed = rt_data.embed
    monkeypatch.setattr(rt_data, 'names', lambda split: list(map(str, range(12))))
    monkeypatch.setattr(rt_data, 'embed', lambda *args: image)
    monkeypatch.setattr(rt_crl, 'text_array', lambda aspect: text)
    want = image.astype(np.float64) @ text.astype(np.float64).T
    raw = rt_data.raw('crl', 'val', 'emotion')
    assert raw.dtype == np.float64
    np.testing.assert_array_equal(raw, want)
    monkeypatch.setattr(rt_data, 'raw', lambda *args: raw)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    np.testing.assert_array_equal(embed('crl', 'val', 'emotion'), want.astype(np.float32))


@pytest.mark.parametrize('split', ['held', 'val'])
def test_crl_needs_its_text_arrays(split, tmp_path, monkeypatch):
    monkeypatch.setattr(rt_data.rt_paths, 'OUTPUTS', tmp_path)
    with pytest.raises(FileNotFoundError):
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
        path = (rt_paths.CT_OUT / f'held/percept/core__k40/P3__s{seed}.npz' if split == 'held' else
                rt_paths.OUTPUTS / f'percept_val/P3__s{seed}.npz')
        with np.load(path) as z:
            return z['prob'][row]
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
    if arm == 'percept' and split == 'val':
        path = rt_paths.OUTPUTS / f'percept_val/P3__s{seed}.npz'
        if not path.is_file() or not path.with_suffix('.json').is_file():
            pytest.skip('Run rt_percept_val.py to export validation from the held mapper')
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
@pytest.mark.parametrize('seed', range(5))
def test_real_percept_validation_weights_match_held_mapper(seed):
    weights = rt_paths.CT_OUT / f'percept/core/stage2/k40/P3__s{seed}.pt'
    original = json.loads(weights.with_suffix('.json').read_text())
    held = json.loads((rt_paths.CT_OUT / f'held/percept/core__k40/P3__s{seed}.json').read_text())
    assert rs_vh.file_sha(weights) == original['weights_sha256'] == held['weights_sha256']
    assert original['config']['fit_paintings'] == held['config']['fit_paintings'] == 13400
    assert original['config']['device'] == held['inference_device'] == 'cuda'


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
