"""D2 graph regression against the frozen AGS arrays; CPU only."""
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

import ct2_graphs


MAIN = Path('/project/CoSiR/src/test')
PICKS = dict(emotion='Gres', style='Fres05', genre='R0o')


@pytest.fixture(scope='module')
def sample_build():
    from itp_data import default_images, split, targets
    from ags_trainimg import aspect_images

    with np.load(MAIN / '20261017_prompted_aspect_views/sample.npz', allow_pickle=False) as z:
        s = {key: z[key] for key in z.files}
    _, fit, dev = split(s)
    spaces = dict(D=targets(s, 'default'), I=default_images(s)[0],
                  E=targets(s, 'emotion_focus'),
                  **{'style.img': aspect_images(s, 'style_word'),
                     'genre.img': aspect_images(s, 'genre_word')})
    graphs, manifest = ct2_graphs.build(spaces, fit, dev)
    return s, graphs, manifest


def test_sample_matches_saved_d2_arrays_and_trainer_config(sample_build):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '20261019_model_sheet'))
    import ms_train
    from ms_rows import ASPECTS, ROWS

    s, graphs, manifest = sample_build
    assert set(graphs) == set(PICKS)
    frozen = MAIN / '20261018_aspect_graph_sheet/graphs'
    reference = json.loads((frozen / 'manifest.json').read_text())
    for aspect, row in PICKS.items():
        graph = graphs[aspect]
        assert set(graph['rows']) == {row}
        path = frozen / f'{aspect}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['archives'][aspect]
        with np.load(path, allow_pickle=False) as z:
            for part in ('fit', 'dev'):
                np.testing.assert_array_equal(graph[f'{part}_node_ids'], z[f'{part}_node_ids'])
                pairs, weights = graph['rows'][row][part]
                np.testing.assert_array_equal(pairs, z[f'{row}__{part}'])
                np.testing.assert_array_equal(weights, z[f'{row}__{part}_w'])
                assert pairs.dtype == np.int64 and weights.dtype == np.float32
        config = ms_train.make_config(s, ROWS['D2'], 'D2', aspect, list(ASPECTS),
                                      dict.fromkeys(ASPECTS, manifest), 0, 'cpu', False, 200)
        assert config['graph_row'] == row
        assert config['fit_pair_count'] == reference['counts'][aspect]['fit'][row]
        assert config['dev_pair_count'] == reference['counts'][aspect]['dev'][row]


def small_spaces(n=104, width=12):
    rng = np.random.default_rng(3)
    spaces = {key: rng.standard_normal((n, width), dtype=np.float32) for key in ct2_graphs.KEYS}
    return {key: x / np.linalg.norm(x, axis=1, keepdims=True) for key, x in spaces.items()}


def test_manifest_records_resources_counts_and_every_array_hash():
    spaces = small_spaces()
    fit, dev = np.arange(70, dtype=np.int64), np.arange(70, 104, dtype=np.int64)
    start = time.monotonic()
    graphs, manifest = ct2_graphs.build(spaces, fit, dev)
    elapsed = time.monotonic() - start
    assert 0 < manifest['seconds'] <= elapsed
    assert set(manifest['part_seconds']) == {'fit', 'dev'}
    assert all(0 < t < manifest['seconds'] for t in manifest['part_seconds'].values())
    assert manifest['peak_rss_bytes'] > 0
    assert manifest['peak_rss_scope'] == 'process lifetime high-water mark (Linux ru_maxrss)'
    assert manifest['source_sha256'] == {
        key: hashlib.sha256(x.tobytes()).hexdigest() for key, x in spaces.items()}
    for aspect, row in PICKS.items():
        graph = graphs[aspect]
        arrays = {f'{part}_node_ids': graph[f'{part}_node_ids'] for part in ('fit', 'dev')}
        for part in ('fit', 'dev'):
            pairs, weights = graph['rows'][row][part]
            arrays[f'{row}__{part}'], arrays[f'{row}__{part}_w'] = pairs, weights
            assert manifest['counts'][aspect][part] == {row: len(pairs)}
        assert manifest['array_sha256'][aspect] == {
            key: hashlib.sha256(x.tobytes()).hexdigest() for key, x in arrays.items()}


def test_build_fits_a_budget_smaller_than_a_dense_matrix():
    # The subprocess gets 192 MiB beyond its warmed-up virtual size. A float32
    # 12,000² matrix alone needs 549 MiB; pav's boolean adjacency temporaries also
    # exceed the budget. Features are narrow so this isolates graph working RAM.
    script = textwrap.dedent('''
        import json, os, resource
        import numpy as np
        import ct2_graphs
        rng = np.random.default_rng(17)
        def inputs(n):
            return {key: rng.standard_normal((n, 8), dtype=np.float32)
                    for key in ct2_graphs.KEYS}
        ct2_graphs.build(inputs(64), np.arange(32), np.arange(32, 64))
        spaces = inputs(12256)
        virtual = int(open('/proc/self/statm').read().split()[0]) * os.sysconf('SC_PAGE_SIZE')
        resource.setrlimit(resource.RLIMIT_AS, (virtual + 192 * 2**20,) * 2)
        graphs, manifest = ct2_graphs.build(spaces, np.arange(12000), np.arange(12000, 12256))
        assert all(len(graphs[a]['rows'][r]['fit'][0]) for a, r in ct2_graphs.ROWS.items())
        print(json.dumps({'seconds': manifest['seconds'], 'peak': manifest['peak_rss_bytes']}))
    ''')
    env = dict(os.environ, PYTHONPATH=str(Path(ct2_graphs.__file__).parent),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               CUDA_VISIBLE_DEVICES='')
    run = subprocess.run([sys.executable, '-c', script], env=env, text=True,
                         capture_output=True, timeout=160)
    assert run.returncode == 0, run.stdout + run.stderr
    metrics = json.loads(run.stdout)
    assert metrics['seconds'] > 0 and metrics['peak'] > 0


def test_saved_sample_loads_through_the_sheet_graph_loader(sample_build, tmp_path):
    from ags_head import load_graph
    from itp_data import fingerprint

    s, graphs, manifest = sample_build
    manifest = dict(manifest, identity=dict(manifest['identity'], sample_sha256=fingerprint(s)))
    out = tmp_path / 'graphs'
    ct2_graphs.save(graphs, manifest, out)
    for aspect, row in PICKS.items():
        loaded, metadata = load_graph(s, tmp_path, aspect, [row])
        assert metadata == manifest
        for part in ('fit', 'dev'):
            np.testing.assert_array_equal(loaded[f'{part}_node_ids'], graphs[aspect][f'{part}_node_ids'])
            for actual, expected in zip(loaded['rows'][row][part], graphs[aspect]['rows'][row][part]):
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('mode', ['sample', 'full'])
def test_cli_uses_mode_providers_and_seals_selected_npzs(mode, tmp_path):
    # ct2_data is ticket 01's file. Exercise that external provider contract in a
    # fresh process without reading or writing any real full-scale feature store.
    provider = textwrap.dedent('''
        import numpy as np
        def split2(mode):
            assert mode in ('sample', 'full')
            nfit = 40 if mode == 'sample' else 48
            n = nfit + 24
            return dict(mode=mode, fit=np.arange(nfit), dev=np.arange(nfit, n),
                        first=np.arange(n), pid=np.arange(n), sample_ids=np.arange(n),
                        all_painting_ids=np.array([f'painting-{i}' for i in range(n)]),
                        subset_pid=np.array([], dtype=np.int64),
                        subset_image_indices=np.array([], dtype=np.int64),
                        names_sha256={'fit': 'fit-names-hash', 'dev': 'dev-names-hash'})
        def space(s, seed):
            x = np.random.default_rng(seed).standard_normal((len(s['first']), 12), dtype=np.float32)
            return x / np.linalg.norm(x, axis=1, keepdims=True)
        def default_images(s):
            return space(s, 1)
        def default_captions(s):
            return space(s, 2)
        def instructed_images(s, key):
            assert key in ('style_word', 'genre_word')
            return space(s, 3 if key == 'style_word' else 4)
        def instructed_captions(s, key):
            assert key == 'emotion_focus'
            return space(s, 5)
    ''')
    (tmp_path / 'ct2_data.py').write_text(provider)
    out = tmp_path / 'graphs'
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), str(Path(ct2_graphs.__file__).parent))),
               PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='')
    run = subprocess.run([sys.executable, '-m', 'ct2_graphs', '--mode', mode, '--out', str(out)],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert {p.name for p in out.iterdir()} == {'manifest.json', 'emotion.npz', 'style.npz', 'genre.npz'}
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['identity']['mode'] == mode
    assert manifest['identity']['names_sha256'] == {'fit': 'fit-names-hash', 'dev': 'dev-names-hash'}
    assert manifest['seconds'] > 0 and manifest['peak_rss_bytes'] > 0
    nfit = 40 if mode == 'sample' else 48
    # Load the test provider as plain data to derive expected source identities.
    providers = {}
    exec(provider, providers)
    s = providers['split2'](mode)
    spaces = dict(D=providers['default_captions'](s), I=providers['default_images'](s),
                  E=providers['instructed_captions'](s, 'emotion_focus'),
                  **{'style.img': providers['instructed_images'](s, 'style_word'),
                     'genre.img': providers['instructed_images'](s, 'genre_word')})
    expected, _ = ct2_graphs.build(spaces, s['fit'], s['dev'])
    for aspect, row in PICKS.items():
        path = out / f'{aspect}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['archives'][aspect]
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {'fit_node_ids', 'dev_node_ids'} | {
                f'{row}__{part}{suffix}' for part in ('fit', 'dev') for suffix in ('', '_w')}
            np.testing.assert_array_equal(z['fit_node_ids'], np.arange(nfit))
            np.testing.assert_array_equal(z['dev_node_ids'], np.arange(nfit, nfit + 24))
            for part in ('fit', 'dev'):
                for key, value in zip((f'{row}__{part}', f'{row}__{part}_w'),
                                      expected[aspect]['rows'][row][part]):
                    np.testing.assert_array_equal(z[key], value)
            assert manifest['array_sha256'][aspect] == {
                key: hashlib.sha256(z[key].tobytes()).hexdigest() for key in z.files}
