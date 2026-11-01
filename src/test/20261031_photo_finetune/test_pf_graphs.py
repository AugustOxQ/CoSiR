"""Emotion graph parity, archive identity, and the controller's r10 replay."""
import hashlib
import json

import numpy as np
import pytest

import pf_paths as P  # establishes the frozen import paths first
import rs_graphs
import pf_graphs


def spaces(n, seed=7, dim=13):
    rng = np.random.default_rng(seed)
    return {key: rng.normal(size=(n, dim)).astype(np.float32)
            for key in rs_graphs.ct2_graphs.KEYS}


@pytest.mark.parametrize('n', [37, 273])
@pytest.mark.parametrize('k', [3, 5])
@pytest.mark.parametrize('correction', [True, False])
def test_emotion_pairs_match_frozen_builder(n, k, correction):
    # Noncontiguous, permuted subsets catch accidental use of global endpoints.
    x = spaces(n + 11)
    nodes = np.random.default_rng(21).permutation(n + 11)[:n]
    if correction:
        expected = rs_graphs._partition_aspect(x, nodes, k)['emotion']
        r0 = rs_graphs._mutual(rs_graphs.ags.l2(x['E'][nodes]), k)
        # The k=3 fixtures exercise the frozen zero-degree lambda fallback.
        if k == 3:
            assert np.any(np.bincount(rs_graphs.ags.pair_array(r0).ravel(), minlength=n) == 0)
    else:
        expected = rs_graphs._partition_k(x, nodes, k=k, variant='uncorrected')['emotion']
    actual = pf_graphs.emotion_pairs(*(x[key][nodes] for key in ('E', 'D', 'I')), k, correction)
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, rs_graphs.ags.pair_array(expected))


def test_emotion_pairs_zero_degree_lambda_fallback_matches_frozen_builder():
    x = spaces(2000, seed=3, dim=13)
    nodes = np.arange(2000)
    r0 = rs_graphs._mutual(rs_graphs.ags.l2(x['E'][nodes]), 3)
    degree = np.bincount(rs_graphs.ags.pair_array(r0).ravel(), minlength=len(nodes))
    assert np.any(degree == 0)
    expected = rs_graphs._partition_aspect(x, nodes, 3)['emotion']
    actual = pf_graphs.emotion_pairs(x['E'], x['D'], x['I'], 3, True)
    np.testing.assert_array_equal(actual, rs_graphs.ags.pair_array(expected))


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    monkeypatch.setenv('PF_OUTPUTS', str(tmp_path))
    monkeypatch.setattr(P, 'OUTPUTS', tmp_path)
    return tmp_path


def photos(n, prefix, seed):
    x = spaces(n, seed)
    return dict(keys=[f'{prefix}/{i:04d}' for i in range(n)], **{key: x[key] for key in ('E', 'D', 'I')})


@pytest.mark.parametrize('correction', [True, False])
def test_build_and_load_independent_fit_and_validation(output_root, correction):
    fit, val = photos(37, 'train', 7), photos(19, 'val', 11)
    path = pf_graphs.build(3, correction, fit, val)
    assert path == output_root / 'graphs' / f'k3__{"on" if correction else "off"}'
    assert {p.name for p in path.iterdir()} == {'emotion.npz', 'manifest.json'}
    graph = pf_graphs.load(3, correction, fit['keys'], val['keys'])
    with np.load(path / 'emotion.npz', allow_pickle=False) as archive:
        assert set(archive.files) == {'fit_pairs', 'dev_pairs'}
        for side, data in (('fit', fit), ('dev', val)):
            expected = rs_graphs._partition_k(spaces(len(data['keys']), 7 if side == 'fit' else 11),
                np.arange(len(data['keys'])), k=3, variant='aspect' if correction else 'uncorrected')['emotion']
            np.testing.assert_array_equal(graph[f'{side}_pairs'], rs_graphs.ags.pair_array(expected))
            np.testing.assert_array_equal(archive[f'{side}_pairs'], graph[f'{side}_pairs'])
    manifest = json.loads((path / 'manifest.json').read_text())
    assert graph['manifest'] == manifest and graph['path'] == path
    assert graph['manifest_sha256'] == rs_graphs.ags.sha_file(path / 'manifest.json')
    assert manifest['k'] == 3 and manifest['correction'] == correction
    assert manifest['fit_keys_sha256'] == hashlib.sha256('\n'.join(fit['keys']).encode()).hexdigest()
    assert manifest['val_keys_sha256'] == hashlib.sha256('\n'.join(val['keys']).encode()).hexdigest()
    assert manifest['counts'] == {side: len(graph[f'{side}_pairs']) for side in ('fit', 'dev')}
    assert manifest['pairs_sha256'] == rs_graphs.ags.sha_file(path / 'emotion.npz')
    assert len(manifest['code_sha256']) == 64 and manifest['seconds'] > 0
    for fit_keys, val_keys in ((fit['keys'][::-1], val['keys']), (fit['keys'], val['keys'][::-1])):
        with pytest.raises(AssertionError, match='keys hash'):
            pf_graphs.load(3, correction, fit_keys, val_keys)


def test_load_survives_unrelated_paths_file_edit(output_root, monkeypatch):
    fit, val = photos(17, 'train', 7), photos(13, 'val', 11)
    folder = pf_graphs.build(3, False, fit, val)
    paths_file = output_root / 'pf_paths.py'
    paths_file.write_text('# Later edits to constants or the sweep grid.\n')
    monkeypatch.setattr(P, '__file__', str(paths_file))
    graph = pf_graphs.load(3, False, fit['keys'], val['keys'])
    with np.load(folder / 'emotion.npz', allow_pickle=False) as archive:
        for key in ('fit_pairs', 'dev_pairs'):
            np.testing.assert_array_equal(graph[key], archive[key])


@pytest.mark.parametrize('mutation,reason', [('archive', 'pairs hash'), ('code', 'code hash'), ('count', 'count mismatch')])
def test_load_refuses_damaged_graph(output_root, mutation, reason):
    fit, val = photos(17, 'train', 7), photos(13, 'val', 11)
    folder = pf_graphs.build(3, False, fit, val)
    marker = folder / 'manifest.json'
    manifest = json.loads(marker.read_text())
    if mutation == 'archive':
        with (folder / 'emotion.npz').open('ab') as stream:
            stream.write(b'corrupt')
    elif mutation == 'code':
        manifest['code_sha256'] = 'wrong'
    else:
        manifest['counts']['fit'] += 1
    marker.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match=reason):
        pf_graphs.load(3, False, fit['keys'], val['keys'])


def replay_source(n, fit_n, dev_n):
    x = spaces(n)
    order = np.random.default_rng(17).permutation(n).astype(np.int64)
    names = np.array([f'painting_{i}' for i in range(n)])
    fit, dev, evaluation = order[:fit_n], order[fit_n:fit_n + dev_n], order[fit_n + dev_n:]
    s = dict(fit=fit, dev=dev, names_sha256={role: rs_graphs.ct2_data._names_sha256(names[ids])
             for role, ids in (('fit', fit), ('dev', dev), ('eval', evaluation))},
             sample_ids=np.arange(n), pid=np.arange(n), first=np.arange(n), all_painting_ids=names,
             subset_pid=evaluation, subset_image_indices=evaluation.copy())
    return s, x


@pytest.mark.parametrize('mutation', [None, 'edges', 'dtype', 'nodes', 'source', 'identity', 'archive'])
def test_gate_iv_compares_frozen_arrays_and_provenance(output_root, mutation):
    sources = {'full': replay_source(111, 57, 47), 'sample': replay_source(61, 37, 17)}
    r10 = output_root / 'fake_r10'
    for mode, k, correction in (('full', 40, True), ('full', 40, False), ('sample', 10, True)):
        s, x = sources[mode]
        row = 'Gres' if correction else 'R0'
        folder = r10 / mode / f'{"aspect" if correction else "uncorrected"}__k{k}' / 'd2/graphs'
        folder.mkdir(parents=True)
        arrays = {}
        for side in ('fit', 'dev'):
            pairs = rs_graphs._partition_k(x, s[side], k=k,
                variant='aspect' if correction else 'uncorrected')['emotion']
            arrays.update({f'{side}_node_ids': s[side].copy(), f'{row}__{side}': rs_graphs.ags.pair_array(pairs),
                           f'{row}__{side}_w': np.ones(len(pairs), dtype=np.float32)})
        target = mode == 'full' and correction
        if target and mutation == 'edges':
            arrays['Gres__fit'] = arrays['Gres__fit'][:-1]
        if target and mutation == 'dtype':
            arrays['Gres__dev'] = arrays['Gres__dev'].astype(np.float64)
        if target and mutation == 'nodes':
            arrays['fit_node_ids'] = arrays['fit_node_ids'][::-1]
        np.savez(folder / 'emotion.npz', **arrays)
        identity = dict(mode=mode, fit_size=len(s['fit']), dev_size=len(s['dev']),
                        names_sha256=s['names_sha256'], sample_sha256=rs_graphs.ags.fingerprint(s))
        manifest = dict(identity=identity, archives={'emotion': rs_graphs.ags.sha_file(folder / 'emotion.npz')},
                        source_sha256={key: rs_graphs.ct2_graphs._sha_array(value) for key, value in x.items()},
                        array_sha256={'emotion': {key: rs_graphs.ct2_graphs._sha_array(value)
                                                  for key, value in arrays.items()}})
        if target and mutation == 'source':
            manifest['source_sha256']['E'] = 'wrong'
        if target and mutation == 'identity':
            manifest['identity']['sample_sha256'] = 'wrong'
        if target and mutation == 'archive':
            manifest['archives']['emotion'] = 'wrong'
        (folder / 'manifest.json').write_text(json.dumps(manifest))
    report = pf_graphs.gate_iv(r10_graphs=r10, inputs_provider=sources.__getitem__)
    assert report['ok'] == (mutation is None)
    assert json.loads((output_root / 'stage0a/graphs.json').read_text()) == report
    assert set(report['graphs']) == {'full/aspect__k40', 'full/uncorrected__k40', 'sample/aspect__k10'}
    for name, result in report['graphs'].items():
        assert result['ok'] == (mutation is None or name != 'full/aspect__k40')
        assert result['seconds'] > 0 and result['build_seconds'] > 0
        assert set(result['parts']) == {'fit', 'dev'}
        for side, part in result['parts'].items():
            assert part['count'] > 0 and part['seconds'] > 0
            assert set(part['arrays']) == {f'{side}_node_ids', f'{result["row"]}__{side}',
                                          f'{result["row"]}__{side}_w'}


@pytest.mark.slow
@pytest.mark.real_data
def test_gate_iv_real_data(output_root):
    """Controller only: all three r10 emotion graphs, fit and dev, on CPU."""
    report = pf_graphs.gate_iv()
    assert report['ok'], report
