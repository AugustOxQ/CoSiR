"""Emotion-only c* graphs on independent fit and validation node sets (r13 t04)."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import pf_paths as P  # frozen modules' import paths
import rs_graphs
from rs_vh import array_sha, names_sha

ags = rs_graphs.ags


@threadpool_limits.wrap(limits=8)
def emotion_pairs(E, D, I, k, correction):
    """Return sorted unique int64 local edges, with both endpoints reciprocal.

    Emotion-only copy of src/test/20261027_recipe_sweep/rs_graphs.py:_partition_aspect;
    omits style/genre work and takes already selected node arrays. All mutual top-k
    calls use k. Frozen helpers retain sampled-SD, argpartition and tie behavior.
    Only 256-by-n similarity blocks are allocated, never an n-by-n matrix.
    """
    assert isinstance(k, (int, np.integer)) and not isinstance(k, bool)
    assert isinstance(correction, (bool, np.bool_))
    assert E.ndim == 2 and E.shape[1] > 0 and 0 < k < len(E)
    for value in (E, D, I):
        assert value.dtype == np.float32 and value.shape == E.shape
        assert np.isfinite(value).all() and (np.linalg.norm(value, axis=1) > 0).all()
    x = {key: np.ascontiguousarray(ags.l2(value)) for key, value in zip(('E', 'D', 'I'), (E, D, I))}
    if not correction:
        return ags.pair_array(rs_graphs._mutual(x['E'], k))
    n = len(E)
    content_pairs = rs_graphs._mutual(x['D'], k) | rs_graphs._mutual(x['I'], k)
    r0 = rs_graphs._mutual(x['E'], k)
    deg = np.bincount(ags.pair_array(r0).ravel(), minlength=n)
    shared_deg = np.bincount(ags.pair_array(r0 & content_pairs).ravel(), minlength=n)
    lam = np.zeros(n, dtype=np.float32)
    lam[deg > 0] = shared_deg[deg > 0] / deg[deg > 0]
    lam[deg == 0] = lam[deg > 0].mean()
    draws = ags.draw_pairs(n)
    sd = {key: ags.similarity_sd(x[key], draws) for key in ('D', 'I', 'E')}
    nn = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, 256):
        end = min(start + 256, n)
        ds, ims = x['D'][start:end] @ x['D'].T, x['I'][start:end] @ x['I'].T
        content = (ds / sd['D'] + ims / sd['I']) / 2
        primary = (x['E'][start:end] @ x['E'].T) / sd['E']
        nn[start:end] = ags.top(primary - lam[start:end, None] * content, start, k)
    return ags.pair_array(ags.reciprocal(nn))


def _folder(k, correction):
    assert isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k > 0
    assert isinstance(correction, (bool, np.bool_))
    return P.OUTPUTS / 'graphs' / f'k{k}__{"on" if correction else "off"}'


def _keys_hash(keys):
    assert isinstance(keys, list) and all(isinstance(key, str) for key in keys)
    assert len(keys) == len(set(keys)) and all(key and '\n' not in key for key in keys)
    return names_sha(keys)


def _code_sha():
    files = dict(rs_graphs.code_files(), pf_graphs=ags.sha_file(Path(__file__)),
                 pf_paths=ags.sha_file(Path(P.__file__)))
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _check_pairs(pairs, n):
    assert pairs.dtype == np.int64 and pairs.ndim == 2 and pairs.shape[1] == 2
    assert np.all((pairs[:, 0] >= 0) & (pairs[:, 0] < pairs[:, 1]) & (pairs[:, 1] < n))
    assert np.all(np.diff(pairs[:, 0] * n + pairs[:, 1]) > 0), 'unsorted or duplicate pairs'


def build(k, correction, fit, val):
    """Write emotion.npz and manifest.json; return their directory.

    fit/val contain E, D, I and a list under 'keys', in exactly their row order.
    pairs_sha256 hashes the NPZ file bytes. Counts use 'fit' and 'dev'.
    """
    tick = time.monotonic()
    folder = _folder(k, correction)
    fit_hash, val_hash = _keys_hash(fit['keys']), _keys_hash(val['keys'])
    assert not set(fit['keys']) & set(val['keys']), 'fit/val overlap'
    arrays = {}
    for side, data in (('fit', fit), ('dev', val)):
        assert all(len(data[key]) == len(data['keys']) for key in ('E', 'D', 'I'))
        arrays[f'{side}_pairs'] = emotion_pairs(data['E'], data['D'], data['I'], k, correction)
    folder.mkdir(parents=True, exist_ok=True)
    ags.atomic_npz(folder / 'emotion.npz', **arrays)
    manifest = dict(k=int(k), correction=bool(correction), fit_keys_sha256=fit_hash, val_keys_sha256=val_hash,
                    counts={side: len(arrays[f'{side}_pairs']) for side in ('fit', 'dev')},
                    pairs_sha256=ags.sha_file(folder / 'emotion.npz'), code_sha256=_code_sha(),
                    seconds=time.monotonic() - tick)
    ags.atomic_json(folder / 'manifest.json', manifest)
    return folder


def load(k, correction, fit_keys, val_keys):
    """Verify provenance; return pairs, manifest, manifest_sha256, directory path.

    Endpoints are local to the corresponding supplied key list. Unit edge weights
    are implicit; callers can batch these arrays directly.
    """
    folder = _folder(k, correction)
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['k'] == k and manifest['correction'] == correction, 'graph settings'
    assert manifest['fit_keys_sha256'] == _keys_hash(fit_keys), 'fit keys hash mismatch'
    assert manifest['val_keys_sha256'] == _keys_hash(val_keys), 'val keys hash mismatch'
    assert manifest['code_sha256'] == _code_sha(), 'graph code hash mismatch'
    assert manifest['pairs_sha256'] == ags.sha_file(folder / 'emotion.npz'), 'graph pairs hash mismatch'
    with np.load(folder / 'emotion.npz', allow_pickle=False) as archive:
        assert set(archive.files) == {'fit_pairs', 'dev_pairs'}
        arrays = {key: archive[key] for key in archive.files}
    for side, keys in (('fit', fit_keys), ('dev', val_keys)):
        _check_pairs(arrays[f'{side}_pairs'], len(keys))
        assert len(arrays[f'{side}_pairs']) == manifest['counts'][side], 'graph count mismatch'
    return dict(arrays, manifest=manifest, manifest_sha256=ags.sha_file(folder / 'manifest.json'), path=folder)


def _replay_inputs(mode):
    """Frozen r10 providers and split order, with no graph materialization/writes."""
    data = rs_graphs.ct2_data
    s = data.split2(mode)
    return s, dict(E=data.instructed_captions(s, 'emotion_focus'),
                   D=data.default_captions(s), I=data.default_images(s))


@threadpool_limits.wrap(limits=8)
def gate_iv(*, r10_graphs=None, inputs_provider=None):
    """Rebuild the three r10 emotion graphs on CPU; write stage0a/graphs.json.

    Returns evidence with top-level 'ok'; the stage runner must stop if false.
    inputs_provider(mode) -> (frozen split dict, E/D/I spaces) and r10_graphs are
    input boundaries for synthetic replay tests. Defaults read r10's real stores.
    Checks include all six saved arrays, source hashes, node identities/order,
    and the reference NPZ checksum. Per-graph build_seconds excludes input I/O.
    """
    tick = time.monotonic()
    root = P.R10_GRAPHS if r10_graphs is None else Path(r10_graphs)
    provider = _replay_inputs if inputs_provider is None else inputs_provider
    report = dict(graphs={}, code_sha256=_code_sha())
    for mode, variants in (('full', ((40, True), (40, False))), ('sample', ((10, True),))):
        load_tick = time.monotonic()
        s, spaces = provider(mode)
        input_seconds = time.monotonic() - load_tick
        n = len(s['all_painting_ids'])
        for key in ('E', 'D', 'I'):
            assert spaces[key].ndim == 2 and len(spaces[key]) == n and spaces[key].dtype == np.float32
        for side in ('fit', 'dev'):
            nodes = s[side]
            assert nodes.dtype == np.int64 and nodes.ndim == 1 and len(np.unique(nodes)) == len(nodes)
            assert ((nodes >= 0) & (nodes < n)).all()
        assert not np.intersect1d(s['fit'], s['dev']).size
        source_hashes = {key: array_sha(spaces[key]) for key in ('E', 'D', 'I')}
        identity = dict(mode=mode, fit_size=len(s['fit']), dev_size=len(s['dev']),
                        names_sha256=s['names_sha256'], sample_sha256=ags.fingerprint(s))
        for k, correction in variants:
            graph_tick = time.monotonic()
            name = f'{mode}/{"aspect" if correction else "uncorrected"}__k{k}'
            row = 'Gres' if correction else 'R0'
            folder = root / name / 'd2/graphs'
            manifest = json.loads((folder / 'manifest.json').read_text())
            checks = dict(identity=all(manifest['identity'][key] == value for key, value in identity.items()),
                          source_hashes=all(manifest['source_sha256'][key] == value
                                            for key, value in source_hashes.items()),
                          archive=manifest['archives']['emotion'] == ags.sha_file(folder / 'emotion.npz'))
            result = dict(k=k, correction=correction, row=row, checks=checks, parts={},
                          input_seconds=input_seconds, build_seconds=0.0)
            with np.load(folder / 'emotion.npz', allow_pickle=False) as reference:
                for side in ('fit', 'dev'):
                    part_tick = time.monotonic()
                    nodes = s[side]
                    pairs = emotion_pairs(*(spaces[key][nodes] for key in ('E', 'D', 'I')), k, correction)
                    seconds = time.monotonic() - part_tick
                    arrays = {f'{side}_node_ids': nodes, f'{row}__{side}': pairs,
                              f'{row}__{side}_w': np.ones(len(pairs), dtype=np.float32)}
                    comparisons = {}
                    for key, actual in arrays.items():
                        expected = reference[key]
                        comparisons[key] = dict(dtype=actual.dtype == expected.dtype, shape=actual.shape == expected.shape,
                                                equal=bool(np.array_equal(actual, expected)),
                                                hash=array_sha(actual) == manifest['array_sha256']['emotion'][key])
                    result['parts'][side] = dict(count=len(pairs), reference_count=len(reference[f'{row}__{side}']),
                                                 arrays=comparisons, seconds=seconds,
                                                 ok=all(all(check.values()) for check in comparisons.values()))
                    result['build_seconds'] += seconds
            result.update(ok=all(checks.values()) and all(part['ok'] for part in result['parts'].values()),
                          seconds=time.monotonic() - graph_tick)
            report['graphs'][name] = result
        # Release full arrays before loading the sample's caption stores.
        del spaces, s
    report.update(ok=all(graph['ok'] for graph in report['graphs'].values()), seconds=time.monotonic() - tick)
    ags.atomic_json(P.OUTPUTS / 'stage0a/graphs.json', report)
    return report
