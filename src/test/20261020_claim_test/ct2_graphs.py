"""D2's Gres / Fres05 / R0o graphs, with independent fit and dev neighbours.

CLI: --mode {sample,full} --out DIR. DIR receives emotion.npz, style.npz,
genre.npz and manifest.json. Each NPZ contains int64 fit_node_ids/dev_node_ids,
int64 <row>__fit/dev (M x 2, sorted unique local endpoints i < j), and float32
<row>__fit_w/dev_w (M unit weights). Only that aspect's selected row is saved.

build returns (graphs, manifest), directly consumable by ms_train.run with
manifests = dict.fromkeys(graphs, manifest). The schema-1 manifest contains
identity (method, partition sizes), counts[aspect][part][row], archives[aspect]
(SHA-256 of the NPZ bytes), array_sha256[aspect][npz_key], source_sha256[space],
seconds (build, including hashes), part_seconds and peak_rss_bytes (Linux process
lifetime high-water mark, including inputs/imports). Array hashes use contiguous
C-order raw bytes. CLI also binds mode, label-free sample fingerprint and the
provider's per-role name hashes in identity. With --out BASE/graphs, sample
archives can also be read by ags_head.load_graph(s, BASE, aspect, [row]).

The CLI expects ct2_data.split2(mode), default_images(s), default_captions(s),
instructed_images(s, key), instructed_captions(s, key). All providers return
normalized float32 painting arrays, with captions pooled to painting means.
"""
import argparse
import hashlib
import io
import resource
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261018_aspect_graph_sheet'))
import ags_graphs as ags

ROWS = dict(emotion='Gres', style='Fres05', genre='R0o')
KEYS = ('D', 'I', 'E', 'style.img', 'genre.img')
METHOD = 'D2 selected AGS Gres/Fres05/R0o; block256; sd200000 rng0 offdiagonal; v1'


def _sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def _arrays(graph):
    arrays = {f'{part}_node_ids': graph[f'{part}_node_ids'] for part in ('fit', 'dev')}
    for row, parts in graph['rows'].items():
        for part, (pairs, weights) in parts.items():
            arrays[f'{row}__{part}'] = pairs
            arrays[f'{row}__{part}_w'] = weights
    return arrays


def _archive_sha(graph):
    with io.BytesIO() as stream:
        np.savez(stream, **_arrays(graph))
        return hashlib.sha256(stream.getbuffer()).hexdigest()


def _mutual(x):
    nn = np.empty((len(x), 20), dtype=np.int64)
    for start in range(0, len(x), 256):
        nn[start:start + 256] = ags.top(x[start:start + 256] @ x.T, start)
    return ags.reciprocal(nn)


def _partition(spaces, nodes):
    n = len(nodes)
    x = {key: np.ascontiguousarray(ags.l2(spaces[key][nodes])) for key in KEYS}
    content_pairs = _mutual(x['D']) | _mutual(x['I'])
    r0 = _mutual(x['E'])
    genre = _mutual(x['genre.img'])
    # Same anchor lambda rule as AGS build_partition, including zero-degree fallback.
    deg = np.bincount(ags.pair_array(r0).ravel(), minlength=n)
    shared_deg = np.bincount(ags.pair_array(r0 & content_pairs).ravel(), minlength=n)
    lam = np.zeros(n, dtype=np.float32)
    lam[deg > 0] = shared_deg[deg > 0] / deg[deg > 0]
    lam[deg == 0] = lam[deg > 0].mean()
    draws = ags.draw_pairs(n)
    sd = {key: ags.similarity_sd(x[key], draws) for key in ('D', 'I', 'E', 'style.img')}
    nn = {a: np.empty((n, 20), dtype=np.int64) for a in ('emotion', 'style')}
    for start in range(0, n, 256):
        end = min(start + 256, n)
        ds, ims = x['D'][start:end] @ x['D'].T, x['I'][start:end] @ x['I'].T
        content = (ds / sd['D'] + ims / sd['I']) / 2
        primary = (x['E'][start:end] @ x['E'].T) / sd['E']
        nn['emotion'][start:end] = ags.top(primary - lam[start:end, None] * content, start)
        primary = (x['style.img'][start:end] @ x['style.img'].T) / sd['style.img']
        nn['style'][start:end] = ags.top(primary - .5 * content, start)
    return dict(emotion=ags.reciprocal(nn['emotion']), style=ags.reciprocal(nn['style']),
                genre=genre)


@threadpool_limits.wrap(limits=8)
def build(spaces, fit, dev):
    """Return (graphs, manifest); spaces are aligned float32 D/I/E/style.img/genre.img.

    D/E are normalized default/emotion caption painting means; I/style.img/genre.img
    are normalized image spaces. Node arrays contain global rows in the caller's
    order; edge endpoints are local indices into each node array, as ms_train.run.
    Only selected rows are returned. The manifest's archive hashes describe NPZs
    containing the node arrays and <row>__<part> / <row>__<part>_w arrays.
    """
    start = time.monotonic()
    shape = spaces['D'].shape
    assert len(shape) == 2 and shape[1] > 0
    for key in KEYS:
        x = spaces[key]
        assert x.shape == shape and x.dtype == np.float32 and np.isfinite(x).all(), key
    parts = dict(fit=fit, dev=dev)
    for nodes in parts.values():
        assert nodes.dtype == np.int64 and nodes.ndim == 1 and len(nodes) > 20
        assert len(np.unique(nodes)) == len(nodes)
        assert (nodes >= 0).all() and (nodes < shape[0]).all()
    assert not np.intersect1d(fit, dev).size
    graphs = {a: dict(fit_node_ids=fit, dev_node_ids=dev, rows={row: {}})
              for a, row in ROWS.items()}
    part_seconds = {}
    for part, nodes in parts.items():
        tick = time.monotonic()
        for aspect, pairs in _partition(spaces, nodes).items():
            pairs = ags.pair_array(pairs)
            graphs[aspect]['rows'][ROWS[aspect]][part] = (pairs, np.ones(len(pairs), dtype=np.float32))
        part_seconds[part] = time.monotonic() - tick
    manifest = dict(schema=1, identity=dict(method=METHOD, fit_size=len(fit), dev_size=len(dev)),
                    archives={a: _archive_sha(g) for a, g in graphs.items()},
                    counts={a: {p: {r: len(graphs[a]['rows'][r][p][0])} for p in parts}
                            for a, r in ROWS.items()},
                    array_sha256={a: {key: _sha_array(x) for key, x in _arrays(g).items()}
                                  for a, g in graphs.items()},
                    source_sha256={key: _sha_array(spaces[key]) for key in KEYS},
                    part_seconds=part_seconds,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    peak_rss_scope='process lifetime high-water mark (Linux ru_maxrss)')
    manifest['seconds'] = time.monotonic() - start
    return graphs, manifest


def save(graphs, manifest, out):
    """Seal three selected-row NPZs, then their manifest, directly in out."""
    out = Path(out)
    assert set(graphs) == set(ROWS)
    for aspect, graph in graphs.items():
        assert set(graph['rows']) == {ROWS[aspect]}
        assert _archive_sha(graph) == manifest['archives'][aspect], 'graph arrays changed after build'
    out.mkdir(parents=True, exist_ok=True)
    for aspect, graph in graphs.items():
        path = out / f'{aspect}.npz'
        ags.atomic_npz(path, **_arrays(graph))
        assert ags.sha_file(path) == manifest['archives'][aspect], 'graph archive checksum mismatch'
    ags.atomic_json(out / 'manifest.json', manifest)


@threadpool_limits.wrap(limits=8)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['sample', 'full'])
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    import ct2_data

    s = ct2_data.split2(args.mode)
    spaces = dict(D=ct2_data.default_captions(s), I=ct2_data.default_images(s),
                  E=ct2_data.instructed_captions(s, 'emotion_focus'),
                  **{'style.img': ct2_data.instructed_images(s, 'style_word'),
                     'genre.img': ct2_data.instructed_images(s, 'genre_word')})
    graphs, manifest = build(spaces, s['fit'], s['dev'])
    manifest['identity'].update(mode=args.mode, sample_sha256=ags.fingerprint(s),
                                names_sha256=s['names_sha256'])
    save(graphs, manifest, args.out)
    print(f'D2 graphs: {args.out}; seconds={manifest["seconds"]:.2f}; '
          f'peak RSS={manifest["peak_rss_bytes"] / 2**20:.1f} MiB', flush=True)


if __name__ == '__main__':
    main()
