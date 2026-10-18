"""Label-free mutual-top-20 graphs; labels are read only by fit graph statistics."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ITP = HERE.parent / '20261017_image_transfer_pilot'
sys.path.insert(0, str(ITP))
from itp_data import atomic_json, default_images, fingerprint, load_frozen, split, targets
from pav_score import graph_stats, mutual_pairs
from itp_score import top20_neighbours

SMOKE = Path('/tmp/claude-0/-project-CoSiR/49c42a7b-8a89-4a25-816b-2232f8f66525/scratchpad/pbp_smoke')
GRAPH_NAMES = ('E', 'D', 'I', 'C', 'P', 'N', 'EC')
METHOD = 'pav mutual cosine top20; C=D|I; P=E-C; N=C-E; local indices; v1'


def root(smoke=False):
    return SMOKE if smoke else HERE


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pair_array(pairs):
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def derive(e, d, i):
    c = d | i
    return dict(E=e, D=d, I=i, C=c, P=e - c, N=c - e, EC=e & c)


def check_sets(graphs, n):
    for name in GRAPH_NAMES:
        ps = graphs[name]
        assert all(0 <= a < b < n for a, b in ps), (name, 'noncanonical/self/out-of-range')
        # Stored once as i<j, expanded symmetrically by every adjacency consumer.
        directed = {(a, b) for a, b in ps} | {(b, a) for a, b in ps}
        assert len(directed) == 2 * len(ps)
        assert all((b, a) in directed for a, b in directed)
    assert graphs == derive(graphs['E'], graphs['D'], graphs['I'])
    assert graphs['P'] <= graphs['E'] and not graphs['P'] & graphs['C']
    assert not graphs['N'] & graphs['E']


def source_pairs(x):
    """Call pav unchanged; independently verify reciprocal top-20 source edges."""
    ps = mutual_pairs(x)
    nn = top20_neighbours(x)
    directed = {(a, int(b)) for a, row in enumerate(nn) for b in row}
    reciprocal = {(a, b) for a, b in directed if a < b and (b, a) in directed}
    assert ps == reciprocal, 'pav pairs disagree with reciprocal source neighbours'
    return ps


def partitions_for(s, smoke):
    _, fit, dev = split(s)
    return dict(fit=fit[:2000] if smoke else fit, dev=dev)


def graph_identity(s, smoke):
    return dict(method=METHOD, sample_sha256=fingerprint(s), smoke=smoke,
                fit_size=2000 if smoke else 13400, dev_size=1489)


def load_graphs(s, smoke=False):
    base = root(smoke) / 'graphs'
    manifest = json.loads((base / 'manifest.json').read_text())
    assert manifest['identity'] == graph_identity(s, smoke), 'graph identity mismatch'
    result = {}
    for side, nodes in partitions_for(s, smoke).items():
        path = base / f'{side}.npz'
        assert sha_file(path) == manifest['archives'][side], 'graph archive changed'
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['node_ids'], nodes), 'graph node order mismatch'
            arrays = {name: z[name] for name in GRAPH_NAMES}
        sets = {}
        for name, a in arrays.items():
            assert a.dtype == np.int64 and a.ndim == 2 and a.shape[1] == 2
            sets[name] = set(map(tuple, a.tolist()))
            assert len(sets[name]) == len(a), 'duplicate pairs'
            assert np.array_equal(a, pair_array(sets[name])), 'noncanonical pair order'
        check_sets(sets, len(nodes))
        result[side] = dict(node_ids=nodes, **arrays)
    return result, manifest


def label_vectors(s, nodes):
    # Only graph statistics and scoring call this function.
    from apc_data import paint_first, paint_majority
    count = len(s['first'])
    return dict(emotion=paint_majority(s['emo'], s['pid'], count, 9)[nodes],
                genre=paint_first(s['gen'], s['pid'], count)[nodes],
                style=paint_first(s['sty'], s['pid'], count)[nodes])


def fit_statistics(graphs, labels, n):
    e = graphs['E']
    stats = dict(n_nodes=n, fractions={
        'E_intersect_C_over_E': len(e & graphs['C']) / len(e),
        'E_intersect_D_over_E': len(e & graphs['D']) / len(e),
        'E_intersect_I_over_E': len(e & graphs['I']) / len(e)}, graphs={})
    for name, pairs in graphs.items():
        deg = np.bincount(pair_array(pairs).ravel(), minlength=n)
        stats['graphs'][name] = dict(graph_stats(pairs, labels),
            degree_zero=int((deg == 0).sum()), mean_degree=float(deg.mean()))
    return stats


def stats_markdown(stats):
    lines = ['| Fit graph | Pairs | Degree 0 | Mean degree | Emotion lift | Genre lift | Style lift |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, g in stats['graphs'].items():
        lifts = [g['labels'][label]['lift'] for label in ('emotion', 'genre', 'style')]
        lines.append(f"| {name} | {g['n_pairs']} | {g['degree_zero']} | {g['mean_degree']:.6f} | " +
                     ' | '.join('—' if v is None else f'{v:.6f}' for v in lifts) + ' |')
    lines += ['', 'E overlap fractions: ' + '; '.join(f'{k}={v:.8f}' for k, v in stats['fractions'].items()) + '.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    start = time.monotonic()
    s = load_frozen()
    base = root(args.smoke) / 'graphs'
    base.mkdir(parents=True, exist_ok=True)
    marker = base / 'manifest.json'
    if marker.exists():
        _, manifest = load_graphs(s, args.smoke)
        print('Graphs verified; skip. ' + json.dumps(manifest['counts']), flush=True)
        return
    spaces = dict(E=targets(s, 'emotion_focus'), D=targets(s, 'default'), I=default_images(s)[0])
    manifest = dict(identity=graph_identity(s, args.smoke), archives={}, counts={}, source_sha256={})
    fit_sets = None
    for side, nodes in partitions_for(s, args.smoke).items():
        ps = {}
        for name, x in spaces.items():
            xx = np.ascontiguousarray(x[nodes])
            manifest['source_sha256'][f'{side}.{name}'] = hashlib.sha256(xx.tobytes()).hexdigest()
            ps[name] = source_pairs(xx)
            print(f'{side} {name}: {len(ps[name])} pairs [{time.monotonic()-start:.1f}s]', flush=True)
        gs = derive(ps['E'], ps['D'], ps['I'])
        check_sets(gs, len(nodes))
        path = base / f'{side}.npz'
        pending = base / f'{side}.pending.npz'
        np.savez(pending, node_ids=nodes, **{name: pair_array(g) for name, g in gs.items()})
        pending.replace(path)
        manifest['archives'][side] = sha_file(path)
        manifest['counts'][side] = {name: len(g) for name, g in gs.items()}
        if side == 'fit':
            fit_sets = gs
    # Graph construction is complete before any label is accessed for statistics.
    stats = fit_statistics(fit_sets, label_vectors(s, partitions_for(s, args.smoke)['fit']),
                           manifest['identity']['fit_size'])
    stats['labels_sha256'] = hashlib.sha256(b''.join(s[k].tobytes() for k in ('emo', 'gen', 'sty'))).hexdigest()
    atomic_json(base / 'stats.json', stats)
    (base / 'stats.md').write_text(stats_markdown(stats))
    manifest['seconds'] = time.monotonic() - start
    atomic_json(marker, manifest)
    load_graphs(s, args.smoke)
    print(stats_markdown(stats), flush=True)
    print(f"Graphs PASS: {manifest['seconds']:.1f}s; {base}", flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
