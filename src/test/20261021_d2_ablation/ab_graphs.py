"""CPU graph store of the D2 ablation: default-view graph (Dview, D14) and random graphs (rand_s*, D15).

Built on fit and early-stopping paintings only, separately per partition. The store layout is the one
`ags_head.load_graph(s, <round folder>, aspect, rows)` reads: <out>/graphs/manifest.json and
{emotion,style,genre}.npz. Dview is the reciprocal top-20 on the content score; rand_s* is a degree-preserving
double-edge-swap randomization of D2's own graph of the aspect. Unit weights, no self-pairs.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
MAIN_TEST = Path('/project/CoSiR/src/test')
MAIN_ROUND = MAIN_TEST / '20261021_d2_ablation'
AGS = MAIN_TEST / '20261018_aspect_graph_sheet'  # D2's graph store (gitignored, main checkout only)
SAMPLE = MAIN_TEST / '20261017_prompted_aspect_views/sample.npz'
sys.path.insert(0, str(TEST / '20261018_aspect_graph_sheet'))
import ags_graphs  # noqa: E402  (its own imports put the earlier-round folders on sys.path)
import ags_head  # noqa: E402

ASPECTS = ('emotion', 'style', 'genre')  # aspect index = position, as in ms_train's [seed, aspect_index]
D2_ROW = dict(emotion='Gres', style='Fres05', genre='R0o')
PARTITIONS = ('fit', 'dev')  # partition index = position
DVIEW, RAND = 'Dview', 'rand_s{seed}'
K = 20
SWAPS_PER_EDGE = 10
METHOD = ('d2 ablation graphs v1; Dview: ags_graphs draw_pairs/similarity_sd/top/reciprocal, content score = mean of '
          'standardized default caption cosine and default image cosine, reciprocal top-20, per partition, block256, '
          'sd200000 rng0 offdiagonal; rand_s{seed}: degree-preserving double-edge swaps of D2 graph (emotion Gres, '
          'style Fres05, genre R0o), 10 accepted swaps per edge, rng [seed, aspect index, partition index], '
          'unit weights')


def check_paths():
    for module in (ags_graphs, ags_head):
        expected = (TEST / '20261018_aspect_graph_sheet' / f'{module.__name__}.py').resolve()
        assert Path(module.__file__).resolve() == expected, module.__file__


def reciprocal_topk(score, k=K, block=256):
    """Reciprocal top-k pairs (a < b, int64, sorted) of an n x n score matrix; self excluded by ags_graphs.top."""
    n = len(score)
    nn = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, block):
        end = min(start + block, n)
        nn[start:end] = ags_graphs.top(np.asarray(score[start:end]), start, k)
    return ags_graphs.pair_array(ags_graphs.reciprocal(nn))


def dview_pairs(d, i, k=K, block=256):
    """Default-view graph of one partition. d, i: unit-norm default caption / image rows of the partition's nodes.

    Same content score as ags_graphs.build_partition: mean of cos/sd for the two spaces, sd from draw_pairs.
    Returns (sorted int64 pairs, dict of the two sd values).
    """
    n = len(d)
    draws = ags_graphs.draw_pairs(n)
    sd_d, sd_i = ags_graphs.similarity_sd(d, draws), ags_graphs.similarity_sd(i, draws)
    nn = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, block):
        end = min(start + block, n)
        content = ((d[start:end] @ d.T) / sd_d + (i[start:end] @ i.T) / sd_i) / 2
        nn[start:end] = ags_graphs.top(content, start, k)
    return ags_graphs.pair_array(ags_graphs.reciprocal(nn)), dict(sd_default_caption=sd_d, sd_default_image=sd_i)


def pair_degrees(pairs, n):
    return np.bincount(np.asarray(pairs).ravel(), minlength=n)


def degree_preserving_swaps(pairs, n, rng, swaps_per_edge=SWAPS_PER_EDGE):
    """Double-edge swaps: (a,b),(c,d) -> (a,d),(c,b) or (a,c),(b,d); rejected if a self pair or an existing pair.

    Runs until swaps_per_edge * m swaps are accepted, or 100 * swaps_per_edge * m attempts are spent (graphs that
    cannot change). Returns (sorted int64 pairs a < b, info).
    """
    m = len(pairs)
    edges = [(int(a), int(b)) for a, b in pairs]
    present = {a * n + b for a, b in edges}
    target, cap = swaps_per_edge * m, 100 * swaps_per_edge * m
    accepted = attempts = 0
    while accepted < target and attempts < cap and m > 1:
        batch = min(1 << 16, cap - attempts)
        ij = rng.integers(m, size=(batch, 2))
        flips = rng.integers(2, size=batch)
        for (i, j), flip in zip(ij.tolist(), flips.tolist()):
            attempts += 1
            if i == j:
                continue
            a, b = edges[i]
            c, d = edges[j]
            x1, y1, x2, y2 = (a, d, c, b) if flip else (a, c, b, d)
            if x1 == y1 or x2 == y2:
                continue
            e1 = (x1, y1) if x1 < y1 else (y1, x1)
            e2 = (x2, y2) if x2 < y2 else (y2, x2)
            c1, c2 = e1[0] * n + e1[1], e2[0] * n + e2[1]
            if c1 == c2 or c1 in present or c2 in present:
                continue
            present.discard(a * n + b)
            present.discard(c * n + d)
            present.add(c1)
            present.add(c2)
            edges[i], edges[j] = e1, e2
            accepted += 1
            if accepted >= target:
                break
    out = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)
    return out, dict(accepted=accepted, attempts=attempts, target=target)


def load_sample():
    import pav_sample  # on sys.path via ags_graphs' imports
    pav_sample.SAMPLE = SAMPLE  # artifacts are gitignored: read the main checkout's sample
    return ags_graphs.load_frozen()


def graph_summary(pairs, n):
    deg = pair_degrees(pairs, n)
    return dict(edges=len(pairs), degree_mean=float(deg.mean()), degree_max=int(deg.max()),
                isolated=int((deg == 0).sum()))


def fmt(summary):
    return (f"edges={summary['edges']} degree mean={summary['degree_mean']:.3f} max={summary['degree_max']} "
            f"isolated={summary['isolated']}")


def as_set(pairs):
    return set(map(tuple, np.asarray(pairs).tolist()))


def build(out, seeds=(0, 1, 2), overwrite=False):
    check_paths()
    tick = time.monotonic()
    folder = Path(out) / 'graphs'
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists() and not overwrite:
        raise SystemExit(f'{manifest_path} exists; pass --overwrite to replace the store')
    s = load_sample()
    _, fit, dev = ags_graphs.split(s)
    parts = dict(fit=fit, dev=dev)
    d_all, i_all = ags_graphs.targets(s, 'default'), ags_graphs.default_images(s)[0]
    unit = lambda x: np.ascontiguousarray(ags_graphs.l2(x))
    dview, metadata = {}, dict(dview={}, rand={})
    for side in PARTITIONS:
        t = time.monotonic()
        dview[side], sds = dview_pairs(unit(d_all[parts[side]]), unit(i_all[parts[side]]))
        summary = graph_summary(dview[side], len(parts[side]))
        metadata['dview'][side] = dict(sds, **summary)
        print(f'{DVIEW} {side}: {fmt(summary)} ({time.monotonic() - t:.1f}s)', flush=True)
    names = [RAND.format(seed=x) for x in seeds]
    arrays, counts, source = {}, {}, {}
    for a_idx, aspect in enumerate(ASPECTS):
        row = D2_ROW[aspect]
        original, original_manifest = ags_head.load_graph(s, AGS, aspect, [row])
        source[aspect] = dict(row=row, archive_sha256=original_manifest['archives'][aspect])
        out_arrays = {f'{side}_node_ids': parts[side] for side in PARTITIONS}
        for p_idx, side in enumerate(PARTITIONS):
            n = len(parts[side])
            base_pairs = original['rows'][row][side][0]
            base_set = as_set(base_pairs)
            both = len(as_set(dview[side]) & base_set) / len(dview[side])
            info_o = dict(graph_summary(base_pairs, n), dview_shared_fraction=both)
            metadata['rand'].setdefault(aspect, {}).setdefault(side, {})['original'] = info_o
            print(f'{aspect} {side}: original {row}: {fmt(info_o)}; {DVIEW} shares {both:.4f} of its edges with it',
                  flush=True)
            out_arrays[f'{DVIEW}__{side}'] = dview[side]
            for seed, name in zip(seeds, names):
                rng = np.random.default_rng([seed, a_idx, p_idx])
                pairs, info = degree_preserving_swaps(base_pairs, n, rng)
                equal = bool(np.array_equal(pair_degrees(pairs, n), pair_degrees(base_pairs, n)))
                assert equal and len(pairs) == len(base_pairs), 'degree sequence or edge count changed'
                shared = len(as_set(pairs) & base_set) / len(pairs)
                chance = len(pairs) / (n * (n - 1) / 2)  # edge density: chance that a random pair is an original edge
                summary = dict(graph_summary(pairs, n), shared_with_original=shared, chance=chance,
                               equal_degree=equal, **info)
                metadata['rand'][aspect][side][name] = summary
                print(f'{aspect} {name} {side}: {fmt(summary)} shared with {row}={shared:.4f} '
                      f'(edge density {chance:.5f}) equal degree={equal} '
                      f'swaps accepted={info["accepted"]}/{info["attempts"]}', flush=True)
                out_arrays[f'{name}__{side}'] = pairs
            for name in [DVIEW] + names:
                out_arrays[f'{name}__{side}_w'] = np.ones(len(out_arrays[f'{name}__{side}']), dtype=np.float32)
        arrays[aspect] = out_arrays
        counts[aspect] = {side: {name: len(out_arrays[f'{name}__{side}']) for name in [DVIEW] + names}
                          for side in PARTITIONS}
    folder.mkdir(parents=True, exist_ok=True)
    archives = {}
    for aspect, out_arrays in arrays.items():
        path = folder / f'{aspect}.npz'
        ags_graphs.atomic_npz(path, **out_arrays)
        archives[aspect] = ags_graphs.sha_file(path)
    identity = dict(method=METHOD, sample_sha256=ags_graphs.fingerprint(s), fit_size=len(fit), dev_size=len(dev),
                    k=K, swaps_per_edge=SWAPS_PER_EDGE, seeds=list(seeds), rows=[DVIEW] + names)
    ags_graphs.atomic_json(manifest_path, dict(identity=identity, archives=archives, counts=counts,
        source_d2_graphs=source, metadata=metadata, seconds=time.monotonic() - tick))  # seals the store: last
    for aspect in ASPECTS:  # the contract: ags_head.load_graph's own asserts
        loaded, _ = ags_head.load_graph(s, Path(out), aspect, [DVIEW] + names)
        assert set(loaded['rows']) == {DVIEW} | set(names)
    print(f'Graph store PASS: {time.monotonic() - tick:.1f}s; {folder}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=MAIN_ROUND, help='round folder; the store goes to <out>/graphs')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=8):
        build(args.out, tuple(args.seeds), args.overwrite)


if __name__ == '__main__':
    main()
