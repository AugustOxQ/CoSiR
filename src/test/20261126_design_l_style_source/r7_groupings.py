"""Round-7 painting groupings (DECISION_RULE L1-L4, L5 U, L6-L9; C5/C6).

CPU only; no episodes or label AMIs. --only accepts comma/space-separated IDs.
--out is an output directory containing r7_groupings.{npz,json}.
--trial --tokstats DIR restricts scorer-train to the extraction's image prefix;
its outputs are explicitly trial objects, never the real shared interface.
"""
import r7_common as C

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_mutual_info_score as ami

STYLE = Path('/data/SSD2/pre_extract/artelingo/style_csd_vitl')
TOKSTATS = Path('/data/SSD2/pre_extract/artelingo/clip_b32_tokstats')
IDS = tuple(f'V{k}' for k in range(1, 7)) + tuple(f'Z{k}' for k in range(1, 7)) + ('T', 'R0')


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def first_rows(g):
    assert g.dtype == np.int64 and g.ndim == 1 and len(g), 'painting codes: shape/dtype'
    u, first = np.unique(g, return_index=True)
    assert np.array_equal(u, np.arange(len(u))), 'painting codes must be compact, sorted IDs'
    return first


def painting_labels(local, g):
    assert local.dtype == np.int64 and local.shape == g.shape, 'local labels: shape/dtype'
    p = local[first_rows(g)]
    assert np.array_equal(p[g], local), 'labels differ within painting'
    assert np.array_equal(np.unique(p), np.arange(len(np.unique(p)))), 'noncompact labels'
    return p


def painting_vectors(source, row_images, st, g):
    """One source vector per first scorer-train row, without expanding image features."""
    image = row_images[st]
    first = first_rows(g)
    assert np.array_equal(image, image[first][g]), 'painting maps to multiple images'
    assert len(np.unique(image[first])) == len(first), 'image maps to multiple paintings'
    xp = np.asarray(source[image[first]], dtype=np.float32)
    assert np.isfinite(xp).all(), 'nonfinite painting vectors'
    # Repeated rows index the very same image-level source row: difference is exactly zero.
    return xp, {'painting_to_image_one_to_one': True, 'paintings_with_differing_rows': 0}


def read_source(directory, data, rs, tokens=False, trial=False):
    """Copy of real_style_features C6 checks, with ticket 02's external join paths."""
    meta_path = directory / 'meta.json'
    assert meta_path.is_file(), f'{meta_path}: extraction incomplete'
    meta = json.loads(meta_path.read_text())
    ann, sids, paths = rs.annotation_image_paths(data)
    assert sids.dtype == np.int64 and sids.shape == (308723,), 'sample IDs shape/dtype'
    assert np.array_equal(np.sort(sids), np.arange(len(ann))), 'sample ID coverage'
    if tokens:
        assert meta['mode'] == ('trial' if trial else 'full'), 'token extraction mode'
        assert meta['kind'] == 'CLIP B/32 token statistics', 'token source kind'
        assert meta['blocks'] == [4, 6, 8] and meta['patch_tokens'] == 49, 'token layout'
        assert meta['cache_check']['min_cosine'] >= 0.9999, 'L5 cache check'
        rm = meta['row_mapping']
        assert Path(rm['source_meta']).resolve() == (STYLE / 'meta.json').resolve(), 'token source meta path'
        assert rm['source_meta_sha256'] == C.INPUTS[str(STYLE / 'meta.json')], 'token source meta SHA'
        r2ip, r2iap = Path(rm['row_to_image']), Path(rm['row_to_image_annotation_order'])
        assert r2ip.resolve() == (STYLE / 'row_to_image.npy').resolve(), 'token row map path'
        assert r2iap.resolve() == (STYLE / 'row_to_image_annotation_order.npy').resolve(), 'token annotation map path'
        emb_path, dim = directory / 'tokstats.npy', 4608
        assert C.sha_file(emb_path) == meta['output']['tokstats.npy']['sha256'], 'token statistics SHA'
        assert meta['output']['tokstats.npy']['dtype'] == 'float32', 'token output dtype record'
    else:
        assert C.sha_file(meta_path) == C.INPUTS[str(meta_path)], 'CSD meta SHA'
        assert meta['plan']['sha256'] == rs.PLAN_SHA and meta['mode'] == 'full', 'CSD plan/mode'
        assert meta['kind'].startswith('CSD style embedding'), 'CSD source kind'
        rm = meta['row_mapping']
        r2ip, r2iap = directory / 'row_to_image.npy', directory / 'row_to_image_annotation_order.npy'
        emb_path, dim = directory / 'embeddings.npy', 768
        assert C.sha_file(emb_path) == C.INPUTS[str(emb_path)], 'CSD embeddings SHA'
    assert C.sha_file(Path(meta['images']['annotations'])) == meta['images']['annotations_sha256'], 'annotations SHA'
    source_meta = json.loads((STYLE / 'meta.json').read_text())
    assert C.sha_file(STYLE / 'meta.json') == C.INPUTS[str(STYLE / 'meta.json')], 'source meta SHA'
    assert meta['images']['annotations_sha256'] == source_meta['images']['annotations_sha256'], 'source annotations SHA'
    img_paths = np.asarray(source_meta['images']['image_paths'], dtype=object)
    assert len(img_paths) == len(set(img_paths)) == meta['images']['n_images_total'] == 61402, 'image coverage'
    n = meta['images']['n_images_in_this_file']
    assert ((0 < n < 61402) if trial else n == 61402), 'source image count'
    assert np.array_equal(meta['images']['image_paths'], img_paths[:n]), 'source image order'
    if tokens:
        path_sha = hashlib.sha256(json.dumps(img_paths.tolist(), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        assert path_sha == rm['image_paths_sha256'], 'image path SHA'
        assert sha_array(sids) == rm['sample_ids_sha256'], 'sample ID SHA'
        assert meta['output']['tokstats.npy']['shape'] == [n, dim], 'token output shape record'
    r2i, r2ia = (np.load(p, allow_pickle=False) for p in (r2ip, r2iap))
    for a, key in ((r2i, 'row_to_image_sha256'), (r2ia, 'row_to_image_annotation_order_sha256')):
        assert a.dtype == np.int32 and a.shape == (len(ann),), 'row mapping shape/dtype'
        assert a.min() == 0 and a.max() == 61401, 'row mapping range'
        assert sha_array(a) == rm[key] == source_meta['row_mapping'][key], 'row mapping array SHA'
    assert np.array_equal(img_paths[r2ia], [a['image'] for a in ann]), 'annotation image join'
    assert np.array_equal(r2i, r2ia[sids]), 'sample ID row join'
    assert np.array_equal(img_paths[r2i], paths), 'data image join'
    assert len(np.unique(r2i)) == 61402, 'every image used'
    _, p = np.unique(np.asarray(data.paintings), return_inverse=True)
    pairs = np.unique(np.stack((p, r2i), axis=1), axis=0)
    assert len(pairs) == len(np.unique(p)) == len(np.unique(r2i)), 'painting/image bijection'
    source = np.load(emb_path, mmap_mode='r', allow_pickle=False)
    assert source.shape == (n, dim) and source.dtype == np.float32, 'source shape/dtype'
    assert np.isfinite(source).all(), 'source finite'
    return source, r2i, {'meta_sha256': C.sha_file(meta_path), 'source_sha256': C.sha_file(emb_path),
                         'row_to_image_sha256': sha_array(r2i), 'C6_checks': True}


def orth(x, content):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0)
    m = np.stack([centered[content == k].mean(axis=0) for k in np.unique(content)], axis=1)
    singular = np.linalg.svd(m, compute_uv=False)
    residual = centered - (m @ (np.linalg.pinv(m) @ centered.T)).T
    before = float(np.abs(m.T @ centered.T).max())
    after = float(np.abs(m.T @ residual.T).max())
    assert after <= 1e-6 * before, 'L3 Orth projection assertion'
    return residual.astype(np.float32), {'singular_values': singular.tolist(),
        'pinv_rank': int(np.sum(singular > 1e-15 * singular[0])), 'pinv_rcond': 1e-15,
        'max_abs_before': before, 'max_abs_after': after}


def preprocess_u(x):
    x = np.asarray(x, dtype=np.float64)
    std = x.std(axis=0, ddof=0)
    z = np.divide(x - x.mean(axis=0), std, out=np.zeros_like(x), where=std != 0)
    assert min(z.shape) >= 256, 'L5 U requires 256 PCA components'
    pca = PCA(n_components=256, svd_solver='full')
    u = pca.fit_transform(z)
    assert np.isfinite(u).all(), 'U finite'
    return u, {'zero_deviation_dimensions': int(np.sum(std == 0)), 'pca_components': 256,
               'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist()}


def ridge_residual(x, clip, unit):
    predictors = unit(np.asarray(clip, dtype=np.float32))
    assert np.allclose(np.linalg.norm(predictors, axis=1), 1, atol=1e-6), 'L4 unit CLIP vectors'
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 6)).fit(predictors, x)
    residual = np.asarray(x - ridge.predict(predictors), dtype=np.float32)
    assert np.isfinite(residual).all(), 'ridge finite'
    return residual, {'alpha': float(ridge.alpha_)}


def leiden(graph, seed, weights=None):
    import leidenalg
    p = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                 weights=weights, resolution_parameter=1.0, seed=seed)
    return np.unique(np.asarray(p.membership, dtype=np.int64), return_inverse=True)[1].astype(np.int64)


def describe(local, g, content):
    p = painting_labels(local, g)
    return {'n_groups': len(np.unique(p)), 'sizes_rows': np.bincount(local).tolist(),
            'sizes_paintings': np.bincount(p).tolist(), 'ami_content_paintings': float(ami(content, p)),
            'ami_content_rows': float(ami(content[g], local)), 'partition_sha256': sha_array(local),
            'stability_ami': None}


def group_vectors(x, g, content, rs, weight=None, graph=None, min_rows=200):
    xp = rs.rc.unit(np.asarray(x, dtype=np.float32))
    assert np.isfinite(xp).all(), 'graph vectors finite'
    graph = rs.rsw.knn_union_graph(xp, 20) if graph is None else graph.copy()
    if weight is not None:
        graph.es['weight'] = [weight if content[a] == content[b] else 1.0 for a, b in graph.get_edgelist()]
    labels, moves = {}, {}
    for seed in (42, 43, 44):
        raw = leiden(graph, seed, 'weight' if weight is not None else None)
        labels[seed], moves[seed] = rs.rto.merge_small(raw[g], xp[g], min_rows)
        painting_labels(labels[seed], g)
    local = labels[42]
    rec = describe(local, g, content)
    rec.update(dropped=rec['n_groups'] < 2, reason='fewer than 2 groups after merge' if rec['n_groups'] < 2 else None,
               weight=weight, merge_moves=moves,
               graph={'k': 20, 'vertices': graph.vcount(), 'edges': graph.ecount(),
                      'resolution': 1.0, 'merge_min_rows': min_rows},
               stability_seeds={str(s): {'n_groups': len(np.unique(labels[s])),
                                        'partition_sha256': sha_array(labels[s])} for s in labels},
               stability_ami={f'{a}_{b}': float(ami(labels[a], labels[b])) for a, b in ((42, 43), (42, 44), (43, 44))})
    return local, rec, graph


def random_control(local, g, k):
    p = painting_labels(local, g)
    sizes = np.bincount(p)
    perm = np.random.default_rng(7000 + k).permutation(len(p))
    z = np.empty(len(p), dtype=np.int64)
    z[perm] = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    assert np.array_equal(np.bincount(z), sizes), 'L8 painting sizes'
    return z[g]


def build(only=IDS, out=None, tokstats=TOKSTATS, trial=False):
    C.assert_rule()
    C.assert_threads()
    only = set(only)
    assert only and only <= set(IDS), 'unknown grouping IDs'
    assert not trial or out is not None, 'trial requires explicit output directory'
    out = Path(out) if out is not None else C.RESULTS / 'groupings'
    targets = [out / f'r7_groupings.{ext}' for ext in ('npz', 'json')]
    assert all(not p.exists() or C.in_smoke(p) for p in targets), 'output exists (write once)'
    rs = C.load('run_step1')
    data, _, st_full, _, g_full = rs.base_data()
    assert st_full.dtype == np.int64 and st_full.shape == (183694,) and (np.diff(st_full) > 0).all(), 'scorer-train rows'
    assert len(first_rows(g_full)) == 36518, 'painting count'
    needed = only | {f'V{name[1:]}' for name in only if name.startswith('Z')}
    sources, provenance = {}, {}
    r2i = None
    if needed & {'V1', 'V2', 'V3', 'V4'}:
        sources['CSD'], r2i, provenance['CSD'] = read_source(STYLE, data, rs)
    if needed & {'V5', 'V6'} or trial:
        sources['tokens'], r2it, provenance['tokens'] = read_source(Path(tokstats), data, rs, True, trial)
        assert r2i is None or np.array_equal(r2i, r2it), 'source joins differ'
        r2i = r2it
    mask = r2i[st_full] < len(sources['tokens']) if trial else np.ones(len(st_full), dtype=bool)
    st = st_full[mask]
    g = np.unique(g_full[mask], return_inverse=True)[1].astype(np.int64)
    arrays = {'scorer_train': st, 'g_st': g}
    content_path = C.path_of('20261031_pseudo_partitions/results/partitions.npz')
    assert C.sha_file(content_path) == C.INPUTS['20261031_pseudo_partitions/results/partitions.npz'], 'E2 SHA'
    with np.load(content_path, allow_pickle=False) as z:
        assert {'image', 'local_groups'} <= set(z.files), 'E2 keys'
        assert np.array_equal(z['local_groups'], g_full), 'E2 alignment'
        assert z['image'].dtype == np.int64 and z['image'].shape == st_full.shape, 'content shape/dtype'
        assert len(np.unique(z['image'])) == 64, 'L2 content count'
        painting_labels(z['image'], g_full)
        c = z['image'][mask][first_rows(g)]
    vectors, transforms, records, graphs = {}, {}, {}, {}
    if 'CSD' in sources:
        vectors['CSD'], provenance['CSD']['painting_vectors'] = painting_vectors(sources['CSD'], r2i, st, g)
    if needed & {'V1', 'V4'}:
        vectors['OrthCSD'], transforms['OrthCSD'] = orth(vectors['CSD'], c)
    if needed & {'V5', 'V6'}:
        tokens, provenance['tokens']['painting_vectors'] = painting_vectors(sources['tokens'], r2i, st, g)
        u, transforms['U'] = preprocess_u(tokens)
        if 'V5' in needed:
            clip = np.asarray(data.img_features)[st[first_rows(g)]]
            vectors['ridgeU'], transforms['ridgeU'] = ridge_residual(u, clip, rs.rc.unit)
        if 'V6' in needed:
            vectors['OrthU'], transforms['OrthU'] = orth(u, c)
    recipes = {'V1': ('OrthCSD', None), 'V2': ('CSD', 0.0), 'V3': ('CSD', 0.5),
               'V4': ('OrthCSD', 0.0), 'V5': ('ridgeU', None), 'V6': ('OrthU', None)}
    for name in IDS[:6]:
        if name not in needed:
            continue
        key, weight = recipes[name]
        print(f'building {name} ({len(first_rows(g))} paintings)', flush=True)
        local, rec, graph = group_vectors(vectors[key], g, c, rs, weight, graphs.get(key))
        graphs[key] = graph
        rec['vector_source'] = key
        records[name] = rec
        if name in only and not rec['dropped']:
            arrays[name] = local
        zk = f'Z{name[1:]}'
        if zk in only:
            zlocal = random_control(local, g, int(name[1:]))
            records[zk] = {**describe(zlocal, g, c), 'dropped': rec['dropped'], 'seed': 7000 + int(name[1:]),
                           'matched_variant': name, 'reason': rec['reason']}
            if not rec['dropped']:
                arrays[zk] = zlocal
    if 'R0' in only:
        key = '20261116_grouping_step1_style/results/step1_group_style.npz'
        path = C.path_of(key)
        assert C.sha_file(path) == C.INPUTS[key], 'R0 SHA'
        with np.load(path, allow_pickle=False) as z:
            assert {'scorer_train', 'g_st', 'style_csd', 'style_csd__seed43', 'style_csd__seed44'} <= set(z.files), 'R0 keys'
            assert np.array_equal(z['scorer_train'], st_full) and np.array_equal(z['g_st'], g_full), 'R0 alignment'
            assert z['style_csd'].dtype == np.int64 and z['style_csd'].shape == st_full.shape, 'R0 shape/dtype'
            assert len(np.unique(z['style_csd'])) == 17, 'R0 group count'
            painting_labels(z['style_csd'], g_full)
            arrays['R0'] = np.unique(z['style_csd'][mask], return_inverse=True)[1].astype(np.int64) if trial else z['style_csd']
            seeds = {42: z['style_csd']}
            for seed in (43, 44):
                seeds[seed] = z[f'style_csd__seed{seed}']
                painting_labels(seeds[seed], g_full)
            stability = {f'{a}_{b}': float(ami(seeds[a][mask], seeds[b][mask]))
                         for a, b in ((42, 43), (42, 44), (43, 44))}
        records['R0'] = {**describe(arrays['R0'], g, c), 'dropped': False, 'source_sha256': C.INPUTS[key]}
        records['R0']['stability_ami'] = stability
    if 'T' in only:
        t = rs.artelingo_aspect_labels(data)['style'][st_full]
        assert t.dtype == np.int64 and t.shape == st_full.shape and (t >= 0).all(), 'T missing code/shape'
        painting_labels(t, g_full)
        assert len(np.unique(t)) == 27 and np.bincount(t).min() >= 200, 'L9 T count/min rows'
        arrays['T'] = np.unique(t[mask], return_inverse=True)[1].astype(np.int64) if trial else t
        records['T'] = {**describe(arrays['T'], g, c), 'dropped': False}
    C.write_npz_once(targets[0], **arrays)
    return C.write_json_once(targets[1], {'trial': trial, 'requested': sorted(only), 'groupings': records,
        'sources': provenance, 'transforms': transforms, 'scorer_train_sha256': sha_array(st),
        'g_st_sha256': sha_array(g), 'n_rows': len(st), 'n_paintings': len(first_rows(g))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', default=list(IDS))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--tokstats', type=Path, default=TOKSTATS)
    parser.add_argument('--trial', action='store_true')
    args = parser.parse_args()
    only = [name for part in args.only for name in part.split(',')]
    build(only, args.out, args.tokstats, args.trial)


if __name__ == '__main__':
    main()
