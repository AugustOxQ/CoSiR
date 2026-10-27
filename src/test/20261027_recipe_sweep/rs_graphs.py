"""Graph variants of the recipe sweep (ticket 03): k11 rebuilds and the ablation's graphs, with their checks.

The graphs are built by the claim test's graph code (`ct2_graphs.build`, read back by `ct2_d2.load_training_data`)
with only the neighbour count (k11) or the variant changed, at both scales:

  aspect        D2's graphs: Gres (emotion), Fres05 (style), R0o (genre); k replaces 20 in every mutual top-k, the
                aspect graphs and the graphs that set the per-node lambda (rule §2.2)
  uncorrected   the same without the content correction: R0 of emotion and of style (the plain mutual top-k of the
                aspect's primary view, as the D2 ablation's R8), R0o of genre (it has none) (rule §2.5)
  default_view  one default-view graph, Dview, for all three aspects (the D2 ablation's R9) (rule §2.5)

`ct2_graphs._partition` hard-codes 20; `_partition_k` is its copy with k (a patch cannot reach the `np.empty((n, 20))`
arrays), swapped in by a patch on the CT's module. The copy is checked three ways: the original's source hash is pinned
(`CT_PARTITION_SHA256`), a test compares the two on random data at k = 20, and the gates below compare every rebuilt
array with the frozen one.

  sample_store(k, variant)   what `rg_train.load_graphs` returns: (graphs, manifests), per aspect, read by the frozen
                             sample loader (`ms_train.head.load_graph`)
  full_graphs(k, variant)    what `ct2_d2.load_training_data('full', ...)` returns without s and inputs:
                             (graphs, manifests); `full_training_data` gives the CT's four values
  row_names(variant)         the row of each aspect in the graphs (the trainer's `base['graphs']`)

(k, variant) = (20, 'aspect') returns the frozen graphs themselves. Any other build refuses without its gate:
  k != 20 at the sample scale -> gates/k11_sample.json (check K); at full scale -> gates/k11_full.json (V4c)
  an ablation variant           -> gates/ablation_graphs.json (its item)
The stores sit under <outputs>/graphs/<scale>/<variant>__k<k>/ in the folder layout of the CT's loaders
(`d2/graphs/{emotion,style,genre}.npz`, `manifest.json`), plus `rs_build.json`.

CLI: rs_graphs build --scale {sample,full} --k K --variant V [--outputs DIR]
     rs_graphs check {k11-sample,k11-full,ablation} [--outputs DIR]
"""
import argparse
import hashlib
import importlib
import json
import subprocess
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import rs_paths
import ab_graphs
import ab_train
import ct2_d2
import ct2_data
import ct2_graphs
import rb_student
import rg_train

ags = ct2_graphs.ags
ms_train = ab_train.ms_train
ASPECTS = rs_paths.ASPECTS
VARIANTS = ('aspect', 'uncorrected', 'default_view')
FROZEN = (20, 'aspect')                       # the CT's graphs
ROWS = dict(aspect=dict(ct2_graphs.ROWS),
            uncorrected=dict(emotion='R0', style='R0', genre='R0o'),
            default_view=dict.fromkeys(ASPECTS, ab_graphs.DVIEW))
AGS_FROZEN = rs_paths.MAIN_TEST / '20261018_aspect_graph_sheet/graphs'      # sample: Gres, Fres05, R0o, R0, ...
AB_FROZEN = rs_paths.MAIN_TEST / '20261021_d2_ablation/graphs'              # sample: Dview
# The frozen sample graphs each variant is compared with: folder, row of each aspect.
REFERENCE = dict(aspect=(AGS_FROZEN, ROWS['aspect']), uncorrected=(AGS_FROZEN, ROWS['uncorrected']),
                 default_view=(AB_FROZEN, ROWS['default_view']))
# SHA-256 of `inspect.getsource(...)` of the CT's functions that `_partition_aspect` and `_mutual` copy: if one
# changes, the copy needs a review (a test pins both).
CT_PARTITION_SHA256 = '8d81a5e267ce2b874698a9e51f2db110ef3a40fb5136d9c0088286c437c2465a'
CT_MUTUAL_SHA256 = '68991d4759677d3d7a17efab2ae65f964e0eea446e5ee8897ff7e05c8af29c83'
# This module and the frozen modules that shape the arrays or their loading (the gates are stamped with their hashes).
CODE_MODULES = ('rs_graphs', 'ct2_graphs', 'ct2_d2', 'ct2_data', 'ab_graphs', 'ags_graphs', 'apc_data', 'pbp_graphs',
                'itp_data', 'ags_head')


class GraphRefusal(RuntimeError):
    """A build or load that its gate does not allow."""


def _mutual(x, k):
    """Copy of ct2_graphs._mutual with k in place of 20."""
    nn = np.empty((len(x), k), dtype=np.int64)
    for start in range(0, len(x), 256):
        nn[start:start + 256] = ags.top(x[start:start + 256] @ x.T, start, k)
    return ags.reciprocal(nn)


def _partition_aspect(spaces, nodes, k):
    """Copy of ct2_graphs._partition with k in place of 20 (the mutual top-ks, the neighbour arrays, ags.top)."""
    n = len(nodes)
    x = {key: np.ascontiguousarray(ags.l2(spaces[key][nodes])) for key in ct2_graphs.KEYS}
    content_pairs = _mutual(x['D'], k) | _mutual(x['I'], k)
    r0 = _mutual(x['E'], k)
    genre = _mutual(x['genre.img'], k)
    # Same anchor lambda rule as AGS build_partition, including zero-degree fallback.
    deg = np.bincount(ags.pair_array(r0).ravel(), minlength=n)
    shared_deg = np.bincount(ags.pair_array(r0 & content_pairs).ravel(), minlength=n)
    lam = np.zeros(n, dtype=np.float32)
    lam[deg > 0] = shared_deg[deg > 0] / deg[deg > 0]
    lam[deg == 0] = lam[deg > 0].mean()
    draws = ags.draw_pairs(n)
    sd = {key: ags.similarity_sd(x[key], draws) for key in ('D', 'I', 'E', 'style.img')}
    nn = {a: np.empty((n, k), dtype=np.int64) for a in ('emotion', 'style')}
    for start in range(0, n, 256):
        end = min(start + 256, n)
        ds, ims = x['D'][start:end] @ x['D'].T, x['I'][start:end] @ x['I'].T
        content = (ds / sd['D'] + ims / sd['I']) / 2
        primary = (x['E'][start:end] @ x['E'].T) / sd['E']
        nn['emotion'][start:end] = ags.top(primary - lam[start:end, None] * content, start, k)
        primary = (x['style.img'][start:end] @ x['style.img'].T) / sd['style.img']
        nn['style'][start:end] = ags.top(primary - .5 * content, start, k)
    return dict(emotion=ags.reciprocal(nn['emotion']), style=ags.reciprocal(nn['style']),
                genre=genre)


def _partition_k(spaces, nodes, *, k, variant):
    """The CT's `_partition` of the variant at k: a dict aspect -> set of local pairs (a < b)."""
    if variant == 'aspect':
        return _partition_aspect(spaces, nodes, k)
    x = {key: np.ascontiguousarray(ags.l2(spaces[key][nodes])) for key in ct2_graphs.KEYS}
    if variant == 'uncorrected':
        # R0 of AGS: the mutual top-k of the aspect's primary view (emotion: the instructed caption mean E, style:
        # the instructed image); genre's graph is R0o, as in the aspect variant.
        return dict(emotion=_mutual(x['E'], k), style=_mutual(x['style.img'], k), genre=_mutual(x['genre.img'], k))
    assert variant == 'default_view'
    # The D2 ablation's Dview (ab_graphs.dview_pairs, unchanged): the CT's content score alone, one graph for all.
    pairs, _ = ab_graphs.dview_pairs(x['D'], x['I'], k)
    pairs = set(map(tuple, pairs.tolist()))
    return dict.fromkeys(ASPECTS, pairs)


def code_files():
    """SHA-256 of each file of the graph code: this module and the frozen modules it drives."""
    files = {m: Path(__file__) if m == 'rs_graphs' else Path(importlib.import_module(m).__file__)
             for m in CODE_MODULES}
    return {m: hashlib.sha256(path.read_bytes()).hexdigest() for m, path in files.items()}


def code_sha256():
    return hashlib.sha256(''.join(code_files().values()).encode()).hexdigest()


def _method(k, variant):
    return f'{ct2_graphs.METHOD}; rs_graphs variant={variant} k={k}; v1'


def enter_patches(stack, k, variant, *, force=False):
    """Swap the CT's graph code for the variant at k. Enters none for the CT's own graphs (k = 20, 'aspect') unless
    force (the checks rebuild them with `_partition_k`)."""
    assert variant in VARIANTS and isinstance(k, int) and k >= 2, (k, variant)
    if (k, variant) == FROZEN and not force:
        return
    stack.enter_context(patch.object(ct2_graphs, '_partition', partial(_partition_k, k=k, variant=variant)))
    # The comparison with the frozen arrays is the gate's, after the build: the CT's own is for D2's three rows only.
    stack.enter_context(patch.object(ct2_graphs, '_check_sample', lambda graphs: None))
    if (k, variant) != FROZEN:
        stack.enter_context(patch.object(ct2_graphs, 'METHOD', _method(k, variant)))
    if variant != 'aspect':
        stack.enter_context(patch.object(ct2_graphs, 'ROWS', dict(ROWS[variant])))


def store_root(outputs, scale, k, variant, *, check=False):
    """A check builds its own store, under graphs/<scale>/check/, so that no earlier build stands in for it."""
    assert scale in ('sample', 'full')
    return Path(outputs) / 'graphs' / scale / ('check' if check else '.') / f'{variant}__k{k}'


def _atomic_json(path, value):
    ags.atomic_json(Path(path), value)


def _git_state():
    def git(*args):
        return subprocess.run(['git', *args], cwd=rs_paths.ROUND, capture_output=True, text=True, check=True).stdout.strip()
    # code_dirty: any tracked or untracked .py file of the repository differs from HEAD.
    return dict(code_commit=git('rev-parse', 'HEAD'), code_dirty=bool(git('status', '--porcelain', '--', ':/*.py')))


def require_absent(root):
    """A check starts from nothing: an existing store was built by an earlier run, whatever its code."""
    if (root / 'rs_build.json').exists() or (root / 'd2').exists():
        raise GraphRefusal(f'{root} exists: move it aside (a check builds its own store)')


def _require_own_store(root, scale, k, variant):
    """An existing store is read only if this code and this request built it (the record is written after the build)."""
    if not (root / 'd2/graphs/manifest.json').exists():
        return
    record_path = root / 'rs_build.json'
    if not record_path.exists():
        raise GraphRefusal(f'{root} has a manifest but no rs_build.json: move it aside')
    record = json.loads(record_path.read_text())
    if record['code_sha256'] != code_sha256():
        raise GraphRefusal(f'{root}: store built by other graph code: move it aside')
    assert (record['scale'], record['k'], record['variant']) == (scale, k, variant), f'{record_path}: another store'


def _materialize(scale, k, variant, outputs, *, force=False, check=False):
    """Build (when the store is absent) and read the variant's graphs by the CT's `load_training_data`. Returns
    (root, (s, inputs, graphs, manifests)). The gates are the callers'. A check (check=True) needs an absent store."""
    root = store_root(outputs, scale, k, variant, check=check)
    if check:
        require_absent(root)
    _require_own_store(root, scale, k, variant)
    wall = time.monotonic()
    with ExitStack() as stack, threadpool_limits(limits=8):
        enter_patches(stack, k, variant, force=force)
        data = ct2_d2.load_training_data(scale, root)
    manifest = json.loads((root / 'd2/graphs/manifest.json').read_text())
    record_path = root / 'rs_build.json'
    if not record_path.exists():
        _atomic_json(record_path, dict(scale=scale, k=k, variant=variant, rows=ROWS[variant],
                                       archives=manifest['archives'], build_seconds=manifest['seconds'],
                                       call_seconds=time.monotonic() - wall, code_sha256=code_sha256(),
                                       **_git_state()))
    else:
        assert json.loads(record_path.read_text())['archives'] == manifest['archives'], f'{record_path}: store changed'
    return root, data


# ---------------------------------------------------------------- gates

def gate_path(outputs, name):
    return Path(outputs) / 'gates' / f'{name}.json'


def require_gate(outputs, name, item=None):
    """Refuse (GraphRefusal, with the reason) unless the gate file passes for the current graph code."""
    path = gate_path(outputs, name)
    if not path.is_file():
        raise GraphRefusal(f'{path} is missing: run `rs_graphs check ...` first')
    gate = json.loads(path.read_text())
    if gate['code_sha256'] != code_sha256():
        raise GraphRefusal(f'{path} was written by other graph code: rerun the check')
    if item is None:
        ok = gate['pass']
    else:
        ok = gate['items'][item]['pass']
    if not ok:
        raise GraphRefusal(f'{path}{"" if item is None else " item " + item} failed: this build does not run')


def require_gates(outputs, scale, k, variant):
    if k != FROZEN[0]:
        require_gate(outputs, 'k11_sample' if scale == 'sample' else 'k11_full')
    if variant != 'aspect':
        require_gate(outputs, 'ablation_graphs', variant)


# ---------------------------------------------------------------- the loaders the trainers swap in

def sample_store(k=20, variant='aspect', *, s=None, outputs=rs_paths.OUTPUTS):
    """(graphs, manifests) per aspect as `rg_train.load_graphs(s, recipe)` returns them. `s` is the frozen sample
    (`ab_train.load_sample()` when None). The trainer's row of each aspect is `row_names(variant)`."""
    s = ab_train.load_sample() if s is None else s
    if (k, variant) == FROZEN:
        return rg_train.load_graphs(s, rg_train.resolved(0))
    require_gates(outputs, 'sample', k, variant)
    root, _ = _materialize('sample', k, variant, outputs)
    graphs, manifests = {}, {}
    for aspect in ASPECTS:
        # The frozen sample loader, on this store's folder (<root>/d2/graphs).
        graphs[aspect], manifests[aspect] = ms_train.head.load_graph(s, root / 'd2', aspect, [ROWS[variant][aspect]])
    return graphs, manifests


def full_training_data(k=20, variant='aspect', *, outputs=rs_paths.OUTPUTS):
    """The CT's (s, inputs, graphs, manifests) of the full scale, as `rb_student.load_scale('full')` returns them."""
    if (k, variant) == FROZEN:
        return rb_student.load_scale('full')
    require_gates(outputs, 'full', k, variant)
    return _materialize('full', k, variant, outputs)[1]


def full_graphs(k=20, variant='aspect', *, outputs=rs_paths.OUTPUTS):
    """(graphs, manifests) of the CT's full-scale loader. The trainer's row of each aspect is `row_names(variant)`."""
    return full_training_data(k, variant, outputs=outputs)[2:]


def row_names(variant):
    return dict(ROWS[variant])


# ---------------------------------------------------------------- comparisons with the frozen arrays

def _sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def _keys(rows_by_aspect, aspect):
    row = rows_by_aspect[aspect]
    return ['fit_node_ids', 'dev_node_ids'] + [f'{row}__{p}{w}' for p in ('fit', 'dev') for w in ('', '_w')]


def compare_arrays(frozen, rebuilt, keys):
    """Per key: both arrays exist and are equal in dtype, shape and every value (hashes of both given)."""
    items = {}
    with np.load(frozen, allow_pickle=False) as f, np.load(rebuilt, allow_pickle=False) as r:
        for key in keys:
            if key not in f.files or key not in r.files:
                items[key] = dict(equal=False, reason='missing', in_frozen=key in f.files, in_rebuilt=key in r.files)
                continue
            a, b = f[key], r[key]
            equal = a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b))
            items[key] = dict(equal=equal, dtype=str(b.dtype), shape=list(b.shape), frozen_sha256=_sha_array(a),
                              rebuilt_sha256=_sha_array(b))
    return items


def _write_gate(outputs, name, passed, items, extra, started):
    gate = dict(name=name, **{'pass': bool(passed)}, items=items, code_sha256=code_sha256(),
                code_files=code_files(),
                seconds=time.monotonic() - started, **_git_state(), **extra)
    _atomic_json(gate_path(outputs, name), gate)
    return gate


def _frozen_archive_ok(folder, aspect):
    """The frozen archive equals its manifest's checksum (the CT's `_check_sample` does this before comparing)."""
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert ags.sha_file(folder / f'{aspect}.npz') == manifest['archives'][aspect], f'{folder}/{aspect}: checksum'
    return manifest['archives'][aspect]


def _compare_variant(scale_root, folder, rows):
    """Rebuilt `scale_root/d2/graphs` against frozen `folder` for the rows of each aspect."""
    items, frozen_archives = {}, {}
    for aspect in ASPECTS:
        frozen_archives[aspect] = _frozen_archive_ok(folder, aspect)
        items[aspect] = compare_arrays(folder / f'{aspect}.npz', scale_root / f'd2/graphs/{aspect}.npz',
                                       _keys(rows, aspect))
    ok = all(v['equal'] for per in items.values() for v in per.values())
    return dict(rows=rows, arrays=items, frozen_archives=frozen_archives, **{'pass': ok})


def _loaded_equal(s, root, frozen_base, rows):
    """Per aspect: the rebuilt store read by the frozen sample loader equals the frozen store read by it (node ids,
    pairs, weights, dtypes)."""
    out = {}
    for aspect in ASPECTS:
        graph, _ = ms_train.head.load_graph(s, root / 'd2', aspect, [rows[aspect]])
        frozen, _ = ms_train.head.load_graph(s, frozen_base, aspect, [rows[aspect]])
        same = set(graph) == set(frozen) and all(
            np.array_equal(frozen[f'{p}_node_ids'], graph[f'{p}_node_ids']) for p in ('fit', 'dev'))
        for part in ('fit', 'dev'):
            for a, b in zip(frozen['rows'][rows[aspect]][part], graph['rows'][rows[aspect]][part]):
                same = same and a.dtype == b.dtype and np.array_equal(a, b)
        out[aspect] = bool(same)
    return out


def check_k11_sample(outputs):
    """Check K (rule §2.2): the sample rebuild at k = 20 equals the frozen sample graphs bit for bit."""
    started = time.monotonic()
    root, _ = _materialize('sample', 20, 'aspect', outputs, force=True, check=True)
    folder, rows = REFERENCE['aspect']
    result = _compare_variant(root, folder, rows)
    # And through the loaders: the rebuilt store read by the frozen sample loader equals what rg_train.load_graphs reads.
    loaded = _loaded_equal(ab_train.load_sample(), root, folder.parent, rows)
    passed = result['pass'] and all(loaded.values())
    return _write_gate(outputs, 'k11_sample', passed, dict(arrays=result['arrays'], loaded_equal=loaded),
                       dict(scale='sample', k=20, variant='aspect', store=str(root), rows=rows,
                            frozen_archives=result['frozen_archives'],
                            note='the rows other than the three D2 rows of the frozen AGS archives are not rebuilt'),
                       started)


def check_k11_full(outputs):
    """V4c: the full-scale rebuild at k = 20 equals the CT's 42,969-node graphs bit for bit (arrays, archives, manifest)."""
    started = time.monotonic()
    root, _ = _materialize('full', 20, 'aspect', outputs, force=True, check=True)
    folder = rs_paths.CT_FULL / 'd2/graphs'
    frozen_manifest = json.loads((folder / 'manifest.json').read_text())
    manifest = json.loads((root / 'd2/graphs/manifest.json').read_text())
    items = {}
    for aspect in ASPECTS:
        with np.load(folder / f'{aspect}.npz', allow_pickle=False) as z:
            keys = sorted(z.files)
        assert ags.sha_file(folder / f'{aspect}.npz') == frozen_manifest['archives'][aspect]
        items[aspect] = compare_arrays(folder / f'{aspect}.npz', root / f'd2/graphs/{aspect}.npz', keys)
        with np.load(root / f'd2/graphs/{aspect}.npz', allow_pickle=False) as z:
            assert sorted(z.files) == keys, f'{aspect}: array names differ'
    manifest_equal = {key: manifest[key] == frozen_manifest[key]
                      for key in ('schema', 'identity', 'archives', 'counts', 'array_sha256', 'source_sha256')}
    passed = (all(v['equal'] for per in items.values() for v in per.values()) and all(manifest_equal.values()))
    return _write_gate(outputs, 'k11_full', passed, dict(arrays=items, manifest_equal=manifest_equal),
                       dict(scale='full', k=20, variant='aspect', store=str(root),
                            frozen_archives=frozen_manifest['archives'], rebuilt_archives=manifest['archives'],
                            frozen_build_seconds=frozen_manifest['seconds'], rebuilt_build_seconds=manifest['seconds']),
                       started)


def check_ablation(outputs):
    """Rule §2.5: at the sample scale the default_view build equals the D2 ablation's frozen Dview graph and the
    uncorrected build equals the AGS sheet's R0 / R0o rows, bit for bit."""
    started = time.monotonic()
    items, built = {}, {}
    variants = ('default_view', 'uncorrected')
    for variant in variants:        # both stores absent before the first build
        require_absent(store_root(outputs, 'sample', 20, variant, check=True))
    s = ab_train.load_sample()
    for variant in variants:
        root, _ = _materialize('sample', 20, variant, outputs, force=True, check=True)
        folder, rows = REFERENCE[variant]
        items[variant] = _compare_variant(root, folder, rows)
        # And through the loader: the rebuilt store read by the frozen sample loader equals the frozen store read by it.
        loaded = _loaded_equal(s, root, folder.parent, rows)
        items[variant].update(loaded_equal=loaded, **{'pass': items[variant]['pass'] and all(loaded.values())})
        built[variant] = str(root)
    passed = all(item['pass'] for item in items.values())
    return _write_gate(outputs, 'ablation_graphs', passed, items,
                       dict(scale='sample', k=20, stores=built), started)


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='build (or read) one store; refuses without its gate')
    b.add_argument('--scale', required=True, choices=['sample', 'full'])
    b.add_argument('--k', required=True, type=int)
    b.add_argument('--variant', default='aspect', choices=VARIANTS)
    c = sub.add_parser('check', help='the reproduction checks; write <outputs>/gates/<gate>.json')
    c.add_argument('which', choices=['k11-sample', 'k11-full', 'ablation'])
    for p in (b, c):
        p.add_argument('--outputs', type=Path, default=rs_paths.OUTPUTS)
    args = parser.parse_args()
    outputs = rs_paths.check_outputs(args.outputs)
    rs_paths.check_imports()
    if args.command == 'build':
        started = time.monotonic()
        try:
            require_gates(outputs, args.scale, args.k, args.variant)
            root, _ = _materialize(args.scale, args.k, args.variant, outputs, force=(args.k, args.variant) == FROZEN)
        except GraphRefusal as error:
            raise SystemExit(f'REFUSED: {error}')
        record = json.loads((root / 'rs_build.json').read_text())
        print(f'{args.scale} {args.variant} k={args.k}: {root}; build {record["build_seconds"]:.1f}s; '
              f'this call {time.monotonic() - started:.1f}s (providers, build or read, checks of the CT loader)',
              flush=True)
        return
    check = dict({'k11-sample': check_k11_sample, 'k11-full': check_k11_full, 'ablation': check_ablation})[args.which]
    try:
        gate = check(outputs)
    except GraphRefusal as error:
        raise SystemExit(f'REFUSED: {error}')
    print(f'{gate["name"]}: {"PASS" if gate["pass"] else "FAIL"} ({gate["seconds"]:.1f}s) '
          f'{gate_path(outputs, gate["name"])}', flush=True)
    raise SystemExit(0 if gate['pass'] else 1)


if __name__ == '__main__':
    main()
