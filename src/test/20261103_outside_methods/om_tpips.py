"""r18 ticket 03: TPIPS-Embed-Qwen3VL-8B image vectors per factor, stores, reader, gate (iv).

Spec D2 (model, factors), D18 (stores), D20 (batching), D23 and §3.5 gate (iv). Plumbing: asserts at every file
boundary, no unit tests. Run in the tpips venv (`om_paths.TPIPS_ENV`); `ns_data.keys` imports there.

Flow (every step is a CLI sub-command, all plain so that DAS6 (ticket 04) can run `extract` and `merge` alone):

  prepare   writes OUT/tpips_inputs/<set>.json for sets artelingo (held + val), mmart (test) and probe (the 48 probe
            images): keys, image paths, visual token counts (from the image size), the length-sorted order.
            The only step that needs r17's key lists; DAS6 gets the json files and the images.
  extract   loads the model once; for the shard's keys (every n-th key of the length-sorted list: `--shard i --of n`)
            embeds each image under each factor, in batches of `--batch` (1 or more; same length-sorted order), in
            chunks of CHUNK images saved as it goes (a rerun skips finished chunks). Writes
            <store>/<factor>/shards/<i>of<n>/{emb.npy, keys.json, meta.json}. `--image-root-map OLD=NEW` rewrites the
            prefix of every image path (DAS6).
  merge     joins the n shards of a (set, factor): <store>/<factor>/{emb.npy, keys.json, meta.json}, rows in the
            manifest's key order, every key once, unit rows, sha256 of emb.npy in meta.json.
  gate-iv   probe stores of each path against the local batch-1 probe store (cosine >= 0.999 per row), then the real
            stores' keys, unit rows and sha256. Writes OUT/stage0/gate_iv.json.

Reader: `load(dataset, factor, keys) -> float32 [n, 4096]`, joined by key.

Store roots: `om_paths.TPIPS_STORE[set]` (smoke, OM_SMOKE=1: OUT/tpips_store_smoke/<set>); `--store-root` overrides it.
Batching (D20): the default is batch 1. Batch > 1 is allowed only after gate (iv) passes for that path; the mode is
recorded in every meta.json. Left padding or right padding does not matter to the model's last-token pooling.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import om_paths as P  # noqa: E402

FACTORS = tuple(P.TPIPS_FACTORS.values())        # emotion, style, genre
WIDTH = P.TPIPS_WIDTH
CHUNK = 256                                      # images per saved chunk
TOL_UNIT = 1e-5                                  # stored rows are renormalised in float64, then cast to float32
PATCH_FACTOR = 32                                # Qwen3-VL: patch 16 times merge 2
MIN_PIXELS, MAX_PIXELS = 4 * 32 * 32, 1800 * 32 * 32     # tpips.embedding.qwen3_vl_embed
SETS = ('artelingo', 'mmart', 'probe')
ARTELINGO_INDEX = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b/index.json')   # painting -> file
ARTELINGO_ROOT = '/data/PDD/wikiart_proj/wikiart'
MMART_INDEX = '/data/SSD/mmart/mmart_index.parquet'


def inputs_dir():
    return P.OUT / 'tpips_inputs'


def store_root(set_, override=None):
    if override:
        return Path(override)
    if P.SMOKE:
        return P.OUT / 'tpips_store_smoke' / set_
    assert set_ in P.TPIPS_STORE, f'no default store for {set_}: pass --store-root'
    return P.TPIPS_STORE[set_]


def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and np.isfinite(x).all()
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _check_unit(x, what):
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == WIDTH and np.isfinite(x).all(), (what, x.shape)
    dev = float(np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1).max()) if len(x) else 0.0
    assert dev < TOL_UNIT, f'{what}: rows not unit norm (max deviation {dev})'


# ----------------------------------------------------------------------------------------------- prepare
def visual_tokens(width, height):
    """Visual token count of an image, from Qwen's resize rule (the length-sort key; D20)."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    h, w = smart_resize(height, width, factor=PATCH_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return (h // 16) * (w // 16) // 4


def _manifest(set_, keys, paths):
    from PIL import Image
    assert len(keys) == len(set(keys)) == len(paths)
    tokens = []
    for p in paths:
        assert os.path.exists(p), p
        with Image.open(p) as im:
            tokens.append(visual_tokens(*im.size))
    order = sorted(range(len(keys)), key=lambda i: (tokens[i], keys[i]))     # length-sorted, ties by key
    return {'set': set_, 'keys': list(keys), 'image': list(paths), 'tokens': tokens, 'order': order}


def prepare():
    import ns_data
    import pyarrow.parquet as pq
    idx = {r['painting']: r['image'] for r in json.loads(ARTELINGO_INDEX.read_text())['images']}
    held, val = ns_data.keys('artelingo', 'held'), ns_data.keys('artelingo', 'val')
    assert (len(held), len(val)) == (12281, 6152) and not set(held) & set(val)
    keys = held + val
    out = {'artelingo': (keys, [f'{ARTELINGO_ROOT}/{idx[k]}' for k in keys])}      # KeyError = a painting w/o file
    tab = {r['key']: r['image_path'] for r in pq.read_table(MMART_INDEX, columns=['key', 'image_path']).to_pylist()}
    mm = ns_data.keys('mmart', 'test')
    assert len(mm) == 16384
    out['mmart'] = (mm, [tab[k] for k in mm])
    pk = sorted(k for k, s in ((r['key'], r['split']) for r in pq.read_table(
        MMART_INDEX, columns=['key', 'split']).to_pylist()) if s == 'new')[:P.TPIPS_PROBE_N]     # as om_tpips_probe
    assert len(pk) == P.TPIPS_PROBE_N
    out['probe'] = (pk, [tab[k] for k in pk])
    total = 0
    for s, (k, paths) in out.items():
        m = _manifest(s, k, paths)
        _write_json(inputs_dir() / f'{s}.json', m)
        print(s, len(k), 'tokens mean', round(float(np.mean(m['tokens'])), 1), 'max', max(m['tokens']))
        total += len(k) if s != 'probe' else 0
    assert total == P.N_TPIPS_IMAGES, total


def load_manifest(set_, path=None):
    m = json.loads(Path(path or inputs_dir() / f'{set_}.json').read_text())
    n = len(m['keys'])
    assert m['set'] == set_ and len(m['image']) == len(m['tokens']) == len(m['order']) == n
    assert len(set(m['keys'])) == n and sorted(m['order']) == list(range(n))
    return m


def shard_indices(manifest, i, n_shards):
    """Manifest rows of shard i of n: every n-th row of the length-sorted list."""
    assert 0 <= i < n_shards
    return manifest['order'][i::n_shards]


def _map_root(path, root_map):
    if root_map:
        old, new = root_map
        assert path.startswith(old), f'{path} does not start with {old}'
        return new + path[len(old):]
    return path


# ----------------------------------------------------------------------------------------------- extract
class FakeModel:
    """Smoke only (--fake): hash-seeded vectors per (image size, factor), no GPU."""
    def embed(self, images, factor, normalized=True):
        import zlib
        rows = []
        for im in images:
            r = np.random.RandomState(zlib.crc32(f'{im.size}{factor}'.encode()))
            rows.append(r.randn(WIDTH))
        import torch
        return torch.tensor(_unit(np.stack(rows)))


def load_model(device, fake=False, snapshot_dir=None):
    if fake:
        assert P.SMOKE, '--fake only under OM_SMOKE=1'
        return FakeModel(), 'fake'
    import tpips
    if snapshot_dir is None:
        from huggingface_hub import snapshot_download
        snapshot_dir = snapshot_download(P.TPIPS_REPO, local_files_only=True)
    assert Path(snapshot_dir).name == P.TPIPS_SNAPSHOT, f'snapshot {snapshot_dir} is not {P.TPIPS_SNAPSHOT}'
    return tpips.load_model('embedding', model_path=str(snapshot_dir), device=device), P.TPIPS_SNAPSHOT


def _embed(model, ims, factor):
    import torch
    with torch.inference_mode():
        e = model.embed(ims, factor=factor, normalized=True)
    e = e.float().cpu().numpy()
    assert e.shape == (len(ims), WIDTH), e.shape
    return _unit(e)


def extract(set_, factors, shard, n_shards, batch, device, root=None, root_map=None, manifest_path=None, limit=None,
            fake=False, snapshot_dir=None):
    from PIL import Image
    assert batch >= 1 and all(f in FACTORS for f in factors)
    man = load_manifest(set_, manifest_path)
    rows = shard_indices(man, shard, n_shards)
    if limit:
        rows = rows[:limit]
    base = store_root(set_, root)
    sdir = {f: base / f / 'shards' / f'{shard}of{n_shards}' for f in factors}
    for d in sdir.values():
        (d / 'chunks').mkdir(parents=True, exist_ok=True)
    model, snap = load_model(device, fake, snapshot_dir)
    t0 = time.time()
    for c0 in range(0, len(rows), CHUNK):
        crows = rows[c0:c0 + CHUNK]
        todo = [f for f in factors if not (sdir[f] / 'chunks' / f'{c0:07d}.npy').exists()]
        if not todo:
            continue
        ims = []
        for i in crows:
            with Image.open(_map_root(man['image'][i], root_map)) as im:
                ims.append(im.convert('RGB'))
        for f in todo:
            parts = [_embed(model, ims[b:b + batch], f) for b in range(0, len(ims), batch)]
            x = np.concatenate(parts)
            _write_json(sdir[f] / 'chunks' / f'{c0:07d}.json', [man['keys'][i] for i in crows])
            np.save(sdir[f] / 'chunks' / f'{c0:07d}.npy', x)
        print(f'{set_} shard {shard}/{n_shards} {c0 + len(crows)}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    for f in factors:                       # finish the shard: chunks -> emb.npy
        d = sdir[f]
        xs, ks = [], []
        for c0 in range(0, len(rows), CHUNK):
            xs.append(np.load(d / 'chunks' / f'{c0:07d}.npy'))
            ks += json.loads((d / 'chunks' / f'{c0:07d}.json').read_text())
        x = np.concatenate(xs) if xs else np.zeros((0, WIDTH), np.float32)
        assert ks == [man['keys'][i] for i in rows], 'chunk keys differ from the shard list'
        _check_unit(x, f'{set_}/{f} shard {shard}')
        np.save(d / 'emb.npy', x)
        _write_json(d / 'keys.json', ks)
        _write_json(d / 'meta.json', {
            'set': set_, 'factor': f, 'snapshot': snap, 'shard': shard, 'of': n_shards, 'n': len(ks),
            'batch': batch, 'batch_mode': 'batch 1' if batch == 1 else f'length-sorted batches of {batch}',
            'host': platform.node(), 'device': device, 'fake': bool(fake), 'limit': limit,
            'sha256': P.sha256_file(d / 'emb.npy')})


# ----------------------------------------------------------------------------------------------- merge
def merge(set_, factors, n_shards, root=None, manifest_path=None, allow_partial=False):
    man = load_manifest(set_, manifest_path)
    base = store_root(set_, root)
    pos = {k: i for i, k in enumerate(man['keys'])}
    for f in factors:
        xs, ks, shards = [], [], []
        for i in range(n_shards):
            d = base / f / 'shards' / f'{i}of{n_shards}'
            meta = json.loads((d / 'meta.json').read_text())
            x = np.load(d / 'emb.npy')
            assert P.sha256_file(d / 'emb.npy') == meta['sha256'] and meta['of'] == n_shards and meta['shard'] == i
            assert meta['snapshot'] in (P.TPIPS_SNAPSHOT, 'fake') and meta['factor'] == f and meta['set'] == set_
            k = json.loads((d / 'keys.json').read_text())
            assert len(k) == len(x) == meta['n']
            _check_unit(x, f'{d}')
            xs.append(x)
            ks += k
            shards.append({k_: meta[k_] for k_ in ('shard', 'of', 'n', 'batch', 'batch_mode', 'host', 'device',
                                                   'sha256', 'fake')})
        assert len(ks) == len(set(ks)), f'{set_}/{f}: a key twice across shards'
        if not allow_partial:
            assert set(ks) == set(man['keys']), f'{set_}/{f}: shards miss {len(set(man["keys"]) - set(ks))} keys'
        assert set(ks) <= set(pos), f'{set_}/{f}: shard keys outside the manifest'
        order = np.argsort([pos[k] for k in ks], kind='stable')
        x = np.concatenate(xs)[order]
        keys = [ks[j] for j in order]
        np.save(base / f / 'emb.npy', x)
        _write_json(base / f / 'keys.json', keys)
        batches = sorted({s['batch'] for s in shards})
        _write_json(base / f / 'meta.json', {
            'set': set_, 'factor': f, 'snapshot': shards and (P.TPIPS_SNAPSHOT if not shards[0]['fake'] else 'fake'),
            'n': len(keys), 'width': WIDTH, 'batch_mode': [s['batch_mode'] for s in shards][0] if len(batches) == 1
            else 'mixed', 'batches': batches, 'shards': shards, 'sha256': P.sha256_file(base / f / 'emb.npy'),
            'fake': any(s['fake'] for s in shards)})
        print(f'merged {set_}/{f}: {len(keys)} rows from {n_shards} shards')


# ----------------------------------------------------------------------------------------------- reader
def load(dataset, factor, keys, root=None):
    """float32 [len(keys), 4096] unit rows of `factor` for `keys`, joined by key (all present)."""
    assert factor in FACTORS, factor
    d = store_root(dataset, root) / factor
    meta = json.loads((d / 'meta.json').read_text())
    assert not meta.get('fake') or P.SMOKE, 'store made with --fake'
    ks = json.loads((d / 'keys.json').read_text())
    x = np.load(d / 'emb.npy', mmap_mode='r')
    assert x.shape == (len(ks), WIDTH) and len(ks) == len(set(ks)) == meta['n'], (d, x.shape)
    pos = {k: i for i, k in enumerate(ks)}
    missing = [k for k in keys if k not in pos]
    assert not missing, f'{dataset}/{factor}: {len(missing)} keys missing, first {missing[:3]}'
    out = np.asarray(x[[pos[k] for k in keys]], dtype=np.float32)
    _check_unit(out, f'{dataset}/{factor}')
    return out


# ----------------------------------------------------------------------------------------------- gate (iv)
def _read_store(d):
    meta = json.loads((d / 'meta.json').read_text())
    ks = json.loads((d / 'keys.json').read_text())
    x = np.load(d / 'emb.npy')
    assert x.shape == (len(ks), WIDTH) and P.sha256_file(d / 'emb.npy') == meta['sha256'], d
    return meta, ks, x


def gate_iv(paths, reference, stores=True):
    """paths: {name: probe store root}; reference: the local batch-1 probe store root."""
    ev = {'cos_min': P.TPIPS_COS, 'paths': {}, 'stores': {}, 'problems': []}
    ref = {f: _read_store(Path(reference) / f) for f in FACTORS}
    ref_metas = {f: ref[f][0] for f in FACTORS}
    for f, m in ref_metas.items():
        if m['batches'] != [1]:
            ev['problems'].append(f'reference {f} is not batch 1: {m["batches"]}')
        if m.get('fake'):
            ev['problems'].append('reference made with --fake')
        assert len(ref[f][1]) == P.TPIPS_PROBE_N, (f, len(ref[f][1]))
    ref_hosts = {s['host'] for m in ref_metas.values() for s in m['shards']}
    covered = {(h, 1) for h in ref_hosts}                  # (host, batch) signatures checked
    for name, root in paths.items():
        rows = {}
        for f in FACTORS:
            meta, ks, x = _read_store(Path(root) / f)
            rk, rx = ref[f][1], ref[f][2]
            if sorted(ks) != sorted(rk):
                ev['problems'].append(f'{name}/{f}: probe keys differ from the reference')
                continue
            pos = {k: i for i, k in enumerate(ks)}
            cos = np.einsum('ij,ij->i', x[[pos[k] for k in rk]].astype(np.float64), rx.astype(np.float64))
            rows[f] = {'min_cos': float(cos.min()), 'mean_cos': float(cos.mean()), 'n': len(cos),
                       'pass': bool(cos.min() >= P.TPIPS_COS)}
            if not rows[f]['pass']:
                ev['problems'].append(f'{name}/{f}: min cosine {cos.min():.6f} < {P.TPIPS_COS}')
            covered |= {(s['host'], s['batch']) for s in meta['shards']}
        ev['paths'][name] = rows
    if stores:
        import ns_data
        expect = {'artelingo': ns_data.keys('artelingo', 'held') + ns_data.keys('artelingo', 'val'),
                  'mmart': ns_data.keys('mmart', 'test')}
        for s, want in expect.items():
            for f in FACTORS:
                tag = f'{s}/{f}'
                try:
                    meta, ks, x = _read_store(store_root(s) / f)
                    _check_unit(x, tag)
                except (FileNotFoundError, AssertionError) as exc:
                    ev['stores'][tag] = {'pass': False, 'error': repr(exc)[:200]}
                    ev['problems'].append(f'{tag}: {exc!r}'[:200])
                    continue
                ok = len(ks) == len(set(ks)) == len(want) and set(ks) == set(want) and not meta.get('fake')
                sigs = {(sh['host'], sh['batch']) for sh in meta['shards']}
                uncovered = sorted(sigs - covered)
                ev['stores'][tag] = {'pass': bool(ok and not uncovered), 'n': len(ks), 'keys_ok': bool(ok),
                                     'signatures': sorted(map(list, sigs)), 'uncovered': [list(u) for u in uncovered]}
                if not ok:
                    ev['problems'].append(f'{tag}: key list is not the expected {len(want)} keys once each')
                if uncovered:
                    ev['problems'].append(f'{tag}: path(s) {uncovered} have no passing probe run')
    ev['pass'] = not ev['problems'] and bool(ev['paths'])
    out = P.OUT / 'stage0' / 'gate_iv.json'
    _write_json(out, ev)
    print('gate (iv):', 'PASS' if ev['pass'] else 'FAIL', *ev['problems'][:6], sep='\n  ')
    return ev


# ----------------------------------------------------------------------------------------------- CLI
def _factors(arg):
    fs = FACTORS if arg == 'all' else tuple(arg.split(','))
    assert all(f in FACTORS for f in fs), fs
    return fs


def _root_map(arg):
    if not arg:
        return None
    old, new = arg.split('=', 1)
    return old, new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare')
    e = sub.add_parser('extract')
    m = sub.add_parser('merge')
    for s in (e, m):
        s.add_argument('--set', required=True, choices=SETS)
        s.add_argument('--factor', default='all', help='all or comma list of emotion,style,genre')
        s.add_argument('--store-root')
        s.add_argument('--manifest', help='inputs json (default OUT/tpips_inputs/<set>.json)')
    e.add_argument('--shard', type=int, default=0)
    e.add_argument('--of', type=int, default=1)
    e.add_argument('--batch', type=int, default=1)
    e.add_argument('--device', default='cuda')
    e.add_argument('--image-root-map', help='OLD=NEW: rewrite this prefix of every image path')
    e.add_argument('--limit', type=int, help='first n rows of the shard (smoke)')
    e.add_argument('--fake', action='store_true', help='smoke only: no model')
    e.add_argument('--snapshot-dir')
    m.add_argument('--of', type=int, default=1)
    m.add_argument('--allow-partial', action='store_true', help='smoke only')
    g = sub.add_parser('gate-iv')
    g.add_argument('--path', action='append', default=[], metavar='NAME=ROOT',
                   help='probe store of a path under test (batched, das6); repeat')
    g.add_argument('--reference', help='local batch-1 probe store (default OUT/stage0/probe_b1)')
    g.add_argument('--skip-stores', action='store_true', help='smoke only')
    a = ap.parse_args(argv)
    if a.cmd == 'prepare':
        prepare()
    elif a.cmd == 'extract':
        extract(a.set, _factors(a.factor), a.shard, a.of, a.batch, a.device, a.store_root, _root_map(a.image_root_map),
                a.manifest, a.limit, a.fake, a.snapshot_dir)
    elif a.cmd == 'merge':
        merge(a.set, _factors(a.factor), a.of, a.store_root, a.manifest, a.allow_partial)
    else:
        paths = dict(p.split('=', 1) for p in a.path)
        ref = a.reference or (P.OUT / 'stage0' / 'probe_b1')
        ev = gate_iv(paths, ref, stores=not a.skip_stores)
        return 0 if ev['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
