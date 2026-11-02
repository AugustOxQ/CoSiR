"""r14 ticket 01: EmoSet unpack, key-list extractor for six views, parity (gate 0(i)) and store check (gate 0(ii)).

Spec 2026-11-02-backbones-sweep-design.md §3.3, §3.5 stage 0 (i)(ii), §9 (Budget); D8, D9, D17.

Views: qwen_default, emotion_focus, emotion_focus_photo (Qwen3-VL-Embedding-2B at the pinned revision, instructions as
r12), clip_l14, siglip2, siglip1 (encoder classes and batch sizes of the stored ArtELingo stores). Datasets: affection,
emoset. Every store is <folder>/{img.npy float16, keys.json, meta.json}, written through <folder>.pending, then renamed;
a finished or pending folder is never touched (a finished target is skipped by `extract-all`, a pending one stops it).

Targets (`targets()`):
  affection  siglip2 -> affection_local/siglip2_so400m (85,007 keys, the Qwen store's index order)
             siglip1 -> affection_local/siglip_so400m  (85,007 keys)
             clip_l14 -> affection_local/clip_l14_train (72,255 train keys; val+test are r12's clip_l14_valtest)
  emoset     all six views, 23,621 keys (val.json then test.json order) -> emoset/<folder of bs_paths>
EmoSet keys are the split-file path strings; the image is EMOSET_ROOT/<key>. No code reads a class from a key.

Commands (env /root/miniconda3/envs/CoSiR, PYTHONDONTWRITEBYTECODE=1; GPU commands under flock -n -o -E 75
/tmp/gpu0.lock; BS_OUTPUTS and BS_STORE_ROOT redirect a smoke run):
  python bs_extract.py unpack [--budget-seconds S]       idempotent; exit 3 when the budget ends first
  python bs_extract.py extract --dataset D --view V [--smoke N]
  python bs_extract.py extract-all                       the full target list, Affection first (skips sealed stores)
  python bs_extract.py parity [--cases a,b,..]           gate 0(i): OUT/stage0/parity_<case>.json; exit 1 on failure
  python bs_extract.py check [--smoke]                   gate 0(ii): OUT/stage0/stores.json; exit 1 on failure
--smoke N: N keys spread evenly over the target's key list; requires BS_STORE_ROOT (a scratch root), meta smoke true.
"""
import argparse
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / 'scripts'))
sys.path.insert(0, str(HERE.parent / '20261030_affection_confirmation'))
import bs_paths as BP  # noqa: E402
import bs_stores  # noqa: E402
import ac_paths  # noqa: E402  (puts every older round, e.g. ct2_*, rs_vh, on sys.path)
import ac_extract  # noqa: E402
import rs_vh  # noqa: E402

VIEWS = bs_stores.VIEWS
WIDTH = bs_stores.WIDTH
BATCH = {'qwen_default': 32, 'emotion_focus': 32, 'emotion_focus_photo': 32, 'clip_l14': 128, 'siglip2': 32,
         'siglip1': 32}        # scripts/extract_features.py BATCH (img) of each encoder
COSINE_MIN = 0.999
UNIT_TOL = 1e-3
N_AFF = {'all': 85007, 'train': 72255, 'val': 4251, 'test': 8501}
N_EMO = BP.EMOSET_SPLITS['val'] + BP.EMOSET_SPLITS['test']
STAGE0 = BP.OUT / 'stage0'


# ---------------------------------------------------------------- EmoSet unpack
def _safe_member(name):
    assert name.startswith(('image/', 'annotation/')) and '..' not in Path(name).parts and not name.startswith('/'), name
    return name


def read_split_entries(root=None, zf=None):
    """{'val': entries, 'test': entries}: each entry [label, image path, annotation path], from the unpacked split
    files (or from the zip). Asserts the counts, 8 classes, unique keys, val and test disjoint."""
    out = {}
    for split in ('val', 'test'):
        raw = zf.read(f'{split}.json') if zf is not None else (Path(root or BP.EMOSET_ROOT) / f'{split}.json').read_bytes()
        out[split] = json.loads(raw)
    keys = {s: [e[1] for e in out[s]] for s in out}
    for s, n in BP.EMOSET_SPLITS.items():
        assert len(out[s]) == n, f'{s}: {len(out[s])} entries, expected {n}'
        assert all(len(e) == 3 for e in out[s])
        assert len(set(keys[s])) == n, f'{s}: duplicate keys'
        assert all(e[2].startswith('annotation/') and e[1].startswith('image/') for e in out[s])
    assert not set(keys['val']) & set(keys['test']), 'val and test overlap'
    labels = {e[0] for s in out for e in out[s]}
    assert labels == set(BP.EMOSET_CLASSES), f'classes: {sorted(labels)}'
    return out


def unpack(budget_seconds=None):
    """Extract info.json, val.json, test.json and the image and annotation member of every val and test entry into
    EMOSET_ROOT (never a train-only member). Idempotent: a member already present with the right size is skipped."""
    t0 = time.monotonic()
    root = BP.EMOSET_ROOT
    with zipfile.ZipFile(BP.EMOSET_ZIP) as zf:
        entries = read_split_entries(zf=zf)
        info = json.loads(zf.read('info.json'))
        assert set(info['label2idx']) == set(BP.EMOSET_CLASSES) and len(info['label2idx']) == 8
        members = ['info.json', 'val.json', 'test.json']
        for s in ('val', 'test'):
            members += [_safe_member(m) for e in entries[s] for m in (e[1], e[2])]
        assert len(members) == 3 + 2 * N_EMO and len(set(members)) == len(members)
        done = skipped = 0
        for name in members:
            zi = zf.getinfo(name)                    # KeyError if a member is missing from the zip
            target = root / name
            if target.exists() and target.stat().st_size == zi.file_size:
                skipped += 1
                continue
            if budget_seconds is not None and time.monotonic() - t0 > budget_seconds:
                print(f'unpack: budget ended; extracted {done}, present {skipped}, left {len(members) - done - skipped}')
                return 3
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + '.part')
            with zf.open(zi) as src, open(tmp, 'wb') as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            assert tmp.stat().st_size == zi.file_size, name
            tmp.rename(target)
            done += 1
    print(f'unpack: extracted {done}, already present {skipped}, total {len(members)} in {time.monotonic() - t0:.0f}s')
    verify_unpack()
    return 0


def verify_unpack():
    """Gate 0(ii), EmoSet part: split counts, classes, and every val and test image and annotation present."""
    entries = read_split_entries(root=BP.EMOSET_ROOT)
    missing = [m for s in entries for e in entries[s] for m in e[1:] if not (BP.EMOSET_ROOT / m).is_file()]
    assert not missing, f'{len(missing)} unpacked files missing, e.g. {missing[:2]}'
    return dict(val=len(entries['val']), test=len(entries['test']), classes=len(BP.EMOSET_CLASSES))


# ---------------------------------------------------------------- key lists
def affection_index():
    images = json.loads((BP.AFF / BP.ENCODERS['qwen']['aff'][0] / 'index.json').read_text())['images']
    assert len(images) == N_AFF['all']
    return images


def affection_keys(which='all'):
    """Keys in the Qwen store's index order; 'train' | 'val' | 'test' keep the images whose splits include it (an
    image flagged `ignored` also carries its split, as in r12)."""
    images = affection_index()
    keys = [f"{m['source']}/{m['image_name']}" for m in images if which == 'all' or which in m['splits']]
    assert len(keys) == N_AFF[which] and len(set(keys)) == len(keys), (which, len(keys))
    return keys


def emoset_keys():
    entries = read_split_entries(root=BP.EMOSET_ROOT)
    keys = [e[1] for s in ('val', 'test') for e in entries[s]]
    assert len(keys) == N_EMO == len(set(keys))
    return keys


def targets():
    """Ordered (dataset, view) list of the full queue: Affection first, then EmoSet (spec §3.5 0, §9)."""
    return [('affection', 'siglip2'), ('affection', 'siglip1'), ('affection', 'clip_l14')] + \
           [('emoset', v) for v in VIEWS]


def target_keys(dataset, view):
    if dataset == 'emoset':
        return emoset_keys()
    return affection_keys('train' if view == 'clip_l14' else 'all')


def pick(keys, smoke):
    if not smoke:
        return list(keys)
    assert 0 < smoke <= len(keys)
    return [keys[int(i)] for i in np.linspace(0, len(keys) - 1, smoke).round().astype(int)]


# ---------------------------------------------------------------- encoders
def load_encoder(view, device='cuda'):
    """(encoder, identity dict, instruction). Qwen and CLIP: r12's ac_extract.load_encoder; SigLIP: the classes of
    src/data/feature_extract.py with the checkpoint revision recorded."""
    assert view in VIEWS
    if view in ('siglip2', 'siglip1'):
        from src.data import feature_extract as fe
        cls, repo = (fe.Siglip2So400m, fe.SIGLIP2_REPO) if view == 'siglip2' else (fe.SiglipSo400m, fe.SIGLIP1_REPO)
        enc = cls(device)
        rev = getattr(enc.m.config, '_commit_hash', None)
        if rev is None:
            from transformers import AutoConfig
            rev = AutoConfig.from_pretrained(repo)._commit_hash
        assert isinstance(rev, str) and len(rev) == 40, 'SigLIP revision not resolved'
        return enc, dict(model_id=repo, model_revision=rev), None
    ac_view = 'default' if view == 'qwen_default' else view
    enc, identity = ac_extract.load_encoder(ac_view, device)
    return enc, identity, ac_extract.instruction_for(ac_view)


class EmoImages:
    """Spec-like image source for EmoSet keys: load_image(i) opens EMOSET_ROOT/<keys[i]> as RGB."""

    def __init__(self, keys):
        self.keys = keys
        self.root = BP.EMOSET_ROOT

    def load_image(self, i):
        from extract_features import _open_rgb
        return _open_rgb(self.root / self.keys[int(i)])


def image_source(dataset, keys):
    """(spec with load_image, rows): rows[i] is the spec image index of keys[i]."""
    if dataset == 'emoset':
        missing = [k for k in keys if not (BP.EMOSET_ROOT / k).is_file()]
        assert not missing, f'{len(missing)} EmoSet images missing (unpack first), e.g. {missing[:2]}'
        return EmoImages(keys), np.arange(len(keys))
    assert dataset == 'affection'
    return ac_extract.spec_for(keys)


def encode_rows(enc, spec, rows, batch, width):
    """Yield (start, float16 [b, width]) in `rows` order. The next batch's images load on a second thread while the
    GPU works; images are closed after use."""
    assert batch > 0 and len(rows) > 0

    def load(start):
        return [spec.load_image(int(i)) for i in rows[start:start + batch]]

    with ThreadPoolExecutor(1) as pool:
        fut = pool.submit(load, 0)
        for start in range(0, len(rows), batch):
            ims = fut.result()
            if start + batch < len(rows):
                fut = pool.submit(load, start + batch)
            try:
                value = np.asarray(enc.encode_images(ims, batch_size=batch), dtype=np.float16)
            finally:
                for im in ims:
                    im.close()
            assert value.shape == (len(ims), width), value.shape
            yield start, value


# ---------------------------------------------------------------- extraction
def seal(pending, out, dataset, view, keys, x, identity, instruction, smoke, seconds):
    """Gate the array, write keys.json and meta.json into `pending`, then rename it to `out`."""
    from extract_features import check_arrays
    pending, out = Path(pending), Path(out)
    gate = check_arrays(x, None, tol=UNIT_TOL)
    assert gate['ok'], f'extraction gate failed: {gate}'
    assert x.dtype == np.float16 and x.shape == (len(keys), WIDTH[view]), x.shape
    (pending / 'keys.json').write_text(json.dumps(keys))
    meta = dict(dataset=dataset, view=view, instruction=instruction, **identity, dtype='float16', shape=list(x.shape),
                keys_sha256=rs_vh.names_sha(keys), img_sha256=rs_vh.array_sha(x), gate=gate, smoke=bool(smoke),
                batch_size=BATCH[view], seconds=seconds, seconds_per_image=seconds / len(keys))
    (pending / 'meta.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    assert not out.exists(), f'existing view: {out}'
    pending.rename(out)
    return meta


def extract(dataset, view, smoke=0, device='cuda'):
    """Extract one target into its folder under the store root. Smoke needs a scratch store root."""
    root = bs_stores.store_root()
    assert bs_stores.new_folder(dataset, view) is not None, f'r14 extracts no {dataset}/{view}'
    assert (smoke == 0) or root != BP.PRE, 'a smoke extraction needs BS_STORE_ROOT (a scratch root)'
    out = bs_stores.new_folder(dataset, view)
    keys = pick(target_keys(dataset, view), smoke)
    pending = ac_extract.check_target(out)          # refuses a finished or pending target
    spec, rows = image_source(dataset, keys)
    enc, identity, instruction = load_encoder(view, device)
    pending.mkdir(parents=True)
    x = np.lib.format.open_memmap(pending / 'img.npy', mode='w+', dtype=np.float16, shape=(len(keys), WIDTH[view]))
    t0, done = time.monotonic(), 0
    for start, value in encode_rows(enc, spec, rows, BATCH[view], WIDTH[view]):
        assert start == done
        x[start:start + len(value)] = value
        done += len(value)
        if (start // BATCH[view]) % 50 == 0 or done == len(keys):
            print(f'{dataset}/{view}: {done}/{len(keys)} {time.monotonic() - t0:.0f}s', flush=True)
    assert done == len(keys)
    x.flush()
    seconds = time.monotonic() - t0
    meta = seal(pending, out, dataset, view, keys, x, identity, instruction, smoke, seconds)
    print(f'{dataset}/{view}: sealed {len(keys)} images at {out}; {seconds / len(keys) * 1000:.1f} ms/image', flush=True)
    return meta


def sealed(dataset, view):
    out = bs_stores.new_folder(dataset, view)
    return out.exists() and (out / 'meta.json').exists()


def extract_all(device='cuda'):
    """The full queue's extractions. A sealed target is skipped (idempotent); a pending folder stops the run."""
    for dataset, view in targets():
        out = bs_stores.new_folder(dataset, view)
        if sealed(dataset, view):
            print(f'{dataset}/{view}: sealed already, skipped', flush=True)
            continue
        extract(dataset, view, 0, device)
    return 0


# ---------------------------------------------------------------- parity (gate 0(i))
def parity_cases():
    """name -> dict(view, dataset, keys fn, expected fn). Gate 0(i): first 100 sorted Affection validation keys for the
    CLIP L/14 valtest store, the default Qwen store and both instructed emotion stores; first 100 ArtELingo rows for the
    SigLIP 2 and SigLIP 1 stores."""
    return {'clip_l14_affection': ('affection', 'clip_l14'), 'qwen_default_affection': ('affection', 'qwen_default'),
            'emotion_focus_affection': ('affection', 'emotion_focus'),
            'emotion_focus_photo_affection': ('affection', 'emotion_focus_photo'),
            'siglip2_artelingo': ('artelingo', 'siglip2'), 'siglip1_artelingo': ('artelingo', 'siglip1')}


def parity_keys(dataset, n=100):
    if dataset == 'affection':
        keys = sorted(affection_keys('val'))[:n]
    else:
        keys = [m['image'] for m in json.loads((BP.ART / BP.ENCODERS['siglip2']['art'] / 'index.json').read_text())['images'][:n]]
    assert len(keys) == n
    return keys


def run_parity(name, device='cuda'):
    dataset, view = parity_cases()[name]
    keys = parity_keys(dataset)
    expected = bs_stores.load(dataset, view, keys)
    if dataset == 'artelingo':       # the second SigLIP store has the same painting order
        art = json.loads((BP.ART / BP.ENCODERS[view]['art'] / 'index.json').read_text())['images'][:len(keys)]
        assert [m['image'] for m in art] == keys, 'ArtELingo rows differ between the two SigLIP stores'
        spec = type('S', (), {'load_image': staticmethod(
            lambda i: __import__('extract_features')._open_rgb(ac_extract.ART_IMAGES / keys[int(i)]))})
        rows = np.arange(len(keys))
    else:
        spec, rows = ac_extract.spec_for(keys)
    enc, identity, _ = load_encoder(view, device)
    batch = BATCH[view]
    actual = np.concatenate([v for _, v in encode_rows(enc, spec, rows, batch, WIDTH[view])])
    del enc
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    report = ac_extract.parity_report(actual, expected)
    path = STAGE0 / f'parity_{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    info = dict(name=name, dataset=dataset, view=view, stores=[str(f) for f in bs_stores.folders(dataset, view) if f],
                keys_sha256=rs_vh.names_sha(keys), **identity, batch_size=batch)
    path.write_text(json.dumps(dict(info, **report), indent=2, allow_nan=False) + '\n')
    print(f'parity {name}: min_cosine={report["min_cosine"]:.6f} max_abs_diff={report["max_abs_diff"]:.3e} '
          f'ok={report["ok"]} -> {path}', flush=True)
    return report


def parity(cases=None, device='cuda'):
    names = cases or list(parity_cases())
    ok = True
    for name in names:
        ok &= run_parity(name, device)['ok']
    return 0 if ok else 1


# ---------------------------------------------------------------- store check (gate 0(ii))
def check_store(dataset, view):
    """Re-read one sealed store from disk: rows, keys (equal to the target's key list, or its smoke subset), finite,
    unit norm within 1e-3, hashes, and the join through bs_stores.load. Returns a dict with `ok`."""
    from extract_features import check_arrays
    out = bs_stores.new_folder(dataset, view)
    res = dict(dataset=dataset, view=view, folder=str(out), ok=False)
    if not (out.exists() and (out / 'meta.json').exists()):
        res['error'] = 'store missing'
        return res
    meta = json.loads((out / 'meta.json').read_text())
    keys = json.loads((out / 'keys.json').read_text())
    x = np.load(out / 'img.npy', mmap_mode='r')
    full = target_keys(dataset, view)
    expected = pick(full, len(keys)) if meta['smoke'] else full
    gate = check_arrays(x, None, tol=UNIT_TOL)
    got = bs_stores.load(dataset, view, keys[:2000] if len(keys) > 2000 else keys, folder_list=[out])
    sub = np.asarray(x[:len(got)], dtype=np.float32)
    checks = dict(
        rows_equal_keys=x.shape == (len(keys), WIDTH[view]), dtype_f16=str(x.dtype) == 'float16',
        keys_unique=len(set(keys)) == len(keys), keys_equal_target=keys == expected,
        count_ok=meta['smoke'] or len(keys) == len(full), keys_sha=meta['keys_sha256'] == rs_vh.names_sha(keys),
        img_sha=meta['img_sha256'] == rs_vh.array_sha(np.asarray(x)), finite_unit=gate['ok'],
        join_ok=bool(np.array_equal(got, sub)))
    res.update(checks=checks, n=len(keys), smoke=meta['smoke'], gate=gate, model_revision=meta.get('model_revision'),
               ok=all(checks.values()))
    return res


def check(smoke=False):
    """Gate 0(ii) on every new store, plus the EmoSet unpack counts. `smoke`: only stores that exist are checked."""
    results = []
    unpack_res = verify_unpack()
    for dataset, view in targets():
        r = check_store(dataset, view)
        if smoke and r.get('error') == 'store missing':
            continue
        results.append(r)
        print(f'check {dataset}/{view}: ok={r["ok"]} {r.get("error", "")}', flush=True)
    ok = all(r['ok'] for r in results) and bool(results)
    path = STAGE0 / 'stores.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(ok=ok, emoset_unpack=unpack_res, stores=results), indent=2, allow_nan=False) + '\n')
    print(f'check: ok={ok} -> {path}', flush=True)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    u = sub.add_parser('unpack')
    u.add_argument('--budget-seconds', type=float)
    e = sub.add_parser('extract')
    e.add_argument('--dataset', required=True, choices=['affection', 'emoset'])
    e.add_argument('--view', required=True, choices=VIEWS)
    e.add_argument('--smoke', type=int, default=0, metavar='N')
    e.add_argument('--skip-sealed', action='store_true', help='exit 0 when the target is already sealed')
    e.add_argument('--device', default='cuda')
    a = sub.add_parser('extract-all')
    a.add_argument('--device', default='cuda')
    p = sub.add_parser('parity')
    p.add_argument('--cases', help='comma list of ' + ','.join(parity_cases()))
    p.add_argument('--device', default='cuda')
    c = sub.add_parser('check')
    c.add_argument('--smoke', action='store_true')
    args = ap.parse_args(argv)
    if args.cmd == 'unpack':
        return unpack(args.budget_seconds)
    if args.cmd == 'extract':
        if args.skip_sealed and sealed(args.dataset, args.view):
            print(f'{args.dataset}/{args.view}: sealed already, skipped')
            return 0
        extract(args.dataset, args.view, args.smoke, args.device)
        return 0
    if args.cmd == 'extract-all':
        return extract_all(args.device)
    if args.cmd == 'parity':
        return parity(args.cases.split(',') if args.cases else None, args.device)
    return check(args.smoke)


if __name__ == '__main__':
    sys.exit(main())
