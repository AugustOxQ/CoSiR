"""Key-list image extractor for the new Affection feature views, and the parity checks of stage 0a(ii) and 0a(vi).

Views (r12 ticket 03; spec §3.2, D3, D11, §3.5 0a(ii), 0a(vi)):
  emotion_focus        Qwen3-VL-Embedding-2B, r10's emotion_focus instruction (unchanged)
  emotion_focus_photo  same encoder, spec §3.2's photo instruction (`ac_paths.PHOTO_INSTRUCTION`)
  default              same encoder, the default instruction (parity (b) only; also a valid view)
  clip_l14             CLIP ViT-L/14, the encoder class and settings of the stored ArtELingo clip_l14 store

Each extraction writes <out>/{img.npy, keys.json, meta.json} (index format of the ticket index), through
<out>.pending, then renames. A finished or pending folder is never touched. Images come from
`extract_features.adapter_affection_local` (COCO prefixes, FI subfolders), joined by key `<dataset>/<image_name>`.

Commands (repo root; env /root/miniconda3/envs/CoSiR; PYTHONDONTWRITEBYTECODE=1; GPU under flock -n -o -E 75
/tmp/gpu0.lock):
  extract  python ac_extract.py extract --view emotion_focus --keys valtest --out <folder>
           python ac_extract.py extract --view clip_l14 --keys '["coco_affective/COCO_..jpg", ...]' --out <folder>
           --smoke N: N keys spread evenly over valtest (covers FI); meta.smoke true; never into a canonical folder.
  parity   python ac_extract.py parity a|b|c   (writes OUTPUTS/stage0a/<name>.json; exit 1 if cosine < 0.999)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import ac_paths as P  # noqa: E402  (also puts every older round, e.g. ct2_*, on sys.path)
import rs_vh  # noqa: E402

VIEWS = ('emotion_focus', 'emotion_focus_photo', 'default', 'clip_l14')
# The real folders, also under AC_SMOKE (when P.VIEWS and P.CLIP_VIEW point at the scratch smoke_views, which are
# not canonical): a smoke extraction may go into smoke_views and never into these.
_VIEWS, _CLIP_VIEW = P.AFF / 'qwen3vl_emb_2b_aspect_prompts', P.AFF / 'clip_l14_valtest'
CANONICAL = {'emotion_focus': _VIEWS / 'emotion_focus', 'emotion_focus_photo': _VIEWS / 'emotion_focus_photo',
             'clip_l14': _CLIP_VIEW}
BATCH = {'emotion_focus': 32, 'emotion_focus_photo': 32, 'default': 32, 'clip_l14': 128}  # clip: BATCH[CLIP_L14]['img']
WIDTH = {'emotion_focus': 2048, 'emotion_focus_photo': 2048, 'default': 2048, 'clip_l14': 768}
COSINE_MIN = 0.999
N_PARITY = 200
ART_CLIP_STORE = Path('/data/SSD2/pre_extract/artelingo_full/clip_l14')
ART_IMAGES = Path('/data/PDD/wikiart_proj/wikiart')


# ---------------------------------------------------------------- keys and paths
def valtest_keys():
    import ac_data
    return ac_data.keys('val') + ac_data.keys('test')


def is_smoke_run(smoke_n, keys):
    """A view is a smoke view when --smoke is given, when AC_SMOKE is set (valtest_keys is then the truncated list,
    so equal keys would otherwise be marked as a full view), or when the keys are not the val and test keys."""
    return bool(smoke_n) or P.SMOKE > 0 or keys != valtest_keys()


def parse_keys(arg):
    """'valtest', a JSON list string, or a path to a JSON list file. Keys are unique `<dataset>/<image_name>`."""
    if arg == 'valtest':
        return valtest_keys()
    keys = json.loads(Path(arg).read_text() if Path(arg).is_file() else arg)
    check_keys(keys)
    return keys


def check_keys(keys):
    assert isinstance(keys, list) and keys and all(isinstance(k, str) and '/' in k for k in keys), 'key list'
    assert len(set(keys)) == len(keys), 'duplicate keys'
    return keys


def split_key(key):
    source, _, name = key.partition('/')
    assert source and name, f'bad key {key}'
    return source, name


def spec_for(keys):
    """(adapter spec, rows): `rows[i]` is the adapter image row of keys[i]. The adapter is the one of
    scripts/extract_features.py with the FI root added (ticket 01); it checks every image file exists."""
    import extract_features as ef
    sources = sorted({split_key(k)[0] for k in keys})
    roots = dict(ef.AFFECTION_IMAGE_ROOTS)
    roots[ef.AFFECTION_FI_SOURCE] = ef.AFFECTION_FI_ROOT
    assert ef.AFFECTION_FI_SOURCE == P.FI_SOURCE and Path(ef.AFFECTION_FI_ROOT) == P.FI_ROOT
    spec = ef.adapter_affection_local(sources=sources, image_roots=roots)
    lookup = {(m['source'], m['image_name']): i for i, m in enumerate(spec.images)}
    missing = [k for k in keys if split_key(k) not in lookup]
    assert not missing, f'keys not in the annotations: {missing[:3]} ({len(missing)})'
    rows = np.asarray([lookup[split_key(k)] for k in keys], dtype=np.int64)
    assert [f"{spec.images[int(i)]['source']}/{spec.images[int(i)]['image_name']}" for i in rows] == keys
    return spec, rows


def smoke_keys(n):
    keys = valtest_keys()
    assert 0 < n <= len(keys)
    return [keys[int(i)] for i in np.linspace(0, len(keys) - 1, n).round().astype(int)]


# ---------------------------------------------------------------- encoders
def instruction_for(view):
    if view == 'emotion_focus':
        from itp_data import PROMPTS
        return PROMPTS['emotion_focus']
    if view == 'emotion_focus_photo':
        return P.PHOTO_INSTRUCTION
    if view == 'default':
        from src.data.feature_extract import QWEN_INSTRUCTION_DEFAULT
        return QWEN_INSTRUCTION_DEFAULT
    assert view == 'clip_l14'
    return None


def load_encoder(view, device='cuda'):
    """(encoder, identity). Qwen: ct2_extract.load_encoder (pinned revision of the ArtELingo stores, asserted) with
    the view's instruction set afterwards, as rb_extract_a28 does. CLIP: src.data.feature_extract.ClipL14."""
    assert view in VIEWS
    if view == 'clip_l14':
        from src.data.feature_extract import CLIP_L14_REPO, ClipL14
        enc = ClipL14(device)
        rev = getattr(enc.m.config, '_commit_hash', None)
        if rev is None:
            from transformers import AutoConfig
            rev = AutoConfig.from_pretrained(CLIP_L14_REPO)._commit_hash
        assert isinstance(rev, str) and len(rev) == 40, 'CLIP revision not resolved'
        return enc, dict(model_id=CLIP_L14_REPO, model_revision=rev)
    import ct2_extract
    enc, identity = ct2_extract.load_encoder('emotion_focus_img', device)
    enc.instruction = instruction_for(view)
    return enc, identity


def encode_rows(enc, spec, rows, batch, width):
    """Yield (start, float16 [b, width]) in `rows` order; images are loaded and closed per batch."""
    assert batch > 0 and len(rows) > 0
    for start in range(0, len(rows), batch):
        ims = [spec.load_image(int(i)) for i in rows[start:start + batch]]
        try:
            value = np.asarray(enc.encode_images(ims, batch_size=batch), dtype=np.float16)
        finally:
            for im in ims:
                im.close()
        assert value.shape == (len(ims), width), value.shape
        yield start, value


# ---------------------------------------------------------------- sealing
def is_canonical(out):
    return Path(out).resolve() in {p.resolve() for p in CANONICAL.values()}


def check_target(out):
    out = Path(out)
    pending = out.with_name(out.name + '.pending')
    assert not out.exists(), f'sealed or existing view, not overwritten: {out}'
    assert not pending.exists(), f'pending extraction exists: {pending}'
    return pending


def seal(pending, out, keys, x, view, identity, instruction, smoke, seconds):
    """Write keys.json and meta.json next to img.npy (already in `pending`), gate it, then rename to `out`."""
    from extract_features import check_arrays
    pending, out = Path(pending), Path(out)
    gate = check_arrays(x, None, tol=1e-3)
    assert gate['ok'], f'extraction gate failed: {gate}'
    assert x.dtype == np.float16 and x.shape == (len(keys), WIDTH[view])
    (pending / 'keys.json').write_text(json.dumps(keys))
    meta = dict(view=view, instruction=instruction, **identity, dtype='float16', shape=list(x.shape),
                keys_sha256=rs_vh.names_sha(keys), img_sha256=rs_vh.array_sha(x), gate=gate, smoke=bool(smoke),
                batch_size=BATCH[view], seconds=seconds)
    (pending / 'meta.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    assert not out.exists(), f'existing view: {out}'
    pending.rename(out)
    return meta


def extract(view, keys, out, device='cuda', smoke=False):
    assert view in VIEWS
    check_keys(keys)
    out = Path(out)
    assert smoke or P.SMOKE == 0, 'AC_SMOKE is set: this must be a smoke extraction'
    assert not (smoke and is_canonical(out)), 'a smoke view cannot go into a canonical folder'
    assert smoke or keys == valtest_keys() or not is_canonical(out), 'a canonical view takes exactly valtest keys'
    pending = check_target(out)
    spec, rows = spec_for(keys)
    enc, identity = load_encoder(view, device)
    pending.mkdir(parents=True)
    x = np.lib.format.open_memmap(pending / 'img.npy', mode='w+', dtype=np.float16, shape=(len(keys), WIDTH[view]))
    t0, done = time.monotonic(), 0
    for start, value in encode_rows(enc, spec, rows, BATCH[view], WIDTH[view]):
        assert start == done
        x[start:start + len(value)] = value
        done += len(value)
        print(f'{view}: {done}/{len(keys)}', flush=True)
    x.flush()
    meta = seal(pending, out, keys, x, view, identity, instruction_for(view), smoke, time.monotonic() - t0)
    print(f'{view}: sealed {len(keys)} images at {out}', flush=True)
    return meta


# ---------------------------------------------------------------- parity
def parity_report(actual, expected):
    a, e = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    assert a.shape == e.shape and a.ndim == 2 and np.isfinite(a).all() and np.isfinite(e).all()
    cos = np.sum(a * e, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(e, axis=1))
    return dict(n=len(a), min_cosine=float(cos.min()), mean_cosine=float(cos.mean()),
                max_abs_diff=float(np.abs(a - e).max()), per_row_max_abs_diff_max=float(np.abs(a - e).max(axis=1).max()),
                threshold=COSINE_MIN, ok=bool((cos >= COSINE_MIN).all()), cosines=cos.tolist())


def write_parity(name, report, **info):
    path = P.OUTPUTS / 'stage0a' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(name=name, **info, **report), indent=2, allow_nan=False) + '\n')
    print(f'{name}: min_cosine={report["min_cosine"]:.6f} max_abs_diff={report["max_abs_diff"]:.3e} '
          f'ok={report["ok"]} -> {path}', flush=True)
    return path


def parity_keys_b(n_each=100):
    """The build's reading of "the first 200 validation keys including 100 FI": the first 100 non-FI and the first
    100 FI validation keys, in sorted order."""
    import ac_data
    val = ac_data.keys('val')
    fi = [k for k in val if split_key(k)[0] == P.FI_SOURCE][:n_each]
    other = [k for k in val if split_key(k)[0] != P.FI_SOURCE][:n_each]
    assert len(fi) == len(other) == n_each
    return other + fi


def store_rows_by_key(store, keys):
    """Rows of an Affection store (index.json images, img.npy), joined by key."""
    store = Path(store)
    images = json.loads((store / 'index.json').read_text())['images']
    lookup = {f"{m['source']}/{m['image_name']}": i for i, m in enumerate(images)}
    missing = [k for k in keys if k not in lookup]
    assert not missing, f'keys not in {store}: {len(missing)} (FI merged yet?) e.g. {missing[:2]}'
    img = np.load(store / 'img.npy', mmap_mode='r')
    return np.stack([np.asarray(img[lookup[k]]) for k in keys])


def parity(which, device='cuda'):
    if which == 'a':      # generalized extractor vs the stored ArtELingo emotion_focus store, first 200 rows
        import ct2_data
        import ct_extract
        path = ct2_data._sealed(ct2_data.ASPECT / 'emotion_focus', 'emotion_focus')
        names, raw = ct2_data._image_store(path)
        names, expected = names[:N_PARITY].tolist(), np.asarray(raw[:N_PARITY])
        spec, rows = ct_extract.image_spec(names)
        enc, identity = load_encoder('emotion_focus', device)
        info = dict(view='emotion_focus', store=str(path), **identity, batch_size=32)
        width = 2048
    elif which == 'b':    # default instruction on 100 non-FI + 100 FI validation keys vs the default store, by key
        keys = parity_keys_b()
        expected = store_rows_by_key(P.STORE, keys)
        spec, rows = spec_for(keys)
        enc, identity = load_encoder('default', device)
        info = dict(view='default', store=str(P.STORE), keys=keys, keys_sha256=rs_vh.names_sha(keys), **identity,
                    batch_size=32)
        width = 2048
    elif which == 'c':    # CLIP L/14 vs 200 stored ArtELingo clip_l14 rows
        from extract_features import Spec, _open_rgb
        images = json.loads((ART_CLIP_STORE / 'index.json').read_text())['images'][:N_PARITY]
        expected = np.asarray(np.load(ART_CLIP_STORE / 'img.npy', mmap_mode='r')[:N_PARITY])
        spec = Spec(images, lambda i: _open_rgb(ART_IMAGES / images[i]['image']))
        rows = np.arange(N_PARITY)
        enc, identity = load_encoder('clip_l14', device)
        info = dict(view='clip_l14', store=str(ART_CLIP_STORE), **identity, batch_size=BATCH['clip_l14'])
        width = 768
    else:
        raise SystemExit('parity a|b|c')
    batch = info['batch_size']
    actual = np.concatenate([v for _, v in encode_rows(enc, spec, rows, batch, width)])
    report = parity_report(actual, expected)
    name = {'a': 'extractor_artelingo', 'b': 'extractor_default_affection', 'c': 'clip_l14_artelingo'}[which]
    write_parity(name, report, **info)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--view', required=True, choices=VIEWS)
    e.add_argument('--keys', default='valtest', help="'valtest' or a JSON list (string or file)")
    e.add_argument('--out', required=True, type=Path)
    e.add_argument('--smoke', type=int, metavar='N', help='N keys spread over valtest; ignores --keys')
    e.add_argument('--device', default='cuda')
    p = sub.add_parser('parity')
    p.add_argument('which', choices=['a', 'b', 'c'])
    p.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    if args.cmd == 'parity':
        report = parity(args.which, args.device)
        return 0 if report['ok'] else 1
    keys = smoke_keys(args.smoke) if args.smoke else parse_keys(args.keys)
    extract(args.view, keys, args.out, args.device, smoke=is_smoke_run(args.smoke, keys))
    return 0


if __name__ == '__main__':
    sys.exit(main())
