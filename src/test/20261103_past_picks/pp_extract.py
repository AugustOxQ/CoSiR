"""r19 extraction: the new stores of spec §9, the parity checks of gate (i), and the store checks.

  pp_extract.py extract --dataset {mmart,affection,semart} --view <view> [--smoke N]
  pp_extract.py parity  --view <view> [--smoke N]
  pp_extract.py check   --dataset <dataset> --view <view>

`extract` writes `pp_paths.NEW_STORE[(dataset, view)]`: `img.npy` (float16 [n, d], unit rows), `keys.json`, `meta.json`
(model id, `config._commit_hash`, n, keys_sha256, img_sha256, writer, seconds), sealed by renaming `<folder>.pending`.
Then it runs `pp_stores.check` and writes `OUT/stage0/gate_i/store_<dataset>_<view>.json`. It takes no lock (run it
under `res-queue gpu`). A sealed or pending target stops it.

Copied from the code that wrote the stores each view is compared with (no frozen file edited):
  encoders   CLIP B/32, B/16, DINOv2: `src.data.feature_extract` classes as `scripts/extract_features.py` builds them
             (batch 256, 256, 128); CLIP L/14 and SigLIP 1: `bs_extract.load_encoder` (batch 128, 32); Qwen
             `genre_word`: `mp_extract.load_encoder('genre_word')` (batch 32). Revision: `enc.m.config._commit_hash`.
  loop       `bs_extract.encode_rows` (one prefetch thread; float16 rows), r14's `extract` (:253) and `seal` (:228)
  keys       MMArt: `mp_extract.target_keys('new')`, images `image_path` of the index; Affection:
             `bs_extract.affection_keys('all')` with `bs_extract.image_source` (85,007, the Qwen store's index order);
             SemArt: the image files of the stored Qwen store's `index.json` (train, val, test CSV order),
             images `SEMART_DATA/Images/<image_file>` opened with `extract_features._open_rgb`.
Parity (gate (i)): the writing code (same loader, batch, precision) re-encodes the first 100 keys of a stored store and
compares row by row with `pp_stores.load` (joined by key); cosine >= `pp_paths.PARITY_COS`. MMArt: the prep's 200 shared
paintings (`mp_extract.target_keys('parity')`) against the ArtELingo store of the same painting.
Smoke: `PP_SMOKE=1` with `PP_OUTPUTS`; stores go to `OUT/stores/<path under pre_extract>`, never to `pre_extract`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import pp_paths as PP
import pp_stores as S

import bs_extract                       # r14's extractor (puts ac_extract, ct2_*, rs_vh on sys.path)
import ac_extract
import mp_extract

BATCH = {'clip_b32': 256, 'clip_b16': 256, 'dinov2_large': 128, 'clip_l14': 128, 'siglip1': 32, 'siglip2': 32,
         'qwen': 32, 'genre_word': 32}      # extract_features.BATCH (CLIP 256, Qwen 32, SigLIP 32); bs/mp BATCH
GATE_DIR = PP.OUT / 'stage0' / 'gate_i'
NEW_VIEWS = {'mmart': PP.NEW_ENCODERS, 'affection': PP.NEW_ENCODERS,
             'semart': ('siglip1', 'clip_l14', 'clip_b16', 'dinov2_large', 'genre_word')}
FE_CLASS = {'clip_b32': ('ClipB32', 'CLIP_REPO'), 'clip_b16': ('ClipB16', 'CLIP_B16_REPO'),
            'dinov2_large': ('Dinov2Large', 'DINOV2_REPO')}
SEMART_QWEN_INDEX = PP.SEMART_PRE / 'qwen3vl_emb_2b' / 'index.json'
WRITER = 'pp_extract.py (r19 build, ticket 02)'


# ------------------------------------------------------------------------------------------------ encoders
def load_encoder(view, device='cuda'):
    """(encoder, identity dict {model_id, model_revision}, instruction or None) for an encoder key or `genre_word`."""
    if view in FE_CLASS:
        from src.data import feature_extract as fe
        cls, repo = FE_CLASS[view]
        enc = getattr(fe, cls)(device)
        repo = getattr(fe, repo)
        rev = getattr(enc.m.config, '_commit_hash', None)
        if rev is None:
            from transformers import AutoConfig
            rev = AutoConfig.from_pretrained(repo)._commit_hash
        assert isinstance(rev, str) and len(rev) == 40, f'{view}: revision not resolved'
        return enc, dict(model_id=repo, model_revision=rev), None
    base = {'siglip1': 'siglip1', 'clip_l14': 'clip_l14', 'siglip2': 'siglip2'}.get(view)
    if base:
        return bs_extract.load_encoder(base, device)
    if view == 'qwen':
        return mp_extract.load_encoder('qwen_default', device)
    assert view == 'genre_word', view
    return mp_extract.load_encoder('genre_word', device)


# ------------------------------------------------------------------------------------------------ keys and images
class PathImages:
    """Spec-like image source: load_image(i) opens paths[i] as RGB."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        missing = [p for p in self.paths if not p.is_file()]
        assert not missing, f'{len(missing)} images missing, e.g. {missing[:2]}'

    def load_image(self, i):
        from extract_features import _open_rgb
        return _open_rgb(self.paths[int(i)])


def semart_keys():
    images = json.loads(SEMART_QWEN_INDEX.read_text())['images']
    keys = [m['image_file'] for m in images]
    assert len(keys) == PP.STORE_ROWS['semart'] == len(set(keys))
    return keys


def target_keys(dataset):
    if dataset == 'mmart':
        return mp_extract.target_keys('new')
    if dataset == 'affection':
        return bs_extract.affection_keys('all')
    assert dataset == 'semart'
    return semart_keys()


def images_for(dataset, keys):
    """(spec, rows): rows[i] is the spec image index of keys[i]."""
    if dataset == 'affection':
        return ac_extract.spec_for(keys)
    if dataset == 'mmart':
        table = mp_extract.S.read_index().set_index('key')
        spec = PathImages(table.loc[keys].image_path.tolist())
    elif dataset == 'semart':
        spec = PathImages([PP.SEMART_DATA / 'Images' / k for k in keys])
    else:
        assert dataset == 'artelingo'
        table = {m['painting']: m['image'] for m in json.loads((PP.ART / PP.ART_STORE['qwen'] / 'index.json')
                                                                .read_text())['images']}
        spec = PathImages([ac_extract.ART_IMAGES / table[k] for k in keys])
    return spec, np.arange(len(keys))


def pick(keys, smoke):
    return bs_extract.pick(keys, smoke)


# ------------------------------------------------------------------------------------------------ store folders
def smoke_guard(smoke):
    assert smoke >= 0
    assert bool(smoke) == bool(PP.SMOKE), 'a smoke run needs PP_SMOKE=1 with PP_OUTPUTS (and PP_SMOKE=1 needs --smoke N)'


def store_folder(dataset, view):
    """The target folder; under smoke a mirror inside OUT, never `pre_extract`."""
    assert view in NEW_VIEWS[dataset], f'{dataset}/{view} is not a new store'
    folder = Path(PP.NEW_STORE[(dataset, view)])
    return PP.OUT / 'stores' / folder.relative_to(PP.PRE) if PP.SMOKE else folder


def write_json(name, value):
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    path = GATE_DIR / name
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return path


# ------------------------------------------------------------------------------------------------ extract
def seal(pending, out, dataset, view, keys, x, identity, instruction, smoke, seconds):
    """r14's seal (bs_extract.seal :228) with this round's meta."""
    pending, out = Path(pending), Path(out)
    assert x.dtype == np.float16 and x.shape == (len(keys), PP.WIDTH.get(view, 2048)), x.shape
    assert len(set(keys)) == len(keys)
    dev = 0.0
    for start in range(0, len(x), 4096):
        block = np.asarray(x[start:start + 4096], dtype=np.float32)
        assert np.isfinite(block).all(), 'non-finite rows'
        dev = max(dev, float(np.abs(np.linalg.norm(block, axis=1) - 1).max()))
    assert dev <= PP.UNIT_TOL, f'rows not unit: {dev:.2e}'
    (pending / 'keys.json').write_text(json.dumps(keys))
    meta = dict(dataset=dataset, view=view, instruction=instruction, **identity, dtype='float16', shape=list(x.shape),
                n=len(keys), d=int(x.shape[1]), keys_sha256=S._sha_keys(keys), img_sha256=S._sha_array(x),
                max_unit_dev=dev, smoke=bool(smoke), batch_size=BATCH[view], writer=WRITER, seconds=seconds,
                seconds_per_image=seconds / len(keys))
    (pending / 'meta.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    assert not out.exists(), f'existing folder: {out}'
    pending.rename(out)
    return meta


def extract(dataset, view, smoke=0, device='cuda'):
    smoke_guard(smoke)
    out = store_folder(dataset, view)
    pending = ac_extract.check_target(out)          # refuses a sealed or pending target
    pending.parent.mkdir(parents=True, exist_ok=True)
    keys = pick(target_keys(dataset), smoke)
    spec, rows = images_for(dataset, keys)
    enc, identity, instruction = load_encoder(view, device)
    width = PP.WIDTH.get(view, 2048)
    pending.mkdir()
    x = np.lib.format.open_memmap(pending / 'img.npy', mode='w+', dtype=np.float16, shape=(len(keys), width))
    t0, done = time.monotonic(), 0
    for start, value in bs_extract.encode_rows(enc, spec, rows, BATCH[view], width):
        assert start == done
        x[start:start + len(value)] = value
        done += len(value)
        if (start // BATCH[view]) % 50 == 0 or done == len(keys):
            print(f'{dataset}/{view}: {done}/{len(keys)} {time.monotonic() - t0:.0f}s', flush=True)
    assert done == len(keys)
    x.flush()
    seconds = time.monotonic() - t0
    meta = seal(pending, out, dataset, view, keys, x, identity, instruction, smoke, seconds)
    print(f'{dataset}/{view}: sealed {len(keys)} images at {out}; {1000 * seconds / len(keys):.1f} ms/image',
          flush=True)
    return meta


def store_check(dataset, view, smoke=0):
    """pp_stores.check of the written store -> OUT/stage0/gate_i/store_<dataset>_<view>.json. Under smoke the store
    is a 64-key mirror: the row count and reference key set of a full store are replaced by the smoke keys."""
    folder = store_folder(dataset, view)
    if smoke:
        keys = pick(target_keys(dataset), smoke)
        S.NEW_STORE[(dataset, view)] = folder
        S.EXPECTED_ROWS[dataset] = len(keys)
        S._reference_keys = lambda d, index=None, _k=set(keys): set(_k)
    report = S.check(dataset, view)
    report['smoke'] = bool(smoke)
    write_json(f'store_{dataset}_{view}.json', report)
    print(f'store check {dataset}/{view}: ok={report["ok"]} n={report["n"]} problems={report["problems"]}', flush=True)
    return report


# ------------------------------------------------------------------------------------------------ parity (gate (i))
def parity_cases(view):
    """name -> (dataset of the stored store, keys fn, image dataset). Each case re-encodes with the writing code."""
    cases = {}
    index = json.loads((PP.ART / PP.ART_STORE['qwen'] / 'index.json').read_text())['images']
    first_art = [m['painting'] for m in index[:PP.PARITY_ROWS]]
    semart = semart_keys()[:PP.PARITY_ROWS]
    if view in ('dinov2_large', 'clip_b32', 'clip_b16', 'siglip1', 'clip_l14'):
        cases[f'art_{view}'] = ('artelingo', 'artelingo', view, first_art)
    if view in PP.NEW_ENCODERS:
        held = mp_extract.target_keys('parity')
        cases[f'mmart_{view}'] = ('mmart', 'mmart', view, held)
    if view == 'clip_b32':
        cases['semart_clip_b32'] = ('semart', 'semart', view, semart)
    if view in ('qwen', 'siglip2'):
        cases[f'semart_{view}'] = ('semart', 'semart', view, semart)
    if view == 'clip_l14':
        cases['aff_clip_l14'] = ('affection', 'affection', view, sorted(bs_extract.affection_keys('val'))[:PP.PARITY_ROWS])
    if view == 'genre_word':
        import rt_data
        cases['genre_word'] = ('artelingo', 'artelingo', view, rt_data.names('val')[:PP.PARITY_ROWS])
    assert cases, f'no parity case for {view!r}'
    return cases


def parity(view, smoke=0, device='cuda'):
    smoke_guard(smoke)
    cases = parity_cases(view)
    enc, identity, instruction = load_encoder(view, device)
    width = PP.WIDTH.get(view, 2048)
    results = {}
    for name, (stored_ds, image_ds, stored_view, keys) in cases.items():
        if smoke:
            keys = pick(keys, min(smoke, len(keys)))
        expected = S.load(stored_ds, stored_view, keys)          # joined by key, L2-normalised
        spec, rows = images_for(image_ds, keys)
        t0 = time.monotonic()
        actual = np.concatenate([v for _, v in bs_extract.encode_rows(enc, spec, rows, BATCH[view], width)])
        seconds = time.monotonic() - t0
        a = actual.astype(np.float32)
        assert a.shape == expected.shape and np.isfinite(a).all(), (a.shape, expected.shape)
        cos = (a * expected).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(expected, axis=1))
        assert np.isfinite(cos).all()
        report = {'pass': bool((cos >= PP.PARITY_COS).all()), 'min_cosine': float(cos.min()),
                  'mean_cosine': float(cos.mean()), 'n': len(keys), 'cosine_bar': PP.PARITY_COS, 'view': view,
                  'stored': f'{stored_ds}/{stored_view}', 'batch_size': BATCH[view], **identity,
                  'seconds': seconds, 'ms_per_image': 1000 * seconds / len(keys), 'smoke': bool(smoke),
                  'first_key': keys[0], 'last_key': keys[-1]}
        write_json(f'{name}.json', report)
        results[name] = report
        print(f'parity {name}: pass={report["pass"]} min={report["min_cosine"]:.5f} mean={report["mean_cosine"]:.5f} '
              f'n={len(keys)} {report["ms_per_image"]:.1f} ms/image', flush=True)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('extract', 'check'):
        s = sub.add_parser(name)
        s.add_argument('--dataset', choices=sorted(NEW_VIEWS), required=True)
        s.add_argument('--view', required=True)
        s.add_argument('--smoke', type=int, default=0)
    s = sub.add_parser('parity')
    s.add_argument('--view', choices=sorted(BATCH), required=True)
    s.add_argument('--smoke', type=int, default=0)
    a = p.parse_args(argv)
    if a.command == 'parity':
        res = parity(a.view, a.smoke)
        return 0 if all(r['pass'] for r in res.values()) else 1
    assert a.view in NEW_VIEWS[a.dataset], f'{a.dataset} has no new {a.view} store'
    if a.command == 'extract':
        extract(a.dataset, a.view, a.smoke)
    return 0 if store_check(a.dataset, a.view, a.smoke)['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
