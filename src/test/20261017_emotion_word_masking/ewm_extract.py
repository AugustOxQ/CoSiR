"""Controller GPU extraction of four masked-caption views. Import/help load no model."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ewm_masks import (HERE, KEYS, MASK_SPEC_SHA256, atomic_json, build_masks,
                       json_hash, load_captions, load_frozen, save_masks)
from pav_extract import OUTPUT, identity, validate_key
from pav_prompts import PROMPTS

SMOKE_OUTPUT = Path('/tmp/claude-0/-project-CoSiR/ewm_smoke')
EMPTY_IMAGE = np.empty((0, 2048), dtype=np.float16)


def validate_masked(path, expected, instruction, mask, caption_sha256, smoke=False):
    """Verify a seal and every feature row; incompatible sealed files are never replaced."""
    from scripts.extract_features import check_arrays
    done = json.loads((path / 'done.json').read_text())
    assert done['schema'] == 1
    assert done['instruction'] == instruction, f'{path}: instruction mismatch'
    assert done['mask'] == mask and done['masked_caption_sha256'] == caption_sha256
    assert done['mask_spec_sha256'] == MASK_SPEC_SHA256
    assert done['smoke'] == smoke, f'{path}: smoke/full mismatch'
    assert done['gate']['ok'], f'{path}: sealed gate failed'
    assert all(done['unmasked_reproduction'][key]['mean_cosine'] > .999 for key in KEYS)
    with np.load(path / 'ids.npz', allow_pickle=False) as ids:
        for name, value in expected.items():
            assert np.array_equal(ids[name], value), f'{path}: {name} mismatch'
    txt = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert txt.shape == (len(expected['sample_ids']), 2048)
    assert txt.dtype == np.float16
    gate = check_arrays(EMPTY_IMAGE, txt)
    assert gate['ok'], (path, gate)
    return txt


def reproduction(enc, texts, s):
    """Encode exactly the first 256 unmasked captions, with both instructions."""
    from apc_data import l2
    out = {}
    for key in KEYS:
        _, stored = validate_key(OUTPUT / key, identity(s, 5000, 99999), PROMPTS[key])
        enc.instruction = PROMPTS[key]
        start = time.monotonic()
        actual = np.asarray(enc.encode_texts(texts[:256], batch_size=128), dtype=np.float32)
        assert actual.shape == (256, 2048) and np.isfinite(actual).all()
        cos = float(np.mean(np.sum(l2(actual) * l2(np.asarray(stored[:256], dtype=np.float32)), axis=1)))
        print(f'unmasked reproduction {key}: mean cosine={cos:.8f} (required >0.999)', flush=True)
        assert cos > .999, (key, cos)
        out[key] = dict(n=256, mean_cosine=cos, seconds=time.monotonic() - start)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true', help='64 captions, both keys, label mask, only /tmp output')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    s = load_frozen()
    texts = load_captions(s)
    nt = 64 if args.smoke else len(texts)
    # Smoke artifacts are also limited to 64 rows and can never replace full artifacts.
    bundle = build_masks(texts[:nt])
    save_masks(bundle, {'sample_ids': s['sample_ids'][:nt]}, SMOKE_OUTPUT if args.smoke else HERE)
    root = SMOKE_OUTPUT if args.smoke else OUTPUT
    masks = ('label',) if args.smoke else ('label', 'broad')
    expected = identity(s, 0, nt)
    pending = []
    for mask in masks:
        for key in KEYS:
            path = root / f'masked_{mask}' / key
            if (path / 'done.json').exists():
                validate_masked(path, expected, PROMPTS[key], mask, bundle['masked_sha256'][mask], args.smoke)
                print(f'{mask}/{key}: verified complete; skip', flush=True)
            else:
                pending.append((mask, key, path))
    if not pending:
        return
    from scripts.extract_features import check_arrays
    from src.data.feature_extract import Qwen3VLEmb
    enc = Qwen3VLEmb(device=args.device, instruction=PROMPTS[KEYS[0]])
    assert enc.dim == 2048
    checks = reproduction(enc, texts, s)
    for mask, key, path in pending:
        path.mkdir(parents=True, exist_ok=True)
        enc.instruction = PROMPTS[key]
        txt = np.lib.format.open_memmap(path / 'txt.npy', mode='w+', dtype=np.float16, shape=(nt, 2048))
        start = time.monotonic()
        masked = bundle['masked'][mask]
        for a in range(0, nt, 4096):
            b = min(a + 4096, nt)
            txt[a:b] = enc.encode_texts(masked[a:b], batch_size=128)
            print(f'{mask}/{key}: captions {b}/{nt}', flush=True)
        txt.flush()
        elapsed = time.monotonic() - start
        gate = check_arrays(EMPTY_IMAGE, txt)
        assert gate['ok'], gate
        np.savez(path / 'ids.npz', **expected)
        atomic_json(path / 'done.json', dict(schema=1, instruction=PROMPTS[key], mask=mask,
            masked_caption_sha256=json_hash(masked), mask_spec_sha256=MASK_SPEC_SHA256,
            source_caption_sha256=bundle['source_sha256'], shape=list(txt.shape), dtype=str(txt.dtype),
            text_seconds=elapsed, text_items_per_second=nt / elapsed, gate=gate,
            unmasked_reproduction=checks, smoke=args.smoke))
        print(f'{mask}/{key}: {elapsed:.1f}s, {nt / elapsed:.2f} captions/s; sealed', flush=True)


if __name__ == '__main__':
    main()
