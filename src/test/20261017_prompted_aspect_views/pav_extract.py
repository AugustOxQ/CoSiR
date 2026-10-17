"""Controller-run Qwen extraction. Import/help never construct a model.

Keys are sealed only after positional ID, shape, finite and unit-norm gates.
Incomplete keys are recomputed; completed incompatible keys raise (no deletion).
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from pav_prompts import PROMPTS
from pav_sample import load_frozen

OUTPUT = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts')
SMOKE_OUTPUT = Path('/tmp/claude-0/-project-CoSiR/pav_smoke')


def identity(s, ni, nt):
    return dict(sample_ids=s['sample_ids'][:nt], caption_painting_ids=s['painting_ids'][:nt],
                image_painting_ids=s['subset_painting_ids'][:ni],
                image_indices=s['subset_image_indices'][:ni])


def validate_key(path, expected, instruction):
    from scripts.extract_features import check_arrays
    with (path / 'done.json').open() as f:
        done = json.load(f)
    assert done['instruction'] == instruction, f'{path}: instruction mismatch'
    with np.load(path / 'ids.npz') as ids:
        for k, v in expected.items():
            assert np.array_equal(ids[k], v), f'{path}: {k} mismatch'
    img = np.load(path / 'img.npy', mmap_mode='r')
    txt = np.load(path / 'txt.npy', mmap_mode='r')
    assert img.shape == (len(expected['image_indices']), 2048)
    assert txt.shape == (len(expected['sample_ids']), 2048)
    assert img.dtype == txt.dtype == np.float16
    gate = check_arrays(img, txt)
    assert gate['ok'], (path, gate)
    return img, txt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true', help='16 images, 64 captions, two keys in isolated /tmp output')
    p.add_argument('--keys', nargs='+', choices=list(PROMPTS))
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    keys = args.keys or (list(PROMPTS)[:2] if args.smoke else list(PROMPTS))
    if args.smoke and len(keys) != 2:
        p.error('--smoke requires exactly two keys')
    s = load_frozen()
    ni, nt = (16, 64) if args.smoke else (5000, 99999)
    expected = identity(s, ni, nt)
    root = SMOKE_OUTPUT if args.smoke else OUTPUT
    pending = []
    for key in keys:
        path = root / key
        if all((path / name).exists() for name in ('img.npy', 'txt.npy', 'ids.npz', 'done.json')):
            validate_key(path, expected, PROMPTS[key])
            print(f'{key}: verified complete; skip', flush=True)
        else:
            if (path / 'done.json').exists():
                raise RuntimeError(f'{path}: sealed key missing files; inspect manually')
            pending.append(key)
    if not pending:
        return
    from scripts.extract_features import adapter_artelingo_full, check_arrays
    from src.data.feature_extract import Qwen3VLEmb
    spec = adapter_artelingo_full(sample_ids=s['sample_ids'], expected=None)
    assert np.array_equal([m['sample_id'] for m in spec.text_meta], s['sample_ids'])
    assert np.array_equal([spec.images[int(i)]['painting'] for i in s['subset_image_indices']],
                          s['subset_painting_ids'])
    enc = Qwen3VLEmb(device=args.device, instruction=PROMPTS[pending[0]])
    for key in pending:
        path = root / key
        path.mkdir(parents=True, exist_ok=True)
        enc.instruction = PROMPTS[key]
        img = np.lib.format.open_memmap(path / 'img.npy', mode='w+', dtype='float16', shape=(ni, enc.dim))
        txt = np.lib.format.open_memmap(path / 'txt.npy', mode='w+', dtype='float16', shape=(nt, enc.dim))
        start = time.monotonic()
        for a in range(0, ni, 32):
            batch = [spec.load_image(int(i)) for i in s['subset_image_indices'][a:min(a + 32, ni)]]
            try:
                img[a:a + len(batch)] = enc.encode_images(batch, batch_size=32)
            finally:
                for im in batch:
                    im.close()
        img.flush()
        image_seconds = time.monotonic() - start
        print(f'{key}: images {image_seconds:.1f}s, {ni / image_seconds:.2f} items/s', flush=True)
        start = time.monotonic()
        for a in range(0, nt, 4096):
            b = min(a + 4096, nt)
            txt[a:b] = enc.encode_texts(spec.texts[a:b], batch_size=128)
            print(f'{key}: captions {b}/{nt}', flush=True)
        txt.flush()
        text_seconds = time.monotonic() - start
        gate = check_arrays(img, txt)
        assert gate['ok'], gate
        np.savez(path / 'ids.npz', **expected)
        done = dict(instruction=PROMPTS[key], image_seconds=image_seconds, text_seconds=text_seconds,
                    image_items_per_second=ni / image_seconds, text_items_per_second=nt / text_seconds,
                    gate=gate, smoke=args.smoke)
        marker = path / 'done.pending.json'
        with marker.open('w') as f:
            json.dump(done, f, indent=2)
        marker.replace(path / 'done.json')
        print(f'{key}: captions {text_seconds:.1f}s, {nt / text_seconds:.2f} items/s; gate passed', flush=True)


if __name__ == '__main__':
    main()
