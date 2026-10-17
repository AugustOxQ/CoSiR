"""Optional controller-run emotion-focus training images; help/import are model-free."""
import argparse
import hashlib
import time

import numpy as np

from itp_data import (OUTPUT, PROMPTS, SMOKE, TRAIN_IMAGES, atomic_json, fingerprint,
                      load_frozen, prompted, train_image_ids, validate_train_images)


def encode(enc, spec, indices):
    batch = []
    try:
        for i in indices:
            batch.append(spec.load_image(int(i)))
        return enc.encode_images(batch, batch_size=32)
    finally:
        for im in batch:
            im.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--smoke', action='store_true', help='16 training images, isolated /tmp; includes 32-image subset gate')
    args = p.parse_args()
    s = load_frozen()
    path = SMOKE / 'extract' if args.smoke else TRAIN_IMAGES
    if (path / 'done.json').exists():
        validate_train_images(s, path, smoke=args.smoke)
        print(f'{path}: verified sealed output; skip', flush=True)
        return
    from scripts.extract_features import adapter_artelingo_full, check_arrays
    spec = adapter_artelingo_full(sample_ids=s['sample_ids'], expected=None)
    assert np.array_equal([m['sample_id'] for m in spec.text_meta], s['sample_ids'])
    adapter_indices = np.asarray([m['img'] for m in spec.text_meta])[s['first']]
    assert np.array_equal([spec.images[int(i)]['painting'] for i in adapter_indices], s['all_painting_ids'])
    assert len(np.unique(adapter_indices)) == len(s['first'])
    assert np.array_equal(adapter_indices[s['subset_pid']], s['subset_image_indices'])
    ids = train_image_ids(s, adapter_indices)
    if args.smoke:
        ids = {k: v[:16] for k, v in ids.items()}
    reference = prompted(s, 'emotion_focus')[0]
    from src.data.feature_extract import Qwen3VLEmb
    start = time.monotonic()
    enc = Qwen3VLEmb(device=args.device, instruction=PROMPTS['emotion_focus'])
    test = np.asarray(encode(enc, spec, s['subset_image_indices'][:32]), dtype=np.float32)
    ref = np.asarray(reference[:32], dtype=np.float32)
    cosine = np.sum(test * ref, axis=1) / (np.linalg.norm(test, axis=1) * np.linalg.norm(ref, axis=1))
    subset_gate = dict(mean_cosine=float(cosine.mean()), min_cosine=float(cosine.min()), n=32)
    print(f"subset re-encode gate: mean_cosine={subset_gate['mean_cosine']:.8f}", flush=True)
    assert subset_gate['mean_cosine'] > 0.999, subset_gate
    path.mkdir(parents=True, exist_ok=True)
    ni = len(ids['dense_indices'])
    img = np.lib.format.open_memmap(path / 'img.npy', mode='w+', dtype=np.float16, shape=(ni, 2048))
    image_start = time.monotonic()
    for a in range(0, ni, 32):
        batch = encode(enc, spec, ids['image_indices'][a:a + 32])
        img[a:a + len(batch)] = batch
    img.flush()
    seconds = time.monotonic() - image_start
    gate = check_arrays(img, np.empty((0, 2048), dtype=np.float16))
    assert gate['ok'], gate
    np.savez(path / 'ids.npz', **ids)
    done = dict(instruction=PROMPTS['emotion_focus'], sample_sha256=fingerprint(s),
                ids_sha256=hashlib.sha256(b''.join(np.ascontiguousarray(ids[k]).tobytes() for k in sorted(ids))).hexdigest(),
                smoke=args.smoke, n_images=ni, image_seconds=seconds, total_seconds=time.monotonic() - start,
                image_items_per_second=ni / seconds, subset_gate=subset_gate, gate=gate)
    atomic_json(path / 'done.json', done)
    validate_train_images(s, path, expected=ids, smoke=args.smoke)
    print(f'{ni} images; {seconds:.1f}s; {ni / seconds:.2f} images/s; sealed {path}', flush=True)


if __name__ == '__main__':
    main()
