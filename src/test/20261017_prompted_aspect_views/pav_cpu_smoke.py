"""Optional bounded offline CPU encoder check: two images and four captions.

Writes no features. Invoke under timeout 230 with CUDA_VISIBLE_DEVICES empty.
"""
import time
import numpy as np
from pav_sample import load_frozen
from pav_prompts import PROMPTS


def main():
    from scripts.extract_features import adapter_artelingo_full, check_arrays
    from src.data.feature_extract import Qwen3VLEmb
    start = time.monotonic()
    s = load_frozen()
    spec = adapter_artelingo_full(sample_ids=s['sample_ids'], expected=None)
    enc = Qwen3VLEmb(device='cpu', instruction=PROMPTS['default'])
    images = [spec.load_image(int(i)) for i in s['subset_image_indices'][:2]]
    try:
        img = enc.encode_images(images, batch_size=1)
    finally:
        for im in images:
            im.close()
    txt = enc.encode_texts(spec.texts[:4], batch_size=4)
    assert img.shape == (2, 2048) and txt.shape == (4, 2048)
    gate = check_arrays(img.astype(np.float16), txt.astype(np.float16))
    assert gate['ok'], gate
    print(f'CPU smoke: {img.shape}, {txt.shape}, gate={gate}, seconds={time.monotonic()-start:.1f}', flush=True)


if __name__ == '__main__':
    main()
