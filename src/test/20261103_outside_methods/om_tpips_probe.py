"""r18 item 13: TPIPS timing probe (released 8B late-fusion model, `sywang/TPIPS-Embed-Qwen3VL-8B`).

Measures ms per image of `model.embed(images, factor=...)` on the local GPU at a few batch sizes, the peak GPU memory,
and the visual token count per image. Also a sanity check: embeddings of one image under two factors differ.
Run in the tpips venv (/root/venvs/tpips, CoSiR packages plus tpips 0.1.2), under the GPU lock:

  flock -n -o -E 75 /tmp/gpu0.lock env PYTHONDONTWRITEBYTECODE=1 /root/venvs/tpips/bin/python om_tpips_probe.py
"""
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

import tpips

ROUND = Path(__file__).resolve().parent
OUT = ROUND / 'records' / 'tpips_probe.json'
N = 48
BATCHES = (1, 4, 8)
FACTORS = ('emotion', 'style', 'genre')


def load_images(n):
    t = pq.read_table('/data/SSD/mmart/mmart_index.parquet', columns=['key', 'image_path', 'split']).to_pylist()
    rows = sorted((r for r in t if r['split'] == 'new'), key=lambda r: r['key'])[:n]
    ims = [Image.open(r['image_path']).convert('RGB') for r in rows]
    return [r['key'] for r in rows], ims


def main():
    keys, ims = load_images(N)
    sizes = [im.size for im in ims]
    t0 = time.time()
    model = tpips.load_model('embedding', device='cuda')
    load_s = time.time() - t0
    attn = getattr(model.backbone.config, '_attn_implementation', None) if hasattr(model, 'backbone') else None

    # visual tokens per image, from the processor's grid (patch 16, merge 2)
    proc = model.processor.image_processor if hasattr(model, 'processor') else None
    tokens = None
    if proc is not None:
        try:
            from tpips.embedding.qwen3_vl_embed import MAX_PIXELS, MIN_PIXELS
            import numpy as np
            grids = []
            for im in ims:
                g = proc(images=[im], size={'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS},
                         return_tensors='pt')['image_grid_thw']
                grids.append(int(g.prod()) // 4)
            tokens = {'mean': float(np.mean(grids)), 'min': int(min(grids)), 'max': int(max(grids))}
        except Exception as exc:  # token count is a side fact; timing must still run
            tokens = f'not measured: {exc!r}'[:200]

    res = {'n_images': N, 'load_s': round(load_s, 1), 'attn': attn, 'visual_tokens': tokens,
           'image_px_mean': sum(w * h for w, h in sizes) / len(sizes), 'gpu': torch.cuda.get_device_name(0),
           'batches': {}}
    with torch.inference_mode():
        model.embed(ims[:2], factor='emotion')  # warm-up
        torch.cuda.synchronize()
        for b in BATCHES:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.time()
            for i in range(0, N, b):
                model.embed(ims[i:i + b], factor='emotion')
            torch.cuda.synchronize()
            dt = time.time() - t
            res['batches'][b] = {'ms_per_image': round(1000 * dt / N, 1),
                                 'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            print(b, res['batches'][b], flush=True)
        # sanity: one image under the three factors (normalized), pairwise cosines
        e = torch.stack([model.embed(ims[0], factor=f, normalized=True) for f in FACTORS])
        cos = (e @ e.T).cpu().tolist()
        res['factor_cos_image0'] = {f'{FACTORS[i]}|{FACTORS[j]}': round(cos[i][j], 4)
                                    for i in range(3) for j in range(i + 1, 3)}
        res['dim'] = int(e.shape[1])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    sys.exit(main())
