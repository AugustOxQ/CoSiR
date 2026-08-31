"""
Held-out image features for the independent-signal NN test (RedCaps).

DINOv2-small (self-supervised, NOT CLIP) — a representation independent of the
CLIP features used to build the buddy graph. Output is aligned to buddy-analysis
row order (data.sample_ids) and cached to dino_feats.npy.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, os.path.dirname(__file__))
import redcaps_buddy as rb

OUT = os.path.join(os.path.dirname(__file__), "dino_feats.npy")
MODEL = "facebook/dinov2-small"
BATCH = 128


def main():
    if os.path.exists(OUT):
        print(f"{OUT} already exists — nothing to do.")
        return

    data = rb.load_data()
    files = [os.path.join(rb.IMG_ROOT, r["image"]) for r in data.records]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    feats = np.zeros((data.n, model.config.hidden_size), dtype=np.float32)
    missing = 0
    with torch.no_grad():
        for start in range(0, data.n, BATCH):
            end = min(start + BATCH, data.n)
            imgs, idx = [], []
            for i in range(start, end):
                try:
                    imgs.append(Image.open(files[i]).convert("RGB"))
                    idx.append(i)
                except (FileNotFoundError, OSError):
                    missing += 1
            if not imgs:
                continue
            inp = proc(images=imgs, return_tensors="pt").to(device)
            out = model(**inp)
            cls = out.pooler_output if out.pooler_output is not None else \
                out.last_hidden_state[:, 0]
            feats[idx] = cls.float().cpu().numpy()
            if (start // BATCH) % 50 == 0:
                print(f"  {end}/{data.n}  (missing so far: {missing})")

    np.save(OUT, feats)
    print(f"Saved {OUT}: {feats.shape}, missing images: {missing}")


if __name__ == "__main__":
    main()
