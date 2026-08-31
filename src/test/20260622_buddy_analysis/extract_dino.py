"""
Held-out image features for the independent-signal NN test (step 3).

DINOv2-small (self-supervised, NOT CLIP) — a representation independent of the
features used to build the buddy graph. Output is aligned to buddy_analysis row
order (data.sample_ids) and cached to dino_feats.npy.
"""
import os, sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

sys.path.insert(0, os.path.dirname(__file__))
import buddy_analysis as ba

OUT = os.path.join(os.path.dirname(__file__), "dino_feats.npy")
MODEL = "facebook/dinov2-small"
BATCH = 64


def main():
    data = ba.load_data()
    files = [os.path.join(ba.IMG_ROOT, r["image"]) for r in data.records]
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
            cls = out.last_hidden_state[:, 0]          # CLS token
            feats[idx] = cls.cpu().float().numpy()
            if start % (BATCH * 20) == 0:
                print(f"  {end}/{data.n}", flush=True)
    np.save(OUT, feats)
    zero = int((np.abs(feats).sum(1) == 0).sum())
    print(f"saved {OUT}  shape={feats.shape}  missing_files={missing}  zero_rows={zero}")


if __name__ == "__main__":
    main()
