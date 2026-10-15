"""Fidelity checks for src/data/feature_extract.py (CVPR plan Task 6, Step 5).

1. CLIP: re-encode 64 ArtELingo SELECTION rows (image and caption); cosine >= 0.999 to the cached load_artelingo() rows.
2. Qwen: 50 CUB images and 50 captions from TRAIN species only, encoded by load_encoder("qwen3vl_emb_2b") and by the
   official Qwen3VLEmbedder (snapshot scripts/qwen3_vl_embedding.py, transformers 4.57.6 from
   /data/SSD2/pyenvs/qwen_official, run in a subprocess). Pass: every cosine >= 0.98 and image-to-caption top-1
   agreement on >= 48 of 50 queries. On failure, one retry with max_pixels=1310720 on both sides.

Run: flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python <this file>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

OFFICIAL_PATH = "/data/SSD2/pyenvs/qwen_official"
SNAPSHOT = ("/data/SSD2/HF_home/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/"
            "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda")
WIKIART = Path("/data/PDD/wikiart_proj/wikiart")
CUB_XLSA = "/data/SSD/cub/xlsa17"
SCRATCH = Path(os.environ.get("FIDELITY_SCRATCH", "/tmp/claude-0/qwen_fidelity"))
N_CLIP, N_QWEN = 64, 50
CLIP_MIN_COS, QWEN_MIN_COS, QWEN_MIN_TOP1 = 0.999, 0.98, 48
MAX_PIXELS_BACKBONE = 512 * 32 * 32
MAX_PIXELS_RETRY = 1_310_720


def official_main(inputs_json, out_npz, max_pixels, instruction):
    """Subprocess entry: the official embedder, nothing from src/."""
    sys.path.insert(0, OFFICIAL_PATH)
    sys.path.insert(0, f"{SNAPSHOT}/scripts")
    import numpy as np
    import torch
    from PIL import Image
    from qwen3_vl_embedding import Qwen3VLEmbedder
    spec = json.load(open(inputs_json))
    emb = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B", max_pixels=max_pixels,
                          default_instruction=instruction, torch_dtype=torch.bfloat16)

    def run(items, bs=8):
        out = []
        for i in range(0, len(items), bs):
            out.append(emb.process(items[i:i + bs]).float().cpu().numpy())
        return np.concatenate(out)

    img = run([{"image": Image.open(p).convert("RGB"), "instruction": instruction} for p in spec["images"]])
    txt = run([{"text": t, "instruction": instruction} for t in spec["texts"]])
    np.savez(out_npz, img=img, txt=txt)
    import transformers
    print(f"official side: transformers {transformers.__version__}, numpy {np.__version__}, max_pixels {max_pixels}")


def clip_check():
    import numpy as np
    from PIL import Image
    from src.data.artelingo import ANNOTATIONS_PATH, join_captions, load_artelingo
    from src.data.artelingo_splits import artelingo_splits
    from src.data.feature_extract import load_encoder
    data = load_artelingo()
    sel = artelingo_splits(data).selection
    rows = np.sort(np.random.default_rng(0).choice(sel, N_CLIP, replace=False))
    assert np.isin(rows, sel).all()
    ann = json.load(open(ANNOTATIONS_PATH))
    sid = data.sample_ids[rows]
    caps = list(join_captions(sid, ann))
    paths = [WIKIART / ann[int(s)]["image"] for s in sid]
    enc = load_encoder("clip_b32", device="cuda")
    img = enc.encode_images([Image.open(p).convert("RGB") for p in paths], batch_size=32)
    txt = enc.encode_texts(caps, batch_size=32)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)       # the cache stores unnormalised projections
    print(f"cached feature norms: image {np.linalg.norm(data.img_features[rows], axis=1).mean():.2f}, "
          f"text {np.linalg.norm(data.txt_features[rows], axis=1).mean():.2f}")
    ci = (img * unit(data.img_features[rows])).sum(1)
    ct = (txt * unit(data.txt_features[rows])).sum(1)
    print(f"CLIP image cosine min/median/max {ci.min():.5f}/{np.median(ci):.5f}/{ci.max():.5f}; "
          f"text cosine min/median/max {ct.min():.5f}/{np.median(ct):.5f}/{ct.max():.5f}")
    ok = min(ci.min(), ct.min()) >= CLIP_MIN_COS
    print("CLIP PASS" if ok else f"CLIP FAIL (min cosine {min(ci.min(), ct.min()):.5f} < {CLIP_MIN_COS})")
    return ok


def cub_inputs():
    import numpy as np
    from src.data.cub import CUB_ROOT, load_cub, zero_shot_split
    cub = load_cub()
    train_idx, test_idx = zero_shot_split(cub.species, CUB_XLSA)
    test_species = set(cub.species[test_idx].tolist())
    pick = np.random.default_rng(0).choice(train_idx, N_QWEN, replace=False)
    assert not test_species & set(cub.species[pick].tolist())
    paths = [str(CUB_ROOT / "CUB_200_2011" / "images" / cub.paths[i]) for i in pick]
    caps = [cub.captions[i][0] for i in pick]
    return paths, caps


def large_inputs(n_huge_min=12):
    """50 ArtELingo SELECTION-row images that need downscaling (> 524,288 px; >= 10 above 1,843,200), distinct paintings,
    seed-42 order, each with the caption of one selection row of its painting."""
    import numpy as np
    from PIL import Image
    from src.data.artelingo import ANNOTATIONS_PATH, join_captions, load_artelingo
    from src.data.artelingo_splits import artelingo_splits
    data = load_artelingo()
    sel = artelingo_splits(data).selection
    ann = json.load(open(ANNOTATIONS_PATH))
    sid = data.sample_ids[sel]
    caps = join_captions(sid, ann)
    first = {}
    for k, row_paint in enumerate(np.asarray(data.paintings)[sel].tolist()):
        first.setdefault(row_paint, k)                                  # one selection row per painting
    keys = sorted(first)
    order = np.random.default_rng(42).permutation(len(keys))
    huge, mid = [], []
    for j in order:
        k = first[keys[j]]
        path = WIKIART / ann[int(sid[k])]["image"]
        with Image.open(path) as im:
            px = im.size[0] * im.size[1]
        if px > 1_843_200 and len(huge) < n_huge_min:
            huge.append((path, caps[k], px))
        elif 524_288 < px <= 1_843_200 and len(mid) < N_QWEN - n_huge_min:
            mid.append((path, caps[k], px))
        if len(huge) == n_huge_min and len(mid) == N_QWEN - n_huge_min:
            break
    items = huge + mid
    assert len(items) == N_QWEN and len(huge) >= 10 and all(px > 524_288 for _, _, px in items)
    assert np.isin(np.array([first[k] for k in keys]), np.arange(len(sel))).all()
    return [str(a) for a, _, _ in items], [str(b) for _, b, _ in items], [px for _, _, px in items]


def qwen_check(max_pixels, tag, paths, caps):
    import gc
    import numpy as np
    import torch
    from PIL import Image
    from src.data import feature_extract as fe
    SCRATCH.mkdir(parents=True, exist_ok=True)
    spec, out = SCRATCH / "inputs_{tag}.json", SCRATCH / f"official_{tag}_{max_pixels}.npz"
    json.dump({"images": paths, "texts": caps}, open(spec, "w"))
    enc = fe.Qwen3VLEmb("cuda", fe.QWEN_INSTRUCTION_DEFAULT, max_pixels)
    ours_img = enc.encode_images([Image.open(p).convert("RGB") for p in paths], batch_size=8)
    ours_txt = enc.encode_texts(caps, batch_size=16)
    del enc
    gc.collect()
    torch.cuda.empty_cache()
    env = dict(os.environ)
    env["PYTHONPATH"] = OFFICIAL_PATH
    r = subprocess.run([sys.executable, __file__, "--official", str(spec), str(out), str(max_pixels),
                        fe.QWEN_INSTRUCTION_DEFAULT], env=env, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError("official side failed")
    off = dict(np.load(out))
    off = {k: v / np.linalg.norm(v, axis=1, keepdims=True) for k, v in off.items()}   # official pools in bf16: renormalise in fp32
    ci, ct = (ours_img * off["img"]).sum(1), (ours_txt * off["txt"]).sum(1)
    allc = np.concatenate([ci, ct])
    s_ours, s_off = ours_img @ ours_txt.T, off["img"] @ off["txt"].T
    agree = int((s_ours.argmax(1) == s_off.argmax(1)).sum())
    agree_t2i = int((s_ours.argmax(0) == s_off.argmax(0)).sum())
    acc_o, acc_f = int((s_ours.argmax(1) == np.arange(N_QWEN)).sum()), int((s_off.argmax(1) == np.arange(N_QWEN)).sum())
    print(f"[{tag}] max_pixels={max_pixels}: image cosine min/median/max {ci.min():.4f}/{np.median(ci):.4f}/{ci.max():.4f}; "
          f"caption cosine {ct.min():.4f}/{np.median(ct):.4f}/{ct.max():.4f}; "
          f"all {allc.min():.4f}/{np.median(allc):.4f}/{allc.max():.4f}")
    print(f"[{tag}] GATED direction: image-to-caption (i2t) top-1 agreement {agree}/{N_QWEN} (rule >= {QWEN_MIN_TOP1}); "
          f"text-to-image (t2i, not gated) {agree_t2i}/{N_QWEN}; "
          f"matched-pair top-1 ours {acc_o}/{N_QWEN}, official {acc_f}/{N_QWEN}")
    ok = bool(allc.min() >= QWEN_MIN_COS and agree >= QWEN_MIN_TOP1)
    return ok, allc, agree


def verdict_line(name, ok, mp, allc, agree):
    import numpy as np
    print(f"{name} {'PASS' if ok else 'FAIL'} (max_pixels={mp}): min/median/max cosine {allc.min():.4f}/"
          f"{np.median(allc):.4f}/{allc.max():.4f}, i2t top-1 agreement {agree}/{N_QWEN}")


def main():
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    clip_check()
    paths, caps = cub_inputs()
    px = []
    for p in paths:
        with Image.open(p) as im:
            px.append(im.size[0] * im.size[1])
    print(f"CUB stage images span {min(px):,}-{max(px):,} px (no downscaling)")
    for mp in (MAX_PIXELS_BACKBONE, MAX_PIXELS_RETRY):
        ok, allc, agree = qwen_check(mp, "CUB", paths, caps)
        verdict_line("QWEN", ok, mp, allc, agree)
        if ok:
            break
    lp, lc, lpx = large_inputs()
    print(f"large stage images span {min(lpx):,}-{max(lpx):,} px; {sum(p > 1_843_200 for p in lpx)} above 1,843,200 "
          f"(all above {MAX_PIXELS_BACKBONE:,}, so every image is downscaled)")
    ok, allc, agree = qwen_check(MAX_PIXELS_BACKBONE, "LARGE", lp, lc)
    verdict_line("QWEN-LARGE", ok, MAX_PIXELS_BACKBONE, allc, agree)
    print("final Qwen features use max_pixels 524,288 (512 tokens), below the official default 1,843,200")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--official":
        official_main(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        main()
