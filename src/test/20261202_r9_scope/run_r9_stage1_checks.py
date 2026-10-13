"""Stage 1 store gates (rule section 3.3); no ranking, episodes or verdicts.

Run locally, after the full store is sealed:
  python src/test/20261202_r9_scope/run_r9_stage1_checks.py --backbone clip_b32
Spot re-encoding defaults to CPU; controller may select --device cuda.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_backbone as B  # imports earlier-round src from MAIN
import r9_data as D
import numpy as np
import pandas as pd

# Load this checkout's builder, even though MAIN supplies earlier-round imports.
module_spec = importlib.util.spec_from_file_location("r9_extract_features", HERE.parents[2] / "scripts/extract_features.py")
E = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = E
module_spec.loader.exec_module(E)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    assert np.isfinite(x).all()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norms > 0).all()
    return x / norms


def cosines(a, b):
    assert a.shape == b.shape
    out = np.sum(unit(a) * unit(b), axis=1)
    assert np.isfinite(out).all()
    return out


def clip_checks(spec, images, read):
    data = D.load_data("clip_b32")
    split = D.load_split(data)
    pa = np.asarray(data.paintings)
    held = set(pa[split.held])
    # CSV's non-held paintings: derive from the pinned CLIP split, never dataset IDs.
    train_side = set(pa[np.union1d(split.train, split.val)])
    assert not held & train_side and held | train_side == set(pa)
    read_names = [r["painting"] for r in spec.images]
    held_read = set(read[read.held].painting)
    assert held_read == set(read_names) & held and len(held_read) == 248
    csv = pd.read_csv(E.A28_CSV)
    assert len(csv) == 173_745 and "painting" in csv.columns
    train_names = sorted(train_side & set(csv.painting))
    assert len(train_names) == 912 and len(read_names) == 746
    assert not set(read_names) & train_side
    # Feature row positions come from load_data's CLIP sample order.
    first = {}
    for i, p in enumerate(pa):
        first.setdefault(p, i)
    assert all(p in first for p in held_read | set(train_names))
    held_positions = [i for i, p in enumerate(read_names) if p in held_read]
    joined = cosines(images[held_positions], data.img_features[[first[read_names[i]] for i in held_positions]])
    painting_join = [{"painting": read_names[i], "cosine": float(c)} for i, c in zip(held_positions, joined)]
    assert len(painting_join) == 248 and (joined >= 0.99).all(), "held painting image join failed"
    sim = unit(images) @ unit(data.img_features[[first[p] for p in train_names]]).T
    assert sim.shape == (746, 912) and np.isfinite(sim).all()
    rr, tt = np.where(sim >= 0.99)
    near = {"cos_099": int((sim.max(axis=1) >= 0.99).sum()),
            "cos_098": int((sim.max(axis=1) >= 0.98).sum()), "train_side": 912, "read_paintings": 746,
            "pairs_099": [[read_names[r], train_names[t], float(sim[r, t])] for r, t in zip(rr, tt)]}
    assert near["pairs_099"] == sorted(near["pairs_099"], key=lambda p: (p[0], p[1]))
    return painting_join, near


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True, choices=B.BACKBONES)
    ap.add_argument("--store-root", type=Path, default=E.PRE_EXTRACT)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    spec = E.adapter_artelingo28()
    store = a.store_root / "artelingo28" / a.backbone
    index = json.loads((store / "index.json").read_text())
    assert {"dataset", "backbone", "smoke", "dtype", "dim", "n_images", "n_texts", "images", "texts"} <= index.keys()
    assert (index["dataset"], index["backbone"], index["smoke"], index["dtype"], index["dim"]) == (
        "artelingo28", a.backbone, False, "float16", B.DIMENSIONS[a.backbone])
    # Exact lists enforce row_id order, unique image rows and every C6 painting join.
    assert index["images"] == spec.images and index["texts"] == spec.text_meta, "C6 store alignment"
    assert (index["n_images"], index["n_texts"]) == (746, len(spec.texts))
    img = np.load(store / "img.npy", mmap_mode="r", allow_pickle=False)
    txt = np.load(store / "txt.npy", mmap_mode="r", allow_pickle=False)
    assert img.dtype == txt.dtype == np.float16
    assert img.shape == (746, B.DIMENSIONS[a.backbone])
    assert txt.shape == (len(spec.texts), B.DIMENSIONS[a.backbone])
    gate = E.check_arrays(img, txt)
    assert gate["ok"], gate
    progress = json.loads((store / "progress.json").read_text())
    assert {"n_img", "n_txt", "img_done", "txt_done", "index_sha256", "gate"} <= progress.keys()
    assert (progress["n_img"], progress["n_txt"], progress["img_done"], progress["txt_done"]) == (
        len(img), len(txt), len(img), len(txt))
    assert progress["index_sha256"] == E._sha256(store / "index.json") and progress["gate"]["ok"]
    assert progress["item_sha256"] == E.a28_item_sha256(spec), "A28 input captions or joins changed"
    rng = np.random.default_rng(0)
    image_rows = rng.choice(len(img), 20, replace=False)
    text_rows = rng.choice(len(txt), 20, replace=False)
    # Use MAIN's encoders as required for earlier-round src imports.
    sys.path.insert(0, str(B.MAIN))
    from src.data.feature_extract import load_encoder
    enc = load_encoder(a.backbone, a.device)
    image_cos = cosines(img[image_rows], enc.encode_images([spec.load_image(int(i)) for i in image_rows]))
    text_cos = cosines(txt[text_rows], enc.encode_texts([spec.texts[int(i)] for i in text_rows]))
    assert (image_cos >= 0.99).all() and (text_cos >= 0.99).all(), "spot re-encoding failed"
    report = {"backbone": a.backbone, "n_images": len(img), "n_texts": len(txt), "finite_norm_gate": gate,
              "row_id_join": True, "painting_join_ok": True, "spot_check": {
                  "device": a.device, "seed": 0,
                  "images": [{"row": int(i), "painting": spec.images[i]["painting"], "cosine": float(c)}
                             for i, c in zip(image_rows, image_cos)],
                  "texts": [{"row": int(i), "row_id": spec.text_meta[i]["row_id"], "cosine": float(c)}
                            for i, c in zip(text_rows, text_cos)]}}
    if a.backbone == "clip_b32":
        read = pd.read_parquet(E.A28_READ_SET)
        report["painting_join"], report["near_duplicates"] = clip_checks(spec, img, read)
    B.RESULTS.mkdir(parents=True, exist_ok=True)
    target = B.RESULTS / f"stage1_checks_{a.backbone}.json"
    target.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    back = json.loads(target.read_text())
    assert back == report and len(back["spot_check"]["images"]) == len(back["spot_check"]["texts"]) == 20
    if a.backbone == "clip_b32":
        near_path = B.RESULTS / "near_duplicates_a28.json"
        near_path.write_text(json.dumps(report["near_duplicates"], indent=1, allow_nan=False) + "\n")
        assert json.loads(near_path.read_text()) == back["near_duplicates"]
    print(f"passed: {target}")


if __name__ == "__main__":
    main()
