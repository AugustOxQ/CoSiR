"""Pick CUB's third aspect (spec 11 E0). Development species only. The 50 zero-shot test species are never used:
load_cub() reads every image's metadata, the test species' entries serve only to exclude them, and only
training-species feature rows are loaded."""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.cub import CUB_ROOT, dev_species, load_cub, zero_shot_split

GROUPS = ("has_shape", "has_wing_pattern", "has_breast_pattern", "has_wing_color")
CLIP_DIR = Path("/data/SSD2/pre_extract/backbone_check/cub/clip")
OUT = Path(__file__).parent / "results" / "cub_third_aspect.json"


def main():
    cub = load_cub()
    train_idx, test_idx = zero_shot_split(cub.species, CUB_ROOT / "xlsa17")
    test_set = set(test_idx.tolist())
    dev = dev_species(cub.species[train_idx], n_dev=30, seed=42)
    in_dev = np.isin(cub.species[train_idx], dev)
    dev_idx, probe_idx = train_idx[in_dev], train_idx[~in_dev]
    assert not (set(dev_idx.tolist()) | set(probe_idx.tolist())) & test_set
    print(f"probe-train species images: {len(probe_idx)} ({len(np.unique(cub.species[probe_idx]))} species); "
          f"dev species images: {len(dev_idx)} ({len(dev)} species); test-species images excluded: {len(test_idx)}")

    clip_paths = json.load(open(CLIP_DIR / "index.json"))["image_paths"]
    key = lambda p: "/".join(Path(p).parts[-2:])
    row_of = {key(p): r for r, p in enumerate(clip_paths)}
    rows = np.array([row_of[key(p)] for p in cub.paths[train_idx]])   # covers every train-species image or KeyError
    assert len(rows) == len(train_idx)
    img_all = np.load(CLIP_DIR / "img.npy", mmap_mode="r")
    txt_all = np.load(CLIP_DIR / "txt.npy", mmap_mode="r")
    where = {int(i): k for k, i in enumerate(train_idx)}              # CUB index -> position in train set
    pick = lambda idx: np.array([rows[where[int(i)]] for i in idx])
    feats = {}
    for name, idx in (("probe", probe_idx), ("dev", dev_idx)):
        r = pick(idx)
        order = np.argsort(r)                                         # sorted reads from the memmap, train rows only
        img = np.empty((len(r), img_all.shape[1]), dtype=np.float32)
        txt = np.empty((len(r), txt_all.shape[2]), dtype=np.float32)
        img[order] = np.asarray(img_all[r[order]])
        txt[order] = np.asarray(txt_all[r[order]]).mean(axis=1)
        feats[name] = (img, txt)

    table = []
    for g in GROUPS:
        lab_p, lab_d = cub.attributes[g][probe_idx], cub.attributes[g][dev_idx]
        mp, md = lab_p >= 0, lab_d >= 0
        majority = float(np.bincount(lab_d[md]).max() / md.sum())
        acc = {}
        for k, modality in enumerate(("image", "caption")):
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(feats["probe"][k][mp], lab_p[mp])
            acc[modality] = float(clf.score(feats["dev"][k][md], lab_d[md]))
        table.append({"group": g, "image_acc": acc["image"], "caption_acc": acc["caption"], "majority": majority,
                      "score": min(acc.values()) - majority, "n_probe": int(mp.sum()), "n_dev": int(md.sum())})
    best = max(table, key=lambda r: r["score"])
    print(f"{'group':20s} {'image':>7s} {'caption':>8s} {'majority':>9s} {'min-maj':>8s} {'n_probe':>8s} {'n_dev':>6s}")
    for r in table:
        print(f"{r['group']:20s} {r['image_acc']:7.3f} {r['caption_acc']:8.3f} {r['majority']:9.3f} "
              f"{r['score']:8.3f} {r['n_probe']:8d} {r['n_dev']:6d}")
    print(f"picked: {best['group']}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"table": table, "picked": best["group"], "n_probe_images": int(len(probe_idx)),
                               "n_dev_images": int(len(dev_idx)), "dev_species": dev.tolist()}, indent=1))


if __name__ == "__main__":
    main()
