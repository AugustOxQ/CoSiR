"""Final review B1: ArtELingo-28 store joins (three backbones), pins re-hash, painting join and near-duplicates.

Own code, no r9_* import. Reads the stores, the read set (proven equal to fr_b1_counts.py's own table), the CSV, the
zip's member list, ArtELingo's CLIP image features (load_artelingo, frozen src) and the frozen split files. Image
features only (the stage-1 checks); no caption feature is compared with anything, nothing is scored.
Writes fr_b1_stores.json beside this file.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/project/CoSiR-wt/r9-fr-b")
sys.path.insert(0, str(ROOT))
from src.data.artelingo import load_artelingo  # noqa: E402

HERE = Path(__file__).resolve().parent
F = HERE.parent
REC = Path("/project/CoSiR/src/test/20261202_r9_scope/results")
A28 = Path("/data/PDD/artelingo/ArtELingo/semeval-artelingo28")
STORE = Path("/data/SSD2/pre_extract/artelingo28")
TEST = Path("/project/CoSiR/src/test")
DIMS = {"clip_b32": 512, "qwen3vl_emb_2b": 2048, "siglip2_so400m": 1152}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


out = {}
# 1. pins re-hash
pins = json.loads((F / "pins_a28.json").read_text())
out["pins_n"] = len(pins["files"])
out["pins_bad"] = [k for k, v in pins["files"].items() if sha(v["path"]) != v["sha256"]]
rule_list = ([f"translations/shard_{i}/{f}" for i in range(4) for f in ("translations.parquet", "coverage.json")]
             + [f"translations/nllb200_3.3b_en/{f}" for f in ("translations.parquet", "coverage.json")]
             + [f"store/{b}/{f}" for b in DIMS for f in ("img.npy", "txt.npy")]
             + [f"results/{f}" for f in ("value_sets_a28.json", "language_flags_a28.json", "read_set_a28.parquet")])
out["pins_missing_rule_items"] = sorted(set(rule_list) - set(pins["files"]))
out["pins_extra_items"] = sorted(set(pins["files"]) - set(rule_list))
meta = json.loads((STORE / "nllb200_3.3b_en/meta.json").read_text())
out["merged_from"] = meta.get("merged_from")
# do the shards reassemble into the merged table?
shards = pd.concat([pd.read_parquet(pins["files"][f"translations/shard_{i}/translations.parquet"]["path"])
                    for i in range(4)]).sort_values("csv_row").reset_index(drop=True)
merged = pd.read_parquet(STORE / "nllb200_3.3b_en/translations.parquet").sort_values("csv_row").reset_index(drop=True)
out["shards_equal_merged"] = bool(list(shards.columns) == list(merged.columns) and shards.equals(merged))

# 2. store joins
read = pd.read_parquet(REC / "read_set_a28.parquet")
kept = read[~read.dropped]
csv = pd.read_csv(A28 / "artelingo28_public_noworkerid_split.csv")
img_name = csv.groupby("painting").image_name.agg(lambda x: sorted(set(x)))
with zipfile.ZipFile(A28 / "images.zip") as z:
    members = z.namelist()
mcount = pd.Series(members).value_counts()
paint_sorted = sorted(set(read.painting))
stores = {}
for b, dim in DIMS.items():
    d = STORE / b
    idx = json.loads((d / "index.json").read_text())
    img = np.load(d / "img.npy", mmap_mode="r")
    txt = np.load(d / "txt.npy", mmap_mode="r")
    texts, images = idx["texts"], idx["images"]
    ipaint = [im["painting"] for im in images]
    r = {"header": [idx[k] for k in ("dataset", "backbone", "smoke", "dtype", "dim", "n_images", "n_texts")],
         "img_shape": list(img.shape), "txt_shape": list(txt.shape), "dtype": [str(img.dtype), str(txt.dtype)],
         "n_texts_eq_kept": len(texts) == len(kept),
         "row_id_order": [t["row_id"] for t in texts] == kept.row_id.tolist(),
         "painting_eq": [t["painting"] for t in texts] == kept.painting.tolist(),
         "img_points_own": all(ipaint[t["img"]] == t["painting"] for t in texts),
         "one_image_per_painting": len(ipaint) == len(set(ipaint)) == 746,
         "images_sorted_eq_read": ipaint == paint_sorted,
         "image_path_eq_csv": all(len(img_name[im["painting"]]) == 1 and im["image"] == "images/" + img_name[im["painting"]][0]
                                  for im in images),
         "zip_member_once": all(mcount.get(im["image"], 0) == 1 for im in images)}
    for name, x in (("img", img), ("txt", txt)):
        x = np.asarray(x, dtype=np.float32)
        nrm = np.linalg.norm(x, axis=1)
        r[f"{name}_finite"] = bool(np.isfinite(x).all())
        r[f"{name}_norm_range"] = [float(nrm.min()), float(nrm.max())]
    prog = json.loads((d / "progress.json").read_text())
    r["progress_index_sha_ok"] = prog.get("index_sha256") == sha(d / "index.json")
    stores[b] = r
out["stores"] = stores

# 3. painting join (248 held) and near-duplicates, CLIP image features
data = load_artelingo()
pa = np.asarray(data.paintings)
with np.load(TEST / "20261014_stage_d_final/cache/held_codes.npz") as z:
    held = set(pa[z["held_rows"]])
first, spread = {}, {}
for i, p in enumerate(pa):
    first.setdefault(p, i)
every = sorted(first)
E = np.asarray(data.img_features[[first[p] for p in every]], dtype=np.float32)
# is a painting's image feature identical over its rows? (max abs deviation from its first row)
dev = np.abs(np.asarray(data.img_features, dtype=np.float32) - np.asarray(data.img_features, dtype=np.float32)[
    np.asarray([first[p] for p in pa])]).max()
out["artelingo_image_same_within_painting_maxdev"] = float(dev)
E /= np.linalg.norm(E, axis=1, keepdims=True)
idx = json.loads((STORE / "clip_b32/index.json").read_text())
names = [im["painting"] for im in idx["images"]]
Z = np.asarray(np.load(STORE / "clip_b32/img.npy"), dtype=np.float32)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
col = {p: j for j, p in enumerate(every)}
hp = [i for i, p in enumerate(names) if p in held]
S = Z[hp] @ E.T
own = np.array([col[names[i]] for i in hp])
oc = S[np.arange(len(hp)), own]
S2 = S.copy()
S2[np.arange(len(hp)), own] = -np.inf
nxt = S2.max(axis=1)
out["join_n"] = len(hp)
out["join_all_own_nearest"] = bool((S.argmax(axis=1) == own).all())
out["join_cos_min_max"] = [float(oc.min()), float(oc.max())]
out["join_cos_below_099"] = int((oc < 0.99).sum())
out["join_margin_min"] = float((oc - nxt).min())
s1 = json.loads((REC / "stage1_checks_clip_b32.json").read_text())
pj = {r["painting"]: r for r in s1["painting_join"]}
out["join_record_maxabs_cos"] = float(max(abs(pj[names[i]]["cosine"] - c) for i, c in zip(hp, oc)))
out["join_record_maxabs_margin"] = float(max(abs(pj[names[i]]["margin"] - (c - m)) for i, c, m in zip(hp, oc, nxt)))
out["join_record_paintings_eq"] = sorted(pj) == sorted(names[i] for i in hp)
train_side = sorted((set(first) - held) & set(csv.painting))
out["train_side_n"] = len(train_side)
T = E[[col[p] for p in train_side]]
mx = (Z @ T.T).max(axis=1)
near = {f"cos_{t}": int((mx >= v).sum()) for t, v in (("099", 0.99), ("098", 0.98), ("092", 0.92))}
near["top5_max_cos"] = sorted(mx.tolist(), reverse=True)[:5]
near["n_090"] = int((mx >= 0.90).sum())
# a cut that a float16-vs-float32 rounding could move: values within 1e-3 of 0.92
near["within_1e3_of_092"] = int((np.abs(mx - 0.92) < 1e-3).sum())
out["near_mine"] = near
nd = json.loads((REC / "near_duplicates_a28.json").read_text())
out["near_record"] = {k: v for k, v in nd.items() if k != "pairs_099"}
out["near_record_pairs_099"] = len(nd.get("pairs_099", []))
# held-only and new-only split of the >= 0.92 count (the report states the version gap)
out["near_092_held_new"] = [int(((mx >= 0.92) & np.isin(names, list(held))).sum()),
                            int(((mx >= 0.92) & ~np.isin(names, list(held))).sum())]
(HERE / "fr_b1_stores.json").write_text(json.dumps(out, indent=1, default=str))
print("written")
