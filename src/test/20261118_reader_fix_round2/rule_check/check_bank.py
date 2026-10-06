"""Read-only: bank/halves/reader structure the draft rule assumes (keys, shapes, dtypes, counts)."""
import json, pickle
from pathlib import Path
import numpy as np
R = Path("/project/CoSiR/src/test/20261117_reader_fix_csd/results")
for cfg in ("A0", "A1"):
    r = json.loads((R / f"rb_reader_{cfg}.json").read_text())
    print(cfg, "sklearn", r["sklearn_version"], "numpy", r["numpy_version"], "C", {j: h["chosen_C"] for j, h in r["halves"].items()},
          "n_examples", {j: h["n_examples"] for j, h in r["halves"].items()}, "runtime_s", r["runtime_s"],
          "fit_s", {j: [row["fit_s"] for row in h["cv_table"]] for j, h in r["halves"].items()},
          "class_counts", {j: h["class_counts"] for j, h in r["halves"].items()})
    print("   feature_names", r["feature_names"])
h = json.loads((R / "rb_halves.json").read_text())
print("halves rows_per_half", h["rows_per_half"], "paintings", h["paintings_per_half"])
zh = np.load(R / "rb_halves.npz")
for k in zh.files: print("  halves", k, zh[k].dtype, zh[k].shape)
for cfg in ("A0", "A1"):
    for j in (0, 1):
        z = np.load(R / f"rb_bank_{cfg}_half{j}.npz")
        print(cfg, j, {k: (str(z[k].dtype), z[k].shape) for k in z.files})
        print("   block_pairs", z["block_pairs"].tolist(), "block_size", int(z["block_size"]))
        rows = zh[f"local_rows_half{j}"]
        paint = zh["painting_of_local_row"]
        A = z["anchor"].astype(np.int64)
        print("   anchor in half rows:", np.isin(A, rows).all(), " pairs img==txt same row share:",
              float(np.mean(z["pairs_a_img"] == z["pairs_a_txt"])),
              " pair img/txt same painting share:", float(np.mean(paint[z["pairs_a_img"]] == paint[z["pairs_a_txt"]])))
        allrows = np.concatenate([A[:, None], z["candidates"], z["pairs_a_img"], z["pairs_a_txt"], z["pairs_b_img"], z["pairs_b_txt"]], axis=1)
        p = paint[allrows]
        sp = np.sort(p, axis=1)
        print("   episodes with all-distinct paintings:", float(np.mean((np.diff(sp, axis=1) != 0).all(axis=1))))
d = json.loads((R / "rb_diag_A0.json").read_text())
smd = d["c_shift_report"]["smd"]
print("diag A0 smd n", len(smd), "all finite", all(v is not None for v in smd.values()))
print("  smd", {k: v for k, v in smd.items()})
print("  n_seed42 n_bank", d["c_shift_report"]["n_seed42"], d["c_shift_report"]["n_bank"])
print("  top prob", d["top_probability"]["seed42_both_conditions"]["mean"], d["top_probability"]["bank_oof_pooled_halves"]["mean"])
zr = np.load(R / "rb_reader_A0.npz")
print("reader npz", {k: (str(zr[k].dtype), zr[k].shape) for k in zr.files})
with open(R / "rb_reader_A0.pkl", "rb") as f:
    pk = pickle.load(f)
for i, hh in enumerate(pk["halves"]):
    sc = hh["scaler"]
    print("half", i, "C", hh["C"], "scaler mean dtype", sc.mean_.dtype, "scale min", sc.scale_.min(), "var min", sc.var_.min(), "n_samples_seen", sc.n_samples_seen_)
# A1 diag
d1 = json.loads((R / "rb_diag_A1.json").read_text())
print("diag A1 smd all finite", all(v is not None for v in d1["c_shift_report"]["smd"].values()), len(d1["c_shift_report"]["smd"]))
