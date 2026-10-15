"""Final review: selection under spec section 4 from the nine metrics.json, and val retrieval recomputed with own code
from every features.npz (best epoch) and features_epoch0.npz (epoch 0), against metrics.json's integer counts.

Run (CPU): CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python fr_selection.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402

JOBS = ROOT / "res/cluster_jobs"
RUNS = ROOT / ".superpowers/sdd/2026-10-07-clip-lightweight-ft/runs.txt"


def retrieval(img_rows, txt_rows, paintings_of_rows):
    """One image per painting (its first row in ascending row order), all captions; ties to the first index."""
    uniq, first, pos = np.unique(paintings_of_rows, return_index=True, return_inverse=True)
    I = img_rows[first].astype(np.float32)
    I /= np.linalg.norm(I, axis=1, keepdims=True)
    T = txt_rows.astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    P, N = len(I), len(T)
    best_val = np.full(P, -np.inf, np.float32)
    best_idx = np.zeros(P, np.int64)
    t2i = 0
    for c0 in range(0, N, 4096):
        S = I @ T[c0:c0 + 4096].T                       # (P, c)
        t2i += int((S.argmax(0) == pos[c0:c0 + 4096]).sum())
        v, ix = S.max(1), S.argmax(1)
        upd = v > best_val
        best_val[upd], best_idx[upd] = v[upd], ix[upd] + c0
    i2t = int((pos[best_idx] == np.arange(P)).sum())
    return {"i2t_correct": i2t, "t2i_correct": t2i, "n_images": P, "n_captions": N,
            "selection": (i2t / P + t2i / N) / 2}


def main():
    data = load_artelingo()
    sp = artelingo_splits(data)
    val = np.sort(np.asarray(sp.val))
    sel = np.sort(np.asarray(sp.selection))
    paint = np.asarray(data.paintings)
    expect_rows = np.sort(np.concatenate([val, sel]))
    runs = []
    for line in RUNS.read_text().split("\n"):
        if not line.strip():
            continue
        node, variant, lr, ok, tag = line.split()[:5]
        d = next((JOBS / tag / "code/outputs/clipft").iterdir())
        m = json.loads((d / "metrics.json").read_text())
        rr = json.loads((d / "run_record.json").read_text())
        runs.append({"node": node, "variant": variant, "lr_txt": lr, "tag": tag, "dir": d, "m": m, "rr": rr})

    out = {"runs": [], "selected": {}}
    # selection rule (own): per variant, max selection over epochs >= 1; ties smaller lr then earlier epoch
    for v in ("LP", "LB", "LoRA"):
        cands = []
        for r in runs:
            if r["variant"] != v:
                continue
            assert float(r["m"]["lr"]) == float(r["lr_txt"])
            for e in r["m"]["epochs"]:
                if e["epoch"] >= 1:
                    # recompute the metric from the integer counts
                    s = (e["i2t_correct"] / e["n_images"] + e["t2i_correct"] / e["n_captions"]) / 2
                    assert abs(s - e["selection"]) < 1e-15
                    cands.append((s, float(r["m"]["lr"]), e["epoch"], r["tag"]))
        top = max(c[0] for c in cands)
        ties = sorted([c for c in cands if c[0] == top], key=lambda c: (c[1], c[2]))
        out["selected"][v] = {"selection": top, "lr": ties[0][1], "epoch": ties[0][2], "tag": ties[0][3],
                              "n_tied": len(ties),
                              "margin_to_next": top - max(c[0] for c in cands if c[0] < top)}

    for r in runs:
        m, rr = r["m"], r["rr"]
        eps = m["epochs"]
        best = max((e for e in eps if e["epoch"] >= 1), key=lambda e: (e["selection"], -e["epoch"]))
        rec = {"tag": r["tag"], "variant": r["variant"], "lr": m["lr"], "best_epoch_metrics": m["best_epoch"],
               "best_epoch_mine": best["epoch"], "best_selection": best["selection"],
               "epoch0": {k: eps[0][k] for k in ("i2t_correct", "t2i_correct", "selection")},
               "per_epoch_selection": [round(100 * e["selection"], 4) for e in eps],
               "i2t_gt_t2i_at_best": best["i2t_r1"] > best["t2i_r1"],
               "rr_best_epoch": rr["best_epoch"], "commit": rr["git_commit"], "dirty": rr["git_dirty_in_code_folder"],
               "n_trainable": rr["n_trainable_params"], "training": rr["training"], "data": rr["data"],
               "checks": rr["checks"], "gpu": rr.get("gpu_name"), "host": rr.get("hostname"),
               "seconds": [e["seconds"] for e in eps]}
        for fname, ep in (("features.npz", best), ("features_epoch0.npz", eps[0])):
            z = np.load(r["dir"] / fname)
            rows = z["rows"]
            assert np.array_equal(rows, expect_rows), f"{r['tag']} {fname}: rows are not exactly val+selection sorted"
            isval = np.isin(rows, val)
            got = retrieval(z["img"][isval], z["txt"][isval], paint[rows[isval]])
            rec[fname] = {"mine": got, "metrics": {k: ep[k] for k in ("i2t_correct", "t2i_correct")},
                          "diff": {k: got[k] - ep[k] for k in ("i2t_correct", "t2i_correct")},
                          "finite": bool(np.isfinite(z["img"]).all() and np.isfinite(z["txt"]).all())}
        out["runs"].append(rec)
        print(r["variant"], m["lr"], "best", m["best_epoch"], best["epoch"], "rr", rr["best_epoch"],
              "sel %.4f" % (100 * best["selection"]),
              "feat diff", rec["features.npz"]["diff"], "ep0 diff", rec["features_epoch0.npz"]["diff"], flush=True)
    print("SELECTED", json.dumps(out["selected"]))
    (HERE / "fr_selection.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
