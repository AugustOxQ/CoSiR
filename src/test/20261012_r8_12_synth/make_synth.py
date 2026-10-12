"""Synthetic result tree for run_r8_descriptive.py (smoke root results/smoke/gate under R8_RESULTS). Not real numbers.
Arrays are R7's seed-42 arrays renamed to r8 scorer names; labels are the content grouping with noise."""
import json, os, sys
from pathlib import Path
import numpy as np
RF = Path(__file__).resolve().parents[1] / "20261129_r8_grouping"
sys.path.insert(0, str(RF))
import r8_common as R
import r8_gate as G

root = R.SMOKE / "gate"
c = R.ctx()
rng = np.random.default_rng(1)
ids = list(G.MAIN[:-1]) + list(G.SCREEN) + ["R0own", "NRK", "TP"]
DROP = {"DI": "ESC1 failed (synthetic)"}
Ks = {x: 17 for x in ids}; Ks["TP"] = 27; Ks["FC4"] = 12
labels, recs = {}, {}
for j, x in enumerate(ids):
    if x in DROP: continue
    K = Ks[x]
    p = (c.content + j) % K
    flip = rng.random(len(p)) < 0.1
    p = np.where(flip, rng.integers(0, K, len(p)), p)
    p = np.unique(p, return_inverse=True)[1].astype(np.int64)
    labels[x] = p
    recs[x] = {"n_groups": int(p.max()) + 1, "evaluated": [{"r": 1.0, "K": int(p.max()) + 1}], "chosen_r": 1.0,
               "K": int(p.max()) + 1, "stage": "grid", "dropped": False, "reason": None}
    if x in G.MAIN or x in G.SCREEN:
        i = len(labels)
        z = rng.permutation(p); labels["Z" + x] = z
        recs["Z" + x] = {"n_groups": int(z.max()) + 1, "dropped": False, "reason": None}
arr_rows = {x: v[c.g_st] for x, v in labels.items()}
(root / "groupings").mkdir(parents=True, exist_ok=True)
np.savez(root / "groupings/r8_groupings.npz", scorer_train=c.scorer_train, g_st=c.g_st, **arr_rows)
dropped = {x: r for x, r in DROP.items()}
(root / "groupings/r8_groupings.json").write_text(json.dumps({"groupings": recs, "dropped": dropped, "smoke": True}))
(root / "spaces").mkdir(exist_ok=True); (root / "heads").mkdir(exist_ok=True)
halves_row = c.halves[c.g_st]
for x in labels:
    if x.startswith("V") or x.startswith("NB"):
        (root / f"spaces/{x}.json").write_text(json.dumps({"training": {"last_epoch_mean_loss": 3.0 + rng.random()}}))
    if x.startswith("Z"): 
        img = np.eye(int(labels[x].max()) + 1, dtype=np.float32)[rng.integers(0, int(labels[x].max()) + 1, len(labels[x]))]
    else:
        v = arr_rows[x]; k = int(v.max()) + 1
        g = np.where(rng.random(len(v)) < 0.2, rng.integers(0, k, len(v)), v)
        img = np.eye(k, dtype=np.float32)[g]
    np.savez(root / f"heads/xheads_{x}.npz", img=img)
    (root / f"heads/head_{x}.json").write_text(json.dumps({"grouping": x, "dropped": False, "text": {"source": "s", "standard": {
        "heldout_accuracy": {"img": 40.0, "txt": 25.0}, "check_majority_share": 10.0, "uniform": 5.0}}}))
# arrays
z7 = np.load(R.MAIN / "src/test/20261126_design_l_style_source/results/seed42/arrays.npz")
srcs = ["V1", "V2", "V3", "V4", "V5", "V6", "R0", "N"]
arr = {k: z7[k] for k in ("cl", "pair_index", "parity") + tuple(k for k in z7.files if k.startswith(("AFF", "B__", "B0__", "B1__")))}
names = [x for x in labels]
for j, x in enumerate(names):
    s = srcs[j % len(srcs)]
    for kind in ("R1", "XAFF"):
        for k in z7.files:
            if k.startswith((f"{kind}_{s}__", f"{kind}_{s}_cf__")):
                arr[k.replace(f"{kind}_{s}", f"{kind}_{x}", 1)] = z7[k]
    for k in z7.files:
        if k.startswith(f"BX_{s}__"):
            arr[k.replace(f"BX_{s}", f"BX_{x}", 1)] = z7[k]
    rd = np.load(R.MAIN / f"src/test/20261126_design_l_style_source/results/seed42/reader_{s}.npz")
    (root / "seed42").mkdir(exist_ok=True)
    np.savez(root / f"seed42/reader_{x}.npz", **{k: rd[k] for k in rd.files})
# stub the dropped ids' absence; gate over all
stats = G.gate_stats(arr, dropped={**dropped, "ZDI": "no control: DI is dropped"} if False else dropped)
def save(a, name):
    np.savez(root / f"seed42/arrays_{name}.npz", **a)
    return R.sha_file(root / f"seed42/arrays_{name}.npz")
sha = save(arr, "main")
(root / "seed42/index.json").write_text(json.dumps({"batches": [{"file": "arrays_main.npz", "sha256": sha}]}))
verdict = {**stats, "smoke": True, "step": "gate", "rule_sha256": R.sha_file(R.RULE)}
(root / "gate_verdict.json").write_text(json.dumps(R.jsonable(verdict)))
t = verdict["main"]["head_target"]; print("pick", verdict["main"]["pick"], "target", t)
# H arm files
x = t["grouping"]; hx = "H" + x
harr = {k: z7[k] for k in ("cl", "pair_index", "parity") + tuple(k for k in z7.files if k.startswith(("AFF", "B__", "B0__", "B1__")))}
for kind in ("R1", "XAFF"):
    for k in z7.files:
        if k.startswith((f"{kind}_V6__", f"{kind}_V6_cf__")):
            harr[k.replace(f"{kind}_V6", f"{kind}_{hx}", 1)] = z7[k]
for k in z7.files:
    if k.startswith("BX_V6__"): harr[k.replace("BX_V6", f"BX_{hx}", 1)] = z7[k]
sha = save(harr, "hx")
(root / "seed42/index_hx.json").write_text(json.dumps({"batches": [{"file": "arrays_hx.npz", "sha256": sha}]}))
rd = np.load(R.MAIN / "src/test/20261126_design_l_style_source/results/seed42/reader_V6.npz")
np.savez(root / f"seed42/reader_{hx}.npz", **{k: rd[k] for k in rd.files})
np.savez(root / f"heads/post_{hx}.npz", img=np.zeros(1))
print("done", root)
