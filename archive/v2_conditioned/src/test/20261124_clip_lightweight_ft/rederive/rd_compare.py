"""Compare the hashed independent recompute (rd_ft.json) with the implementation's results/eval.json.

Agreement = every selection identical and every shared point and 95% bound within 1e-9 percentage points. Also
identifies which run's features.npz reproduces each ft:* scorer in eval.json (eval.json does not name its inputs).

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261124_clip_lightweight_ft/rederive/rd_compare.py \
    [--prep-cache <scratch>/rd_prep.npz]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rd_ft  # noqa: E402

TOL = 1e-9
EVAL = HERE.parent / "results/eval.json"
NAME = {"plain": "plain", "LP": "ft:LP", "LB": "ft:LB", "LoRA": "ft:LoRA", "LB_epoch0": "ft:CLIPcache",
        "AFF": "AFF", "B": "B", "Bp_A0": "Bp0", "Bp_A1": "Bp1"}
VNAME = {"LP": "LP", "LB": "LB", "LoRA": "LoRA", "LB_epoch0": "CLIPcache"}
LABEL = {"AFF_minus_ft": "AFF_minus_ft", "ft_minus_plain": "ft_minus_plain", "ft_minus_BpA0": "ft_minus_Bp0"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-cache", type=Path, default=None)
    args = ap.parse_args()
    recorded = (HERE / "rd_ft.json.sha256").read_text().split()[0]
    assert rd_ft.sha256(HERE / "rd_ft.json") == recorded, "rd_ft.json changed after its hash was recorded"
    mine, theirs = json.loads((HERE / "rd_ft.json").read_text()), json.loads(EVAL.read_text())
    rows, worst = [], 0.0

    def cmp(where: str, a: float, b: float) -> None:
        nonlocal worst
        d = abs(a - b)
        worst = max(worst, d)
        rows.append((where, a, b, d))

    # means per seed and pooled (r1, either, other)
    for blk, tb in [(str(s), theirs["seeds"][str(s)]) for s in rd_ft.SEEDS] + [("pooled_49_51", theirs["pooled"])]:
        for k, v in mine["means"][blk].items():
            for m in ("r1", "either", "other"):
                if m in tb["scorers"][NAME[k]]:
                    cmp(f"mean {blk} {k} {m}", v[m], tb["scorers"][NAME[k]][m]["point"])
        assert set(tb["scorers"]) == {NAME[k] for k in mine["means"][blk]}, (blk, set(tb["scorers"]))
    # differences: point, both bounds, cluster count
    for v, d in mine["differences"].items():
        for lab, blocks in d.items():
            for blk, x in blocks.items():
                tb = theirs["pooled"] if blk == "pooled_49_51" else theirs["seeds"][blk]
                y = tb["comparisons"][VNAME[v]][LABEL[lab]]
                for m in ("r1", "either"):
                    cmp(f"diff {v} {lab} {blk} {m} point", x[m]["point"], y[m]["point"])
                    cmp(f"diff {v} {lab} {blk} {m} lo", x[m]["ci95"][0], y[m]["ci95"][0])
                    cmp(f"diff {v} {lab} {blk} {m} hi", x[m]["ci95"][1], y[m]["ci95"][1])
                    assert x[m]["n_clusters"] == y[m]["n_clusters"], (v, lab, blk, m)
    # per aspect pair pooled over 49-51 (points)
    for i, name in enumerate(rd_ft.PAIR_NAMES):
        tp = theirs["pairs"][str(i)]
        assert tp["seeds"] == list(rd_ft.POOL)
        for k, v in mine["per_pair_pooled_49_51"][name].items():
            for m in ("r1", "either"):
                cmp(f"pair {name} {k} {m}", v[m], tp["scorers"][NAME[k]][m]["point"])

    # which run's features.npz does each ft:* in eval.json come from? (R@1 mean on all four seeds)
    prep = rd_ft.prepare(args.prep_cache)
    n, selection = int(prep["n_rows"]), prep["selection"]
    in_sel = np.zeros(n, bool)
    in_sel[selection] = True
    eps = {s: rd_ft.load_episodes(s, in_sel) for s in rd_ft.SEEDS}
    ident = {}
    for run in rd_ft.read_runs():
        d = rd_ft.run_dir(run)
        for fname in ("features.npz", "features_epoch0.npz"):
            if fname == "features_epoch0.npz" and run["variant"] == "LP":
                continue
            z = np.load(d / fname, allow_pickle=False)
            img = rd_ft.full_array(n, z["rows"], z["img"], selection)
            txt = rd_ft.full_array(n, z["rows"], z["txt"], selection)
            r1 = {s: 100 * float(rd_ft.cosine_metrics(img, txt, eps[s])["r1"].mean()) for s in rd_ft.SEEDS}
            hits = [nm for nm in ("ft:LP", "ft:LB", "ft:LoRA", "ft:CLIPcache")
                    if all(abs(r1[s] - theirs["seeds"][str(s)]["scorers"][nm]["r1"]["point"]) <= TOL
                           for s in rd_ft.SEEDS)]
            ident[f"{run['variant']}_lr{run['lr_str']}/{fname}"] = hits
    used = {nm: [k for k, h in ident.items() if nm in h] for nm in ("ft:LP", "ft:LB", "ft:LoRA", "ft:CLIPcache")}
    want = {f"ft:{v}": [f"{v}_lr{s['dir'].split('_lr')[-1]}/features.npz"] for v, s in mine["selection"].items()}
    sel_ok = all(used[k] == want[k] for k in want)
    cache_ok = any(k.startswith(f"LB_lr{mine['selection']['LB']['dir'].split('_lr')[-1]}/features_epoch0")
                   for k in used["ft:CLIPcache"])

    bad = [r for r in rows if r[3] > TOL]
    out = {"rd_ft_sha256": recorded, "eval_json_sha256": rd_ft.sha256(EVAL), "n_compared": len(rows),
           "max_abs_diff_pp": worst, "n_over_tol": len(bad), "over_tol": [list(r) for r in bad[:50]],
           "features_identification": ident, "eval_ft_sources": used, "selected_features": want,
           "selection_identical": sel_ok, "cache_reference_is_selected_LB_epoch0": cache_ok,
           "agree": not bad and sel_ok and cache_ok}
    (HERE / "rd_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "features_identification"}, indent=1))


if __name__ == "__main__":
    main()
