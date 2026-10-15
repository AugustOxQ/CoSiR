"""Phase 2, step 4: compare phase2_results.json and out/rd_held_arrays.npz (my numbers, written and hashed first) with
the runner's held outputs: held_started.json (episode and head-coefficient hashes), held_pass.json, held_arrays.npz,
sensitivity_held.json; plus the seed-42 pick records the held run pinned. Agreement as rule §8.4: discrete
quantities identical; points and bounds within 1e-9 R@1 points; per-anchor arrays exact (shape, dtype, values).
Writes phase2_compare.json.

  cd /project/CoSiR && PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python \
  src/test/20261125_artelingo_held_test/rederive/rd_compare_held.py \
  > src/test/20261125_artelingo_held_test/rederive/out/rd_compare_held.log 2>&1
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_common as R  # noqa: E402
import numpy as np  # noqa: E402

RES = R.F / "results"
TOL = 1e-9
ITEMS, PENDING = [], []
SEEDS = ("52", "53", "54")
P_IDS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
S_IDS = ("S1", "S2")


def add(name, mine, theirs, kind="exact"):
    if kind == "exact":
        ok = mine == theirs
    elif kind == "tol":
        ok = (isinstance(mine, (int, float)) and isinstance(theirs, (int, float))
              and math.isfinite(mine) and math.isfinite(theirs) and abs(mine - theirs) <= TOL)
    elif kind == "rel":
        ok = abs(mine - theirs) <= TOL * max(1.0, abs(theirs))
    elif kind == "array":
        a, b = np.asarray(mine), np.asarray(theirs)
        ok = bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b))
        mine, theirs = f"array{list(a.shape)} {a.dtype}", f"array{list(b.shape)} {b.dtype}"
    else:
        raise ValueError(kind)
    item = {"name": name, "result": "equal" if ok else "diff", "mine": mine, "theirs": theirs, "kind": kind}
    if kind in ("tol", "rel") and isinstance(mine, (int, float)) and isinstance(theirs, (int, float)):
        item["abs_diff"] = abs(mine - theirs)
    ITEMS.append(item)


def load(p):
    if not p.exists():
        PENDING.append(p.name)
        return None
    return json.loads(p.read_text())


def main():
    sha_res = R.sha_file(R.RD / "phase2_results.json")
    res = json.loads((R.RD / "phase2_results.json").read_text())
    eps, sc, st = res["episodes"], res["scores"], res["stats"]
    mine_eps = {s: eps["seeds"][s]["episodes_sha256"] for s in SEEDS}

    # held_started.json: the episode hashes and the head-coefficient hashes
    hs = load(RES / "held_started.json")
    if hs is not None:
        att = hs["attempts"]
        add("held_started/n_attempts", 1, len(att))
        a = att[-1]
        for s in SEEDS:
            for pair, h in mine_eps[s].items():
                add(f"held_started/episodes_sha256/{s}/{pair}", h, a["episodes_sha256"][s][pair])
        for g, mm in sc["heads"]["coef_sha256"].items():
            for m, h in mm.items():
                add(f"held_started/coef_sha256/{g}/{m}", h, a["coef_sha256"][g][m])
        add("held_started/rule_sha256", R.sha_file(R.F / "DECISION_RULE.md"), hs["rule_sha256"])
        # the seed-42 pick records the held run pinned: same files, same picks as my frozen ones
        rec42 = a["seed42_records_sha256"]
        for name in ("picks_seed42.json", "regression_seed42.json", "sensitivity_seed42.json"):
            add(f"held_started/seed42_records_sha256/{name} (file on disk)", R.sha_file(RES / name), rec42[name])
        pk = json.loads((RES / "picks_seed42.json").read_text())
        for b, v in sc["frozen"]["B_picks"].items():
            for h in (0, 1):
                add(f"frozen/B_picks/{b}/tune_half_{h}", v[f"tune_half_{h}_scores_parity_{1 - h}"],
                    [float(x) for x in pk[b][str(h)]])
        lam = pk["lambda_picks"]["rca"]
        for h in (0, 1):
            add(f"frozen/rca_lambda/tune_half_{h}", sc["frozen"]["rca_lambda"][f"tune_half_{h}_scores_parity_{1 - h}"],
                float("inf") if lam[str(h)] == "inf" else float(lam[str(h)]))
        rg = json.loads((RES / "regression_seed42.json").read_text())
        for key, v in sc["frozen"]["cells"].items():
            reader, kind = key.split("_")
            add(f"frozen/cells/{key}", [v["tune_half_0_scores_parity_1"]["cell"], v["tune_half_1_scores_parity_0"]["cell"]],
                rg["cells"][reader][kind])

    # held_pass.json
    hp = load(RES / "held_pass.json")
    if hp is not None:
        add("held_pass/rule_sha256", R.sha_file(R.F / "DECISION_RULE.md"), hp["rule_sha256"])
        add("held_pass/seeds", [52, 53, 54], hp["seeds"])
        add("held_pass/n_episodes", st["n_episodes"], hp["n_episodes"])
        add("held_pass/n_clusters", st["n_clusters"], hp["n_clusters"])
        add("held_pass/holm_order", st["holm_order_P"], hp["holm_order"])
        for s in SEEDS:
            for pair, h in mine_eps[s].items():
                add(f"held_pass/episodes_sha256/{s}/{pair}", h, hp["episodes_sha256"][s][pair])
        for j in P_IDS + S_IDS:
            mine = st["checks"][j]
            theirs = hp["checks"][j] if j in P_IDS else hp["secondary"][j]
            add(f"held_pass/{j}/n_j", mine["n_j"], theirs["n"])
            add(f"held_pass/{j}/point", mine["point"], theirs["point"], "tol")
            for lab, x, y in zip(("lo95", "hi95"), mine["ci95"], theirs["ci95"]):
                add(f"held_pass/{j}/{lab}", x, y, "tol")
            add(f"held_pass/{j}/holm_k", mine["holm_k"], theirs["holm_k"])
            add(f"held_pass/{j}/level_two_sided", mine["holm_level_two_sided"], theirs["level_two_sided"], "tol")
            for lab, x, y in zip(("lo_holm", "hi_holm"), mine["ci_holm"], theirs["ci_holm"]):
                add(f"held_pass/{j}/{lab}", x, y, "tol")
            add(f"held_pass/{j}/near_boundary", mine["within_one_of_boundary"], theirs["near_boundary"])
            for mk, tk in (("pass", "passes"), ("own_count_passes", "own_count_passes")):
                if tk in theirs:
                    add(f"held_pass/{j}/{tk}", mine[mk], theirs[tk])
        for name in ("held_arrays.npz", "sensitivity_held.json"):
            add(f"held_pass/outputs/{name} (file on disk)", R.sha_file(RES / name), hp["outputs"][name])

    # held_arrays.npz
    p = RES / "held_arrays.npz"
    if p.exists():
        mine_z, theirs_z = np.load(R.OUT / "rd_held_arrays.npz"), np.load(p)
        add("held_arrays/keys present in mine", sorted(set(theirs_z.files) - set(mine_z.files)), [])
        for k in theirs_z.files:
            add(f"held_arrays/{k}", mine_z[k], theirs_z[k], "array")
    else:
        PENDING.append(p.name)

    # sensitivity_held.json
    sh = load(RES / "sensitivity_held.json")
    if sh is not None:
        ms = st["sensitivity"]
        add("sensitivity_held/N", ms["N"], sh["N"])
        add("sensitivity_held/n_paintings", ms["n_paintings"], sh["n_paintings"])
        add("sensitivity_held/seeds", [52, 53, 54], sh["seeds"])
        for s in SEEDS:
            for pair, h in mine_eps[s].items():
                add(f"sensitivity_held/episodes_sha256/{s}/{pair}", h, sh["episodes_sha256"][s][pair])
        for j in P_IDS + S_IDS:
            mine, theirs = ms["checks"][j], sh[j]
            add(f"sensitivity_held/{j}/sigma_a2", mine["sigma_a2"], theirs["sigma_a2"], "rel")
            add(f"sensitivity_held/{j}/sigma_eps2", mine["sigma_eps2"], theirs["sigma_eps2"], "rel")
            for q in ("SE", "x95", "x" if j in P_IDS else "x2"):
                add(f"sensitivity_held/{j}/{q}", mine[q], theirs[q], "tol")
        add("sensitivity_held/sigma_source sha (file on disk)", R.sha_file(RES / sh["sigma_source"]["file"]),
            sh["sigma_source"]["sha256"])

    out = {"n_items": len(ITEMS), "n_equal": sum(i["result"] == "equal" for i in ITEMS),
           "n_diff": sum(i["result"] != "equal" for i in ITEMS),
           "diffs": [i["name"] for i in ITEMS if i["result"] != "equal"], "pending": PENDING,
           "phase2_results_sha256": sha_res,
           "tolerance": "discrete exact; points and bounds within 1e-9 R@1 points; arrays exact (shape, dtype, values); "
                        "sigma^2 within 1e-9 relative",
           "items": ITEMS, "time": R.now_ams()}
    R.write_json(R.RD / "phase2_compare.json", out)
    R.log(f"items {out['n_items']}, equal {out['n_equal']}, diff {out['n_diff']} {out['diffs'][:10]}, "
          f"pending {PENDING}")


if __name__ == "__main__":
    main()
