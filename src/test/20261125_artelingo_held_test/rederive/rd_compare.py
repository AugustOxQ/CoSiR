"""Phase 1, step 5: assemble phase1_results.json from rd_heads / rd_episodes / rd_seed42 / rd_stats, and compare it
with the runner's result files in ../results/ (values only). Writes phase1_results.json and phase1_compare.json.
A runner file that does not exist yet is listed as pending.

  PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python \
  src/test/20261125_artelingo_held_test/rederive/rd_compare.py
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


def load(name):
    return json.loads((R.RD / name).read_text())


def add(name, mine, runner, kind="exact"):
    if kind == "exact":
        ok = mine == runner
    elif kind == "tol":
        ok = (isinstance(mine, (int, float)) and isinstance(runner, (int, float))
              and math.isfinite(mine) and math.isfinite(runner) and abs(mine - runner) <= TOL)
    elif kind == "rel":
        ok = abs(mine - runner) <= TOL * max(1.0, abs(runner))
    elif kind == "array":
        a, b = np.asarray(mine), np.asarray(runner)
        ok = bool(a.shape == b.shape and np.array_equal(a, b))
        mine, runner = f"array{list(a.shape)}", f"array{list(b.shape)}"
    else:
        raise ValueError(kind)
    ITEMS.append({"quantity": name, "mine": mine, "runner": runner, "kind": kind,
                  "result": "equal" if ok else "DIFF"})


def fnum(x):
    return float("inf") if x == "inf" else float(x)


def cmp_refit(heads):
    p = RES / "refit_check.json"
    if not p.exists():
        PENDING.append(p.name)
        return
    r = json.loads(p.read_text())
    add("refit/passed (all bitwise equal)", heads["all_bitwise_equal"], r["passed"])
    for k, v in heads["bitwise"].items():
        add(f"refit/{k}/equal", v["equal"], r["items"][k]["equal"])
        add(f"refit/{k}/n_diff", v["n_diff"], r["items"][k]["n_diff"])
    add("refit/affect_identity", heads["affect_identity"], r["affect_identity"])
    for g in ("affect", "affect_km", "image", "caption", "csd"):
        for m in ("img", "txt"):
            theirs = r["coef_sha256"][g][m]
            variants = {k: v for k, v in heads["fits"][g][m].items() if isinstance(v, str)}
            match = [k for k, v in variants.items() if v == theirs]
            ITEMS.append({"quantity": f"refit/coef_sha256/{g}/{m}", "mine": variants, "runner": theirs,
                          "kind": "any-variant", "matched_variant": match,
                          "result": "equal" if match else "DIFF (hash format unknown)"})
    prov = {"affect": heads["prov"]["affect"], "csd": heads["prov"]["csd"],
            "affect_km": heads["prov"]["e2"]["affect"], "image": heads["prov"]["e2"]["image"],
            "caption": heads["prov"]["e2"]["caption"]}
    for g, pv in prov.items():
        for k, v in pv.items():
            if k in r["prov"][g]:
                add(f"refit/prov/{g}/{k}", v, r["prov"][g][k])
        if "draw_rows_sha256" not in pv and "draw_rows_sha256" in r["prov"][g]:
            add(f"refit/prov/{g}/draw_rows_sha256", heads["prov"]["e2"]["draw_rows_sha256"],
                r["prov"][g]["draw_rows_sha256"])
        if "n_draw" not in pv and "n_draw" in r["prov"][g]:
            add(f"refit/prov/{g}/n_draw", heads["prov"]["e2"]["n_draw"], r["prov"][g]["n_draw"])


def cmp_picks(s42, eps, heads):
    p = RES / "picks_seed42.json"
    if not p.exists():
        PENDING.append(p.name)
        return
    r = json.loads(p.read_text())
    for b in ("B", "B0", "B1"):
        for h in ("0", "1"):
            add(f"picks/{b}/tune_half_{h}", s42["B_picks"][b]["picks"][h], [float(x) for x in r[b][h]])
        add(f"picks/{b}/mean_r1", s42["B_picks"][b]["mean_r1"], r["mean_r1"][b])
    for name, v in s42["lambda"].items():
        for h in ("0", "1"):
            add(f"picks/lambda/{name}/{h}", v["recorded"][h], fnum(r["lambda_picks"][name][h]))
    for k, v in eps["seeds"]["42"]["sha256"].items():
        add(f"picks/episodes_sha256/{k}", v, r["episodes_sha256"][k])
    add("picks/fit_rows_sha256", s42["fit_rows_sha256"], r["fit_rows_sha256"])


def cmp_values(eps):
    p = RES / "value_sets.json"
    if not p.exists():
        PENDING.append(p.name)
        return
    r = json.loads(p.read_text())
    for x in ("emotion", "style", "genre"):
        theirs = r[x]
        add(f"value_sets/{x}/codes", eps["value_sets"][x]["codes"],
            [e["code"] for e in theirs] if isinstance(theirs, list) else theirs["codes"])
        add(f"value_sets/{x}/names", eps["value_sets"][x]["names"],
            [e["name"] for e in theirs] if isinstance(theirs, list) else theirs["names"])


def main():
    heads, eps, s42, st = load("rd_heads.json"), load("rd_episodes.json"), load("rd_seed42.json"), load("rd_stats.json")
    results = {"what": "C13 phase-1 re-derivation of round 6 (rule §8.4), own code; seed 42 on selection rows",
               "heads": heads, "episodes": eps, "seed42": s42, "stats": st, "time": R.now_ams()}
    R.write_json(R.RD / "phase1_results.json", results)
    cmp_refit(heads)
    cmp_picks(s42, eps, heads)
    cmp_values(eps)
    import rd_compare_late as late  # mappings for the files written after the picks
    late.compare(add, PENDING, s42, st)
    out = {"n_compared": len(ITEMS), "n_equal": sum(i["result"] == "equal" for i in ITEMS),
           "diffs": [i["quantity"] for i in ITEMS if i["result"] != "equal"], "pending_files": PENDING,
           "tolerance": "discrete exact; points and bounds within 1e-9 R@1 points; arrays exact",
           "items": ITEMS, "time": R.now_ams()}
    R.write_json(R.RD / "phase1_compare.json", out)
    R.log(f"compared {out['n_compared']}, equal {out['n_equal']}, diffs {out['diffs'][:10]}, pending {PENDING}")


if __name__ == "__main__":
    main()
