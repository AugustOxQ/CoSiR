"""EXPLORATORY (decides nothing; seed 42 development episodes only). Describes the N6 result that missed its pass rule
(results/n6_seed42.*): E1 bootstrap-seed sensitivity, E2/E3 N6 on A3's centered condition-free base (T_N1u) with the hard
and soft readers, E4 per aspect pair. CPU only; read-only on every stored result.

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python src/test/20261108_new_method_quick_checks/explore_n6.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_n6 as rn  # noqa: E402
import run_checks as rc  # noqa: E402

from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import METRICS, cluster_bootstrap, per_anchor  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import centered_term  # noqa: E402

rg = rc.rg
RES = HERE / "results"


def c(x):
    return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f}, {x['ci95'][1]:6.2f}]"


def main():
    outs = [RES / "explore_n6.json", RES / "explore_n6.txt"]
    if any(p.exists() for p in outs):
        raise SystemExit("explore_n6 outputs exist; refusing to overwrite")
    z = np.load(RES / "per_anchor_n6_seed42.npz")
    cl, pidx = z["anchor_group"], z["pair_index"]
    out, lines = {}, ["EXPLORATORY: N6 on seed-42 development episodes (decides nothing)"]

    # E1
    d = {m: z[f"nested__{m}"].astype(np.float64) - z[f"control__{m}"].astype(np.float64) for m in ("r1", "gain")}
    lo = {m: np.array([cluster_bootstrap(d[m], cl, seed=s)["ci95"][0] for s in range(100)]) for m in d}
    both = (lo["r1"] > 0) & (lo["gain"] > 0)
    out["E1"] = {"share_lb_gt0": {"r1": float((lo["r1"] > 0).mean()), "gain": float((lo["gain"] > 0).mean()),
                                  "both": float(both.mean())},
                 "lower_bound_pp": {m: {"min": 100 * float(lo[m].min()), "median": 100 * float(np.median(lo[m])),
                                        "max": 100 * float(lo[m].max())} for m in lo}}
    e = out["E1"]
    lines.append("E1 bootstrap seeds 0..99, nested - control lower bound > 0: "
                 f"R@1 {100*e['share_lb_gt0']['r1']:.0f}%, gain {100*e['share_lb_gt0']['gain']:.0f}%, both {100*e['share_lb_gt0']['both']:.0f}%")
    for m in lo:
        v = e["lower_bound_pp"][m]
        lines.append(f"   {m} lower bound pp min {v['min']:.3f} median {v['median']:.3f} max {v['max']:.3f}")

    # E2/E3
    ctx = rg.EvalContext(42, False)
    ep = ctx.pooled
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    inp, _, _ = rc.model_inputs(ctx, "A3", scorer_train, False)[:3]
    tn1u = centered_term(inp, ep, uniform=True)
    post = rn.load_posteriors(RES / "n6_posteriors.npz", ctx)
    t6, t6s, t6u, _, _ = rn.n6_terms(post, ep)
    nested0, _, _ = crossfit_nested(ctx.cos, t6u, t6, ctx.parity)
    pn0 = per_anchor(nested0)
    for m in METRICS:
        if not np.array_equal(np.asarray(pn0[m]), z[f"nested__{m}"]):
            raise AssertionError(f"recomputed N6-nested differs from stored ({m})")
    lines.append("input check: recomputed N6 nested equals stored nested arrays (bit-equal)")
    cos_pa, rca_pa = rn.e1_arrays(ctx, False, 42)
    rep = rc.Report(ctx, cos_pa, rca_pa)
    a3 = np.load(RES / "per_anchor_addendum1.npz")
    pm = {m: a3[f"dev__A3__matched__{m}"].astype(np.float64) for m in METRICS}
    for tag, term in (("E2 hard T6", t6), ("E3 soft T6soft", t6s)):
        nested, control, picks = crossfit_nested(ctx.cos, tn1u, term, ctx.parity)
        pn, pc = per_anchor(nested), per_anchor(control)
        r = {"nested": rep.describe(pn), "control": rep.describe(pc), "picks": picks,
             "nested_vs_control": rep.paired(pn, pc), "nested_vs_A3_matched": rep.paired(pn, pm)}
        out[tag] = r
        lines.append(f"{tag} on T_N1u base: picks {picks}")
        for k in ("nested", "control"):
            s = r[k]["summary"]
            lines.append(f"   {k:8s} R@1 {c(s['r1'])} gain {c(s['gain'])} either {c(r[k]['either'])}")
        for k in ("nested_vs_control", "nested_vs_A3_matched"):
            lines.append(f"   {k:21s} R@1 {c(r[k]['r1'])} gain {c(r[k]['gain'])}")

    # E4
    out["E4"] = {}
    for i, name in enumerate(("emotion__style", "emotion__genre", "style__genre")):
        mk = pidx == i
        out["E4"][name] = {m: {"point": 100 * float(d[m][mk].mean()),
                               "ci95": [100 * v for v in cluster_bootstrap(d[m][mk], cl[mk])["ci95"]]} for m in d}
        lines.append(f"E4 {name:15s} nested - control R@1 {c(out['E4'][name]['r1'])} gain {c(out['E4'][name]['gain'])}")

    rg.assert_finite_tree(out)
    outs[0].write_text(json.dumps(out, indent=1))
    outs[1].write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
