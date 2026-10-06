"""R-a (DECISION_RULE.md section 4.1) and the AR check of section 5, item 2.

  1. regression check (inside load_bundle), 2. the five sigma_h from the seed-42 support/contrast pair agreements,
  written to results/ra_sigma.json BEFORE any R-a score, 3. R-a on A1, A0 and AR (saved as Ra_A1, Ra_A0, Ra_AR),
  4. results/ra_summary.json/.txt: A1 - A0 under R-a and the AR check (share of picks that go to rand under R-a and
  under the step-1 arg-max reader).

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261117_reader_fix_csd/run_ra.py [--smoke | --check-only]
Non-smoke outputs are never overwritten. --smoke: 200 episodes per aspect pair, results/smoke/, numbers are not results.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CANDS = (("A1", "Ra_A1"), ("A0", "Ra_A0"), ("AR", "Ra_AR"))


def compute_sigma(bundle):
    ep = bundle.ctx.pooled
    si, st, ci, ct = C.condition_sets(ep, "a")
    sup, con = C.pair_agreements(bundle.post, si, st, ci, ct, C.GROUPINGS)
    sig = C.sigma_from_agreements(sup, con)
    return {h: float(s) for h, s in zip(C.GROUPINGS, sig)}, int(len(si))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="build the bundle (regression check) and exit")
    args = ap.parse_args()
    C.assert_rule()
    out = C.res_dir(args.smoke)
    if not args.smoke and not args.check_only:
        for n in ("ra_sigma.json", "ra_summary.json", "ra_summary.txt"):
            if (out / n).exists():
                raise SystemExit(f"{out / n} exists; refusing to overwrite")
        for _, nm in CANDS:
            if any(p.exists() for p in C._paths(nm, False)[1].values()):
                raise SystemExit(f"candidate {nm} exists; refusing to overwrite")
    bundle = C.load_bundle(smoke=args.smoke)
    print(json.dumps(bundle.checks, indent=1, default=str))
    if args.check_only:
        print("check-only: bundle built, nothing written")
        return
    ctx, cl = bundle.ctx, bundle.cl

    # ---- sigma_h first, written before any R-a score
    sigma, n_ep = compute_sigma(bundle)
    C.write_json_once(out / "ra_sigma.json", {
        "sigma": sigma, "formula": "sqrt(mean over episodes of (s2_S,h + s2_C,h)/4), ddof 1, condition a's sets, standard heads",
        "n_episodes": n_ep, "groupings": list(C.GROUPINGS)}, args.smoke)
    C.log(f"sigma written: {sigma}" if not args.smoke else "sigma written (smoke)")
    sig_file = json.loads((out / "ra_sigma.json").read_text())["sigma"]
    if sig_file != sigma:
        raise AssertionError("sigma_h did not round-trip through ra_sigma.json at full precision")

    # ---- R-a
    results, arrays, picks_of = {}, {}, {}
    for cfg, name in CANDS:
        parts = C.CONFIGS[cfg]
        si, st, ci, ct = C.condition_sets(ctx.pooled, "a")
        sup, con = C.pair_agreements(bundle.post, si, st, ci, ct, parts)
        delta_a = sup.mean(axis=1) - con.mean(axis=1)
        # the same Delta as aspect_deltas (the step-1 reader's), bit for bit
        from src.eval.aspect_quick_checks import aspect_deltas
        if not np.array_equal(delta_a, aspect_deltas({h: bundle.post[h] for h in parts}, ctx.pooled, "a", parts)):
            raise AssertionError("Delta differs from aspect_quick_checks.aspect_deltas")
        sg = np.array([sigma[h] for h in parts])
        picks, margins, _ = C.scaled_delta_picks(delta_a, sg)
        stack = C.grouping_stack(bundle.post, ctx.pooled, parts)
        T = C.hard_term(stack, picks)
        # consistency: with sigma = 1 the pick would be the step-1 arg max, and T would be its term
        p1, _, _ = C.scaled_delta_picks(delta_a, np.ones(len(parts)))
        if not np.array_equal(p1["a"], bundle.argmax[cfg]["picks"]["a"]) or not np.array_equal(p1["b"], bundle.argmax[cfg]["picks"]["b"]):
            raise AssertionError(f"{cfg}: unscaled pick differs from the step-1 arg-max reader's picks")
        summary, arr = C.evaluate(bundle, cfg, T, picks, name)
        summary["smoke"] = bool(args.smoke)
        summary["sigma"] = {h: sigma[h] for h in parts}
        C.save_candidate(name, summary, arr, T, picks, margins, smoke=args.smoke)
        results[cfg], arrays[cfg], picks_of[cfg] = summary, arr, picks

    # ---- summary: A1 - A0 under R-a, the AR check
    a1, a0 = arrays["A1"], arrays["A0"]
    diff = {m: C.paired_diff(a1, a0, m, cl) for m in ("fused_r1", "bar_r1", "margin_r1")}
    ar_idx = C.CONFIGS["AR"].index("rand")
    ar_check = {"reader_Ra": results["AR"]["pick_share"],
                "reader_step1_argmax": C.pick_shares(bundle.argmax["AR"]["picks"], C.CONFIGS["AR"], ctx.pair_index),
                "rand_share_chance": 100.0 / len(C.CONFIGS["AR"]),
                "Ra_AR": {"margin_r1": results["AR"]["margin"]["r1"], "bar_margin_r1": results["AR"]["bar"]["r1"],
                          "bar_comparator": results["AR"]["bar"]["comparator"]},
                "step1_argmax_AR": {"margin_r1": bundle.argmax["AR"]["margin_r1"],
                                    "bar_margin_r1": bundle.argmax["AR"]["bar"]["r1"],
                                    "bar_comparator": bundle.argmax["AR"]["bar"]["comparator"]}}
    rec = {"smoke": bool(args.smoke), "A1_minus_A0_under_Ra": diff, "ar_check": ar_check,
           "candidates": {nm: {"bar_margin": results[c]["bar"]["r1"], "bar_comparator": results[c]["bar"]["comparator"],
                               "gain_statistic": results[c]["gain_statistic"], "clears_bar": results[c]["clears_bar"]}
                          for c, nm in CANDS},
           "regression_checks": bundle.checks, "sigma": sigma}
    C.write_json_once(out / "ra_summary.json", rec, args.smoke)
    f = C._fmt
    L = [f"R-a summary{' [SMOKE: not a result]' if args.smoke else ''} (rule {C.RULE_SHA[:12]})",
         "A1 - A0 under R-a (paired per anchor): fused R@1 " + f(diff["fused_r1"]) + "; bar margin " + f(diff["bar_r1"])
         + "; margin " + f(diff["margin_r1"]), "AR check: share of (episode, condition) picks going to rand (chance 25%)"]
    for lab, sh in (("R-a", ar_check["reader_Ra"]), ("step-1 arg-max", ar_check["reader_step1_argmax"])):
        L.append(f"  {lab}: overall {sh['overall']['rand']:.2f}%; per condition a/b "
                 f"{sh['per_condition']['a']['rand']:.2f}/{sh['per_condition']['b']['rand']:.2f}; per pair a/b "
                 + "; ".join(f"{p} {v['a']['rand']:.1f}/{v['b']['rand']:.1f}" for p, v in sh["per_pair_condition"].items()))
    L.append(f"  R-a on AR: margin {f(ar_check['Ra_AR']['margin_r1'])}, bar margin {f(ar_check['Ra_AR']['bar_margin_r1'])} "
             f"({ar_check['Ra_AR']['bar_comparator']}); step-1 arg-max on AR: margin {f(ar_check['step1_argmax_AR']['margin_r1'])}, "
             f"bar margin {f(ar_check['step1_argmax_AR']['bar_margin_r1'])} ({ar_check['step1_argmax_AR']['bar_comparator']})")
    (out / "ra_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
