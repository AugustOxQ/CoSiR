"""EXPLORATORY (decides nothing; seed 42 development episodes only). Condition-free counterparts for the fix
diagnostics (diagnose_fixes.py, same folder, imported, not modified): (1) D0's condition-free counterpart D0uni (mean of
the three label-probe dots) against D0told and D0 inferred-hard; (2) for T in (T6, T6oracle, D0told), T fused on B with
the nested min-margin cross-fit against T_cf = 0.5(T_a + T_b) fused on B with the condition-free max-R@1 cross-fit, so
fused-T minus fused-T_cf is the margin attributable to reading the condition; (3) the either change of the told and the
N6 fusions on the "both picks correct" subset and its complement; (4) the pick-accuracy ceiling of an argmax-Delta reader
under the told mapping, enumerated over all orderings of Delta given Delta_b = -Delta_a. 95% CIs resample paintings.
CPU only. No other episode seed is read.

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261109_fix_diagnostics/diagnose_counterparts.py
    -> results/diagnose_counterparts.json, results/diagnose_counterparts.txt (refuses to overwrite)
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnose_fixes as df  # noqa: E402  (also puts run_checks, run_n6 and the repo root on sys.path)

from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (ASPECTS, centered_term, crossfit_condition_free,  # noqa: E402
                                          inferred_scores, told_scores, uniform_probe_scores)

rc, rg, n6, n6c = df.rc, df.rg, df.n6, df.n6c
OUT_JSON, OUT_TXT = df.OUT / "diagnose_counterparts.json", df.OUT / "diagnose_counterparts.txt"
CF_TERMS = ("T6", "T6oracle", "D0told")


def cf_version(t):
    return {cnd: {d: (0.5 * (t["a"][d].astype(np.float64) + t["b"][d].astype(np.float64))).astype(np.float32)
                  for d in DIRECTIONS} for cnd in CONDITIONS}


def ceiling(pair_index, told_part):
    """Argmax-Delta reader (pick_a = argmax D, pick_b = argmax(-D) = argmin D) over the 6 strict orderings of D."""
    out, shares = {}, {}
    for i, name in enumerate(rg.POOLED_ORDER):
        a_asp, b_asp = rg.PAIRS[i]
        ta, tb = df.PARTS.index(told_part[a_asp]), df.PARTS.index(told_part[b_asp])
        best_both, best_mean = 0, 0.0
        for perm in itertools.permutations(range(3)):
            d = np.array(perm, float)                      # a strict ordering of the 3 partition deltas
            ok_a, ok_b = int(d.argmax() == ta), int((-d).argmax() == tb)
            best_both, best_mean = max(best_both, ok_a * ok_b), max(best_mean, 0.5 * (ok_a + ok_b))
        out[name] = {"both_correct_possible": bool(best_both), "max_accuracy": 100 * best_mean}
        shares[name] = float((pair_index == i).mean())
    out["pooled_max_accuracy"] = sum(shares[k] * out[k]["max_accuracy"] for k in shares)
    out["pooled_max_both_correct_share"] = 100 * sum(shares[k] for k in shares if out[k]["both_correct_possible"])
    out["pair_shares"] = shares
    return out


def text(r):
    c, c3 = df.c, df.c3
    L = [f"EXPLORATORY diagnose_counterparts: decides nothing; seed 42 dev episodes only (n={r['n_episodes']}, "
         f"{r['n_clusters']} paintings); 95% CIs resample paintings (5,000). Columns R@1 | gain | either (pp).",
         f"B (= C2) {c3(r['B'])}; checks: B equals stored C2 True, D0told equals stored {r['checks']['D0told_equals_stored']}, "
         f"max |Delta_b + Delta_a| = {r['checks']['max_abs_delta_sum']:.2e}",
         "1. D0 condition-free counterpart (D0uni = mean of the 3 probe dots, same under both conditions)"]
    for k in ("D0uni", "D0told", "D0hard"):
        L.append(f"  {k:7s} {c3(r['d0'][k])}")
    for k in ("D0told_minus_D0uni", "D0hard_minus_D0uni"):
        L.append(f"  {k:19s} {c3(r['d0'][k])}")
    L.append("2. Matched counterparts on B (Delta vs B unless stated; cf = 0.5(T_a+T_b), cross-fit max R@1)")
    for t in CF_TERMS:
        x = r["matched"][t]
        L.append(f"  {t}: fused T - B     {c3(x['fusedT_vs_B'])}")
        L.append(f"  {t}: fused T_cf - B  {c3(x['fusedTcf_vs_B'])}  picks {x['cf_picks']}")
        L.append(f"  {t}: fused T - fused T_cf (condition margin) {c3(x['fusedT_vs_fusedTcf'])}")
    L.append("3. Either change vs B, told fusion (T6oracle on B) and N6 reader fusion (T6 on B), by subset where N6's hard "
             "pick = told partition under both conditions")
    s = r["subset"]
    L.append("  composition by pair (both-correct / all): " + "; ".join(
        f"{df.PAIR_SHORT[p]} {v['both_correct']}/{v['all']}" for p, v in s["composition"].items())
        + f"; total {s['n_both']}/{s['n_all']}")
    for key in ("both correct", "not both"):
        for t in ("T6oracle", "T6"):
            x = s["split"][key][t]
            L.append(f"  {key:12s} (n={x['n']}) {t:9s} dR@1 {c(x['r1'])} dgain {c(x['gain'])} deither {c(x['either'])}")
    k = r["ceiling"]
    L.append("4. Pick-accuracy ceiling (argmax-Delta reader, told mapping; style x genre -> image under both conditions)")
    L.append("  " + "; ".join(f"{df.PAIR_SHORT[p]} {k[p]['max_accuracy']:.1f}% (both correct possible: "
                               f"{k[p]['both_correct_possible']})" for p in rg.POOLED_ORDER)
             + f"; pooled max accuracy {k['pooled_max_accuracy']:.1f}%; max both-correct share "
               f"{k['pooled_max_both_correct_share']:.1f}% (observed 52.4% and 28.1%)")
    return "\n".join(L)


def main():
    if OUT_JSON.exists() or OUT_TXT.exists():
        raise SystemExit(f"{OUT_JSON} or {OUT_TXT} exists; refusing to overwrite.")
    t0 = time.time()
    ctx = rg.EvalContext(df.SEED, False)
    ep, cl = ctx.pooled, ctx.anchor_group
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    inp, _, a3_prov = rc.model_inputs(ctx, "A3", scorer_train, False)
    t_n1u = centered_term(inp, ep, uniform=True)
    if rg.sha_file(df.POSTERIORS) != n6c.POSTERIORS_SHA:
        raise AssertionError("posteriors SHA differs")
    post = n6.load_posteriors(df.POSTERIORS, ctx)
    t6, _, t6u, info, _ = n6.n6_terms(post, ep)
    B, picks_B = crossfit_condition_free(ctx.cos, t_n1u, t6u, ctx.parity)
    pB = per_anchor(B)
    gate = np.load(df.GATE)
    if not all(np.array_equal(pB[m], gate[f"matched__{m}"]) for m in df.METRICS):
        raise SystemExit("per_anchor(B) differs from stored C2")

    aspect_of = {"a": np.array([rg.PAIRS[i][0] for i in ctx.pair_index]),
                 "b": np.array([rg.PAIRS[i][1] for i in ctx.pair_index])}
    part_idx = {c: np.array([df.PARTS.index(df.TOLD_PART[a]) for a in aspect_of[c]]) for c in CONDITIONS}
    asp_idx = {c: np.array([ASPECTS.index(a) for a in aspect_of[c]]) for c in CONDITIONS}
    st6 = df.stacked(post, ep, df.PARTS)
    t6oracle = {c: df.pick(st6, part_idx[c]) for c in CONDITIONS}
    post_d0, probe_prov = rc.fit_probes(ctx, artelingo_aspect_labels(ctx.data), scorer_train, rc.PROBE_ROWS)
    d0told = told_scores(post_d0, ep, asp_idx["a"], asp_idx["b"])
    d0uni = uniform_probe_scores(post_d0, ep, ASPECTS)
    d0hard, _ = inferred_scores(post_d0, ep, "hard", ASPECTS)
    stored = np.load(df.CHECKS)
    d0ok = df.same_metrics(per_anchor(d0told), {m: stored[f"d0_told__{m}"] for m in df.METRICS})
    terms = {"T6": df.as_f32(t6), "T6oracle": df.as_f32(t6oracle), "D0told": df.as_f32(d0told)}

    delta_sum = max(float(np.abs(df.aspect_deltas(post, ep, "a", df.PARTS)
                                 + df.aspect_deltas(post, ep, "b", df.PARTS)).max()), 0.0)
    res = {"exploratory": "decides nothing; seed 42 development episodes only", "n_episodes": int(ctx.n),
           "n_clusters": int(len(np.unique(cl))),
           "provenance": {"script_sha256": rg.sha_file(Path(__file__)), "diagnose_fixes_sha256": rg.sha_file(Path(df.__file__)),
                          "posteriors_sha256": rg.sha_file(df.POSTERIORS), "a3": a3_prov, "d0_probes": probe_prov},
           "checks": {"B_equals_stored_C2": True, "D0told_equals_stored": bool(d0ok), "max_abs_delta_sum": delta_sum},
           "B": df.describe(pB, cl), "B_picks": picks_B}

    # 1. D0 counterpart
    pu, pt, ph = per_anchor(d0uni), per_anchor(d0told), per_anchor(d0hard)
    res["d0"] = {"D0uni": df.describe(pu, cl), "D0told": df.describe(pt, cl), "D0hard": df.describe(ph, cl),
                 "D0told_minus_D0uni": df.diff(pt, pu, cl), "D0hard_minus_D0uni": df.diff(ph, pu, cl)}

    # 2. matched counterparts on B
    res["matched"], fused_pa = {}, {}
    for t, s in terms.items():
        nested, _, picks = crossfit_nested(B, B, s, ctx.parity)
        cf, cf_picks = crossfit_condition_free(B, B, cf_version(s), ctx.parity)
        pn, pc = per_anchor(nested), per_anchor(cf)
        fused_pa[t] = pn
        res["matched"][t] = {"fusedT_vs_B": df.diff(pn, pB, cl), "fusedTcf_vs_B": df.diff(pc, pB, cl),
                             "fusedT_vs_fusedTcf": df.diff(pn, pc, cl), "T_picks": picks, "cf_picks": cf_picks}
        rc.log(f"matched {t} done")

    # 3. both-correct subset
    correct = {c: info[c]["weights"].argmax(axis=1) == part_idx[c] for c in CONDITIONS}
    both = correct["a"] & correct["b"]
    comp = {name: {"both_correct": int((both & (ctx.pair_index == i)).sum()),
                   "all": int((ctx.pair_index == i).sum())} for i, name in enumerate(rg.POOLED_ORDER)}
    split = {}
    for key, mask in (("both correct", both), ("not both", ~both)):
        split[key] = {}
        for t in ("T6oracle", "T6"):
            pn = fused_pa[t]
            split[key][t] = {"n": int(mask.sum()),
                             "r1": rc.point_ci((pn["r1"] - pB["r1"])[mask], cl[mask]),
                             "gain": rc.point_ci((pn["gain"] - pB["gain"])[mask], cl[mask]),
                             "either": rc.point_ci((df.either(pn) - df.either(pB))[mask], cl[mask])}
    res["subset"] = {"composition": comp, "n_both": int(both.sum()), "n_all": int(ctx.n), "split": split}

    # 4. ceiling
    res["ceiling"] = ceiling(ctx.pair_index, df.TOLD_PART)
    res["runtime_s"] = round(time.time() - t0, 1)
    rg.assert_finite_tree(res)
    OUT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    txt = text(res)
    OUT_TXT.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
