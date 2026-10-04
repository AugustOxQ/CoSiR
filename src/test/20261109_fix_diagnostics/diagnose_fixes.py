"""EXPLORATORY (decides nothing; seed 42 development episodes only). Ranks candidate fixes for the either-rate cost
found by the new-method quick checks (docs/reports/auto/v2/2026-11-08_new_method_quick_checks.md, Sections 8 and 9)
for the next handoff: a contrastive partition reader (hard and soft), the told ("oracle") partition, a label-probe
contrastive upper bound, each scored alone, fused on top of the matched condition-free control C2 (called B here) with
A′'s min-margin cross-fit, and as a find-then-select cascade on B; plus the hard reader's correctness and per-direction
metrics. No other episode seed is read. CPU only. Reuses the reviewed quick-check code; modifies nothing.

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261109_fix_diagnostics/diagnose_fixes.py
    -> results/diagnose_fixes.json, results/diagnose_fixes.txt (refuses to overwrite)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QC = HERE.parent / "20261108_new_method_quick_checks"
sys.path.insert(0, str(QC))
import run_checks as rc  # noqa: E402  (puts run_gonogo and the repo root on sys.path)
import run_n6 as n6  # noqa: E402
import run_n6c as n6c  # noqa: E402

from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, compare, per_anchor, summarize  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (ASPECTS, aspect_deltas, cascade_scores, centered_term,  # noqa: E402
                                          crossfit_condition_free, probe_dots, told_scores)

rg = rc.rg
SEED = 42
PARTS = n6.PARTS                                              # ("affect", "image", "caption")
TOLD_PART = {"emotion": "affect", "style": "image", "genre": "image"}
RES_QC = QC / "results"
GATE = RES_QC / "per_anchor_n6c_gate.npz"
POSTERIORS = RES_QC / "n6_posteriors.npz"
CHECKS = RES_QC / "per_anchor_checks_seed42.npz"
OUT = HERE / "results"
OUT_JSON, OUT_TXT = OUT / "diagnose_fixes.json", OUT / "diagnose_fixes.txt"
TERMS = ("T6", "T6soft", "T6pm", "T6pm_soft", "T6oracle", "T6oracle_pm", "D0pm", "D0told")
CASCADE_TERMS, CASCADE_KS = ("T6", "T6pm", "T6pm_soft"), (2, 3)
PAIR_SHORT = {"emotion__style": "e×s", "emotion__genre": "e×g", "style__genre": "s×g"}


# ---------------------------------------------------------------- terms

def stacked(post, ep, parts):
    """{dir: (E, H, K)} probe dot products p_h(query) . p_h(candidate), h in ``parts``."""
    dots = probe_dots(post, ep, parts)
    return {d: np.stack([dots[h][d] for h in parts], axis=1).astype(np.float64) for d in DIRECTIONS}


def pick(stack, idx):
    """{dir: (E, K)}: the dot product of partition/aspect idx[e] in episode e."""
    rows = np.arange(len(idx))
    return {d: stack[d][rows, idx] for d in DIRECTIONS}


def minus(x, y):
    return {d: x[d] - y[d] for d in DIRECTIONS}


def contrastive_terms(post, ep, stack):
    """T6pm: s_{argmax Δ} − s_{argmin Δ}; T6pm_soft: Σ_h Δ_h s_h / Σ_h |Δ_h| (0 where Σ|Δ| = 0). Also returns Δ."""
    hard, soft, deltas = {}, {}, {}
    for c in CONDITIONS:
        delta = aspect_deltas(post, ep, c, PARTS).astype(np.float64)
        tot = np.abs(delta).sum(axis=1)
        w = np.where(tot[:, None] > 0, delta / np.where(tot > 0, tot, 1.0)[:, None], 0.0)
        hard[c] = minus(pick(stack, delta.argmax(axis=1)), pick(stack, delta.argmin(axis=1)))
        soft[c] = {d: np.einsum("nh,nhk->nk", w, stack[d]) for d in DIRECTIONS}
        deltas[c] = delta
    return hard, soft, deltas


def as_f32(scores):
    return {c: {d: np.asarray(scores[c][d], dtype=np.float32) for d in DIRECTIONS} for c in CONDITIONS}


def one_direction(scores, d):
    """Direction d duplicated into both direction slots: per_anchor then returns exactly direction d's metrics."""
    return {c: {dd: scores[c][d] for dd in DIRECTIONS} for c in CONDITIONS}


# ---------------------------------------------------------------- summaries

def either(pa):
    return np.asarray(pa["r1"]) + np.asarray(pa["other"])


def sub(pa, mask):
    return {m: np.asarray(pa[m])[mask] for m in METRICS}


def describe(pa, cl):
    return {**summarize(pa, cl), "either": rc.point_ci(either(pa), cl)}


def diff(pa, pb, cl):
    return {"r1": compare(pa, pb, cl, "r1"), "gain": compare(pa, pb, cl, "gain"),
            "either": rc.point_ci(either(pa) - either(pb), cl)}


def per_pair_diff(pa, pb, ctx):
    out = {}
    for i, name in enumerate(rg.POOLED_ORDER):
        mask = ctx.pair_index == i
        a, b, cl = sub(pa, mask), sub(pb, mask), ctx.anchor_group[mask]
        out[name] = {m: compare(a, b, cl, m) for m in ("r1", "gain")}
    return out


def same_metrics(pa, pb):
    return all(np.array_equal(np.asarray(pa[m]), np.asarray(pb[m])) for m in METRICS)


# ---------------------------------------------------------------- text

def c(x):
    return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f},{x['ci95'][1]:6.2f}]"


def c3(x):
    return f"{c(x['r1'])} {c(x['gain'])} {c(x['either'])}"


def picks_text(p):
    return " ".join(f"h{h}:σ{v['sigma']:g}({v['cell'][0]:g},{v['cell'][1]:g})" for h, v in p.items())


def text(r):
    b, cos = r["B"]["alone"], r["cosine"]
    L = [f"EXPLORATORY diagnose_fixes: decides nothing; seed 42 dev episodes only (n={r['n_episodes']}, "
         f"{r['n_clusters']} paintings); 95% CIs resample paintings (5,000); Δ = paired difference vs B.",
         f"B = C2 (stored matched__* reproduced exactly: {r['checks']['B_equals_stored_C2']}; picks {r['B']['picks']}): "
         f"R@1/gain/either {c3(b)} | cosine {c3(cos)} | D0 probes refit = stored D0 told: "
         f"{r['checks']['D0told_equals_stored']}",
         "Columns: R@1 | gain | either (pp). alone = term only (flat = % rankings all tied, a miss); fused = "
         "z(B)(1+λu)+λa·z(T), A′ min-margin cross-fit vs B, picks = σ(λu,λa) per tuning half; per-direction = "
         "per_anchor on one direction duplicated into both slots."]
    for t in TERMS:
        x = r["alone"][t]
        L.append(f"alone {t:12s} {c3(x)} flat {x['flat_share']:5.2f}"
                 + ("  (≡0 on s×g)" if t == "T6oracle_pm" else ""))
    for t in TERMS:
        x = r["fused"][t]
        L.append(f"fused {t:12s} {c3(x)} | Δ {c3(x['vs_B'])} | {picks_text(x['picks'])}")
    for t in TERMS:
        pp = r["fused"][t]["per_pair_vs_B"]
        L.append(f"pair  {t:12s} ΔR@1/Δgain " + " | ".join(
            f"{PAIR_SHORT[p]} {c(v['r1'])} {c(v['gain'])}" for p, v in pp.items()))
    for t in CASCADE_TERMS:
        for k in CASCADE_KS:
            x = r["cascade"][f"{t}_k{k}"]
            L.append(f"casc  {t:9s} k={k} {c3(x)} | Δ {c3(x['vs_B'])}")
    rd = r["reader"]
    L.append(f"reader T6 hard pick = told partition (s×g: image counts for both): pooled {c(rd['correct_share'])}; "
             + "; ".join(f"{PAIR_SHORT[p]} a {v['a']:.1f} b {v['b']:.1f}" for p, v in rd["per_pair_condition"].items())
             + f"; both conditions correct in {rd['both_correct_share']:.1f}% of episodes")
    sp = rd["fused_T6_split"]
    L.append("reader fused T6 − B per anchor: " + " | ".join(
        f"{k} (n={v['n']}) ΔR@1 {c(v['r1'])} Δgain {c(v['gain'])} Δeither {c(v['either'])}"
        for k, v in sp.items()))
    for name in ("B", *r["directions"]["best_two"]):
        x = r["directions"]["scores"][name]
        L.append(f"dir   {name:12s} " + " | ".join(f"{d} {c3(x[d])}" for d in DIRECTIONS))
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main():
    if OUT_JSON.exists() or OUT_TXT.exists():
        raise SystemExit(f"{OUT_JSON} or {OUT_TXT} exists; refusing to overwrite.")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------- step 0: context, inputs, base B = C2
    ctx = rg.EvalContext(SEED, False)
    ep, cl = ctx.pooled, ctx.anchor_group
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    inp, _, a3_prov = rc.model_inputs(ctx, "A3", scorer_train, False)
    t_n1u = centered_term(inp, ep, uniform=True)
    if rg.sha_file(POSTERIORS) != n6c.POSTERIORS_SHA:
        raise AssertionError(f"{POSTERIORS}: SHA-256 differs from ADDENDUM_3_N6C.md")
    post = n6.load_posteriors(POSTERIORS, ctx)
    t6, t6soft, t6u, info, _ = n6.n6_terms(post, ep)
    B, picks_B = crossfit_condition_free(ctx.cos, t_n1u, t6u, ctx.parity)
    pB = per_anchor(B)
    gate = np.load(GATE)
    if not (np.array_equal(gate["anchor_group"], cl) and np.array_equal(gate["pair_index"], ctx.pair_index)
            and all(np.array_equal(pB[m], gate[f"matched__{m}"]) for m in METRICS)):
        raise SystemExit("per_anchor(B) differs from the stored C2 (matched__*) arrays: stopping")
    rc.log(f"B reproduces C2 exactly ({ctx.n} episodes)")
    cos_pa, _ = n6.e1_arrays(ctx, False, SEED)

    # ---------------------------------------------------------- step 1: terms
    aspect_of = {"a": np.array([rg.PAIRS[i][0] for i in ctx.pair_index]),
                 "b": np.array([rg.PAIRS[i][1] for i in ctx.pair_index])}
    other_of = {"a": aspect_of["b"], "b": aspect_of["a"]}
    part_idx = {c: np.array([PARTS.index(TOLD_PART[a]) for a in aspect_of[c]]) for c in CONDITIONS}
    part_idx_other = {c: np.array([PARTS.index(TOLD_PART[a]) for a in other_of[c]]) for c in CONDITIONS}
    asp_idx = {c: np.array([ASPECTS.index(a) for a in aspect_of[c]]) for c in CONDITIONS}
    asp_idx_other = {c: np.array([ASPECTS.index(a) for a in other_of[c]]) for c in CONDITIONS}

    st6 = stacked(post, ep, PARTS)
    t6pm, t6pm_soft, deltas = contrastive_terms(post, ep, st6)
    for cnd in CONDITIONS:
        if not np.array_equal(deltas[cnd].argmax(axis=1), info[cnd]["weights"].argmax(axis=1)):
            raise AssertionError("T6pm's argmax differs from N6's hard reader")
    t6oracle = {cnd: pick(st6, part_idx[cnd]) for cnd in CONDITIONS}
    t6oracle_pm = {cnd: minus(pick(st6, part_idx[cnd]), pick(st6, part_idx_other[cnd])) for cnd in CONDITIONS}
    sg = ctx.pair_index == rg.POOLED_ORDER.index("style__genre")
    oracle_pm_zero_sg = all(np.all(t6oracle_pm[cnd][d][sg] == 0) for cnd in CONDITIONS for d in DIRECTIONS)

    rc.log("fitting D0 label probes (as run_checks)")
    post_d0, probe_prov = rc.fit_probes(ctx, artelingo_aspect_labels(ctx.data), scorer_train, rc.PROBE_ROWS)
    std0 = stacked(post_d0, ep, ASPECTS)
    d0told = told_scores(post_d0, ep, asp_idx["a"], asp_idx["b"])
    d0pm = {cnd: minus(pick(std0, asp_idx[cnd]), pick(std0, asp_idx_other[cnd])) for cnd in CONDITIONS}
    stored = np.load(CHECKS)
    d0told_ok = same_metrics(per_anchor(d0told), {m: stored[f"d0_told__{m}"] for m in METRICS})

    terms = {"T6": t6, "T6soft": t6soft, "T6pm": t6pm, "T6pm_soft": t6pm_soft, "T6oracle": t6oracle,
             "T6oracle_pm": t6oracle_pm, "D0pm": d0pm, "D0told": d0told}
    terms = {k: as_f32(v) for k, v in terms.items()}
    result = {"exploratory": "decides nothing; seed 42 development episodes only", "n_episodes": int(ctx.n),
              "n_clusters": int(len(np.unique(cl))),
              "provenance": {"script_sha256": rg.sha_file(Path(__file__)), "a3": a3_prov,
                             "posteriors_sha256": rg.sha_file(POSTERIORS), "c2_arrays": str(GATE.relative_to(rc.ROOT)),
                             "c2_arrays_sha256": rg.sha_file(GATE), "episodes_sha256": ctx.shas, "d0_probes": probe_prov},
              "checks": {"B_equals_stored_C2": True, "D0told_equals_stored": bool(d0told_ok),
                         "T6oracle_pm_zero_on_style_genre": bool(oracle_pm_zero_sg)},
              "cosine": describe(cos_pa, cl), "B": {"alone": describe(pB, cl), "picks": picks_B}}
    result["alone"] = {}
    for t, s in terms.items():
        rc.assert_finite_scores(s, t)
        result["alone"][t] = {**describe(per_anchor(s), cl), "flat_share": rc.flat_share(s)}
        rc.log(f"alone {t}: R@1 {result['alone'][t]['r1']['point']:.2f}")

    # ---------------------------------------------------------- step 2: fusion on B
    result["fused"], fused_scores, fused_pa = {}, {}, {}
    for t, s in terms.items():
        nested, ctrl, picks = crossfit_nested(B, B, s, ctx.parity)
        pn = per_anchor(nested)
        fused_scores[t], fused_pa[t] = nested, pn
        result["fused"][t] = {**describe(pn, cl), "vs_B": diff(pn, pB, cl), "per_pair_vs_B": per_pair_diff(pn, pB, ctx),
                              "picks": picks, "control_ranks_as_B": same_metrics(per_anchor(ctrl), pB)}
        rc.log(f"fused {t}: ΔR@1 {result['fused'][t]['vs_B']['r1']['point']:.2f}")

    # ---------------------------------------------------------- step 3: cascades on B
    result["cascade"] = {}
    for t in CASCADE_TERMS:
        for k in CASCADE_KS:
            casc = cascade_scores(B, terms[t], k)
            rc.assert_finite_scores(casc, f"cascade {t} k={k}")
            pa = per_anchor(casc)
            result["cascade"][f"{t}_k{k}"] = {**describe(pa, cl), "vs_B": diff(pa, pB, cl)}
    rc.log("cascades done")

    # ---------------------------------------------------------- step 4: reader correctness (T6 hard)
    correct = {cnd: info[cnd]["weights"].argmax(axis=1) == part_idx[cnd] for cnd in CONDITIONS}
    both = correct["a"] & correct["b"]
    reader = {"correct_share": rc.point_ci(0.5 * (correct["a"].astype(float) + correct["b"].astype(float)), cl),
              "per_pair_condition": {name: {cnd: 100 * float(correct[cnd][ctx.pair_index == i].mean())
                                            for cnd in CONDITIONS} for i, name in enumerate(rg.POOLED_ORDER)},
              "both_correct_share": 100 * float(both.mean()), "fused_T6_split": {}}
    pn = fused_pa["T6"]
    for key, mask in (("both correct", both), ("not both", ~both)):
        reader["fused_T6_split"][key] = {
            "n": int(mask.sum()),
            "r1": rc.point_ci((pn["r1"] - pB["r1"])[mask], cl[mask]),
            "gain": rc.point_ci((pn["gain"] - pB["gain"])[mask], cl[mask]),
            "either": rc.point_ci((either(pn) - either(pB))[mask], cl[mask])}
    result["reader"] = reader

    # ---------------------------------------------------------- step 5: directions
    def dir_block(scores):
        out = {}
        for d in DIRECTIONS:
            pa = per_anchor(one_direction(scores, d))
            out[d] = {"r1": rc.point_ci(pa["r1"], cl), "gain": rc.point_ci(pa["gain"], cl),
                      "either": rc.point_ci(either(pa), cl)}
        return out

    best_two = sorted(TERMS, key=lambda t: -result["fused"][t]["vs_B"]["r1"]["point"])[:2]
    result["directions"] = {"method": "per_anchor on a score dict holding one direction duplicated into both "
                                      "direction slots (per_anchor averages the two identical directions)",
                            "best_two": best_two,
                            "scores": {"B": dir_block(B), **{t: dir_block(fused_scores[t]) for t in best_two}}}
    result["runtime_s"] = round(time.time() - t0, 1)
    rg.assert_finite_tree(result)
    OUT_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    txt = text(result)
    OUT_TXT.write_text(txt + "\n")
    print(txt)
    rc.log(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
