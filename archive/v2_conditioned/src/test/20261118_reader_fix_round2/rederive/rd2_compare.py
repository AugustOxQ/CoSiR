"""Phase-2 comparison: our re-derived numbers (out/rd2_phase2.*, out/rd2_r2_A0.*, out/rd2_r3_A0.*, out/rd2_regress.*)
against the main run's result files in ../results/ (JSON, npz and txt only; the implementation's .py files are never
read). Tolerances of rule §7: discrete quantities identical; mu42, sigma42, pi_hat, tau, D(k) within 1e-12 relative
(absolute 1e-12 where the value is exactly 0, the Delta entries of mu42); bar margins, gain statistics and their bounds
within 1e-9 pp. Further quantities (per-anchor arrays, probabilities, T, gates, integer criteria, SMDs, side
comparisons) are compared as well and labelled as checks outside §7. Writes out/rd2_compare.json and
out/rd2_compare_table.md."""
import json

import numpy as np

import rd2_core as K
import rd2_family as FAM
from rd2_core import COND, DIRS, RC

RES = K.R2DIR / "results"
ROWS = []


def add(group, qty, stored, mine, kind, rule7=True):
    """kind: 'eq' (identical), 'rel' (1e-12 relative, absolute 1e-12 at 0), 'pp' (1e-9 pp), 'arr' (arrays identical),
    'arr_info' (arrays, max |diff| reported, identical required), 'info' (reported only)."""
    if kind == "eq":
        diff = None
        ok = stored == mine
    elif kind == "rel":
        a, b = float(stored), float(mine)
        diff = abs(a - b)
        scale = max(abs(a), abs(b))
        ok = diff <= 1e-12 * scale if scale > 0 else diff <= 1e-12
    elif kind == "pp":
        a, b = float(stored), float(mine)
        diff = abs(a - b)
        ok = diff <= 1e-9
    elif kind in ("arr", "arr_info"):
        a, b = np.asarray(stored), np.asarray(mine)
        same_shape = a.shape == b.shape
        diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if same_shape and a.size else None
        ok = bool(same_shape and np.array_equal(a.astype(np.float64), b.astype(np.float64)))
        stored, mine = f"array {a.shape} {a.dtype}", f"array {b.shape} {b.dtype}"
    else:
        diff, ok = None, None
    ROWS.append({"group": group, "quantity": qty, "stored": stored, "rederived": mine, "abs_diff": diff,
                 "kind": kind, "rule7": rule7, "agree": ok})
    return ok


def ci_rows(group, name, st, mi, rule7=True):
    add(group, f"{name} point", st["point"], mi["point"], "pp", rule7)
    add(group, f"{name} lower", st["ci95"][0], mi["ci95"][0], "pp", rule7)
    add(group, f"{name} upper", st["ci95"][1], mi["ci95"][1], "pp", rule7)


def main():
    RC.assert_rule()
    cache = K.load_cache()
    p2 = json.loads((K.OUT / "rd2_phase2.json").read_text())
    z2 = np.load(K.OUT / "rd2_phase2.npz")
    # ------------------------------------------------ regression check (main run's file)
    rc = json.loads((RES / "regression_check.json").read_text())
    mine_rc = json.loads((K.OUT / "rd2_regress.json").read_text())["checks"]
    add("regression", "passed", rc["passed"], mine_rc["PASSED"], "eq")
    add("regression", "all 34 comparisons ok", all(c["ok"] for c in rc["comparisons"]), True, "eq")
    # ------------------------------------------------ per candidate
    for name in ("R1", "R2", "R3"):
        g = f"{name}/A0"
        sj = json.loads((RES / f"cand_{name}_A0.json").read_text())
        sz = np.load(RES / f"cand_{name}_A0.npz")
        tj = json.loads((RES / f"tau_{name}_A0.json").read_text())
        ev = p2["candidates"][name]
        fam = ev["family"]
        # tau
        for t in range(4):
            add(g, f"tau_{t} (tau_{name}_A0.json)", tj["taus"][t], fam["taus"][t], "rel")
        add(g, "tau (cand json) equal tau json", sj["taus"] == tj["taus"], True, "eq")
        # cells, sigma*
        for h in ("0", "1"):
            sf, mf = sj["crossfit"]["cells"]["fused"][h], fam["fused_cells"][h]
            sc, mc = sj["crossfit"]["cells"]["counterpart"][h], fam["counterpart_cells"][h]
            add(g, f"fused cell, tune half {h}", sf["cell"], mf["cell"], "eq")
            add(g, f"fused cell settings, half {h} (k_top, tau idx, lam_u, lam_a)",
                (sf["k_top"], sf["tau_index"], sf["lambda_u"], sf["lambda_a"]),
                (mf["k_top"], mf["tau_index"], mf["lambda_u"], mf["lambda_a"]), "eq")
            add(g, f"counterpart cell, tune half {h}", sc["cell"], mc["cell"], "eq")
            add(g, f"counterpart cell settings, half {h}",
                (sc["k_top"], sc["tau_index"], sc["lambda_u"], sc["lambda_a"]),
                (mc["k_top"], mc["tau_index"], mc["lambda_u"], mc["lambda_a"]), "eq")
            add(g, f"control sigma*, half {h}", sj["crossfit"]["cells"]["control"][h]["sigma"],
                fam["control"][h]["sigma"], "eq")
            ic = sj["crossfit"]["cells"]["integer_criteria"]
            add(g, f"integer rho / gamma / rho_ctrl of the fused pick, half {h}",
                (ic["fused"][h]["rho"], ic["fused"][h]["gamma"], ic["fused"][h]["rho_ctrl"]),
                (mf["rho"], mf["gamma"], fam["control"][h]["rho_ctrl"]), "eq", False)
            add(g, f"integer rho of the counterpart pick, half {h}", ic["counterpart"][h]["rho"], mc["rho"], "eq", False)
        npz_cells = (sz["fused_cells"].tolist(), sz["cf_cells"].tolist(), sz["ctrl_sigma"].tolist())
        add(g, "npz fused_cells / cf_cells / ctrl_sigma",
            npz_cells, ([fam["fused_cells"][h]["cell"] for h in "01"], [fam["counterpart_cells"][h]["cell"] for h in "01"],
                        [fam["control"][h]["sigma"] for h in "01"]), "eq")
        # bar comparator, bar margin, gain statistic, clauses
        add(g, "bar comparator", sj["bar"]["comparator"], ev["bar_comparator"], "eq")
        ci_rows(g, "bar margin", sj["bar"]["r1"], ev["bar_margin"])
        ci_rows(g, "gain statistic", sj["gain_statistic"], ev["gain_statistic"])
        cb = sj["clears_bar"]
        add(g, "clause 1 (bar point >= 0.5)", cb["clause1_bar_point_at_least_0.5"], ev["clauses"]["c1_bar_point_ge_0.5"], "eq")
        add(g, "clause 2 (bar lower > 0)", cb["clause2_bar_lower_above_0"], ev["clauses"]["c2_bar_lower_gt_0"], "eq")
        add(g, "clause 3 (gain lower > 0)", cb["clause3_gain_statistic_lower_above_0"], ev["clauses"]["c3_gain_lower_gt_0"], "eq")
        add(g, "clears_bar", cb["clears_bar"], ev["clauses"]["clears_bar"], "eq")
        # comparator means and side comparisons (outside §7)
        for k in ("fused", "counterpart", "B", "B_prime"):
            add(g, f"mean R@1 {k}", sj["r1_means"][k], ev["r1_means"][k], "pp", False)
        for blk in ("margin", "fused_vs_B", "fused_vs_Bprime", "counterpart_vs_B"):
            for met in ("r1", "gain", "either"):
                ci_rows(g, f"{blk}.{met}", sj[blk][met], ev[blk][met], False)
        for p in K.PAIR_NAMES:
            ci_rows(g, f"bar margin {p}", sj["bar"]["per_pair_r1"][p], ev["per_pair"][p]["bar_margin_r1"], False)
        ci_rows(g, "pick accuracy (diagnostic)", sj["pick_accuracy"]["correct_share"], ev["pick_accuracy"]["correct_share"],
                False)
        # per-anchor arrays
        for part, pre in (("fused", "fused"), ("cf", "cf")):
            for m in K.METRICS:
                add(g, f"per-anchor {part}__{m}", sz[f"{part}__{m}"], z2[f"{name}__{pre}__{m}"], "arr",
                    m in ("r1", "gain"))
        add(g, "per-anchor bar_v", sz["bar_v"], z2[f"{name}__bar_v"], "arr")
        # probabilities, picks, margins, T, gates (outside §7)
        P = {c: z2[f"{name}__P__{c}"] for c in COND}
        T, picks, margins = FAM.reader_parts(cache, P, "A0")
        for c in COND:
            add(g, f"probs__{c} (cand npz)", sz[f"probs__{c}"], P[c], "arr_info", False)
            add(g, f"pick__{c}", sz[f"pick__{c}"], picks[c], "arr", False)
            add(g, f"margin__{c}", sz[f"margin__{c}"], margins[c], "arr_info", False)
            for d in DIRS:
                add(g, f"T__{c}__{d}", sz[f"T__{c}__{d}"], T[c][d], "arr_info", False)
            gates = np.stack([(margins[c] >= t) for t in fam["taus"]]).astype(np.float32)
            add(g, f"gate__{c} (4 x E)", sz[f"gate__{c}"], gates, "arr", False)
        add(g, "anchor_group / pair_index / parity", all(np.array_equal(sz[k], cache[kk]) for k, kk in
                                                          (("anchor_group", "cl"), ("pair_index", "pair_index"),
                                                           ("parity", "parity"))), True, "eq", False)
    # ------------------------------------------------ R2's frozen parameters
    pj = json.loads((RES / "probs_R2_A0.json").read_text())
    pz = np.load(RES / "probs_R2_A0.npz")
    r2 = json.loads((K.OUT / "rd2_r2_A0.json").read_text())
    r2z = np.load(K.OUT / "rd2_r2_A0.npz")
    names = r2["feature_names"]
    for i, n in enumerate(names):
        add("R2 mu42", n, pj["mu42"][i], r2["mu42"][i], "rel")
    for i, n in enumerate(names):
        add("R2 sigma42", n, pj["sigma42"][i], r2["sigma42"][i], "rel")
    add("R2 sigma42", "zero entries replaced", pj["sigma42_zero_replaced_by_1"], r2["sigma42_zero_entries"], "eq")
    for i, h in enumerate(K.CONFIGS["A0"]):
        add("R2 EM", f"pi_hat[{h}]", pj["pi_hat"][i], r2["em"]["pi_hat"][i], "rel")
    add("R2 EM", "iterations (stored n_iter vs our updates computed)", pj["em"]["n_iter"], r2["em"]["n_updates"], "eq")
    add("R2 EM", "cap reached", pj["em"]["cap_reached"], r2["em"]["cap_reached"], "eq")
    add("R2 EM", "last max |change|", pj["em"]["last_max_abs_change"], r2["em"]["last_max_abs_change"], "rel", False)
    add("R2 code check", "passed", pj["code_check"]["passed"], r2["code_check"]["passed"], "eq")
    for c in COND:
        add("R2 probabilities", f"P__{c} (probs_R2_A0.npz)", pz[f"P__{c}"], r2z[f"P__{c}"], "arr_info", False)
    # ------------------------------------------------ R3
    kj = json.loads((RES / "r3_k_A0.json").read_text())
    p3 = json.loads((RES / "probs_R3_A0.json").read_text())
    p3z = np.load(RES / "probs_R3_A0.npz")
    r3 = json.loads((K.OUT / "rd2_r3_A0.json").read_text())
    r3z = np.load(K.OUT / "rd2_r3_A0.npz")
    for k in "1234":
        add("R3 D(k)", f"D({k})", kj["D"][k], r3["D_table"][k]["D"], "rel")
    for k in "1234":
        for n in names:
            add("R3 SMD", f"k={k} {n}", kj["smd"][k][n], r3["D_table"][k]["smd"][n], "rel", False)
    add("R3", "k*", kj["k_star"], r3["k_star"], "eq")
    add("R3", "k* (probs_R3_A0.json)", p3["k_star"], r3["k_star"], "eq")
    for j in "01":
        for a in ("order", "img", "cap"):
            add("R3 SHA-256", f"half {j} {a}", kj["sha256"][f"half{j}"][a], r3["halves"][j]["sha256"][a], "eq")
        for k in "1234":
            add("R3 SHA-256", f"half {j} features purity {k}", kj["sha256"][f"half{j}"][f"features_purity{k}"],
                r3["halves"][j]["purity"][k]["sha256_X"], "eq")
    for j in "01":
        hr = p3["half_readers"][j]
        tr = r3["training"][j]
        add("R3 training", f"chosen C, half {j}", hr["chosen_C"], tr["chosen_C"], "eq")
        for row in tr["cv_table"]:
            add("R3 training", f"half {j} CV mean log loss, C {row['C']}", hr["cv_mean_log_loss"][str(row["C"])],
                row["mean_log_loss"], "rel", False)
        add("R3 training", f"half {j} OOF accuracy at chosen C", hr["oof_accuracy_at_chosen_C"],
            tr["oof_accuracy_at_chosen_C"], "pp", False)
    for c in COND:
        add("R3 probabilities", f"P__{c} (probs_R3_A0.npz)", p3z[f"P__{c}"], r3z[f"P__{c}"], "arr_info", False)
    for j in "01":
        d = kj["draws"][f"half{j}"]
        hi = r3["halves"][j]["draw_info"]
        add("R3 draws", f"half {j} redrawn entries (img, cap)", (d["img_entries_redrawn"], d["cap_entries_redrawn"]),
            (hi["img_redrawn_entries"], hi["cap_redrawn_entries"]), "eq", False)
    # ------------------------------------------------ rule application (our own; both sides)
    clears = {n: json.loads((RES / f"cand_{n}_A0.json").read_text())["clears_bar"]["clears_bar"] for n in ("R1", "R2", "R3")}
    add("rule", "candidates clearing the bar", [n for n in ("R1", "R2", "R3") if clears[n]], p2["rule"]["clear_the_bar"], "eq")

    out = {"rows": ROWS}
    r7 = [r for r in ROWS if r["rule7"]]
    out["n_rows"] = len(ROWS)
    out["n_rule7_rows"] = len(r7)
    out["rule7_all_agree"] = all(r["agree"] for r in r7)
    out["all_agree"] = all(r["agree"] for r in ROWS if r["agree"] is not None)
    out["disagreements"] = [r for r in ROWS if r["agree"] is False]
    K.save_json(K.OUT / "rd2_compare.json", out)

    def fmt(v):
        if isinstance(v, float):
            return repr(v)
        return str(v)
    L = ["| group | quantity | stored | re-derived | abs diff | §7 | agree |", "|---|---|---|---|---|---|---|"]
    for r in ROWS:
        d = "" if r["abs_diff"] is None else f"{r['abs_diff']:.3g}"
        L.append(f"| {r['group']} | {r['quantity']} | {fmt(r['stored'])} | {fmt(r['rederived'])} | {d} | "
                 f"{'yes' if r['rule7'] else 'no'} | {'yes' if r['agree'] else ('n/a' if r['agree'] is None else 'NO')} |")
    (K.OUT / "rd2_compare_table.md").write_text("\n".join(L) + "\n")
    print(f"rows {len(ROWS)}, rule-7 rows {len(r7)}, rule-7 all agree {out['rule7_all_agree']}, all agree "
          f"{out['all_agree']}; disagreements: {[(r['group'], r['quantity'], r['abs_diff']) for r in out['disagreements']]}")


if __name__ == "__main__":
    main()
