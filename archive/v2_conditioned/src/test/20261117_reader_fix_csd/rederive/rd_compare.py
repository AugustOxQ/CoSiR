"""Generic full-precision comparison of a re-derived candidate with its stored results/cand_<name>.{json,npz}.

  python rd_compare.py rb      -> compares out/rd_rb.{json,npz} with every stored Rb candidate
  python rd_compare.py rc      -> compares out/rd_rc.{json,npz} with the stored Rc candidate
"""
import json
import sys

import numpy as np

import rd_core as K

METRICS = ("r1", "gain", "other", "swap", "strict")


def flat(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{prefix}/{k}" if prefix else str(k)))
    elif isinstance(x, list) and all(isinstance(i, (int, float, bool)) for i in x):
        for i, v in enumerate(x):
            out[f"{prefix}[{i}]"] = v
    else:
        out[prefix] = x
    return out


def diffrow(label, s, m):
    if isinstance(s, (bool, str)) or isinstance(m, (bool, str)) or s is None or m is None:
        return (label, s, m, 0.0 if s == m else float("nan"))
    return (label, s, m, abs(float(s) - float(m)))


def keymap():
    km = {"r1_means/fused": "r1_means/fused", "r1_means/counterpart": "r1_means/counterpart", "r1_means/B": "r1_means/B",
          "r1_means/B_prime": "r1_means/B_prime", "bar/comparator": "bar/comparator",
          "clears_bar/clause1_bar_point_at_least_0.5": "clears_bar/c1",
          "clears_bar/clause2_bar_lower_above_0": "clears_bar/c2",
          "clears_bar/clause3_gain_statistic_lower_above_0": "clears_bar/c3", "clears_bar/clears_bar": "clears_bar/clears",
          "pick_accuracy/both_correct_share": "pick_accuracy/both_correct_share",
          "crossfit/control_ranks_as_B": "crossfit/control_ranks_as_B",
          "vs_step1_argmax/argmax_reader/fused_r1": "vs_step1_argmax/step1_fused_r1",
          "vs_step1_argmax/argmax_reader/bar_comparator": "vs_step1_argmax/step1_bar_comparator"}
    q3 = ("point", "ci95[0]", "ci95[1]")
    for q in q3:
        for a, b in (("gain_statistic", "gain_statistic"), ("bar/r1", "bar/r1"),
                     ("pick_accuracy/correct_share", "pick_accuracy/correct_share"),
                     ("vs_step1_argmax/fused_r1", "vs_step1_argmax/fused_r1"),
                     ("vs_step1_argmax/margin_r1", "vs_step1_argmax/margin_r1"),
                     ("vs_step1_argmax/bar_margin_r1", "vs_step1_argmax/bar_margin_r1"),
                     ("vs_step1_argmax/argmax_reader/bar_r1", "vs_step1_argmax/step1_bar_r1"),
                     ("vs_step1_argmax/argmax_reader/margin_r1", "vs_step1_argmax/step1_margin_r1")):
            km[f"{a}/{q}"] = f"{b}/{q}"
        for blk in ("margin", "fused_vs_B", "fused_vs_Bprime", "counterpart_vs_B"):
            for m in ("r1", "gain", "either"):
                km[f"{blk}/{m}/{q}"] = f"{blk}/{m}/{q}"
        for p in K.PAIR_NAMES:
            km[f"bar/per_pair_r1/{p}/{q}"] = f"bar/per_pair_r1/{p}/{q}"
            km[f"per_pair/{p}/fused_vs_B_r1/{q}"] = f"per_pair/{p}/fused_vs_B_r1/{q}"
            km[f"per_pair/{p}/fused_vs_Bprime_r1/{q}"] = f"per_pair/{p}/fused_vs_Bprime_r1/{q}"
            for m in ("r1", "gain", "either"):
                km[f"per_pair/{p}/margin/{m}/{q}"] = f"per_pair/{p}/margin/{m}/{q}"
    for p in K.PAIR_NAMES:
        for c in "ab":
            km[f"pick_accuracy/per_pair_condition/{p}/{c}"] = f"pick_accuracy/per_pair_condition/{p}/{c}"
    for h in range(2):
        km[f"crossfit/T_picks/{h}/sigma"] = f"crossfit/T_picks/{h}/sigma"
        for i in range(2):
            km[f"crossfit/T_picks/{h}/cell[{i}]"] = f"crossfit/T_picks/{h}/cell[{i}]"
            km[f"crossfit/cf_picks/{h}[{i}]"] = f"crossfit/cf_picks/{h}[{i}]"
    return km


def extras(mi, mz, prefix, cfg, comp):
    """Quantities stored by the implementation that we derive from our own arrays."""
    z1 = np.load(K.STEP1_EVAL)
    cl, pi = z1["anchor_group"], z1["pair_index"]
    ck = {"B_prime": f"{cfg}__Bprime", "counterpart": f"{prefix}__cf"}.get(comp)
    comp_gain = mz[f"{ck}__gain"] if ck else mz["B__gain"]
    fg = mz[f"{prefix}__fused__gain"]
    ex = {"bar/comparator_mean_r1/B_prime": mi["r1_means/B_prime"],
          "bar/comparator_mean_r1/counterpart": mi["r1_means/counterpart"],
          "bar/comparator_mean_r1/B": mi["r1_means/B"], "bar/fused_r1": mi["r1_means/fused"],
          "pick_accuracy/chance": 100.0 / len(K.CONFIGS[cfg])}
    gv = K.point_ci(fg - comp_gain, cl)
    ex.update({"bar/gain_vs_comparator/point": gv["point"], "bar/gain_vs_comparator/ci95[0]": gv["ci95"][0],
               "bar/gain_vs_comparator/ci95[1]": gv["ci95"][1]})
    for i, p in enumerate(K.PAIR_NAMES):
        msk = pi == i
        for lab, v in (("bar/per_pair_gain/" + p, fg[msk] - comp_gain[msk]),
                       (f"per_pair/{p}/counterpart_vs_B_r1", mz[f"{prefix}__cf__r1"][msk] - mz["B__r1"][msk]),
                       (f"per_pair/{p}/bar_margin_r1", mz[f"{prefix}__bar_v"][msk])):
            r = K.point_ci(v, cl[msk])
            ex.update({f"{lab}/point": r["point"], f"{lab}/ci95[0]": r["ci95"][0], f"{lab}/ci95[1]": r["ci95"][1]})
    return ex


def compare_candidate(name, cfg, mine_cand, mz, prefix, extra_arrays=()):
    st = flat(json.loads((K.RES / f"cand_{name}.json").read_text()))
    mi = flat(mine_cand)
    rows, arr_rows = [], []
    km = keymap()
    for sk, mk in km.items():
        if sk in st:
            rows.append(diffrow(f"{name}/{sk}", st[sk], mi.get(mk, "MISSING")))
    for k in st:
        if k.startswith("pick_share/"):
            rows.append(diffrow(f"{name}/{k}", st[k], mi.get(k, "MISSING")))
        if k.startswith("AR_check_rand_share/"):
            tail = k[len("AR_check_rand_share/"):]
            mk = "pick_share/overall/rand" if tail == "overall" else f"pick_share/{tail}/rand"
            rows.append(diffrow(f"{name}/{k}", st[k], mi.get(mk, "MISSING")))
    for c in "ab":
        for j in range(len(K.CONFIGS[cfg])):
            if f"picks/{c}[{j}]" in st:
                rows.append(diffrow(f"{name}/picks/{c}[{j}]", st[f"picks/{c}[{j}]"], mine_cand["picks_count"][c][j]))
    ex = extras(mi, mz, prefix, cfg, mine_cand["bar"]["comparator"])
    for k, v in ex.items():
        rows.append(diffrow(f"{name}/{k}", st.get(k, "MISSING"), v))
    skip = ("pick_share/", "picks/", "provenance", "sigma/", "extra/", "groupings", "tau", "rc/", "AR_check_rand_share/",
            "crossfit/cells/", "gate_open_share/")
    unc = [k for k in st if k not in km and k not in ex and not k.startswith(skip) and not k.endswith("n_clusters")
           and k not in ("name", "config", "n_episodes", "smoke", "scoring", "parent")]
    sz = np.load(K.RES / f"cand_{name}.npz")
    pairs = [(f"fused__{m}", f"{prefix}__fused__{m}") for m in METRICS] + \
            [(f"cf__{m}", f"{prefix}__cf__{m}") for m in METRICS] + [("bar_v", f"{prefix}__bar_v")] + \
            [(f"pick__{c}", f"{prefix}__pick__{c}") for c in "ab"] + \
            [(f"margin__{c}", f"{prefix}__margin__{c}") for c in "ab"] + \
            [(f"T__{c}__{d}", f"{prefix}__T__{c}__{d}") for c in "ab" for d in ("i2t", "t2i")] + list(extra_arrays)
    for sk, mk in pairs:
        if sk not in sz.files:
            arr_rows.append((f"{name}/{sk}", "NOT STORED", float("nan"), "", ""))
            continue
        a, b = np.asarray(sz[sk]), np.asarray(mz[mk])
        eq = a.shape == b.shape and np.array_equal(a, b)
        mx = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape else float("nan")
        arr_rows.append((f"{name}/{sk}", bool(eq), mx, str(a.dtype), str(b.dtype)))
    return rows, arr_rows, unc


def report(rows, arr_rows, unc_all, out_name):
    nonzero = [r for r in rows if r[3] != 0.0]
    bad = [r for r in arr_rows if r[1] is not True]
    print(f"scalars compared {len(rows)}; nonzero/unmatched {len(nonzero)}")
    for r in nonzero:
        print("  ", r)
    print(f"arrays compared {len(arr_rows)}; not identical {len(bad)}")
    for r in bad:
        print("  ", r)
    for n, u in unc_all.items():
        if u:
            print(f"  {n}: stored keys not compared: {u}")
    K.save_json(K.OUT / out_name, {"n_scalars": len(rows), "nonzero": [list(map(str, r)) for r in nonzero],
                                   "n_arrays": len(arr_rows), "arrays_not_identical": [list(map(str, r)) for r in bad],
                                   "uncompared": unc_all})


def main_rb(tag="rd_rb"):
    mine = json.loads((K.OUT / f"{tag}.json").read_text())
    mz = np.load(K.OUT / f"{tag}.npz")
    rows, arr_rows, unc = [], [], {}
    for cfg in ("A1", "A0", "AR"):
        for scoring in ("argmax", "expected"):
            name = f"Rb_{scoring}_{cfg}"
            r, a, u = compare_candidate(name, cfg, mine["candidates"][name], mz, name,
                                        extra_arrays=[("extra__probs_a", f"{cfg}__P__a"), ("extra__probs_b", f"{cfg}__P__b")])
            rows += r
            arr_rows += a
            unc[name] = u
    sm = flat(json.loads((K.RES / "rb_summary.json").read_text()))
    mm = flat(mine["A1_minus_A0"])
    for scoring in ("argmax", "expected"):
        for q in ("point", "ci95[0]", "ci95[1]"):
            rows.append(diffrow(f"rb_summary/A1_minus_A0/{scoring}/fused_r1/{q}",
                                sm[f"A1_minus_A0/{scoring}/fused_r1/{q}"], mm[f"{scoring}/fused_r1/{q}"]))
            rows.append(diffrow(f"rb_summary/A1_minus_A0/{scoring}/bar_margin_r1/{q}",
                                sm[f"A1_minus_A0/{scoring}/bar_margin_r1/{q}"], mm[f"{scoring}/bar_r1/{q}"]))
        mAR = flat(mine["candidates"][f"Rb_{scoring}_AR"])
        rows.append(diffrow(f"rb_summary/AR_check/{scoring}/rand_share/overall",
                            sm[f"AR_check/{scoring}/rand_share/overall"], mAR["pick_share/overall/rand"]))
        for p in K.PAIR_NAMES:
            for c in "ab":
                rows.append(diffrow(f"rb_summary/AR_check/{scoring}/rand_share/{p}/{c}",
                                    sm[f"AR_check/{scoring}/rand_share/per_pair_condition/{p}/{c}"],
                                    mAR[f"pick_share/per_pair_condition/{p}/{c}/rand"]))
        for q in ("point", "ci95[0]", "ci95[1]"):
            rows.append(diffrow(f"rb_summary/AR_check/{scoring}/margin_r1/{q}", sm[f"AR_check/{scoring}/margin_r1/{q}"],
                                mAR[f"margin/r1/{q}"]))
            rows.append(diffrow(f"rb_summary/AR_check/{scoring}/bar_r1/{q}", sm[f"AR_check/{scoring}/bar_r1/{q}"],
                                mAR[f"bar/r1/{q}"]))
        rows.append(diffrow(f"rb_summary/AR_check/{scoring}/bar_comparator", sm[f"AR_check/{scoring}/bar_comparator"],
                            mAR["bar/comparator"]))
        for nm in ("Rb_argmax_AR", "Rb_expected_AR"):
            pass
    report(rows, arr_rows, unc, f"compare_{tag}.json")
    print("readers", json.dumps(mine["readers"]))


def main_rc():
    mine = json.loads((K.OUT / "rd_rc.json").read_text())
    mz = np.load(K.OUT / "rd_rc.npz")
    stj = flat(json.loads((K.RES / "cand_Rc_Rb_expected_A0.json").read_text()))
    tauj = json.loads((K.RES / "rc_tau.json").read_text())
    rows_all, arr_all, unc = [], [], {}
    for label in ("Rc_mine", "Rc_mine_f64feat", "Rc_stored_parent"):
        cand = dict(mine[label])
        cand["picks_count"] = {c: np.bincount(mz[f"{label}__pick__{c}"].astype(int), minlength=3).tolist() for c in "ab"}
        rows, arr_rows, u = compare_candidate("Rc_Rb_expected_A0", "A0", cand, mz, label,
                                              extra_arrays=[("extra__gate_a", f"{label}__gate__a"),
                                                            ("extra__gate_b", f"{label}__gate__b"),
                                                            ("extra__taus", f"{label}__taus")])
        rows = [(f"[{label}] " + r[0],) + tuple(r[1:]) for r in rows]
        arr_rows = [(f"[{label}] " + r[0],) + tuple(r[1:]) for r in arr_rows]
        mi = flat(cand)
        for h in ("0", "1"):
            for part, mk in (("fused", "fused"), ("counterpart", "counterpart")):
                for q in ("tau_index", "tau", "lambda_u", "lambda_a"):
                    rows.append(diffrow(f"[{label}] crossfit/{part}/{h}/{q}", stj[f"crossfit/cells/{part}/{h}/{q}"],
                                        mi[f"crossfit/{mk}/{h}/{q}"]))
            rows.append(diffrow(f"[{label}] B_r1_on_tune_half/{h}", stj[f"crossfit/cells/control/{h}/r1_on_tune_half"],
                                mi[f"crossfit/B_r1_on_tune_half/{h}"]))
        for i in range(4):
            rows.append(diffrow(f"[{label}] tau_{i} (rc_tau.json)", tauj["taus"][i], cand["taus"][i]))
            for c in ("overall", "a", "b"):
                rows.append(diffrow(f"[{label}] gate_open_share/tau_{i}/{c}", stj[f"gate_open_share/tau_{i}/{c}"],
                                    mi[f"gate_open_share/tau_{i}/{c}"]))
        for sk, mk in (("tau0_sanity/fused_score_arrays_equal_parent_cells", "tau0_sanity/fused_score_arrays_equal_parent_cells"),
                       ("tau0_sanity/tau0_restricted_counterpart_r1_mean", "tau0_sanity/tau0_restricted_counterpart_r1_mean"),
                       ("tau0_sanity/counterpart_max_abs_score_diff", "tau0_sanity/counterpart_max_abs_score_diff_vs_parent_cf_cells")):
            rows.append(diffrow(f"[{label}] {sk}", stj[sk], mi[mk]))
        rows_all += rows
        arr_all += arr_rows
        unc[label] = u
    report(rows_all, arr_all, unc, "rd_compare_rc.json")
    for k in ("gate_flips_mine_vs_stored_margins", "tau_mine_minus_stored", "gate_flips_f64feat_vs_mine",
              "tau_f64feat_minus_mine", "gate_flips_f64_vs_f32_margins",
              "taus_f32_margins", "nearest_margin_distance_to_tau"):
        print(k, mine[k])
    for label in ("Rc_mine", "Rc_mine_f64feat", "Rc_stored_parent"):
        print(label, "tau0_sanity", mine[label]["tau0_sanity"], "G_cf condition-free", mine[label]["G_cf_condition_free"])


if __name__ == "__main__":
    {"rb": main_rb, "rb_ad": lambda: main_rb("rd_rb_ad"), "rc": main_rc}[sys.argv[1]]()
