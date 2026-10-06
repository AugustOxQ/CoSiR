"""Compare rederive/out/rd_ra.{json,npz} with the stored run_ra outputs (results/cand_Ra_*.json/.npz, ra_sigma.json,
ra_summary.json) at full precision. Writes rederive/out/rd_compare_ra.json and prints a table."""
import json

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


def cmp_pairs(rows, label, s, m):
    if isinstance(s, bool) or isinstance(m, bool) or isinstance(s, str) or isinstance(m, str):
        rows.append((label, s, m, 0.0 if s == m else float("nan")))
    else:
        rows.append((label, s, m, abs(float(s) - float(m))))


def main():
    mine = json.loads((K.OUT / "rd_ra.json").read_text())
    mz = np.load(K.OUT / "rd_ra.npz")
    rows, arr_rows = [], []
    sig = json.loads((K.RES / "ra_sigma.json").read_text())["sigma"]
    for h in K.GROUPINGS:
        cmp_pairs(rows, f"sigma/{h} (ours float64 agreements)", sig[h], mine["sigma"][h])
        cmp_pairs(rows, f"sigma/{h} (ours float32 agreements)", sig[h], mine["sigma_f32_agreements"][h])
        rows.append((f"sigma/{h} relative diff (float64 agreements)", sig[h], mine["sigma"][h],
                     abs(sig[h] - mine["sigma"][h]) / sig[h]))

    keymap = {  # stored path -> our path
        "r1_means/fused": "r1_means/fused", "r1_means/counterpart": "r1_means/counterpart",
        "r1_means/B": "r1_means/B", "r1_means/B_prime": "r1_means/B_prime",
        "bar/comparator": "bar/comparator", "gain_statistic/point": "gain_statistic/point",
        "gain_statistic/ci95[0]": "gain_statistic/ci95[0]", "gain_statistic/ci95[1]": "gain_statistic/ci95[1]",
        "bar/r1/point": "bar/r1/point", "bar/r1/ci95[0]": "bar/r1/ci95[0]", "bar/r1/ci95[1]": "bar/r1/ci95[1]",
        "clears_bar/clause1_bar_point_at_least_0.5": "clears_bar/c1", "clears_bar/clause2_bar_lower_above_0": "clears_bar/c2",
        "clears_bar/clause3_gain_statistic_lower_above_0": "clears_bar/c3", "clears_bar/clears_bar": "clears_bar/clears",
        "pick_accuracy/correct_share/point": "pick_accuracy/correct_share/point",
        "pick_accuracy/correct_share/ci95[0]": "pick_accuracy/correct_share/ci95[0]",
        "pick_accuracy/correct_share/ci95[1]": "pick_accuracy/correct_share/ci95[1]",
        "pick_accuracy/both_correct_share": "pick_accuracy/both_correct_share",
        "crossfit/control_ranks_as_B": "crossfit/control_ranks_as_B",
        "vs_step1_argmax/fused_r1/point": "vs_step1_argmax/fused_r1/point",
        "vs_step1_argmax/fused_r1/ci95[0]": "vs_step1_argmax/fused_r1/ci95[0]",
        "vs_step1_argmax/fused_r1/ci95[1]": "vs_step1_argmax/fused_r1/ci95[1]",
        "vs_step1_argmax/margin_r1/point": "vs_step1_argmax/margin_r1/point",
        "vs_step1_argmax/margin_r1/ci95[0]": "vs_step1_argmax/margin_r1/ci95[0]",
        "vs_step1_argmax/margin_r1/ci95[1]": "vs_step1_argmax/margin_r1/ci95[1]",
        "vs_step1_argmax/bar_margin_r1/point": "vs_step1_argmax/bar_margin_r1/point",
        "vs_step1_argmax/bar_margin_r1/ci95[0]": "vs_step1_argmax/bar_margin_r1/ci95[0]",
        "vs_step1_argmax/bar_margin_r1/ci95[1]": "vs_step1_argmax/bar_margin_r1/ci95[1]",
        "vs_step1_argmax/argmax_reader/fused_r1": "vs_step1_argmax/step1_fused_r1",
        "vs_step1_argmax/argmax_reader/bar_comparator": "vs_step1_argmax/step1_bar_comparator",
        "vs_step1_argmax/argmax_reader/bar_r1/point": "vs_step1_argmax/step1_bar_r1/point",
        "vs_step1_argmax/argmax_reader/bar_r1/ci95[0]": "vs_step1_argmax/step1_bar_r1/ci95[0]",
        "vs_step1_argmax/argmax_reader/bar_r1/ci95[1]": "vs_step1_argmax/step1_bar_r1/ci95[1]",
        "vs_step1_argmax/argmax_reader/margin_r1/point": "vs_step1_argmax/step1_margin_r1/point",
        "vs_step1_argmax/argmax_reader/margin_r1/ci95[0]": "vs_step1_argmax/step1_margin_r1/ci95[0]",
        "vs_step1_argmax/argmax_reader/margin_r1/ci95[1]": "vs_step1_argmax/step1_margin_r1/ci95[1]",
    }
    for blk in ("margin", "fused_vs_B", "fused_vs_Bprime", "counterpart_vs_B"):
        for m in ("r1", "gain", "either"):
            for q in ("point", "ci95[0]", "ci95[1]"):
                keymap[f"{blk}/{m}/{q}"] = f"{blk}/{m}/{q}"
    for p in K.PAIR_NAMES:
        for q in ("point", "ci95[0]", "ci95[1]"):
            keymap[f"bar/per_pair_r1/{p}/{q}"] = f"bar/per_pair_r1/{p}/{q}"
            keymap[f"per_pair/{p}/fused_vs_B_r1/{q}"] = f"per_pair/{p}/fused_vs_B_r1/{q}"
            keymap[f"per_pair/{p}/fused_vs_Bprime_r1/{q}"] = f"per_pair/{p}/fused_vs_Bprime_r1/{q}"
            for m in ("r1", "gain", "either"):
                keymap[f"per_pair/{p}/margin/{m}/{q}"] = f"per_pair/{p}/margin/{m}/{q}"
        for c in "ab":
            keymap[f"pick_accuracy/per_pair_condition/{p}/{c}"] = f"pick_accuracy/per_pair_condition/{p}/{c}"
    for h in range(2):
        keymap[f"crossfit/T_picks/{h}/sigma"] = f"crossfit/T_picks/{h}/sigma"
        keymap[f"crossfit/T_picks/{h}/cell[0]"] = f"crossfit/T_picks/{h}/cell[0]"
        keymap[f"crossfit/T_picks/{h}/cell[1]"] = f"crossfit/T_picks/{h}/cell[1]"
        keymap[f"crossfit/cf_picks/{h}[0]"] = f"crossfit/cf_picks/{h}[0]"
        keymap[f"crossfit/cf_picks/{h}[1]"] = f"crossfit/cf_picks/{h}[1]"

    for cfg in ("A1", "A0", "AR"):
        name = f"Ra_{cfg}"
        st = flat(json.loads((K.RES / f"cand_{name}.json").read_text()))
        mi = flat(mine["candidates"][name])
        sh = {k: v for k, v in st.items() if k.startswith("pick_share/")}
        for k in sh:
            keymap_k = k
            if keymap_k in mi:
                cmp_pairs(rows, f"{name}/{k}", st[k], mi[keymap_k])
            else:
                rows.append((f"{name}/{k}", st[k], "MISSING", float("nan")))
        for c in "ab":
            for j, v in enumerate(st.get(f"picks/{c}", [])) if isinstance(st.get(f"picks/{c}"), list) else []:
                pass
        for c in "ab":
            for j in range(len(K.CONFIGS[cfg])):
                k = f"picks/{c}[{j}]"
                if k in st:
                    cmp_pairs(rows, f"{name}/{k}", st[k], mine["candidates"][name]["picks_count"][c][j])
        for sk, mk in keymap.items():
            if sk in st:
                cmp_pairs(rows, f"{name}/{sk}", st[sk], mi.get(mk, "MISSING"))
        # extra quantities derived from our own arrays (our comparator)
        z1 = np.load(K.STEP1_EVAL)
        cl, pi = z1["anchor_group"], z1["pair_index"]
        comp = mine["candidates"][name]["bar"]["comparator"]
        ck = {"B_prime": f"{cfg}__Bprime", "counterpart": f"{cfg}__cf"}.get(comp)
        comp_gain = mz[f"{ck}__gain"] if ck else mz["B__gain"]
        fg = mz[f"{cfg}__fused__gain"]
        extra = {"bar/comparator_mean_r1/B_prime": mi["r1_means/B_prime"],
                 "bar/comparator_mean_r1/counterpart": mi["r1_means/counterpart"],
                 "bar/comparator_mean_r1/B": mi["r1_means/B"], "bar/fused_r1": mi["r1_means/fused"],
                 "pick_accuracy/chance": 100.0 / len(K.CONFIGS[cfg])}
        gv = K.point_ci(fg - comp_gain, cl)
        extra.update({"bar/gain_vs_comparator/point": gv["point"], "bar/gain_vs_comparator/ci95[0]": gv["ci95"][0],
                      "bar/gain_vs_comparator/ci95[1]": gv["ci95"][1]})
        for i, p in enumerate(K.PAIR_NAMES):
            msk = pi == i
            for lab, v in (("bar/per_pair_gain/" + p, fg[msk] - comp_gain[msk]),
                           (f"per_pair/{p}/counterpart_vs_B_r1", mz[f"{cfg}__cf__r1"][msk] - mz["B__r1"][msk]),
                           (f"per_pair/{p}/bar_margin_r1", mz[f"{cfg}__bar_v"][msk])):
                r = K.point_ci(v, cl[msk])
                extra.update({f"{lab}/point": r["point"], f"{lab}/ci95[0]": r["ci95"][0], f"{lab}/ci95[1]": r["ci95"][1]})
        for k, v in extra.items():
            cmp_pairs(rows, f"{name}/{k}", st.get(k, "MISSING"), v)
        keymap_extra = set(extra)
        missing = [k for k in st if k not in keymap and k not in keymap_extra
                   and not k.startswith(("pick_share/", "picks/", "provenance", "sigma/"))
                   and k not in ("name", "config", "n_episodes", "n_clusters", "smoke") and not k.endswith("n_clusters")
                   and not k.startswith("groupings")]
        if missing:
            rows.append((f"{name}/UNCOMPARED_KEYS", ";".join(missing), "", float("nan")))
        # arrays
        sz = np.load(K.RES / f"cand_{name}.npz")
        pairs = [(f"fused__{m}", f"{cfg}__fused__{m}") for m in METRICS] + \
                [(f"cf__{m}", f"{cfg}__cf__{m}") for m in METRICS] + [("bar_v", f"{cfg}__bar_v")] + \
                [(f"pick__{c}", f"{cfg}__pick__{c}") for c in "ab"] + [(f"margin__{c}", f"{cfg}__margin__{c}") for c in "ab"] + \
                [(f"T__{c}__{d}", f"{cfg}__T__{c}__{d}") for c in "ab" for d in ("i2t", "t2i")]
        for sk, mk in pairs:
            a, b = np.asarray(sz[sk]), np.asarray(mz[mk])
            eq = a.shape == b.shape and np.array_equal(a, b)
            mx = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape else float("nan")
            arr_rows.append((f"{name}/{sk}", bool(eq), mx, str(a.dtype), str(b.dtype)))

    sm = flat(json.loads((K.RES / "ra_summary.json").read_text()))
    for k in ("fused_r1", "bar_r1", "margin_r1"):
        for q in ("point", "ci95[0]", "ci95[1]"):
            cmp_pairs(rows, f"summary/A1_minus_A0/{k}/{q}", sm[f"A1_minus_A0_under_Ra/{k}/{q}"],
                      flat(mine["A1_minus_A0_under_Ra"])[f"{k}/{q}"])
    mAR = flat(mine["candidates"]["Ra_AR"])
    for k, v in sm.items():
        if k.startswith("ar_check/reader_Ra/"):
            cmp_pairs(rows, f"summary/{k}", v, mAR.get("pick_share/" + k[len("ar_check/reader_Ra/"):], "MISSING"))
        elif k.startswith("ar_check/reader_step1_argmax/"):
            cmp_pairs(rows, f"summary/{k}", v,
                      mAR.get("ar_check_step1_argmax_shares/" + k[len("ar_check/reader_step1_argmax/"):], "MISSING"))
    for q in ("point", "ci95[0]", "ci95[1]"):
        cmp_pairs(rows, f"summary/ar_check/Ra_AR/margin_r1/{q}", sm[f"ar_check/Ra_AR/margin_r1/{q}"], mAR[f"margin/r1/{q}"])
        cmp_pairs(rows, f"summary/ar_check/Ra_AR/bar_margin_r1/{q}", sm[f"ar_check/Ra_AR/bar_margin_r1/{q}"],
                  mAR[f"bar/r1/{q}"])
        cmp_pairs(rows, f"summary/ar_check/step1_argmax_AR/margin_r1/{q}", sm[f"ar_check/step1_argmax_AR/margin_r1/{q}"],
                  mAR[f"vs_step1_argmax/step1_margin_r1/{q}"])
        cmp_pairs(rows, f"summary/ar_check/step1_argmax_AR/bar_margin_r1/{q}",
                  sm[f"ar_check/step1_argmax_AR/bar_margin_r1/{q}"], mAR[f"vs_step1_argmax/step1_bar_r1/{q}"])
    for cfg in ("A1", "A0", "AR"):
        mi = flat(mine["candidates"][f"Ra_{cfg}"])
        for q in ("point", "ci95[0]", "ci95[1]"):
            cmp_pairs(rows, f"summary/candidates/Ra_{cfg}/bar_margin/{q}", sm[f"candidates/Ra_{cfg}/bar_margin/{q}"],
                      mi[f"bar/r1/{q}"])
            cmp_pairs(rows, f"summary/candidates/Ra_{cfg}/gain_statistic/{q}",
                      sm[f"candidates/Ra_{cfg}/gain_statistic/{q}"], mi[f"gain_statistic/{q}"])
    for h in K.GROUPINGS:
        cmp_pairs(rows, f"summary/sigma/{h} (ours f32)", sm[f"sigma/{h}"], mine["sigma_f32_agreements"][h])

    nonzero = [r for r in rows if not (r[3] == 0.0)]
    bad_arr = [r for r in arr_rows if not r[1]]
    out = {"n_scalar_compared": len(rows), "n_scalar_nonzero_or_unmatched": len(nonzero),
           "nonzero": [list(map(str, r)) for r in nonzero], "n_arrays": len(arr_rows),
           "arrays_not_identical": [list(map(str, r)) for r in bad_arr], "checks": mine["checks"],
           "picks_f32_vs_f64": {k: v["picks_f32_vs_f64_delta_disagree"] for k, v in mine["candidates"].items()},
           "step1_argmax_picks_reproduced": {k: v["step1_argmax_picks_reproduced"] for k, v in mine["candidates"].items()}}
    K.save_json(K.OUT / "rd_compare_ra.json", out)
    print(f"scalars compared {len(rows)}; nonzero/unmatched {len(nonzero)}")
    for r in nonzero:
        print("  ", r)
    print(f"arrays compared {len(arr_rows)}; not identical {len(bad_arr)}")
    for r in bad_arr:
        print("  ", r)
    print("checks", mine["checks"])
    print("picks f32 vs f64", out["picks_f32_vs_f64"], "step1 picks reproduced", out["step1_argmax_picks_reproduced"])


if __name__ == "__main__":
    main()
