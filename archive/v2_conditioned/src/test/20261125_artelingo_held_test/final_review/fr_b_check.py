"""Final review B: my own comparisons on what the r6 runners wrote and computed (no r6 module imported).

Inputs: argv[1] = temp dir holding capture.npz (fr_b_regression_capture.py), r3.npz and r3.json
(fr_b_r3_independent.py); the snapshot's F/results/*.json|npz written by the runners. Writes argv[1]/check.json and
prints one line per check group.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path("/project/CoSiR")
TEST = MAIN / "src/test"
sys.path.insert(0, str(MAIN))
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402

RES = Path("/project/CoSiR-r6-fr/src/test/20261125_artelingo_held_test/results")
TMP = Path(sys.argv[1])
METRICS = ("r1", "gain", "other", "swap", "strict")
PM = ("diag", "diag_relu", "bilinear", "kissme", "xing", "wang", "probe", "tip", "value_prototype")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return a.tobytes() == b.tobytes()            # bit for bit (NaN-safe)
    return bool(np.array_equal(a, b))


def pci(v, cl):
    r = cluster_bootstrap(np.asarray(v, np.float64), cl)
    return [100 * r["point"], 100 * r["ci95"][0], 100 * r["ci95"][1]]


def main():
    out = {}
    cap = np.load(TMP / "capture.npz")
    meta = json.loads(str(cap["meta"]))
    r3 = np.load(TMP / "r3.npz")
    r3s = json.loads((TMP / "r3.json").read_text())

    # ---- section 6.2: refit posteriors on selection rows vs the stored arrays, bit for bit
    refit = {}
    with np.load(TEST / "20261108_new_method_quick_checks/results/n6_posteriors.npz") as z:
        assert np.array_equal(z["selection"], cap["selection"])
        for stored, mine in (("affect", "affect_km"), ("image", "image"), ("caption", "caption")):
            for m in ("img", "txt"):
                refit[f"n6/{stored}__{m}"] = same(cap[f"selpost__{mine}__{m}"], z[f"{stored}__{m}"])
    with np.load(TEST / "20261116_grouping_step1_style/results/step1_heads_style.npz") as z:
        assert np.array_equal(z["selection"], cap["selection"])
        for m in ("img", "txt"):
            refit[f"step1/style_csd__{m}"] = same(cap[f"selpost__csd__{m}"], z[f"style_csd__{m}"])
    for m in ("img", "txt"):     # D1 affect: r6 refit vs round 3's own fit_one_head refit (no stored array exists)
        refit[f"r3_fit_one_head/affect__{m}"] = same(cap[f"selpost__affect__{m}"], r3[f"r3post__affect__{m}"])
        for h in ("image", "caption"):
            refit[f"r3_bundle_post/{h}__{m}"] = same(cap[f"selpost__{h}__{m}"], r3[f"r3post__{h}__{m}"])
    rc = json.loads((RES / "refit_check.json").read_text())
    refit["coef_sha_capture_eq_refit_check"] = meta["coef_sha256"] == rc["coef_sha256"]
    refit["runner_items_all_equal"] = all(v["equal"] for v in rc["items"].values()) and rc["passed"] is True
    out["refit"] = refit

    # ---- section 6.3: B, B0, B1
    tg = {"B": 18.341064453125, "B0": 18.436686197916664, "B1": 18.804931640625}
    pk = json.loads((RES / "picks_seed42.json").read_text())
    nb = {}
    for k, t in tg.items():
        mine = 100 * float(np.mean(cap[f"scored__{k}__r1"]))
        nb[f"{k}_r6_scored_mean_eq_target"] = mine == t
        nb[f"{k}_picks_json_eq_target"] = pk["mean_r1"][k] == t
        nb[f"{k}_r3_independent_eq_target"] = r3s[f"{k}_mean"] == t
        nb[f"{k}_r6_frozen_arrays_eq_r3_crossfit"] = all(same(cap[f"scored__{k}__{m}"], r3[f"{k}__{m}"]) for m in METRICS)
        nb[f"{k}_picks_eq_r3_crossfit_picks"] = ({h: [float(x) for x in v] for h, v in pk[k].items()}
                                                 == r3s[f"{k}_picks"])
    out["nested"] = nb

    # ---- bundle on selection: r6 vs round 3
    bd = {}
    for k in [k for k in r3.files if k.startswith("bundle__")]:
        bd[k] = same(cap[k], r3[k])
    for k in ("cl", "pair_index", "anchor"):
        bd[k] = same(cap[f"bundle__{k}"], r3[k])
    for k in [k for k in r3.files if k.startswith("ep__")]:
        bd[k] = same(cap[k], r3[k])
    out["bundle_vs_r3"] = bd

    # ---- section 6.4: per-anchor arrays vs round 3's seed42_arrays.npz and AB's per_anchor_seed42.npz
    p3 = TEST / "20261121_round3_affect_gate/results/seed42_arrays.npz"
    assert sha(p3) == "5ea4b09a4161a5eac6ca78942cf0f4b99b9c634edca651fba4c2689c7c24ab8a"
    reg = json.loads((RES / "regression_seed42.json").read_text())
    arr, disagree = {}, []
    with np.load(p3) as z:
        for who in ("aff", "r1"):
            for part in ("fused", "cf"):
                for m in METRICS:
                    k = f"{who}_{part}__{m}"
                    arr[f"seed42_arrays/{k}"] = same(cap[f"scored__{k}"], z[k])
                    arr[f"r3_score_frozen/{k}"] = same(cap[f"scored__{k}"], r3[k])
            for c in ("a", "b"):
                arr[f"seed42_arrays/{who}_gate__{c}"] = same(cap[f"scored__gates__{who}__{c}"], z[f"{who}_gate__{c}"])
        for k in ("cl", "pair_index", "parity"):
            arr[f"seed42_arrays/{k}"] = same(cap[f"bundle__{k}"], z[k])
        for c in ("a", "b"):
            arr[f"seed42_arrays/pick__{c}"] = same(cap[f"scored__reader__pick__{c}"].astype(np.int64), z[f"pick__{c}"])
            arr[f"seed42_arrays/margin__{c}"] = same(cap[f"scored__reader__m__{c}"], z[f"margin__{c}"])
            arr[f"seed42_arrays/P__{c}"] = same(cap[f"scored__reader__P__{c}"], z[f"P__{c}"])
    pa = TEST / "20261030_aspect_baselines/results/per_anchor_seed42.npz"
    assert sha(pa).startswith("a4818ba0")
    with np.load(pa) as z:
        for s in ("cosine", "rca") + PM:
            for m in METRICS:
                arr[f"per_anchor_seed42/{s}__{m}"] = same(cap[f"scored__{s}__{m}"], z[f"{s}__{m}"])
        arr["per_anchor_seed42/anchor_group"] = same(cap["bundle__cl"], z["anchor_group"])
        arr["per_anchor_seed42/pair_index"] = same(cap["bundle__pair_index"], z["pair_index"])
    for k, v in arr.items():
        if k in reg["items"] and reg["items"][k]["equal"] != v:
            disagree.append(k)
    names_rule = [f"seed42_arrays/{w}_{p}__{m}" for w in ("aff", "r1") for p in ("fused", "cf") for m in METRICS] + \
        [f"seed42_arrays/{w}_gate__{c}" for w in ("aff", "r1") for c in ("a", "b")] + \
        [f"per_anchor_seed42/{s}__{m}" for s in ("cosine", "rca") + PM for m in METRICS]
    out["arrays"] = {"n": len(arr), "n_equal": sum(arr.values()), "unequal": [k for k, v in arr.items() if not v],
                     "rule_keys_in_runner_record": all(k in reg["items"] for k in names_rule),
                     "runner_vs_mine_disagree": disagree}

    # ---- section 6.4 scalars, my own code from the per-anchor arrays (fractions; x100 after the percentile)
    cl = cap["bundle__cl"]
    g = lambda k: cap[f"scored__{k}"].astype(np.float64)  # noqa: E731
    means = {"B_prime": g("B0__r1").mean(), "counterpart": g("aff_cf__r1").mean(), "B": g("B__r1").mean()}
    order = ["B_prime", "counterpart", "B"]
    comp = order[0]
    for nm in order[1:]:
        if means[nm] > means[comp]:
            comp = nm
    comp_key = {"B_prime": "B0", "counterpart": "aff_cf", "B": "B"}[comp]
    r1m = {"B_prime": g("B0__r1").mean(), "counterpart": g("r1_cf__r1").mean(), "B": g("B__r1").mean()}
    comp_r1 = order[0]
    for nm in order[1:]:
        if r1m[nm] > r1m[comp_r1]:
            comp_r1 = nm
    mine = {
        "aff_fused_r1": 100 * float(np.mean(g("aff_fused__r1"))),
        "aff_cf_r1": 100 * float(np.mean(g("aff_cf__r1"))),
        "aff_bar_comparator": comp,
        "aff_bar_margin": pci(g("aff_fused__r1") - g(f"{comp_key}__r1"), cl),
        "aff_gain_statistic": pci(g("aff_fused__gain") - g("aff_cf__gain"), cl),
        "aff_minus_r1_fused": pci(g("aff_fused__r1") - g("r1_fused__r1"), cl),
        "r1_bar_comparator": comp_r1,
        "r1_margin_vs_counterpart": pci(g("r1_fused__r1") - g("r1_cf__r1"), cl),
        "r1_gain_statistic": pci(g("r1_fused__gain") - g("r1_cf__gain"), cl),
        "aff_minus_b1": pci(g("aff_fused__r1") - g("B1__r1"), cl),
        "aff_quarter_hits": int(np.rint(4 * g("aff_fused__r1")).astype(np.int64).sum()),
    }
    rule = {"aff_fused_r1": 19.136555989583336, "aff_cf_r1": 18.39599609375, "aff_bar_comparator": "B_prime",
            "aff_bar_margin": [0.6998697916666667, 0.4598852740816973, 0.9371680126852968],
            "aff_gain_statistic": [3.110758463541667, 2.780005709854805, 3.4559584315470384],
            "aff_minus_r1_fused": [0.21769205729166666, 0.06425880757348419, 0.3709597330984391],
            "r1_bar_comparator": "counterpart",
            "r1_margin_vs_counterpart": [0.4435221354166667, 0.21646171563312194, 0.6735669710776852],
            "r1_gain_statistic": [2.667236328125, 2.325087836946873, 3.012361650695922],
            "aff_minus_b1": [0.33162434895833337, 0.048231414333532084, 0.6246158772581268],
            "aff_quarter_hits": 9406}
    sc = {}
    for k, want in rule.items():
        sc[k] = {"mine_eq_rule": mine[k] == want, "runner_got_eq_mine": reg["items"][k]["got"] == mine[k],
                 "runner_equal_flag": reg["items"][k]["equal"],
                 "max_abs_vs_rule": (float(np.max(np.abs(np.subtract(mine[k], want))))
                                     if not isinstance(want, str) else None)}
    out["scalars"] = sc
    out["regression_record"] = {"passed": reg["passed"], "n_items": reg["n_items"], "n_failed": reg["n_failed"]}

    # ---- per-episode diffs (the sensitivity input) vs mine; section 6.5 sigma split, my own code
    defs = {"P1": ("r1", "cosine"), "P2": ("r1", "rca"), "P3": ("r1", "B"), "P4": ("r1", "B0"), "P5": ("r1", "aff_cf"),
            "P6": ("gain", "aff_cf"), "P7": ("gain", "rca"), "S1": ("r1", "B1"), "S2": ("r1", "r1_fused")}
    pe = np.load(RES / "seed42_per_episode.npz")
    sens = json.loads((RES / "sensitivity_seed42.json").read_text())
    sg = {}
    _, idx = np.unique(cl, return_inverse=True)
    P = idx.max() + 1
    mcount = np.bincount(idx).astype(np.float64)
    n = len(cl)
    for c, (met, other) in defs.items():
        d = g(f"aff_fused__{met}") - g(f"{other}__{met}")
        eq_file = same(d, pe[f"diff__{c}"])
        x = 100 * d
        mp = np.bincount(idx, weights=x) / mcount
        msw = float(np.sum((x - mp[idx]) ** 2)) / (n - P)
        msb = float(np.sum(mcount * (mp - x.mean()) ** 2)) / (P - 1)
        n0 = (n - float(np.sum(mcount ** 2)) / n) / (P - 1)
        sa = max(0.0, (msb - msw) / n0)
        got = sens[c]
        sg[c] = {"diff_eq_file": eq_file, "sigma_a2_rel": abs(got["sigma_a2"] - sa) / max(abs(sa), 1e-300),
                 "sigma_eps2_rel": abs(got["sigma_eps2"] - msw) / msw}
    sg["N"] = sens["N"] == n
    sg["n_paintings"] = sens["n_paintings"] == int(P)
    out["sensitivity"] = sg
    out["pe_cl_eq"] = same(pe["cl"], cl)

    # ---- episodes: original builder (r3 script) vs the stored hashes, and the runner's identity items
    ep = r3s["episodes"]
    out["episodes"] = {"original_builder_equal": sum(v["equal"] for v in ep.values()), "n": len(ep),
                       "runner_items_equal": all(reg["items"][f"episodes_seed{k.split('__')[0]}__{k.split('__', 1)[1]}"]["equal"]
                                                 for k in ep),
                       "runner_sha_eq_mine": all(reg["items"][f"episodes_seed{k.split('__')[0]}__{k.split('__', 1)[1]}"]["got"]
                                                 == v["got"] for k, v in ep.items()),
                       "value_sets_eq": meta["value_sets"] == r3s["dev_values"], "dev_counts": r3s["dev_counts"]}
    out["split"] = r3s["split"]
    (TMP / "check.json").write_text(json.dumps(out, indent=1, default=str))
    print("refit", sum(refit.values()), "/", len(refit))
    print("nested", sum(nb.values()), "/", len(nb))
    print("bundle_vs_r3", sum(bd.values()), "/", len(bd))
    print("arrays", out["arrays"]["n_equal"], "/", out["arrays"]["n"], "disagree", disagree,
          "rule keys in record", out["arrays"]["rule_keys_in_runner_record"])
    print("scalars mine==rule", sum(v["mine_eq_rule"] for v in sc.values()), "/", len(sc),
          "runner==mine", sum(v["runner_got_eq_mine"] for v in sc.values()))
    print("sensitivity max rel", max(max(v["sigma_a2_rel"], v["sigma_eps2_rel"]) for k, v in sg.items()
                                     if isinstance(v, dict)),
          "diffs eq file", all(v["diff_eq_file"] for v in sg.values() if isinstance(v, dict)), sg["N"], sg["n_paintings"])
    print("episodes", out["episodes"])


if __name__ == "__main__":
    main()
