"""Mappings for the runner files written after the picks: regression_seed42.json, seed42_pass_counts.json,
seed42_per_episode.npz, sensitivity_seed42.json. Each comparison is added only when its file exists; my side comes
from rd_seed42.json, rd_stats.json, rd_episodes.json and out/rd_arrays_seed42.npz."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_common as R  # noqa: E402
import numpy as np  # noqa: E402

RES = R.F / "results"
NAMES = {"B0": "B_prime", "aff_cf": "counterpart", "r1_cf": "counterpart", "B": "B"}
CHECKS = {"P1": ("r1", "aff_fused", "cosine"), "P2": ("r1", "aff_fused", "rca"), "P3": ("r1", "aff_fused", "B"),
          "P4": ("r1", "aff_fused", "B0"), "P5": ("r1", "aff_fused", "aff_cf"), "P6": ("gain", "aff_fused", "aff_cf"),
          "P7": ("gain", "aff_fused", "rca"), "S1": ("r1", "aff_fused", "B1"), "S2": ("r1", "aff_fused", "r1_fused")}


def bar(z, fused, cf):
    means = {k: float(np.mean(z[f"{k}__r1"])) for k in ("B0", cf, "B")}
    comp = "B0"
    for k in (cf, "B"):
        if means[k] > means[comp]:
            comp = k
    return comp


def compare(add, pending, s42, st):
    z = np.load(R.OUT / "rd_arrays_seed42.npz")
    eps = json.loads((R.RD / "rd_episodes.json").read_text())
    reg_mine = s42["regression"]

    p = RES / "regression_seed42.json"
    if p.exists():
        r = json.loads(p.read_text())
        it = r["items"]
        add("regression/aff_fused_r1", reg_mine["aff_fused_r1"], it["aff_fused_r1"]["got"], "tol")
        add("regression/aff_cf_r1", reg_mine["cf_r1"], it["aff_cf_r1"]["got"], "tol")
        add("regression/aff_bar_comparator", NAMES[reg_mine["bar_comparator"]], it["aff_bar_comparator"]["got"])
        for mine_k, their_k in (("bar_margin", "aff_bar_margin"), ("gain_statistic", "aff_gain_statistic"),
                                ("aff_minus_r1", "aff_minus_r1_fused"), ("r1_margin_vs_cf", "r1_margin_vs_counterpart"),
                                ("r1_gain_statistic", "r1_gain_statistic"), ("aff_minus_b1", "aff_minus_b1")):
            mine = [reg_mine[mine_k]["point"], *reg_mine[mine_k]["ci95"]]
            for lab, a, b in zip(("point", "lo95", "hi95"), mine, it[their_k]["got"]):
                add(f"regression/{their_k}/{lab}", a, b, "tol")
        add("regression/r1_bar_comparator", NAMES[bar(z, "r1_fused", "r1_cf")], it["r1_bar_comparator"]["got"])
        add("regression/r1_fused_r1", 100 * float(np.mean(z["r1_fused__r1"])), it["r1_fused_r1"]["got"], "tol")
        add("regression/r1_cf_r1", 100 * float(np.mean(z["r1_cf__r1"])), it["r1_cf_r1"]["got"], "tol")
        add("regression/aff_quarter_hits", int(np.rint(4 * z["aff_fused__r1"]).sum()), it["aff_quarter_hits"]["got"])
        for b in ("B", "B0", "B1"):
            add(f"regression/{b}_mean_r1", s42["B_picks"][b]["mean_r1"], it[f"{b}_mean_r1"]["got"], "tol")
            for h in ("0", "1"):
                add(f"regression/picks/{b}/{h}", s42["B_picks"][b]["picks"][h], [float(x) for x in r["picks"][b][h]])
        for s in ("42", "9001", "9002", "9003"):
            for pair, sha in eps["seeds"][s]["sha256"].items():
                add(f"regression/episodes_seed{s}__{pair}", sha, it[f"episodes_seed{s}__{pair}"]["got"])
                if s == "42":
                    add(f"regression/bundle_episodes_seed42__{pair}", sha, it[f"bundle_episodes_seed42__{pair}"]["got"])
        for name, v in s42["lambda"].items():
            for h in ("0", "1"):
                theirs = r["lambdas"][name][h]
                add(f"regression/lambdas/{name}/{h}", v["recorded"][h],
                    float("inf") if theirs == "inf" else float(theirs))
        for reader in ("aff", "r1"):
            for kind in ("fused", "cf"):
                cells = s42["frozen_cells"][f"{reader}_{kind}"]
                add(f"regression/cells/{reader}/{kind}", [cells["0"]["cell"], cells["1"]["cell"]],
                    r["cells"][reader][kind])
        add("regression/fit_rows_sha256", s42["fit_rows_sha256"], r["fit_rows_sha256"])
        # array verdicts: the runner's 'equal' against the stored files vs mine against the same files
        mine_checks = dict(s42["array_checks"])
        r3 = np.load(R.R3DIR / "results/seed42_arrays.npz")
        mine_checks["seed42_arrays/aff_bar_v"] = mine_checks.pop("bar_v/aff")
        comp = bar(z, "r1_fused", "r1_cf")
        mine_checks["seed42_arrays/r1_bar_v"] = bool(np.array_equal(z["r1_fused__r1"] - z[f"{comp}__r1"],
                                                                    r3["r1_bar_v"]))
        for k, v in it.items():
            if k.startswith(("seed42_arrays/", "per_anchor_seed42/")):
                if k in mine_checks:
                    add(f"regression/array_verdict/{k}", mine_checks[k], v["equal"])
                else:
                    add(f"regression/array_verdict/{k} (not in my checks)", None, v["equal"])
        add("regression/heads_selection_bits_equal_stored",
            json.loads((R.RD / "rd_heads.json").read_text())["all_bitwise_equal"],
            it["heads_selection_bits_equal_stored"]["got"])
    else:
        pending.append(p.name)

    p = RES / "seed42_pass_counts.json"
    if p.exists():
        r = json.loads(p.read_text())
        add("pass_counts/n_episodes", st["n_episodes"], r["n_episodes"])
        add("pass_counts/n_clusters", st["n_clusters"], r["n_clusters"])
        add("pass_counts/holm_order_P", st["holm_order_P"], r["holm_order"])
        for j in CHECKS:
            mine = st["checks"][j]
            theirs = r["checks"][j] if j.startswith("P") else r["secondary"][j]
            add(f"pass_counts/{j}/n_j", mine["n_j"], theirs["n"])
            add(f"pass_counts/{j}/point", mine["point"], theirs["point"], "tol")
            for lab, a, b in zip(("lo95", "hi95"), mine["ci95"], theirs["ci95"]):
                add(f"pass_counts/{j}/{lab}", a, b, "tol")
            add(f"pass_counts/{j}/holm_k", mine["holm_k"], theirs["holm_k"])
            add(f"pass_counts/{j}/level_two_sided", mine["holm_level_two_sided"], theirs["level_two_sided"], "tol")
            for lab, a, b in zip(("lo_holm", "hi_holm"), mine["ci_holm"], theirs["ci_holm"]):
                add(f"pass_counts/{j}/{lab}", a, b, "tol")
            add(f"pass_counts/{j}/near_boundary", mine["within_one_of_boundary"], theirs["near_boundary"])
            if "passes" in theirs:
                add(f"pass_counts/{j}/pass", mine["pass"], theirs["passes"])
            if "own_count_passes" in theirs:
                add(f"pass_counts/{j}/own_count_passes", mine["own_count_passes"], theirs["own_count_passes"])
    else:
        pending.append(p.name)

    p = RES / "seed42_per_episode.npz"
    if p.exists():
        r = np.load(p)
        add("per_episode/cl", z["cl"], r["cl"], "array")
        add("per_episode/pair_index", z["pair_index"], r["pair_index"], "array")
        for j, (metric, x, y) in CHECKS.items():
            add(f"per_episode/diff__{j}", z[f"{x}__{metric}"] - z[f"{y}__{metric}"], r[f"diff__{j}"], "array")
    else:
        pending.append(p.name)

    p = RES / "sensitivity_seed42.json"
    if p.exists():
        r = json.loads(p.read_text())
        add("sensitivity/N", st["n_episodes"], r["N"])
        add("sensitivity/n_paintings", st["sensitivity"]["P1"]["P"], r["n_paintings"])
        for j in CHECKS:
            add(f"sensitivity/{j}/sigma_a2", st["sensitivity"][j]["sigma_a2"], r[j]["sigma_a2"], "rel")
            add(f"sensitivity/{j}/sigma_eps2", st["sensitivity"][j]["sigma_e2"], r[j]["sigma_eps2"], "rel")
    else:
        pending.append(p.name)
