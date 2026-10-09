"""Phase 1, step 4: on seed 42 alone (selection rows, 12,288 episodes), the pass record of P1 to P7 and S1, S2 with the
rule's integer bootstrap counts and Holm (rule §3, §4), and the variance split of each check's per-episode difference
(rule §6 item 5, R3 rule §6.1). Reads out/rd_arrays_seed42.npz (rd_seed42.py). Writes rd_stats.json.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/rederive/rd_stats.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_common as R  # noqa: E402
import numpy as np  # noqa: E402

from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402

CHECKS = {"P1": ("r1", "aff_fused", "cosine"), "P2": ("r1", "aff_fused", "rca"), "P3": ("r1", "aff_fused", "B"),
          "P4": ("r1", "aff_fused", "B0"), "P5": ("r1", "aff_fused", "aff_cf"), "P6": ("gain", "aff_fused", "aff_cf"),
          "P7": ("gain", "aff_fused", "rca"), "S1": ("r1", "aff_fused", "B1"), "S2": ("r1", "aff_fused", "r1_fused")}
R3_NAMES = {"P1": "r1_vs_cosine", "P2": "r1_vs_rca", "P3": "r1_vs_B", "P4": "r1_vs_Bprime", "P5": "r1_vs_counterpart",
            "P6": "gain_statistic", "P7": "gain_vs_rca", "S2": "secondary"}


def main():
    z = np.load(R.OUT / "rd_arrays_seed42.npz")
    cl = z["cl"]
    for k in ("cosine", "B", "B0", "B1", "aff_cf"):
        if not (z[f"{k}__gain"] == 0).all():
            raise AssertionError(f"{k}: condition gain not 0")
    draws = R.Draws(cl)
    out = {"n_episodes": int(len(cl)), "n_clusters": draws.k, "checks": {}}
    for j, (metric, x, y) in CHECKS.items():
        d = z[f"{x}__{metric}"] - z[f"{y}__{metric}"]
        boots = draws.boots(d)
        ints = draws.int_sums(d)
        n_j = int((ints <= 0).sum())
        if n_j != int((boots <= 0).sum()):
            raise AssertionError(f"{j}: integer and float counts differ")
        ref = cluster_bootstrap(d, cl)
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        if ci != ref["ci95"]:
            raise AssertionError(f"{j}: own draws differ from cluster_bootstrap")
        out["checks"][j] = {"quantity": f"{metric}: {x} - {y}", "n_j": n_j, "p_j": (n_j + 1) / 5001,
                            "point": 100 * float(d.mean()), "ci95": [100 * c for c in ci],
                            "int_sum_min": int(ints.min()), "int_sum_max": int(ints.max()),
                            "_boots": boots}
    for fam, ids, m in (("P", ["P1", "P2", "P3", "P4", "P5", "P6", "P7"], 7), ("S", ["S1", "S2"], 2)):
        order, passed, level = R.holm([out["checks"][j]["n_j"] for j in ids], m)
        for pos, j in enumerate(ids):
            c = out["checks"][j]
            k = level[pos]["k"]
            a = 0.025 / (m + 1 - k)
            b = c["_boots"]
            c["holm_k"] = k
            c["holm_level_two_sided"] = 1 - 2 * a
            c["ci_holm"] = [100 * float(np.percentile(b, 100 * a)), 100 * float(np.percentile(b, 100 * (1 - a)))]
            c["own_count_passes"] = level[pos]["own_count_passes"]
            c["pass"] = bool(passed[pos])
            nstar = 5001 // (40 * (m + 1 - k)) - 1
            c["n_star_k"] = nstar
            c["within_one_of_boundary"] = abs(c["n_j"] - nstar) <= 1
        out[f"holm_order_{fam}"] = [ids[i] for i in order]
    out["all_P_pass_seed42"] = all(out["checks"][j]["pass"] for j in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"))
    out["S_tested_on_seed42"] = out["all_P_pass_seed42"]
    for c in out["checks"].values():
        c.pop("_boots")

    # sensitivity inputs (pp units, as R3's sensitivity.json)
    r3 = json.loads((R.R3DIR / "results/sensitivity.json").read_text())["checks"]
    sens, vs_r3 = {}, {}
    for j, (metric, x, y) in CHECKS.items():
        yv = 100 * (z[f"{x}__{metric}"] - z[f"{y}__{metric}"])
        sens[j] = R.variance_split(yv, cl)
        if j in R3_NAMES:
            ref = r3[R3_NAMES[j]]
            vs_r3[j] = {q: {"mine": sens[j][q], "round3": ref[q],
                            "rel_diff": abs(sens[j][q] - ref[q]) / max(abs(ref[q]), 1e-300)}
                        for q in ("sigma_a2", "sigma_e2", "n0")}
            vs_r3[j]["P_n_equal"] = sens[j]["P"] == ref["P"] and sens[j]["n"] == ref["n"]
    out["sensitivity"] = sens
    out["sensitivity_vs_round3"] = vs_r3
    out["time"] = R.now_ams()
    R.write_json(R.RD / "rd_stats.json", out)
    R.log(f"Holm P order {out['holm_order_P']}; all P pass {out['all_P_pass_seed42']}; "
          f"S order {out['holm_order_S']}")


if __name__ == "__main__":
    main()
