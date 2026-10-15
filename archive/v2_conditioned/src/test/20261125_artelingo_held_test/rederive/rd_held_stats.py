"""Phase 2, step 3: the pooled checks on held seeds 52, 53, 54 (rule §3, §4) and the sensitivity of rule §8.2, own
code; then phase2_results.json (every phase-2 quantity, with the time) and its SHA-256.

P1 to P7 and S1, S2 from out/rd_held_arrays.npz (36,864 pooled episodes; clusters = anchor paintings): n_j from the
integer 4·Σ cluster sums of the shared 5,000 draws, p_j, point, 95% and Holm-level intervals, Holm order (P: m = 7;
S: m = 2, computed though read only after a GO), pass or fail, boundary flags (rule §8.5). Sensitivity: SE² =
(σ_a²·Σ_p M_p² + σ_ε²·N) / N² with σ² from phase 1's seed-42 split (rd_stats.json, pp units); x = 3.532·SE,
x₉₅ = 2.80·SE (P), x₂ = 3.083·SE, x₉₅ (S).

  cd /project/CoSiR && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/rederive/rd_held_stats.py \
  > src/test/20261125_artelingo_held_test/rederive/out/rd_held_stats.log 2>&1
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
FAMILIES = (("P", ["P1", "P2", "P3", "P4", "P5", "P6", "P7"], 7), ("S", ["S1", "S2"], 2))
N_EXPECTED = 36_864


def main():
    z = np.load(R.OUT / "rd_held_arrays.npz")
    cl = z["cl"]
    if len(cl) != N_EXPECTED:
        raise AssertionError("pooled episode count")
    for k in ("cosine", "B", "B0", "B1", "aff_cf"):
        if not (z[f"{k}__gain"] == 0).all():
            raise AssertionError(f"{k}: condition gain not 0")
    draws = R.Draws(cl)
    out = {"n_episodes": int(len(cl)), "n_clusters": draws.k, "checks": {}}
    boots_of = {}
    for j, (metric, x, y) in CHECKS.items():
        d = z[f"{x}__{metric}"] - z[f"{y}__{metric}"]
        boots = draws.boots(d)
        ints = draws.int_sums(d)
        n_j = int((ints <= 0).sum())
        n_float = int((boots <= 0).sum())
        ref = cluster_bootstrap(d, cl)
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        if ci != ref["ci95"] or ref["n_clusters"] != draws.k:
            raise AssertionError(f"{j}: own draws differ from cluster_bootstrap")
        boots_of[j] = boots
        out["checks"][j] = {"quantity": f"{metric}: {x} - {y}", "n_j": n_j, "n_j_float_form": n_float,
                            "p_j": (n_j + 1) / 5001, "point": 100 * float(d.mean()), "ci95": [100 * c for c in ci],
                            "int_sum_min": int(ints.min()), "int_sum_max": int(ints.max()),
                            "quarter_hit_sum": int(np.rint(4 * d).sum())}
    for fam, ids, m in FAMILIES:
        order, passed, level = R.holm([out["checks"][j]["n_j"] for j in ids], m)
        for pos, j in enumerate(ids):
            c = out["checks"][j]
            k = level[pos]["k"]
            a = 0.025 / (m + 1 - k)
            b = boots_of[j]
            c["holm_k"] = k
            c["holm_level_two_sided"] = 1 - 0.05 / (m + 1 - k)
            c["ci_holm"] = [100 * float(np.percentile(b, 100 * 0.025 / (m + 1 - k))),
                            100 * float(np.percentile(b, 100 * (1 - a)))]
            c["own_count_passes"] = level[pos]["own_count_passes"]
            c["pass"] = bool(passed[pos])
            nstar = 5001 // (40 * (m + 1 - k)) - 1
            c["n_star_k"] = nstar
            c["within_one_of_boundary"] = abs(c["n_j"] - nstar) <= 1
            c["not_reached"] = bool(c["own_count_passes"] and not c["pass"])
        out[f"holm_order_{fam}"] = [ids[i] for i in order]
    out["all_P_pass"] = all(out["checks"][j]["pass"] for j in FAMILIES[0][1])
    out["S_tested"] = out["all_P_pass"]
    out["boundary_cases"] = [j for j, c in out["checks"].items() if c["within_one_of_boundary"]]

    # sensitivity (rule §8.2)
    st42 = json.loads((R.RD / "rd_stats.json").read_text())["sensitivity"]
    _, inv = np.unique(cl, return_inverse=True)
    M = np.bincount(inv).astype(np.float64)
    N = float(len(cl))
    sum_m2 = float(np.sum(M ** 2))
    sens = {"N": int(N), "n_paintings": int(len(M)), "sum_Mp2": sum_m2, "checks": {}}
    for j in CHECKS:
        sa, se = st42[j]["sigma_a2"], st42[j]["sigma_e2"]
        SE = float(np.sqrt((sa * sum_m2 + se * N) / N ** 2))
        row = {"sigma_a2": sa, "sigma_eps2": se, "SE": SE, "x95": 2.80 * SE}
        if j.startswith("P"):
            row["x"] = 3.532 * SE
        else:
            row["x2"] = 3.083 * SE
        sens["checks"][j] = row
    out["sensitivity"] = sens
    out["time"] = R.now_ams()
    R.write_json(R.RD / "rd_held_stats.json", out)

    # phase2_results.json
    eps = json.loads((R.RD / "rd_held_episodes.json").read_text())
    sc = json.loads((R.RD / "rd_held_scores.json").read_text())
    res = {"what": "C13 phase-2 re-derivation of round 6 (rule §8.4): held seeds 52, 53, 54, own code; written "
                   "before any runner held output was opened",
           "episodes": eps, "scores": sc, "stats": out, "time": R.now_ams()}
    path = R.RD / "phase2_results.json"
    if path.exists():
        raise SystemExit(f"{path} exists; not overwritten")
    R.write_json(path, res)
    sha = R.sha_file(path)
    R.log(f"Holm order P {out['holm_order_P']}; all P pass {out['all_P_pass']}; S order {out['holm_order_S']}; "
          f"boundary cases {out['boundary_cases']}")
    print(f"phase2_results.json SHA-256 {sha}", flush=True)


if __name__ == "__main__":
    main()
