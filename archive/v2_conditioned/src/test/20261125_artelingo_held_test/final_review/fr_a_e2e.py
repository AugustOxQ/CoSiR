"""Final review A: the held read end to end on the synthetic real-shaped env of test_r6_held_synthetic.py (its own
fixtures: run_r6_held.run_held with the real RowContext, build_bundle_r6, score_seed, pass_record), then held_pass.json
and sensitivity_held.json re-derived with my own code from held_arrays.npz and the rule's text (section 3, 8.2, 8.3).

    pytest -q -p no:cacheprovider final_review/fr_a_e2e.py   (collected because it is named explicitly)
"""
import json
import sys
from pathlib import Path

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r6_common as R  # noqa: E402
import r6_score as S  # noqa: E402
import r6_stats as ST  # noqa: E402
import test_r6_held as TH  # noqa: E402
from test_r6_held_synthetic import env, read  # noqa: E402,F401  (fixtures)
from fr_a_stats import my_draws, my_holm  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import METRICS  # noqa: E402

COMP = {"P1": ("r1", "cosine"), "P2": ("r1", "rca"), "P3": ("r1", "B"), "P4": ("r1", "B0"), "P5": ("r1", "aff_cf"),
        "P6": ("gain", "aff_cf"), "P7": ("gain", "rca"), "S1": ("r1", "B1"), "S2": ("r1", "r1_fused")}
OUT = Path(__file__).with_name("fr_a_e2e.json")


def test_held_pass_rederived(read):
    assert read.code == 0
    rec = json.loads((read.res / "held_pass.json").read_text())
    sens = json.loads((read.res / "sensitivity_held.json").read_text())
    with np.load(read.res / "held_arrays.npz") as z:
        arr = {k: z[k] for k in z.files}
    want_keys = {f"{s}__{m}" for s in S.CORE_SCORERS for m in METRICS} | {"cl", "pair_index", "seed_index"}
    report = {"arrays_keys_ok": set(arr) == want_keys, "extra_keys": sorted(set(arr) - want_keys),
              "missing_keys": sorted(want_keys - set(arr)), "core": list(S.CORE_SCORERS),
              "pass_top_keys": sorted(rec), "n_episodes": rec["n_episodes"]}
    cl = arr["cl"]
    assert len(cl) == 36864
    assert np.array_equal(arr["seed_index"], np.repeat([0, 1, 2], 12288))
    assert np.all(arr["aff_cf__gain"] == 0)
    mine, diffs = {}, []
    for c, (m, key) in COMP.items():
        d = arr[f"aff_fused__{m}"] - arr[f"{key}__{m}"]
        b, n_int, n_float, k = my_draws(d, cl)
        mine[c] = {"b": b, "n": n_int, "n_float": n_float, "k": k, "point": 100 * float(d.mean()),
                   "ci95": [100 * float(np.percentile(b, 2.5)), 100 * float(np.percentile(b, 97.5))]}
    for fam, names, m in (("checks", ST.CHECKS, 7), ("secondary", ST.SECONDARY, 2)):
        order, h = my_holm({c: mine[c]["n"] for c in names}, tuple(names), m)
        if fam == "checks" and rec["holm_order"] != order:
            diffs.append(("holm_order", rec["holm_order"], order))
        for c in names:
            g, w, kk = rec[fam][c], mine[c], h[c]["k"]
            want = {"n": w["n"], "point": w["point"], "ci95": w["ci95"], "holm_k": kk,
                    "ci_holm": [100 * float(np.percentile(w["b"], 100 * 0.025 / (m + 1 - kk))),
                                100 * float(np.percentile(w["b"], 100 * (1 - 0.025 / (m + 1 - kk))))],
                    "level_two_sided": 1 - 0.05 / (m + 1 - kk),
                    "near_boundary": abs(w["n"] - (5001 // (40 * (m + 1 - kk)) - 1)) <= 1}
            if fam == "checks":
                want.update(passes=h[c]["passes"], own_count_passes=h[c]["own"])
            for f, v in want.items():
                if g.get(f) != v:
                    diffs.append((c, f, g.get(f), v))
            if fam == "secondary" and "passes" in g:
                diffs.append((c, "pass field in secondary"))
    if rec["n_clusters"] != mine["P1"]["k"]:
        diffs.append(("n_clusters", rec["n_clusters"], mine["P1"]["k"]))
    # section 8.2 from the pooled anchors and the stand-in sigma of the seed-42 file
    _, M = np.unique(cl, return_counts=True)
    worst = 0.0
    for c in ST.CHECKS + ST.SECONDARY:
        sa, se_ = TH.SIGMA[c]["sigma_a2"], TH.SIGMA[c]["sigma_eps2"]
        se = float(np.sqrt((sa * float(np.sum(M.astype(np.float64) ** 2)) + se_ * 36864) / 36864 ** 2))
        xk, z = ("x", 3.532) if c in ST.CHECKS else ("x2", 3.083)
        for f, v in (("SE", se), (xk, z * se), ("x95", 2.80 * se), ("sigma_a2", sa), ("sigma_eps2", se_)):
            worst = max(worst, abs(sens[c][f] - v) / max(abs(v), 1e-300))
        if (xk == "x" and "x2" in sens[c]) or (xk == "x2" and "x" in sens[c]):
            diffs.append((c, "wrong margin key"))
    report.update({"field_diffs": diffs, "counts": {c: mine[c]["n"] for c in COMP},
                   "float_vs_int": {c: [mine[c]["n_float"], mine[c]["n"]] for c in COMP
                                    if mine[c]["n_float"] != mine[c]["n"]},
                   "sens_worst_rel": worst, "sens_N": sens["N"], "sens_paintings": sens["n_paintings"],
                   "paintings_mine": int(M.size), "sens_top_keys": sorted(sens)})
    OUT.write_text(json.dumps(report, indent=1, default=str))
    assert not diffs, diffs[:5]
    assert worst < 1e-12 and sens["N"] == 36864 and sens["n_paintings"] == M.size
    assert report["arrays_keys_ok"], report
