"""Print the report's markdown tables from results/D0..D6.json (no training)."""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
NAMES = ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
GATES = ["participation_ratio", "redundancy", "readout", "sparsity", "dead",
         "modality_private", "usage_concentration", "community_spanning", "pair_retrieval"]
res = {n: json.loads((RESULTS / f"{n}.json").read_text()) for n in NAMES}


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main() -> None:
    print("## Headline geometry (eval = val rows)")
    print(row(["ID", "PR img", "PR txt", "max abs r", "pairs >= 0.9", "PC1 img", "PC1 txt",
               "active img", "active txt"]))
    print(row(["---"] * 9))
    for n in NAMES:
        r, v = res[n], res[n]["values"]
        c = v["correlation"]
        print(row([n, f"{v['participation_ratio_img']:.4f}", f"{v['participation_ratio_txt']:.4f}",
                   f"{c['max_abs']:.5f}", f"{c['pairs_at_or_above']}/{c['pairs_total']}",
                   f"{r['pc1_share_img']:.4f}", f"{r['pc1_share_txt']:.4f}",
                   f"{v['active_fraction_img']:.4f}", f"{v['active_fraction_txt']:.4f}"]))
    print("\n## Remaining values")
    print(row(["ID", "readout img (PCA-10)", "readout txt (PCA-10)", "dead", "private",
               "top-2 share", "spanning", "code R@10", "CLIP R@10", "ratio"]))
    print(row(["---"] * 10))
    for n in NAMES:
        v = res[n]["values"]
        print(row([n, f"{v['readout_img']:.4f} ({v['pca10_img']:.4f})",
                   f"{v['readout_txt']:.4f} ({v['pca10_txt']:.4f})",
                   len(v["dead_indices"]), len(v["private_indices"]), f"{v['top2_mass_share']:.4f}",
                   f"{v['community']['spanning_fraction']:.3f}",
                   f"{v['code_retrieval_recall']:.4f}", f"{v['clip_retrieval_recall']:.4f}",
                   f"{v['retrieval_ratio']:.4f}"]))
    print("\n## Gate pass/fail")
    print(row(["ID"] + GATES + ["passed"]))
    print(row(["---"] * (len(GATES) + 2)))
    for n in NAMES:
        p = res[n]["passed"]
        print(row([n] + ["PASS" if p[g] else "FAIL" for g in GATES] + [f"{sum(p.values())}/9"]))
    print("\n## Pre-registered rule (removal vs D0)")
    d0 = res["D0"]["values"]
    d0_min = min(d0["participation_ratio_img"], d0["participation_ratio_txt"])
    print(row(["ID", "min PR", "x D0 min PR", "max abs r", "doubled?", "r above .9 -> below .9?", "implicated"]))
    print(row(["---"] * 7))
    for n in ("D3", "D4", "D5"):
        v = res[n]["values"]
        mn = min(v["participation_ratio_img"], v["participation_ratio_txt"])
        doubled = mn >= 2 * d0_min
        crossed = d0["correlation"]["max_abs"] > 0.9 and v["correlation"]["max_abs"] < 0.9
        print(row([n, f"{mn:.4f}", f"{mn / d0_min:.2f}", f"{v['correlation']['max_abs']:.5f}",
                   "yes" if doubled else "no", "yes" if crossed else "no",
                   "YES" if doubled or crossed else "no"]))
    print("\n## Other run facts")
    print(row(["ID", "loss first", "loss last", "mean code img", "mean code txt", "dead idx", "seconds"]))
    print(row(["---"] * 7))
    for n in NAMES:
        r, v = res[n], res[n]["values"]
        print(row([n, f"{r['loss_first']:.5f}", f"{r['loss_last']:.5f}", f"{r['mean_code_img']:.4f}",
                   f"{r['mean_code_txt']:.4f}", len(v["dead_indices"]), f"{r['runtime_seconds']:.0f}"]))


if __name__ == "__main__":
    main()
