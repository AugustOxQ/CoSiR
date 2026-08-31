"""
Phase 1 buddy analysis (offline): identity split, type confusion (lift), and the
held-out DINOv2 NN test. Produces stats.json + figures under docs/reports/assets/.

Run extract_dino.py first (writes dino_feats.npy) to enable step 3.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import buddy_analysis as ba

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                      "docs", "reports", "assets", "buddy_analysis"))
DINO = os.path.join(os.path.dirname(__file__), "dino_feats.npy")
os.makedirs(ASSETS, exist_ok=True)


def _heatmap(ax, M, title, cmap="RdBu_r", center1=True):
    vmax = np.nanmax(np.abs(np.log2(M))) if center1 else np.nanmax(M)
    im = ax.imshow(np.log2(M) if center1 else M, cmap=cmap,
                   vmin=-vmax if center1 else 0, vmax=vmax)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(ba.TYPE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ba.TYPE_NAMES, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black")
    ax.set_title(title, fontsize=9)
    return im


def main():
    data = ba.load_data()
    G = ba.build_graphs(data, K=30)
    stats = {"dataset": "impressions", "n": data.n, "n_source_images": len(data.source_names),
             "K": 30, "type_names": ba.TYPE_NAMES, "graphs": {}}

    heldout = np.load(DINO) if os.path.exists(DINO) else None
    if heldout is not None:
        valid = np.abs(heldout).sum(1) > 0
        print(f"held-out DINO loaded: {heldout.shape}, valid rows {valid.sum()}/{len(valid)}")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for col, name in enumerate(["B", "E"]):
        e = ba.edges(G[name])
        st = ba.identity_stats(data, e)
        wm = st.pop("within_mask")
        counts, lift, chi2, p = ba.type_confusion(data, e)
        _, lift_cross, chi2c, pc = ba.type_confusion(data, e[~wm])

        g = {**st, "type_lift_overall": lift.tolist(), "chi2_overall": chi2,
             "type_lift_cross_image": lift_cross.tolist(), "chi2_cross_image": chi2c,
             "diag_lift_overall": float(np.nanmean(np.diag(lift))),
             "offdiag_lift_overall": float(np.nanmean(lift[~np.eye(4, dtype=bool)])),
             "diag_lift_cross": float(np.nanmean(np.diag(lift_cross))),
             "offdiag_lift_cross": float(np.nanmean(lift_cross[~np.eye(4, dtype=bool)]))}

        if heldout is not None:
            ev = e[valid[e[:, 0]] & valid[e[:, 1]]]
            wmv = data.source_id[ev[:, 0]] == data.source_id[ev[:, 1]]
            g["heldout_dino"] = ba.heldout_distance_test(heldout, data, ev, wmv)

        stats["graphs"][name] = g
        _heatmap(axes[0, col], lift, f"{name}: type lift (all edges)  χ²={chi2:.0f}")
        _heatmap(axes[1, col], lift_cross, f"{name}: type lift (cross-image)  χ²={chi2c:.0f}")

    fig.suptitle("Buddy type confusion — log2(lift over random); red = enriched, blue = depleted", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "type_confusion.png"), dpi=130)
    print("wrote type_confusion.png")

    # identity + held-out summary bars
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    names = ["B", "E"]
    within = [stats["graphs"][n]["frac_within_image"] for n in names]
    a1.bar(names, within, color=["#c44", "#48a"])
    a1.set_ylabel("fraction of buddy edges within same source image")
    a1.set_ylim(0, 1); a1.set_title("Same-photo dominance of buddy edges")
    for i, v in enumerate(within):
        a1.text(i, v + 0.02, f"{v:.0%}", ha="center")

    if heldout is not None:
        x = np.arange(len(names)); w = 0.25
        bc = [stats["graphs"][n]["heldout_dino"]["buddy_cross"] for n in names]
        rc = [stats["graphs"][n]["heldout_dino"]["random_cross"] for n in names]
        a2.bar(x - w/2, bc, w, label="buddy (cross-image)", color="#48a")
        a2.bar(x + w/2, rc, w, label="random (cross-image)", color="#bbb")
        a2.set_xticks(x); a2.set_xticklabels(names)
        a2.set_ylabel("mean DINOv2 distance (held-out)")
        a2.set_title("Independent-signal test: lower = buddies more alike")
        a2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(ASSETS, "identity_heldout.png"), dpi=130)
    print("wrote identity_heldout.png")

    with open(os.path.join(ASSETS, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("wrote stats.json")
    print(json.dumps({n: {k: stats["graphs"][n][k] for k in
                          ["n_edges", "avg_degree", "frac_within_image",
                           "diag_lift_overall", "diag_lift_cross"]}
                      for n in names}, indent=2))
    if heldout is not None:
        for n in names:
            print(n, "DINO held-out:", json.dumps(stats["graphs"][n]["heldout_dino"]))


if __name__ == "__main__":
    main()
