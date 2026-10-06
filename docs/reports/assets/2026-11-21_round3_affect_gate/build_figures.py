"""Figures and figure data for docs/reports/auto/v2/2026-11-21_round3_affect_gate.md (reader fix, round 3).

Reads stored outputs only (nothing under src/ is written) and asserts every plotted number against its source:
  src/test/20261121_round3_affect_gate/results/test_verdict.json   the verdict, seven GO checks and the secondary
  src/test/20261121_round3_affect_gate/results/go_pooled.json      the same checks as the GO pass wrote them
  src/test/20261121_round3_affect_gate/results/descriptive.json    rule section 7: per seed, per pair, R1, control
  src/test/20261121_round3_affect_gate/results/sensitivity.json    detectable margins x (rule 6.1)
  src/test/20261121_round3_affect_gate/results/regression_check.json   seed-42 regression checks (rule section 5)
  src/test/20261121_round3_affect_gate/results/go_seed{49,50,51}.npz   per-anchor arrays of the GO pass
  src/test/20261121_round3_affect_gate/results/seed42_arrays.npz        seed-42 per-anchor arrays of R1 and AFF
  src/test/20261120_r1_levers_brainstorm/results/bs_10_subsets.json     seed-42 random-share control (SHA-256 asserted
                                                                         against the rule's D15 table)

Quantities the results files do not hold are computed here from the per-anchor arrays with
src.eval.aspect_metrics.cluster_bootstrap (5,000 resamples, seed 42; clusters = anchor paintings, shared across seeds)
and written to figure_data.json beside this script, which the report cites as "computed by the figure script".

Run from the repo root:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-21_round3_affect_gate/build_figures.py
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402

RES = ROOT / "src/test/20261121_round3_affect_gate/results"
BS10 = ROOT / "src/test/20261120_r1_levers_brainstorm/results/bs_10_subsets.json"
BS10_SHA = "c9d64815c1c6dba9e84177f7a33fb4e917a326f801ed9b115ae770ce1a1988f1"  # rule D15
RULE_SHA = "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925"
OUT = Path(__file__).resolve().parent
DPI = 150
SEEDS = (49, 50, 51)
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
PAIR_LABEL = {"emotion__style": "emotion × style", "emotion__genre": "emotion × genre",
              "style__genre": "style × genre"}
CHECKS = ("r1_vs_cosine", "r1_vs_rca", "r1_vs_B", "r1_vs_Bprime", "r1_vs_counterpart", "gain_statistic",
          "gain_vs_rca")

# dataviz reference palette, first three categorical slots (validated all-pairs in light mode); grey for references
C_AFF = "#2a78d6"   # slot 1, blue: AFF
C_R1 = "#eb6834"    # slot 2, orange: R1
C_CTL = "#1baf7a"   # slot 3, aqua: random-share control (both draws; told apart by marker)
C_REF = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


def same_ci(x, y, tol=1e-9):
    close(x["point"], y["point"], tol)
    close(x["ci95"][0], y["ci95"][0], tol)
    close(x["ci95"][1], y["ci95"][1], tol)


def pp(values, clusters):
    """Mean in percentage points with the painting-clustered 95% interval (round 1's common.point_ci)."""
    r = cluster_bootstrap(values, clusters)
    return {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]]}


def fmt(x):
    return f"{x['point']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


# ---------------------------------------------------------------- data and cross-checks

def data():
    verdict = load(RES / "test_verdict.json")
    gop = load(RES / "go_pooled.json")
    desc = load(RES / "descriptive.json")
    sens = load(RES / "sensitivity.json")
    reg = load(RES / "regression_check.json")
    for d in (verdict, gop, desc, sens, reg):
        assert d["provenance"]["rule_sha256"] == RULE_SHA
    assert verdict["verdict"] == "GO" and gop["go"] is True and desc["verdict"] == "GO"
    assert reg["passed"] is True and reg["n_comparisons"] == 134
    for k in CHECKS:
        same_ci(verdict["checks"][k], gop["checks"][k], 0.0)
    same_ci(verdict["secondary"], gop["secondary"], 0.0)
    assert sha(BS10) == BS10_SHA
    bs10 = load(BS10)

    # per-anchor arrays of the GO pass (seed order 49, 50, 51, as the rule pools them)
    z = {s: np.load(RES / f"go_seed{s}.npz", allow_pickle=False) for s in SEEDS}
    for s in SEEDS:
        assert json.loads(str(z[s]["meta"]))["rule_sha256"] == RULE_SHA
        assert np.all(z[s]["aff_cf__gain"] == 0.0)
    cat = lambda key: np.concatenate([z[s][key] for s in SEEDS])  # noqa: E731
    arr = {k: cat(k) for k in z[SEEDS[0]].files if k.endswith(("__r1", "__gain", "__other"))}
    cl, pidx = cat("cl"), cat("pair_index")
    assert len(cl) == 36864 == gop["n_episodes_pooled"] and len(np.unique(cl)) == gop["n_clusters_pooled"] == 5195

    # re-derive the seven pooled checks and the secondary from the per-anchor arrays (points and intervals)
    diffs = {
        "r1_vs_cosine": arr["aff_fused__r1"] - arr["cosine__r1"],
        "r1_vs_rca": arr["aff_fused__r1"] - arr["rca__r1"],
        "r1_vs_B": arr["aff_fused__r1"] - arr["B__r1"],
        "r1_vs_Bprime": arr["aff_fused__r1"] - arr["Bp__r1"],
        "r1_vs_counterpart": arr["aff_fused__r1"] - arr["aff_cf__r1"],
        "gain_statistic": arr["aff_fused__gain"] - arr["aff_cf__gain"],
        "gain_vs_rca": arr["aff_fused__gain"] - arr["rca__gain"],
    }
    for k, v in diffs.items():
        same_ci(pp(v, cl), gop["checks"][k])
    same_ci(pp(arr["aff_fused__r1"] - arr["r1_fused__r1"], cl), gop["secondary"])

    # the bar comparator is B' for AFF, R1 and both control draws, pooled and per seed
    i2, i3, i7 = desc["item2_bar_margin_cells_frozen"], desc["item3_R1_checks"], desc["item7_random_share_control"]
    assert i2["AFF_bar_margin_pooled"]["comparator"] == "B_prime"
    assert i3["bar_margin_pooled"]["comparator"] == "B_prime"
    for r in ("r0", "r1"):
        assert i7["draws"][r]["bar_margin_pooled"]["comparator"] == "B_prime"
    for s in SEEDS:
        assert i2["per_seed"][str(s)]["AFF"]["comparator"] == "B_prime"
        assert i2["per_seed"][str(s)]["R1"]["comparator"] == "B_prime"
    same_ci(i2["AFF_bar_margin_pooled"]["r1"], gop["checks"]["r1_vs_Bprime"])

    # per pair, pooled over seeds: AFF and R1 against B' (bar margins equal the descriptive pass's); other-aspect and
    # either-rate differences are new here (figure script)
    per_pair = {}
    for p_i, p in enumerate(PAIRS):
        m = pidx == p_i
        row = {}
        for who, key in (("AFF", "aff_fused"), ("R1", "r1_fused")):
            r1 = pp(arr[f"{key}__r1"][m] - arr["Bp__r1"][m], cl[m])
            gain = pp(arr[f"{key}__gain"][m] - arr["Bp__gain"][m], cl[m])
            other = pp(arr[f"{key}__other"][m] - arr["Bp__other"][m], cl[m])
            either = pp((arr[f"{key}__r1"][m] + arr[f"{key}__other"][m])
                        - (arr["Bp__r1"][m] + arr["Bp__other"][m]), cl[m])
            row[who] = {"r1": r1, "gain": gain, "other": other, "either": either}
        same_ci(row["AFF"]["r1"], i2["AFF_bar_margin_pooled"]["per_pair_r1"][p])
        same_ci(row["AFF"]["gain"], i2["AFF_bar_margin_pooled"]["per_pair_gain"][p])
        same_ci(row["R1"]["r1"], i3["bar_margin_pooled"]["per_pair_r1"][p])
        same_ci(row["R1"]["gain"], i3["bar_margin_pooled"]["per_pair_gain"][p])
        per_pair[p] = row
    assert np.all(arr["Bp__gain"] == 0.0)

    # pooled mean R@1 of every scorer (points; comparators cross-checked against the descriptive pass)
    means = {k: 100 * float(arr[f"{k}__r1"].mean()) for k in
             ("aff_fused", "aff_cf", "r1_fused", "B", "Bp", "cosine", "rca")}
    cm = i2["AFF_bar_margin_pooled"]["comparator_mean_r1"]
    close(means["Bp"], cm["B_prime"]), close(means["B"], cm["B"]), close(means["aff_cf"], cm["counterpart"])
    close(means["aff_fused"], i2["AFF_bar_margin_pooled"]["fused_r1"])
    close(means["r1_fused"], i3["bar_margin_pooled"]["fused_r1"])

    # seed 42: R1 minus B'(A0). bar_v is the per-anchor bar-margin difference (fused minus bar comparator); AFF's
    # bar comparator on seed 42 was B', so B' per anchor = aff_fused - aff_bar_v
    s42 = np.load(RES / "seed42_arrays.npz", allow_pickle=False)
    reg3 = {c["name"]: c for c in reg["comparisons"] if c["item"] == 3}
    reg2 = {c["name"]: c for c in reg["comparisons"] if c["item"] == 2}
    assert reg3["comparator"]["got"] == "B_prime" and reg2["comparator"]["got"] == "counterpart"
    same_ci(pp(s42["aff_bar_v"], s42["cl"]), {"point": reg3["bar_margin"]["got"][0], "ci95": reg3["bar_margin"]["got"][1:]})
    same_ci(pp(s42["r1_bar_v"], s42["cl"]), {"point": reg2["bar_margin"]["got"][0], "ci95": reg2["bar_margin"]["got"][1:]})
    bp_42 = s42["aff_fused__r1"] - s42["aff_bar_v"]
    close(100 * bp_42.mean(), 18.436686197916664)  # rule D11
    r1_minus_bp_42 = pp(s42["r1_fused__r1"] - bp_42, s42["cl"])

    # either cost per unit of gain against the own counterpart; R1's test-seed either change from margin and gain
    # (R@1 = (either + gain) / 2 per episode, so either difference = 2 * margin - gain exactly)
    aff_cf = i2["AFF_vs_counterpart_pooled"]
    r1_margin, r1_gain = i3["checks"]["r1_vs_counterpart"]["point"], i3["checks"]["gain_statistic"]["point"]
    r1_either_test = 2 * r1_margin - r1_gain
    close(2 * aff_cf["r1"]["point"] - aff_cf["gain"]["point"], aff_cf["either"]["point"])
    either_ratio = {
        "AFF_test": -aff_cf["either"]["point"] / aff_cf["gain"]["point"],
        "AFF_seed42": -reg3["either_vs_counterpart"]["got"] / reg3["gain_statistic"]["got"][0],
        "R1_test": -r1_either_test / r1_gain,
    }

    # projected half-width and x beside each realised pooled half-width (rule 6.1, 6.7)
    proj = {}
    for k in CHECKS + ("secondary",):
        c = gop["secondary"] if k == "secondary" else gop["checks"][k]
        proj[k] = {"x": sens["checks"][k]["x"], "projected_half_width": sens["checks"][k]["half_width"],
                   "realised_half_width": (c["ci95"][1] - c["ci95"][0]) / 2, "point": c["point"],
                   "point_over_x": c["point"] / sens["checks"][k]["x"]}

    # random-share control per pair: AFF minus control per pair is the difference of the per-pair bar margins (both
    # against the pooled comparator B', so the difference of means equals the paired mean difference; no interval)
    ctl_pp = {}
    for r in ("r0", "r1"):
        b = i7["draws"][r]["bar_margin_pooled"]["per_pair_r1"]
        ctl_pp[r] = {p: {"control_bar": b[p]["point"],
                         "AFF_minus_control": i2["AFF_bar_margin_pooled"]["per_pair_r1"][p]["point"] - b[p]["point"]}
                     for p in PAIRS}
    # AFF's tau_0 gate open share per pair and condition (counts / 12,288 pooled episodes per pair)
    g0 = desc["item4_gate_open_shares"]["AFF"]["pooled"]["tau_0"]
    open_pp = {p: {c: 100 * g0["per_pair_open_count"][p][c] / 12288 for c in ("a", "b")} for p in PAIRS}

    fig_data = {
        "what": "quantities computed by build_figures.py from the round-3 results (rule section 7 descriptive; "
                "decides nothing); pp, 95% painting-bootstrap intervals (5,000 resamples, seed 42)",
        "pooled_mean_r1": means,
        "per_pair_vs_Bprime": per_pair,
        "seed42_R1_minus_Bprime": r1_minus_bp_42,
        "R1_test_either_vs_own_counterpart": r1_either_test,
        "either_cost_per_unit_gain": either_ratio,
        "sensitivity_vs_realised": proj,
        "random_share_per_pair": ctl_pp,
        "AFF_tau0_open_share_per_pair_condition": open_pp,
        "inputs_sha256": {f.name: sha(f) for f in
                          [RES / "test_verdict.json", RES / "go_pooled.json", RES / "descriptive.json",
                           RES / "sensitivity.json", RES / "regression_check.json", RES / "seed42_arrays.npz",
                           BS10] + [RES / f"go_seed{s}.npz" for s in SEEDS]},
    }
    (OUT / "figure_data.json").write_text(json.dumps(fig_data, indent=1, ensure_ascii=False) + "\n")
    return verdict, gop, desc, sens, reg3, reg2, bs10, fig_data


# ---------------------------------------------------------------- figure (a): the checks

def fig_checks(gop, sens):
    left = [("r1_vs_cosine", "R@1 vs cosine"), ("r1_vs_rca", "R@1 vs RCA"),
            ("gain_statistic", "gain statistic\n(vs B, B′, counterpart)"), ("gain_vs_rca", "condition gain vs RCA")]
    right = [("r1_vs_B", "R@1 vs B"), ("r1_vs_Bprime", "R@1 vs B′(A0)"),
             ("r1_vs_counterpart", "R@1 vs matched\ncounterpart"), ("secondary", "secondary: R@1 vs R1\n(never changes GO)")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.2, wspace=0.62)
    for ax, rows, xmax in ((axes[0], left, 7.4), (axes[1], right, 1.25)):
        ys = -np.arange(len(rows), dtype=float)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for y, (k, lab) in zip(ys, rows):
            c = gop["secondary"] if k == "secondary" else gop["checks"][k]
            assert c["pass"] is True
            x = sens["checks"][k]["x"]
            lo, hi = c["ci95"]
            ax.plot([lo, hi], [y, y], color=C_AFF, lw=2.2, solid_capstyle="round", zorder=3)
            ax.plot(c["point"], y, "o", ms=8, color=C_AFF, zorder=4)
            ax.plot(x, y, marker="D", ms=7, mfc="white", mec=INK, mew=1.4, ls="none", zorder=5)
            ax.text(1.0, y + 0.22, f"{c['point']:+.3f} [{lo:+.3f}, {hi:+.3f}]", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8.5, color=INK)
            ax.text(1.0, y - 0.22, f"x = {x:.3f}", transform=ax.get_yaxis_transform(), ha="right", va="top",
                    fontsize=8.5, color=INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[1] for r in rows])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(ys.min() - 0.6, 0.6)
        ax.set_xlim(-0.04 * xmax, xmax)
        ax.set_xlabel("AFF minus comparator, pooled (percentage points)")
    axes[0].set_title("Against the external baselines, and condition gain", fontsize=10, loc="left", color=INK2)
    axes[1].set_title("Against the condition-free comparators, and R1", fontsize=10, loc="left", color=INK2)
    handles = [Line2D([0], [0], color=C_AFF, lw=2.2, marker="o", ms=7,
                      label="pooled point and 95% interval, seeds 49 to 51 (36,864 episodes)"),
               Line2D([0], [0], color="none", marker="D", ms=7, mfc="white", mec=INK, mew=1.4,
                      label="detectable margin x, projected from seed 42 before the build")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Every GO check and the secondary check cleared 0, each point at least 1.5 times its detectable margin",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.savefig(OUT / "checks.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure (b): per seed and per pair

def fig_per_seed_pair(desc):
    i2, i3 = desc["item2_bar_margin_cells_frozen"], desc["item3_R1_checks"]
    seed_cats = [str(s) for s in SEEDS] + ["pooled"]
    aff_seed = [i2["per_seed"][s]["AFF"]["r1"] for s in seed_cats[:3]] + [i2["AFF_bar_margin_pooled"]["r1"]]
    r1_seed = [i2["per_seed"][s]["R1"]["r1"] for s in seed_cats[:3]] + [i3["bar_margin_pooled"]["r1"]]
    for s in seed_cats[:3]:  # R1's per-seed bar margin is the same number in items 2 and 3
        same_ci(i2["per_seed"][s]["R1"]["r1"], i3["per_seed"][s]["bar"]["r1"], 0.0)
    aff_pair = [i2["AFF_bar_margin_pooled"]["per_pair_r1"][p] for p in PAIRS]
    r1_pair = [i3["bar_margin_pooled"]["per_pair_r1"][p] for p in PAIRS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True, gridspec_kw={"width_ratios": [4, 3]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.2, wspace=0.08)
    for ax, cats, aff, r1, labels in (
            (axes[0], seed_cats, aff_seed, r1_seed, ["seed 49", "seed 50", "seed 51", "pooled"]),
            (axes[1], PAIRS, aff_pair, r1_pair, [PAIR_LABEL[p] for p in PAIRS])):
        xs = np.arange(len(cats), dtype=float)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for off, vals, col, mk in ((-0.13, aff, C_AFF, "o"), (0.13, r1, C_R1, "s")):
            for x, v in zip(xs + off, vals):
                ax.plot([x, x], v["ci95"], color=col, lw=2.2, solid_capstyle="round", zorder=3)
                ax.plot(x, v["point"], marker=mk, ms=8, color=col, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(cats) - 0.4)
    # selective direct labels: the pooled values and style x genre
    for (ax, x, v, col, off) in ((axes[0], 3, aff_seed[3], C_AFF, -0.13), (axes[0], 3, r1_seed[3], C_R1, 0.13),
                                 (axes[1], 2, aff_pair[2], C_AFF, -0.13), (axes[1], 2, r1_pair[2], C_R1, 0.13)):
        ax.annotate(f"{v['point']:+.3f}", (x + off, v["point"]), xytext=(-8 if off < 0 else 8, 0),
                    textcoords="offset points", ha="right" if off < 0 else "left", va="center", fontsize=8.5,
                    color=INK)
    axes[0].axvline(2.5, color=GRID, lw=1.0)
    axes[0].set_ylabel("Bar margin against B′(A0), R@1 (percentage points)")
    axes[0].set_title("Per test seed, and pooled", fontsize=10, loc="left", color=INK2)
    axes[1].set_title("Per aspect pair, pooled over the seeds (not tested)", fontsize=10, loc="left", color=INK2)
    handles = [Line2D([0], [0], color=C_AFF, lw=2.2, marker="o", ms=7, label="AFF (gate open only on affect picks)"),
               Line2D([0], [0], color=C_R1, lw=2.2, marker="s", ms=7, label="R1 (gate open on any confident pick)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("AFF stayed above B′ on every seed and on both emotion pairs, and fell below it on style × genre",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.savefig(OUT / "bar_per_seed_pair.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure (c): random-share control

def fig_random_share(desc, reg3, reg2, bs10):
    i2, i3, i7 = desc["item2_bar_margin_cells_frozen"], desc["item3_R1_checks"], desc["item7_random_share_control"]
    same_ci(i7["draws"]["r0"]["AFF_minus_control_fused_r1"], i7["draws"]["r0"]["AFF_minus_control_bar_margin"], 0.0)
    same_ci(i7["draws"]["r1"]["AFF_minus_control_fused_r1"], i7["draws"]["r1"]["AFF_minus_control_bar_margin"], 0.0)
    assert bs10["random_seed0"]["comparator"] == bs10["random_seed1"]["comparator"] == "B_prime"
    s42_aff = {"point": reg3["bar_margin"]["got"][0], "ci95": reg3["bar_margin"]["got"][1:]}
    s42_r1 = {"point": reg2["bar_margin"]["got"][0], "ci95": reg2["bar_margin"]["got"][1:]}
    rows = [
        ("AFF", i2["AFF_bar_margin_pooled"]["r1"], s42_aff, C_AFF, "o", None),
        ("random-share control, draw 0", i7["draws"]["r0"]["bar_margin_pooled"]["r1"], bs10["random_seed0"]["bar"],
         C_CTL, "^", i7["draws"]["r0"]["AFF_minus_control_bar_margin"]),
        ("random-share control, draw 1", i7["draws"]["r1"]["bar_margin_pooled"]["r1"], bs10["random_seed1"]["bar"],
         C_CTL, "v", i7["draws"]["r1"]["AFF_minus_control_bar_margin"]),
        ("R1", i3["bar_margin_pooled"]["r1"], s42_r1, C_R1, "s", None),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    fig.subplots_adjust(left=0.22, right=0.66, top=0.84, bottom=0.24)
    ys = -np.arange(len(rows), dtype=float)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for y, (lab, test, s42, col, mk, dif) in zip(ys, rows):
        lo, hi = test["ci95"]
        ax.plot([lo, hi], [y + 0.1, y + 0.1], color=col, lw=2.2, solid_capstyle="round", zorder=3)
        ax.plot(test["point"], y + 0.1, marker=mk, ms=9, color=col, zorder=4)
        ax.plot(s42["point"], y - 0.18, marker=mk, ms=8, mfc="white", mec=col, mew=1.8, ls="none", zorder=4)
        txt = f"{test['point']:+.3f} [{lo:+.3f}, {hi:+.3f}]   seed 42: {s42['point']:+.3f}"
        if dif is not None:
            txt += f"\nAFF minus this control: {fmt(dif)}"
        ax.text(1.02, y, txt, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ys.min() - 0.6, 0.6)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("Bar margin, R@1 (percentage points)")
    handles = [Line2D([0], [0], color=INK2, lw=2.2, marker="o", ms=7,
                      label="seeds 49 to 51 pooled, against B′(A0), with 95% interval"),
               Line2D([0], [0], color="none", marker="o", ms=7, mfc="white", mec=INK2, mew=1.8,
                      label="seed 42 (recorded; R1's comparator there was its counterpart)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle("A random gate with AFF's per-condition open shares matched AFF; R1 trailed both",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.savefig(OUT / "random_share.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure (d): what changed

def box(ax, x, y, w, h, text, kind, fs=8.6):
    fill = {"same": ("#ecebe8", "#8a8984"), "unchanged": ("#e3e0f4", "#4a3aa7"), "replaced": ("#fde3d7", "#eb6834"),
            "new": ("#d3f1e6", "#12876a")}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fill[0],
                                ec=fill[1], lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=INK2))


def fig_what_changed():
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(-0.012, 1.012), ax.set_ylim(0.03, 0.9), ax.axis("off")
    h = 0.16
    y1 = 0.71
    box(ax, 0.005, y1, 0.16, h, "Episodes of fresh seeds\n49, 50, 51 (12,288 each),\nbuilt once; seed 42\nonly as a regression check",
        "new")
    box(ax, 0.195, y1, 0.17, h, "A0 groupings and heads\naffect, image, caption;\ngrouping scores s_h\n[frozen]", "same")
    box(ax, 0.395, y1, 0.16, h, "18 reader features\nper grouping: S, C, Δ,\ntwo spreads, match share\n[per seed]", "same")
    box(ax, 0.585, y1, 0.19, h, "R1's reader: round 1's two\nA0 half-readers, P^c(h),\npick π^c, top-two margin m^c\n[frozen; run per seed]",
        "unchanged")
    box(ax, 0.805, y1, 0.19, h, "weighted term\nT^c = Σ_h P^c(h)·s_h,\nz-scored per ranking row\n[per seed]", "unchanged")
    for x0, x1 in ((0.165, 0.195), (0.365, 0.395), (0.555, 0.585)):
        arrow(ax, x0, y1 + h / 2, x1, y1 + h / 2)
    arrow(ax, 0.775, y1 + h / 2, 0.805, y1 + h / 2)
    y2 = 0.36
    box(ax, 0.005, y2, 0.30, h, "gate (the one change)\nR1:  g^c = 1[m^c ≥ τ]\nAFF: g^c = 1[m^c ≥ τ] · 1[π^c = affect]\n"
        "τ_0..τ_3 frozen from R1's seed-42 margins", "replaced", fs=9)
    box(ax, 0.335, y2, 0.20, h, "fused score\nz(B) + λ_u·z(B) + λ_a·g^c·z(T^c),\n224 cells (4 τ × 7 λ_u × 8 λ_a)\n[cells frozen]",
        "unchanged")
    box(ax, 0.565, y2, 0.20, h, "integer cross-fit on the\nseed's own parity halves\n(min-margin rule; nested\ncontrol σ*) [per seed]",
        "unchanged")
    box(ax, 0.795, y2, 0.20, h, "matched counterpart\nG_cf = (g^a z(T^a) + g^b z(T^b)) / 2\nwith AFF's own gates,\nmax-R@1 cross-fit [per seed]",
        "unchanged")
    arrow(ax, 0.66, y1 - 0.012, 0.20, y2 + h + 0.012)   # pick and margin -> gate
    arrow(ax, 0.90, y1 - 0.012, 0.47, y2 + h + 0.012)   # weighted term -> fused score
    ax.text(0.36, 0.645, "π^c, m^c", fontsize=8.5, color=INK2, ha="center")
    ax.text(0.74, 0.668, "z(T^c)", fontsize=8.5, color=INK2, ha="center")
    for x0, x1 in ((0.305, 0.335), (0.535, 0.565), (0.765, 0.795)):
        arrow(ax, x0, y2 + h / 2, x1, y2 + h / 2)
    y3 = 0.07
    box(ax, 0.005, y3, 0.36, 0.17, "comparators: cosine, RCA, B, B′(A0) (B and B′ refitted\non each seed's halves), the matched counterpart;\n"
        "round 2's seven GO checks, pooled over 3 seeds,\npainting bootstrap with clusters shared across seeds", "same")
    box(ax, 0.395, y3, 0.27, 0.17, "secondary check (pre-registered,\nnever changes GO): AFF minus R1,\nfused R@1, paired per anchor;\n"
        "R1 run beside AFF on the same seeds", "new")
    box(ax, 0.695, y3, 0.30, 0.17, "random-share control (descriptive):\nR1's gate times a random mask with AFF's\n"
        "τ_0 open share per condition; it reads\nwhich condition is a, so it is never a method", "new")
    handles = [Patch(fc="#ecebe8", ec="#8a8984", label="grey: same as rounds 1 and 2 (heads, features, comparators, GO list)"),
               Patch(fc="#e3e0f4", ec="#4a3aa7", label="purple: unchanged from R1"),
               Patch(fc="#fde3d7", ec="#eb6834", label="orange: replaced (the gate)"),
               Patch(fc="#d3f1e6", ec="#12876a", label="teal: new in round 3")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("What round 3 changed: AFF is R1 with one extra factor in the gate, tested on fresh seeds",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.925, "[frozen]: fixed on seed 42 or earlier and reused unchanged; [per seed]: recomputed on each "
             "test seed (the readers' weights are frozen, their outputs are per seed)", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.07)
    fig.savefig(OUT / "what_changed.png", dpi=DPI)
    plt.close(fig)


def main():
    verdict, gop, desc, sens, reg3, reg2, bs10, fd = data()
    fig_checks(gop, sens)
    fig_per_seed_pair(desc)
    fig_random_share(desc, reg3, reg2, bs10)
    fig_what_changed()
    pr = fd["per_pair_vs_Bprime"]
    for p in PAIRS:
        print(p, {w: {m: fmt(pr[p][w][m]) for m in ("r1", "gain", "other", "either")} for w in ("AFF", "R1")})
    print("means", {k: round(v, 3) for k, v in fd["pooled_mean_r1"].items()})
    print("seed42 R1 - B'", fmt(fd["seed42_R1_minus_Bprime"]))
    print("R1 either (test)", round(fd["R1_test_either_vs_own_counterpart"], 4), fd["either_cost_per_unit_gain"])
    for k, v in fd["sensitivity_vs_realised"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
