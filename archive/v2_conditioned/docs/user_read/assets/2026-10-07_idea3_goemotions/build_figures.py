"""Build the three briefing figures, validating every source value first.

Run from the repository root using the command supplied in the figure brief.
Expected numbers below are assertion targets only; all marks and labels use
the values loaded from the source JSON files.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
A_PATH = "docs/reports/assets/2026-11-23_idea3_goemotions/figure_data.json"
D_PATH = "src/test/20261123_idea3_goemotions/results/dev_seed42.json"
METHODS = {
    "G-T": "GoEmotions, steering term only",
    "G-TF": "GoEmotions, term and reader",
}
COLORS = {"G-T": "#0072B2", "G-TF": "#D55E00"}
GREY = "#777777"
CAPTIONS = {
    "fig1_vs_affect_steering.png": "R@1 differences from affect steering, with 95% intervals and net ranking changes, for both GoEmotions versions.",
    "fig2_signal_by_side.png": "Same-emotion lift on caption, image, and image–caption sides under both placements; dashed line: the most a caption placement could reach (2.71).",
    "fig3_same_weights.png": "Changes in condition gain, either rate, and R@1, with 95% intervals, when both GoEmotions versions use affect steering’s weights.",
}


def read_key(document, key):
    value = document
    for part in key.split("."):
        value = value[part]
    return value


def close(actual, expected, key, tolerance=1e-9):
    assert math.isfinite(actual), f"Nonfinite value: {key}"
    assert abs(actual - expected) <= tolerance, (
        f"{key}: read {actual!r}; expected {expected!r} within {tolerance}"
    )


def source_value(doc, path, key, expected, tolerance=1e-9):
    value = read_key(doc, key)
    close(value, expected, key, tolerance)
    return {"value": value, "source_file": path, "source_key": key}


def interval(doc, key, expected_point, expected_ci, tolerance=1e-9):
    value = read_key(doc, key)
    close(value["point"], expected_point, key + ".point", tolerance)
    assert len(value["ci95"]) == 2, f"Invalid interval: {key}"
    for index, bound in enumerate(value["ci95"]):
        close(bound, expected_ci[index], f"{key}.ci95[{index}]", tolerance)
    assert value["ci95"][0] <= value["point"] <= value["ci95"][1], key
    return {
        "point": value["point"], "ci95": value["ci95"],
        "source_file": A_PATH,
        "source_keys": {"point": key + ".point", "ci95": key + ".ci95"},
    }


def load_data():
    a = json.loads((ROOT / A_PATH).read_text())
    d = json.loads((ROOT / D_PATH).read_text())
    data = {"fig1": {"methods": {}}, "fig2": {"bars": {}}, "fig3": {"methods": {}}}
    expected_first = {
        "G-T": (-0.34383138020833337, [-0.5222166846883708, -0.16297832514540103], -169, 1e-9),
        "G-TF": (-0.2726, [-0.4565, -0.0924], -134, 1e-4),
    }
    for method, (point, bounds, net, tolerance) in expected_first.items():
        key = f"descriptive.minus_AFF.{method}.pooled"
        record = interval(a, key, point, bounds, tolerance)
        record["label"] = METHODS[method]
        record["net_rankings"] = source_value(d, D_PATH, f"candidates.{method}.delta_int", net)
        # D is read only for delta_int. Recover the denominator from the exact
        # percentage-point change and net count rather than typing a plotted count.
        denominator = round(record["net_rankings"]["value"] * 100 / record["point"])
        close(denominator, 49152, f"{method}: derived ranking count")
        close(record["point"], 100 * net / denominator, f"{method}: net/total consistency")
        record["total_rankings"] = {
            "value": denominator,
            "derived_from": [
                {"source_file": D_PATH, "source_key": f"candidates.{method}.delta_int"},
                {"source_file": A_PATH, "source_key": key + ".point"},
            ],
            "operation": "round(100 * delta_int / pooled.point)",
        }
        data["fig1"]["methods"][METHODS[method]] = record
    lifts = {
        "caption_x_caption_CLIP": 1.6563540987645193,
        "caption_x_caption_GE": 2.5747361780729103,
        "image_x_image": 1.0432515762928676,
        "image_x_caption_CLIP": 1.1445184466303795,
        "image_x_caption_GE": 1.207331272357703,
    }
    for key, expected in lifts.items():
        data["fig2"]["bars"][key] = source_value(
            a, A_PATH, f"descriptive.pair_lifts.{key}.ratio_same_over_diff", expected
        )
    data["fig2"]["communities"] = source_value(
        a, A_PATH, "diagnostics.pair_lift.groups", 2.7111312041209863
    )
    expected_fixed = {
        "G-T": {
            "gain_minus_AFF": (0.8219401041666666, [0.5473940598100887, 1.0997916944842665]),
            "either_minus_AFF": (-1.416015625, [-1.6943691582730787, -1.1312102278963105]),
            "minus_AFF": (-0.29703776041666663, [-0.4887166820807385, -0.1042526519860795]),
        },
        "G-TF": {
            "gain_minus_AFF": (0.7100423177083333, [0.43546029770007066, 1.0090379985629752]),
            "either_minus_AFF": (-1.2227376302083335, [-1.5031144998153474, -0.9467061404557124]),
            "minus_AFF": (-0.25634765625, [-0.4457005329630872, -0.060688573018442195]),
        },
    }
    for method, measures in expected_fixed.items():
        record = {}
        for measure, (point, bounds) in measures.items():
            record[measure] = interval(a, f"descriptive.fixed_cells.{method}_at_AFF_cells.{measure}", point, bounds)
        close((record["gain_minus_AFF"]["point"] + record["either_minus_AFF"]["point"]) / 2,
              record["minus_AFF"]["point"], f"{method}: R@1 average")
        data["fig3"]["methods"][METHODS[method]] = record
    for figure in ("fig1", "fig3"):
        data[figure]["reference"] = {"value": 0.0, "source_key": "brief: change against affect steering"}
        data[figure]["units"] = "percentage points"
    data["fig2"]["baseline"] = {"value": 1.0, "source_key": "brief: no-signal lift"}
    data["fig2"]["units"] = "same-emotion / different-emotion lift (ratio)"
    print("PASS: all 32 plotted source values match expectations (1e-9; Figure 1 term-and-reader point/bounds 1e-4).")
    print("PASS: ranking denominators are 49,152; both fixed-weight R@1 points equal the average of gain and either changes.")
    return data


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAAAAA")
    ax.tick_params(length=3, color="#AAAAAA")


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Reject text that would be clipped by the fixed 1400 x 800 output canvas.
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        assert text.get_fontsize() >= 10, f"Small text: {text.get_text()}"
        bbox = text.get_window_extent(renderer)
        assert bbox.x0 >= 0 and bbox.y0 >= 0 and bbox.x1 <= fig.bbox.width and bbox.y1 <= fig.bbox.height, (
            f"Clipped text in {name}: {text.get_text()!r}"
        )
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Rendered {name}: 1400 x 800 pixels; fonts >= 10 pt; text inside canvas.")


def method_handles():
    return [Line2D([], [], marker="o", color=COLORS[m], linestyle="none", markersize=6, label=METHODS[m]) for m in METHODS]


def fig1(data):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    fig.subplots_adjust(left=0.40, right=0.73, bottom=0.22, top=0.73)
    fig.suptitle("Neither GoEmotions version beat affect steering", x=0.04, y=0.96, ha="left", fontsize=14, fontweight="bold")
    fig.legend(handles=method_handles(), loc="upper left", bbox_to_anchor=(0.035, 0.89), frameon=False, fontsize=10)
    ax.axvline(data["reference"]["value"], color=GREY, linewidth=1)
    ax.annotate("affect steering", (0, 1.5), xytext=(0, 5), textcoords="offset points", color=GREY, ha="center", va="bottom", fontsize=10)
    for row, method in zip([1, 0], METHODS):
        item = data["methods"][METHODS[method]]
        point, (low, high) = item["point"], item["ci95"]
        ax.errorbar(point, row, xerr=[[point - low], [high - point]], fmt="o", color=COLORS[method], capsize=4, linewidth=1.8, markersize=7)
        ax.text(1.05, row, f'{item["net_rankings"]["value"]:+,} of\n{item["total_rankings"]["value"]:,} rankings', transform=ax.get_yaxis_transform(), va="center", fontsize=10.5)
    ax.set_yticks([1, 0], ["GoEmotions,\nsteering term only", "GoEmotions,\nterm and reader"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.60, 0.13)
    ax.set_xticks([-0.6, -0.4, -0.2, 0.0])
    fig.text(0.5, 0.10, "R@1 minus affect steering (percentage points)", ha="center", fontsize=11)
    fig.text(0.5, 0.035, "Points and 95% intervals · affect steering (current best)", ha="center", color=GREY, fontsize=10)
    style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    save(fig, "fig1_vs_affect_steering.png")


def fig2(data):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.35, top=0.77)
    fig.suptitle("A sharper caption placement barely sharpened\nwhat the method reads", x=0.04, y=0.97, ha="left", fontsize=14, fontweight="bold")
    baseline = data["baseline"]["value"]
    bars = [
        (0 - 0.18, "caption_x_caption_CLIP", GREY, None),
        (0 + 0.18, "caption_x_caption_GE", COLORS["G-T"], None),
        (1, "image_x_image", GREY, "///"),
        (2 - 0.18, "image_x_caption_CLIP", GREY, None),
        (2 + 0.18, "image_x_caption_GE", COLORS["G-T"], None),
    ]
    for x, key, color, hatch in bars:
        value = data["bars"][key]["value"]
        ax.bar(x, value - baseline, bottom=baseline, width=0.30, color=color, hatch=hatch, edgecolor="white" if hatch else color, linewidth=0.6)
        inside = value > data["communities"]["value"] - 0.25
        ax.annotate(f"{value:.2f}", (x, value), xytext=(0, -5 if inside else 4), textcoords="offset points", ha="center", va="top" if inside else "bottom", color="white" if inside else "black", fontsize=10)
    ax.axhline(baseline, color=GREY, linewidth=1)
    ax.text(0.99, baseline - 0.10, f"no signal ({baseline:.1f})", transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=10, color=GREY)
    communities = data["communities"]["value"]
    ax.hlines(communities, -0.43, 0.43, color=GREY, linestyle=(0, (4, 3)), linewidth=1)
    ax.annotate("a perfect caption placement (2.71)", (0.43, communities), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=10, color=GREY)
    ax.set_ylim(0.60, 3.12)
    ax.set_xlim(-0.55, 2.55)
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xticks([0, 1, 2], ["caption side\nalone", "image side\nalone", "image x caption\n(what the method reads)"])
    ax.set_ylabel("lift: same emotion over\ndifferent emotion (ratio)", fontsize=10.5)
    ax.tick_params(axis="x", length=0, pad=7)
    fig.legend(handles=[
        Patch(facecolor=GREY, label="current placement (CLIP caption head)"),
        Patch(facecolor=COLORS["G-T"], label="GoEmotions placement"),
        Patch(facecolor=GREY, edgecolor="white", hatch="///", label="image head (the same for both)"),
    ], loc="lower left", bbox_to_anchor=(0.12, 0.01), frameon=False, fontsize=10)
    style_axes(ax)
    save(fig, "fig2_signal_by_side.png")


def fig3(data):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.22, top=0.66)
    fig.suptitle("At the same weights, the sharper term bought gain\nbut lost more either rate", x=0.04, y=0.97, ha="left", fontsize=14, fontweight="bold")
    fig.legend(handles=method_handles(), loc="upper left", bbox_to_anchor=(0.14, 0.82), frameon=False, fontsize=10)
    ax.axhline(data["reference"]["value"], color=GREY, linewidth=1)
    measures = ["gain_minus_AFF", "either_minus_AFF", "minus_AFF"]
    for method, offset in zip(METHODS, [-0.13, 0.13]):
        records = [data["methods"][METHODS[method]][key] for key in measures]
        points = np.array([item["point"] for item in records])
        errors = np.array([[item["point"] - item["ci95"][0], item["ci95"][1] - item["point"]] for item in records]).T
        ax.errorbar(np.arange(3) + offset, points, yerr=errors, fmt="o", color=COLORS[method], capsize=4, linewidth=1.8, markersize=7)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-1.9, 1.35)
    ax.set_xticks([0, 1, 2], ["condition gain", "either rate", "R@1 (= average\nof the two)"])
    ax.set_ylabel("change against affect steering\n(percentage points)", fontsize=10.5)
    ax.tick_params(axis="x", length=0, pad=7)
    fig.text(0.54, 0.045, "Affect steering’s own weights · points and 95% intervals", ha="center", color=GREY, fontsize=10)
    style_axes(ax)
    save(fig, "fig3_same_weights.png")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "figure.facecolor": "white", "axes.facecolor": "white"})
    data = load_data()  # All assertions happen before any plotting or output.
    (OUT / "figure_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    assert all(len(caption.split()) <= 25 for caption in CAPTIONS.values())
    (OUT / "captions.md").write_text("".join(f"{name}: {caption}\n" for name, caption in CAPTIONS.items()))
    fig1(data["fig1"])
    fig2(data["fig2"])
    fig3(data["fig3"])


if __name__ == "__main__":
    main()
