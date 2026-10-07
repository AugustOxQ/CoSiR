"""Figures and figure data for docs/reports/auto/v2/2026-11-24_clip_lightweight_ft.md (lightweight CLIP fine-tuning
comparator: linear probe, last block, LoRA).

Reads stored outputs only (nothing under src/ or res/ is written) and re-derives every plotted number:
  res/cluster_jobs/<tag>/code/outputs/clipft/<variant>_lr<lr>/metrics.json      val retrieval per epoch (9 runs)
  res/cluster_jobs/<tag>/code/outputs/clipft/<variant>_lr<lr>/run_record.json   best epoch, trainable parameters
  res/cluster_jobs/<tag>/.../features.npz, features_epoch0.npz                   selected models' features and the
                                                                                 untrained cache-path reference
  src/test/20261030_aspect_baselines/results/episodes_seed{s}.npz              aspect episodes (SHA-256 asserted
                                                                                 against baselines_seed{s}.json)
  src/test/20261030_aspect_baselines/results/per_anchor_seed{s}.npz            stored plain cosine (asserted equal)
  src/test/20261121_round3_affect_gate/results/go_seed{49,50,51}.npz           AFF, B, B'(A0) per-anchor arrays
  src/test/20261122_round4_aff_vetoes/results/seed42_arrays.npz                 the same on seed 42, plus B'(A1)
and asserts the re-derived numbers against src/test/20261124_clip_lightweight_ft/results/eval.json (SHA-256 asserted):
every point of every scope, scorer, comparison and metric, and the 95% intervals of the plotted quantities.

The selection is re-derived from metrics.json with the spec's rule (spec section 4: the highest mean of val
image->caption R@1 and caption->image R@1 over epochs 1 to 10, ties to the smaller learning rate, then the earlier
epoch), in exact integer arithmetic. Quantities eval.json does not hold (the per-side first-place rates of the
condition-free cosines, the cache-path feature cosines, the R@1 decomposition, the grid-edge arithmetic) are computed
here and written to figure_data.json beside this script; the report cites them as "computed by the figure script".

Run from the repo root (CPU only, about two minutes):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-24_clip_lightweight_ft/build_figures.py
"""
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_episodes import AspectEpisodes, concat_episodes, episodes_sha256  # noqa: E402
from src.eval.aspect_metrics import DIRECTIONS, cluster_bootstrap, first_place, per_anchor  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores  # noqa: E402

OUT = Path(__file__).resolve().parent
FT = ROOT / "src/test/20261124_clip_lightweight_ft"
EVAL = FT / "results/eval.json"
EVAL_SHA = "4204be404feb2cfa32a187aa6551dd2db46ffea1117d286c742a8d38e0c6a18b"
E1 = ROOT / "src/test/20261030_aspect_baselines/results"
R3 = ROOT / "src/test/20261121_round3_affect_gate/results"
R4 = ROOT / "src/test/20261122_round4_aff_vetoes/results"
RUNS_TXT = ROOT / ".superpowers/sdd/2026-10-07-clip-lightweight-ft/runs.txt"
JOBS = ROOT / "res/cluster_jobs"

# (variant, lr string, cluster tag (UTC), node): the nine runs at commit 65bb2f4, as in the ledger's runs.txt
RUNS = [("LP", "1e-4", "20261007-070919-65bb2f4", "node404"), ("LP", "3e-4", "20261007-070950-65bb2f4", "node404"),
        ("LP", "1e-3", "20261007-071021-65bb2f4", "node404"), ("LB", "3e-6", "20261007-071049-65bb2f4", "node405"),
        ("LB", "1e-5", "20261007-071120-65bb2f4", "node405"), ("LB", "3e-5", "20261007-071151-65bb2f4", "node405"),
        ("LoRA", "3e-5", "20261007-071219-65bb2f4", "node411"), ("LoRA", "1e-4", "20261007-071251-65bb2f4", "node411"),
        ("LoRA", "3e-4", "20261007-071321-65bb2f4", "node411")]
VARIANTS = ("LP", "LB", "LoRA")
EXPECTED_SELECTION = {"LP": ("3e-4", 9), "LB": ("3e-5", 8), "LoRA": ("1e-4", 10)}   # run log 09:35
EXPECTED_PARAMS = {"LP": 524_289, "LB": 10_898_177, "LoRA": 1_966_081}
CACHE_REF_RUN = ("LB", "3e-5")   # its features_epoch0.npz reproduces eval.json's ft:CLIPcache (rd_ft_report.md)

SEEDS = (42, 49, 50, 51)
POOLED = (49, 50, 51)
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
PAIR_KEYS = [f"{a}__{b}" for a, b in PAIRS]
PAIR_LABEL = {"emotion__style": "emotion × style", "emotion__genre": "emotion × genre", "style__genre": "style × genre"}
FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
METRICS = ("r1", "either", "gain", "other")
STORED = {"49-51": {"AFF": (R3 / "go_seed{s}.npz", "aff_fused"), "B": (R3 / "go_seed{s}.npz", "B"),
                    "Bp0": (R3 / "go_seed{s}.npz", "Bp")},
          "42": {"AFF": (R4 / "seed42_arrays.npz", "aff_fused"), "B": (R4 / "seed42_arrays.npz", "B"),
                 "Bp0": (R4 / "seed42_arrays.npz", "Bp0"), "Bp1": (R4 / "seed42_arrays.npz", "Bp1")}}
COMPARISONS = {"AFF_minus_ft": ("AFF", "ft"), "ft_minus_plain": ("ft", "plain"), "ft_minus_Bp0": ("ft", "Bp0")}
TOL = 1e-9

# dataviz reference palette (light mode): grey for references, slot 2 orange for the fine-tuned family (this report),
# slot 1 blue for AFF; the learning rates of one variant use one orange ramp, light to dark (an ordinal magnitude)
C_AFF = "#2a78d6"
C_FT = "#eb6834"
C_FT_RAMP = ("#f5b392", "#eb6834", "#a8401a")
C_REF = "#8a8984"
C_REF_DARK = "#52514e"
C_GAIN = "#2a78d6"
C_EITHER = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DPI = 150

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_dir(variant, lr):
    tag = next(t for v, l, t, _ in RUNS if v == variant and l == lr)
    return JOBS / tag / "code/outputs/clipft" / f"{variant}_lr{lr}"


# ----- 1. training curves and the selection --------------------------------------------------------------------------

def check_runs_txt():
    if not RUNS_TXT.exists():
        return "runs.txt not present; tags taken from this script"
    rows = [ln.split()[:5] for ln in RUNS_TXT.read_text().splitlines() if ln.strip()]
    got = [(v, lr, tag, node) for node, v, lr, ok, tag in rows]
    assert sorted(got) == sorted(RUNS), "run tags differ from the ledger's runs.txt"
    assert all(ok == "True" for _, _, _, ok, _ in rows)
    return "tags equal the ledger's runs.txt"


def training_curves():
    runs = {}
    for variant, lr, tag, node in RUNS:
        d = run_dir(variant, lr)
        m = json.loads((d / "metrics.json").read_text())
        rec = json.loads((d / "run_record.json").read_text())
        assert rec["variant"] == variant and rec["hostname"] == node and rec["git_commit"].startswith("65bb2f4")
        assert rec["n_trainable_params"] == EXPECTED_PARAMS[variant]
        exact, sel = [], []
        for e in m["epochs"]:
            q = Fraction(e["i2t_correct"], e["n_images"]) + Fraction(e["t2i_correct"], e["n_captions"])
            exact.append(q / 2)
            assert abs(float(q / 2) - e["selection"]) < 1e-12, (variant, lr, e["epoch"])
            sel.append(e["selection"])
        assert [e["epoch"] for e in m["epochs"]] == list(range(11))
        best = max(range(1, 11), key=lambda k: (exact[k], -k))          # epochs 1..10, ties to the earlier epoch
        assert best == rec["best_epoch"] and abs(float(exact[best]) - rec["best_selection"]) < 1e-12
        runs[(variant, lr)] = {
            "tag": tag, "node": node, "epochs": list(range(11)), "selection": sel,
            "i2t_r1": [e["i2t_r1"] for e in m["epochs"]], "t2i_r1": [e["t2i_r1"] for e in m["epochs"]],
            "i2t_correct": [e["i2t_correct"] for e in m["epochs"]],
            "t2i_correct": [e["t2i_correct"] for e in m["epochs"]],
            "n_images": m["epochs"][0]["n_images"], "n_captions": m["epochs"][0]["n_captions"],
            "train_loss": [e["train_loss"] for e in m["epochs"]], "seconds": [e["seconds"] for e in m["epochs"]],
            "best_epoch": best, "best_selection": float(exact[best]), "exact_best": exact[best],
            "n_trainable_params": rec["n_trainable_params"], "duration_s": rec["duration_s"],
            "start_time": rec["start_time"], "end_time": rec["end_time"]}
    selected = {}
    for v in VARIANTS:
        lrs = [lr for vv, lr, _, _ in RUNS if vv == v]
        # highest selection; ties to the smaller learning rate, then the earlier epoch
        cands = [(runs[(v, lr)]["exact_best"], -float(lr), -runs[(v, lr)]["best_epoch"], lr) for lr in lrs]
        top = max(cands)
        n_top = sum(1 for c in cands if c[0] == top[0])
        selected[v] = {"lr": top[3], "epoch": runs[(v, top[3])]["best_epoch"],
                       "selection": runs[(v, top[3])]["best_selection"], "ties_at_max": n_top - 1}
        assert (selected[v]["lr"], selected[v]["epoch"]) == EXPECTED_SELECTION[v], (v, selected[v])
    for r in runs.values():
        r.pop("exact_best")
    return runs, selected


# ----- 2. episode evaluation -----------------------------------------------------------------------------------------

def place(rows, img, txt, n_rows, selection):
    """Selection-row features in feature-row order, NaN elsewhere."""
    rows = np.asarray(rows, dtype=np.int64)
    assert len(np.unique(rows)) == len(rows) and np.isin(selection, rows).all()
    keep = np.isin(rows, selection)
    out = []
    for x in (img, txt):
        full = np.full((n_rows, x.shape[1]), np.nan, dtype=np.float32)
        full[rows[keep]] = np.asarray(x, dtype=np.float32)[keep]
        assert np.isfinite(full[selection]).all()
        out.append(full)
    return out


def load_episodes(seed, selection):
    z = np.load(E1 / f"episodes_seed{seed}.npz")
    base = json.loads((E1 / f"baselines_seed{seed}.json").read_text())
    assert list(z["pair_order"]) == PAIR_KEYS == base["pair_order"]
    parts = []
    for a, b in PAIRS:
        ep = AspectEpisodes(a, b, *(z[f"{a}__{b}__{k}"].astype(np.int64) for k in FIELDS))
        assert episodes_sha256(ep) == base["episodes_sha256"][f"{a}__{b}"], (seed, a, b)
        assert np.isin(ep.rows(), selection).all()
        parts.append(ep)
    n = len(parts[0].anchor)
    assert n == base["n_per_pair"] and all(len(p.anchor) == n for p in parts)
    return concat_episodes(parts), np.repeat(np.arange(len(PAIRS)), n)


def cosine_metrics(img, txt, ep):
    """Per-anchor r1/other of the cosine, plus the per-side first-place rates (p_A first, p_B first)."""
    sc = cosine_scores(EvalInputs(img, txt), ep)
    pa = per_anchor(sc)
    side_a = 0.5 * sum(first_place(sc["a"][d], 0) for d in DIRECTIONS)
    side_b = 0.5 * sum(first_place(sc["a"][d], 1) for d in DIRECTIONS)
    assert np.array_equal(0.5 * (side_a + side_b), pa["r1"])     # condition-free: R@1 is the mean of the two sides
    assert np.array_equal(pa["r1"], pa["other"])
    return {"r1": pa["r1"], "other": pa["other"]}, {"A": side_a, "B": side_b}


def with_derived(d):
    r1, other = np.asarray(d["r1"], np.float64), np.asarray(d["other"], np.float64)
    return {"r1": r1, "other": other, "either": r1 + other, "gain": r1 - other}


def episodes_block():
    data = load_artelingo()
    sp = artelingo_splits(data)
    n_rows, sel, val = len(sp.groups), np.asarray(sp.selection), np.asarray(sp.val)
    frozen = place(sel, data.img_features[sel], data.txt_features[sel], n_rows, sel)
    feats, sources = {}, {}
    for v in VARIANTS:
        lr, _ = EXPECTED_SELECTION[v]
        p = run_dir(v, lr) / "features.npz"
        z = np.load(p)
        feats[f"ft:{v}"] = place(z["rows"], z["img"], z["txt"], n_rows, sel)
        sources[f"ft:{v}"] = str(p.relative_to(ROOT))
    p0 = run_dir(*CACHE_REF_RUN) / "features_epoch0.npz"
    z0 = np.load(p0)
    feats["ft:CLIPcache"] = place(z0["rows"], z0["img"], z0["txt"], n_rows, sel)
    sources["ft:CLIPcache"] = str(p0.relative_to(ROOT))

    # cache-path image features against the frozen cache (unit vectors, float64), per split, per row
    cache_cos = {}
    for split, rows_split in (("selection", sel), ("val", val)):
        m = np.isin(z0["rows"], rows_split)
        rr = z0["rows"][m]
        for mod, key, ref in (("img", "img", data.img_features), ("txt", "txt", data.txt_features)):
            a = z0[key][m].astype(np.float64)
            b = ref[rr].astype(np.float64)
            c = (a / np.linalg.norm(a, axis=1, keepdims=True) * (b / np.linalg.norm(b, axis=1, keepdims=True))).sum(1)
            cache_cos[f"{split}_{mod}"] = {
                "n_rows": int(m.sum()), "mean": float(c.mean()), "min": float(c.min()),
                "p0_1": float(np.quantile(c, 0.001)), "p1": float(np.quantile(c, 0.01)),
                "n_rows_below_0_999": int((c < 0.999).sum()),
                "n_paintings_below_0_999": int(len(np.unique(np.asarray(data.paintings)[rr[c < 0.999]])))}

    per_seed, sides = {}, {}
    for s in SEEDS:
        ep, pair_index = load_episodes(s, sel)
        cl = sp.groups[ep.anchor]
        scorers, side = {}, {}
        plain, side["plain"] = cosine_metrics(*frozen, ep)
        stored_plain = np.load(E1 / f"per_anchor_seed{s}.npz")
        assert all(np.array_equal(plain[m], stored_plain[f"cosine__{m}"]) for m in ("r1", "other"))
        scorers["plain"] = with_derived(plain)
        src = STORED["42"] if s == 42 else STORED["49-51"]
        for name, (path, prefix) in src.items():
            z = np.load(str(path).format(s=s))
            assert np.array_equal(z["cl"], cl) and np.array_equal(z["pair_index"], pair_index), (s, name)
            assert np.array_equal(z["cosine__r1"], plain["r1"]), (s, name)
            scorers[name] = with_derived({"r1": z[f"{prefix}__r1"], "other": z[f"{prefix}__other"]})
            assert np.allclose(scorers[name]["gain"], z[f"{prefix}__gain"], atol=1e-12)
        for name, (fi, ft) in feats.items():
            pa, side[name] = cosine_metrics(fi, ft, ep)
            scorers[name] = with_derived(pa)
        per_seed[s] = {"cl": cl, "pair_index": pair_index, "scorers": scorers}
        sides[s] = side
    return per_seed, sides, sources, cache_cos


def scope_values(per_seed, seeds, mask_pair=None):
    cl = np.concatenate([per_seed[s]["cl"] for s in seeds])
    pi = np.concatenate([per_seed[s]["pair_index"] for s in seeds])
    mask = np.ones(len(cl), bool) if mask_pair is None else pi == mask_pair
    names = [n for n in per_seed[seeds[0]]["scorers"] if all(n in per_seed[s]["scorers"] for s in seeds)]
    vals = {n: {m: np.concatenate([per_seed[s]["scorers"][n][m] for s in seeds])[mask] for m in METRICS}
            for n in names}
    return vals, cl[mask]


def comparisons(vals, variant):
    ft = f"ft:{variant}"
    pick = lambda t: ft if t == "ft" else t  # noqa: E731
    return {c: {m: vals[pick(a)][m] - vals[pick(b)][m] for m in METRICS} for c, (a, b) in COMPARISONS.items()}


def boot(values, clusters):
    r = cluster_bootstrap(np.asarray(values, dtype=np.float64), clusters)
    return {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]], "n_clusters": r["n_clusters"]}


def assert_points(rep, per_seed):
    """Every point in eval.json (scorers and comparisons, all metrics, every scope) equals the re-derived one."""
    scopes = [(rep["seeds"][str(s)], (s,), None) for s in SEEDS] + [(rep["pooled"], POOLED, None)]
    scopes += [(rep["pairs"][str(i)], POOLED, i) for i in range(3)]
    scopes += [(rep["pairs_seed42"][str(i)], (42,), i) for i in range(3)]
    n, worst = 0, 0.0
    for block, seeds, pair in scopes:
        vals, cl = scope_values(per_seed, seeds, pair)
        assert block["n_episodes"] == len(cl)
        assert set(block["scorers"]) == set(vals), (seeds, pair, set(block["scorers"]) ^ set(vals))
        for name, ms in block["scorers"].items():
            for m in METRICS:
                d = abs(100 * vals[name][m].mean() - ms[m]["point"])
                worst, n = max(worst, d), n + 1
                assert d < TOL, (seeds, pair, name, m, d)
        for v, comps in block["comparisons"].items():
            mine = comparisons(vals, v)
            for c, ms in comps.items():
                for m in METRICS:
                    d = abs(100 * mine[c][m].mean() - ms[m]["point"])
                    worst, n = max(worst, d), n + 1
                    assert d < TOL, (seeds, pair, v, c, m, d)
    return n, worst


def assert_interval(mine, theirs, what):
    assert abs(mine["point"] - theirs["point"]) < TOL, what
    assert all(abs(a - b) < TOL for a, b in zip(mine["ci95"], theirs["ci95"])), (what, mine, theirs)
    assert mine["n_clusters"] == theirs["n_clusters"], what


# ----- 3. figures ----------------------------------------------------------------------------------------------------

def fig_curves(runs, selected, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    plain_frozen = 100 * runs[("LP", "1e-4")]["selection"][0]
    for ax, v in zip(axes, VARIANTS):
        lrs = [lr for vv, lr, _, _ in RUNS if vv == v]
        for col, lr in zip(C_FT_RAMP, lrs):
            r = runs[(v, lr)]
            ax.plot(r["epochs"], [100 * x for x in r["selection"]], color=col, lw=2, marker="o", ms=3.5,
                    label=f"lr {lr}")
        sv = selected[v]
        ax.plot([sv["epoch"]], [100 * sv["selection"]], marker="*", ms=15, color=C_FT_RAMP[2], mec="white", mew=1.2,
                zorder=5, ls="none")
        ax.annotate(f"selected: lr {sv['lr']}, epoch {sv['epoch']} ({100 * sv['selection']:.2f})",
                    (sv["epoch"], 100 * sv["selection"]), xytext=(0.22, 0.93), textcoords="axes fraction",
                    fontsize=8, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
        ax.axhline(plain_frozen, color=C_REF, ls="--", lw=1.2)
        ax.set_title({"LP": "linear probe (LP)", "LB": "last block (LB)", "LoRA": "LoRA"}[v])
        ax.set_xlabel("epoch (0 = untrained)")
        ax.set_xticks(range(0, 11, 2))
        ax.set_ylim(6, 17.6)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("val selection metric (%)")
    for ax in axes:
        ax.text(1.3, plain_frozen + 0.2, "plain CLIP (epoch 0)", color=INK2, fontsize=7.5, ha="left")
    fig.suptitle("Every run beat plain CLIP on val retrieval; LoRA reached the highest selection metric",
                 fontsize=10.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


LADDER = [("plain", "plain CLIP", C_REF), ("ft:LP", "LP", C_FT), ("ft:LB", "LB", C_FT), ("ft:LoRA", "LoRA", C_FT),
          ("B", "B", C_REF_DARK), ("Bp0", "B′(A0)", C_REF_DARK), ("AFF", "AFF", C_AFF)]


def fig_ladder(pooled_ci, path):
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    xs = np.arange(len(LADDER))
    for x, (key, label, col) in zip(xs, LADDER):
        r = pooled_ci[key]
        ax.bar(x, r["point"], width=0.62, color=col, edgecolor="white", linewidth=2)
        ax.errorbar(x, r["point"], yerr=[[r["point"] - r["ci95"][0]], [r["ci95"][1] - r["point"]]], color=INK,
                    capsize=3, lw=1)
        ax.text(x, r["ci95"][1] + 0.25, f"{r['point']:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(xs, [lab for _, lab, _ in LADDER])
    ax.set_ylabel("R@1 (%), pooled seeds 49 to 51")
    ax.set_ylim(0, 21.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(100 / 13, color=C_REF, ls=":", lw=1)
    ax.set_xlim(-1.05, len(LADDER) - 0.5)
    ax.text(-1.0, 100 / 13 + 0.2, "chance\n7.69", color=INK2, fontsize=7.5, ha="left", va="bottom")
    ax.legend(handles=[Patch(color=C_REF, label="backbone only (frozen CLIP)"),
                       Patch(color=C_FT, label="fine-tuned CLIP, cosine (this report)"),
                       Patch(color=C_REF_DARK, label="condition-free fusions (round 3)"),
                       Patch(color=C_AFF, label="AFF (reads the condition)")],
              loc="upper left", fontsize=7.8)
    ax.set_title("Fine-tuning recovered about 2 of the 5.2 points between plain CLIP and B′(A0)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


PER_PAIR = [("plain", "plain CLIP", C_REF, None), ("ft:LP", "LP", C_FT, None), ("ft:LB", "LB", C_FT, "////"),
            ("ft:LoRA", "LoRA", C_FT, "...."), ("Bp0", "B′(A0)", C_REF_DARK, None), ("AFF", "AFF", C_AFF, None)]


def fig_pairs(pair_ci, path):
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    w = 0.13
    for j, pk in enumerate(PAIR_KEYS):
        for k, (key, label, col, hatch) in enumerate(PER_PAIR):
            r = pair_ci[pk][key]
            x = j + (k - (len(PER_PAIR) - 1) / 2) * w
            ax.bar(x, r["point"], width=w * 0.92, color=col, hatch=hatch, edgecolor="white", linewidth=1,
                   label=label if j == 0 else None)
            ax.errorbar(x, r["point"], yerr=[[r["point"] - r["ci95"][0]], [r["ci95"][1] - r["point"]]], color=INK,
                        capsize=2, lw=0.9)
        lift = np.mean([pair_ci[pk][f"ft:{v}"]["point"] for v in VARIANTS]) - pair_ci[pk]["plain"]["point"]
        ax.text(j, 25.2, f"fine-tuned − plain: +{lift:.2f} (mean of 3)", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(3), [PAIR_LABEL[p] for p in PAIR_KEYS])
    ax.set_ylabel("R@1 (%), pooled seeds 49 to 51")
    ax.set_ylim(0, 27)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    ax.set_title("Fine-tuning lifted the genre pairs most and emotion × style least", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


DECOMP = [("plain", "plain CLIP"), ("ft:LP", "LP"), ("ft:LB", "LB"), ("ft:LoRA", "LoRA"), ("B", "B"),
          ("Bp0", "B′(A0)")]


def fig_decomp(decomp, path):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ys = np.arange(len(DECOMP))[::-1]
    for y, (key, label) in zip(ys, DECOMP):
        d = decomp[key]
        e, g = d["half_either"], d["half_gain"]
        # either part from 0; gain part stacked after it (to the right of 0 when the either part is negative)
        ax.barh(y, e, height=0.55, color=C_EITHER, edgecolor="white", linewidth=2)
        start = e if e > 0 else 0.0
        ax.barh(y, g, left=start, height=0.55, color=C_GAIN, edgecolor="white", linewidth=2)
        ax.plot([d["total"]], [y], marker="|", ms=16, mew=2.2, color=INK, ls="none")
        ax.text(max(start + g, 0) + 0.1, y, f"net {d['total']:+.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys, [f"AFF − {lab}" for _, lab in DECOMP])
    ax.set_xlabel("R@1 difference (points), pooled seeds 49 to 51")
    ax.set_xlim(-1.4, 6.6)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=C_EITHER, label="½ × either-rate difference"),
                       Patch(color=C_GAIN, label="½ × condition gain of AFF (3.32)"),
                       Line2D([], [], marker="|", ms=12, mew=2, color=INK, ls="none", label="net R@1 margin")],
              loc="lower right", fontsize=8)
    ax.set_title("Against fine-tuned CLIP, more than half of AFF's margin is either rate", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ----- main ----------------------------------------------------------------------------------------------------------

def main():
    assert sha256(EVAL) == EVAL_SHA, "eval.json changed since the report was written"
    rep = json.loads(EVAL.read_text())
    runs_txt = check_runs_txt()
    runs, selected = training_curves()
    per_seed, sides, sources, cache_cos = episodes_block()
    n_points, worst = assert_points(rep, per_seed)

    # intervals of the plotted quantities, re-derived and asserted against eval.json
    vals, cl = scope_values(per_seed, POOLED)
    pooled_ci = {}
    for key in [k for k, _, _ in LADDER] + ["ft:CLIPcache"]:
        pooled_ci[key] = boot(vals[key]["r1"], cl)
        assert_interval(pooled_ci[key], rep["pooled"]["scorers"][key]["r1"], ("pooled", key))
    pooled_cmp = {}
    for v in VARIANTS + ("CLIPcache",):
        comps = comparisons(vals, v)
        pooled_cmp[v] = {}
        for c in COMPARISONS:
            for m in ("r1", "either"):
                pooled_cmp[v][f"{c}__{m}"] = boot(comps[c][m], cl)
                assert_interval(pooled_cmp[v][f"{c}__{m}"], rep["pooled"]["comparisons"][v][c][m], (v, c, m))
    pair_ci = {}
    for i, pk in enumerate(PAIR_KEYS):
        pv, pcl = scope_values(per_seed, POOLED, i)
        pair_ci[pk] = {}
        for key, *_ in PER_PAIR:
            pair_ci[pk][key] = boot(pv[key]["r1"], pcl)
            assert_interval(pair_ci[pk][key], rep["pairs"][str(i)]["scorers"][key]["r1"], (pk, key))
        pair_ci[pk]["B"] = {"point": 100 * pv["B"]["r1"].mean()}

    # R@1 = (either + gain) / 2, so AFF minus a condition-free scorer = half the either difference + half AFF's gain
    decomp = {}
    for key, _ in DECOMP:
        de = 100 * (vals["AFF"]["either"].mean() - vals[key]["either"].mean())
        dg = 100 * (vals["AFF"]["gain"].mean() - vals[key]["gain"].mean())
        tot = 100 * (vals["AFF"]["r1"].mean() - vals[key]["r1"].mean())
        assert abs(0.5 * (de + dg) - tot) < TOL
        decomp[key] = {"either_diff": de, "gain_diff": dg, "half_either": de / 2, "half_gain": dg / 2, "total": tot,
                       "either_share": (de / 2) / tot}

    # per-side first-place rates of the condition-free cosines (pooled 49-51), points only
    side_rates = {}
    for i, pk in enumerate(PAIR_KEYS):
        a_name, b_name = pk.split("__")
        side_rates[pk] = {}
        for key in ("plain", "ft:CLIPcache") + tuple(f"ft:{v}" for v in VARIANTS):
            pi = np.concatenate([per_seed[s]["pair_index"] for s in POOLED]) == i
            A = 100 * np.concatenate([sides[s][key]["A"] for s in POOLED])[pi].mean()
            Bv = 100 * np.concatenate([sides[s][key]["B"] for s in POOLED])[pi].mean()
            side_rates[pk][key] = {f"{a_name}_candidate_first": A, f"{b_name}_candidate_first": Bv,
                                   "r1": 0.5 * (A + Bv)}
            assert abs(0.5 * (A + Bv) - rep["pairs"][str(i)]["scorers"][key]["r1"]["point"]) < TOL
        for v in VARIANTS:
            side_rates[pk][f"lift_{v}"] = {s_: side_rates[pk][f"ft:{v}"][s_] - side_rates[pk]["plain"][s_]
                                          for s_ in (f"{a_name}_candidate_first", f"{b_name}_candidate_first")}

    # grid edges against the gap to B'(A0): val-selection points per R@1 point, and the edges' last steps
    edges = {}
    for v in VARIANTS:
        lr, ep = EXPECTED_SELECTION[v]
        sel_pts = 100 * selected[v]["selection"]
        base_pts = 100 * runs[(v, lr)]["selection"][0]
        lift = rep["pooled"]["comparisons"][v]["ft_minus_plain"]["r1"]["point"]
        edges[v] = {"val_gain_points_over_epoch0": sel_pts - base_pts, "r1_lift_over_plain": lift,
                    "r1_per_val_point": lift / (sel_pts - base_pts),
                    "gap_to_Bp0": -rep["pooled"]["comparisons"][v]["ft_minus_Bp0"]["r1"]["point"]}
    lb = {lr: runs[("LB", lr)]["best_selection"] for lr in ("3e-6", "1e-5", "3e-5")}
    lora = runs[("LoRA", "1e-4")]["selection"]
    steep = max(e["r1_per_val_point"] for e in edges.values())
    edges["LB_last_lr_step_val_points"] = 100 * (lb["3e-5"] - lb["1e-5"])
    edges["LB_previous_lr_step_val_points"] = 100 * (lb["1e-5"] - lb["3e-6"])
    edges["LoRA_last_epoch_step_val_points"] = 100 * (lora[10] - lora[9])
    edges["LoRA_epochs_8_to_10_val"] = [100 * x for x in lora[8:11]]
    edges["steepest_r1_per_val_point"] = steep
    edges["LB_last_step_r1_if_repeated"] = steep * edges["LB_last_lr_step_val_points"]
    edges["LoRA_last_step_r1_if_repeated"] = steep * edges["LoRA_last_epoch_step_val_points"]
    edges["val_points_needed_to_close_smallest_gap_at_steepest_rate"] = \
        min(e["gap_to_Bp0"] for k, e in edges.items() if k in VARIANTS) / steep
    lp_r1, lora_r1 = (rep["pooled"]["scorers"][f"ft:{v}"]["r1"]["point"] for v in ("LP", "LoRA"))
    edges["cross_variant_r1_per_val_point_LoRA_vs_LP"] = (lora_r1 - lp_r1) / (
        100 * (selected["LoRA"]["selection"] - selected["LP"]["selection"]))
    edges["note"] = ("linear extrapolation at the steepest observed ratio of episode R@1 lift to val-selection lift; "
                     "an upper-side illustration, not a measurement")

    # share of the plain-to-B'(A0) gap that each fine-tune recovers (pooled), and the seed-42 gap to B'(A1)
    ps = rep["pooled"]["scorers"]
    gap_share = {v: (ps[f"ft:{v}"]["r1"]["point"] - ps["plain"]["r1"]["point"])
                 / (ps["Bp0"]["r1"]["point"] - ps["plain"]["r1"]["point"]) for v in VARIANTS}
    s42 = rep["seeds"]["42"]["scorers"]
    bp1_minus_ft_seed42 = {v: s42["Bp1"]["r1"]["point"] - s42[f"ft:{v}"]["r1"]["point"] for v in VARIANTS}
    # what a condition-free scorer's either rate must be to match AFF's pooled R@1 (R@1 = either / 2)
    either_needed = 2 * ps["AFF"]["r1"]["point"]

    fig_curves(runs, selected, OUT / "val_curves.png")
    fig_ladder(pooled_ci, OUT / "ladder_pooled.png")
    fig_pairs(pair_ci, OUT / "per_pair.png")
    fig_decomp(decomp, OUT / "decomposition.png")

    out = {
        "generated_by": "docs/reports/assets/2026-11-24_clip_lightweight_ft/build_figures.py",
        "eval_json": {"path": str(EVAL.relative_to(ROOT)), "sha256": EVAL_SHA},
        "runs_txt_check": runs_txt,
        "eval_points_compared": n_points, "eval_points_max_abs_diff_pp": worst,
        "feature_sources": sources,
        "runs": {f"{v}_lr{lr}": r for (v, lr), r in runs.items()},
        "selected": selected,
        "pooled_r1_ci": pooled_ci,
        "pooled_comparisons_ci": pooled_cmp,
        "per_pair_r1_ci": pair_ci,
        "decomposition_AFF_minus": decomp,
        "per_side_first_place_rates_pooled": side_rates,
        "cache_path_vs_frozen_cosine": cache_cos,
        "grid_edges": edges,
        "gap_share_recovered_plain_to_Bp0": gap_share,
        "seed42_Bp1_minus_ft_r1": bp1_minus_ft_seed42,
        "either_needed_by_a_condition_free_scorer_to_match_AFF_r1": either_needed,
        "figures": {"val_curves.png": "val selection metric per epoch for the nine runs; selected models starred",
                    "ladder_pooled.png": "pooled R@1 (seeds 49-51) with 95% painting-bootstrap intervals",
                    "per_pair.png": "per-pair R@1 (pooled 49-51) with 95% intervals",
                    "decomposition.png": "AFF minus each condition-free scorer = half either difference + half gain"},
    }
    (OUT / "figure_data.json").write_text(json.dumps(out, indent=1))
    print(f"OK: {n_points} eval.json points re-derived (max abs diff {worst:.2e} pp); "
          f"{len(pooled_ci) + sum(len(x) for x in pooled_cmp.values()) + 3 * len(PER_PAIR)} intervals asserted; selection {selected}; figures written to {OUT}")


if __name__ == "__main__":
    main()
