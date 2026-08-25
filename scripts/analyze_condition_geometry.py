"""
Post-hoc condition-embedding geometry diagnostic (Experiment 11.1, spec
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md S4).

Retrieval numbers (test_oracle/*_R1) can miss a real difference between the frozen and
trained arms of Experiment 11.1 -- this script inspects the actual embedding geometry
instead: how much does conditioning shift the combine-side embedding, how does that shift
distribution compare across epochs/arms, which samples are moved the most/least, and --
via a condition-vs-text cross grid -- whether conditions are interchangeable/null for a
given text (low diversity across conditions) or one condition dominates and collapses
every text to nearly the same output (low diversity across texts). Retrieval numbers alone
cannot distinguish those two failure modes from each other or from a healthy grid.

Two modes:
  --exp-dir PATH   analyze one run's condition_viz/ snapshots, write
                   condition_geometry/summary.json + a plot inside that run's directory.
  --compare A B    load two already-produced summary.json files (e.g. a frozen run and a
                   trained run, same seed) and print a paired diff.
  --selftest       offline arithmetic check of the pure math helpers, no data needed.

Usage:
  python scripts/analyze_condition_geometry.py --exp-dir <run_dir>
  python scripts/analyze_condition_geometry.py --compare <frozen_run_dir> <trained_run_dir>
  python scripts/analyze_condition_geometry.py --selftest
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

_REDCAPS_BUDDY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "test", "20260623_redcaps_buddy"
)
if _REDCAPS_BUDDY_DIR not in sys.path:
    sys.path.insert(0, _REDCAPS_BUDDY_DIR)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.regularizer import reorder_features_to_z
from src.model.combiner import Combiner_new


def compute_shift(comb_emb: torch.Tensor, unconditioned_emb: torch.Tensor) -> np.ndarray:
    """Per-sample 1 - cos(conditioned, unconditioned). Both [N, D]; neither need be
    pre-normalized (this normalizes both)."""
    cond_n = F.normalize(comb_emb, dim=-1)
    uncond_n = F.normalize(unconditioned_emb, dim=-1)
    return (1.0 - (cond_n * uncond_n).sum(dim=-1)).cpu().numpy()


def effective_dims(x: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Number of PCA components needed to explain >= variance_threshold of variance.
    x: [N, D]. Falls back to D if N <= D (PCA undefined)."""
    n, d = x.shape
    if n <= d:
        return d
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    var = s ** 2
    ratio = var / var.sum()
    cumsum = np.cumsum(ratio)
    return int(np.argmax(cumsum >= variance_threshold) + 1)


def pairwise_sim_spread(x: np.ndarray, n_sample: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean/std of pairwise cosine similarity over a random subsample (full N^2 is wasteful
    at N~120k). x rows need not be pre-normalized."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    xs = x[idx]
    xs = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
    sims = xs @ xs.T
    iu = np.triu_indices(len(idx), k=1)
    off = sims[iu]
    return {"mean": float(off.mean()), "std": float(off.std())}


def rank_most_least_changed(shift: np.ndarray, sample_ids: List[int], k: int = 20) -> Dict[str, List[Dict]]:
    """Top-k / bottom-k samples by shift magnitude, paired with their sample id."""
    order = np.argsort(shift)  # ascending: least-changed first
    least = [{"sample_id": int(sample_ids[i]), "shift": float(shift[i])} for i in order[:k]]
    most = [{"sample_id": int(sample_ids[i]), "shift": float(shift[i])} for i in order[::-1][:k]]
    return {"most_changed": most, "least_changed": least}


def correlate_shift(shift: np.ndarray, other: np.ndarray) -> Dict[str, float]:
    """Pearson r between per-sample shift and another per-sample scalar (condition norm,
    buddy-graph degree, ...). Returns r=0/p=1 if either array has ~zero variance."""
    if shift.std() < 1e-8 or other.std() < 1e-8 or len(shift) < 2:
        return {"r": 0.0, "p": 1.0}
    r, p = pearsonr(shift, other)
    return {"r": float(r), "p": float(p)}


def grid_diversity(comb_grid: torch.Tensor) -> Dict[str, np.ndarray]:
    """comb_grid[i, j] = combine(text_i, condition_j), shape [n_text, n_cond, D]. Splits
    diversity two ways to distinguish two collapse failure modes retrieval numbers can't
    tell apart:

      row_diversity[i] = 1 - mean pairwise cosine sim among {comb_grid[i, j] for all j}
                         (low => for text i, varying the condition barely changes the
                         output => conditions are interchangeable/null for this text)
      col_diversity[j] = 1 - mean pairwise cosine sim among {comb_grid[i, j] for all i}
                         (low => for condition j, varying the text barely changes the
                         output => condition j dominates/collapses the combination)

    Returns {"row_diversity": np.ndarray [n_text], "col_diversity": np.ndarray [n_cond]}.
    NaN for a row/column of length < 2 (nothing to compare pairwise).
    """
    n_text, n_cond, _ = comb_grid.shape
    g = F.normalize(comb_grid, dim=-1)

    row_diversity = np.full(n_text, np.nan)
    iu_cond = torch.triu_indices(n_cond, n_cond, offset=1)
    if iu_cond.shape[1] > 0:
        for i in range(n_text):
            sims = g[i] @ g[i].T
            row_diversity[i] = float(1.0 - sims[iu_cond[0], iu_cond[1]].mean())

    col_diversity = np.full(n_cond, np.nan)
    iu_text = torch.triu_indices(n_text, n_text, offset=1)
    if iu_text.shape[1] > 0:
        for j in range(n_cond):
            col = g[:, j]
            sims = col @ col.T
            col_diversity[j] = float(1.0 - sims[iu_text[0], iu_text[1]].mean())

    return {"row_diversity": row_diversity, "col_diversity": col_diversity}


def _load_redcaps_train_features():
    """Frozen CLIP (img, txt) features + sample_ids for RedCaps-150k, in FeatureManager's
    own row order. Scope is RedCaps-150k only per spec S4 Experiment 11.1."""
    import redcaps_buddy as rb
    data = rb.load_data()
    return data.img, data.txt, data.sample_ids


def _rebuild_combiner(epoch_snapshot: dict) -> Combiner_new:
    cfg = epoch_snapshot["combiner_config"]
    combiner = Combiner_new(
        clip_feature_dim=cfg["clip_feature_dim"],
        projection_dim=cfg["projection_dim"],
        label_dim=cfg["label_dim"],
        hidden_dim=512,  # unused by Combiner_new's forward; harmless placeholder
        num_heads=8,     # unused by Combiner_new's forward; harmless placeholder
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    combiner.load_state_dict(epoch_snapshot["combiner_state_dict"])
    combiner.eval()
    return combiner


def _compute_comb_emb(combiner: Combiner_new, text_feat: torch.Tensor, conditions: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Chunked forward through the combiner. text_feat/conditions: [N, *], row-aligned."""
    n = text_feat.shape[0]
    out = None
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            c = combiner(text_feat[s:e], None, conditions[s:e])
            if out is None:
                out = torch.empty((n,) + c.shape[1:], dtype=c.dtype)
            out[s:e] = c
    return out


def compute_condition_text_grid(combiner: Combiner_new, text_sample: torch.Tensor, cond_sample: torch.Tensor) -> torch.Tensor:
    """comb_grid[i, j] = combine(text_sample[i], cond_sample[j]). text_sample: [n_text, D],
    cond_sample: [n_cond, D_cond]. Small grid (n_text x n_cond combiner calls, batched over
    j per i) -- feeds grid_diversity's null-condition-vs-dominant-condition check."""
    n_text = text_sample.shape[0]
    n_cond = cond_sample.shape[0]
    out = None
    with torch.no_grad():
        for i in range(n_text):
            t_rep = text_sample[i : i + 1].expand(n_cond, -1)
            c = combiner(t_rep, None, cond_sample)  # [n_cond, D]
            if out is None:
                out = torch.empty((n_text, n_cond) + c.shape[1:], dtype=c.dtype)
            out[i] = c
    return out


def analyze_run(exp_dir: str, k_ranked: int = 20, n_text_sample: int = 30, n_cond_sample: int = 30, grid_seed: int = 0) -> dict:
    """Analyze one run's condition_viz/ snapshots. Writes condition_geometry/summary.json
    and a shift-trajectory plot inside exp_dir. Returns the summary dict."""
    exp_path = Path(exp_dir)
    cond_viz_dir = exp_path / "condition_viz"
    epoch_files = sorted(cond_viz_dir.glob("epoch_*.pt"))
    if not epoch_files:
        raise FileNotFoundError(f"no condition_viz/epoch_*.pt snapshots under {exp_dir}")

    img_np, txt_np, feat_sample_ids = _load_redcaps_train_features()
    img_t = torch.from_numpy(np.ascontiguousarray(img_np)).float()
    txt_t = torch.from_numpy(np.ascontiguousarray(txt_np)).float()

    edges_path = exp_path / "training_embeddings" / "buddy_edges.npy"
    buddy_edges = np.load(edges_path) if edges_path.exists() else None

    rng = np.random.default_rng(grid_seed)

    # Condition-vs-text cross grid: sample the text/condition indices ONCE per run (not
    # once per epoch), from the first snapshot's N -- N is constant across a run's epochs
    # (asserted below) -- so the same fixed indices are reused every epoch and the
    # row/col-diversity trajectory in _plot_trajectory reflects real training dynamics
    # rather than resampling noise.
    n_ref = torch.load(epoch_files[0], map_location="cpu")["label_embeddings_all"].shape[0]
    text_idx = rng.choice(n_ref, size=min(n_text_sample, n_ref), replace=False)
    cond_idx = rng.choice(n_ref, size=min(n_cond_sample, n_ref), replace=False)

    per_epoch = []
    for ef in epoch_files:
        snap = torch.load(ef, map_location="cpu")
        epoch = snap["epoch"]
        conditions = snap["label_embeddings_all"]  # [N, D]
        sample_ids = snap["sample_ids"]             # [N], added in Task 1
        n = conditions.shape[0]
        assert n == n_ref, (
            f"epoch {epoch} has N={n} samples but the run's first epoch had N={n_ref}; "
            f"the fixed text/condition grid indices assume constant N across a run's epochs"
        )

        combine_side = snap.get("combine_side", "txt")
        raw_feat = img_t if combine_side == "img" else txt_t
        combine_feat = reorder_features_to_z(raw_feat, feat_sample_ids, sample_ids)

        combiner = _rebuild_combiner(snap)
        comb_emb = _compute_comb_emb(combiner, combine_feat, conditions)

        shift = compute_shift(comb_emb, combine_feat)
        cond_np = conditions.numpy()
        comb_np = comb_emb.numpy()
        raw_np = F.normalize(combine_feat, dim=-1).numpy()

        cond_norm = np.linalg.norm(cond_np, axis=1)
        norm_corr = correlate_shift(shift, cond_norm)

        degree_corr = {"r": None, "p": None}
        if buddy_edges is not None:
            degree = np.bincount(buddy_edges.flatten(), minlength=n).astype(float)
            degree_corr = correlate_shift(shift, degree)

        # text_idx/cond_idx were sampled once, before the loop, and reused unchanged here.
        comb_grid = compute_condition_text_grid(combiner, combine_feat[text_idx], conditions[cond_idx])
        diversity = grid_diversity(comb_grid)

        per_epoch.append({
            "epoch": int(epoch),
            "n_samples": int(n),
            "shift_mean": float(shift.mean()),
            "shift_std": float(shift.std()),
            "shift_p10": float(np.percentile(shift, 10)),
            "shift_p90": float(np.percentile(shift, 90)),
            "conditioned_effective_dims": effective_dims(comb_np),
            "unconditioned_effective_dims": effective_dims(raw_np),
            "condition_effective_dims": effective_dims(cond_np),
            "conditioned_pairwise_sim": pairwise_sim_spread(comb_np),
            "unconditioned_pairwise_sim": pairwise_sim_spread(raw_np),
            "shift_vs_condition_norm": norm_corr,
            "shift_vs_buddy_degree": degree_corr,
            "ranked": rank_most_least_changed(shift, sample_ids, k=k_ranked),
            "grid_diagnostic": {
                "n_text_sample": int(len(text_idx)),
                "n_cond_sample": int(len(cond_idx)),
                "row_diversity_mean": float(np.nanmean(diversity["row_diversity"])),
                "row_diversity_min": float(np.nanmin(diversity["row_diversity"])),
                "col_diversity_mean": float(np.nanmean(diversity["col_diversity"])),
                "col_diversity_min": float(np.nanmin(diversity["col_diversity"])),
            },
        })

    out_dir = exp_path / "condition_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"exp_dir": str(exp_path), "per_epoch": per_epoch}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_trajectory(per_epoch, exp_path / "plots" / "condition_geometry_trajectory.png")
    print(f"Wrote {out_dir / 'summary.json'} ({len(per_epoch)} epochs)")
    return summary


def _plot_trajectory(per_epoch: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [e["epoch"] for e in per_epoch]
    shift_mean = [e["shift_mean"] for e in per_epoch]
    shift_std = [e["shift_std"] for e in per_epoch]
    eff_dims = [e["conditioned_effective_dims"] for e in per_epoch]
    row_div = [e["grid_diagnostic"]["row_diversity_mean"] for e in per_epoch]
    col_div = [e["grid_diagnostic"]["col_diversity_mean"] for e in per_epoch]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].errorbar(epochs, shift_mean, yerr=shift_std, marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("shift = 1 - cos(conditioned, unconditioned)")
    axes[0].set_title("Conditioning shift over training")

    axes[1].plot(epochs, eff_dims, marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("PCA effective dims (95% var)")
    axes[1].set_title("Conditioned-embedding effective dimensionality")

    axes[2].plot(epochs, row_div, marker="o", label="row (across conditions, per text)")
    axes[2].plot(epochs, col_div, marker="s", label="col (across texts, per condition)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("mean diversity (1 - mean pairwise cos sim)")
    axes[2].set_title("Grid diagnostic: low row => conditions null; low col => condition dominates")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compare_runs(dir_a: str, dir_b: str) -> None:
    """Print a paired epoch-by-epoch diff between two already-analyzed runs (e.g. frozen vs
    trained, same seed). Requires both to already have condition_geometry/summary.json (run
    analyze_run on each first)."""
    with open(Path(dir_a) / "condition_geometry" / "summary.json") as f:
        a = json.load(f)
    with open(Path(dir_b) / "condition_geometry" / "summary.json") as f:
        b = json.load(f)

    a_by_epoch = {e["epoch"]: e for e in a["per_epoch"]}
    b_by_epoch = {e["epoch"]: e for e in b["per_epoch"]}
    common = sorted(set(a_by_epoch) & set(b_by_epoch))
    if not common:
        print("No overlapping epochs between the two runs.")
        return

    print(f"\n{'='*78}\nCondition geometry comparison\n  A: {dir_a}\n  B: {dir_b}\n{'='*78}")
    for ep in common:
        ea, eb = a_by_epoch[ep], b_by_epoch[ep]
        d_mean = eb["shift_mean"] - ea["shift_mean"]
        d_dims = eb["conditioned_effective_dims"] - ea["conditioned_effective_dims"]
        ids_a = {r["sample_id"] for r in ea["ranked"]["most_changed"]}
        ids_b = {r["sample_id"] for r in eb["ranked"]["most_changed"]}
        overlap = len(ids_a & ids_b) / max(len(ids_a | ids_b), 1)
        d_row = eb["grid_diagnostic"]["row_diversity_mean"] - ea["grid_diagnostic"]["row_diversity_mean"]
        d_col = eb["grid_diagnostic"]["col_diversity_mean"] - ea["grid_diagnostic"]["col_diversity_mean"]
        print(f"  epoch {ep:>4}: shift_mean B-A={d_mean:+.4f}  eff_dims B-A={d_dims:+d}  "
              f"most-changed-set Jaccard(A,B)={overlap:.2f}  "
              f"row_div B-A={d_row:+.4f}  col_div B-A={d_col:+.4f}")


def _selftest():
    torch.manual_seed(0)
    # compute_shift: identical vectors -> shift 0; orthogonal -> shift 1.
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    s = compute_shift(a, b)
    assert abs(s[0] - 0.0) < 1e-6, s
    assert abs(s[1] - 1.0) < 1e-6, s

    # effective_dims: a rank-1 signal embedded in D=5 needs 1 component for 95% variance.
    rng = np.random.default_rng(0)
    direction = rng.normal(size=5)
    x = np.outer(rng.normal(size=500), direction) + rng.normal(scale=1e-4, size=(500, 5))
    assert effective_dims(x) == 1, effective_dims(x)

    # pairwise_sim_spread: identical rows -> mean sim == 1, std == 0.
    same = np.tile(rng.normal(size=(1, 8)), (100, 1))
    spread = pairwise_sim_spread(same, n_sample=50)
    assert abs(spread["mean"] - 1.0) < 1e-5, spread
    assert spread["std"] < 1e-5, spread

    # rank_most_least_changed: correct extremes and ids.
    shift = np.array([0.1, 0.9, 0.5, 0.0, 1.0])
    ids = [10, 11, 12, 13, 14]
    ranks = rank_most_least_changed(shift, ids, k=2)
    assert [r["sample_id"] for r in ranks["most_changed"]] == [14, 11], ranks
    assert [r["sample_id"] for r in ranks["least_changed"]] == [13, 10], ranks

    # correlate_shift: perfectly correlated inputs -> r ~= 1.
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    c = correlate_shift(x1, x2)
    assert abs(c["r"] - 1.0) < 1e-6, c

    # grid_diversity: three constructed scenarios covering both failure modes + healthy.
    n_text, n_cond, d = 6, 5, 8
    base_text = F.normalize(torch.randn(n_text, d), dim=-1)
    base_cond = F.normalize(torch.randn(n_cond, d), dim=-1)

    # Case A: output depends only on text, ignores condition entirely -> conditions "null".
    # row_diversity ~ 0 (no variation across j, for each i); col_diversity high (varies across i).
    grid_null_cond = base_text.unsqueeze(1).expand(n_text, n_cond, d).clone()
    ga = grid_diversity(grid_null_cond)
    assert np.nanmax(ga["row_diversity"]) < 1e-5, ga["row_diversity"]
    assert np.nanmean(ga["col_diversity"]) > 0.1, ga["col_diversity"]

    # Case B: output depends only on condition, ignores text entirely -> condition dominates.
    # col_diversity ~ 0 (no variation across i, for each j); row_diversity high (varies across j).
    grid_dominant_cond = base_cond.unsqueeze(0).expand(n_text, n_cond, d).clone()
    gb = grid_diversity(grid_dominant_cond)
    assert np.nanmax(gb["col_diversity"]) < 1e-5, gb["col_diversity"]
    assert np.nanmean(gb["row_diversity"]) > 0.1, gb["row_diversity"]

    # Case C: healthy -- independent random grid, both diversities nontrivially positive.
    grid_healthy = F.normalize(torch.randn(n_text, n_cond, d), dim=-1)
    gc = grid_diversity(grid_healthy)
    assert np.nanmean(gc["row_diversity"]) > 0.3, gc["row_diversity"]
    assert np.nanmean(gc["col_diversity"]) > 0.3, gc["col_diversity"]

    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=None, help="analyze one run's condition_viz/ snapshots")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("DIR_A", "DIR_B"),
                     help="print a paired diff between two already-analyzed run directories")
    ap.add_argument("--k-ranked", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.exp_dir:
        analyze_run(args.exp_dir, k_ranked=args.k_ranked)
        return
    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return
    ap.print_help()


if __name__ == "__main__":
    main()
