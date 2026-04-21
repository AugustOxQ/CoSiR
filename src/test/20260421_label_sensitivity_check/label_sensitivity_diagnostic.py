"""
Diagnostic: does the trained Combiner_new actually use label_features?

Four tests:
  1. Delta magnitude  — ||delta|| vs ||text||, are they in the same ballpark?
  2. Zero-out ablation — cosine similarity of output(label) vs output(zeros)
  3. Permutation test  — cosine similarity of output(label) vs output(shuffled_label)
  4. Gradient attribution — ||∂output/∂label|| vs ||∂output/∂text||, normalised by L2 norm

Usage:
    conda activate CoSiR
    python src/test/20260421_label_sensitivity_check/label_sensitivity_diagnostic.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import torch.nn.functional as F
import numpy as np

from src.model.combiner import Combiner_new

# ─── Config ───────────────────────────────────────────────────────────────────
CKPT_PATH = "res/CoSiR_Experiment/impressions/20260421_013733_CoSiR_Experiment/checkpoints/phase_1_model_20260421014328.pt"
FIXED_DATA_PATH = "res/CoSiR_Experiment/impressions/20260421_013733_CoSiR_Experiment/condition_viz/fixed_data.pt"

LABEL_DIM = 2
CLIP_DIM = 512
NUM_LAYERS = 6   # inferred from checkpoint: 6 Linear layers in label_decoder.network
DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 256   # how many text vectors to use for stats
SWEEP_GRID = 20   # grid size for 2-D label sweep
# ──────────────────────────────────────────────────────────────────────────────


def load_combiner(ckpt_path: str) -> Combiner_new:
    combiner = Combiner_new(
        clip_feature_dim=CLIP_DIM,
        projection_dim=CLIP_DIM,
        label_dim=LABEL_DIM,
        hidden_dim=CLIP_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    combiner.load_state_dict(state)
    combiner.eval()
    return combiner.to(DEVICE)


def load_text_features(fixed_data_path: str, n: int) -> torch.Tensor:
    data = torch.load(fixed_data_path, map_location="cpu")
    txt = data["all_txt_emb"]         # [N, 512]
    txt = F.normalize(txt, dim=-1)    # ensure unit vectors
    return txt[:n].to(DEVICE)


def sep() -> None:
    print("-" * 60)


# ── Test 1: Delta magnitude ────────────────────────────────────────────────────
def test_delta_magnitude(combiner: Combiner_new, txt: torch.Tensor, lbl: torch.Tensor) -> None:
    sep()
    print("TEST 1 — Delta magnitude")
    with torch.no_grad():
        _, delta = combiner(txt, None, lbl, return_delta=True)
    delta_norm = delta.norm(dim=-1)          # [B]
    txt_norm   = txt.norm(dim=-1)            # [B]  (≈1 after normalize)
    ratio      = delta_norm / txt_norm

    print(f"  ||delta|| : mean={delta_norm.mean():.4f}  std={delta_norm.std():.4f}  "
          f"min={delta_norm.min():.4f}  max={delta_norm.max():.4f}")
    print(f"  ||text||  : mean={txt_norm.mean():.4f}")
    print(f"  ratio ||delta||/||text|| : mean={ratio.mean():.4f}  max={ratio.max():.4f}")
    print("  Verdict: ratio < 0.05 → labels mostly ignored; > 0.3 → labels have real influence")


# ── Test 2: Zero-out ablation ──────────────────────────────────────────────────
def test_zero_ablation(combiner: Combiner_new, txt: torch.Tensor, lbl: torch.Tensor) -> None:
    sep()
    print("TEST 2 — Zero-out ablation  (label → zeros)")
    zeros = torch.zeros_like(lbl)
    with torch.no_grad():
        out_real  = combiner(txt, None, lbl)
        out_zeros = combiner(txt, None, zeros)
    cos_sim = F.cosine_similarity(out_real, out_zeros, dim=-1)
    print(f"  cosine_sim(output_real, output_zeros): "
          f"mean={cos_sim.mean():.4f}  std={cos_sim.std():.4f}  "
          f"min={cos_sim.min():.4f}  max={cos_sim.max():.4f}")
    print("  Verdict: cos_sim > 0.999 → labels are ignored; < 0.99 → labels matter")


# ── Test 3: Permutation test ───────────────────────────────────────────────────
def test_permutation(combiner: Combiner_new, txt: torch.Tensor, lbl: torch.Tensor) -> None:
    sep()
    print("TEST 3 — Permutation test  (shuffle labels within batch)")
    perm = torch.randperm(lbl.size(0))
    lbl_shuffled = lbl[perm]
    with torch.no_grad():
        out_real     = combiner(txt, None, lbl)
        out_shuffled = combiner(txt, None, lbl_shuffled)
    cos_sim = F.cosine_similarity(out_real, out_shuffled, dim=-1)
    print(f"  cosine_sim(output_real, output_shuffled): "
          f"mean={cos_sim.mean():.4f}  std={cos_sim.std():.4f}  "
          f"min={cos_sim.min():.4f}  max={cos_sim.max():.4f}")
    print("  Verdict: cos_sim > 0.999 → labels are ignored; < 0.99 → labels matter")


# ── Test 4: Gradient attribution ──────────────────────────────────────────────
def test_gradient_attribution(combiner: Combiner_new, txt: torch.Tensor, lbl: torch.Tensor) -> None:
    sep()
    print("TEST 4 — Gradient attribution  ||∂output/∂input|| / ||input||")

    txt_req  = txt.clone().detach().requires_grad_(True)
    lbl_req  = lbl.clone().detach().requires_grad_(True)

    out = combiner(txt_req, None, lbl_req)      # [B, 512]
    scalar = out.sum()
    scalar.backward()

    grad_txt = txt_req.grad.detach()    # [B, 512]
    grad_lbl = lbl_req.grad.detach()   # [B, 2]

    # Jacobian-vector product norm, normalised by input magnitude
    gnorm_txt = grad_txt.norm(dim=-1) / txt.norm(dim=-1).clamp(min=1e-8)
    gnorm_lbl = grad_lbl.norm(dim=-1) / lbl.norm(dim=-1).clamp(min=1e-8)

    ratio = gnorm_lbl / gnorm_txt.clamp(min=1e-8)

    print(f"  ||∂/∂text||  / ||text|| : mean={gnorm_txt.mean():.4f}  std={gnorm_txt.std():.4f}")
    print(f"  ||∂/∂label|| / ||label||: mean={gnorm_lbl.mean():.4f}  std={gnorm_lbl.std():.4f}")
    print(f"  ratio (label/text sensitivity): mean={ratio.mean():.4f}  max={ratio.max():.4f}")
    print("  Verdict: ratio < 0.1 → network mostly ignores labels; > 0.5 → labels have strong gradient influence")


# ── Test 5: 2-D label sweep on a single text vector ───────────────────────────
def test_label_sweep(combiner: Combiner_new, txt: torch.Tensor) -> None:
    sep()
    print("TEST 5 — 2-D label sweep  (fix one text vector, sweep label over a grid)")

    txt1 = txt[0:1]   # [1, 512]

    # Build a grid of label values spanning [-3, 3] in both dims
    g = torch.linspace(-3, 3, SWEEP_GRID)
    grid_x, grid_y = torch.meshgrid(g, g, indexing="ij")
    labels_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).to(DEVICE)
    B_grid = labels_grid.size(0)
    txt_rep = txt1.expand(B_grid, -1)

    with torch.no_grad():
        out_grid = combiner(txt_rep, None, labels_grid)  # [G*G, 512]

    # Spread: std of output cosine similarities to the first grid point
    ref = out_grid[0:1]
    cos_all = F.cosine_similarity(out_grid, ref.expand_as(out_grid), dim=-1)
    spread_cos = 1 - cos_all    # distance from reference

    print(f"  Output cosine-distance from grid[0,0] across {B_grid} label values:")
    print(f"  mean={spread_cos.mean():.4f}  std={spread_cos.std():.4f}  "
          f"max={spread_cos.max():.4f}")
    print("  Verdict: max < 0.001 → label sweep has no effect; > 0.01 → labels visibly shift the output")

    # Also report unique retrieval results if labels cause ranking changes:
    # compute pairwise std of outputs across grid
    out_np = out_grid.cpu().float().numpy()
    col_std = out_np.std(axis=0)   # std per output dimension
    print(f"  Per-dim output std across grid: mean={col_std.mean():.5f}  max={col_std.max():.5f}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Loading combiner from: {CKPT_PATH}")
    combiner = load_combiner(CKPT_PATH)

    print(f"Loading text features from: {FIXED_DATA_PATH}")
    txt = load_text_features(FIXED_DATA_PATH, N_SAMPLES)
    B = txt.size(0)

    # Generate synthetic labels from a reasonable distribution
    # (unit-normalised 2D → range [-1, 1] roughly)
    torch.manual_seed(42)
    lbl = torch.randn(B, LABEL_DIM, device=DEVICE)

    print(f"\nUsing {B} text vectors, label_dim={LABEL_DIM}")
    sep()

    test_delta_magnitude(combiner, txt, lbl)
    test_zero_ablation(combiner, txt, lbl)
    test_permutation(combiner, txt, lbl)
    test_gradient_attribution(combiner, txt, lbl)
    test_label_sweep(combiner, txt)

    sep()
    print("\nSUMMARY")
    print("If tests 1-5 all show near-zero effect:")
    print("  → The 2D label signal is being swamped by the 512D text features.")
    print("  → Solutions: (a) up-project labels before concatenation,")
    print("                (b) use FiLM conditioning (scale+shift) instead of concat,")
    print("                (c) increase label_dim so it's proportional to clip_dim.")


if __name__ == "__main__":
    main()
