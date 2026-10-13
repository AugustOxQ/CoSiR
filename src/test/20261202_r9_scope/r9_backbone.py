"""Round 9 backbone names, frozen CLIP inputs and recipe part paths."""

import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[2]
_git = subprocess.run(
    ["git", "-C", str(HERE), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    capture_output=True, text=True, check=True,
)
MAIN = Path(_git.stdout.strip()).resolve().parent
TEST = MAIN / "src/test"
RESULTS = HERE / "results"
assert TEST.is_dir(), f"not CoSiR's main checkout: {MAIN}"

# The editable install otherwise leaves MAIN behind this worktree's cwd.
sys.path[:] = [str(MAIN)] + [p for p in sys.path if not (p and Path(p).resolve() == MAIN)]
for _name, _module in tuple(sys.modules.items()):
    if _name == "src" or _name.startswith("src."):
        _file = getattr(_module, "__file__", None)
        if _file and not Path(_file).resolve().is_relative_to(MAIN / "src"):
            raise ImportError(f"{_name} was imported from {_file}; import r9_backbone before src")

BACKBONES = ("clip_b32", "qwen3vl_emb_2b", "siglip2_so400m")
LEDGER_NAMES = dict(zip(BACKBONES, (
    "CLIP ViT-B/32", "Qwen3-VL-Embedding-2B", "SigLIP 2 so400m/14-384",
)))
DIMENSIONS = dict(zip(BACKBONES, (512, 2048, 1152)))
STORE_PATHS = {
    "clip_b32": Path(os.environ.get("COSIR_ARTELINGO_FEATURES") or
                     "/data/SSD2/pre_extract/artelingo/features"),
    **{b: Path("/data/SSD2/pre_extract/artelingo_full") / b for b in BACKBONES[1:]},
}


# CLIP file names are the shared interface, including both reader formats.
CLIP_PARTS = {
    "prepare": TEST / "20261013_stage_d_selection/cache/prepare.npz",
    "held_codes": TEST / "20261014_stage_d_final/cache/held_codes.npz",
    "partition_L": TEST / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz",
    "told_oracle": TEST / "20261111_community_told_oracle/results/told_oracle.json",
    "step1_group_style": TEST / "20261116_grouping_step1_style/results/step1_group_style.npz",
    "partitions": TEST / "20261031_pseudo_partitions/results/partitions.npz",
    "affect_km": TEST / "20261031_pseudo_partitions/results/partitions.npz",
    "graph": TEST / "20261016_factor_learning_grid/cache/graph.npz",
    "bank_AIC": TEST / "20261031_pseudo_partitions/results/bank_AIC.npz",
    "n6_posteriors": TEST / "20261108_new_method_quick_checks/results/n6_posteriors.npz",
    "step1_heads_style": TEST / "20261116_grouping_step1_style/results/step1_heads_style.npz",
    "A3_seed42": TEST / "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt",
    "picked": TEST / "20261101_aspect_factor_gonogo/results/picked.json",
    "rb_halves": TEST / "20261117_reader_fix_csd/results/rb_halves.npz",
    "rb_reader_A0": TEST / "20261117_reader_fix_csd/results/rb_reader_A0.pkl",
    "rb_reader_A0_json": TEST / "20261117_reader_fix_csd/results/rb_reader_A0.json",
    "rc_tau": TEST / "20261117_reader_fix_csd/results/rc_tau.json",
    "baselines_seed42": TEST / "20261030_aspect_baselines/results/baselines_seed42.json",
    "pm_fit_rows": TEST / "20261030_aspect_baselines/results/baselines_seed42.json",
    "episodes_seed42": TEST / "20261030_aspect_baselines/results/episodes_seed42.npz",
    "picks_seed42": TEST / "20261125_artelingo_held_test/results/picks_seed42.json",
    # Ticket 06 materialises CLIP's constants here; no R6 cells file exists.
    "cells_seed42": RESULTS / "recipe/clip_b32/cells_seed42.json",
    **{f"rb_heads_{group}": TEST / f"20261117_reader_fix_csd/results/rb_heads_{group}.npz"
       for group in ("affect", "image", "caption", "csd", "rand")},
    **{f"rb_bank_A0_half{half}": TEST / f"20261117_reader_fix_csd/results/rb_bank_A0_half{half}.npz"
       for half in (0, 1)},
}
FROZEN_ROLES = frozenset(("prepare", "held_codes", "partition_L", "told_oracle", "affect_km",
                          "step1_group_style", "rb_halves", "pm_fit_rows", "episodes_seed42"))
RECIPE_ROLES = frozenset(CLIP_PARTS) - FROZEN_ROLES


def parts(backbone) -> dict[str, Path]:
    """CLIP's stored parts or the same file names under results/recipe/<backbone>."""
    assert backbone in BACKBONES, f"unknown backbone: {backbone}"
    return {role: (path if backbone == "clip_b32" or role in FROZEN_ROLES
                   else RESULTS / "recipe" / backbone / path.name)
            for role, path in CLIP_PARTS.items()}
