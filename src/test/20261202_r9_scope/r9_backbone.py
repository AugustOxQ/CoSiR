"""Round 9 backbone names and paths; recipe roles are filled by tickets 06/07."""

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


def parts(backbone) -> dict[str, Path]:
    """Role resolver stub: tickets 06 and 07 supply the pinned and recipe roles."""
    assert backbone in BACKBONES, f"unknown backbone: {backbone}"
    raise NotImplementedError("backbone roles are supplied by tickets 06 and 07")
