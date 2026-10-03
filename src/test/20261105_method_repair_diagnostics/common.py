"""Shared constants and helpers of the method-repair diagnostics (PREREGISTRATION.md in this folder)."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E3 = ROOT / "src/test/20261101_aspect_factor_gonogo"
sys.path.insert(0, str(E3))
import run_gonogo as rg  # noqa: E402  (puts the repo root on sys.path)

from src.eval.aspect_episodes import AspectEpisodes  # noqa: E402
from src.train.train_factors import R3_CONFIG  # noqa: E402

AFFECT_CACHE = ROOT / "src/test/20261018_affect_factor_learning/cache/affect_prepare.npz"
C0_CKPT = ROOT / "src/test/20261016_factor_learning_grid/checkpoints/C0_seed42.pt"
BANK_SIZE = 65_536
BANK_SEEDS = {"LAB": 1042, "MK": 2042}
FRESH_LABEL_SEED = 3042
N_FRESH = 4096
LAB_PARTS = ("emotion", "genre", "style")
MK_K = {"affect8": 8, "caption10": 10, "image23": 23}
A3 = {"lambda_aspect": 3.0, "lambda_swap": 1.0}
RUNS = {
    "L3": ("LAB", A3),
    "L5": ("LAB", {"lambda_aspect": 1.0, "lambda_swap": 1.0, "aspect_beta": 0.0}),
    "LT": ("LAB", {**A3, "aspect_tau_fixed": True}),
    "MK3": ("MK", A3),
}
LABEL_RUNS = ("L3", "L5", "LT")


def folders(smoke: bool) -> dict:
    sub = "smoke" if smoke else ""
    return {"ckpt": HERE / "checkpoints" / sub, "res": HERE / "results" / sub}


def run_config(run: str, seed: int, steps: int = rg.FULL_STEPS):
    """E3's C0 base recipe (rg.run_config without its RUNS table) plus this stage's run fields."""
    base = dataclasses.replace(R3_CONFIG, painting_batches=True, seed=seed, epochs=steps, **rg.C0_CELL)
    return dataclasses.replace(base, **RUNS[run][1])


def local_rows(data, splits):
    """(scorer-train rows, local painting groups), aligned with E2's partitions.npz (asserted)."""
    st = np.asarray(splits.scorer_train)
    local_groups = np.unique(splits.groups[st], return_inverse=True)[1].astype(np.int64)
    e2 = np.load(rg.E2 / "partitions.npz")
    if not np.array_equal(e2["local_groups"], local_groups):
        raise AssertionError("local_groups differ from E2's partitions.npz")
    return st, local_groups


def load_bank(name: str, smoke: bool) -> tuple:
    res = folders(smoke)["res"]
    path = res / f"bank_{name}.npz"
    record = json.loads((res / "build_record.json").read_text())
    sha = rg.sha_file(path)
    if sha != record["sha256"][f"bank_{name}.npz"]:
        raise AssertionError(f"{path}: SHA-256 differs from build_record.json")
    z = np.load(path)
    return AspectEpisodes(str(z["aspect_a"]), str(z["aspect_b"]), *(z[f].astype(np.int64) for f in rg.FIELDS)), sha
