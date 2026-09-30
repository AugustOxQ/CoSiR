"""Early V3 check (controller, before Task 6): does the PercepT Stage 1 port
(`h2h_percept.fit_percept_stage1`, default config, k_target=40, seed 42)
reproduce §6g's PercepT refit on DAS6?

Stage 2 here is the pilots' OWN mapper and training loop
(`run_percept_mapper_symmetric_sweep_pilot.train_and_eval`) on the pilots'
own multi-hot targets (`base.soft_assignments` on the surviving centers,
argmax one-hot plus q > 2/K, as `s2.multi_hot_targets` builds them), so this
isolates the Stage 1 port. References: K=40; held-out native AMI
0.1094 / 0.2798 (fixed snapshot, §6b); macro AUC 0.9226 at lr 1e-2 / 400
epochs (4-seed mean, §6g) and 0.5843 at lr 1e-3 / 100 epochs (§6g's refit;
the fixed snapshot had 0.5925). §6g notes the refit is not bit-reproducible.

Usage (DAS6 via scripts/run_h2h_early_v3.sh):
    python early_v3_check.py --seed 42
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from scripts.buddy_percept_sweep.h2h_percept import (
    PerceptStage1Config, fit_percept_stage1, load_percept_modules,
)
from scripts.buddy_percept_sweep.h2h_store import load_or_build_store
from scripts.buddy_percept_sweep.pilot_metrics import ami_emotion_genre

SYMMETRIC_SWEEP_PATH = REPO / "src/test/20260927_deep_stage_analysis/run_percept_mapper_symmetric_sweep_pilot.py"
K_TARGET = 40
STRESS_SEEDS = (42, 7, 123, 2024)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pilot_multi_hot(base, latent, centers, threshold):
    """Exactly s2.multi_hot_targets, applied to already-computed latents."""
    with torch.no_grad():
        q = base.soft_assignments(torch.as_tensor(latent), torch.as_tensor(centers))
        targets = torch.zeros_like(q)
        targets.scatter_(1, q.argmax(dim=1, keepdim=True), 1.0)
        targets[q > threshold] = 1.0
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = load_or_build_store()
    mods = load_percept_modules()
    start = time.monotonic()
    out = fit_percept_stage1(PerceptStage1Config(), store, args.seed, K_TARGET, mods, device)
    stage1_seconds = time.monotonic() - start
    emotion, genre = ami_emotion_genre(out.heldout_native, store.heldout_emotion, store.heldout_genre)

    base = mods.s2.base
    threshold = 2.0 / K_TARGET
    centers = out.info["surviving_centers"]
    train_targets = pilot_multi_hot(base, out.train_embedding, centers, threshold)
    heldout_targets = pilot_multi_hot(base, out.heldout_embedding, centers, threshold)
    multi_labelled = float((heldout_targets.sum(dim=1) > 1).float().mean())

    sym = load_module("symmetric_sweep_for_early_v3", SYMMETRIC_SWEEP_PATH)
    tuned = [sym.train_and_eval(mods.s2, device, store.train_patches, train_targets,
                                store.heldout_patches, heldout_targets, seed, 1e-2, 400, f"tuned-{seed}")
             for seed in STRESS_SEEDS]
    untuned = sym.train_and_eval(mods.s2, device, store.train_patches, train_targets,
                                 store.heldout_patches, heldout_targets, args.seed, 1e-3, 100, "untuned")
    result = {
        "seed": args.seed, "k": int(out.info["n_train_topics"]),
        "dec_epochs": out.info.get("dec_epochs"), "dec_stop_reason": out.info.get("dec_stop_reason"),
        "heldout_unmapped": out.info.get("heldout_unmapped"), "stage1_seconds": round(stage1_seconds, 1),
        "native_emo": emotion, "native_genre": genre, "heldout_multi_labelled_fraction": multi_labelled,
        "auc_tuned_per_seed": [float(a) for a in tuned], "auc_tuned_mean": float(np.mean(tuned)),
        "auc_untuned": float(untuned),
        "reference": {"k": 40, "native_emo": 0.1094, "native_genre": 0.2798,
                      "auc_tuned_mean": 0.9226, "auc_untuned": 0.5843},
    }
    print("EARLY_V3 " + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
