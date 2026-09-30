"""Controller smoke test of the full H2H trial path on real data (DAS6),
before any sweep is launched: one `run_h2h_trial` on the val subset for a
given flat config, printing the result and wall-clock timings (used to size
TRIALS_PER_CELL, spec ruling R4).

Configs are named presets (no JSON on the command line: the cluster launcher
re-parses quotes and spaces over ssh).

Usage (DAS6 via scripts/run_h2h_smoke.sh):
    python smoke_trial.py --preset buddy_pilot_k16 --seeds 1001,1002
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from scripts.buddy_percept_sweep.h2h_split import make_split
from scripts.buddy_percept_sweep.h2h_store import load_or_build_store
from scripts.buddy_percept_sweep.h2h_trial import resolve_h2h_config, run_h2h_trial
from scripts.buddy_percept_sweep.pilot_metrics import load_pilot_modules

PRESETS = {
    "buddy_pilot_k16": {"system": "buddy", "k_target": 16, "buddy_impl": "pilot",
                        "leiden_graph": "mknn", "merge_small_threshold": 0.01},
    "buddy_pilot_k40": {"system": "buddy", "k_target": 40, "buddy_impl": "pilot",
                        "leiden_graph": "pilot_repaired", "merge_small_threshold": 0.005},
    "buddy_harness_k40": {"system": "buddy", "k_target": 40, "buddy_impl": "harness",
                          "buddy_heads": "attn1", "buddy_num_heads": 4, "buddy_d_shared": 64,
                          "leiden_graph": "pilot_repaired", "merge_small_threshold": 0.005,
                          "target_cutoff": 0.15, "class_balanced_loss": True},
    "percept_k16": {"system": "percept", "k_target": 16},
    "percept_k40": {"system": "percept", "k_target": 40},
}


def _scalar(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value if isinstance(value, (int, float, str)) or value is None else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--seeds", default="1001")
    parser.add_argument("--subset", default="val")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = PRESETS[args.preset]
    cfg = resolve_h2h_config(raw)
    load_start = time.monotonic()
    store = load_or_build_store()
    split = make_split(store.heldout_emotion, store.heldout_genre)
    pilot = load_pilot_modules()
    percept_mods = None
    if cfg.system == "percept":
        from scripts.buddy_percept_sweep.h2h_percept import load_percept_modules
        percept_mods = load_percept_modules()
    load_seconds = time.monotonic() - load_start

    seeds = tuple(int(s) for s in args.seeds.split(","))
    trial_start = time.monotonic()
    result = run_h2h_trial(cfg, store, split, args.subset, seeds, pilot, percept_mods, device)
    trial_seconds = time.monotonic() - trial_start
    out = {
        "preset": args.preset, "config": raw, "seeds": list(seeds), "subset": args.subset,
        "objective": _scalar(result["objective"]), "split_digest": result["split_digest"],
        "load_seconds": round(load_seconds, 1), "trial_seconds": round(trial_seconds, 1),
        "per_seed": [{k: _scalar(v) for k, v in row.items()} for row in result["per_seed"]],
    }
    print("SMOKE " + json.dumps(out, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
