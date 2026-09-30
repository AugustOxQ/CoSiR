"""QC2 runner: fit the pilots' unchanged Attention-h1 baseline at one seed,
keep its snapshot, and score the held-out embedding with both yardsticks --
the pilots' independent re-clustering (reproduces the four-seed table in
attention_h1_baseline_seed_stress_pilot_report.md) and the sweep harness's
k-NN transfer / small-topic merge (spec: master report section 6i).

Usage (DAS6 via scripts/run_baseline_seed_snapshot.sh):
    python run_baseline_seed_snapshot.py --seed 42 --out-dir <dir>
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

BASELINE_PATH = (Path(__file__).resolve().parents[1] / "20260923_artelingo_buddy_analysis"
                 / "run_attention_h1_embedding_snapshot_pilot.py")
TRANSFER_KS = (10, 20, 30, 40)
MERGED_TRANSFER_KS = (20, 40)
MERGE_MIN_FRACTION = 0.02


def load_baseline():
    spec = importlib.util.spec_from_file_location("attention_h1_unchanged_baseline", str(BASELINE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline script: {BASELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route_heldout_paths(baseline):
    """The arch-sweep pilot hardcodes HELDOUT_STORAGE_DIR/HELDOUT_JSON to local
    paths; on DAS6 route them through the same env vars real_data.py uses.
    With PERCEPT_FEATURE_ROOT unset this is a no-op (local run)."""
    original = baseline.load_module

    def load_module(module_name, path):
        module = original(module_name, path)
        if path == baseline.ARCH_SWEEP_PATH and os.environ.get("PERCEPT_FEATURE_ROOT"):
            json_root = os.environ.get("PERCEPT_RAW_JSON_ROOT", "/data/PDD/artelingo")
            module.HELDOUT_STORAGE_DIR = f"{os.environ['PERCEPT_FEATURE_ROOT']}/artelingo_heldout/features"
            module.HELDOUT_JSON = f"{json_root}/artelingo_val_test.json"
        return module

    baseline.load_module = load_module


def score_snapshot(snapshot: Mapping[str, np.ndarray]) -> dict:
    from scripts.buddy_percept_sweep.clustering import merge_small_communities
    from scripts.buddy_percept_sweep.pilot_metrics import ami_emotion_genre
    from scripts.buddy_percept_sweep.targets import assign_to_train_communities

    train_embedding = snapshot["train_embedding_post"]
    train_community = np.asarray(snapshot["train_community_post"])
    heldout_embedding = snapshot["heldout_embedding_post"]
    heldout_community = np.asarray(snapshot["heldout_community_post"])
    emotion, genre = snapshot["heldout_emotion"], snapshot["heldout_genre"]

    def amis(labels):
        emo, gen = ami_emotion_genre(labels, emotion, genre)
        return {"emotion": emo, "genre": gen}

    def transfer(labels_train, ks):
        return {str(k): amis(assign_to_train_communities(
            train_embedding, labels_train, heldout_embedding, k)) for k in ks}

    merged, _ = merge_small_communities(train_embedding, train_community, MERGE_MIN_FRACTION)
    return {
        "independent": {
            **amis(heldout_community),
            "k": int(len(np.unique(heldout_community))),
            "train_k": int(len(np.unique(train_community))),
        },
        "transfer": transfer(train_community, TRANSFER_KS),
        "merged_transfer": {
            "k_after_merge": int(len(np.unique(merged))),
            "by_k": transfer(merged, MERGED_TRANSFER_KS),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline()
    route_heldout_paths(baseline)
    baseline.SEED = args.seed
    baseline.REPORT_OUT_DIR = str(out_dir)
    baseline.REPORT_PATH = str(out_dir / f"snapshot_seed{args.seed}_report.md")
    baseline.NPZ_PATH = str(out_dir / f"snapshot_seed{args.seed}.npz")
    baseline.main()

    with np.load(baseline.NPZ_PATH, allow_pickle=True) as snapshot:
        if int(snapshot["seed"]) != args.seed:
            raise RuntimeError(f"Snapshot seed mismatch for {args.seed}")
        result = {"seed": args.seed, **score_snapshot({k: snapshot[k] for k in snapshot.files})}
    line = json.dumps(result, separators=(",", ":"))
    (out_dir / f"baseline_seed{args.seed}.json").write_text(line + "\n")
    print("BASELINE_RESULT " + line, flush=True)


if __name__ == "__main__":
    main()
