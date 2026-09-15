"""Task 9 Step 3: interpretability probe for the prototype_pooled arm.

Reuses 17.1's control-task-gated probe harness (checkpoint_probe.py's
probe_selectivity), but computes the [N, D] condition array fresh via
model.prototype_bank(query_features) over the full RedCaps-150k cached
CLIP features, rather than reading final_embeddings/embeddings.npy — that
table is the buddy-graph-init snapshot from epoch 0 and is never updated
again in prototype_pooled mode (same staleness issue as the oracle-eval bug,
see progress.md's 2026-09-15 "Design questions answered mid-sweep" entry).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/project/CoSiR-buddy_prototype_conditioning")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src/test/20260915_condition_space_audit"))

from src.model.prototype_bank import PrototypeBank
from src.utils import FeatureManager
import axis_definitions as ax
from checkpoint_probe import probe_selectivity

RESULTS_ROOT = Path("/project/CoSiR/res/CoSiR_Experiment")
SEEDS = ["seed1", "seed2", "seed3"]
REDCAPS_META_PATH = "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"


def find_checkpoint(seed: str) -> Path:
    run_dirs = sorted((RESULTS_ROOT / "exp18_prototype_pooled" / seed).glob("2026*"))
    assert len(run_dirs) == 1, run_dirs
    ckpts = sorted((run_dirs[0] / "checkpoints").glob("phase_1_model_*.pt"))
    assert ckpts, run_dirs[0]
    return ckpts[-1]


def main():
    print("Loading FeatureManager + full 150k cached CLIP features...")
    fm = FeatureManager(storage_dir="/data/SSD2/pre_extract/redcaps_150k/features")
    sample_ids = np.array(fm.get_all_sample_ids())
    feats = fm.load_all_to_ram(["img_features", "txt_features"])
    query_features = 0.5 * (feats["img_features"] + feats["txt_features"])  # [N, 512]
    print(f"query_features: {query_features.shape}, sample_ids: {sample_ids.shape}")

    print(f"Loading RedCaps-150k metadata from {REDCAPS_META_PATH} ...")
    redcaps_meta = json.load(open(REDCAPS_META_PATH))
    records_by_sample_id = redcaps_meta
    records = [records_by_sample_id[int(s)] for s in sample_ids]

    results = {}
    for seed in SEEDS:
        ckpt_path = find_checkpoint(seed)
        print(f"\n=== {seed}: {ckpt_path} ===")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pc = ckpt["prototype_bank_config"]
        bank = PrototypeBank(
            num_prototypes=pc["num_prototypes"], condition_dim=pc["condition_dim"],
            query_dim=pc["query_dim"], temperature_init=pc.get("temperature_init", 1.0),
        )
        bank.load_state_dict(ckpt["prototype_bank_state_dict"])
        bank.eval()

        with torch.no_grad():
            emb = bank(query_features).numpy()  # [N, 16], fresh, from the trained bank

        seed_result = {}
        for axis_name in ax.REDCAPS_AXES.keys():
            keep, labels = ax.redcaps_binary_labels(records, axis_name)
            if len(np.unique(labels)) < 2 or len(labels) < 20:
                seed_result[axis_name] = {"verdict": "skipped", "reason": "insufficient samples"}
                continue
            seed_result[axis_name] = probe_selectivity(emb[keep], labels)
            print(f"  {axis_name}: {seed_result[axis_name]}")
        results[seed] = seed_result

    out_path = Path(__file__).parent / "prototype_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
