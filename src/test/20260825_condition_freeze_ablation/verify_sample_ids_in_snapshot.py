"""
Verify Task 1 of the condition-freeze-ablation plan (docs/superpowers/plans/
2026-08-25-condition-freeze-ablation.md): condition_viz/epoch_*.pt now carries a
'sample_ids' field whose length and values exactly match the run's own persisted
training_embeddings/sample_ids.npy (the TrainableEmbeddingManager's ground-truth z-table
order) -- the CLAUDE.md 'sample ID consistency' check, applied to this new field.

Run against a completed smoke run's experiment directory:
    python src/test/20260825_condition_freeze_ablation/verify_sample_ids_in_snapshot.py <exp_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch


def verify(exp_dir: str) -> None:
    exp_path = Path(exp_dir)
    cond_viz_dir = exp_path / "condition_viz"
    epoch_files = sorted(cond_viz_dir.glob("epoch_*.pt"))
    assert epoch_files, f"no condition_viz/epoch_*.pt under {exp_dir}"

    truth_path = exp_path / "training_embeddings" / "sample_ids.npy"
    assert truth_path.exists(), f"missing ground-truth {truth_path}"
    truth_ids = [int(x) for x in np.load(truth_path).tolist()]

    for ef in epoch_files:
        snap = torch.load(ef, map_location="cpu")
        assert "sample_ids" in snap, f"{ef} missing 'sample_ids' key"
        ids = snap["sample_ids"]
        n_emb = snap["label_embeddings_all"].shape[0]
        assert len(ids) == n_emb, (
            f"{ef}: sample_ids length {len(ids)} != label_embeddings_all rows {n_emb}"
        )
        ids_int = [int(x) for x in ids]
        assert ids_int == truth_ids, (
            f"{ef}: sample_ids do not match training_embeddings/sample_ids.npy in row order "
            f"(expected positional equality; len {len(ids_int)} vs {len(truth_ids)}, or values/order differ)"
        )
        print(f"PASS {ef.name}: {len(ids)} sample_ids, all match ground truth")

    print(f"ALL {len(epoch_files)} SNAPSHOT(S) VERIFIED")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: verify_sample_ids_in_snapshot.py <exp_dir>"
    verify(sys.argv[1])
