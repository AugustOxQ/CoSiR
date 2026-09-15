"""Evaluate real and deliberately mismatched conditions for Exp18 checkpoints.

This is a read-only diagnostic: it reconstructs saved components and invokes
the existing training-set evaluator without altering training code or state.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/project/CoSiR-buddy_prototype_conditioning")
sys.path.insert(0, str(REPO))

from src.eval import EvaluationConfig, EvaluationManager
from src.model.combiner import CombinerLowRankAdapter
from src.model.prototype_bank import PrototypeBank
from src.utils import FeatureManager


RESULTS_ROOT = Path("/project/CoSiR/res/CoSiR_Experiment")
FEATURES_DIR = "/data/SSD2/pre_extract/redcaps_150k/features"
ARMS = ["baseline", "prototype_pooled"]
SEEDS = ["seed1", "seed2", "seed3"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIRTUAL_BATCH_SIZE = 4096


class ModelStub:
    """Exposes exactly the evaluator-facing CoSiR model interface."""

    def __init__(self, combiner, other_proj, combine_side):
        self.combiner = combiner
        self.other_proj = other_proj
        self.combine_side = combine_side

    def combine(self, emb, emb_full, labels, epoch=None, return_label_proj=False,
                return_delta=False, return_scalar=False):
        return self.combiner(emb, emb_full, labels, return_delta=return_delta, return_scalar=return_scalar)

    def project_other(self, emb):
        return self.other_proj(emb)

    def eval(self):
        """The contained modules are already in eval mode; satisfy TrainEvaluator."""
        return self


class StaticConditionAdapter:
    """Looks up the final trained free-vector condition for each global ID."""

    def __init__(self, embeddings: torch.Tensor, sample_id_to_row: Dict[int, int]):
        self.embeddings = embeddings
        self.sample_id_to_row = sample_id_to_row

    def get_embeddings(self, sample_ids: List[int]) -> torch.Tensor:
        row_indices = [self.sample_id_to_row[int(sample_id)] for sample_id in sample_ids]
        return self.embeddings[row_indices]


class PrototypeConditionAdapter:
    """Computes fresh conditions from trained prototypes and cached CLIP features."""

    def __init__(
        self,
        prototype_bank: PrototypeBank,
        query_features: torch.Tensor,
        sample_id_to_row: Dict[int, int],
        device: str,
    ):
        self.prototype_bank = prototype_bank
        self.query_features = query_features
        self.sample_id_to_row = sample_id_to_row
        self.device = device

    def get_embeddings(self, sample_ids: List[int]) -> torch.Tensor:
        row_indices = [self.sample_id_to_row[int(sample_id)] for sample_id in sample_ids]
        query_subset = self.query_features[row_indices].to(self.device, non_blocking=True)
        with torch.no_grad():
            return self.prototype_bank(query_subset)


class VirtualFeatureManager:
    """Read-only, in-memory virtual batches for TrainEvaluator.

    The persisted store has 100k-sample shards, but TrainEvaluator performs
    quadratic work within each chunk. This presents the same global tensors in
    bounded slices while retaining the original global sample IDs and order.
    """

    def __init__(
        self,
        img_features: torch.Tensor,
        txt_features: torch.Tensor,
        sample_ids: torch.Tensor,
        batch_size: int = VIRTUAL_BATCH_SIZE,
    ):
        if not (len(img_features) == len(txt_features) == len(sample_ids)):
            raise ValueError("virtual feature tensors must have matching lengths")
        self.img_features = img_features
        self.txt_features = txt_features
        self.sample_ids = sample_ids
        self.batch_size = batch_size

    def get_num_chunks(self) -> int:
        return (len(self.sample_ids) + self.batch_size - 1) // self.batch_size

    def get_features_by_chunk(self, chunk_id: int) -> dict:
        start = chunk_id * self.batch_size
        end = min(start + self.batch_size, len(self.sample_ids))
        if start >= len(self.sample_ids):
            raise IndexError(f"virtual chunk {chunk_id} is out of range")
        return {
            "img_features": self.img_features[start:end],
            "txt_features": self.txt_features[start:end],
            "sample_ids": self.sample_ids[start:end],
        }


def find_run_dir(arm: str, seed: str) -> Path:
    seed_dir = RESULTS_ROOT / f"exp18_{arm}" / seed
    candidates = sorted(seed_dir.glob("2026*"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one run dir under {seed_dir}, found {candidates}")
    return candidates[0]


def find_checkpoint(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "checkpoints").glob("phase_1_model_*.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one checkpoint under {run_dir / 'checkpoints'}, found {candidates}")
    return candidates[0]


def build_model_stub(ckpt: dict) -> ModelStub:
    cc = ckpt["combiner_config"]
    combiner = CombinerLowRankAdapter(
        clip_feature_dim=cc["clip_feature_dim"], label_dim=cc["label_dim"], dropout=0.0,
    )
    combiner.load_state_dict(ckpt["combiner_state_dict"])
    combiner.eval().to(DEVICE)

    feature_dim = cc["clip_feature_dim"]
    other_proj = nn.Linear(feature_dim, feature_dim)
    other_proj.load_state_dict(ckpt["other_proj_state_dict"])
    other_proj.eval().to(DEVICE)

    return ModelStub(combiner, other_proj, ckpt["combine_side"])


def build_prototype_bank(ckpt: dict) -> PrototypeBank:
    pc = ckpt["prototype_bank_config"]
    prototype_bank = PrototypeBank(
        num_prototypes=pc["num_prototypes"],
        condition_dim=pc["condition_dim"],
        query_dim=pc["query_dim"],
        temperature_init=pc.get("temperature_init", 1.0),
    )
    prototype_bank.load_state_dict(ckpt["prototype_bank_state_dict"])
    return prototype_bank.eval().to(DEVICE)


def build_static_adapter(run_dir: Path, sample_id_to_row: Dict[int, int]) -> StaticConditionAdapter:
    emb_dir = run_dir / "final_embeddings"
    embeddings = torch.from_numpy(np.load(emb_dir / "embeddings.npy")).float()
    saved_sample_ids = np.load(emb_dir / "sample_ids.npy")
    if len(embeddings) != len(saved_sample_ids):
        raise RuntimeError(f"mismatched embedding and sample-id lengths in {emb_dir}")

    saved_id_to_row = {int(sample_id): row for row, sample_id in enumerate(saved_sample_ids)}
    if set(saved_id_to_row) != set(sample_id_to_row):
        raise RuntimeError(f"saved baseline sample IDs do not match FeatureManager IDs for {run_dir}")
    reordered_embeddings = embeddings[[saved_id_to_row[sample_id] for sample_id in sample_id_to_row]]
    return StaticConditionAdapter(reordered_embeddings, sample_id_to_row)


def evaluate_run(
    arm: str,
    seed: str,
    sample_id_to_row: Dict[int, int],
    query_features: torch.Tensor,
    feature_manager: VirtualFeatureManager,
) -> dict:
    run_dir = find_run_dir(arm, seed)
    checkpoint = find_checkpoint(run_dir)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_stub(ckpt)
    if arm == "baseline":
        embedding_manager = build_static_adapter(run_dir, sample_id_to_row)
    elif arm == "prototype_pooled":
        embedding_manager = PrototypeConditionAdapter(
            build_prototype_bank(ckpt), query_features, sample_id_to_row, DEVICE,
        )
    else:
        raise ValueError(f"unknown arm: {arm}")

    max_batches = feature_manager.get_num_chunks()
    evaluator = EvaluationManager(
        EvaluationConfig(device=DEVICE, train_max_batches=max_batches, print_metrics=True)
    )
    result = evaluator.evaluate_train(
        model=model,
        feature_manager=feature_manager,
        embedding_manager=embedding_manager,
        dataloader=range(feature_manager.get_num_chunks()),
        device=DEVICE,
        epoch=0,
        max_batches=max_batches,
    )
    print(result.metrics)

    output = {"run_dir": str(run_dir), "checkpoint": str(checkpoint)}
    output.update(result.metrics)
    return output


def main():
    print("Loading global RedCaps-150k cached CLIP features...")
    feature_store = FeatureManager(storage_dir=FEATURES_DIR)
    sample_ids = feature_store.get_all_sample_ids()
    sample_id_to_row = {int(sample_id): row for row, sample_id in enumerate(sample_ids)}
    features = feature_store.load_all_to_ram(["img_features", "txt_features"])
    query_features = 0.5 * (features["img_features"] + features["txt_features"])
    print(f"query_features: {tuple(query_features.shape)}, sample_ids: {len(sample_ids)}")
    feature_manager = VirtualFeatureManager(
        features["img_features"],
        features["txt_features"],
        torch.tensor(sample_ids, dtype=torch.long),
    )
    print(
        f"Virtual evaluation batches: {feature_manager.get_num_chunks()} "
        f"x <= {feature_manager.batch_size} samples"
    )

    results = {}
    for arm in ARMS:
        results[arm] = {}
        for seed in SEEDS:
            print(f"\n=== {arm} / {seed} ===")
            results[arm][seed] = evaluate_run(
                arm, seed, sample_id_to_row, query_features, feature_manager,
            )

    out_path = Path(__file__).parent / "condition_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
