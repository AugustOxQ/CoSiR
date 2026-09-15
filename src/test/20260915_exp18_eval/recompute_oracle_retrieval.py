"""Task 9 Step 2: correctly recomputed oracle retrieval (t2i/i2t R1) for both
Exp18 arms (baseline=free_vector, prototype_pooled), 3 seeds each.

Bypasses the ungated stale-representative-selection bug found mid-sweep
(progress.md, "Design questions answered mid-sweep" entry): for
prototype_pooled, `_eval_snapshot` FPS-samples representatives from
`embedding_manager.embeddings`, which is buddy-graph-initialized once at
epoch 0 and never updated again in this mode (batch_indices=None in the
training loop). This script instead uses `model.prototype_bank.values` — the
actual trained 16 prototypes — directly as representatives for that arm.
For baseline (free_vector), embedding_manager.embeddings IS updated every
step, so FPS-sampling it (same method training's own in-loop eval uses,
k=30 to match the final-epoch convention in train_cosir.py) remains valid.

Reads only saved checkpoints + each run's cached test_backbone_embeddings.pt
(frozen-CLIP test features, same for every run) — no CLIP backbone, no raw
test images, no live training-loop state needed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/project/CoSiR-buddy_prototype_conditioning")
sys.path.insert(0, str(REPO))

from src.model.combiner import CombinerLowRankAdapter
from src.model.prototype_bank import PrototypeBank
from src.eval.metrics import OracleMetrics, RecallMetrics
from src.eval.config import EvaluationConfig
from src.utils.tools import get_representatives_fps

RESULTS_ROOT = Path("/project/CoSiR/res/CoSiR_Experiment")
ARMS = ["baseline", "prototype_pooled"]
SEEDS = ["seed1", "seed2", "seed3"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINAL_EPOCH_N_REP = 30  # matches train_cosir.py's final-epoch representative_number


class ModelStub:
    """Exposes exactly what OracleMetrics.compute_oracle_recall_average needs:
    .combine(...), .project_other(...), .combine_side — replicates
    CoSiRModel.combine/project_other exactly (label_encoder is nn.Identity
    in the real model, so skipping it here is not an approximation).
    """

    def __init__(self, combiner, other_proj, combine_side):
        self.combiner = combiner
        self.other_proj = other_proj
        self.combine_side = combine_side

    def combine(self, emb, emb_full, labels, epoch=None, return_label_proj=False,
                return_delta=False, return_scalar=False):
        return self.combiner(emb, emb_full, labels, return_delta=return_delta, return_scalar=return_scalar)

    def project_other(self, emb):
        return self.other_proj(emb)


def find_run_dir(arm: str, seed: str) -> Path:
    seed_dir = RESULTS_ROOT / f"exp18_{arm}" / seed
    candidates = sorted(seed_dir.glob("2026*"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one run dir under {seed_dir}, found {candidates}")
    return candidates[0]


def find_checkpoint(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "checkpoints").glob("phase_1_model_*.pt"))
    if not candidates:
        raise RuntimeError(f"no checkpoint found under {run_dir / 'checkpoints'}")
    return candidates[-1]


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


def get_representatives(arm: str, run_dir: Path, ckpt: dict) -> torch.Tensor:
    if arm == "baseline":
        emb_dir = run_dir / "final_embeddings"
        emb = np.load(emb_dir / "embeddings.npy")
        reps = get_representatives_fps(torch.from_numpy(emb).float(), k=FINAL_EPOCH_N_REP)
        return reps.to(DEVICE)
    elif arm == "prototype_pooled":
        pc = ckpt["prototype_bank_config"]
        bank = PrototypeBank(
            num_prototypes=pc["num_prototypes"], condition_dim=pc["condition_dim"],
            query_dim=pc["query_dim"], temperature_init=pc.get("temperature_init", 1.0),
        )
        bank.load_state_dict(ckpt["prototype_bank_state_dict"])
        bank.eval()
        return bank.values.detach().to(DEVICE)
    else:
        raise ValueError(arm)


def evaluate_run(arm: str, seed: str) -> dict:
    run_dir = find_run_dir(arm, seed)
    ckpt_path = find_checkpoint(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build_model_stub(ckpt)
    representatives = get_representatives(arm, run_dir, ckpt)

    cache = torch.load(run_dir / "test_backbone_embeddings.pt", map_location="cpu", weights_only=False)
    img_emb, txt_emb, txt_full = cache["img_emb"], cache["txt_emb"], cache["txt_full"]
    t2i_map, i2t_map = cache["t2i_map"].to(DEVICE), cache["i2t_map"].to(DEVICE)

    cfg = EvaluationConfig(device=DEVICE, k_vals=[1, 5, 10], batch_size=512, cpu_offload=True)
    oracle = OracleMetrics(cfg)
    oracle_metrics = oracle.compute_oracle_recall_average(
        model, representatives, img_emb, txt_emb, txt_full, t2i_map, i2t_map,
        prefix="oracle", aggregation="max",
    )

    recall = RecallMetrics(cfg)
    raw_metrics = recall.compute_all_recalls(img_emb.to(DEVICE), txt_emb.to(DEVICE), t2i_map, i2t_map, prefix="raw")

    out = {"run_dir": str(run_dir), "checkpoint": str(ckpt_path), "n_representatives": representatives.shape[0]}
    out.update(oracle_metrics)
    out.update(raw_metrics)
    return out


def main():
    results = {}
    for arm in ARMS:
        results[arm] = {}
        for seed in SEEDS:
            print(f"\n=== {arm} / {seed} ===")
            results[arm][seed] = evaluate_run(arm, seed)
            print(json.dumps(results[arm][seed], indent=2, default=str))

    out_path = Path(__file__).parent / "oracle_retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
