"""Final analysis of the winning prototype-fix combo (temperature_init=0.3,
lr_prototype=1e-2) across its 3 seeds. Mixed layout: seed1 is the screen's own
temp0.3_lrproto1e-2 run; seeds 2/3 are the dedicated 3-seed confirmation runs.
Computes both the coherence/probe battery and oracle retrieval for all 3, and
writes one combined JSON for the report to cite.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

REPO = Path("/project/CoSiR-buddy_prototype_conditioning")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src/test/20260915_condition_space_audit"))

from src.model.combiner import CombinerLowRankAdapter
from src.model.prototype_bank import PrototypeBank
from src.eval.metrics import OracleMetrics, RecallMetrics
from src.eval.config import EvaluationConfig
from src.utils.tools import get_representatives_fps
from src.utils import FeatureManager
import axis_definitions as ax
from checkpoint_probe import probe_selectivity

RESULTS_ROOT = Path("/project/CoSiR/res/CoSiR_Experiment")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILHOUETTE_SUBSAMPLE = 20000
DIVERSITY_SUBSAMPLE = 2000
RNG_SEED = 0
REDCAPS_META_PATH = "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"

SEED_RUN_DIRS = {
    1: RESULTS_ROOT / "exp18_prototype_fix_screen" / "temp0.3_lrproto1e-2",
    2: RESULTS_ROOT / "exp18_prototype_fix_3seed" / "seed2",
    3: RESULTS_ROOT / "exp18_prototype_fix_3seed" / "seed3",
}


class ModelStub:
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
        pass


def find_checkpoint(seed: int) -> Path:
    run_dirs = sorted(SEED_RUN_DIRS[seed].glob("2026*"))
    assert len(run_dirs) == 1, f"seed{seed}: expected 1 run dir, found {run_dirs}"
    ckpts = sorted((run_dirs[0] / "checkpoints").glob("phase_1_model_*.pt"))
    assert ckpts, f"seed{seed}: no checkpoint under {run_dirs[0]}"
    return ckpts[-1], run_dirs[0]


def mean_pairwise_cosine(emb: np.ndarray, n_sub: int, rng: np.random.RandomState) -> float:
    idx = rng.choice(len(emb), size=min(n_sub, len(emb)), replace=False)
    x = emb[idx]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = x @ x.T
    mask = ~np.eye(len(x), dtype=bool)
    return float(sim[mask].mean())


def build_model_stub(ckpt: dict) -> ModelStub:
    cc = ckpt["combiner_config"]
    combiner = CombinerLowRankAdapter(clip_feature_dim=cc["clip_feature_dim"], label_dim=cc["label_dim"], dropout=0.0)
    combiner.load_state_dict(ckpt["combiner_state_dict"])
    combiner.eval().to(DEVICE)
    import torch.nn as nn
    feature_dim = cc["clip_feature_dim"]
    other_proj = nn.Linear(feature_dim, feature_dim)
    other_proj.load_state_dict(ckpt["other_proj_state_dict"])
    other_proj.eval().to(DEVICE)
    return ModelStub(combiner, other_proj, ckpt["combine_side"])


def main():
    print("Loading FeatureManager + full 150k cached CLIP features...")
    fm = FeatureManager(storage_dir="/data/SSD2/pre_extract/redcaps_150k/features")
    sample_ids = np.array(fm.get_all_sample_ids())
    feats = fm.load_all_to_ram(["img_features", "txt_features"])
    query_features = 0.5 * (feats["img_features"] + feats["txt_features"])
    n = query_features.shape[0]

    print(f"Loading RedCaps-150k metadata from {REDCAPS_META_PATH} ...")
    redcaps_meta = json.load(open(REDCAPS_META_PATH))
    records = [redcaps_meta[int(s)] for s in sample_ids]

    rng = np.random.RandomState(RNG_SEED)
    results = {}

    for seed in [1, 2, 3]:
        ckpt_path, run_dir = find_checkpoint(seed)
        print(f"\n=== seed{seed}: {ckpt_path.name} ===")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # --- Coherence + probe (bank forward pass) ---
        pc = ckpt["prototype_bank_config"]
        bank = PrototypeBank(
            num_prototypes=pc["num_prototypes"], condition_dim=pc["condition_dim"],
            query_dim=pc["query_dim"], temperature_init=pc.get("temperature_init", 1.0),
        )
        bank.load_state_dict(ckpt["prototype_bank_state_dict"])
        bank.eval()

        with torch.no_grad():
            emb = bank(query_features)
            entropy = bank.usage_entropy().item()
            log_p = float(np.log(pc["num_prototypes"]))
            attn = bank._last_attn
            argmax_ids = attn.argmax(dim=-1).numpy()

        emb_np = emb.numpy()
        usage_counts = np.bincount(argmax_ids, minlength=pc["num_prototypes"])

        sil_idx = rng.choice(n, size=min(SILHOUETTE_SUBSAMPLE, n), replace=False)
        sil_labels = argmax_ids[sil_idx]
        silhouette = float(silhouette_score(emb_np[sil_idx], sil_labels)) if len(np.unique(sil_labels)) >= 2 else None

        pca = PCA()
        pca.fit(emb_np[sil_idx])
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_effective_dims = int(np.argmax(cumsum >= 0.95) + 1)
        near_origin_ratio = float((np.linalg.norm(emb_np, axis=1) < 0.5).mean())
        diversity = mean_pairwise_cosine(emb_np, DIVERSITY_SUBSAMPLE, rng)

        probe_results = {}
        for axis_name in ax.REDCAPS_AXES.keys():
            keep, labels = ax.redcaps_binary_labels(records, axis_name)
            probe_results[axis_name] = probe_selectivity(emb_np[keep], labels) if len(np.unique(labels)) >= 2 and len(labels) >= 20 else {"verdict": "skipped"}

        # --- Oracle retrieval ---
        model = build_model_stub(ckpt)
        representatives = bank.values.detach().to(DEVICE)
        cache = torch.load(run_dir / "test_backbone_embeddings.pt", map_location="cpu", weights_only=False)
        img_emb, txt_emb, txt_full = cache["img_emb"], cache["txt_emb"], cache["txt_full"]
        t2i_map, i2t_map = cache["t2i_map"].to(DEVICE), cache["i2t_map"].to(DEVICE)
        cfg = EvaluationConfig(device=DEVICE, k_vals=[1, 5, 10], batch_size=512, cpu_offload=True)
        oracle_metrics = OracleMetrics(cfg).compute_oracle_recall_average(
            model, representatives, img_emb, txt_emb, txt_full, t2i_map, i2t_map, prefix="oracle", aggregation="max",
        )
        raw_metrics = RecallMetrics(cfg).compute_all_recalls(img_emb.to(DEVICE), txt_emb.to(DEVICE), t2i_map, i2t_map, prefix="raw")

        seed_result = {
            "checkpoint": str(ckpt_path),
            "usage_entropy": entropy,
            "entropy_ratio": entropy / log_p,
            "argmax_num_prototypes_used": int((usage_counts > 0).sum()),
            "argmax_top_share": float(usage_counts.max() / usage_counts.sum()),
            "silhouette_score_argmax_labels": silhouette,
            "n_effective_dims_95pct": n_effective_dims,
            "near_origin_ratio": near_origin_ratio,
            "mean_pairwise_cosine_sim": diversity,
            "probe_selectivity": probe_results,
        }
        seed_result.update(oracle_metrics)
        seed_result.update(raw_metrics)
        print(json.dumps(seed_result, indent=2, default=str))
        results[f"seed{seed}"] = seed_result

    out_path = Path(__file__).parent / "prototype_fix_3seed_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
