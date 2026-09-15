"""Task 9 Step 4 (+ collapse diagnostic prompted by prototype_probe.py's
near-null selectivity result): for each prototype_pooled seed's final
checkpoint, compute:

  1. usage_entropy over the FULL 150k dataset (the smoke test's entropy
     check only covered 2 epochs / 1000 samples — this is the real,
     final, 100-epoch number).
  2. argmax-prototype cluster assignment per sample + condition_space_
     evaluator.py's silhouette_score machinery (Task 9 Step 4), using that
     assignment in place of the HDBSCAN labels it was originally built
     around.
  3. per-sample condition-vector diversity (mean pairwise cosine sim over
     a random subsample) — a direct, seed-independent check for whether
     the per-sample outputs are collapsing toward a small number of near-
     identical vectors, which is what would explain prototype_probe.py's
     near-null selectivity despite non-collapsed (near-uniform) *attention*
     entropy: near-uniform attention over a *shared* small prototype set
     produces near-constant `attn @ values` for every sample regardless of
     content, which is a different failure mode than argmax-collapse onto
     one prototype.
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

from src.model.prototype_bank import PrototypeBank
from src.utils import FeatureManager

RESULTS_ROOT = Path("/project/CoSiR/res/CoSiR_Experiment")
SEEDS = ["seed1", "seed2", "seed3"]
SILHOUETTE_SUBSAMPLE = 20000  # full 150k silhouette is O(n^2) memory/time — subsample
DIVERSITY_SUBSAMPLE = 2000
RNG_SEED = 0


def find_checkpoint(seed: str) -> Path:
    run_dirs = sorted((RESULTS_ROOT / "exp18_prototype_pooled" / seed).glob("2026*"))
    assert len(run_dirs) == 1, run_dirs
    ckpts = sorted((run_dirs[0] / "checkpoints").glob("phase_1_model_*.pt"))
    assert ckpts, run_dirs[0]
    return ckpts[-1]


def find_baseline_final_embeddings(seed: str) -> np.ndarray:
    run_dirs = sorted((RESULTS_ROOT / "exp18_baseline" / seed).glob("2026*"))
    assert len(run_dirs) == 1, run_dirs
    return np.load(run_dirs[0] / "final_embeddings" / "embeddings.npy")


def mean_pairwise_cosine(emb: np.ndarray, n_sub: int, rng: np.random.RandomState) -> float:
    idx = rng.choice(len(emb), size=min(n_sub, len(emb)), replace=False)
    x = emb[idx]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = x @ x.T
    mask = ~np.eye(len(x), dtype=bool)
    return float(sim[mask].mean())


def main():
    print("Loading FeatureManager + full 150k cached CLIP features...")
    fm = FeatureManager(storage_dir="/data/SSD2/pre_extract/redcaps_150k/features")
    feats = fm.load_all_to_ram(["img_features", "txt_features"])
    query_features = 0.5 * (feats["img_features"] + feats["txt_features"])  # [N, 512]
    n = query_features.shape[0]
    print(f"query_features: {query_features.shape}")

    rng = np.random.RandomState(RNG_SEED)

    results = {"prototype_pooled": {}, "baseline": {}}

    for seed in SEEDS:
        ckpt_path = find_checkpoint(seed)
        print(f"\n=== prototype_pooled / {seed}: {ckpt_path.name} ===")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pc = ckpt["prototype_bank_config"]
        bank = PrototypeBank(
            num_prototypes=pc["num_prototypes"], condition_dim=pc["condition_dim"],
            query_dim=pc["query_dim"], temperature_init=pc.get("temperature_init", 1.0),
        )
        bank.load_state_dict(ckpt["prototype_bank_state_dict"])
        bank.eval()

        with torch.no_grad():
            emb = bank(query_features)  # [N, D], triggers _last_attn
            entropy = bank.usage_entropy().item()
            log_p = float(np.log(pc["num_prototypes"]))
            attn = bank._last_attn  # [N, P]
            argmax_ids = attn.argmax(dim=-1).numpy()

        emb_np = emb.numpy()
        usage_counts = np.bincount(argmax_ids, minlength=pc["num_prototypes"])

        # Silhouette on a subsample using argmax-prototype id as cluster label
        sil_idx = rng.choice(n, size=min(SILHOUETTE_SUBSAMPLE, n), replace=False)
        sil_labels = argmax_ids[sil_idx]
        unique_labels = np.unique(sil_labels)
        silhouette = None
        if len(unique_labels) >= 2:
            silhouette = float(silhouette_score(emb_np[sil_idx], sil_labels))

        # PCA effective dims (95% variance), near-origin ratio — same fields
        # compute_condition_space_quality reports
        pca = PCA()
        pca.fit(emb_np[sil_idx])
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_effective_dims = int(np.argmax(cumsum >= 0.95) + 1)
        norms = np.linalg.norm(emb_np, axis=1)
        near_origin_ratio = float((norms < 0.5).mean())

        diversity = mean_pairwise_cosine(emb_np, DIVERSITY_SUBSAMPLE, rng)

        seed_result = {
            "usage_entropy": entropy,
            "log_num_prototypes": log_p,
            "entropy_ratio": entropy / log_p,
            "argmax_usage_counts": usage_counts.tolist(),
            "argmax_num_prototypes_used": int((usage_counts > 0).sum()),
            "silhouette_score_argmax_labels": silhouette,
            "n_effective_dims_95pct": n_effective_dims,
            "near_origin_ratio": near_origin_ratio,
            "mean_pairwise_cosine_sim": diversity,
        }
        print(json.dumps(seed_result, indent=2))
        results["prototype_pooled"][seed] = seed_result

        # Baseline comparison point: same diversity metric on the trained
        # free_vector table (per-sample condition vectors, updated every
        # step, not subject to the staleness bug) for direct contrast.
        base_emb = find_baseline_final_embeddings(seed)
        base_diversity = mean_pairwise_cosine(base_emb, DIVERSITY_SUBSAMPLE, rng)
        results["baseline"][seed] = {"mean_pairwise_cosine_sim": base_diversity}
        print(f"  baseline/{seed} mean_pairwise_cosine_sim: {base_diversity:.4f}")

    out_path = Path(__file__).parent / "prototype_coherence_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
