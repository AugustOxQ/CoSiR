"""Diagnostic battery for the exp18 prototype-fix screen (lr_prototype x
temperature_init grid), generalized from prototype_coherence.py +
prototype_probe.py for the new run-dir naming:
  res/CoSiR_Experiment/exp18_prototype_fix_screen/temp{T}_lrproto{L}/<run>/

Computes, per combo: usage_entropy (real, full 150k), argmax concentration,
silhouette on argmax labels, PCA effective dims, near-origin ratio, mean
pairwise cosine similarity, and the 17.1 probe-selectivity harness on
warmth/register — the same battery already run against the original 3-seed
prototype_pooled sweep, so results are directly comparable.

Usage: python prototype_coherence_screen.py <combo_dir_name> [<combo_dir_name> ...]
e.g. python prototype_coherence_screen.py temp1.0_lrproto1e-5 temp0.3_lrproto1e-5
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

from src.model.prototype_bank import PrototypeBank
from src.utils import FeatureManager
import axis_definitions as ax
from checkpoint_probe import probe_selectivity

RESULTS_ROOT = Path("/project/CoSiR/res/CoSiR_Experiment/exp18_prototype_fix_screen")
SILHOUETTE_SUBSAMPLE = 20000
DIVERSITY_SUBSAMPLE = 2000
RNG_SEED = 0
REDCAPS_META_PATH = "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"


def find_checkpoint(combo: str) -> Path:
    run_dirs = sorted((RESULTS_ROOT / combo).glob("2026*"))
    assert len(run_dirs) == 1, f"{combo}: expected 1 run dir, found {run_dirs}"
    ckpts = sorted((run_dirs[0] / "checkpoints").glob("phase_1_model_*.pt"))
    assert ckpts, f"{combo}: no checkpoint found under {run_dirs[0]}"
    return ckpts[-1]


def mean_pairwise_cosine(emb: np.ndarray, n_sub: int, rng: np.random.RandomState) -> float:
    idx = rng.choice(len(emb), size=min(n_sub, len(emb)), replace=False)
    x = emb[idx]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = x @ x.T
    mask = ~np.eye(len(x), dtype=bool)
    return float(sim[mask].mean())


def main():
    combos = sys.argv[1:]
    if not combos:
        combos = sorted(d.name for d in RESULTS_ROOT.iterdir() if d.is_dir())
    print(f"Combos to analyze: {combos}")

    print("Loading FeatureManager + full 150k cached CLIP features...")
    fm = FeatureManager(storage_dir="/data/SSD2/pre_extract/redcaps_150k/features")
    sample_ids = np.array(fm.get_all_sample_ids())
    feats = fm.load_all_to_ram(["img_features", "txt_features"])
    query_features = 0.5 * (feats["img_features"] + feats["txt_features"])
    n = query_features.shape[0]
    print(f"query_features: {query_features.shape}")

    print(f"Loading RedCaps-150k metadata from {REDCAPS_META_PATH} ...")
    redcaps_meta = json.load(open(REDCAPS_META_PATH))
    records = [redcaps_meta[int(s)] for s in sample_ids]

    rng = np.random.RandomState(RNG_SEED)
    results = {}

    for combo in combos:
        try:
            ckpt_path = find_checkpoint(combo)
        except AssertionError as e:
            print(f"SKIP {combo}: {e}")
            results[combo] = {"status": "not_ready", "reason": str(e)}
            continue

        print(f"\n=== {combo}: {ckpt_path.name} ===")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
        unique_labels = np.unique(sil_labels)
        silhouette = None
        if len(unique_labels) >= 2:
            silhouette = float(silhouette_score(emb_np[sil_idx], sil_labels))

        pca = PCA()
        pca.fit(emb_np[sil_idx])
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_effective_dims = int(np.argmax(cumsum >= 0.95) + 1)
        norms = np.linalg.norm(emb_np, axis=1)
        near_origin_ratio = float((norms < 0.5).mean())
        diversity = mean_pairwise_cosine(emb_np, DIVERSITY_SUBSAMPLE, rng)

        probe_results = {}
        for axis_name in ax.REDCAPS_AXES.keys():
            keep, labels = ax.redcaps_binary_labels(records, axis_name)
            if len(np.unique(labels)) < 2 or len(labels) < 20:
                probe_results[axis_name] = {"verdict": "skipped"}
                continue
            probe_results[axis_name] = probe_selectivity(emb_np[keep], labels)

        combo_result = {
            "status": "ok",
            "checkpoint": str(ckpt_path),
            "temperature_init_config": pc.get("temperature_init"),
            "usage_entropy": entropy,
            "entropy_ratio": entropy / log_p,
            "argmax_usage_counts": usage_counts.tolist(),
            "argmax_num_prototypes_used": int((usage_counts > 0).sum()),
            "argmax_top_share": float(usage_counts.max() / usage_counts.sum()),
            "silhouette_score_argmax_labels": silhouette,
            "n_effective_dims_95pct": n_effective_dims,
            "near_origin_ratio": near_origin_ratio,
            "mean_pairwise_cosine_sim": diversity,
            "probe_selectivity": probe_results,
        }
        print(json.dumps(combo_result, indent=2, default=str))
        results[combo] = combo_result

    out_path = Path(__file__).parent / "prototype_fix_screen_results.json"
    existing = {}
    if out_path.exists():
        existing = json.load(open(out_path))
    existing.update(results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
