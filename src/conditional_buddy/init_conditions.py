"""
Standalone runner for conditional-buddies condition initialization.

Examples
--------
2D sanity check (Steps 1–5 at N_DIM=2, saves debug_init_2d.png, no template write):
    python -m src.conditional_buddy.init_conditions dataset=impressions +n_dim=2 +visualize=true

Full init at cfg.model.embedding_dim, written to the dataset's template_embeddings/:
    python -m src.conditional_buddy.init_conditions dataset=impressions

Optional: +subsample=2000 to operate on a random subset (handy for quick 2D checks).

The init vectors are written in FeatureManager.get_all_sample_ids() order so the
existing TrainableEmbeddingManager template-loading path (load_imgtxt_template)
picks them up unchanged.
"""

import json
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from src.utils import FeatureManager
from src.conditional_buddy.buddy_graph import mutual_knn
from src.conditional_buddy.compute_buddies import _l2_normalize, compute_buddy_init
from src.conditional_buddy.embedding_methods import normalise_embedding, spectral_embedding
from src.conditional_buddy.buddy_graph import (
    ensure_min_degree,
    mix_distances,
    rank_normalise_sparse,
    sparse_cosine_distance,
    union_graph,
)
from src.conditional_buddy.visualize import plot_2d_buddies


def _buddy_cfg(cfg: DictConfig) -> dict:
    bud = cfg.train.get("buddies", {}) if "train" in cfg else {}
    return {
        "K": int(bud.get("k", 30)),
        "alpha": float(bud.get("alpha", 0.5)),
        "method": str(bud.get("method", "spectral")),
        "knn_batch_size": int(bud.get("knn_batch_size", 1024)),
        "normalize_method": str(bud.get("normalize_method", "rank")),
        "b_weight": float(bud.get("b_weight", 1.0)),
    }


def _load_features(cfg: DictConfig):
    storage_dir = cfg.featuremanager.storage_dir
    if not os.path.exists(os.path.join(storage_dir, "metadata.json")):
        raise FileNotFoundError(
            f"No feature store at '{storage_dir}'. Extract features first "
            "(run training with load_existing_features: false)."
        )
    fm = FeatureManager(storage_dir)
    data = fm.load_all_to_ram(["img_features", "txt_features"])
    img = data["img_features"].numpy().astype(np.float32)
    txt = data["txt_features"].numpy().astype(np.float32)
    sample_ids = [int(s) for s in data["sample_ids"].tolist()]
    print(f"[buddies] Loaded {len(sample_ids):,} samples from {storage_dir}")
    return img, txt, sample_ids


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    device = cfg.device if cfg.get("device") else "cuda"
    n_dim = int(cfg.get("n_dim") or cfg.model.embedding_dim)
    visualize = bool(cfg.get("visualize", False)) or n_dim == 2
    subsample = cfg.get("subsample")
    bud = _buddy_cfg(cfg)
    seed = int(cfg.get("seed", 42))

    img, txt, sample_ids = _load_features(cfg)

    if subsample is not None and int(subsample) < len(sample_ids):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(sample_ids), size=int(subsample), replace=False)
        img, txt = img[sel], txt[sel]
        sample_ids = [sample_ids[i] for i in sel]
        print(f"[buddies] Subsampled to {len(sample_ids):,} samples")

    debug_dir = Path(cfg.featuremanager.storage_dir).parent / "buddies_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if visualize:
        # Steps 1–5 at the requested n_dim, then the buddy-vs-random check.
        img_n, txt_n = _l2_normalize(img), _l2_normalize(txt)
        A_img = mutual_knn(img_n, bud["K"], device, bud["knn_batch_size"])
        A_txt = mutual_knn(txt_n, bud["K"], device, bud["knn_batch_size"])
        E, deg_stats = ensure_min_degree(union_graph(A_img, A_txt), img_n, txt_n, device)
        print(
            f"[buddies] E avg degree={E.nnz / E.shape[0]:.2f}, "
            f"isolated fixed={deg_stats['num_isolated']}"
        )
        D_mixed = mix_distances(
            rank_normalise_sparse(sparse_cosine_distance(img_n, E)),
            rank_normalise_sparse(sparse_cosine_distance(txt_n, E)),
            bud["alpha"],
        )
        # Strict buddies (intersection): the "are buddies neighbours?" test, and the
        # B-lean affinity boost when b_weight != 1.0.
        B = A_img.multiply(A_txt)
        B.data[:] = 1.0
        emb = normalise_embedding(
            spectral_embedding(D_mixed, n_dim, seed=seed,
                               b_edges=B.tocsr() if bud["b_weight"] != 1.0 else None,
                               b_weight=bud["b_weight"]),
            method=bud["normalize_method"],
        )

        out_path = str(debug_dir / f"debug_init_{n_dim}d.png")
        if n_dim == 2:
            plot_2d_buddies(emb, B.tocsr(), out_path)
        else:
            from src.conditional_buddy.visualize import buddy_vs_random_distance

            stats = buddy_vs_random_distance(emb, B.tocsr(), seed=seed)
            print(f"[buddies] buddy-vs-random distance report ({n_dim}D): {stats}")
        print("[buddies] Visualization-only run complete (no template written).")
        return

    # Full init → write template_embeddings/ in feature-store order.
    # return_edges=True persists the union graph E so the buddy-graph smoothness
    # regularizer (cfg.loss.lambda_buddy>0) works when reusing this template; the
    # edges are in the same row order as emb (no output reorder here).
    emb, edges = compute_buddy_init(
        img,
        txt,
        n_dim=n_dim,
        method=bud["method"],
        K=bud["K"],
        alpha=bud["alpha"],
        device=device,
        knn_batch_size=bud["knn_batch_size"],
        normalize_method=bud["normalize_method"],
        seed=seed,
        b_weight=bud["b_weight"],
        return_edges=True,
    )

    template_dir = Path(cfg.experiment.results_dir) / "template_embeddings"
    template_dir.mkdir(parents=True, exist_ok=True)
    np.lib.format.open_memmap(
        template_dir / "embeddings.npy", mode="w+", dtype=np.float32, shape=emb.shape
    )[:] = emb
    np.save(template_dir / "buddy_edges.npy", edges.astype(np.int64))
    ids_arr = np.array(sample_ids, dtype=np.int64)
    np.lib.format.open_memmap(
        template_dir / "sample_ids.npy", mode="w+", dtype=np.int64, shape=ids_arr.shape
    )[:] = ids_arr
    with open(template_dir / "metadata.json", "w") as f:
        json.dump({"total_samples": emb.shape[0], "embedding_dim": emb.shape[1]}, f, indent=2)
    # template_config.json must match the keys train_cosir checks on load.
    config = {
        "strategy": "buddies",
        "embedding_dim": n_dim,
        "factor": float(cfg.train.get("imgtxt_factor", 1.0)),
        "normalize": bool(cfg.train.get("normalize", False)),
        "extra": {"k": bud["K"], "alpha": bud["alpha"], "method": bud["method"],
                  "b_weight": bud["b_weight"]},
    }
    with open(template_dir / "template_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[buddies] Wrote template → {template_dir}")
    print(f"[buddies] template_config: {config}")


if __name__ == "__main__":
    main()
