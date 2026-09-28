"""Manual verification (not a pytest): runs 3 hardcoded configs through the
real ArtELingo data on the local GPU, confirming the full sweep pipeline
(Tasks 1-8) produces sane, non-degenerate metrics before creating the real
W&B sweep and dispatching to DAS6 (spec §8 step 1).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.buddy_percept_sweep.cache import FixedInputCache
from scripts.buddy_percept_sweep.config import resolve_trial_config
from scripts.buddy_percept_sweep.pipeline import run_trial
from scripts.buddy_percept_sweep.real_data import load_real_raw_inputs

BASE_CONFIG = dict(
    heads="attn1", num_heads=1, d_shared=32, lr=1e-3, noise_std=0.0,
    lambda_affect=1.0, batch_size=1024, weight_decay=0.0,
    teacher_graph_K=20, teacher_graph_alpha=0.5,
    leiden_resolution=1.0, merge_small_threshold=0.01,
    mapper_lr=1e-2, mapper_epochs=400, num_queries=1, mlp_head="linear",
    transfer_k=20, target_cutoff="single_label", class_balanced_loss=True,
    weight_decay_stage2=0.0,
)


def main() -> None:
    cache = FixedInputCache()
    variants = [
        {},  # baseline core config
        {"heads": "mlp128", "num_heads": 1},
        {"target_cutoff": 0.15, "num_queries": 4, "mlp_head": "one_hidden"},
    ]
    for i, overrides in enumerate(variants):
        raw = {**BASE_CONFIG, **overrides}
        config = resolve_trial_config(raw)
        fixed_inputs = cache.get(content_pca_dim=50, raw_loader=load_real_raw_inputs)
        start = time.monotonic()
        result = run_trial(config, fixed_inputs)
        elapsed = time.monotonic() - start
        print(
            f"[variant {i}] emotion_ami={result.emotion_ami:.4f} genre_ami={result.genre_ami:.4f} "
            f"stage2_auc={result.stage2_macro_auc:.4f} objective={result.objective:.4f} "
            f"n_topics={result.n_topics_after_merge} elapsed={elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
